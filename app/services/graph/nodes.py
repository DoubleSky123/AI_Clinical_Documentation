"""
LangGraph node functions for the clinical documentation pipeline.

Node execution order:
  retrieve_context
       │
  extract_soap
  ┌────┴────────────────┐
  │         │           │
audit_icd  audit_meds  detect_flags   (parallel)
  └────┬────────────────┘
  supervise
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from app.models.schemas import SOAPNote
from .state import ClinicalState

logger = logging.getLogger(__name__)

# ── Shared LLM ────────────────────────────────────────────────────────────────

_llm = ChatOllama(model="qwen2.5:7b", temperature=0)


# ── Pydantic schemas for specialized agents ───────────────────────────────────

class IcdAuditResult(BaseModel):
    """Corrected ICD-10-CM codes for this encounter."""
    icd10_codes: list[str]


class MedicationAuditResult(BaseModel):
    """Validated medication list with dose/route/frequency."""
    medications: list[str]


class ClinicalFlagsResult(BaseModel):
    """High-risk clinical indicators identified in the encounter."""
    clinical_flags: list[str]


# ── Helper ────────────────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({doc.metadata.get('source', 'kb')})\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )


# ── Node 1: Retrieval ─────────────────────────────────────────────────────────

def retrieve_context(state: ClinicalState) -> dict[str, Any]:
    """MMR retrieval over FAISS — fetch diverse clinical guideline chunks."""
    retriever = state["vectorstore"].as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.6},
    )
    docs = retriever.invoke(state["transcript"])
    return {"context": _format_docs(docs)}


# ── Node 2: Extraction Agent ──────────────────────────────────────────────────

_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior clinical documentation specialist.
Convert the physician-patient transcript into a structured SOAP note.
Scope: primary care and general dentistry encounters.

RETRIEVED CLINICAL GUIDELINES:
{context}

Rules:
- Use only information stated or clinically inferable from the transcript.
- Assign ICD-10-CM codes; include Z-exposure codes when applicable.
- Flag any high-risk features (SpO2 <94%, dyspnea, elderly, immunocompromised).
- Write all narrative fields (subjective, objective, assessment, plan, follow_up) in the SAME language as the transcript. If the transcript is in Chinese, respond in Chinese. If in English, respond in English.
"""),
    ("human", "TRANSCRIPT:\n{transcript}"),
])

def extract_soap(state: ClinicalState) -> dict[str, Any]:
    """Stage 1: RAG-grounded SOAP note extraction."""
    chain = _EXTRACTION_PROMPT | _llm.with_structured_output(SOAPNote)
    note: SOAPNote = chain.invoke({
        "context":    state["context"],
        "transcript": state["transcript"],
    })
    return {"soap_draft": note.model_dump()}


# ── Node 3: ICD Coding Agent (parallel) ──────────────────────────────────────

_ICD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an ICD-10-CM coding specialist.
Review the SOAP note and return the complete, accurate list of ICD-10-CM codes.
Include:
- Primary diagnosis codes
- Z-codes for exposure/contact when applicable
- Secondary diagnosis codes if clinically relevant
Return ONLY the corrected code list — do not repeat the full note.
"""),
    ("human", "SOAP NOTE:\n{soap_json}\n\nTRANSCRIPT:\n{transcript}"),
])

def audit_icd_codes(state: ClinicalState) -> dict[str, Any]:
    """ICD Coding Agent: audits and corrects ICD-10 codes."""
    import json
    chain = _ICD_PROMPT | _llm.with_structured_output(IcdAuditResult)
    result: IcdAuditResult = chain.invoke({
        "soap_json":  json.dumps(state["soap_draft"], indent=2),
        "transcript": state["transcript"],
    })
    # Strip any description text the model appended after the code (e.g. "U07.1 - COVID-19")
    cleaned = [code.split(" ")[0].split("-")[0].strip() for code in result.icd10_codes]
    return {"icd_codes": cleaned}


# ── Node 4: Medication Agent (parallel) ──────────────────────────────────────

_MED_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical pharmacist reviewing an AI-generated SOAP note.
Validate each medication for:
1. Correct dose and route for the diagnosis
2. Appropriate frequency and duration
3. Any obvious contraindications given the transcript

Return the corrected medication list. Each entry must include: drug name, dose, route, frequency, duration.
Respond in the SAME language as the transcript.
"""),
    ("human", "SOAP NOTE:\n{soap_json}\n\nTRANSCRIPT:\n{transcript}"),
])

def audit_medications(state: ClinicalState) -> dict[str, Any]:
    """Medication Agent: validates drug doses, routes, and durations."""
    import json
    chain = _MED_PROMPT | _llm.with_structured_output(MedicationAuditResult)
    result: MedicationAuditResult = chain.invoke({
        "soap_json":  json.dumps(state["soap_draft"], indent=2),
        "transcript": state["transcript"],
    })
    return {"medications": result.medications}


# ── Node 5: Clinical Flags Agent (parallel) ───────────────────────────────────

_FLAGS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical risk assessment specialist.
Identify ALL high-risk indicators present in this encounter:
- SpO2 < 94% or respiratory distress
- Age ≥ 65 or immunocompromised status
- Comorbidities (diabetes, heart disease, obesity, etc.)
- Severe symptoms (high fever, altered mental status, etc.)
- Social risk factors (no follow-up access, lives alone)

If no high-risk flags are present, return an empty list.
Respond in the SAME language as the transcript.
"""),
    ("human", "SOAP NOTE:\n{soap_json}\n\nTRANSCRIPT:\n{transcript}"),
])

def detect_clinical_flags(state: ClinicalState) -> dict[str, Any]:
    """Clinical Flags Agent: identifies high-risk indicators."""
    import json
    chain = _FLAGS_PROMPT | _llm.with_structured_output(ClinicalFlagsResult)
    result: ClinicalFlagsResult = chain.invoke({
        "soap_json":  json.dumps(state["soap_draft"], indent=2),
        "transcript": state["transcript"],
    })
    return {"clinical_flags": result.clinical_flags}


# ── Node 6: Supervisor ────────────────────────────────────────────────────────

def supervise(state: ClinicalState) -> dict[str, Any]:
    """
    Supervisor Node: merges outputs from the three parallel agents
    into the final validated SOAP note.
    """
    note = dict(state["soap_draft"])
    note["icd10_codes"]    = state.get("icd_codes",      note.get("icd10_codes", []))
    note["medications"]    = state.get("medications",    note.get("medications", []))
    note["clinical_flags"] = state.get("clinical_flags", note.get("clinical_flags", []))
    return {"final_note": note}
