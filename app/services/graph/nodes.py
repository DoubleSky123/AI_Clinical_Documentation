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

from langdetect import detect as _detect_lang

from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from app.models.schemas import SOAPNote
from .state import ClinicalState

logger = logging.getLogger(__name__)

# ── Shared LLM ────────────────────────────────────────────────────────────────

_llm = ChatOllama(model="qwen2.5:7b", temperature=0)

# ── CrossEncoder (lazy-loaded on first use) ───────────────────────────────────
# BAAI/bge-reranker-base: ~270 MB, CPU inference, ~2-5s for 15 pairs

_cross_encoder: CrossEncoder | None = None

def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading CrossEncoder (BAAI/bge-reranker-base)...")
        _cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
        logger.info("CrossEncoder loaded.")
    return _cross_encoder


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


def _lang_instruction(transcript: str) -> str:
    """Detect transcript language and return an explicit output-language directive.

    Uses Unicode block check first (reliable for CJK), falls back to langdetect.
    """
    import re
    if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', transcript):
        return "所有输出字段（subjective、objective、assessment、plan、follow_up、medications、clinical_flags）必须用中文书写。"
    try:
        lang = _detect_lang(transcript[:400])
        if lang.startswith("zh"):
            return "所有输出字段（subjective、objective、assessment、plan、follow_up、medications、clinical_flags）必须用中文书写。"
    except Exception:
        pass
    return "Write all output fields (subjective, objective, assessment, plan, follow_up, medications, clinical_flags) in English."


# ── Node 1: Hybrid Retrieval ──────────────────────────────────────────────────

def retrieve_context(state: ClinicalState) -> dict[str, Any]:
    """Hybrid retrieval: BM25 (keyword) + FAISS MMR (semantic), fused with RRF.

    BM25  → precise term matching (drug names, ICD codes, exact phrases)
    FAISS → semantic similarity (concept-level matching)
    RRF   → rank fusion, no manual weight tuning needed
    Returns raw_docs (unranked candidates) for the Evaluator Agent to rerank.
    """
    faiss_retriever = state["vectorstore"].as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.6},
    )
    bm25_retriever = state["bm25_retriever"]
    bm25_retriever.k = 10

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6],
    )
    docs = ensemble.invoke(state["transcript"])
    logger.info("Hybrid retrieval returned %d candidate chunks", len(docs))
    return {"raw_docs": docs}


# ── Node 2: Evaluator Agent ───────────────────────────────────────────────────

def evaluate_context(state: ClinicalState) -> dict[str, Any]:
    """Evaluator Agent: CrossEncoder reranking + retrieval quality gating.

    Scores each (transcript, chunk) pair with a cross-encoder.
    Selects top-6 by score, then classifies context quality:
      good  (avg >= 0.5): high-confidence retrieval
      low   (avg >= 0.1): relevant chunks found but weak signal
      none  (avg <  0.1): no relevant context — LLM will run without grounding
    """
    docs = state.get("raw_docs", [])
    if not docs:
        return {"context": "", "retrieval_scores": [], "context_quality": "none"}

    ce = _get_cross_encoder()
    # Truncate transcript to 512 chars for CrossEncoder input limit
    query = state["transcript"][:512]
    pairs = [(query, doc.page_content) for doc in docs]
    scores: list[float] = ce.predict(pairs).tolist()

    # Rank by descending score and take top-6
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_pairs = ranked[:6]
    top_scores = [s for s, _ in top_pairs]
    top_docs   = [d for _, d in top_pairs]

    avg_score = sum(top_scores) / len(top_scores)
    if avg_score >= 0.5:
        quality = "good"
    elif avg_score >= 0.1:
        quality = "low"
    else:
        quality = "none"

    logger.info(
        "Evaluator Agent: quality=%s  avg_score=%.3f  top_scores=%s",
        quality, avg_score, [round(s, 3) for s in top_scores],
    )
    return {
        "context":          _format_docs(top_docs),
        "retrieval_scores": top_scores,
        "context_quality":  quality,
    }


# ── Node 3: Extraction Agent ──────────────────────────────────────────────────

_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior clinical documentation specialist.
Convert the physician-patient transcript into a structured SOAP note.
Scope: primary care and general dentistry encounters.

LANGUAGE: {language_instruction}

RETRIEVED CLINICAL GUIDELINES:
{context}

Rules:
- Use only information stated or clinically inferable from the transcript.
- Assign ICD-10-CM codes; include Z-exposure codes when applicable.
- Flag any high-risk features (SpO2 <94%, dyspnea, elderly, immunocompromised).
"""),
    ("human", "TRANSCRIPT:\n{transcript}"),
])

def extract_soap(state: ClinicalState) -> dict[str, Any]:
    """Stage 1: RAG-grounded SOAP note extraction."""
    chain = _EXTRACTION_PROMPT | _llm.with_structured_output(SOAPNote)
    note: SOAPNote = chain.invoke({
        "context":             state["context"],
        "transcript":          state["transcript"],
        "language_instruction": _lang_instruction(state["transcript"]),
    })
    return {"soap_draft": note.model_dump()}


# ── Node 4: ICD Coding Agent (parallel) ──────────────────────────────────────

_ICD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an ICD-10-CM coding specialist for primary care and general dentistry.

CODING RULES (strictly enforced):
1. Only assign codes supported by explicit evidence in the transcript. Do NOT infer complications (e.g., do not assign E11.2 diabetic nephropathy unless kidney disease is documented).
2. Use the most general code when specificity is not documented (e.g., E11.9 for Type 2 diabetes with no stated complications).
3. Return codes in order: primary diagnosis first, then secondary diagnoses, then Z-codes.

CRITICAL CODE REFERENCE — common errors to avoid:
| Condition | CORRECT code | WRONG code |
|-----------|-------------|------------|
| COVID-19 confirmed | U07.1 | J09.X2, J10, J11 |
| COVID-19 exposure/contact | Z20.828 | Z20.41, Z20.49 |
| Influenza confirmed | J09.X2 / J10.1 / J11.1 | — |
| Type 2 diabetes, unspecified | E11.9 | E11.2 (requires kidney disease) |
| Fever, unspecified | R50.9 | — |
| Acute pharyngitis | J02.9 | — |
| Dental caries | K02.9 | — |

Return ONLY the corrected code list — no descriptions, no full note.
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


# ── Node 5: Medication Agent (parallel) ──────────────────────────────────────

_MED_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical pharmacist reviewing an AI-generated SOAP note.
Validate each medication for:
1. Correct dose and route for the diagnosis
2. Appropriate frequency and duration
3. Any obvious contraindications given the transcript

Return the corrected medication list. Each entry must include: drug name, dose, route, frequency, duration.
LANGUAGE: {language_instruction}
"""),
    ("human", "SOAP NOTE:\n{soap_json}\n\nTRANSCRIPT:\n{transcript}"),
])

def audit_medications(state: ClinicalState) -> dict[str, Any]:
    """Medication Agent: validates drug doses, routes, and durations."""
    import json
    chain = _MED_PROMPT | _llm.with_structured_output(MedicationAuditResult)
    result: MedicationAuditResult = chain.invoke({
        "soap_json":            json.dumps(state["soap_draft"], indent=2),
        "transcript":           state["transcript"],
        "language_instruction": _lang_instruction(state["transcript"]),
    })
    return {"medications": result.medications}


# ── Node 6: Clinical Flags Agent (parallel) ───────────────────────────────────

_FLAGS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical risk assessment specialist.
Identify ALL high-risk indicators present in this encounter:
- SpO2 < 94% or respiratory distress
- Age ≥ 65 or immunocompromised status
- Comorbidities (diabetes, heart disease, obesity, etc.)
- Severe symptoms (high fever, altered mental status, etc.)
- Social risk factors (no follow-up access, lives alone)

If no high-risk flags are present, return an empty list.
LANGUAGE: {language_instruction}
"""),
    ("human", "SOAP NOTE:\n{soap_json}\n\nTRANSCRIPT:\n{transcript}"),
])

def detect_clinical_flags(state: ClinicalState) -> dict[str, Any]:
    """Clinical Flags Agent: identifies high-risk indicators."""
    import json
    chain = _FLAGS_PROMPT | _llm.with_structured_output(ClinicalFlagsResult)
    result: ClinicalFlagsResult = chain.invoke({
        "soap_json":            json.dumps(state["soap_draft"], indent=2),
        "transcript":           state["transcript"],
        "language_instruction": _lang_instruction(state["transcript"]),
    })
    return {"clinical_flags": result.clinical_flags}


# ── Node 7: Supervisor ────────────────────────────────────────────────────────

def supervise(state: ClinicalState) -> dict[str, Any]:
    """
    Supervisor Node: merges outputs from the three parallel agents
    into the final validated SOAP note.
    Injects retrieval quality warnings when context was weak or absent.
    """
    import re
    note = dict(state["soap_draft"])
    note["icd10_codes"]    = state.get("icd_codes",      note.get("icd10_codes", []))
    note["medications"]    = state.get("medications",    note.get("medications", []))
    note["clinical_flags"] = state.get("clinical_flags", note.get("clinical_flags", []))

    quality = state.get("context_quality", "good")
    is_chinese = bool(re.search(r'[\u4e00-\u9fff]', state.get("transcript", "")))

    if quality == "low":
        warning = (
            "[低召回警告] 检索到的指南相关性较低，建议人工复核" if is_chinese
            else "[LOW RECALL] Retrieved guidelines have weak relevance — manual review recommended"
        )
        note["clinical_flags"] = [warning] + note["clinical_flags"]
    elif quality == "none":
        warning = (
            "[无召回警告] 未找到相关临床指南，SOAP 内容基于纯模型推理" if is_chinese
            else "[NO RECALL] No relevant guidelines found — SOAP based on model knowledge only"
        )
        note["clinical_flags"] = [warning] + note["clinical_flags"]

    return {"final_note": note}
