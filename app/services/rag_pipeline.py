"""
RAG Pipeline — Core retrieval + generation logic.
Resume highlights: two-stage chain (extract → refine), MMR diversity retrieval,
structured JSON output enforcement via Pydantic.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any


from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama

from app.models.schemas import SOAPNote

logger = logging.getLogger(__name__)

# ── Concurrency control ───────────────────────────────────────────────────────
# Limits simultaneous LangGraph pipeline runs to prevent Ollama overload.
# Tune MAX_CONCURRENT_PIPELINES based on your GPU/CPU capacity.
_MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_PIPELINES", "3"))
_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)


# ── Prompt Templates ─────────────────────────────────────────────────────────

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior clinical documentation specialist.
Convert the physician-patient transcript into a structured SOAP note.

SCOPE: Respiratory infections only — Influenza, COVID-19, URI.

RETRIEVED CLINICAL GUIDELINES:
{context}

RULES:
- Use only information stated or clinically inferable from the transcript.
- Assign ICD-10-CM codes precisely; include Z-exposure codes when applicable.
- Flag any high-risk features (SpO2 <94%, dyspnea, elderly, immunocompromised).
"""),
    ("human", "TRANSCRIPT:\n{transcript}"),
])

REFINEMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a medical coding auditor. Review the SOAP note for:
1. ICD-10 code accuracy and completeness
2. Medication dose/route plausibility
3. Missing clinical flags

Return the corrected note. If no changes needed, return it unchanged.
"""),
    ("human", "SOAP NOTE TO REVIEW:\n{soap_json}"),
])


# ── Retriever ─────────────────────────────────────────────────────────────────

def _build_retriever(vectorstore: FAISS):
    """
    MMR (Maximal Marginal Relevance) retrieval:
    balances relevance + diversity to avoid redundant chunks.
    fetch_k=20 candidates → re-ranked → top k=6 returned.
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.6},
    )


def _format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "clinical_kb")
        parts.append(f"[{i}] ({src})\n{doc.page_content}")
    return "\n\n".join(parts)


# ── Two-Stage Chain ───────────────────────────────────────────────────────────

def build_pipeline(vectorstore: FAISS) -> Any:
    """
    Stage 1 — RAG-grounded extraction (structured output via with_structured_output)
    Stage 2 — Self-refinement / audit pass
    Returns a single LangChain runnable.
    """
    llm           = ChatOllama(model="qwen2.5:7b", temperature=0)
    structured_llm = llm.with_structured_output(SOAPNote)
    retriever     = _build_retriever(vectorstore)

    # Stage 1
    extraction_chain = (
        RunnableParallel({
            "context":    retriever | RunnableLambda(_format_docs),
            "transcript": RunnablePassthrough(),
        })
        | EXTRACTION_PROMPT
        | structured_llm
        | RunnableLambda(lambda note: note.model_dump())
    )

    # Stage 2 — self-refinement (catches ICD coding errors, missing flags)
    def refine(soap_dict: dict) -> dict:
        try:
            refine_chain = REFINEMENT_PROMPT | structured_llm
            result = refine_chain.invoke({
                "soap_json": json.dumps(soap_dict, indent=2),
            })
            return result.model_dump()
        except Exception as e:
            logger.warning("Refinement skipped: %s", e)
            return soap_dict

    return extraction_chain | RunnableLambda(refine)


async def generate_soap_note(transcript: str, vectorstore: FAISS) -> dict:
    """
    Async entry point called by FastAPI.
    1. Check Redis cache — return immediately on hit.
    2. Acquire semaphore — max _MAX_CONCURRENT pipelines run simultaneously.
    3. Run LangGraph pipeline — write result back to cache.
    """
    from app.cache import get_soap, set_soap
    from app.services.graph.pipeline import run_pipeline

    cached = await get_soap(transcript)
    if cached is not None:
        return cached

    async with _semaphore:
        # Re-check cache: another request may have populated it while we waited
        cached = await get_soap(transcript)
        if cached is not None:
            return cached

        result = await run_pipeline(transcript, vectorstore)

    await set_soap(transcript, result)
    return result
