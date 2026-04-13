"""
MCP Server — Clinical Documentation AI
Exposes three tools via the Model Context Protocol (stdio transport).

Tools:
  1. search_clinical_guidelines  — MMR retrieval over FAISS knowledge base
  2. generate_soap_note          — Full LangGraph pipeline (RAG + multi-agent)
  3. evaluate_soap_quality       — Score a generated note against expected outputs

Run:
    python -m app.mcp_server          # stdio (Claude Desktop / MCP Inspector)
    mcp dev app/mcp_server.py         # interactive dev mode
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ── FastMCP app ───────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="clinical-documentation-assistant",
    instructions=(
        "A clinical AI assistant. Use search_clinical_guidelines to look up "
        "CDC/WHO treatment guidelines, generate_soap_note to produce structured "
        "SOAP documentation from a transcript, and evaluate_soap_quality to "
        "audit note accuracy."
    ),
)

# ── Lazy-loaded vectorstore (loaded once on first tool call) ──────────────────

_vectorstore = None


def _get_vectorstore():
    """Load FAISS index from disk on first call; reuse thereafter."""
    global _vectorstore
    if _vectorstore is None:
        from app.vectorstore.build_index import load_vectorstore
        _vectorstore = load_vectorstore()
        logger.info("Vectorstore loaded for MCP server.")
    return _vectorstore


# ── Tool 1: Clinical Guidelines Search ───────────────────────────────────────

@mcp.tool()
async def search_clinical_guidelines(query: str, k: int = 6) -> list[dict]:
    """
    Search the clinical knowledge base (COVID-19, Influenza, URI guidelines)
    using MMR retrieval to return diverse, relevant chunks.

    Args:
        query: Clinical question or symptom description, e.g. "Paxlovid dosing for COVID-19"
        k:     Number of chunks to return (default 6, max 10)

    Returns:
        List of dicts with 'source' (guideline filename) and 'content' (chunk text).
    """
    from app.cache import get_guidelines, set_guidelines

    cached = await get_guidelines(query)
    if cached is not None:
        return cached

    k = min(k, 10)
    vs = _get_vectorstore()
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.6},
    )
    docs = retriever.invoke(query)
    results = [
        {
            "source":  doc.metadata.get("source", "clinical_kb"),
            "content": doc.page_content,
        }
        for doc in docs
    ]
    await set_guidelines(query, results)
    return results


# ── Tool 2: SOAP Note Generation ──────────────────────────────────────────────

@mcp.tool()
async def generate_soap_note(transcript: str, patient_id: Optional[str] = None) -> dict:
    """
    Generate a structured SOAP note from a de-identified physician-patient transcript.

    Runs the full LangGraph pipeline:
      retrieve_context → extract_soap → [audit_icd | audit_meds | detect_flags] → supervise

    Args:
        transcript:  De-identified dialogue text (minimum 50 characters).
        patient_id:  Optional de-identified patient identifier for tracking.

    Returns:
        SOAP note dict with keys: subjective, objective, assessment, plan,
        icd10_codes, medications, follow_up, clinical_flags.
    """
    if len(transcript.strip()) < 50:
        return {"error": "Transcript too short. Minimum 50 characters required."}

    vs = _get_vectorstore()

    from app.services.rag_pipeline import generate_soap_note as _generate
    soap = await _generate(transcript, vs)

    if patient_id:
        soap["patient_id"] = patient_id

    return soap


# ── Tool 3: SOAP Note Quality Evaluation ──────────────────────────────────────

@mcp.tool()
async def evaluate_soap_quality(
    transcript:               str,
    icd10_must_include:       list[str] = [],
    icd10_must_not_include:   list[str] = [],
    medications_must_include: list[str] = [],
    plan_must_mention:        list[str] = [],
    flags_must_include:       list[str] = [],
) -> dict:
    """
    Generate a SOAP note then score it against expected clinical outputs.
    Useful for regression testing after prompt changes or model upgrades.

    Args:
        transcript:               De-identified physician-patient dialogue.
        icd10_must_include:       ICD-10 codes that must appear, e.g. ["U07.1"].
        icd10_must_not_include:   ICD-10 codes that must NOT appear.
        medications_must_include: Drug names that must appear, e.g. ["Paxlovid"].
        plan_must_mention:        Keywords that must appear in the plan field.
        flags_must_include:       High-risk indicators that must be flagged.

    Returns:
        Dict with overall score (0.0-1.0), pass/fail counts,
        per-check details, and the full generated SOAP note.
    """
    vs = _get_vectorstore()

    from app.services.rag_pipeline import generate_soap_note as _generate
    from app.routers.evaluate import ExpectedOutputs, evaluate_soap

    soap = await _generate(transcript, vs)

    expected = ExpectedOutputs(
        icd10_must_include=icd10_must_include,
        icd10_must_NOT_include=icd10_must_not_include,
        medications_must_include=medications_must_include,
        plan_must_mention=plan_must_mention,
        flags_must_include=flags_must_include,
    )

    result = evaluate_soap(soap, expected)

    return {
        "score":      result.score,
        "passed":     result.passed,
        "failed":     result.failed,
        "checks":     [c.model_dump() for c in result.checks],
        "soap_note":  result.soap_note,
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
