"""
Shared state for the clinical documentation LangGraph pipeline.
All agent nodes read from and write to this TypedDict.
"""
from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class ClinicalState(TypedDict):
    transcript:       str
    vectorstore:      Any        # FAISS instance, injected at invocation; not serialized
    bm25_retriever:   Any        # BM25Retriever instance, injected at invocation
    raw_docs:         list       # Raw Document objects from hybrid retrieval (pre-rerank)
    retrieval_scores: list[float] # CrossEncoder scores for top-k chunks
    context_quality:  str        # "good" | "low" | "none" — set by Evaluator Agent
    context:          str        # Formatted context injected into LLM prompt
    soap_draft:       dict       # Extraction Agent output
    icd_codes:        list[str]  # ICD Coding Agent output
    medications:      list[str]  # Medication Agent output
    clinical_flags:   list[str]  # Clinical Flags Agent output
    final_note:       dict       # Supervisor merged output
