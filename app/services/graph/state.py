"""
Shared state for the clinical documentation LangGraph pipeline.
All agent nodes read from and write to this TypedDict.
"""
from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class ClinicalState(TypedDict):
    transcript:     str
    vectorstore:    Any       # FAISS instance, injected at invocation; not serialized
    context:        str       # RAG retrieval output
    soap_draft:     dict      # Extraction Agent output
    icd_codes:      list[str] # ICD Coding Agent output
    medications:    list[str] # Medication Agent output
    clinical_flags: list[str] # Clinical Flags Agent output
    final_note:     dict      # Supervisor merged output
