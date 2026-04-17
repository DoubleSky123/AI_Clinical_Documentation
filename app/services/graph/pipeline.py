"""
LangGraph pipeline — assembles the clinical documentation StateGraph.

Graph topology:
    START
      │
  retrieve_context    (BM25 + FAISS hybrid → raw_docs)
      │
  evaluate_context    (CrossEncoder rerank → context + quality score)
      │
  extract_soap
      │
  ┌───┴──────────────┐──────────────────┐
  │                  │                  │
audit_icd_codes  audit_medications  detect_clinical_flags
  │                  │                  │
  └───────┬──────────┘──────────────────┘
      supervise       (merges + injects quality warnings)
          │
         END
"""
from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langgraph.graph import END, START, StateGraph

from .nodes import (
    audit_icd_codes,
    audit_medications,
    detect_clinical_flags,
    evaluate_context,
    extract_soap,
    retrieve_context,
    supervise,
)
from .state import ClinicalState


def build_graph() -> StateGraph:
    """Compile the clinical documentation StateGraph."""
    graph = StateGraph(ClinicalState)

    # Register nodes
    graph.add_node("retrieve",      retrieve_context)
    graph.add_node("evaluate",      evaluate_context)
    graph.add_node("extract",       extract_soap)
    graph.add_node("audit_icd",     audit_icd_codes)
    graph.add_node("audit_meds",    audit_medications)
    graph.add_node("detect_flags",  detect_clinical_flags)
    graph.add_node("supervise",     supervise)

    # Sequential: retrieve → evaluate → extract
    graph.add_edge(START,       "retrieve")
    graph.add_edge("retrieve",  "evaluate")
    graph.add_edge("evaluate",  "extract")

    # Fan-out: extract → three parallel agents
    graph.add_edge("extract",   "audit_icd")
    graph.add_edge("extract",   "audit_meds")
    graph.add_edge("extract",   "detect_flags")

    # Fan-in: all three agents → supervisor
    graph.add_edge("audit_icd",    "supervise")
    graph.add_edge("audit_meds",   "supervise")
    graph.add_edge("detect_flags", "supervise")

    graph.add_edge("supervise", END)

    return graph.compile()


async def run_pipeline(transcript: str, vectorstore: FAISS, bm25_retriever: any) -> dict:
    """Async entry point — called by FastAPI routers."""
    compiled = build_graph()
    result = await compiled.ainvoke({
        "transcript":       transcript,
        "vectorstore":      vectorstore,
        "bm25_retriever":   bm25_retriever,
        "raw_docs":         [],
        "retrieval_scores": [],
        "context_quality":  "none",
        "context":          "",
        "soap_draft":       {},
        "icd_codes":        [],
        "medications":      [],
        "clinical_flags":   [],
        "final_note":       {},
    })
    return result["final_note"]
