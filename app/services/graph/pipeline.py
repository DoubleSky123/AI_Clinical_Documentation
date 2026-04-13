"""
LangGraph pipeline — assembles the clinical documentation StateGraph.

Graph topology:
    START
      │
  retrieve_context
      │
  extract_soap
      │
  ┌───┴──────────────┐──────────────────┐
  │                  │                  │
audit_icd_codes  audit_medications  detect_clinical_flags
  │                  │                  │
  └───────┬──────────┘──────────────────┘
      supervise
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
    extract_soap,
    retrieve_context,
    supervise,
)
from .state import ClinicalState


def build_graph(vectorstore: FAISS) -> StateGraph:
    """
    Compile the clinical documentation StateGraph.
    The vectorstore is injected into the initial state at invocation time.
    """
    graph = StateGraph(ClinicalState)

    # Register nodes
    graph.add_node("retrieve",      retrieve_context)
    graph.add_node("extract",       extract_soap)
    graph.add_node("audit_icd",     audit_icd_codes)
    graph.add_node("audit_meds",    audit_medications)
    graph.add_node("detect_flags",  detect_clinical_flags)
    graph.add_node("supervise",     supervise)

    # Sequential edges
    graph.add_edge(START,       "retrieve")
    graph.add_edge("retrieve",  "extract")

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


async def run_pipeline(transcript: str, vectorstore: FAISS) -> dict:
    """Async entry point — called by FastAPI routers."""
    compiled = build_graph(vectorstore)
    result = await compiled.ainvoke({
        "transcript":     transcript,
        "vectorstore":    vectorstore,
        "context":        "",
        "soap_draft":     {},
        "icd_codes":      [],
        "medications":    [],
        "clinical_flags": [],
        "final_note":     {},
    })
    return result["final_note"]
