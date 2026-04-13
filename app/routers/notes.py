"""
Clinical Notes Router — /api/v1/notes
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import field_validator

from app.models.schemas import SOAPResponse, TranscriptRequest
from app.services.rag_pipeline import generate_soap_note

router = APIRouter(prefix="/api/v1/notes", tags=["Clinical Notes"])


class ValidatedTranscriptRequest(TranscriptRequest):
    @field_validator("transcript")
    @classmethod
    def no_phi(cls, v: str) -> str:
        """
        Minimal PHI guard: reject transcripts with SSN-like patterns.
        Production → AWS Comprehend Medical for full de-identification.
        """
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", v):
            raise ValueError("Transcript may contain SSN. De-identify before submission.")
        return v


@router.post("/generate", response_model=SOAPResponse)
async def generate_note(body: ValidatedTranscriptRequest, request: Request):
    """
    Generate a structured SOAP note from a de-identified transcript.

    Pipeline:
      1. MMR retrieval over clinical guidelines (FAISS)
      2. LLM extraction → structured JSON (Stage 1)
      3. LLM self-refinement / coding audit (Stage 2)
    """
    vs = getattr(request.app.state, "vectorstore", None)
    if vs is None:
        raise HTTPException(status_code=503, detail="Vectorstore not ready.")

    t0 = time.perf_counter()
    try:
        soap = await generate_soap_note(body.transcript, vs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SOAPResponse(
        request_id=str(uuid.uuid4()),
        patient_id=body.patient_id,
        soap_note=soap,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@router.get("/health")
async def health():
    return {"status": "ok"}
