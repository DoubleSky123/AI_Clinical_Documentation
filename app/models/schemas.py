"""
Shared Pydantic schemas used across the API.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class SOAPNote(BaseModel):
    """Structured SOAP note for respiratory infection encounters."""
    subjective: str = Field(description="Chief complaint, HPI, symptom onset/duration/severity")
    objective: str = Field(description="Vitals, exam findings, rapid-test results if mentioned")
    assessment: str = Field(description="Primary diagnosis, differential, reasoning")
    plan: str = Field(description="Pharmacological and non-pharmacological treatment plan")
    icd10_codes: List[str] = Field(description="ICD-10-CM codes e.g. ['U07.1','Z20.822']")
    medications: List[str] = Field(description="Medications with dose/route/frequency")
    follow_up: str = Field(description="Return precautions and follow-up instructions")
    clinical_flags: List[str] = Field(
        default_factory=list,
        description="High-risk indicators: O2 sat <94%, dyspnea, elderly, immunocompromised"
    )


class TranscriptRequest(BaseModel):
    transcript: str = Field(..., min_length=50, description="Raw physician-patient dialogue")
    patient_id: Optional[str] = Field(default=None, description="De-identified patient ID")
    encounter_type: str = Field(default="respiratory")


class SOAPResponse(BaseModel):
    request_id: str
    patient_id: Optional[str]
    soap_note: dict
    latency_ms: float
    model_version: str = "qwen2.5:7b + FAISS-mmr"
