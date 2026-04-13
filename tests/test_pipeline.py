"""
Tests for RAG pipeline and API endpoints.
Run: pytest tests/ -v
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ── Unit: SOAP generation ─────────────────────────────────────────────────────

SAMPLE_TRANSCRIPT = """
Doctor: What brings you in today?
Patient: I've had a fever of 101.5, sore throat, and body aches for 2 days.
         Also really tired and have a mild cough.
Doctor: Any COVID exposure recently?
Patient: Yes, my coworker tested positive 4 days ago.
Doctor: We'll run a rapid flu and COVID test.
Results: COVID-19 POSITIVE, Influenza NEGATIVE.
Doctor: Given the positive result and 2-day onset, you qualify for Paxlovid.
"""

EXPECTED_ICD_CODES = {"U07.1"}  # COVID-19 confirmed


@pytest.mark.asyncio
async def test_generate_soap_note_structure():
    """SOAP output must contain all required keys."""
    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = MagicMock()

    mock_result = {
        "subjective": "2-day fever 101.5F, sore throat, myalgia, fatigue. COVID exposure 4 days ago.",
        "objective":  "COVID-19 rapid antigen: POSITIVE. Influenza rapid: NEGATIVE. Temp 101.5F.",
        "assessment": "COVID-19 confirmed (U07.1). Mild-moderate severity.",
        "plan":       "Paxlovid 300/100mg BID x5 days. Acetaminophen PRN. Isolation x5 days.",
        "icd10_codes": ["U07.1", "Z20.822"],
        "medications": ["Nirmatrelvir/ritonavir (Paxlovid) 300mg/100mg BID x5 days"],
        "follow_up":   "Return if SpO2 <94% or dyspnea develops.",
        "clinical_flags": [],
    }

    with patch("app.services.rag_pipeline.build_pipeline") as mock_pipeline:
        mock_chain = AsyncMock(return_value=mock_result)
        mock_pipeline.return_value = mock_chain

        from app.services.rag_pipeline import generate_soap_note
        result = await generate_soap_note(SAMPLE_TRANSCRIPT, mock_vs)

    required_keys = {"subjective", "objective", "assessment", "plan",
                     "icd10_codes", "medications", "follow_up"}
    assert required_keys.issubset(result.keys()), "Missing required SOAP keys"
    assert EXPECTED_ICD_CODES.issubset(set(result["icd10_codes"])), "Missing ICD-10 U07.1"


def test_transcript_too_short():
    """API should reject transcripts under 50 chars."""
    from app.routers.notes import ValidatedTranscriptRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ValidatedTranscriptRequest(transcript="short")


def test_phi_guard_ssn():
    """API should reject transcripts containing SSN patterns."""
    from app.routers.notes import ValidatedTranscriptRequest
    from pydantic import ValidationError
    bad_transcript = "Patient SSN is 123-45-6789. " * 5
    with pytest.raises(ValidationError, match="SSN"):
        ValidatedTranscriptRequest(transcript=bad_transcript)
