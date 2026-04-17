"""
Voice Router — /api/v1/voice
End-to-end: audio file → transcript → SOAP Note

Pipeline:
  1. Whisper ASR  : audio (.wav/.mp3/.m4a/.webm) → transcript text
  2. RAG Pipeline : transcript → structured SOAP Note JSON
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
import uuid

import os
import whisper
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

# ── ffmpeg PATH fix (Windows local dev only) ──────────────────────────────────
# In Docker, ffmpeg is installed via apt — FFMPEG_DIR env var is left empty.
_FFMPEG_DIR = os.getenv("FFMPEG_DIR", r"D:\ffmpeg\ffmpeg-8.1-essentials_build\bin")
if _FFMPEG_DIR and _FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

from pydantic import BaseModel

from app.services.rag_pipeline import generate_soap_note

router = APIRouter(prefix="/api/v1/voice", tags=["Voice & Transcription"])

# WHISPER_CACHE: local dev → D:\ai_cache\whisper, Docker → /root/.cache/whisper
_WHISPER_CACHE = os.getenv("WHISPER_CACHE", r"D:\ai_cache\whisper") or None
_whisper_model = whisper.load_model("base", download_root=_WHISPER_CACHE)

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".webm", ".mp4", ".mpeg", ".ogg"}

# ── Response Schema ───────────────────────────────────────────────────────────

class VoiceDocumentResponse(BaseModel):
    request_id:    str
    transcript:    str        # raw Whisper output
    soap_note:     dict       # structured SOAP Note
    asr_latency_ms:   float   # time for Whisper step
    llm_latency_ms:   float   # time for RAG pipeline step
    total_latency_ms: float


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/transcribe-and-document", response_model=VoiceDocumentResponse)
async def transcribe_and_document(
    audio: UploadFile = File(..., description="Audio file of physician-patient encounter"),
    request: Request = None,
):
    """
    Full ambient documentation pipeline:

    1. Receive audio file (wav/mp3/m4a/webm)
    2. Whisper-1 ASR → transcript
    3. RAG pipeline  → structured SOAP Note

    Mirrors the core pipeline used by Abridge and Microsoft DAX.
    """
    # ── Validate file format ──────────────────────────────────────────────────
    suffix = os.path.splitext(audio.filename or "")[1].lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format '{suffix}'. Supported: {SUPPORTED_FORMATS}"
        )

    vs = getattr(request.app.state, "vectorstore", None)
    if vs is None:
        raise HTTPException(status_code=503, detail="Vectorstore not ready")
    bm25 = getattr(request.app.state, "bm25_retriever", None)
    if bm25 is None:
        raise HTTPException(status_code=503, detail="BM25 retriever not ready")

    t_total_start = time.perf_counter()

    # ── Step 1: Whisper ASR ───────────────────────────────────────────────────
    audio_bytes = await audio.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        t_asr_start = time.perf_counter()

        # Run synchronous whisper in a thread so we don't block the event loop
        result     = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _whisper_model.transcribe(tmp_path, language="en")
        )
        asr_ms     = round((time.perf_counter() - t_asr_start) * 1000, 2)
        transcript = result["text"]

    finally:
        os.unlink(tmp_path)  # always clean up temp file

    if not transcript or len(transcript.strip()) < 20:
        raise HTTPException(
            status_code=422,
            detail="Transcript too short — audio may be silent or inaudible."
        )

    # ── Step 2: RAG Pipeline ──────────────────────────────────────────────────
    t_llm_start = time.perf_counter()
    soap        = await generate_soap_note(transcript, vs, bm25)
    llm_ms      = round((time.perf_counter() - t_llm_start) * 1000, 2)

    total_ms = round((time.perf_counter() - t_total_start) * 1000, 2)

    return VoiceDocumentResponse(
        request_id=str(uuid.uuid4()),
        transcript=transcript,
        soap_note=soap,
        asr_latency_ms=asr_ms,
        llm_latency_ms=llm_ms,
        total_latency_ms=total_ms,
    )


# ── Transcript-only endpoint (lightweight, no SOAP generation) ────────────────

@router.post("/transcribe-only")
async def transcribe_only(
    audio: UploadFile = File(...),
):
    """
    Whisper ASR only — returns raw transcript without SOAP generation.
    Useful for previewing transcription quality before full pipeline.
    """
    suffix = os.path.splitext(audio.filename or "")[1].lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=415, detail=f"Unsupported format '{suffix}'")

    audio_bytes = await audio.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        t0     = time.perf_counter()
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _whisper_model.transcribe(tmp_path, language="en")
        )
        ms = round((time.perf_counter() - t0) * 1000, 2)
    finally:
        os.unlink(tmp_path)

    return {
        "transcript":  result["text"],
        "latency_ms":  ms,
        "word_count":  len(transcription.text.split())
    }
