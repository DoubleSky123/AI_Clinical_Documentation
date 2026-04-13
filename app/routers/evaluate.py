"""
Evaluation Router — /api/v1/evaluate
Exposes SOAP note quality scoring as an API endpoint.

Two modes:
  1. /evaluate/single  — score one transcript against expected outputs
  2. /evaluate/batch   — run all built-in test cases, return aggregate metrics
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.services.rag_pipeline import generate_soap_note

router = APIRouter(prefix="/api/v1/evaluate", tags=["Evaluation"])


# ── Request / Response Schemas ────────────────────────────────────────────────

class ExpectedOutputs(BaseModel):
    """What the caller expects the SOAP note to contain."""
    icd10_must_include:       list[str] = []  # e.g. ["U07.1"]
    icd10_must_NOT_include:   list[str] = []
    medications_must_include: list[str] = []  # e.g. ["Paxlovid"]
    medications_must_NOT_include: list[str] = []  # e.g. ["amoxicillin"]
    plan_must_mention:        list[str] = []  # keyword checks
    flags_must_include:       list[str] = []  # e.g. ["SpO2", "elderly"]


class EvaluateRequest(BaseModel):
    transcript: str
    expected:   ExpectedOutputs
    case_id:    Optional[str] = None


class CheckDetail(BaseModel):
    check:  str   # human-readable description
    passed: bool
    detail: str   # what was found vs expected


class EvaluateResponse(BaseModel):
    request_id:    str
    case_id:       Optional[str]
    score:         float          # 0.0 – 1.0
    passed:        int
    failed:        int
    checks:        list[CheckDetail]
    soap_note:     dict
    latency_ms:    float


class BatchEvaluateResponse(BaseModel):
    request_id:       str
    total_cases:      int
    overall_score:    float
    cases_passed:     int
    cases_failed:     int
    per_case_results: list[EvaluateResponse]
    latency_ms:       float


# ── Core Evaluation Logic ─────────────────────────────────────────────────────

def _flatten(soap: dict) -> str:
    return json.dumps(soap).lower()


def evaluate_soap(soap: dict, expected: ExpectedOutputs, case_id: str = "") -> EvaluateResponse:
    """
    Run all checks against a generated SOAP note.
    Returns structured results with per-check pass/fail details.
    """
    checks: list[CheckDetail] = []
    text   = _flatten(soap)
    codes  = [c.upper() for c in soap.get("icd10_codes", [])]
    flags  = " ".join(soap.get("clinical_flags", [])).lower()

    # ── 1. ICD codes that must be present ────────────────────────────────────
    for code in expected.icd10_must_include:
        hit = any(c.startswith(code.upper()) for c in codes)
        checks.append(CheckDetail(
            check=f"ICD-10 must include: {code}",
            passed=hit,
            detail=f"Found: {codes}" if hit else f"Missing. Got: {codes}"
        ))

    # ── 2. ICD codes that must NOT be present ────────────────────────────────
    for code in expected.icd10_must_NOT_include:
        hit = any(c.startswith(code.upper()) for c in codes)
        checks.append(CheckDetail(
            check=f"ICD-10 must NOT include: {code}",
            passed=not hit,
            detail="Correctly absent" if not hit else f"Should not appear. Got: {codes}"
        ))

    # ── 3. Medications that must be present ──────────────────────────────────
    for med in expected.medications_must_include:
        hit = med.lower() in text
        checks.append(CheckDetail(
            check=f"Medication must include: {med}",
            passed=hit,
            detail="Found in output" if hit else f"Not mentioned. Medications: {soap.get('medications', [])}"
        ))

    # ── 4. Medications that must NOT be present ───────────────────────────────
    for med in expected.medications_must_NOT_include:
        hit = med.lower() in text
        checks.append(CheckDetail(
            check=f"Medication must NOT include: {med}",
            passed=not hit,
            detail="Correctly absent" if not hit else f"Inappropriately present: {med}"
        ))

    # ── 5. Plan keyword checks ────────────────────────────────────────────────
    for kw in expected.plan_must_mention:
        hit = kw.lower() in text
        checks.append(CheckDetail(
            check=f"Plan must mention: '{kw}'",
            passed=hit,
            detail="Found" if hit else f"Missing keyword '{kw}' in plan/follow_up"
        ))

    # ── 6. Clinical flags ─────────────────────────────────────────────────────
    for kw in expected.flags_must_include:
        hit = kw.lower() in flags or kw.lower() in text
        checks.append(CheckDetail(
            check=f"Clinical flag must include: '{kw}'",
            passed=hit,
            detail="Flagged correctly" if hit else f"High-risk indicator '{kw}' not flagged"
        ))

    # ── 7. Structural completeness ────────────────────────────────────────────
    for f_name in ["subjective", "objective", "assessment", "plan", "icd10_codes", "medications"]:
        val  = soap.get(f_name)
        ok   = bool(val and (len(val) > 0 if isinstance(val, list) else len(str(val)) > 10))
        checks.append(CheckDetail(
            check=f"Field '{f_name}' is populated",
            passed=ok,
            detail="OK" if ok else f"Field '{f_name}' is empty or missing"
        ))

    passed = sum(1 for c in checks if c.passed)
    failed = len(checks) - passed
    score  = round(passed / len(checks), 3) if checks else 0.0

    return EvaluateResponse(
        request_id=str(uuid.uuid4()),
        case_id=case_id,
        score=score,
        passed=passed,
        failed=failed,
        checks=checks,
        soap_note=soap,
        latency_ms=0.0,  # set by caller
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/single", response_model=EvaluateResponse)
async def evaluate_single(body: EvaluateRequest, request: Request):
    """
    Evaluate one transcript against caller-supplied expected outputs.

    Example use cases:
    - CI/CD regression test after prompt change
    - A/B testing two prompt versions
    - QA review of a specific encounter type
    """
    vs = getattr(request.app.state, "vectorstore", None)
    if vs is None:
        raise HTTPException(status_code=503, detail="Vectorstore not ready")

    t0   = time.perf_counter()
    soap = await generate_soap_note(body.transcript, vs)
    ms   = round((time.perf_counter() - t0) * 1000, 2)

    result            = evaluate_soap(soap, body.expected, body.case_id or "")
    result.latency_ms = ms
    return result


@router.post("/batch", response_model=BatchEvaluateResponse)
async def evaluate_batch(request: Request):
    """
    Run all built-in clinical test cases and return aggregate metrics.

    Use this endpoint to:
    - Benchmark a new model version
    - Run nightly regression tests
    - Track quality trends over time
    """
    from tests.sample_transcripts import CASES

    vs = getattr(request.app.state, "vectorstore", None)
    if vs is None:
        raise HTTPException(status_code=503, detail="Vectorstore not ready")

    t0      = time.perf_counter()
    results = []

    for case in CASES:
        exp = case.get("expected", {})

        # Map sample_transcripts format → ExpectedOutputs schema
        expected = ExpectedOutputs(
            icd10_must_include=exp.get("icd10_codes_must_include", []),
            medications_must_include=exp.get("medications_must_include", []),
            medications_must_NOT_include=exp.get("medications_must_NOT_include", []),
            plan_must_mention=exp.get("plan_must_mention", []),
            flags_must_include=exp.get("clinical_flags_must_include", []),
        )

        case_t0 = time.perf_counter()
        try:
            soap = await generate_soap_note(case["transcript"], vs)
        except Exception as e:
            soap = {}

        case_ms          = round((time.perf_counter() - case_t0) * 1000, 2)
        result           = evaluate_soap(soap, expected, case["id"])
        result.latency_ms = case_ms
        results.append(result)

    total_checks  = sum(r.passed + r.failed for r in results)
    total_passed  = sum(r.passed for r in results)
    overall_score = round(total_passed / total_checks, 3) if total_checks else 0.0
    cases_passed  = sum(1 for r in results if r.failed == 0)

    return BatchEvaluateResponse(
        request_id=str(uuid.uuid4()),
        total_cases=len(results),
        overall_score=overall_score,
        cases_passed=cases_passed,
        cases_failed=len(results) - cases_passed,
        per_case_results=results,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
