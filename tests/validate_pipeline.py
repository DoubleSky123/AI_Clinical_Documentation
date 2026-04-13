"""
Validation script — runs all sample cases and checks output quality.

Usage:
    python -m tests.validate_pipeline

What it checks:
    - Required ICD-10 codes are present
    - Correct medications are mentioned
    - Forbidden medications are absent (e.g. antibiotics for viral URI)
    - Clinical flags are triggered for high-risk cases
    - All required SOAP fields are populated
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()

from app.vectorstore.build_index import load_vectorstore
from app.services.rag_pipeline import generate_soap_note
from tests.sample_transcripts import CASES


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    case_id: str
    label: str
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    soap: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.failed) == 0


# ── Individual checks ─────────────────────────────────────────────────────────

def _soap_text(soap: dict) -> str:
    """Flatten all SOAP fields into one lowercase string for easy search."""
    return json.dumps(soap).lower()


def run_checks(case: dict, soap: dict) -> CheckResult:
    result = CheckResult(case_id=case["id"], label=case["label"], soap=soap)
    exp = case.get("expected", {})
    text = _soap_text(soap)
    codes = [c.upper() for c in soap.get("icd10_codes", [])]
    meds  = text  # medications also checked via full-text

    # 1. ICD codes that MUST be present
    for code in exp.get("icd10_codes_must_include", []):
        # match prefix, e.g. "J10" matches "J10.1"
        if any(c.startswith(code.upper()) for c in codes):
            result.passed.append(f"ICD {code} ✅")
        else:
            result.failed.append(f"ICD {code} missing ❌  (got: {codes})")

    # 2. ICD codes that SHOULD be present (soft check — warning only)
    for code in exp.get("icd10_codes_should_include", []):
        if any(c.startswith(code.upper()) for c in codes):
            result.passed.append(f"ICD {code} (optional) ✅")
        else:
            result.passed.append(f"ICD {code} (optional) ⚠️  not included")

    # 3. Medications that MUST be mentioned
    for med in exp.get("medications_must_include", []):
        if med.lower() in meds:
            result.passed.append(f"Medication '{med}' ✅")
        else:
            result.failed.append(f"Medication '{med}' missing ❌")

    # 4. Medications that must NOT appear (e.g. antibiotics for viral URI)
    for med in exp.get("medications_must_NOT_include", []):
        if med.lower() in meds:
            result.failed.append(f"Forbidden medication '{med}' present ❌")
        else:
            result.passed.append(f"Forbidden medication '{med}' absent ✅")

    # 5. Plan keywords
    for kw in exp.get("plan_must_mention", []):
        if kw.lower() in text:
            result.passed.append(f"Plan mentions '{kw}' ✅")
        else:
            result.failed.append(f"Plan missing '{kw}' ❌")

    # 6. Follow-up keywords
    for kw in exp.get("follow_up_must_mention", []):
        if kw.lower() in text:
            result.passed.append(f"Follow-up mentions '{kw}' ✅")
        else:
            result.failed.append(f"Follow-up missing '{kw}' ❌")

    # 7. Clinical flags (critical for high-risk cases)
    flag_text = " ".join(soap.get("clinical_flags", [])).lower()
    for kw in exp.get("clinical_flags_must_include", []):
        if kw.lower() in flag_text or kw.lower() in text:
            result.passed.append(f"Clinical flag '{kw}' ✅")
        else:
            result.failed.append(f"Clinical flag '{kw}' missing ❌")

    for kw in exp.get("clinical_flags_should_include", []):
        if kw.lower() in flag_text or kw.lower() in text:
            result.passed.append(f"Clinical flag '{kw}' (optional) ✅")
        else:
            result.passed.append(f"Clinical flag '{kw}' (optional) ⚠️  not flagged")

    # 8. All SOAP fields populated
    required_fields = ["subjective", "objective", "assessment", "plan",
                       "icd10_codes", "medications", "follow_up"]
    for f_name in required_fields:
        val = soap.get(f_name)
        if val and (isinstance(val, str) and len(val) > 10 or isinstance(val, list) and len(val) > 0):
            result.passed.append(f"Field '{f_name}' populated ✅")
        else:
            result.failed.append(f"Field '{f_name}' empty or missing ❌")

    return result


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_all():
    print("\n" + "="*60)
    print("  Clinical Documentation AI — Validation Suite")
    print("="*60)

    vs = load_vectorstore()
    results: list[CheckResult] = []

    for case in CASES:
        print(f"\n▶ Running: [{case['id']}] {case['label']}")
        try:
            soap = await generate_soap_note(case["transcript"], vs)
            result = run_checks(case, soap)
        except Exception as e:
            print(f"  ERROR: {e}")
            result = CheckResult(case_id=case["id"], label=case["label"])
            result.failed.append(f"Pipeline error: {e}")

        results.append(result)

        # Print check results
        for msg in result.passed:
            print(f"  {msg}")
        for msg in result.failed:
            print(f"  {msg}")

        # Print generated SOAP note
        print(f"\n  Generated SOAP Note:")
        print(f"  S: {result.soap.get('subjective', 'N/A')[:120]}...")
        print(f"  A: {result.soap.get('assessment', 'N/A')[:120]}...")
        print(f"  ICD: {result.soap.get('icd10_codes', [])}")
        print(f"  Meds: {result.soap.get('medications', [])}")
        print(f"  Flags: {result.soap.get('clinical_flags', [])}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    total_passed = sum(len(r.passed) for r in results)
    total_failed = sum(len(r.failed) for r in results)

    for r in results:
        status = "✅ PASS" if r.ok else "❌ FAIL"
        print(f"  {status}  [{r.case_id}]  {r.label}")

    print(f"\n  Checks passed: {total_passed}")
    print(f"  Checks failed: {total_failed}")
    accuracy = total_passed / (total_passed + total_failed) * 100 if (total_passed + total_failed) > 0 else 0
    print(f"  Check accuracy: {accuracy:.1f}%")
    print("="*60 + "\n")

    return results


if __name__ == "__main__":
    asyncio.run(run_all())
