"""
Simulated physician-patient transcripts for validation.
Each case includes: transcript, expected outputs, and clinical reasoning notes.
"""

CASES = [

    # ── Case 1: COVID-19, clear Paxlovid candidate ─────────────────────────
    {
        "id": "COVID_001",
        "label": "COVID-19 confirmed, high-risk, Paxlovid indicated",
        "transcript": """
Doctor: What brings you in today?
Patient: I've had a fever of 101.5, sore throat, and body aches for 2 days.
         Also really tired and have a mild cough.
Doctor: Any COVID exposure recently?
Patient: Yes, my coworker tested positive 4 days ago.
Doctor: Any chronic conditions I should know about?
Patient: I have Type 2 diabetes, controlled with metformin.
Doctor: We ran a rapid test. COVID-19 is positive, flu is negative.
         Given your diabetes and 2-day onset, you qualify for Paxlovid.
         I'll also check for drug interactions with metformin.
Patient: Should I be worried?
Doctor: Your oxygen looks fine at 98%. We'll monitor closely.
         Isolate for 5 days, rest, fluids, acetaminophen for fever.
         Come back immediately if you feel short of breath.
""",
        "expected": {
            "icd10_codes_must_include": ["U07.1"],       # COVID confirmed
            "icd10_codes_should_include": ["Z20.822"],   # exposure code
            "medications_must_include": ["Paxlovid"],
            "clinical_flags_should_include": ["diabetes"],
            "plan_must_mention": ["isolation", "5 days"],
        }
    },

    # ── Case 2: Influenza, antiviral within 48h window ─────────────────────
    {
        "id": "FLU_001",
        "label": "Influenza A confirmed, Tamiflu within 48h",
        "transcript": """
Doctor: Hello, what's going on today?
Patient: Started yesterday with a sudden high fever, 102 degrees,
         chills, terrible muscle aches, and a bad headache. 
         I can barely get out of bed.
Doctor: Any cough or sore throat?
Patient: Mild cough, no sore throat.
Doctor: Any COVID or flu exposure?
Patient: My kid had flu last week.
Doctor: Rapid flu test shows Influenza A positive, COVID negative.
         Since it's only been about 24 hours, Tamiflu will help.
         75mg twice a day for 5 days.
Patient: Anything else I should do?
Doctor: Rest, fluids, ibuprofen or Tylenol for the fever and aches.
         You should start feeling better by day 3.
         Come back if you develop difficulty breathing.
""",
        "expected": {
            "icd10_codes_must_include": ["J10"],          # Influenza A
            "medications_must_include": ["Tamiflu", "oseltamivir"],
            "plan_must_mention": ["5 days"],
            "follow_up_must_mention": ["breathing"],
        }
    },

    # ── Case 3: URI / Common Cold, antibiotics NOT appropriate ─────────────
    {
        "id": "URI_001",
        "label": "Viral URI, no antibiotics indicated",
        "transcript": """
Doctor: What seems to be the problem?
Patient: I've had a runny nose, mild sore throat, and sneezing for 3 days.
         No fever, just a bit tired.
Doctor: Any cough?
Patient: A little dry cough. Not bad.
Doctor: Let me take a look. Throat is mildly red, no exudate.
         No swollen lymph nodes. Lungs are clear.
         COVID and flu tests are both negative.
Patient: Can I get antibiotics? I want to get better faster.
Doctor: This looks like a viral cold. Antibiotics won't help here
         and can cause side effects. 
         I recommend rest, fluids, saline nasal rinse,
         and pseudoephedrine for the congestion if needed.
Patient: How long will this last?
Doctor: Usually 7 to 10 days. You're already on day 3, so halfway there.
""",
        "expected": {
            "icd10_codes_must_include": ["J00", "J06.9"],  # URI / cold
            "medications_must_NOT_include": ["amoxicillin", "antibiotic", "azithromycin"],
            "plan_must_mention": ["saline", "rest"],
        }
    },

    # ── Case 4: COVID-19, HIGH RISK — flags must be triggered ──────────────
    {
        "id": "COVID_002",
        "label": "COVID-19, elderly + low O2 — clinical flags critical",
        "transcript": """
Doctor: Mr. Chen, your family called saying you weren't feeling well?
Patient: Yes, I've been feverish for 3 days. Very tired. Hard to breathe.
Doctor: How old are you?
Patient: 72.
Doctor: Any medical history?
Patient: Heart failure. On lisinopril and furosemide.
Doctor: Let me check your oxygen. It's 91% on room air. 
         COVID test is positive.
         Given your age, heart condition, and low oxygen,
         I'm going to send you to the ED for further evaluation.
         This needs more monitoring than we can do here.
""",
        "expected": {
            "icd10_codes_must_include": ["U07.1"],
            "clinical_flags_must_include": ["SpO2", "91", "elderly", "heart"],
            "plan_must_mention": ["ED", "emergency", "hospital"],
        }
    },

]
