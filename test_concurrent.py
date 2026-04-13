import asyncio
import httpx
import time

BASE_URL = "http://localhost:8000"

# 6 个不同 transcript，绕过缓存，强制跑 pipeline
TRANSCRIPTS = [
    "Doctor: How long have you had fever? Patient: Two days, 38 degrees. Doctor: Any cough? Patient: Yes dry cough. Doctor: Likely COVID-19, prescribing Paxlovid.",
    "Doctor: What are your symptoms? Patient: Runny nose and sore throat for 3 days. Doctor: Sounds like URI, recommending rest and fluids.",
    "Doctor: Any fever? Patient: Yes 39 degrees for one day. Doctor: Flu test positive, prescribing Tamiflu 75mg twice daily.",
    "Doctor: How are you feeling? Patient: Cough and fatigue for 5 days, tested positive for COVID. Doctor: Prescribing Paxlovid and rest.",
    "Doctor: Any shortness of breath? Patient: Yes, SpO2 is 92%. Doctor: High risk, recommending hospital admission immediately.",
    "Doctor: Symptoms? Patient: Sore throat and mild fever since yesterday. Doctor: Likely viral URI, symptomatic treatment recommended.",
]

async def send_request(client: httpx.AsyncClient, i: int, transcript: str):
    t0 = time.perf_counter()
    resp = await client.post(
        f"{BASE_URL}/api/v1/notes/generate",
        json={"transcript": transcript},
        timeout=120,
    )
    ms = round((time.perf_counter() - t0) * 1000)
    print(f"Request {i+1}: status={resp.status_code}  time={ms}ms")

async def main():
    print(f"Sending {len(TRANSCRIPTS)} concurrent requests...\n")
    t0 = time.perf_counter()

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[
            send_request(client, i, t)
            for i, t in enumerate(TRANSCRIPTS)
        ])

    total = round((time.perf_counter() - t0) * 1000)
    print(f"\nAll done in {total}ms")

asyncio.run(main())
