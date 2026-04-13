# AI Clinical Documentation Assistant

RAG-powered, multi-agent SOAP note generation for primary care and general dentistry encounters.  
Runs fully **offline** — no OpenAI API key required.

---

## Demo

**Chat interface** — paste a transcript or record live audio, get a structured SOAP note instantly.

```
Input:  Physician-patient transcript  (text or audio)
Output: Structured SOAP Note (S / O / A / P + ICD-10 codes + medications + clinical flags)
```

Automatically responds in the **same language as the transcript** — Chinese input produces a Chinese note, English input produces an English note.

---

## Architecture

```
Audio / Text Input
        │
        ▼
┌───────────────────┐
│   FastAPI (REST)  │  POST /api/v1/notes/generate
│   Voice pipeline  │  POST /api/v1/voice/transcribe-and-document
└────────┬──────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                  LangGraph StateGraph                    │
│                                                         │
│  ┌─────────────┐     ┌──────────────────┐               │
│  │  Retrieval  │────▶│   Extraction     │               │
│  │    Node     │     │     Agent        │               │
│  │ (FAISS MMR) │     │  (RAG-grounded   │               │
│  └─────────────┘     │   SOAP draft)    │               │
│                      └────────┬─────────┘               │
│              ┌────────────────┼────────────────┐        │
│              ▼                ▼                ▼        │
│     ┌──────────────┐ ┌──────────────┐ ┌─────────────┐  │
│     │  ICD Coding  │ │  Medication  │ │  Clinical   │  │
│     │    Agent     │ │    Agent     │ │  Flags      │  │
│     │              │ │              │ │  Agent      │  │
│     └──────┬───────┘ └──────┬───────┘ └──────┬──────┘  │
│            └────────────────┼─────────────────┘        │
│                             ▼                           │
│                    ┌─────────────────┐                  │
│                    │   Supervisor    │                  │
│                    │     Node        │                  │
│                    │  (merge + QA)   │                  │
│                    └────────┬────────┘                  │
└─────────────────────────────┼───────────────────────────┘
                              ▼
               Structured SOAP Note (JSON)
                              │
                              ▼
              ┌───────────────────────────┐
              │    React Chat UI          │
              │  Chat bubbles · SOAP card │
              │  Live mic · File upload   │
              └───────────────────────────┘
```

### Agent Responsibilities

| Agent | Role |
|-------|------|
| **Retrieval Node** | MMR retrieval over FAISS — fetches diverse CDC/WHO guideline chunks (k=6, fetch_k=20) |
| **Extraction Agent** | Generates initial SOAP note grounded in retrieved clinical context |
| **ICD Coding Agent** | Audits and corrects ICD-10-CM codes; adds Z-exposure codes when applicable |
| **Medication Agent** | Validates drug name, dose, route, frequency, and duration against transcript |
| **Clinical Flags Agent** | Identifies high-risk indicators — SpO2 <94%, elderly, immunocompromised, comorbidities |
| **Supervisor Node** | Merges parallel agent outputs into a single validated SOAP note |

---

## Tech Stack

### Backend

| Layer | Technology |
|-------|------------|
| **Agent Framework** | LangGraph 1.1 — StateGraph with parallel fan-out |
| **LLM** | Qwen 2.5 7B via Ollama (local, no API cost) |
| **Embeddings** | BAAI/bge-small-en-v1.5 (HuggingFace, local) |
| **Vector Store** | FAISS with MMR retrieval |
| **Structured Output** | Pydantic v2 + `with_structured_output` |
| **API** | FastAPI with async lifespan vectorstore management |
| **ASR** | OpenAI Whisper (local, `base` model) |
| **Knowledge Base** | CDC/WHO respiratory infection guidelines (.txt) |
| **Cache** | Redis — SOAP note cache (TTL 1h), guidelines cache (TTL 24h), graceful degradation |
| **Concurrency** | asyncio Semaphore — max 3 concurrent LLM pipelines, double-checked locking |
| **Monitoring** | Prometheus metrics (`/metrics`) + Grafana dashboard |
| **MCP Server** | 3 tools: `search_clinical_guidelines`, `generate_soap_note`, `evaluate_soap_quality` |
| **Infrastructure** | Docker Compose — Redis, Prometheus, Grafana |

### Frontend

| Layer | Technology |
|-------|------------|
| **Framework** | React 18 |
| **UI Pattern** | Chat interface — message bubbles with SOAP card rendering |
| **Audio** | MediaRecorder API (WebM) — live mic recording + file upload |
| **Components** | `App` · `ChatMessage` · `SOAPDisplay` · `Recorder` |
| **Proxy** | CRA dev proxy → `http://localhost:8000` |

---

## Project Structure

```
aiclinical/
├── app/
│   ├── main.py                          # FastAPI app + lifespan
│   ├── models/schemas.py                # Pydantic schemas (SOAPNote, SOAPResponse)
│   ├── routers/
│   │   ├── notes.py                     # POST /api/v1/notes/generate
│   │   ├── voice.py                     # POST /api/v1/voice/transcribe-and-document
│   │   └── evaluate.py                  # POST /api/v1/evaluate/single|batch
│   ├── services/
│   │   ├── rag_pipeline.py              # Entry point — delegates to LangGraph
│   │   └── graph/
│   │       ├── state.py                 # ClinicalState TypedDict
│   │       ├── nodes.py                 # Six agent node functions
│   │       └── pipeline.py             # StateGraph assembly + compile
│   └── vectorstore/
│       └── build_index.py              # FAISS build + incremental update
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.js                       # Main chat app — state, routing, API calls
│   │   ├── App.css                      # Chat UI styles
│   │   └── components/
│   │       ├── ChatMessage.js           # Message bubble (doctor / AI)
│   │       ├── SOAPDisplay.js           # Structured SOAP note card + stats
│   │       └── Recorder.js             # Mic recording state machine
│   └── package.json
├── data/medical_kb/                     # CDC/WHO guideline .txt files
├── vectorstore/faiss_index/             # Persisted FAISS index (auto-generated)
├── tests/
│   ├── test_pipeline.py
│   ├── validate_pipeline.py
│   └── sample_transcripts.py
├── docker-compose.yml                   # Redis + Prometheus + Grafana
├── prometheus.yml                       # Prometheus scrape config
├── Dockerfile                           # API container (for full deployment)
├── test_concurrent.py                   # Concurrency stress test
└── requirements.txt
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com) installed and running
- [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) installed and on PATH (required for Whisper audio decoding)

### 1. Install backend dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Pull the LLM

```bash
ollama pull qwen2.5:7b
```

### 3. Build the FAISS vector index

```bash
python -m app.vectorstore.build_index
```

This downloads `BAAI/bge-small-en-v1.5` (~130 MB) on first run.

### 4. Start infrastructure (Redis + Prometheus + Grafana)

```bash
docker compose up redis prometheus grafana -d
```

### 5. Start the backend

```bash
uvicorn app.main:app --reload --port 8000
```

### 6. Start the frontend

```bash
cd frontend
npm install
npm start
```

The React app opens at `http://localhost:3000` and proxies API calls to port 8000.

### 7. Generate a SOAP note (API)

```bash
curl -X POST http://localhost:8000/api/v1/notes/generate \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Doctor: What brings you in? Patient: Fever 101.5, sore throat, body aches for 2 days. Coworker tested COVID positive 4 days ago. Doctor: Rapid COVID test: POSITIVE. Starting Paxlovid.",
    "patient_id": "PT-001"
  }'
```

Interactive API docs: `http://localhost:8000/docs`

---

## Frontend Features

| Feature | Description |
|---------|-------------|
| **Chat interface** | Conversation-style layout — doctor messages on the right, AI notes on the left |
| **SOAP card** | Colour-coded S/O/A/P sections, ICD-10 tags, medication tags, clinical flag badges |
| **Live recording** | Click mic button to record encounter audio directly in browser (WebM) |
| **File upload** | Upload pre-recorded audio files (.wav / .mp3 / .m4a / .webm / .mp4 / .ogg) |
| **Latency stats** | Shows ASR, LLM, and total processing time on each response |
| **Raw transcript** | Collapsible panel shows Whisper transcript for audio inputs |
| **Sample transcript** | "Load sample" button pre-fills a COVID-19 encounter for quick testing |
| **Multilingual** | SOAP note language matches the transcript — Chinese in, Chinese out |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/notes/generate` | Transcript → SOAP Note |
| `POST` | `/api/v1/voice/transcribe-and-document` | Audio file → Transcript → SOAP Note |
| `POST` | `/api/v1/voice/transcribe-only` | Audio file → Transcript |
| `POST` | `/api/v1/evaluate/single` | Score one note against expected outputs |
| `POST` | `/api/v1/evaluate/batch` | Run all test cases, return aggregate metrics |
| `GET`  | `/api/v1/notes/health` | Health check |

### Sample Response

```json
{
  "request_id": "8e8c528d-...",
  "patient_id": "PT-001",
  "soap_note": {
    "subjective": "Fever 101.5°F, sore throat, body aches x2 days. Coworker COVID+ 4 days ago.",
    "objective": "Rapid antigen test: Positive for SARS-CoV-2.",
    "assessment": "Confirmed COVID-19. Meets criteria for antiviral therapy — within 5 days of symptom onset.",
    "plan": "Initiate Paxlovid (nirmatrelvir 300mg + ritonavir 100mg BID x5 days). Supportive care, isolation x5 days.",
    "icd10_codes": ["U07.1", "Z20.822"],
    "medications": ["Nirmatrelvir/ritonavir (Paxlovid) 300mg/100mg BID x5 days"],
    "follow_up": "Return in 24-48h if SpO2 <94% or symptoms worsen.",
    "clinical_flags": []
  },
  "latency_ms": 28243.0,
  "model_version": "qwen2.5:7b + FAISS-mmr"
}
```

---

## Evaluation

The `/evaluate` endpoints score generated notes against expected outputs:

```bash
curl -X POST http://localhost:8000/api/v1/evaluate/single \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "...",
    "expected": {
      "icd10_must_include": ["U07.1"],
      "medications_must_include": ["Paxlovid"],
      "medications_must_NOT_include": ["amoxicillin"],
      "plan_must_mention": ["isolat"]
    }
  }'
```

Returns a `score` (0.0–1.0) with per-check pass/fail details — useful for regression testing after prompt or model changes.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Design Decisions

**Why LangGraph over a single chain?**  
Each agent is scoped to one task (ICD coding, medication validation, risk flagging). This makes prompt tuning, debugging, and model swapping independent — changing the ICD agent doesn't affect medication validation.

**Why local models (Ollama) over OpenAI?**  
Clinical transcripts contain PHI. Running inference locally eliminates data-sharing concerns without requiring de-identification preprocessing before every API call.

**Why FAISS + MMR retrieval?**  
MMR (Maximal Marginal Relevance) balances relevance and diversity — with `fetch_k=20` candidates re-ranked to `k=6`, the pipeline avoids returning redundant guideline chunks that repeat the same clinical advice.

**Why `with_structured_output` over `JsonOutputParser`?**  
Local models don't reliably follow "return only JSON" instructions. `with_structured_output` enforces the Pydantic schema at the framework level, eliminating parsing failures. This reduced pipeline latency from ~65s to ~28s by removing retry overhead.

**Why a chat UI over a form-based UI?**  
Clinical encounters are conversational by nature. A chat layout lets doctors review the history of a session and run multiple encounters in sequence without page reloads.

**Why Redis with graceful degradation?**  
Same transcript submitted twice (network retry, batch evaluation repeated cases) would otherwise re-run the full 10s pipeline. Redis caches results by `md5(transcript)` with a 1-hour TTL. All Redis operations are wrapped in `try/except` — if Redis is unavailable the app falls through to the pipeline transparently, keeping availability at 100%.

**Why Semaphore for concurrency control?**  
Ollama runs one model inference at a time. Without a limit, 10 concurrent requests would queue internally and all time out. A `Semaphore(3)` caps simultaneous pipeline runs, keeping response times predictable. Double-checked locking inside the semaphore prevents redundant pipeline runs when multiple identical requests arrive at the same time.

**Why MCP alongside FastAPI?**  
FastAPI serves human users via HTTP. MCP exposes the same capabilities as tools for AI agents (Claude Desktop, multi-agent systems). The two interfaces coexist — adding MCP didn't modify a single line of the existing API.
