# Deriv RAG Assistant

A hybrid-retrieval RAG system that answers technical questions about **Deriv synthetic indices, CFD margin rules, and WebSocket API payloads**, with a built-in **numeric guardrail** that rejects any LLM answer containing numbers not found in the retrieved context.

Backend: FastAPI + Qdrant + BM25 + cross-encoder reranking + Gemini. Frontend: React (Vite) + Tailwind.

> **Disclaimer:** This is an educational project. Parts of the knowledge base are hand-typed and explicitly flagged as unverified (see [Known Limitations](#known-limitations)). Nothing here is financial advice. Always confirm figures against the official Deriv documentation before trading.

---

## Features

- **Hybrid retrieval**: dense vector search (Qdrant, `all-MiniLM-L6-v2`) fused with sparse keyword search (BM25) using **Reciprocal Rank Fusion (RRF)**.
- **Cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`) over fused candidates for final top-k selection.
- **Custom tokenizer** that keeps `stop_loss`, `R_100`, `take_profit` intact and splits `50%` into `50` and `%`, so parameter names and percentages match properly in BM25.
- **Structured LLM output**: Gemini runs in JSON mode and the response is validated with Pydantic (`answer`, `exact_parameters`, `risk_warning`), with retries on malformed output and exponential backoff on rate limits.
- **Hallucination guardrail**: every value in `exact_parameters`, and every number in the answer text, must appear in the retrieved chunks (thousands separators normalised, no partial matches such as `5` inside `1.5`). Otherwise the API returns HTTP 422 with a safe fallback message.
- **Per-parameter source attribution**: the UI shows which retrieved document supports each extracted spec.
- **Multi-source ingestion**:
  - Live Deriv API JSON schemas (`active_symbols`, `proposal`, `buy`, `ticks_history`, `statement`, `contracts_for`)
  - Synthetic index spec tables (R_100, CRASH_500, BOOM_1000) stored as entity blocks
  - Header-aware Markdown chunking (`#`, `##`, `###`) for policy docs such as margin call and stop-out rules
- **Safe degradation on ingest**: per-endpoint retry, verified offline fallbacks where available, and an explicit "schema unavailable, check the docs" placeholder chunk instead of a silent gap.

---

## Architecture

```
                    ┌──────────────────────── Ingestion (offline) ────────────────────────┐
                    │  Deriv API schemas  +  Synthetic index specs  +  Markdown policies  │
                    │              │  chunk + metadata  │                                 │
                    │              ▼                    ▼                                 │
                    │     MiniLM embeddings  ───►  Qdrant (local, on disk)                │
                    └──────────────────────────────────────────────────────────────────────┘

 React UI ──POST /api/v1/ask──► FastAPI
                                  │
                                  ├─ BM25 (rebuilt from Qdrant payloads at startup)
                                  ├─ Dense search (Qdrant)
                                  ├─ Reciprocal Rank Fusion
                                  ├─ Cross-encoder rerank  ──► top_k chunks
                                  ├─ Gemini (JSON mode)    ──► Pydantic validation
                                  ├─ Numeric guardrail     ──► 422 if unverified numbers
                                  └─ Response + parameter sources
```

---

## Project Structure

```
.
├── deriv_rag_pipeline.py        # Ingestion: fetch, chunk, embed, write to Qdrant
├── deriv_trading_assistant.py   # FastAPI app: retrieval, reranking, LLM, guardrails
├── frontend/
│   └── src/
│       ├── App.jsx              # Search UI, answer, specs, risk warning, sources
│       └── App.css
├── .env                         # GEMINI_API_KEY (not committed)
└── README.md
```

Adjust paths to match your repository layout.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) |
| Vector DB | Qdrant (local persistent mode) |
| Sparse search | `rank_bm25` (BM25Okapi) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Google Gemini via `google-generativeai` (default `gemini-2.5-flash`) |
| API | FastAPI, Pydantic, Uvicorn |
| Frontend | React, Vite, Tailwind CSS, `lucide-react` |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A [Gemini API key](https://aistudio.google.com/app/apikey)

### 1. Backend setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install fastapi uvicorn pydantic python-dotenv numpy requests beautifulsoup4 \
            qdrant-client sentence-transformers rank-bm25 google-generativeai
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
# Optional: override the default model
# GEMINI_MODEL=gemini-2.5-flash
```

### 2. Ingest the knowledge base

```bash
python deriv_rag_pipeline.py
```

This creates the on-disk Qdrant store at `./deriv_rag_db` and runs a sample hybrid-retrieval query to verify it.

> Local-mode Qdrant takes a file lock. **Stop the API before re-running ingestion.**

### 3. Run the API

```bash
python deriv_trading_assistant.py
```

The API starts on `http://localhost:8000` (interactive docs at `/docs`).

### 4. Run the frontend

```bash
cd frontend
npm install
npm install lucide-react
npm run dev
```

Open `http://localhost:5173`. CORS is configured for the default Vite origin.

> `App.jsx` uses Tailwind utility classes, so make sure Tailwind is set up in the Vite project.

---

## API Reference

### `POST /api/v1/ask`

**Request**

```json
{
  "query": "What happens when margin hits 50% on CFDs?",
  "top_k": 2
}
```

`top_k` is an integer from 1 to 10 (default 2).

**Success response (200)**

```json
{
  "status": "success",
  "sources": ["Margin Calls and Stop-Out Mechanism on CFDs", "..."],
  "parameter_sources": { "stop_out_level": ["Margin Calls and Stop-Out Mechanism on CFDs"] },
  "structured_output": {
    "answer": "...",
    "exact_parameters": { "stop_out_level": "50%" },
    "risk_warning": "..."
  }
}
```

**Error responses**

| Code | Meaning |
|---|---|
| 422 | Guardrail failure: the generated answer contained numbers not present in the retrieved context. |
| 502 | No relevant context found, or LLM generation failed after retries. |
| 500 | RAG engine failed to initialise (for example, the database is empty or the API key is missing). |

---

## How the Guardrail Works

1. Normalise the retrieved context (lowercase, strip thousands separators).
2. For each entry in `exact_parameters`, check that the value (numeric or string) appears in the context as a standalone token. `5` does not match `1.5`.
3. Scan the answer text for numbers and verify each one the same way.
4. If anything is unsupported, discard the answer and return HTTP 422 with a pointer to the official docs.

The frontend surfaces this as a "Query Failed" card with a safe fallback.

---

## Known Limitations

- **Hand-typed spec data.** The synthetic index table (leverage caps, tick frequency, spike/drop probabilities) and the Markdown policy text in `run_ingestion()` are hardcoded and not pulled from an official source. Spec chunks carry an `[UNVERIFIED / ILLUSTRATIVE DATA]` disclaimer in both text and metadata. Replace them with a verified source before relying on them.
- **Limited offline fallbacks.** Only `proposal` and `buy` have hardcoded fallback schemas. Other endpoints that fail to fetch get an explicit "unavailable" placeholder chunk.
- **Guardrail scope.** The check verifies that numbers appear in the retrieved context. It does not verify that the context itself is correct, or that the number is used in the right place.
- **Small corpus.** The knowledge base is a demo-scale set of chunks, so retrieval quality on out-of-scope questions will be limited.
- **Local Qdrant** is single-process. For multi-worker or production use, run a Qdrant server.

---

## Roadmap

- [ ] Replace hardcoded specs with scraped or official Deriv documentation
- [ ] Evaluation set with retrieval metrics (hit rate, MRR) and answer-faithfulness checks
- [ ] Streaming responses in the UI
- [ ] Dockerised setup (API, Qdrant server, frontend)
- [ ] Expand API schema coverage and add offline fallbacks for more endpoints

---

## Author

Built by **Anurag Parasar Mund**.

## License

Add a license of your choice (for example MIT) and update this section.
