import os
import re
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, GoogleAPIError

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

# Gemini 1.5 models are retired (that's the 404). Override via GEMINI_MODEL in .env if needed.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ---------------------------------------------------------
# 1. Schemas & Tokenizer
# ---------------------------------------------------------
def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9_]+|%", text)


class TradingAssistantResponse(BaseModel):
    answer: str = Field(description="Direct, concise explanation of the trading rule or API spec.")
    exact_parameters: Dict[str, Union[float, str]] = Field(
        default_factory=dict,
        description="Extracted specs: numeric (tick intervals, margin rates, multipliers) or string (contract_type, basis, symbol codes)."
    )
    risk_warning: str = Field(description="Mandatory risk disclosure regarding the contract specs.")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=2, ge=1, le=10)


class QueryResponse(BaseModel):
    status: str
    sources: List[str]
    parameter_sources: Dict[str, List[str]]
    structured_output: Dict[str, Any]


# ---------------------------------------------------------
# 2. RAG Engine
# ---------------------------------------------------------
class DerivTradingAssistant:
    def __init__(self, db_path: str = "./deriv_rag_db", collection_name: str = "deriv_knowledge_base"):
        print("Loading local embedding and reranking models...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        print(f"Connecting to persistent Qdrant database at '{db_path}'...")
        # NOTE: local-mode Qdrant takes a file lock; don't run ingestion while the API is up.
        self.vector_db = QdrantClient(path=db_path)
        self.collection_name = collection_name

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        genai.configure(api_key=api_key)
        self.llm_model = genai.GenerativeModel(GEMINI_MODEL)

        self.documents: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None

        self._rebuild_sparse_index()

    def _rebuild_sparse_index(self):
        print("Scrolling Qdrant database to rebuild BM25 sparse index...")
        self.documents = []  # avoid duplicates if called twice
        next_page_offset = None

        while True:
            records, next_page_offset = self.vector_db.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False
            )
            for record in records:
                self.documents.append(record.payload)

            if next_page_offset is None:
                break

        if not self.documents:
            raise ValueError("No documents found in the database. Run the ingestion pipeline first.")

        tokenized_corpus = [tokenize(doc["text"]) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"Sparse index built successfully for {len(self.documents)} semantic chunks.")

    def retrieve_and_rerank(self, query: str, top_k: int = 3, candidate_k: int = 10, rrf_k: int = 60) -> List[Dict[str, Any]]:
        # A. Sparse retrieval (BM25) - skip zero-score docs so they don't pollute fusion
        bm25_scores = self.bm25.get_scores(tokenize(query))
        sparse_ranked_idx = [
            int(i) for i in np.argsort(bm25_scores)[::-1][:candidate_k] if bm25_scores[i] > 0
        ]

        # B. Dense retrieval (Qdrant)
        query_vector = self.embedding_model.encode(query).tolist()
        dense_results = self.vector_db.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=candidate_k
        ).points

        # C. Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        doc_by_id: Dict[str, Dict[str, Any]] = {}

        for rank, idx in enumerate(sparse_ranked_idx):
            doc = self.documents[idx]
            cid = doc["chunk_id"]
            doc_by_id[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, hit in enumerate(dense_results):
            cid = hit.payload["chunk_id"]
            doc_by_id[cid] = hit.payload
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        fused_ids = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
        candidate_docs = [doc_by_id[cid] for cid, _ in fused_ids]
        if not candidate_docs:
            return []

        # D. Cross-encoder reranking
        pairs = [[query, doc["text"]] for doc in candidate_docs]
        rerank_scores = self.reranker.predict(pairs)

        ranked = sorted(zip(rerank_scores, candidate_docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked][:top_k]

    # ---------------- Guardrail helpers ----------------
    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and strip thousands separators (1,000 -> 1000)."""
        return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text.lower())

    @staticmethod
    def _number_present(candidate: str, context_text: str) -> bool:
        # Not part of a longer number on either side (so "5" doesn't match "1.5" or "5.5").
        pattern = r"(?<![0-9.])" + re.escape(candidate) + r"(?![0-9]|\.[0-9])"
        return re.search(pattern, context_text) is not None

    @staticmethod
    def _as_number(text: str) -> Optional[float]:
        cleaned = text.strip().replace(",", "").replace("$", "").rstrip("%").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None

    @staticmethod
    def _number_forms(value: float) -> set:
        forms = {str(value), format(value, "g")}
        if float(value).is_integer():
            forms.add(str(int(value)))
        return forms

    def _value_supported(self, value: Union[float, str], normalized_text: str) -> bool:
        """True if the value (numeric, numeric-looking string, or plain string) appears in the text."""
        if isinstance(value, str):
            stripped = value.strip().lower()
            num = self._as_number(stripped)
            if num is None:
                return stripped in normalized_text
            value = num
        return any(self._number_present(c, normalized_text) for c in self._number_forms(value))

    def validate_guardrails(self, response: TradingAssistantResponse, context_chunks: List[str]) -> bool:
        context_text = self._normalize(" ".join(context_chunks))

        for key, value in response.exact_parameters.items():
            if not self._value_supported(value, context_text):
                print(f"[GUARDRAIL ALERT] Unverified parameter: {key} = {value!r}")
                return False

        for num in re.findall(r"\d+(?:\.\d+)?", self._normalize(response.answer)):
            if not self._value_supported(float(num), context_text):
                print(f"[GUARDRAIL ALERT] Unverified number in answer text: {num}")
                return False

        return True

    # ---------------- Generation ----------------
    def ask(self, query: str, top_k: int = 2) -> Dict[str, Any]:
        best_docs = self.retrieve_and_rerank(query, top_k=top_k)
        if not best_docs:
            return {"status": "error", "message": "No relevant context found for this query."}

        context_chunks = [doc["text"] for doc in best_docs]
        context_str = "\n\n".join(
            f"Source ({doc.get('source_type', 'unknown')} - {doc.get('source_name', 'unknown')}):\n{doc['text']}"
            for doc in best_docs
        )

        prompt = f"""You are a highly precise technical assistant for Deriv algorithmic trading.
Use ONLY the provided context to answer the user's query. If the context does not contain the answer,
say so in "answer" and leave "exact_parameters" empty. Never invent numbers.

Extract exact numerical or string specifications (tick intervals, margins, multipliers, contract types,
symbol codes, API parameters) into "exact_parameters", copying values exactly as written in the context.

Respond with ONLY a JSON object of this shape:
{{
  "answer": "<concise explanation>",
  "exact_parameters": {{"<parameter_name>": <number or string>}},
  "risk_warning": "<mandatory risk disclosure>"
}}

Context:
{context_str}

User Query: {query}
"""

        # response_schema is deliberately NOT used: Gemini's schema format rejects
        # Dict[str, Union[float, str]] (free-form objects). JSON mode + Pydantic validation instead.
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,
        )

        response_data: Optional[TradingAssistantResponse] = None
        last_error: Optional[Exception] = None

        for attempt in range(3):
            try:
                response = self.llm_model.generate_content(prompt, generation_config=generation_config)
                raw_json = json.loads(response.text)
                response_data = TradingAssistantResponse(**raw_json)
                break
            except (ResourceExhausted, ServiceUnavailable) as e:
                last_error = e
                time.sleep(2 ** attempt)
            except (json.JSONDecodeError, ValidationError, TypeError) as e:
                last_error = e  # malformed output; retry
            except GoogleAPIError as e:
                last_error = e
                break
            except Exception as e:  # e.g. blocked response -> response.text raises ValueError
                last_error = e
                break

        if response_data is None:
            return {"status": "error", "message": f"LLM Generation failed: {last_error}"}

        if not self.validate_guardrails(response_data, context_chunks):
            return {
                "status": "failed_guardrail",
                "error": "The generated response contained unverified numerical parameters.",
                "safe_fallback": "Please refer directly to the Deriv official documentation."
            }

        param_sources: Dict[str, List[str]] = {}
        for key, value in response_data.exact_parameters.items():
            param_sources[key] = [
                doc.get("source_name", "unknown")
                for doc in best_docs
                if self._value_supported(value, self._normalize(doc["text"]))
            ]

        return {
            "status": "success",
            "sources": [doc.get("source_name", "unknown") for doc in best_docs],
            "parameter_sources": param_sources,
            "structured_output": response_data.model_dump()
        }


# ---------------------------------------------------------
# 3. FastAPI Application Setup
# ---------------------------------------------------------
state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["assistant"] = DerivTradingAssistant(db_path="./deriv_rag_db")
        print("API is ready to accept requests.")
    except Exception as e:
        print(f"Failed to initialize DerivTradingAssistant: {e}")
        state["assistant"] = None

    yield

    state.clear()
    print("Shutting down API...")


app = FastAPI(
    title="Deriv RAG API",
    description=f"Hybrid retrieval trading assistant powered by {GEMINI_MODEL}",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    assistant: Optional[DerivTradingAssistant] = state.get("assistant")
    if not assistant:
        raise HTTPException(status_code=500, detail="RAG Engine is not initialized properly.")

    result = assistant.ask(req.query, top_k=req.top_k)

    if result["status"] == "failed_guardrail":
        raise HTTPException(status_code=422, detail=result)
    elif result["status"] == "error":
        raise HTTPException(status_code=502, detail=result)

    return result


if __name__ == "__main__":
    import uvicorn
    # Run the ingestion script first to create ./deriv_rag_db
    uvicorn.run(app, host="0.0.0.0", port=8000)
