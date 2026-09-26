import os
import re
import time
import numpy as np
from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------
# 0. Tokenizer (must match the ingestion pipeline exactly)
# ---------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """
    FIX: previously this file used `doc["text"].lower().split()` while the
    ingestion pipeline could use a different tokenizer entirely — BM25 scores
    are only meaningful if the corpus was tokenized the same way at index
    time and query time. This must be kept identical to the tokenizer used
    in the ingestion script (deriv_rag_pipeline.py).
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_]+|%", text)
    return tokens


# ---------------------------------------------------------
# 1. Output Schema Definition (Pydantic)
# ---------------------------------------------------------
class TradingAssistantResponse(BaseModel):
    answer: str = Field(description="Direct, concise explanation of the trading rule or API spec.")
    # FIX: was Dict[str, float]. Real extracted specs aren't all numeric —
    # things like contract_type ('CALL'/'PUT'), basis ('payout'/'stake'), or
    # symbol codes ('R_100') are legitimate parameters too. Forcing float-only
    # either makes the LLM drop these fields or coerce them into meaningless
    # numbers to satisfy the schema. Union[float, str] lets both kinds through
    # so the guardrail below can actually check what was extracted.
    exact_parameters: Dict[str, Union[float, str]] = Field(
        default_factory=dict,
        description="Extracted specs: numeric (tick intervals, margin rates, multipliers) or string (contract_type, basis, symbol codes)."
    )
    risk_warning: str = Field(description="Mandatory risk disclosure regarding the contract specs.")


# ---------------------------------------------------------
# 2. RAG Execution & Guardrail Engine
# ---------------------------------------------------------
class DerivTradingAssistant:
    def __init__(self, db_path: str = "./deriv_rag_db", collection_name: str = "deriv_knowledge_base"):
        print("Loading local embedding and reranking models...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        print(f"Connecting to persistent Qdrant database at '{db_path}'...")
        self.vector_db = QdrantClient(path=db_path)
        self.collection_name = collection_name

        # FIX: fail fast and clearly if the API key is missing, instead of
        # letting it surface later as an opaque OpenAI error inside ask().
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it before instantiating DerivTradingAssistant."
            )
        self.llm_client = OpenAI(api_key=api_key)

        self.documents: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi = None

        self._rebuild_sparse_index()

    def _rebuild_sparse_index(self):
        """Scrolls through the persistent Qdrant DB to load text payloads and rebuild the BM25 index."""
        print("Scrolling Qdrant database to rebuild BM25 sparse index...")
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

        # FIX: use the shared tokenizer so BM25 term matching is consistent
        # with however the ingestion pipeline tokenized these same texts.
        tokenized_corpus = [tokenize(doc["text"]) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"Sparse index built successfully for {len(self.documents)} semantic chunks.")

    def retrieve_and_rerank(self, query: str, top_k: int = 3, candidate_k: int = 10, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining dense vector search and sparse keyword match,
        fused via Reciprocal Rank Fusion, then reranked with a cross-encoder.

        FIX (RRF implemented but unused): the ingestion pipeline defines RRF
        fusion but the actual query path in this class used a cruder
        "take top 5 from each, dedup by chunk_id" merge with no fusion
        scoring at all — so RRF only ever ran in the pipeline's own demo,
        never in the code that actually serves queries. This now applies the
        same RRF approach here, so a chunk that ranks well in *both*
        retrievers is prioritized into the reranker's candidate pool over one
        that only barely made either list.
        """
        # A. Sparse Retrieval (BM25) — full ranked order, not just top 5
        bm25_scores = self.bm25.get_scores(tokenize(query))
        sparse_ranked_idx = np.argsort(bm25_scores)[::-1][:candidate_k]

        # B. Dense Retrieval (Qdrant)
        # FIX: `.search()` is deprecated in recent qdrant-client versions;
        # use `.query_points()` and pull `.points` off the result.
        query_vector = self.embedding_model.encode(query).tolist()
        dense_results = self.vector_db.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=candidate_k
        ).points

        # C. Reciprocal Rank Fusion across the two ranked lists
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

        # D. Cross-Encoder Reranking of the fused candidate pool
        pairs = [[query, doc["text"]] for doc in candidate_docs]
        rerank_scores = self.reranker.predict(pairs)

        ranked_pairs = sorted(zip(rerank_scores, candidate_docs), key=lambda x: x[0], reverse=True)
        final_docs = [doc for score, doc in ranked_pairs][:top_k]

        return final_docs

    @staticmethod
    def _number_present(candidate_text: str, context_text: str) -> bool:
        """Word-boundary check that a numeric value appears as an actual
        standalone number in context_text (optionally with a trailing '%')."""
        pattern = r"(?<![0-9.])" + re.escape(candidate_text) + r"%?(?![0-9])"
        return re.search(pattern, context_text) is not None

    def validate_guardrails(self, response: TradingAssistantResponse, context_chunks: List[str]) -> bool:
        """
        DETERMINISTIC GUARDRAIL: Validates against hallucinated specs.

        FIX 1 (substring matching): the original check used plain substring
        matching (`str(value) not in context_text`), so a hallucinated "10"
        would pass simply because context containing "100" or "1000" also
        contains the substring "10". Numeric values are now matched on word
        boundaries, so a value only passes if it appears as an actual
        standalone number in the context (optionally followed by '%').

        FIX 2 (guardrail doesn't inspect answer numbers): the original only
        checked `exact_parameters` — but `answer` is free text the LLM wrote
        itself, and could easily state a number nowhere in
        exact_parameters (e.g. "...which is about 25% higher than usual").
        That number was never checked against anything. This now also scans
        `answer` for numeric literals and validates each one the same way.
        A small stopword-style allowlist avoids flagging incidental numbers
        that aren't really "specs" (e.g. "step 1", "either of 2 options").
        """
        context_text = " ".join(context_chunks).lower()

        # --- exact_parameters: numeric values checked against context;
        # string values checked as case-insensitive substrings.
        for key, value in response.exact_parameters.items():
            if isinstance(value, str):
                if value.lower() not in context_text:
                    print(f"[GUARDRAIL ALERT] Hallucinated parameter detected: {key} = {value!r}")
                    return False
                continue

            candidates = {str(value)}
            if float(value).is_integer():
                candidates.add(str(int(value)))

            if not any(self._number_present(c, context_text) for c in candidates):
                print(f"[GUARDRAIL ALERT] Hallucinated parameter detected: {key} = {value}")
                return False

        # --- answer text: any numeric literal mentioned in prose must also
        # be traceable to the retrieved context.
        answer_numbers = re.findall(r"\d+(?:\.\d+)?", response.answer)
        for num in answer_numbers:
            candidates = {num}
            if "." not in num:
                pass
            elif num.endswith(".0"):
                candidates.add(num[:-2])
            if not any(self._number_present(c, context_text) for c in candidates):
                print(f"[GUARDRAIL ALERT] Unverified number in answer text: {num}")
                return False

        return True

    def ask(self, query: str) -> Dict[str, Any]:
        """Executes the full pipeline: Retrieval -> Generation -> Guardrail validation."""
        print(f"\n--- Processing Query: '{query}' ---")

        # 1. Retrieve & Rerank
        best_docs = self.retrieve_and_rerank(query, top_k=2)
        context_chunks = [doc["text"] for doc in best_docs]

        context_str = "\n\n".join([f"Source ({doc['source_type']} - {doc['source_name']}):\n{doc['text']}"
                                   for doc in best_docs])

        # 2. LLM Generation
        prompt = f"""
        You are a highly precise technical assistant for Deriv algorithmic trading.
        Use ONLY the provided context to answer the user's query. Extract any exact numerical
        specifications (like tick intervals, margins, or API parameters) into the exact_parameters dictionary.

        Context:
        {context_str}

        User Query: {query}
        """

        print("Generating structured response...")
        # FIX: added a small retry loop with backoff for transient
        # connection/rate-limit errors, instead of failing on the first hiccup.
        response_data = None
        last_error = None
        for attempt in range(3):
            try:
                completion = self.llm_client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format=TradingAssistantResponse,
                    temperature=0.0
                )
                response_data = completion.choices[0].message.parsed
                break
            except (RateLimitError, APIConnectionError) as e:
                last_error = e
                wait = 2 ** attempt
                print(f"  Transient error ({e}), retrying in {wait}s...")
                time.sleep(wait)
            except APIError as e:
                last_error = e
                break  # non-transient API error, don't retry
            except Exception as e:
                last_error = e
                break

        if response_data is None:
            return {"status": "error", "message": f"LLM Generation failed: {str(last_error)}"}

        # 3. Guardrail Execution
        passed_guardrail = self.validate_guardrails(response_data, context_chunks)

        if not passed_guardrail:
            return {
                "status": "failed_guardrail",
                "error": "The generated response contained unverified numerical parameters.",
                "safe_fallback": "Please refer directly to the Deriv official documentation."
            }

        # FIX (no claim-level source attribution): previously only an
        # undifferentiated list of every source consulted was returned,
        # with no way to tell which specific chunk backs which extracted
        # parameter. This does a best-effort per-parameter lookup: for each
        # extracted value, find which of the retrieved docs actually
        # contains it. Best-effort because a value could legitimately appear
        # in more than one chunk, or (rarely) be a paraphrase the guardrail's
        # substring/number check still accepted.
        param_sources: Dict[str, List[str]] = {}
        for key, value in response_data.exact_parameters.items():
            matches = []
            for doc in best_docs:
                text_lower = doc["text"].lower()
                if isinstance(value, str):
                    hit = value.lower() in text_lower
                else:
                    candidates = {str(value)}
                    if float(value).is_integer():
                        candidates.add(str(int(value)))
                    hit = any(self._number_present(c, text_lower) for c in candidates)
                if hit:
                    matches.append(doc["source_name"])
            param_sources[key] = matches

        return {
            "status": "success",
            "sources": [doc["source_name"] for doc in best_docs],
            "parameter_sources": param_sources,
            "structured_output": response_data.model_dump()
        }

# ---------------------------------------------------------
# 3. Execution Example
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        # Connects to the database created by your ingestion script
        assistant = DerivTradingAssistant(db_path="./deriv_rag_db")

        test_queries = [
            "What happens when margin hits 50% on CFDs?",
            "What is the required API payload for a proposal, and what is the Boom 1000 index tick frequency?"
        ]

        for q in test_queries:
            result = assistant.ask(q)

            if result["status"] == "success":
                print("\n[Final Output Validated]")
                print(f"Answer: {result['structured_output']['answer']}")
                print(f"Extracted Params: {result['structured_output']['exact_parameters']}")
                print(f"Risk Warning: {result['structured_output']['risk_warning']}")
                print(f"Sources Used: {result['sources']}\n")
            else:
                print(f"\n[Request Failed] {result}\n")

    except Exception as e:
        print(f"Pipeline Error: {e}")
