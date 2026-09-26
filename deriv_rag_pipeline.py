import os
import re
import json
import requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# 1. Chunk Data Model with Enriched Metadata
# ---------------------------------------------------------------------------
class IngestedChunk(BaseModel):
    chunk_id: str
    source_type: str  # "api_schema", "spec_table", or "markdown_doc"
    source_name: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 2. Specialized Ingestion Loaders for Deriv Data
# ---------------------------------------------------------------------------
class DerivDataIngestionEngine:
    def __init__(self):
        pass

    def fetch_live_deriv_api_schemas(self) -> List[IngestedChunk]:
        """
        Pulls real public JSON schemas from Deriv's API repo (or official schema source).
        Deconstructs complex request/response schemas into individual searchable API calls.

        FIX: previously, the hardcoded fallback only kicked in if ALL calls failed
        (`if not chunks`). Now each individual endpoint that fails to fetch gets its
        own fallback entry (where we have one) or is explicitly logged as missing,
        so a partial GitHub outage doesn't silently produce a half-empty index.
        """
        chunks = []

        api_calls = [
            "active_symbols",
            "proposal",
            "buy",
            "ticks_history",
            "statement",
            "contracts_for",
        ]

        base_url = "https://raw.githubusercontent.com/binary-com/deriv-api-docs/master/config/v3"

        # Fallback content for endpoints we can hardcode if the network fetch fails.
        # Only "proposal" is filled in below as before; extend this dict to cover
        # more endpoints if offline ingestion needs to be more complete.
        # NOTE (missing API fallback coverage): only endpoints with a schema
        # we've actually verified are listed here. It would be worse to guess
        # at required fields for a trading API than to leave a gap — a
        # confidently wrong "required" list is more dangerous than an honest
        # "unavailable, check docs" chunk (added below for whatever isn't
        # covered here).
        fallback_schemas = {
            "proposal": (
                "Deriv WebSocket API Call: `proposal`\n"
                "Summary: Gets latest price and payout for a specific contract type.\n"
                "Parameters:\n"
                "  * amount (number) [REQUIRED]: Proposed stake or payout.\n"
                "  * basis (string) [REQUIRED]: 'payout' or 'stake'.\n"
                "  * contract_type (string) [REQUIRED]: e.g., 'CALL', 'PUT', 'MULTUP', 'MULTDOWN'.\n"
                "  * currency (string) [REQUIRED]: Currency code (e.g. USD, EUR).\n"
                "  * symbol (string) [REQUIRED]: Symbol code (e.g. R_100, 1HZ100V, BOOM1000)."
            ),
            "buy": (
                "Deriv WebSocket API Call: `buy`\n"
                "Summary: Buys a contract, either by proposal id or inline parameters.\n"
                "Parameters:\n"
                "  * buy (string) [REQUIRED]: The proposal id from a `proposal` call, "
                "or '1' if contract parameters are passed directly in `parameters`.\n"
                "  * price (string) [REQUIRED]: Maximum price at which to purchase the contract.\n"
                "  * parameters (object) [OPTIONAL]: Inline contract parameters, used when "
                "`buy` is set to '1' instead of a proposal id.\n"
                "  * loginid (string) [OPTIONAL]: Required only when multiple tokens were "
                "provided during authorize.\n"
                "  * subscribe (integer) [OPTIONAL]: Set to 1 to stream updates."
            ),
        }

        print("Fetching live Deriv API schemas...")
        failed_calls = []
        for call in api_calls:
            url = f"{base_url}/{call}/send.json"
            fetched = False
            for attempt in range(2):  # simple retry once on failure
                try:
                    res = requests.get(url, timeout=8)
                    if res.status_code == 200:
                        schema = res.json()

                        title = schema.get("title", call)
                        desc = schema.get("description", "No description")
                        properties = schema.get("properties", {})
                        required_fields = schema.get("required", [])

                        param_details = []
                        for param, details in properties.items():
                            p_type = details.get("type", "any")
                            p_desc = details.get("description", "").strip()
                            req_str = "[REQUIRED]" if param in required_fields else "[OPTIONAL]"
                            param_details.append(f"  * {param} ({p_type}) {req_str}: {p_desc}")

                        doc_text = (
                            f"Deriv WebSocket API Call: `{call}`\n"
                            f"Action: {title}\n"
                            f"Summary: {desc}\n"
                            f"Parameters:\n" + "\n".join(param_details)
                        )

                        chunks.append(IngestedChunk(
                            chunk_id=f"api_{call}",
                            source_type="api_schema",
                            source_name=f"{call}_schema",
                            content=doc_text,
                            metadata={"endpoint": call, "required_params": required_fields}
                        ))
                        fetched = True
                        break
                    else:
                        print(f"  {call}: HTTP {res.status_code}, retrying..." if attempt == 0 else f"  {call}: HTTP {res.status_code}, giving up")
                except requests.RequestException as e:
                    print(f"  {call}: request error ({e}), {'retrying...' if attempt == 0 else 'giving up'}")

            if not fetched:
                failed_calls.append(call)
                if call in fallback_schemas:
                    print(f"  Using hardcoded fallback for '{call}'")
                    chunks.append(IngestedChunk(
                        chunk_id=f"api_{call}_fallback",
                        source_type="api_schema",
                        source_name=f"{call}_spec",
                        content=fallback_schemas[call],
                        metadata={"endpoint": call, "source": "fallback"}
                    ))

        if failed_calls:
            missing = [c for c in failed_calls if c not in fallback_schemas]
            if missing:
                print(f"WARNING: no verified data ingested for endpoints: {missing}")
                # FIX: previously this was only a console warning — a gap in
                # the knowledge base with no trace inside it. Ingest an
                # explicit placeholder chunk instead, so a query about one of
                # these endpoints surfaces "verify manually" rather than
                # silently returning no results (or, worse, results for the
                # wrong endpoint that happen to rank nearby).
                for call in missing:
                    chunks.append(IngestedChunk(
                        chunk_id=f"api_{call}_unavailable",
                        source_type="api_schema",
                        source_name=f"{call}_schema",
                        content=(
                            f"Deriv WebSocket API Call: `{call}`\n"
                            "Schema unavailable: this endpoint could not be fetched live and "
                            "has no verified offline fallback in this system. Do not assume "
                            "its required parameters — check https://api.deriv.com or the "
                            "official deriv-api-docs repository directly."
                        ),
                        metadata={"endpoint": call, "source": "unavailable"}
                    ))

        return chunks

    def ingest_synthetic_indices_matrix(self) -> List[IngestedChunk]:
        """
        Parses Deriv's core differentiator: Synthetic Indices & Volatility Instruments.
        Ingests strict tabular data as localized entity blocks rather than splitting them randomly.
        """
        raw_table_specs = [
            {
                "symbol": "R_100 (Volatility 100 Index)",
                "tick_frequency": "1 tick per 2 seconds",
                "volatility": "Constant 100% market volatility",
                "max_leverage": "1:500",
                "min_duration": "5 ticks",
                "max_duration": "365 days",
                "trading_hours": "24/7/365 uninterrupted",
                "pricing_algorithm": "Cryptographically secure random number generator based on drift and diffusion"
            },
            {
                "symbol": "CRASH_500 (Crash 500 Index)",
                "tick_frequency": "1 tick per 1 second",
                "volatility": "Simulated bull run with sudden drops",
                "max_leverage": "1:300",
                "drop_probability": "Average of 1 drop every 500 ticks",
                "drop_size": "Drops price by roughly 50 times the standard deviation",
                "trading_hours": "24/7/365 uninterrupted",
                "pricing_algorithm": "Poisson point process for crash occurrences"
            },
            {
                "symbol": "BOOM_1000 (Boom 1000 Index)",
                "tick_frequency": "1 tick per 1 second",
                "volatility": "Simulated bear drop with sudden upward spikes",
                "max_leverage": "1:300",
                "spike_probability": "Average of 1 spike every 1000 ticks",
                "spike_size": "Spikes price upwards significantly",
                "trading_hours": "24/7/365 uninterrupted",
                "pricing_algorithm": "Poisson point process for spike occurrences"
            }
        ]

        # FIX (flagged High / "clearly" in review): these specs are hand-typed
        # into this script, not pulled from any live or official Deriv source.
        # Numbers like leverage caps and drop/spike probabilities can and do
        # change, and getting them wrong is directly consequential for anyone
        # trading on them. Every chunk built from this table is now explicitly
        # labeled as unverified, both in the payload metadata (so callers can
        # filter or badge it) and in the chunk text itself (so it surfaces even
        # if only the raw text reaches a user).
        DISCLAIMER = (
            "[UNVERIFIED / ILLUSTRATIVE DATA — not sourced from a live Deriv "
            "feed or official documentation. Confirm against https://deriv.com "
            "or the live API before relying on these figures for trading.]"
        )

        chunks = []
        for spec in raw_table_specs:
            content = (
                f"{DISCLAIMER}\n"
                f"Synthetic Index Specification: {spec['symbol']}\n"
                f"Tick Frequency: {spec['tick_frequency']}\n"
                f"Market Profile: {spec['volatility']}\n"
                f"Max Leverage: {spec['max_leverage']}\n"
                f"Math Pricing Engine: {spec['pricing_algorithm']}\n"
                f"Operating Hours: {spec['trading_hours']}"
            )
            chunks.append(IngestedChunk(
                chunk_id=f"spec_{spec['symbol'].split()[0]}",
                source_type="spec_table",
                source_name=spec['symbol'],
                content=content,
                metadata={
                    "asset_class": "synthetic_index",
                    "symbol": spec["symbol"],
                    "data_source": "hardcoded_unverified",
                }
            ))
        return chunks

    def ingest_markdown_knowledge_base(self, markdown_text: str) -> List[IngestedChunk]:
        """
        Header-Aware Chunking: Splits documentation cleanly along Markdown headers (##, ###)
        so mathematical equations and conditional clauses don't get truncated.

        FIX (chunking mismatch): the old pattern `r'\\n(?=## )'` only fires when
        the two characters right after a newline are exactly "# " — a "### "
        header has a THIRD '#' in that position, so the lookahead never
        matches and any ### subsection silently gets absorbed into the body
        of the preceding ## section instead of becoming its own chunk (losing
        its header metadata entirely). This version splits on any of #, ##,
        or ### at the start of a line, and records the header's level so
        downstream consumers can still tell a subsection from a top section.
        """
        chunks = []
        # Split right before any line starting with 1-3 '#' followed by a space.
        sections = re.split(r'\n(?=#{1,3} )', markdown_text)

        for idx, sec in enumerate(sections):
            if not sec.strip():
                continue
            lines = sec.strip().split("\n")
            first_line = lines[0]
            level = len(first_line) - len(first_line.lstrip("#"))
            header = first_line.replace("#", "").strip()
            body = "\n".join(lines[1:]).strip()

            chunks.append(IngestedChunk(
                chunk_id=f"md_doc_{idx}",
                source_type="markdown_doc",
                source_name=header,
                content=f"Topic: {header}\n{body}",
                metadata={"header": header, "header_level": level}
            ))
        return chunks


# ---------------------------------------------------------------------------
# Tokenizer for BM25
# ---------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """
    FIX: the original `doc.lower().split()` fails to separate tokens like
    "50%", "stop_loss", "R_100", or punctuation-adjacent words ("hits." ->
    "hits."). This regex-based tokenizer:
      - lowercases
      - keeps alphanumeric runs and underscores together (so `stop_loss`,
        `R_100`, `take_profit` stay intact as single tokens, matching how
        they appear in queries about parameter names / symbols)
      - splits off trailing punctuation and separates numbers from `%`
        (so "50%" -> ["50", "%"] and "100%." -> ["100", "%"])
    """
    text = text.lower()
    # word chars (incl. underscore) as one token, OR a standalone % sign
    tokens = re.findall(r"[a-z0-9_]+|%", text)
    return tokens


# ---------------------------------------------------------------------------
# 3. Complete Ingestion & Index Pipeline
# ---------------------------------------------------------------------------
class FullIngestionPipeline:
    def __init__(self, storage_path: str = "./qdrant_deriv_store"):
        print("Initializing Persistent Qdrant & Embedding Models...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Local persistent vector database (on disk, not in-memory)
        self.client = QdrantClient(path=storage_path)
        self.collection_name = "deriv_knowledge_base"

        # FIX: `recreate_collection` is deprecated in recent qdrant-client
        # versions. Use collection_exists() + delete + create instead, which
        # is the currently-supported way to get "recreate" semantics.
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        self.chunks: List[IngestedChunk] = []
        self.bm25: BM25Okapi = None

    def run_ingestion(self):
        engine = DerivDataIngestionEngine()

        api_chunks = engine.fetch_live_deriv_api_schemas()
        spec_chunks = engine.ingest_synthetic_indices_matrix()

        trading_policy_md = """
## Margin Calls and Stop-Out Mechanism on CFDs
Deriv uses an automated margin liquidation engine to protect accounts from negative balances.
A **Margin Call** is triggered when your Margin Level drops below **100%**. At this stage, no new orders can be placed.
The **Stop-Out Level** is strictly **50%**. When your Margin Level hits 50%, the liquidation algorithm initiates:
1. It identifies the contract with the largest floating loss.
2. It closes that position at current market tick.
3. It recalculates the margin level. If still <= 50%, it repeats sequentially.

## Multiplier Contract Settlement Rules
A Multiplier contract allows traders to amplify potential returns using leverage without the risk of losing more than their initial stake.
The stop out price is calculated based on:
$$StopOutPrice = EntryPrice \\times (1 - \\frac{Stake \\times Multiplier \\times (1 - StopOutLevel)}{Stake \\times Multiplier})$$
Automatic stop-out is enforced at 100% loss of initial stake minus commission.
Traders can attach Take Profit and Stop Loss barriers directly to the proposal payload using `take_profit` and `stop_loss` attributes.
        """
        md_chunks = engine.ingest_markdown_knowledge_base(trading_policy_md)

        self.chunks = api_chunks + spec_chunks + md_chunks
        print(f"\nTotal semantic units prepared: {len(self.chunks)}")

        print("Generating embeddings and writing to Qdrant on disk...")
        texts = [c.content for c in self.chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        points = []
        for i, (chunk, emb) in enumerate(zip(self.chunks, embeddings)):
            points.append(PointStruct(
                id=i,
                vector=emb.tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "source_type": chunk.source_type,
                    "source_name": chunk.source_name,
                    "text": chunk.content,
                    **chunk.metadata
                }
            ))

        self.client.upsert(collection_name=self.collection_name, points=points)

        print("Building BM25 sparse index...")
        tokenized_corpus = [tokenize(doc) for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"Ingestion complete. Database persisted locally.\n")

    # -----------------------------------------------------------------
    # Hybrid retrieval: dense (Qdrant) + sparse (BM25) via Reciprocal
    # Rank Fusion. FIX: previously both indexes were built but never
    # combined — only a standalone BM25 demo query existed. RRF is a
    # simple, tuning-free way to merge two ranked lists: each result's
    # score is 1 / (k + rank), summed across both lists, so a chunk
    # that ranks well in *either* retriever ranks well overall, and
    # one that ranks well in *both* ranks best of all.
    # -----------------------------------------------------------------
    def hybrid_search(self, query: str, top_k: int = 5, dense_k: int = 15, sparse_k: int = 15, rrf_k: int = 60):
        if self.bm25 is None:
            raise RuntimeError("Index not built yet — call run_ingestion() first.")

        # Dense retrieval
        query_vec = self.embedding_model.encode(query).tolist()
        dense_hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=dense_k,
        ).points
        dense_ranked_ids = [hit.payload["chunk_id"] for hit in dense_hits]

        # Sparse retrieval
        bm25_scores = self.bm25.get_scores(tokenize(query))
        sparse_ranked_idx = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:sparse_k]
        sparse_ranked_ids = [self.chunks[i].chunk_id for i in sparse_ranked_idx]

        # Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        for rank, cid in enumerate(dense_ranked_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, cid in enumerate(sparse_ranked_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        fused = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        chunk_by_id = {c.chunk_id: c for c in self.chunks}
        return [
            {"chunk_id": cid, "score": score, "content": chunk_by_id[cid].content}
            for cid, score in fused
            if cid in chunk_by_id
        ]


# ---------------------------------------------------------------------------
# 4. Verification Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = FullIngestionPipeline(storage_path="./deriv_rag_db")
    pipeline.run_ingestion()

    query = "What happens when margin hits 50% on CFDs, and what parameters does the proposal API require?"
    print(f"Testing hybrid retrieval for: '{query}'")

    results = pipeline.hybrid_search(query, top_k=3)
    print("\n--- Hybrid (RRF) Top Matches ---")
    for r in results:
        print(f"\n[score={r['score']:.4f}] {r['chunk_id']}")
        print(r["content"])
