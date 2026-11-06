import os
import re
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
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
    source_type: str  # "api_schema", "api_response", "official_doc", "spec_table", or "markdown_doc"
    source_name: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Source repository + local download cache
# ---------------------------------------------------------------------------
DERIV_REPO = "binary-com/deriv-api-docs"
DERIV_BRANCH = "master"
RAW_BASE = f"https://raw.githubusercontent.com/{DERIV_REPO}/{DERIV_BRANCH}"
TREE_URL = f"https://api.github.com/repos/{DERIV_REPO}/git/trees/{DERIV_BRANCH}?recursive=1"
CACHE_DIR = os.getenv("DERIV_SOURCE_CACHE", "./deriv_source_cache")

# Used only if the repo file listing can't be fetched AND isn't cached.
DEFAULT_API_CALLS = ["active_symbols", "proposal", "buy", "ticks_history", "statement", "contracts_for"]

MAX_CHUNK_CHARS = 1200  # all-MiniLM-L6-v2 truncates at 256 word pieces (~1000-1200 chars)


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, *key.split("/"))


def fetch_text(url: str, cache_key: str, retries: int = 2, timeout: int = 15) -> Optional[str]:
    """GET a URL, caching the body on disk. Falls back to the cached copy if the network fails."""
    headers = {}
    if "api.github.com" in url and os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.getenv('GITHUB_TOKEN')}"

    path = _cache_path(cache_key)
    for _ in range(retries):
        try:
            res = requests.get(url, timeout=timeout, headers=headers)
            if res.status_code == 200:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(res.text)
                return res.text
            if res.status_code == 404:
                return None
        except requests.RequestException:
            pass

    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return None


def list_repo_files() -> List[str]:
    text = fetch_text(TREE_URL, "_tree.json")
    if not text:
        return []
    try:
        return [item["path"] for item in json.loads(text).get("tree", []) if item.get("type") == "blob"]
    except (json.JSONDecodeError, KeyError):
        return []


# ---------------------------------------------------------------------------
# JSON-schema -> readable text
# ---------------------------------------------------------------------------
def _fmt_enum(values: List[Any], limit: int = 30) -> str:
    shown = ", ".join(str(v) for v in values[:limit])
    return shown + (f", ... (+{len(values) - limit} more)" if len(values) > limit else "")


def describe_properties(properties: Dict[str, Any], required: List[str], prefix: str = "",
                        depth: int = 0, max_depth: int = 2) -> List[str]:
    lines = []
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            continue
        p_type = spec.get("type", "any")
        if isinstance(p_type, list):
            p_type = "|".join(p_type)
        req = "[REQUIRED]" if name in required else "[OPTIONAL]"
        desc = re.sub(r"^\[Optional\]\s*", "", spec.get("description", "").strip())
        extras = []
        if "enum" in spec:
            extras.append(f"Allowed values: {_fmt_enum(spec['enum'])}")
        for key, label in (("default", "Default"), ("minimum", "Minimum"), ("maximum", "Maximum")):
            if key in spec:
                extras.append(f"{label}: {spec[key]}")
        if "examples" in spec:
            extras.append(f"Example: {_fmt_enum(spec['examples'], 3)}")
        line = f"{'  ' * depth}  * {prefix}{name} ({p_type}) {req}: {desc}"
        if extras:
            line += " " + "; ".join(extras) + "."
        lines.append(line)

        if depth < max_depth:
            nested = spec.get("properties")
            if not nested and isinstance(spec.get("items"), dict):
                nested = spec["items"].get("properties")
                req_nested = spec["items"].get("required", [])
            else:
                req_nested = spec.get("required", [])
            if nested:
                lines.extend(describe_properties(nested, req_nested, f"{prefix}{name}.", depth + 1, max_depth))
    return lines


def pack_lines(header: str, lines: List[str], max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Group lines into chunks under max_chars, repeating the header in each chunk."""
    parts, current = [], []
    size = len(header)
    for line in lines:
        if len(line) > max_chars:
            line = line[: max_chars - 3] + "..."
        if current and size + len(line) + 1 > max_chars:
            parts.append(header + "\n".join(current))
            current, size = [], len(header)
        current.append(line)
        size += len(line) + 1
    if current or not parts:
        parts.append(header + "\n".join(current))
    return parts


# ---------------------------------------------------------------------------
# Markdown / MDX cleanup + splitting
# ---------------------------------------------------------------------------
def clean_markdown(raw: str) -> Tuple[str, str]:
    """Strip front-matter, MDX imports and JSX/HTML tags. Returns (title, body)."""
    title = ""
    fm = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.S)
    if fm:
        t = re.search(r"^title:\s*(.+)$", fm.group(1), re.M)
        if t:
            title = t.group(1).strip().strip("'\"")
        raw = raw[fm.end():]
    raw = re.sub(r"^\s*(import|export)\s.*$", "", raw, flags=re.M)
    raw = re.sub(r"<[A-Za-z][^>]*/>", "", raw)              # self-closing JSX
    raw = re.sub(r"</?[A-Za-z][^>]*>", "", raw)             # remaining tags
    raw = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", raw)          # images
    raw = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", raw)      # links -> text
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return title, raw.strip()


def split_markdown_sections(body: str, doc_title: str) -> List[Tuple[str, str]]:
    """Split on #/##/### headers, then by paragraphs so no section exceeds MAX_CHUNK_CHARS."""
    out = []
    for sec in re.split(r"\n(?=#{1,4} )", "\n" + body):
        sec = sec.strip()
        if not sec:
            continue
        if sec.startswith("#"):
            first, _, rest = sec.partition("\n")
            header = first.lstrip("#").strip()
            text = rest.strip()
        else:
            header, text = doc_title, sec
        if not text:
            continue
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        buf = ""
        for p in paragraphs:
            if buf and len(buf) + len(p) + 2 > MAX_CHUNK_CHARS:
                out.append((header, buf))
                buf = ""
            buf = f"{buf}\n\n{p}" if buf else p[: MAX_CHUNK_CHARS * 2]
        if buf:
            out.append((header, buf))
    return out


# ---------------------------------------------------------------------------
# 2. Specialized Ingestion Loaders for Deriv Data
# ---------------------------------------------------------------------------
class DerivDataIngestionEngine:
    # Hand-written schemas used only when an endpoint can't be fetched live or from cache.
    FALLBACK_SCHEMAS = {
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

    def __init__(self, max_workers: int = 16):
        self.max_workers = max_workers
        self.repo_files = list_repo_files()
        if self.repo_files:
            print(f"Found {len(self.repo_files)} files in {DERIV_REPO}.")
        else:
            print(f"WARNING: could not list {DERIV_REPO}; using the default endpoint list only.")

    def _api_calls(self) -> List[str]:
        calls = sorted({
            p.split("/")[2] for p in self.repo_files
            if p.startswith("config/v3/") and p.endswith("/send.json")
        })
        return calls or DEFAULT_API_CALLS

    def _schema_chunks(self, call: str) -> List[IngestedChunk]:
        chunks: List[IngestedChunk] = []
        send_raw = fetch_text(f"{RAW_BASE}/config/v3/{call}/send.json", f"config/v3/{call}/send.json")
        recv_raw = fetch_text(f"{RAW_BASE}/config/v3/{call}/receive.json", f"config/v3/{call}/receive.json")

        # ---- Request schema ----
        send = None
        if send_raw:
            try:
                send = json.loads(send_raw)
            except json.JSONDecodeError:
                send = None

        if send:
            required = send.get("required", [])
            auth = send.get("auth_required", 0)
            scopes = send.get("auth_scopes", [])
            header = (
                f"Deriv WebSocket API Call: `{call}` (request)\n"
                f"Action: {send.get('title', call)}\n"
                f"Summary: {send.get('description', 'No description')}\n"
                f"Authorization required: {'yes' if auth else 'no'}"
                + (f" (scopes: {', '.join(scopes)})" if scopes else "") + "\n"
                f"Required fields: {', '.join(required) if required else 'none'}\n"
                "Parameters:\n"
            )
            lines = describe_properties(send.get("properties", {}), required)
            for i, text in enumerate(pack_lines(header, lines)):
                chunks.append(IngestedChunk(
                    chunk_id=f"api_{call}" if i == 0 else f"api_{call}_p{i}",
                    source_type="api_schema",
                    source_name=f"{call}_schema",
                    content=text,
                    metadata={"endpoint": call, "required_params": required, "part": i,
                              "url": f"https://github.com/{DERIV_REPO}/blob/{DERIV_BRANCH}/config/v3/{call}/send.json"},
                ))
        elif call in self.FALLBACK_SCHEMAS:
            print(f"  Using hardcoded fallback for '{call}'")
            chunks.append(IngestedChunk(
                chunk_id=f"api_{call}_fallback", source_type="api_schema", source_name=f"{call}_spec",
                content=self.FALLBACK_SCHEMAS[call], metadata={"endpoint": call, "source": "fallback"},
            ))
        else:
            # An explicit "unavailable" chunk beats a silent gap: queries about this endpoint
            # surface "verify manually" instead of results for a neighbouring endpoint.
            chunks.append(IngestedChunk(
                chunk_id=f"api_{call}_unavailable", source_type="api_schema", source_name=f"{call}_schema",
                content=(
                    f"Deriv WebSocket API Call: `{call}`\n"
                    "Schema unavailable: this endpoint could not be fetched live and has no verified "
                    "offline fallback in this system. Do not assume its required parameters - check "
                    "https://api.deriv.com or the official deriv-api-docs repository directly."
                ),
                metadata={"endpoint": call, "source": "unavailable"},
            ))

        # ---- Response schema (fields of the main response object) ----
        recv = None
        if recv_raw:
            try:
                recv = json.loads(recv_raw)
            except json.JSONDecodeError:
                recv = None
        if recv:
            props = recv.get("properties", {})
            body = props.get(call) if isinstance(props.get(call), dict) else {}
            items = body.get("items") if isinstance(body.get("items"), dict) else {}
            if body.get("properties"):
                lines = describe_properties(body["properties"], body.get("required", []), max_depth=1)
            elif items.get("properties"):
                lines = describe_properties(items["properties"], items.get("required", []), max_depth=1)
            else:
                top = {k: v for k, v in props.items() if k not in ("echo_req", "req_id", "passthrough")}
                lines = describe_properties(top, recv.get("required", []), max_depth=1)
            header = (
                f"Deriv WebSocket API Call: `{call}` (response)\n"
                f"Action: {recv.get('title', call)}\n"
                f"Summary: {recv.get('description', 'No description')}\n"
                "Response fields:\n"
            )
            for i, text in enumerate(pack_lines(header, lines)):
                chunks.append(IngestedChunk(
                    chunk_id=f"api_{call}_response" + (f"_p{i}" if i else ""),
                    source_type="api_response",
                    source_name=f"{call}_response",
                    content=text,
                    metadata={"endpoint": call, "part": i},
                ))
        return chunks

    def fetch_live_deriv_api_schemas(self) -> List[IngestedChunk]:
        """Every endpoint in deriv-api-docs/config/v3: request params + response fields."""
        calls = self._api_calls()
        print(f"Fetching request/response schemas for {len(calls)} API endpoints...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(self._schema_chunks, calls))
        chunks = [c for group in results for c in group]
        unavailable = [c.metadata["endpoint"] for c in chunks if c.metadata.get("source") == "unavailable"]
        if unavailable:
            print(f"WARNING: no verified schema for endpoints: {unavailable}")
        return chunks

    def _doc_chunks(self, path: str) -> List[IngestedChunk]:
        raw = fetch_text(f"{RAW_BASE}/{path}", path)
        if not raw:
            return []
        title, body = clean_markdown(raw)
        if not title:
            title = path.rsplit("/", 2)[-2].replace("-", " ").strip("_").capitalize()
        slug = re.sub(r"[^a-z0-9]+", "_", path.lower()).strip("_")
        chunks = []
        for i, (header, text) in enumerate(split_markdown_sections(body, title)):
            if len(text) < 40:  # skip stubs like "Coming soon"
                continue
            name = title if header == title else f"{title} - {header}"
            chunks.append(IngestedChunk(
                chunk_id=f"doc_{slug}_{i}",
                source_type="official_doc",
                source_name=name,
                content=f"Topic: {name}\n{text}",
                metadata={"doc_title": title, "header": header, "repo_path": path,
                          "url": f"https://github.com/{DERIV_REPO}/blob/{DERIV_BRANCH}/{path}"},
            ))
        return chunks

    def fetch_official_docs(self) -> List[IngestedChunk]:
        """Markdown/MDX pages from deriv-api-docs/docs (concepts, guides, FAQ, terminology, examples)."""
        paths = [
            p for p in self.repo_files
            if p.startswith("docs/") and p.endswith((".md", ".mdx")) and "backup" not in p
        ]
        if not paths:
            print("WARNING: no documentation pages found to ingest.")
            return []
        print(f"Fetching {len(paths)} documentation pages...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(self._doc_chunks, paths))
        return [c for group in results for c in group]

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
# 2b. Live market data from Deriv's public WebSocket (no auth required)
# ---------------------------------------------------------------------------
# Docs: https://developers.deriv.com/llms/active-symbols.md (and contracts-for, trading-times)
DERIV_PUBLIC_WS = os.getenv("DERIV_PUBLIC_WS", "wss://api.derivws.com/trading/v1/options/ws/public")

# Readable names for contract_category codes returned by contracts_for.
CONTRACT_CATEGORY_NAMES = {
    "accumulator": "Accumulators",
    "asian": "Asians",
    "callput": "Rise/Fall",
    "callputequal": "Rise/Fall Equal",
    "digits": "Digits",
    "endsinout": "Ends Between/Ends Outside",
    "higherlower": "Higher/Lower",
    "highlowticks": "High/Low Ticks",
    "multiplier": "Multipliers",
    "reset": "Reset Call/Reset Put",
    "runs": "Only Ups/Only Downs",
    "staysinout": "Stays Between/Goes Outside",
    "touchnotouch": "Touch/No Touch",
    "turbos": "Turbos",
    "vanilla": "Vanilla Options",
}

# Fields that are rules of the contract (stable). Price-dependent fields such as
# barrier, high_barrier, low_barrier, barrier_choices, stakes and barrier offsets
# change with every tick, so they are deliberately NOT stored.
RANGE_FIELDS = {
    "multiplier_range": "Multipliers",
    "cancellation_range": "Deal cancellation options",
    "growth_rate_range": "Growth rates",
    "last_digit_range": "Last digit predictions",
    "payout_choices": "Payout per point choices",
}


def _fmt_duration(value: str) -> str:
    if value in (None, "", "0"):
        return "no fixed expiry"
    units = {"t": "ticks", "s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
    m = re.fullmatch(r"(\d+)([tsmhd])", str(value))
    if not m:
        return str(value)
    n, unit = m.group(1), units[m.group(2)]
    return f"{n} {unit[:-1] if n == '1' else unit}"


class DerivLiveMarketIngestion:
    """Symbols, available contracts and trading hours from Deriv's public market-data WebSocket.

    Fetched once per ingestion run (not per question). Raw responses are cached so a failed
    run falls back to the last successful fetch; nothing here is hand-written.
    """

    def __init__(self, url: str = DERIV_PUBLIC_WS, timeout: int = 20):
        self.url = url
        self.timeout = timeout
        self.fetched_at: Optional[str] = None

    # ---------------- transport ----------------
    def _fetch_all(self) -> Optional[Dict[str, Any]]:
        import websocket  # websocket-client
        from datetime import datetime, timezone

        ws = websocket.create_connection(self.url, timeout=self.timeout)
        req_id = 0

        def call(payload: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal req_id
            req_id += 1
            ws.send(json.dumps({**payload, "req_id": req_id}))
            while True:  # skip anything that isn't the reply to this request
                msg = json.loads(ws.recv())
                if msg.get("req_id") == req_id:
                    return msg

        try:
            symbols_res = call({"active_symbols": "full"})
            if "error" in symbols_res:
                raise RuntimeError(f"active_symbols: {symbols_res['error']}")
            symbols = symbols_res.get("active_symbols", [])
            if not symbols:
                raise RuntimeError("active_symbols returned no symbols")

            times_res = call({"trading_times": "today"})
            trading_times = times_res.get("trading_times", {}) if "error" not in times_res else {}

            contracts: Dict[str, Any] = {}
            for i, s in enumerate(symbols, 1):
                sym = s["underlying_symbol"]
                res = call({"contracts_for": sym})
                contracts[sym] = res.get("contracts_for", {}).get("available", []) if "error" not in res else []
                if i % 20 == 0:
                    print(f"  contracts_for: {i}/{len(symbols)} symbols")
        finally:
            ws.close()

        return {
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "active_symbols": symbols,
            "trading_times": trading_times,
            "contracts_for": contracts,
        }

    def _load(self) -> Optional[Dict[str, Any]]:
        cache = _cache_path("live/market_snapshot.json")
        try:
            data = self._fetch_all()
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return data
        except Exception as e:
            print(f"WARNING: live market fetch failed ({e}).")
            if os.path.exists(cache):
                with open(cache, encoding="utf-8") as f:
                    data = json.load(f)
                print(f"  Using cached market snapshot from {data.get('fetched_at')}.")
                return data
            return None

    # ---------------- chunk building ----------------
    @staticmethod
    def _hours_index(trading_times: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        index = {}
        for market in trading_times.get("markets", []):
            for sub in market.get("submarkets", []):
                for s in sub.get("symbols", []):
                    index[s.get("underlying_symbol")] = {
                        "market_name": market.get("name"),
                        "submarket_name": sub.get("name"),
                        **s,
                    }
        return index

    @staticmethod
    def _contract_lines(available: List[Dict[str, Any]]) -> List[str]:
        lines = []
        by_cat: Dict[str, List[Dict[str, Any]]] = {}
        for c in available:
            by_cat.setdefault(c.get("contract_category", "other"), []).append(c)
        for cat in sorted(by_cat):
            name = CONTRACT_CATEGORY_NAMES.get(cat, cat)
            lines.append(f"  * {name} (category `{cat}`):")
            seen = set()
            for c in sorted(by_cat[cat], key=lambda c: (c.get("contract_type", ""), c.get("expiry_type", ""))):
                key = (c.get("contract_type"), c.get("expiry_type"), c.get("start_type"),
                       c.get("min_contract_duration"), c.get("max_contract_duration"))
                if key in seen:
                    continue
                seen.add(key)
                expiry = c.get("expiry_type", "?")
                if expiry == "no_expiry":
                    duration = "no fixed expiry"
                else:
                    duration = (f"duration {_fmt_duration(c.get('min_contract_duration'))} to "
                                f"{_fmt_duration(c.get('max_contract_duration'))}")
                extras = [f"{label}: {', '.join(str(v) for v in c[field])}"
                          for field, label in RANGE_FIELDS.items() if c.get(field)]
                if c.get("start_type") == "forward":
                    extras.append("forward-starting")
                line = f"    - contract_type `{c.get('contract_type')}` ({expiry} expiry): {duration}"
                if extras:
                    line += "; " + "; ".join(extras)
                lines.append(line)
        return lines

    def fetch_market_chunks(self) -> List[IngestedChunk]:
        print(f"Fetching live market data from {self.url} ...")
        data = self._load()
        if not data:
            print("WARNING: no live market data available; symbol/contract questions will be unanswerable.")
            return []

        fetched_at = data["fetched_at"]
        hours = self._hours_index(data.get("trading_times", {}))
        source_note = f"Source: Deriv public API (active_symbols, contracts_for, trading_times), fetched {fetched_at}."
        chunks: List[IngestedChunk] = []
        submarkets: Dict[Tuple[str, str], List[str]] = {}

        for s in data["active_symbols"]:
            sym = s["underlying_symbol"]
            name = s.get("underlying_symbol_name", sym)
            h = hours.get(sym, {})
            market_name = h.get("market_name") or s.get("market")
            submarket_name = h.get("submarket_name") or s.get("submarket")
            submarkets.setdefault((market_name, submarket_name), []).append(f"{sym} ({name})")

            available = data["contracts_for"].get(sym, [])
            categories = sorted({CONTRACT_CATEGORY_NAMES.get(c.get("contract_category"), c.get("contract_category"))
                                 for c in available})
            times = h.get("times", {})
            hours_line = ""
            if times:
                opens, closes = times.get("open", []), times.get("close", [])
                if opens == ["00:00:00"] and closes == ["23:59:59"]:
                    hours_line = "Trading hours: open all day (00:00-23:59:59 UTC)"
                else:
                    hours_line = "Trading hours (UTC): " + ", ".join(f"{o}-{c}" for o, c in zip(opens, closes))
                if h.get("trading_days"):
                    hours_line += f" on {', '.join(h['trading_days'])}"
            events = "; ".join(f"{e.get('descrip')} ({e.get('dates')})" for e in h.get("events", []) if e.get("descrip"))

            overview = (
                f"Deriv Market Symbol: {sym}\n"
                f"Name: {name}\n"
                f"Market: {market_name} > {submarket_name} (market code `{s.get('market')}`, "
                f"submarket code `{s.get('submarket')}`)\n"
                f"Use `{sym}` as the `underlying_symbol` / symbol value in API requests "
                f"(e.g. `contracts_for`, `proposal`, `ticks`).\n"
                f"Pip size: {s.get('pip_size')}\n"
                + (hours_line + "\n" if hours_line else "")
                + (f"Market events: {events}\n" if events else "")
                + (f"Contract types available: {', '.join(categories)}\n" if categories
                   else "No contracts were offered for this symbol at fetch time.\n")
                + source_note
            )
            chunks.append(IngestedChunk(
                chunk_id=f"live_symbol_{sym}",
                source_type="live_market",
                source_name=f"{sym} ({name}) - market info",
                content=overview,
                metadata={"symbol": sym, "market": s.get("market"), "submarket": s.get("submarket"),
                          "fetched_at": fetched_at},
            ))

            if available:
                header = (f"Deriv contracts available on {sym} ({name}), {market_name} > {submarket_name}.\n"
                          f"{source_note}\n")
                for i, text in enumerate(pack_lines(header, self._contract_lines(available))):
                    chunks.append(IngestedChunk(
                        chunk_id=f"live_contracts_{sym}_p{i}",
                        source_type="live_market",
                        source_name=f"{sym} ({name}) - contracts",
                        content=text,
                        metadata={"symbol": sym, "part": i, "fetched_at": fetched_at},
                    ))

        for (market_name, submarket_name), members in sorted(submarkets.items()):
            header = f"Deriv {market_name} > {submarket_name}: symbols available ({len(members)}).\n{source_note}\n"
            slug = re.sub(r"[^a-z0-9]+", "_", f"{market_name}_{submarket_name}".lower()).strip("_")
            for i, text in enumerate(pack_lines(header, [f"  * {m}" for m in sorted(members)])):
                chunks.append(IngestedChunk(
                    chunk_id=f"live_submarket_{slug}_p{i}",
                    source_type="live_market",
                    source_name=f"{market_name} > {submarket_name} - symbol list",
                    content=text,
                    metadata={"market": market_name, "submarket": submarket_name, "fetched_at": fetched_at},
                ))

        print(f"  {len(data['active_symbols'])} symbols -> {len(chunks)} chunks (data as of {fetched_at}).")
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
        doc_chunks = engine.fetch_official_docs()
        live_chunks = DerivLiveMarketIngestion().fetch_market_chunks()
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

        self.chunks = api_chunks + doc_chunks + live_chunks + spec_chunks + md_chunks
        print(f"\nTotal semantic units prepared: {len(self.chunks)} "
              f"(api={len(api_chunks)}, docs={len(doc_chunks)}, live={len(live_chunks)}, specs={len(spec_chunks)}, policy={len(md_chunks)})")

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
