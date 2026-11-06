"""
Agentic Deriv assistant built with LangGraph.

The agent decides per question which tools to call:
  * search_deriv_knowledge  -> the existing Qdrant knowledge base (read-only, same hybrid search + reranker)
  * get_latest_price        -> live spot price       (Deriv public WebSocket, ticks_history)
  * get_market_status       -> live open/suspended    (Deriv public WebSocket, active_symbols)
  * get_price_quote         -> live contract quote    (Deriv public WebSocket, proposal)

Live results are passed straight to the model and never stored. The final answer goes through
the same "every number must appear in the tool outputs" guardrail as the classic RAG endpoint.
"""
import os
import re
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal

import websocket  # websocket-client
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END

DERIV_PUBLIC_WS = os.getenv("DERIV_PUBLIC_WS", "wss://api.derivws.com/trading/v1/options/ws/public")
MAX_TOOL_ROUNDS = int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "5"))


# ---------------------------------------------------------------------------
# Live Deriv WebSocket helper (public market data, no auth)
# ---------------------------------------------------------------------------
def _deriv_call(payload: Dict[str, Any], timeout: int = 15) -> Dict[str, Any]:
    ws = websocket.create_connection(DERIV_PUBLIC_WS, timeout=timeout)
    try:
        ws.send(json.dumps({**payload, "req_id": 1}))
        while True:
            msg = json.loads(ws.recv())
            if msg.get("req_id") == 1:
                return msg
    finally:
        ws.close()


def _utc(epoch: Optional[int]) -> str:
    if not epoch:
        return "unknown time"
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _error_text(res: Dict[str, Any]) -> Optional[str]:
    err = res.get("error")
    return f"Deriv API error {err.get('code')}: {err.get('message')}" if err else None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def build_tools(rag_assistant) -> List:
    """rag_assistant: the already-initialised DerivTradingAssistant (shares its Qdrant client)."""

    @tool
    def search_deriv_knowledge(query: str) -> str:
        """Search the Deriv knowledge base: API endpoint parameters and responses, official API guides,
        and the daily market snapshot (symbol codes and names, which contract types each symbol supports,
        duration limits, multiplier ranges, trading hours). Use this for any 'how do I / what is / which
        parameters / what can I trade' question, and to look up a symbol code from its name
        (e.g. 'Volatility 50 Index' -> R_50)."""
        docs = rag_assistant.retrieve_and_rerank(query, top_k=4)
        if not docs:
            return "No relevant documents found."
        return "\n\n".join(
            f"[Source: {d.get('source_name', 'unknown')}]\n{d['text']}" for d in docs
        )

    @tool
    def get_latest_price(symbol: str) -> str:
        """LIVE: get the latest spot price of a Deriv symbol right now. `symbol` must be the symbol code,
        e.g. R_50, 1HZ100V, CRASH500, frxEURUSD (use search_deriv_knowledge first if you only know the name)."""
        try:
            res = _deriv_call({"ticks_history": symbol, "end": "latest", "count": 1, "style": "ticks"})
        except Exception as e:
            return f"Live price lookup failed (connection error: {e})."
        if _error_text(res):
            return _error_text(res)
        prices, times = res.get("history", {}).get("prices", []), res.get("history", {}).get("times", [])
        if not prices:
            return f"No recent ticks returned for {symbol}."
        pip = res.get("pip_size")
        price = f"{prices[-1]:.{int(pip)}f}" if isinstance(pip, (int, float)) else str(prices[-1])
        return (f"LIVE latest price for {symbol}: {price} (tick time {_utc(times[-1])}; "
                f"checked {_now()}; source: Deriv ticks_history).")

    @tool
    def get_market_status(symbol: str) -> str:
        """LIVE: check whether a Deriv symbol's market is open right now and whether trading is suspended.
        `symbol` must be the symbol code, e.g. R_100, frxEURUSD, OTC_NDX."""
        try:
            res = _deriv_call({"active_symbols": "brief"})
        except Exception as e:
            return f"Live market status lookup failed (connection error: {e})."
        if _error_text(res):
            return _error_text(res)
        for s in res.get("active_symbols", []):
            if s.get("underlying_symbol", "").lower() == symbol.lower():
                return (f"LIVE status for {s['underlying_symbol']} ({s.get('underlying_symbol_name')}): "
                        f"market {'OPEN' if s.get('exchange_is_open') else 'CLOSED'}, "
                        f"trading {'SUSPENDED' if s.get('is_trading_suspended') else 'not suspended'} "
                        f"(checked {_now()}; source: Deriv active_symbols).")
        return f"Symbol {symbol} is not in Deriv's list of active symbols right now."

    @tool
    def get_price_quote(
        symbol: str,
        contract_type: str,
        amount: float,
        basis: Literal["stake", "payout"] = "stake",
        currency: str = "USD",
        duration: Optional[int] = None,
        duration_unit: Optional[Literal["t", "s", "m", "h", "d"]] = None,
        multiplier: Optional[float] = None,
        growth_rate: Optional[float] = None,
        barrier: Optional[str] = None,
    ) -> str:
        """LIVE: get a real-time price quote (ask price, payout) for a specific contract via Deriv `proposal`.
        Examples: Rise on R_50 for 5 ticks with 10 USD stake -> contract_type='CALL', duration=5, duration_unit='t',
        amount=10. Multiplier up on R_100 -> contract_type='MULTUP', multiplier=100, no duration.
        Accumulator -> contract_type='ACCU', growth_rate=0.03, no duration. Digit contracts need `barrier`
        (the digit). Use search_deriv_knowledge first if unsure which contract types/durations a symbol allows."""
        payload: Dict[str, Any] = {
            "proposal": 1, "underlying_symbol": symbol, "contract_type": contract_type,
            "amount": amount, "basis": basis, "currency": currency,
        }
        for key, val in (("duration", duration), ("duration_unit", duration_unit), ("multiplier", multiplier),
                         ("growth_rate", growth_rate), ("barrier", barrier)):
            if val is not None:
                payload[key] = val
        try:
            res = _deriv_call(payload)
        except Exception as e:
            return f"Live quote failed (connection error: {e})."
        if _error_text(res):
            return _error_text(res)
        p = res.get("proposal", {})
        parts = [f"LIVE quote for {contract_type} on {symbol} (checked {_now()}; source: Deriv proposal):",
                 f"ask_price: {p.get('ask_price')} {currency}"]
        if p.get("payout") is not None:
            parts.append(f"payout: {p.get('payout')} {currency}")
        if p.get("spot") is not None:
            parts.append(f"spot: {p.get('spot')} at {_utc(p.get('spot_time'))}")
        if p.get("longcode"):
            parts.append(f"description: {p['longcode']}")
        parts.append("Quotes change every tick; this is indicative only and not an order.")
        return "\n".join(parts)

    return [search_deriv_knowledge, get_latest_price, get_market_status, get_price_quote]


# ---------------------------------------------------------------------------
# Final structured answer (list of name/value pairs: Gemini rejects free-form dict schemas)
# ---------------------------------------------------------------------------
class Parameter(BaseModel):
    name: str = Field(description="Parameter name, e.g. 'contract_type', 'latest_price', 'max_duration'.")
    value: str = Field(description="Concrete value copied exactly from the tool outputs (not a type name).")


class FinalAnswer(BaseModel):
    answer: str = Field(description="Direct, concise answer to the user's question.")
    exact_parameters: List[Parameter] = Field(
        default_factory=list,
        description="Concrete values from the tool outputs (prices, codes, limits, durations, allowed values). "
                    "Never JSON types like 'integer' or words like 'required'. Empty if none.")
    risk_warning: str = Field(description="Mandatory, brief risk disclosure about trading derivatives.")


SYSTEM_PROMPT = """You are a precise technical assistant for the Deriv trading API and Deriv markets.

Tools:
- search_deriv_knowledge: documentation, API parameters, symbol codes, which contracts a symbol supports,
  durations, multiplier ranges, trading hours (daily snapshot).
- get_latest_price / get_market_status / get_price_quote: LIVE data from Deriv, fetched now.

Rules:
- Use tools; never answer from memory. Never invent numbers, symbol codes or parameters.
- If you only know an instrument's name, find its symbol code with search_deriv_knowledge first.
- For anything that changes in real time (current price, is it open now, current payout/ask price),
  call the live tools. For rules and how-to questions, use search_deriv_knowledge.
- Copy numbers exactly as the tools return them (no rounding, no reformatting).
- If the tools don't provide the answer, say so plainly.
- You give information, not trading advice or recommendations.
- When you have what you need, call FinalAnswer (say which data is live and when it was checked;
  put concrete values in exact_parameters, never type names like 'integer')."""

FINAL_PROMPT = """Write the final answer to the user's question using ONLY the tool results above.
State which data is live and when it was checked. Copy numbers exactly. If the tools did not answer
the question, say so and leave exact_parameters empty."""


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
class AgentState(MessagesState):
    tool_rounds: int


class DerivAgent:
    def __init__(self, rag_assistant, model: str, api_key: str):
        self.rag = rag_assistant
        self.tools = build_tools(rag_assistant)
        self.tools_by_name = {t.name: t for t in self.tools}

        # max_retries: the Gemini client backs off and retries on 429 rate-limit errors.
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0, max_retries=4)
        # FinalAnswer is exposed as a tool and every turn must call some tool, so the agent ends by
        # calling FinalAnswer itself: one model call fewer per question than a separate formatting step.
        self.llm_with_tools = llm.bind_tools(self.tools + [FinalAnswer], tool_choice="any")
        self.final_llm = llm.with_structured_output(FinalAnswer)  # fallback only
        self.graph = self._build_graph()

    # -- nodes --
    def _agent(self, state: AgentState) -> Dict[str, Any]:
        messages = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            messages.append(HumanMessage("Tool budget used up. Call FinalAnswer now."))
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _tools(self, state: AgentState) -> Dict[str, Any]:
        last: AIMessage = state["messages"][-1]
        outputs = []
        for call in last.tool_calls:
            if call["name"] == "FinalAnswer":
                continue
            fn = self.tools_by_name.get(call["name"])
            try:
                result = fn.invoke(call["args"]) if fn else f"Unknown tool {call['name']}"
            except Exception as e:
                result = f"Tool {call['name']} failed: {e}"
            outputs.append(ToolMessage(content=str(result), tool_call_id=call["id"], name=call["name"]))
        return {"messages": outputs, "tool_rounds": state.get("tool_rounds", 0) + 1}

    def _route(self, state: AgentState) -> str:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return END
        if any(c["name"] == "FinalAnswer" for c in last.tool_calls):
            return END
        if state.get("tool_rounds", 0) > MAX_TOOL_ROUNDS:
            return END
        return "tools"

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("agent", self._agent)
        g.add_node("tools", self._tools)
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", self._route, {"tools": "tools", END: END})
        g.add_edge("tools", "agent")
        return g.compile()

    # -- guardrail: every number in the answer must appear in some tool output --
    def _numbers_supported(self, final: FinalAnswer, context: str) -> List[str]:
        norm = self.rag._normalize(context)
        bad = []
        values = [p.value for p in final.exact_parameters]
        tokens = re.findall(r"\d+(?:\.\d+)?", self.rag._normalize(final.answer))
        for v in values:
            if not self.rag._value_supported(v, norm) and not self._raw_present(v, norm):
                bad.append(v)
        for t in tokens:
            if not self._raw_present(t, norm) and not self.rag._value_supported(float(t), norm):
                bad.append(t)
        return bad

    @staticmethod
    def _raw_present(token: str, text: str) -> bool:
        token = token.strip().lower()
        return bool(token) and re.search(r"(?<![0-9.])" + re.escape(token) + r"(?![0-9])", text) is not None

    # -- public --
    def ask(self, query: str) -> Dict[str, Any]:
        state = self.graph.invoke(
            {"messages": [HumanMessage(query)], "tool_rounds": 0},
            {"recursion_limit": 2 * MAX_TOOL_ROUNDS + 4},
        )
        messages = state["messages"]
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        trace, final_args = [], None
        for m in messages:
            if isinstance(m, AIMessage):
                for c in m.tool_calls:
                    if c["name"] == "FinalAnswer":
                        final_args = c["args"]
                    else:
                        trace.append({"tool": c["name"], "args": c["args"]})

        if not tool_msgs:
            return {"status": "error", "message": "The agent did not consult any data source.", "tool_trace": trace}

        final: Optional[FinalAnswer] = None
        if final_args:
            try:
                final = FinalAnswer(**final_args)
            except Exception:
                final = None
        if final is None:  # fallback: separate formatting call
            final = self.final_llm.invoke(
                [SystemMessage(SYSTEM_PROMPT)] + messages + [HumanMessage(FINAL_PROMPT)]
            )
        context = "\n".join(str(m.content) for m in tool_msgs)
        unsupported = self._numbers_supported(final, context)
        if unsupported:
            print(f"[AGENT GUARDRAIL] unsupported values: {unsupported}")
            return {
                "status": "failed_guardrail",
                "error": "The generated response contained values not found in the retrieved or live data.",
                "unsupported_values": unsupported,
                "rejected_answer": final.answer,
                "safe_fallback": "Please refer directly to the Deriv official documentation.",
                "tool_trace": trace,
            }

        sources = re.findall(r"\[Source: ([^\]]+)\]", context)
        live_used = sorted({t["tool"] for t in trace if t["tool"] != "search_deriv_knowledge"})
        sources = list(dict.fromkeys(sources + [f"LIVE: {t}" for t in live_used]))

        # (label, text) per source block: each search hit separately, each live result as a whole
        blocks = []
        for m in tool_msgs:
            if m.name == "search_deriv_knowledge":
                for part in str(m.content).split("\n\n[Source: "):
                    hit = re.match(r"\[?Source: ([^\]]+)\]\n?(.*)", part if part.startswith("[") else "[Source: " + part, re.S)
                    if hit:
                        blocks.append((hit.group(1), self.rag._normalize(hit.group(2))))
            else:
                blocks.append((f"LIVE: {m.name}", self.rag._normalize(str(m.content))))

        params = {p.name: p.value for p in final.exact_parameters}
        param_sources = {}
        for name, value in params.items():
            labels = [label for label, text in blocks
                      if self._raw_present(str(value), text) or self.rag._value_supported(value, text)]
            param_sources[name] = list(dict.fromkeys(labels))[:3]

        return {
            "status": "success",
            "sources": sources,
            "parameter_sources": param_sources,
            "structured_output": {
                "answer": final.answer,
                "exact_parameters": params,
                "risk_warning": final.risk_warning,
            },
            "tool_trace": trace,
        }
