"""LLM provider adapters, fallback chain and per-provider circuit breakers.

The chain (``llm_chain``) is mutated in place under ``llm_chain_lock`` so that
modules which imported it (including ``replay.py`` via ``app``) keep seeing the
live list.
"""

import asyncio
import json
import sys
import threading
import time

import aiohttp

from services.runtime import metrics, metrics_lock
from settings import (
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    HUGIN_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_RECAP_MODEL,
    VALID_PROVIDERS,
)


def _default_chain() -> list[dict]:
    """Build the default LLM fallback chain from whichever providers have
    credentials configured at startup. Order is: local (Hugin) → Gemini →
    Anthropic. Anthropic is last-resort because it's the slowest link to
    reach over the public internet and the most expensive per token."""
    chain: list[dict] = []
    if HUGIN_BASE_URL:
        chain.append({"provider": "hugin", "model": OLLAMA_MODEL})
    if GEMINI_API_KEY:
        chain.append({"provider": "gemini", "model": "gemini-2.5-flash"})
    if ANTHROPIC_API_KEY:
        chain.append({"provider": "anthropic", "model": "claude-sonnet-4-20250514"})
    if not chain:
        # No credentials at all — keep Hugin as a placeholder so the server
        # still boots; calls will fail closed with a clear error.
        chain.append({"provider": "hugin", "model": OLLAMA_MODEL})
    return chain


llm_chain: list[dict] = _default_chain()


llm_chain_lock = threading.Lock()


_last_serving_provider: str = ""


# Back-compat alias: some code paths (replay.py) still import _active_llm /
# _active_llm_lock. Expose them as views onto the head of the chain.
_active_llm_lock = llm_chain_lock


class _ActiveLLMView:
    def __getitem__(self, key):
        with llm_chain_lock:
            return llm_chain[0][key]

    def get(self, key, default=None):
        with llm_chain_lock:
            return llm_chain[0].get(key, default)

    def __iter__(self):
        with llm_chain_lock:
            return iter(dict(llm_chain[0]))


_active_llm = _ActiveLLMView()


# Extra system prompt for non-Claude models to improve graph quality
_SMALL_MODEL_GRAPH_PREFIX = """CRITICAL: Output ONLY the raw JSON object. No thinking, no reasoning, no explanation, no markdown fences.

GRAPH EVOLUTION (follow strictly):
- You MUST add new nodes for every new concept, person, or topic in the NEW SEGMENT
- Always evolve the graph — never return it unchanged. The conversation is progressing, the graph must too.
- Every node MUST connect to at least 2 different nodes — no orphans
- NEVER create a star/hub where all nodes link to one central node
- Create cross-connections between related concepts, not just to the main topic
- Vary relationship labels: "enables", "requires", "part of", "contrasts", "drives", "informs", "blocks"
- Use "group" field (not "type") for node category

"""


# Per-provider circuit breakers. Each tier in the chain has its own state so
# one provider tripping doesn't lock the others out.
CB_MAX_BACKOFF = 60.0


CB_FAILURE_THRESHOLD = 3


_cb_lock = threading.Lock()


_cb: dict[str, dict] = {
    p: {"state": "closed", "failures": 0, "backoff_until": 0.0, "backoff_secs": 5.0}
    for p in VALID_PROVIDERS
}


def _cb_can_attempt(provider: str) -> bool:
    """Return True if the provider's breaker permits an attempt.
    Transitions open→half_open when backoff has elapsed."""
    now = time.time()
    with _cb_lock:
        s = _cb[provider]
        if s["state"] == "open":
            if now < s["backoff_until"]:
                return False
            s["state"] = "half_open"
        return True


def _cb_record_success(provider: str) -> None:
    with _cb_lock:
        s = _cb[provider]
        s["state"] = "closed"
        s["failures"] = 0
        s["backoff_secs"] = 5.0
        s["backoff_until"] = 0.0


def _cb_record_failure(provider: str, hard: bool = False) -> None:
    """Record a failed attempt. hard=True trips the breaker immediately
    (e.g. 429 rate limit); otherwise opens after CB_FAILURE_THRESHOLD."""
    with _cb_lock:
        s = _cb[provider]
        s["failures"] += 1
        if hard or s["failures"] >= CB_FAILURE_THRESHOLD:
            s["state"] = "open"
            s["backoff_secs"] = min(s["backoff_secs"] * 2, CB_MAX_BACKOFF)
            s["backoff_until"] = time.time() + s["backoff_secs"]


def cb_snapshot() -> list[dict]:
    """Return a serializable snapshot of every tier's breaker state,
    in chain order."""
    now = time.time()
    with llm_chain_lock:
        chain = [dict(t) for t in llm_chain]
    with _cb_lock:
        out = []
        for tier in chain:
            p = tier["provider"]
            s = _cb.get(p, {})
            out.append(
                {
                    "provider": p,
                    "model": tier["model"],
                    "state": s.get("state", "closed"),
                    "failures": s.get("failures", 0),
                    "retry_in": max(0.0, s.get("backoff_until", 0.0) - now),
                }
            )
        return out


def cb_summary_state() -> str:
    """Overall breaker health: 'closed' if any tier is attempting,
    'degraded' if primary is open but a fallback is attempting,
    'open' if every tier is open."""
    with _cb_lock:
        states = [_cb[p]["state"] for p in VALID_PROVIDERS]
    if all(s == "open" for s in states):
        return "open"
    with llm_chain_lock:
        primary = llm_chain[0]["provider"] if llm_chain else ""
    with _cb_lock:
        primary_state = _cb.get(primary, {}).get("state", "closed")
    if primary_state != "closed":
        return "degraded"
    return "closed"


async def check_ollama_models():
    """Warn at startup if Ollama is unreachable or configured models aren't pulled."""
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"{HUGIN_BASE_URL}/api/tags", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp,
        ):
            data = await resp.json()
    except Exception as e:
        print(
            f"  WARNING: Ollama not reachable at {HUGIN_BASE_URL}: {e}", file=sys.stderr
        )
        return
    have = {m.get("name") for m in data.get("models", [])}
    have |= {n.removesuffix(":latest") for n in have}
    for m in sorted({OLLAMA_MODEL, OLLAMA_RECAP_MODEL}):
        if m not in have:
            print(
                f"  WARNING: Ollama model '{m}' not found — run: ollama pull {m}",
                file=sys.stderr,
            )


def validate_tier(tier: dict) -> str | None:
    """Return None if OK, else error string."""
    provider = (tier.get("provider") or "").strip()
    model = (tier.get("model") or "").strip()
    if provider not in VALID_PROVIDERS:
        return f"Unknown provider: {provider}"
    if not model:
        return "Model is required"
    if provider == "hugin" and not HUGIN_BASE_URL:
        return "HUGIN_BASE_URL not configured"
    if provider == "gemini" and not GEMINI_API_KEY:
        return "Gemini API key not configured"
    if provider == "anthropic" and not ANTHROPIC_API_KEY:
        return "Anthropic API key not configured"
    return None


# Pricing per million tokens (input, output)
LLM_PRICING = {
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.0-flash": (0.10, 0.40),
}


DEFAULT_PRICING = (0.0, 0.0)  # self-hosted = free


def extract_usage(data: dict, provider: str) -> dict:
    """Extract token usage from API response, normalised."""
    if provider == "anthropic":
        u = data.get("usage", {})
        return {"input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0)}
    else:
        # OpenAI-compatible (Gemini, Hugin/Ollama) — check _usage (stashed) or usage
        u = data.get("_usage", data.get("usage", {}))
        return {
            "input": u.get("prompt_tokens", 0),
            "output": u.get("completion_tokens", 0),
        }


async def _call_anthropic(body: dict) -> tuple[int, dict]:
    """POST to Anthropic Messages API. Returns (status, response_dict).
    Bounded timeout so a dead Anthropic endpoint can't stall the chain."""
    if not ANTHROPIC_API_KEY:
        return 503, {"error": {"message": "Anthropic: no API key configured"}}
    timeout = aiohttp.ClientTimeout(total=60, connect=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json=body,
        ) as resp:
            data = await resp.json()
            return resp.status, data


async def _call_hugin(body: dict) -> tuple[int, dict]:
    """Translate Anthropic-format request to OpenAI-compatible, call Hugin,
    translate response back to Anthropic format."""
    model = body.get("model") or ""

    # Build OpenAI-compatible messages from Anthropic format
    oai_messages = []
    system_text = body.get("system", "")
    # Always prepend the structured-output prefix for local models
    system_text = _SMALL_MODEL_GRAPH_PREFIX + system_text
    oai_messages.append({"role": "system", "content": system_text})
    for msg in body.get("messages", []):
        oai_messages.append({"role": msg["role"], "content": msg["content"]})

    # Use Ollama native /api/chat — supports think:false and format:json
    ollama_body = {
        "model": model,
        "messages": oai_messages,
        "stream": False,
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "num_predict": 2048,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }

    print(
        f"  Hugin request: model={model}, think={ollama_body['think']}, num_predict=2048"
    )

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{HUGIN_BASE_URL}/api/chat",
            headers={"Content-Type": "application/json"},
            json=ollama_body,
            timeout=aiohttp.ClientTimeout(total=120),
            ssl=False,
        ) as resp:
            data = await resp.json()

            if resp.status != 200:
                err_msg = data.get("error", "") or str(data)
                return resp.status, {"error": {"message": f"Hugin: {err_msg}"}}

            # Ollama native response: data.message.content
            msg = data.get("message", {})
            text = msg.get("content") or ""
            if not text:
                print(
                    f"  Hugin: empty content. Message keys: {list(msg.keys())}",
                    file=sys.stderr,
                )
                print(
                    f"  Hugin: raw response keys: {list(data.keys())}", file=sys.stderr
                )
                print(f"  Hugin: raw message: {json.dumps(msg)[:500]}", file=sys.stderr)

            # Translate to Anthropic format; extract usage from Ollama metrics
            result = {"content": [{"type": "text", "text": text}]}
            if "prompt_eval_count" in data:
                result["_usage"] = {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                }
            return 200, result


async def _call_gemini(body: dict) -> tuple[int, dict]:
    """Translate Anthropic-format request to OpenAI-compatible, call Gemini,
    translate response back to Anthropic format."""
    model = body.get("model") or ""

    # Build OpenAI-compatible messages
    oai_messages = []
    system_text = body.get("system", "")
    if system_text:
        # Gemini is good at JSON but benefits from the same structural hints
        system_text = _SMALL_MODEL_GRAPH_PREFIX + system_text
        oai_messages.append({"role": "system", "content": system_text})
    for msg in body.get("messages", []):
        oai_messages.append({"role": msg["role"], "content": msg["content"]})

    oai_body = {
        "model": model,
        "messages": oai_messages,
        "temperature": 0,
    }
    # Gemini 2.5 Flash uses thinking tokens from the max_tokens budget.
    # The graph JSON needs ~1-2k tokens, but thinking can consume 2-4k.
    # Set a generous budget so thinking doesn't starve the actual output.
    oai_body["max_tokens"] = 16384

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{GEMINI_BASE_URL}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GEMINI_API_KEY}",
            },
            json=oai_body,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            data = await resp.json()

            if resp.status != 200:
                err_msg = data.get("error", {}).get("message", "") or str(data)
                return resp.status, {"error": {"message": f"Gemini: {err_msg}"}}

            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            # Gemini 2.5 Flash thinking mode: content can be null,
            # actual text may be in reasoning_content or parts
            text = msg.get("content") or ""
            if not text:
                text = msg.get("reasoning_content") or ""
            if not text:
                print(
                    f"  Gemini: empty content. Keys: {list(msg.keys())}",
                    file=sys.stderr,
                )
                print(f"  Gemini: choice: {json.dumps(choice)[:500]}", file=sys.stderr)
            result = {"content": [{"type": "text", "text": text}]}
            if "usage" in data:
                result["_usage"] = data["usage"]
            return 200, result


def extract_graph_json(raw_text: str) -> dict:
    """Extract a JSON object containing 'nodes' from LLM output.
    Handles markdown fences, thinking preamble, and extra text."""
    import re

    cleaned = raw_text.replace("```json", "").replace("```", "")

    # Strategy 1: find {"nodes" and parse from there (handles thinking preamble)
    for pattern in [r'\{\s*"nodes"\s*:', r"\{\s*'nodes'\s*:"]:
        match = re.search(pattern, cleaned)
        if match:
            start = match.start()
            # Find matching closing brace by counting depth
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == "{":
                    depth += 1
                elif cleaned[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cleaned[start : i + 1])
                        except json.JSONDecodeError:
                            break  # try next strategy

    # Strategy 2: try the whole thing stripped
    stripped = cleaned.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Strategy 3: first { to last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError(
        f"No valid graph JSON found in {len(raw_text)} chars", raw_text[:200], 0
    )


async def call_provider(provider: str, body: dict) -> tuple[int, dict]:
    """Dispatch a single LLM call to the named provider."""
    if provider == "hugin":
        return await _call_hugin(body)
    if provider == "gemini":
        return await _call_gemini(body)
    if provider == "anthropic":
        return await _call_anthropic(body)
    return 400, {"error": {"message": f"unknown provider: {provider}"}}


async def call_llm_chain(body: dict) -> tuple[int, dict, str, str]:
    """Walk the configured LLM chain in order. For each tier whose circuit
    breaker permits an attempt, try the call. Return on first 200. On any
    failure record the tier's breaker and try the next tier.

    Returns (status, data, served_by_provider, served_by_model). If every
    tier failed, served_by_provider is "" and data holds the last error.
    """
    global _last_serving_provider

    with llm_chain_lock:
        chain = [dict(t) for t in llm_chain]

    last_status = 503
    last_data: dict = {"error": {"message": "No LLM tiers configured"}}
    attempts: list[str] = []

    for tier in chain:
        provider = tier["provider"]
        model = tier["model"]

        if not _cb_can_attempt(provider):
            attempts.append(f"{provider}:skip(breaker)")
            continue

        tier_body = dict(body)
        tier_body["model"] = model
        t0 = time.time()
        try:
            status, data = await call_provider(provider, tier_body)
        except asyncio.TimeoutError:
            dt = time.time() - t0
            print(f"  LLM [{provider}]: TIMEOUT after {dt:.1f}s", file=sys.stderr)
            _cb_record_failure(provider)
            attempts.append(f"{provider}:timeout")
            last_status = 504
            last_data = {"error": {"message": f"{provider}: timeout"}}
            continue
        except Exception as e:
            dt = time.time() - t0
            print(
                f"  LLM [{provider}]: EXCEPTION ({dt:.1f}s) — {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            _cb_record_failure(provider)
            attempts.append(f"{provider}:{type(e).__name__}")
            last_status = 500
            last_data = {"error": {"message": f"{provider}: {type(e).__name__}: {e}"}}
            continue

        dt = time.time() - t0

        if status == 200:
            _cb_record_success(provider)
            _last_serving_provider = provider
            if attempts:
                print(
                    f"  LLM chain: demoted through [{', '.join(attempts)}] → served by {provider}/{model} ({dt:.1f}s)"
                )
            return status, data, provider, model

        # Non-200: record failure and try the next tier.
        err_msg = ""
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                err_msg = err.get("message", "")
            elif isinstance(err, str):
                err_msg = err
        hard = status == 429
        _cb_record_failure(provider, hard=hard)
        attempts.append(f"{provider}:{status}")
        print(f"  LLM [{provider}]: {status} ({dt:.1f}s) — {err_msg}", file=sys.stderr)
        last_status = status
        last_data = data

    print(f"  LLM chain: ALL TIERS FAILED — [{', '.join(attempts)}]", file=sys.stderr)
    return last_status, last_data, "", ""


def publish_llm_state() -> None:
    """Copy chain + breaker state into the metrics dict so the monitor can
    render a live view of which tier is currently serving."""
    tiers = cb_snapshot()
    with metrics_lock:
        metrics["llm_tiers"] = tiers
        metrics["llm_serving"] = _last_serving_provider
        metrics["cb_state"] = cb_summary_state()
        total_failures = sum(t.get("failures", 0) for t in tiers)
        metrics["cb_failures"] = total_failures


def get_last_serving_provider() -> str:
    """Provider that served the most recent successful chain call."""
    return _last_serving_provider
