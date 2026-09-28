"""LLM/STT provider selection and metrics endpoints."""

import asyncio
import os
import sys
import time

import aiohttp
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from services.llm import (
    cb_snapshot,
    get_last_serving_provider,
    llm_chain,
    llm_chain_lock,
    publish_llm_state,
    validate_tier,
)
from services.runtime import metrics, metrics_lock, reconciler
from services.session_runtime import live_session
from settings import (
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    HUGIN_BASE_URL,
    HUGIN_CF_ID,
    HUGIN_CF_SECRET,
)
from stt_worker import configure_stt, get_stt_config

router = APIRouter()


@router.get("/v1/metrics")
async def get_metrics_rest():
    with metrics_lock:
        m = {**metrics, "uptime": time.time() - metrics["started_at"]}
    churn = reconciler.get_churn_metrics()
    m.update(churn)
    m["current_session_id"] = live_session.current_session_id
    m["active_nodes"] = len(
        [ns for ns in reconciler.nodes.values() if ns.state == "active"]
    )
    with llm_chain_lock:
        head = dict(llm_chain[0]) if llm_chain else {"provider": "", "model": ""}
    m["llm_provider"] = head.get("provider", "")
    m["llm_model"] = head.get("model", "")
    m["llm_tiers"] = cb_snapshot()
    m["llm_serving"] = get_last_serving_provider()
    stt = get_stt_config()
    m["stt_backend"] = stt["backend"]
    m["stt_remote_url"] = stt.get("remote_url", "")
    return JSONResponse(m)


@router.get("/v1/llm/providers")
async def list_llm_providers():
    """Return available LLM providers and models."""
    providers = {
        "anthropic": {
            "label": "Anthropic (Claude)",
            "models": [
                {
                    "id": "claude-sonnet-4-20250514",
                    "label": "Claude Sonnet 4",
                    "note": "Default",
                },
                {
                    "id": "claude-haiku-4-5-20251001",
                    "label": "Claude Haiku 4.5",
                    "note": "Fast",
                },
            ],
            "available": bool(ANTHROPIC_API_KEY),
        },
    }
    if HUGIN_BASE_URL:
        providers["hugin"] = {
            "label": "Ollama (Local)",
            "models": [],
            "available": True,
        }
        # Fetch live model list from Ollama
        try:
            headers = {}
            if HUGIN_CF_ID and HUGIN_CF_SECRET:
                headers["CF-Access-Client-Id"] = HUGIN_CF_ID
                headers["CF-Access-Client-Secret"] = HUGIN_CF_SECRET
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{HUGIN_BASE_URL}/api/tags",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                    ssl=False,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for m in data.get("models", []):
                            size = m.get("details", {}).get("parameter_size", "")
                            providers["hugin"]["models"].append(
                                {
                                    "id": m["name"],
                                    "label": m["name"],
                                    "note": size,
                                }
                            )
        except Exception as e:
            print(f"  Hugin model list failed: {e}", file=sys.stderr)
            providers["hugin"]["available"] = False
            providers["hugin"]["error"] = str(e)

    if GEMINI_API_KEY:
        providers["gemini"] = {
            "label": "Google (Gemini)",
            "models": [
                {
                    "id": "gemini-2.5-flash",
                    "label": "Gemini 2.5 Flash",
                    "note": "$0.075/$0.30 per M tok",
                },
                {
                    "id": "gemini-2.0-flash",
                    "label": "Gemini 2.0 Flash",
                    "note": "$0.10/$0.40 per M tok",
                },
            ],
            "available": True,
        }

    with llm_chain_lock:
        chain = [dict(t) for t in llm_chain]
    tiers_state = cb_snapshot()
    return JSONResponse(
        {
            "providers": providers,
            "chain": chain,
            "tiers": tiers_state,
            "serving": get_last_serving_provider(),
            # Back-compat: legacy clients still read `active`.
            "active": chain[0] if chain else {},
        }
    )


@router.post("/v1/llm/active")
async def set_active_llm(request: Request):
    """Update the LLM fallback chain.

    Accepts:
      - {"chain": [{"provider": "...", "model": "..."}, ...]} — replace the
        full chain. Order is priority (first is primary).
      - {"provider": "...", "model": "..."} — back-compat: update that
        provider's tier in-place (or add it as primary if not yet present).
    """
    body = await request.json()

    if isinstance(body.get("chain"), list):
        new_chain: list[dict] = []
        seen: set[str] = set()
        for raw in body["chain"]:
            if not isinstance(raw, dict):
                return JSONResponse(
                    {"error": "chain entries must be objects"}, status_code=400
                )
            err = validate_tier(raw)
            if err:
                return JSONResponse({"error": err}, status_code=400)
            p = raw["provider"].strip()
            m = raw["model"].strip()
            if p in seen:
                return JSONResponse(
                    {"error": f"duplicate provider in chain: {p}"}, status_code=400
                )
            seen.add(p)
            new_chain.append({"provider": p, "model": m})
        if not new_chain:
            return JSONResponse({"error": "chain must not be empty"}, status_code=400)

        with llm_chain_lock:
            old = [dict(t) for t in llm_chain]
            llm_chain.clear()
            llm_chain.extend(new_chain)

        print(
            f"  LLM chain: {[f'{t["provider"]}/{t["model"]}' for t in old]} → {[f'{t["provider"]}/{t["model"]}' for t in new_chain]}"
        )
        publish_llm_state()
        return JSONResponse({"chain": new_chain, "tiers": cb_snapshot()})

    # Back-compat single-tier update
    provider = (body.get("provider") or "").strip()
    model = (body.get("model") or "").strip()
    err = validate_tier({"provider": provider, "model": model})
    if err:
        return JSONResponse({"error": err}, status_code=400)

    with llm_chain_lock:
        found = False
        for tier in llm_chain:
            if tier["provider"] == provider:
                tier["model"] = model
                found = True
                break
        if not found:
            llm_chain.insert(0, {"provider": provider, "model": model})
        new_chain = [dict(t) for t in llm_chain]

    print(f"  LLM chain: updated tier {provider} → {model}")
    publish_llm_state()
    return JSONResponse(
        {"chain": new_chain, "active": {"provider": provider, "model": model}}
    )


@router.get("/v1/stt/backends")
async def list_stt_backends():
    """Return available STT backends with health checks."""
    stt_cfg = get_stt_config()
    backends = {
        "remote": {
            "label": "faster-whisper",
            "note": "99 languages, proven",
            "url": stt_cfg["remote_url"],
            "available": False,
        },
        "parakeet": {
            "label": "Parakeet TDT 0.6b v3",
            "note": "25 EU languages, highest throughput",
            "url": stt_cfg["parakeet_url"],
            "available": False,
        },
        "canary": {
            "label": "Canary 1b v2",
            "note": "25 EU languages, best accuracy",
            "url": stt_cfg["canary_url"],
            "available": False,
        },
    }

    # Health check each backend in parallel.
    # Uses requests (via executor) so CF-Access headers survive cross-domain redirects
    # that aiohttp strips by default.
    import requests as _requests

    loop = asyncio.get_event_loop()

    def _check_health_sync(key, url):
        try:
            headers = {}
            if key == "remote":
                cf_id = os.environ.get("STT_CF_ID") or os.environ.get("HUGIN_CF_ID", "")
                cf_secret = os.environ.get("STT_CF_SECRET") or os.environ.get(
                    "HUGIN_CF_SECRET", ""
                )
                if cf_id and cf_secret:
                    headers["CF-Access-Client-Id"] = cf_id
                    headers["CF-Access-Client-Secret"] = cf_secret
            resp = _requests.get(
                f"{url}/health", headers=headers, timeout=5, verify=False
            )
            if resp.status_code == 200:
                backends[key]["available"] = True
                data = resp.json()
                if "model" in data:
                    backends[key]["model"] = data["model"]
        except Exception:
            pass

    async def check_health(key, url):
        await loop.run_in_executor(None, _check_health_sync, key, url)

    await asyncio.gather(
        check_health("remote", stt_cfg["remote_url"]),
        check_health("parakeet", stt_cfg["parakeet_url"]),
        check_health("canary", stt_cfg["canary_url"]),
    )

    active = stt_cfg
    return JSONResponse({"backends": backends, "active": active})


@router.post("/v1/stt/active")
async def set_active_stt(request: Request):
    """Switch the active STT backend and/or language."""
    body = await request.json()
    backend = body.get("backend")
    language = body.get("language")

    old = get_stt_config()

    # If backend is being changed, validate it
    if backend is not None:
        backend = backend.strip()
        if backend not in ("remote", "parakeet", "canary"):
            return JSONResponse(
                {"error": f"Unknown backend: {backend}"}, status_code=400
            )
    else:
        backend = old["backend"]

    lang = language.strip() if language is not None else old.get("language", "")
    configure_stt(backend, "", lang)
    new = get_stt_config()

    print(
        f"  STT config: backend={new['backend']}, language={new['language'] or 'auto'}"
    )
    return JSONResponse({"active": new})
