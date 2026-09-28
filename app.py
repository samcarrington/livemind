#!/usr/bin/env python3
"""
LiveMind — FastAPI Server
Real-time conversation visualization. Receives audio from browser,
dispatches to STT, proxies LLM calls, manages graph reconciliation.

This module wires the app together: lifespan, router registration and the
CLI entry point. Behaviour lives in ``routes/`` (HTTP/WS surface) and
``services/`` (runtime state, LLM chain, live pipeline, post-session jobs).
"""

import argparse
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

# db must be imported before settings: settings loads .env, and db.DB_PATH
# has always been read from the process environment only.
import db
from stt_worker import configure_stt, configure_stt_urls

from settings import (
    HUGIN_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_RECAP_MODEL,
    STT_SERVER_URL,
)
from routes import live, pages, post_session, providers, sessions
from services.live_pipeline import broadcast_loop, snapshot_loop
from services.llm import check_ollama_models

# Back-compat re-exports: replay.py (and older tooling) imports these from app.
from services.llm import (  # noqa: F401
    _active_llm,
    _active_llm_lock,
    call_llm_chain,
    llm_chain as _llm_chain,
    llm_chain_lock as _llm_chain_lock,
    call_provider as _call_provider,
    extract_graph_json as _extract_graph_json,
)


# ─── Lifespan ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    # Default STT backend
    configure_stt_urls(
        remote_url=STT_SERVER_URL,
        parakeet_url=os.environ.get("PARAKEET_URL", ""),
        canary_url=os.environ.get("CANARY_URL", ""),
    )
    configure_stt(os.environ.get("STT_BACKEND") or "remote")
    print(f"  STT: faster-whisper ({STT_SERVER_URL})")
    print(
        f"  LLM: Ollama ({HUGIN_BASE_URL}, model={OLLAMA_MODEL}, "
        f"recap={OLLAMA_RECAP_MODEL}, num_ctx={OLLAMA_NUM_CTX})"
    )
    await check_ollama_models()
    asyncio.create_task(broadcast_loop())
    asyncio.create_task(snapshot_loop())
    print("  Server ready — audio arrives from browser via WebSocket")
    print(f"  Main:     http://127.0.0.1:{WS_PORT}/")
    print(f"  Monitor:  http://127.0.0.1:{WS_PORT}/monitor")
    print(f"  Sessions: http://127.0.0.1:{WS_PORT}/sessions\n")
    yield
    await db.close_db()


app = FastAPI(lifespan=lifespan)
WS_PORT = 8765

# Order matters: post_session registers the literal /v1/sessions/synthesis
# routes, which must match before sessions' /v1/sessions/{session_id}.
app.include_router(pages.router)
app.include_router(post_session.router)
app.include_router(sessions.router)
app.include_router(providers.router)
app.include_router(live.router)


# ─── Entry point ───
if __name__ == "__main__":
    import uvicorn

    p = argparse.ArgumentParser(description="LiveMind Server")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=8765, help="Bind port")
    args = p.parse_args()

    print("=" * 50)
    print("  LiveMind : Server")
    print("=" * 50 + "\n")

    WS_PORT = args.port

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
