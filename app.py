#!/usr/bin/env python3
"""
LiveMind — FastAPI Server
Real-time conversation visualization. Receives audio from browser,
dispatches to STT, proxies LLM calls, manages graph reconciliation.
"""

import asyncio, json, time, threading, queue, sys, argparse, os, uuid, base64
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import aiohttp

import db
import stt_worker
from stt_worker import configure_stt, configure_stt_urls, get_stt_config
from reconciler import GraphReconciler

# ─── Load .env ───
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ─── Hugin (Ollama) ───
# Defaults to a local Ollama on this machine. Set HUGIN_BASE_URL (or
# OLLAMA_BASE_URL) to point at a remote/tunnelled Ollama instead.
HUGIN_BASE_URL = (
    os.environ.get("HUGIN_BASE_URL")
    or os.environ.get("OLLAMA_BASE_URL")
    or "http://localhost:11434"
).rstrip("/")
# Model used for live graph extraction (default chain head)
OLLAMA_MODEL = (os.environ.get("OLLAMA_MODEL") or "gemma4:e4b").strip()
# Model used for recaps / synthesis / transcript cleanup (heavier jobs)
OLLAMA_RECAP_MODEL = (os.environ.get("OLLAMA_RECAP_MODEL") or OLLAMA_MODEL).strip()
# Context window requested from Ollama. Ollama's own default (4096) silently
# truncates long recap prompts; larger values cost more RAM.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX") or 16384)
# How long Ollama keeps the model loaded between calls (avoids reload stalls)
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE") or "30m"
# Parallel requests for batch jobs (transcript cleanup). 1 suits a single laptop.
OLLAMA_MAX_CONCURRENCY = max(1, int(os.environ.get("OLLAMA_MAX_CONCURRENCY") or 1))
# Rough transcript budget for recaps: ~3 chars/token, leaving room for
# the system prompt and 4096 output tokens.
_RECAP_MAX_CHARS = max(8000, min(150000, (OLLAMA_NUM_CTX - 4096 - 2000) * 3))
HUGIN_CF_ID = os.environ.get("HUGIN_CF_ID", "")
HUGIN_CF_SECRET = os.environ.get("HUGIN_CF_SECRET", "")

# ─── Gemini ───
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

# ─── STT (faster-whisper compatible /v1/transcribe server) ───
# Defaults to the bundled local server (stt_server.py) on :8766.
STT_SERVER_URL = os.environ.get("STT_SERVER_URL", "http://localhost:8766").rstrip("/")

# LLM fallback chain — ordered list of tiers, tried in order.
# First tier is primary; each subsequent tier is a fallback.
# Mutable at runtime via /v1/llm/active.
_VALID_PROVIDERS = ("hugin", "gemini", "anthropic")


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


_llm_chain: list[dict] = _default_chain()
_llm_chain_lock = threading.Lock()
_last_serving_provider: str = ""

# Back-compat alias: some code paths (replay.py) still import _active_llm /
# _active_llm_lock. Expose them as views onto the head of the chain.
_active_llm_lock = _llm_chain_lock


class _ActiveLLMView:
    def __getitem__(self, key):
        with _llm_chain_lock:
            return _llm_chain[0][key]

    def get(self, key, default=None):
        with _llm_chain_lock:
            return _llm_chain[0].get(key, default)

    def __iter__(self):
        with _llm_chain_lock:
            return iter(dict(_llm_chain[0]))


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

# ─── Global state ───
transcript_queue: queue.Queue = queue.Queue()
connected_clients: set[WebSocket] = set()
client_sessions: dict[WebSocket, str] = {}  # ws → session_id

metrics = {
    "started_at": time.time(),
    "chunks_processed": 0,
    "chunks_skipped_silent": 0,
    "chunks_skipped_catchup": 0,
    "stt_last_duration": 0.0,
    "stt_avg_duration": 0.0,
    "stt_total_time": 0.0,
    "stt_last_text": "",
    "stt_empty_results": 0,
    "stt_partials_emitted": 0,
    "audio_buffer_seconds": 0.0,
    "audio_rms": 0.0,
    "tokenizer_recreations": 0,
    "tokenizer_last_ms": 0.0,
    "stt_e2e_last": 0.0,
    "stt_e2e_avg": 0.0,
    "stt_e2e_total": 0.0,
    "claude_calls": 0,
    "claude_errors": 0,
    "claude_last_duration": 0.0,
    "claude_avg_duration": 0.0,
    "claude_total_time": 0.0,
    "ws_clients": 0,
    "transcript_queue_size": 0,
    "chunk_seconds": 2,
    "cb_state": "closed",
    "cb_failures": 0,
    "vad_state": "silent",
    "ws_reconnects": 0,
    "last_restore_ms": 0.0,
    "frontend_fps": 0.0,
    "nodes_added_per_min": 0,
    "nodes_removed_per_min": 0,
    "edge_churn_per_min": 0,
    "analysis_queue_depth": 0,
    "claude_last_error": "",
}
metrics_lock = threading.Lock()

# Activity log for post-session operations (recap, clean, synthesis)
_activity_log: list[dict] = []  # capped at 50 entries
_activity_lock = threading.Lock()

# In-flight transcript cleaning jobs, keyed by session_id. Each entry tracks
# progress so the frontend can poll instead of waiting on a single long
# request — the cloudflare tunnel would otherwise kill it at ~100s.
_clean_jobs: dict[str, dict] = {}
_clean_jobs_lock = threading.Lock()


def log_activity(
    event_type: str, session_id: str = "", status: str = "started", detail: str = ""
):
    """Log a post-session operation (recap, clean-transcript, synthesis)."""
    entry = {
        "type": event_type,
        "session_id": session_id,
        "status": status,
        "detail": detail,
        "timestamp": time.time(),
    }
    with _activity_lock:
        _activity_log.append(entry)
        if len(_activity_log) > 50:
            _activity_log.pop(0)


# Per-provider circuit breakers. Each tier in the chain has its own state so
# one provider tripping doesn't lock the others out.
CB_MAX_BACKOFF = 60.0
CB_FAILURE_THRESHOLD = 3
_cb_lock = threading.Lock()
_cb: dict[str, dict] = {
    p: {"state": "closed", "failures": 0, "backoff_until": 0.0, "backoff_secs": 5.0}
    for p in _VALID_PROVIDERS
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


def _cb_snapshot() -> list[dict]:
    """Return a serializable snapshot of every tier's breaker state,
    in chain order."""
    now = time.time()
    with _llm_chain_lock:
        chain = [dict(t) for t in _llm_chain]
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


def _cb_summary_state() -> str:
    """Overall breaker health: 'closed' if any tier is attempting,
    'degraded' if primary is open but a fallback is attempting,
    'open' if every tier is open."""
    with _cb_lock:
        states = [_cb[p]["state"] for p in _VALID_PROVIDERS]
    if all(s == "open" for s in states):
        return "open"
    with _llm_chain_lock:
        primary = _llm_chain[0]["provider"] if _llm_chain else ""
    with _cb_lock:
        primary_state = _cb.get(primary, {}).get("state", "closed")
    if primary_state != "closed":
        return "degraded"
    return "closed"


# Monotonic sequence counter for transcript messages
_seq_counter = 0
_seq_lock = threading.Lock()

# Graph reconciler
reconciler = GraphReconciler()
_current_session_id: str | None = None
_summary: str = ""

# Monotonic session-generation counter. Bumped every time /v1/sessions/new or
# /v1/sessions/{id}/end resets state. In-flight _proxy_claude tasks capture it
# at request time; if it has changed by the time the LLM call returns, the
# response is stale and must be discarded (otherwise a late response can
# repopulate the reconciler with nodes from the previous session).
_session_gen: int = 0
_session_gen_lock = threading.Lock()


def _bump_session_gen() -> int:
    global _session_gen
    with _session_gen_lock:
        _session_gen += 1
        return _session_gen


def _next_seq() -> int:
    global _seq_counter
    with _seq_lock:
        _seq_counter += 1
        return _seq_counter


async def _check_ollama_models():
    """Warn at startup if Ollama is unreachable or configured models aren't pulled."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{HUGIN_BASE_URL}/api/tags", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                data = await resp.json()
    except Exception as e:
        print(f"  WARNING: Ollama not reachable at {HUGIN_BASE_URL}: {e}", file=sys.stderr)
        return
    have = {m.get("name") for m in data.get("models", [])}
    have |= {n.removesuffix(":latest") for n in have}
    for m in sorted({OLLAMA_MODEL, OLLAMA_RECAP_MODEL}):
        if m not in have:
            print(f"  WARNING: Ollama model '{m}' not found — run: ollama pull {m}", file=sys.stderr)


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
    await _check_ollama_models()
    asyncio.create_task(broadcast_loop())
    asyncio.create_task(snapshot_loop())
    print(f"  Server ready — audio arrives from browser via WebSocket")
    print(f"  Main:     http://0.0.0.0:{WS_PORT}/")
    print(f"  Monitor:  http://0.0.0.0:{WS_PORT}/monitor")
    print(f"  Sessions: http://0.0.0.0:{WS_PORT}/sessions\n")
    yield
    await db.close_db()


app = FastAPI(lifespan=lifespan)
WS_PORT = 8765


# ─── Static serving ───
_STATIC = os.path.join(os.path.dirname(__file__), "static")


def _html(name: str) -> FileResponse:
    return FileResponse(os.path.join(_STATIC, name))


@app.get("/livemind-favicon.svg")
async def serve_favicon():
    return FileResponse(
        os.path.join(_STATIC, "livemind-favicon.svg"), media_type="image/svg+xml"
    )


@app.get("/livemind-logo-clean.svg")
async def serve_logo():
    return FileResponse(
        os.path.join(_STATIC, "livemind-logo-clean.svg"), media_type="image/svg+xml"
    )


@app.get("/", response_class=HTMLResponse)
async def serve_main():
    return _html("live-mindmap.html")


@app.get("/monitor", response_class=HTMLResponse)
async def serve_monitor():
    return _html("monitor.html")


@app.get("/doc", response_class=HTMLResponse)
async def serve_doc():
    return _html("doc.html")


@app.get("/doc/admin", response_class=HTMLResponse)
async def serve_doc_admin():
    return _html("doc-admin.html")


@app.get("/sessions", response_class=HTMLResponse)
async def serve_sessions():
    return _html("sessions.html")


@app.get("/sessions/archive", response_class=HTMLResponse)
async def serve_archive():
    return _html("sessions.html")


# Legacy redirects
from starlette.responses import RedirectResponse


@app.get("/admin/sessions")
async def redirect_admin_sessions():
    return RedirectResponse("/sessions")


@app.get("/admin", response_class=HTMLResponse)
async def serve_admin():
    return _html("admin.html")


# ─── REST endpoints ───


@app.get("/v1/sessions")
async def list_sessions(archived: bool = False):
    sessions = await db.list_sessions(archived=archived)
    return JSONResponse(sessions)


@app.post("/v1/sessions/archive")
async def archive_sessions(request: Request):
    """Move sessions to archive."""
    body = await request.json()
    session_ids = body.get("session_ids", [])
    if not session_ids:
        return JSONResponse({"error": "No session IDs provided"}, status_code=400)
    await db.archive_sessions(session_ids)
    return JSONResponse({"ok": True, "archived": len(session_ids)})


@app.post("/v1/sessions/unarchive")
async def unarchive_sessions(request: Request):
    """Move sessions out of archive."""
    body = await request.json()
    session_ids = body.get("session_ids", [])
    if not session_ids:
        return JSONResponse({"error": "No session IDs provided"}, status_code=400)
    await db.unarchive_sessions(session_ids)
    return JSONResponse({"ok": True, "unarchived": len(session_ids)})


@app.delete("/v1/sessions")
async def delete_sessions(request: Request):
    """Permanently delete sessions and all associated data."""
    body = await request.json()
    session_ids = body.get("session_ids", [])
    if not session_ids:
        return JSONResponse({"error": "No session IDs provided"}, status_code=400)
    await db.delete_sessions(session_ids)
    return JSONResponse({"ok": True, "deleted": len(session_ids)})


@app.post("/v1/sessions")
async def create_session(request: Request):
    body = await request.json()
    session_id = str(uuid.uuid4())[:8]
    topic = body.get("topic", "")
    session = await db.create_session(session_id, topic)
    global _current_session_id
    _current_session_id = session_id
    return JSONResponse(session)


@app.post("/v1/sessions/new")
async def new_session(request: Request):
    """End current session (if any) and start a fresh one. Returns the new session."""
    global _current_session_id, _summary, _seq_counter
    body = (
        await request.json()
        if request.headers.get("content-length", "0") != "0"
        else {}
    )
    topic = body.get("topic", "")

    # Bump the session generation FIRST, so any in-flight _proxy_claude task
    # that resumes during our awaits below will see a newer gen and discard
    # its response. Without this, a late LLM response can race with the
    # reconciler clear and repopulate it with stale nodes from the previous
    # session.
    _bump_session_gen()

    # End current session
    if _current_session_id:
        if reconciler.nodes:
            await db.store_snapshot(
                _current_session_id, _seq_counter, reconciler.get_full_state(), "end"
            )
        await db.end_session(_current_session_id, _summary)
    # Reset all state
    reconciler.nodes.clear()
    reconciler.edges.clear()
    reconciler._mention_log.clear()
    reconciler._churn_log.clear()
    _summary = ""
    with _seq_lock:
        _seq_counter = 0
    # Drain any leftover transcript messages from STT queue
    while not transcript_queue.empty():
        try:
            transcript_queue.get_nowait()
        except Exception:
            break
    # Reset session metrics
    with metrics_lock:
        keep = {
            "started_at",
            "ws_clients",
            "cb_state",
            "cb_failures",
            "llm_tiers",
            "llm_serving",
        }
        for k, v in metrics.items():
            if k not in keep:
                if isinstance(v, (int, float)):
                    metrics[k] = 0 if isinstance(v, int) else 0.0
                elif isinstance(v, str):
                    metrics[k] = ""
        metrics["started_at"] = time.time()
    # Create new
    session_id = str(uuid.uuid4())[:8]
    session = await db.create_session(session_id, topic)
    _current_session_id = session_id
    # Belt-and-braces: re-clear reconciler AFTER the create_session await, in
    # case a racing _proxy_claude resumed during the yield and repopulated it.
    # The gen-bump above should already have caused such tasks to discard, but
    # this closes the window with zero cost.
    reconciler.nodes.clear()
    reconciler.edges.clear()
    reconciler._mention_log.clear()
    reconciler._churn_log.clear()
    # Notify all connected frontends
    msg = json.dumps(
        {"type": "session_reset", "session_id": session_id, "topic": topic}
    )
    for ws in list(connected_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            pass
    return JSONResponse(session)


@app.post("/v1/sessions/{session_id}/end")
async def end_session(session_id: str, request: Request):
    """End a session: flush final snapshot, store summary, reset reconciler."""
    global _current_session_id, _summary, _seq_counter
    body = (
        await request.json()
        if request.headers.get("content-length", "0") != "0"
        else {}
    )
    summary = body.get("summary", _summary)
    # Invalidate any in-flight LLM responses before we touch reconciler state.
    _bump_session_gen()
    # Final snapshot
    if reconciler.nodes:
        await db.store_snapshot(
            session_id, _seq_counter, reconciler.get_full_state(), "end"
        )
    await db.end_session(session_id, summary)
    # Reset server state
    reconciler.nodes.clear()
    reconciler.edges.clear()
    reconciler._mention_log.clear()
    reconciler._churn_log.clear()
    _summary = ""
    with _seq_lock:
        _seq_counter = 0
    while not transcript_queue.empty():
        try:
            transcript_queue.get_nowait()
        except Exception:
            break
    if _current_session_id == session_id:
        _current_session_id = None
    # Notify all connected frontends
    msg = json.dumps({"type": "session_ended", "session_id": session_id})
    for ws in list(connected_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            pass
    return JSONResponse({"ok": True, "session_id": session_id})


@app.get("/v1/sessions/synthesis")
async def list_synthesis():
    """List all cross-session synthesis recaps."""
    return JSONResponse(await db.list_synthesis_recaps())


@app.get("/v1/sessions/synthesis/{synthesis_id}")
async def get_synthesis(synthesis_id: int):
    """Get a single cross-session synthesis recap."""
    result = await db.get_synthesis_recap(synthesis_id)
    if not result:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(result)


@app.delete("/v1/sessions/synthesis/{synthesis_id}")
async def delete_synthesis(synthesis_id: int):
    """Delete a cross-session synthesis recap."""
    await db.delete_synthesis_recap(synthesis_id)
    return JSONResponse({"ok": True})


@app.get("/v1/sessions/{session_id}/restore")
async def restore_session(session_id: str, from_seq: int = 0):
    t0 = time.time()
    snapshot = await db.get_latest_snapshot(session_id)
    segments = await db.get_segments_since(session_id, from_seq)
    restore_ms = (time.time() - t0) * 1000
    with metrics_lock:
        metrics["last_restore_ms"] = restore_ms
    return JSONResponse(
        {
            "snapshot": snapshot,
            "segments": segments,
            "restore_ms": round(restore_ms, 1),
        }
    )


@app.post("/v1/sessions/{session_id}/actions")
async def session_action(session_id: str, request: Request):
    body = await request.json()
    action_type = body.get("action")
    payload = body.get("payload", {})
    await db.store_action(session_id, action_type, payload)
    graph = reconciler.apply_action(action_type, payload)
    # Broadcast updated graph to all connected clients
    msg = json.dumps(
        {"type": "graph_update", "graph": graph, "session_id": _current_session_id}
    )
    for ws in list(connected_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            pass
    return JSONResponse({"ok": True, "graph": graph})


@app.get("/v1/sessions/{session_id}/snapshots")
async def get_session_snapshots_endpoint(session_id: str):
    """Get all snapshots for playback."""
    snapshots = await db.get_session_snapshots(session_id)
    return JSONResponse(
        {
            "session_id": session_id,
            "count": len(snapshots),
            "snapshots": snapshots,
        }
    )


_export_tasks: dict[str, dict] = {}  # session_id -> {task, status, path, error}


@app.post("/v1/sessions/{session_id}/export/{fmt}")
async def start_export(session_id: str, fmt: str, request: Request):
    """Start a PDF or video export. Returns immediately; poll status endpoint."""
    if fmt not in ("pdf", "video"):
        return JSONResponse(
            {"error": "Format must be 'pdf' or 'video'"}, status_code=400
        )

    task_key = f"{session_id}_{fmt}"
    if task_key in _export_tasks and _export_tasks[task_key].get("status") == "running":
        return JSONResponse(
            {"status": "running", "message": "Export already in progress"}
        )

    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    import tempfile

    ext = "pdf" if fmt == "pdf" else "mp4"
    outfile = os.path.join(tempfile.gettempdir(), f"livemind-{session_id}.{ext}")

    _export_tasks[task_key] = {"status": "running", "path": outfile, "error": None}

    async def run_export():
        try:
            from export import export_pdf, export_video

            if fmt == "pdf":
                await export_pdf(session_id, outfile)
            else:
                await export_video(
                    session_id,
                    outfile,
                    speed=body.get("speed", 2.0),
                    max_hold=body.get("max_hold", 3.0),
                )
            _export_tasks[task_key]["status"] = "done"
        except Exception as e:
            _export_tasks[task_key]["status"] = "error"
            _export_tasks[task_key]["error"] = str(e)
            print(f"Export error ({fmt} {session_id}): {e}", file=sys.stderr)

    asyncio.create_task(run_export())
    return JSONResponse({"status": "started", "format": fmt})


@app.get("/v1/sessions/{session_id}/export/{fmt}/status")
async def export_status(session_id: str, fmt: str):
    """Check export status."""
    task_key = f"{session_id}_{fmt}"
    info = _export_tasks.get(task_key)
    if not info:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return JSONResponse({"status": info["status"], "error": info.get("error")})


@app.get("/v1/sessions/{session_id}/export/{fmt}/download")
async def download_export(session_id: str, fmt: str):
    """Download the exported file."""
    task_key = f"{session_id}_{fmt}"
    info = _export_tasks.get(task_key)
    if not info or info["status"] != "done":
        return JSONResponse({"error": "Export not ready"}, status_code=404)

    ext = "pdf" if fmt == "pdf" else "mp4"
    media = "application/pdf" if fmt == "pdf" else "video/mp4"
    return FileResponse(
        info["path"],
        media_type=media,
        filename=f"livemind-{session_id}.{ext}",
    )


@app.get("/v1/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """Get session detail: transcript, final snapshot, and recap.
    Works for both live and archived sessions."""
    session_meta = await db.get_session(session_id)
    if session_meta is None:
        return JSONResponse(
            {"error": f"Session {session_id} not found"}, status_code=404
        )
    transcript = await db.get_session_transcript(session_id)
    snapshot = await db.get_latest_snapshot(session_id)
    recap = await db.get_recap(session_id)
    return JSONResponse(
        {
            "session": session_meta,
            "transcript": transcript,
            "snapshot": snapshot,
            "recap": recap,
        }
    )


@app.post("/v1/sessions/{session_id}/recap")
async def generate_recap(session_id: str):
    """Generate an AI recap for a session (on-demand, server-side LLM call)."""
    log_activity("recap", session_id, "started")
    segments = await db.get_session_transcript(session_id)
    if not segments:
        log_activity("recap", session_id, "error", "No transcript found")
        return JSONResponse({"error": "No transcript found"}, status_code=404)

    full_text = " ".join(seg.get("cleaned_text") or seg["text"] for seg in segments)

    # Detect session language from STT metadata (majority vote)
    lang_counts: dict[str, int] = {}
    for seg in segments:
        lang = seg.get("stt_language", "")
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    session_lang = max(lang_counts, key=lang_counts.get) if lang_counts else "en"

    # Get final snapshot — full graph with nodes and edges
    snapshot = await db.get_latest_snapshot(session_id)
    graph_context = ""
    if snapshot and snapshot.get("graph"):
        graph = snapshot["graph"]
        nodes = graph.get("nodes", {})
        edges = graph.get("edges", [])
        # Build structured graph description
        active_nodes = {
            nid: n for nid, n in nodes.items() if n.get("state") == "active"
        }
        if active_nodes:
            node_lines = [
                f"  - {n.get('label', nid)} (category: {n.get('group', 'unknown')})"
                for nid, n in active_nodes.items()
            ]
            # Map node IDs to labels for edge descriptions
            id_to_label = {nid: n.get("label", nid) for nid, n in nodes.items()}
            edge_lines = []
            for e in edges:
                src = e.get("source", "")
                tgt = e.get("target", "")
                lbl = e.get("label", "relates to")
                src_label = id_to_label.get(src, src)
                tgt_label = id_to_label.get(tgt, tgt)
                if src in active_nodes or tgt in active_nodes:
                    edge_lines.append(f"  - {src_label} --[{lbl}]--> {tgt_label}")
            graph_context = "\n\nFINAL KNOWLEDGE GRAPH:\nNodes:\n" + "\n".join(
                node_lines
            )
            if edge_lines:
                graph_context += "\n\nEdges (relationships):\n" + "\n".join(edge_lines)

    # Compute stats
    duration_minutes = 0.0
    if len(segments) >= 2:
        duration_minutes = (segments[-1]["timestamp"] - segments[0]["timestamp"]) / 60
    stats = {
        "total_segments": len(segments),
        "total_chars": len(full_text),
        "duration_minutes": round(duration_minutes, 1),
    }

    # Truncate if very long
    max_chars = _RECAP_MAX_CHARS
    transcript_for_recap = full_text[:max_chars]
    if len(full_text) > max_chars:
        transcript_for_recap += (
            f"\n\n[Transcript truncated at {max_chars} chars out of {len(full_text)}]"
        )

    lang_name = {
        "en": "English",
        "no": "Norwegian",
        "sv": "Swedish",
        "da": "Danish",
        "de": "German",
        "fr": "French",
    }.get(session_lang, "English")

    system_prompt = f"""You generate structured session recap documents that surface insight, not meeting minutes.
You have access to both the full transcript AND the final knowledge graph (nodes and their relationships).

Return ONLY valid JSON with this exact structure:
{{
  "elevator_pitch": "2-3 sentences a participant could say out loud after the session about what it means. First person plural is fine. Written in {lang_name}.",
  "non_obvious_connections": [
    {{"topics": ["Topic A", "Topic B"], "insight": "What the link reveals that wasn't stated explicitly."}}
  ],
  "retain": ["First thing worth remembering a week from now.", "Second.", "Third."],
  "contradictions": ["Where the discussion diverged from stated positions or earlier claims."],
  "summary": "One paragraph reference summary.",
  "decisions": ["Decisions made, if any."],
  "open_threads": ["Unresolved tensions or threads worth following up."]
}}

LANGUAGE: Write ALL fields in {lang_name}. Every field — elevator_pitch, retain, non_obvious_connections insights, contradictions, summary, decisions, open_threads — must be written in {lang_name}. Do NOT switch to English for any field.

Rules:
- elevator_pitch: Write in {lang_name}, in the voice of a participant (first person). 2-3 sentences someone could actually say out loud.
- non_obvious_connections: 0 to 3 items ONLY. Draw on the knowledge graph edges to find links participants likely didn't notice in the room. Return an EMPTY ARRAY rather than fabricate connections. One real connection beats three plausible ones. Write the insight in {lang_name}.
- retain: EXACTLY 3 items in {lang_name}. The three ideas that should survive the week. Forces you to prioritize.
- contradictions: Often empty — that's fine. Only include when there's genuine divergence between what was said vs. stated positions, slides, or earlier claims. Return an EMPTY ARRAY if none. Write in {lang_name}.
- summary: One concise paragraph in {lang_name}. This is reference material, not the headline.
- decisions: Empty array if none. Name the people involved where possible. Write in {lang_name}.
- open_threads: Unresolved tensions or questions worth following up. Empty array if none. Write in {lang_name}.
- Prefer empty arrays over speculation. Never invent or pad.
- Use specific names, not "the user" or "the participant"."""

    user_prompt = f"SESSION TRANSCRIPT:\n\n{transcript_for_recap}{graph_context}\n\nGenerate the recap."

    model = OLLAMA_RECAP_MODEL
    ollama_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "num_predict": 4096,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }

    # Retry once on parse failure with stricter reminder
    max_attempts = 2
    last_error = None

    for attempt in range(max_attempts):
        try:
            if attempt > 0:
                # Add stricter reminder on retry
                ollama_body["messages"] = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt
                        + "\n\nIMPORTANT: Return ONLY the raw JSON object. No markdown, no explanation, no code fences.",
                    },
                ]

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{HUGIN_BASE_URL}/api/chat",
                    headers={"Content-Type": "application/json"},
                    json=ollama_body,
                    timeout=aiohttp.ClientTimeout(total=180),
                    ssl=False,
                ) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        err = data.get("error", "") or str(data)
                        return JSONResponse(
                            {"error": f"Ollama: {err}"}, status_code=502
                        )

            raw_text = data.get("message", {}).get("content", "")
            # Strip markdown code fences if present
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            recap = json.loads(cleaned.strip())

            # Add metadata
            recap["language"] = session_lang
            recap["schema_version"] = 2
            recap["transcript_stats"] = stats

            await db.store_recap(session_id, recap, model)
            log_activity(
                "recap",
                session_id,
                "completed",
                f"{len(segments)} segments, model={model}",
            )
            return JSONResponse(
                {"recap": recap, "model": model, "created_at": time.time()}
            )

        except json.JSONDecodeError as e:
            last_error = e
            print(f"  Recap parse attempt {attempt + 1} failed: {e}", file=sys.stderr)
            if attempt < max_attempts - 1:
                continue
            # Final failure — store error state
            error_recap = {
                "schema_version": 2,
                "language": session_lang,
                "error": f"Failed to parse LLM response after {max_attempts} attempts: {str(last_error)}",
                "raw_response": raw_text[:2000],
                "transcript_stats": stats,
            }
            await db.store_recap(session_id, error_recap, model)
            log_activity("recap", session_id, "error", str(last_error))
            return JSONResponse(
                {"error": f"Failed to parse LLM response: {last_error}"},
                status_code=502,
            )
        except Exception as e:
            log_activity("recap", session_id, "error", str(e))
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


# ─── Transcript cleaning ───


@app.get("/v1/sessions/{session_id}/clean-transcript/status")
async def clean_transcript_status(session_id: str):
    """Return the current state of an in-flight (or recently finished) clean
    job for this session. Frontend polls this while a job is running."""
    with _clean_jobs_lock:
        job = _clean_jobs.get(session_id)
        snapshot = dict(job) if job else None
    if snapshot is None:
        return JSONResponse({"status": "idle"})
    return JSONResponse(snapshot)


@app.post("/v1/sessions/{session_id}/clean-transcript")
async def clean_transcript(session_id: str):
    """Kick off a transcript-cleaning job in the background. Returns 202
    immediately so the cloudflare tunnel doesn't time out; the frontend
    polls /clean-transcript/status for progress."""
    # Refuse if a job is already running for this session
    with _clean_jobs_lock:
        existing = _clean_jobs.get(session_id)
        if existing and existing.get("status") == "running":
            return JSONResponse(
                {"status": "running", "progress": existing.get("progress", {})},
                status_code=202,
            )

    segments = await db.get_session_transcript(session_id)
    if not segments:
        return JSONResponse({"error": "No transcript found"}, status_code=404)

    # Record an initial job state and fire the background task
    with _clean_jobs_lock:
        _clean_jobs[session_id] = {
            "status": "running",
            "progress": {"done": 0, "total": 0},
            "started_at": time.time(),
        }
    asyncio.create_task(_run_clean_job(session_id, segments))
    return JSONResponse(
        {"status": "running", "progress": {"done": 0, "total": 0}}, status_code=202
    )


async def _run_clean_job(session_id: str, segments: list[dict]):
    """Background worker for clean-transcript. Updates _clean_jobs as it
    progresses. Never raises — all errors land in the job state."""
    try:
        result = await _clean_transcript_impl(session_id, segments)
        with _clean_jobs_lock:
            _clean_jobs[session_id] = {
                "status": "done",
                "result": result,
                "finished_at": time.time(),
            }
    except Exception as e:
        print(
            f"  Clean job {session_id} crashed: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        log_activity("clean", session_id, "error", f"{type(e).__name__}: {e}")
        with _clean_jobs_lock:
            _clean_jobs[session_id] = {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "finished_at": time.time(),
            }


def _clean_job_progress(session_id: str, done: int, total: int):
    with _clean_jobs_lock:
        job = _clean_jobs.get(session_id)
        if job and job.get("status") == "running":
            job["progress"] = {"done": done, "total": total}


async def _clean_transcript_impl(session_id: str, segments: list[dict]) -> dict:
    """The actual cleaning work. Separated from the HTTP layer so it can run
    as a background task."""
    log_activity("clean", session_id, "started")

    # Detect language from segments
    lang_counts: dict[str, int] = {}
    for seg in segments:
        lang = seg.get("stt_language", "")
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    session_lang = max(lang_counts, key=lang_counts.get) if lang_counts else "en"
    lang_name = {
        "en": "English",
        "no": "Norwegian",
        "sv": "Swedish",
        "da": "Danish",
        "de": "German",
        "fr": "French",
    }.get(session_lang, "English")

    # Pre-filter: catch repetition loops and obvious garbage before LLM
    import re as _re

    def pre_clean(text: str) -> str:
        """Catch STT hallucination loops and garbage that the LLM shouldn't waste tokens on."""
        words = text.split()
        if len(words) >= 10:
            # Find the longest repetition loop and replace it
            best = None  # (start, end, count)
            for plen in (1, 2, 3):
                for start in range(len(words)):
                    if start + plen * 5 > len(words):
                        break
                    pattern = (
                        " ".join(words[start : start + plen]).lower().rstrip(".,!?")
                    )
                    if not pattern:
                        continue
                    pos = start
                    while pos + plen <= len(words):
                        chunk = " ".join(words[pos : pos + plen]).lower().rstrip(".,!?")
                        if chunk == pattern:
                            pos += plen
                        else:
                            break
                    count = (pos - start) // plen
                    if count >= 5:
                        span = pos - start
                        if not best or span > (best[1] - best[0]):
                            best = (start, pos, count)
            if best:
                start, end, _ = best
                before = " ".join(words[:start]).strip()
                after = " ".join(words[end:]).strip()
                parts = [p for p in [before, "[inaudible]", after] if p]
                return pre_clean(" ".join(parts))
        # Single repeated character sequences
        text = _re.sub(r"(.)\1{20,}", "[inaudible]", text)
        return text

    for seg in segments:
        seg["text"] = pre_clean(seg["text"])

    system_prompt = f"""You are a transcript cleaner. You receive raw speech-to-text segments and fix obvious transcription errors.

Rules:
- Fix misspelled words, garbled text, and wrong language fragments
- Add missing punctuation and capitalization
- Fix obvious name misspellings (be consistent across segments)
- Preserve the speaker's original words — do NOT rephrase, summarize, or paraphrase
- If a segment contains "[inaudible]", keep that marker as-is
- If a segment is mostly noise or completely unintelligible, replace it with "[inaudible]"
- If a segment is fine, return it unchanged
- The transcript is in {lang_name}. Some segments may contain English terms or code-switching — preserve those naturally
- Return EXACTLY the same number of items as the input, in the same order
- Return ONLY a JSON array of strings, one per segment: ["cleaned segment 1", "cleaned segment 2", ...]
- Do NOT add any explanation, just the JSON array"""

    # Process in chunks with capped concurrency so we don't queue requests
    # behind each other on a single local Ollama.
    CHUNK_SIZE = 40
    CHUNK_TIMEOUT = 120
    MAX_CONCURRENCY = OLLAMA_MAX_CONCURRENCY
    model = OLLAMA_RECAP_MODEL

    chunks = [segments[i : i + CHUNK_SIZE] for i in range(0, len(segments), CHUNK_SIZE)]
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def clean_one(
        idx: int, chunk: list[dict]
    ) -> tuple[int, list[dict], str | None]:
        """Clean a single chunk. Returns (idx, cleaned_items, error_msg).
        On any failure, falls back to originals and reports the error — never
        raises, so one chunk can't abort the whole run."""
        texts = [seg["text"] for seg in chunk]
        user_prompt = "Clean these transcript segments:\n" + json.dumps(
            texts, ensure_ascii=False
        )
        ollama_body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "format": "json",
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"temperature": 0, "num_predict": 4096, "num_ctx": OLLAMA_NUM_CTX},
        }
        fallback = [{"seq": seg["seq"], "cleaned_text": seg["text"]} for seg in chunk]

        async with sem:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{HUGIN_BASE_URL}/api/chat",
                        headers={"Content-Type": "application/json"},
                        json=ollama_body,
                        timeout=aiohttp.ClientTimeout(total=CHUNK_TIMEOUT),
                        ssl=False,
                    ) as resp:
                        if resp.status != 200:
                            body = await resp.text()
                            return idx, fallback, f"http {resp.status}: {body[:200]}"
                        data = await resp.json()
            except asyncio.TimeoutError:
                return idx, fallback, f"timeout after {CHUNK_TIMEOUT}s"
            except Exception as e:
                return idx, fallback, f"{type(e).__name__}: {e}"

        raw_text = data.get("message", {}).get("content", "")
        cleaned_str = raw_text.strip()
        if cleaned_str.startswith("```"):
            cleaned_str = (
                cleaned_str.split("\n", 1)[1]
                if "\n" in cleaned_str
                else cleaned_str[3:]
            )
        if cleaned_str.endswith("```"):
            cleaned_str = cleaned_str[:-3]

        try:
            parsed = json.loads(cleaned_str.strip())
        except json.JSONDecodeError as e:
            return idx, fallback, f"json parse: {e}"

        # LLM may wrap the array in an object — unwrap if so
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break

        if not isinstance(parsed, list) or len(parsed) != len(chunk):
            got = len(parsed) if isinstance(parsed, list) else "non-list"
            return idx, fallback, f"length mismatch: expected {len(chunk)} got {got}"

        items = []
        for j, seg in enumerate(chunk):
            ct = parsed[j] if isinstance(parsed[j], str) else seg["text"]
            items.append({"seq": seg["seq"], "cleaned_text": ct})
        return idx, items, None

    # Initial progress state now that we know how many chunks we have
    total_chunks = len(chunks)
    done_count = 0
    _clean_job_progress(session_id, done_count, total_chunks)

    # Launch all chunks as tasks, reporting progress as they complete.
    # The semaphore caps real concurrency; the gather just lets us observe
    # completions for UI updates.
    tasks = [asyncio.create_task(clean_one(i, c)) for i, c in enumerate(chunks)]
    results: list[tuple[int, list[dict], str | None]] = []
    for fut in asyncio.as_completed(tasks):
        r = await fut
        results.append(r)
        done_count += 1
        _clean_job_progress(session_id, done_count, total_chunks)

    results.sort(key=lambda r: r[0])
    all_cleaned: list[dict] = []
    failed_chunks: list[dict] = []
    for idx, items, err in results:
        all_cleaned.extend(items)
        if err:
            print(f"  Clean chunk {idx + 1}: {err}", file=sys.stderr)
            failed_chunks.append({"chunk": idx + 1, "error": err})

    await db.store_cleaned_segments(session_id, all_cleaned)

    changed = sum(
        1 for c, seg in zip(all_cleaned, segments) if c["cleaned_text"] != seg["text"]
    )

    status_detail = f"{changed}/{len(segments)} changed, model={model}"
    if failed_chunks:
        status_detail += f", {len(failed_chunks)} chunks failed"
    log_activity("clean", session_id, "completed", status_detail)

    return {
        "ok": True,
        "total_segments": len(segments),
        "changed": changed,
        "model": model,
        "failed_chunks": failed_chunks,
    }


# ─── Cross-session synthesis ───


@app.post("/v1/sessions/synthesis")
async def generate_synthesis(request: Request):
    """Generate a cross-session synthesis recap from multiple sessions' individual recaps."""
    log_activity("synthesis", "", "started")
    body = await request.json()
    session_ids = body.get("session_ids", [])
    if len(session_ids) < 2:
        return JSONResponse({"error": "Need at least 2 sessions"}, status_code=400)

    # Load each session's recap + graph
    sessions_data = []
    missing_recaps = []
    for sid in session_ids:
        recap = await db.get_recap(sid)
        if not recap:
            missing_recaps.append(sid)
            continue
        snapshot = await db.get_latest_snapshot(sid)
        # Get session metadata
        all_sessions = await db.list_sessions()
        meta = next((s for s in all_sessions if s["id"] == sid), {})
        sessions_data.append(
            {
                "id": sid,
                "topic": meta.get("topic", ""),
                "created_at": meta.get("created_at", 0),
                "ended_at": meta.get("ended_at"),
                "recap": recap["recap"],
                "snapshot": snapshot,
            }
        )

    if missing_recaps:
        return JSONResponse(
            {
                "error": f"Sessions missing recaps: {', '.join(missing_recaps)}. Generate individual recaps first."
            },
            status_code=400,
        )

    # Detect language (majority from session recaps)
    lang_counts: dict[str, int] = {}
    for sd in sessions_data:
        lang = sd["recap"].get("language", "en")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    session_lang = max(lang_counts, key=lang_counts.get) if lang_counts else "en"
    lang_name = {
        "en": "English",
        "no": "Norwegian",
        "sv": "Swedish",
        "da": "Danish",
        "de": "German",
        "fr": "French",
    }.get(session_lang, "English")

    # Build per-session blocks
    session_blocks = []
    for i, sd in enumerate(sessions_data, 1):
        duration = ""
        if sd["ended_at"] and sd["created_at"]:
            mins = round((sd["ended_at"] - sd["created_at"]) / 60)
            duration = f" ({mins} min)"

        r = sd["recap"]
        block = f'SESSION {i}: "{sd["topic"] or "Untitled"}"{duration}\n'
        block += f'ID: {sd["id"]}\n'

        # Include recap highlights
        if r.get("elevator_pitch"):
            block += f'PITCH: {r["elevator_pitch"]}\n'
        if r.get("retain"):
            block += (
                "KEY TAKEAWAYS:\n"
                + "\n".join(f"  - {item}" for item in r["retain"])
                + "\n"
            )
        if r.get("non_obvious_connections"):
            block += "CONNECTIONS:\n"
            for conn in r["non_obvious_connections"]:
                topics = " ↔ ".join(conn.get("topics", []))
                block += f'  - {topics}: {conn.get("insight", "")}\n'
        if r.get("summary"):
            block += f'SUMMARY: {r["summary"]}\n'
        if r.get("contradictions"):
            block += (
                "CONTRADICTIONS:\n"
                + "\n".join(f"  - {c}" for c in r["contradictions"])
                + "\n"
            )

        # Include graph
        if sd["snapshot"] and sd["snapshot"].get("graph"):
            graph = sd["snapshot"]["graph"]
            nodes = graph.get("nodes", {})
            edges = graph.get("edges", [])
            active = {nid: n for nid, n in nodes.items() if n.get("state") == "active"}
            if active:
                id_to_label = {nid: n.get("label", nid) for nid, n in nodes.items()}
                block += (
                    "GRAPH NODES: "
                    + ", ".join(n.get("label", nid) for nid, n in active.items())
                    + "\n"
                )
                edge_strs = []
                for e in edges:
                    src = id_to_label.get(e.get("source", ""), e.get("source", ""))
                    tgt = id_to_label.get(e.get("target", ""), e.get("target", ""))
                    edge_strs.append(f'{src} --[{e.get("label", "")}]--> {tgt}')
                if edge_strs:
                    block += "GRAPH EDGES: " + "; ".join(edge_strs) + "\n"

        session_blocks.append(block)

    all_sessions_text = "\n---\n\n".join(session_blocks)

    system_prompt = f"""You synthesize insights across multiple session recaps from the same event or day.
You have access to each session's recap (elevator pitch, key takeaways, connections, summary) AND its knowledge graph.

Your job is to find the threads that run BETWEEN sessions — ideas that evolved, echoed, or contradicted each other across different conversations.

Return ONLY valid JSON with this exact structure:
{{
  "elevator_pitch": "The day/event in 2-3 sentences. What would a participant tell a colleague? Written in {lang_name}, first person plural.",
  "cross_connections": [
    {{"sessions": ["id1", "id2"], "topics": ["Topic A", "Topic B"], "insight": "What the link across these sessions reveals."}}
  ],
  "evolution": ["How an idea or theme evolved from one session to the next."],
  "tensions": ["Where one session contradicted or complicated another's conclusions."],
  "synthesis": "2-3 paragraph narrative of the day's arc — what emerged across all sessions taken together.",
  "language": "{session_lang}"
}}

Rules:
- elevator_pitch: Written in {lang_name}, first person. Something a participant would actually say.
- cross_connections: 0 to 5 items. Reference the specific session IDs. Draw on graph edges across sessions to find themes that link different conversations. Return an EMPTY ARRAY rather than fabricate.
- evolution: How ideas developed across the timeline of sessions. Empty array if nothing evolved.
- tensions: Where sessions disagreed or complicated each other. Often empty — that's fine.
- synthesis: A narrative, not a list. This is the "big picture" view of the day.
- Prefer empty arrays over speculation. Never invent connections."""

    user_prompt = f"{all_sessions_text}\n\nGenerate the cross-session synthesis."

    model = OLLAMA_RECAP_MODEL
    ollama_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "num_predict": 4096,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{HUGIN_BASE_URL}/api/chat",
                headers={"Content-Type": "application/json"},
                json=ollama_body,
                timeout=aiohttp.ClientTimeout(total=300),
                ssl=False,
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    err = data.get("error", "") or str(data)
                    return JSONResponse({"error": f"Ollama: {err}"}, status_code=502)

        raw_text = data.get("message", {}).get("content", "")
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        synthesis = json.loads(cleaned.strip())
        synthesis["schema_version"] = 1
        synthesis["session_count"] = len(session_ids)

        row_id = await db.store_synthesis(session_ids, synthesis, model)
        log_activity(
            "synthesis",
            ",".join(session_ids),
            "completed",
            f"{len(session_ids)} sessions, model={model}",
        )
        return JSONResponse(
            {
                "id": row_id,
                "session_ids": session_ids,
                "recap": synthesis,
                "model": model,
                "created_at": time.time(),
            }
        )

    except json.JSONDecodeError as e:
        return JSONResponse(
            {"error": f"Failed to parse LLM response: {e}"}, status_code=502
        )
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.get("/v1/metrics")
async def get_metrics_rest():
    with metrics_lock:
        m = {**metrics, "uptime": time.time() - metrics["started_at"]}
    churn = reconciler.get_churn_metrics()
    m.update(churn)
    m["current_session_id"] = _current_session_id
    m["active_nodes"] = len(
        [ns for ns in reconciler.nodes.values() if ns.state == "active"]
    )
    with _llm_chain_lock:
        head = dict(_llm_chain[0]) if _llm_chain else {"provider": "", "model": ""}
    m["llm_provider"] = head.get("provider", "")
    m["llm_model"] = head.get("model", "")
    m["llm_tiers"] = _cb_snapshot()
    m["llm_serving"] = _last_serving_provider
    stt = get_stt_config()
    m["stt_backend"] = stt["backend"]
    m["stt_remote_url"] = stt.get("remote_url", "")
    return JSONResponse(m)


# ─── LLM provider endpoints ───
@app.get("/v1/llm/providers")
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

    with _llm_chain_lock:
        chain = [dict(t) for t in _llm_chain]
    tiers_state = _cb_snapshot()
    return JSONResponse(
        {
            "providers": providers,
            "chain": chain,
            "tiers": tiers_state,
            "serving": _last_serving_provider,
            # Back-compat: legacy clients still read `active`.
            "active": chain[0] if chain else {},
        }
    )


def _validate_tier(tier: dict) -> str | None:
    """Return None if OK, else error string."""
    provider = (tier.get("provider") or "").strip()
    model = (tier.get("model") or "").strip()
    if provider not in _VALID_PROVIDERS:
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


@app.post("/v1/llm/active")
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
            err = _validate_tier(raw)
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

        with _llm_chain_lock:
            old = [dict(t) for t in _llm_chain]
            _llm_chain.clear()
            _llm_chain.extend(new_chain)

        print(
            f"  LLM chain: {[f'{t['provider']}/{t['model']}' for t in old]} → {[f'{t['provider']}/{t['model']}' for t in new_chain]}"
        )
        _publish_llm_state()
        return JSONResponse({"chain": new_chain, "tiers": _cb_snapshot()})

    # Back-compat single-tier update
    provider = (body.get("provider") or "").strip()
    model = (body.get("model") or "").strip()
    err = _validate_tier({"provider": provider, "model": model})
    if err:
        return JSONResponse({"error": err}, status_code=400)

    with _llm_chain_lock:
        found = False
        for tier in _llm_chain:
            if tier["provider"] == provider:
                tier["model"] = model
                found = True
                break
        if not found:
            _llm_chain.insert(0, {"provider": provider, "model": model})
        new_chain = [dict(t) for t in _llm_chain]

    print(f"  LLM chain: updated tier {provider} → {model}")
    _publish_llm_state()
    return JSONResponse(
        {"chain": new_chain, "active": {"provider": provider, "model": model}}
    )


# ─── STT backend endpoints ───
@app.get("/v1/stt/backends")
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


@app.post("/v1/stt/active")
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


# ─── LLM cost tracking ───
# Pricing per million tokens (input, output)
_LLM_PRICING = {
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.0-flash": (0.10, 0.40),
}
_DEFAULT_PRICING = (0.0, 0.0)  # self-hosted = free


def _extract_usage(data: dict, provider: str) -> dict:
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


# ─── LLM proxy ───
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


def _extract_graph_json(raw_text: str) -> dict:
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


async def _broadcast_llm_response(status_code, data, req_id):
    """Send LLM response to ALL connected clients (guards against disconnected sockets)."""
    payload = json.dumps(
        {
            "type": "claude_response",
            "status": status_code,
            "data": data,
            "req_id": req_id,
        }
    )
    for ws in list(connected_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            pass


async def _call_provider(provider: str, body: dict) -> tuple[int, dict]:
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

    with _llm_chain_lock:
        chain = [dict(t) for t in _llm_chain]

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
            status, data = await _call_provider(provider, tier_body)
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


def _publish_llm_state() -> None:
    """Copy chain + breaker state into the metrics dict so the monitor can
    render a live view of which tier is currently serving."""
    tiers = _cb_snapshot()
    with metrics_lock:
        metrics["llm_tiers"] = tiers
        metrics["llm_serving"] = _last_serving_provider
        metrics["cb_state"] = _cb_summary_state()
        total_failures = sum(t.get("failures", 0) for t in tiers)
        metrics["cb_failures"] = total_failures


async def _proxy_claude(websocket: WebSocket, req: dict):
    """Proxy an LLM request server-side, walking the fallback chain."""
    global _summary
    _req_session_id = _current_session_id
    _req_gen = _session_gen

    t0 = time.time()
    try:
        body = req.get("body", {})
        status_code, data, provider, model = await call_llm_chain(body)
        dt = time.time() - t0

        # Token usage / cost attributed to whichever tier actually served.
        if provider and model:
            usage = _extract_usage(data, provider)
            pricing = _LLM_PRICING.get(model, _DEFAULT_PRICING)
            call_cost = (
                usage["input"] * pricing[0] + usage["output"] * pricing[1]
            ) / 1_000_000
        else:
            usage = {"input": 0, "output": 0}
            call_cost = 0.0

        with metrics_lock:
            metrics["claude_calls"] += 1
            metrics["claude_last_duration"] = dt
            metrics["claude_total_time"] += dt
            metrics["claude_avg_duration"] = (
                metrics["claude_total_time"] / metrics["claude_calls"]
            )
            if provider:
                metrics["llm_input_tokens"] = (
                    metrics.get("llm_input_tokens", 0) + usage["input"]
                )
                metrics["llm_output_tokens"] = (
                    metrics.get("llm_output_tokens", 0) + usage["output"]
                )
                metrics["llm_session_cost"] = (
                    metrics.get("llm_session_cost", 0.0) + call_cost
                )
                metrics["llm_last_cost"] = call_cost
                metrics["llm_last_input_tokens"] = usage["input"]
                metrics["llm_last_output_tokens"] = usage["output"]

        _publish_llm_state()

        if status_code == 200:
            cost_str = f"${call_cost:.4f}" if call_cost > 0 else "free"
            print(
                f"  LLM [{provider}/{model}]: 200 OK ({dt:.1f}s, {usage['input']}+{usage['output']} tok, {cost_str})"
            )
            with metrics_lock:
                metrics["claude_last_error"] = ""

            try:
                raw_text = "".join(c.get("text", "") for c in data.get("content", []))
                parsed = _extract_graph_json(raw_text)
                if parsed.get("nodes") and parsed.get("edges") is not None:
                    # Generation check catches the None→None race that the
                    # session_id check alone can't: if a session reset happened
                    # while the LLM was responding, _session_gen has advanced
                    # and we must discard this response.
                    if _session_gen != _req_gen:
                        print(
                            f"  LLM: discarding stale response (gen {_req_gen} → {_session_gen}, session {_req_session_id} → {_current_session_id})"
                        )
                        return
                    if _current_session_id != _req_session_id:
                        print(
                            f"  LLM: discarding stale response (session changed {_req_session_id} → {_current_session_id})"
                        )
                        return
                    n_before = len(reconciler.nodes)
                    graph = reconciler.reconcile(parsed)
                    n_after = len(reconciler.nodes)
                    if parsed.get("summary"):
                        _summary = parsed["summary"]
                    if _current_session_id:
                        await db.store_snapshot(
                            _current_session_id,
                            _seq_counter,
                            reconciler.get_full_state(),
                            "analysis",
                        )
                    data = {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        **graph,
                                        "summary": _summary,
                                    }
                                ),
                            }
                        ],
                    }
                    churn = reconciler.get_churn_metrics()
                    with metrics_lock:
                        metrics["nodes_added_per_min"] = churn["nodes_added_per_min"]
                        metrics["nodes_removed_per_min"] = churn[
                            "nodes_removed_per_min"
                        ]
                        metrics["edge_churn_per_min"] = churn["edge_churn_per_min"]
                        metrics["llm_parse_ok"] = metrics.get("llm_parse_ok", 0) + 1
                        metrics["llm_last_node_count"] = len(parsed.get("nodes", []))
                    print(
                        f"  LLM: parsed {len(parsed['nodes'])} nodes, {len(parsed.get('edges',[]))} edges (reconciler: {n_before}→{n_after})"
                    )

                    graph_msg = json.dumps(
                        {
                            "type": "graph_update",
                            "graph": graph,
                            "session_id": _current_session_id,
                        }
                    )
                    for ws in list(connected_clients):
                        try:
                            await ws.send_text(graph_msg)
                        except Exception:
                            pass
                else:
                    print(
                        f"  LLM: parsed JSON but missing nodes/edges keys",
                        file=sys.stderr,
                    )
                    with metrics_lock:
                        metrics["llm_parse_no_graph"] = (
                            metrics.get("llm_parse_no_graph", 0) + 1
                        )
            except (json.JSONDecodeError, KeyError) as parse_err:
                print(f"  LLM: response parse error: {parse_err}", file=sys.stderr)
                print(f"  LLM: raw text: {raw_text[:500]}", file=sys.stderr)
                with metrics_lock:
                    metrics["llm_parse_fail"] = metrics.get("llm_parse_fail", 0) + 1
                    metrics["llm_last_raw_fail"] = raw_text[:500]
        else:
            err_msg = ""
            if isinstance(data, dict):
                err = data.get("error")
                if isinstance(err, dict):
                    err_msg = err.get("message", "")
                elif isinstance(err, str):
                    err_msg = err
            with metrics_lock:
                metrics["claude_errors"] += 1
                metrics["claude_last_error"] = (
                    f"{status_code}: {err_msg or 'chain exhausted'}"
                )

        await _broadcast_llm_response(status_code, data, req.get("req_id"))
    except Exception as e:
        dt = time.time() - t0
        print(
            f"  LLM: EXCEPTION in proxy ({dt:.1f}s) — {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        with metrics_lock:
            metrics["claude_calls"] += 1
            metrics["claude_errors"] += 1
            metrics["claude_last_error"] = f"{type(e).__name__}: {e}"
        _publish_llm_state()
        await _broadcast_llm_response(500, {"error": str(e)}, req.get("req_id"))


# ─── Audio chunk handler ───
async def _handle_audio_chunk(audio_arr: np.ndarray, source_rate: int):
    """Process an audio chunk from the browser: STT → store segment → broadcast."""
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            stt_worker.transcribe_audio_chunk,
            audio_arr,
            source_rate,
            metrics,
            metrics_lock,
        )
    except Exception as e:
        print(f"  STT error: {e}", file=sys.stderr)
        return

    if result:
        seq = _next_seq()
        msg = {
            "type": "transcript",
            "text": result["text"],
            "seq": seq,
            "timestamp": time.time(),
        }

        # Persist with STT metadata
        if _current_session_id:
            await db.store_segment(
                _current_session_id,
                seq,
                result["text"],
                is_partial=False,
                timestamp=msg["timestamp"],
                stt_language=result.get("language", ""),
                stt_backend=result.get("backend", ""),
                stt_latency_ms=result.get("latency_ms"),
                stt_raw_text=result.get("raw_text"),
            )

        # Broadcast to all clients
        payload = json.dumps(msg)
        for ws in list(connected_clients):
            try:
                await ws.send_text(payload)
            except Exception:
                pass


# ─── WebSocket ───
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    with metrics_lock:
        metrics["ws_clients"] = len(connected_clients)
    print(f"Browser connected ({len(connected_clients)} clients)")
    await websocket.send_json(
        {
            "type": "status",
            "status": "connected",
            "message": "STT server ready",
        }
    )
    # If a session is active, send restore so late-joining displays catch up
    if _current_session_id:
        client_sessions[websocket] = _current_session_id
        snapshot = await db.get_latest_snapshot(_current_session_id)
        segments = await db.get_segments_since(_current_session_id, 0)
        await websocket.send_json(
            {
                "type": "restore",
                "session_id": _current_session_id,
                "snapshot": snapshot,
                "segments": segments,
                "restore_ms": 0,
            }
        )
    try:
        while True:
            raw = await websocket.receive_text()
            d = json.loads(raw)
            msg_type = d.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "get_metrics":
                with metrics_lock:
                    m = {**metrics, "uptime": time.time() - metrics["started_at"]}
                churn = reconciler.get_churn_metrics()
                m.update(churn)
                m["current_session_id"] = _current_session_id
                m["active_nodes"] = len(
                    [ns for ns in reconciler.nodes.values() if ns.state == "active"]
                )
                with _llm_chain_lock:
                    head = (
                        dict(_llm_chain[0])
                        if _llm_chain
                        else {"provider": "", "model": ""}
                    )
                m["llm_provider"] = head.get("provider", "")
                m["llm_model"] = head.get("model", "")
                m["llm_tiers"] = _cb_snapshot()
                m["llm_serving"] = _last_serving_provider
                stt = get_stt_config()
                m["stt_backend"] = stt["backend"]
                m["stt_remote_url"] = stt.get("remote_url", "")
                with _activity_lock:
                    m["activity_log"] = list(_activity_log[-20:])
                await websocket.send_json({"type": "metrics", **m})

            elif msg_type == "claude_request":
                asyncio.create_task(_proxy_claude(websocket, d))

            elif msg_type == "connect_session":
                session_id = d.get("session_id")
                last_seq = d.get("last_seq", 0)
                if session_id:
                    client_sessions[websocket] = session_id
                    with metrics_lock:
                        metrics["ws_reconnects"] += 1
                    # Send restore data
                    t0 = time.time()
                    snapshot = await db.get_latest_snapshot(session_id)
                    segments = await db.get_segments_since(session_id, last_seq)
                    restore_ms = (time.time() - t0) * 1000
                    with metrics_lock:
                        metrics["last_restore_ms"] = restore_ms
                    await websocket.send_json(
                        {
                            "type": "restore",
                            "snapshot": snapshot,
                            "segments": segments,
                            "restore_ms": round(restore_ms, 1),
                        }
                    )

            elif msg_type == "frontend_metrics":
                fps = d.get("fps", 0)
                with metrics_lock:
                    metrics["frontend_fps"] = fps

            elif msg_type == "audio_chunk":
                audio_bytes = base64.b64decode(d["data"])
                audio_arr = np.frombuffer(audio_bytes, dtype=np.float32)
                source_rate = d.get("sample_rate", 48000)
                asyncio.create_task(_handle_audio_chunk(audio_arr, source_rate))

    except (WebSocketDisconnect, Exception):
        pass
    finally:
        connected_clients.discard(websocket)
        client_sessions.pop(websocket, None)
        with metrics_lock:
            metrics["ws_clients"] = len(connected_clients)
        print(f"Browser disconnected ({len(connected_clients)} clients)")


# ─── Background loops ───
async def broadcast_loop():
    """Poll transcript_queue and broadcast to all WS clients."""
    while True:
        try:
            msg = transcript_queue.get_nowait()
            seq = _next_seq()
            msg["seq"] = seq

            # Persist segment
            if _current_session_id and msg.get("type") in (
                "transcript",
                "partial_transcript",
            ):
                await db.store_segment(
                    _current_session_id,
                    seq,
                    msg["text"],
                    is_partial=(msg["type"] == "partial_transcript"),
                    timestamp=msg.get("timestamp", time.time()),
                )

            if connected_clients:
                p = json.dumps(msg)
                for ws in list(connected_clients):
                    try:
                        await ws.send_text(p)
                    except Exception:
                        pass
        except queue.Empty:
            pass
        await asyncio.sleep(0.05)


async def snapshot_loop():
    """Periodic graph snapshot every 60s."""
    while True:
        await asyncio.sleep(60)
        if _current_session_id and reconciler.nodes:
            try:
                await db.store_snapshot(
                    _current_session_id,
                    _seq_counter,
                    reconciler.get_full_state(),
                    "periodic",
                )
            except Exception as e:
                print(f"  Snapshot error: {e}", file=sys.stderr)


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
