"""Session lifecycle, restore/playback, graph actions and export endpoints."""

import asyncio
import os
import sys
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

import db
from services.runtime import (
    metrics,
    metrics_lock,
    broadcast_json,
    reconciler,
)
from services.session_runtime import clear_graph, live_session, reset_live_state

router = APIRouter()


@router.get("/v1/sessions")
async def list_sessions(archived: bool = False):
    sessions = await db.list_sessions(archived=archived)
    return JSONResponse(sessions)


@router.post("/v1/sessions/archive")
async def archive_sessions(request: Request):
    """Move sessions to archive."""
    body = await request.json()
    session_ids = body.get("session_ids", [])
    if not session_ids:
        return JSONResponse({"error": "No session IDs provided"}, status_code=400)
    await db.archive_sessions(session_ids)
    return JSONResponse({"ok": True, "archived": len(session_ids)})


@router.post("/v1/sessions/unarchive")
async def unarchive_sessions(request: Request):
    """Move sessions out of archive."""
    body = await request.json()
    session_ids = body.get("session_ids", [])
    if not session_ids:
        return JSONResponse({"error": "No session IDs provided"}, status_code=400)
    await db.unarchive_sessions(session_ids)
    return JSONResponse({"ok": True, "unarchived": len(session_ids)})


@router.delete("/v1/sessions")
async def delete_sessions(request: Request):
    """Permanently delete sessions and all associated data."""
    body = await request.json()
    session_ids = body.get("session_ids", [])
    if not session_ids:
        return JSONResponse({"error": "No session IDs provided"}, status_code=400)
    await db.delete_sessions(session_ids)
    return JSONResponse({"ok": True, "deleted": len(session_ids)})


@router.post("/v1/sessions")
async def create_session(request: Request):
    body = await request.json()
    session_id = str(uuid.uuid4())[:8]
    topic = body.get("topic", "")
    session = await db.create_session(session_id, topic)
    live_session.current_session_id = session_id
    return JSONResponse(session)


@router.post("/v1/sessions/new")
async def new_session(request: Request):
    """End current session (if any) and start a fresh one. Returns the new session."""
    body = (
        await request.json()
        if request.headers.get("content-length", "0") != "0"
        else {}
    )
    topic = body.get("topic", "")

    # Bump the session generation FIRST, so any in-flight proxy_claude task
    # that resumes during our awaits below will see a newer gen and discard
    # its response. Without this, a late LLM response can race with the
    # reconciler clear and repopulate it with stale nodes from the previous
    # session.
    live_session.bump_gen()

    # End current session
    if live_session.current_session_id:
        if reconciler.nodes:
            await db.store_snapshot(
                live_session.current_session_id,
                live_session.seq_counter,
                reconciler.get_full_state(),
                "end",
            )
        await db.end_session(live_session.current_session_id, live_session.summary)
    # Reset all state (graph, summary, seq, pending transcripts)
    reset_live_state()
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
    live_session.current_session_id = session_id
    # Belt-and-braces: re-clear reconciler AFTER the create_session await, in
    # case a racing proxy_claude resumed during the yield and repopulated it.
    # The gen-bump above should already have caused such tasks to discard, but
    # this closes the window with zero cost.
    clear_graph()
    # Notify all connected frontends
    await broadcast_json(
        {"type": "session_reset", "session_id": session_id, "topic": topic}
    )
    return JSONResponse(session)


@router.post("/v1/sessions/{session_id}/end")
async def end_session(session_id: str, request: Request):
    """End a session: flush final snapshot, store summary, reset reconciler."""
    body = (
        await request.json()
        if request.headers.get("content-length", "0") != "0"
        else {}
    )
    summary = body.get("summary", live_session.summary)
    # Invalidate any in-flight LLM responses before we touch reconciler state.
    live_session.bump_gen()
    # Final snapshot
    if reconciler.nodes:
        await db.store_snapshot(
            session_id, live_session.seq_counter, reconciler.get_full_state(), "end"
        )
    await db.end_session(session_id, summary)
    # Reset server state
    reset_live_state()
    if live_session.current_session_id == session_id:
        live_session.current_session_id = None
    # Notify all connected frontends
    await broadcast_json({"type": "session_ended", "session_id": session_id})
    return JSONResponse({"ok": True, "session_id": session_id})


@router.get("/v1/sessions/{session_id}/restore")
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


@router.post("/v1/sessions/{session_id}/actions")
async def session_action(session_id: str, request: Request):
    body = await request.json()
    action_type = body.get("action")
    payload = body.get("payload", {})
    await db.store_action(session_id, action_type, payload)
    graph = reconciler.apply_action(action_type, payload)
    # Broadcast updated graph to all connected clients
    await broadcast_json(
        {
            "type": "graph_update",
            "graph": graph,
            "session_id": live_session.current_session_id,
        }
    )
    return JSONResponse({"ok": True, "graph": graph})


@router.get("/v1/sessions/{session_id}/snapshots")
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


@router.post("/v1/sessions/{session_id}/export/{fmt}")
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


@router.get("/v1/sessions/{session_id}/export/{fmt}/status")
async def export_status(session_id: str, fmt: str):
    """Check export status."""
    task_key = f"{session_id}_{fmt}"
    info = _export_tasks.get(task_key)
    if not info:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return JSONResponse({"status": info["status"], "error": info.get("error")})


@router.get("/v1/sessions/{session_id}/export/{fmt}/download")
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


@router.get("/v1/sessions/{session_id}")
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
