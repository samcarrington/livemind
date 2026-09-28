"""Shared in-process runtime state for the LiveMind server.

Holds the objects every layer touches: the STT transcript queue, the set of
connected WebSocket clients, live metrics, the post-session activity log and
the graph reconciler. Mutate containers in place; never rebind these names.
"""

import json
import queue
import threading
import time

from fastapi import WebSocket

from reconciler import GraphReconciler


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
activity_log: list[dict] = []  # capped at 50 entries


activity_lock = threading.Lock()


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
    with activity_lock:
        activity_log.append(entry)
        if len(activity_log) > 50:
            activity_log.pop(0)


# Graph reconciler (one live graph per server process)
reconciler = GraphReconciler()


async def broadcast_json(payload: dict) -> None:
    """Send a JSON message to every connected client, ignoring dead sockets."""
    msg = json.dumps(payload)
    for ws in list(connected_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            pass
