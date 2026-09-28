"""Live WebSocket endpoint (``/ws``) for audience, monitor and audio clients."""

import asyncio
import base64
import json
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import db
from services.live_pipeline import handle_audio_chunk, proxy_claude
from services.llm import (
    cb_snapshot,
    get_last_serving_provider,
    llm_chain,
    llm_chain_lock,
)
from services.runtime import (
    activity_lock,
    activity_log,
    client_sessions,
    connected_clients,
    metrics,
    metrics_lock,
    reconciler,
)
from services.session_runtime import live_session
from stt_worker import get_stt_config

router = APIRouter()


@router.websocket("/ws")
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
    if live_session.current_session_id:
        client_sessions[websocket] = live_session.current_session_id
        snapshot = await db.get_latest_snapshot(live_session.current_session_id)
        segments = await db.get_segments_since(live_session.current_session_id, 0)
        await websocket.send_json(
            {
                "type": "restore",
                "session_id": live_session.current_session_id,
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
                m["current_session_id"] = live_session.current_session_id
                m["active_nodes"] = len(
                    [ns for ns in reconciler.nodes.values() if ns.state == "active"]
                )
                with llm_chain_lock:
                    head = (
                        dict(llm_chain[0])
                        if llm_chain
                        else {"provider": "", "model": ""}
                    )
                m["llm_provider"] = head.get("provider", "")
                m["llm_model"] = head.get("model", "")
                m["llm_tiers"] = cb_snapshot()
                m["llm_serving"] = get_last_serving_provider()
                stt = get_stt_config()
                m["stt_backend"] = stt["backend"]
                m["stt_remote_url"] = stt.get("remote_url", "")
                with activity_lock:
                    m["activity_log"] = list(activity_log[-20:])
                await websocket.send_json({"type": "metrics", **m})

            elif msg_type == "claude_request":
                asyncio.create_task(proxy_claude(websocket, d))

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
                asyncio.create_task(handle_audio_chunk(audio_arr, source_rate))

    except (WebSocketDisconnect, Exception):
        pass
    finally:
        connected_clients.discard(websocket)
        client_sessions.pop(websocket, None)
        with metrics_lock:
            metrics["ws_clients"] = len(connected_clients)
        print(f"Browser disconnected ({len(connected_clients)} clients)")
