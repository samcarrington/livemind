"""Live pipeline: audio → STT → transcript broadcast, and LLM graph ingestion.

Also hosts the background ``broadcast_loop`` / ``snapshot_loop`` started by the
app lifespan.
"""

import asyncio
import json
import queue
import sys
import time

import numpy as np
from fastapi import WebSocket

import db
import stt_worker
from services.llm import (
    DEFAULT_PRICING,
    LLM_PRICING,
    call_llm_chain,
    extract_graph_json,
    extract_usage,
    publish_llm_state,
)
from services.runtime import (
    connected_clients,
    metrics,
    metrics_lock,
    reconciler,
    transcript_queue,
)
from services.session_runtime import live_session


async def broadcast_llm_response(status_code, data, req_id):
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


async def proxy_claude(websocket: WebSocket, req: dict):
    """Proxy an LLM request server-side, walking the fallback chain."""
    _req_session_id = live_session.current_session_id
    _req_gen = live_session.session_gen

    t0 = time.time()
    try:
        body = req.get("body", {})
        status_code, data, provider, model = await call_llm_chain(body)
        dt = time.time() - t0

        # Token usage / cost attributed to whichever tier actually served.
        if provider and model:
            usage = extract_usage(data, provider)
            pricing = LLM_PRICING.get(model, DEFAULT_PRICING)
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

        publish_llm_state()

        if status_code == 200:
            cost_str = f"${call_cost:.4f}" if call_cost > 0 else "free"
            print(
                f"  LLM [{provider}/{model}]: 200 OK ({dt:.1f}s, {usage['input']}+{usage['output']} tok, {cost_str})"
            )
            with metrics_lock:
                metrics["claude_last_error"] = ""

            try:
                raw_text = "".join(c.get("text", "") for c in data.get("content", []))
                parsed = extract_graph_json(raw_text)
                if parsed.get("nodes") and parsed.get("edges") is not None:
                    # Generation check catches the None→None race that the
                    # session_id check alone can't: if a session reset happened
                    # while the LLM was responding, live_session.session_gen has advanced
                    # and we must discard this response.
                    if live_session.session_gen != _req_gen:
                        print(
                            f"  LLM: discarding stale response (gen {_req_gen} → {live_session.session_gen}, session {_req_session_id} → {live_session.current_session_id})"
                        )
                        return
                    if live_session.current_session_id != _req_session_id:
                        print(
                            f"  LLM: discarding stale response (session changed {_req_session_id} → {live_session.current_session_id})"
                        )
                        return
                    n_before = len(reconciler.nodes)
                    graph = reconciler.reconcile(parsed)
                    n_after = len(reconciler.nodes)
                    if parsed.get("summary"):
                        live_session.summary = parsed["summary"]
                    if live_session.current_session_id:
                        await db.store_snapshot(
                            live_session.current_session_id,
                            live_session.seq_counter,
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
                                        "summary": live_session.summary,
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
                        f"  LLM: parsed {len(parsed['nodes'])} nodes, {len(parsed.get('edges', []))} edges (reconciler: {n_before}→{n_after})"
                    )

                    graph_msg = json.dumps(
                        {
                            "type": "graph_update",
                            "graph": graph,
                            "session_id": live_session.current_session_id,
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

        await broadcast_llm_response(status_code, data, req.get("req_id"))
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
        publish_llm_state()
        await broadcast_llm_response(500, {"error": str(e)}, req.get("req_id"))


async def handle_audio_chunk(audio_arr: np.ndarray, source_rate: int):
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
        seq = live_session.next_seq()
        msg = {
            "type": "transcript",
            "text": result["text"],
            "seq": seq,
            "timestamp": time.time(),
        }

        # Persist with STT metadata
        if live_session.current_session_id:
            await db.store_segment(
                live_session.current_session_id,
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


async def broadcast_loop():
    """Poll transcript_queue and broadcast to all WS clients."""
    while True:
        try:
            msg = transcript_queue.get_nowait()
            seq = live_session.next_seq()
            msg["seq"] = seq

            # Persist segment
            if live_session.current_session_id and msg.get("type") in (
                "transcript",
                "partial_transcript",
            ):
                await db.store_segment(
                    live_session.current_session_id,
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
        if live_session.current_session_id and reconciler.nodes:
            try:
                await db.store_snapshot(
                    live_session.current_session_id,
                    live_session.seq_counter,
                    reconciler.get_full_state(),
                    "periodic",
                )
            except Exception as e:
                print(f"  Snapshot error: {e}", file=sys.stderr)
