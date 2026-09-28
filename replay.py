#!/usr/bin/env python3
"""
LiveMind — File Replay Module
Feed a pre-recorded audio file through the STT → LLM → reconciler pipeline.
Produces a session in the DB identical to a live recording.

Usage:
    python replay.py recording.mp3
    python replay.py recording.mp3 --topic "Q2 Strategy"
    python replay.py recording.mp3 --llm-provider hugin --llm-model gemma4:e4b
    python replay.py recording.mp3 --stt-backend canary --chunk-seconds 10
"""

import asyncio, argparse, json, os, subprocess, sys, threading, time, uuid
import numpy as np

import db
import stt_worker
from reconciler import GraphReconciler


# ─── Audio Loading ───

def load_audio_file(path: str) -> tuple[np.ndarray, int]:
    """Decode any audio format to 16kHz float32 mono via ffmpeg.
    Returns (audio_array, 16000)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    cmd = [
        "ffmpeg", "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", "16000", "-ac", "1",
        "-loglevel", "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if len(audio) == 0:
        raise RuntimeError("ffmpeg produced no audio output")

    return audio, 16000


# ─── Audio Chunking ───

def chunk_audio(audio: np.ndarray, sample_rate: int = 16000,
                chunk_seconds: float = 5.0) -> list[np.ndarray]:
    """Split audio into chunks, preferring to cut at silence boundaries.
    Skips chunks that are entirely silent."""
    chunk_size = int(chunk_seconds * sample_rate)
    scan_window = int(1.0 * sample_rate)  # ±1 second scan for quiet points
    rms_window = int(0.1 * sample_rate)   # 100ms RMS window
    silence_threshold = 0.005

    chunks = []
    pos = 0

    while pos < len(audio):
        if pos + chunk_size >= len(audio):
            # Last chunk — take everything remaining
            chunk = audio[pos:]
        else:
            # Scan ±1s around the target boundary for the quietest point
            target = pos + chunk_size
            scan_start = max(pos + chunk_size // 2, target - scan_window)
            scan_end = min(len(audio) - rms_window, target + scan_window)

            best_pos = target
            best_rms = float("inf")

            for p in range(scan_start, scan_end, rms_window // 2):
                window = audio[p:p + rms_window]
                rms = float(np.sqrt(np.mean(window ** 2)))
                if rms < best_rms:
                    best_rms = rms
                    best_pos = p

            chunk = audio[pos:best_pos]

        # Skip silent chunks
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms >= silence_threshold and len(chunk) > sample_rate // 4:  # at least 250ms
            chunks.append(chunk)

        pos += len(chunk)

    return chunks


# ─── Server-Side Analysis Pipeline ───

class AnalysisPipeline:
    """Server-side equivalent of live-mindmap.html analyze().
    Accumulates transcript text and triggers LLM analysis."""

    def __init__(self, topic: str = "", max_nodes: int = 30):
        self.topic = topic
        self.max_nodes = max_nodes
        self.full_text = ""
        self.sent_pos = 0
        self.summary = ""

    def add_text(self, text: str):
        self.full_text += " " + text

    def has_enough(self, min_chars: int = 50) -> bool:
        fresh = self.full_text[self.sent_pos:].strip()
        return len(fresh) >= min_chars

    def build_prompt(self, reconciler: GraphReconciler) -> dict:
        """Construct the LLM request body (Anthropic format).
        Mirrors live-mindmap.html lines 771-794."""
        cur_graph = reconciler.get_active_graph()
        graph_json = {
            "nodes": [{"id": n["id"], "label": n["label"], "group": n["group"]}
                      for n in cur_graph.get("nodes", [])],
            "edges": [{"source": e["source"], "target": e["target"], "label": e.get("label", "")}
                      for e in cur_graph.get("edges", [])],
        }

        topic_line = f'- Meeting context: "{self.topic}"' if self.topic else ""
        system = f"""You generate mind-map graphs from meeting transcripts. Return ONLY valid JSON.

Rules:
- Max {self.max_nodes} nodes. Merge lesser concepts to stay under limit.
- Node: {{"id":"n1","label":"Short Name","group":"Category"}}
  - label: 2-4 words, title case
  - group: broad category for color clustering
- Edge: {{"source":"n1","target":"n2","label":"verb"}}
  - label: 1-2 word relationship (e.g. "drives", "enables", "part of")
- Preserve existing node IDs. Add only genuinely important new concepts.
- Remove nodes that are no longer relevant as the conversation evolves.
- Create edges that reveal the STRUCTURE of the discussion, not just proximity.
{topic_line}

Return: {{"nodes":[...],"edges":[...],"summary":"<2-3 sentence summary of all key points discussed so far>"}}"""

        fresh = self.full_text[self.sent_pos:].strip()
        segment = fresh[-3000:] if len(fresh) > 3000 else fresh

        if self.summary:
            context_block = f'CONTEXT SUMMARY:\n"{self.summary}"'
        elif self.full_text.strip():
            context_block = f'INITIAL TRANSCRIPT:\n"{self.full_text.strip()[:3000]}"'
        else:
            context_block = ""

        user_msg = f"CURRENT GRAPH:\n{json.dumps(graph_json)}\n\nNEW SEGMENT:\n\"{segment}\"\n\n{context_block}\n\nReturn the updated graph."

        return {
            "max_tokens": 2000,
            "system": system,
            "messages": [{"role": "user", "content": user_msg}],
        }

    def mark_sent(self):
        self.sent_pos = len(self.full_text)


# ─── Main Orchestrator ───

async def replay_file(
    file_path: str,
    topic: str = "",
    chunk_seconds: float = 5.0,
    analysis_min_chars: int = 50,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    stt_backend: str | None = None,
    stt_language: str | None = None,
    db_path: str = "livemind.db",
) -> dict:
    """Replay an audio file through the full STT→LLM→reconciler pipeline.
    Returns {"session_id", "segments", "snapshots", "duration_s"}."""

    # Load and chunk audio
    print(f"Loading {file_path}...")
    audio, sr = load_audio_file(file_path)
    duration_s = len(audio) / sr
    print(f"  Duration: {duration_s:.1f}s ({duration_s/60:.1f}m)")

    chunks = chunk_audio(audio, sr, chunk_seconds)
    print(f"  Chunks: {len(chunks)} (≈{chunk_seconds}s each)")

    # Init DB
    await db.init_db(db_path)

    # Create session
    session_id = str(uuid.uuid4())[:8]
    filename = os.path.basename(file_path)
    session_topic = topic or filename
    await db.create_session(session_id, session_topic, source="replay")
    print(f"  Session: {session_id} — {session_topic}")

    # Init reconciler (own instance, not the global)
    reconciler = GraphReconciler()

    # Init analysis pipeline
    pipeline = AnalysisPipeline(topic=session_topic, max_nodes=30)

    # Configure STT if overrides specified
    if stt_backend or stt_language:
        stt_worker.configure_stt(
            stt_backend or stt_worker.get_stt_config()["backend"],
            language=stt_language or "",
        )

    # Reset STT state to avoid cross-contamination
    stt_worker.reset_state()

    # Resolve LLM config. Replay can either pin a single tier (via the
    # --llm-provider / --llm-model flags) or walk the server's full chain.
    from app import call_llm_chain, _llm_chain, _llm_chain_lock, _extract_graph_json

    pinned = bool(llm_provider)
    if pinned:
        provider = llm_provider
        model = llm_model or ""
        print(f"  LLM: pinned to {provider}/{model or '(default)'}")
    else:
        with _llm_chain_lock:
            chain_desc = [f"{t['provider']}/{t['model']}" for t in _llm_chain]
        provider = ""
        model = ""
        print(f"  LLM chain: {' → '.join(chain_desc)}")
    print(f"  STT: {stt_worker.get_stt_config()['backend']}")
    print()

    # Process chunks
    seq = 0
    snapshot_count = 0
    metrics = {}
    metrics_lock = threading.Lock()
    loop = asyncio.get_event_loop()

    for i, chunk in enumerate(chunks):
        # STT (blocking call, run in executor)
        result = await loop.run_in_executor(
            None, stt_worker.transcribe_audio_chunk,
            chunk, sr, metrics, metrics_lock,
        )

        if result:
            seq += 1
            ts = time.time()

            # Store segment
            await db.store_segment(
                session_id, seq, result["text"], False, ts,
                stt_language=result.get("language", ""),
                stt_backend=result.get("backend", ""),
                stt_latency_ms=result.get("latency_ms"),
                stt_raw_text=result.get("raw_text"),
            )

            pipeline.add_text(result["text"])

        # Trigger analysis when enough text accumulated
        if pipeline.has_enough(analysis_min_chars):
            body = pipeline.build_prompt(reconciler)
            if model:
                body["model"] = model

            try:
                if pinned:
                    from app import _call_provider
                    status_code, data = await _call_provider(provider, body)
                    served_by = provider
                else:
                    status_code, data, served_by, _ = await call_llm_chain(body)

                if status_code == 200:
                    raw_text = "".join(c.get("text", "") for c in data.get("content", []))
                    parsed = _extract_graph_json(raw_text)
                    if parsed.get("nodes"):
                        reconciler.reconcile(parsed)
                        snapshot_count += 1
                        await db.store_snapshot(
                            session_id, seq, reconciler.get_full_state(), "analysis"
                        )
                        if parsed.get("summary"):
                            pipeline.summary = parsed["summary"]
                        pipeline.mark_sent()
                else:
                    print(f"  LLM error {status_code}: {data}", file=sys.stderr)

            except Exception as e:
                print(f"  LLM call failed: {e}", file=sys.stderr)

        # Progress
        pct = (i + 1) / len(chunks) * 100
        print(f"\r  [{pct:5.1f}%] chunk {i+1}/{len(chunks)} | {seq} segments | {snapshot_count} snapshots", end="", flush=True)

    # Final analysis pass on any remaining text
    if pipeline.has_enough(1):
        body = pipeline.build_prompt(reconciler)
        if model:
            body["model"] = model

        try:
            if pinned:
                from app import _call_provider
                status_code, data = await _call_provider(provider, body)
            else:
                status_code, data, _, _ = await call_llm_chain(body)

            if status_code == 200:
                raw_text = "".join(c.get("text", "") for c in data.get("content", []))
                parsed = _extract_graph_json(raw_text)
                if parsed.get("nodes"):
                    reconciler.reconcile(parsed)
                    snapshot_count += 1
                    await db.store_snapshot(
                        session_id, seq, reconciler.get_full_state(), "final"
                    )
                    if parsed.get("summary"):
                        pipeline.summary = parsed["summary"]
        except Exception as e:
            print(f"\n  Final analysis failed: {e}", file=sys.stderr)

    # End session
    await db.end_session(session_id, pipeline.summary)
    await db.store_snapshot(session_id, seq, reconciler.get_full_state(), "end")

    print(f"\n\nDone. Session: {session_id}")
    print(f"  Segments: {seq}")
    print(f"  Snapshots: {snapshot_count}")
    print(f"  Audio duration: {duration_s:.1f}s")

    return {
        "session_id": session_id,
        "segments": seq,
        "snapshots": snapshot_count,
        "duration_s": duration_s,
    }


# ─── CLI ───

def main():
    parser = argparse.ArgumentParser(
        description="Replay an audio file through LiveMind's STT→LLM→graph pipeline"
    )
    parser.add_argument("file", help="Path to audio file (wav, mp3, ogg, flac, m4a, ...)")
    parser.add_argument("--topic", default="", help="Session topic (default: filename)")
    parser.add_argument("--chunk-seconds", type=float, default=5.0, help="Audio chunk duration in seconds (default: 5)")
    parser.add_argument("--analysis-min-chars", type=int, default=50, help="Min new chars before triggering LLM analysis (default: 50)")
    parser.add_argument("--llm-provider", choices=["hugin", "gemini", "anthropic"], help="LLM provider (default: server config)")
    parser.add_argument("--llm-model", help="LLM model name (default: server config)")
    parser.add_argument("--stt-backend", choices=["remote", "parakeet", "canary"], help="STT backend (default: server config)")
    parser.add_argument("--stt-language", default=None, help="STT language code, e.g. 'no' for Norwegian (default: auto-detect)")
    parser.add_argument("--db", default="livemind.db", help="Database path (default: livemind.db)")
    args = parser.parse_args()

    asyncio.run(replay_file(
        file_path=args.file,
        topic=args.topic,
        chunk_seconds=args.chunk_seconds,
        analysis_min_chars=args.analysis_min_chars,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        stt_backend=args.stt_backend,
        stt_language=args.stt_language,
        db_path=args.db,
    ))


if __name__ == "__main__":
    main()
