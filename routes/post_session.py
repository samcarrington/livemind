"""Post-session endpoints: recap, cross-session synthesis, transcript cleaning."""

import asyncio
import json
import sys
import time

import aiohttp
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import db
from services.post_session import clean_jobs, clean_jobs_lock, run_clean_job
from services.runtime import log_activity
from settings import (
    HUGIN_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_CTX,
    OLLAMA_RECAP_MODEL,
    RECAP_MAX_CHARS,
)

router = APIRouter()


@router.get("/v1/sessions/synthesis")
async def list_synthesis():
    """List all cross-session synthesis recaps."""
    return JSONResponse(await db.list_synthesis_recaps())


@router.get("/v1/sessions/synthesis/{synthesis_id}")
async def get_synthesis(synthesis_id: int):
    """Get a single cross-session synthesis recap."""
    result = await db.get_synthesis_recap(synthesis_id)
    if not result:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(result)


@router.delete("/v1/sessions/synthesis/{synthesis_id}")
async def delete_synthesis(synthesis_id: int):
    """Delete a cross-session synthesis recap."""
    await db.delete_synthesis_recap(synthesis_id)
    return JSONResponse({"ok": True})


@router.post("/v1/sessions/{session_id}/recap")
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
    max_chars = RECAP_MAX_CHARS
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

            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"{HUGIN_BASE_URL}/api/chat",
                    headers={"Content-Type": "application/json"},
                    json=ollama_body,
                    timeout=aiohttp.ClientTimeout(total=180),
                    ssl=False,
                ) as resp,
            ):
                data = await resp.json()
                if resp.status != 200:
                    err = data.get("error", "") or str(data)
                    return JSONResponse({"error": f"Ollama: {err}"}, status_code=502)

            raw_text = data.get("message", {}).get("content", "")
            # Strip markdown code fences if present
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            cleaned = cleaned.removesuffix("```")
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


@router.get("/v1/sessions/{session_id}/clean-transcript/status")
async def clean_transcript_status(session_id: str):
    """Return the current state of an in-flight (or recently finished) clean
    job for this session. Frontend polls this while a job is running."""
    with clean_jobs_lock:
        job = clean_jobs.get(session_id)
        snapshot = dict(job) if job else None
    if snapshot is None:
        return JSONResponse({"status": "idle"})
    return JSONResponse(snapshot)


@router.post("/v1/sessions/{session_id}/clean-transcript")
async def clean_transcript(session_id: str):
    """Kick off a transcript-cleaning job in the background. Returns 202
    immediately so the cloudflare tunnel doesn't time out; the frontend
    polls /clean-transcript/status for progress."""
    # Refuse if a job is already running for this session
    with clean_jobs_lock:
        existing = clean_jobs.get(session_id)
        if existing and existing.get("status") == "running":
            return JSONResponse(
                {"status": "running", "progress": existing.get("progress", {})},
                status_code=202,
            )

    segments = await db.get_session_transcript(session_id)
    if not segments:
        return JSONResponse({"error": "No transcript found"}, status_code=404)

    # Record an initial job state and fire the background task
    with clean_jobs_lock:
        clean_jobs[session_id] = {
            "status": "running",
            "progress": {"done": 0, "total": 0},
            "started_at": time.time(),
        }
    asyncio.create_task(run_clean_job(session_id, segments))
    return JSONResponse(
        {"status": "running", "progress": {"done": 0, "total": 0}}, status_code=202
    )


@router.post("/v1/sessions/synthesis")
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
        block += f"ID: {sd['id']}\n"

        # Include recap highlights
        if r.get("elevator_pitch"):
            block += f"PITCH: {r['elevator_pitch']}\n"
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
                block += f"  - {topics}: {conn.get('insight', '')}\n"
        if r.get("summary"):
            block += f"SUMMARY: {r['summary']}\n"
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
                    edge_strs.append(f"{src} --[{e.get('label', '')}]--> {tgt}")
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
