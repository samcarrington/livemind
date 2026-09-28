"""Post-session workflows: background transcript-cleaning jobs.

Recap and synthesis generation still live in ``routes/post_session.py``.
"""

import asyncio
import json
import sys
import threading
import time

import aiohttp

import db
from services.runtime import log_activity
from settings import (
    HUGIN_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MAX_CONCURRENCY,
    OLLAMA_NUM_CTX,
    OLLAMA_RECAP_MODEL,
)


# In-flight transcript cleaning jobs, keyed by session_id. Each entry tracks
# progress so the frontend can poll instead of waiting on a single long
# request — the cloudflare tunnel would otherwise kill it at ~100s.
clean_jobs: dict[str, dict] = {}


clean_jobs_lock = threading.Lock()


async def run_clean_job(session_id: str, segments: list[dict]):
    """Background worker for clean-transcript. Updates clean_jobs as it
    progresses. Never raises — all errors land in the job state."""
    try:
        result = await _clean_transcript_impl(session_id, segments)
        with clean_jobs_lock:
            clean_jobs[session_id] = {
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
        with clean_jobs_lock:
            clean_jobs[session_id] = {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "finished_at": time.time(),
            }


def _clean_job_progress(session_id: str, done: int, total: int):
    with clean_jobs_lock:
        job = clean_jobs.get(session_id)
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
            "options": {
                "temperature": 0,
                "num_predict": 4096,
                "num_ctx": OLLAMA_NUM_CTX,
            },
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
