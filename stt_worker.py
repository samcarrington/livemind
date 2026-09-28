"""
STT Worker — Audio transcription dispatch.
Receives PCM audio from the browser (via WebSocket), resamples to 16kHz,
and dispatches to the active STT backend:
  - remote: faster-whisper at localhost:8766
  - parakeet: Parakeet TDT at localhost:8010
  - canary: Canary at localhost:8011
All backends use the same /v1/transcribe raw PCM endpoint with query params.
"""

import time, threading, os
import numpy as np

SAMPLE_RATE = 16000  # All STT backends expect 16kHz

# ─── STT backend config (mutable at runtime) ───
_active_stt = {
    "backend": "remote",                            # "remote" | "parakeet" | "canary"
    "remote_url": "http://localhost:8766",        # faster-whisper
    "parakeet_url": "http://localhost:8010",      # Parakeet TDT
    "canary_url": "http://localhost:8011",        # Canary
    "language": "",                               # ISO 639-1, empty = auto-detect
}
_active_stt_lock = threading.Lock()

# Rolling context for Whisper prompt conditioning (last N chars of transcript)
_transcript_context = {"text": ""}
_transcript_context_lock = threading.Lock()
CONTEXT_MAX_CHARS = 200

# Known Whisper hallucination patterns (exact matches after stripping)
_WHISPER_HALLUCINATIONS = {
    "", ".", "..", "...", "thank you.", "thank you", "thanks.", "thanks",
    "thanks for watching.", "thanks for watching", "thank you for watching.",
    "thank you for watching", "subscribe.", "subscribe",
    "like and subscribe.", "please subscribe.", "bye.", "bye",
    "you", "okay.", "okay", "ok.", "so.",
    "the end.", "the end", "subtitles by the amara.org community",
    "subs by www.teletext.ch",
}

# Previous result for deduplication
_prev_result = {"text": "", "time": 0.0}


def _clean_whisper_output(text: str) -> str:
    """Filter hallucinations and clean up Whisper output. Returns empty string if hallucinated."""
    stripped = text.strip()
    if not stripped:
        return ""

    # Exact match against known hallucinations
    if stripped.lower().rstrip(".!,") in _WHISPER_HALLUCINATIONS or stripped.lower() in _WHISPER_HALLUCINATIONS:
        print(f"  [Filter] hallucination dropped: '{stripped}'")
        return ""

    # Repetition detection — if the same short phrase repeats 3+ times, it's looping
    words = stripped.split()
    if len(words) >= 6:
        chunk = " ".join(words[:2]).lower()
        count = stripped.lower().count(chunk)
        if count >= 3:
            print(f"  [Filter] repetition loop dropped: '{stripped[:80]}...'")
            return ""

    return stripped


def _update_context(text: str):
    """Append text to rolling context for Whisper prompt conditioning."""
    with _transcript_context_lock:
        _transcript_context["text"] = (_transcript_context["text"] + " " + text).strip()
        if len(_transcript_context["text"]) > CONTEXT_MAX_CHARS:
            _transcript_context["text"] = _transcript_context["text"][-CONTEXT_MAX_CHARS:]


def _get_context() -> str:
    """Get rolling context for initial_prompt conditioning."""
    with _transcript_context_lock:
        return _transcript_context["text"]


def _is_duplicate(text: str) -> bool:
    """Check if this is a duplicate of the previous result."""
    now = time.time()
    if text == _prev_result["text"] and (now - _prev_result["time"]) < 10.0:
        print(f"  [Filter] duplicate dropped: '{text[:60]}'")
        return True
    _prev_result["text"] = text
    _prev_result["time"] = now
    return False


def configure_stt(backend: str, remote_url: str = "", language: str = ""):
    """Set the active STT backend. Called from app.py."""
    with _active_stt_lock:
        _active_stt["backend"] = backend
        if remote_url:
            if backend == "remote":
                _active_stt["remote_url"] = remote_url
            elif backend == "parakeet":
                _active_stt["parakeet_url"] = remote_url
            elif backend == "canary":
                _active_stt["canary_url"] = remote_url
        if language is not None:
            _active_stt["language"] = language


def configure_stt_urls(remote_url: str = "", parakeet_url: str = "", canary_url: str = ""):
    """Override backend base URLs without changing the active backend."""
    with _active_stt_lock:
        if remote_url:
            _active_stt["remote_url"] = remote_url.rstrip("/")
        if parakeet_url:
            _active_stt["parakeet_url"] = parakeet_url.rstrip("/")
        if canary_url:
            _active_stt["canary_url"] = canary_url.rstrip("/")


def get_stt_config() -> dict:
    """Return current STT config. Called from app.py."""
    with _active_stt_lock:
        return {**_active_stt}


def reset_state():
    """Clear rolling context and dedup caches. Call before replay to avoid cross-contamination."""
    global _prev_result
    with _transcript_context_lock:
        _transcript_context["text"] = ""
    _prev_result = {"text": "", "time": 0.0}


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio from source_rate to target_rate using linear interpolation."""
    if source_rate == target_rate:
        return audio
    ratio = target_rate / source_rate
    n_out = max(1, int(len(audio) * ratio))
    return np.interp(
        np.linspace(0, len(audio) - 1, n_out),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def _transcribe_backend(audio_arr: np.ndarray, metrics: dict, metrics_lock,
                        url: str, label: str) -> dict:
    """POST raw PCM float32 to /v1/transcribe on any backend. All three servers
    (faster-whisper, Parakeet, Canary) support this endpoint with the same
    query-param interface: sample_rate, language, initial_prompt.
    Returns dict with keys: text, language, latency_ms."""
    import requests
    from urllib.parse import urlencode

    with _active_stt_lock:
        language = _active_stt["language"]

    headers = {"Content-Type": "application/octet-stream"}
    # Cloudflare Access headers for remote (faster-whisper) backend
    if label == "Remote":
        cf_id = os.environ.get("STT_CF_ID") or os.environ.get("HUGIN_CF_ID", "")
        cf_secret = os.environ.get("STT_CF_SECRET") or os.environ.get("HUGIN_CF_SECRET", "")
        if cf_id and cf_secret:
            headers["CF-Access-Client-Id"] = cf_id
            headers["CF-Access-Client-Secret"] = cf_secret

    params = {"sample_rate": SAMPLE_RATE}
    if language:
        params["language"] = language
    context = _get_context()
    if context:
        params["initial_prompt"] = context

    t0 = time.time()
    resp = requests.post(
        f"{url}/v1/transcribe?{urlencode(params)}",
        headers=headers,
        data=audio_arr.tobytes(),
        timeout=30,
        verify=False,
    )
    dt = time.time() - t0

    if resp.status_code != 200:
        print(f"  {label} STT error: {resp.status_code} {resp.text[:200]}")
        return {"text": "", "language": "", "latency_ms": int(dt * 1000)}

    data = resp.json()
    raw_text = data.get("text", "").strip()
    lang = data.get("language", "")
    proc = data.get("processing_s", 0)

    with metrics_lock:
        metrics["stt_last_duration"] = dt
        if proc:
            metrics["stt_remote_processing_s"] = proc
        if lang:
            metrics["stt_language"] = lang

    print(f"  {label} STT ({dt:.1f}s, lang={lang}): '{raw_text}'")
    return {"text": raw_text, "language": lang, "latency_ms": int(dt * 1000)}


# ─── Unified dispatch ───

def transcribe_audio_chunk(
    audio_data: np.ndarray,
    source_sample_rate: int,
    metrics: dict,
    metrics_lock: threading.Lock,
) -> dict | None:
    """
    Accept a PCM audio chunk, resample to 16kHz, dispatch to active STT backend.
    Returns dict with keys: text, language, backend, latency_ms, raw_text.
    Returns None if silence/error/hallucination/duplicate.
    """
    # 1. Resample to 16kHz
    audio_16k = _resample(audio_data, source_sample_rate, SAMPLE_RATE)

    # Skip if too quiet
    rms = float(np.sqrt(np.mean(audio_16k ** 2)))
    if rms < 0.005:
        with metrics_lock:
            metrics["chunks_skipped_silent"] = metrics.get("chunks_skipped_silent", 0) + 1
        return None

    # 2. Dispatch to active backend
    with _active_stt_lock:
        backend = _active_stt["backend"]
        if backend == "parakeet":
            url, label = _active_stt["parakeet_url"], "Parakeet"
        elif backend == "canary":
            url, label = _active_stt["canary_url"], "Canary"
        else:
            url, label = _active_stt["remote_url"], "Remote"

    t_e2e = time.time()

    result = _transcribe_backend(audio_16k, metrics, metrics_lock, url, label)
    raw_text = result["text"]

    e2e = time.time() - t_e2e

    # 3. Filter hallucinations, dedup
    text = _clean_whisper_output(raw_text)
    if not text:
        return None
    if _is_duplicate(text):
        return None

    # 4. Update rolling context
    _update_context(text)

    # 5. Update metrics
    with metrics_lock:
        metrics["chunks_processed"] = metrics.get("chunks_processed", 0) + 1
        metrics["stt_total_time"] = metrics.get("stt_total_time", 0) + metrics.get("stt_last_duration", 0)
        cp = metrics["chunks_processed"]
        metrics["stt_avg_duration"] = metrics["stt_total_time"] / cp if cp else 0
        metrics["stt_last_text"] = text
        metrics["stt_e2e_last"] = e2e
        metrics["stt_e2e_total"] = metrics.get("stt_e2e_total", 0) + e2e
        metrics["stt_e2e_avg"] = metrics["stt_e2e_total"] / cp if cp else 0

    return {
        "text": text,
        "language": result["language"],
        "backend": backend,
        "latency_ms": result["latency_ms"],
        "raw_text": raw_text if raw_text != text else None,
    }
