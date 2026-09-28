"""LiveMind server configuration: ``.env`` loading and environment-derived constants.

Import this module *after* ``db`` so ``db.DB_PATH`` keeps its existing
precedence (process env only, not ``.env``).
"""

import os

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
RECAP_MAX_CHARS = max(8000, min(150000, (OLLAMA_NUM_CTX - 4096 - 2000) * 3))
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
VALID_PROVIDERS = ("hugin", "gemini", "anthropic")
