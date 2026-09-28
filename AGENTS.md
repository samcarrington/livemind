# AGENTS.md

Guidance for AI coding agents working in this repository. (`CLAUDE.md` imports this file.)

## What This Is

LiveMind: a real-time conversation visualization system that captures live speech, transcribes it, and builds an animated knowledge graph of concepts and relationships as people talk.

Runs locally with all inference on-host (cloud LLMs available as fallback). No external API calls required for core operation.

## Architecture Overview

```
Browser (Monitor View)              Server (this machine, or a Linux/GPU host)
────────────────────                ──────────────────────────────────────────
getUserMedia → Web Audio API
  → VAD (energy threshold)
  → PCM chunks via WebSocket ──→  app.py (FastAPI)
                                    → POST to STT service
                                      (stt_server.py :8766 [mlx-whisper | faster-whisper]
                                       | parakeet :8010 | canary :8011)
                                    → transcript to LLM proxy
                                      (Ollama :11434 | Gemini API | Claude API)
                                    → reconciler (scoring, decay, budget)
                                    → graph update via WebSocket ──→  Browser (Main View)
                                    → SQLite persistence
```

Audio capture happens in the browser (monitor view), NOT on the server. The server never touches a microphone. This is the key architectural difference from the original LiveMind.

## Three Views

| URL | Purpose | Audience |
|-----|---------|----------|
| `/` | Clean visualization: D3.js force graph + transcript sidebar. No controls, no chrome. | Projected on screen for audience |
| `/monitor` | Full control surface: audio device picker, gain meter, VAD indicator, status panel (STT/LLM/WS health), session controls, model switching, live metrics, small graph mirror | Technician's laptop |
| `/admin/sessions` | Post-session: browse past sessions, replay, generate AI recaps, export | After the event |

## Running the Project

Fully local by default: Ollama for the LLM and the bundled `stt_server.py` for speech-to-text. No tunnel or API keys required.

```bash
# One-off setup
uv sync --extra stt-mlx --extra stt   # mlx-whisper (Apple Silicon) + faster-whisper
cp .env.example .env                  # optional; defaults point at localhost
ollama pull gemma4:e4b                # or whatever OLLAMA_MODEL is set to

# Each run (three processes)
ollama serve                          # :11434, if not already running
make stt                              # local STT server on :8766 (stt_server.py)
make run                              # LiveMind on :8765 (uv run python app.py --host 0.0.0.0 --port 8765)
```

At startup the server warns if Ollama is unreachable or a configured model hasn't been pulled.
Open `/monitor` on the technician's device, `/` on the audience-facing screen.

To use a remote/tunnelled STT or Ollama instead, point `STT_SERVER_URL` / `HUGIN_BASE_URL` at it and set the Cloudflare Access service-token vars (`STT_CF_ID`/`STT_CF_SECRET`, `HUGIN_CF_ID`/`HUGIN_CF_SECRET`).

## File Structure

```
app.py              — Entry point: lifespan, router registration, CLI; re-exports LLM helpers for replay.py
settings.py         — .env loading + environment-derived config constants
routes/             — HTTP/WS surface (APIRouters; registered in app.py, order matters)
  pages.py          — Static HTML/SVG pages
  sessions.py       — Session lifecycle, restore/playback, graph actions, export
  post_session.py   — Recap, cross-session synthesis, transcript cleaning endpoints
  providers.py      — LLM/STT provider switching, /v1/metrics
  live.py           — /ws WebSocket endpoint
services/           — Runtime state and behaviour (never import from app.py)
  runtime.py        — Shared state: transcript queue, clients, metrics, activity log, reconciler
  session_runtime.py — Current session, seq/generation counters, reset helpers (`live_session`)
  llm.py            — Provider adapters, fallback chain, circuit breakers, pricing, JSON extraction
  live_pipeline.py  — Audio → STT, LLM graph ingestion, broadcast/snapshot loops
  post_session.py   — Background transcript-cleaning jobs
stt_worker.py       — Audio chunk → STT dispatch (remote/local whisper, Parakeet, Canary), hallucination filter
stt_server.py       — Local STT server (:8766): mlx-whisper on Apple Silicon, else faster-whisper
replay.py           — Replay a stored transcript through the LLM pipeline (imports helpers from app.py)
export.py           — PDF/video export of a session graph (Playwright)
db.py               — SQLite persistence (sessions, segments, snapshots, actions, recaps)
reconciler.py       — Deterministic graph reconciler (scoring, decay, budget enforcement)
static/             — All HTML/SVG files served by FastAPI
  live-mindmap.html — Audience view: D3.js force graph + transcript sidebar (NO controls)
  monitor.html      — Technician view: audio capture, device picker, status panel, session/model controls
  sessions.html     — Session browser: list, detail, recap generation, export
  admin.html        — Legacy admin dashboard
  doc.html          — User documentation
  doc-admin.html    — Admin documentation
  export-graph.html — Export helper page
```

## Infrastructure

### STT Services (choose via monitor panel)
- **Local whisper** (`stt_server.py`) at `STT_SERVER_URL` (default `http://localhost:8766`). Backend name `remote`, the default (`STT_BACKEND`). Exposes `GET /health` and `POST /v1/transcribe` (raw float32 PCM).
  - Engine `auto` (default): **mlx-whisper** `mlx-community/whisper-large-v3-turbo` on Apple Silicon, else **faster-whisper** `small` (CPU int8, or CUDA).
  - Override with `--engine`/`--model`/`--device`/`--compute-type`/`--host`/`--port` or `STT_ENGINE`, `STT_MODEL`, `STT_DEVICE`, `STT_COMPUTE_TYPE`, `STT_HOST`, `STT_PORT`.
  - Any faster-whisper-compatible `/v1/transcribe` server works (e.g. a tunnelled GPU host).
- **Parakeet TDT 0.6b v3** at `PARAKEET_URL` (default `localhost:8010`): 25 European languages, very high throughput, NeMo-based. External service, not bundled.
- **Canary 1b v2** at `CANARY_URL` (default `localhost:8011`): 25 European languages, best accuracy, NeMo-based. External service, not bundled.

### LLM Services (choose via monitor panel)
Default fallback chain is built at startup from what's configured: **Ollama → Gemini → Anthropic**.
- **Ollama** (provider `hugin`) at `HUGIN_BASE_URL` / `OLLAMA_BASE_URL` (default `http://localhost:11434`):
  - `OLLAMA_MODEL`: live graph extraction, default `gemma4:e4b`.
  - `OLLAMA_RECAP_MODEL`: recaps, synthesis and transcript cleanup (blank = `OLLAMA_MODEL`).
  - `OLLAMA_NUM_CTX` (16384), `OLLAMA_KEEP_ALIVE` (30m), `OLLAMA_MAX_CONCURRENCY` (1).
- **Gemini** via API (`GEMINI_API_KEY`): gemini-2.5-flash. Added to the chain only if the key is set.
- **Claude** via API (`ANTHROPIC_API_KEY`): claude-sonnet-4. Last resort; added only if the key is set.

### Networking (optional, remote deployments)
- Cloudflare Tunnel can expose LiveMind (e.g. `livemind.btrbot.com` → :8765).
- Remote STT/Ollama behind Cloudflare Access use the `*_CF_ID` / `*_CF_SECRET` service-token env vars.

## Key Configuration

### Frontend (`live-mindmap.html`)
Located in the `C` object:
- `C.interval`: Claude/LLM analysis interval in ms (default 20000)
- `C.minLen`: minimum new chars before triggering analysis (default 50)
- `C.maxN`: max nodes in LLM prompt (default 30)

### Reconciler (`reconciler.py`)
- `MAX_ACTIVE`: 24 nodes max
- `DECAY_SECONDS`: 720 (12 min to parked)
- Scoring: `0.45*recency + 0.35*frequency + 0.20*centrality + pin_bonus`

### Environment (`.env`, see `.env.example`)
`settings.py` and `stt_server.py` both load `.env` from the repo root (existing process env wins). `DB_PATH` is read from the process env only (default `livemind.db`).

### LLM Proxy (`services/llm.py`)
- Provider switching: anthropic / hugin (Ollama) / gemini
- Circuit breaker: 3 failures → open, exponential backoff to 60s max
- Server-side only: no API keys in browser

## WebSocket Protocol

Server → Browser:
- `{"type":"transcript","text":"...","seq":N,"timestamp":T}` — final transcript
- `{"type":"partial_transcript","text":"...","seq":N,"timestamp":T}` — partial
- `{"type":"claude_response","status":200,"data":{...},"req_id":"..."}` — LLM result (reconciled)
- `{"type":"graph_update","graph":{...}}` — graph update from user action
- `{"type":"restore","snapshot":{...},"segments":[...],"restore_ms":N}` — session restore
- `{"type":"session_reset","session_id":"..."}` — new session started
- `{"type":"status","status":"connected","message":"..."}` — connection status
- `{"type":"metrics",...}` — metrics response

Browser → Server:
- `{"type":"ping"}` — keepalive
- `{"type":"get_metrics"}` — request metrics
- `{"type":"claude_request","req_id":"...","body":{...}}` — LLM proxy request
- `{"type":"connect_session","session_id":"...","last_seq":N}` — reconnect
- `{"type":"frontend_metrics","fps":N}` — FPS report
- `{"type":"audio_chunk","data":"<base64 PCM>","sample_rate":48000}` — audio from monitor (NEW)

## Dependencies

Python (managed by uv: `pyproject.toml` + `uv.lock`):
- Server: fastapi, uvicorn, aiosqlite, aiohttp, numpy, requests
- Extra `stt`: faster-whisper (local STT on CPU/CUDA)
- Extra `stt-mlx`: mlx-whisper (local STT on Apple Silicon)
- NO sounddevice, NO moshi_mlx, NO sentencepiece (removed: Mac-only). `requirements.txt` is stale; don't use it.

External processes: Ollama (`ollama serve`); optionally Parakeet/Canary NeMo services.

Browser (no build step):
- D3.js (force graph)
- Web Audio API (mic capture, VAD)
- Vanilla HTML/CSS/JS

## Design Principles

- No frameworks beyond FastAPI. Vanilla frontend.
- No build step. HTML files served directly.
- Single SQLite database. No external DB servers.
- All LLM keys server-side only. Browser gets session tokens.
- Audience view is distraction-free. All controls live in monitor view.
- Circuit breaker on all external calls. Graceful degradation (transcript keeps flowing).
- The graph must stay readable: max 24 active nodes, automatic decay, importance scoring.
