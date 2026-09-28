# Live Mind Map

Real-time conversation visualization. Browser-captured audio is transcribed on-host, then AI extracts concepts and relationships into an animated, interactive mind map — live as people speak.

![Python](https://img.shields.io/badge/python-3.12+-blue)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Docker-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## What it does

As people talk in a meeting, workshop, or lecture, the system:

1. **Captures audio** in the browser — no server-side microphone needed
2. **Transcribes locally** on-host via faster-whisper, Parakeet, or Canary — nothing leaves your infrastructure
3. **Extracts concepts** every 20 seconds via LLM, identifying key ideas and their relationships
4. **Renders a live map** — an animated force-directed graph that grows, evolves, and self-manages as the conversation unfolds

Concepts that stop being discussed fade out. Important ideas stick around. The map stays clean even in a 2-hour session.

## Quick start

```bash
# Clone
git clone https://github.com/mctar/livemind.git
cd livemind

# Install dependencies with uv
uv sync

# Copy and fill in environment variables
cp .env.example .env
# Set ANTHROPIC_API_KEY and/or GEMINI_API_KEY as needed

# Run
uv run python app.py --host 0.0.0.0 --port 8765
```

### Fully local (Ollama + local Whisper, no tunnel)

```bash
uv sync --extra stt-mlx --extra stt   # mlx-whisper (Apple Silicon) + faster-whisper
ollama pull gemma4:e4b                # or set OLLAMA_MODEL in .env
ollama serve                          # if not already running (:11434)
make stt                              # local STT server on :8766 (stt_server.py)
make run                              # LiveMind on :8765
```

Defaults: `HUGIN_BASE_URL=http://localhost:11434`, `STT_SERVER_URL=http://localhost:8766`,
`OLLAMA_MODEL=gemma4:e4b` (live graph), `OLLAMA_RECAP_MODEL` (recaps/cleanup; blank = `OLLAMA_MODEL`),
`OLLAMA_NUM_CTX=16384`, `OLLAMA_KEEP_ALIVE=30m`, `OLLAMA_MAX_CONCURRENCY=1`. The server warns at
startup if a configured model hasn't been pulled.
`stt_server.py` picks mlx-whisper (`whisper-large-v3-turbo`) on Apple Silicon, otherwise
faster-whisper (`small`, CPU int8) — override with `--engine` / `--model` or `STT_ENGINE` / `STT_MODEL`.

Open `/monitor` on the technician's device to select audio input and start a session.  
Open `/` on the audience-facing screen for the clean visualization.

## Three views

| URL | Purpose | Who uses it |
|-----|---------|-------------|
| [`/`](http://localhost:8765/) | Clean visualization: D3.js force graph + transcript sidebar. No controls. | Projected for the audience |
| [`/monitor`](http://localhost:8765/monitor) | Full control surface: audio device picker, gain meter, VAD indicator, STT/LLM/WS status, session controls, model switching, live metrics | Technician's laptop |
| [`/admin/sessions`](http://localhost:8765/admin/sessions) | Browse past sessions, replay, generate AI recaps, export | After the event |

## Requirements

- **Python 3.12+** and [uv](https://docs.astral.sh/uv/)
- **STT service** — one of faster-whisper, Parakeet TDT, or Canary running locally or via tunnel
- **LLM** — Ollama (local), Gemini API key, or Anthropic API key

Audio capture is entirely browser-based. No microphone access is needed on the server.

## Architecture

```
Browser (Monitor View)              Server (Linux / Docker)
──────────────────────              ───────────────────────
getUserMedia → Web Audio API
  → VAD (energy threshold)
  → PCM chunks via WebSocket ──→  app.py (FastAPI)
                                    → POST to localhost STT service
                                      (faster-whisper | Parakeet | Canary)
                                    → transcript to LLM
                                      (Ollama | Gemini API | Claude API)
                                    → reconciler (scoring, decay, budget)
                                    → graph update via WebSocket ──→  Browser (Main View)
                                    → SQLite persistence
```

```
app.py              FastAPI server — WebSocket + REST, LLM proxy, STT relay, broadcast loops
stt_worker.py       WebSocket audio receiver + STT dispatch
db.py               SQLite persistence (sessions, segments, snapshots, actions, recaps)
reconciler.py       Deterministic graph reconciler — scoring, decay, budget enforcement
static/
  live-mindmap.html Audience view — D3.js force-directed graph + transcript sidebar
  monitor.html      Technician view — audio capture, device picker, status panel, session controls
  sessions.html     Session browser — list, detail, recap generation, export
  admin.html        Legacy admin dashboard
  doc.html / doc-admin.html  User and admin documentation
  export-graph.html Export helper page
```

## STT services

| Service | Languages | Notes |
|---------|-----------|-------|
| **faster-whisper** | 99 | Proven, hallucination filtering included |
| **Parakeet TDT 0.6b v3** | 25 European | 3300× RTFx, extreme throughput |
| **Canary 1b v2** | 25 European | Best accuracy (8.1% avg WER) |

Switch between them live from the monitor panel.

## LLM providers

| Provider | Model | Notes |
|----------|-------|-------|
| **Ollama** (local) | gemma4:e4b (default, set `OLLAMA_MODEL`) | Zero egress, recommended |
| **Gemini API** | gemini-2.5-flash | Cloud fallback |
| **Claude API** | claude-sonnet-4 | Recap generation, cloud fallback |

Switch between them live from the monitor panel. All API keys stay server-side.

## Configuration

Key settings in `live-mindmap.html` (the `C` object):

| Setting | Default | Description |
|---------|---------|-------------|
| `interval` | 20000 | LLM analysis interval (ms) |
| `minLen` | 50 | Min new chars before triggering analysis |
| `maxN` | 30 | Max nodes in LLM prompt |

Reconciler settings in `reconciler.py`: `MAX_ACTIVE = 24`, `DECAY_SECONDS = 720`.

## Docker

```bash
docker build -t livemind .
docker run -p 8765:8765 --env-file .env -v /data:/data livemind
```

Mount a persistent volume at `/data` to preserve `livemind.db` across restarts.

## Azure deployment

The `Makefile` targets build and push to Azure Container Registry, then deploy to Azure Container Apps.

```bash
# Set in .env (or export):
# AZURE_RESOURCE_GROUP, ACR_NAME

make login    # az acr login
make build    # docker build + tag
make push     # docker push to ACR
make deploy   # login + build + push in one step
```

The app is served via Cloudflare Tunnel (`livemind.btrbot.com`) with Cloudflare Access for auth.

## Session management

Sessions store everything: transcripts, graph snapshots, user actions. Use **New Session** in the monitor to archive the current session and start fresh. Browse and replay past sessions at `/admin/sessions`.

To wipe all history: delete `livemind.db` and restart.

## License

MIT

## Contact

Thordur Arnason (Lead)
Ian Davies (AI Factory Deployment)
