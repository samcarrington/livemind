# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## What This Is

LiveMind: a real-time conversation visualization system that captures live speech, transcribes it, and builds an animated knowledge graph of concepts and relationships as people talk.

Runs locally with all inference on-host (cloud LLMs available as fallback). No external API calls required for core operation.

## Architecture Overview

```
Browser (Monitor View)              DGX Spark (hugin.local)
────────────────────                ────────────────────────
getUserMedia → Web Audio API
  → VAD (energy threshold)
  → PCM chunks via WebSocket ──→  app.py (FastAPI)
                                    → POST to localhost STT service
                                      (faster-whisper :8766 | parakeet :8010 | canary :8011)
                                    → transcript to LLM proxy
                                      (Ollama localhost:11434 | Gemini API | Claude API)
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

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the server (no device picker needed, audio comes from browser)
python app.py --host 0.0.0.0 --port 8765

# Or with specific port
python app.py --host 0.0.0.0 --port 8080
```

Server starts and is accessible via Cloudflare tunnel at `livemind.btrbot.com`.
Open `/monitor` on the technician's device, `/` on the audience-facing screen.

## File Structure

```
app.py              — FastAPI server: WS + REST routes, LLM proxy, STT relay, broadcast/snapshot loops
stt_worker.py       — WebSocket audio receiver + STT dispatch (faster-whisper, Parakeet, Canary)
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

## Infrastructure (DGX Spark)

### STT Services (choose via admin/monitor panel)
- **faster-whisper** at `localhost:8766` (also via `stt.btrbot.com`): 99 languages, proven, hallucination filtering in place
- **Parakeet TDT 0.6b v3** at `localhost:8010`: 25 European languages, extreme throughput (3300x RTFx), NeMo-based
- **Canary 1b v2** at `localhost:8011`: 25 European languages, best accuracy (8.1% avg WER), NeMo-based

### LLM Services (choose via admin/monitor panel)
- **Ollama** at `localhost:11434`: gemma4:26b (primary, fast MoE), gemma4:31b (quality fallback)
- **Gemini** via API: gemini-2.5-flash (current default, will migrate to local)
- **Claude** via API: claude-sonnet-4 (recap generation, fallback)

### Networking
- Cloudflare Tunnel: `livemind.btrbot.com` → localhost:8765
- Cloudflare Access: service token auth (same credentials as munin/stt)
- Related services: `munin.btrbot.com` (Ollama API), `stt.btrbot.com` (faster-whisper)

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

### LLM Proxy (`app.py`)
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

Python (server):
- fastapi, uvicorn, aiosqlite, aiohttp, numpy
- NO sounddevice, NO moshi_mlx, NO sentencepiece (removed: Mac-only)

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
