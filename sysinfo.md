# LiveMind — System Overview

*Real-time conversation visualization: live speech → transcription → AI analysis → animated knowledge graph*

Runs locally with all inference on-host. No external API calls required for core operation (cloud LLMs available as fallback).

---

## How It Works

1. A technician opens **`/monitor`** on their laptop and starts a session (picks topic, STT backend, LLM, language, audio device)
2. The monitor captures audio from the browser microphone, runs voice activity detection, and sends speech chunks to the server over WebSocket
3. The server dispatches audio to the active STT backend (faster-whisper, Parakeet, or Canary) and broadcasts the transcript to all connected clients
4. Every 20 seconds (or when enough new text accumulates), the **display view** (`/`) sends the current graph + new transcript to the LLM for analysis
5. The LLM returns an updated graph (nodes, edges, summary) which is reconciled through a scoring/decay system to keep the graph readable
6. The audience sees a live, animated D3 force graph that evolves as the conversation progresses
7. After the session, recaps can be generated and sessions can be browsed at **`/admin/sessions`**

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
                                    → graph update via WebSocket ──→  Browser (Display View)
                                    → SQLite persistence
```

Audio capture happens in the browser (monitor view), NOT on the server. The server never touches a microphone.

---

## Three Views

| URL | Purpose | Audience |
|-----|---------|----------|
| `/` | Clean visualization: D3.js force graph + transcript sidebar. No controls, no chrome. Defaults to big-screen mode. | Projected on screen for audience |
| `/monitor` | Two-phase control surface: session setup (topic, models, language) → live monitoring (audio, metrics, graph mirror) | Technician's laptop |
| `/admin/sessions` | Post-session: browse past sessions, replay graph evolution, generate AI recaps, export PDF/JSON | After the event |

---

## Components

### Server — `app.py`

FastAPI application serving REST + WebSocket endpoints. Single process, async throughout.

**REST Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve display view |
| GET | `/monitor` | Serve monitor view |
| GET | `/admin/sessions` | Serve session browser |
| POST | `/v1/sessions` | Create new session |
| POST | `/v1/sessions/new` | End current + create new session |
| POST | `/v1/sessions/{id}/end` | End session (final snapshot, summary, reset) |
| GET | `/v1/sessions/{id}/restore` | Get snapshot + segments for reconnection |
| GET | `/v1/sessions/{id}` | Full session detail (transcript, snapshot, recap) |
| GET | `/v1/sessions/{id}/snapshots` | All snapshots for playback |
| POST | `/v1/sessions/{id}/actions` | User actions (pin, hide, rename, merge, promote) |
| POST | `/v1/sessions/{id}/recap` | Generate AI recap |
| GET | `/v1/sessions` | List all sessions |
| GET | `/v1/metrics` | System metrics |
| GET | `/v1/llm/providers` | List LLM providers with health |
| POST | `/v1/llm/active` | Switch LLM provider/model |
| GET | `/v1/stt/backends` | List STT backends with health |
| POST | `/v1/stt/active` | Switch STT backend/language |

**WebSocket Protocol (`/ws`):**

Server → Browser:
- `transcript` — final transcript segment (`text`, `seq`, `timestamp`)
- `partial_transcript` — interim result
- `claude_response` — LLM analysis result (reconciled graph)
- `graph_update` — graph update from user action
- `restore` — session restore (snapshot + segments, sent on connect if session active)
- `session_reset` — new session started (ID + topic)
- `session_ended` — session ended
- `metrics` — metrics response
- `status` — connection status
- `pong` — heartbeat response

Browser → Server:
- `audio_chunk` — base64 PCM float32 audio from monitor
- `claude_request` — LLM proxy request (from display analysis)
- `connect_session` — reconnect to session
- `get_metrics` — request metrics
- `frontend_metrics` — FPS report from display
- `ping` — heartbeat

**LLM Proxy:**

All LLM calls are server-side (no API keys in browser). Three providers with a common wrapper:

| Provider | Endpoint | Models | Cost |
|----------|----------|--------|------|
| Hugin (Ollama) | `localhost:11434/api/chat` | gemma4:26b (live), gemma4:31b (recap) | Free (local) |
| Gemini | Google API | gemini-2.5-flash | Per-token |
| Claude | Anthropic API | claude-sonnet-4 | Per-token |

Live analysis uses the small/fast model (gemma4:26b). Recap generation uses the quality model (gemma4:31b, 4096 token budget). All responses normalized to Anthropic message format internally.

**Circuit Breaker:**

Protects against LLM failures cascading:
- `closed` → normal operation
- 3 consecutive failures → `open` (all requests short-circuited, exponential backoff 5s → 60s)
- After backoff → `half_open` (one test request)
- Success → `closed`; Failure → `open` again

Rate limit (429) responses immediately open the breaker.

**Background Tasks:**
- `broadcast_loop()` — polls transcript queue every 50ms, assigns sequence numbers, persists, broadcasts
- `snapshot_loop()` — persists full graph state every 60s

**Session Lifecycle:**
1. Monitor creates session via `/v1/sessions/new` — server assigns 8-char UUID, resets all state, broadcasts `session_reset`
2. During session — segments persisted with STT metadata, periodic snapshots, metrics accumulate
3. Monitor ends session via `/v1/sessions/{id}/end` — final snapshot stored, summary persisted, state cleared, `session_ended` broadcast
4. Metrics reset on new session (keeps WS clients, circuit breaker state)

---

### STT Worker — `stt_worker.py`

Audio transcription dispatch. Receives PCM from browser, resamples, sends to active backend.

**Pipeline:**
1. **Resample** — linear interpolation from source rate (typically 48kHz) to 16kHz
2. **Silence detection** — RMS < 0.005 → skip
3. **Backend dispatch** — POST raw float32 PCM to `/v1/transcribe` with query params (`sample_rate`, `language`, `initial_prompt`)
4. **Hallucination filter** — exact match against known Whisper hallucinations ("thank you.", "subscribe.", etc.) + repetition loop detection (same 2-word phrase 3+ times)
5. **Deduplication** — identical text within 10s → dropped
6. **Context update** — rolling 200-char window for Whisper prompt conditioning
7. **Return** — dict with `text`, `language`, `backend`, `latency_ms`, `raw_text` (or None if filtered)

**Backends:**

| Backend | Port | Languages | Strength |
|---------|------|-----------|----------|
| faster-whisper | 8766 | 99 | Proven, wide language coverage |
| Parakeet TDT 0.6b v3 | 8010 | 25 EU | Extreme throughput (3300x RTFx) |
| Canary 1b v2 | 8011 | 25 EU | Best accuracy (8.1% avg WER) |

All share the same HTTP interface. Language can be auto-detected or explicitly set. Canary's auto-detection struggles with Norwegian; works well for English and Swedish when set explicitly.

---

### Graph Reconciler — `reconciler.py`

Deterministic graph lifecycle management. Ensures the graph stays readable and reflects the conversation's evolution.

**Node States:** `active` → `parked` → `archived` / `hidden`

**Scoring Formula (active nodes only):**
```
importance = 0.45 × recency + 0.35 × frequency + 0.20 × centrality + pin_bonus

recency    = 2^(-age_seconds / 300)     # 5-minute half-life
frequency  = mention_count / max_mentions
centrality = edge_count / max_edges
pin_bonus  = 0.15 if pinned, else 0.0
```

**Reconciliation (on each LLM response):**
1. Update existing nodes / create new ones from LLM proposal
2. Check reactivation — parked nodes mentioned 2+ times in last 3 minutes get reactivated
3. Decay — active nodes NOT in proposal, older than 12 minutes, not pinned → parked
4. Update edges — filter to valid active/parked nodes, remove orphans
5. Score all active nodes
6. Budget enforcement — if more than 24 active, park lowest-scoring non-pinned nodes
7. Track churn (additions, removals, edge changes) for metrics

**User Actions:** pin/unpin, hide, rename, merge (redirect edges), promote (boost importance)

**Key Thresholds:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MAX_ACTIVE` | 24 | Hard budget for active nodes |
| `DECAY_SECONDS` | 720 (12 min) | Time before inactive nodes get parked |
| `REACTIVATION_WINDOW` | 180 (3 min) | Window to check for reactivation |
| `REACTIVATION_MENTIONS` | 2 | Mentions needed to reactivate |
| `POSITION_CLAMP` | 30 px | Max position shift per update (anti-jitter) |

---

### Database — `db.py`

SQLite with WAL mode via aiosqlite. Single file: `livemind.db`.

**Tables:**

```
sessions
├── id (TEXT PK)          — 8-char UUID
├── topic (TEXT)
├── created_at (REAL)
├── ended_at (REAL)
└── summary (TEXT)

segments
├── id (INTEGER PK)
├── session_id (FK)
├── seq (INTEGER)         — monotonic per session
├── text (TEXT)
├── is_partial (INTEGER)
├── timestamp (REAL)
├── stt_language (TEXT)   — detected language code
├── stt_backend (TEXT)    — which STT service
├── stt_latency_ms (INT) — processing time
└── stt_raw_text (TEXT)   — before hallucination filtering
    UNIQUE(session_id, seq)

snapshots
├── id (INTEGER PK)
├── session_id (FK)
├── seq_at (INTEGER)      — segment seq at snapshot time
├── graph_json (TEXT)     — full reconciler state
├── created_at (REAL)
└── trigger (TEXT)        — "periodic" | "analysis" | "end"

actions
├── id (INTEGER PK)
├── session_id (FK)
├── action_type (TEXT)    — pin | hide | rename | merge | promote
├── payload (TEXT)        — JSON
└── created_at (REAL)

recaps
├── id (INTEGER PK)
├── session_id (FK)       UNIQUE
├── recap_json (TEXT)     — structured recap
├── model (TEXT)
└── created_at (REAL)
```

---

### Display View — `live-mindmap.html`

Audience-facing, distraction-free visualization. Purely passive — never creates or ends sessions.

**Startup:** Auto-connects WebSocket on page load. Shows "Waiting for session to start" until the server sends a `restore` or `session_reset` message. Transitions to the graph view when a session is active; returns to waiting when the session ends.

**D3 Force Graph:**
- Nodes: circles with glow aura, labels, category tags. Birth animation (scale 0→1.2→1). Pinned nodes get thicker strokes.
- Edges: curved paths with relationship labels, draw-on animation.
- Context menu (right-click): pin, promote, rename, merge, hide.
- Big-screen mode (default): larger fonts, thicker strokes, enhanced contrast for projection.

**LLM Analysis Loop:**
- Runs every 20s (`C.interval`) if ≥50 chars of new text
- Sends current graph + last 3000 chars of new transcript + rolling summary to LLM
- System prompt enforces: max 30 nodes, preserve IDs, vary edge labels, return JSON
- Max 2 retries with exponential backoff on failure
- Enters degraded mode on rate limit (30s recovery) or server error (15s recovery)
- On success: applies graph via `applyGraph()`, updates summary, advances transcript pointer

**Sidebar:** Live transcript (chunks fade as new ones arrive) + concept list (colored dots with labels).

---

### Monitor View — `monitor.html`

Technician's control surface. Two-phase UI:

**Phase 1 — Session Setup** (shown when no session is active):
- Topic name input
- STT backend picker (radio buttons with health indicators)
- LLM picker (dropdown)
- Language picker (auto-detect, English, Norwegian, Swedish, Danish, German, French)
- Audio device picker
- "Start Session" button → configures STT + LLM on server, creates session, auto-starts audio capture, transitions to Phase 2

**Phase 2 — Live Monitor** (shown when session is active):
- **Header:** session ID, topic, duration timer, "End Session" button
- **Left column:** audio level meter, VAD indicator, waveform, language picker (changeable live), start/stop/mute
- **Center column:** connection health dots (WS/STT/LLM), segment count, metrics grid (STT latency, LLM latency, graph size, churn, parse success, cost, circuit breaker, audience FPS), STT backend picker (changeable live), LLM picker
- **Right column:** graph mirror (read-only D3 replica)
- **Bottom bar:** live transcript

**Browser-Side Audio Capture + VAD:**
- Web Audio API: `AudioContext` (48kHz) → `AnalyserNode` + `ScriptProcessorNode`
- VAD state machine with smoothed RMS:
  - `alpha = 0.3` (smoothing)
  - `onset threshold = 0.02`, requires 3 consecutive frames
  - `offset threshold = 0.008`, requires 25 consecutive frames
  - Pre-roll buffer: 0.5s (captures speech start)
  - Max speech: 5.0s (auto-flush)
  - Min speech: 1.0s (discard shorter)
- Audio chunks: float32 PCM → base64 → WebSocket `audio_chunk` message

**Metrics (polled every 2s via WebSocket):**
All metrics reset when a new session starts.

---

### Session Browser — `sessions.html`

Post-session analysis interface.

**Session List:** Table of all sessions (topic, ID, segment count, duration, recap status). Click to open detail.

**Session Detail (3 tabs):**

- **Recap:** AI-generated structured summary (executive summary, key topics, decisions, action items, open questions). "Generate Recap" button if none exists. Export to PDF or JSON.
- **Transcript:** All segments with timestamps. Each segment shows STT metadata badges (detected language, backend used, latency).
- **Mind Map:** Graph evolution playback — step through snapshots chronologically with play/pause and timeline slider.

Recap generation uses `gemma4:31b` (quality model) with up to 4096 output tokens. Transcript truncated at 150k chars if needed.

---

## Infrastructure

### Hardware: NVIDIA DGX Spark (hugin.local)
- GPU: NVIDIA GB10 (unified memory shared between CPU and GPU)
- CPU: 20 cores
- RAM: 119 GB (unified with GPU)
- OS: Linux (Ubuntu)

### Services

| Service | Port | Type | GPU Memory |
|---------|------|------|------------|
| LiveMind server (app.py) | 8765 | Python process | — |
| faster-whisper STT | 8766 | Python process | ~2.2 GB |
| Parakeet TDT | 8010 | Docker container | ~2.9 GB |
| Canary 1b v2 | 8011 | Docker container | ~10 GB |
| Ollama | 11434 | System service | ~36 GB (gemma4:26b loaded) |

Total GPU memory when all services loaded: ~51 GB. On unified memory architecture, all compete for bandwidth. Unloading unused STT backends (`docker stop canary-asr`) frees memory and can improve LLM inference speed.

### Networking
- **Cloudflare Tunnel:** `livemind.btrbot.com` → localhost:8765
- **Cloudflare Access:** service token auth
- **Related services:** `munin.btrbot.com` (Ollama API), `stt.btrbot.com` (faster-whisper)

### Dependencies

**Python** (server): fastapi, uvicorn, aiosqlite, aiohttp, numpy, requests

**Browser** (no build step): D3.js v7.8.5, html2pdf.js v0.10.1, Web Audio API, vanilla HTML/CSS/JS

---

## Key Design Principles

- **No frameworks beyond FastAPI.** Vanilla frontend, no build step.
- **Single SQLite database.** No external DB servers.
- **All LLM keys server-side only.** Browser gets session tokens.
- **Audience view is distraction-free.** All controls live in monitor view.
- **Circuit breaker on all external calls.** Graceful degradation — transcript keeps flowing even when LLM is down.
- **The graph must stay readable:** max 24 active nodes, automatic decay, importance scoring.
- **Audio capture in the browser, not on the server.** No microphone permissions needed on the server.
- **Session lifecycle owned by the monitor.** Display is passive — connects, receives, renders.
