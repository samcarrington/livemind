# LiveMind — Quick Start for Tomorrow

LiveMind builds a live knowledge graph from your conversations. It transcribes speech in real time, identifies key concepts and relationships, and displays an animated map that evolves as the discussion progresses. After the session, it generates AI-powered recaps that surface non-obvious connections and key takeaways.

---

## Access

All views are at **livemind.btrbot.com**. You'll be prompted to authenticate via Cloudflare Access on first visit.

| View | URL | Who |
|------|-----|-----|
| **Display** | [livemind.btrbot.com](https://livemind.btrbot.com) | Projected for the audience — the live knowledge graph |
| **Monitor** | [livemind.btrbot.com/monitor](https://livemind.btrbot.com/monitor) | The operator running the session — audio, controls, metrics |
| **Sessions** | [livemind.btrbot.com/sessions](https://livemind.btrbot.com/sessions) | Post-session — recaps, transcripts, exports |

---

## How a Session Works

1. The **operator** opens `/monitor` and fills in the session topic, language, and audio device
2. Click **Start Session** — audio capture begins, the display wakes up
3. The audience sees the knowledge graph build in real time on the **display** (`/`)
4. When the session ends, the operator clicks **End Session** in the monitor header
5. After the session, go to `/sessions` to generate a recap, clean the transcript, and export

---

## What You Get After a Session

- **AI Recap** — elevator pitch, three things to retain, non-obvious connections, contradictions, decisions, open threads
- **Clean Transcript** — AI-cleaned version of the raw transcription, with timestamps
- **Cross-Session Synthesis** — select multiple sessions and synthesise connections across them
- **Export** — PDF and Markdown for both recaps and transcripts

---

## Key Things to Know

- **Language**: set it explicitly in the monitor setup (auto-detect works for English, not great for Norwegian)
- **The display is passive** — it just shows what's happening. All controls are in the monitor
- **Everything is saved** — transcripts, graph snapshots, recaps. Nothing is lost when a session ends
- **The graph manages itself** — max 24 nodes, automatic decay, importance scoring. No manual cleanup needed

---

## Operator Documentation

Full guide at **[livemind.btrbot.com/doc](https://livemind.btrbot.com/doc)** — covers setup, monitoring, post-session workflow, and troubleshooting.

---

*Questions? Reach out to the system operator or check the docs.*
