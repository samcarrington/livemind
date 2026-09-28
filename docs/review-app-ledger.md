# Review Decision Ledger — app

Tracks findings from reviews of `app.py`. Finding IDs remain stable across review iterations.

| ID | Introduced | Status | Resolved in | Summary |
|---|---:|---|---:|---|
| R1-001 | 1 | open |  | Ending a non-current session can persist the active graph under the wrong ID and clear active runtime state. |
| R1-002 | 1 | open |  | Sensitive session REST routes and the WebSocket lack application-level authentication/authorization and WebSocket Origin validation. |
| R1-003 | 1 | open |  | STT and Ollama calls disable TLS certificate verification, risking credential and transcript exposure for remote HTTPS endpoints. |
| R1-004 | 1 | open |  | Session reset/lifecycle logic is duplicated across handlers and reaches into reconciler internals. |
| R1-005 | 1 | open |  | Provider/fallback handling, accounting, graph processing, persistence, and broadcasts are coupled in the LLM proxy path. |
| R1-006 | 1 | open |  | Recap, synthesis, and transcript-cleaning workflows are embedded in endpoint handling rather than a post-session service/job boundary. |
| R1-007 | 1 | open |  | Detached audio transcription tasks can complete after a session reset and write results into the new session. |
| R1-008 | 1 | open |  | `session_action` can persist an action under one session ID while mutating the global active graph. |
| R1-009 | 1 | open |  | WebSocket audio and LLM messages can create unbounded work without queue, concurrency, or payload limits. |
