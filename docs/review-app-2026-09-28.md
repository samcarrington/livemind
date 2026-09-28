# Review Report — app — 2026-09-28

## Summary

**Overall vote: Needs revision.** The review identified nine findings: three High and six Medium. The highest risks are cross-session state corruption, missing application-level access controls, and disabled TLS verification. No Critical or Low findings were identified.

## Top Fixes

1. [R1-001] Prevent ending a historical session from persisting or clearing active-session state.
2. [R1-002] Add application-level access controls to sensitive REST routes and the WebSocket.
3. [R1-003] Restore TLS certificate verification for remote STT and Ollama connections.

## Reviewer Votes

| Reviewer | Role | Vote |
|---|---|---|
| Software architect | Architect | Acceptable with caveats |
| Maintainability reviewer | Maintainability | Needs revision |
| QA engineer | QA | Needs revision |
| Security analyst | Security | Needs revision |

## Recommendations

### Critical

None identified.

### High

#### R1-001 — Cross-session corruption in `end_session`

**Reviewer:** QA engineer  
**Location:** `app.py:580-600`

The snapshot uses the requested session ID, while reconciler clearing and sequence reset occur unconditionally. Only the current-session ID reset is guarded. Ending a historical or non-current session can therefore save the live graph under the old ID and wipe active state.

**Recommendation:** Reject attempts to end a non-current session, or ensure global runtime state is modified only when the requested ID is current. Test that the active ID, graph, and sequence survive ending another session.

#### R1-002 — Sensitive routes lack application-level access control

**Reviewer:** Security analyst  
**Location:** `app.py:2347-2368` (`/ws`); `app.py:438-485` (session REST routes)

`/ws` accepts clients without identity or Origin validation and sends active snapshots and transcripts. Session REST routes also lack application-level authentication. Reachable clients can read conversations and invoke session-changing or LLM operations; the allowed origin is broad.

**Recommendation:** Enforce authentication and authorization on sensitive API routes and validate WebSocket Origin. Preserve these controls during modularisation.

#### R1-003 — TLS verification disabled

**Reviewer:** Security analyst  
**Location:** `app.py:1725-1737` (STT health); `app.py:1867-1874` (Ollama requests)

The STT health request sends `CF-Access-Client-Secret` with `verify=False`, while Ollama requests use `ssl=False`. For remote HTTPS endpoints, endpoint impersonation could expose service credentials or transcripts.

**Recommendation:** Enable certificate verification and configure a trusted CA bundle where necessary.

### Medium

#### R1-004 — Session-runtime responsibilities are duplicated and coupled

**Reviewers:** Software architect, Maintainability reviewer  
**Location:** `app.py:488-608`, especially `513-520`, `544-555`, and `586-598`

Session generation, snapshots, reconciler clearing, summaries, sequence resets, and queue draining are coordinated inside endpoints. Reset logic is duplicated, `new_session` repeats clears, and handlers clear private reconciler logs.

**Recommendation:** Introduce a session-runtime coordinator that owns session ID, generation, sequence, reset, and end persistence. Add a public `GraphReconciler.reset()` and preserve endpoint behavior contracts.

#### R1-005 — Proxy/provider and graph-processing concerns are entangled

**Reviewers:** Software architect, Maintainability reviewer  
**Location:** `app.py:1815-1963`, `2028-2294`, including `_proxy_claude` at `2131-2294`

Provider translation, selection, circuit breaking, metrics, token/cost accounting, parsing, stale checks, reconciliation, snapshots, broadcasts, and errors share one path.

**Recommendation:** Separate provider adapters and fallback policy from graph ingestion. Split accounting, graph processing, and broadcasting, leaving a thin proxy orchestrator.

#### R1-006 — Post-session processing belongs behind a service/job boundary

**Reviewer:** Software architect  
**Location:** recap `app.py:785-990`; synthesis `996-1278`; transcript cleaning `1284-1490`

Endpoints contain prompt construction, LLM calls, parsing, and persistence; transcript cleaning also manages job/progress state.

**Recommendation:** Move post-session processing into a service/job boundary while retaining API response and status contracts.

#### R1-007 — Detached audio tasks can cross session resets

**Reviewer:** QA engineer  
**Location:** `app.py:2302-2327`, `2441-2445`

Detached `_handle_audio_chunk` tasks await executor transcription and then read the global current session. Queue draining does not invalidate in-flight tasks, so an old chunk can enter a new session.

**Recommendation:** Capture session ID/generation on receipt and discard stale results. Add a deterministic test that pauses transcription across a session reset.

#### R1-008 — `session_action` can mutate the wrong session

**Reviewer:** QA engineer  
**Location:** `app.py:650-660`

The action is persisted under the path ID but mutates and broadcasts the global active graph. A mismatched target can change the live graph while logging the action against another session.

**Recommendation:** Reject mismatched IDs or introduce session-scoped state, with a regression test.

#### R1-009 — WebSocket task creation lacks backpressure

**Reviewer:** Security analyst  
**Location:** `app.py:2410-2411`, `2441-2445`

Claude and audio messages spawn tasks without concurrency, queue, payload-size, or rate limits, risking CPU, memory, and provider-cost exhaustion.

**Recommendation:** Add bounded concurrency/queues, payload and rate validation, and explicit overload responses.

### Low

None identified.

## Adjudicator Notes

Reviewer votes were normalized to the report rubric. The architect's “acceptable with caveats” vote was consistent with its medium-severity structural findings. The maintainability and QA reviewers' “request_changes” votes were normalized to “needs revision”; the QA vote is supported by a High correctness finding. The security analyst voted “needs revision,” consistent with the two High findings. The overall vote is “needs revision.”

Duplicate merges: `ARCH-001` and `MAINT-001` were merged into R1-004, retaining both recommendations about shared reset ownership and a public reconciler reset API. `ARCH-002` and `MAINT-002` were merged into R1-005, retaining both provider/fallback separation and decomposition of proxy responsibilities. R1-001 remains a distinct correctness defect.

No reviewer directly covered performance/concurrency; a focused follow-up on task lifecycle, backpressure, and executor behavior is advisable. The caller did not independently validate reviewer evidence or line ranges. Graph coverage reported no recorded issue, but that check is best-effort; recheck cited ranges when implementing, especially R1-001, R1-002, and R1-003. R1-002 concerns application-level controls only; network-layer protections were outside the reviewed evidence and are not assumed.
