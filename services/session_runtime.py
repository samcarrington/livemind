"""Live-session runtime: current session id, transcript sequence, rolling
summary, session generation, and the reset helpers used on session start/end.

State lives on the ``live_session`` singleton; always access it as
``live_session.<attr>`` (never ``from ... import`` the scalar values).
"""

import threading

from services.runtime import reconciler, transcript_queue


class SessionState:
    def __init__(self) -> None:
        self.current_session_id: str | None = None
        self.summary: str = ""
        # Monotonic sequence counter for transcript messages
        self.seq_counter: int = 0
        self.seq_lock = threading.Lock()
        # Monotonic session-generation counter. Bumped every time
        # /v1/sessions/new or /v1/sessions/{id}/end resets state. In-flight
        # proxy_claude tasks capture it at request time; if it has changed by
        # the time the LLM call returns, the response is stale and must be
        # discarded (otherwise a late response can repopulate the reconciler
        # with nodes from the previous session).
        self.session_gen: int = 0
        self.session_gen_lock = threading.Lock()

    def bump_gen(self) -> int:
        with self.session_gen_lock:
            self.session_gen += 1
            return self.session_gen

    def next_seq(self) -> int:
        with self.seq_lock:
            self.seq_counter += 1
            return self.seq_counter


live_session = SessionState()


def clear_graph() -> None:
    """Empty the live reconciler graph and its churn/mention history."""
    reconciler.nodes.clear()
    reconciler.edges.clear()
    reconciler._mention_log.clear()
    reconciler._churn_log.clear()


def drain_transcript_queue() -> None:
    """Drop any leftover transcript messages from the STT queue."""
    while not transcript_queue.empty():
        try:
            transcript_queue.get_nowait()
        except Exception:
            break


def reset_live_state() -> None:
    """Clear graph, summary, sequence counter and pending transcripts."""
    clear_graph()
    live_session.summary = ""
    with live_session.seq_lock:
        live_session.seq_counter = 0
    drain_transcript_queue()
