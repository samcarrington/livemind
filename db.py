"""
SQLite persistence for Live Mind Map sessions.
Uses aiosqlite for async access with WAL mode for concurrent reads.
"""

import json, time, os
import aiosqlite

DB_PATH = os.environ.get("DB_PATH", "livemind.db")
_db: aiosqlite.Connection | None = None


async def init_db(path: str = DB_PATH):
    """Initialize database connection and create schema."""
    global _db
    _db = await aiosqlite.connect(path)
    _db.row_factory = aiosqlite.Row
    await _db.execute("PRAGMA journal_mode=WAL")
    await _db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            topic TEXT,
            created_at REAL NOT NULL,
            ended_at REAL,
            summary TEXT
        );
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            seq INTEGER NOT NULL,
            text TEXT NOT NULL,
            is_partial INTEGER NOT NULL DEFAULT 0,
            timestamp REAL NOT NULL,
            UNIQUE(session_id, seq)
        );
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            seq_at INTEGER NOT NULL,
            graph_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            trigger TEXT
        );
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            action_type TEXT NOT NULL,
            payload TEXT,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS recaps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            recap_json TEXT NOT NULL,
            model TEXT,
            created_at REAL NOT NULL,
            UNIQUE(session_id)
        );
    """)
    await _db.commit()

    # ─── Migrations ───
    # Add STT metadata columns to segments (idempotent)
    cursor = await _db.execute("PRAGMA table_info(segments)")
    existing_cols = {row[1] for row in await cursor.fetchall()}
    migrations = [
        ("stt_language", "TEXT DEFAULT ''"),
        ("stt_backend", "TEXT DEFAULT ''"),
        ("stt_latency_ms", "INTEGER"),
        ("stt_raw_text", "TEXT"),
        ("cleaned_text", "TEXT"),
    ]
    for col_name, col_def in migrations:
        if col_name not in existing_cols:
            await _db.execute(f"ALTER TABLE segments ADD COLUMN {col_name} {col_def}")

    # Add archived and source columns to sessions (idempotent)
    cursor = await _db.execute("PRAGMA table_info(sessions)")
    session_cols = {row[1] for row in await cursor.fetchall()}
    if "archived" not in session_cols:
        await _db.execute("ALTER TABLE sessions ADD COLUMN archived INTEGER DEFAULT 0")
    if "source" not in session_cols:
        await _db.execute("ALTER TABLE sessions ADD COLUMN source TEXT DEFAULT 'live'")

    # Cross-session synthesis recaps table
    await _db.execute("""
        CREATE TABLE IF NOT EXISTS synthesis_recaps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_ids TEXT NOT NULL,
            recap_json TEXT NOT NULL,
            model TEXT,
            created_at REAL NOT NULL
        )
    """)
    await _db.commit()

    return _db


async def close_db():
    """Close the database connection."""
    global _db
    if _db:
        await _db.close()
        _db = None


async def create_session(session_id: str, topic: str = "", source: str = "live") -> dict:
    """Create a new session. source: 'live' or 'replay'. Returns session dict."""
    now = time.time()
    await _db.execute(
        "INSERT INTO sessions (id, topic, created_at, source) VALUES (?, ?, ?, ?)",
        (session_id, topic, now, source),
    )
    await _db.commit()
    return {"id": session_id, "topic": topic, "created_at": now, "source": source}


async def end_session(session_id: str, summary: str = ""):
    """Mark a session as ended."""
    await _db.execute(
        "UPDATE sessions SET ended_at = ?, summary = ? WHERE id = ?",
        (time.time(), summary, session_id),
    )
    await _db.commit()


async def store_segment(session_id: str, seq: int, text: str, is_partial: bool, timestamp: float,
                        stt_language: str = "", stt_backend: str = "", stt_latency_ms: int | None = None,
                        stt_raw_text: str | None = None):
    """Store a transcript segment with optional STT metadata."""
    await _db.execute(
        "INSERT OR REPLACE INTO segments (session_id, seq, text, is_partial, timestamp, stt_language, stt_backend, stt_latency_ms, stt_raw_text)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, seq, text, int(is_partial), timestamp, stt_language, stt_backend, stt_latency_ms, stt_raw_text),
    )
    await _db.commit()


async def get_segments_since(session_id: str, from_seq: int) -> list[dict]:
    """Get all segments with seq > from_seq for a session."""
    cursor = await _db.execute(
        "SELECT seq, text, is_partial, timestamp FROM segments WHERE session_id = ? AND seq > ? ORDER BY seq",
        (session_id, from_seq),
    )
    rows = await cursor.fetchall()
    return [{"seq": r["seq"], "text": r["text"], "is_partial": bool(r["is_partial"]), "timestamp": r["timestamp"]} for r in rows]


async def store_snapshot(session_id: str, seq_at: int, graph: dict, trigger: str = "periodic"):
    """Store a graph snapshot."""
    await _db.execute(
        "INSERT INTO snapshots (session_id, seq_at, graph_json, created_at, trigger) VALUES (?, ?, ?, ?, ?)",
        (session_id, seq_at, json.dumps(graph), time.time(), trigger),
    )
    await _db.commit()


async def get_latest_snapshot(session_id: str) -> dict | None:
    """Get the most recent snapshot for a session."""
    cursor = await _db.execute(
        "SELECT seq_at, graph_json, created_at, trigger FROM snapshots WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
        (session_id,),
    )
    row = await cursor.fetchone()
    if not row:
        return None
    return {
        "seq_at": row["seq_at"],
        "graph": json.loads(row["graph_json"]),
        "created_at": row["created_at"],
        "trigger": row["trigger"],
    }


async def store_action(session_id: str, action_type: str, payload: dict):
    """Store a user action (pin, hide, rename, merge, promote)."""
    await _db.execute(
        "INSERT INTO actions (session_id, action_type, payload, created_at) VALUES (?, ?, ?, ?)",
        (session_id, action_type, json.dumps(payload), time.time()),
    )
    await _db.commit()


async def get_session(session_id: str) -> dict | None:
    """Fetch a single session's metadata by id, regardless of archive state.
    Returns the same shape as list_sessions() entries."""
    cursor = await _db.execute("""
        SELECT s.id, s.topic, s.created_at, s.ended_at, s.summary,
               COALESCE(s.archived, 0) AS archived,
               COALESCE(s.source, 'live') AS source,
               (SELECT COUNT(*) FROM segments WHERE session_id = s.id AND is_partial = 0) AS segment_count,
               (SELECT SUM(LENGTH(text)) FROM segments WHERE session_id = s.id AND is_partial = 0) AS total_chars,
               (SELECT COUNT(*) FROM snapshots WHERE session_id = s.id) AS snapshot_count,
               (SELECT COUNT(*) FROM recaps WHERE session_id = s.id) AS has_recap
        FROM sessions s
        WHERE s.id = ?
    """, (session_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None


async def list_sessions(archived: bool = False) -> list[dict]:
    """List sessions with computed metadata. archived=True returns archived sessions."""
    cursor = await _db.execute("""
        SELECT s.id, s.topic, s.created_at, s.ended_at, s.summary,
               COALESCE(s.archived, 0) AS archived,
               COALESCE(s.source, 'live') AS source,
               (SELECT COUNT(*) FROM segments WHERE session_id = s.id AND is_partial = 0) AS segment_count,
               (SELECT SUM(LENGTH(text)) FROM segments WHERE session_id = s.id AND is_partial = 0) AS total_chars,
               (SELECT COUNT(*) FROM snapshots WHERE session_id = s.id) AS snapshot_count,
               (SELECT COUNT(*) FROM recaps WHERE session_id = s.id) AS has_recap
        FROM sessions s
        WHERE COALESCE(s.archived, 0) = ?
        ORDER BY s.created_at DESC
    """, (1 if archived else 0,))
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def archive_sessions(session_ids: list[str]):
    """Move sessions to archive."""
    for sid in session_ids:
        await _db.execute("UPDATE sessions SET archived = 1 WHERE id = ?", (sid,))
    await _db.commit()


async def unarchive_sessions(session_ids: list[str]):
    """Move sessions out of archive."""
    for sid in session_ids:
        await _db.execute("UPDATE sessions SET archived = 0 WHERE id = ?", (sid,))
    await _db.commit()


async def delete_sessions(session_ids: list[str]):
    """Permanently delete sessions and all associated data."""
    for sid in session_ids:
        await _db.execute("DELETE FROM segments WHERE session_id = ?", (sid,))
        await _db.execute("DELETE FROM snapshots WHERE session_id = ?", (sid,))
        await _db.execute("DELETE FROM actions WHERE session_id = ?", (sid,))
        await _db.execute("DELETE FROM recaps WHERE session_id = ?", (sid,))
        await _db.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    await _db.commit()


async def get_session_transcript(session_id: str) -> list[dict]:
    """Get all final transcript segments for a session."""
    cursor = await _db.execute(
        "SELECT seq, text, timestamp, stt_language, stt_backend, stt_latency_ms, stt_raw_text, cleaned_text"
        " FROM segments WHERE session_id = ? AND is_partial = 0 ORDER BY seq",
        (session_id,),
    )
    rows = await cursor.fetchall()
    return [{
        "seq": r["seq"], "text": r["text"], "timestamp": r["timestamp"],
        "stt_language": r["stt_language"] or "", "stt_backend": r["stt_backend"] or "",
        "stt_latency_ms": r["stt_latency_ms"], "stt_raw_text": r["stt_raw_text"],
        "cleaned_text": r["cleaned_text"],
    } for r in rows]


async def store_cleaned_segments(session_id: str, cleaned: list[dict]):
    """Store cleaned text for segments. Each item: {seq, cleaned_text}."""
    for item in cleaned:
        await _db.execute(
            "UPDATE segments SET cleaned_text = ? WHERE session_id = ? AND seq = ?",
            (item["cleaned_text"], session_id, item["seq"]),
        )
    await _db.commit()


async def store_recap(session_id: str, recap: dict, model: str = ""):
    """Store a generated recap (replaces any existing)."""
    await _db.execute(
        "INSERT OR REPLACE INTO recaps (session_id, recap_json, model, created_at) VALUES (?, ?, ?, ?)",
        (session_id, json.dumps(recap), model, time.time()),
    )
    await _db.commit()


async def get_recap(session_id: str) -> dict | None:
    """Get the stored recap for a session, if any."""
    cursor = await _db.execute(
        "SELECT recap_json, model, created_at FROM recaps WHERE session_id = ?",
        (session_id,),
    )
    row = await cursor.fetchone()
    if not row:
        return None
    return {
        "recap": json.loads(row["recap_json"]),
        "model": row["model"],
        "created_at": row["created_at"],
    }


async def get_session_snapshots(session_id: str) -> list[dict]:
    """Get all snapshots for a session, chronologically, with consecutive duplicate suppression."""
    cursor = await _db.execute(
        "SELECT seq_at, graph_json, created_at, trigger FROM snapshots "
        "WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    )
    rows = await cursor.fetchall()
    results = []
    prev_json = None
    for r in rows:
        json_str = r["graph_json"]
        if json_str == prev_json:
            continue
        prev_json = json_str
        results.append({
            "seq_at": r["seq_at"],
            "graph": json.loads(json_str),
            "created_at": r["created_at"],
            "trigger": r["trigger"],
        })
    return results


# ─── Cross-session synthesis ───

async def store_synthesis(session_ids: list[str], recap: dict, model: str = "") -> int:
    """Store a cross-session synthesis recap. Returns the new row ID."""
    cursor = await _db.execute(
        "INSERT INTO synthesis_recaps (session_ids, recap_json, model, created_at) VALUES (?, ?, ?, ?)",
        (json.dumps(session_ids), json.dumps(recap), model, time.time()),
    )
    await _db.commit()
    return cursor.lastrowid


async def list_synthesis_recaps() -> list[dict]:
    """List all cross-session synthesis recaps."""
    cursor = await _db.execute(
        "SELECT id, session_ids, recap_json, model, created_at FROM synthesis_recaps ORDER BY created_at DESC"
    )
    rows = await cursor.fetchall()
    return [{
        "id": r["id"],
        "session_ids": json.loads(r["session_ids"]),
        "recap": json.loads(r["recap_json"]),
        "model": r["model"],
        "created_at": r["created_at"],
    } for r in rows]


async def get_synthesis_recap(synthesis_id: int) -> dict | None:
    """Get a single cross-session synthesis recap by ID."""
    cursor = await _db.execute(
        "SELECT id, session_ids, recap_json, model, created_at FROM synthesis_recaps WHERE id = ?",
        (synthesis_id,),
    )
    row = await cursor.fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "session_ids": json.loads(row["session_ids"]),
        "recap": json.loads(row["recap_json"]),
        "model": row["model"],
        "created_at": row["created_at"],
    }


async def delete_synthesis_recap(synthesis_id: int):
    """Delete a cross-session synthesis recap."""
    await _db.execute("DELETE FROM synthesis_recaps WHERE id = ?", (synthesis_id,))
    await _db.commit()
