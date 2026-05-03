"""
backend/api/sessions_meta.py — per-session metadata (title / pinned).

LangGraph's SqliteSaver tables (`checkpoints`, `writes`, `checkpoint_blobs`)
are keyed by `thread_id` and own the conversation state. They have no slot
for human-friendly labels or pin flags, so we keep that metadata in a
sibling table in the SAME SQLite file.

Why same file: the SqliteSaver schema and this metadata both live and die
together — backups, container volumes, and Cloud Run mounts handle one
file instead of two, and a `DELETE FROM checkpoints WHERE thread_id=?`
can be wrapped with a parallel `DELETE FROM session_meta WHERE thread_id=?`
in one transaction.

Concurrency: WAL mode (set by SqliteSaver on first connect) lets multiple
connections write to the same DB. Each helper here opens a short-lived
connection and closes it — no long-lived state.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

import config


SCHEMA = """
CREATE TABLE IF NOT EXISTS session_meta (
    thread_id   TEXT PRIMARY KEY,
    title       TEXT,
    pinned      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    last_active TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_meta_pinned_active
    ON session_meta(pinned DESC, last_active DESC);
"""


def init_schema() -> None:
    """Create the session_meta table if it doesn't exist. Idempotent."""
    with _connect() as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Short-lived connection. Always commits-or-closes."""
    # Ensure the parent directory exists. SQLite errors with
    # "unable to open database file" if the dir is missing, which
    # bites when SESSIONS_DB_PATH points at a path Cloud Render's
    # persistent disk mounts (e.g. /var/data/sessions.db) and the
    # mount root exists but no file has been written yet — and
    # also in fresh local checkouts where backend/data/ wasn't
    # populated. Mirrors graph_builder._build_checkpointer.
    import os
    parent = os.path.dirname(config.SESSIONS_DB_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(config.SESSIONS_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert(thread_id: str, *, title: str | None = None) -> None:
    """Insert a new session_meta row, or touch last_active if it exists."""
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO session_meta (thread_id, title, pinned,
                                       created_at, last_active)
            VALUES (?, ?, 0, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                last_active = excluded.last_active,
                title = COALESCE(session_meta.title, excluded.title)
            """,
            (thread_id, title, now, now),
        )
        conn.commit()


def touch(thread_id: str) -> None:
    """Update last_active to now. Used after each /chat turn so recency
    ordering reflects actual usage (not just creation time)."""
    with _connect() as conn:
        conn.execute(
            "UPDATE session_meta SET last_active = ? WHERE thread_id = ?",
            (_now(), thread_id),
        )
        conn.commit()


def update(thread_id: str, *, title: str | None = None,
           pinned: bool | None = None) -> dict | None:
    """Patch title and/or pinned. Returns the updated row or None if missing."""
    sets: list[str] = []
    args: list = []
    if title is not None:
        sets.append("title = ?")
        args.append(title.strip()[:120] or None)
    if pinned is not None:
        sets.append("pinned = ?")
        args.append(1 if pinned else 0)
    if not sets:
        return get(thread_id)
    args.append(thread_id)
    with _connect() as conn:
        cur = conn.execute(
            f"UPDATE session_meta SET {', '.join(sets)} WHERE thread_id = ?",
            args,
        )
        conn.commit()
        if cur.rowcount == 0:
            return None
    return get(thread_id)


def delete(thread_id: str) -> None:
    """Hard-delete: remove the meta row AND all SqliteSaver checkpoint
    rows for this thread. Wrapped in a single transaction so the two
    can't drift out of sync. Idempotent — missing rows are fine."""
    with _connect() as conn:
        try:
            conn.execute("BEGIN")
            for table in ("checkpoints", "writes", "checkpoint_blobs"):
                try:
                    conn.execute(
                        f"DELETE FROM {table} WHERE thread_id = ?",
                        (thread_id,),
                    )
                except sqlite3.OperationalError:
                    # Table doesn't exist yet (graph never ran for this id)
                    pass
            conn.execute(
                "DELETE FROM session_meta WHERE thread_id = ?",
                (thread_id,),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def get(thread_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM session_meta WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None


def list_all() -> list[dict]:
    """All sessions, pinned first, then most-recently-active first."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT thread_id, title, pinned, created_at, last_active
            FROM session_meta
            ORDER BY pinned DESC, last_active DESC
            """,
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id":          row["thread_id"],
        "title":       row["title"],
        "pinned":      bool(row["pinned"]),
        "created_at":  row["created_at"],
        "last_active": row["last_active"],
    }
