"""
backend/api/user_weak_topics.py — per-user persistent weak-topics store.

The brief asks the system to "proactively revisit" topics the student
struggled with. SqliteSaver's GraphState.weak_topics handles this
WITHIN a session — but every "New chat" mints a fresh session_id and
starts that list at []. This module fixes that by promoting weak
topics to a per-user store: a topic stays open until the student
answers it correctly, regardless of how many sessions they spread
across.

Lives in the same SQLite file as session_meta + SqliteSaver
checkpoints so a single backup/volume covers all persistence. Schema
is intentionally narrow — just enough to drive the sidebar + the
rapport/teach prompts.

  user_weak_topics(user_id, concept, ...)

When a topic is added: upsert (miss_count++, last_seen=now).
When a topic is solved: row is deleted.
When the sidebar renders or rapport runs: list_for_user(user_id).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

import config


SCHEMA = """
CREATE TABLE IF NOT EXISTS user_weak_topics (
    user_id      TEXT NOT NULL,
    concept      TEXT NOT NULL,
    miss_count   INTEGER NOT NULL DEFAULT 1,
    mastery_level TEXT,         -- 'failed' | 'needs_review' | NULL
    first_seen   TEXT NOT NULL,
    last_seen    TEXT NOT NULL,
    PRIMARY KEY (user_id, concept)
);
CREATE INDEX IF NOT EXISTS idx_user_weak_topics_user
    ON user_weak_topics(user_id, last_seen DESC);
"""


def init_schema() -> None:
    """Idempotent — safe to call on every backend boot."""
    with _connect() as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.SESSIONS_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_weak(user_id: str, concept: str,
             mastery_level: str | None = "failed") -> None:
    """Mark `concept` as weak for `user_id`. On repeat misses,
    miss_count increments and last_seen advances. New rows pin
    first_seen.

    Idempotent — fine to call multiple times; counts the misses.
    Silent no-op when user_id or concept is empty (don't pollute the
    table with garbage rows from anonymous traffic).
    """
    user_id = (user_id or "").strip()
    concept = (concept or "").strip()
    if not user_id or not concept:
        return
    now = _now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_weak_topics
                (user_id, concept, miss_count, mastery_level,
                 first_seen, last_seen)
            VALUES (?, ?, 1, ?, ?, ?)
            ON CONFLICT(user_id, concept) DO UPDATE SET
                miss_count    = miss_count + 1,
                mastery_level = COALESCE(excluded.mastery_level, mastery_level),
                last_seen     = excluded.last_seen
            """,
            (user_id, concept, mastery_level, now, now),
        )
        conn.commit()


def remove_weak(user_id: str, concept: str) -> None:
    """Drop `concept` from `user_id`'s weak list — student finally
    answered it correctly. No-op when row doesn't exist. The row goes
    away entirely so list_for_user stays small."""
    user_id = (user_id or "").strip()
    concept = (concept or "").strip()
    if not user_id or not concept:
        return
    with _connect() as conn:
        conn.execute(
            "DELETE FROM user_weak_topics WHERE user_id = ? AND concept = ?",
            (user_id, concept),
        )
        conn.commit()


def list_for_user(user_id: str) -> list[str]:
    """Return concept names ordered by most-recently-missed first.
    Used by the sidebar (after merge with session-level state) and by
    api/main.py for hydrating fresh sessions."""
    user_id = (user_id or "").strip()
    if not user_id:
        return []
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT concept FROM user_weak_topics
            WHERE user_id = ?
            ORDER BY last_seen DESC
            """,
            (user_id,),
        ).fetchall()
        return [r["concept"] for r in rows]


def clear_user(user_id: str) -> None:
    """Drop ALL weak topics for a user. Currently unused; useful for a
    future "reset my progress" admin flow."""
    user_id = (user_id or "").strip()
    if not user_id:
        return
    with _connect() as conn:
        conn.execute(
            "DELETE FROM user_weak_topics WHERE user_id = ?", (user_id,),
        )
        conn.commit()
