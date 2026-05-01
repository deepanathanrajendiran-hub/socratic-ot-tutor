"""
backend/memory/mem0_client.py

Optional cross-session memory layer (mem0.ai).

Usage:
  from backend.memory.mem0_client import client
  client.add(messages, user_id="..."[, metadata=...])
  hits = client.search(query, user_id="...")

Behavior:
  - If MEMORY_BACKEND != "mem0", every method is a no-op (returns empty
    result, doesn't raise). Lets call sites use the client unconditionally.
  - If the `mem0ai` package isn't installed OR MEM0_API_KEY isn't set,
    we log a one-time warning and degrade to no-op. Never crashes the
    chat flow on a memory failure.
  - Network errors during add/search are swallowed and logged. Cross-
    session memory is best-effort, not load-bearing.

Why not replace SqliteSaver:
  LangGraph's SqliteSaver owns *conversation state* — turn_count, weak
  topics, classifier outputs, message history. That's a graph-internal
  concern. mem0 owns *user-level facts* across sessions — what topics
  the student keeps struggling with, learning preferences, prior topic
  exposure. Different shapes, different lifetimes. They live alongside.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import config

logger = logging.getLogger(__name__)

# ── Try to import mem0 lazily. If it's missing, fall back to no-op. ─────────
try:
    from mem0 import MemoryClient as _MemoryClient  # type: ignore[import-not-found]
    _MEM0_AVAILABLE = True
except ImportError:
    _MemoryClient = None  # type: ignore[assignment]
    _MEM0_AVAILABLE = False


_warned = False


def _warn_once(msg: str) -> None:
    global _warned
    if not _warned:
        logger.warning("[mem0] %s — running with no-op memory layer", msg)
        _warned = True


class _NoopClient:
    """Returned when mem0 isn't enabled, isn't installed, or has no API key.
    Every method matches the real client's signature so call sites don't
    need to branch on backend."""

    enabled = False

    def add(self, messages: Any, user_id: str | None = None,
            metadata: dict | None = None, **_: Any) -> dict:
        return {"results": [], "_skipped": "memory backend disabled"}

    def search(self, query: str, user_id: str | None = None,
               limit: int = 5, **_: Any) -> list[dict]:
        return []

    def get_all(self, user_id: str | None = None, **_: Any) -> list[dict]:
        return []

    def delete_all(self, user_id: str | None = None, **_: Any) -> dict:
        return {"_skipped": "memory backend disabled"}


class _Mem0Cloud:
    """Thin wrapper around the mem0 cloud client. Catches network errors
    so a flaky memory service can't break the chat path."""

    enabled = True

    def __init__(self, api_key: str):
        if _MemoryClient is None:                            # pragma: no cover
            raise RuntimeError("mem0ai package not importable")
        self._client = _MemoryClient(api_key=api_key)
        # Mutex around add/search — mem0 cloud client is fine concurrent,
        # but we serialize fire-and-forget adds so a slow API call doesn't
        # interleave with a fast one in confusing ways during demos.
        self._lock = threading.Lock()

    def add(self, messages: Any, user_id: str | None = None,
            metadata: dict | None = None, **kwargs: Any) -> dict:
        if not user_id:
            return {"results": [], "_skipped": "no user_id"}
        try:
            with self._lock:
                # mem0 cloud accepts a list of {role, content} dicts and
                # extracts memories via its own LLM. metadata is attached
                # to every extracted memory.
                return self._client.add(
                    messages,
                    user_id=user_id,
                    metadata=metadata or {},
                    **kwargs,
                )
        except Exception as exc:                             # pragma: no cover
            logger.warning("[mem0] add failed (non-fatal): %s", exc)
            return {"results": [], "_error": str(exc)[:200]}

    def search(self, query: str, user_id: str | None = None,
               limit: int = 5, **kwargs: Any) -> list[dict]:
        if not user_id or not query:
            return []
        try:
            with self._lock:
                # mem0 cloud v2.x rejects top-level user_id on search/get_all
                # — must use `filters={"user_id": ...}`. Add forces the
                # opposite (top-level user_id), so we keep them split.
                results = self._client.search(
                    query, filters={"user_id": user_id}, limit=limit,
                    **kwargs,
                )
            # The cloud client returns either a list or {"results": [...]}
            # depending on version — normalize to a list.
            if isinstance(results, dict):
                return results.get("results", []) or []
            return results or []
        except Exception as exc:                             # pragma: no cover
            logger.warning("[mem0] search failed (non-fatal): %s", exc)
            return []

    def get_all(self, user_id: str | None = None, **kwargs: Any) -> list[dict]:
        if not user_id:
            return []
        try:
            with self._lock:
                results = self._client.get_all(
                    filters={"user_id": user_id}, **kwargs,
                )
            if isinstance(results, dict):
                return results.get("results", []) or []
            return results or []
        except Exception as exc:                             # pragma: no cover
            logger.warning("[mem0] get_all failed: %s", exc)
            return []

    def delete_all(self, user_id: str | None = None, **kwargs: Any) -> dict:
        if not user_id:
            return {"_skipped": "no user_id"}
        try:
            with self._lock:
                # mem0 cloud's API is inconsistent — search/get_all use
                # filters=, but delete_all wants top-level user_id (a
                # 400 fires if you pass filters here). Mirroring add().
                return self._client.delete_all(
                    user_id=user_id, **kwargs,
                ) or {}
        except Exception as exc:                             # pragma: no cover
            logger.warning("[mem0] delete_all failed: %s", exc)
            return {"_error": str(exc)[:200]}


def _build_client() -> _NoopClient | _Mem0Cloud:
    """Resolve the active client based on config.MEMORY_BACKEND. Falls
    back to _NoopClient with a warning rather than raising — the demo
    must not break because the optional memory layer is misconfigured."""
    if config.MEMORY_BACKEND != "mem0":
        return _NoopClient()
    if not _MEM0_AVAILABLE:
        _warn_once("MEMORY_BACKEND=mem0 but `pip install mem0ai` not found")
        return _NoopClient()
    if not config.MEM0_API_KEY:
        _warn_once("MEMORY_BACKEND=mem0 but MEM0_API_KEY is empty")
        return _NoopClient()
    try:
        return _Mem0Cloud(api_key=config.MEM0_API_KEY)
    except Exception as exc:                                 # pragma: no cover
        _warn_once(f"failed to build mem0 client: {exc}")
        return _NoopClient()


# Module-level singleton. Built once at import; safe to share across
# requests because we serialize add/search inside the wrapper.
client = _build_client()


def is_enabled() -> bool:
    return client.enabled


def memories_to_text(memories: list[dict]) -> str:
    """Format a list of mem0 memory dicts into a short bulleted string
    suitable for injecting into a prompt slot. Keeps each line short so
    the rapport node doesn't blow its token budget. Falls back to an
    empty string when there are no memories — the prompt should treat
    that as "no prior context"."""
    if not memories:
        return ""
    lines: list[str] = []
    for m in memories[: config.MEM0_TOP_K]:
        # Cloud results are typically {"memory": "...", "score": 0.x, ...}
        # OSS shape is similar. Be defensive about the field name.
        text = (
            m.get("memory") or m.get("text") or m.get("content") or ""
        ).strip()
        if not text:
            continue
        # Trim to one sentence so we don't dump prior conversation back.
        if len(text) > 160:
            text = text[:157] + "…"
        lines.append(f"- {text}")
    return "\n".join(lines)
