"""
test_stream_chat.py — unit tests for the per-request streaming sink and
teacher_socratic's streaming helper.

These tests deliberately avoid spinning up the full graph (which would
require mocking Haiku for the manager/classifier and Sonnet for Dean).
We verify the building blocks instead:

  1. graph._stream — sink lifecycle, emit_* push the right SSE shapes,
     no-ops when no sink is installed.
  2. teacher_socratic._stream_completion — drains a fake Anthropic
     stream, suppresses <thinking>...</thinking> from live emission,
     and returns the full raw text.

Run from the repo root with:
    PYTHONPATH=backend python3 -m pytest backend/test_stream_chat.py -v
or as a script:
    PYTHONPATH=backend python3 backend/test_stream_chat.py
"""
from __future__ import annotations

import asyncio
import sys
import types
from typing import Iterator
from unittest.mock import patch


# ── _stream sink primitives ──────────────────────────────────────────────────

def _drain(q: asyncio.Queue) -> list[dict]:
    out: list[dict] = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_emit_token_no_sink_is_noop():
    """Calls without a sink installed must not raise. This guarantees nodes
    can call emit_token unconditionally without checking has_sink first."""
    from graph import _stream

    # No sink installed → silent no-op.
    _stream.emit_token("hello")
    _stream.emit_step("generation", "start")
    _stream.emit_replace("done")
    assert _stream.has_sink() is False


def test_sink_lifecycle_and_event_shapes():
    """set_sink installs, emit_* pushes correctly-shaped frames, drain
    removes the sink so subsequent emits become no-ops again."""
    from graph import _stream

    async def go() -> list[dict]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        token = _stream.set_sink(q, loop)
        try:
            assert _stream.has_sink() is True
            _stream.emit_step("generation", "start")
            _stream.emit_token("Hello ")
            _stream.emit_token("world")
            _stream.emit_step("generation", "done")
            _stream.emit_replace("Hello world!")
            # Yield once so the call_soon_threadsafe-scheduled put_nowait
            # callbacks all run before we drain. emit_* is sync but it
            # schedules onto the loop via call_soon_threadsafe.
            await asyncio.sleep(0)
        finally:
            _stream.drain(token)

        assert _stream.has_sink() is False
        # After drain emits go nowhere.
        _stream.emit_token("ignored")
        await asyncio.sleep(0)
        return _drain(q)

    events = asyncio.run(go())
    assert events == [
        {"event": "step",    "step": "generation", "status": "start"},
        {"event": "token",   "delta": "Hello "},
        {"event": "token",   "delta": "world"},
        {"event": "step",    "step": "generation", "status": "done"},
        {"event": "replace", "response": "Hello world!"},
    ], events


def test_emit_token_empty_delta_is_skipped():
    """The Anthropic stream occasionally yields empty deltas at start/end.
    We must not pollute the SSE stream with empty token frames."""
    from graph import _stream

    async def go() -> list[dict]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        token = _stream.set_sink(q, loop)
        try:
            _stream.emit_token("")
            _stream.emit_token("real")
            _stream.emit_token("")
            await asyncio.sleep(0)
        finally:
            _stream.drain(token)
        return _drain(q)

    events = asyncio.run(go())
    assert events == [{"event": "token", "delta": "real"}], events


# ── teacher_socratic._stream_completion ───────────────────────────────────────

class _FakeStream:
    """Minimal stand-in for `client.messages.stream(...)` return value.
    Behaves as a context manager and exposes a `text_stream` iterable."""

    def __init__(self, deltas: list[str]):
        self._deltas = deltas

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    @property
    def text_stream(self) -> Iterator[str]:
        return iter(self._deltas)


def _patch_stream(deltas: list[str]):
    """Patch teacher_socratic._client.messages.stream to return _FakeStream."""
    from graph.nodes import teacher_socratic as ts
    fake_messages = types.SimpleNamespace(
        stream=lambda **kwargs: _FakeStream(deltas)
    )
    return patch.object(ts._client, "messages", fake_messages)


def test_stream_completion_suppresses_thinking_then_emits_visible():
    """The teacher prompt asks for <thinking>...</thinking>visible_text.
    The streaming helper must NOT emit any token deltas until the closing
    </thinking> tag appears, then emit only what comes after it."""
    from graph import _stream
    from graph.nodes.teacher_socratic import _stream_completion

    deltas = [
        "<thinking>",
        "private chain of thought ",
        "more reasoning ",
        "</thinking>",
        "Visible reply ",
        "to the student.",
    ]

    async def go():
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        token = _stream.set_sink(q, loop)
        try:
            with _patch_stream(deltas):
                # Run the (sync) helper on a thread, like the API does.
                raw = await asyncio.to_thread(
                    _stream_completion,
                    {"model": "stub", "messages": []},
                    True,
                )
            await asyncio.sleep(0)
        finally:
            _stream.drain(token)
        return raw, _drain(q)

    raw, events = asyncio.run(go())

    # Full raw text is unchanged — preserves the <thinking> block for
    # log_thinking and strip_thinking_block downstream.
    assert raw == "".join(deltas)

    token_events = [e for e in events if e.get("event") == "token"]
    assert len(token_events) >= 1, f"expected ≥1 token event, got {events}"

    # Concatenated emitted tokens are exactly the post-</thinking> tail,
    # left-stripped (the helper does .lstrip() once).
    emitted = "".join(e["delta"] for e in token_events)
    assert emitted == "Visible reply to the student.", emitted

    # No fragment of the thinking block leaked.
    for e in token_events:
        assert "<thinking>" not in e["delta"]
        assert "</thinking>" not in e["delta"]
        assert "private chain" not in e["delta"]


def test_stream_completion_emit_tokens_false_runs_silently():
    """When emit_tokens=False (length retry path) the helper must drain
    the stream into raw text but NOT push any frames to the sink."""
    from graph import _stream
    from graph.nodes.teacher_socratic import _stream_completion

    deltas = ["<thinking>x</thinking>", "Hello ", "world."]

    async def go():
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        token = _stream.set_sink(q, loop)
        try:
            with _patch_stream(deltas):
                raw = await asyncio.to_thread(
                    _stream_completion,
                    {"model": "stub", "messages": []},
                    False,
                )
            await asyncio.sleep(0)
        finally:
            _stream.drain(token)
        return raw, _drain(q)

    raw, events = asyncio.run(go())
    assert raw == "".join(deltas)
    assert events == [], f"emit_tokens=False should be silent, got {events}"


def test_stream_completion_handles_split_close_tag():
    """The </thinking> tag may straddle two stream chunks. The helper must
    detect it once both chunks land in the buffer."""
    from graph import _stream
    from graph.nodes.teacher_socratic import _stream_completion

    deltas = ["<thinking>reason</thi", "nking>visible reply"]

    async def go():
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        token = _stream.set_sink(q, loop)
        try:
            with _patch_stream(deltas):
                await asyncio.to_thread(
                    _stream_completion,
                    {"model": "stub", "messages": []},
                    True,
                )
            await asyncio.sleep(0)
        finally:
            _stream.drain(token)
        return _drain(q)

    events = asyncio.run(go())
    token_events = [e for e in events if e.get("event") == "token"]
    emitted = "".join(e["delta"] for e in token_events)
    assert emitted == "visible reply", emitted


# ── Manual runner so the file is also executable directly ───────────────────

if __name__ == "__main__":
    tests = [
        test_emit_token_no_sink_is_noop,
        test_sink_lifecycle_and_event_shapes,
        test_emit_token_empty_delta_is_skipped,
        test_stream_completion_suppresses_thinking_then_emits_visible,
        test_stream_completion_emit_tokens_false_runs_silently,
        test_stream_completion_handles_split_close_tag,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"ERROR {t.__name__}: {exc!r}")
    if failed:
        sys.exit(1)
