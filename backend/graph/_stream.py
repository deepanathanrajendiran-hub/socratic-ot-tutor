"""
graph/_stream.py — opt-in per-request live SSE sink for /chat.

Mirrors graph/_trace.py but produces realtime events (token deltas,
step start/done, replace) rather than a post-hoc record. The graph
runs on a worker thread (asyncio.to_thread(graph.invoke, ...)); the
FastAPI handler drains an asyncio.Queue from the event loop. Producers
inside the graph push onto that queue via loop.call_soon_threadsafe.

Usage from api/main.py /chat:

    from graph._stream import set_sink, drain
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    token = set_sink(q, loop)
    try:
        result = await asyncio.to_thread(graph.invoke, state, cfg)
    finally:
        drain(token)

Producers inside nodes call emit_token / emit_step / emit_replace; if
no sink is active the calls are no-ops, so nodes don't need to know
whether they're being streamed.

Event shapes match the SSE frames the frontend consumes:
    {"event": "step",    "step": "generation", "status": "start"}
    {"event": "step",    "step": "generation", "status": "done"}
    {"event": "token",   "delta": "Before "}
    {"event": "replace", "response": "Final cleaned text after revision"}
"""
from __future__ import annotations

import asyncio
import contextvars
from typing import Optional


class _Sink:
    """Holds the asyncio.Queue + loop reference together so contextvars
    only stores one object."""

    __slots__ = ("queue", "loop")

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.queue = queue
        self.loop = loop

    def push(self, event: dict) -> None:
        # Producers run on a worker thread (asyncio.to_thread). Use
        # call_soon_threadsafe to schedule the put_nowait on the FastAPI
        # event loop. Failures are swallowed — losing a token frame is
        # better than crashing the graph mid-turn.
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, event)
        except RuntimeError:
            # Event loop closed — happens if the client disconnected and
            # FastAPI tore down the consumer. Drop the event silently.
            pass


_sink: contextvars.ContextVar[Optional[_Sink]] = contextvars.ContextVar(
    "stream_sink", default=None
)


def set_sink(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> contextvars.Token:
    """Install the per-request sink. Call drain() with the returned token."""
    return _sink.set(_Sink(queue, loop))


def drain(token: contextvars.Token) -> None:
    """Remove the active sink. Always call in a finally block."""
    _sink.reset(token)


def emit_token(delta: str) -> None:
    """Push one token delta to the SSE consumer. No-op if no sink active."""
    if not delta:
        return
    s = _sink.get()
    if s is None:
        return
    s.push({"event": "token", "delta": delta})


def emit_step(step: str, status: str) -> None:
    """Push a step lifecycle event. status ∈ {"start", "done", "error"}."""
    s = _sink.get()
    if s is None:
        return
    s.push({"event": "step", "step": step, "status": status})


def emit_replace(text: str) -> None:
    """Tell the consumer to overwrite the current bubble with `text`.

    Used when a node retries (e.g. teacher length-retry produces a fresh
    draft) or when post-stream cleanup mutates the streamed text.
    """
    s = _sink.get()
    if s is None:
        return
    s.push({"event": "replace", "response": text})


def emit_usage(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_create_tokens: int = 0,
) -> None:
    """Push an Anthropic usage record from a single LLM call. No-op if
    no sink is active. Producers (the wrapped Anthropic client in
    _llm_client.py) call this after every messages.create() / stream()
    so the trace endpoint can sum exact token spend per turn — used
    by evaluation/token_budget.py --live for billable estimates."""
    s = _sink.get()
    if s is None:
        return
    s.push({
        "event":               "usage",
        "model":               model,
        "input_tokens":        int(input_tokens or 0),
        "output_tokens":       int(output_tokens or 0),
        "cache_read_tokens":   int(cache_read_tokens or 0),
        "cache_create_tokens": int(cache_create_tokens or 0),
    })


def has_sink() -> bool:
    """True if a sink is currently installed. Lets nodes branch between
    streaming and non-streaming code paths without raising."""
    return _sink.get() is not None
