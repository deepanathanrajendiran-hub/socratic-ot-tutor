"""
graph/_trace.py — opt-in per-step trace collector for the architecture
visualizer.

Per-request collection is tracked in a `contextvars.ContextVar` so the
collector is isolated even if FastAPI serves concurrent requests. Modules
that don't activate the collector pay zero cost — emit() is a no-op when
no list is set.

Usage from api/main.py /chat/trace:

    from graph._trace import set_collector, drain
    events: list[dict] = []
    token = set_collector(events)
    try:
        result = await asyncio.to_thread(graph.invoke, state, cfg)
    finally:
        drain(token)
    # events now contains per-step records in invocation order

Usage from a wrapped node (graph_builder uses the with_trace decorator
so node files don't change):

    @with_trace("concept_extraction", capture=lambda state: {
        "student_message": ...,
        "history_len":     len(state.get("messages", [])),
    })
    def manager_agent(state):
        ...

Each emitted event has the shape spec'd in docs/website.md §5.4:
    {
        "step":        str,           # canonical step name
        "input":       dict,          # captured input snapshot
        "output":      dict,          # captured output snapshot
        "duration_ms": int,
        "model":       str | None,    # if known
    }
"""
from __future__ import annotations

import contextvars
import functools
import time
from typing import Any, Callable, Optional


_collector: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar(
    "trace_collector", default=None
)


def set_collector(events: list[dict]) -> contextvars.Token:
    """Install the per-request collector. Call drain() with the returned
    token to remove it."""
    return _collector.set(events)


def drain(token: contextvars.Token) -> None:
    """Remove the active collector. Always call in a finally block to
    avoid leaking a collector across requests."""
    _collector.reset(token)


def emit(step: str, input_: dict, output: dict, duration_ms: int,
         model: Optional[str] = None, **extra: Any) -> None:
    """Append a step record to the active collector. No-op if none active."""
    sink = _collector.get()
    if sink is None:
        return
    record = {
        "step":        step,
        "input":       input_,
        "output":      output,
        "duration_ms": duration_ms,
    }
    if model:
        record["model"] = model
    record.update(extra)
    sink.append(record)


def with_trace(step: str,
               capture_input: Callable[[dict], dict] | None = None,
               capture_output: Callable[[dict, dict], dict] | None = None,
               model: Optional[str] = None) -> Callable:
    """Wrap a node function so each invocation appends a trace event.

    capture_input(state)   → dict snapshot of relevant input fields
    capture_output(state, result) → dict snapshot of relevant output fields

    Defaults: input is shallow state keys minus large blobs; output is the
    full result dict (which is small for most nodes).
    """
    def _default_input(state: dict) -> dict:
        # Skip messages + retrieved_chunks (potentially large) by default
        return {k: v for k, v in state.items()
                if k not in ("messages", "retrieved_chunks")
                and isinstance(v, (str, int, float, bool, list, type(None)))}

    def _default_output(state: dict, result: dict) -> dict:
        # Same exclusions as input. `messages` is an add_messages reducer
        # value containing langchain BaseMessage instances which aren't
        # JSON-serializable — emitting them breaks the SSE writer.
        return {k: v for k, v in (result or {}).items()
                if k not in ("messages", "retrieved_chunks")
                and isinstance(v, (str, int, float, bool, list, type(None)))}

    cap_in  = capture_input  or _default_input
    cap_out = capture_output or _default_output

    # Lazy import — avoid a hard dep cycle if _stream evolves.
    from graph import _stream

    def decorator(node_fn: Callable) -> Callable:
        @functools.wraps(node_fn)
        def wrapper(state: dict) -> dict:
            input_snapshot = cap_in(state)
            _stream.emit_step(step, "start")
            t0 = time.time()
            try:
                result = node_fn(state)
            except Exception:
                _stream.emit_step(step, "error")
                raise
            duration_ms = int((time.time() - t0) * 1000)
            _stream.emit_step(step, "done")
            try:
                output_snapshot = cap_out(state, result)
            except Exception:
                output_snapshot = {}
            emit(step, input_snapshot, output_snapshot,
                 duration_ms, model=model)
            return result
        return wrapper
    return decorator
