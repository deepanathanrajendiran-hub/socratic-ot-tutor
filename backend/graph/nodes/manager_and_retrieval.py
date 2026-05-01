"""
graph/nodes/manager_and_retrieval.py

Combo node that runs manager_agent and retrieval_node concurrently.

Why this exists:
  Pre-streaming-rewrite, the graph ran manager → retrieval sequentially,
  burning ~2.5s of TTFT each turn (~2.6s manager Haiku + ~2.4s retrieval
  CRAG eval + embed + rerank). The two are mostly independent — manager
  produces current_concept, retrieval needs a search query but works
  fine seeded with the raw student message (CRAG synonym expansion +
  cross-encoder rerank handle the precision drop).

Implementation:
  Both inner functions remain wrapped by with_trace so they fire their
  own step.start / step.done events (the architecture visualizer sees
  them as separate panels with overlapping spans). We use a stdlib
  ThreadPoolExecutor with contextvars.copy_context() so the per-request
  _stream sink and _trace collector propagate into the worker threads.

  After both finish we merge the results: manager wins on overlapping
  keys (manager's current_concept overrides the raw-message seed used
  for retrieval).
"""
from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from graph.state import GraphState
from graph.nodes._helpers import msg_text


def _last_student_message(state: GraphState) -> str:
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", None) == "human":
            return msg_text(getattr(m, "content", ""))
    return ""


def make_manager_and_retrieval(
    manager_traced: Callable[[GraphState], dict],
    retrieval_traced: Callable[[GraphState], dict],
) -> Callable[[GraphState], dict]:
    """Build the combo node by closing over the trace-wrapped versions
    of manager_agent and retrieval_node from graph_builder.

    We accept already-wrapped callables (rather than wrapping here) so
    the architecture visualizer keeps seeing the same step names —
    "concept_extraction" and "retrieval" — and the trace decorator's
    timing/io snapshots are unchanged.

    ContextVars propagation: ThreadPoolExecutor workers do NOT inherit
    the calling thread's ContextVar values by default. We snapshot a
    fresh context PER CHILD in the parent thread (where _stream.set_sink
    and _trace.set_collector were installed by the API handler), then
    invoke each child via that context's `.run()`. Snapshotting in the
    parent is critical — calling `copy_context()` inside the worker
    captures the worker's empty context, which is what we hit on the
    first attempt and broke step-event propagation.
    """

    def combo(state: GraphState) -> dict:
        # Seed retrieval with the raw student message when no concept is
        # locked yet. retrieval_node falls back to state["current_concept"]
        # both for the CRAG query and inside _build_turn_query, so we
        # patch a temporary copy of state — not the caller's state —
        # with the raw message acting as the concept stand-in.
        retrieval_state: GraphState = dict(state)  # type: ignore[assignment]
        if not retrieval_state.get("current_concept"):
            seed = _last_student_message(state)
            if seed:
                retrieval_state["current_concept"] = seed

        # Snapshot the parent context BEFORE submitting — one copy per
        # child so the two threads never share a Context (ContextVar
        # mutations would otherwise race).
        ctx_manager = contextvars.copy_context()
        ctx_retrieval = contextvars.copy_context()

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_manager = ex.submit(ctx_manager.run, manager_traced, state)
            f_retrieval = ex.submit(ctx_retrieval.run, retrieval_traced, retrieval_state)
            retrieval_out: dict = f_retrieval.result()
            manager_out: dict = f_manager.result()

        # Merge: manager-side updates (the authoritative current_concept,
        # student_phase reset, dean_revisions reset) override the
        # retrieval seed. retrieval-side updates carry the chunks +
        # chunk_sources + crag_decision into the merged state.
        return {**retrieval_out, **manager_out}

    return combo
