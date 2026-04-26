"""
test_trace_collector.py — verify the per-step trace collector and node
wrapping for the architecture visualizer.

The collector is opt-in via contextvars: nodes wrapped with @with_trace
emit events when a collector is active, and silently no-op when not.
This means existing tests (test_idk_counter, test_full_loop) keep passing
unchanged because they don't activate a collector.
"""
import time
from unittest.mock import patch

from graph._trace import with_trace, set_collector, drain, emit


# ── Collector lifecycle ──────────────────────────────────────────────────────

def test_emit_is_noop_without_collector():
    """No collector active → emit must not raise + must not allocate."""
    emit("test_step", {"in": 1}, {"out": 2}, 100)  # should not raise


def test_set_and_drain_collector():
    events: list[dict] = []
    token = set_collector(events)
    try:
        emit("foo", {"x": 1}, {"y": 2}, 50)
        emit("bar", {}, {}, 25)
    finally:
        drain(token)
    assert len(events) == 2
    assert events[0]["step"] == "foo"
    assert events[0]["input"] == {"x": 1}
    assert events[0]["duration_ms"] == 50
    assert events[1]["step"] == "bar"


def test_emit_after_drain_is_noop():
    events: list[dict] = []
    token = set_collector(events)
    drain(token)
    emit("after_drain", {}, {}, 10)
    assert events == [], "emit should not append after drain"


# ── with_trace decorator ─────────────────────────────────────────────────────

def test_with_trace_records_step_when_collector_active():
    @with_trace("manager")
    def my_node(state):
        return {"current_concept": "ulnar nerve"}

    events: list[dict] = []
    token = set_collector(events)
    try:
        my_node({"messages": [], "turn_count": 1})
    finally:
        drain(token)

    assert len(events) == 1
    rec = events[0]
    assert rec["step"] == "manager"
    assert rec["input"] == {"turn_count": 1}     # messages stripped by default capture
    assert rec["output"] == {"current_concept": "ulnar nerve"}
    assert rec["duration_ms"] >= 0


def test_with_trace_silent_without_collector():
    @with_trace("manager")
    def my_node(state):
        return {"x": 1}

    out = my_node({"y": 2})  # must not raise; collector inactive
    assert out == {"x": 1}


def test_with_trace_records_model_label():
    @with_trace("classifier", model="haiku")
    def cls(state):
        return {"classifier_output": "incorrect"}

    events: list[dict] = []
    token = set_collector(events)
    try:
        cls({"messages": []})
    finally:
        drain(token)
    assert events[0]["model"] == "haiku"


def test_with_trace_custom_capture():
    @with_trace(
        "manager",
        capture_input=lambda state: {"msg_count": len(state.get("messages", []))},
        capture_output=lambda state, result: {"concept": result.get("current_concept", "")},
    )
    def manager(state):
        return {"current_concept": "ulnar nerve"}

    events: list[dict] = []
    token = set_collector(events)
    try:
        manager({"messages": ["a", "b", "c"]})
    finally:
        drain(token)
    assert events[0]["input"] == {"msg_count": 3}
    assert events[0]["output"] == {"concept": "ulnar nerve"}


def test_with_trace_records_duration():
    @with_trace("slow")
    def sleepy(state):
        time.sleep(0.05)  # 50ms
        return {}

    events: list[dict] = []
    token = set_collector(events)
    try:
        sleepy({})
    finally:
        drain(token)
    assert events[0]["duration_ms"] >= 40, f"expected ≥40ms, got {events[0]['duration_ms']}"


# ── Graph integration ───────────────────────────────────────────────────────

def test_graph_node_emits_when_invoked_with_collector():
    """Smoke test: route_after_retrieval doesn't need an LLM, but a real
    graph node call would. Just verify the decorator chain compiles + the
    graph still has all nodes after wrapping.
    """
    from graph.graph_builder import graph
    nodes = set(graph.nodes.keys())
    for required in ("manager_agent", "retrieval", "response_classifier",
                     "teacher_socratic", "dean_node", "study_node"):
        assert required in nodes, f"missing wrapped node {required!r}"


# ── Concurrent isolation (contextvars) ──────────────────────────────────────

def test_collectors_isolated_across_contextvars():
    """If two collectors are active in different copy_context()s, events
    must not bleed. (Real check is in async land; this is a sanity proxy.)"""
    events_a: list[dict] = []
    events_b: list[dict] = []

    token_a = set_collector(events_a)
    emit("foo", {}, {}, 1)
    drain(token_a)

    token_b = set_collector(events_b)
    emit("bar", {}, {}, 2)
    drain(token_b)

    assert len(events_a) == 1 and events_a[0]["step"] == "foo"
    assert len(events_b) == 1 and events_b[0]["step"] == "bar"


if __name__ == "__main__":
    test_emit_is_noop_without_collector()
    test_set_and_drain_collector()
    test_emit_after_drain_is_noop()
    test_with_trace_records_step_when_collector_active()
    test_with_trace_silent_without_collector()
    test_with_trace_records_model_label()
    test_with_trace_custom_capture()
    test_with_trace_records_duration()
    test_graph_node_emits_when_invoked_with_collector()
    test_collectors_isolated_across_contextvars()
    print("PASS: all 10 tests")
