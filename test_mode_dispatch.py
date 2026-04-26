"""
test_mode_dispatch.py — Phase 5 mode-dispatch routing.

State.mode='study' must take the bypass path (retrieval → study_node → END);
state.mode='socratic' (default) takes the normal path through response_classifier.
"""
from graph.edges import route_after_input, route_after_retrieval


# ── route_after_input — mode does NOT change entry point ────────────────────

def test_socratic_mode_routes_to_manager():
    state = {"mode": "socratic", "student_phase": "learning"}
    assert route_after_input(state) == "manager_agent"


def test_study_mode_also_routes_to_manager():
    """Study mode still needs concept extraction + retrieval first."""
    state = {"mode": "study", "student_phase": "learning"}
    assert route_after_input(state) == "manager_agent"


def test_image_pending_overrides_mode():
    state = {"mode": "study", "student_phase": "learning", "image_pending": True}
    assert route_after_input(state) == "vlm_node"


def test_phase_gate_overrides_mode():
    """A mid-Socratic-flow choice_pending phase must not be derailed by mode."""
    state = {"mode": "study", "student_phase": "choice_pending"}
    assert route_after_input(state) == "mastery_choice_classifier"


# ── route_after_retrieval — the actual mode split ───────────────────────────

def test_retrieval_socratic_goes_to_classifier():
    assert route_after_retrieval({"mode": "socratic"}) == "response_classifier"


def test_retrieval_study_goes_to_study_node():
    assert route_after_retrieval({"mode": "study"}) == "study_node"


def test_retrieval_default_is_socratic():
    """Missing mode field → default to socratic for backward compat."""
    assert route_after_retrieval({}) == "response_classifier"


# ── Graph compiles with study_node + edge ───────────────────────────────────

def test_graph_includes_study_node():
    from graph.graph_builder import graph
    nodes = set(graph.nodes.keys())
    assert "study_node" in nodes, f"graph missing study_node; has {nodes}"


if __name__ == "__main__":
    test_socratic_mode_routes_to_manager()
    test_study_mode_also_routes_to_manager()
    test_image_pending_overrides_mode()
    test_phase_gate_overrides_mode()
    test_retrieval_socratic_goes_to_classifier()
    test_retrieval_study_goes_to_study_node()
    test_retrieval_default_is_socratic()
    test_graph_includes_study_node()
    print("PASS: all 8 tests")
