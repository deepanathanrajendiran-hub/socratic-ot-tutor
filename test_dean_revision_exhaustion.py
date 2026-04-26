"""
test_dean_revision_exhaustion.py

When Dean rejects a draft DEAN_MAX_REVISIONS times, the graph routes to
fallback_scaffold (not back to the source node). The fallback must:
  1. Deliver an AIMessage so the student isn't left stranded.
  2. Reset student_phase to "learning" so stale phase state doesn't corrupt
     routing on the next turn.
  3. Reset concept_mastered + dean_revisions + dean_revision_instruction.

Without a working fallback, a student stuck in choice_pending who repeatedly
trips Dean's GROUNDING_CHECK would loop forever or get a malformed response.
"""
import config

from graph.edges import route_after_dean


def test_dean_at_max_revisions_routes_to_fallback():
    state = {
        "dean_passed":       False,
        "dean_revisions":    config.DEAN_MAX_REVISIONS,
        "draft_source_node": "teacher_socratic",
    }
    assert route_after_dean(state) == "fallback_scaffold"


def test_dean_above_max_revisions_routes_to_fallback():
    """Defensive: if dean_revisions exceeds the gate (e.g., off-by-one bug
    upstream), still go to fallback, not silently to deliver_response."""
    state = {
        "dean_passed":       False,
        "dean_revisions":    config.DEAN_MAX_REVISIONS + 5,
        "draft_source_node": "teacher_socratic",
    }
    assert route_after_dean(state) == "fallback_scaffold"


def test_dean_below_max_routes_to_source_node():
    state = {
        "dean_passed":       False,
        "dean_revisions":    0,
        "draft_source_node": "teacher_socratic",
    }
    assert route_after_dean(state) == "teacher_socratic"


def test_dean_passed_routes_to_deliver():
    state = {
        "dean_passed":       True,
        "dean_revisions":    1,
        "draft_source_node": "teacher_socratic",
    }
    assert route_after_dean(state) == "deliver_response"


def test_missing_draft_source_node_raises():
    """Generation node forgot to set draft_source_node — fail loudly."""
    state = {
        "dean_passed":    False,
        "dean_revisions": 0,
    }
    try:
        route_after_dean(state)
        assert False, "Expected ValueError for missing draft_source_node"
    except ValueError as e:
        assert "draft_source_node" in str(e)


def test_fallback_scaffold_resets_state_and_delivers():
    """The fallback callable wired into graph_builder must produce an
    AIMessage and reset phase + dean state."""
    from graph.graph_builder import _stub_fallback_scaffold
    out = _stub_fallback_scaffold({
        "student_phase":   "choice_pending",
        "concept_mastered": True,
        "dean_revisions": config.DEAN_MAX_REVISIONS,
        "dean_revision_instruction": "stale reason",
        "messages":       [],
        "current_concept": "ulnar nerve",
        "turn_count":     5,
    })
    assert "messages" in out and len(out["messages"]) >= 1, "no AIMessage delivered"
    msg = out["messages"][0]
    assert msg.type == "ai"
    assert len(msg.content) > 10, "fallback message empty"
    # State resets
    assert out.get("student_phase") == "learning"
    assert out.get("concept_mastered") is False
    assert out.get("dean_revisions") == 0
    assert out.get("dean_revision_instruction") == ""
    # Turn count must increment so the next routing isn't tripped by the same turn
    assert out.get("turn_count", 0) == 6


if __name__ == "__main__":
    test_dean_at_max_revisions_routes_to_fallback()
    test_dean_above_max_revisions_routes_to_fallback()
    test_dean_below_max_routes_to_source_node()
    test_dean_passed_routes_to_deliver()
    test_missing_draft_source_node_raises()
    test_fallback_scaffold_resets_state_and_delivers()
    print("PASS: all 6 tests")
