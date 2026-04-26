"""
test_study_node.py — pin the study_node JSON envelope contract + the
implicit weak-topic counter.
"""
from unittest.mock import patch
from langchain_core.messages import HumanMessage

from graph.nodes import study_node


def _state(question: str, prior_topic: str = "", count: int = 0,
           weak: list | None = None) -> dict:
    return {
        "messages":           [HumanMessage(content=question)],
        "current_concept":    "ulnar nerve",
        "retrieved_chunks":   ["The ulnar nerve passes posterior to the medial epicondyle."],
        "domain":             "OT_anatomy",
        "study_active_topic": prior_topic,
        "study_topic_count":  count,
        "weak_topics":        weak or [],
        "mode":               "study",
        "turn_count":         0,
    }


def _mock(envelope_json: str):
    return patch.object(study_node, "_call_llm", return_value=envelope_json)


def test_study_node_emits_direct_answer_as_AIMessage():
    payload = ('{"answer": "The ulnar nerve causes the funny-bone sensation.", '
               '"active_topic": "ulnar nerve", "is_continuation": false, '
               '"citations": ["c1"]}')
    with _mock(payload):
        out = study_node.study_node(_state("What is the funny bone nerve?"))
    assert "messages" in out
    assert len(out["messages"]) == 1
    msg = out["messages"][0]
    assert msg.type == "ai"
    assert "ulnar nerve" in msg.content.lower()


def test_study_node_increments_continuation_counter():
    payload = ('{"answer": "It innervates intrinsic hand muscles.", '
               '"active_topic": "ulnar nerve", "is_continuation": true, '
               '"citations": []}')
    with _mock(payload):
        out = study_node.study_node(_state("What does it innervate?",
                                           prior_topic="ulnar nerve",
                                           count=2))
    assert out["study_topic_count"] == 3
    assert out["study_active_topic"] == "ulnar nerve"


def test_study_node_resets_counter_on_topic_change():
    payload = ('{"answer": "The median nerve runs through the carpal tunnel.", '
               '"active_topic": "median nerve", "is_continuation": false, '
               '"citations": []}')
    with _mock(payload):
        out = study_node.study_node(_state("What about the median nerve?",
                                           prior_topic="ulnar nerve",
                                           count=4))
    assert out["study_topic_count"] == 1
    assert out["study_active_topic"] == "median nerve"


def test_study_node_promotes_to_weak_topic_at_threshold():
    """Five consecutive same-topic Qs → append to weak_topics (idempotent)."""
    payload = ('{"answer": "More on ulnar.", "active_topic": "ulnar nerve", '
               '"is_continuation": true, "citations": []}')
    with _mock(payload):
        # prior count = 4 → after this turn count = 5 → threshold hit
        out = study_node.study_node(_state("More?",
                                           prior_topic="ulnar nerve", count=4))
    assert out["study_topic_count"] == 5
    assert "ulnar nerve" in out["weak_topics"]


def test_study_node_does_not_double_add_existing_weak_topic():
    payload = ('{"answer": "...", "active_topic": "ulnar nerve", '
               '"is_continuation": true, "citations": []}')
    with _mock(payload):
        out = study_node.study_node(_state("More?", prior_topic="ulnar nerve",
                                           count=4, weak=["ulnar nerve"]))
    # Should appear once, not twice
    assert out["weak_topics"].count("ulnar nerve") == 1


def test_study_node_recomputes_continuation_in_python():
    """If LLM lies about is_continuation but active_topic differs from prior,
    Python wins — counter does NOT continue."""
    payload = ('{"answer": "...", "active_topic": "median nerve", '
               '"is_continuation": true, "citations": []}')
    with _mock(payload):
        out = study_node.study_node(_state("Different topic?",
                                           prior_topic="ulnar nerve",
                                           count=4))
    # Topic changed → counter resets to 1, NOT 5
    assert out["study_topic_count"] == 1
    assert "ulnar nerve" not in out["weak_topics"]


def test_study_node_increments_turn_count():
    payload = ('{"answer": "test", "active_topic": "x", "is_continuation": '
               'false, "citations": []}')
    state = _state("Q?")
    state["turn_count"] = 3
    with _mock(payload):
        out = study_node.study_node(state)
    assert out["turn_count"] == 4


def test_study_node_handles_malformed_json():
    """Parse failures must not crash; return raw text as answer."""
    with _mock("not valid json at all"):
        out = study_node.study_node(_state("Q?"))
    assert out["messages"][0].type == "ai"
    # Counter still advances cleanly
    assert "study_topic_count" in out


if __name__ == "__main__":
    test_study_node_emits_direct_answer_as_AIMessage()
    test_study_node_increments_continuation_counter()
    test_study_node_resets_counter_on_topic_change()
    test_study_node_promotes_to_weak_topic_at_threshold()
    test_study_node_does_not_double_add_existing_weak_topic()
    test_study_node_recomputes_continuation_in_python()
    test_study_node_increments_turn_count()
    test_study_node_handles_malformed_json()
    print("PASS: all 8 tests")
