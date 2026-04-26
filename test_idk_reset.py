"""
test_idk_reset.py — pin the help-abuse jailbreak guard.

idk_count must reset to 0 when the student makes a real attempt (correct,
incorrect, questioning). Without this, a student who alternated "idk" and
"is it median?" would still trigger reveal at the threshold.

Tests use the rule-based idk pre-classifier path where we can (no LLM call
needed) and mock _client.messages.create for non-idk paths.
"""
from unittest.mock import patch, MagicMock

from langchain_core.messages import HumanMessage

from graph.nodes.response_classifier import response_classifier


def _state(message: str, idk_count: int = 0,
           student_attempted: bool = False) -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "idk_count":         idk_count,
        "current_concept":   "ulnar nerve",
        "turn_count":        1,
        "student_attempted": student_attempted,
    }


# ── Rule-based path (no LLM mock needed) ─────────────────────────────────────

def test_idk_message_increments_counter():
    out = response_classifier(_state("i don't know", idk_count=0))
    assert out["classifier_output"] == "idk", f"got {out['classifier_output']!r}"
    assert out["idk_count"] == 1


def test_consecutive_idks_keep_counting():
    out = response_classifier(_state("idk", idk_count=2))
    assert out["classifier_output"] == "idk"
    assert out["idk_count"] == 3


def test_give_up_phrase_is_idk():
    out = response_classifier(_state("just tell me the answer", idk_count=1))
    assert out["classifier_output"] == "idk"
    assert out["idk_count"] == 2


def test_idk_does_not_set_student_attempted():
    out = response_classifier(_state("i don't know", idk_count=0,
                                     student_attempted=False))
    assert out["student_attempted"] is False


# ── Non-idk path (mock LLM) ──────────────────────────────────────────────────

def _mock_llm_returning(label: str):
    fake_resp = MagicMock()
    fake_resp.content = [MagicMock(text=label)]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_resp
    return fake_client


def test_engagement_resets_counter():
    """Wrong attempt after 2 idks → counter resets to 0."""
    fake = _mock_llm_returning("incorrect")
    with patch("graph.nodes.response_classifier._client", fake):
        out = response_classifier(_state("is it the median nerve?", idk_count=2))
    assert out["classifier_output"] == "incorrect"
    assert out["idk_count"] == 0


def test_correct_resets_counter():
    fake = _mock_llm_returning("correct")
    with patch("graph.nodes.response_classifier._client", fake):
        out = response_classifier(_state("the ulnar nerve", idk_count=2))
    assert out["classifier_output"] == "correct"
    assert out["idk_count"] == 0


def test_questioning_resets_counter():
    fake = _mock_llm_returning("questioning")
    with patch("graph.nodes.response_classifier._client", fake):
        out = response_classifier(_state("can you explain that more?", idk_count=2))
    assert out["classifier_output"] == "questioning"
    assert out["idk_count"] == 0


def test_engagement_makes_student_attempted_sticky():
    fake = _mock_llm_returning("incorrect")
    with patch("graph.nodes.response_classifier._client", fake):
        out = response_classifier(_state("is it median?", idk_count=2,
                                         student_attempted=False))
    assert out["student_attempted"] is True


if __name__ == "__main__":
    test_idk_message_increments_counter()
    test_consecutive_idks_keep_counting()
    test_give_up_phrase_is_idk()
    test_idk_does_not_set_student_attempted()
    test_engagement_resets_counter()
    test_correct_resets_counter()
    test_questioning_resets_counter()
    test_engagement_makes_student_attempted_sticky()
    print("PASS: all 8 tests")
