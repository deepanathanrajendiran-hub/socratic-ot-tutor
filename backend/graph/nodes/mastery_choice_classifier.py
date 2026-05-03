"""
graph/nodes/mastery_choice_classifier.py

Classifies the student's post-mastery intent.
Called when: student_phase == "choice_pending".

Reads the student's response to the three-way choice offer and classifies it
as: clinical | next | done | other

Model: FAST_MODEL (claude-haiku-4-5)
Input:  messages (last student message)
Output: mastery_choice (str)
"""

from graph._llm_client import Anthropic

import config
from graph.state import GraphState
from graph.nodes._helpers import load_prompt, msg_text

_client = Anthropic()


def _extract_last_student_message(messages) -> str:
    for msg in reversed(messages):
        if msg.type == "human":
            return msg_text(msg.content)
    return ""


VALID = {"clinical", "next", "done", "analyze", "other"}


# Deterministic shortcuts: when the student types a single letter or a
# crystal-clear analysis-request phrase, skip the Haiku call and route
# directly. Saves one LLM call on the most common bare-pick interactions
# and guarantees the literal "D" → analyze mapping that the new D pill
# button on the frontend produces.
_LETTER_SHORTCUT = {
    "a": "clinical", "b": "next", "c": "done", "d": "analyze",
}
_ANALYZE_PHRASES = (
    "review my attempts",
    "review where i went wrong",
    "where did i go wrong",
    "where i went wrong",
    "analyze my answers",
    "analyze my attempts",
    "analysis please",
    "show me my mistakes",
)


def _shortcut_choice(student_message: str) -> str | None:
    """Return a fixed choice label for unambiguous student messages, or
    None when we should fall through to Haiku."""
    if not student_message:
        return None
    stripped = student_message.strip().rstrip(".!?,").lower()
    if stripped in _LETTER_SHORTCUT:
        return _LETTER_SHORTCUT[stripped]
    if any(p in stripped for p in _ANALYZE_PHRASES):
        return "analyze"
    return None


def mastery_choice_classifier(state: GraphState) -> dict:
    messages = state.get("messages", [])
    student_message = _extract_last_student_message(messages)

    shortcut = _shortcut_choice(student_message)
    if shortcut is not None:
        choice = shortcut
    else:
        prompt = load_prompt("mastery_choice.txt").format(student_message=student_message)
        response = _client.messages.create(
            model=config.FAST_MODEL,
            max_tokens=config.CLASSIFIER_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip().lower().split()[0] if response.content[0].text.strip() else "other"
        choice = raw if raw in VALID else "other"

    # Always reset student_phase — the classifier's job is done after one call.
    # Downstream nodes (clinical_question_node, topic_choice_node, END) set
    # their own next phase. This prevents the "done" infinite loop where
    # student_phase stayed "choice_pending" and re-routed every subsequent
    # message back into this classifier.
    base: dict = {"mastery_choice": choice, "student_phase": "learning"}

    # Choice C ("done") routes straight to END — no downstream LLM-generation
    # node fires. Without an AI message in the result, the API's
    # _last_ai_text() falls back to the previous turn's mastery menu and the
    # frontend sees the menu repeat. Bake a friendly close message here so
    # the user gets a clear signal that the session is winding down while
    # leaving the door open for them to come back.
    if choice == "done":
        from langchain_core.messages import AIMessage
        close_msg = (
            "Great work — that's a wrap for now. If you have any doubts "
            "about this concept later, or want to revisit it, come back "
            "here and ask me again anytime. I'm always around."
        )
        base["messages"] = [AIMessage(content=close_msg)]
        base["draft_response"] = close_msg
        base["turn_count"] = state.get("turn_count", 0) + 1

    return base
