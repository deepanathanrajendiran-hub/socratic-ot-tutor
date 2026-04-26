"""
graph/nodes/topic_choice_classifier.py

Classifies the student's topic preference.
Called when: student_phase == "topic_choice_pending".

Classifies: weak | own | other
Resets student_phase to "learning" and clears mastery_choice.

Model: FAST_MODEL (claude-haiku-4-5)
Input:  messages (last student message), weak_topics
Output: topic_choice (str), student_phase ("learning"), mastery_choice ("")
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


VALID = {"weak", "own", "other"}


def topic_choice_classifier(state: GraphState) -> dict:
    messages = state.get("messages", [])
    student_message = _extract_last_student_message(messages)

    weak = state.get("weak_topics", [])
    weak_text = ", ".join(weak) if weak else "(none)"

    prompt = load_prompt("topic_choice_classifier.txt").format(
        weak_topics=weak_text,
        student_message=student_message,
    )

    response = _client.messages.create(
        model=config.FAST_MODEL,
        max_tokens=config.CLASSIFIER_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip().lower().split()[0] if response.content[0].text.strip() else "other"
    choice = raw if raw in VALID else "other"

    return {
        "topic_choice": choice,
        "student_phase": "learning",
        "mastery_choice": "",
    }
