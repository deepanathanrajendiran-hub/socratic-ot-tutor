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

import os

from anthropic import Anthropic

import config
from graph.state import GraphState

_client = Anthropic()


def _load_prompt() -> str:
    path = os.path.join(config.PROMPTS_DIR, "topic_choice_classifier.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _msg_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def _extract_last_student_message(messages) -> str:
    for msg in reversed(messages):
        if msg.type == "human":
            return _msg_text(msg.content)
    return ""


VALID = {"weak", "own", "other"}


def topic_choice_classifier(state: GraphState) -> dict:
    messages = state.get("messages", [])
    student_message = _extract_last_student_message(messages)

    weak = state.get("weak_topics", [])
    weak_text = ", ".join(weak) if weak else "(none)"

    prompt = _load_prompt().format(
        weak_topics=weak_text,
        student_message=student_message,
    )

    response = _client.messages.create(
        model=config.FAST_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip().lower().split()[0] if response.content[0].text.strip() else "other"
    choice = raw if raw in VALID else "other"

    return {
        "topic_choice": choice,
        "student_phase": "learning",
        "mastery_choice": "",
    }
