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

import os

from anthropic import Anthropic

import config
from graph.state import GraphState

_client = Anthropic()


def _load_prompt() -> str:
    path = os.path.join(config.PROMPTS_DIR, "mastery_choice.txt")
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


VALID = {"clinical", "next", "done", "other"}


def mastery_choice_classifier(state: GraphState) -> dict:
    messages = state.get("messages", [])
    student_message = _extract_last_student_message(messages)

    prompt = _load_prompt().format(student_message=student_message)

    response = _client.messages.create(
        model=config.FAST_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip().lower().split()[0] if response.content[0].text.strip() else "other"
    choice = raw if raw in VALID else "other"

    return {"mastery_choice": choice}
