"""
graph/nodes/response_classifier.py

Classifies the student's latest message into exactly one label:
  irrelevant | questioning | incorrect | correct

Model: FAST_MODEL (claude-haiku-4-5) — single-word output, max_tokens=10.
Input:  state["current_concept"], state["messages"]
Output: state["classifier_output"]
"""

import os

from anthropic import Anthropic

import config
from graph.state import GraphState

_client = Anthropic()


def _load_prompt() -> str:
    path = os.path.join(config.PROMPTS_DIR, "response_classifier.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _msg_text(content) -> str:
    """Safe string extraction — content can be str or list[dict] for multimodal."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return " ".join(parts)
    return str(content)


def _format_last_two_turns(messages) -> str:
    """Format up to 4 prior messages (2 exchanges) as plain text for context."""
    if not messages:
        return "(no prior context)"
    lines = []
    for msg in messages:
        role = "Student" if msg.type == "human" else "Tutor"
        lines.append(f"{role}: {_msg_text(msg.content)}")
    return "\n".join(lines)


def response_classifier(state: GraphState) -> dict:
    messages = state.get("messages", [])

    # Latest message is the student's current input
    student_message = ""
    if messages and messages[-1].type == "human":
        student_message = _msg_text(messages[-1].content)

    # Prior context: up to 4 messages before the current one (2 full exchanges)
    prior = messages[:-1][-4:] if len(messages) > 1 else []
    last_two_turns = _format_last_two_turns(prior)

    prompt = _load_prompt().format(
        current_concept=state.get("current_concept", ""),
        student_message=student_message,
        last_two_turns=last_two_turns,
    )

    response = _client.messages.create(
        model=config.FAST_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip().lower()

    valid = {"irrelevant", "questioning", "incorrect", "correct"}
    label = raw if raw in valid else "incorrect"

    return {"classifier_output": label}
