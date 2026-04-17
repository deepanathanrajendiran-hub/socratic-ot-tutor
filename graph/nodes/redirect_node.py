"""
graph/nodes/redirect_node.py

Redirect generator for off-topic student messages.
Called when: classifier_output == "irrelevant".

Acknowledges the off-topic message briefly, then bridges back to the
current anatomy concept with a Socratic question.

The Dean node checks this draft before delivery (absolute rule).

Model: PRIMARY_MODEL (claude-sonnet-4-5)
Input:  current_concept, turn_count, messages, dean_revision_instruction
Output: state["draft_response"]
"""

import os

from anthropic import Anthropic

import config
from graph.state import GraphState

_client = Anthropic()


def _load_prompt() -> str:
    path = os.path.join(config.PROMPTS_DIR, "redirect.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _msg_text(content) -> str:
    """Safe text extraction — content may be str or list[dict] for multimodal."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def _extract_last_student_message(messages) -> str:
    """Return the most recent human message — the off-topic message to redirect from."""
    for msg in reversed(messages):
        if msg.type == "human":
            return _msg_text(msg.content)
    return "(no student message found)"


def redirect_node(state: GraphState) -> dict:
    concept = state.get("current_concept", "")
    turn_count = state.get("turn_count", 0)

    messages = state.get("messages", [])
    student_message = _extract_last_student_message(messages)

    prompt = _load_prompt().format(
        current_concept=concept,
        student_message=student_message,
        turn_count=turn_count,
    )

    # On revision pass, inject Dean's instruction as a system message
    revision_instruction = state.get("dean_revision_instruction", "")
    system_msg = (
        f"REVISION REQUIRED: {revision_instruction}\n"
        "Fix the issue above and rewrite the response."
        if revision_instruction
        else None
    )

    api_kwargs: dict = dict(
        model=config.PRIMARY_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    if system_msg:
        api_kwargs["system"] = system_msg

    response = _client.messages.create(**api_kwargs)
    draft = response.content[0].text.strip()

    return {"draft_response": draft, "draft_source_node": "redirect_node"}
