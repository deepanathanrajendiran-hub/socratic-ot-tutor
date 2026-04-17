"""
graph/nodes/explain_node.py

Clarification responder for student questions.
Called when: classifier_output == "questioning".

Answers the student's clarifying question with a Socratic scaffold — still
guides toward the answer rather than giving it directly.

The Dean node checks this draft before delivery (absolute rule).

Model: PRIMARY_MODEL (claude-sonnet-4-5)
Input:  current_concept, retrieved_chunks, turn_count, student_attempted,
        messages, domain, dean_revision_instruction
Output: state["draft_response"]
"""

import os

from anthropic import Anthropic

import config
from graph.state import GraphState

_client = Anthropic()


def _load_prompt() -> str:
    path = os.path.join(config.PROMPTS_DIR, "explain.txt")
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
    """Return the most recent human message — the clarifying question."""
    for msg in reversed(messages):
        if msg.type == "human":
            return _msg_text(msg.content)
    return "(no student message found)"


def _should_reveal(state: GraphState) -> bool:
    if state.get("concept_mastered", False):
        return True
    if state.get("student_phase", "learning") != "learning":
        return True
    return state.get("turn_count", 0) >= config.SOCRATIC_TURN_GATE


def explain_node(state: GraphState) -> dict:
    domain = state.get("domain", config.DOMAIN)
    domain_ctx = config.DOMAIN_CONFIG.get(domain, {}).get(
        "system_context", domain
    )

    concept = state.get("current_concept", "")
    chunks = state.get("retrieved_chunks", [])
    retrieved_text = (
        "\n\n---\n\n".join(chunks) if chunks else "(no content retrieved)"
    )

    turn_count = state.get("turn_count", 0)
    reveal_permitted = _should_reveal(state)

    messages = state.get("messages", [])
    student_message = _extract_last_student_message(messages)

    prompt = _load_prompt().format(
        domain_context=domain_ctx,
        current_concept=concept,
        retrieved_chunks=retrieved_text,
        student_message=student_message,
        turn_count=turn_count,
        reveal_permitted=reveal_permitted,
        max_sentences=config.MAX_RESPONSE_SENTENCES,
    )

    revision_instruction = state.get("dean_revision_instruction", "")
    system_msg = (
        f"REVISION REQUIRED: {revision_instruction}\n"
        "Fix the issue above and rewrite the response."
        if revision_instruction
        else None
    )

    api_kwargs: dict = dict(
        model=config.PRIMARY_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    if system_msg:
        api_kwargs["system"] = system_msg

    response = _client.messages.create(**api_kwargs)
    draft = response.content[0].text.strip()

    return {"draft_response": draft, "draft_source_node": "explain_node"}
