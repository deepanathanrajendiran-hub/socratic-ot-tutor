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

from graph._llm_client import Anthropic

import config
from graph.state import GraphState
from graph.nodes._helpers import (
    load_prompt,
    log_thinking,
    msg_text,
    strip_thinking_block,
)

_client = Anthropic()


def _extract_last_student_message(messages) -> str:
    """Return the most recent human message — the clarifying question."""
    for msg in reversed(messages):
        if msg.type == "human":
            return msg_text(msg.content)
    return "(no student message found)"


from graph.edges import should_reveal as _should_reveal  # canonical reveal gate


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

    prompt = load_prompt("explain.txt").format(
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
        max_tokens=config.EXPLAIN_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    if system_msg:
        api_kwargs["system"] = system_msg

    response = _client.messages.create(**api_kwargs)
    raw = response.content[0].text
    draft, thinking = strip_thinking_block(raw)
    log_thinking(
        thinking,
        node="explain_node",
        session_id=state.get("session_id", ""),
        turn_count=turn_count,
        concept=concept,
        classifier_output=state.get("classifier_output", ""),
        reveal_permitted=reveal_permitted,
    )

    return {"draft_response": draft, "draft_source_node": "explain_node"}
