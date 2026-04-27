"""
graph/nodes/clinical_question_node.py

Clinical OT application question generator.
Called when: mastery_choice == "clinical".

Generates one clinical scenario question grounded in retrieved chunks,
then sets student_phase = "clinical_pending" so the next student message
routes directly to synthesis_assessor.

The Dean node checks this draft before delivery (absolute rule).

Model: PRIMARY_MODEL (claude-sonnet-4-5)
Input:  current_concept, retrieved_chunks, domain, dean_revision_instruction
Output: draft_response, student_phase ("clinical_pending")
"""

from graph._llm_client import Anthropic

import config
from graph.state import GraphState
from graph.nodes._helpers import (
    fill_prompt,
    load_prompt,
    log_thinking,
    strip_thinking_block,
)

_client = Anthropic()


def clinical_question_node(state: GraphState) -> dict:
    domain = state.get("domain", config.DOMAIN)
    domain_ctx = config.DOMAIN_CONFIG.get(domain, {}).get(
        "system_context", domain
    )

    concept = state.get("current_concept", "")
    chunks = state.get("retrieved_chunks", [])
    retrieved_text = (
        "\n\n---\n\n".join(chunks) if chunks else "(no content retrieved)"
    )

    prompt = fill_prompt(
        load_prompt("clinical_question.txt"),
        domain_context=domain_ctx,
        current_concept=concept,
        retrieved_chunks=retrieved_text,
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
        max_tokens=config.CLINICAL_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    if system_msg:
        api_kwargs["system"] = system_msg

    response = _client.messages.create(**api_kwargs)
    raw = response.content[0].text
    draft, thinking = strip_thinking_block(raw)
    log_thinking(
        thinking,
        node="clinical_question_node",
        session_id=state.get("session_id", ""),
        turn_count=state.get("turn_count", 0),
        concept=concept,
        classifier_output=state.get("classifier_output", ""),
        reveal_permitted=False,
    )

    return {
        "draft_response": draft,
        "draft_source_node": "clinical_question_node",
        "student_phase": "clinical_pending",
    }
