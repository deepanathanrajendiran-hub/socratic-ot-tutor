"""
graph/nodes/teach_node.py

Failed-mastery reveal and explainer.
Called when: classifier_output == "incorrect" AND turn_count >= SOCRATIC_TURN_GATE.

Reveals the concept directly, provides a grounded explanation, then offers
the same post-mastery choice as step_advancer (clinical Q / next topic / done).

The Dean node checks this draft before delivery (absolute rule).

Model: PRIMARY_MODEL (claude-sonnet-4-5)
Input:  current_concept, retrieved_chunks, turn_count, domain,
        dean_revision_instruction
Output: draft_response, concept_mastered (False), mastery_level ("failed"),
        student_phase ("choice_pending")
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


def teach_node(state: GraphState) -> dict:
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

    prompt = fill_prompt(
        load_prompt("teach.txt"),
        domain_context=domain_ctx,
        current_concept=concept,
        retrieved_chunks=retrieved_text,
        turn_count=turn_count,
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
        max_tokens=config.TEACH_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    if system_msg:
        api_kwargs["system"] = system_msg

    response = _client.messages.create(**api_kwargs)
    raw = response.content[0].text
    draft, thinking = strip_thinking_block(raw)
    log_thinking(
        thinking,
        node="teach_node",
        session_id=state.get("session_id", ""),
        turn_count=turn_count,
        concept=concept,
        classifier_output=state.get("classifier_output", ""),
        reveal_permitted=True,
    )

    return {
        "draft_response": draft,
        "draft_source_node": "teach_node",
        "concept_mastered": False,
        "mastery_level": "failed",
        "student_phase": "choice_pending",
    }
