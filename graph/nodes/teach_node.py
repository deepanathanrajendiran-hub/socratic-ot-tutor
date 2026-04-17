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

import os

from anthropic import Anthropic

import config
from graph.state import GraphState

_client = Anthropic()


def _load_prompt() -> str:
    path = os.path.join(config.PROMPTS_DIR, "teach.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _fill_prompt(template: str, **kwargs) -> str:
    """Safe placeholder substitution — avoids str.format() on prompts that
    may contain literal JSON braces in the retrieved_chunks text."""
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


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

    prompt = _fill_prompt(
        _load_prompt(),
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
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    if system_msg:
        api_kwargs["system"] = system_msg

    response = _client.messages.create(**api_kwargs)
    draft = response.content[0].text.strip()

    return {
        "draft_response": draft,
        "draft_source_node": "teach_node",
        "concept_mastered": False,
        "mastery_level": "failed",
        "student_phase": "choice_pending",
    }
