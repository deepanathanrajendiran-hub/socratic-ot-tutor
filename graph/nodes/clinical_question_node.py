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

import os

from anthropic import Anthropic

import config
from graph.state import GraphState

_client = Anthropic()


def _load_prompt() -> str:
    path = os.path.join(config.PROMPTS_DIR, "clinical_question.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _fill_prompt(template: str, **kwargs) -> str:
    """Safe substitution for prompts that may include retrieved chunks with
    curly braces (e.g. chemical notation, JSON examples in textbook content)."""
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


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

    prompt = _fill_prompt(
        _load_prompt(),
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
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    if system_msg:
        api_kwargs["system"] = system_msg

    response = _client.messages.create(**api_kwargs)
    draft = response.content[0].text.strip()

    return {
        "draft_response": draft,
        "draft_source_node": "clinical_question_node",
        "student_phase": "clinical_pending",
    }
