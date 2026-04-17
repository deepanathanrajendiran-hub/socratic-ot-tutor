"""
graph/nodes/step_advancer.py

Mastery confirmer and post-mastery navigator.
Called when: classifier_output == "correct".

Sets mastery tier (pure Python), then generates a confirmation message
that offers the student three choices: clinical question, next topic, or done.

Mastery tiers (Python only — never in prompts):
  strong = correct before turn 3 (turn_count < SOCRATIC_TURN_GATE)
  weak   = correct at turn 3 (turn_count >= SOCRATIC_TURN_GATE)

The Dean node checks this draft before delivery (absolute rule).

Model: PRIMARY_MODEL (claude-sonnet-4-5)
Input:  current_concept, turn_count, domain, dean_revision_instruction
Output: draft_response, concept_mastered, mastery_level, student_phase
"""

import os

from anthropic import Anthropic

import config
from graph.state import GraphState

_client = Anthropic()


def _load_prompt() -> str:
    path = os.path.join(config.PROMPTS_DIR, "step_advancer.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def step_advancer(state: GraphState) -> dict:
    domain = state.get("domain", config.DOMAIN)
    domain_ctx = config.DOMAIN_CONFIG.get(domain, {}).get(
        "system_context", domain
    )

    concept = state.get("current_concept", "")
    turn_count = state.get("turn_count", 0)

    # Mastery tier — pure Python, never delegated to LLM
    mastery_level = (
        "strong" if turn_count < config.SOCRATIC_TURN_GATE else "weak"
    )

    prompt = _load_prompt().format(
        domain_context=domain_ctx,
        current_concept=concept,
        mastery_level=mastery_level,
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
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    if system_msg:
        api_kwargs["system"] = system_msg

    response = _client.messages.create(**api_kwargs)
    draft = response.content[0].text.strip()

    return {
        "draft_response": draft,
        "draft_source_node": "step_advancer",
        "concept_mastered": True,
        "mastery_level": mastery_level,
        "student_phase": "choice_pending",
    }
