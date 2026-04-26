"""
graph/nodes/topic_choice_node.py

Topic navigation prompt generator.
Called when: mastery_choice in ("next", "other").

Asks the student whether they want to practice from their weak topics list
or name a specific topic of their own.

The Dean node checks this draft before delivery (absolute rule).

Model: PRIMARY_MODEL (claude-sonnet-4-5)
Input:  weak_topics, domain, dean_revision_instruction
Output: draft_response, student_phase ("topic_choice_pending")
"""

from graph._llm_client import Anthropic

import config
from graph.state import GraphState
from graph.nodes._helpers import load_prompt

_client = Anthropic()


def topic_choice_node(state: GraphState) -> dict:
    domain = state.get("domain", config.DOMAIN)
    domain_ctx = config.DOMAIN_CONFIG.get(domain, {}).get(
        "system_context", domain
    )

    weak = state.get("weak_topics", [])
    weak_text = ", ".join(weak) if weak else "(none)"

    prompt = load_prompt("topic_choice.txt").format(
        domain_context=domain_ctx,
        weak_topics=weak_text,
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
        max_tokens=config.TOPIC_CHOICE_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    if system_msg:
        api_kwargs["system"] = system_msg

    response = _client.messages.create(**api_kwargs)
    draft = response.content[0].text.strip()

    return {
        "draft_response": draft,
        "draft_source_node": "topic_choice_node",
        "student_phase": "topic_choice_pending",
    }
