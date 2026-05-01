"""
graph/nodes/clinical_question_node.py

Clinical OT application question generator.
Called when: mastery_choice == "clinical".

Generates one clinical scenario question grounded in retrieved chunks,
then sets student_phase = "clinical_pending" so the next student message
routes directly to synthesis_assessor.

The Dean node checks this draft before delivery (absolute rule).

Runs its own retrieval on `current_concept` so the scenario is grounded
in fresh chunks and the dashboard's CRAG / chunk_sources state reflects
this turn's retrieval — not stale values left over from prior turns
(e.g. the "It's the synapse" turn whose CRAG was INCORRECT because
"It's the synapse" alone is a weak retrieval query).

Model: PRIMARY_MODEL (claude-sonnet-4-5)
Input:  current_concept, domain, weak_topics, dean_revision_instruction
Output: draft_response, retrieved_chunks, chunk_sources, crag_decision,
        student_phase ("clinical_pending")
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

    # Fresh retrieval keyed on the locked concept. We don't reuse the
    # prior turn's chunks because that turn's query was the student's
    # short confirmation ("It's the synapse"), which CRAG often scores
    # INCORRECT — leaving the scenario ungrounded.
    fresh_chunks: list[str] = []
    fresh_sources: list[str] = []
    crag_decision: str = ""
    try:
        from retrieval.crag import corrective_retrieve
        reranked, section_texts, crag_log = corrective_retrieve(
            query=concept,
            weak_topics=state.get("weak_topics", []),
            turn_query=concept,
        )
        fresh_chunks = section_texts
        fresh_sources = [c.get("id", "") for c in reranked]
        crag_decision = crag_log.get("crag_decision", "")
    except Exception as exc:
        print(f"[clinical_question_node] WARNING: retrieval failed — {exc}")
        crag_decision = "FAILED"

    retrieved_text = (
        "\n\n---\n\n".join(fresh_chunks) if fresh_chunks
        else "(no content retrieved)"
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
        "retrieved_chunks": fresh_chunks,
        "chunk_sources": fresh_sources,
        "crag_decision": crag_decision,
    }
