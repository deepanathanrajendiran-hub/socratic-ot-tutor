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

import sys

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

    # Conditionally include the D pill ("Review where you went wrong"):
    # only when the student made at least one real attempt this loop
    # AND analysis hasn't already fired for this loop. On an IDK-only
    # ladder the student has no attempts to analyze, so we keep the
    # menu at the standard A/B/C — no point offering a button that
    # would surface the canned "you didn't make a content guess" line.
    student_attempted = bool(state.get("student_attempted", False))
    analysis_used     = bool(state.get("analysis_used", False))
    if student_attempted and not analysis_used:
        choice_menu = (
            "    A) Try a clinical application question for this concept\n"
            "    B) Move on to the next topic\n"
            "    C) Stop here for now\n"
            "    D) Review where you went wrong"
        )
    else:
        choice_menu = (
            "    A) Try a clinical application question for this concept\n"
            "    B) Move on to the next topic\n"
            "    C) Stop here for now"
        )

    discovery_target = (state.get("discovery_target") or "name").strip()
    prompt_file = (
        "teach_function.txt"
        if discovery_target == "function"
        else "teach.txt"
    )
    prompt = fill_prompt(
        load_prompt(prompt_file),
        domain_context=domain_ctx,
        current_concept=concept,
        retrieved_chunks=retrieved_text,
        turn_count=turn_count,
        choice_menu=choice_menu,
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

    # Function-mode placeholder back-substitution. Mirrors teacher_socratic
    # and hint_error_node: if the LLM emitted "[the target structure]" /
    # "[this structure]" instead of the literal concept name, replace it.
    if discovery_target == "function" and concept and draft:
        import re as _re
        placeholder_re = _re.compile(
            r"\[\s*(?:(?:the|this|that)\s+)?"
            r"(?:target\s+)?"
            r"(?:structure|concept|region|area)"
            r"\s*\]",
            _re.IGNORECASE,
        )
        if placeholder_re.search(draft):
            draft = placeholder_re.sub(concept, draft)
            print(
                f"[teach_node] placeholder-leak strip: replaced bracketed "
                f"template with {concept!r}",
                file=sys.stderr,
            )

    # Deterministic guard: the LLM is occasionally given chunks covering
    # multiple structures (the OT supplement has Section 1 ulnar /
    # Section 3 median / Section 2 radial) and picks the wrong one despite
    # the prompt saying "name {current_concept}". If the rendered concept
    # name is not present in the visible draft, prepend the correct reveal
    # so the student doesn't get a confidently-wrong identification.
    if concept and concept.lower() not in (draft or "").lower():
        print(
            f"[teach_node] WRONG-CONCEPT-REVEAL guard: concept={concept!r} "
            f"absent from draft → prepending correct reveal | "
            f"draft_head={draft[:120]!r}",
            file=sys.stderr,
        )
        draft = f"The answer is **{concept}**. " + (draft or "").lstrip()

    # Record the failed concept in weak_topics so the student can revisit
    # it via Choice B → 'weak' later in the session. teach_node fires when
    # the student couldn't reach the answer themselves (IDK ladder or
    # turn-gate reveal) — that's the exact signal the weak-topic dashboard
    # cares about. Idempotent: only append if the concept isn't already
    # there.
    weak_topics = list(state.get("weak_topics", []) or [])
    if concept and concept not in weak_topics:
        weak_topics.append(concept)
        print(
            f"[teach_node] weak-topic recorded: {concept!r} "
            f"(weak_topics now: {weak_topics!r})",
            file=sys.stderr,
        )
    # Persist the same concept at the USER level so it follows the
    # student into future sessions — not just this one. Best-effort;
    # a DB hiccup must not break the chat response.
    user_id = (state.get("user_id") or "").strip()
    if user_id and concept:
        try:
            from api import user_weak_topics as _uwt
            _uwt.add_weak(user_id, concept, mastery_level="failed")
        except Exception as exc:
            print(
                f"[teach_node] user_weak_topics.add_weak failed "
                f"(non-fatal): {exc!r}",
                file=sys.stderr,
            )

    return {
        "draft_response": draft,
        "draft_source_node": "teach_node",
        "concept_mastered": False,
        "mastery_level": "failed",
        "student_phase": "choice_pending",
        "weak_topics": weak_topics,
        # Reset per-loop counters — the answer was just revealed, so
        # whatever's next is a fresh interaction, not a continuation of
        # the failed attempt cycle. Mirrors the reset in step_advancer.
        "turn_count": 0,
        "student_attempted": False,
        "idk_count": 0,
        "dean_revisions": 0,
        "dean_revision_instruction": "",
    }
