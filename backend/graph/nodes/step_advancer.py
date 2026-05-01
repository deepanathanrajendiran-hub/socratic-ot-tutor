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

import sys

from graph._llm_client import Anthropic

import config
from graph.state import GraphState
from graph.nodes._helpers import (
    load_prompt,
    log_thinking,
    strip_thinking_block,
)

_client = Anthropic()


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

    # If the student got this concept correct (any tier), drop it from
    # their persistent user-level weak list — the brief's "until they
    # solve it" condition. Only "strong" is unambiguous mastery; "weak"
    # (correct at turn-gate) is borderline. We currently clear on BOTH
    # since the alternative (keeping a passed-at-turn-3 topic in the
    # sidebar forever) is more annoying than the converse. Adjust here
    # if the policy needs to be stricter.
    user_id = (state.get("user_id") or "").strip()
    if user_id and concept:
        try:
            from api import user_weak_topics as _uwt
            _uwt.remove_weak(user_id, concept)
        except Exception as exc:
            print(
                f"[step_advancer] user_weak_topics.remove_weak failed "
                f"(non-fatal): {exc!r}",
                file=sys.stderr,
            )

    # Mirror the removal in the in-session state so the sidebar drops
    # the chip immediately — without this, the student sees the just-
    # mastered concept hanging around until the next /sessions/{id} GET.
    session_weak = list(state.get("weak_topics", []) or [])
    if concept and concept in session_weak:
        session_weak = [c for c in session_weak if c != concept]

    # Detect misspelling so the confirmation can gently include the
    # correct form. Reuses the classifier's fuzzy detector so the two
    # surfaces stay consistent: if the classifier promoted on a fuzzy
    # match, this fires the spelling note; otherwise it stays empty.
    last_student = ""
    for m in reversed(state.get("messages", []) or []):
        if getattr(m, "type", None) == "human":
            from graph.nodes._helpers import msg_text
            last_student = msg_text(m.content).strip()
            break
    spelling_note = ""
    if last_student and concept:
        from graph.nodes.response_classifier import _is_misspelled_concept
        if _is_misspelled_concept(last_student, concept):
            spelling_note = (
                f"SPELLING NOTE: the student wrote {last_student!r} but the "
                f"correct term is {concept!r}. Their answer is right; weave "
                f"the correct spelling into the confirmation naturally — for "
                f"example: \"That's right — the term is **{concept}**.\" Keep "
                f"it brief, one sentence, no scolding."
            )

    prompt = load_prompt("step_advancer.txt").format(
        domain_context=domain_ctx,
        current_concept=concept,
        mastery_level=mastery_level,
        spelling_note=spelling_note,
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
        max_tokens=config.STEP_ADVANCER_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    if system_msg:
        api_kwargs["system"] = system_msg

    response = _client.messages.create(**api_kwargs)
    raw = response.content[0].text
    draft, thinking = strip_thinking_block(raw)
    log_thinking(
        thinking,
        node="step_advancer",
        session_id=state.get("session_id", ""),
        turn_count=turn_count,
        concept=concept,
        classifier_output=state.get("classifier_output", ""),
        reveal_permitted=False,
    )

    return {
        "draft_response": draft,
        "draft_source_node": "step_advancer",
        "concept_mastered": True,
        "mastery_level": mastery_level,
        "student_phase": "choice_pending",
        "weak_topics": session_weak,  # mastered concept dropped above
        # Reset per-loop counters now that the student has mastered this
        # concept. The next interaction is the A/B/C menu — not a Socratic
        # turn — and any subsequent topic switch starts at turn 0 with
        # no inherited "attempted" / "idk" baggage from the prior concept.
        "turn_count": 0,
        "student_attempted": False,
        "idk_count": 0,
        "dean_revisions": 0,
        "dean_revision_instruction": "",
    }
