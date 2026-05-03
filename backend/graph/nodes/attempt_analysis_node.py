"""
graph/nodes/attempt_analysis_node.py

Optional fourth-option (D) handler that fires AFTER teach_node has
already revealed the answer. The student explicitly chose to review
where they went wrong; we extract their attempts from this Socratic
loop, hand them to Sonnet under a tight prompt, and emit a brief
per-attempt diagnostic followed by the standard A/B/C menu.

Triggered by: mastery_choice_classifier returning "analyze".

Reads: messages, current_concept, retrieved_chunks
Writes: draft_response, draft_source_node, student_phase=choice_pending,
        analysis_used=True (so the D pill doesn't reappear after the
        re-offered menu).

Dean still gates the draft (the prompt has hard rules against
fabrication, but Dean is the load-bearing check on grounding).

Model: PRIMARY_MODEL (Sonnet) — analysis tone matches teach_node.
"""
from __future__ import annotations

import sys

from graph._llm_client import Anthropic

import config
from graph.state import GraphState
from graph.nodes._helpers import (
    fill_prompt,
    load_prompt,
    log_thinking,
    msg_text,
    strip_thinking_block,
)

_client = Anthropic()


# Phrases that count as "no real attempt" and are skipped from the
# attempt block. Mirrors the prompt's rule 5 so the LLM never even
# sees the IDK lines as attempts to be referenced.
_IDK_SUBSTRINGS = (
    "i don't know", "i dont know", "i do not know",
    "no clue", "no idea", "not sure", "idk",
    "tell me", "just tell me",
)


# Phrases that mark an AI message as a Socratic content opener
# (teacher_socratic / hint_error / explain). The student's FIRST
# answer attempt is the human message right AFTER one of these.
# rapport / topic-choice / mastery-menu replies don't contain these
# cues, so they get correctly skipped — even when the rapport reply
# itself ends with a question mark.
_SOCRATIC_OPENER_CUES = (
    "before we",
    "let's start with",
    "what do you already know",
    "what do you think",
    "what's your understanding",
    "what has to happen",
    "tell me what you",
    "let's think about",
    "let's explore",
)


def _find_socratic_loop_start(messages: list) -> int:
    """Return the index of the FIRST AI message that looks like a
    Socratic teaching opener. Human messages with index > this are the
    student's actual attempts on the locked concept. Returns -1 when
    no opener has fired yet (caller should treat all human messages
    as pre-loop, i.e. there are no attempts to analyze).
    """
    for i, m in enumerate(messages):
        if getattr(m, "type", None) != "ai":
            continue
        text = (msg_text(m.content) or "").lower()
        if any(c in text for c in _SOCRATIC_OPENER_CUES):
            return i
    return -1


def _extract_attempts(
    messages: list,
    current_concept: str = "",
    max_keep: int = 6,
) -> list[str]:
    """Pull recent student attempts at answering a Socratic question.

    An "attempt" is a student message that is RESPONDING to a Socratic
    teaching question. The student's TOPIC-OPENER ("What's the gap
    between neurons?"), TOPIC-PICK after Choice B ("cerebellum"), or
    rapport-mode questions ("Is there a topic you'd like to explore?")
    are NOT attempts — they happen BEFORE the Socratic loop properly
    begins.

    Strategy: locate the index of the first AI message that looks like
    a Socratic opener (using cue-phrase heuristics — "before we", "what
    do you already know", etc.). Then only consider human messages
    that came AFTER that index. Apply the standard filters
    (IDK / single-letter / "review" trigger / bare concept-match)
    on top.

    Edge cases:
      - No Socratic opener fired yet → returns []. The prompt's rule 5
        emits the canned "you didn't make a content guess" line.
      - Domain change mid-session — the rapport rejection AI message
        before the domain switch doesn't contain Socratic cues, so the
        student's pre-switch question is correctly excluded. Reported
        and fixed 2026-05-02.

    Capped at max_keep — the prompt's <thinking> step picks the most
    informative 3 of those for the visible bullets.
    """
    opener_idx = _find_socratic_loop_start(messages)
    if opener_idx < 0:
        return []

    out: list[str] = []
    concept_lower = (current_concept or "").strip().lower()
    for i, m in enumerate(messages):
        if i <= opener_idx:
            continue  # pre-loop messages, including the opener itself
        if getattr(m, "type", None) != "human":
            continue
        text = (msg_text(m.content) or "").strip()
        if not text:
            continue
        lower = text.lower()
        if lower in {"d", "a", "b", "c", "yes", "sure", "ok", "okay"}:
            continue
        if "review" in lower and ("wrong" in lower or "attempt" in lower):
            continue
        if any(s in lower for s in _IDK_SUBSTRINGS):
            continue
        # Bare exact concept match → topic announcement caught by the
        # classifier's topic-naming guard, not an attempt.
        if concept_lower and lower == concept_lower:
            continue
        out.append(text)
    return out[-max_keep:]


def _format_attempts_block(attempts: list[str]) -> str:
    """Render attempts as a numbered list for the prompt. When empty,
    return an explicit marker so the LLM falls into the all-IDK branch
    of its instructions."""
    if not attempts:
        return "(no content guesses — student only used IDK / give-up turns)"
    return "\n".join(f"{i + 1}. {text}" for i, text in enumerate(attempts))


def attempt_analysis_node(state: GraphState) -> dict:
    domain = state.get("domain", config.DOMAIN)
    domain_ctx = config.DOMAIN_CONFIG.get(domain, {}).get(
        "system_context", domain
    )
    concept = state.get("current_concept", "")
    chunks = state.get("retrieved_chunks", []) or []
    retrieved_text = (
        "\n\n---\n\n".join(chunks) if chunks else "(no content retrieved)"
    )
    attempts = _extract_attempts(
        state.get("messages", []) or [],
        current_concept=concept,
    )
    attempts_block = _format_attempts_block(attempts)

    prompt = fill_prompt(
        load_prompt("attempt_analysis.txt"),
        domain_context=domain_ctx,
        current_concept=concept,
        retrieved_chunks=retrieved_text,
        attempts_block=attempts_block,
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
        # Reuse the teach-node budget — analysis is the same length
        # ballpark (3 bullets + close + A/B/C menu).
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
        node="attempt_analysis_node",
        session_id=state.get("session_id", ""),
        turn_count=state.get("turn_count", 0),
        concept=concept,
        classifier_output=state.get("classifier_output", ""),
        reveal_permitted=True,  # we're post-reveal by definition
    )
    if not draft.strip():
        # Defensive: empty draft → fall back to a minimal message rather
        # than ship an empty bubble. Dean would catch it but this saves
        # the round trip.
        draft = (
            "Here's the short version: a few of your guesses were close "
            f"to {concept!r} but didn't quite name it. The reveal above "
            "captures the canonical definition."
        )
        print(
            f"[attempt_analysis_node] empty Sonnet draft — using fallback "
            f"| concept={concept!r}",
            file=sys.stderr,
        )

    # Deterministic menu append. The prompt asks for a closing A/B/C
    # menu (rule 7) but Sonnet sometimes truncates after the bullet
    # diagnostics — leaving the student with a wall of analysis but no
    # navigation. We append the menu in code so it's guaranteed to be
    # there. ChoiceButtons.tsx parses A/B/C from this exact format.
    menu_block = (
        "\n\nWhat would you like to do next?\n"
        "A) Try a clinical application question for this concept\n"
        "B) Move on to the next topic\n"
        "C) Stop here for now"
    )
    # Skip if the LLM already produced a menu (avoid duplicate) — check
    # for a contiguous A) ... B) ... C) run in the existing draft.
    has_menu = (
        "A)" in draft and "B)" in draft and "C)" in draft
        and draft.find("A)") < draft.find("B)") < draft.find("C)")
    )
    if not has_menu:
        draft = draft.rstrip() + menu_block

    return {
        "draft_response": draft,
        "draft_source_node": "attempt_analysis_node",
        "student_phase": "choice_pending",
        "analysis_used": True,
        # Reset the per-turn Dean state so the next /chat turn starts
        # cleanly. mastery_choice / topic_choice already cleared.
        "dean_revisions": 0,
        "dean_revision_instruction": "",
    }
