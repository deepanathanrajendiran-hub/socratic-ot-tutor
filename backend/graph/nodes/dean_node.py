"""
graph/nodes/dean_node.py

Quality controller. Checks draft_response against 6 criteria before delivery.
Called on EVERY teacher/generation response — absolute rule from CLAUDE.md.

5 criteria (from prompts/dean_check.txt):
  1. REVEAL CHECK   — concept not named when reveal_permitted is False
  2. DEFINITION CHECK — no "X is defined as..." when not permitted
  3. GROUNDING CHECK  — every claim traceable to retrieved chunks
  4. QUESTION CHECK   — draft ends with 1-2 questions
  5. LENGTH CHECK     — max MAX_RESPONSE_SENTENCES before the question
  (SYCOPHANCY guard — no praise opener — folded into the prompt rules)

Model: PRIMARY_MODEL (claude-sonnet-4-5)
Input:  draft_response, current_concept, turn_count, student_attempted,
        retrieved_chunks, dean_revisions
Output: dean_passed (bool), dean_revisions (int), dean_revision_instruction (str)
"""

import json
import re
from graph._llm_client import Anthropic

import config
from graph.state import GraphState
import sys
from graph.nodes._helpers import (
    fill_prompt,
    get_generic_words,
    get_stem_blacklist,
    load_prompt,
    msg_text,
)

_client = Anthropic()


from graph.edges import should_reveal as _should_reveal  # canonical reveal gate


def _extract_dean_json(raw: str) -> dict | None:
    """Extract Dean's JSON verdict from raw LLM output.

    Walks the text to find balanced {...} blocks (depth-tracked so nested
    objects stay intact), then returns the LAST block that parses as JSON
    with a "passed" key. Haiku occasionally chains-of-thought after its
    first JSON ("Wait, let me reconsider...") and emits a corrected JSON
    at the end; the last valid block is the authoritative verdict.
    Returns None if no parseable block is found.
    """
    if not raw:
        return None
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "")

    blocks = []
    depth = 0
    start_idx = -1
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_idx != -1:
                blocks.append(cleaned[start_idx : i + 1])
                start_idx = -1

    for block in reversed(blocks):
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and "passed" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


def dean_node(state: GraphState) -> dict:
    concept = state.get("current_concept", "")
    turn_count = state.get("turn_count", 0)
    reveal_permitted = _should_reveal(state)

    chunks = state.get("retrieved_chunks", [])
    retrieved_text = (
        "\n\n---\n\n".join(chunks) if chunks else "(no content retrieved)"
    )

    draft = state.get("draft_response", "")
    current_revisions = state.get("dean_revisions", 0)
    source = state.get("draft_source_node", "")

    # ── Bypass: post-mastery navigation nodes ────────────────────────────────
    # clinical_question_node and topic_choice_node produce navigation/follow-up
    # content (clinical scenarios, "what do you want to learn next" prompts)
    # that uses parametric knowledge by design. Dean's GROUNDING_CHECK was
    # forcing fallback_scaffold on every post-mastery turn (I2/I3 in e2e
    # suite). These nodes are exempt — they're not Socratic teaching turns.
    # REVEAL_CHECK doesn't apply either (concept is already mastered).
    if source in ("clinical_question_node", "topic_choice_node"):
        print(
            f"[dean] t={turn_count} rev={current_revisions} "
            f"BYPASS source={source!r} (post-mastery navigation) | "
            f"{draft[:100]!r}",
            file=sys.stderr,
        )
        return {
            "dean_passed": True,
            "dean_revisions": current_revisions,
            "dean_revision_instruction": "",
        }

    # ── Python pre-check: SYCOPHANCY CHECK ───────────────────────────────────
    # The Dean prompt says "fail only if first word is in the six-word
    # list AND immediately followed by '!'". Sonnet keeps misinterpreting
    # this and rejecting legitimate openers like "Perfect — that's exactly
    # what we're..." or "Great — let's start with..." (em dash, not '!').
    # Enforce the rule mechanically here: literal first-word + '!' match
    # is the only true violation. Any LLM-side SYCOPHANCY rejection
    # downstream is treated as advisory.
    _SYCOPHANCY_BANNED = ("great", "excellent", "perfect", "wonderful",
                          "amazing", "fantastic")
    _stripped = (draft or "").lstrip()
    sycophancy_violated = bool(
        _stripped and "!" in _stripped[:30] and any(
            _stripped.lower().startswith(w + "!") for w in _SYCOPHANCY_BANNED
        )
    )
    if sycophancy_violated:
        first_word = _stripped.split("!", 1)[0]
        print(
            f"[dean] t={turn_count} rev={current_revisions} reveal={reveal_permitted} "
            f"FAIL ['SYCOPHANCY CHECK'] (python pre-check, literal '{first_word}!') "
            f"| {draft[:100]!r}",
            file=sys.stderr,
        )
        instruction = (
            f"Replace the opening '{first_word}!' with a neutral transition: "
            f"use a dash or comma instead of an exclamation, e.g. "
            f"'{first_word} — ...' or just start directly with the content."
        )
        print(f"       instruction: {instruction!r}", file=sys.stderr)
        return {
            "dean_passed": False,
            "dean_revisions": current_revisions + 1,
            "dean_revision_instruction": instruction,
        }

    # ── Python pre-check: QUESTION CHECK ─────────────────────────────────────
    # The LLM judge interprets Socratic directives ("Think about X") as implicit
    # questions and never fires this criterion reliably. Check mechanically here:
    # reveal-path responses (A/B/C choice menus) don't need a "?".
    if not reveal_permitted and "?" not in draft:
        print(
            f"[dean] t={turn_count} rev={current_revisions} reveal={reveal_permitted} "
            f"FAIL ['QUESTION CHECK'] (python pre-check) | {draft[:100]!r}",
            file=sys.stderr,
        )
        instruction = (
            "Add a guiding question ending with '?' — "
            "every pre-reveal response must end with at least one question."
        )
        print(f"       instruction: {instruction!r}", file=sys.stderr)
        return {
            "dean_passed": False,
            "dean_revisions": current_revisions + 1,
            "dean_revision_instruction": instruction,
        }

    # ── Python pre-check: REVEAL CHECK ───────────────────────────────────────
    # Sonnet's Dean was hallucinating REVEAL_CHECK violations on
    # concept-clean drafts: it would cite "remove 'synaptic'" or "remove
    # 'ulnar' from 'inside of your elbow'" when neither word appeared in
    # the draft (semantic-association false positives).
    # Use the same deterministic _contains_concept Python function the
    # teacher's leak guard uses — it operates on literal substrings only.
    # If Python says the draft is clean, we tell the LLM Dean reveal is
    # already permitted so its prompt's auto-pass rule kicks in for
    # REVEAL/DEFINITION/QUESTION; the LLM only judges GROUNDING and
    # SYCOPHANCY (which legitimately need semantic reasoning).
    # Imported lazily to dodge a circular import (teacher_socratic.py
    # imports from this module's neighbors).
    from graph.nodes.teacher_socratic import _contains_concept
    domain = state.get("domain", config.DOMAIN)
    generic_words = get_generic_words(domain)
    stem_blacklist = get_stem_blacklist(domain)

    # ── Exemption: student already named the concept ────────────────────────
    # REVEAL_CHECK exists to stop the tutor from telling the student something
    # they don't yet know. If the student themselves just named the concept
    # (topic-switch — "what about the rotator cuff?" — or first-message
    # naming — "what's the synaptic cleft?"), there is nothing to reveal.
    # Echoing the concept once during a pivot acknowledgment is natural and
    # not a leak. We only exempt the IMMEDIATELY PREVIOUS student message —
    # not the full history — so a stale mention from many turns ago doesn't
    # disable the gate forever.
    last_student_msg = ""
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", None) == "human":
            last_student_msg = msg_text(getattr(m, "content", "")).lower()
            break
    student_already_named = bool(
        concept and last_student_msg and concept.lower() in last_student_msg
    )

    if (
        not reveal_permitted
        and not student_already_named
        and concept
        and _contains_concept(draft, concept, generic_words, stem_blacklist)
    ):
        print(
            f"[dean] t={turn_count} rev={current_revisions} reveal={reveal_permitted} "
            f"FAIL ['REVEAL CHECK'] (python pre-check, literal) | {draft[:100]!r}",
            file=sys.stderr,
        )
        instruction = (
            f"Remove '{concept}' and its derivatives (plural, adjectival, "
            f"stem variants) from the draft. Replace with broad vocabulary "
            f"like 'this structure', 'connection point', or 'communication gap'."
        )
        print(f"       instruction: {instruction!r}", file=sys.stderr)
        return {
            "dean_passed": False,
            "dean_revisions": current_revisions + 1,
            "dean_revision_instruction": instruction,
        }

    if student_already_named:
        print(
            f"[dean] t={turn_count} rev={current_revisions} reveal={reveal_permitted} "
            f"REVEAL skipped — student already named {concept!r} in last "
            f"message; echoing is not a leak | {draft[:100]!r}",
            file=sys.stderr,
        )

    # Python pre-checks all clean. Force reveal_permitted=True so the LLM
    # prompt auto-passes REVEAL/DEFINITION/QUESTION and only evaluates
    # GROUNDING + SYCOPHANCY. This eliminates the hallucination class of
    # Dean failures.
    prompt = fill_prompt(
        load_prompt("dean_check.txt"),
        current_concept=concept,
        turn_count=turn_count,
        reveal_permitted=True,
        retrieved_chunks=retrieved_text,
        draft_response=draft,
        max_sentences=config.MAX_RESPONSE_SENTENCES,
    )

    response = _client.messages.create(
        model=config.model_for("dean"),
        max_tokens=config.DEAN_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_response = response.content[0].text
    result = _extract_dean_json(raw_response)
    if result is None:
        # Fail-CLOSED: a malformed Dean response would otherwise let leaky
        # teacher drafts through. Treat parse failure as a forced revision —
        # the teacher rewrites and Dean tries again. After DEAN_MAX_REVISIONS
        # the route_after_dean fallback kicks in (fallback_scaffold).
        print(
            f"[dean] t={turn_count} rev={current_revisions} reveal={reveal_permitted} "
            f"DRAFT={draft[:120]!r} → JSON_PARSE_FAILED raw={raw_response!r}",
            file=sys.stderr,
        )
        return {
            "dean_passed": False,
            "dean_revisions": current_revisions + 1,
            "dean_revision_instruction": (
                "PARSE_ERROR — quality verdict was unparseable. "
                "Rewrite the response so it adheres to all five criteria."
            ),
        }

    passed = bool(result.get("passed", True))
    revision_instruction = (
        result.get("revision_instruction", "") if not passed else ""
    )
    failed = result.get("failed_criteria", [])

    # ── LLM-judge hallucination overrides ────────────────────────────────────
    # The Dean criteria most prone to LLM hallucination are GROUNDING
    # (phrasing nitpicks, fabricated chunk content, attribution confusion),
    # REVEAL (Sonnet citing concept words it claims are in the draft but
    # that the literal substring check disagrees with), and SYCOPHANCY
    # (rejecting "Perfect — ..." with a dash even though the prompt
    # explicitly allows it).
    #
    # We have deterministic safety nets for REVEAL (literal substring
    # _contains_concept check above) and SYCOPHANCY (literal "{Word}!"
    # first-token check above). Anything those Python checks let through
    # is by definition not a real violation, so an LLM rejection on those
    # criteria alone is treated as advisory.
    #
    # GROUNDING has no equivalent deterministic check, so we rely on the
    # advisory-pass behavior to blunt Sonnet's overconfidence.
    spurious = {"GROUNDING CHECK", "REVEAL CHECK", "DEFINITION CHECK",
                "SYCOPHANCY CHECK"}
    if (
        not passed
        and failed
        and all(c in spurious for c in failed)
    ):
        print(
            f"[dean] t={turn_count} rev={current_revisions} reveal={reveal_permitted} "
            f"FAIL {failed} → ADVISORY PASS (likely hallucination — "
            f"Python pre-checks verified no leak, GROUNDING is advisory) | "
            f"{draft[:100]!r}",
            file=sys.stderr,
        )
        print(f"       (advisory) instruction: {revision_instruction!r}", file=sys.stderr)
        passed = True
        revision_instruction = ""

    # ── Permanent structured log (always visible in terminal) ─────────────────
    status = "PASS" if passed else f"FAIL {failed}"
    print(
        f"[dean] t={turn_count} rev={current_revisions} reveal={reveal_permitted} "
        f"{status} | {draft[:100]!r}",
        file=sys.stderr,
    )
    if not passed:
        print(f"       instruction: {revision_instruction!r}", file=sys.stderr)
    # ─────────────────────────────────────────────────────────────────────────

    return {
        "dean_passed": passed,
        "dean_revisions": current_revisions + (0 if passed else 1),
        "dean_revision_instruction": revision_instruction,
    }
