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

    if not reveal_permitted and concept and _contains_concept(
        draft, concept, generic_words, stem_blacklist,
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
        model=config.PRIMARY_MODEL,
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
