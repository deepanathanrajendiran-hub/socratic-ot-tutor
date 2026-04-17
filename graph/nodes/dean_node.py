"""
graph/nodes/dean_node.py

Quality controller. Checks draft_response against 6 criteria before delivery.
Called on EVERY teacher/generation response — absolute rule from CLAUDE.md.

6 criteria (from prompts/dean_check.txt):
  1. REVEAL CHECK   — concept not named when reveal_permitted is False
  2. DEFINITION CHECK — no "X is defined as..." when not permitted
  3. GROUNDING CHECK  — every claim traceable to retrieved chunks
  4. QUESTION CHECK   — draft ends with 1-2 questions
  5. LENGTH CHECK     — max MAX_RESPONSE_SENTENCES before the question
  6. SYCOPHANCY CHECK — no praise opener, no contradiction of locked_answer

Model: PRIMARY_MODEL (claude-sonnet-4-5)
Input:  draft_response, current_concept, turn_count, student_attempted,
        retrieved_chunks, dean_revisions
Output: dean_passed (bool), dean_revisions (int), dean_revision_instruction (str)
"""

import json
import os

from anthropic import Anthropic

import config
from graph.state import GraphState
import sys

_client = Anthropic()


def _load_prompt() -> str:
    path = os.path.join(config.PROMPTS_DIR, "dean_check.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _should_reveal(state: GraphState) -> bool:
    """Reveal is permitted when any of these is true:
    1. Mastery confirmed (step_advancer set concept_mastered=True).
    2. Post-mastery phase — clinical_question, topic_choice, etc.
       (student_phase != "learning" means we are past the reveal gate).
    3. Past the Socratic turn gate (teach_node path for struggling students).
    """
    if state.get("concept_mastered", False):
        return True
    if state.get("student_phase", "learning") != "learning":
        return True
    return state.get("turn_count", 0) >= config.SOCRATIC_TURN_GATE



def _fill_prompt(template: str, **kwargs) -> str:
    """Replace named placeholders without str.format() — safe for prompts that
    contain literal JSON braces (e.g. dean_check.txt, synthesis_assessor.txt).
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


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

    prompt = _fill_prompt(
        _load_prompt(),
        current_concept=concept,
        turn_count=turn_count,
        reveal_permitted=reveal_permitted,
        retrieved_chunks=retrieved_text,
        draft_response=draft,
        max_sentences=config.MAX_RESPONSE_SENTENCES,
    )

    response = _client.messages.create(
        model=config.PRIMARY_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Extract the JSON object — handles code fences and trailing text
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fail-open: unparseable verdict → pass the draft rather than blocking
        print(
            f"[dean] t={turn_count} rev={current_revisions} reveal={reveal_permitted} "
            f"DRAFT={draft[:120]!r} → JSON_PARSE_FAILED raw={raw!r}",
            file=sys.stderr,
        )
        return {
            "dean_passed": True,
            "dean_revisions": current_revisions,
            "dean_revision_instruction": "",
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
