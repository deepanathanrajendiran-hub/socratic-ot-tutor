"""
graph/nodes/hint_error_node.py

Hint generator for incorrect student answers (before the reveal gate).
Called when: classifier_output == "incorrect" AND turn_count < SOCRATIC_TURN_GATE.

Acknowledges the wrong attempt without sycophancy, provides a targeted
scaffold that nudges the student toward the correct answer.

The Dean node checks this draft before delivery (absolute rule).

Model: PRIMARY_MODEL (claude-sonnet-4-5)
Input:  current_concept, retrieved_chunks, turn_count, student_attempted,
        messages, domain, dean_revision_instruction
Output: state["draft_response"]
"""

import os

from anthropic import Anthropic

import config
from graph.state import GraphState
import re
import sys

_client = Anthropic()


def _load_prompt() -> str:
    path = os.path.join(config.PROMPTS_DIR, "hint_error.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _msg_text(content) -> str:
    """Safe text extraction — content may be str or list[dict] for multimodal."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def _format_messages(messages) -> str:
    """Format conversation history as plain text for the prompt."""
    if not messages:
        return "(no prior conversation)"
    lines = []
    for msg in messages:
        role = "Student" if msg.type == "human" else "Tutor"
        lines.append(f"{role}: {_msg_text(msg.content)}")
    return "\n".join(lines)


def _extract_last_student_message(messages) -> str:
    """Return the most recent human message — the wrong answer to hint against."""
    for msg in reversed(messages):
        if msg.type == "human":
            return _msg_text(msg.content)
    return "(no student message found)"


def _contains_concept(draft: str, concept: str) -> bool:
    """Return True if draft contains the concept word or obvious derivatives."""
    if not concept:
        return False
    draft_lower = draft.lower()
    concept_lower = concept.lower()
    if concept_lower in draft_lower:
        return True
    for word in concept_lower.split():
        if len(word) < 5:
            continue
        stem = word[: max(4, len(word) - 2)]
        if re.search(r"\b" + re.escape(stem), draft_lower):
            return True
    return False


def _should_reveal(state: GraphState) -> bool:
    if state.get("concept_mastered", False):
        return True
    if state.get("student_phase", "learning") != "learning":
        return True
    return state.get("turn_count", 0) >= config.SOCRATIC_TURN_GATE


def hint_error_node(state: GraphState) -> dict:
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
    reveal_permitted = _should_reveal(state)

    messages = state.get("messages", [])
    student_last_message = _extract_last_student_message(messages)
    messages_text = _format_messages(messages)

    classifier_output = state.get("classifier_output", "incorrect")
    student_mode = "idk" if classifier_output == "idk" else "incorrect"

    prompt = _load_prompt().format(
        domain_context=domain_ctx,
        current_concept=concept,
        retrieved_chunks=retrieved_text,
        student_last_message=student_last_message,
        student_mode=student_mode,
        turn_count=turn_count,
        reveal_permitted=reveal_permitted,
        max_sentences=config.MAX_RESPONSE_SENTENCES,
        messages=messages_text,
    )

    revision_instruction = state.get("dean_revision_instruction", "")
    revision_system = (
        f"REVISION REQUIRED: {revision_instruction}\n"
        "Fix the issue above and rewrite the response."
        if revision_instruction
        else None
    )

    api_kwargs: dict = dict(
        model=config.PRIMARY_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    if revision_system:
        api_kwargs["system"] = revision_system

    response = _client.messages.create(**api_kwargs)
    draft = response.content[0].text.strip()

    # ── Concept-leak guard ────────────────────────────────────────────────────
    # idk mode quotes retrieved chunks verbatim — those chunks may contain the
    # concept name. Catch it here before Dean sees it, same pattern as teacher.
    MAX_LEAK_RETRIES = 2
    if not reveal_permitted and concept:
        for attempt in range(MAX_LEAK_RETRIES):
            if not _contains_concept(draft, concept):
                break
            forbidden: list[str] = [concept, f"{concept}s"]
            for word in concept.split():
                if len(word) >= 5:
                    stem = word[: max(4, len(word) - 2)]
                    forbidden += [word, f"{word}s", f"{stem}ic", f"{stem}al"]
            forbidden_str = ", ".join(f"'{w}'" for w in sorted(set(forbidden)))
            leak_instruction = (
                f"CRITICAL: The word '{concept}' and ALL its forms "
                f"({forbidden_str}) are STRICTLY FORBIDDEN — do NOT use them "
                "anywhere, not even when paraphrasing the textbook. "
                "Replace with: 'this structure', 'the connection point', "
                "'the specialized gap', 'where nerve meets muscle'."
            )
            combined = (
                f"{revision_system}\n\n{leak_instruction}"
                if revision_system
                else leak_instruction
            )
            retry = _client.messages.create(
                model=config.PRIMARY_MODEL,
                max_tokens=500,
                system=combined,
                messages=[{"role": "user", "content": prompt}],
            )
            new_draft = retry.content[0].text.strip()
            print(
                f"[hint] leak_retry attempt={attempt + 1}: concept='{concept}' "
                f"| {new_draft[:80]!r}",
                file=sys.stderr,
            )
            draft = new_draft

        if _contains_concept(draft, concept):
            for word in [concept] + concept.split():
                if len(word) < 4:
                    continue
                stem = word[: max(4, len(word) - 2)]
                draft = re.sub(
                    r"\b" + re.escape(stem) + r"\w*",
                    "[this structure]",
                    draft,
                    flags=re.IGNORECASE,
                )
            print(f"[hint] deterministic_strip: {draft[:80]!r}", file=sys.stderr)
    # ─────────────────────────────────────────────────────────────────────────

    return {"draft_response": draft, "draft_source_node": "hint_error_node"}
