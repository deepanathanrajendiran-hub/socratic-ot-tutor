"""
graph/nodes/teacher_socratic.py

Main generation node. Produces a Socratic tutoring draft written to
state["draft_response"]. The Dean node checks it before delivery.

This node is called:
  - On the primary teaching path (after retrieval, when student needs guidance)
  - On the revision path (when Dean rejects — dean_revisions > 0 in state)

Model: PRIMARY_MODEL (claude-sonnet-4-5)
Input:  current_concept, retrieved_chunks, turn_count, student_attempted,
        weak_topics, messages, domain, question_bank (optional file)
Output: state["draft_response"]
"""

import json
import os

from anthropic import Anthropic

import config
from graph.state import GraphState
import re
import sys

_client = Anthropic()


def _load_prompt() -> str:
    path = os.path.join(config.PROMPTS_DIR, "teacher_socratic.txt")
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


def _load_question_bank(concept: str) -> str:
    """Load pre-generated Socratic questions for this concept if available.
    Falls back to '(none)' — question_bank_builder.py is deferred to Step 9.
    """
    if not concept:
        return "(none)"
    safe = concept.lower().replace(" ", "_").replace("/", "_")
    path = os.path.join(config.QUESTION_BANK_DIR, f"{safe}.json")
    if not os.path.exists(path):
        return "(none)"
    try:
        with open(path, encoding="utf-8") as f:
            questions = json.load(f)
        if isinstance(questions, list):
            return "\n".join(
                f"- {q}" for q in questions[: config.QUESTION_BANK_PER_CONCEPT]
            )
    except (json.JSONDecodeError, OSError):
        pass
    return "(none)"


def _count_preamble_sentences(draft: str) -> int:
    """Count prose sentences that appear before the first '?' in the draft.

    Skips blank lines and A)/B)/C) choice labels.
    Used by the length guard to enforce MAX_RESPONSE_SENTENCES without
    spending a Dean revision slot on a mechanical counting task.
    """
    first_q = draft.find("?")
    preamble = draft[:first_q] if first_q != -1 else draft

    # Split on sentence-ending punctuation followed by whitespace
    segments = re.split(r"(?<=[.!])\s+", preamble)
    count = 0
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if re.match(r"^[A-Ca-c][).]", seg):  # skip choice labels
            continue
        count += 1
    return count


def _contains_concept(draft: str, concept: str) -> bool:
    """Return True if draft contains the concept word or obvious derivatives.

    Catches both exact matches and morphological variants:
      "synapse"  →  matches "synapse", "synapses", "synaptic", "synaptosome"
      "reflex"   →  matches "reflex", "reflexes", "reflexive"
      "neuron"   →  matches "neuron", "neurons", "neuronal"

    Multi-word concepts: each word >= 5 chars is checked independently.
    """
    if not concept:
        return False
    draft_lower = draft.lower()
    concept_lower = concept.lower()

    # 1. Exact phrase match
    if concept_lower in draft_lower:
        return True

    # 2. Per-word stem check (covers plurals and adjectival derivatives)
    for word in concept_lower.split():
        if len(word) < 5:
            continue
        # Stem = first max(4, len-2) chars — "synapse"→"synap", "reflex"→"refl"
        stem = word[: max(4, len(word) - 2)]
        if re.search(r'\b' + re.escape(stem), draft_lower):
            return True

    return False

def _should_reveal(state: GraphState) -> bool:
    if state.get("concept_mastered", False):
        return True
    if state.get("student_phase", "learning") != "learning":
        return True
    return state.get("turn_count", 0) >= config.SOCRATIC_TURN_GATE


def teacher_socratic(state: GraphState) -> dict:
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

    weak = state.get("weak_topics", [])
    weak_text = ", ".join(weak) if weak else "(none)"

    question_bank = _load_question_bank(concept)
    messages_text = _format_messages(state.get("messages", []))

    prompt = _load_prompt().format(
        domain_context=domain_ctx,
        current_concept=concept,
        retrieved_chunks=retrieved_text,
        turn_count=turn_count,
        reveal_permitted=reveal_permitted,
        max_sentences=config.MAX_RESPONSE_SENTENCES,
        question_bank=question_bank,
        weak_topics=weak_text,
        messages=messages_text,
    )

    # On revision pass, prepend the Dean's instruction as a system message
    # so the teacher knows exactly what to fix without changing the prompt file.
    revision_instruction = state.get("dean_revision_instruction", "")
    revision_system = (
        f"REVISION REQUIRED: {revision_instruction}\n"
        "Fix the issue above and rewrite the response."
        if revision_instruction
        else None
    )

    api_kwargs: dict = dict(
        model=config.PRIMARY_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    if revision_system:
        api_kwargs["system"] = revision_system

    response = _client.messages.create(**api_kwargs)
    draft = response.content[0].text.strip()

    # ── Length guard (replaces Dean criterion 5) ──────────────────────────────
    # Count prose sentences before the first "?" in Python — no revision slot
    # consumed, no LLM tokens spent on a mechanical counting task.
    # One retry with a tight system message if over the limit.
    preamble_count = _count_preamble_sentences(draft)
    if preamble_count > config.MAX_RESPONSE_SENTENCES:
        length_instruction = (
            f"Your response had {preamble_count} sentences before the question "
            f"(limit is {config.MAX_RESPONSE_SENTENCES}). "
            "Rewrite with at most 1 brief sentence of context, then your Socratic "
            "question. Do NOT open with meta-commentary about the teaching approach."
        )
        combined = (
            f"{revision_system}\n\n{length_instruction}"
            if revision_system
            else length_instruction
        )
        length_response = _client.messages.create(
            model=config.PRIMARY_MODEL,
            max_tokens=600,
            system=combined,
            messages=[{"role": "user", "content": prompt}],
        )
        new_draft = length_response.content[0].text.strip()
        print(
            f"[teacher] length_retry: preamble={preamble_count} > "
            f"{config.MAX_RESPONSE_SENTENCES} | {new_draft[:80]!r}",
            file=sys.stderr,
        )
        draft = new_draft
    # ─────────────────────────────────────────────────────────────────────────

    # ── Concept-leak guard ────────────────────────────────────────────────────
    # If reveal is not permitted and the draft still contains the concept word
    # (or a derivative), retry with an explicit forbidden-word system message.
    # Retries are combined with any active revision instruction so that both
    # constraints are honoured simultaneously.
    # After MAX_LEAK_RETRIES attempts, strip deterministically as last resort.
    MAX_LEAK_RETRIES = 2
    if not reveal_permitted and concept:
        for attempt in range(MAX_LEAK_RETRIES):
            if not _contains_concept(draft, concept):
                break

            # Build forbidden forms (handles multi-word concepts correctly)
            forbidden: list[str] = [concept, f"{concept}s"]
            for word in concept.split():
                if len(word) >= 5:
                    stem = word[: max(4, len(word) - 2)]
                    forbidden += [word, f"{word}s", f"{stem}ic", f"{stem}al"]
            forbidden_str = ", ".join(f"'{w}'" for w in sorted(set(forbidden)))

            leak_instruction = (
                f"CRITICAL: The word '{concept}' and ALL its forms "
                f"({forbidden_str}) are STRICTLY FORBIDDEN — do NOT use them "
                "anywhere in your response, not even inside a question. "
                "Replace with only broad process vocabulary: "
                "'the connection', 'where nerve meets muscle', 'the gap', "
                "'the signal crossing point', 'the communication interface'."
            )
            combined_system = (
                f"{revision_system}\n\n{leak_instruction}"
                if revision_system
                else leak_instruction
            )

            retry_response = _client.messages.create(
                model=config.PRIMARY_MODEL,
                max_tokens=600,
                system=combined_system,
                messages=[{"role": "user", "content": prompt}],
            )
            new_draft = retry_response.content[0].text.strip()
            print(
                f"[teacher] leak_retry attempt={attempt + 1}: concept='{concept}' "
                f"| old={draft[:60]!r} → new={new_draft[:60]!r}",
                file=sys.stderr,
            )
            draft = new_draft

        # Deterministic strip if LLM still didn't comply
        if _contains_concept(draft, concept):
            stripped = draft
            stripped = re.sub(
                re.escape(concept), "[the target structure]", stripped,
                flags=re.IGNORECASE,
            )
            for word in concept.split():
                if len(word) < 5:
                    continue
                stem = word[: max(4, len(word) - 2)]
                stripped = re.sub(
                    r"\b" + re.escape(stem) + r"\w*",
                    "[the target structure]",
                    stripped,
                    flags=re.IGNORECASE,
                )
            print(
                f"[teacher] deterministic_strip: concept='{concept}' "
                f"| {stripped[:80]!r}",
                file=sys.stderr,
            )
            draft = stripped
    # ─────────────────────────────────────────────────────────────────────────

    return {"draft_response": draft, "draft_source_node": "teacher_socratic"}
