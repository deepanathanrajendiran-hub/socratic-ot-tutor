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

from graph._llm_client import Anthropic

import config
from graph.state import GraphState
import re
import sys
from graph.nodes._helpers import (
    get_generic_words,
    get_stem_blacklist,
    load_prompt,
    log_thinking,
    msg_text,
    strip_thinking_block,
)

_client = Anthropic()


def _format_messages(messages) -> str:
    """Format conversation history as plain text for the prompt."""
    if not messages:
        return "(no prior conversation)"
    lines = []
    for msg in messages:
        role = "Student" if msg.type == "human" else "Tutor"
        lines.append(f"{role}: {msg_text(msg.content)}")
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


# ── Meta-language strip ─────────────────────────────────────────────────────
# Sonnet sometimes leaks system-side language ("the retrieved content
# mentions...", "Now that we're at turn 6...") into the student-facing
# response, even with explicit prompt rules forbidding it. These patterns
# are deterministically removed AFTER generation as a defense in depth.
# Each entry is (pattern, replacement). Most strip to "", but the linker
# pattern strips to ". " so the two clauses on either side become
# separate sentences instead of jamming together.
_META_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "the retrieved content/chunks/text mentions/notes/describes/states ..."
    (re.compile(
        r"(?:the\s+)?retrieved\s+(?:content|chunks?|text|material|"
        r"passages?|facts?)\s+(?:mention(?:s|ed)?|note(?:s|d)?|"
        r"describe(?:s|d)?|state(?:s|d)?|say(?:s|ing)?|"
        r"indicate(?:s|d)?|show(?:s|n|ed)?|tell(?:s|ing)?)\s+"
        r"(?:(?:that|how|us)\s+)?",
        re.IGNORECASE,
    ), ""),
    # "according to the textbook/content/source/material/passage/chunk"
    (re.compile(
        r"\baccording\s+to\s+(?:what\s+i\s+(?:have|was\s+given)|"
        r"the\s+(?:retrieved\s+)?(?:textbook|content|passage|source|"
        r"material|chunk))[,\s]*",
        re.IGNORECASE,
    ), ""),
    # "the textbook/content/source/passage/material says/mentions/etc"
    (re.compile(
        r"\b(?:the\s+)?(?:textbook|content|passage|source|material)\s+"
        r"(?:says?|notes?|describes?|mentions?|states?|indicates?|"
        r"shows?|tells?\s+us)\s+(?:that\s+)?",
        re.IGNORECASE,
    ), ""),
    # "based on (what I have | what I was given | the retrieved | the provided)"
    (re.compile(
        r"\bbased\s+on\s+(?:what\s+i\s+(?:have|was\s+given)|"
        r"the\s+(?:retrieved|provided|material))[,\s]*",
        re.IGNORECASE,
    ), ""),
    # "my knowledge base / training data / database / reference material"
    (re.compile(
        r"\b(?:my|the)\s+(?:knowledge\s+base|training\s+data|"
        r"database|reference\s+material)[,\s]*",
        re.IGNORECASE,
    ), ""),
    # "now that we're at turn N" / "we're at turn N" / "this is turn N"
    (re.compile(
        r"\b(?:now\s+that\s+we'?re|we'?re|this\s+is)\s+"
        r"(?:now\s+)?at\s+turn\s+\d+[,\s]*",
        re.IGNORECASE,
    ), ""),
    (re.compile(r"\bat\s+turn\s+\d+[,\s]*", re.IGNORECASE), ""),
    (re.compile(
        r"\bsince\s+(?:we'?re\s+)?(?:now\s+)?(?:at|on)\s+turn\s+\d+[,\s]*",
        re.IGNORECASE,
    ), ""),
    # Linker pronouns referring back to a just-stripped source noun:
    # ", and it also notes that ..." → ". " (becomes a fresh sentence).
    (re.compile(
        r"[,;]\s+(?:and\s+)?(?:it|this|that)\s+(?:also\s+)?"
        r"(?:notes?|mentions?|describes?|states?|says|indicates?|"
        r"points?\s+out|tells?\s+us)\s+(?:that\s+)?",
        re.IGNORECASE,
    ), ". "),
]


def _strip_meta_language(draft: str) -> str:
    """Remove system meta-references that break the tutor persona.

    Targets phrasings the LLM uses that expose the underlying retrieval
    system or internal turn counter. Replaces matches with empty string
    (or ". " for clause-linker patterns so the two halves stay readable
    as separate sentences), then cleans up orphan punctuation and
    recapitalizes any sentence opener the strip exposed.
    """
    if not draft:
        return ""
    cleaned = draft
    for pat, repl in _META_PATTERNS:
        cleaned = pat.sub(repl, cleaned)
    # Cleanup
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"^[,.\s]+", "", cleaned)
    cleaned = cleaned.strip()
    # Recapitalize after any period+space (in case the strip exposed a
    # mid-sentence lowercase opener) and at the very start.
    cleaned = re.sub(
        r"([.!?]\s+)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        cleaned,
    )
    if cleaned and cleaned[0].isalpha() and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _contains_concept(
    draft: str,
    concept: str,
    generic_words: set[str] | None = None,
    stem_blacklist: set[str] | None = None,
) -> bool:
    """Return True if draft contains the concept word or obvious derivatives.

    Pipeline:
      1. Exact full-phrase match — catches the canonical concept name.
      2. Per-word check, skipping `generic_words` (so "nerve" alone never
         triggers in a draft about the ulnar nerve concept):
         a. Exact word match for words ≥4 chars — catches "ulnar" in
            "medial (ulnar) side".
         b. Stem-prefix match for words ≥5 chars — catches morphological
            variants ("synapse" → "synaptic"; "ulnar" → "ulnaris").
            Stems landing in `stem_blacklist` are skipped.

    `generic_words` and `stem_blacklist` default to the active domain's
    sets from config.DOMAIN_CONFIG via _helpers.get_*. Callers in other
    domains (or tests) can pass explicit sets.
    """
    if not concept:
        return False
    if generic_words is None:
        generic_words = get_generic_words()
    if stem_blacklist is None:
        stem_blacklist = get_stem_blacklist()

    draft_lower = draft.lower()
    concept_lower = concept.lower()

    if concept_lower in draft_lower:
        return True

    for word in concept_lower.split():
        if word in generic_words:
            continue
        if len(word) >= 4 and re.search(
            r"\b" + re.escape(word) + r"\b", draft_lower
        ):
            return True
        if len(word) >= 5:
            stem = word[: max(4, len(word) - 2)]
            if stem in stem_blacklist:
                continue
            if re.search(r"\b" + re.escape(stem), draft_lower):
                return True

    return False

from graph.edges import should_reveal as _should_reveal  # canonical reveal gate


def teacher_socratic(state: GraphState) -> dict:
    domain = state.get("domain", config.DOMAIN)
    domain_ctx = config.DOMAIN_CONFIG.get(domain, {}).get(
        "system_context", domain
    )
    generic_words = get_generic_words(domain)
    stem_blacklist = get_stem_blacklist(domain)

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

    prompt = load_prompt("teacher_socratic.txt").format(
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
        max_tokens=config.TEACHER_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    if revision_system:
        api_kwargs["system"] = revision_system

    response = _client.messages.create(**api_kwargs)
    raw = response.content[0].text
    draft, thinking = strip_thinking_block(raw)
    log_thinking(
        thinking,
        node="teacher_socratic",
        session_id=state.get("session_id", ""),
        turn_count=turn_count,
        concept=concept,
        classifier_output=state.get("classifier_output", ""),
        reveal_permitted=reveal_permitted,
    )

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
            max_tokens=config.TEACHER_MAX_TOKENS,
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
            if not _contains_concept(draft, concept, generic_words, stem_blacklist):
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
                max_tokens=config.TEACHER_MAX_TOKENS,
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

        # Deterministic strip if LLM still didn't comply.
        # Skip generic_words — replacing every "nerve" / "lateral" mention
        # would mangle a draft that's already concept-clean elsewhere.
        if _contains_concept(draft, concept, generic_words, stem_blacklist):
            stripped = draft
            stripped = re.sub(
                re.escape(concept), "[the target structure]", stripped,
                flags=re.IGNORECASE,
            )
            for word in concept.split():
                word_l = word.lower()
                if word_l in generic_words or len(word_l) < 4:
                    continue
                if len(word_l) >= 5:
                    stem = word_l[: max(4, len(word_l) - 2)]
                    pattern = r"\b" + re.escape(stem) + r"\w*"
                else:
                    pattern = r"\b" + re.escape(word_l) + r"\b"
                stripped = re.sub(
                    pattern, "[the target structure]", stripped,
                    flags=re.IGNORECASE,
                )
            print(
                f"[teacher] deterministic_strip: concept='{concept}' "
                f"| {stripped[:80]!r}",
                file=sys.stderr,
            )
            draft = stripped
    # ─────────────────────────────────────────────────────────────────────────

    # ── Meta-language strip ──────────────────────────────────────────────────
    # Always run, regardless of reveal_permitted — these phrasings break
    # the tutor persona at any turn. Independent of the concept-leak guard.
    pre_strip_len = len(draft)
    draft = _strip_meta_language(draft)
    if len(draft) != pre_strip_len:
        print(
            f"[teacher] meta_strip: removed {pre_strip_len - len(draft)} chars "
            f"| {draft[:80]!r}",
            file=sys.stderr,
        )
    # ─────────────────────────────────────────────────────────────────────────

    return {"draft_response": draft, "draft_source_node": "teacher_socratic"}
