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
from graph import _stream

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


_THINKING_CLOSE_RE = re.compile(r"</\s*thinking\s*>", re.IGNORECASE)


def _stream_completion(api_kwargs: dict, emit_tokens: bool) -> str:
    """Run a streaming Anthropic completion, returning the full raw text.

    When `emit_tokens` is True AND a stream sink is installed by the API
    layer, calls _stream.emit_token for each chunk of student-visible text
    (i.e. text after the </thinking> close tag). Anything inside a leading
    <thinking>...</thinking> block is suppressed from live emission and
    only ends up in the post-stream raw text for log_thinking.

    To give the user feedback during the (often multi-second) thinking
    suppression window, we also emit a "thinking" step.start as soon as
    the first delta arrives that contains an opening <thinking> tag, and
    a "thinking" step.done when </thinking> closes. The frontend turns
    the typing-bubble label into "Thinking through your question…" while
    that pseudo-step is active, then flips to "Writing response…" when
    the visible tokens start flowing.

    When `emit_tokens` is False, the call still streams (so we can use
    the same SDK path) but no SSE frames are pushed — used for length
    retries that overwrite the streamed bubble via emit_replace at the end.

    Returns the full raw text exactly as it would have appeared from a
    non-streaming `messages.create` call. Callers run strip_thinking_block
    on it as before.
    """
    raw = ""
    started_emitting = False
    thinking_announced = False  # have we emitted a thinking step.start yet?

    with _client.messages.stream(**api_kwargs) as stream:
        for delta in stream.text_stream:
            raw += delta
            if not emit_tokens:
                continue
            if started_emitting:
                _stream.emit_token(delta)
                continue
            # Still inside (or before) the <thinking> block. Surface the
            # thinking pseudo-step as soon as we know the model has begun
            # producing private CoT, so the user sees a status change
            # instead of a static "Writing response…" label.
            if not thinking_announced and "<thinking" in raw.lower():
                _stream.emit_step("thinking", "start")
                thinking_announced = True
            # Look for the closing tag in the accumulated buffer; once
            # found, announce thinking done, emit any text past it, and
            # start emitting subsequent deltas directly.
            m = _THINKING_CLOSE_RE.search(raw)
            if m:
                if thinking_announced:
                    _stream.emit_step("thinking", "done")
                started_emitting = True
                tail = raw[m.end():].lstrip()
                if tail:
                    _stream.emit_token(tail)

    # If we announced thinking but never saw a close tag (unusual — model
    # ignored the prompt format), close out the step so the frontend
    # doesn't get stuck on the "thinking" label.
    if thinking_announced and not started_emitting:
        _stream.emit_step("thinking", "done")

    return raw


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

    # Discovery-mode branch. When the student named the concept upfront
    # (manager_agent set discovery_target="function"), load the
    # function-discovery prompt — it asks about what the concept DOES,
    # not what it IS. Empty/"name" → existing name-discovery prompt.
    discovery_target = (state.get("discovery_target") or "name").strip()
    prompt_file = (
        "teacher_socratic_function.txt"
        if discovery_target == "function"
        else "teacher_socratic.txt"
    )
    prompt = load_prompt(prompt_file).format(
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

    # System prompt selection. Name-mode uses the original
    # teacher_socratic_system.txt which forbids stating "the target
    # concept" (correct for name-discovery). That same forbid-pattern
    # makes Sonnet emit "[the target structure]" placeholders in
    # function-mode replies — exactly what the student called out
    # 2026-05-03. In function mode we drop the system prompt entirely;
    # the function-mode user prompt has the full rule set inline.
    system_blocks: list[dict]
    if discovery_target == "function":
        system_blocks = []
    else:
        # The teacher's persona/rules/format prompt is identical across
        # every name-mode turn (~80 lines, ~1.5K tokens). System message
        # with `cache_control: ephemeral` so Anthropic memos the prefix
        # and shaves ~30-40% off input-processing latency on turn 2+
        # within a 5-minute window.
        system_blocks = [
            {
                "type": "text",
                "text": load_prompt("teacher_socratic_system.txt"),
                "cache_control": {"type": "ephemeral"},
            },
        ]
    if revision_system:
        # Revision instructions vary per call; append uncached.
        system_blocks.append({"type": "text", "text": revision_system})

    api_kwargs: dict = dict(
        model=config.PRIMARY_MODEL,
        max_tokens=config.TEACHER_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    if system_blocks:
        api_kwargs["system"] = system_blocks

    # Stream the primary call so the FastAPI /chat handler can forward
    # token deltas to the browser as they arrive. On Dean revisions
    # (revision_system set) we suppress live emission — the user already
    # saw the original streamed draft; revising silently and overwriting
    # with the API's final replace event avoids confusing flicker.
    emit_live = not revision_system and _stream.has_sink()
    raw = _stream_completion(api_kwargs, emit_tokens=emit_live)
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
    # One retry with a tight system message if over the limit. The retry
    # streams silently and overwrites the live bubble via emit_replace.
    preamble_count = _count_preamble_sentences(draft)
    if preamble_count > config.MAX_RESPONSE_SENTENCES:
        length_instruction = (
            f"Your response had {preamble_count} sentences before the question "
            f"(limit is {config.MAX_RESPONSE_SENTENCES}). "
            "Rewrite with at most 1 brief sentence of context, then your Socratic "
            "question. Do NOT open with meta-commentary about the teaching approach."
        )
        # Reuse the cached static prefix; append the length instruction as
        # an uncached system block alongside any active revision instruction.
        retry_system_blocks: list[dict] = [
            {
                "type": "text",
                "text": load_prompt("teacher_socratic_system.txt"),
                "cache_control": {"type": "ephemeral"},
            },
        ]
        if revision_system:
            retry_system_blocks.append({"type": "text", "text": revision_system})
        retry_system_blocks.append({"type": "text", "text": length_instruction})

        new_raw = _stream_completion(
            dict(
                model=config.model_for("teacher"),
                max_tokens=config.TEACHER_MAX_TOKENS,
                system=retry_system_blocks,
                messages=[{"role": "user", "content": prompt}],
            ),
            emit_tokens=False,
        )
        new_draft, _ = strip_thinking_block(new_raw)
        print(
            f"[teacher] length_retry: preamble={preamble_count} > "
            f"{config.MAX_RESPONSE_SENTENCES} | {new_draft[:80]!r}",
            file=sys.stderr,
        )
        draft = new_draft
        if emit_live:
            _stream.emit_replace(draft)
    # ─────────────────────────────────────────────────────────────────────────

    # ── Concept-leak guard ────────────────────────────────────────────────────
    # Streaming-path simplification: the original code retried the LLM up
    # to 2 times when the draft still contained the locked concept. Each
    # retry was a full Sonnet round-trip, blocking TTFT. With streaming we
    # cannot rerun the LLM without flicker, so we drop the retry and rely
    # on (a) the deterministic strip below as the final guarantee and
    # (b) the Dean node's REVEAL_CHECK (which still runs after this node)
    # to catch any leak that survives. If the strip fires, the API's
    # final replace event overwrites the streamed bubble.
    if not reveal_permitted and concept and _contains_concept(
        draft, concept, generic_words, stem_blacklist,
    ):
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

    # Deterministic placeholder-leak guard. In function-mode Sonnet
    # sometimes emits bracketed template phrases like "[the target
    # structure]" / "[the structure]" / "[the concept]" — likely
    # priming from the cached name-mode system prompt's "the target
    # concept" wording. Substitute any such bracketed placeholder with
    # the actual concept name. Idempotent and safe in name mode too
    # (those bracketed forms are never desired in either mode).
    if concept and draft:
        placeholder_re = re.compile(
            r"\[\s*(?:(?:the|this|that)\s+)?"
            r"(?:target\s+)?"
            r"(?:structure|concept|region|area)"
            r"\s*\]",
            re.IGNORECASE,
        )
        if placeholder_re.search(draft):
            new_draft = placeholder_re.sub(concept, draft)
            print(
                f"[teacher_socratic] placeholder-leak strip: replaced "
                f"bracketed template with {concept!r}",
                file=sys.stderr,
            )
            draft = new_draft

    return {"draft_response": draft, "draft_source_node": "teacher_socratic"}
