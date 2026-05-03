"""
graph/nodes/response_classifier.py

Classifies the student's latest message into exactly one label:
  irrelevant | questioning | incorrect | correct | idk

Model: FAST_MODEL (claude-haiku-4-5) — single-word output, max_tokens=config.CLASSIFIER_MAX_TOKENS.
Input:  state["current_concept"], state["messages"]
Output: state["classifier_output"]
"""

import difflib
import re
import sys

from graph._llm_client import Anthropic

import config
from graph.state import GraphState
from graph.nodes._helpers import get_generic_words, load_prompt, msg_text


# ── Misspelling helpers ───────────────────────────────────────────────────────
# Threshold tuning: 0.82 still catches the common typos
# ("synapis"/"synaps"/"sinapse" → "synapse" at 0.85/0.92/0.83;
#  "cerebelum"/"cerebellem" → "cerebellum" at 0.95/0.90;
#  "brachail" → "brachial" at 0.875) while leaving extra headroom over
# wrong-structure near-misses ("cerebrum" vs "cerebellum" 0.78). One
# trade-off: 4-or-5-letter typos with a single vowel swap slip through
# as "incorrect" (e.g. "ulner" vs "ulnar" lands at 0.80) — the student
# then takes the normal Socratic re-engage path. Tighten further to
# 0.85 if you ever observe a false-positive in the wild.
_MISSPELL_THRESHOLD = 0.82


def _is_misspelled_concept(student_message: str, concept: str) -> bool:
    """Did the student type a recognizably-misspelled form of `concept`?
    Conservative fuzzy match: at least one student token is ≥ THRESHOLD
    similar to a discriminating concept token (or the whole concept
    if it's a single word). Used as a Python safety net so an honest
    typo doesn't get downgraded to "incorrect" by Haiku.
    """
    if not student_message or not concept:
        return False
    student_l = student_message.lower()
    concept_l = concept.lower()
    if concept_l in student_l:
        return False  # spelled correctly — not "misspelled"
    student_tokens = re.findall(r"[a-z]+", student_l)
    if not student_tokens:
        return False
    student_set = set(student_tokens)
    # Only fuzzy-match the discriminating concept tokens — skip tokens
    # the student has already typed verbatim. Without this, a multi-word
    # concept like "ulnar nerve" would false-positive on "median nerve"
    # (the shared "nerve" alone scores 1.0 against itself).
    concept_tokens = [
        t for t in re.findall(r"[a-z]+", concept_l)
        if len(t) >= 5 and t not in student_set
    ]
    if not concept_tokens:
        return False
    for st in student_tokens:
        if len(st) < 4:
            continue
        for ct in concept_tokens:
            ratio = difflib.SequenceMatcher(None, st, ct).ratio()
            if ratio >= _MISSPELL_THRESHOLD:
                return True
    return False

_client = Anthropic()


# ── Rule-based idk pre-classifier ─────────────────────────────────────────────
# Catches obvious idk phrases before the LLM call. Closes the haiku
# misclassification gap (idk often labelled as "incorrect") and saves a
# round-trip on the most common short-circuit. The LLM still runs for
# everything that doesn't match these patterns.
_IDK_PATTERNS = [
    r"\bi\s+don'?t\s+know\b",
    r"\bi\s+do\s+not\s+know\b",
    r"\bi\s+have\s+no\s+idea\b",
    r"\bno\s+idea\b",
    r"\bno\s+clue\b",
    r"\bnot\s+sure\b",
    r"\bstill\s+(?:no|don'?t|dont|nothing)\b",
    r"\bgive\s+up\b",
    r"\bgive\s+me\s+the\s+answer\b",
    r"\bjust\s+tell\s+(?:me|us)\b",
    r"\btell\s+me\s+already\b",
    r"\b(?:idk|dunno)\b",
    r"\bi\s+don'?t\s+(?:get|understand)\s+(?:it|this)\b",
    r"\bskip\s+(?:this|it)\b",
]
_IDK_REGEX = re.compile("|".join(_IDK_PATTERNS), re.IGNORECASE)


def _detect_idk(message: str) -> bool:
    """Return True if the message is an obvious 'I don't know' / give-up phrase."""
    if not message or not message.strip():
        return False
    return bool(_IDK_REGEX.search(message))


# ── correct-label defensive guard ───────────────────────────────────────────
# Haiku occasionally over-classifies a wrong-entity guess ("Is it the median
# nerve?") as "correct" when current_concept is "ulnar nerve" — because the
# message has the SHAPE of a correct answer (names a structure) without
# Haiku actually verifying that the named entity matches the concept. This
# is documented in evaluation/results — Exp D classifier accuracy ~75%.
# Programmatic guard: if label is "correct" but neither the full concept
# phrase nor a discriminating-word stem appears in the student's message,
# override to "incorrect" so the flow goes to hint_error_node instead of
# step_advancer (which would falsely confirm mastery).


def _verify_correct_label(
    label: str,
    student_message: str,
    concept: str,
    generic_words: set[str] | None = None,
) -> str:
    """Override 'correct' to 'incorrect' when concept is absent from message.

    Pass-through for non-correct labels and for empty inputs (defensive —
    don't override on missing data).

    Match logic (any of the following → keep "correct"):
      1. Full concept phrase appears in message (case-insensitive).
      2. A discriminating concept word's stem appears in message.
         "Discriminating" = not in `generic_words` and len ≥ 5.
         Stem = word[:max(4, len(word)-2)] — same convention as the
         leak-detection in teacher_socratic.
    Otherwise → return "incorrect".

    `generic_words` defaults to the active domain's set from
    config.DOMAIN_CONFIG via _helpers.get_generic_words().
    """
    if label != "correct":
        return label
    if not student_message or not concept:
        return label
    if generic_words is None:
        generic_words = get_generic_words()
    student_l = student_message.lower()
    concept_l = concept.lower()
    if concept_l in student_l:
        return label
    discriminating = [
        w for w in concept_l.split()
        if w not in generic_words and len(w) >= 5
    ]
    for word in discriminating:
        stem = word[: max(4, len(word) - 2)]
        if stem in student_l:
            return label
    return "incorrect"


def _format_last_two_turns(messages) -> str:
    """Format up to 4 prior messages (2 exchanges) as plain text for context."""
    if not messages:
        return "(no prior context)"
    lines = []
    for msg in messages:
        role = "Student" if msg.type == "human" else "Tutor"
        lines.append(f"{role}: {msg_text(msg.content)}")
    return "\n".join(lines)


def response_classifier(state: GraphState) -> dict:
    messages = state.get("messages", [])

    # Latest message is the student's current input
    student_message = ""
    if messages and messages[-1].type == "human":
        student_message = msg_text(messages[-1].content)

    # Rule-based pre-classifier: short-circuit obvious idk phrases
    if _detect_idk(student_message):
        label = "idk"
    else:
        # Prior context: up to 4 messages before the current one (2 full exchanges)
        prior = messages[:-1][-4:] if len(messages) > 1 else []
        last_two_turns = _format_last_two_turns(prior)

        # Discovery-mode branch. In function-mode the student's
        # response is judged against the concept's function (per the
        # textbook chunks), not against the concept name. We use a
        # different prompt that gets the chunks as reference, and
        # SKIP the name-based verifier and misspelling promotion
        # below — those guards are tuned for name-discovery.
        domain = state.get("domain", config.DOMAIN)
        concept = state.get("current_concept", "")
        discovery_target = (state.get("discovery_target") or "name").strip()
        is_function_mode = discovery_target == "function"

        if is_function_mode:
            chunks = state.get("retrieved_chunks", []) or []
            chunks_text = (
                "\n\n---\n\n".join(chunks) if chunks
                else "(no textbook content retrieved — fall back to standard knowledge)"
            )
            prompt = load_prompt("response_classifier_function.txt").format(
                current_concept=concept,
                student_message=student_message,
                retrieved_chunks=chunks_text,
                recent_history=last_two_turns,
            )
        else:
            prompt = load_prompt("response_classifier.txt").format(
                current_concept=concept,
                student_message=student_message,
                last_two_turns=last_two_turns,
            )

        response = _client.messages.create(
            model=config.FAST_MODEL,
            max_tokens=config.CLASSIFIER_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip().lower().rstrip(".!?,'\"")

        valid = {"irrelevant", "questioning", "incorrect", "correct", "idk"}
        label = raw if raw in valid else "incorrect"

        # Name-discovery guards skip in function mode — concept name
        # appearing or not appearing in the message is not a useful
        # signal there. (Student saying "cerebellum" again doesn't
        # demonstrate function understanding; classifier handles it
        # via the prompt's STEP 0 questioning rule.)
        if not is_function_mode:
            # Defensive guard against Haiku mislabel — see
            # _verify_correct_label.
            verified = _verify_correct_label(
                label,
                student_message,
                concept,
                generic_words=get_generic_words(domain),
            )
            if verified != label:
                print(
                    f"[classifier] override correct→incorrect "
                    f"| concept={concept!r} "
                    f"student={student_message[:80]!r}",
                    file=sys.stderr,
                )
                label = verified

        # Misspelling safety net — promote 'incorrect' → 'correct' when
        # the student wrote a fuzzy-near match of the locked concept
        # ("synapis" → "synapse"). Name-mode only; in function mode
        # there's nothing to fuzzy-match.
        if (not is_function_mode
                and label == "incorrect"
                and _is_misspelled_concept(student_message, concept)):
            print(
                f"[classifier] promote incorrect→correct (misspelling) "
                f"| concept={concept!r} student={student_message[:80]!r}",
                file=sys.stderr,
            )
            label = "correct"

        # ── Topic-naming guard ────────────────────────────────────────────
        # Any of these signals means the student is ANNOUNCING a topic
        # rather than ANSWERING a Socratic question. In all three cases,
        # a label of "correct" is wrong — there's been no prior question
        # to be correct against — so we force "questioning" and let
        # route_after_classifier send the turn to teacher_socratic for
        # a fresh Socratic opener.
        #   1. Previous tutor draft came from rapport_node or
        #      topic_choice_node — explicit topic-pick handoff.
        #   2. turn_count is 0 — fresh loop, no Socratic question
        #      has been asked yet by definition. Catches the case
        #      where draft_source_node has been overwritten by a
        #      downstream node (e.g. teach_node from a prior loop
        #      whose state lingered through topic-switch).
        #   3. The bare-token form: student_message is one or two
        #      tokens AND those tokens make up the concept itself.
        #      A student typing "cerebellum" is announcing a topic;
        #      they would write "it's the cerebellum" or "cerebellum?"
        #      to actually answer.
        prior_source = state.get("draft_source_node", "")
        turn_count_now = state.get("turn_count", 0)
        student_attempted_now = state.get("student_attempted", False)
        bare_concept_pick = False
        if concept and student_message:
            tokens = re.findall(r"[a-zA-Z']+", student_message)
            if 1 <= len(tokens) <= 4:
                joined = " ".join(t.lower() for t in tokens)
                bare_concept_pick = joined == concept.lower()
        is_topic_announcement = (
            prior_source in {"rapport_node", "topic_choice_node"}
            or (turn_count_now == 0 and not student_attempted_now)
            or bare_concept_pick
        )
        # The guard now also catches mis-labels other than "correct" —
        # function-mode classifier sometimes returns "idk" for a learn
        # request like "I'd like to learn about the cerebellum",
        # because the message doesn't describe a function. But that's
        # the loop opener, not a give-up: forcing "questioning" routes
        # to teacher_socratic with a clean opener and keeps idk_count
        # at 0 (so the IDK ladder counts the *real* IDKs that follow).
        # We exclude rule-based IDKs (caught by _detect_idk on regex
        # match) since those are deterministically a give-up phrase.
        prelabeled_by_regex = label == "idk" and _detect_idk(student_message)
        if is_topic_announcement and label != "questioning" and not prelabeled_by_regex:
            print(
                f"[classifier] override {label}→questioning "
                f"| prior_source={prior_source!r} turn={turn_count_now} "
                f"attempted={student_attempted_now} bare_pick={bare_concept_pick} "
                f"| student={student_message[:80]!r}",
                file=sys.stderr,
            )
            label = "questioning"

    # idk counter: increments on idk, resets to 0 on any engagement (correct,
    # incorrect attempt, questioning). Used by route_after_classifier to gate
    # teach_node reveal after IDK_REVEAL_THRESHOLD consecutive idks.
    prior_idk_count = state.get("idk_count", 0)
    new_idk_count = prior_idk_count + 1 if label == "idk" else 0

    # student_attempted: sticky once True. Set on any real engagement
    # (correct/incorrect/questioning). Used by edges.should_reveal to gate
    # the turn-based reveal path — prevents a student who has only ever
    # said "idk" from triggering a reveal at turn 2 via the incorrect path.
    prior_attempted = state.get("student_attempted", False)
    student_attempted = prior_attempted or label in {
        "correct", "incorrect", "questioning"
    }

    out: dict = {
        "classifier_output": label,
        "idk_count":         new_idk_count,
        "student_attempted": student_attempted,
    }

    # ── Rapport → Socratic transition: reset the per-loop counters ──────────
    # If the previous tutor response came from rapport_node or
    # topic_choice_node, this turn is the FIRST turn of the actual
    # Socratic loop on the newly-named topic. The rapport messages
    # already burned a few turns and possibly set student_attempted=True
    # via "questioning" labels; if we don't reset, route_after_classifier
    # sees turn_count >= 1 and routes "questioning" → explain_node
    # (which dumps the answer) instead of teacher_socratic (which asks
    # the Socratic opener). Wipe the per-loop counters so the new topic
    # starts cleanly.
    prior_source = state.get("draft_source_node", "")
    if prior_source in {"rapport_node", "topic_choice_node"}:
        print(
            f"[classifier] rapport→Socratic transition: resetting "
            f"turn_count, idk_count, student_attempted "
            f"| prior_source={prior_source!r}",
            file=sys.stderr,
        )
        out["turn_count"] = 0
        out["idk_count"] = 0
        out["student_attempted"] = False

    return out
