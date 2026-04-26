"""
graph/nodes/response_classifier.py

Classifies the student's latest message into exactly one label:
  irrelevant | questioning | incorrect | correct | idk

Model: FAST_MODEL (claude-haiku-4-5) — single-word output, max_tokens=config.CLASSIFIER_MAX_TOKENS.
Input:  state["current_concept"], state["messages"]
Output: state["classifier_output"]
"""

import re

from graph._llm_client import Anthropic

import config
from graph.state import GraphState
from graph.nodes._helpers import load_prompt, msg_text

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
    r"\bnot\s+sure\b",
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

        prompt = load_prompt("response_classifier.txt").format(
            current_concept=state.get("current_concept", ""),
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

    return {
        "classifier_output": label,
        "idk_count":         new_idk_count,
        "student_attempted": student_attempted,
    }
