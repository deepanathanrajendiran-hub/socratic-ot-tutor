"""
graph/nodes/synthesis_assessor.py  —  Step 25

Clinical application scorer.
Called when: student_phase == "clinical_pending".

Scores the student's clinical OT response on three dimensions (0-2 each):
  1. Structure accuracy
  2. Functional consequence
  3. OT relevance

Resets student_phase to "learning" after scoring.
If weak_topic_flag is True (total <= 2), adds concept to weak_topics list.

Model: PRIMARY_MODEL (claude-sonnet-4-5)
Input:  current_concept, retrieved_chunks, messages (last = clinical response)
Output: draft_response (feedback), student_phase ("learning"),
        weak_topics (possibly updated)
"""

import json
from langchain_core.messages import AIMessage

from graph._llm_client import Anthropic

import config
from graph.state import GraphState
from graph.nodes._helpers import fill_prompt, load_prompt, msg_text

_client = Anthropic()


def _extract_last_student_message(messages) -> str:
    for msg in reversed(messages):
        if msg.type == "human":
            return msg_text(msg.content)
    return "(no response)"


def synthesis_assessor(state: GraphState) -> dict:
    concept = state.get("current_concept", "")
    chunks = state.get("retrieved_chunks", [])
    retrieved_text = (
        "\n\n---\n\n".join(chunks) if chunks else "(no content retrieved)"
    )

    messages = state.get("messages", [])
    student_clinical_response = _extract_last_student_message(messages)

    prompt = fill_prompt(
        load_prompt("synthesis_assessor.txt"),
        confirmed_concept=concept,
        retrieved_chunks=retrieved_text,
        student_clinical_response=student_clinical_response,
    )

    response = _client.messages.create(
        model=config.PRIMARY_MODEL,
        max_tokens=config.SYNTHESIS_MAX_TOKENS,
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
        # Fail-open: deliver a generic feedback message rather than crashing.
        # Append AIMessage directly (bypasses Dean — this is a scoring response,
        # not a Socratic teaching draft) and route to choice_pending so the
        # student's next reply lands in mastery_choice_classifier.
        fallback = (
            "Thanks for your response. Let's continue exploring this concept. "
            "Would you like to try another topic, practice a weak area, "
            "or are you done?"
        )
        return {
            "messages": [AIMessage(content=fallback)],
            "turn_count": state.get("turn_count", 0) + 1,
            "student_phase": "choice_pending",
            "concept_mastered": False,
        }

    total = result.get("total", 0)
    passed = result.get("passed", False)
    feedback = result.get("feedback", "")
    weak_topic_flag = result.get("weak_topic_flag", False)

    # Build a readable feedback message for the student
    score_label = "strong" if passed else "needs review"
    feedback_msg = (
        f"Score: {total}/6 ({score_label}). {feedback} "
        "Would you like to try another topic, practice a weak area, or are you done?"
    )

    # Update weak_topics if flagged
    current_weak = list(state.get("weak_topics", []))
    if weak_topic_flag and concept and concept not in current_weak:
        current_weak.append(concept)

    # Append AIMessage so the score reaches the user (was previously written
    # to draft_response and dropped at END). turn_count increments so the
    # scoring counts as a system turn. student_phase="choice_pending" routes
    # the student's next message to mastery_choice_classifier.
    return {
        "messages": [AIMessage(content=feedback_msg)],
        "turn_count": state.get("turn_count", 0) + 1,
        "student_phase": "choice_pending",
        "weak_topics": current_weak,
        "concept_mastered": False,
    }
