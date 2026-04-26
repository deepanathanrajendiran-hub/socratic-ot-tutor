"""
graph/nodes/mastery_choice_classifier.py

Classifies the student's post-mastery intent.
Called when: student_phase == "choice_pending".

Reads the student's response to the three-way choice offer and classifies it
as: clinical | next | done | other

Model: FAST_MODEL (claude-haiku-4-5)
Input:  messages (last student message)
Output: mastery_choice (str)
"""

from graph._llm_client import Anthropic

import config
from graph.state import GraphState
from graph.nodes._helpers import load_prompt, msg_text

_client = Anthropic()


def _extract_last_student_message(messages) -> str:
    for msg in reversed(messages):
        if msg.type == "human":
            return msg_text(msg.content)
    return ""


VALID = {"clinical", "next", "done", "other"}


def mastery_choice_classifier(state: GraphState) -> dict:
    messages = state.get("messages", [])
    student_message = _extract_last_student_message(messages)

    prompt = load_prompt("mastery_choice.txt").format(student_message=student_message)

    response = _client.messages.create(
        model=config.FAST_MODEL,
        max_tokens=config.CLASSIFIER_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip().lower().split()[0] if response.content[0].text.strip() else "other"
    choice = raw if raw in VALID else "other"

    # Always reset student_phase — the classifier's job is done after one call.
    # Downstream nodes (clinical_question_node, topic_choice_node, END) set
    # their own next phase. This prevents the "done" infinite loop where
    # student_phase stayed "choice_pending" and re-routed every subsequent
    # message back into this classifier.
    return {"mastery_choice": choice, "student_phase": "learning"}
