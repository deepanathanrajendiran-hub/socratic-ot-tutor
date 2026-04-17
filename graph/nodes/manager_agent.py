"""
graph/nodes/manager_agent.py  —  Step 23

Concept extractor and session manager.
Called from: route_after_input (when student_phase == "learning")
             route_after_topic_choice (when student picks next topic)

Responsibilities:
  1. If topic_choice == "weak" and weak_topics is non-empty:
       pick the first weak topic directly (no LLM needed)
  2. Otherwise: call FAST_MODEL with manager_agent.txt to extract the
       primary anatomy concept from the student's message.
  3. If chitchat / no concept found: set current_concept = "" so
       route_after_manager sends to chitchat_response.

Model: FAST_MODEL (claude-haiku-4-5)
Input:  messages, domain, weak_topics, topic_choice
Output: current_concept (str)
"""

import json
import os

from anthropic import Anthropic

import config
from graph.state import GraphState

_client = Anthropic()


def _load_prompt() -> str:
    path = os.path.join(config.PROMPTS_DIR, "manager_agent.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _fill_prompt(template: str, **kwargs) -> str:
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def _msg_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def _format_recent_history(messages, n: int = 4) -> str:
    if not messages:
        return "(no prior conversation)"
    recent = messages[-n:]
    lines = []
    for msg in recent:
        role = "Student" if msg.type == "human" else "Tutor"
        lines.append(f"{role}: {_msg_text(msg.content)}")
    return "\n".join(lines)


def _extract_last_student_message(messages) -> str:
    for msg in reversed(messages):
        if msg.type == "human":
            return _msg_text(msg.content)
    return ""


def manager_agent(state: GraphState) -> dict:
    # Fast path: student chose weak topics — no LLM extraction needed
    # Fast path: student chose weak topics — no LLM extraction needed
    # Fast path: student chose weak topics — no LLM extraction needed
    # Fast path: student chose weak topics — no LLM extraction needed
    topic_choice = state.get("topic_choice", "")
    weak_topics = state.get("weak_topics", [])

    if topic_choice == "weak" and weak_topics:
        return {
            "current_concept": weak_topics[0],
            "topic_choice": "",
            "dean_revisions": 0,
            "dean_revision_instruction": "",
            "student_phase": "learning",
        }

    # Concept-preservation path: once a concept is established and the
    # conversation is ongoing, student follow-up answers (e.g. "spinal cord",
    # "two", "the anterior horn") must NOT overwrite the session concept.
    # Only skip extraction when:
    #   - a concept is already set, AND
    #   - there is prior conversation (messages > 1), AND
    #   - topic_choice is empty (not a deliberate topic switch)
    existing_concept = state.get("current_concept", "")
    messages = state.get("messages", [])
    if existing_concept and len(messages) > 1 and not topic_choice:
        return {
            "current_concept": existing_concept,
            "topic_choice": "",
            "dean_revisions": 0,
            "dean_revision_instruction": "",
            "student_phase": "learning",
        }

    # Concept-extraction path: first turn, new session, or explicit topic switch
    domain = state.get("domain", config.DOMAIN)
    domain_ctx = config.DOMAIN_CONFIG.get(domain, {}).get("system_context", domain)

    student_message = _extract_last_student_message(messages)
    recent_history = _format_recent_history(messages)

    prompt = _fill_prompt(
        _load_prompt(),
        domain_context=domain_ctx,
        student_message=student_message,
        recent_history=recent_history,
    )

    response = _client.messages.create(
        model=config.FAST_MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Extract the JSON object — handles code fences and trailing rationale text
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]

    try:
        result = json.loads(raw)
        concept = result.get("current_concept") or ""
        if concept == "null":
            concept = ""
    except (json.JSONDecodeError, AttributeError):
        concept = ""

    return {
        "current_concept": concept,
        "topic_choice": "",
        "dean_revisions": 0,
        "dean_revision_instruction": "",
        "student_phase": "learning",
    }
