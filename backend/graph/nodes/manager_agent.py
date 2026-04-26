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
from graph._llm_client import Anthropic

import config
from graph.state import GraphState
from graph.nodes._helpers import fill_prompt, load_prompt, msg_text

_client = Anthropic()


def _format_recent_history(messages, n: int = 4) -> str:
    if not messages:
        return "(no prior conversation)"
    recent = messages[-n:]
    lines = []
    for msg in recent:
        role = "Student" if msg.type == "human" else "Tutor"
        lines.append(f"{role}: {msg_text(msg.content)}")
    return "\n".join(lines)


def _extract_last_student_message(messages) -> str:
    for msg in reversed(messages):
        if msg.type == "human":
            return msg_text(msg.content)
    return ""


def manager_agent(state: GraphState) -> dict:
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

    # Concept-extraction path: every turn (no short-circuit).
    # Previously, the manager skipped extraction whenever a concept was set
    # and there was prior conversation — but that blocked legitimate topic
    # switches mid-session ("what about the rotator cuff?"). The manager
    # prompt is responsible for preserving the concept on follow-ups
    # ("spinal cord", "two") AND switching it on new topics. Trust the LLM.
    messages = state.get("messages", [])
    domain = state.get("domain", config.DOMAIN)
    domain_ctx = config.DOMAIN_CONFIG.get(domain, {}).get("system_context", domain)

    student_message = _extract_last_student_message(messages)
    recent_history = _format_recent_history(messages)

    prompt = fill_prompt(
        load_prompt("manager_agent.txt"),
        domain_context=domain_ctx,
        student_message=student_message,
        recent_history=recent_history,
    )

    response = _client.messages.create(
        model=config.FAST_MODEL,
        max_tokens=config.MANAGER_MAX_TOKENS,
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
