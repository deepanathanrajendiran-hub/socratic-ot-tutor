"""
graph/nodes/study_node.py

Study-mode answerer. Calls Sonnet with prompts/study.txt, parses the JSON
envelope, and updates the implicit weak-topic counter.

Per ABSOLUTE RULE #5 in CLAUDE.md, Dean only checks LLM-generated *teacher*
responses (Socratic mode). Study mode is direct teaching by design — Dean
is bypassed and a separate grounding contract is enforced via the prompt's
"every claim must be supported" rule plus the live retrieval pipeline.

Implicit weak-topic counter: after STUDY_WEAK_TOPIC_THRESHOLD consecutive
questions on the same active_topic, that topic is appended to weak_topics
so subsequent Socratic sessions can prioritize it.
"""
import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

import config
from graph._llm_client import Anthropic
from graph.nodes._helpers import load_prompt, fill_prompt, msg_text


STUDY_WEAK_TOPIC_THRESHOLD = 5

_client = Anthropic()


def _call_llm(prompt: str) -> str:
    """Single Sonnet call, deterministic. Subclassable via test patch."""
    resp = _client.messages.create(
        model=config.PRIMARY_MODEL,
        max_tokens=config.SYNTHESIS_MAX_TOKENS,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text if resp.content else ""


def _parse_envelope(raw: str) -> dict[str, Any]:
    """Extract the JSON envelope from the LLM output. Tolerates leading/
    trailing whitespace and stray markdown fences (despite the prompt
    forbidding them — LLMs sometimes ignore the rule).
    """
    text = raw.strip()
    if text.startswith("```"):
        # Strip leading fence and optional 'json' tag
        text = text.lstrip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
        # Strip trailing fence
        end = text.rfind("```")
        if end != -1:
            text = text[:end].strip()
    # Find the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return {"answer": raw, "active_topic": "", "is_continuation": False,
                "citations": [], "_parse_failed": True}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"answer": raw, "active_topic": "", "is_continuation": False,
                "citations": [], "_parse_failed": True}


def study_node(state: dict) -> dict:
    """Run one Study-mode turn. Returns state updates only."""
    last_human = next((m for m in reversed(state.get("messages", []))
                       if isinstance(m, HumanMessage)), None)
    question = msg_text(last_human.content) if last_human else ""

    chunks = state.get("retrieved_chunks", []) or []
    chunks_text = "\n\n---\n\n".join(chunks) if chunks else "(no content retrieved)"

    prior_topic = state.get("study_active_topic", "")
    domain_ctx = config.DOMAIN_CONFIG.get(
        state.get("domain", config.DOMAIN), {}
    ).get("system_context", "")

    prompt = fill_prompt(
        load_prompt("study.txt"),
        question=question,
        retrieved_chunks=chunks_text,
        domain_context=domain_ctx,
        prior_active_topic=prior_topic,
    )
    envelope = _parse_envelope(_call_llm(prompt))

    new_topic = envelope.get("active_topic", "") or ""
    # Recompute is_continuation in Python — don't trust the LLM's flag.
    # The contract is "active_topic equals prior_active_topic", so derive it.
    is_continuation = bool(prior_topic) and prior_topic == new_topic

    if is_continuation:
        new_count = state.get("study_topic_count", 0) + 1
    else:
        new_count = 1 if new_topic else 0

    weak_topics = list(state.get("weak_topics", []))
    if (new_count >= STUDY_WEAK_TOPIC_THRESHOLD and new_topic
            and new_topic not in weak_topics):
        weak_topics.append(new_topic)

    return {
        "messages": [AIMessage(content=envelope.get("answer", ""))],
        "study_active_topic":  new_topic,
        "study_topic_count":   new_count,
        "weak_topics":         weak_topics,
        "turn_count":          state.get("turn_count", 0) + 1,
    }
