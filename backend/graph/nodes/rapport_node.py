"""
graph/nodes/rapport_node.py

LLM-driven rapport / chitchat handler for sessions where the manager
agent hasn't yet locked a specific concept.

Replaces the previous static template ("Let's get back to anatomy — is
there a specific topic you'd like to explore?") with a Haiku-powered
multi-turn conversation that:

  - Greets / acknowledges the student naturally
  - Probes for what they want to learn (exam prep, curiosity, etc.)
  - Surfaces one weak topic from this session as a "want to revisit?"
  - Offers narrowing options when the student gestures vaguely
    ("the brain", "nerves") instead of dead-ending the conversation

The node does NOT extract a concept itself — that's manager_agent's job.
It just keeps the conversation alive until the student names something
specific. On the very next turn, manager_agent will pick the concept up
from the message and flow the request through the normal Socratic path.

Model: FAST_MODEL (claude-haiku-4-5) — chat-quality but cheap. The
output is a plain string, not JSON, so no parsing needed.

Input:  messages (history), domain, weak_topics
Output: messages (AIMessage appended), turn_count incremented, current
        per-loop counters left untouched (turn_count grows so the
        eventual Socratic loop sees realistic prior context, but we
        DON'T touch student_attempted / idk_count — those still mean
        "this Socratic loop" and start from False/0 the first time
        teacher_socratic actually fires).
"""
import sys

from langchain_core.messages import AIMessage

from graph._llm_client import Anthropic

import config
from graph.state import GraphState
from graph.nodes._helpers import fill_prompt, load_prompt, msg_text

_client = Anthropic()


def _format_recent_history(messages, n: int = 6) -> str:
    if not messages:
        return "(no prior conversation — fresh session)"
    recent = messages[-n:]
    lines = []
    for msg in recent:
        role = "Student" if getattr(msg, "type", None) == "human" else "Tutor"
        lines.append(f"{role}: {msg_text(getattr(msg, 'content', ''))}")
    return "\n".join(lines)


def _extract_last_student_message(messages) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human":
            return msg_text(getattr(msg, "content", ""))
    return ""


# Static fallback if the LLM call fails (Bedrock 503, network blip, etc.).
# Better than crashing the turn — graceful degradation matches the
# original chitchat_response behavior.
_FALLBACK_REPLY = (
    "Hey — I'm here to help you work through OT anatomy and neuroscience. "
    "What's on your mind today? Anything specific you'd like to dig into, "
    "or want me to suggest a few topics?"
)


def rapport_node(state: GraphState) -> dict:
    messages = state.get("messages", [])
    domain = state.get("domain", config.DOMAIN)
    domain_cfg = config.DOMAIN_CONFIG.get(domain, {})
    domain_ctx       = domain_cfg.get("system_context", domain)
    target_exam      = domain_cfg.get("target_exam", "")
    textbook         = domain_cfg.get("textbook", "")
    rapport_examples = domain_cfg.get("rapport_examples", "a specific concept")

    student_message = _extract_last_student_message(messages)
    recent_history = _format_recent_history(messages)
    weak = state.get("weak_topics", []) or []
    weak_text = ", ".join(weak) if weak else "(none yet)"

    # Cross-session memory layer (no-op when MEMORY_BACKEND=sqlite).
    # Surfaces relevant facts from prior sessions ("user is prepping for
    # NBCOT", "user struggled with brachial plexus last time") so the
    # rapport reply feels continuous rather than amnesiac.
    user_memories_text = "(none — first time, or memory layer disabled)"
    user_id = state.get("user_id", "") or ""
    if user_id:
        try:
            from memory.mem0_client import client as mem0_client, memories_to_text
            if mem0_client.enabled:
                hits = mem0_client.search(
                    query=student_message or "tutoring session",
                    user_id=user_id,
                    limit=config.MEM0_TOP_K,
                )
                rendered = memories_to_text(hits)
                if rendered:
                    user_memories_text = rendered
                    print(
                        f"[rapport_node] mem0 injected {len(hits)} memory(ies)",
                        file=sys.stderr,
                    )
        except Exception as exc:
            print(
                f"[rapport_node] memory lookup failed (non-fatal): {exc!r}",
                file=sys.stderr,
            )

    prompt = fill_prompt(
        load_prompt("rapport.txt"),
        domain_context=domain_ctx,
        target_exam=target_exam,
        textbook=textbook,
        rapport_examples=rapport_examples,
        student_message=student_message,
        recent_history=recent_history,
        weak_topics=weak_text,
        user_memories=user_memories_text,
    )

    try:
        response = _client.messages.create(
            model=config.FAST_MODEL,
            max_tokens=getattr(config, "RAPPORT_MAX_TOKENS", 256),
            messages=[{"role": "user", "content": prompt}],
        )
        reply = (response.content[0].text or "").strip()
        if not reply:
            reply = _FALLBACK_REPLY
    except Exception as exc:
        # Don't crash the turn on transient API failures — fallback
        # message preserves the conversation.
        print(
            f"[rapport_node] LLM call failed ({exc!r}); using fallback reply",
            file=sys.stderr,
        )
        reply = _FALLBACK_REPLY

    return {
        "messages": [AIMessage(content=reply)],
        "draft_response": reply,
        "draft_source_node": "rapport_node",
        "turn_count": state.get("turn_count", 0) + 1,
    }
