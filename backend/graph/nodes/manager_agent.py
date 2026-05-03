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
        # Defensive net: even when topic_choice="weak", check the
        # student's message for a named concept. If they named one of
        # the weak topics specifically (e.g. "cerebellum" instead of
        # generic "weak"), honor that pick instead of grabbing
        # weak_topics[0]. Without this, the classifier mislabel
        # "named-from-list = weak" silently rerouted the student to
        # the wrong topic. Reported 2026-05-01.
        last_msg = _extract_last_student_message(state.get("messages", []))
        if last_msg:
            ml = last_msg.lower()
            for t in weak_topics:
                if t and t.lower() in ml:
                    return {
                        "current_concept": t,
                        "topic_choice": "",
                        "dean_revisions": 0,
                        "dean_revision_instruction": "",
                        "student_phase": "learning",
                    }
        return {
            "current_concept": weak_topics[0],
            "topic_choice": "",
            "dean_revisions": 0,
            "dean_revision_instruction": "",
            "student_phase": "learning",
        }

    # Concept-extraction path: every turn (no short-circuit).
    # The LLM extracts the concept fresh each turn but receives the
    # currently-locked concept so it can preserve on follow-ups ("idk",
    # "spinal cord", "two") and only switch on a clear topic shift
    # ("what about the rotator cuff?"). Without an explicit locked_concept
    # the LLM drifts mid-loop (synapse → neurotransmitters at turn 3).
    messages = state.get("messages", [])
    domain = state.get("domain", config.DOMAIN)
    domain_cfg = config.DOMAIN_CONFIG.get(domain, {})
    domain_ctx       = domain_cfg.get("system_context", domain)
    textbook         = domain_cfg.get("textbook", "")
    subject_noun     = domain_cfg.get("subject_noun", "domain")
    example_concepts = domain_cfg.get("example_concepts", "")
    reject_examples  = domain_cfg.get("reject_examples", "")

    student_message = _extract_last_student_message(messages)
    recent_history  = _format_recent_history(messages)
    locked_concept  = state.get("current_concept", "") or "(none yet)"

    prompt = fill_prompt(
        load_prompt("manager_agent.txt"),
        domain_context=domain_ctx,
        textbook=textbook,
        subject_noun=subject_noun,
        example_concepts=example_concepts,
        reject_examples=reject_examples,
        student_message=student_message,
        recent_history=recent_history,
        locked_concept=locked_concept,
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

    # Discovery-target detection. If the concept (or one of its
    # discriminating stems) appears in the student's current message,
    # they NAMED the topic upfront — switch the loop into function-
    # discovery mode. Otherwise stay in name-discovery (the default).
    # We also preserve a non-empty existing target so a multi-turn loop
    # doesn't flip mode mid-conversation: the choice is locked once
    # the concept is first locked.
    discovery_target = (state.get("discovery_target") or "").strip()
    if not discovery_target and concept and student_message:
        if _student_named_concept(student_message, concept):
            discovery_target = "function"
        else:
            discovery_target = "name"

    return {
        "current_concept": concept,
        "discovery_target": discovery_target,
        "topic_choice": "",
        "dean_revisions": 0,
        "dean_revision_instruction": "",
        "student_phase": "learning",
    }


def _student_named_concept(student_message: str, concept: str) -> bool:
    """True only when the student's message contains the FULL concept
    (or its simple plural). Used to gate Path-B function-discovery mode
    — must be conservative because the Dean's REVEAL_CHECK skips
    function-mode loops, so a false-positive here lets the concept name
    leak through.

    Why exact match (not stems): adjectival/related forms ('synaptic'
    for 'synapse', 'neuronal' for 'neuron') are NOT the student naming
    the concept upfront. Path B's contract is "the student typed the
    concept name verbatim". Stem-matching mis-fires on:
      - 'What is the synaptic cleft?' → concept 'synapse' (stem 'synap'
        matches 'synaptic'); student is asking about a related but
        distinct concept, not naming the synapse upfront.
      - 'Which nerve is compressed?' → concept 'median nerve' (stem
        'nerv' matches 'nerve'); generic noun, not a name.

    For multi-word concepts the student must include enough of the
    discriminating words for the LLM concept-extractor to settle on
    the same concept anyway — exact-match is sufficient in practice.
    (2026-05-03: tightened from stem-match after regression report.)
    """
    if not student_message or not concept:
        return False
    student_l = student_message.lower()
    concept_l = concept.lower().strip()
    if concept_l in student_l:
        return True
    # Simple plural: "synapses" should also count as naming "synapse".
    if (concept_l + "s") in student_l:
        return True
    return False
