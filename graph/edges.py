"""
graph/edges.py — ALL conditional routing logic for the Socratic-OT graph.

Every routing decision lives here as a pure Python function.
No LLM calls, no imports of node logic — only state reads.

Routing entry point: route_after_input
  Phase-gate runs first: if student_phase is not "learning", bypass the
  normal classifier flow and route directly to the appropriate node.
"""

from langgraph.graph import END

from config import SOCRATIC_TURN_GATE, DEAN_MAX_REVISIONS
from graph.state import GraphState


# ── Entry point ───────────────────────────────────────────────────────────────

def route_after_input(state: GraphState) -> str:
    """Phase-gate checked before anything else.

    If the student is in a post-mastery choice or clinical flow, skip the
    normal manager → retrieval → classifier path.
    """
    phase = state.get("student_phase", "learning")
    if phase == "choice_pending":
        return "mastery_choice_classifier"
    if phase == "topic_choice_pending":
        return "topic_choice_classifier"
    if phase == "clinical_pending":
        return "synthesis_assessor"
    if state.get("image_pending"):
        return "vlm_node"
    return "manager_agent"


# ── Normal teaching flow ──────────────────────────────────────────────────────

def route_after_manager(state: GraphState) -> str:
    if state.get("current_concept"):
        return "retrieval"
    return "chitchat_response"


def route_after_classifier(state: GraphState) -> str:
    label = state.get("classifier_output", "")
    turn = state.get("turn_count", 0)
    if label == "irrelevant":
        return "redirect_node"
    if label == "questioning":
        # Turn 0: student opened with a question — treat as the initial prompt,
        # send to teacher_socratic for a Socratic response.
        # Turn 1+: student is asking for clarification mid-session → explain_node.
        if turn == 0:
            return "teacher_socratic"
        return "explain_node"
    if label == "incorrect":
        # Last turn and still wrong → reveal path
        if turn >= SOCRATIC_TURN_GATE:
            return "teach_node"
        return "hint_error_node"
    if label == "idk":
        # Student made no attempt — progressive scaffold, same turn gate as incorrect
        if turn >= SOCRATIC_TURN_GATE:
            return "teach_node"
        return "hint_error_node"
    if label == "correct":
        return "step_advancer"
    return "hint_error_node"  # safe default


# ── Dean quality gate ─────────────────────────────────────────────────────────

def route_after_dean(state: GraphState) -> str:
    if state.get("dean_passed"):
        return "deliver_response"
    if state.get("dean_revisions", 0) >= DEAN_MAX_REVISIONS:
        return "fallback_scaffold"
    # Route revision back to the node that originally wrote the draft.
    # Prevents step_advancer/teach_node failures from being revised by
    # teacher_socratic (wrong format — Socratic hint instead of mastery offer).
    source = state.get("draft_source_node", "teacher_socratic")
    # Only route back to nodes that have a dean→node edge in the graph.
    # All generation nodes are registered, so any valid source is safe.
    return source


# ── Post-mastery navigation ───────────────────────────────────────────────────

def route_after_step_advancer(state: GraphState) -> str:
    """Choice-prompt draft always goes through Dean before delivery."""
    return "dean_node"


def route_after_teach(state: GraphState) -> str:
    """Reveal+choice draft always goes through Dean before delivery."""
    return "dean_node"


def route_after_mastery_choice(state: GraphState) -> str:
    choice = state.get("mastery_choice", "other")
    if choice == "clinical":
        return "clinical_question_node"
    if choice == "done":
        return END
    return "topic_choice_node"          # "next" or "other"


def route_after_topic_choice(state: GraphState) -> str:
    """Both 'weak' and 'own' route to manager_agent.
    manager_agent reads topic_choice to prioritise weak topics if needed.
    """
    return "manager_agent"


# ── Utility ───────────────────────────────────────────────────────────────────

def should_reveal(state: GraphState) -> bool:
    """Shared helper used by teacher nodes to compute reveal_permitted."""
    return (
        state.get("turn_count", 0) >= SOCRATIC_TURN_GATE
        and state.get("student_attempted", False)
    )
