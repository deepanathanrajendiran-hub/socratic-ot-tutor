"""
graph/edges.py — ALL conditional routing logic for the Socratic-OT graph.

Every routing decision lives here as a pure Python function.
No LLM calls, no imports of node logic — only state reads.

Routing entry point: route_after_input
  Phase-gate runs first: if student_phase is not "learning", bypass the
  normal classifier flow and route directly to the appropriate node.
"""

from langgraph.graph import END

from config import SOCRATIC_TURN_GATE, DEAN_MAX_REVISIONS, IDK_REVEAL_THRESHOLD
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


def route_after_retrieval(state: GraphState) -> str:
    """Mode dispatch after retrieval. Study mode bypasses the
    classifier+Dean+teacher chain and goes straight to study_node, which
    delivers a direct answer + JSON envelope. Socratic mode (default) takes
    the normal classifier-driven path.
    """
    if state.get("mode", "socratic") == "study":
        return "study_node"
    return "response_classifier"


def route_after_manager_and_retrieval(state: GraphState) -> str:
    """Fused router for the parallelized manager+retrieval combo.
    Combines the prior `route_after_manager` (no-concept → chitchat)
    and `route_after_retrieval` (study-mode short-circuit) decisions
    into a single edge so the graph stays a DAG.
    """
    if not state.get("current_concept"):
        return "chitchat_response"
    if state.get("mode", "socratic") == "study":
        return "study_node"
    return "response_classifier"


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
        # Reveal path requires BOTH past the turn gate AND prior engagement.
        # student_attempted prevents a disengaged session (all idks) from
        # accidentally landing on teach_node via a single "incorrect" label.
        if (
            turn >= SOCRATIC_TURN_GATE
            and state.get("student_attempted", False)
        ):
            return "teach_node"
        return "hint_error_node"
    if label == "idk":
        # Progressive scaffold gated by idk_count, NOT by turn_count.
        # The student must accumulate IDK_REVEAL_THRESHOLD consecutive idks
        # before reveal — engagement (any non-idk classification) resets the
        # counter to 0. This prevents the help-abuse jailbreak where a student
        # spams "I don't know" to skip the work, while still giving a way out
        # for a genuinely-stuck student after several attempts.
        if state.get("idk_count", 0) >= IDK_REVEAL_THRESHOLD:
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
    source = state.get("draft_source_node")
    if not source:
        # Fail loudly — silent fallback to teacher_socratic hides the bug
        # (a generation node forgot to set draft_source_node). Better to
        # surface the contract violation than produce a wrong-format revision.
        raise ValueError(
            "route_after_dean: draft_source_node is missing from state. "
            "Every generation node MUST set it in its return dict."
        )
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
    if choice == "analyze":
        return "attempt_analysis_node"  # post-reveal "where did I go wrong" branch
    return "topic_choice_node"          # "next" or "other"


def route_after_topic_choice(state: GraphState) -> str:
    """Both 'weak' and 'own' route to manager_agent.
    manager_agent reads topic_choice to prioritise weak topics if needed.
    """
    return "manager_agent"


# ── Utility ───────────────────────────────────────────────────────────────────

def should_reveal(state: GraphState) -> bool:
    """Canonical reveal-gate helper. Used by every generation node's
    `_should_reveal` shim so there's exactly one reveal policy in the codebase.

    Reveal is permitted when ANY of:
    1. Mastery already confirmed (step_advancer set concept_mastered=True).
    2. Post-mastery phase — clinical_question, topic_choice, etc.
       (student_phase != "learning" means we are past the reveal gate).
    3. Past the Socratic turn gate AND the student made at least one
       real attempt. student_attempted prevents help-abuse (idk-only
       sessions never reach reveal regardless of turn count).
    """
    if state.get("concept_mastered", False):
        return True
    if state.get("student_phase", "learning") != "learning":
        return True
    return (
        state.get("turn_count", 0) >= SOCRATIC_TURN_GATE
        and state.get("student_attempted", False)
    )
