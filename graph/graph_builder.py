"""
graph/graph_builder.py — assembles the full Socratic-OT LangGraph.

Stubs for Phase 3 nodes (manager_agent, retrieval, vlm_node,
synthesis_assessor, deliver_response, fallback_scaffold, chitchat_response)
are included as passthrough functions so the graph compiles and can be
smoke-tested end-to-end before those nodes are built.

Phase 3 nodes to replace stubs:
  Step 22: input_router  (currently inlined in route_after_input edge)
  Step 23: manager_agent
  Step 24: vlm_node
  Step 25: synthesis_assessor
  Step 26: memory/session_store (deliver_response logs to SQLite)
  Step 28: api/main.py
"""

from langgraph.graph import StateGraph, END

from graph.state import GraphState
from graph.edges import (
    route_after_input,
    route_after_manager,
    route_after_classifier,
    route_after_dean,
    route_after_step_advancer,
    route_after_teach,
    route_after_mastery_choice,
    route_after_topic_choice,
)

# ── Real nodes ────────────────────────────────────────────────────────────────
from graph.nodes.response_classifier import response_classifier
from graph.nodes.teacher_socratic import teacher_socratic
from graph.nodes.dean_node import dean_node
from graph.nodes.hint_error_node import hint_error_node
from graph.nodes.redirect_node import redirect_node
from graph.nodes.explain_node import explain_node
from graph.nodes.step_advancer import step_advancer
from graph.nodes.teach_node import teach_node
from graph.nodes.mastery_choice_classifier import mastery_choice_classifier
from graph.nodes.topic_choice_node import topic_choice_node
from graph.nodes.topic_choice_classifier import topic_choice_classifier
from graph.nodes.clinical_question_node import clinical_question_node
from graph.nodes.manager_agent import manager_agent
from graph.nodes.retrieval_node import retrieval_node
from graph.nodes.synthesis_assessor import synthesis_assessor


# ── Phase 3 stubs (Step 24 vlm_node still pending) ───────────────────────────

def _stub_vlm_node(state: GraphState) -> dict:
    """Stub: returns state unchanged. Real node built in Step 24."""
    return {}


def _stub_deliver_response(state: GraphState) -> dict:
    """Stub: appends draft_response to messages and increments turn_count.
    Real deliver_response also logs to SQLite (Step 26).
    """
    from langchain_core.messages import AIMessage
    draft = state.get("draft_response", "")
    turn_count = state.get("turn_count", 0)
    updates: dict = {"turn_count": turn_count + 1}
    if draft:
        updates["messages"] = [AIMessage(content=draft)]
    return updates


def _stub_fallback_scaffold(state: GraphState) -> dict:
    """Stub: delivers a safe generic scaffold when Dean max revisions exceeded.
    Always resets student_phase to 'learning' so stale phase state doesn't
    corrupt routing on the next student turn.
    """
    from langchain_core.messages import AIMessage
    msg = (
        "Let's take a step back. Based on what we've covered, "
        "can you describe what you know so far about this topic?"
    )
    return {
        "messages": [AIMessage(content=msg)],
        "turn_count": state.get("turn_count", 0) + 1,
        "draft_response": msg,
        "student_phase": "learning",
        "concept_mastered": False,
        "dean_revisions": 0,
        "dean_revision_instruction": "",
    }


def _stub_chitchat_response(state: GraphState) -> dict:
    """Stub: handles off-topic small talk when manager finds no concept."""
    from langchain_core.messages import AIMessage
    msg = (
        "That's an interesting thought! Let's get back to anatomy — "
        "is there a specific topic you'd like to explore?"
    )
    return {
        "messages": [AIMessage(content=msg)],
        "turn_count": state.get("turn_count", 0) + 1,
    }


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(GraphState)

    # Real nodes
    g.add_node("response_classifier", response_classifier)
    g.add_node("teacher_socratic", teacher_socratic)
    g.add_node("dean_node", dean_node)
    g.add_node("hint_error_node", hint_error_node)
    g.add_node("redirect_node", redirect_node)
    g.add_node("explain_node", explain_node)
    g.add_node("step_advancer", step_advancer)
    g.add_node("teach_node", teach_node)
    g.add_node("mastery_choice_classifier", mastery_choice_classifier)
    g.add_node("topic_choice_node", topic_choice_node)
    g.add_node("topic_choice_classifier", topic_choice_classifier)
    g.add_node("clinical_question_node", clinical_question_node)

    # Real Phase 3 nodes
    g.add_node("manager_agent", manager_agent)
    g.add_node("retrieval", retrieval_node)
    g.add_node("synthesis_assessor", synthesis_assessor)

    # Remaining stubs (Step 24)
    g.add_node("vlm_node", _stub_vlm_node)
    g.add_node("deliver_response", _stub_deliver_response)
    g.add_node("fallback_scaffold", _stub_fallback_scaffold)
    g.add_node("chitchat_response", _stub_chitchat_response)

    # Entry point — phase gate (single conditional edge from __start__)
    g.add_conditional_edges(
        "__start__",
        route_after_input,
        {
            "mastery_choice_classifier": "mastery_choice_classifier",
            "topic_choice_classifier": "topic_choice_classifier",
            "synthesis_assessor": "synthesis_assessor",
            "vlm_node": "vlm_node",
            "manager_agent": "manager_agent",
        },
    )

    # Normal teaching flow
    g.add_conditional_edges(
        "manager_agent",
        route_after_manager,
        {"retrieval": "retrieval", "chitchat_response": "chitchat_response"},
    )
    g.add_edge("retrieval", "response_classifier")
    g.add_conditional_edges(
        "response_classifier",
        route_after_classifier,
        {
            "redirect_node": "redirect_node",
            "explain_node": "explain_node",
            "hint_error_node": "hint_error_node",
            "step_advancer": "step_advancer",
            "teach_node": "teach_node",
            "teacher_socratic": "teacher_socratic",   # turn-0 question path
        },
    )

    # All generation nodes feed into Dean
    for node in ("teacher_socratic", "hint_error_node", "redirect_node",
                 "explain_node", "clinical_question_node", "topic_choice_node"):
        g.add_edge(node, "dean_node")

    g.add_conditional_edges(
        "step_advancer",
        route_after_step_advancer,
        {"dean_node": "dean_node"},
    )
    g.add_conditional_edges(
        "teach_node",
        route_after_teach,
        {"dean_node": "dean_node"},
    )

    # Dean gate
    g.add_conditional_edges(
        "dean_node",
        route_after_dean,
        {
            "deliver_response":        "deliver_response",
            "fallback_scaffold":       "fallback_scaffold",
            "teacher_socratic":        "teacher_socratic",
            "hint_error_node":         "hint_error_node",
            "explain_node":            "explain_node",
            "redirect_node":           "redirect_node",
            "step_advancer":           "step_advancer",
            "teach_node":              "teach_node",
            "clinical_question_node":  "clinical_question_node",
            "topic_choice_node":       "topic_choice_node",
        },
    )

    # Post-mastery navigation
    g.add_conditional_edges(
        "mastery_choice_classifier",
        route_after_mastery_choice,
        {
            "clinical_question_node": "clinical_question_node",
            "topic_choice_node": "topic_choice_node",
            END: END,
        },
    )
    g.add_conditional_edges(
        "topic_choice_classifier",
        route_after_topic_choice,
        {"manager_agent": "manager_agent"},
    )

    # Terminal nodes
    g.add_edge("deliver_response", END)
    g.add_edge("fallback_scaffold", END)
    g.add_edge("chitchat_response", END)
    g.add_edge("synthesis_assessor", END)
    g.add_edge("vlm_node", END)

    return g.compile()


# Singleton for import
graph = build_graph()
