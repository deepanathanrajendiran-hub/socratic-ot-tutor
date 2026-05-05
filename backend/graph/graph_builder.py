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
    route_after_manager_and_retrieval,
    route_after_classifier,
    route_after_dean,
    route_after_step_advancer,
    route_after_teach,
    route_after_mastery_choice,
    route_after_topic_choice,
)

# ── Real nodes ────────────────────────────────────────────────────────────────
from graph.nodes.response_classifier import response_classifier
from graph.nodes.study_node import study_node
from graph.nodes.teacher_socratic import teacher_socratic
from graph.nodes.dean_node import dean_node
from graph.nodes.hint_error_node import hint_error_node
from graph.nodes.redirect_node import redirect_node
from graph.nodes.explain_node import explain_node
from graph.nodes.step_advancer import step_advancer
from graph.nodes.teach_node import teach_node
from graph.nodes.attempt_analysis_node import attempt_analysis_node
from graph.nodes.mastery_choice_classifier import mastery_choice_classifier
from graph.nodes.topic_choice_node import topic_choice_node
from graph.nodes.topic_choice_classifier import topic_choice_classifier
from graph.nodes.clinical_question_node import clinical_question_node
from graph.nodes.manager_agent import manager_agent
from graph.nodes.retrieval_node import retrieval_node
from graph.nodes.manager_and_retrieval import make_manager_and_retrieval
from graph.nodes.synthesis_assessor import synthesis_assessor


# ── Phase 3 stubs (Step 24 vlm_node still pending) ───────────────────────────

def _stub_vlm_node(state: GraphState) -> dict:
    """Adapter: forward to the real Sonnet-vision vlm_node.

    Kept under the legacy name so the existing graph wiring (the
    `vlm_node` target in route_after_input + the END edge below)
    continues to resolve. Actual identification + Socratic-opener
    logic lives in graph/nodes/vlm_node.py.
    """
    from graph.nodes.vlm_node import vlm_node
    return vlm_node(state)


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
    """Adapter: forward to the real LLM-driven rapport_node.

    Kept under the legacy name so existing graph wiring (route_after_*
    targets, `chitchat_response` edge) continues to resolve. The actual
    behavior — context-aware multi-turn rapport that lets the student
    converge on a topic naturally instead of being told to "get back to
    anatomy" — lives in graph/nodes/rapport_node.py.
    """
    from graph.nodes.rapport_node import rapport_node
    return rapport_node(state)


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(GraphState)

    # Real nodes — wrapped with trace decorators where the architecture
    # visualizer surfaces them (concept extraction, retrieval, classifier,
    # generation, Dean, study). See graph/_trace.py.
    from graph._trace import with_trace
    g.add_node("response_classifier",
               with_trace("classifier", model="haiku")(response_classifier))
    g.add_node("teacher_socratic",
               with_trace("generation", model="sonnet")(teacher_socratic))
    g.add_node("dean_node",
               with_trace("dean", model="sonnet")(dean_node))
    g.add_node("hint_error_node", hint_error_node)
    g.add_node("redirect_node", redirect_node)
    g.add_node("explain_node", explain_node)
    g.add_node("step_advancer", step_advancer)
    g.add_node("teach_node", teach_node)
    g.add_node("attempt_analysis_node",
               with_trace("attempt_analysis", model="sonnet")(attempt_analysis_node))
    g.add_node("mastery_choice_classifier", mastery_choice_classifier)
    g.add_node("topic_choice_node", topic_choice_node)
    g.add_node("topic_choice_classifier", topic_choice_classifier)
    g.add_node("clinical_question_node", clinical_question_node)

    # Real Phase 3 nodes — manager_agent and retrieval_node are wrapped
    # individually for the architecture visualizer (each emits its own
    # trace event), then composed into a single combo node that runs them
    # concurrently to cut ~2.5s off TTFT every turn. We still register
    # the inner names for backwards compatibility with anything that
    # references them directly (currently nothing in the live graph;
    # routing uses "manager_and_retrieval"). The duplicate-node calls
    # are harmless because LangGraph deduplicates by name.
    _traced_manager = with_trace("concept_extraction", model="haiku")(manager_agent)
    _traced_retrieval = with_trace("retrieval")(retrieval_node)
    g.add_node("manager_and_retrieval",
               make_manager_and_retrieval(_traced_manager, _traced_retrieval))
    g.add_node("synthesis_assessor", synthesis_assessor)

    # Phase 5: study mode answerer (bypasses classifier + Dean)
    g.add_node("study_node",
               with_trace("study", model="sonnet")(study_node))

    # Remaining stubs (Step 24)
    g.add_node("vlm_node", _stub_vlm_node)
    g.add_node("deliver_response", _stub_deliver_response)
    g.add_node("fallback_scaffold", _stub_fallback_scaffold)
    g.add_node("chitchat_response", _stub_chitchat_response)

    # Entry point — phase gate (single conditional edge from __start__).
    # Note: route_after_input still returns "manager_agent" for the
    # learning-phase route; we map it to the parallel combo node here so
    # callers (input router, edges.py) don't need to change.
    g.add_conditional_edges(
        "__start__",
        route_after_input,
        {
            "mastery_choice_classifier": "mastery_choice_classifier",
            "topic_choice_classifier": "topic_choice_classifier",
            "synthesis_assessor": "synthesis_assessor",
            "vlm_node": "vlm_node",
            "manager_agent": "manager_and_retrieval",
        },
    )

    # Normal teaching flow — combined manager+retrieval node fans out into
    # one of three downstream paths (chitchat / study / classifier).
    g.add_conditional_edges(
        "manager_and_retrieval",
        route_after_manager_and_retrieval,
        {
            "chitchat_response":   "chitchat_response",
            "study_node":          "study_node",
            "response_classifier": "response_classifier",
        },
    )
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
                 "explain_node", "clinical_question_node", "topic_choice_node",
                 "attempt_analysis_node"):
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
            "attempt_analysis_node":   "attempt_analysis_node",
        },
    )

    # Post-mastery navigation
    g.add_conditional_edges(
        "mastery_choice_classifier",
        route_after_mastery_choice,
        {
            "clinical_question_node": "clinical_question_node",
            "topic_choice_node": "topic_choice_node",
            "attempt_analysis_node": "attempt_analysis_node",
            END: END,
        },
    )
    g.add_conditional_edges(
        "topic_choice_classifier",
        route_after_topic_choice,
        # Old "manager_agent" target now resolves to the parallelized
        # manager+retrieval combo node — same downstream behavior.
        {"manager_agent": "manager_and_retrieval"},
    )

    # Terminal nodes
    g.add_edge("deliver_response", END)
    g.add_edge("fallback_scaffold", END)
    g.add_edge("chitchat_response", END)
    g.add_edge("synthesis_assessor", END)
    g.add_edge("vlm_node", END)
    g.add_edge("study_node", END)

    return g.compile(checkpointer=_make_checkpointer())


def _make_checkpointer():
    """Build a long-lived checkpointer. Persists session state across requests
    so a Cloud Run instance handling turn N+1 picks up where turn N left off.

    Backend chosen by DB_BACKEND env var:
      - "sqlite" (default): file-based, persists across the process lifetime.
        On Cloud Run, the filesystem is ephemeral except /tmp, and /tmp is
        wiped between cold starts — fine for local dev, lossy in production.
      - "postgres": Cloud SQL Postgres or any psycopg-reachable Postgres,
        configured via DATABASE_URL. Survives cold starts and instance
        recycling. Same LangGraph API surface; existing checkpoints in
        SQLite are NOT migrated automatically.

    NOTE on sync vs async: SYNC saver in both branches. For FastAPI's
    /chat route the graph is invoked via asyncio.to_thread (see api/main.py),
    not graph.astream_events. The async equivalents need an event loop at
    construction time which conflicts with module-level compilation.
    """
    import os
    backend = os.getenv("DB_BACKEND", "sqlite").lower()

    if backend == "postgres":
        # Cloud Run + Cloud SQL: DATABASE_URL points at the unix-socket
        # connector — postgresql://user:pass@/dbname?host=/cloudsql/conn-name
        # Locally with the Cloud SQL Auth Proxy: host=127.0.0.1:5432.
        from psycopg_pool import ConnectionPool
        from langgraph.checkpoint.postgres import PostgresSaver
        pool = ConnectionPool(
            conninfo=os.environ["DATABASE_URL"],
            max_size=20,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        saver = PostgresSaver(pool)
        saver.setup()  # idempotent — creates checkpoint tables on first run
        return saver

    # ── SQLite path (default, local dev) ──────────────────────────────────
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    import config
    os.makedirs(os.path.dirname(config.SESSIONS_DB_PATH), exist_ok=True)
    # check_same_thread=False because uvicorn/FastAPI may serve requests
    # from different threads in the same process.
    conn = sqlite3.connect(config.SESSIONS_DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)


# Singleton for import
graph = build_graph()
