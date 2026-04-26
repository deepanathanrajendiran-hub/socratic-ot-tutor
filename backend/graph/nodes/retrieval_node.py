"""
graph/nodes/retrieval_node.py

Thin wrapper around corrective_retrieve().
Called after manager_agent extracts current_concept.

Invokes the full CRAG pipeline:
  synonym expand → embed → ChromaDB search → CRAG eval → rerank → threshold

On failure (ollama not running, ChromaDB empty): returns empty chunks so the
graph degrades gracefully rather than crashing — the teacher will respond with
"(no content retrieved)" and Dean will flag the grounding issue.

Input:  current_concept, weak_topics, messages (for turn-aware query)
Output: retrieved_chunks, chunk_sources, crag_decision
"""

import config
from graph.state import GraphState


def _build_turn_query(state: GraphState) -> str:
    """Turn-aware query: concept-anchored, faceted by turn number.
    Extracts last student message from state.messages and calls
    build_turn_query() with all four required positional arguments.
    """
    try:
        from retrieval.turn_aware import build_turn_query
        from langchain_core.messages import HumanMessage
        messages = state.get("messages", [])
        last_student = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
            "",
        )
        return build_turn_query(
            original_query=state.get("current_concept", ""),
            student_response=last_student,
            target_concept=state.get("current_concept", ""),
            turn_count=state.get("turn_count", 0),
        )
    except Exception:
        return state.get("current_concept", "")


def retrieval_node(state: GraphState) -> dict:
    concept = state.get("current_concept", "")
    weak_topics = state.get("weak_topics", [])

    try:
        from retrieval.crag import corrective_retrieve
        turn_query = _build_turn_query(state)
        reranked, section_texts, crag_log = corrective_retrieve(
            query=concept,
            weak_topics=weak_topics,
            turn_query=turn_query,
        )
        return {
            "retrieved_chunks": section_texts,
            "chunk_sources": [c.get("id", "") for c in reranked],
            "crag_decision": crag_log.get("crag_decision", ""),
        }
    except Exception as exc:
        # Graceful degradation: log and continue with empty chunks
        print(f"[retrieval_node] WARNING: retrieval failed — {exc}")
        return {
            "retrieved_chunks": [],
            "chunk_sources": [],
            "crag_decision": "FAILED",
        }
