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
            domain=state.get("domain", config.DOMAIN),
        )
    except Exception:
        return state.get("current_concept", "")


def retrieval_node(state: GraphState) -> dict:
    concept = state.get("current_concept", "")
    weak_topics = state.get("weak_topics", [])

    # ── Per-concept cache ────────────────────────────────────────────────
    # On follow-up turns of the same Socratic loop, the locked concept
    # doesn't change — manager_agent is explicitly designed to resist
    # drift. CRAG therefore re-retrieves the same chunks every turn,
    # spending 2-7 s on grader + rerank for no information gain. Skip
    # when the cache is hot.
    #
    # Cache key: (current_concept). Hit when the concept matches what
    # the chunks were last retrieved for AND the chunks list is non-
    # empty. Miss → run CRAG and refresh the cache.
    #
    # Trade-off: turn_aware.py builds slightly different queries at
    # turn 0 / 1 / 2+ ("synapse" → "synapse anatomy location structure"
    # → "synapse function clinical significance OT"). Caching forfeits
    # those facets — but the chunks usually overlap heavily on the
    # same concept, and the latency win on a 6-8 s step is large.
    cached_concept = state.get("chunks_for_concept", "")
    cached_chunks  = state.get("retrieved_chunks", []) or []
    cache_disabled = bool(getattr(config, "RETRIEVAL_CACHE_DISABLE", False))
    if (
        not cache_disabled
        and concept
        and concept == cached_concept
        and cached_chunks
    ):
        # Cache hit: return the prior values. We re-emit them so the
        # trace event captures crag_decision (preserved from prior
        # retrieval) and downstream nodes see a non-empty chunk list.
        return {
            "retrieved_chunks": cached_chunks,
            "chunk_sources":    state.get("chunk_sources", []) or [],
            "crag_decision":    state.get("crag_decision", "CACHED"),
            "chunks_for_concept": concept,
        }

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
            "chunks_for_concept": concept,
        }
    except Exception as exc:
        # Graceful degradation: log and continue with empty chunks
        print(f"[retrieval_node] WARNING: retrieval failed — {exc}")
        return {
            "retrieved_chunks": [],
            "chunk_sources": [],
            "crag_decision": "FAILED",
            "chunks_for_concept": "",
        }
