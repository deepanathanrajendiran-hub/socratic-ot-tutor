"""
retrieval/crag.py — Corrective Retrieval-Augmented Generation.

Single entry point (corrective_retrieve) for all LangGraph nodes.
ABSOLUTE RULE: No graph node imports chromadb or calls vs.query() directly.

Pipeline:
  1. OT synonym expansion (retrieval/ot_synonyms.py)
  2. Embed expanded query (Ollama nomic-embed-text, search_query: prefix)
  3. ChromaDB vector search → top-10 late chunks
  4. Weak topic boost (session memory)
  5. CRAG evaluation (claude-haiku, prompts/crag_evaluator.txt)
     → CORRECT:   proceed
     → AMBIGUOUS: refine query once, re-retrieve, take better result
     → INCORRECT: return out_of_scope=True → redirect node handles it
  6. Cross-encoder rerank → top-3
  7. Confidence threshold check (OUT_OF_SCOPE_THRESHOLD)
  8. Fetch full section texts from chunk metadata
  9. Log to retrieval_logs.jsonl (includes crag_decision)

Returns:
  reranked      — list[dict]  top-3 chunk dicts with rerank scores
  section_texts — list[str]   full section texts for LLM generation
  crag_log      — dict        full audit trail (paper/demo panel)
"""

import json
import os

from anthropic import Anthropic
import ollama

import config
from ingest.vector_store import VectorStore
from ingest.reranker     import Reranker

_anthropic_client = Anthropic()

# ── Module-level singletons ────────────────────────────────────────────────────
_vs:       VectorStore | None = None
_reranker: Reranker     | None = None


def get_vs() -> VectorStore:
    global _vs
    if _vs is None:
        _vs = VectorStore(config.CHROMA_DIR, config.DOMAIN)
    return _vs


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_prompt(name: str) -> str:
    path = os.path.join(config.PROMPTS_DIR, name)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _embed_query(text: str) -> list[float]:
    """Embed with search_query prefix (nomic asymmetric retrieval)."""
    response = ollama.embed(
        model="nomic-embed-text",
        input=f"search_query: {text}",
    )
    return response["embeddings"][0]


def _vector_search(embedding: list[float]) -> list[dict]:
    """Raw ChromaDB cosine search. Returns structured result dicts."""
    vs  = get_vs()
    raw = vs.chunks_col.query(
        query_embeddings=[embedding],
        n_results=config.TOP_K_RETRIEVE,
        include=["documents", "metadatas", "distances"],
    )
    results = []
    for i in range(len(raw["ids"][0])):
        meta = raw["metadatas"][0][i]
        results.append({
            "id":               raw["ids"][0][i],
            "text":             raw["documents"][0][i],
            "distance":         raw["distances"][0][i],
            "section_id":       meta.get("section_id", ""),
            "section_title":    meta.get("section_title", ""),
            "chapter_num":      meta.get("chapter_num", ""),
            "full_section_text": meta.get("full_section_text",
                                          raw["documents"][0][i]),
        })
    return results


def _evaluate_retrieval(query: str, results: list[dict]) -> dict:
    prompt_template = _load_prompt("crag_evaluator.txt")
    chunks_preview  = "\n\n".join(
        f"[{r['section_title']}]: {r['text'][:400]}"
        for r in results[:3]
    )
    response = _anthropic_client.messages.create(
        model=config.FAST_MODEL,
        max_tokens=config.CRAG_EVAL_MAX_TOKENS,
        messages=[{
            "role":    "user",
            "content": prompt_template.format(
                query=query,
                chunks=chunks_preview,
            ),
        }],
    )
    raw = response.content[0].text.strip()
    # Strip markdown code fences that haiku often wraps around JSON
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        raw = raw.rsplit("```", 1)[0].strip()
    # Extract JSON object if there's surrounding text
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "score":            0.5,
            "decision":         "AMBIGUOUS",
            "refinement_query": f"{query} anatomy occupational therapy",
            "reason":           "JSON parse failed — defaulting to AMBIGUOUS",
        }


def _append_log(log: dict) -> None:
    path = os.path.join(config.PROCESSED_DIR, "retrieval_logs.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")


# ── Main entry point ───────────────────────────────────────────────────────────

def corrective_retrieve(
    query:        str,
    weak_topics:  list[str] | None = None,
    turn_query:   str | None       = None,
) -> tuple[list[dict], list[str], dict]:
    from retrieval.ot_synonyms import expand_query

    vs       = get_vs()
    reranker = get_reranker()

    # ── Step 1: Determine search query ────────────────────────────────────────
    # turn_query is concept-anchored (e.g. "ulnar nerve anatomy location structure")
    # and is used for both retrieval and reranking so the whole pipeline is
    # consistent. When no turn_query is provided, fall back to raw query.
    search_query = turn_query if turn_query else query
    expanded     = expand_query(search_query)

    # ── Step 2: Embed and search ──────────────────────────────────────────────
    embedding = _embed_query(expanded)
    results   = _vector_search(embedding)

    # ── Step 3: Weak topic boost (pre-CRAG, distance-based) ──────────────────
    # Boosts weak-topic chunks into CRAG's evaluation window so CRAG sees
    # the right content. A second boost is applied at rerank time (logit-based).
    if weak_topics:
        results = vs.boost_and_rerank_by_weak_topics(
            results, weak_topics, config.WEAK_TOPIC_BOOST
        )

    # ── Step 4: CRAG evaluation ───────────────────────────────────────────────
    eval_result   = _evaluate_retrieval(expanded, results)
    crag_decision = eval_result["decision"]
    refined       = False

    if crag_decision == "INCORRECT":
        log = {
            "query":         query,
            "expanded":      expanded,
            "crag_decision": "INCORRECT",
            "score":         eval_result["score"],
            "out_of_scope":  True,
            "reason":        eval_result.get("reason", ""),
        }
        _append_log(log)
        return [], [], log

    if crag_decision == "AMBIGUOUS":
        refined_query   = eval_result.get(
            "refinement_query",
            f"{query} anatomy occupational therapy",
        )
        refined_embed   = _embed_query(refined_query)
        refined_results = _vector_search(refined_embed)
        re_eval         = _evaluate_retrieval(refined_query, refined_results)
        if re_eval["score"] > eval_result["score"]:
            results       = refined_results
            crag_decision = "AMBIGUOUS→REFINED"
            refined       = True

    # ── Step 5: Cross-encoder rerank + weak-topic logit boost ────────────────
    # Use search_query (concept-anchored) rather than the raw student query so
    # the reranker scores chunk-to-concept relevance. This prevents meta-questions
    # like "how does this apply to OT?" from triggering the low-confidence guard
    # when the retrieved OT-content chunks are genuinely relevant.
    reranked, rerank_log = reranker.rerank_with_logging(
        search_query, results, top_k=config.TOP_K_RERANK, weak_topics=weak_topics
    )

    # ── Step 6: Confidence threshold ─────────────────────────────────────────
    if (not reranked or
            reranked[0].get("rerank_score", 0) < config.OUT_OF_SCOPE_THRESHOLD):
        log = {
            "query":         query,
            "expanded":      expanded,
            "crag_decision": "LOW_CONFIDENCE",
            "max_score":     reranked[0].get("rerank_score", 0) if reranked else 0,
            "out_of_scope":  True,
        }
        _append_log(log)
        return [], [], log

    # ── Step 7: Fetch full section texts ──────────────────────────────────────
    section_texts: list[str] = []
    for r in reranked:
        full_text = r.get("full_section_text", "")
        if not full_text:
            try:
                full_text = vs.get_full_section(r["section_id"])
            except ValueError:
                full_text = r["text"]
        section_texts.append(full_text)

    # ── Step 8: Build and save audit log ─────────────────────────────────────
    crag_log = {
        "query":          query,
        "search_query":   search_query,
        "expanded_query": expanded,
        "crag_decision":  crag_decision,
        "crag_score":     eval_result["score"],
        "refined":        refined,
        "out_of_scope":   False,
        "top_sections":   [r.get("section_title", "") for r in reranked],
        "rerank_log":     rerank_log,
    }
    _append_log(crag_log)

    return reranked, section_texts, crag_log
