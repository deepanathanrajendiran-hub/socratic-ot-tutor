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

from graph._llm_client import Anthropic

import config
from ingest.vector_store import VectorStore
from ingest.reranker     import Reranker

# ── Module-level singletons ────────────────────────────────────────────────────
_anthropic_client = None
_vs:       VectorStore | None = None
_reranker: Reranker     | None = None


def get_anthropic():
    """Lazy-init so importing this module doesn't fail when API keys aren't
    yet loaded (e.g. tests that don't make API calls, or eager imports
    before .env is read)."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = Anthropic()
    return _anthropic_client


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
    """Embed with search_query prefix (nomic asymmetric retrieval).

    Delegates to retrieval.embedder.embed_query, which selects the backend
    (transformers for prod / Cloud Run, ollama for dev) via config.EMBED_BACKEND.
    """
    from retrieval.embedder import embed_query as _embed_query_external
    return _embed_query_external(text)


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
    response = get_anthropic().messages.create(
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
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Treat parse failure as CORRECT (skip refinement) and tag the result
        # so the audit log can distinguish a genuine CORRECT from a fallback.
        # Refinement-on-parse-failure was wasteful: extra LLM call + extra
        # ChromaDB roundtrip with a guessed-up query, indistinguishable from
        # genuine AMBIGUOUS in the logs.
        return {
            "score":        1.0,
            "decision":     "CORRECT",
            "reason":       "JSON parse failed — treating as CORRECT, no refinement",
            "parse_failed": True,
        }

    # Apply Python guards so the LLM's text label can't drift away from the
    # numeric thresholds in config.py. The score is the source of truth;
    # the LLM's "decision" string is advisory.
    score = float(parsed.get("score", 0.5))
    if score >= config.CRAG_CORRECT_THRESHOLD:
        parsed["decision"] = "CORRECT"
    elif score <= config.CRAG_INCORRECT_THRESHOLD:
        parsed["decision"] = "INCORRECT"
    else:
        parsed["decision"] = "AMBIGUOUS"
    parsed.setdefault("parse_failed", False)
    return parsed


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

    # ── Step 3: (no cosine-stage boost — moved to logit stage in reranker) ───
    # Personalisation lives in one place: the cross-encoder logit boost
    # (see ingest/reranker.py). The wider TOP_K_RETRIEVE pool ensures
    # weak-topic chunks ranked 11-15 by cosine still reach the reranker.

    # ── Step 4: CRAG evaluation ───────────────────────────────────────────────
    eval_result   = _evaluate_retrieval(expanded, results)
    crag_decision = eval_result["decision"]
    parse_failed  = eval_result.get("parse_failed", False)
    refined       = False

    # Diagnostic logging — surfaces what the evaluator saw when it
    # returned INCORRECT (often the cause of degraded teacher output
    # in production). Goes to stderr so it appears in Render logs.
    import sys
    print(
        f"[crag] decision={crag_decision} score={eval_result.get('score')} "
        f"query={expanded!r} top_chunks="
        + str([
            f"{r['section_title'][:40]}|d={r['distance']:.3f}"
            for r in results[:3]
        ]),
        file=sys.stderr,
    )

    if crag_decision == "INCORRECT":
        # Don't bypass the reranker. The CRAG evaluator (Haiku LLM) is
        # noisy and sometimes flags on-topic retrievals as INCORRECT,
        # but the cross-encoder reranker reliably surfaces the right
        # chunks from the top-K cosine pool — e.g. for the query
        # "what's the gap between neurons?" the textbook chunk
        # "synapse is the gap between nerve cells" sits at cosine rank
        # ~12 but reranker score +5.5 (next best −3.2). Skipping rerank
        # was the prior bug: cosine top-3 (intro chunks) reached the
        # teacher instead of the gold chunk. Continue to rerank;
        # crag_decision stays "INCORRECT" so dashboards still flag it.
        # (2026-05-03)
        print(
            f"[crag] INCORRECT verdict — continuing to reranker anyway "
            f"(cross-encoder is more reliable than the LLM evaluator)",
            file=sys.stderr,
        )
        # Fall through to rerank step below.

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

    # ── Step 7: Fetch full section texts (dedup by section_id) ───────────────
    # Multiple top-ranked chunks can belong to the same section (different
    # anchor positions inside one passage). Returning the full section text
    # once per chunk would feed identical copies to the LLM and clutter the
    # architecture dashboard. Keep the first occurrence per section, in
    # rerank order.
    section_texts: list[str] = []
    seen_sections: set[str] = set()
    for r in reranked:
        sec_id = r.get("section_id", "")
        if sec_id and sec_id in seen_sections:
            continue
        full_text = r.get("full_section_text", "")
        if not full_text:
            try:
                full_text = vs.get_full_section(r["section_id"])
            except ValueError:
                full_text = r["text"]
        section_texts.append(full_text)
        if sec_id:
            seen_sections.add(sec_id)

    # ── Step 8: Build and save audit log ─────────────────────────────────────
    crag_log = {
        "query":          query,
        "search_query":   search_query,
        "expanded_query": expanded,
        "crag_decision":  crag_decision,
        "crag_score":     eval_result["score"],
        "refined":        refined,
        "out_of_scope":   False,
        "parse_failed":   parse_failed,
        "top_sections":   [r.get("section_title", "") for r in reranked],
        "rerank_log":     rerank_log,
    }
    _append_log(crag_log)

    return reranked, section_texts, crag_log
