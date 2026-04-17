"""
ingest/reranker.py — cross-encoder reranking for retrieved chunks.

Why a reranker?
────────────────
Vector search (Step 6) uses approximate nearest-neighbour lookup on
nomic-embed-text embeddings.  Cosine similarity sees token-level overlap —
so "funny bone" matches "bone remodeling" before "ulnar nerve" because
"bone" appears in both query and section title.

The cross-encoder reads the FULL (query, document) pair and scores semantic
relevance.  It knows that "funny bone" is about the ulnar nerve hitting the
medial epicondyle — not about osteoclast activity.  It re-ranks the top-10
cosine candidates and returns only the top-3 that truly answer the question.

Model
──────
cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~85 MB, cached after first download
  - Returns logit scores (range roughly −12 to +5)
  - Higher score = more relevant (opposite convention from cosine distance)
  - Warm-up first call ~5 s, subsequent batches ~250 ms for 10 pairs

Pipeline position
──────────────────
  VectorStore.query()   → 10 medium chunks (cosine, fast)
  boost_by_weak_topics  → optional re-order before reranking
  Reranker.rerank()     → top-3 semantically relevant chunks
  VectorStore.get_large_chunk() → full generation context for LLM
"""

import time
from typing import Any

from sentence_transformers import CrossEncoder

import config

# Default model — changeable only in config.py (CLAUDE.md rule #3)
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Cross-encoder reranker wrapping sentence-transformers CrossEncoder.

    Usage:
        reranker = Reranker()
        reranked = reranker.rerank(query, vs_results, top_k=3)
    """

    # ── init ───────────────────────────────────────────────────────────────────

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        """
        Load the CrossEncoder model.  Downloads ~85 MB on first run,
        then uses the HuggingFace local cache.

        The 'UNEXPECTED position_ids' warning from newer sentence-transformers
        is benign — it reflects an architecture note, not a loading error.
        """
        self.model_name = model_name
        t0 = time.time()
        self._model: CrossEncoder = CrossEncoder(model_name)
        elapsed = time.time() - t0
        print(f"Reranker loaded: {model_name}  ({elapsed:.1f}s)")

    # ── rerank ────────────────────────────────────────────────────────────────

    def rerank(
        self,
        query:   str,
        results: list[dict],
        top_k:   int = config.TOP_K_RERANK, weak_topics: list[str] | None = None,
    ) -> list[dict]:
        if not results:
            return []

        # Build (query, doc_text) pairs and batch score
        pairs      = [(query, r["text"]) for r in results]
        raw_scores = self._model.predict(pairs)   # numpy float32 array

        # Build weak-topic set for O(1) lookup
        weak_set = {t.lower().strip() for t in weak_topics} if weak_topics else set()

        # Attach score + apply weak-topic logit boost before sorting
        scored = []
        for result, score in zip(results, raw_scores):
            r     = dict(result)
            logit = float(score)
            if weak_set and any(
                t in r.get("section_title", "").lower() for t in weak_set
            ):
                logit += config.WEAK_TOPIC_LOGIT_BOOST
                r["weak_topic_boosted"] = True
            else:
                r["weak_topic_boosted"] = False
            r["rerank_score"] = logit
            scored.append(r)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]

    # ── rerank_with_logging ────────────────────────────────────────────────────

    def rerank_with_logging(
        self,
        query:   str,
        results: list[dict],
        top_k:   int = config.TOP_K_RERANK, weak_topics: list[str] | None = None,
    ) -> tuple[list[dict], dict]:
        if not results:
            empty_log: dict[str, Any] = {
                "query":             query,
                "pre_rerank_order":  [],
                "post_rerank_order": [],
                "top_k_returned":    top_k,
                "order_changed":     False,
            }
            return [], empty_log

        # Pre-rerank snapshot (original cosine order)
        pre_order = [
            {
                "rank":          i + 1,
                "id":            r["id"],
                "section_title": r["section_title"],
                "distance":      r["distance"],
            }
            for i, r in enumerate(results)
        ]

        # Batch score all (query, chunk) pairs
        pairs      = [(query, r["text"]) for r in results]
        raw_scores = self._model.predict(pairs)

        # Build weak-topic set for O(1) lookup
        weak_set = {t.lower().strip() for t in weak_topics} if weak_topics else set()

        # Attach score + apply weak-topic logit boost before sorting
        scored = []
        for result, score in zip(results, raw_scores):
            r     = dict(result)
            logit = float(score)
            if weak_set and any(
                t in r.get("section_title", "").lower() for t in weak_set
            ):
                logit += config.WEAK_TOPIC_LOGIT_BOOST
                r["weak_topic_boosted"] = True
            else:
                r["weak_topic_boosted"] = False
            r["rerank_score"] = logit
            scored.append(r)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Post-rerank snapshot (all results, new order, boost flag visible)
        post_order = [
            {
                "rank":               i + 1,
                "id":                 r["id"],
                "section_title":      r["section_title"],
                "rerank_score":       r["rerank_score"],
                "weak_topic_boosted": r["weak_topic_boosted"],
                "distance":           r["distance"],
            }
            for i, r in enumerate(scored)
        ]

        log: dict[str, Any] = {
            "query":             query,
            "pre_rerank_order":  pre_order,
            "post_rerank_order": post_order,
            "top_k_returned":    top_k,
            "order_changed":     (
                bool(pre_order) and bool(post_order)
                and pre_order[0]["id"] != post_order[0]["id"]
            ),
        }

        return scored[:top_k], log
