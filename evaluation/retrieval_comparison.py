"""
evaluation/retrieval_comparison.py — Experiment B: Retrieval Quality

Compares two retrieval modes on 10 anatomy queries from well-covered chapters.

Mode 1 — standard_rag:
  ChromaDB cosine top-3, no synonym expansion, no CRAG eval, no reranking.
  This is the baseline: a plain bi-encoder retrieval system.

Mode 2 — full_pipeline:
  OT synonym expansion → ChromaDB top-10 → CRAG eval → cross-encoder rerank → top-3.
  This is our approach.

Metrics:
  - CRAG decision distribution (CORRECT / AMBIGUOUS→REFINED / INCORRECT)
  - Top rerank score (cross-encoder logit, full pipeline only)
  - Top cosine distance (standard RAG only; lower = more similar in cosine space)
  - manual_relevant: fill after running (1 = relevant, 0 = not relevant)
    — add these to the JSON and re-run with --summarize to get final accuracy

Output: table to stdout + JSON to evaluation/results/retrieval_results.json

Usage:
    PYTHONPATH=. python3 evaluation/retrieval_comparison.py
    PYTHONPATH=. python3 evaluation/retrieval_comparison.py --summarize
"""

import json
import os
import sys

import ollama

import config
from ingest.vector_store import VectorStore
from retrieval.crag import corrective_retrieve, get_vs

# ── 10 queries from well-covered chapters (avoiding peripheral nerve gap) ─────
# Scoped to Ch 9–16 per CLAUDE.md data gap warning.
QUERIES = [
    "What is the structure and function of a motor neuron?",
    "How is the spinal cord gray matter organized into horns?",
    "What are the components of a reflex arc?",
    "How does the autonomic nervous system differ from the somatic nervous system?",
    "What is the role of the cerebellum in coordinating voluntary movement?",
    "How do mechanoreceptors transduce touch stimuli into nerve signals?",
    "What is the thalamus role in relaying sensory information to the cortex?",
    "How does the primary motor cortex control voluntary movement?",
    "What is the difference between flexor and extensor muscles at the elbow joint?",
    "How does the synaptic cleft facilitate neurotransmitter signaling?",
]


# ── Embed helper (same model as production for fair comparison) ───────────────

def _embed(query: str) -> list[float]:
    """Embed with search_query prefix — identical to crag._embed_query."""
    response = ollama.embed(
        model="nomic-embed-text",
        input=f"search_query: {query}",
    )
    return response["embeddings"][0]


# ── Mode 1: Standard RAG ──────────────────────────────────────────────────────

def run_standard_rag(query: str, vs: VectorStore) -> dict:
    """
    Plain cosine search — no synonym expansion, no CRAG eval, no reranking.
    Represents a typical off-the-shelf RAG retriever.
    """
    embedding = _embed(query)
    raw = vs.chunks_col.query(
        query_embeddings=[embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    top_meta = raw["metadatas"][0]
    top_dist = raw["distances"][0]

    return {
        "mode":             "standard_rag",
        "query":            query,
        "top_sections":     [m.get("section_title", "—") for m in top_meta],
        "top_distances":    [round(d, 4) for d in top_dist],
        "top_score":        round(top_dist[0], 4) if top_dist else None,
        "crag_decision":    "N/A",
        "crag_score":       None,
        "out_of_scope":     None,
        "manual_relevant":  None,   # fill after running: 1 = relevant, 0 = not
    }


# ── Mode 2: Full pipeline ─────────────────────────────────────────────────────

def run_full_pipeline(query: str) -> dict:
    """
    Full corrective_retrieve(): synonym expand → CRAG eval → rerank → top-3.
    """
    reranked, section_texts, crag_log = corrective_retrieve(query)

    top_sections = [r.get("section_title", "—") for r in reranked]
    top_scores   = [round(r.get("rerank_score", 0), 4) for r in reranked]

    return {
        "mode":             "full_pipeline",
        "query":            query,
        "top_sections":     top_sections,
        "top_scores":       top_scores,
        "top_score":        top_scores[0] if top_scores else None,
        "crag_decision":    crag_log.get("crag_decision", "—"),
        "crag_score":       crag_log.get("crag_score", None),
        "out_of_scope":     crag_log.get("out_of_scope", False),
        "manual_relevant":  None,   # fill after running: 1 = relevant, 0 = not
    }


# ── Summarize helper (run after manual annotation) ────────────────────────────

def summarize(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        results = json.load(f)

    print("\n" + "=" * 60)
    print("EXPERIMENT B — Final Summary (with manual relevance)")
    print("=" * 60)

    for mode in ("standard_rag", "full_pipeline"):
        rows = [r for r in results if r.get("mode") == mode and "error" not in r]
        if not rows:
            continue
        annotated = [r for r in rows if r.get("manual_relevant") is not None]
        if annotated:
            relevant = sum(r["manual_relevant"] for r in annotated)
            print(f"  {mode:<16}  relevant={relevant}/{len(annotated)} "
                  f"({relevant/len(annotated):.0%})")
        else:
            print(f"  {mode:<16}  no manual annotations yet")

        if mode == "full_pipeline":
            correct  = sum(1 for r in rows if r.get("crag_decision") == "CORRECT")
            ambig    = sum(1 for r in rows if "AMBIGUOUS" in str(r.get("crag_decision", "")))
            incorrect= sum(1 for r in rows if r.get("crag_decision") == "INCORRECT")
            oos      = sum(1 for r in rows if r.get("out_of_scope"))
            print(f"    CRAG: CORRECT={correct}  AMBIGUOUS/REFINED={ambig}  "
                  f"INCORRECT={incorrect}  out_of_scope={oos}")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_experiment():
    vs = get_vs()
    all_results = []

    print("\n" + "=" * 80)
    print("EXPERIMENT B — Retrieval Quality: Standard RAG vs. Full Pipeline")
    print("10 queries from well-covered chapters (Ch 9–16)")
    print("=" * 80)

    for i, query in enumerate(QUERIES, 1):
        print(f"\nQ{i}: {query}")
        print("─" * 70)

        # Standard RAG
        try:
            std = run_standard_rag(query, vs)
            all_results.append(std)
            secs  = " | ".join(std["top_sections"][:2])
            dists = ", ".join(str(d) for d in std["top_distances"][:2])
            print(f"  [standard_rag  ]  dist={dists}  sections={secs[:55]}")
        except Exception as exc:
            print(f"  [standard_rag  ]  ERROR: {exc}", file=sys.stderr)
            all_results.append({"mode": "standard_rag", "query": query, "error": str(exc)})

        # Full pipeline
        try:
            full = run_full_pipeline(query)
            all_results.append(full)
            secs  = " | ".join(full["top_sections"][:2])
            crag  = full["crag_decision"]
            score = full["top_score"]
            oos   = " OUT-OF-SCOPE ⚠" if full.get("out_of_scope") else ""
            print(f"  [full_pipeline ]  CRAG={crag:<22} score={score}  "
                  f"sections={secs[:40]}{oos}")
        except Exception as exc:
            print(f"  [full_pipeline ]  ERROR: {exc}", file=sys.stderr)
            all_results.append({"mode": "full_pipeline", "query": query, "error": str(exc)})

    # ── Summary ───────────────────────────────────────────────────────────────
    std_rows  = [r for r in all_results if r.get("mode") == "standard_rag" and "error" not in r]
    full_rows = [r for r in all_results if r.get("mode") == "full_pipeline" and "error" not in r]

    print("\n" + "=" * 70)
    print("EXPERIMENT B — Summary")
    print("=" * 70)

    if full_rows:
        correct   = sum(1 for r in full_rows if r.get("crag_decision") == "CORRECT")
        ambig     = sum(1 for r in full_rows if "AMBIGUOUS" in str(r.get("crag_decision", "")))
        incorrect = sum(1 for r in full_rows if r.get("crag_decision") == "INCORRECT")
        oos       = sum(1 for r in full_rows if r.get("out_of_scope"))
        n         = len(full_rows)
        print(f"  Full pipeline CRAG decisions ({n} queries):")
        print(f"    CORRECT={correct} ({correct/n:.0%})  "
              f"AMBIGUOUS/REFINED={ambig} ({ambig/n:.0%})  "
              f"INCORRECT={incorrect} ({incorrect/n:.0%})  "
              f"out_of_scope={oos}")
        scores = [r["top_score"] for r in full_rows if r.get("top_score") is not None]
        if scores:
            print(f"    Avg top rerank score: {sum(scores)/len(scores):.4f}")

    if std_rows:
        dists = [r["top_distances"][0] for r in std_rows if r.get("top_distances")]
        if dists:
            print(f"  Standard RAG avg top cosine distance: {sum(dists)/len(dists):.4f}")

    print("\n  Next step: open evaluation/results/retrieval_results.json,")
    print("  set manual_relevant=1/0 for each query, then re-run with --summarize")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join("evaluation", "results", "retrieval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved → {out_path}")


if __name__ == "__main__":
    if "--summarize" in sys.argv:
        path = os.path.join("evaluation", "results", "retrieval_results.json")
        summarize(path)
    else:
        run_experiment()
