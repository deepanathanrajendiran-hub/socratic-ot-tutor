"""
ingest/run_ingest_pipeline.py — single entry point for full v3 ingestion.

Run this after deleting the old chroma_db (or let it auto-delete).

Steps:
  1. Late-chunk all sections (late_chunker.py)
  2. Load into ChromaDB — one {domain}_chunks collection
  3. Verify chunk count
  4. Smoke test: corrective_retrieve() with 3 queries

Usage:
  PYTHONPATH=. python3 ingest/run_ingest_pipeline.py
"""

import json
import os
import shutil
import sys

import config
from ingest.late_chunker  import run_late_chunker
from ingest.vector_store  import VectorStore

SECTIONS_DIR = os.path.join(config.CHUNKS_DIR, "raw_sections")
CHUNKS_DIR   = config.CHUNKS_DIR
LATE_CHUNKS  = os.path.join(CHUNKS_DIR, f"late_chunks_{config.DOMAIN}.json")


def main() -> None:

    # ── Step 0: Wipe old ChromaDB ─────────────────────────────────────────────
    if os.path.exists(config.CHROMA_DIR):
        print(f"Deleting old ChromaDB at {config.CHROMA_DIR} …")
        shutil.rmtree(config.CHROMA_DIR)
        print("  Deleted. ✓")

    # ── Step 1: Late chunking ─────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("Step 1: Late Chunking")
    print("═" * 60)
    result = run_late_chunker(SECTIONS_DIR, CHUNKS_DIR)
    print(f"  Result: {result}")
    if result["failed"] > 0:
        print(f"  WARNING: {result['failed']} sections failed. Check logs above.")

    # ── Step 2: Load into ChromaDB ────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("Step 2: Loading into ChromaDB")
    print("═" * 60)
    vs          = VectorStore(config.CHROMA_DIR, config.DOMAIN)
    load_result = vs.load_from_late_chunks(LATE_CHUNKS)
    print(f"  Result: {load_result}")

    # ── Step 3: Verify count ──────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("Step 3: Verification")
    print("═" * 60)
    stats    = vs.get_collection_stats()
    print(f"  Collection stats: {stats}")

    with open(LATE_CHUNKS, encoding="utf-8") as f:
        expected = len(json.load(f))
    actual = stats["chunks_count"]

    assert actual == expected, \
        f"Count mismatch: expected {expected}, got {actual}"
    print(f"  Count verified: {actual} chunks ✓")

    # ── Step 4: Smoke test ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("Step 4: Smoke Test — corrective_retrieve()")
    print("═" * 60)

    from retrieval.crag import corrective_retrieve

    test_queries = [
        "What nerve causes the funny bone sensation?",
        "rotator cuff tear OT intervention",
        "carpal tunnel syndrome median nerve compression",
    ]

    all_passed = True
    for q in test_queries:
        print(f"\n  Query: {q!r}")
        reranked, sections, log = corrective_retrieve(query=q)
        decision = log.get("crag_decision", "UNKNOWN")
        print(f"  CRAG decision: {decision}")

        if log.get("out_of_scope"):
            print(f"  ⚠  Out of scope — redirect node would fire")
            # Still a valid result — not a failure
        else:
            print(f"  Top section: {reranked[0]['section_title']!r}")
            print(f"  Section text length: {len(sections[0])} chars")
            if len(sections[0]) == 0:
                print(f"  ✗  FAIL: section text is empty")
                all_passed = False
            else:
                print(f"  ✓  PASS")

    # ── Criterion checks ──────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("Acceptance Criteria")
    print("═" * 60)

    # Re-run the primary query for criteria checks
    reranked, sections, log = corrective_retrieve(
        query="What nerve causes the funny bone sensation?"
    )

    c1 = len(reranked) > 0
    c2 = log.get("crag_decision") in ("CORRECT", "AMBIGUOUS→REFINED", "AMBIGUOUS")
    c3 = len(sections) > 0 and len(sections[0]) > 0
    log_path = os.path.join(config.PROCESSED_DIR, "retrieval_logs.jsonl")
    c4 = os.path.exists(log_path)
    if c4:
        with open(log_path) as f:
            last = json.loads(f.readlines()[-1])
        c4 = "crag_decision" in last

    criteria = {
        "1 — corrective_retrieve returns non-empty reranked list": c1,
        "2 — crag_decision is CORRECT or AMBIGUOUS (not INCORRECT)": c2,
        "3 — section_texts[0] is non-empty string": c3,
        "4 — retrieval_logs.jsonl has crag_decision field": c4,
    }
    for name, passed in criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All criteria PASSED — Step 8b-8i complete.")
        print("Next: Step 9 — ingest/question_bank_builder.py")
    else:
        print("Some criteria FAILED — see details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
