"""
ingest/run_metadata_pipeline.py — Step 8 runner.

Builds:
  1. data/processed/diagram_chunk_links.json   (10 diagrams → chunk links)
  2. data/processed/concept_tags.json          (574 large chunks → concept labels)

Then runs 5 smoke tests (acceptance criteria).

Usage:
    python -m ingest.run_metadata_pipeline
"""

import json
import re

from ingest.metadata_builder import build_diagram_chunk_links, tag_large_chunks_with_concepts

# ── Step 1 ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Building diagram → chunk links...")
print("=" * 60)
result1 = build_diagram_chunk_links(
    diagram_metadata_path="data/raw/diagrams/grays/metadata.json",
    output_path="data/processed/diagram_chunk_links.json",
)
print(f"  Result: {result1}")

# ── Step 2 ─────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Step 2: Tagging large chunks with concept labels...")
print("=" * 60)
result2 = tag_large_chunks_with_concepts(
    chunks_dir="data/processed/chunks/",
    output_path="data/processed/concept_tags.json",
)
print(f"  Result: {result2}")

# ── Step 3: Smoke tests ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Step 3: Smoke tests...")
print("=" * 60)

# ── Test A: Every diagram has at least 1 chunk link ────────────────────────────
with open("data/processed/diagram_chunk_links.json") as f:
    diagrams = json.load(f)

all_linked = all(len(d["chunk_links"]) >= 1 for d in diagrams)
print(f"  [A] All diagrams linked: {all_linked}")
if not all_linked:
    missing = [d["filename"] for d in diagrams if len(d["chunk_links"]) < 1]
    print(f"      Missing links for: {missing}")

# ── Test B: Blind test diagrams are flagged correctly ─────────────────────────
blind = [d for d in diagrams if d["blind_test"]]
print(f"  [B] Blind test diagrams: {len(blind)} (expected 3)")
print(f"      Blind test files: {[d['filename'] for d in blind]}")

# ── Test C: Brachial plexus diagram links to nerve content ────────────────────
bp = next((d for d in diagrams if "brachial" in d["filename"]), None)
if bp:
    pc = bp["primary_concept"]
    has_nerve_or_plexus = "nerve" in pc.lower() or "plexus" in pc.lower()
    print(f"  [C] Brachial plexus primary concept: {pc!r}")
    print(f"      Contains 'nerve' or 'plexus': {has_nerve_or_plexus}")
else:
    print("  [C] ERROR: brachial plexus diagram not found")

# ── Test D: OT priority chunk count is reasonable (80–200) ───────────────────
with open("data/processed/concept_tags.json") as f:
    concepts = json.load(f)

ot = [c for c in concepts if c["OT_priority"]]
ot_count = len(ot)
in_range = 80 <= ot_count <= 200
print(f"  [D] OT priority chunks: {ot_count} out of {len(concepts)}  "
      f"(in range 80-200: {in_range})")

# ── Test E: Concept labels are clean (no leftover chapter number prefixes) ────
messy = [c for c in concepts if re.match(r"^\d+\.\d+", c["concept_label"])]
print(f"  [E] Concept labels with leftover chapter numbers: {len(messy)} (expected 0)")
if messy:
    print(f"      Examples: {[c['concept_label'] for c in messy[:3]]}")

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("ACCEPTANCE CRITERIA SUMMARY")
print("=" * 60)
criteria = {
    "A — All 10 diagrams have ≥1 chunk link":                 all_linked,
    "B — Exactly 3 blind test diagrams":                      len(blind) == 3,
    "C — Brachial plexus primary_concept has 'nerve'/'plexus'":
        bp is not None and ("nerve" in bp["primary_concept"].lower()
                            or "plexus" in bp["primary_concept"].lower()),
    "D — OT priority chunks between 80 and 200":              in_range,
    "E — Zero concept labels with leftover chapter numbers":  len(messy) == 0,
}
all_pass = True
for name, passed in criteria.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}  {name}")
    if not passed:
        all_pass = False

print()
if all_pass:
    print("All 5 criteria PASSED — Step 8 complete.")
else:
    print("Some criteria FAILED — see details above.")
