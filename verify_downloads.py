"""
verify_downloads.py — checks all expected data files exist and are valid.
Run from the socratic-ot project root:
    python3 verify_downloads.py
"""

import json
import os

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"

results = []   # (label, passed, detail)


def check(label: str, passed: bool, detail: str = ""):
    tag = PASS if passed else FAIL
    msg = f"{tag}  {label}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)
    results.append((label, passed, detail))


# ── 1. Textbooks ──────────────────────────────────────────────────────────────

print("\n═══ TEXTBOOKS ══════════════════════════════════════════════════════════")

textbooks = [
    ("OpenStax A&P 2e",     "data/raw/textbooks/openStax_AP2e.pdf",       40),
    ("OpenStax Physics V1", "data/raw/physics/openStax_physics_v1.pdf",   25),
]

for name, path, min_mb in textbooks:
    exists = os.path.exists(path)
    if not exists:
        check(name, False, f"MISSING: {path}")
        continue
    size_mb = os.path.getsize(path) / 1e6
    ok = size_mb >= min_mb
    check(name, ok, f"{size_mb:.1f} MB  (min {min_mb} MB)")


# ── 2. Diagrams ───────────────────────────────────────────────────────────────

print("\n═══ GRAY'S ANATOMY DIAGRAMS ════════════════════════════════════════════")

diagrams = [
    ("brachial_plexus",          "data/raw/diagrams/grays/brachial_plexus.png",          20),
    ("ulnar_nerve_hand",         "data/raw/diagrams/grays/ulnar_nerve_hand.png",          10),
    ("median_nerve",             "data/raw/diagrams/grays/median_nerve.png",              10),
    ("radial_nerve",             "data/raw/diagrams/grays/radial_nerve.png",              20),
    ("spinal_cord_cross_section","data/raw/diagrams/grays/spinal_cord_cross_section.png", 10),
    ("dermatomes_upper_limb",    "data/raw/diagrams/grays/dermatomes_upper_limb.png",     10),
    ("rotator_cuff",             "data/raw/diagrams/grays/rotator_cuff.png",              10),
    ("carpal_tunnel",            "data/raw/diagrams/grays/carpal_tunnel.png",             10),
    ("motor_cortex_homunculus",  "data/raw/diagrams/grays/motor_cortex_homunculus.png",  200),
    ("hand_intrinsic_muscles",   "data/raw/diagrams/grays/hand_intrinsic_muscles.png",    10),
]

for name, path, min_kb in diagrams:
    exists = os.path.exists(path)
    if not exists:
        check(name, False, f"MISSING: {path}")
        continue
    size_kb = os.path.getsize(path) / 1e3
    ok = size_kb >= min_kb
    check(name, ok, f"{size_kb:.1f} KB  (min {min_kb} KB)")


# ── 3. Metadata JSON ──────────────────────────────────────────────────────────

print("\n═══ METADATA JSON ══════════════════════════════════════════════════════")

meta_path = "data/raw/diagrams/grays/metadata.json"
exists = os.path.exists(meta_path)
if not exists:
    check("metadata.json exists", False, f"MISSING: {meta_path}")
else:
    check("metadata.json exists", True, meta_path)
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        check("metadata.json is valid JSON", True, f"{len(meta)} entries")

        expected_files = {e["filename"] for e in meta}
        actual_files   = {os.path.basename(d[1]) for d in diagrams}
        missing_in_meta = actual_files - expected_files
        extra_in_meta   = expected_files - actual_files

        check("All diagrams covered in metadata",
              not missing_in_meta,
              f"missing from JSON: {missing_in_meta}" if missing_in_meta else "all 10 present")

        blind_count = sum(1 for e in meta if e.get("blind_test"))
        check("Blind-test diagram count",
              blind_count == 3,
              f"{blind_count} marked blind_test=true (expected 3)")

        required_keys = {"filename", "local_path", "structure", "region",
                         "ot_relevance", "chunk_tags", "blind_test"}
        schema_ok = all(required_keys.issubset(e.keys()) for e in meta)
        check("All entries have required keys", schema_ok,
              str(required_keys) if not schema_ok else "OK")

    except json.JSONDecodeError as e:
        check("metadata.json is valid JSON", False, str(e))


# ── 4. Directory skeleton ─────────────────────────────────────────────────────

print("\n═══ DIRECTORY SKELETON ═════════════════════════════════════════════════")

dirs = [
    "data/raw/textbooks",
    "data/raw/diagrams/grays",
    "data/raw/diagrams/medpix",
    "data/raw/ot_specific",
    "data/raw/physics",
    "data/processed/chunks/raw_sections",
    "data/processed/chroma_db",
    "data/processed/question_bank",
]

for d in dirs:
    check(d, os.path.isdir(d))


# ── 5. Summary ────────────────────────────────────────────────────────────────

print("\n═══ TOTAL DATA ON DISK ═════════════════════════════════════════════════")

total_bytes = 0
for root, _, files in os.walk("data"):
    for fname in files:
        total_bytes += os.path.getsize(os.path.join(root, fname))
print(f"  Total data size: {total_bytes/1e6:.1f} MB")

print("\n═══ RESULT SUMMARY ═════════════════════════════════════════════════════")

passed  = [r for r in results if r[1]]
failed  = [r for r in results if not r[1]]
total   = len(results)

print(f"  {len(passed)}/{total} checks passed")

if failed:
    print("\n  FAILED CHECKS:")
    for label, _, detail in failed:
        print(f"    ✗  {label}: {detail}")
else:
    print("  All checks passed — ready to build ingest/parse_pdf.py")

# Textbook manual-download reminder
tb_paths = ["data/raw/textbooks/openStax_AP2e.pdf",
            "data/raw/physics/openStax_physics_v1.pdf"]
missing_tbs = [p for p in tb_paths if not os.path.exists(p)]
if missing_tbs:
    print("\n  ⚠  MANUAL DOWNLOAD REQUIRED for OpenStax PDFs (CDN blocks scripted downloads):")
    print("     1. Go to https://openstax.org/details/books/anatomy-and-physiology-2e")
    print("        Click 'Download a PDF' → save as data/raw/textbooks/openStax_AP2e.pdf")
    print("     2. Go to https://openstax.org/details/books/university-physics-volume-1")
    print("        Click 'Download a PDF' → save as data/raw/physics/openStax_physics_v1.pdf")
