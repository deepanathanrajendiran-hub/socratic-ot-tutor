"""
backend/evaluation/multimodal_blind_test.py

Task 4 / Multimodal Performance evaluation.

Loads the held-out blind diagram set from
`evaluation/test_sets/blind_diagrams.json`, calls vlm_node directly on
each image (no graph plumbing — we want the VLM step in isolation),
and scores each prediction against the diagram's `accept` list of
synonyms.

Pass criterion (per CLAUDE.md `BLIND_TEST_PASS_THRESHOLD`):
  At least 4 of 5 diagrams correctly identified.

Each diagram tests a different anatomical category so the VLM can't
ride peripheral-nerve recognition across the whole set:
  - CNS structure        (spinal cord)
  - sensory map          (dermatomes)
  - cortical map         (homunculus)
  - muscle group         (rotator cuff)
  - anatomical region    (carpal tunnel)

Match logic:
  Correct iff the predicted concept matches any string in `accept`,
  using a normalized substring check (lowercase, whitespace collapsed,
  hyphens treated as spaces). The VLM has freedom in phrasing — we
  don't penalize "spinal cord transverse section" vs gold
  "spinal cord cross section" because both refer to the same drawing.

Usage:
    PYTHONPATH=. python3 evaluation/multimodal_blind_test.py
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Project root resolution: this file lives under backend/evaluation/.
ROOT = Path(__file__).resolve().parent.parent
TEST_SET_PATH = ROOT / "evaluation" / "test_sets" / "blind_diagrams.json"
RESULTS_PATH  = ROOT / "evaluation" / "results" / "multimodal_blind_results.json"
PASS_THRESHOLD = 4   # per config.BLIND_TEST_PASS_THRESHOLD


def _normalize(s: str) -> str:
    """Lowercase, collapse whitespace, treat hyphens like spaces."""
    s = (s or "").lower().replace("-", " ").strip()
    return re.sub(r"\s+", " ", s)


def _is_match(predicted: str, accept: list[str]) -> tuple[bool, str]:
    """Return (passed, matched_phrase). A prediction passes if it equals
    or contains any accepted synonym, OR vice versa — VLM phrasing
    drift in either direction is OK as long as the answer is the same
    structure."""
    p = _normalize(predicted)
    if not p:
        return False, ""
    for a in accept:
        n = _normalize(a)
        if not n:
            continue
        if n == p or n in p or p in n:
            return True, a
    return False, ""


def _encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _run_one(diagram: dict) -> dict:
    """Run vlm_node on one diagram and return a result row."""
    # Lazy imports — keep CLI startup fast and avoid hard-failing this
    # script on a missing API key when the user just wants `--help`.
    from graph.nodes.vlm_node import vlm_node

    img_path = ROOT / diagram["path"]
    if not img_path.exists():
        return {
            "id":       diagram["id"],
            "category": diagram["category"],
            "passed":   False,
            "error":    f"image not found: {img_path}",
        }

    state = {
        "image_b64":      _encode_image(img_path),
        "image_pending":  True,
        "messages":       [],
        "session_id":     f"blind-test-{diagram['id']}",
        "domain":         "OT_anatomy",
        "turn_count":     0,
    }
    t0 = time.time()
    try:
        out = vlm_node(state)
    except Exception as exc:
        return {
            "id":        diagram["id"],
            "category":  diagram["category"],
            "passed":    False,
            "error":     f"vlm_node raised: {exc!r}",
            "duration_ms": int((time.time() - t0) * 1000),
        }
    duration_ms = int((time.time() - t0) * 1000)

    predicted = (out.get("current_concept") or "").strip()
    opener    = (out.get("draft_response")  or "").strip()
    passed, matched = _is_match(predicted, diagram["accept"])

    return {
        "id":          diagram["id"],
        "category":    diagram["category"],
        "gold":        diagram["gold"],
        "predicted":   predicted,
        "matched_synonym": matched,
        "passed":      passed,
        "opener":      opener[:240],
        "duration_ms": duration_ms,
    }


def main() -> int:
    if not TEST_SET_PATH.exists():
        print(f"Test set not found: {TEST_SET_PATH}", file=sys.stderr)
        return 2
    spec = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    diagrams = spec.get("diagrams", [])
    if not diagrams:
        print("Test set is empty.", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for d in diagrams:
        print(f"  → {d['id']:<14} ({d['category']})", file=sys.stderr, flush=True)
        row = _run_one(d)
        sym = "PASS" if row.get("passed") else "FAIL"
        print(f"      {sym}  predicted={row.get('predicted','')!r}",
              file=sys.stderr)
        rows.append(row)

    n_pass  = sum(1 for r in rows if r.get("passed"))
    n_total = len(rows)
    overall_pass = n_pass >= PASS_THRESHOLD

    report = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "n_total":         n_total,
        "n_pass":          n_pass,
        "pass_threshold":  PASS_THRESHOLD,
        "overall_pass":    overall_pass,
        "results":         rows,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("", file=sys.stderr)
    print(f"  Score: {n_pass}/{n_total}  (need ≥ {PASS_THRESHOLD})",
          file=sys.stderr)
    print(f"  Overall: {'PASS' if overall_pass else 'FAIL'}", file=sys.stderr)
    print(f"  Wrote {RESULTS_PATH}", file=sys.stderr)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
