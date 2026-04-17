"""
test_step_advancer.py — acceptance tests for Step 18

Run from project root:
    PYTHONPATH=. python3 test_step_advancer.py

4 acceptance criteria:
  1. Strong mastery: turn 0 correct → mastery_level="strong", concept_mastered=True
  2. Weak mastery: turn 2 correct → mastery_level="weak"
  3. student_phase="choice_pending" in result
  4. Draft contains the three-way choice offer

All 4 must pass before Step 18b.
"""

import sys
from langchain_core.messages import HumanMessage, AIMessage

import config
from graph.nodes.step_advancer import step_advancer

CONCEPT = "ulnar nerve"


def make_state(turn_count: int, dean_revision_instruction: str = ""):
    return {
        "domain": config.DOMAIN,
        "current_concept": CONCEPT,
        "turn_count": turn_count,
        "dean_revision_instruction": dean_revision_instruction,
    }


passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if condition:
        passed += 1
    else:
        failed += 1


# ── Test 1: Strong mastery (turn 0) ──────────────────────────────────────────
print("  [INFO] Testing strong mastery (turn_count=0)...")
result1 = step_advancer(make_state(turn_count=0))
check(
    "Strong mastery: mastery_level='strong', concept_mastered=True",
    result1.get("mastery_level") == "strong" and result1.get("concept_mastered") is True,
    f"mastery_level={result1.get('mastery_level')!r}, concept_mastered={result1.get('concept_mastered')}",
)

# ── Test 2: Weak mastery (turn >= SOCRATIC_TURN_GATE) ────────────────────────
print(f"  [INFO] Testing weak mastery (turn_count={config.SOCRATIC_TURN_GATE})...")
result2 = step_advancer(make_state(turn_count=config.SOCRATIC_TURN_GATE))
check(
    "Weak mastery: mastery_level='weak'",
    result2.get("mastery_level") == "weak",
    f"mastery_level={result2.get('mastery_level')!r}",
)

# ── Test 3: student_phase set to choice_pending ───────────────────────────────
check(
    "student_phase='choice_pending'",
    result1.get("student_phase") == "choice_pending",
    f"student_phase={result1.get('student_phase')!r}",
)

# ── Test 4: Draft contains the three-way choice ───────────────────────────────
draft = result1.get("draft_response", "")
lower = draft.lower()
has_clinical = "clinical" in lower
has_next = "next" in lower or "topic" in lower
has_done = "done" in lower or "stop" in lower
check(
    "Draft contains three-way choice offer",
    has_clinical and has_next and has_done,
    f"clinical={has_clinical}, next/topic={has_next}, done/stop={has_done} | {draft[:120]!r}",
)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print(f"Results: {passed}/4 passed")

if failed:
    print("FAILED — fix before Step 18b.")
    sys.exit(1)
else:
    print("All tests passed. Step 18 complete.")
