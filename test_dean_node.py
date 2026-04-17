"""
test_dean_node.py — acceptance tests for Step 14

Run from project root:
    PYTHONPATH=. python3 test_dean_node.py

4 acceptance criteria:
  1. Teacher-generated draft (turn 0) passes Dean       → dean_passed = True
  2. Draft that reveals concept at turn 0               → dean_passed = False
  3. dean_revisions increments on failure               → +1
  4. revision_instruction is non-empty on failure       → Dean gives specific guidance

Test 1 uses a real teacher_socratic output so we test the actual Teacher→Dean
pipeline rather than a hand-crafted draft that may not meet grounding criteria.

All 4 must pass before Step 15.
"""

import sys
from langchain_core.messages import HumanMessage

import config
from graph.nodes.teacher_socratic import teacher_socratic
from graph.nodes.dean_node import dean_node

CONCEPT = "ulnar nerve"
CHUNKS = [
    (
        "Several major nerves supply the upper extremity. The brachial plexus "
        "originates from nerve roots C5 through T1 and gives rise to multiple "
        "terminal branches. Each branch follows a specific anatomical course "
        "through the arm and forearm to supply muscles and skin of the upper limb. "
        "The medial cord of the brachial plexus gives rise to nerves that travel "
        "along the medial aspect of the arm."
    ),
    (
        "The ulnar nerve is a branch of the medial cord of the brachial plexus "
        "and arises from nerve roots C8 and T1. It travels along the medial "
        "aspect of the arm, passes posterior to the medial epicondyle of the "
        "humerus, and continues into the forearm and hand. The nerve innervates "
        "the flexor carpi ulnaris and the medial half of the flexor digitorum "
        "profundus in the forearm."
    ),
]

BASE_MESSAGES = [HumanMessage(content="What nerve causes the funny bone sensation?")]


def make_teacher_state():
    return {
        "domain": config.DOMAIN,
        "current_concept": CONCEPT,
        "retrieved_chunks": CHUNKS,
        "turn_count": 0,
        "student_attempted": False,
        "weak_topics": [],
        "messages": BASE_MESSAGES,
        "dean_revision_instruction": "",
    }


def make_dean_state(draft: str, turn_count: int = 0,
                    student_attempted: bool = False, dean_revisions: int = 0):
    return {
        "current_concept": CONCEPT,
        "turn_count": turn_count,
        "student_attempted": student_attempted,
        "retrieved_chunks": CHUNKS,
        "draft_response": draft,
        "dean_revisions": dean_revisions,
        "messages": BASE_MESSAGES,
    }


passed_count = 0
failed_count = 0


def check(name, condition, detail=""):
    global passed_count, failed_count
    status = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if condition:
        passed_count += 1
    else:
        failed_count += 1


# ── Test 1: Teacher-generated draft passes Dean (integration test) ─────────────
print("  [INFO] Generating teacher draft (turn 0, no reveal)...")
teacher_result = teacher_socratic(make_teacher_state())
teacher_draft = teacher_result["draft_response"]
print(f"  [INFO] Draft: {teacher_draft[:120]!r}...")

result1 = dean_node(make_dean_state(draft=teacher_draft, turn_count=0))
check(
    "Teacher draft (turn 0) passes Dean",
    result1["dean_passed"] is True,
    f"dean_passed={result1['dean_passed']}, "
    f"instruction={result1.get('dean_revision_instruction', '')!r}",
)

# ── Test 2: Draft that explicitly names the concept at turn 0 → Dean rejects ──
revealing_draft = (
    "The ulnar nerve is the nerve that causes the funny bone sensation. "
    "It travels along the medial aspect of the arm. "
    "Can you describe what the ulnar nerve innervates in the forearm?"
)
result2 = dean_node(make_dean_state(draft=revealing_draft, turn_count=0))
check(
    "Reveal violation caught (turn 0, reveal_permitted=False)",
    result2["dean_passed"] is False,
    f"dean_passed={result2['dean_passed']}",
)

# ── Test 3: dean_revisions increments on failure ──────────────────────────────
check(
    "dean_revisions increments on failure",
    result2["dean_revisions"] == 1,
    f"dean_revisions={result2['dean_revisions']} (expected 1)",
)

# ── Test 4: revision_instruction non-empty on failure ─────────────────────────
instruction = result2.get("dean_revision_instruction", "")
check(
    "revision_instruction non-empty on failure",
    bool(instruction),
    f"instruction={instruction!r}",
)

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print(f"Results: {passed_count}/4 passed")

if failed_count:
    print("FAILED — fix before Step 15.")
    sys.exit(1)
else:
    print("All tests passed. Step 14 complete.")
