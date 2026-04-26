"""
test_explain_node.py — acceptance tests for Step 17

Run from project root:
    PYTHONPATH=. python3 test_explain_node.py

4 acceptance criteria:
  1. Smoke test: non-empty draft produced
  2. No-reveal (turn 0): concept name NOT in draft
  3. Draft ends with exactly one question mark
  4. No sycophantic opener ("Great question!" etc.)

All 4 must pass before Step 18.
"""

import sys
from langchain_core.messages import HumanMessage, AIMessage

import config
from graph.nodes.explain_node import explain_node

CONCEPT = "ulnar nerve"

BASE_CHUNKS = [
    (
        "Several major nerves supply the upper extremity. The brachial plexus "
        "originates from nerve roots C5 through T1 and gives rise to multiple "
        "terminal branches. The medial cord of the brachial plexus gives rise "
        "to nerves that travel along the medial aspect of the arm."
    ),
    (
        "The ulnar nerve is a branch of the medial cord of the brachial plexus "
        "and arises from nerve roots C8 and T1. It travels along the medial "
        "aspect of the arm, passes posterior to the medial epicondyle of the "
        "humerus, and continues into the forearm and hand."
    ),
]

BASE_MESSAGES = [
    HumanMessage(content="What nerve causes the funny bone sensation?"),
    AIMessage(content="Think about nerves that travel along the medial aspect of the arm. Which cord of the brachial plexus runs there?"),
    HumanMessage(content="What does the medial cord actually do? I'm confused about its role."),
]


def make_state(turn_count: int = 1, student_attempted: bool = True,
               dean_revision_instruction: str = ""):
    return {
        "domain": config.DOMAIN,
        "current_concept": CONCEPT,
        "retrieved_chunks": BASE_CHUNKS,
        "turn_count": turn_count,
        "student_attempted": student_attempted,
        "messages": BASE_MESSAGES,
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


# ── Test 1: Smoke test ────────────────────────────────────────────────────────
print("  [INFO] Calling explain_node (student asks clarifying question)...")
result1 = explain_node(make_state())
draft1 = result1.get("draft_response", "")
check("Smoke test: non-empty draft", bool(draft1), repr(draft1[:80]))

# ── Test 2: No-reveal at turn 1 ───────────────────────────────────────────────
concept_in_draft = CONCEPT.lower() in draft1.lower()
check(
    "No-reveal at turn 1: concept name absent",
    not concept_in_draft,
    f"draft preview: {draft1[:120]!r}",
)

# ── Test 3: Exactly one question mark ─────────────────────────────────────────
qmarks = draft1.count("?")
check(
    "Ends with exactly one question mark",
    qmarks == 1,
    f"found {qmarks} question mark(s) in: {draft1[-80:]!r}",
)

# ── Test 4: No sycophantic opener ─────────────────────────────────────────────
SYCOPHANTIC = [
    "great question", "excellent question", "good question", "wonderful question",
    "great!", "excellent!", "good point", "fantastic",
]
lower_start = draft1.lower()[:60]
found = [p for p in SYCOPHANTIC if p in lower_start]
check(
    "No sycophantic opener",
    not found,
    f"found {found!r} in: {draft1[:80]!r}",
)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print(f"Results: {passed}/4 passed")

if failed:
    print("FAILED — fix before Step 18.")
    sys.exit(1)
else:
    print("All tests passed. Step 17 complete.")
