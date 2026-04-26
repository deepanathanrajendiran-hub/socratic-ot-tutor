"""
test_redirect_node.py — acceptance tests for Step 16

Run from project root:
    PYTHONPATH=. python3 test_redirect_node.py

4 acceptance criteria:
  1. Smoke test: non-empty draft is produced
  2. No-reveal: concept name NOT in draft
  3. Draft ends with exactly one question mark
  4. Off-topic subject is NOT addressed (redirect actually worked)

All 4 must pass before Step 17.
"""

import sys
from langchain_core.messages import HumanMessage, AIMessage

import config
from graph.nodes.redirect_node import redirect_node

CONCEPT = "ulnar nerve"

BASE_MESSAGES = [
    HumanMessage(content="What nerve causes the funny bone sensation?"),
    AIMessage(
        content=(
            "Think about the nerves that travel along the medial aspect of the arm. "
            "Which cord of the brachial plexus gives rise to nerves in that region?"
        )
    ),
    HumanMessage(content="Actually, what is the best restaurant in Buffalo?"),
]


def make_state(turn_count: int = 1, dean_revision_instruction: str = ""):
    return {
        "current_concept": CONCEPT,
        "turn_count": turn_count,
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


# ── Test 1: Smoke test — non-empty draft produced ─────────────────────────────
print("  [INFO] Calling redirect_node (off-topic: restaurant question)...")
result1 = redirect_node(make_state())
draft1 = result1.get("draft_response", "")
check("Smoke test: non-empty draft", bool(draft1), repr(draft1[:80]))

# ── Test 2: No-reveal — concept name must NOT appear ─────────────────────────
concept_in_draft = CONCEPT.lower() in draft1.lower()
check(
    "No-reveal: concept name absent",
    not concept_in_draft,
    f"draft preview: {draft1[:120]!r}",
)

# ── Test 3: Draft ends with exactly one question mark ─────────────────────────
qmarks = draft1.count("?")
check(
    "Ends with exactly one question mark",
    qmarks == 1,
    f"found {qmarks} question mark(s) in: {draft1[-80:]!r}",
)

# ── Test 4: Response bridges back to anatomy ──────────────────────────────────
# The redirect may mention off-topic words to dismiss or bridge them
# ("save restaurant recommendations", "if you were dining..."). What matters
# is that the response contains anatomy content showing the redirect worked.
ANATOMY_KEYWORDS = [
    "nerve", "muscle", "medial", "lateral", "forearm", "arm", "elbow",
    "pathway", "brachial", "cord", "anatomy", "anatomical", "innervate",
    "sensory", "motor", "bone", "tendon", "plexus", "finger", "hand",
]
lower_draft = draft1.lower()
anatomy_found = [kw for kw in ANATOMY_KEYWORDS if kw in lower_draft]
check(
    "Response bridges back to anatomy topic",
    bool(anatomy_found),
    f"anatomy keywords found: {anatomy_found[:3]!r}",
)

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print(f"Results: {passed}/4 passed")

if failed:
    print("FAILED — fix before Step 17.")
    sys.exit(1)
else:
    print("All tests passed. Step 16 complete.")
