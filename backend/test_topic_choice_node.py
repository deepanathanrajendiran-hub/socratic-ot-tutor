"""
test_topic_choice_node.py — acceptance tests for Step 18d

Run from project root:
    PYTHONPATH=. python3 test_topic_choice_node.py

5 acceptance criteria:
  1. Smoke test: non-empty draft produced
  2. student_phase="topic_choice_pending" in result
  3. Draft ends with exactly one question mark
  4. Draft mentions weak topics when the list is non-empty
  5. No returning-session framing (e.g. "Welcome back", "previous sessions")
     — the student is mid-session, having just chosen "move on" after
     mastering a concept. Treating the weak-topic list as a returning-
     visitor greeting is a UX bug.
"""

import sys
import config
from graph.nodes.topic_choice_node import topic_choice_node

passed = 0
failed = 0


def make_state(weak_topics=None):
    return {
        "domain": config.DOMAIN,
        "weak_topics": weak_topics or [],
        "dean_revision_instruction": "",
    }


def check(name, condition, detail=""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if condition:
        passed += 1
    else:
        failed += 1


WEAK = ["ulnar nerve", "brachial plexus", "carpal tunnel"]

print("  [INFO] Calling topic_choice_node with weak topics...")
result1 = topic_choice_node(make_state(weak_topics=WEAK))
draft1 = result1.get("draft_response", "")

check("Smoke test: non-empty draft", bool(draft1), repr(draft1[:80]))
check("student_phase='topic_choice_pending'", result1.get("student_phase") == "topic_choice_pending",
      f"got {result1.get('student_phase')!r}")
check("Ends with exactly one question mark", draft1.count("?") == 1,
      f"found {draft1.count('?')} question mark(s)")

# Check weak topics listed
any_weak_mentioned = any(t.lower() in draft1.lower() for t in WEAK)
check("Weak topics mentioned in draft", any_weak_mentioned, f"draft: {draft1[:120]!r}")

# No returning-session framing — the student is mid-session.
BANNED_PHRASES = [
    "welcome back",
    "welcome,",
    "welcome!",
    "good to see you again",
    "previous session",
    "past session",
    "last time we",
    "returning to",
    "from your prior",
]
lower = draft1.lower()
hits = [p for p in BANNED_PHRASES if p in lower]
check(
    "No returning-session framing",
    not hits,
    f"banned phrase(s) found: {hits!r}" if hits else "clean",
)

print()
print(f"Results: {passed}/5 passed")
if failed:
    print("FAILED — fix before Step 18e.")
    sys.exit(1)
else:
    print("All tests passed. Step 18d complete.")
