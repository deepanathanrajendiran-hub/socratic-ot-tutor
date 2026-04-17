"""
test_topic_choice_node.py — acceptance tests for Step 18d

Run from project root:
    PYTHONPATH=. python3 test_topic_choice_node.py

4 acceptance criteria:
  1. Smoke test: non-empty draft produced
  2. student_phase="topic_choice_pending" in result
  3. Draft ends with exactly one question mark
  4. Draft mentions weak topics when the list is non-empty
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

print()
print(f"Results: {passed}/4 passed")
if failed:
    print("FAILED — fix before Step 18e.")
    sys.exit(1)
else:
    print("All tests passed. Step 18d complete.")
