"""
test_clinical_question_node.py — acceptance tests for Step 18f

Run from project root:
    PYTHONPATH=. python3 test_clinical_question_node.py

4 acceptance criteria:
  1. Smoke test: non-empty draft produced
  2. student_phase="clinical_pending" in result
  3. Draft ends with exactly one question mark
  4. Draft contains clinical/OT scenario context
"""

import sys
import config
from graph.nodes.clinical_question_node import clinical_question_node

CONCEPT = "ulnar nerve"
CHUNKS = [
    (
        "The ulnar nerve is a branch of the medial cord of the brachial plexus "
        "and arises from nerve roots C8 and T1. It passes posterior to the medial "
        "epicondyle of the humerus and continues into the forearm and hand. "
        "The nerve innervates the flexor carpi ulnaris and the medial half of "
        "the flexor digitorum profundus in the forearm."
    ),
]

passed = 0
failed = 0


def make_state():
    return {
        "domain": config.DOMAIN,
        "current_concept": CONCEPT,
        "retrieved_chunks": CHUNKS,
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


print("  [INFO] Calling clinical_question_node...")
result1 = clinical_question_node(make_state())
draft1 = result1.get("draft_response", "")

check("Smoke test: non-empty draft", bool(draft1), repr(draft1[:80]))
check("student_phase='clinical_pending'", result1.get("student_phase") == "clinical_pending",
      f"got {result1.get('student_phase')!r}")
check("Ends with exactly one question mark", draft1.count("?") == 1,
      f"found {draft1.count('?')} question mark(s) in: {draft1[-80:]!r}")

# Clinical/OT context: should mention patient scenario or OT-related terms
OT_TERMS = ["patient", "ot ", "occupational", "function", "assess", "adl",
            "grip", "pinch", "wrist", "hand", "finger", "forearm"]
lower = draft1.lower()
found_ot = [t for t in OT_TERMS if t in lower]
check("Draft contains clinical/OT scenario context", bool(found_ot),
      f"OT terms found: {found_ot[:3]!r} | draft: {draft1[:120]!r}")

print()
print(f"Results: {passed}/4 passed")
if failed:
    print("FAILED — fix before Step 19.")
    sys.exit(1)
else:
    print("All tests passed. Step 18f complete.")
