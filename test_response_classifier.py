"""
test_response_classifier.py — acceptance tests for Step 12

Run from project root:
    PYTHONPATH=. python3 test_response_classifier.py

4 acceptance criteria:
  1. Student correct answer  → "correct"
  2. Student wrong answer    → "incorrect"
  3. Off-topic message       → "irrelevant"
  4. Clarifying question     → "questioning"

All 4 must pass before Step 13.
"""

import sys
from langchain_core.messages import HumanMessage, AIMessage
from graph.nodes.response_classifier import response_classifier

CONCEPT = "ulnar nerve"

PRIOR = [
    HumanMessage(content="What nerve causes the funny bone sensation?"),
    AIMessage(content=(
        "Think about which nerve passes close to the surface at the medial "
        "side of the elbow. That region is sometimes called the cubital tunnel. "
        "Which nerve travels through that tunnel?"
    )),
]

CASES = [
    {
        "name": "correct answer",
        "student_msg": "Is it the ulnar nerve?",
        "expected": "correct",
    },
    {
        "name": "incorrect answer",
        "student_msg": "Is it the median nerve?",
        "expected": "incorrect",
    },
    {
        "name": "off-topic (irrelevant)",
        "student_msg": "What is the best restaurant in Buffalo?",
        "expected": "irrelevant",
    },
    {
        "name": "clarifying question (questioning)",
        "student_msg": "Can you explain what the cubital tunnel actually is?",
        "expected": "questioning",
    },
]

passed = 0
failed = 0

for case in CASES:
    state = {
        "current_concept": CONCEPT,
        "messages": PRIOR + [HumanMessage(content=case["student_msg"])],
    }
    result = response_classifier(state)
    label = result["classifier_output"]
    ok = label == case["expected"]
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {case['name']}: got '{label}' (expected '{case['expected']}')")
    if ok:
        passed += 1
    else:
        failed += 1

print()
print(f"Results: {passed}/4 passed")

if failed:
    print("FAILED — fix before Step 13.")
    sys.exit(1)
else:
    print("All tests passed. Step 12 complete.")
