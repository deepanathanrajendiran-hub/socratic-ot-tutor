"""
test_mastery_choice_classifier.py — acceptance tests for Step 18c

Run from project root:
    PYTHONPATH=. python3 test_mastery_choice_classifier.py

4 acceptance criteria:
  1. "clinical question" intent → "clinical"
  2. "next topic" intent → "next"
  3. "done" intent → "done"
  4. Unclear/other → "other"
"""

import sys
from langchain_core.messages import HumanMessage, AIMessage

import config
from graph.nodes.mastery_choice_classifier import mastery_choice_classifier

passed = 0
failed = 0


def make_state(last_message: str):
    return {
        "messages": [
            HumanMessage(content="What nerve causes the funny bone sensation?"),
            AIMessage(content="Correct! You identified it. Would you like: A) clinical question, B) next topic, C) done?"),
            HumanMessage(content=last_message),
        ]
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


r1 = mastery_choice_classifier(make_state("I'd like to try the clinical application question please"))
check("Clinical intent → 'clinical'", r1["mastery_choice"] == "clinical", f"got {r1['mastery_choice']!r}")

r2 = mastery_choice_classifier(make_state("Let's move on to the next topic"))
check("Next intent → 'next'", r2["mastery_choice"] == "next", f"got {r2['mastery_choice']!r}")

r3 = mastery_choice_classifier(make_state("I'm done for now, thanks"))
check("Done intent → 'done'", r3["mastery_choice"] == "done", f"got {r3['mastery_choice']!r}")

r4 = mastery_choice_classifier(make_state("umm not sure what to do"))
check("Unclear → 'other'", r4["mastery_choice"] == "other", f"got {r4['mastery_choice']!r}")

print()
print(f"Results: {passed}/4 passed")
if failed:
    print("FAILED — fix before Step 18d.")
    sys.exit(1)
else:
    print("All tests passed. Step 18c complete.")
