"""
test_topic_choice_classifier.py — acceptance tests for Step 18e

Run from project root:
    PYTHONPATH=. python3 test_topic_choice_classifier.py

4 acceptance criteria:
  1. "weak topics" intent → "weak"
  2. Specific own topic → "own"
  3. student_phase reset to "learning" in result
  4. mastery_choice reset to "" in result
"""

import sys
from langchain_core.messages import HumanMessage, AIMessage

import config
from graph.nodes.topic_choice_classifier import topic_choice_classifier

WEAK = ["ulnar nerve", "brachial plexus"]

passed = 0
failed = 0


def make_state(last_message: str):
    return {
        "weak_topics": WEAK,
        "messages": [
            AIMessage(content="Would you like to practice a weak topic or name one of your own?"),
            HumanMessage(content=last_message),
        ],
        "student_phase": "topic_choice_pending",
        "mastery_choice": "next",
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


r1 = topic_choice_classifier(make_state("Let's work on my weak topics"))
check("Weak intent → 'weak'", r1["topic_choice"] == "weak", f"got {r1['topic_choice']!r}")

r2 = topic_choice_classifier(make_state("I want to study the rotator cuff"))
check("Own topic intent → 'own'", r2["topic_choice"] == "own", f"got {r2['topic_choice']!r}")

check("student_phase reset to 'learning'", r1.get("student_phase") == "learning",
      f"got {r1.get('student_phase')!r}")

check("mastery_choice reset to ''", r1.get("mastery_choice") == "",
      f"got {r1.get('mastery_choice')!r}")

print()
print(f"Results: {passed}/4 passed")
if failed:
    print("FAILED — fix before Step 18f.")
    sys.exit(1)
else:
    print("All tests passed. Step 18e complete.")
