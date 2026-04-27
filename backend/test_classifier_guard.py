"""
test_classifier_guard.py

Regression: Haiku classifier labeled "Is it the median nerve?" as
"correct" when current_concept was "ulnar nerve". The Socratic flow
then routed to step_advancer (mastery confirmation) instead of
hint_error_node, so the student got "That's correct — you've identified
the median nerve" — a hallucinated agreement on a wrong answer.

The guard is a programmatic post-check on the classifier's verdict:
if the label is "correct" but the concept word (or its stem) is not
present in the student's message, override to "incorrect". This is
deterministic — no LLM round-trip.

Run from project root:
    PYTHONPATH=backend python3 backend/test_classifier_guard.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

for k in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy",
         "HTTP_PROXY", "http_proxy", "FTP_PROXY", "ftp_proxy",
         "GRPC_PROXY", "grpc_proxy"):
    os.environ.pop(k, None)

from graph.nodes.response_classifier import _verify_correct_label as guard


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


# ── Direct full-phrase match ─────────────────────────────────────────────────
check(
    "Full concept phrase in message → correct stays correct",
    guard("correct", "Yes, it's the ulnar nerve", "ulnar nerve") == "correct",
)
check(
    "Concept word alone (stem) in message → correct stays correct",
    guard("correct", "I think it's ulnar", "ulnar nerve") == "correct",
)
check(
    "Morphological variant — 'ulnaris' → correct stays correct",
    guard("correct", "the ulnaris one", "ulnar nerve") == "correct",
)


# ── The funny-bone bug: wrong nerve named ────────────────────────────────────
check(
    "REGRESSION: 'Is it the median nerve?' (concept=ulnar nerve) → override to incorrect",
    guard("correct", "Is it the median nerve?", "ulnar nerve") == "incorrect",
    "verbatim user-session bug",
)
check(
    "Wrong nerve: 'Maybe the radial nerve?' → override to incorrect",
    guard("correct", "Maybe the radial nerve?", "ulnar nerve") == "incorrect",
)
check(
    "Wrong region: 'the spinal cord' → override to incorrect",
    guard("correct", "the spinal cord", "ulnar nerve") == "incorrect",
)


# ── Single-word concept ──────────────────────────────────────────────────────
check(
    "Concept 'synapse', student says 'synaptic cleft' → correct stays",
    guard("correct", "the synaptic cleft", "synapse") == "correct",
)
check(
    "Concept 'synapse', student says 'the membrane' → override to incorrect",
    guard("correct", "the membrane", "synapse") == "incorrect",
)


# ── Pass-through for non-correct labels ──────────────────────────────────────
check(
    "label=incorrect passes through unchanged",
    guard("incorrect", "anything", "ulnar nerve") == "incorrect",
)
check(
    "label=idk passes through unchanged",
    guard("idk", "I don't know", "ulnar nerve") == "idk",
)
check(
    "label=questioning passes through unchanged",
    guard("questioning", "what's a nerve?", "ulnar nerve") == "questioning",
)
check(
    "label=irrelevant passes through unchanged",
    guard("irrelevant", "what's for lunch", "ulnar nerve") == "irrelevant",
)


# ── Edge cases ───────────────────────────────────────────────────────────────
check(
    "Empty student message + correct → no override (defensive)",
    guard("correct", "", "ulnar nerve") == "correct",
)
check(
    "Empty concept + correct → no override (no concept to check against)",
    guard("correct", "anything", "") == "correct",
)


# ── Concept where every word is generic (e.g. 'anterior horn') ───────────────
# Both "anterior" and "horn" are generic anatomical descriptors; the only
# disambiguator is the full phrase. Guard requires the full phrase.
check(
    "Concept 'anterior horn', student says full phrase → correct stays",
    guard("correct", "the anterior horn of the spinal cord", "anterior horn") == "correct",
)
check(
    "Concept 'anterior horn', student says 'something anterior' → override (no full phrase)",
    guard("correct", "something anterior", "anterior horn") == "incorrect",
)


# ── Case insensitivity ───────────────────────────────────────────────────────
check(
    "Case-insensitive match: 'ULNAR NERVE' in caps → correct stays",
    guard("correct", "ULNAR NERVE", "ulnar nerve") == "correct",
)


# ── Summary ──────────────────────────────────────────────────────────────────
print()
print(f"Results: {passed_count}/{passed_count + failed_count} passed")
if failed_count:
    print("FAILED — classifier guard needs work.")
    sys.exit(1)
else:
    print("All tests passed.")
