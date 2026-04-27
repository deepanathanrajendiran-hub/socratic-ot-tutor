"""
test_concept_leak_detection.py

_contains_concept must catch the bare concept word AND morphological
variants. Regression from yesterday's user session:
  concept = "ulnar nerve"
  draft   = "medial (ulnar) side ..."
  → _contains_concept returned False because both "ulnar" and "nerve"
    were 5 chars (skipped by the 6-char stem-check threshold), and
    drafts with "ulnar" leaked through to Dean. Dean rejected, teacher
    revised, leak persisted, fallback_scaffold fired → "Let's take a
    step back" loop.

This test asserts: the leak guard catches the bare concept word, and
ALSO that generic anatomical vocabulary ("nerve" alone, "medial") does
NOT trip a false positive (which would over-strip the draft).

Run from project root:
    PYTHONPATH=backend python3 backend/test_concept_leak_detection.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

for k in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy",
         "HTTP_PROXY", "http_proxy", "FTP_PROXY", "ftp_proxy",
         "GRPC_PROXY", "grpc_proxy"):
    os.environ.pop(k, None)

from graph.nodes.teacher_socratic import _contains_concept as t_check
from graph.nodes.hint_error_node import _contains_concept as h_check


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


def both(draft: str, concept: str) -> bool:
    """Both teacher and hint must agree (they share the contract)."""
    t = t_check(draft, concept)
    h = h_check(draft, concept)
    if t != h:
        print(f"    [WARN] teacher={t} hint={h} disagree", file=sys.stderr)
    return t


# ── Concept "ulnar nerve" — 5-char specific word ─────────────────────────────
check(
    "Catches 'ulnar' as parenthetical (the funny-bone bug)",
    both("It travels on the medial (ulnar) side of the arm.", "ulnar nerve") is True,
    "verbatim regression case from user log",
)
check(
    "Catches bare 'ulnar' adjective",
    both("This nerve is ulnar in distribution.", "ulnar nerve") is True,
)
check(
    "Catches 'ulnaris' derivative",
    both("It innervates the flexor carpi ulnaris.", "ulnar nerve") is True,
)
check(
    "Catches 'Ulnar' (case-insensitive)",
    both("Note the ULNAR distribution along the forearm.", "ulnar nerve") is True,
)
check(
    "Catches full phrase 'ulnar nerve'",
    both("The ulnar nerve passes through here.", "ulnar nerve") is True,
)


# ── Concept "ulnar nerve" — false-positive guards ────────────────────────────
check(
    "Does NOT flag 'nerve' alone (generic anatomical vocab)",
    both("This nerve travels along the medial cord.", "ulnar nerve") is False,
    "Dean PASSes 'nerve' as generic vocab; over-stripping would mangle drafts",
)
check(
    "Does NOT flag 'medial' (anatomical descriptor)",
    both("Posterior to the medial epicondyle of the humerus.", "ulnar nerve") is False,
)
check(
    "Does NOT flag clean Socratic hint",
    both(
        "Think about the bony bump on the inner side of your elbow. "
        "What structure runs just behind it?",
        "ulnar nerve",
    ) is False,
)


# ── Concept "synapse" — single word, ≥6 chars (existing stem path) ───────────
check(
    "Catches 'synaptic' (stem variant)",
    both("It happens at the synaptic cleft.", "synapse") is True,
)
check(
    "Catches 'synapses'",
    both("Multiple synapses fire in sequence.", "synapse") is True,
)


# ── Edge cases ───────────────────────────────────────────────────────────────
check("Empty concept returns False", both("anything", "") is False)
check("Empty draft returns False", both("", "ulnar nerve") is False)


# ── Summary ──────────────────────────────────────────────────────────────────
print()
print(f"Results: {passed_count}/{passed_count + failed_count} passed")
if failed_count:
    print("FAILED — concept-leak detection is broken.")
    sys.exit(1)
else:
    print("All tests passed.")
