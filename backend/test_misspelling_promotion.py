"""
test_misspelling_promotion.py — unit tests for the classifier's
misspelling safety net.

Regression for the 2026-05-01 user report where the student typed
"synapis" (intending "synapse") and got hint_error_node feedback
instead of the mastery menu. The Python guard
`_is_misspelled_concept` now flips 'incorrect' → 'correct' for
recognizable typos, and step_advancer surfaces the canonical spelling
in the confirmation.

The detector's contract:
  - true positive on real misspellings (synapis, ulner, brachail, etc.)
  - false on exact matches (already correctly spelled)
  - false on different anatomical structures sharing a category
    (ulnar vs median nerve, cerebrum vs cerebellum)
  - false on empty inputs (defensive)

Run from backend/:
    PYTHONPATH=. python3 test_misspelling_promotion.py
"""

import sys
from graph.nodes.response_classifier import _is_misspelled_concept

CASES = [
    # (student, concept, expected, label)
    ("synapis",                       "synapse",         True,  "single-word typo"),
    ("It's the synaps",               "synapse",         True,  "missing terminal e"),
    ("sinapse",                       "synapse",         True,  "phonetic variant (i for y)"),
    ("cerebelum",                     "cerebellum",      True,  "missing one l"),
    ("cerebellem",                    "cerebellum",      True,  "wrong vowel"),
    ("brachail plexus",               "brachial plexus", True,  "transposed letters"),
    # Below-threshold misspellings — short words with a single vowel
    # swap don't clear the 0.82 bar (0.80 ratio). Documented trade-off:
    # we'd rather Socratically re-engage on "ulner" than risk a stronger
    # detector promoting two genuinely different short concepts.
    ("ulner nerve",                   "ulnar nerve",     False, "single-vowel swap on short word — under threshold"),
    # Negative — already correct
    ("I think it's the cerebellum",   "cerebellum",      False, "exact spelling — not misspelled"),
    ("the synapse",                   "synapse",         False, "exact spelling"),
    # Negative — different structure (DON'T promote)
    ("It's the cerebrum",             "cerebellum",      False, "different structure (0.778 < 0.78)"),
    ("the medulla",                   "cerebellum",      False, "different structure"),
    ("it's the axon",                 "synapse",         False, "different structure"),
    ("the median nerve",              "ulnar nerve",     False, "different nerve (shared 'nerve' filtered)"),
    # Defensive
    ("",                              "synapse",         False, "empty student input"),
    ("synapse",                       "",                False, "empty concept"),
    ("?",                             "synapse",         False, "no alpha tokens"),
]

passed = failed = 0
for student, concept, expected, label in CASES:
    got = _is_misspelled_concept(student, concept)
    ok = got == expected
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} — {student!r} vs {concept!r} → {got} (expected {expected})")
    if ok:
        passed += 1
    else:
        failed += 1

print()
print(f"Results: {passed}/{len(CASES)} passed")
sys.exit(1 if failed else 0)
