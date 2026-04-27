"""
test_dean_json_parser.py — unit tests for Dean's JSON extraction.

Reproduces the chain-of-thought bug where Haiku self-corrects mid-response
and emits two JSON blocks separated by markdown fences and prose. The
parser must extract the LAST valid JSON block (the corrected verdict),
not splice across blocks (which produces garbage).

Run from project root:
    PYTHONPATH=backend python3 backend/test_dean_json_parser.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from graph.nodes.dean_node import _extract_dean_json


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


# ── Test 1: pure JSON ────────────────────────────────────────────────────────
raw1 = '{"passed": true, "failed_criteria": [], "revision_instruction": ""}'
r1 = _extract_dean_json(raw1)
check(
    "Pure JSON parses correctly",
    r1 is not None and r1.get("passed") is True,
    f"got {r1!r}",
)


# ── Test 2: markdown-fenced JSON ─────────────────────────────────────────────
raw2 = """```json
{"passed": false, "failed_criteria": ["REVEAL CHECK"], "revision_instruction": "Remove the term."}
```"""
r2 = _extract_dean_json(raw2)
check(
    "Markdown-fenced JSON parses correctly",
    r2 is not None and r2.get("passed") is False,
    f"got {r2!r}",
)


# ── Test 3: chain-of-thought self-correction (THE BUG) ──────────────────────
# Real Haiku output captured from user's session. Initial verdict says false,
# then Haiku reconsiders and emits a corrected verdict. Parser must take the
# LAST block (Haiku's final answer after self-correction).
raw3 = '''{
  "passed": false,
  "failed_criteria": ["REVEAL CHECK"],
  "revision_instruction": "Remove 'nerve pathway' which too directly references the ulnar nerve concept."
}
```

Wait, let me reconsider. Looking at the REVEAL CHECK criterion more carefully:

The phrase "nerve pathway" falls under process vocabulary similar to "motor pathway." Let me re-evaluate:

```json
{
  "passed": true,
  "failed_criteria": [],
  "revision_instruction": ""
}'''
r3 = _extract_dean_json(raw3)
check(
    "Chain-of-thought self-correction takes LAST JSON block",
    r3 is not None and r3.get("passed") is True,
    f"got {r3!r}",
)


# ── Test 4: nested object preserved (depth tracking) ─────────────────────────
raw4 = '{"passed": true, "failed_criteria": [], "revision_instruction": "", "meta": {"score": 0.95}}'
r4 = _extract_dean_json(raw4)
check(
    "Nested objects preserved",
    (
        r4 is not None
        and r4.get("passed") is True
        and r4.get("meta", {}).get("score") == 0.95
    ),
    f"got {r4!r}",
)


# ── Test 5: garbage input returns None ───────────────────────────────────────
r5 = _extract_dean_json("no json here at all")
check("Garbage input returns None", r5 is None, f"got {r5!r}")


# ── Test 6: truncated JSON returns None ──────────────────────────────────────
r6 = _extract_dean_json('{"passed": true, "failed_criteria":')
check("Truncated JSON returns None", r6 is None, f"got {r6!r}")


# ── Test 7: last block malformed → falls back to first valid ─────────────────
raw7 = '{"passed": false, "failed_criteria": ["REVEAL CHECK"], "revision_instruction": "Fix it."}\n\nHmm, but actually:\n\n{"passed": true, "failed'
r7 = _extract_dean_json(raw7)
check(
    "Last-block-truncated falls back to first valid",
    r7 is not None and r7.get("passed") is False,
    f"got {r7!r}",
)


# ── Test 8: empty input ──────────────────────────────────────────────────────
r8 = _extract_dean_json("")
check("Empty string returns None", r8 is None, f"got {r8!r}")


# ── Test 9: realistic two-JSON exact reproduction from logs ──────────────────
# Verbatim from user's terminal session last night.
raw9 = (
    '{\n  "passed": false,\n'
    '  "failed_criteria": ["REVEAL CHECK"],\n'
    '  "revision_instruction": "Remove \'a nerve\'..."\n'
    '}\n'
    '```\n'
    '\n'
    'Wait, let me reconsider. Looking at the REVEAL CHECK criterion more carefully:\n\n'
    'The draft says "it\'s definitely a nerve at the elbow" but does NOT contain '
    '"ulnar nerve" or any derivative.\n\n'
    'Let me re-evaluate:\n\n'
    '```json\n'
    '{\n  "passed": true,\n'
    '  "failed_criteria": [],\n'
    '  "revision_instruction": ""\n'
    '}'
)
r9 = _extract_dean_json(raw9)
check(
    "Verbatim Haiku log reproduction (passed=true after self-correction)",
    r9 is not None and r9.get("passed") is True,
    f"got {r9!r}",
)


# ── Summary ──────────────────────────────────────────────────────────────────
print()
print(f"Results: {passed_count}/{passed_count + failed_count} passed")

if failed_count:
    print("FAILED — Dean JSON parser broken.")
    sys.exit(1)
else:
    print("All tests passed.")
