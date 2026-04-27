"""
test_idk_counter.py — acceptance tests for the idk-counter reveal gate

Run from project root:
    PYTHONPATH=. python3 test_idk_counter.py

Tests two layers:

  Routing layer (pure Python, no API calls) — 6 cases:
    R1. idk + count=1 → hint_error_node
    R2. idk + count=2 → hint_error_node
    R3. idk + count=3 (== threshold) → teach_node (REVEAL)
    R4. idk + count=5 (above threshold) → teach_node
    R5. incorrect + low turn → hint_error_node (respects turn gate, not idk gate)
    R6. correct → step_advancer regardless of idk_count

  Classifier layer (one API call per case) — 3 cases:
    C1. 'I don't know'        → label=idk, count increments (0 → 1)
    C2. Engagement after idk  → count resets to 0 (prior=2, attempt → 0)
    C3. Third idk in a row    → count = 3 (prior=2 + idk → 3)

All 9 must pass.
"""

import sys
from langchain_core.messages import HumanMessage, AIMessage

import config
from graph.edges import route_after_classifier
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


passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if condition:
        passed += 1
    else:
        failed += 1


# ── Routing layer (no API calls) ──────────────────────────────────────────────

print("[Routing layer]")

THRESHOLD = config.IDK_REVEAL_THRESHOLD  # default 3

# R1
state = {"classifier_output": "idk", "turn_count": 0, "idk_count": 1}
got = route_after_classifier(state)
check("R1 idk count=1 → hint_error_node", got == "hint_error_node", f"got {got!r}")

# R2
state = {"classifier_output": "idk", "turn_count": 1, "idk_count": 2}
got = route_after_classifier(state)
check("R2 idk count=2 → hint_error_node", got == "hint_error_node", f"got {got!r}")

# R3 — exactly at threshold
state = {"classifier_output": "idk", "turn_count": 0, "idk_count": THRESHOLD}
got = route_after_classifier(state)
check(
    f"R3 idk count={THRESHOLD} (==threshold) → teach_node (REVEAL)",
    got == "teach_node",
    f"got {got!r}",
)

# R4 — above threshold
state = {"classifier_output": "idk", "turn_count": 0, "idk_count": THRESHOLD + 2}
got = route_after_classifier(state)
check(
    f"R4 idk count={THRESHOLD + 2} (>threshold) → teach_node",
    got == "teach_node",
    f"got {got!r}",
)

# R5 — incorrect path is unaffected by idk_count, gated by turn_count
state = {
    "classifier_output": "incorrect",
    "turn_count": 0,
    "idk_count": THRESHOLD + 5,  # high idk count must NOT trigger reveal
}
got = route_after_classifier(state)
check(
    "R5 incorrect at turn 0 (high idk_count irrelevant) → hint_error_node",
    got == "hint_error_node",
    f"got {got!r}",
)

# R6 — correct ignores idk_count
state = {
    "classifier_output": "correct",
    "turn_count": 1,
    "idk_count": THRESHOLD,
}
got = route_after_classifier(state)
check(
    "R6 correct (any idk_count) → step_advancer",
    got == "step_advancer",
    f"got {got!r}",
)


# ── Classifier layer (live API calls) ─────────────────────────────────────────

print()
print("[Classifier layer]")


def _make_state(student_msg: str, prior_idk_count: int = 0):
    return {
        "current_concept": CONCEPT,
        "idk_count": prior_idk_count,
        "messages": PRIOR + [HumanMessage(content=student_msg)],
    }


# C1 — first idk increments 0 → 1
result = response_classifier(_make_state("I don't know.", prior_idk_count=0))
ok = result.get("classifier_output") == "idk" and result.get("idk_count") == 1
check(
    "C1 'I don't know' → label=idk, count 0 → 1",
    ok,
    f"got label={result.get('classifier_output')!r} count={result.get('idk_count')!r}",
)

# C2 — engagement (wrong attempt) resets count
result = response_classifier(
    _make_state("Is it the median nerve?", prior_idk_count=2)
)
ok = result.get("classifier_output") == "incorrect" and result.get("idk_count") == 0
check(
    "C2 wrong attempt after 2 idks → label=incorrect, count → 0",
    ok,
    f"got label={result.get('classifier_output')!r} count={result.get('idk_count')!r}",
)

# C3 — third consecutive idk reaches threshold
result = response_classifier(
    _make_state("I really have no idea, just give me the answer.", prior_idk_count=2)
)
ok = result.get("classifier_output") == "idk" and result.get("idk_count") == 3
check(
    "C3 third consecutive idk → label=idk, count 2 → 3 (== threshold)",
    ok,
    f"got label={result.get('classifier_output')!r} count={result.get('idk_count')!r}",
)


# ── Summary ────────────────────────────────────────────────────────────────────

print()
print(f"Results: {passed}/9 passed")

if failed:
    print("FAILED — idk-counter wiring needs review.")
    sys.exit(1)
else:
    print("All tests passed. idk-counter is wired correctly.")
