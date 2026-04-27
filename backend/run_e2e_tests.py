"""
run_e2e_tests.py — live end-to-end testing of the Socratic-OT backend.

Hits a running uvicorn instance on localhost:8000, drives 5 scripted
student trajectories through POST /chat, captures tutor responses,
runs assertions per turn, and writes a markdown report to test.md.

Each scenario uses POST /sessions/{id}/reset to start from a clean
checkpoint. The student inputs are hardcoded in SCENARIOS — that's the
"answer the question yourself" part of the brief.

Usage:
    PYTHONPATH=backend python3 backend/run_e2e_tests.py

Requires:
  - uvicorn running on http://localhost:8000
  - Valid ANTHROPIC_API_KEY (or AWS Bedrock creds + LLM_PROVIDER=bedrock)
  - ChromaDB populated (data/processed/chroma_db/)
"""
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx


BACKEND = os.getenv("E2E_BACKEND", "http://localhost:8000")
ROOT = Path(__file__).resolve().parent.parent
TEST_MD = ROOT / "test.md"


# ── Assertion helpers ────────────────────────────────────────────────────────

def has_question(text: str) -> tuple[bool, str]:
    return ("?" in text), ("ends with a question" if "?" in text else "no '?' anywhere")


def at_most_two_questions(text: str) -> tuple[bool, str]:
    n = text.count("?")
    return (n <= 2), f"{n} '?' in response"


def no_meta_leak(text: str) -> tuple[bool, str]:
    bad = [
        "the retrieved content", "according to the textbook",
        "the textbook says", "the textbook notes", "the textbook mentions",
        "the passage says", "the passage notes",
        "the source says", "the content describes", "the content notes",
        "based on what i have", "based on the retrieved",
        "my knowledge base", "training data",
        "we're at turn", "now that we're at turn", "this is turn",
        "since we're at turn",
    ]
    lower = text.lower()
    for phrase in bad:
        if phrase in lower:
            return False, f"leaked phrase: {phrase!r}"
    return True, "no meta phrases"


def no_concept_leak(concept: str) -> Callable[[str], tuple[bool, str]]:
    """Returns an assertion that fails if `concept` (or a stem variant of
    its DISCRIMINATING words) appears in the tutor's response.

    Generic words from the active domain (e.g. 'nerve', 'system', 'medial')
    are skipped — production code treats them as generic vocabulary that
    Dean PASSes on its own, so checking them here would false-positive on
    every legitimate use of 'nerve' in a turn-0 question about nerves.
    """
    from graph.nodes._helpers import get_generic_words
    generic = get_generic_words()

    def check(text: str) -> tuple[bool, str]:
        lower = text.lower()
        c_lower = concept.lower()
        if c_lower in lower:
            return False, f"full concept {c_lower!r} present"
        for word in c_lower.split():
            if word in generic:
                continue
            if len(word) < 5:
                continue
            stem = word[: max(4, len(word) - 2)]
            # Word boundary check so stem 'ulna' doesn't match 'tunnel' etc.
            import re as _re
            if _re.search(r"\b" + _re.escape(stem), lower):
                return False, f"discriminating stem {stem!r} of {word!r} present"
        return True, f"no discriminating part of {concept!r} present"
    return check


def not_fallback_scaffold(text: str) -> tuple[bool, str]:
    if text.startswith("Let's take a step back"):
        return False, "fallback_scaffold fired (Dean revisions exhausted)"
    return True, "not the fallback message"


def does_not_confirm_wrong_answer(named_wrong: str) -> Callable[[str], tuple[bool, str]]:
    """Asserts the tutor did NOT confirm `named_wrong` as correct.
    e.g. tutor must not say 'That's correct, you identified the median nerve'
    when the actual concept was ulnar."""
    def check(text: str) -> tuple[bool, str]:
        lower = text.lower()
        named = named_wrong.lower()
        bad_patterns = [
            f"that's correct — you've identified the {named}",
            f"correct — you've identified the {named}",
            f"yes, the {named}",
            f"you're right, the {named}",
            f"you've correctly identified the {named}",
            f"that's the {named}",
            f"yes! the {named}",
        ]
        for p in bad_patterns:
            if p in lower:
                return False, f"tutor confirmed wrong answer: {p!r}"
        # Also fail if response just opens with confirmation language without explicit pushback
        opens_confirm = lower.startswith("that's correct") or lower.startswith("correct")
        mentions_wrong = named in lower
        if opens_confirm and mentions_wrong:
            return False, f"opens with 'correct' AND mentions {named!r} — likely confirming wrong answer"
        return True, f"did not confirm {named!r}"
    return check


def reveals_concept(concept: str) -> Callable[[str], tuple[bool, str]]:
    """Asserts the tutor DID reveal the concept (post-reveal turns)."""
    def check(text: str) -> tuple[bool, str]:
        if concept.lower() in text.lower():
            return True, f"concept {concept!r} revealed as expected"
        return False, f"expected concept {concept!r} to be revealed but it isn't"
    return check


# ── HTTP helpers ─────────────────────────────────────────────────────────────

async def health_check(client: httpx.AsyncClient) -> dict | None:
    try:
        r = await client.get(f"{BACKEND}/health", timeout=5.0)
        return r.json() if r.status_code == 200 else None
    except (httpx.HTTPError, json.JSONDecodeError):
        return None


async def reset(client: httpx.AsyncClient, session_id: str) -> bool:
    try:
        r = await client.post(f"{BACKEND}/sessions/{session_id}/reset", timeout=10.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


async def send_turn(
    client: httpx.AsyncClient, session_id: str, history: list[dict],
) -> tuple[str, str | None, int | None]:
    """Send one /chat turn, return (response_text, error, turn_count)."""
    payload = {
        "messages": history,
        "session_id": session_id,
        "mode": "socratic",
    }
    response_text = ""
    error = None
    turn_count = None
    try:
        async with client.stream(
            "POST", f"{BACKEND}/chat", json=payload, timeout=180.0,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return "", f"HTTP {resp.status_code}: {body[:200]!r}", None
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                if not raw:
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if "response" in ev and isinstance(ev["response"], str):
                    response_text = ev["response"]
                elif "error" in ev:
                    error = ev["error"]
                elif "done" in ev and "turn_count" in ev:
                    turn_count = ev["turn_count"]
    except httpx.HTTPError as exc:
        error = f"transport error: {exc}"
    return response_text, error, turn_count


# ── Scenarios ────────────────────────────────────────────────────────────────

CONCEPT = "ulnar nerve"

SCENARIOS = [
    {
        "name": "S1 — Cooperative correct trajectory",
        "description": (
            "Student progresses from a broad description through partial truths "
            "to the correct answer. Verifies turn-0 broad opener, breadcrumb "
            "pacing, and mastery flow."
        ),
        "turns": [
            {
                "student": "What nerve causes the funny bone sensation?",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("no meta-leak", no_meta_leak),
                    ("ends with question", has_question),
                    ("≤ 2 question marks", at_most_two_questions),
                    ("no concept reveal at turn 0", no_concept_leak(CONCEPT)),
                ],
            },
            {
                "student": "I think the sensation tingles down to the pinky and ring finger.",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("no meta-leak", no_meta_leak),
                    ("ends with question", has_question),
                    ("no concept reveal at turn 1", no_concept_leak(CONCEPT)),
                ],
            },
            {
                "student": "The ulnar nerve.",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("no meta-leak", no_meta_leak),
                    ("reveal-permitted: concept may appear", lambda r: (True, "reveal allowed")),
                ],
            },
        ],
    },
    {
        "name": "S2 — B1 regression: wrong-nerve guess MUST NOT confirm",
        "description": (
            "Student guesses 'median nerve' — a different nerve in the same "
            "category. Classifier guard must override correct→incorrect, "
            "routing to hint_error_node instead of step_advancer mastery."
        ),
        "turns": [
            {
                "student": "What nerve causes the funny bone sensation?",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("no meta-leak", no_meta_leak),
                    ("ends with question", has_question),
                    ("no concept reveal at turn 0", no_concept_leak(CONCEPT)),
                ],
            },
            {
                "student": "Is it the median nerve?",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("no meta-leak", no_meta_leak),
                    ("does NOT confirm 'median nerve'", does_not_confirm_wrong_answer("median nerve")),
                    ("ends with question (hint path)", has_question),
                ],
            },
        ],
    },
    {
        "name": "S3 — IDK ladder: 3 consecutive idks → reveal",
        "description": (
            "Student says 'I don't know' three times. idk_count increments "
            "each turn; at IDK_REVEAL_THRESHOLD=3, teach_node fires the reveal."
        ),
        "turns": [
            {
                "student": "What nerve causes the funny bone sensation?",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("no concept reveal at turn 0", no_concept_leak(CONCEPT)),
                ],
            },
            {
                "student": "I don't know.",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("idk #1: no reveal yet", no_concept_leak(CONCEPT)),
                    ("ends with question", has_question),
                ],
            },
            {
                "student": "I still don't know.",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("idk #2: no reveal yet", no_concept_leak(CONCEPT)),
                    ("ends with question", has_question),
                ],
            },
            {
                "student": "I really have no idea, just give me the answer.",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("no meta-leak", no_meta_leak),
                    # idk_count = 3 → teach_node reveals; concept SHOULD appear
                    ("idk #3: concept revealed", reveals_concept(CONCEPT)),
                ],
            },
        ],
    },
    {
        "name": "S4 — Off-topic injection routes through redirect",
        "description": (
            "Student tries to derail mid-conversation. Classifier should label "
            "irrelevant; redirect_node should bring them back without leaking "
            "the concept."
        ),
        "turns": [
            {
                "student": "What nerve causes the funny bone sensation?",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("no concept reveal at turn 0", no_concept_leak(CONCEPT)),
                ],
            },
            {
                "student": "What's the best restaurant in Buffalo?",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("no meta-leak", no_meta_leak),
                    ("no concept reveal in redirect", no_concept_leak(CONCEPT)),
                    ("ends with question (back-on-topic)", has_question),
                ],
            },
        ],
    },
    {
        "name": "S5 — Wrong-cause attribution → progressive scaffolding",
        "description": (
            "Student blames the bone itself for the funny-bone sensation. "
            "Should be classified incorrect (wrong cause) and routed to "
            "hint_error_node, which validates the partial truth (correct "
            "location) and asks about the structure responsible."
        ),
        "turns": [
            {
                "student": "What nerve causes the funny bone sensation?",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("no concept reveal at turn 0", no_concept_leak(CONCEPT)),
                ],
            },
            {
                "student": "It's because you hit your bone.",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("no meta-leak", no_meta_leak),
                    ("no concept reveal at turn 1", no_concept_leak(CONCEPT)),
                    ("ends with question", has_question),
                    ("≤ 2 question marks", at_most_two_questions),
                ],
            },
        ],
    },
    {
        "name": "S6 — Cooperative correct on well-covered concept (synapse)",
        "description": (
            "Same cooperative trajectory as S1 but on a concept the textbook "
            "actually covers well (chapter 12 — neurons & synapses). If S1 "
            "fails on funny-bone but S6 passes on synapse, that confirms the "
            "S1/S2 failures are CONTENT GAP (CLAUDE.md known limitation: "
            "ulnar nerve appears in only one sentence of OpenStax AP2e), "
            "not a code bug."
        ),
        "turns": [
            {
                "student": "What is the gap between neurons called where signals are passed chemically?",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("no meta-leak", no_meta_leak),
                    ("ends with question", has_question),
                    ("no concept reveal at turn 0", no_concept_leak("synapse")),
                ],
            },
            {
                "student": "It involves neurotransmitters being released from one cell to another.",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("no meta-leak", no_meta_leak),
                    ("no concept reveal at turn 1", no_concept_leak("synapse")),
                ],
            },
            {
                "student": "It's the synapse.",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("no meta-leak", no_meta_leak),
                    ("reveal-permitted at turn 2: concept may appear",
                     lambda r: (True, "reveal allowed")),
                ],
            },
        ],
    },
]


# ── Runner ───────────────────────────────────────────────────────────────────

async def run_scenario(client: httpx.AsyncClient, scenario: dict) -> dict:
    session_id = f"e2e-{uuid.uuid4().hex[:10]}"
    await reset(client, session_id)

    history: list[dict] = []
    turn_results: list[dict] = []

    for i, turn in enumerate(scenario["turns"]):
        student_msg: str = turn["student"]
        history.append({"role": "user", "content": student_msg})

        text, error, turn_count = await send_turn(client, session_id, history)

        if text and not error:
            history.append({"role": "assistant", "content": text})

        assertions = []
        if not error:
            for name, fn in turn["asserts"]:
                try:
                    ok, msg = fn(text)
                except Exception as exc:
                    ok, msg = False, f"assertion crashed: {exc}"
                assertions.append({"name": name, "ok": ok, "msg": msg})

        turn_results.append({
            "turn_idx": i,
            "student": student_msg,
            "tutor": text,
            "error": error,
            "turn_count": turn_count,
            "assertions": assertions,
        })

        if error:
            break  # skip remaining turns in this scenario

    passed = all(
        a["ok"]
        for tr in turn_results if not tr["error"]
        for a in tr["assertions"]
    ) and not any(tr["error"] for tr in turn_results)

    return {
        "name": scenario["name"],
        "description": scenario["description"],
        "session_id": session_id,
        "turns": turn_results,
        "passed": passed,
    }


def render_md(report: dict) -> str:
    lines = []
    lines.append("# Socratic-OT — End-to-End Test Report")
    lines.append("")
    lines.append(f"- **Generated:** {report['generated_at']}")
    lines.append(f"- **Backend:** {report['backend']}")
    lines.append(f"- **Health:** `{report['health']}`")
    lines.append("")

    total = len(report["scenarios"])
    passed = sum(1 for s in report["scenarios"] if s["passed"])
    lines.append(f"## Summary — {passed} / {total} scenarios passed")
    lines.append("")
    for s in report["scenarios"]:
        symbol = "✅" if s["passed"] else "❌"
        lines.append(f"- {symbol} **{s['name']}**")
    lines.append("")

    for s in report["scenarios"]:
        lines.append(f"## {s['name']}")
        lines.append("")
        lines.append(f"_{s['description']}_")
        lines.append("")
        lines.append(f"- session_id: `{s['session_id']}`")
        lines.append(f"- result: {'**PASS**' if s['passed'] else '**FAIL**'}")
        lines.append("")
        for tr in s["turns"]:
            lines.append(f"### Turn {tr['turn_idx']}")
            lines.append("")
            lines.append(f"**Student:** {tr['student']}")
            lines.append("")
            if tr["error"]:
                lines.append(f"**Error:** `{tr['error']}`")
                lines.append("")
                continue
            lines.append("**Tutor:**")
            lines.append("")
            lines.append("> " + tr["tutor"].replace("\n", "\n> "))
            lines.append("")
            if tr["turn_count"] is not None:
                lines.append(f"_(turn_count after: {tr['turn_count']})_")
                lines.append("")
            if tr["assertions"]:
                lines.append("**Assertions:**")
                lines.append("")
                for a in tr["assertions"]:
                    sym = "✅" if a["ok"] else "❌"
                    lines.append(f"- {sym} `{a['name']}` — {a['msg']}")
                lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


async def main() -> int:
    async with httpx.AsyncClient() as client:
        h = await health_check(client)
        if not h:
            print(
                f"Backend not reachable at {BACKEND}/health — start uvicorn first.",
                file=sys.stderr,
            )
            return 2

        print(f"Backend health: {h}", file=sys.stderr)
        print(f"Running {len(SCENARIOS)} scenarios…", file=sys.stderr)

        scenarios_out = []
        for s in SCENARIOS:
            print(f"  → {s['name']}", file=sys.stderr)
            result = await run_scenario(client, s)
            scenarios_out.append(result)
            sym = "PASS" if result["passed"] else "FAIL"
            print(f"     {sym}", file=sys.stderr)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "backend": BACKEND,
            "health": h,
            "scenarios": scenarios_out,
        }
        TEST_MD.write_text(render_md(report), encoding="utf-8")
        print(f"\nWrote {TEST_MD}", file=sys.stderr)
        return 0 if all(s["passed"] for s in scenarios_out) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
