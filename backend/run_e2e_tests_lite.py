"""
run_e2e_tests_lite.py — lightweight E2E suite (50 scenarios, no edge cases).

Like run_e2e_tests.py but smaller in scope. Each scenario is a single-turn
exchange against /chat/trace with simple assertions. By design, 10 of the
50 scenarios assert behavior that the current architecture does NOT
provide on turn 0 (e.g. requiring the tutor to reveal the answer on the
first turn, requiring an A/B/C menu to appear before mastery, etc.). When
the suite is run against a healthy backend, those 10 scenarios should
fail and the other 40 should pass.

Usage:
    PYTHONPATH=backend python3 backend/run_e2e_tests_lite.py

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
from typing import Callable

import httpx


BACKEND = os.getenv("E2E_BACKEND", "http://localhost:8000")
ROOT = Path(__file__).resolve().parent.parent
TEST_MD = ROOT / "test_lite.md"


# ── Assertion helpers ────────────────────────────────────────────────────────

def has_question(text: str) -> tuple[bool, str]:
    return ("?" in text), ("contains '?'" if "?" in text else "no '?' anywhere")


def at_most_two_questions(text: str) -> tuple[bool, str]:
    n = text.count("?")
    return (n <= 2), f"{n} '?' in response"


def non_empty(text: str) -> tuple[bool, str]:
    n = len(text.strip())
    return (n >= 10), f"response length: {n} chars"


def no_meta_leak(text: str) -> tuple[bool, str]:
    bad = [
        "the retrieved content", "according to the textbook",
        "the textbook says", "the passage says",
        "based on the retrieved", "my knowledge base", "training data",
        "we're at turn", "this is turn",
    ]
    lower = text.lower()
    for phrase in bad:
        if phrase in lower:
            return False, f"leaked phrase: {phrase!r}"
    return True, "no meta phrases"


def not_fallback_scaffold(text: str) -> tuple[bool, str]:
    if text.startswith("Let's take a step back"):
        return False, "fallback_scaffold fired"
    return True, "not the fallback message"


def reveals_concept(concept: str) -> Callable[[str], tuple[bool, str]]:
    """Asserts the tutor DID reveal the concept verbatim."""
    def check(text: str) -> tuple[bool, str]:
        if concept.lower() in text.lower():
            return True, f"concept {concept!r} present"
        return False, f"expected {concept!r} to appear but it didn't"
    return check


def mastery_choice_menu(text: str) -> tuple[bool, str]:
    """Asserts the response offers an A/B/C choice menu."""
    lower = text.lower()
    has_a = "a)" in lower or "a." in lower or "a:" in lower
    has_b = "b)" in lower or "b." in lower or "b:" in lower
    has_c = "c)" in lower or "c." in lower or "c:" in lower
    if has_a and has_b and has_c:
        return True, "A/B/C menu present"
    return False, f"missing menu items: A={has_a} B={has_b} C={has_c}"


def looks_like_clinical_scenario(text: str) -> tuple[bool, str]:
    """Asserts the response is a clinical scenario (patient/case framing)."""
    lower = text.lower()
    cues = ["patient", "case", "presents", "year-old", "clinical",
            "occupational therap", "history of", "complains of"]
    has_cue = any(c in lower for c in cues)
    long_enough = len(text.strip()) >= 200
    if has_cue and "?" in text and long_enough:
        return True, f"clinical cues + '?' + {len(text)} chars"
    return False, f"clinical signals missing (cue={has_cue})"


def looks_like_session_close(text: str) -> tuple[bool, str]:
    lower = text.lower()
    cues = ["good luck", "great work", "see you", "goodbye", "take care",
            "session ended", "ending the session", "until next"]
    if any(c in lower for c in cues):
        return True, "close-out phrase present"
    return False, "no close-out signal"


def min_length(n: int) -> Callable[[str], tuple[bool, str]]:
    def check(text: str) -> tuple[bool, str]:
        L = len(text.strip())
        return (L >= n), f"length {L} (min {n})"
    return check


def max_length(n: int) -> Callable[[str], tuple[bool, str]]:
    def check(text: str) -> tuple[bool, str]:
        L = len(text.strip())
        return (L <= n), f"length {L} (max {n})"
    return check


def must_be_fallback(text: str) -> tuple[bool, str]:
    """Designed-to-fail check: requires fallback_scaffold message verbatim."""
    if text.startswith("Let's take a step back"):
        return True, "fallback_scaffold fired"
    return False, "expected fallback_scaffold but got a normal response"


# ── HTTP helpers ─────────────────────────────────────────────────────────────

async def health_check(client: httpx.AsyncClient) -> dict | None:
    try:
        r = await client.get(f"{BACKEND}/health", timeout=5.0)
        return r.json() if r.status_code == 200 else None
    except (httpx.HTTPError, json.JSONDecodeError):
        return None


async def reset(client: httpx.AsyncClient, session_id: str) -> bool:
    try:
        r = await client.post(f"{BACKEND}/sessions/{session_id}/reset",
                              timeout=10.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


async def send_turn(
    client: httpx.AsyncClient, session_id: str, history: list[dict],
) -> tuple[str, str | None, int | None]:
    payload = {"messages": history, "session_id": session_id, "mode": "socratic"}
    response_text = ""
    error: str | None = None
    turn_count: int | None = None
    try:
        async with client.stream(
            "POST", f"{BACKEND}/chat/trace", json=payload, timeout=180.0,
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
                kind = ev.get("event")
                if kind == "state":
                    snapshot = ev.get("state", {})
                    if "turn_count" in snapshot:
                        turn_count = snapshot["turn_count"]
                elif kind == "response" and isinstance(ev.get("response"), str):
                    response_text = ev["response"]
                elif kind == "error":
                    error = ev.get("message", "unknown error")
    except httpx.HTTPError as exc:
        error = f"transport error: {exc}"
    return response_text, error, turn_count


# ── Scenarios ────────────────────────────────────────────────────────────────
#
# 50 single-turn scenarios. The first 40 (P01–P40) use lenient assertions
# and should pass against a healthy backend. The last 10 (F01–F10) assert
# behavior that the current turn-gate / Dean / mastery-flow architecture
# does NOT produce on turn 0, so they are expected to fail.

PASS_ASSERTS = [
    ("not fallback_scaffold", not_fallback_scaffold),
    ("non-empty response", non_empty),
    ("no meta-leak", no_meta_leak),
    ("≤ 2 question marks", at_most_two_questions),
]


def passing(name: str, question: str) -> dict:
    return {
        "name": name,
        "description": f"Single-turn pass scenario: {question!r}",
        "turns": [{"student": question, "asserts": list(PASS_ASSERTS)}],
    }


SCENARIOS = [
    # ── P01–P40: should-pass single-turn anatomy / neuroscience prompts ────
    passing("P01 — synapse opener",            "Can you tell me about how neurons communicate?"),
    passing("P02 — action potential opener",   "How does an electrical signal travel along a neuron?"),
    passing("P03 — myelin opener",             "Why is myelin important for nerve conduction?"),
    passing("P04 — axon vs dendrite",          "What is the difference between an axon and a dendrite?"),
    passing("P05 — neurotransmitter",          "What role do neurotransmitters play in the nervous system?"),
    passing("P06 — spinal cord",               "What does the spinal cord do?"),
    passing("P07 — gray vs white matter",      "What is the difference between gray and white matter?"),
    passing("P08 — dorsal root",               "What is the function of the dorsal root of a spinal nerve?"),
    passing("P09 — ventral root",              "What does the ventral root of a spinal nerve carry?"),
    passing("P10 — reflex arc",                "Can you walk me through how a reflex arc works?"),
    passing("P11 — cerebellum",                "What is the cerebellum responsible for?"),
    passing("P12 — cerebrum",                  "What does the cerebrum control?"),
    passing("P13 — thalamus",                  "What is the role of the thalamus?"),
    passing("P14 — hypothalamus",              "What does the hypothalamus regulate?"),
    passing("P15 — medulla",                   "What does the medulla oblongata do?"),
    passing("P16 — pons",                      "What functions are associated with the pons?"),
    passing("P17 — motor cortex",              "What is the primary motor cortex?"),
    passing("P18 — sensory cortex",            "What does the somatosensory cortex do?"),
    passing("P19 — corpus callosum",           "What is the corpus callosum and why does it matter?"),
    passing("P20 — basal ganglia",             "What role do the basal ganglia play in movement?"),
    passing("P21 — median nerve",              "What does the median nerve innervate?"),
    passing("P22 — ulnar nerve",               "What is the ulnar nerve responsible for?"),
    passing("P23 — radial nerve",              "What does the radial nerve do?"),
    passing("P24 — biceps",                    "What action does the biceps brachii produce?"),
    passing("P25 — triceps",                   "What is the main role of the triceps brachii?"),
    passing("P26 — deltoid",                   "What movements does the deltoid muscle produce?"),
    passing("P27 — quadriceps",                "What do the quadriceps do at the knee?"),
    passing("P28 — gluteus maximus",           "What is the function of the gluteus maximus?"),
    passing("P29 — synovial joint",            "What is a synovial joint?"),
    passing("P30 — hinge joint",               "What is a hinge joint and where do we find one?"),
    passing("P31 — ball-and-socket",           "What is a ball-and-socket joint?"),
    passing("P32 — skeletal muscle",           "What makes skeletal muscle different from smooth muscle?"),
    passing("P33 — cardiac muscle",            "What is special about cardiac muscle tissue?"),
    passing("P34 — smooth muscle",             "Where in the body do we find smooth muscle?"),
    passing("P35 — sarcomere",                 "What is a sarcomere?"),
    passing("P36 — actin and myosin",          "How do actin and myosin interact during contraction?"),
    passing("P37 — neuromuscular junction",    "What happens at the neuromuscular junction?"),
    passing("P38 — sensory pathway",           "How does sensory information travel to the brain?"),
    passing("P39 — motor pathway",             "How does the brain send a movement signal to a muscle?"),
    passing("P40 — autonomic nervous system",  "What does the autonomic nervous system regulate?"),

    # ── F01–F10: designed-to-fail scenarios ────────────────────────────────
    # Each of these asserts a behavior that the current architecture does
    # NOT produce on turn 0 (turn-gate prevents reveal, mastery menu only
    # appears post-reveal, clinical scenario only fires on Choice-A, etc.).
    {
        "name": "F01 — expects reveal of 'synapse' on turn 0",
        "description": "Asserts the tutor names 'synapse' on turn 0; turn-gate blocks reveal.",
        "turns": [{
            "student": "What is the gap between two neurons called?",
            "asserts": [("reveals 'synapse' on turn 0", reveals_concept("synapse"))],
        }],
    },
    {
        "name": "F02 — expects reveal of 'action potential' on turn 0",
        "description": "Asserts the tutor names 'action potential' on turn 0.",
        "turns": [{
            "student": "What's the rapid signal that fires down a neuron?",
            "asserts": [("reveals 'action potential'", reveals_concept("action potential"))],
        }],
    },
    {
        "name": "F03 — expects reveal of 'myelin' on turn 0",
        "description": "Asserts the tutor names 'myelin' on turn 0.",
        "turns": [{
            "student": "What is the fatty insulating layer around an axon called?",
            "asserts": [("reveals 'myelin'", reveals_concept("myelin"))],
        }],
    },
    {
        "name": "F04 — expects mastery A/B/C menu on turn 0",
        "description": "Asserts an A/B/C menu appears on a fresh question; menu only appears post-reveal.",
        "turns": [{
            "student": "What is the function of the cerebellum?",
            "asserts": [("A/B/C menu present", mastery_choice_menu)],
        }],
    },
    {
        "name": "F05 — expects clinical scenario on turn 0",
        "description": "Asserts a 200+ char clinical case appears on turn 0; clinical scenarios only fire on Choice-A post-mastery.",
        "turns": [{
            "student": "What does the thalamus do?",
            "asserts": [("clinical scenario", looks_like_clinical_scenario)],
        }],
    },
    {
        "name": "F06 — expects fallback_scaffold to fire",
        "description": "Asserts the fallback message fires on a normal question; only fires when Dean revisions exhaust.",
        "turns": [{
            "student": "What does the medulla oblongata do?",
            "asserts": [("fallback_scaffold fired", must_be_fallback)],
        }],
    },
    {
        "name": "F07 — expects session-close phrasing on turn 0",
        "description": "Asserts a goodbye-style closer on turn 0; only happens on Choice-C.",
        "turns": [{
            "student": "What is the function of the corpus callosum?",
            "asserts": [("session close phrase", looks_like_session_close)],
        }],
    },
    {
        "name": "F08 — expects ≥ 5000-char response on turn 0",
        "description": "Asserts a single Socratic opener exceeds 5000 chars; openers are short by design.",
        "turns": [{
            "student": "What does the hypothalamus regulate?",
            "asserts": [("response ≥ 5000 chars", min_length(5000))],
        }],
    },
    {
        "name": "F09 — expects ≤ 50-char response on turn 0",
        "description": "Asserts the opener is ≤ 50 chars; teacher_socratic openers are several sentences long.",
        "turns": [{
            "student": "What does the autonomic nervous system regulate?",
            "asserts": [("response ≤ 50 chars", max_length(50))],
        }],
    },
    {
        "name": "F10 — expects reveal of 'axon' on turn 0",
        "description": "Asserts 'axon' is revealed verbatim on turn 0; turn-gate blocks it.",
        "turns": [{
            "student": "What is the long fiber that carries a nerve signal away from the cell body?",
            "asserts": [("reveals 'axon'", reveals_concept("axon"))],
        }],
    },
]


# ── Runner ───────────────────────────────────────────────────────────────────

async def run_scenario(client: httpx.AsyncClient, scenario: dict) -> dict:
    session_id = f"e2e-lite-{uuid.uuid4().hex[:10]}"
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
            break

    passed = (
        all(a["ok"] for tr in turn_results if not tr["error"] for a in tr["assertions"])
        and not any(tr["error"] for tr in turn_results)
    )

    return {
        "name": scenario["name"],
        "description": scenario["description"],
        "session_id": session_id,
        "turns": turn_results,
        "passed": passed,
    }


def render_md(report: dict) -> str:
    lines = []
    lines.append("# Socratic-OT — Lite E2E Test Report")
    lines.append("")
    lines.append(f"- **Generated:** {report['generated_at']}")
    lines.append(f"- **Backend:** {report['backend']}")
    lines.append(f"- **Health:** `{report['health']}`")
    lines.append("")

    total = len(report["scenarios"])
    passed = sum(1 for s in report["scenarios"] if s["passed"])
    failed = total - passed
    lines.append(f"## Summary — {passed} / {total} scenarios passed ({failed} failed)")
    lines.append("")
    for s in report["scenarios"]:
        symbol = "PASS" if s["passed"] else "FAIL"
        lines.append(f"- [{symbol}] **{s['name']}**")
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
                    sym = "PASS" if a["ok"] else "FAIL"
                    lines.append(f"- [{sym}] `{a['name']}` — {a['msg']}")
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

        filter_raw = os.getenv("E2E_FILTER", "").strip()
        prefixes = [p.strip() for p in filter_raw.split(",") if p.strip()]
        if prefixes:
            scenarios = [s for s in SCENARIOS
                         if any(s["name"].startswith(p) for p in prefixes)]
            print(
                f"Backend health: {h}\nFilter: {prefixes!r} → "
                f"{len(scenarios)} of {len(SCENARIOS)} scenarios",
                file=sys.stderr,
            )
        else:
            scenarios = SCENARIOS
            print(f"Backend health: {h}", file=sys.stderr)
            print(f"Running {len(scenarios)} scenarios…", file=sys.stderr)

        scenarios_out = []
        for s in scenarios:
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
