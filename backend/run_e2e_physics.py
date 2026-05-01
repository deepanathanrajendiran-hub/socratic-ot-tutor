"""
backend/run_e2e_physics.py — physics-domain E2E suite.

Companion to run_e2e_tests.py (OT). Same harness, same helpers, same
report shape — but 10 physics scenarios that exercise the core Socratic
behaviors against the `physics_chunks` collection.

Pre-requisites:
  - .env has DOMAIN=physics
  - uvicorn restarted with that env so DOMAIN_CONFIG['physics'] is live
  - 1141 physics chunks loaded (run scripts/parse_pdf.py physics
    + ingest/late_chunker.py --domain physics + load into ChromaDB)

The point of this suite is the brief's generalizability requirement:
  "Demonstrate that your code can work for a different subject (e.g.,
  Physics) simply by swapping the Vector Database."

We don't try to re-prove the entire 86-scenario OT suite passes on
physics — that would be redundant. Instead we hit each major behavior
once: turn-0 no-reveal, wrong-attempt scaffold, IDK ladder reveal,
off-topic redirect, vague-rapport narrowing, first-try mastery,
multi-turn wrong mechanism, mastery menu after correct, weak-topic
recording on failed concept, and concept-lock preservation.

Run:
    PYTHONPATH=. python3 run_e2e_physics.py

Output:
    test_physics.md  — markdown report (same render path as test.md)
"""
import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Reuse the OT runner's helpers + the runner main loop. Note: helpers
# like `no_concept_leak` and `is_redirect_or_chitchat` import from
# `graph.nodes._helpers` which reads DOMAIN_CONFIG — they Just Work for
# physics as long as `DOMAIN=physics` is set in .env when this file runs.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_e2e_tests import (  # noqa: E402
    has_question,
    at_most_two_questions,
    no_meta_leak,
    no_concept_leak,
    not_fallback_scaffold,
    does_not_confirm_wrong_answer,
    reveals_concept,
    non_empty,
    mastery_choice_menu,
    weak_topics_contains,
    is_redirect_or_chitchat,
    looks_like_rapport,
    run_scenario,
    health_check,
    BACKEND,
)


# ── Helper override: physics-aware redirect cues ─────────────────────────────
# The OT version of `is_redirect_or_chitchat` lists "anatomy" /
# "neuroscience" / "occupational therapy" as cues. For physics we want
# different cue words. We DO want to keep the short-with-question
# fallback in the OT helper, so we wrap rather than replace.
def is_physics_redirect_or_chitchat(text: str) -> tuple[bool, str]:
    lower = text.lower()
    cues = [
        "let's get back", "back to ", "let's stay focused",
        "let's stick with", "stay with", "physics", "mechanics",
        "this session", "today's session", "different direction",
    ]
    if any(p in lower for p in cues):
        return True, "physics-redirect phrasing present"
    if len(text) < 350 and "?" in text:
        return True, "short response with a question — likely redirect"
    return False, "no redirect signal in response"


# ── Scenarios (10) ───────────────────────────────────────────────────────────

SCENARIOS = [
    # ────────────────────────────────────────────────────────────────────
    # PA — Cooperative trajectories
    # ────────────────────────────────────────────────────────────────────
    {
        "name": "PA1 — Cooperative trajectory (Newton's second law)",
        "description": (
            "Open-ended physics question → progressive scaffolding → "
            "correct answer. Verifies turn-0 no-reveal + breadcrumb "
            "pacing on the new domain."
        ),
        "turns": [
            {
                "student": "What does Newton's second law actually say?",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("no meta-leak", no_meta_leak),
                    ("ends with question", has_question),
                    ("≤ 2 question marks", at_most_two_questions),
                    ("no concept reveal", no_concept_leak("Newton's second law")),
                ],
            },
            {
                "student": "Force equals mass times acceleration?",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("non-empty response", non_empty),
                ],
            },
        ],
    },
    {
        "name": "PA2 — Cooperative trajectory (kinetic energy)",
        "description": "Different physics concept — verifies the lock generalizes.",
        "turns": [
            {
                "student": "What's the formula that relates an object's mass and speed to its energy of motion?",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("ends with question", has_question),
                    ("no concept reveal", no_concept_leak("kinetic energy")),
                ],
            },
            {
                "student": "Half m v squared.",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("non-empty response", non_empty),
                ],
            },
        ],
    },
    {
        "name": "PA3 — Cooperative trajectory (projectile motion)",
        "description": "Concept lock + turn-0 broad opener for projectile motion.",
        "turns": [
            {
                "student": "How does gravity affect a ball thrown horizontally?",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("ends with question", has_question),
                    ("no concept reveal", no_concept_leak("projectile motion")),
                ],
            },
        ],
    },

    # ────────────────────────────────────────────────────────────────────
    # PB — Wrong attempts (no premature confirmation, no reveal)
    # ────────────────────────────────────────────────────────────────────
    {
        "name": "PB1 — Wrong attempt (force = mass)",
        "description": (
            "Common misconception — student says force = mass. Tutor "
            "must NOT confirm and must NOT reveal F=ma yet."
        ),
        "turns": [
            {
                "student": "What does Newton's second law actually say?",
                "asserts": [("no concept reveal", no_concept_leak("Newton's second law"))],
            },
            {
                "student": "Is it just that force equals mass?",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("does NOT confirm 'force equals mass'",
                     does_not_confirm_wrong_answer("force equals mass")),
                    ("ends with question", has_question),
                ],
            },
        ],
    },

    # ────────────────────────────────────────────────────────────────────
    # PC — IDK ladder → reveal via teach_node
    # ────────────────────────────────────────────────────────────────────
    {
        "name": "PC1 — Three IDKs → reveal of Newton's second law",
        "description": (
            "Three consecutive 'I don't know' messages must trigger "
            "reveal via teach_node (idk_count >= IDK_REVEAL_THRESHOLD)."
        ),
        "turns": [
            {"student": "What does Newton's second law actually say?",
             "asserts": [("no reveal", no_concept_leak("Newton's second law"))]},
            {"student": "I don't know.",
             "asserts": [("no reveal", no_concept_leak("Newton's second law"))]},
            {"student": "Still no idea.",
             "asserts": [("no reveal", no_concept_leak("Newton's second law"))]},
            {"student": "Just tell me, I give up.",
             "asserts": [("reveals concept",
                           reveals_concept("Newton's second law"))]},
        ],
    },

    # ────────────────────────────────────────────────────────────────────
    # PD — Off-topic / redirect
    # ────────────────────────────────────────────────────────────────────
    {
        "name": "PD1 — Off-topic injection (restaurant)",
        "description": (
            "Student injects a totally off-topic question. Tutor must "
            "redirect back to physics (not the previous OT 'anatomy' "
            "framing, which would be a generalizability bug)."
        ),
        "turns": [
            {
                "student": "What's the best restaurant in Buffalo?",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("non-empty response", non_empty),
                    ("redirect or short", is_physics_redirect_or_chitchat),
                ],
            },
        ],
    },

    # ────────────────────────────────────────────────────────────────────
    # PG — Vague rapport opener
    # ────────────────────────────────────────────────────────────────────
    {
        "name": "PG1 — Vague opener ('tell me about physics')",
        "description": (
            "Pure-vague opener — manager rejects the concept and rapport "
            "narrows. Must NOT crash, must NOT lock onto a wrong concept."
        ),
        "turns": [
            {
                "student": "Tell me about physics.",
                "asserts": [
                    ("not fallback_scaffold", not_fallback_scaffold),
                    ("non-empty response", non_empty),
                    ("rapport-style reply", looks_like_rapport),
                ],
            },
        ],
    },

    # ────────────────────────────────────────────────────────────────────
    # PI — First-try mastery
    # ────────────────────────────────────────────────────────────────────
    {
        "name": "PI1 — First-try mastery menu (kinetic energy)",
        "description": (
            "Student names the concept correctly on attempt 1 → mastery "
            "menu (A/B/C) fires."
        ),
        "turns": [
            {"student": "What's the formula that relates an object's mass and speed to its energy of motion?",
             "asserts": [("no reveal at turn 0",
                           no_concept_leak("kinetic energy"))]},
            {"student": "It's kinetic energy, KE = 1/2 m v squared.",
             "asserts": [("mastery menu present", mastery_choice_menu)]},
        ],
    },

    # ────────────────────────────────────────────────────────────────────
    # PL — Wrong-mechanism multi-turn scaffold
    # ────────────────────────────────────────────────────────────────────
    {
        "name": "PL2 — Wrong mechanism (force = velocity)",
        "description": (
            "Student claims force = velocity (a common conflation). "
            "Tutor scaffolds without revealing F=ma. Verifies multi-turn "
            "concept-lock preservation on physics."
        ),
        "turns": [
            {"student": "What does Newton's second law actually say?",
             "asserts": [("no reveal", no_concept_leak("Newton's second law"))]},
            {"student": "Doesn't force just equal velocity?",
             "asserts": [
                 ("not fallback_scaffold", not_fallback_scaffold),
                 ("no reveal", no_concept_leak("Newton's second law")),
                 ("ends with question", has_question),
                 ("≤ 2 question marks", at_most_two_questions),
             ]},
        ],
    },

    # ────────────────────────────────────────────────────────────────────
    # PW — Weak-topic recording
    # ────────────────────────────────────────────────────────────────────
    {
        "name": "PW1 — IDK reveal seeds weak_topics",
        "description": (
            "Concept revealed via IDK ladder must land in weak_topics so "
            "the dashboard can prioritize it next session."
        ),
        "turns": [
            {"student": "What does Newton's second law actually say?",
             "asserts": [("no reveal", no_concept_leak("Newton's second law"))]},
            {"student": "I don't know.",
             "asserts": [("no reveal", no_concept_leak("Newton's second law"))]},
            {"student": "Still no idea.",
             "asserts": [("no reveal", no_concept_leak("Newton's second law"))]},
            {"student": "Just tell me.",
             "asserts": [("reveals concept",
                           reveals_concept("Newton's second law"))]},
        ],
        "post_checks": [
            ("Newton's second law added to weak_topics after IDK reveal",
             weak_topics_contains("Newton's second law")),
        ],
    },
]


# ── Render + main ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
TEST_MD = ROOT / "test_physics.md"


def render_md(report: dict) -> str:
    """Reuse the OT renderer but retitle the document."""
    from run_e2e_tests import render_md as ot_render
    md = ot_render(report)
    return md.replace(
        "# Socratic-OT — End-to-End Test Report",
        "# Socratic-OT — Physics generalizability E2E report",
    )


async def main() -> int:
    async with httpx.AsyncClient() as client:
        h = await health_check(client)
        if not h:
            print(f"Backend not reachable at {BACKEND}/health — start uvicorn first.",
                  file=sys.stderr)
            return 2

        # Quick guard: warn loudly if the backend isn't actually on physics.
        # We can't introspect the running server's DOMAIN, so we read the
        # local config (same .env the server reads) and rely on the user
        # having restarted uvicorn after switching .env.
        try:
            import config
            if config.DOMAIN != "physics":
                print(
                    f"⚠️  Local config has DOMAIN={config.DOMAIN!r}, not 'physics'. "
                    f"If the backend wasn't restarted after .env change, "
                    f"these scenarios will hit the wrong collection.",
                    file=sys.stderr,
                )
        except Exception:
            pass

        print(f"Backend health: {h}", file=sys.stderr)
        print(f"Running {len(SCENARIOS)} physics scenarios…", file=sys.stderr)

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
        n_pass = sum(1 for s in scenarios_out if s["passed"])
        print(f"Score: {n_pass}/{len(scenarios_out)}", file=sys.stderr)
        return 0 if all(s["passed"] for s in scenarios_out) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
