"""
backend/evaluation/physics_generalizability.py

Generalizability proof for the brief's "swap the vector DB to teach a
different subject" requirement.

What this does:
  Drives a scripted 4-turn Socratic conversation about Newton's second
  law against the live backend (POST /chat/trace) and captures the
  per-step trace + assistant turns to a markdown transcript.

Why a separate script:
  We don't want to add physics scenarios to run_e2e_tests.py — that
  suite is OT-anatomy specific, and physics is a different domain
  config. This is a one-shot artifact for the final report.

Pre-requisites:
  - .env has DOMAIN=physics (so the graph routes through physics_chunks)
  - uvicorn restarted with that env so the in-process config matches
  - 1141 physics chunks loaded (run_late_chunker → vector_store.load)

Output:
  evaluation/results/physics_transcript.md  — human-readable transcript
  evaluation/results/physics_generalizability.json — raw events
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx


BACKEND = "http://localhost:8000"
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "evaluation" / "results"
TRANSCRIPT_PATH = RESULTS_DIR / "physics_transcript.md"
RAW_PATH = RESULTS_DIR / "physics_generalizability.json"


# Cooperative student trajectory through Newton's second law:
#  1. Asks the topic
#  2. Wrong attempt (force = mass)
#  3. Closer attempt
#  4. Correct
SCRIPT = [
    "What does Newton's second law actually say?",
    "Is it just that force equals mass?",
    "Force equals mass times acceleration?",
    "F = m * a",
]


def _parse_sse(body: str) -> list[dict]:
    out = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        try:
            out.append(json.loads(payload))
        except json.JSONDecodeError:
            pass
    return out


async def _send(client: httpx.AsyncClient, session_id: str, message: str) -> dict:
    payload = {
        "messages":   [{"role": "user", "content": message}],
        "session_id": session_id,
        "mode":       "socratic",
    }
    async with client.stream("POST", f"{BACKEND}/chat/trace", json=payload,
                             timeout=180.0) as resp:
        resp.raise_for_status()
        body = "".join([chunk async for chunk in resp.aiter_text()])
    events = _parse_sse(body)
    response = next((e["response"] for e in events if e.get("event") == "response"), "")
    state    = next((e["state"]    for e in events if e.get("event") == "state"),    {})
    traces   = [e for e in events if e.get("event") == "trace"]
    return {
        "student":       message,
        "tutor":         response,
        "state_snapshot": {
            "current_concept":  state.get("current_concept"),
            "classifier_output": state.get("classifier_output"),
            "crag_decision":    state.get("crag_decision"),
            "turn_count":       state.get("turn_count"),
            "concept_mastered": state.get("concept_mastered"),
            "mastery_level":    state.get("mastery_level"),
        },
        "traces": [
            {
                "step":        t.get("step"),
                "duration_ms": t.get("duration_ms"),
                "decision":    t.get("output", {}).get("crag_decision"),
                "concept":     t.get("output", {}).get("current_concept"),
            }
            for t in traces
        ],
    }


def _render_md(turns: list[dict], domain_info: dict) -> str:
    lines: list[str] = []
    lines.append("# Generalizability — Physics domain swap")
    lines.append("")
    lines.append(f"_Generated_: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("")
    lines.append("**Setup**")
    lines.append("")
    lines.append(f"- `DOMAIN` = `{domain_info.get('domain')}`")
    lines.append(f"- `COLLECTION_NAME` = `{domain_info.get('collection')}`")
    lines.append(f"- Chunks loaded: **{domain_info.get('chunks_count')}**")
    lines.append(f"- Textbook: `{domain_info.get('textbook')}`")
    lines.append("")
    lines.append(
        "Same Python code path as the OT demo — only the vector-DB collection "
        "and DOMAIN_CONFIG entry changed. No node code, no prompt, no "
        "retrieval logic, no graph wiring was touched."
    )
    lines.append("")

    for i, t in enumerate(turns):
        lines.append(f"## Turn {i}")
        lines.append("")
        lines.append(f"**Student:** {t['student']}")
        lines.append("")
        lines.append("**Tutor:**")
        lines.append("")
        lines.append("> " + (t["tutor"] or "(empty)").replace("\n", "\n> "))
        lines.append("")
        snap = t.get("state_snapshot", {})
        lines.append("**State after turn:**")
        lines.append(
            f"- `current_concept`: `{snap.get('current_concept')!r}` "
            f"· `crag_decision`: `{snap.get('crag_decision')}` "
            f"· `classifier`: `{snap.get('classifier_output')}` "
            f"· `mastery_level`: `{snap.get('mastery_level')}` "
            f"· `concept_mastered`: `{snap.get('concept_mastered')}`"
        )
        steps = " → ".join(tr.get("step", "?") for tr in t.get("traces", []))
        lines.append(f"- node trace: `{steps}`")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


async def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Pull live domain info from the backend so the report doesn't lie if
    # the env wasn't actually swapped.
    async with httpx.AsyncClient() as client:
        # Backend doesn't expose /config; we rely on the runner's local
        # config import which reads the same .env the server reads.
        sys.path.insert(0, str(ROOT))
        import config
        from ingest.vector_store import VectorStore
        vs = VectorStore(config.CHROMA_DIR, config.DOMAIN)
        stats = vs.get_collection_stats()
        domain_info = {
            "domain":       config.DOMAIN,
            "collection":   config.COLLECTION_NAME,
            "chunks_count": stats.get("chunks_count"),
            "textbook":     config.DOMAIN_CONFIG.get(
                config.DOMAIN, {}
            ).get("textbook"),
        }
        print(f"Domain info: {domain_info}", file=sys.stderr)

        if config.DOMAIN != "physics":
            print(
                f"  WARNING: backend was started with DOMAIN={config.DOMAIN!r}, "
                f"not 'physics'. Restart uvicorn with DOMAIN=physics in .env "
                f"or this transcript will hit the OT collection.",
                file=sys.stderr,
            )

        session_id = f"physics-gen-{uuid.uuid4()}"
        print(f"Session: {session_id}", file=sys.stderr)

        turns: list[dict] = []
        for i, msg in enumerate(SCRIPT):
            print(f"  Turn {i}: {msg!r}", file=sys.stderr)
            row = await _send(client, session_id, msg)
            print(
                f"    decision={row['state_snapshot'].get('crag_decision')} "
                f"concept={row['state_snapshot'].get('current_concept')!r} "
                f"mastery={row['state_snapshot'].get('mastery_level')}",
                file=sys.stderr,
            )
            turns.append(row)

    RAW_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domain_info":  domain_info,
        "session_id":   session_id,
        "turns":        turns,
    }, indent=2), encoding="utf-8")
    TRANSCRIPT_PATH.write_text(_render_md(turns, domain_info), encoding="utf-8")

    print(f"\n  Wrote {TRANSCRIPT_PATH}", file=sys.stderr)
    print(f"  Wrote {RAW_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
