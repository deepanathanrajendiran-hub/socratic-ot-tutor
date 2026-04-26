"""
scripts/record_demo_trace.py — hit POST /chat/trace and save the SSE
event stream to data/demo_traces/<id>.json for canonical replay.

Run this against a running backend (uvicorn api.main:app on localhost:8000)
with a working ANTHROPIC_API_KEY (or AWS Bedrock creds + LLM_PROVIDER=bedrock)
in the environment.

Usage:
    PYTHONPATH=. python3 scripts/record_demo_trace.py \\
        --id funny_bone \\
        --label "Funny bone — clean Socratic flow" \\
        --message "What nerve causes the funny bone sensation?"

Replay later via GET /demo/traces and GET /demo/traces/<id>.

Five canonical traces planned (per docs/website.md §9):
    funny_bone        — clean Socratic flow
    idk_x3            — three idks → reveal via teach_node
    off_topic         — out-of-scope (CRAG INCORRECT) → redirect
    ambiguous_refine  — CRAG AMBIGUOUS → refinement
    clinical_synth    — clinical synthesis + Dean grounding
"""
import argparse
import json
import os
import sys
import uuid

import httpx


def parse_sse_stream(body_text: str) -> list[dict]:
    """Parse a text/event-stream body into a list of decoded JSON events."""
    events: list[dict] = []
    for line in body_text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload:
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError as e:
            print(f"  [warn] could not parse SSE line: {payload!r} ({e})",
                  file=sys.stderr)
    return events


def record(backend_url: str, trace_id: str, label: str,
           message: str, mode: str = "socratic") -> dict:
    session_id = str(uuid.uuid4())
    payload = {
        "messages":   [{"role": "user", "content": message}],
        "session_id": session_id,
        "mode":       mode,
    }
    print(f"  POST {backend_url}/chat/trace  (session={session_id})")
    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", f"{backend_url}/chat/trace",
                           json=payload) as resp:
            resp.raise_for_status()
            body = "".join(resp.iter_text())
    events = parse_sse_stream(body)
    return {
        "id":         trace_id,
        "label":      label,
        "input":      payload,
        "events":     events,
        "event_count": len(events),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True,
                        help="canonical trace id (e.g. funny_bone)")
    parser.add_argument("--label", required=True,
                        help="human-readable label")
    parser.add_argument("--message", required=True,
                        help="student question to record")
    parser.add_argument("--mode", default="socratic",
                        choices=("socratic", "study"))
    parser.add_argument("--backend", default="http://localhost:8000")
    parser.add_argument("--out-dir",
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "backend", "data", "demo_traces"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.id}.json")
    if os.path.exists(out_path):
        print(f"  [warn] {out_path} exists; overwriting")

    trace = record(args.backend, args.id, args.label, args.message, args.mode)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
    print(f"  saved {len(trace['events'])} events → {out_path}")


if __name__ == "__main__":
    main()
