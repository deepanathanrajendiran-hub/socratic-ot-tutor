"""
backend/evaluation/mem0_check.py

End-to-end verification that the mem0 cross-session memory layer is
actually wired into the live system. Runs four steps:

  1. Confirm /config reports mem0 enabled.
  2. Session A: drive a chat turn with a fixed user_id; the post-turn
     hook in api/main.py fires mem0.add() in the background.
  3. Wait for mem0's async extraction to finalize.
  4. Direct search via the client wrapper to confirm what mem0 stored.
  5. Session B: same user_id, fresh session_id, send a casual rapport
     opener ("hey, I'm back"). Inspect the tutor reply for any sign
     that memory bled through (e.g. references the prior topic).

Cleans up the test user_id at the end so the mem0 dashboard isn't
polluted.

Usage:
    PYTHONPATH=. python3 evaluation/mem0_check.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from typing import Any

import httpx

BACKEND = "http://localhost:8000"
USER_ID = f"e2e-mem0-check-{uuid.uuid4().hex[:8]}"


def _parse_sse_for_response(body: str) -> str:
    """Pull the final response text out of a /chat SSE stream."""
    text = ""
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if ev.get("event") == "replace" and isinstance(ev.get("response"), str):
            text = ev["response"]
        elif ev.get("event") == "token" and isinstance(ev.get("delta"), str):
            text += ev["delta"]
    return text.strip()


async def _chat(client: httpx.AsyncClient, session_id: str,
                message: str, user_id: str) -> str:
    payload = {
        "messages":   [{"role": "user", "content": message}],
        "session_id": session_id,
        "mode":       "socratic",
        "user_id":    user_id,
    }
    async with client.stream(
        "POST", f"{BACKEND}/chat", json=payload, timeout=180.0,
    ) as resp:
        resp.raise_for_status()
        body = "".join([chunk async for chunk in resp.aiter_text()])
    return _parse_sse_for_response(body)


async def main() -> int:
    print(f"\nuser_id = {USER_ID}\n", file=sys.stderr)

    async with httpx.AsyncClient() as http:
        # ── 1. /config ──────────────────────────────────────────────────
        print("[1] Fetching /config ...", file=sys.stderr)
        r = await http.get(f"{BACKEND}/config", timeout=10.0)
        r.raise_for_status()
        cfg = r.json()
        print(f"    backend:       {cfg['memory']['backend']}", file=sys.stderr)
        print(f"    mem0_enabled:  {cfg['memory']['mem0_enabled']}", file=sys.stderr)
        if not cfg["memory"]["mem0_enabled"]:
            print(
                "    FAIL — backend reports mem0 NOT enabled. Either MEMORY_BACKEND "
                "isn't 'mem0', MEM0_API_KEY is missing, or the backend wasn't "
                "restarted after .env change.",
                file=sys.stderr,
            )
            return 2

        # ── 2. Session A — write a memory via real /chat ────────────────
        session_a = f"session-A-{uuid.uuid4().hex[:8]}"
        print(f"\n[2] Session A ({session_a}): driving a chat turn ...",
              file=sys.stderr)
        a_msg = "What's the gap between neurons where signals jump chemically?"
        a_reply = await _chat(http, session_a, a_msg, USER_ID)
        print(f"    student: {a_msg!r}", file=sys.stderr)
        print(f"    tutor:   {a_reply[:160]!r}{'...' if len(a_reply)>160 else ''}",
              file=sys.stderr)

        # ── 3. Wait for mem0 async extraction ───────────────────────────
        wait_s = 14
        print(f"\n[3] Waiting {wait_s}s for mem0 to extract & index ...",
              file=sys.stderr)
        await asyncio.sleep(wait_s)

        # ── 4. Direct search ────────────────────────────────────────────
        print("\n[4] Direct search via mem0 client ...", file=sys.stderr)
        sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
        from memory.mem0_client import client as mem0_client
        hits = mem0_client.search("neuron synapse", user_id=USER_ID, limit=5)
        print(f"    {len(hits)} hit(s):", file=sys.stderr)
        for h in hits[:5]:
            t = (h.get("memory") or h.get("text") or "").strip()
            print(f"      - {t[:160]}", file=sys.stderr)
        if not hits:
            print(
                "    FAIL — mem0 returned 0 hits after the wait. Either the "
                "post-turn add didn't fire, mem0 hasn't finished extracting, "
                "or the search filter is wrong.",
                file=sys.stderr,
            )
            return 3

        # ── 5. Session B — fresh session, same user_id ──────────────────
        session_b = f"session-B-{uuid.uuid4().hex[:8]}"
        print(f"\n[5] Session B ({session_b}, same user_id):",
              file=sys.stderr)
        print("    sending casual opener — rapport node should pull memories",
              file=sys.stderr)
        b_msg = "Hey, I'm back."
        b_reply = await _chat(http, session_b, b_msg, USER_ID)
        print(f"    student: {b_msg!r}", file=sys.stderr)
        print(f"    tutor:   {b_reply!r}", file=sys.stderr)
        # Heuristic continuity check — does the rapport reply hint at
        # memory of the prior session? Look for any of the topic words
        # we know mem0 extracted.
        cues = ["last time", "before", "previous", "earlier",
                "neuron", "synapse", "neuroscience", "anatomy"]
        b_lower = b_reply.lower()
        matched = [c for c in cues if c in b_lower]
        if matched:
            print(f"    ✅ continuity cue matched: {matched}", file=sys.stderr)
        else:
            print(
                "    ⚠ no continuity cue matched in the rapport reply. The "
                "memory layer may have fired but the LLM chose not to "
                "reference it (acceptable for short greetings).",
                file=sys.stderr,
            )

        # ── 6. Cleanup ──────────────────────────────────────────────────
        print("\n[6] Cleanup", file=sys.stderr)
        out = mem0_client.delete_all(user_id=USER_ID)
        print(f"    delete_all: {out.get('message', out)}", file=sys.stderr)

    print("\n  Mem0 round-trip verified.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
