"""
api/main.py — FastAPI entry point for the Socratic-OT backend.

Exposes:
  GET  /health                    — liveness probe (Cloud Run health checks)
  POST /sessions                  — create a fresh session (returns session_id)
  GET  /sessions/{session_id}     — fetch persisted state for a session
  POST /chat                      — Socratic or Study turn; SSE response
  POST /chat/trace                — same input as /chat, plus per-step trace events

CORS allows localhost dev + any *.vercel.app preview/prod deployment.

Run locally:
    PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

import config

logger = logging.getLogger(__name__)
app = FastAPI(title="Socratic-OT API", version="0.1.0")

# CORS — Vercel preview URLs are always *.vercel.app, prod is fixed.
# allow_origin_regex covers preview deploys (socratic-ot-<hash>.vercel.app).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://socratic-ot.vercel.app",
    ],
    allow_origin_regex=r"https://socratic-ot[a-z0-9-]*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str    # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    session_id: str
    mode: str = "socratic"   # "socratic" | "study"
    domain: str = config.DOMAIN


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Liveness probe. Cloud Run hits this every 10s — must stay cheap."""
    return {"status": "ok", "version": app.version}


# ── Canonical demo traces ───────────────────────────────────────────────────

DEMO_TRACES_DIR = os.path.join(config._BASE_DIR, "data", "demo_traces")


@app.get("/demo/traces")
async def list_demo_traces() -> dict:
    """List the canonical pre-recorded trace files. The architecture
    visualizer's replay mode shows a dropdown of these.

    Each trace is a JSON file at data/demo_traces/<id>.json with shape:
        {"id": "<id>", "label": "<human label>", "events": [...]}

    The recording script (scripts/record_demo_trace.py) generates these
    by hitting POST /chat/trace and saving the SSE event stream.
    """
    if not os.path.isdir(DEMO_TRACES_DIR):
        return {"traces": []}
    out = []
    for fname in sorted(os.listdir(DEMO_TRACES_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(DEMO_TRACES_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            out.append({
                "id":          data.get("id", fname[:-5]),
                "label":       data.get("label", fname[:-5]),
                "event_count": len(data.get("events", [])),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return {"traces": out}


@app.get("/demo/traces/{trace_id}")
async def get_demo_trace(trace_id: str) -> dict:
    """Return the full event sequence for one canonical trace."""
    # Sanity: prevent path traversal
    if "/" in trace_id or ".." in trace_id:
        raise HTTPException(status_code=400, detail="invalid trace_id")
    path = os.path.join(DEMO_TRACES_DIR, f"{trace_id}.json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"trace {trace_id!r} not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Session lifecycle ───────────────────────────────────────────────────────

class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: str


@app.post("/sessions")
async def create_session() -> SessionCreateResponse:
    """Mint a fresh session id. The frontend stores it in localStorage and
    sends it as the thread_id on every subsequent /chat call. SqliteSaver
    creates the per-thread state on first /chat invocation; this endpoint
    just allocates the id.
    """
    return SessionCreateResponse(
        session_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _msg_to_dict(m: Any) -> dict:
    role = "user" if getattr(m, "type", "") == "human" else "assistant"
    content = m.content
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content
                           if isinstance(p, dict))
    return {"role": role, "content": content}


@app.post("/sessions/{session_id}/reset")
async def reset_session(session_id: str) -> dict:
    """Clear all checkpoint state for this session_id. Idempotent.

    Used by the frontend's "Reset session" button and by the demo
    runbook to clear stuck state mid-screencast without terminal
    access. Calling reset on an unknown session_id is a no-op (still
    200) — clients don't need to check existence first.

    Subsequent GET /sessions/{session_id} returns 404 (state cleared)
    until the next /chat call materializes a fresh checkpoint.

    SECURITY: this endpoint accepts any session_id from any caller. A
    public deployment must gate it behind auth; the demo doesn't.
    """
    if "/" in session_id or ".." in session_id:
        raise HTTPException(status_code=400, detail="invalid session_id")
    import sqlite3
    from graph.graph_builder import graph
    cp = graph.checkpointer
    delete = getattr(cp, "delete_thread", None)
    if callable(delete):
        try:
            delete(session_id)
            return {"session_id": session_id, "reset": True}
        except Exception:
            logger.exception("delete_thread raised; falling back to SQL")
    # Fallback: raw SQL across the canonical SqliteSaver tables. Older
    # langgraph-checkpoint-sqlite versions don't expose delete_thread.
    for table in ("checkpoints", "writes", "checkpoint_blobs"):
        try:
            cp.conn.execute(
                f"DELETE FROM {table} WHERE thread_id = ?", (session_id,))
        except sqlite3.OperationalError:
            pass  # Table absent in this schema version
    cp.conn.commit()
    return {"session_id": session_id, "reset": True}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    """Fetch the persisted state for a session. Used by the frontend to
    restore context after page reload (sidebar weak-topics, mode, turn count).

    Returns 404 if no checkpoint exists for this session_id yet (frontend
    should call POST /chat at least once before).
    """
    from graph.graph_builder import graph
    cfg = {"configurable": {"thread_id": session_id}}
    snapshot = graph.get_state(cfg)
    if snapshot is None or not snapshot.values:
        raise HTTPException(
            status_code=404,
            detail=f"No state for session_id={session_id!r}",
        )
    state = snapshot.values
    return {
        "session_id":      session_id,
        "mode":            state.get("mode", "socratic"),
        "turn_count":      state.get("turn_count", 0),
        "current_concept": state.get("current_concept", ""),
        "weak_topics":     state.get("weak_topics", []),
        "student_phase":   state.get("student_phase", "learning"),
        "concept_mastered": state.get("concept_mastered", False),
        "mastery_level":   state.get("mastery_level", ""),
        "messages":        [_msg_to_dict(m) for m in state.get("messages", [])],
    }


def _to_lc_messages(msgs: list[ChatMessage]) -> list[Any]:
    out = []
    for m in msgs:
        if m.role == "user":
            out.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            out.append(AIMessage(content=m.content))
    return out


def _initial_state(req: ChatRequest) -> dict:
    """Build the input dict for graph.invoke.

    Only include fields the request can legitimately change per turn.
    Everything else (turn_count, student_phase, weak_topics, idk_count,
    study_topic_count, …) is owned by the SqliteSaver checkpoint —
    passing defaults here OVERWRITES the persisted values, which would
    reset the reveal gate, clear the weak-topics sidebar, and break the
    post-mastery routing every turn. For new sessions the checkpoint is
    empty; nodes use state.get(key, default) so missing keys fall back
    safely to per-call defaults.

    Messages: the frontend sends the FULL conversation each /chat call
    for resilience; the checkpoint already holds the prior turns, so
    forwarding all of them would let add_messages duplicate the history
    every turn (state bloat, wrong message counts in dashboard). Take
    only the last user message — the new turn — and let add_messages
    append it to the persisted history.
    """
    last_user = next(
        (m for m in reversed(req.messages) if m.role == "user"),
        None,
    )
    new_msgs = [last_user] if last_user else []
    return {
        "messages":   _to_lc_messages(new_msgs),
        "session_id": req.session_id,
        "domain":     req.domain,
        "mode":       req.mode,
    }


def _last_ai_text(state: dict) -> str:
    """Pull the most recent AIMessage text out of returned graph state."""
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", "") == "ai":
            content = m.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(p.get("text", "") for p in content
                                if isinstance(p, dict))
    return ""


async def _invoke_graph(state: dict, thread_cfg: dict) -> dict:
    """Run the (sync) compiled graph on a thread pool so the FastAPI event
    loop stays responsive. Necessary because graph_builder uses the sync
    SqliteSaver — see graph/graph_builder.py:_make_checkpointer for why."""
    import asyncio
    from graph.graph_builder import graph
    return await asyncio.to_thread(graph.invoke, state, thread_cfg)


@app.post("/chat")
async def chat(req: ChatRequest):
    """Run one Socratic / Study turn and stream the response as SSE.

    Implementation note: the underlying SqliteSaver is sync (see
    graph_builder), so we cannot use astream_events (async) for token-level
    streaming. The whole turn is executed via asyncio.to_thread, then the
    final AIMessage is emitted as a single 'response' event followed by
    'done'. This is functional and safe; per-token streaming is a Phase 5
    polish item once AsyncSqliteSaver wiring is sorted.
    """
    thread_cfg = {"configurable": {"thread_id": req.session_id}}

    async def event_stream():
        try:
            result = await _invoke_graph(_initial_state(req), thread_cfg)
            text = _last_ai_text(result)
            if text:
                yield f"data: {json.dumps({'response': text})}\n\n"
            yield f"data: {json.dumps({'done': True, 'turn_count': result.get('turn_count', 0)})}\n\n"
        except Exception as exc:
            logger.exception("chat error")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def _invoke_with_trace(state: dict, thread_cfg: dict) -> tuple[dict, list]:
    """Run the graph with the trace collector active. Returns (final_state,
    trace_events). The collector is per-request via contextvars."""
    import asyncio
    from graph._trace import set_collector, drain
    from graph.graph_builder import graph

    events: list[dict] = []

    def _run():
        token = set_collector(events)
        try:
            return graph.invoke(state, thread_cfg)
        finally:
            drain(token)

    result = await asyncio.to_thread(_run)
    return result, events


@app.post("/chat/trace")
async def chat_trace(req: ChatRequest):
    """Same input as /chat but also emits one 'trace' SSE event per node
    that the architecture visualizer cares about (concept_extraction,
    retrieval, classifier, generation, dean, study). After the trace
    events, emits 'state', 'response', and 'done'.

    See docs/website.md §5.4 for the event schema.
    """
    thread_cfg = {"configurable": {"thread_id": req.session_id}}

    async def event_stream():
        try:
            result, trace_events = await _invoke_with_trace(
                _initial_state(req), thread_cfg)
            # Per-step trace events (in invocation order)
            for ev in trace_events:
                yield f"data: {json.dumps({'event': 'trace', **ev})}\n\n"
            # Final state summary (visualizer 'last card' + sidebar update)
            visible = {k: v for k, v in result.items()
                       if k != "messages" and isinstance(
                           v, (str, int, float, bool, list, dict, type(None)))}
            yield f"data: {json.dumps({'event': 'state', 'state': visible})}\n\n"
            text = _last_ai_text(result)
            if text:
                yield f"data: {json.dumps({'event': 'response', 'response': text})}\n\n"
            yield f"data: {json.dumps({'event': 'done'})}\n\n"
        except Exception as exc:
            logger.exception("trace error")
            yield f"data: {json.dumps({'event': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
