"""
api/main.py — FastAPI entry point for the Socratic-OT backend.

Exposes:
  GET  /health           — liveness probe (Cloud Run health checks)
  POST /chat             — Socratic or Study turn; SSE token stream
  POST /chat/trace       — same input as /chat, but emits step-by-step
                           events for the architecture visualizer

CORS allows localhost dev + any *.vercel.app preview/prod deployment.

Run locally:
    PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import json
import logging
from typing import Any

from fastapi import FastAPI
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


def _to_lc_messages(msgs: list[ChatMessage]) -> list[Any]:
    out = []
    for m in msgs:
        if m.role == "user":
            out.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            out.append(AIMessage(content=m.content))
    return out


def _initial_state(req: ChatRequest) -> dict:
    return {
        "messages":          _to_lc_messages(req.messages),
        "session_id":        req.session_id,
        "domain":            req.domain,
        "mode":              req.mode,
        "student_phase":     "learning",
        "turn_count":        0,
        "weak_topics":       [],
        "study_active_topic": "",
        "study_topic_count":  0,
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


@app.post("/chat/trace")
async def chat_trace(req: ChatRequest):
    """Same as /chat but also emits a 'state' event with the full final
    state (minus messages payload) so the architecture visualizer can show
    which path the graph took (crag_decision, mastery_level, classifier
    output, etc.).
    """
    thread_cfg = {"configurable": {"thread_id": req.session_id}}

    async def event_stream():
        try:
            result = await _invoke_graph(_initial_state(req), thread_cfg)
            # Strip non-JSON-serializable bits
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
