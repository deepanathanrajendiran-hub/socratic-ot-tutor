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
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage

import config
from api import sessions_meta, user_weak_topics

logger = logging.getLogger(__name__)
app = FastAPI(title="Socratic-OT API", version="0.1.0")


@app.on_event("startup")
def _bootstrap_meta_schema() -> None:
    """Ensure persistent metadata tables exist before serving any request."""
    sessions_meta.init_schema()
    user_weak_topics.init_schema()


def _mem0_enabled() -> bool:
    """Helper for /config — never raises even if mem0ai isn't installed."""
    try:
        from memory.mem0_client import client as mem0_client
        return bool(mem0_client.enabled)
    except Exception:
        return False

# CORS — Vercel preview URLs are always *.vercel.app, prod is fixed.
# allow_origin_regex covers preview deploys (socratic-ot-<hash>.vercel.app).
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_origin_regex=config.CORS_ORIGIN_REGEX,
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
    # Optional base64-encoded image (the frontend strips the data: prefix
    # before sending). When present, the graph routes through vlm_node
    # for identification + Socratic opener instead of the text path.
    image_b64: str | None = None
    # Optional stable per-browser user id (from localStorage). Used by
    # the cross-session memory layer (mem0) to scope facts to a user
    # across sessions. Ignored when MEMORY_BACKEND=sqlite.
    user_id: str | None = None


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Liveness probe. Cloud Run hits this every 10s — must stay cheap."""
    return {"status": "ok", "version": app.version}


@app.get("/config")
async def get_config() -> dict:
    """Resolved deployment configuration — useful for verifying that a
    Cloud Run / Vercel deploy is reading the env vars you think it is.

    Returns ONLY non-secret values:
      - the active provider, domain, model IDs (so you can confirm the
        right Sonnet/Haiku / Bedrock-vs-Anthropic resolution)
      - per-node `model_for(...)` resolutions (confirms overrides)
      - server / CORS / DB-path config
      - Socratic gates, retrieval thresholds, eval targets

    NEVER returns:
      - any API key
      - AWS credentials
      - the contents of any file (just paths)

    Auth: NONE. Safe to expose since secrets are filtered, but if you
    want to lock it down behind auth before public deploy, do so in the
    proxy / gateway layer.
    """
    nodes = (
        "teacher", "dean", "vlm", "study", "clinical", "teach",
        "explain", "hint", "redirect", "step_advancer",
        "topic_choice", "synthesis",
    )
    return {
        "version": app.version,
        "build_time": datetime.now(timezone.utc).isoformat(),
        "domain": {
            "active":            config.DOMAIN,
            "available":         list(config.DOMAIN_CONFIG.keys()),
            "collection_name":   config.COLLECTION_NAME,
            "system_context":    config.DOMAIN_CONFIG.get(
                config.DOMAIN, {}
            ).get("system_context", ""),
            "textbook":          config.DOMAIN_CONFIG.get(
                config.DOMAIN, {}
            ).get("textbook", ""),
        },
        "llm": {
            "provider":      config.LLM_PROVIDER,
            "primary_model": config.PRIMARY_MODEL,
            "fast_model":    config.FAST_MODEL,
            "vision_model":  config.VISION_MODEL,
            "embed_model":   config.EMBED_MODEL,
            "embed_backend": config.EMBED_BACKEND,
            "node_models":   {n: config.model_for(n) for n in nodes},
        },
        "server": {
            "host":              config.API_HOST,
            "port":              config.API_PORT,
            "cors_origins":      config.CORS_ORIGINS,
            "cors_origin_regex": config.CORS_ORIGIN_REGEX,
            "sessions_db_path":  config.SESSIONS_DB_PATH,
            "chroma_dir":        config.CHROMA_DIR,
        },
        "socratic": {
            "turn_gate":            config.SOCRATIC_TURN_GATE,
            "idk_reveal_threshold": config.IDK_REVEAL_THRESHOLD,
            "dean_max_revisions":   config.DEAN_MAX_REVISIONS,
            "max_response_sentences": config.MAX_RESPONSE_SENTENCES,
        },
        "retrieval": {
            "top_k_retrieve":          config.TOP_K_RETRIEVE,
            "top_k_rerank":            config.TOP_K_RERANK,
            "weak_topic_logit_boost":  config.WEAK_TOPIC_LOGIT_BOOST,
            "out_of_scope_threshold":  config.OUT_OF_SCOPE_THRESHOLD,
            "crag_correct_threshold":   config.CRAG_CORRECT_THRESHOLD,
            "crag_incorrect_threshold": config.CRAG_INCORRECT_THRESHOLD,
            "crag_max_refinements":     config.CRAG_MAX_REFINEMENTS,
        },
        "eval": {
            "faithfulness_threshold":     config.FAITHFULNESS_THRESHOLD,
            "blind_test_pass_threshold":  config.BLIND_TEST_PASS_THRESHOLD,
        },
        "token_budgets": {
            "teacher":     config.TEACHER_MAX_TOKENS,
            "dean":        config.DEAN_MAX_TOKENS,
            "classifier":  config.CLASSIFIER_MAX_TOKENS,
            "manager":     config.MANAGER_MAX_TOKENS,
            "explain":     config.EXPLAIN_MAX_TOKENS,
            "hint":        config.HINT_MAX_TOKENS,
            "teach":       config.TEACH_MAX_TOKENS,
            "synthesis":   config.SYNTHESIS_MAX_TOKENS,
            "clinical":    config.CLINICAL_MAX_TOKENS,
            "rapport":     config.RAPPORT_MAX_TOKENS,
            "vlm":         config.VLM_MAX_TOKENS,
        },
        "memory": {
            "backend":       config.MEMORY_BACKEND,
            "mem0_enabled":  _mem0_enabled(),
            "mem0_top_k":    config.MEM0_TOP_K,
        },
        "secrets_present": {
            # Just whether the var is set, never the value.
            "anthropic_api_key": bool(config.ANTHROPIC_API_KEY),
            "openai_api_key":    bool(config.OPENAI_API_KEY),
            "aws_access_key_id": bool(os.getenv("AWS_ACCESS_KEY_ID")),
            "aws_profile":       bool(os.getenv("AWS_PROFILE")),
            "mem0_api_key":      bool(config.MEM0_API_KEY),
        },
    }


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
    creates the per-thread state on first /chat invocation.

    NOTE: deliberately does NOT create a `session_meta` row here. The chat
    only becomes "real" — and shows up in the recent-chats sidebar — once
    the user sends their first message. /chat lazily upserts the meta row
    on the first turn, so visiting the page without typing anything no
    longer leaves an empty "New chat" stub behind.
    """
    return SessionCreateResponse(
        session_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ── Session list / metadata (sidebar) ────────────────────────────────────────

class SessionMetaPatch(BaseModel):
    title:  str  | None = None
    pinned: bool | None = None


@app.get("/sessions")
async def list_sessions() -> dict:
    """List every session this DB knows about, pinned first then most-
    recently-active. Used by the recent-chats sidebar.

    SECURITY: returns every session globally — fine for single-user demo,
    NOT fine for any deployment with multiple users. Add a user_id column
    on session_meta and filter here before public deploy.
    """
    return {"sessions": sessions_meta.list_all()}


@app.patch("/sessions/{session_id}")
async def patch_session(session_id: str, patch: SessionMetaPatch) -> dict:
    """Rename and/or pin a session. Either field is optional; sending
    `{}` is a no-op that returns the current row."""
    if "/" in session_id or ".." in session_id:
        raise HTTPException(status_code=400, detail="invalid session_id")
    row = sessions_meta.update(
        session_id, title=patch.title, pinned=patch.pinned,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="session not found")
    return row


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Hard-delete a session: drops the metadata row AND every checkpoint
    row for this thread_id. Idempotent — deleting a non-existent session
    still returns 200."""
    if "/" in session_id or ".." in session_id:
        raise HTTPException(status_code=400, detail="invalid session_id")
    sessions_meta.delete(session_id)
    return {"session_id": session_id, "deleted": True}


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
async def get_session(session_id: str, user_id: str | None = None) -> dict:
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
    # Merge session-level + user-level weak topics so the sidebar
    # always shows the persistent set even right after "New chat".
    # Session-level wins on duplicates (newer in-session add), but
    # user-level entries that aren't in this session still show.
    session_weak = state.get("weak_topics", []) or []
    if user_id:
        try:
            user_weak = user_weak_topics.list_for_user(user_id)
            merged: list[str] = list(session_weak)
            for c in user_weak:
                if c not in merged:
                    merged.append(c)
            weak = merged
        except Exception:
            logger.exception("user_weak_topics merge failed (non-fatal)")
            weak = session_weak
    else:
        weak = session_weak
    return {
        "session_id":      session_id,
        "mode":            state.get("mode", "socratic"),
        "turn_count":      state.get("turn_count", 0),
        "current_concept": state.get("current_concept", ""),
        "weak_topics":     weak,
        "student_phase":   state.get("student_phase", "learning"),
        "concept_mastered": state.get("concept_mastered", False),
        "mastery_level":   state.get("mastery_level", ""),
        # Debug-panel fields — internal state useful for diagnosing
        # what the graph is doing turn-by-turn. Safe to expose: none
        # of these contain prompt/secret data.
        "idk_count":          state.get("idk_count", 0),
        "student_attempted":  state.get("student_attempted", False),
        "classifier_output":  state.get("classifier_output", ""),
        "crag_decision":      state.get("crag_decision", ""),
        "draft_source_node":  state.get("draft_source_node", ""),
        "topic_choice":       state.get("topic_choice", ""),
        "mastery_choice":     state.get("mastery_choice", ""),
        "dean_revisions":     state.get("dean_revisions", 0),
        "messages":        [_msg_to_dict(m) for m in state.get("messages", [])],
    }


@app.delete("/sessions/{session_id}/messages/from/{index}")
async def truncate_messages(session_id: str, index: int) -> dict:
    """Rewind a session: drop the message at `index` and everything after.
    Maps to the chat-app pattern where deleting a message un-does that
    exchange and lets the student type a different reply.

    `index` is 0-based against the message list returned by GET
    /sessions/{id}. Out-of-range index returns 200 with no-op (so the
    UI doesn't have to worry about timing races).

    Implementation uses langgraph's RemoveMessage update so the
    SqliteSaver writes a fresh checkpoint with the rest of the graph
    state (turn_count, current_concept, weak_topics) preserved — only
    the message list shrinks. Subsequent /chat calls run on the
    truncated transcript.
    """
    if "/" in session_id or ".." in session_id:
        raise HTTPException(status_code=400, detail="invalid session_id")
    if index < 0:
        raise HTTPException(status_code=400, detail="index must be >= 0")
    from graph.graph_builder import graph
    cfg = {"configurable": {"thread_id": session_id}}
    snapshot = graph.get_state(cfg)
    if snapshot is None or not snapshot.values:
        raise HTTPException(
            status_code=404,
            detail=f"No state for session_id={session_id!r}",
        )
    messages = snapshot.values.get("messages", []) or []
    if index >= len(messages):
        return {"session_id": session_id, "removed": 0,
                "remaining": len(messages)}
    removes = []
    for m in messages[index:]:
        mid = getattr(m, "id", None)
        if mid:
            removes.append(RemoveMessage(id=mid))
    if removes:
        graph.update_state(cfg, {"messages": removes})
    return {"session_id": session_id, "removed": len(removes),
            "remaining": index}


@app.get("/users/{user_id}/weak_topics")
async def get_user_weak_topics(user_id: str) -> dict:
    """User-scoped weak-topics list — independent of any single session.
    The sidebar polls this so its display is stable when the student
    switches between chats. Returns an empty list (not 404) for unknown
    users so the UI doesn't have to special-case first-visit state.
    """
    try:
        topics = user_weak_topics.list_for_user(user_id)
    except Exception:
        logger.exception("user_weak_topics.list_for_user failed")
        topics = []
    return {"user_id": user_id, "weak_topics": topics}


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
    state: dict = {
        "messages":   _to_lc_messages(new_msgs),
        "session_id": req.session_id,
        "domain":     req.domain,
        "mode":       req.mode,
    }
    # Pass user_id through state so rapport_node can fetch cross-session
    # memories. Only useful when MEMORY_BACKEND=mem0 — otherwise nodes
    # ignore the field.
    if req.user_id:
        state["user_id"] = req.user_id
        # Hydrate persistent weak topics for this user. The sidebar +
        # rapport prompt + teach_node all read state.weak_topics, so
        # surfacing the user-level list here makes them visible from
        # turn 0 of every new session — the brief's "proactively
        # revisits past mistakes" requirement, beyond a single chat.
        # We merge with whatever's already in state (set later by the
        # checkpoint via add/dedupe in nodes) so in-session additions
        # aren't lost on the next turn.
        try:
            persistent = user_weak_topics.list_for_user(req.user_id)
            if persistent:
                state["weak_topics"] = persistent
        except Exception:
            logger.exception("user_weak_topics.list_for_user failed (non-fatal)")
    # Image-upload turn: route_after_input checks image_pending and
    # dispatches to vlm_node. The flag must be reset by vlm_node itself
    # so subsequent turns don't accidentally re-route through vision.
    if req.image_b64:
        state["image_pending"] = True
        state["image_b64"] = req.image_b64
    return state


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


def _maybe_add_memory(req: ChatRequest, final_state: dict, tutor_text: str) -> None:
    """Fire-and-forget: send this turn to the cross-session memory layer.

    No-op when MEMORY_BACKEND=sqlite (Mem0 client is the _NoopClient stub
    in that case, so this still runs cleanly without a feature-flag check
    here). When enabled, runs on a thread so the chat response doesn't
    wait on the mem0 cloud API.

    Metadata captured:
      - session_id  (lets users delete a single session's facts later)
      - concept     (anchors the memory to a specific anatomy/physics topic)
      - mastery_level / mastery / classifier_output
        (lets future sessions surface "you struggled with X")
    """
    if not req.user_id:
        return  # mem0 needs a stable user id; without it, skip silently.
    try:
        from memory.mem0_client import client as mem0_client
        if not mem0_client.enabled:
            return
        # Build the message pair for mem0 to extract from.
        last_user = next(
            (m for m in reversed(req.messages) if m.role == "user"), None,
        )
        if not last_user or not tutor_text:
            return
        msgs = [
            {"role": "user",      "content": last_user.content},
            {"role": "assistant", "content": tutor_text},
        ]
        meta = {
            "session_id":        req.session_id,
            "domain":            final_state.get("domain", req.domain),
            "current_concept":   final_state.get("current_concept", ""),
            "classifier_output": final_state.get("classifier_output", ""),
            "mastery_level":     final_state.get("mastery_level", ""),
        }
        # Strip empty values so the metadata stays clean.
        meta = {k: v for k, v in meta.items() if v}
        import asyncio
        async def _bg():
            await asyncio.to_thread(mem0_client.add, msgs, req.user_id, meta)
        # Fire-and-forget on the running loop. Errors are caught inside
        # the client wrapper.
        asyncio.create_task(_bg())
    except Exception:
        logger.exception("[mem0] post-turn add failed (non-fatal)")


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

    Streams in three layers:

    1. `step` frames (start/done per traced node) drive the typing-bubble
       status label so the user sees "Reading your question…" / "Pulling
       textbook context…" / "Writing response…" as those nodes run.
    2. `token` frames carry teacher_socratic delta chunks live to the
       browser via graph._stream — the bubble fills incrementally.
    3. A final unconditional `replace` frame followed by `done` reconciles
       the visible text with whatever the graph actually delivered (covers
       Dean revisions, deterministic strips, and fallback_scaffold paths
       where the streamed tokens differ from the final AI message).

    The legacy single-`response` frame is intentionally dropped; clients
    should switch to the new event types declared in
    frontend/lib/api-types.ts (which still falls through to a `response`
    handler for older backends).
    """
    import asyncio
    from graph import _stream
    from graph.graph_builder import graph

    thread_cfg = {"configurable": {"thread_id": req.session_id}}
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = {"event": "_done"}

    def _run_graph() -> dict:
        token = _stream.set_sink(queue, loop)
        try:
            return graph.invoke(_initial_state(req), thread_cfg)
        finally:
            _stream.drain(token)
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    async def event_stream():
        producer = asyncio.create_task(asyncio.to_thread(_run_graph))
        try:
            while True:
                ev = await queue.get()
                if ev is sentinel:
                    break
                yield f"data: {json.dumps(ev)}\n\n"

            try:
                result = await producer
            except Exception as exc:
                logger.exception("chat graph error")
                yield f"data: {json.dumps({'event': 'error', 'error': str(exc)})}\n\n"
                return

            text = _last_ai_text(result)
            # Update sidebar recency so this chat moves to the top of the
            # recent-chats list. Best-effort — never let a meta failure
            # break the chat response.
            try:
                # Lazy-create the meta row on first turn; subsequent turns
                # just bump last_active. `upsert(title=None)` is a no-op on
                # an existing row's title (COALESCE preserves it), so the
                # frontend's PATCH /sessions/{id} after turn 1 still wins.
                sessions_meta.upsert(req.session_id, title=None)
            except Exception:
                logger.exception("sessions_meta.upsert failed (non-fatal)")
            # Cross-session memory layer (no-op when MEMORY_BACKEND=sqlite).
            # Fire-and-forget on a thread so the chat response isn't held
            # waiting on the mem0 cloud.
            _maybe_add_memory(req, result, text)
            # Final reconciliation: idempotent — if streamed tokens already
            # match `text` the frontend just sets the same string. Required
            # for Dean revisions / deterministic strips / fallback_scaffold.
            yield f"data: {json.dumps({'event': 'replace', 'response': text})}\n\n"
            yield (
                f"data: {json.dumps({'event': 'done', 'turn_count': result.get('turn_count', 0)})}\n\n"
            )
        except Exception as exc:
            logger.exception("chat error")
            yield f"data: {json.dumps({'event': 'error', 'error': str(exc)})}\n\n"
            producer.cancel()

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
            try:
                # Lazy-create the meta row on first turn; subsequent turns
                # just bump last_active. `upsert(title=None)` is a no-op on
                # an existing row's title (COALESCE preserves it), so the
                # frontend's PATCH /sessions/{id} after turn 1 still wins.
                sessions_meta.upsert(req.session_id, title=None)
            except Exception:
                logger.exception("sessions_meta.upsert failed (non-fatal)")
            text = _last_ai_text(result)
            _maybe_add_memory(req, result, text)
            if text:
                yield f"data: {json.dumps({'event': 'response', 'response': text})}\n\n"
            yield f"data: {json.dumps({'event': 'done'})}\n\n"
        except Exception as exc:
            logger.exception("trace error")
            yield f"data: {json.dumps({'event': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
