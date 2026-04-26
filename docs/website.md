# Phase 5 — Production Website Build Spec

**Owner:** Deepa Nathan Rajendiran (deepanathanrajendiran@gmail.com)
**Created:** 2026-04-25
**Demo deadline:** 2026-05-06 (screen-recorded)
**Status:** Spec locked, implementation pending

This document is the canonical reference for the production website rewrite.
It supersedes any Streamlit references in CLAUDE.md once Phase 5 ships.

---

## 1. Overview

The Socratic-OT system currently runs as a Streamlit app (`frontend/app.py`).
For the May 6 final demo we are rebuilding the user-facing surface as two
related properties:

1. **Tutor site** — Socratic mode (existing graph) + Study mode (new node).
   Per-session mode toggle. Weak-topics dashboard.
2. **Architecture visualizer** — pipeline observability dashboard. Five
   panels show input/output for every stage as a query flows through the
   system. Hybrid live + replay (canonical pre-recorded traces).
3. **Compare page** — sequential reveal of Socratic vs Study response to
   the same question (NOT split-pane).

All three live in one Next.js 14 app. Backend is a FastAPI server in front
of the existing LangGraph graph. Deployed on GCP Cloud Run + Vercel.

---

## 2. Locked decisions

| Decision | Choice |
|---|---|
| Frontend stack | Next.js 14 + TypeScript + Framer Motion |
| Frontend hosting | Vercel |
| Backend stack | FastAPI + existing LangGraph |
| Backend hosting | GCP Cloud Run |
| LLM provider | Anthropic API direct (Bedrock kept as fallback in code) |
| Vector DB | ChromaDB baked into container image |
| Query-time embedder | `nomic-ai/nomic-embed-text-v1.5` via transformers (no ollama in cloud) |
| Mode dispatch | `state["mode"] = "socratic" \| "study"`; per-session, no history shared |
| Study mode retrieval | Same as Socratic — corrective_retrieve() |
| Study mode Dean | None (or grounding-only lite Dean) |
| Weak-topic tracking (Socratic) | Existing — step_advancer + synthesis_assessor |
| Weak-topic tracking (Study) | Implicit topic-similarity counter, ≥5 same-topic Qs → weak |
| Compare layout | Sequential reveal, not split-pane |
| Visualizer mode | Hybrid: live by default, 5 canonical pre-recorded traces as fallback |
| Visualizer panels | 5 (concept extract, retrieval, CRAG, reranker, generation+Dean) |
| Auth | None for demo; session header `X-Session-Id` |
| Repo layout | Monorepo: `backend/` + `frontend/` |
| Streaming | Server-Sent Events (SSE) — token stream for chat, event stream for traces |

---

## 3. System architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Vercel  (Next.js 14 + TS + Framer Motion)                      │
│  /tutor              chat, mode toggle, weak-topics sidebar     │
│  /tutor/study        same shell, study-mode prompt              │
│  /architecture       5-panel pipeline visualizer                │
│  /compare            sequential Socratic-then-Study reveal      │
│  /dashboard          weak topics + session history              │
└──────────────────────────┬──────────────────────────────────────┘
                           │  HTTPS, SSE
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  GCP Cloud Run  (FastAPI + LangGraph)                           │
│  POST /chat              SSE — token stream                     │
│  POST /chat/trace        SSE — pipeline-step events             │
│  GET  /demo/traces       canonical pre-recorded traces          │
│  GET  /sessions/{id}     session state + weak topics            │
│  POST /sessions          create new session                     │
│                                                                 │
│  graph/_llm_client.py  ──► Anthropic API direct                 │
│  retrieval/crag.py     ──► ChromaDB (baked in container)        │
│  graph/nodes/*         ──► all existing + new study_node        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
                   Anthropic API (Sonnet 4.5 + Haiku 4.5)
                   Models: claude-sonnet-4-5, claude-haiku-4-5
```

---

## 4. Backend changes (`backend/`)

### 4.1 Monorepo migration
- Move project root → `backend/`. Keep all existing tree intact under it.
- Init `frontend/` at repo root for Next.js.
- Update `.gitignore` for `frontend/node_modules`, `frontend/.next`.
- Update `requirements.txt` paths.

### 4.2 New module: `backend/graph/nodes/study_node.py`
- Same retrieval input as Socratic (`corrective_retrieve`).
- Different prompt (`prompts/study.txt`): direct explanation, grounded in chunks.
- Returns JSON envelope (see §6).
- No Dean (study mode bypasses the gate; the prompt's grounding constraint
  + retrieval is the safety story).

### 4.3 New module: `backend/api/main.py`
FastAPI app exposing the graph.
- CORS allowlist: `https://*.vercel.app`, `https://socratic-ot.vercel.app`,
  and `http://localhost:3000` for dev.
- Routes mounted from `backend/api/routes/`.

### 4.4 New module: `backend/api/routes/chat.py`
See §5 for full contracts.

### 4.5 New module: `backend/api/routes/trace.py`
See §5 — emits step-by-step events for the architecture visualizer.

### 4.6 Embedder swap
- Add `backend/retrieval/embedder.py` with a single `embed_query(text: str)`
  function that uses `nomic-ai/nomic-embed-text-v1.5` via `transformers`.
- Update `backend/retrieval/crag.py:_embed_query` to use this instead of
  ollama. Keep ollama as a configurable fallback (`EMBED_BACKEND` env var:
  `transformers` | `ollama`).
- Bake the model weights into the container image (saves ~10s cold start).

### 4.7 Checkpointer
- Add `SqliteSaver` to `graph_builder.compile()`.
- Sessions persisted to `/data/sessions.db` (mounted volume on Cloud Run).
- This unblocks A3 from the code review.

### 4.8 Mode dispatch
- Extend `GraphState`: `mode: str  # "socratic" | "study"`.
- Mode dispatch fires AFTER retrieval (not at `route_after_input`). Both modes
  need concept extraction + retrieval first; only the post-retrieval node
  diverges. New edge `route_after_retrieval`:
  - `mode == "study"`    → `study_node` → END
  - `mode == "socratic"` (default) → `response_classifier` → existing flow
- `route_after_input` is unchanged (still phase-gated on student_phase + image).

### 4.9 Implicit weak-topic counter (Study mode)
- New state fields:
  - `study_active_topic: str` — current sticky topic
  - `study_topic_count: int` — consecutive related-question count
- After each `study_node` call, the node updates state from the JSON envelope:
  - `active_topic == prior study_active_topic` → `study_topic_count += 1`
  - else → `study_active_topic = active_topic`, `study_topic_count = 1`
  - `study_topic_count >= STUDY_WEAK_TOPIC_THRESHOLD` (5) → append
    `study_active_topic` to `weak_topics` (idempotent).
- Continuation is computed in Python from `active_topic == prior`, NOT trusted
  from the LLM's `is_continuation` flag (LLMs can lie about it).
- All state updates happen inside the node — no separate similarity LLM call.

---

## 5. API contracts

### 5.1 `POST /sessions` — create session
```json
Request:  {}
Response: {"session_id": "uuid-v4", "created_at": "2026-04-25T12:00:00Z"}
```

### 5.2 `GET /sessions/{session_id}` — fetch state
```json
Response: {
  "session_id": "uuid-v4",
  "mode": "socratic",
  "turn_count": 3,
  "current_concept": "ulnar nerve",
  "weak_topics": ["brachial plexus", "rotator cuff"],
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### 5.3 `POST /chat` — stream tutor response (SSE)
```json
Request: {
  "messages": [                                // full chat thread the
    {"role": "user", "content": "..."},        // frontend has on hand;
    {"role": "assistant", "content": "..."}    // graph state's checkpointer
  ],                                           // is the source of truth
  "session_id": "uuid-v4",                     // for sticky state, but the
  "mode": "socratic",                          // request carries messages
  "domain": "OT_anatomy"                       // for cold-start and audit.
}
```
Response: `text/event-stream`. Implementation note: token-level streaming
is deferred (sync SqliteSaver doesn't support `astream_events`); the response
arrives as one event for now. Event shape:
```
data: {"response": "<full assistant text>"}

data: {"done": true, "turn_count": 4}
```
Frontend should treat the `response` event as the full answer (no
incremental concatenation needed). Per-token streaming is a Phase 5
follow-up once AsyncSqliteSaver wiring lands.

### 5.4 `POST /chat/trace` — same as /chat but with step events
```json
Request: same as /chat
```
SSE events emitted **before** the token stream starts:
```
event: trace
data: {
  "step": "concept_extraction",
  "input": {"student_message": "...", "history": [...]},
  "output": {"current_concept": "ulnar nerve"},
  "duration_ms": 412,
  "model": "claude-haiku-4-5"
}

event: trace
data: {
  "step": "retrieval",
  "input": {"query": "...", "expanded_query": "...", "weak_topics": []},
  "output": {
    "candidates_count": 15,
    "top_3": [
      {"section_title": "...", "score": 4.21, "text": "...", "boosted": false},
      ...
    ]
  },
  "duration_ms": 178
}

event: trace
data: {
  "step": "crag",
  "input": {"query": "...", "chunks_preview": [...]},
  "output": {"score": 0.82, "decision": "CORRECT", "refined": false, "parse_failed": false},
  "duration_ms": 1024,
  "prompt": "..."  // collapsible in UI
}

event: trace
data: {
  "step": "reranker",
  "input": {"candidates": 15, "query": "..."},
  "output": {
    "top_3": [{"section_title": "...", "logit": 4.21, "weak_topic_boosted": false}, ...]
  },
  "duration_ms": 350
}

event: trace
data: {
  "step": "generation",
  "input": {"node": "teacher_socratic", "reveal_permitted": false, "turn_count": 1},
  "output": {"draft": "...", "dean_passed": true, "dean_revisions": 0},
  "duration_ms": 2103,
  "prompt": "..."
}
```
After all `trace` events, the normal `token` stream begins.

### 5.5 `GET /demo/traces` — canonical traces
```json
Response: {
  "traces": [
    {"id": "funny_bone", "label": "Funny bone — clean Socratic flow", "events": [...]},
    {"id": "idk_x3", "label": "Three idks → reveal via teach_node", "events": [...]},
    {"id": "off_topic", "label": "Out-of-scope (CRAG INCORRECT) → redirect", "events": [...]},
    {"id": "ambiguous_refine", "label": "CRAG AMBIGUOUS → refinement", "events": [...]},
    {"id": "clinical_synth", "label": "Clinical synthesis + Dean grounding", "events": [...]}
  ]
}
```
The `events` array is the recorded output of `/chat/trace` for that question.

---

## 6. Study mode JSON envelope

Study mode's `study_node` calls Sonnet 4.5 with a prompt that demands JSON
output instead of free text. The envelope:

```json
{
  "answer": "The ulnar nerve runs posterior to the medial epicondyle...",
  "active_topic": "ulnar nerve",
  "is_continuation": true,
  "citations": ["chunk_id_1", "chunk_id_2"]
}
```

- `answer` — markdown-formatted explanation (rendered in the chat bubble).
- `active_topic` — the canonical topic this question is about (1-3 words).
  Used for the continuity counter — Python compares it to prior state.
- `is_continuation` — advisory; the LLM's self-judgment. Python recomputes
  this from `active_topic == prior_active_topic` and ignores the LLM's flag
  (LLMs are unreliable about this).
- `citations` — optional list of chunk IDs the answer drew from. Logged for
  audit; not surfaced in UI v1.

The backend strips the JSON envelope and surfaces `answer` to the client.
Metadata (`active_topic`, count, weak-topic promotion) updates session
state and is included in the final `done` event.

**Note on `confidence`:** earlier drafts of this spec included a self-reported
`confidence` float. Dropped because LLM self-assessed confidence is poorly
calibrated and adds noise without informing any decision.

See `prompts/study.txt` for the actual template (rendered with `{question}`,
`{retrieved_chunks}`, `{domain_context}`, `{prior_active_topic}` placeholders).
The contract is: output ONLY a JSON object with keys `answer`, `active_topic`,
`is_continuation`, `citations`.

---

## 7. Frontend pages

### 7.1 `/tutor` (and `/tutor/study`)
**Goal:** Primary chat surface.

**Layout:**
- Top bar: app title, mode toggle (segmented control: `Socratic | Study`),
  session-id chip (copyable for debugging).
- Main: chat thread (user right, assistant left), message bubbles with
  Framer Motion fade-in.
- Bottom: input box + send button. Optional image upload (defer to v2).
- Right sidebar (collapsible): weak topics list, concept-of-the-turn,
  mastery level, turn count.

**State:**
- `useChatStream` hook handles SSE consumption, token aggregation,
  optimistic user message appending, scroll-to-bottom.
- Session ID stored in `localStorage` (created on first visit).

**Mode toggle:**
- Switching mode mid-session creates a NEW session (no history shared).
- Confirmation modal: "Switch to Study mode? This starts a fresh conversation."

### 7.2 `/architecture`
**Goal:** Pipeline observability.

**Layout:**
- Top: query input + "Run live" / "Replay canonical" toggle.
- Canonical mode: dropdown of 5 pre-recorded traces.
- Main: 5 stacked panels with Framer Motion accordion expansion. Click "Next"
  to advance, or click any panel header to jump.

**5 panels:**
1. **Concept Extraction** (manager_agent)
2. **Retrieval** (corrective_retrieve up to vector search)
3. **CRAG Evaluation** (LLM judge → CORRECT/AMBIGUOUS/INCORRECT)
4. **Cross-encoder Rerank** (top-15 → top-3 with logit scores + weak-topic boost flag)
5. **Generation + Dean** (selected node → draft → Dean verdict → revision if any)

**Per-panel components:**
- Header: step name, duration, model used, success/fail indicator
- Input section (collapsible)
- Output section (always visible)
- "Show LLM prompt" button (collapsible) — for transparency
- Sub-event timeline if there are sub-steps (e.g., CRAG refinement)

**Final card after panel 5:**
- Delivered AI message (the actual output the user would see)

### 7.3 `/compare`
**Goal:** Show that the Socratic system produces a meaningfully different
experience than a generic study chatbot.

**Layout:**
- Top: query input + "Compare" button.
- Below input: two stacked cards.
  - **Card 1 (Socratic mode):** appears first with Framer Motion slide-up.
  - **Card 2 (Study mode):** appears 800ms later, slide-up.
- Each card shows: full assistant response, mode label, turn metadata.
- Below cards: "Try another" button to reset.

**Sequential, not split-pane.** This was an explicit design choice — the
goal is presentation clarity, not direct A/B comparison.

### 7.4 `/dashboard`
**Goal:** Session history + weak-topic accumulation.

**Layout:**
- Header: session ID, started_at, total turns, mode breakdown (e.g.,
  "12 Socratic / 3 Study").
- Weak topics list with click-to-resume (loads `/tutor` with that concept).
- Conversation log (expand to see full message thread).

**Defer if time-limited.** Not blocking for the demo.

---

## 8. Architecture visualizer panel detail

Each panel implements a common component:

```tsx
<TracePanel
  step="retrieval"
  title="2. Retrieval"
  duration_ms={178}
  status="success"
  input={...}
  output={...}
  prompt={null}
  active={panelIndex === 1}
  onAdvance={() => setPanelIndex(2)}
/>
```

### Panel 1 — Concept Extraction
- Input: student message + last 4 turns of history
- Output: `current_concept` (or empty if chitchat)
- Prompt: `prompts/manager_agent.txt` filled
- Visual: animated arrow from input box → LLM → extracted concept chip

### Panel 2 — Retrieval
- Input: expanded query (after OT synonym expansion), weak_topics
- Output: top-15 candidates as a table (rank, section title, distance,
  weak-topic-boost flag)
- Visual: 15 cards fly in from left, sorted by cosine distance

### Panel 3 — CRAG Evaluation
- Input: query + top-3 chunk previews
- Output: `{score, decision, parse_failed, reason}`
- Sub-event: if `decision == "AMBIGUOUS"`, show refinement query + re-retrieval
- Visual: traffic-light badge (green CORRECT / yellow AMBIGUOUS / red INCORRECT)

### Panel 4 — Cross-encoder Rerank
- Input: 15 candidates + query
- Output: top-3 with logits, weak-topic-boost flags
- Visual: side-by-side "before" (cosine order) and "after" (rerank order)
  with arrows showing position changes

### Panel 5 — Generation + Dean
- Input: target node (teacher_socratic / hint_error / etc), reveal_permitted,
  turn_count, retrieved chunks
- Output: draft response + Dean verdict + revisions (if any)
- Sub-events: each Dean revision shown as a chevron expansion
- Visual: draft text → Dean badge (PASS / FAIL [criteria]) → final response

---

## 9. Canonical demo traces

Five pre-recorded traces to ship with the build. Capture once, replay
deterministically. Each trace is the full event stream from `/chat/trace`.

| ID | Question | What it demonstrates |
|---|---|---|
| `funny_bone` | "What nerve causes the funny bone sensation?" | Clean retrieval + Socratic hint, CRAG CORRECT, Dean PASS |
| `idk_x3` | (3-turn session) "I don't know" × 3 | idk-counter ramp 1→2→3, hint intensity 1→2→3, REVEAL via teach_node at turn 3 |
| `off_topic` | "Tell me about pancreatic enzymes" | CRAG INCORRECT → out-of-scope flag → redirect_node fires |
| `ambiguous_refine` | "What's the difference between flexor and extensor muscles at the elbow?" | CRAG AMBIGUOUS → refinement query → improved score → AMBIGUOUS→REFINED |
| `clinical_synth` | (post-mastery) "If the ulnar nerve is compressed, the patient would have difficulty with pinch grip" | synthesis_assessor 3-dim scoring, weak-topics update, AIMessage feedback |

**Capture process:**
1. Run each scenario locally with `LLM_PROVIDER=anthropic` and a stable seed.
2. Tee the SSE event stream to `backend/data/demo_traces/{id}.json`.
3. Replay endpoint `GET /demo/traces/{id}` reads from disk.
4. Frontend `/architecture?trace=funny_bone` loads it on mount.

---

## 10. Deployment

### 10.1 Backend on GCP Cloud Run

**Container:**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY backend/ /app/
RUN pip install --no-cache-dir -r requirements.txt
# Pre-download models so cold start is faster
RUN python -c "from sentence_transformers import CrossEncoder; \
               CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
RUN python -c "from transformers import AutoModel; \
               AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', \
                                          trust_remote_code=True)"
# Bake ChromaDB into image (50MB; 2246 chunks)
COPY backend/data/processed/chroma_db /app/data/processed/chroma_db
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Cloud Run service config:**
- Region: `us-central1`
- Memory: 4 GiB (models + ChromaDB)
- CPU: 2
- `min-instances=1` to skip cold start during demo (~$5/day)
- `max-instances=5`
- Concurrency: 5 (LLM calls are I/O bound; can share)
- Timeout: 300s (long for streaming)
- Secrets:
  - `ANTHROPIC_API_KEY` from Secret Manager
  - `OPENAI_API_KEY` (for VLM, if used)

**Deploy command:**
```
gcloud run deploy socratic-ot-backend \
  --source backend/ \
  --region us-central1 \
  --memory 4Gi --cpu 2 \
  --min-instances 1 --max-instances 5 \
  --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest \
  --allow-unauthenticated
```

### 10.2 Frontend on Vercel

**Environment variables:**
- `NEXT_PUBLIC_BACKEND_URL=https://socratic-ot-backend-xxx.run.app`
- `NEXT_PUBLIC_DEMO_MODE=false` (set true to default to canonical replays)

**Deploy:**
- Connect GitHub repo to Vercel.
- Project root: `frontend/`.
- Build command: `npm run build` (auto-detected).
- Output: `.next/`.

### 10.3 CORS
Cloud Run app must allow `https://*.vercel.app` and the production custom
domain (if any). Set in `backend/api/main.py` via `CORSMiddleware`.

### 10.4 Cold start mitigation
- `min-instances=1` keeps one warm container.
- Models pre-downloaded in the image (no first-request stall).
- Lazy-load reranker only on first retrieval call (already done in code).

---

## 11. Build order — day-by-day

Today is **2026-04-25**. Demo is **2026-05-06**. **11 days.**

| Day | Focus | Deliverables |
|---|---|---|
| 1 | Backend monorepo migration | `backend/` move, `frontend/` Next.js init, gitignore, requirements.txt paths verified |
| 2 | FastAPI + SSE | `api/main.py`, `api/routes/chat.py` with token streaming, `useChatStream` hook on frontend, basic chat UI |
| 3 | Study mode + weak counter + transformers embedder | `study_node.py`, `study.txt`, JSON envelope parsing, embedder swap, mode dispatch in graph |
| 4 | Architecture visualizer scaffold | `/architecture` page, 5 empty panels, SSE consumer for trace events, Framer Motion transitions |
| 5 | Visualizer panels populated | Each panel rendering input/output from real trace events; live mode working |
| 6 | Canonical traces capture + replay | Run 5 scenarios, save JSON traces, `/demo/traces` endpoint, replay loader |
| 7 | Compare page + dashboard | `/compare` sequential reveal, `/dashboard` weak topics view |
| 8 | GCP Cloud Run deploy | Dockerfile, model pre-download, ChromaDB baking, deploy, smoke-test from local browser |
| 9 | Vercel frontend deploy | Environment vars, custom domain (optional), end-to-end test from production URL |
| 10 | Polish | Loading states, error states, empty states, copy editing, mobile responsive (basic) |
| 11 | Demo recording | Run through all 3 sites, record 5-10 min screencast |

**Slack: zero.** If anything slips:
- Cut Day 7 dashboard (or stub it as a static "coming soon" page).
- Cut compare page (just toggle modes manually in the recording).
- Visualizer in replay-only mode (skip live SSE streaming for traces).

---

## 12. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Cloud Run cold start ≥ 10s | High | Demo lag | `min-instances=1`; pre-download models in image |
| Anthropic quota issue during demo | Low | Demo blocked | Bedrock fallback already wired; canonical traces don't need live API |
| ChromaDB persistence on Cloud Run | Medium | Lost sessions | Bake into image (read-only is fine); SqliteSaver on persistent volume |
| Next.js + SSE quirks | Medium | Token streaming bugs | Use `EventSource` polyfill or fetch with ReadableStream; test early on day 2 |
| LangGraph `astream_events` shape | Low | Visualizer events wrong | Tap directly in node code via callbacks instead; cleaner anyway |
| 11-day schedule slip | Medium | Demo cut features | Day 11 is recording — if features unfinished, fall back to local-only demo with the same screencast |
| TypeScript types for SSE events | Low | Frontend friction | Define shared types in `frontend/lib/api-types.ts`, mirror backend Pydantic models |
| Mobile responsive | Low | Some users see broken layout | Stretch goal; demo will be desktop screen-recorded |

---

## 13. Open questions

- [ ] Custom domain or `*.vercel.app` is fine?
- [ ] Need any analytics (PostHog, etc)?
- [ ] Image upload (vlm_node) wired up in v1, or defer to v2?
- [ ] Streaming intermediate trace events during the live chat, or only show
      them on `/architecture`? (Recommend: only `/architecture` to keep
      `/tutor` clean.)
- [ ] Replay traces — share state with frontend via `searchParams`
      (`/architecture?trace=funny_bone`) or via app state? `searchParams` is
      shareable for the demo recording.
- [ ] Color scheme / branding — should I just use Tailwind defaults, or do
      you have a palette?

---

## 14. References

- CLAUDE.md — overall project spec
- docs/milestone2_report.md — Milestone 2 submission (post-hoc audit done in
  the 2026-04-25 review marathon; numbers in §3 are stale per the review)
- docs/JOURNAL.md — architectural decision log
- docs/phase-1-report.md — Phase 1 foundation report

---

## 15. Glossary

- **SSE** — Server-Sent Events. One-way streaming over HTTP; simpler than
  WebSockets when client doesn't need to send mid-stream.
- **CRAG** — Corrective RAG; the LLM judge that scores retrieval quality
  and triggers refinement on AMBIGUOUS results.
- **Dean** — quality gate node that evaluates teacher drafts against 5
  criteria (REVEAL, DEFINITION, GROUNDING, QUESTION, SYCOPHANCY).
- **Trace event** — pipeline step output emitted by the backend during a
  query. Drives the architecture visualizer.
- **Canonical trace** — pre-recorded trace event sequence shipped with the
  app for deterministic demo replays.
