# SOCRATIC-OT MULTIMODAL AI TUTOR

CSE 635 (NLP and Text Mining) semester project, University at Buffalo.
Final demo: **May 6**.

## What this system does

Students ask anatomy / neuroscience questions. The tutor NEVER reveals the
answer in the first two turns — it retrieves textbook chunks and asks
leading questions ("Tutor, not Teller"). After turn 2, if the student has
genuinely attempted, the system may reveal and immediately ask a clinical
application question.

---

## ABSOLUTE RULES — never violate

1. Turn-gate logic lives in **Python edges**. Prompts may receive
   `reveal_permitted` and `turn_count` but must not compute the gate.
2. Use **LangGraph**, not LangChain agents.
3. No model names hardcoded outside `config.py`.
4. No OT-specific logic in retrieval or routing — domain is config-driven.
5. **Dean runs on every LLM-generated teacher response** before delivery.
   Static-text nodes (chitchat, fallback_scaffold) bypass Dean.
6. "I don't know" gets progressive scaffolding (intensity 1→2→3) for the
   first `IDK_REVEAL_THRESHOLD-1` IDKs. The Nth consecutive IDK reveals
   via teach_node — bounded escape valve. Any non-IDK resets the counter.
7. Every node has one job. No node does retrieval AND generation AND eval.
8. All prompts live in `backend/prompts/*.txt`, never inline.

---

## Tech stack

- **Orchestration**: LangGraph + SqliteSaver checkpointer
- **LLMs**: Claude Sonnet 4.5 (Teacher, Dean, Synthesis, Study);
  Claude Haiku 4.5 (Classifier, Manager, CRAG)
- **Provider**: Anthropic API direct (default) | Bedrock fallback
  (selected via `LLM_PROVIDER`; client factory in
  `backend/graph/_llm_client.py`)
- **Vision**: Claude Sonnet 4.5 vision (native image blocks) for VLM node
- **Embeddings**: nomic-embed-text-v1.5 via transformers (Cloud Run friendly)
- **Retrieval**: `corrective_retrieve()` — synonym expand → CRAG → cross-encoder rerank
- **Vector DB**: ChromaDB single collection `{domain}_chunks`, baked into container
- **Memory**: SQLite session checkpointer + per-user `user_weak_topics` table;
  optional mem0 cross-session via `MEMORY_BACKEND=mem0`
- **API**: FastAPI + SSE streaming (token / step / replace / done envelope)
- **Frontend**: Next.js 14 + TS + Framer Motion (Vercel); legacy Streamlit kept
- **Deploy**: Backend on GCP Cloud Run, frontend on Vercel

---

## Repository layout

```
socratic-ot/
├── backend/
│   ├── api/main.py            # FastAPI + SSE
│   ├── graph/
│   │   ├── state.py           # GraphState TypedDict (central contract)
│   │   ├── edges.py           # ALL routing logic
│   │   ├── graph_builder.py   # graph assembly + checkpointer
│   │   ├── _llm_client.py     # Anthropic | Bedrock factory
│   │   ├── _stream.py         # contextvar token/step sink
│   │   └── nodes/             # one job per node
│   ├── retrieval/
│   │   ├── crag.py            # corrective_retrieve() — single entry point
│   │   ├── ot_synonyms.py
│   │   └── turn_aware.py
│   ├── ingest/                # PDF parse → late-chunk → ChromaDB
│   ├── memory/mem0_client.py  # feature-flagged cross-session memory
│   ├── prompts/*.txt          # all LLM prompts
│   ├── evaluation/            # ragas, socratic_purity, blind_test, etc.
│   ├── data/{raw,processed}/  # textbooks + chunks + chroma_db
│   └── config.py              # ALL settings — single source of truth
└── frontend/                  # Next.js 14 app router
    ├── app/{tutor,architecture,compare,dashboard}/page.tsx
    ├── components/, lib/
    └── package.json
```

---

## Where to look

| Need | File |
|---|---|
| Settings, thresholds, model IDs, paths | `backend/config.py` |
| What state flows through the graph | `backend/graph/state.py` |
| Routing decisions / edge logic | `backend/graph/edges.py` |
| LLM instructions | `backend/prompts/*.txt` |
| Per-node model overrides + env vars | `.env.example` (knob-board) |
| Retrieval pipeline | `backend/retrieval/crag.py` |

`config.model_for(node_name)` reads `<NODE>_MODEL_OVERRIDE` env vars so each
node's model is hot-swappable without code changes.

---

## Domain config

`DOMAIN` env var picks the active corpus. `DOMAIN_CONFIG[DOMAIN]` provides
prompt slots: `subject_noun`, `example_concepts`, `reject_examples`,
`rapport_examples`, `stem_blacklist` (used by leak detector).

Currently shipped: `OT_anatomy` (OpenStax AP2e), `physics` (OpenStax University Physics).

---

## Known data gap

OpenStax AP2e has thin peripheral-nerve clinical content. Demo questions are
scoped to well-covered chapters (Ch 9–16: CNS, spinal cord, sensory/motor
pathways, joints, muscles). Avoid peripheral-nerve injury questions until a
supplementary clinical reference is added to `data/raw/textbooks/` and
ingest is re-run.

---

## CURRENT PROJECT STATUS

**Phase**: Phase 5 — full brief satisfied (5 tasks + bonuses); production
frontend live; mem0 cross-session memory verified end-to-end.

**Last completed**: Choice-buttons UX polish — A/B/C menu rendered as
clickable pills, prose stripped of menu lines so options aren't duplicated;
inline markdown (`**bold**` + `*italic*`) renders cleanly (2026-04-30).

**Eval results on file** (`backend/evaluation/results/`):
- Socratic Purity: 0% premature reveal (full system) vs 80% (no Dean baseline)
- Faithfulness: 0.85+ with Dean grounding check
- Classifier accuracy: 75% (Haiku)
- Multimodal blind test: 4/5 PASS
- Generalizability (physics): 10/10 E2E
- 86/86 OT regression PASS
- Teacher A/B: Sonnet (0% leak, 0.88 faithfulness, 8.6s) vs Haiku
  (0% leak, 0.81 faithfulness, 3.9s) — keep Sonnet for demo

**Next**: P1.1 — record 5 canonical demo traces for `/architecture`
replay panels, then Milestone 3 final report + screencast.

**Deferred**: `ingest/question_bank_builder.py` (nice-to-have, not on critical path).

**Blockers**: None.

---

## When asked "what next"

1. Check this status block, then `backend/config.py` and `graph/state.py`.
2. Complete one step fully before moving to the next.
3. Run tests after every node change.
4. Update this status block when a step lands.
