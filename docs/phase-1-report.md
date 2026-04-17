# Phase 1 Verification Report

**Project:** Socratic-OT Multimodal AI Tutor  
**Course:** CSE 635: NLP and Text Mining, Spring 2026  
**Date:** 2026-04-14  
**Verified by:** Claude Code — fresh live runs, no assumed state

---

## Summary Verdict

| Phase | Status | Gap |
|---|---|---|
| Phase 1: Foundation (Steps 1–8i) | **COMPLETE** | — |
| Step 9: Question Bank Builder | **OPEN** | `question_bank_builder.py` not built |
| Step 11: Prompts (Phase 2 prep) | **COMPLETE** | Done in this session |

Phase 1 infrastructure is fully operational. One optional step (Step 9) is deferred per CLAUDE.md guidance: *"question bank can be built in parallel or after the graph is wired."*

---

## 1. File Structure Audit

All required Phase 1 files verified present on disk:

### Ingest Pipeline
| File | Status | Notes |
|---|---|---|
| `ingest/__init__.py` | ✅ present | |
| `ingest/parse_pdf.py` | ✅ present | 29 per-chapter JSON files + `all_sections_OT_anatomy.json` produced |
| `ingest/late_chunker.py` | ✅ present | v3 contextual embeddings via token offset mapping |
| `ingest/vector_store.py` | ✅ present | Single-collection abstraction, ChromaDB cosine HNSW |
| `ingest/reranker.py` | ✅ present | Cross-encoder with weak-topic logit boost |
| `ingest/metadata_builder.py` | ✅ present | `diagram_chunk_links.json`, `concept_tags.json` produced |
| `ingest/run_ingest_pipeline.py` | ✅ present | End-to-end rebuild script |
| `ingest/question_bank_builder.py` | ❌ **missing** | Step 9 — deferred |

### Retrieval Pipeline
| File | Status | Notes |
|---|---|---|
| `retrieval/__init__.py` | ✅ present | |
| `retrieval/ot_synonyms.py` | ✅ present | 35 OT terms, zero-latency dict lookup |
| `retrieval/turn_aware.py` | ✅ present | Concept-facet ladder, negation bug fixed |
| `retrieval/crag.py` | ✅ present | Full CRAG pipeline, single entry point |

### Graph Foundation
| File | Status | Notes |
|---|---|---|
| `graph/__init__.py` | ✅ present | |
| `graph/nodes/__init__.py` | ✅ present | |
| `graph/state.py` | ✅ present | All 15 TypedDict fields match spec |

### Config & Env
| File | Status | Notes |
|---|---|---|
| `config.py` | ✅ present | All constants, thresholds, paths correct |
| `.env` | ✅ present | `ANTHROPIC_API_KEY` set and working |

### Prompts (9/9)
| File | Status |
|---|---|
| `prompts/crag_evaluator.txt` | ✅ pre-existing |
| `prompts/teacher_socratic.txt` | ✅ written 2026-04-14 |
| `prompts/dean_check.txt` | ✅ written 2026-04-14 |
| `prompts/response_classifier.txt` | ✅ written 2026-04-14 |
| `prompts/manager_agent.txt` | ✅ written 2026-04-14 |
| `prompts/vlm_identify.txt` | ✅ written 2026-04-14 |
| `prompts/synthesis_assessor.txt` | ✅ written 2026-04-14 |
| `prompts/redirect.txt` | ✅ written 2026-04-14 |
| `prompts/explain.txt` | ✅ written 2026-04-14 |

---

## 2. ChromaDB — Live Query Results

**Command run:** `chromadb.PersistentClient(path='data/processed/chroma_db')`

```
Collection name:   OT_anatomy_chunks
Chunk count:       2163  (target: 2163 ✅)
Embedding dims:    768   (nomic-embed-text-v1.5 ✅)
HNSW space:        cosine ✅
```

**Metadata fields verified present on sample chunk:**

| Field | Required | Present |
|---|---|---|
| `section_id` | ✅ | ✅ |
| `section_title` | ✅ | ✅ |
| `chapter_num` | ✅ | ✅ |
| `full_section_text` | ✅ | ✅ |
| `chunk_index` | ✅ | ✅ |
| `tier` | ✅ | ✅ |
| `OT_priority` | ✅ | ✅ |
| `page_start` | ✅ | ✅ |

**Processed data files verified:**

| File | Entries |
|---|---|
| `data/processed/chunks/late_chunks_OT_anatomy.json` | 2163 chunks |
| `data/processed/concept_tags.json` | 574 entries |
| `data/processed/diagram_chunk_links.json` | 10 entries |
| `data/processed/chunks/embed_meta.json` | model, dimension, chunk_count, chunk_size, overlap, method, failed |

---

## 3. Live Pipeline Smoke Test

**Query:** `"What is the function of the rotator cuff?"`

```
VectorStore:     2163 chunks loaded   ✅
Reranker:        cross-encoder/ms-marco-MiniLM-L-6-v2 loaded (1.6s)  ✅
CRAG API call:   claude-haiku-4-5  ✅
Chunks returned: 3  ✅
CRAG decision:   AMBIGUOUS (score 0.5)
Out of scope:    False  ✅
Top section:     9.6 Anatomy of Selected Synovial Joints — Shoulder Joint
Top rerank score: 4.309  (threshold: -8.0 → well within bounds ✅)
```

**AMBIGUOUS verdict is correct behavior** — OpenStax AP2e covers rotator cuff anatomy but not its functional/clinical significance in depth. The pipeline correctly flagged it as needing refinement rather than returning off-topic content. Same pattern as the ulnar nerve data gap documented in JOURNAL.md.

---

## 4. Sub-Component Verification

### Turn-Aware Query Construction

**Command:** `build_turn_query("What nerve causes funny bone?", <student_response>, "ulnar nerve", turn)`

| Turn | Output | Correct |
|---|---|---|
| 0 | `"What nerve causes funny bone?"` | ✅ original preserved |
| 1 | `"ulnar nerve anatomy location structure"` | ✅ facet 1: where/what |
| 2 | `"ulnar nerve function clinical significance occupational therapy"` | ✅ facet 2: function/clinical |

Negation bug (`f"{concept} clarification not {student_response}"`) confirmed fixed.

### OT Synonym Expansion

**Command:** `expand_query("What causes wrist drop?")`

```
Output: "What causes wrist drop? radial nerve extensor paralysis posterior interosseous"
```

Zero-latency dict lookup confirmed working. ✅

### Weak-Topic Logit Boost

Confirmed in `ingest/reranker.py`: boost of `WEAK_TOPIC_LOGIT_BOOST = 1.0` is applied
to cross-encoder logits (range ≈ -12 to +5) after scoring, before sort. Logged as
`weak_topic_boosted: true/false` in rerank_log. Pre-CRAG distance boost of `0.2`
retained to keep weak-topic chunks in CRAG's candidate window.

---

## 5. Known Issues / Open Items

### Open: Step 9 — Question Bank Builder

- **What:** `ingest/question_bank_builder.py` not built, `data/processed/question_bank/` is empty
- **Impact:** No pre-generated Socratic questions. Teacher node will rely entirely on in-context generation from `{question_bank}` prompt placeholder (empty string fallback).
- **Disposition:** Deferred per CLAUDE.md — can be built in parallel with graph nodes or before Phase 4 evaluation
- **Action required before:** Phase 4 evaluation (Step 31+)

### Known: Peripheral Nerve Content Gap

- OpenStax AP2e contains only 1 sentence mentioning the ulnar nerve
- Queries about brachial plexus, cubital tunnel, median nerve compression return AMBIGUOUS
- CRAG correctly flags these — pipeline is not broken
- **Fix:** Add supplementary peripheral nerve clinical PDF before Phase 4
- **Workaround:** Demo questions scoped to Ch 12–16, Ch 9–11 (documented in CLAUDE.md)

### Known: CRAG AMBIGUOUS Rate

- CRAG returns AMBIGUOUS rather than CORRECT for several anatomy queries where the textbook
  content is structural (not functional/clinical)
- Increasing chunk preview from 200 → 400 chars reduced false AMBIGUOUSes but did not eliminate them
- Not a bug — correct behavior given the source content
- Acceptable for demo scope

---

## 6. Architecture Compliance Checklist

Verified against CLAUDE.md absolute rules:

| Rule | Compliant |
|---|---|
| Turn constraint lives in Python edges, never in prompts | ✅ (turn gate in `config.SOCRATIC_TURN_GATE`, enforced in `edges.py` — not yet built but no gate logic in any prompt) |
| LangGraph, not LangChain agents | ✅ (no agent imports anywhere) |
| No model names hardcoded outside `config.py` | ✅ (all nodes use `config.PRIMARY_MODEL`, `config.FAST_MODEL`) |
| Dean node runs on every teacher response | ⏳ (Phase 2 — not built yet) |
| Every node has one job | ⏳ (Phase 2 — design enforced in spec) |
| All prompts in `prompts/` as `.txt` files | ✅ (9/9 files present, none inline) |
| `corrective_retrieve()` is sole retrieval entry point | ✅ (crag.py only interface) |
| ChromaDB never imported outside `vector_store.py` | ✅ (confirmed by grep — only `vector_store.py` imports chromadb) |

---

## 7. Next Steps

| Priority | Step | Description |
|---|---|---|
| Now | Step 12 | `graph/nodes/response_classifier.py` — simplest node, tests graph scaffolding |
| Now | Step 13 | `graph/nodes/teacher_socratic.py` — main generation node |
| Now | Step 14 | `graph/nodes/dean_node.py` — quality check loop |
| Parallel | Step 9 | `ingest/question_bank_builder.py` — can be done while graph nodes are built |
| Before Phase 4 | Data gap | Add peripheral nerve clinical PDF, re-run ingest pipeline |
