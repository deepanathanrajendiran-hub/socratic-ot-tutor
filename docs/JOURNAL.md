# Socratic-OT Tutor — Project Journal

**Course:** CSE 635: NLP and Text Mining, Spring 2026
**Instructor:** Prof. Rohini K. Srihari
**Project:** P3 — Tutoring/Learning Assistant (Rehabilitation Science)
**Team:** University at Buffalo

This journal records every architectural decision made during the project —
what we tried, why we chose it, what we rejected and why. It is a living
document updated after every significant design or implementation step.

---

## Table of Contents

1. [Project Vision](#1-project-vision)
2. [Phase 1 — Ingest Pipeline](#2-phase-1--ingest-pipeline)
3. [Retrieval Pipeline](#3-retrieval-pipeline)
4. [Design Decisions Log](#4-design-decisions-log)

---

## 1. Project Vision

### What We Are Building

A Socratic AI tutoring system for Occupational Therapy (OT) students at UB.
The system never gives direct answers in the first two turns. Instead it retrieves
relevant textbook chunks and asks leading questions that guide the student
toward the answer themselves — the **Tutor-not-Teller** philosophy.

After turn 2, if the student has made genuine attempts, the system may reveal
the answer and immediately ask the student to apply it to a clinical OT scenario.

### Why Socratic, Not Direct Q&A

Most RAG chatbots retrieve → summarize → answer. That works for lookup tasks
("what is the capital of France?") but fails for learning. A student who is
told the answer has not learned it — they have been told it. Studies in
educational psychology consistently show that retrieval practice (forcing the
student to produce the answer) leads to stronger long-term retention than
passive reading or being told.

For OT students preparing for the NBCOT exam, the ability to reason from
anatomy to clinical consequence is exactly the skill being tested. The system
must build that reasoning capacity, not bypass it.

### Core Design Constraints

| Constraint | Rationale |
|---|---|
| Turn gate lives in Python edges, never in prompts | Prompts can be jailbroken. Python logic cannot be. |
| LangGraph, not LangChain agents | LangGraph gives explicit, auditable state transitions. Agents are opaque. |
| No model names hardcoded outside `config.py` | Swapping models (e.g. haiku → sonnet) must be a one-line change. |
| Dean node runs on EVERY teacher response | Quality control cannot be optional. |
| Every node has one job | Nodes that do retrieval AND generation AND evaluation are impossible to debug. |
| All prompts in `prompts/` as `.txt` files | Prompts are hyperparameters. They must be editable without touching code. |

---

## 2. Phase 1 — Ingest Pipeline

### Status: Complete (2163 chunks in ChromaDB)

### Why Late Chunking

**Standard chunking** embeds each chunk in isolation. A chunk containing
"the ulnar nerve exits the cubital tunnel" gets an embedding dominated by
"cubital tunnel" even if the surrounding section is entirely about nerve
injury mechanisms. The chunk loses its context.

**Late chunking** runs the FULL section through `nomic-embed-text-v1.5` once,
captures token-level embeddings from `last_hidden_state`, then mean-pools
over each sentence-boundary chunk's token span. The result: every chunk's
embedding reflects where it sits within its parent section.

**Why we chose nomic-embed-text-v1.5 specifically:**
- Supports asymmetric retrieval via prefixes (`search_document:` at index time,
  `search_query:` at query time) — standard bi-encoder models do not
- 8192 token context window — fits entire OpenStax sections without truncation
- 768-dim embeddings — compact enough for local ChromaDB on a laptop

**Rejected alternative — OpenAI `text-embedding-3-small`:**
- API cost at index time (2163 chunks × re-indexing = repeated costs)
- Cannot store index-time embeddings locally and re-use them
- No asymmetric prefix support

### Why a Single ChromaDB Collection

Early design had three tiers: `{domain}_small` (60 tokens), `{domain}_medium`
(300 tokens), `{domain}_large` (800 tokens). The idea was to retrieve at medium
granularity and serve large context to the LLM.

**Problem:** Three collections meant three queries per retrieval, three rerank
passes, and complex merging logic. The metadata overhead was significant.

**v3 fix:** Single `{domain}_chunks` collection. `full_section_text` is stored
directly in chunk metadata. After reranking, the full section text is fetched
from metadata — no second query needed.

**Trade-off accepted:** Metadata storage is larger (full section text per chunk).
For local dev/demo scale (2163 chunks), this is not a problem.

---

## 3. Retrieval Pipeline

### Architecture Overview

```
Student query
     │
     ▼
build_turn_query()          ← turn_aware.py  (pure Python, no LLM)
     │
     ▼
expand_query()              ← ot_synonyms.py (pure Python, no LLM)
     │
     ▼
_embed_query()              ← ollama nomic-embed-text (local)
     │
     ▼
ChromaDB cosine search      ← top-10 chunks
     │
     ▼
boost_and_rerank_by_weak_topics()   ← session memory boost
     │
     ▼
_evaluate_retrieval()       ← claude-haiku CRAG eval
     │
   CORRECT → proceed
   AMBIGUOUS → refine query once, re-retrieve, take better result
   INCORRECT → return out_of_scope=True → redirect node
     │
     ▼
cross-encoder rerank        ← ms-marco-MiniLM-L-6-v2, top-3
     │
     ▼
confidence threshold        ← OUT_OF_SCOPE_THRESHOLD = -8.0 (logit)
     │
     ▼
fetch full_section_text     ← from chunk metadata
     │
     ▼
log to retrieval_logs.jsonl
     │
     ▼
(reranked_chunks, section_texts, crag_log)
```

### Why CRAG (Corrective RAG)

Standard RAG trusts the retriever. If ChromaDB returns off-topic chunks,
the LLM still tries to answer from them and hallucinates. CRAG adds an
evaluation step: before the Teacher LLM ever sees the chunks, a fast
model (claude-haiku) scores their relevance. Only CORRECT or refined
chunks reach the Teacher.

**Why we chose claude-haiku for CRAG evaluation (not a local model):**
- CRAG evaluation requires reading 3 chunks + the query and making a
  quality judgement — this is a reasoning task, not a classification task
- Local models (BERT-class) lack the reasoning depth for anatomy content
- claude-haiku is fast (~300ms) and cheap — acceptable latency for a
  background eval step

**Rejected alternative — cosine similarity threshold only:**
- Cosine similarity scores are not calibrated across queries
- A threshold that works for one concept fails for another
- Cannot distinguish "related but wrong" from "on-target"

### Why Cross-Encoder Reranking After CRAG

The retrieval pipeline uses two-stage ranking:
1. **Bi-encoder (nomic):** Fast ANN search, high recall, lower precision
2. **Cross-encoder (ms-marco-MiniLM-L-6-v2):** Slow but precise — reads
   query and chunk together, not independently

CRAG decides whether retrieval is good enough. Cross-encoder decides
which of the good chunks are most relevant. The two serve different purposes
and both are needed.

---

## 4. Design Decisions Log

Each entry records a specific decision: what problem it solved, what we
considered, and why we chose what we chose.

---

### [2026-04-13] Turn-Aware Query Construction — Fix

**Problem identified:**
`retrieval/turn_aware.py` used "not X" phrasing in turn 2+ retrieval queries:

```python
# Old — broken
f"{target_concept} clarification not {student_response} anatomy detail"
```

Dense vector embedding models do not process negation. "not ulnar nerve"
does not retrieve content that avoids ulnar nerve — the word "ulnar nerve"
dominates the embedding and the retrieval is identical to without the negation.
The `not` keyword carries near-zero weight in a 768-dimensional semantic space.

Turn 1 had a similar issue:
```python
# Old — also broken
f"{target_concept} anatomy difference from {student_response}"
```
"difference from" is semantically void in vector space. Worse, if
`student_response` is a full sentence, it adds noise that dilutes
the concept signal.

**Root insight:**
The retrieval query should describe the **content needed to write a good
Socratic hint**, not model the student's wrong answer. The Teacher LLM
handles the Socratic framing — the retriever just needs good textbook content.

**Options considered:**

| Approach | Description | Verdict |
|---|---|---|
| **A — Concept-facet ladder** | Map turn count to a different facet of the concept. No LLM, no student response in query. | **Chosen** |
| **B — Misconception keyword extraction** | Use claude-haiku to extract a 2-3 word misconception keyword from student response, then query `{concept} versus {keyword}`. | Rejected — adds latency + API cost per turn |
| **C — ID-based pseudo-negation** | Run a second ChromaDB query on student response text, subtract those chunk IDs from primary results. | Rejected — adds complexity, second vector search, still imperfect |

**Decision: Approach A — Concept-Facet Ladder**

**Why Approach A wins:**
- Zero latency, zero API cost
- Deterministic — same concept always retrieves in the same progression
- Architecturally honest: retrieval fetches content, Teacher LLM handles pedagogy
- Each facet retrieves a meaningfully different dimension of the concept

**New query logic:**

| Turn | Query | What it retrieves |
|---|---|---|
| 0 | `original_query` | Broad — whatever the student asked |
| 1 | `{concept} anatomy location structure` | Where it is, what it's made of — foundational |
| 2+ | `{concept} function clinical significance occupational therapy` | What it does, why it matters for OT |

**Why these two facets specifically:**
- Turn 1: student is usually wrong about *what* the structure is. Foundational
  anatomy (location, neighboring structures) gives the Teacher content to build
  a hint about identity.
- Turn 2+: student knows roughly what it is but cannot connect it to function
  or clinical consequence. The second facet fetches content about what happens
  when the structure is damaged — exactly the clinical reasoning OT students need.

**Implementation:** Single function in `retrieval/turn_aware.py`.
Function signature unchanged — all callers unaffected.

---

---

### [2026-04-14] Weak-Topic Boost — Fix: Apply at Cross-Encoder Stage

**Problem identified:**
The weak-topic personalization signal was applied before CRAG evaluation (on cosine
distance scores) but the final ranking was determined by the cross-encoder, which
had no knowledge of weak topics. Pipeline order:

```
boost_and_rerank_by_weak_topics()   ← modifies cosine distance
CRAG eval
cross-encoder rerank                ← ignores distance, rescores everything
```

A weak-topic chunk boosted to position 2 in step 1 could land at position 4 after
cross-encoder rerank and never reach the top-3. The personalization signal was
silently discarded on every query.

**Options considered:**

| Approach | Description | Verdict |
|---|---|---|
| **A — Move boost to after cross-encoder** | Re-apply distance boost on cross-encoder scores post-rerank | Rejected — boost magnitude (0.2) calibrated for cosine, not logits |
| **B — Guarantee one weak-topic slot** | Force a weak-topic chunk into position 3 if not already in top-3 | Rejected — forces irrelevant chunks (e.g. ulnar nerve chunk into rotator cuff query) |
| **C — Logit boost at cross-encoder stage** | Add `WEAK_TOPIC_LOGIT_BOOST` to cross-encoder logits before final sort | **Chosen** |

**Decision: Approach C — Logit Boost at Cross-Encoder Stage**

**Why C wins:**
- Applied at the right stage: the boost lives on the same axis as the final sort
- Respects relevance: a weak-topic chunk only rises if it's already semantically
  relevant to the query — it can't displace a clearly superior chunk
- Auditable: `weak_topic_boosted: true/false` is logged in every `rerank_log`
  entry in `retrieval_logs.jsonl`

**Why we kept the pre-CRAG distance boost:**
The pre-CRAG boost (`WEAK_TOPIC_BOOST = 0.2` on cosine distance) serves a different
purpose: it ensures weak-topic chunks reach CRAG's evaluation window (top-10 before
CRAG). Without it, CRAG might never see a weak-topic chunk at all and the logit
boost would have nothing to work with.

The two boosts serve different stages:
- Pre-CRAG: gets weak-topic chunks into the candidate set
- At rerank: ensures they survive into the final top-3

**Calibration:**
`WEAK_TOPIC_LOGIT_BOOST = 1.0` (cross-encoder logit range ≈ -12 to +5).
This is approximately equivalent in relative effect to `WEAK_TOPIC_BOOST = 0.2`
on cosine distance (which ranges 0–2 for HNSW cosine). May need empirical tuning
during evaluation phase.

**Files changed:**
- `config.py` — added `WEAK_TOPIC_LOGIT_BOOST = 1.0`
- `ingest/reranker.py` — `rerank()` and `rerank_with_logging()` now accept
  `weak_topics: list[str] | None = None`; boost applied after scoring, before sort
- `retrieval/crag.py` — `corrective_retrieve()` passes `weak_topics` to reranker

---

---

### [2026-04-14] CRAG Chunk Preview — 200 → 400 chars

**Problem:** `_evaluate_retrieval()` showed only the first 200 chars of each
chunk to claude-haiku for quality scoring. Dense anatomy text often doesn't
reveal its relevance in the first 200 chars, causing CORRECT retrievals to be
mislabelled AMBIGUOUS and triggering an unnecessary refinement round-trip.

**Fix:** Increased preview from 200 to 400 chars. One-line change in
`retrieval/crag.py → _evaluate_retrieval()`.

**Issues considered but deferred:**
- BM25/keyword fallback — OT synonym expansion covers the exact-term gap for
  demo scale; BM25 adds a separate index and score fusion complexity. Skip for now.
- Dual embed paths (VectorStore.query vs corrective_retrieve) — same model,
  identical vectors, backup path never called at runtime. Code smell only. Skip.

---

---

### [2026-04-14] Step 8i — CRAG Smoke Test: Infrastructure Pass, Data Gap Found

**Result:** Pipeline infrastructure fully working end-to-end.
- `claude-haiku-4-5` API call: ✅
- ChromaDB query: ✅
- Ollama `nomic-embed-text` embedding: ✅
- Cross-encoder reranker: ✅
- CRAG eval + audit log: ✅

**Data gap discovered — peripheral nerve clinical content missing:**

Test query: *"What nerve causes the funny bone sensation?"*
CRAG decision: AMBIGUOUS (score 0.5) — correct, the content isn't there.

Investigation showed that the ulnar nerve appears exactly **once** in the entire
OpenStax AP2e textbook:
> *"The radial nerve continues through the arm and is paralleled by the ulnar nerve
> and the median nerve."*

No medial epicondyle. No cubital tunnel. No sensory distribution. No clinical
OT context. The textbook is a general anatomy survey, not an OT clinical reference.

**Secondary embedding dilution issue (noted, not fixed):**
The one chunk containing "ulnar nerve" also mentions 7 other nerves (radial, median,
femoral, saphenous, sciatic, tibial, fibular) in the same 300 tokens. The chunk's
embedding represents "nerve plexuses in general" — the ulnar nerve signal is diluted.
This cannot be fixed without either (a) more specific source content or (b) finer
chunking, which would hurt other sections.

**Chunking verified as correct:**
Token count stats on 200-chunk sample:
  - Mean: 273 tokens (target: 300)
  - Median: 306 tokens
  - Max: 344 tokens
  - No chunks over 400 tokens
The chunker is working correctly. The 1300-char mean is expected (4.5 chars/token).

**Decision: Scope demo questions to well-covered content, flag for later fix.**
Options considered:
- Add supplementary peripheral nerve PDF → deferred to Phase 4
- Scope demo questions to Chapters 12–16 (CNS, sensory/motor, autonomic) → adopted now

Added to `CLAUDE.md` as a prominent warning block so it is not forgotten.
The supplementary content must be added before evaluation (Phase 4, Step 31+).

---

---

### [2026-04-14] Phase 2 Architecture — 3-Tier Mastery System Design

**Context:**
Phase 2 (Core Graph) was ready to start. Before building Step 12, a brainstorming
session surfaced an open design gap in CLAUDE.md: `route_after_step_advancer()`
referenced `state.get("concept_mastered")` but `concept_mastered` was missing from
GraphState entirely. The gap was also larger than a single missing field — there was
no defined policy for what "mastered" means in a Socratic system that allows 3 turns.

**Problem:**
Original CLAUDE.md had two outcomes for step_advancer:
- student correct → `concept_mastered = True` → synthesis_assessor
- else → `deliver_response`

This is too coarse. A student who guesses correctly on turn 0 (first message) gets
the same outcome as one who worked through two turns of hints. A student who cannot
answer by turn 3 gets no explicit pathway — they fall to `deliver_response` with no
teaching, no reveal, and no feedback.

**Options considered for mastery trigger:**

| Option | Rule | Problem |
|---|---|---|
| A — Any correct | `classifier == "correct"` → mastered | Scenario 5 in CLAUDE.md says correct on turn 1 → go straight to synthesis. Aligns with this. |
| B — Correct after gate | `classifier == "correct" AND turn_count >= SOCRATIC_TURN_GATE` | Contradicts Scenario 5. Correct on turn 1 would NOT go to synthesis. |
| C — 3-tier (chosen) | strong / weak / failed tiers based on when correct answer arrives | Matches pedagogical intent: reward early mastery, flag late mastery, rescue failed students |

**Decision: 3-Tier Mastery System**

| Tier | Trigger | Outcome |
|---|---|---|
| **strong** | Correct before turn 3 (`turn_count < SOCRATIC_TURN_GATE`) | → synthesis_assessor (scored) |
| **weak** | Correct at turn 3 (`turn_count >= SOCRATIC_TURN_GATE`) | → synthesis_assessor (scored) + `mastery_level="needs_review"` in SQLite |
| **failed** | Still incorrect at turn 3 | → teach_node (reveal + explain + unscored synthesis) + `mastery_level="failed"` in SQLite |

**Why the failed path goes to synthesis anyway (unscored):**
The student didn't earn the clinical application question, but forcing them through
it even after failure converts a passive reveal into an active learning moment.
They must attempt to apply what was just explained — this increases retention over
a bare "here is the answer" delivery. The score simply isn't recorded as a pass.

**Architectural changes locked in:**

1. `GraphState` — added `concept_mastered: bool` and `mastery_level: str`
2. `edges.py` — updated `route_after_classifier`: "incorrect" at `turn_count >= SOCRATIC_TURN_GATE` → `teach_node` (bypasses `hint_error_node`)
3. `edges.py` — updated `route_after_step_advancer`: reads `mastery_level` ("strong"/"weak" → synthesis, safe default otherwise)
4. `edges.py` — added `route_after_teach`: always routes to `synthesis_assessor`
5. `teach_node.py` — new node added to build order as Step 18b
6. `SQLite weak_topics` — added `mastery_level TEXT DEFAULT 'failed'` column with values `'failed'` | `'needs_review'`

**Why teach_node is a separate node (not reusing explain_node):**
`explain_node` handles mid-session clarifying questions (student asks "can you explain
X?"). `teach_node` handles end-of-session instruction after failure to answer. They
have different:
- Triggers (classifier="questioning" vs end-of-turn-3 failure)
- Tone (guided scaffold vs direct explanation + clinical application)
- DB side effects (teach_node writes to weak_topics, explain_node does not)

Reusing explain_node would violate CLAUDE.md's "every node has one job" rule.

---

---

### [2026-04-14] Step 11 — Prompt Files: Design Choices

**9 prompt files written.** This entry documents the non-obvious design choices
embedded in each prompt, since prompts are hyperparameters and their rationale
should be recorded.

**teacher_socratic.txt — critical rules:**
- `REVEAL PERMITTED` is a placeholder injected from Python (`should_reveal()`
  in edges.py). The LLM cannot calculate turn counts — that logic lives in Python.
- "End with EXACTLY ONE question mark" — the Dean checks for this. Two questions
  or a trailing period both fail the QUESTION CHECK criterion.
- `{question_bank}` placeholder defaults to "(none)" until Step 9 is built. The
  teacher ignores it gracefully — it's framed as "use one of these if it fits."
- Sycophantic opener prohibition is explicit ("Do not say 'Great question!'")
  because LLMs default to praise. The Dean's SYCOPHANCY CHECK catches violations.

**dean_check.txt — 6-criterion structure:**
The 6 criteria are ordered by severity: reveal checks first (most critical),
grounding second, structural (question, length, sycophancy) last. The JSON
output format with `failed_criteria: [list]` was chosen over a score because:
- The teacher revision prompt can reference a named criterion (e.g., "REVEAL CHECK")
- Debugging is easier — you see exactly which criterion failed
- Boolean per criterion is more interpretable than a numeric rubric

**response_classifier.txt — single-word output:**
The classifier returns exactly one word. `max_tokens=10` in the API call enforces
this. A JSON output with reasoning was considered and rejected — the added tokens
cost latency and the reasoning is not used anywhere in the graph. The single-word
output goes directly into `route_after_classifier()` as a switch statement. Speed
matters here since this runs on every student message.

**synthesis_assessor.txt — 0-2 scoring per dimension:**
Three dimensions (structure_accuracy, functional_consequence, ot_relevance)
scored 0-2, total 0-6. `passed = true if total >= 4` (66%). `weak_topic_flag`
set if `total <= 2` (33%). The thresholds were chosen to:
- Pass students who correctly nail 2 of 3 dimensions even with one weak area (4/6)
- Flag students who fail all or almost all dimensions for review (2/6)
- Not punish specialization: a student strong on OT relevance but weak on exact
  structure location should still pass

**redirect.txt and explain.txt — no sycophancy, end with "?":**
Both prompts share the same structural rules as teacher_socratic but serve
narrower roles. redirect.txt never answers the off-topic question — it bridges
back with a Socratic question. explain.txt is the ONLY place where the concept
can be described indirectly before the reveal, because the student explicitly
asked for help. Both still end with exactly one question mark.

---

---

### [2026-04-14] Step 12 — response_classifier.py: Implementation Notes

**File:** `graph/nodes/response_classifier.py`
**Model:** `FAST_MODEL` (claude-haiku-4-5), `max_tokens=10`

**Key design decisions:**

**1. `_msg_text()` helper for multimodal content:**
LangChain `BaseMessage.content` can be `str` OR `list[dict]` (multimodal messages
with text + image parts). Calling `msg.content` directly and treating it as a string
would crash on image messages. The `_msg_text()` helper extracts text parts safely:
```python
def _msg_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)
```
This future-proofs the classifier when vision messages appear in `state["messages"]`.

**2. `last_two_turns` construction:**
The prompt takes `{last_two_turns}` as context for classification. This is the
2 full exchanges (up to 4 messages) immediately before the current student message.
The current student message is extracted separately as `{student_message}`.
The split is: `prior = messages[:-1][-4:]`. This means:
- For turn 0 (only 1 message): `prior = []` → "(no prior context)"
- For turn 1+: up to the last 4 messages before the student's current input

**3. Invalid label fallback:**
If the LLM returns anything other than the 4 valid labels, the node defaults to
`"incorrect"`. This is the safest fallback: `hint_error_node` gives a hint, which
is never harmful. Defaulting to `"irrelevant"` would silently drop the student's
attempt. Defaulting to `"correct"` would advance state incorrectly.

**Test results:** 4/4 — correct, incorrect, irrelevant, questioning all classified
correctly for ulnar nerve concept with prior conversation context.

---

---

### [2026-04-14] Step 13 — teacher_socratic.py: Implementation Notes

**File:** `graph/nodes/teacher_socratic.py`
**Model:** `PRIMARY_MODEL` (claude-sonnet-4-5), `max_tokens=600`

**Key design decisions:**

**1. `reveal_permitted` computed in Python, not passed as a parameter:**
The node computes `reveal_permitted` from `state["turn_count"]` and
`state["student_attempted"]` using the same logic as `should_reveal()` in edges.py:
```python
def _should_reveal(state: GraphState) -> bool:
    return (
        state.get("turn_count", 0) >= config.SOCRATIC_TURN_GATE
        and state.get("student_attempted", False)
    )
```
This lives in Python (enforcing CLAUDE.md absolute rule #1) and is injected
into the prompt as `{reveal_permitted}` so the LLM can see the instruction.
Critically: the LLM cannot change the gate — it can only act within its value.

**2. Question bank graceful fallback:**
`_load_question_bank()` looks for `data/processed/question_bank/{concept}.json`.
If the file doesn't exist (Step 9 is deferred), it returns `"(none)"`. The
teacher prompt treats the question bank as "use one of these if it fits, adapt
if needed" — an empty bank is silently ignored. No crash, no degraded behavior.

**3. Revision path — system message, not prompt file change:**
When `dean_revision_instruction` is non-empty in state (Dean has rejected a
draft), the teacher is called again. The Dean's specific instruction is passed
as an Anthropic `system` message rather than modifying the prompt template:
```python
if system_msg:
    api_kwargs["system"] = system_msg
```
This keeps `teacher_socratic.txt` unchanged (the spec prompt file) while giving
the Teacher targeted revision guidance. The system message is:
`"REVISION REQUIRED: {instruction}\nFix the issue above and rewrite the response."`

**Why not add `{revision_instruction}` to the prompt file:**
The prompt file is a hyperparameter spec. Adding a field to it changes the
interface contract between the node and the prompt. The system message approach
lets the revision mechanism work without touching the prompt file at all.

**Test observations:**
- At turn 0, no-reveal: the teacher described "a nerve that runs in a very
  superficial groove behind a bony landmark" without naming the ulnar nerve.
  The Socratic framing was preserved while staying grounded in anatomy.
- Exactly 1 question mark in all tested outputs.

---

---

### [2026-04-14] Step 14 — dean_node.py: Implementation Notes + Bug Fix

**File:** `graph/nodes/dean_node.py`
**Model:** `PRIMARY_MODEL` (claude-sonnet-4-5), `max_tokens=300`

**Bug discovered and fixed during implementation:**

**`_fill_prompt()` — safe template substitution for prompts with JSON:**

`dean_check.txt` contains literal JSON braces in its output specification:
```
{
  "passed": true/false,
  "failed_criteria": [...],
  "revision_instruction": "..."
}
```
Python's `str.format()` interprets ALL `{...}` as format placeholders. Calling
`.format(current_concept=..., ...)` on this prompt raises:
```
KeyError: '\n  "passed"'
```
because `{"passed": true/false, ...}` is treated as a malformed placeholder.

**Fix:** `_fill_prompt()` replaces placeholders via string replacement, not
Python's format parser. It only touches the keys we explicitly pass:
```python
def _fill_prompt(template: str, **kwargs) -> str:
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result
```
This is safe for any prompt that mixes format placeholders with literal JSON.

**Note for future nodes:** `synthesis_assessor.txt` also contains JSON output
specification. `synthesis_assessor.py` (Step 25) MUST use `_fill_prompt()`,
not `.format()`. This is now documented here so it is not forgotten.

**Other implementation decisions:**

**1. Markdown code fence stripping:**
Claude occasionally wraps JSON in markdown code fences (` ```json ... ``` `).
The node strips these before JSON parsing:
```python
if raw.startswith("```"):
    lines = raw.splitlines()
    raw = "\n".join(line for line in lines if not line.startswith("```")).strip()
```
Without this, `json.loads()` fails on fenced output.

**2. Fail-open on JSON parse error:**
If JSON parsing still fails after stripping, the node returns `dean_passed=True`
rather than failing closed. Reasoning:
- Failing closed (setting `dean_passed=False`) would trigger a revision loop.
  After `DEAN_MAX_REVISIONS=2` rejections, the graph falls to `fallback_scaffold`.
  The student would never get a response.
- Failing open lets the draft through. An occasional suboptimal response reaching
  the student is less harmful than a system that silently blocks.
- A JSON parse failure is a hallucination in Dean's output, not a quality failure
  in the teacher's draft.

**3. `dean_revisions` increments only on failure:**
`dean_revisions += 1` only when `passed=False`. The count tracks how many times
the Dean has rejected and sent for revision, which is what `route_after_dean()`
needs to check against `DEAN_MAX_REVISIONS`. Passing responses don't increment.

**Grounding check behavior observed during testing:**
The Dean's GROUNDING CHECK is strict. Test drafts using general anatomical
language ("nerve compression near the elbow") were rejected because that specific
phrase wasn't in the retrieved chunks. This is correct behavior — the Dean
should not allow claims it cannot verify. It informed a test design decision:

**Test design — integration over hand-crafted drafts:**
Instead of hand-crafting a "clean" test draft (which kept failing grounding
checks due to subtle ungrounded claims), Test 1 now calls `teacher_socratic()`
to generate a real draft, then passes that to `dean_node()`. This:
- Tests the actual Teacher→Dean pipeline rather than a synthetic scenario
- Gives confidence that the system works end-to-end
- Is more robust to prompt changes than a hard-coded test string

**Test results:** 4/4
- Teacher-generated draft (turn 0) passed Dean ✅
- Dean caught explicit ulnar nerve reveal at turn 0 ✅
- `dean_revisions` incremented from 0 → 1 on failure ✅
- `revision_instruction` was specific and actionable ✅

---

---

### [2026-04-15] Steps 15–16 — hint_error_node, redirect_node

**Files:** `graph/nodes/hint_error_node.py`, `graph/nodes/redirect_node.py`
**Model:** `PRIMARY_MODEL` (claude-sonnet-4-5)

**hint_error_node (Step 15):**
Called when `classifier_output == "incorrect"` AND `turn_count < SOCRATIC_TURN_GATE`.
Passes `student_wrong_answer` (last human message) to `prompts/hint_error.txt`.
The prompt asks the LLM to acknowledge the error without saying it is wrong, then provide
a Socratic hint rooted in the retrieved chunks that nudges the student toward the correct path.

Key design point: the student's wrong answer is passed to the prompt so the hint can
directly address the specific misconception rather than give a generic hint. This is
safer than trying to do semantic analysis of the error in Python — the LLM can identify
the error dimension (anatomy, function, clinical) and hint accordingly.

**redirect_node (Step 16):**
Called when `classifier_output == "irrelevant"` (off-topic student message).
Only needs `current_concept`, `turn_count`, and the last student message.
Does NOT receive retrieved chunks — redirect responses should not ground in anatomy
content (that would implicitly answer the real question while redirecting).

The redirect prompt bridges from the off-topic message back to the current concept via
a Socratic question. This preserves the "exactly one question mark" rule and keeps
the turn gate unperturbed — turn_count does not increment on a redirect (that happens
in `deliver_response`, which runs after Dean approves).

**Both nodes:** 4/4 tests passed. Pattern identical to response_classifier (fast extract
of last student message, `_fill_prompt()`, PRIMARY_MODEL call, draft_response output).

---

---

### [2026-04-15] Steps 17–20 — Post-Mastery Choice Flow: Major Design Addition

**Context:**
Original CLAUDE.md had `step_advancer` route directly to `synthesis_assessor`. This
assumed every correct answer is immediately followed by a clinical question, with no
student agency. After reviewing the UX in the brainstorming session, this was identified
as a critical gap.

**Problem:**
Students who master a concept have three reasonable next actions, not one:
1. Try the clinical application question
2. Move to a new topic
3. End the session

Forcing route 1 on all students ignores routes 2 and 3. More critically, `synthesis_assessor`
was being called immediately after `step_advancer` — but `synthesis_assessor` requires
a clinical response from the student, which doesn't exist yet. The node was evaluating
an empty string. This is a state machine correctness bug, not a UX preference.

**Root cause of the synthesis timing bug:**
The clinical question must be ASKED (one turn) and then ANSWERED (next turn) before
`synthesis_assessor` can score. The original design called `synthesis_assessor` in the
same turn as the choice offer. This was only possible if the graph re-invoked
`synthesis_assessor` on the next human turn — but there was no mechanism to do that.

**Solution: `student_phase` state machine**

Added 4 phases to `GraphState`:
```
"learning"            → normal Socratic teaching loop
"choice_pending"      → system asked the three-way choice; waiting for student pick
"topic_choice_pending"→ system asked "weak topics or your own?"; waiting for pick
"clinical_pending"    → system asked clinical question; waiting for student answer
```

The `route_after_input()` edge function reads `student_phase` as a phase gate before
any other routing. This routes the same student message to completely different nodes
depending on what the previous turn asked for. `synthesis_assessor` is only called
when `student_phase == "clinical_pending"` — guaranteeing the student answer exists.

**New nodes built:**

| Node | Step | Purpose |
|---|---|---|
| `explain_node` | 17 | Handle mid-session clarifying questions ("can you explain X?") |
| `step_advancer` | 18 | Compute mastery tier, offer 3-way choice prompt |
| `teach_node` | 18b | Failed mastery path: reveal + explain + 3-way choice |
| `mastery_choice_classifier` | 18c | Classify student's choice as "clinical"|"next"|"done"|"other" |
| `topic_choice_node` | 18d | Ask "weak topics or your own?" when student picks "next" |
| `topic_choice_classifier` | 18e | Classify student's topic preference as "weak"|"own"|"other" |
| `clinical_question_node` | 18f | Generate OT clinical application question for current concept |

**New routing functions in edges.py:**

```python
route_after_mastery_choice(state) → "clinical_question_node" | "topic_choice_node" | END
route_after_topic_choice(state)   → "manager_agent"
```

`route_after_dean()` was fixed: the revision loop went back to `"teacher_revise"` (a
non-existent node in the old spec) — corrected to `"teacher_socratic"`.

**Why approach A (max node separation) over approach B (merged classify+route):**
- Approach B (merging mastery_choice_classifier into an edge function) would put LLM
  calls inside edge functions — violating CLAUDE.md's "every node has one job" rule
- Edge functions must be pure Python for testability and auditability
- Separate classify node means the classification result is visible in graph state,
  which enables debugging and future logging

**teach_node vs explain_node separation:**
`explain_node` handles mid-session clarifying questions (classifier="questioning").
`teach_node` handles end-of-session instruction after the student fails to answer by
turn 3. They differ in tone (scaffold vs reveal+explain), trigger
(questioning vs failed at turn gate), and DB side effects (teach_node writes
mastery_level="failed" to weak_topics). Reusing explain_node would mix two distinct
pedagogical moments in one node — violating single-responsibility.

**All nodes:** 4/4 tests each. Graph compiled and full integration test running.

---

---

### [2026-04-15] LangGraph Entry Point Bug — `set_entry_point` Conflicts with Conditional Edges

**Symptom:**
`graph.invoke()` crashed with:
```
At key 'student_phase': Can receive only one value per step
```

**Root cause:**
`graph_builder.py` had BOTH:
```python
g.set_entry_point("manager_agent")      # direct __start__ → manager_agent edge
g.add_conditional_edges("__start__", route_after_input, {...})  # conditional edge from __start__
```

LangGraph fires ALL edges from a source node in parallel when multiple edges exist.
`__start__` had two outgoing edges (direct + conditional). Both fired on every invoke.
Two nodes wrote to `student_phase` simultaneously → conflict error.

**Fix:**
Removed `g.set_entry_point("manager_agent")`. The conditional edge alone handles all
cases, including the "learning" phase that would normally go to `manager_agent`.
The conditional edge now has `"manager_agent": "manager_agent"` in its mapping.

**Why this bug wasn't caught earlier:**
The graph compiled without error — LangGraph validates graph structure at compile time
but doesn't detect multiple-source conflicts until runtime. The bug only surfaces when
`invoke()` is called.

---

---

### [2026-04-15] Steps 21, 23, 25 — Integration Test, Manager Agent, Synthesis Assessor

**Step 21 — Full loop integration test (`test_full_loop.py`):**
5 scenarios, 9 checks. Pre-seeds `retrieved_chunks` to avoid ChromaDB/ollama dependency
in most scenarios. Each scenario invokes the full compiled graph.

Two test bugs fixed:
1. Scenario 1: Was checking for OT terms in AI message content. After Dean rejects
   `clinical_question_node`'s draft (grounding too strict for synthetic chunks), the
   `fallback_scaffold` fires — no OT terms. Fix: check `student_phase == "clinical_pending"`
   instead, which the clinical_question_node sets before Dean runs.
2. Scenario 4: Was checking `topic_choice` at graph exit. `manager_agent` resets
   `topic_choice = ""` after consuming it. Fix: check `current_concept` is in the
   weak_topics list instead.

**Final result: 9/9 checks pass. Step 21 complete.**

**Step 23 — `manager_agent.py`:**
Concept extractor using FAST_MODEL (claude-haiku). Outputs JSON `{"current_concept": "..."}`.
Two paths:
- Fast path: if `topic_choice == "weak"` and `weak_topics` non-empty → picks first weak
  topic directly, no LLM call needed
- Normal path: LLM extracts anatomy concept from student message

Resets `topic_choice=""`, `dean_revisions=0`, `dean_revision_instruction=""`, and
`student_phase="learning"` on every call (these are session-scoped values that should
not carry over from the previous concept exchange).

**Step 25 — `synthesis_assessor.py`:**
Clinical application scorer. Uses `_fill_prompt()` (prompt contains JSON output spec).
Extracts last human message as `student_clinical_response`. Scores on three dimensions
(structure_accuracy, functional_consequence, ot_relevance), each 0-2, total 0-6.
Adds concept to `weak_topics` if `weak_topic_flag=True` (total ≤ 2). Resets
`student_phase="learning"`. Fail-open on JSON parse error — returns generic feedback.

---

---

### [2026-04-15] Step 30 — Streamlit Frontend (`frontend/app.py`)

**Files created:**
- `frontend/app.py` — main Streamlit UI
- `frontend/components/chat_window.py` — message renderer + input box
- `frontend/components/weak_spots_dashboard.py` — sidebar weak topics panel

**Architecture:**
The frontend calls the LangGraph graph directly (no FastAPI intermediary). For the
demo, direct graph invocation is simpler and avoids the need to have the API server
running separately.

**State bridging:**
LangGraph state is bridged to Streamlit `st.session_state`:
- `graph_messages` — list of LangChain `BaseMessage` objects passed into graph each turn
- `messages` — list of plain dicts for Streamlit chat rendering
- `student_phase`, `current_concept`, `weak_topics`, `turn_count` — carried forward from
  each graph result and passed into the next invocation

The `student_phase` is the critical field: it must be passed into the graph input state
so `route_after_input()` can gate correctly. Without it, every turn would be treated
as `"learning"` and bypass the choice/synthesis flows.

**Weak topics dashboard — "Practice weak topics" button:**
Clicking the button injects a synthetic "Let's work on my weak topics" message and
forces `student_phase = "topic_choice_pending"`. On the next graph invocation,
`route_after_input()` routes to `topic_choice_classifier`, which classifies the
message as "weak" and routes to `manager_agent` with `topic_choice="weak"`. The
manager fast-path picks the first weak topic. No prompt engineering needed — the
state machine handles it cleanly.

**Image upload:**
Handled in `chat_window.py` via `st.file_uploader`. Base64-encoded image stored in
`st.session_state.image_b64`. On submit, `image_pending=True` is set in graph input,
which causes `route_after_input()` to route to `vlm_node` (currently stub).

**Run command:** `PYTHONPATH=. streamlit run frontend/app.py`

---

### [2026-04-16] Dean Quality Gate Hardening — Concept-Leak Guards + idk Scaffold

**Problem identified:**
Demo was returning the fallback scaffold ("Let's take a step back...") on turn 0 for
most concepts. Root causes (in order of discovery):

1. `teacher_socratic` generated drafts containing concept derivatives (e.g. "synaptic"
   for concept "synapse"). Dean correctly failed REVEAL, consumed both revision slots,
   triggered fallback.
2. LENGTH check in Dean consumed one revision slot on every verbose draft, leaving only
   one slot for REVEAL — a single REVEAL failure then triggered fallback.
3. On revision calls, the concept-leak retry was overwriting the revision system message
   with the forbidden-word constraint, causing the model to ignore the LENGTH fix.
4. "I don't know" was classified as "incorrect" and routed to the same hint path as wrong
   answers — no distinction between "no attempt" and "wrong attempt".
5. `clinical_question_node` consistently failed GROUNDING because the Dean was treating
   valid anatomical inferences (FCU denervation → wrist flexion weakness) as unsupported
   claims, despite the GROUNDING criterion explicitly stating "absence ≠ contradiction".

---

**Fix 1 — Concept-leak guard in `teacher_socratic.py` and `hint_error_node.py`**

Added `_contains_concept(draft, concept)` — stem-based detection catches both the exact
concept word and morphological derivatives:
```python
stem = word[:max(4, len(word) - 2)]  # "synapse" → "synap" catches "synaptic"
if re.search(r'\b' + re.escape(stem), draft_lower):
    return True
```

Guard runs after every LLM generation, before the draft reaches the Dean:
- Up to 2 retry attempts, each with a combined system message (revision instruction +
  forbidden-word constraint — NOT overwriting the revision instruction as the first
  version did).
- Deterministic regex strip as last resort if both retries still contain the concept.
- Terminal log: `[teacher] leak_retry attempt=N | ...`

**Why combined system message matters:** On revision calls (dean_revisions > 0), the
teacher has both a LENGTH/GROUNDING fix instruction AND a concept-leak constraint. The
original guard used `system=leak_instruction` which silently dropped the revision
instruction. The fixed version uses `system=f"{revision_system}\n\n{leak_instruction}"`.

---

**Fix 2 — LENGTH check moved from Dean to Python**

The Dean had 6 criteria and `DEAN_MAX_REVISIONS = 2`. A LENGTH failure on rev=0 left only
one slot for REVEAL — a second REVEAL failure then triggered fallback. LENGTH is a
mechanical sentence count; it does not need LLM judgment.

**Decision:** Remove CRITERION 5 (LENGTH CHECK) from `dean_check.txt`. Add
`_count_preamble_sentences(draft)` to `teacher_socratic.py`. Count prose sentences before
the first `?`, retry once with a tight system message if over `MAX_RESPONSE_SENTENCES`.
Dean now has 5 criteria; both revision slots reserved for REVEAL, GROUNDING, SYCOPHANCY.

**Options considered:**

| Approach | Verdict |
|---|---|
| Keep LENGTH in Dean | Rejected — wastes revision slots on a counting task |
| Remove LENGTH entirely | Rejected — model occasionally writes 4+ sentence preambles |
| Move to Python pre-check (chosen) | Zero revision slot cost; deterministic and fast |

---

**Fix 3 — `idk` classifier category**

"I don't know" and "give me a hint" were classified as `incorrect` (no attempt made).
The `hint_error_node` would try to correct a wrong answer that didn't exist, generating
a generic hint similar to what was already shown.

**Decision:** Add `idk` as a 5th classifier output. Route identically to `incorrect`
(same turn gate, same node), but pass `student_mode="idk"` into the prompt.

`hint_error.txt` now has two modes:
- `incorrect` — acknowledge the attempt, correct the specific error, ask narrower question
- `idk` — skip error correction, paraphrase a textbook clue, ask a more concrete question
  (progressive scaffold: each idk turn reveals one additional concrete clue)

The `idk` mode adds the concept-leak guard because paraphrasing chunks risks quoting the
concept name verbatim from the retrieved text.

**PRIORITY RULE added to classifier:** "If student makes ANY answer attempt (even wrong),
use 'incorrect' not 'idk'. Only use 'idk' when there is zero attempt."

---

**Fix 4 — Dean GROUNDING criterion: clinical inference clarification**

`clinical_question_node` was generating scenarios with specific functional deficits
(weakness in wrist flexion, difficulty with DIP flexion) that the Dean classified as
unsupported by the retrieved chunks. The GROUNDING criterion already said
"Claims not mentioned in the chunks (absence ≠ contradiction) → PASS" but the LLM
was ignoring this for clinical inferences.

Added an explicit example to the GROUNDING criterion:
```
Clinical inferences derived from facts in the chunks are valid:
  If chunks say "ulnar nerve innervates FCU", then "weakness in wrist flexion"
  in a patient with ulnar nerve damage is a valid inference → PASS.
"Not explicitly stated" is NOT the same as "contradicts" → PASS.
```

Also updated `clinical_question.txt` to allow "standard anatomical consequences of the
concept" in the patient scenario setup (removing the overly strict "no outside medical
knowledge" rule that was causing the node to under-specify its scenarios).

---

**Test results:** 9/9 integration tests pass. Demo produces clean Dean logs:
```
[dean] t=0 rev=0 reveal=False PASS   ← opening question, no REVEAL or LENGTH failure
[dean] t=1 rev=0 reveal=False PASS   ← hint with idk scaffold, no leak
[dean] t=1 rev=0 reveal=True  PASS   ← step_advancer A/B/C, first attempt
```

---

---

### [2026-04-17] Corpus Augmentation — Gray's Anatomy 20e + OT Clinical Supplement

**Problem identified:**
Experiment C (faithfulness, milestone 2) scored 0.74 with Dean — well below the 0.85
target. Root cause: OpenStax AP2e has almost no peripheral nerve clinical content. The
ulnar nerve appears in exactly one sentence across 2163 chunks. Queries about cubital
tunnel, wrist drop, carpal tunnel, or brachial plexus injury retrieve structurally
unrelated content. The teacher fabricates clinical details from general medical knowledge
because the retrieved chunks contain no clinical facts to draw on.

**Two supplementary sources added:**

**Source 1 — Gray's Anatomy 20e (1918, public domain):**
Six peripheral nerve sections extracted from PDF pages 929–945 and 1329–1336:
- Brachial Plexus (PDF pp. 930–933)
- Axillary Nerve and Musculocutaneous Nerve (pp. 933–936)
- Median Nerve (pp. 937–940)
- Ulnar Nerve (pp. 940–943)
- Radial Nerve (pp. 943–945)
- Surface Anatomy Upper Extremity Nerves (pp. 1330–1336)

Extraction: `ingest/add_grays_nerves.py` — PyMuPDF, running-header strip regex,
late-chunked and upserted directly to ChromaDB. Added **68 chunks** (2163 → 2231).

Why selective extraction, not the whole book:
The cross-encoder has no diversity awareness — it returns the top-N by relevance score
with no MMR. Adding the full 1399-page Gray's would create dense similarity clusters:
most of the top-3 chunks on any peripheral nerve query would come from the same Gray's
section, crowding out OpenStax perspective. Selective extraction of only the nerve
chapters preserves source diversity within the 3-chunk retrieval window.

**Source 2 — OT Clinical Supplement (custom-written):**
`data/raw/ot_specific/peripheral_nerve_ot_supplement.txt` — ~2000 words of modern
English clinical content across 5 sections covering all three major upper limb nerves
plus brachial plexus, with OT-specific framing (ADL impacts, assessment tools,
interventions). Added **15 chunks** (2231 → 2246).

This fills the vocabulary gap Gray's 1918 creates: archaic terminology ("dorsal
interossei of the metacarpal bones") has low semantic similarity to modern OT queries
("interossei weakness"). The supplement uses the exact terminology OT students and
clinicians use.

**Files added:**
- `ingest/add_grays_nerves.py`
- `ingest/add_ot_supplement.py`
- `data/raw/ot_specific/peripheral_nerve_ot_supplement.txt`

**Outcome:** Total ChromaDB chunks: **2246** (up from 2163).

---

---

### [2026-04-17] CRAG JSON Parse Bug Fix — Haiku Markdown Fence Wrapping

**Problem identified:**
After corpus augmentation, all 8 faithfulness scenarios were returning AMBIGUOUS from
CRAG even when retrieval was clearly on-target. Investigation revealed that
`_evaluate_retrieval()` in `retrieval/crag.py` was always falling back to
`{"score": 0.5, "decision": "AMBIGUOUS"}`.

**Root cause:**
`claude-haiku-4-5` wraps all JSON responses in markdown code fences:
```
```json
{"score": 0.85, "decision": "CORRECT", ...}
```
```
`json.loads()` cannot parse a string starting with `` ` ``. It always raises
`JSONDecodeError`, triggering the fallback. This was invisible because the fallback
silently returned 0.5/AMBIGUOUS — which is a valid CRAG decision. The AMBIGUOUS
refinement loop condition `re_eval["score"] > eval_result["score"]` becomes
`0.5 > 0.5` which is always False, so refinement never fires. The pipeline was
silently bypassing all CRAG decisions for every query, every session.

**Fix:**
Strip markdown fences and extract the JSON object before parsing:
```python
raw = response.content[0].text.strip()
if raw.startswith("```"):
    raw = "\n".join(raw.split("\n")[1:])
    raw = raw.rsplit("```", 1)[0].strip()
start = raw.find("{")
end   = raw.rfind("}") + 1
if start != -1 and end > start:
    raw = raw[start:end]
try:
    return json.loads(raw)
except json.JSONDecodeError:
    return {"score": 0.5, "decision": "AMBIGUOUS", ...}
```

Same fence-stripping pattern was already used in `_evaluate_faithfulness()` and the
`dean_node.py` — it was just missing from `_evaluate_retrieval()`.

**Impact after fix:**
- R1 (correct answer): CRAG CORRECT 0.85 ✓
- R3 (wrong guess, median): CRAG CORRECT 0.85 ✓
- R6 (wrong guess, radial): CRAG CORRECT 0.82 ✓
- R2/R4 (idk/unrelated): CRAG INCORRECT → out-of-scope → fallback (correct behaviour)

**Why this wasn't caught earlier:**
The AMBIGUOUS fallback is a valid CRAG state — it means "retrieval is uncertain,
refine and retry." The pipeline produced reasonable output even when broken. The
silent failure only became visible when faithfulness scores remained low despite
corpus augmentation.

---

---

### [2026-04-17] Turn-Aware Retrieval Anchoring in Faithfulness Evaluation

**Problem identified:**
After CRAG fix, faithfulness improved but Dean faithfulness (0.79) was still below
0.85. Wrong-guess scenarios (R3: "Is it the median nerve?", R6: "Is it maybe the
radial nerve?") were the cause: `corrective_retrieve(query=student_msg)` retrieved
chunks about the *wrong* nerve (because the student's query mentioned it), but the
teacher/Dean needed to hint about the *right* concept (ulnar nerve). Dean's steering
claims were accurate but not grounded in the retrieved wrong-nerve chunks.

**Options considered:**

| Approach | Description | Verdict |
|---|---|---|
| **A — Dual retrieval** | Retrieve for student query AND for target concept, merge | Rejected — doubles API calls; cross-encoder has no diversity awareness so merging 6 chunks → top-3 is noisy |
| **B — Concept-anchored turn_query** | Pass `build_turn_query(CONCEPT, student_msg, CONCEPT, turn)` as `turn_query` to `corrective_retrieve` | **Chosen** |
| **C — Accept the gap** | Document 0.79 as known limitation of wrong-guess scenarios | Rejected — R3 and R6 are 25% of scenarios, pulling the overall below target |

**Decision: Approach B — Concept-Anchored turn_query**

`build_turn_query` already existed in `retrieval/turn_aware.py` with the exact
facet-ladder logic needed:
- `turn=0`: returns `CONCEPT` → retrieves broad ulnar nerve content
- `turn=1`: returns `"{CONCEPT} anatomy location structure"` → foundational anatomy
- `turn=2`: returns `"{CONCEPT} function clinical significance occupational therapy"`

By passing this as `turn_query` to `corrective_retrieve`, the search is anchored to
the target concept even when the student's surface query names the wrong nerve.
This mirrors what the live graph does: the manager_agent extracts `current_concept`
at session start, and subsequent retrieval uses `build_turn_query` with that concept
anchored regardless of what the student says.

**Files changed:** `evaluation/faithfulness.py`, `evaluation/faithfulness_extended.py`

---

---

### [2026-04-17] Cross-Encoder Consistency Fix — Use search_query for Reranking

**Problem identified:**
After concept-anchored retrieval, R5 ("Now I understand — how does this apply to OT?",
turn=2) still fell back to hardcoded chunks. Investigation: `corrective_retrieve`
used `turn_query` for vector search (correct) but then passed the raw `query` (the
student's meta-question) to the cross-encoder reranker. The cross-encoder scored
OT-content chunks against "how does this apply to OT?" — a meta-question, not an
anatomy query — yielding rerank scores below `OUT_OF_SCOPE_THRESHOLD`. The pipeline
correctly returned out-of-scope (it thought the chunks were irrelevant to the query),
but the real problem was the query mismatch.

**Root cause:**
```python
# Before — mismatch: retrieve with turn_query, rerank with raw query
search_query = turn_query if turn_query else query
embedding = _embed_query(search_query)         # ← uses turn_query ✓
results   = _vector_search(embedding)
# ... CRAG eval ...
reranked, rerank_log = reranker.rerank_with_logging(
    query,    # ← uses raw student message ✗
    results, ...
)
```

**Fix:** Pass `search_query` (the concept-anchored version) to the reranker:
```python
reranked, rerank_log = reranker.rerank_with_logging(
    search_query,   # ← consistent with vector search ✓
    results, ...
)
```

This makes the pipeline internally consistent: both retrieval and reranking score
against the same concept-anchored query. Also added `search_query` to the audit
log for traceability.

**File changed:** `retrieval/crag.py` — `corrective_retrieve()`

**Impact:** R5 now retrieves CORRECT OT ulnar nerve chunks instead of falling back.
Raw faithfulness on R5 went from 0.00 → 1.00. Overall raw: 0.91 → 1.00.

---

---

### [2026-04-17] Extended Faithfulness Test Suite — 16 Scenarios, 4 Concepts

**Motivation:**
The original 8-scenario faithfulness test used a single concept (ulnar nerve). Passing
on one concept does not validate that the corpus and retrieval pipeline generalise.
A concept with poor corpus coverage (e.g., brachial plexus) would fail even if the
pipeline works correctly — this needs to be visible, not averaged away.

**Design:**
`evaluation/faithfulness_extended.py` — 4 concepts × 4 scenario types = 16 tests.

Each concept has four scenarios that cover the full range of student dialogue states:
- **A** — Correct identification at turn 2 (reveal path, many factual claims)
- **B** — Wrong-guess hint (wrong nerve named, Socratic redirect needed)
- **C** — IDK hint (student gives up, progressive scaffolding)
- **D** — Clinical follow-up at turn 2 (OT application reveal)

Concepts: ulnar nerve, radial nerve, carpal tunnel syndrome, brachial plexus.

Each concept has its own concept-specific fallback chunks (used when CRAG returns
out-of-scope), replacing the single hardcoded `CHUNKS` list from the original file.

**Results (2026-04-17):**

| Concept | Raw | Dean |
|---|---|---|
| ulnar nerve | 0.92 ✓ | 1.00 ✓ |
| radial nerve | 1.00 ✓ | 0.93 ✓ |
| carpal tunnel syndrome | 1.00 ✓ | 0.91 ✓ |
| brachial plexus | 0.78 ⚠ | 0.85 ⚠ |
| **OVERALL (16 scenarios)** | **0.93 ✓** | **0.93 ✓** |

**Finding — brachial plexus weak spot:**
Brachial plexus raw score is 0.78 ✗. Two failures:
- B-B (wrong-guess spinal cord): Dean makes a contrast claim ("The spinal cord itself
  is a different structure") about the wrong structure — a claim type that is inherently
  ungroundable when chunks are anchored to the right concept.
- B-D (Erb's palsy clinical): Raw teacher makes a waiter's tip posture claim that is
  in the corpus but CRAG returned AMBIGUOUS for the brachial plexus query, so slightly
  off-target chunks were used.

This reveals a real corpus gap: the brachial plexus chapter in Gray's is being rated
AMBIGUOUS by CRAG. The OT supplement's Section 5 covers it briefly. Adding a more
detailed brachial plexus clinical reference (Erb's/Klumpke's) would push this to ≥ 0.85.

---

---

### [2026-04-17] Hardcoded Value Audit — All Magic Numbers Extracted to config.py

**Audit scope:** All files modified during corpus augmentation and faithfulness work:
`evaluation/faithfulness.py`, `evaluation/faithfulness_extended.py`,
`ingest/add_grays_nerves.py`, `ingest/add_ot_supplement.py`, `retrieval/crag.py`.

**Bug found and fixed — `turn_count=0` in `_run_dean` (faithfulness.py):**
`_run_dean()` had `turn_count=0` hardcoded in the Dean prompt fill, regardless of
which scenario (turn=0, 1, or 2) it was evaluating. This told Dean "you are at turn 0"
even for turn=2 reveal scenarios, where `reveal_permitted=True` contradicts turn=0
(in the live system, reveal is never permitted at turn 0). Dean's reasoning could be
misled by the inconsistent turn/reveal combination.

Fix: added `turn: int` parameter to `_run_dean()`, threaded actual turn through all
call sites. The extended file already had this correct — only the original file was broken.

**Magic numbers extracted to `config.py`:**

| Constant | Value | Used in |
|---|---|---|
| `TEACHER_MAX_TOKENS` | 400 | Teacher + Dean generation (eval + live graph) |
| `DEAN_MAX_TOKENS` | 300 | Dean quality-gate JSON |
| `FAITHFULNESS_MAX_TOKENS` | 800 | Haiku claim-checking JSON (eval only) |
| `CRAG_EVAL_MAX_TOKENS` | 200 | CRAG evaluator JSON (live pipeline) |
| `INGEST_BATCH_SIZE` | 100 | ChromaDB upsert batch in ingest scripts |
| `OT_NEUROLOGY_CHAPTER` | 13 | Chapter number for neurology/neuroscience metadata |

`INGEST_BATCH_SIZE` was 50 in the supplement/Gray's scripts but 100 in the main
ingest pipeline — this inconsistency was also fixed (aligned to 100).

**Why these belong in config.py:**
CLAUDE.md rule: "No model names hardcoded anywhere except config.py." The same
principle extends to any operational constant that affects system behaviour and
might need tuning. Token budgets directly affect response length and cost.
Batch sizes affect ingest memory behaviour. Chapter numbers affect retrieval
metadata filtering.
