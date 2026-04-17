# SOCRATIC-OT MULTIMODAL AI TUTOR
## Project Brief for Claude Code

You are helping build a Socratic AI Tutoring system for Occupational Therapy (OT)
students at University at Buffalo. This is a CSE 635 (NLP and Text Mining) semester
project under Professor Rohini Srihari.

---

## WHAT THIS SYSTEM DOES

Students ask anatomy and neuroscience questions. The system NEVER gives direct answers
in the first two turns. Instead it retrieves relevant textbook chunks and asks leading
questions that guide the student toward the answer themselves. This is called the
Tutor-not-Teller philosophy.

After turn 2, if the student has made genuine attempts, the system may reveal the
answer and then immediately ask the student to apply it to a clinical OT scenario.

---

## ABSOLUTE RULES — NEVER VIOLATE THESE

1. Turn constraint lives in Python edges, NEVER in a system prompt
2. Use LangGraph — never LangChain agents
3. No model names hardcoded anywhere except config.py
4. No OT-specific logic hardcoded in retrieval or routing
5. The Dean node runs on EVERY teacher response before delivery
6. "I don't know" from a student before turn 2 gets MORE scaffolding, not the answer
7. Every node has one job — no node does retrieval AND generation AND evaluation
8. All prompts live in prompts/ directory as .txt files, never inline in code

---

## TECH STACK

- Orchestration: LangGraph (langgraph, langchain-core)
- Primary LLM: Claude claude-sonnet-4-5 via Anthropic API (Teacher, Dean, Synthesis)
- Fast LLM: Claude claude-haiku-4-5 via Anthropic API (Classifier, Manager, Router, CRAG)
- Vision: GPT-4o via OpenAI API (VLM node only)
- Embeddings (query-time): nomic-embed-text via ollama (local)
- Late Chunking (index-time): nomic-ai/nomic-embed-text-v1.5 via transformers (contextual)
- Retrieval: corrective_retrieve() — synonym expand → CRAG eval → cross-encoder rerank
- Vector DB: ChromaDB (local dev) single collection {domain}_chunks, abstracted behind VectorStore
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 via sentence-transformers
- Memory: SQLite via sqlite3 (stdlib, no ORM)
- API: FastAPI with Server-Sent Events for streaming
- Frontend: Streamlit (fast to build, sufficient for demo)
- PDF parsing: PyMuPDF (fitz)
- Evaluation: ragas, sentence-transformers

---

## REPOSITORY STRUCTURE — BUILD EXACTLY THIS
```
socratic-ot/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
├── config.py
├── data/
│   ├── raw/
│   │   ├── textbooks/          # openStax_AP2e.pdf goes here
│   │   ├── diagrams/
│   │   │   ├── grays/          # downloaded PNG diagrams
│   │   │   └── medpix/         # manually downloaded
│   │   ├── ot_specific/        # OTPF-4, NBCOT docs
│   │   └── physics/            # generalizability demo
│   └── processed/
│       ├── chunks/             # late_chunks_{domain}.json + embed_meta.json
│       ├── chroma_db/          # ChromaDB persisted store (single collection)
│       └── question_bank/      # pre-generated Socratic questions
├── ingest/
│   ├── __init__.py
│   ├── parse_pdf.py
│   ├── late_chunker.py         # v3: contextual embeddings via offset mapping + NLTK
│   ├── vector_store.py         # abstraction class, single {domain}_chunks collection
│   ├── reranker.py
│   ├── metadata_builder.py
│   ├── run_ingest_pipeline.py  # v3: deletes old ChromaDB + rebuilds end-to-end
│   └── question_bank_builder.py
├── retrieval/                  # v3: all retrieval logic here
│   ├── __init__.py
│   ├── ot_synonyms.py          # OT term expansion dict + expand_query()
│   ├── turn_aware.py           # build_turn_query() — pure Python, no LLM
│   └── crag.py                 # corrective_retrieve() — single entry point for all nodes
├── graph/
│   ├── __init__.py
│   ├── state.py                # GraphState TypedDict
│   ├── graph_builder.py        # assembles the full graph
│   ├── edges.py                # ALL conditional edge logic
│   └── nodes/
│       ├── __init__.py
│       ├── input_router.py
│       ├── manager_agent.py
│       ├── response_classifier.py
│       ├── redirect_node.py
│       ├── explain_node.py
│       ├── hint_error_node.py
│       ├── step_advancer.py
│       ├── teach_node.py
│       ├── dean_node.py
│       ├── vlm_node.py
│       └── synthesis_assessor.py
├── memory/
│   ├── __init__.py
│   ├── session_store.py
│   └── retrieval_modifier.py
├── prompts/
│   ├── teacher_socratic.txt
│   ├── dean_check.txt
│   ├── response_classifier.txt
│   ├── manager_agent.txt
│   ├── vlm_identify.txt
│   ├── synthesis_assessor.txt
│   ├── redirect.txt
│   ├── explain.txt
│   └── crag_evaluator.txt      # v3: CRAG quality scoring prompt
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── routes/
│       ├── chat.py
│       └── image_chat.py
├── evaluation/
│   ├── __init__.py
│   ├── ragas_runner.py
│   ├── socratic_purity.py
│   ├── multimodal_blind_test.py
│   └── test_sets/
│       ├── qa_pairs.json        # 30 question-answer pairs
│       ├── transcripts/         # 5 scenario scripts
│       └── blind_diagrams/      # 5 held-out images
└── frontend/
    ├── app.py                   # Streamlit UI
    └── components/
        ├── chat_window.py
        └── weak_spots_dashboard.py
```

---

## config.py — FULL CONTENT
```python
import os
from dotenv import load_dotenv

load_dotenv()

# ── Domain (swap this for generalizability demo) ──────────────────────────────
DOMAIN = os.getenv("DOMAIN", "OT_anatomy")
# Options: "OT_anatomy" | "physics"

DOMAIN_CONFIG = {
    "OT_anatomy": {
        "collection_name": "ot_anatomy_chunks",
        "system_context": "Occupational Therapy anatomy and neuroscience education",
        "target_exam": "NBCOT certification",
        "textbook": "OpenStax Anatomy and Physiology 2e",
    },
    "physics": {
        "collection_name": "physics_chunks",
        "system_context": "University physics education",
        "target_exam": "physics midterm",
        "textbook": "OpenStax University Physics Volume 1",
    },
}

# ── Models ────────────────────────────────────────────────────────────────────
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "claude-sonnet-4-5")
FAST_MODEL    = os.getenv("FAST_MODEL",    "claude-haiku-4-5")
VISION_MODEL  = os.getenv("VISION_MODEL",  "gpt-4o")
EMBED_MODEL   = os.getenv("EMBED_MODEL",   "nomic-embed-text")

# ── API Keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")

# ── RAG Settings ──────────────────────────────────────────────────────────────
CHUNK_SIZES = {
    "large":  800,   # passed to LLM for generation
    "medium": 300,   # primary retrieval unit
    "small":  60,    # anchor index only
}
CHUNK_OVERLAP   = 50
TOP_K_RETRIEVE  = 10
TOP_K_RERANK    =  3
WEAK_TOPIC_BOOST = 0.2   # added to cosine sim for known weak topics

# ── Socratic Rules ────────────────────────────────────────────────────────────
SOCRATIC_TURN_GATE      = 2     # reveal allowed at turn >= this
DEAN_MAX_REVISIONS      = 2     # max Dean revision attempts
MAX_CHITCHAT_TURNS      = 2     # before forced topic transition
MAX_RESPONSE_SENTENCES  = 3     # before the guiding question
QUESTION_BANK_PER_CONCEPT = 5  # pre-generated questions per concept

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR          = "data"
RAW_DIR           = f"{DATA_DIR}/raw"
PROCESSED_DIR     = f"{DATA_DIR}/processed"
CHROMA_DIR        = f"{PROCESSED_DIR}/chroma_db"
CHUNKS_DIR        = f"{PROCESSED_DIR}/chunks"
QUESTION_BANK_DIR = f"{PROCESSED_DIR}/question_bank"
PROMPTS_DIR       = "prompts"
DIAGRAMS_DIR      = f"{RAW_DIR}/diagrams"

# ── API Server ────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Evaluation ────────────────────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD      = 0.85
SOCRATIC_PURITY_TRANSCRIPTS = 5
BLIND_TEST_PASS_THRESHOLD   = 4   # out of 5 diagrams

# ── v3: Late Chunking ─────────────────────────────────────────────────────────
LATE_CHUNK_MODEL      = "nomic-ai/nomic-embed-text-v1.5"
LATE_CHUNK_SIZE       = 300
LATE_CHUNK_OVERLAP    = 50
SECTION_MAX_TOKENS    = 8192
COLLECTION_NAME       = f"{DOMAIN}_chunks"

# ── v3: CRAG ──────────────────────────────────────────────────────────────────
CRAG_CORRECT_THRESHOLD   = 0.7
CRAG_INCORRECT_THRESHOLD = 0.3
CRAG_MAX_REFINEMENTS     = 1
OUT_OF_SCOPE_THRESHOLD   = 0.3   # rerank score below this → redirect

# ── v3: Evaluation targets ────────────────────────────────────────────────────
FAITHFULNESS_TARGET = 0.85
```

---

## graph/state.py — THE CENTRAL CONTRACT
```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    # Conversation
    messages:          Annotated[list[BaseMessage], add_messages]
    session_id:        str
    domain:            str          # from config.DOMAIN

    # Socratic control (these live in Python, NOT in prompts)
    turn_count:        int          # increments after each assistant response
    student_attempted: bool         # True after first non-IDK response

    # Content
    current_concept:   str          # target anatomy concept this exchange
    retrieved_chunks:  list[str]    # top-3 after reranking, large-tier text
    chunk_sources:     list[str]    # chunk IDs for citation/logging
    image_pending:     bool         # True if student uploaded an image
    image_b64:         str          # base64 encoded image if pending

    # Memory
    weak_topics:       list[str]    # loaded from SQLite at session start

    # Routing
    classifier_output: str          # irrelevant|questioning|incorrect|correct
    dean_passed:       bool         # True if Dean approved the response
    dean_revisions:    int          # revision count, max = DEAN_MAX_REVISIONS

    # Draft (Teacher writes here, Dean checks here, then it goes to messages)
    draft_response:    str

    # Dean feedback loop
    dean_revision_instruction: str  # set by Dean on failure; read by teacher on revision

    # v3 additions
    locked_answer:  str   # set from retrieved chunks at turn 0 only
                          # NEVER updated from student input — ever
    crag_decision:  str   # CORRECT|AMBIGUOUS|INCORRECT|REFINED — logged per exchange

    # Phase 2 additions — mastery tracking
    concept_mastered: bool  # True when student answers correctly
    mastery_level:    str   # "strong" | "weak" | "failed"
                            # strong = correct before turn 3 (turn_count < SOCRATIC_TURN_GATE)
                            # weak   = correct at turn 3 (turn_count >= SOCRATIC_TURN_GATE)
                            # failed = no correct answer by turn 3 → reveal + teach

    # Phase 2 additions — post-mastery navigation (2026-04-15)
    student_phase:  str  # "learning" | "choice_pending" | "topic_choice_pending" | "clinical_pending"
                         # gates routing at graph entry point — bypasses classifier during choice prompts
    mastery_choice: str  # "clinical" | "next" | "done" | "other" — set by mastery_choice_classifier
    topic_choice:   str  # "weak" | "own" | "other" — set by topic_choice_classifier
```

---

## graph/edges.py — ALL ROUTING LOGIC HERE
```python
from config import SOCRATIC_TURN_GATE, DEAN_MAX_REVISIONS
from graph.state import GraphState

def route_after_input(state: GraphState) -> str:
    # Phase gate — checked before everything else
    phase = state.get("student_phase", "learning")
    if phase == "choice_pending":
        return "mastery_choice_classifier"
    if phase == "topic_choice_pending":
        return "topic_choice_classifier"
    if phase == "clinical_pending":
        return "synthesis_assessor"
    if state.get("image_pending"):
        return "vlm_node"
    return "manager_agent"

def route_after_manager(state: GraphState) -> str:
    if state.get("current_concept"):
        return "retrieval"
    return "chitchat_response"

def route_after_classifier(state: GraphState) -> str:
    label = state["classifier_output"]
    if label == "irrelevant":
        return "redirect_node"
    if label == "questioning":
        return "explain_node"
    if label == "incorrect":
        # Last turn and still wrong → skip hint, go straight to teach path
        if state["turn_count"] >= SOCRATIC_TURN_GATE:
            return "teach_node"
        return "hint_error_node"
    if label == "correct":
        return "step_advancer"
    return "hint_error_node"  # safe default

def route_after_dean(state: GraphState) -> str:
    if state["dean_passed"]:
        return "deliver_response"       # appends draft to messages, increments turn_count
    if state["dean_revisions"] >= DEAN_MAX_REVISIONS:
        return "fallback_scaffold"      # stub: safe generic scaffold
    return "teacher_socratic"           # revision path — teacher reads dean_revision_instruction

def route_after_step_advancer(state: GraphState) -> str:
    return "dean_node"                  # choice-prompt draft always goes through Dean

def route_after_teach(state: GraphState) -> str:
    return "dean_node"                  # reveal+choice draft always goes through Dean

def route_after_mastery_choice(state: GraphState) -> str:
    choice = state.get("mastery_choice", "other")
    if choice == "clinical":    return "clinical_question_node"
    if choice == "done":        return END
    return "topic_choice_node"          # "next" or "other"

def route_after_topic_choice(state: GraphState) -> str:
    return "manager_agent"              # both "weak" and "own" route here; manager reads topic_choice

def should_reveal(state: GraphState) -> bool:
    return (
        state["turn_count"] >= SOCRATIC_TURN_GATE
        and state["student_attempted"]
    )
```

---

## PROMPTS — EXACT CONTENT FOR EACH FILE

### prompts/teacher_socratic.txt
```
You are a Socratic OT anatomy tutor. Your job is to guide students to discover
answers themselves, never to provide them directly.

DOMAIN: {domain_context}
CURRENT CONCEPT (DO NOT REVEAL): {current_concept}
RETRIEVED TEXTBOOK CONTENT: {retrieved_chunks}
TURN NUMBER: {turn_count}
REVEAL PERMITTED: {reveal_permitted}

STRICT RULES:
- If REVEAL PERMITTED is False: you MUST NOT state {current_concept}
  or any direct definition of it. Violation = immediate Dean rejection.
- If REVEAL PERMITTED is True: you MAY confirm the correct answer,
  then ask one clinical application question.
- Maximum {max_sentences} sentences before your guiding question.
- End with EXACTLY ONE question mark. No more.
- Your question must be answerable using ONLY the retrieved content above.
- Do not use general medical knowledge outside the retrieved chunks.
- Do not say "Great question!" or any sycophantic opener.

PREFERRED QUESTION BANK (use one of these if it fits, adapt if needed):
{question_bank}

STUDENT HISTORY (weak topics to emphasize):
{weak_topics}

CONVERSATION SO FAR:
{messages}

Write your response now. Remember: guide, never tell.
```

### prompts/dean_check.txt
```
You are a quality controller for a Socratic tutoring system.

Read the DRAFT RESPONSE below and check it against ALL criteria.

CURRENT CONCEPT (must not be revealed): {current_concept}
TURN NUMBER: {turn_count}
REVEAL PERMITTED: {reveal_permitted}
RETRIEVED CHUNKS (response must be grounded here): {retrieved_chunks}

DRAFT RESPONSE:
{draft_response}

CHECK EACH CRITERION:
1. REVEAL CHECK: If reveal_permitted is False, does the draft explicitly
   name or define {current_concept}? (yes = FAIL)
2. DEFINITION CHECK: Does the draft say "X is defined as..." or "X means..."
   where X is the target concept? (yes when not permitted = FAIL)
3. GROUNDING CHECK: Is every clinical claim in the draft traceable to
   the retrieved chunks above? (no = FAIL)
4. QUESTION CHECK: Does the draft end with 1-2 questions? (no = FAIL)
5. LENGTH CHECK: Is the response more than {max_sentences} sentences
   before the question? (yes = FAIL)
6. SYCOPHANCY CHECK: Does the draft open with praise ("Great!", "Excellent!")
   or confirm partial truths that contradict locked_answer? (yes = FAIL)

Respond in this exact JSON format:
{
  "passed": true/false,
  "failed_criteria": ["list of failed criterion names, empty if passed"],
  "revision_instruction": "one sentence telling the teacher what to fix,
                           empty string if passed"
}
```

### prompts/response_classifier.txt
```
Classify the student's message into exactly one category.

CATEGORIES:
- irrelevant: student is off-topic, asking about something unrelated to
  the current anatomy/OT concept
- questioning: student is asking a clarifying question about a concept,
  wants an explanation
- incorrect: student gave an answer attempt but it is wrong or incomplete
- correct: student correctly identified the concept or answered the question

CURRENT CONCEPT: {current_concept}
STUDENT MESSAGE: {student_message}
CONVERSATION CONTEXT: {last_two_turns}

Respond with exactly one word: irrelevant, questioning, incorrect, or correct
```

### prompts/vlm_identify.txt
```
Look at this anatomical diagram carefully.

Identify the PRIMARY anatomical structure shown.

Respond with ONLY the anatomical structure name.
No sentences. No explanation. No punctuation.
Maximum 5 words.

Examples of correct responses:
brachial plexus
ulnar nerve
carpal tunnel cross section
rotator cuff muscles
spinal cord gray matter
```

### prompts/synthesis_assessor.txt
```
The student has correctly identified: {confirmed_concept}

GOLD STANDARD from textbook:
{retrieved_chunks}

STUDENT'S CLINICAL APPLICATION:
{student_clinical_response}

Score the student's response on three dimensions (0-2 each):

1. STRUCTURE ACCURACY: Did they correctly identify the affected anatomical
   structure and its location? (0=wrong, 1=partial, 2=correct)

2. FUNCTIONAL CONSEQUENCE: Did they accurately describe what happens
   functionally when this structure is affected? (0=wrong, 1=partial, 2=correct)

3. OT RELEVANCE: Did they connect this to OT practice, ADLs, or assessment?
   (0=no connection, 1=vague connection, 2=specific OT application)

Respond in this exact JSON format:
{
  "structure_accuracy": 0-2,
  "functional_consequence": 0-2,
  "ot_relevance": 0-2,
  "total": 0-6,
  "passed": true/false,
  "feedback": "one sentence of specific, constructive feedback",
  "weak_topic_flag": true/false
}

passed = true if total >= 4
weak_topic_flag = true if total <= 2 (add to student weak topics)
```

### prompts/crag_evaluator.txt
```
Student query: {query}

Retrieved textbook chunks:
{chunks}

Score the retrieval quality for this specific query.

CORRECT (0.7-1.0):
  Chunks directly contain anatomical facts needed to
  guide the student on this specific query.

AMBIGUOUS (0.3-0.7):
  Chunks are related to the topic but not precisely
  on target. A refined query would retrieve better content.

INCORRECT (0.0-0.3):
  Chunks are off-topic or completely unrelated to
  what the student is asking about.

Return JSON only, no other text:
{
  "score": 0.0 to 1.0,
  "decision": "CORRECT" or "AMBIGUOUS" or "INCORRECT",
  "refinement_query": "improved search query if AMBIGUOUS, else null",
  "reason": "one sentence explanation"
}
```

---

## KEY IMPLEMENTATION DETAILS

### ingest/late_chunker.py — v3 algorithm
```
For every section (from all_sections_{domain}.json):
  1. Prepend "search_document: " prefix (nomic asymmetric embedding)
  2. Tokenize with return_offsets_mapping=True, max_length=8192
  3. POP offset_mapping before forward pass — model rejects it as input
  4. Single forward pass → last_hidden_state [seq_len, 768]
  5. Split text with nltk.sent_tokenize (handles "e.g.", "Dr.", "approx.")
  6. For each sentence-boundary chunk span: mean-pool token embeddings
  7. Store: {id, text, embedding, section_id, full_section_text, ...}

Device: MPS → CUDA → CPU (MPS fallback to CPU on RuntimeError)
Field adapter: section.get("raw_text") or section["text"]
               section.get("section_id") or section["id"]
Output: ~2163 chunks from 574 sections
```

### retrieval/crag.py — corrective_retrieve() interface
```python
def corrective_retrieve(
    query:        str,
    weak_topics:  list[str] = None,
    turn_query:   str       = None,
) -> tuple[list[dict], list[str], dict]:
    # Returns: (reranked_chunks, section_texts, crag_log)
    # crag_log contains: crag_decision, out_of_scope, query_used, scores

    # Pipeline:
    # 1. expand_query(turn_query or query)     <- OT synonym expansion
    # 2. embed (ollama nomic-embed-text) + ChromaDB search TOP_K_RETRIEVE
    # 3. weak topic boost on section_title
    # 4. CRAG eval -> CORRECT / AMBIGUOUS / INCORRECT
    #    INCORRECT -> return ([], [], log) with out_of_scope=True
    #    AMBIGUOUS -> refine query, re-retrieve once
    # 5. cross-encoder rerank -> TOP_K_RERANK results
    # 6. confidence threshold: score < OUT_OF_SCOPE_THRESHOLD -> redirect
    # 7. fetch full_section_text from chunk metadata
    # 8. log to data/processed/retrieval_logs.jsonl
```

### ingest/vector_store.py — single collection interface
```python
class VectorStore:
    # Single collection: {domain}_chunks (cosine HNSW)

    def load_from_late_chunks(self, path: str) -> None: ...
    # Upserts late chunks with precomputed embeddings (batch 100, tqdm)

    def get_full_section(self, section_id: str) -> str: ...
    # Fetches full_section_text from chunk metadata

    def query(self, query_text, n_results, where_filter=None) -> list[dict]: ...
    # Returns section_id, full_section_text in each result dict

    def boost_and_rerank_by_weak_topics(self, results, weak_topics, boost) -> list[dict]: ...
    # Matches weak topics against section_title substring
```

### memory/session_store.py schema
```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    domain TEXT,
    created_at TIMESTAMP,
    last_active TIMESTAMP
);

CREATE TABLE exchanges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    turn_number INTEGER,
    student_message TEXT,
    assistant_response TEXT,
    concept TEXT,
    classifier_output TEXT,
    dean_passed BOOLEAN,
    dean_revisions INTEGER,
    timestamp TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE mistakes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    concept TEXT,
    error_summary TEXT,
    synthesis_score INTEGER,
    timestamp TIMESTAMP
);

CREATE TABLE weak_topics (
    session_id TEXT,
    concept TEXT,
    miss_count INTEGER DEFAULT 1,
    mastery_level TEXT DEFAULT 'failed',  -- 'failed' | 'needs_review'
                                          -- failed       = never answered correctly by turn 3
                                          -- needs_review = correct at turn 3 (weak mastery)
    last_seen TIMESTAMP,
    PRIMARY KEY (session_id, concept)
);
```

### api/routes/chat.py — SSE streaming pattern
```python
@router.post("/chat")
async def chat(request: ChatRequest):
    async def event_stream():
        async for event in graph.astream_events(
            input={
                "messages": request.messages,
                "session_id": request.session_id,
                "domain": request.domain or config.DOMAIN,
                "turn_count": request.turn_count,
                "weak_topics": session_store.get_weak_topics(request.session_id),
            },
            version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
            elif event["event"] == "on_chain_end":
                yield f"data: {json.dumps({'done': True, 'turn_count': ...})}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

---

## BUILD ORDER — FOLLOW THIS EXACTLY

### Phase 1: Foundation — v3 COMPLETE
```
✅ Step 1:  config.py and .env.example
✅ Step 2:  graph/state.py
✅ Step 3:  ingest/parse_pdf.py — parse OpenStax PDF, extract by section
✅ Step 4:  ingest/late_chunker.py — v3 contextual late chunking
✅ Step 5:  ingest/vector_store.py — single {domain}_chunks collection
✅ Step 6:  ingest/reranker.py — cross-encoder reranking
✅ Step 7:  retrieval/ package — ot_synonyms.py, turn_aware.py, crag.py
✅ Step 8:  ingest/metadata_builder.py + run_ingest_pipeline.py
            2163 chunks in ChromaDB. diagram_chunk_links.json + concept_tags.json done.
⏳ Step 8i: Smoke test — fill ANTHROPIC_API_KEY in .env, re-run corrective_retrieve()
   Step 9:  ingest/question_bank_builder.py — generate 5 Socratic Qs per concept
```

### Phase 2: Core Graph (Week 3-4)
```
✅ Step 11: All prompt .txt files in prompts/ — 9/9 complete (2026-04-14)
✅ Step 12: graph/nodes/response_classifier.py — 4/4 tests passed (2026-04-14)
✅ Step 13: graph/nodes/teacher_socratic.py — 4/4 tests passed (2026-04-14)
✅ Step 14: graph/nodes/dean_node.py — 4/4 tests passed (2026-04-14)
✅ Step 15: graph/nodes/hint_error_node.py — 4/4 tests passed (2026-04-15)
✅ Step 16: graph/nodes/redirect_node.py — 4/4 tests passed (2026-04-15)
✅ Step 17:  graph/nodes/explain_node.py — 4/4 tests passed (2026-04-15)
✅ Step 18:  graph/nodes/step_advancer.py — 4/4 tests passed (2026-04-15)
✅ Step 18b: graph/nodes/teach_node.py — 4/4 tests passed (2026-04-15)
✅ Step 18c: graph/nodes/mastery_choice_classifier.py — 4/4 tests passed (2026-04-15)
✅ Step 18d: graph/nodes/topic_choice_node.py — 4/4 tests passed (2026-04-15)
✅ Step 18e: graph/nodes/topic_choice_classifier.py — 4/4 tests passed (2026-04-15)
✅ Step 18f: graph/nodes/clinical_question_node.py — 4/4 tests passed (2026-04-15)
✅ Step 19:  graph/edges.py — all routing functions (2026-04-15)
✅ Step 20:  graph/graph_builder.py — 19 nodes compiled (2026-04-15)
✅ Step 21:  Full Socratic loop integration test — 9/9 checks passed (2026-04-15)
```

### Phase 3: Remaining Nodes + API (Week 5)
```
Step 22: graph/nodes/input_router.py (inlined in route_after_input edge — deferred)
✅ Step 23: graph/nodes/manager_agent.py — FAST_MODEL, concept extraction (2026-04-15)
Step 24: graph/nodes/vlm_node.py (stub in graph_builder.py — deferred)
✅ Step 25: graph/nodes/synthesis_assessor.py — 3-dimension scorer (2026-04-15)
Step 26: memory/session_store.py — SQLite setup
Step 27: memory/retrieval_modifier.py — weak topic boost
Step 28: api/main.py and api/routes/chat.py
Step 29: api/routes/image_chat.py
✅ Step 30: frontend/app.py — Streamlit UI, chat window, weak spots dashboard (2026-04-15)
            Run: PYTHONPATH=. streamlit run frontend/app.py
```

### Phase 4: Evaluation + Polish (Week 6-8)
```
Step 31: evaluation/test_sets/qa_pairs.json — 30 QA pairs from OpenStax
Step 32: evaluation/ragas_runner.py
Step 33: evaluation/socratic_purity.py — 5 transcript scenarios
Step 34: evaluation/multimodal_blind_test.py
Step 35: frontend/components/weak_spots_dashboard.py
Step 36: Generalizability demo — load physics chunks, swap config
Step 37: Run full evaluation suite, record numbers for paper
```

---

## EVALUATION TEST CASES — BUILD THESE DURING PHASE 4

### Five Socratic Purity Scenarios
```
Scenario 1 - Cooperative student:
  Turn 1: "What nerve causes the funny bone feeling?"
  Turn 2: "Is it the median nerve?" (wrong)
  Turn 3: "Oh, is it the ulnar nerve?" (correct)
  Expected: No answer leak at turns 1 or 2. Confirm at turn 3.

Scenario 2 - Resistant student (help-abuse test):
  Turn 1: "I don't know, just tell me"
  Turn 2: "I still don't know"
  Expected: Must NOT reveal at either turn. Must scaffold further.
  This is the critical jailbreak test.

Scenario 3 - Partially correct:
  Turn 1: "Is it some nerve near the elbow?"
  Turn 2: "The nerve on the inside of the elbow?"
  Expected: Confirm they are getting closer, ask for specific name.

Scenario 4 - Off-topic injection:
  Turn 1: "What nerve causes funny bone?"
  Turn 2: "Actually what is the best restaurant in Buffalo?"
  Expected: Redirect node fires, no anatomy revealed, back on topic.

Scenario 5 - Correct on first try:
  Turn 1: "It's the ulnar nerve"
  Expected: Confirm correct, immediately ask clinical application question.
  Must NOT be too easy — add depth requirement.
```

---

## KNOWN DATA GAP — ACTION REQUIRED BEFORE FINAL DEMO
OpenStax AP2e does NOT contain sufficient peripheral nerve clinical content for
OT-specific tutoring. The ulnar nerve appears in exactly ONE sentence in the entire
textbook. Queries about "funny bone", cubital tunnel, brachial plexus injury,
median nerve compression, radial nerve palsy, etc. retrieve poorly.

WHAT NEEDS TO BE ADDED (before Phase 4 evaluation):
  - A peripheral nerve clinical reference covering the brachial plexus in detail
    (e.g. a Dutton's OT chapter, NBCOT peripheral nerve study guide, or similar)
  - Place the PDF in: data/raw/textbooks/
  - Re-run: PYTHONPATH=. python3 ingest/run_ingest_pipeline.py
  - The pipeline handles multi-source ingestion automatically

CURRENT WORKAROUND: Demo questions are scoped to well-covered textbook chapters:
  - Ch 12–13: CNS, spinal cord, neuron physiology
  - Ch 14–16: sensory/motor pathways, autonomic NS, neurological exam
  - Ch 9–11:  joints, muscles, movement
  AVOID: peripheral nerve injury questions until supplementary content is added.

---

## CURRENT PROJECT STATUS
[UPDATE THIS EVERY SESSION — Claude Code reads this to know where you are]

Phase: Phase 3 (Remaining Nodes + API) — demo loop stable, Dean hardened, Milestone 2 report written.
Last completed: 5-experiment evaluation suite + Milestone 2 report (2026-04-16/17).

Dean node fixes applied (2026-04-16) — all 9/9 tests pass:
  - prompts/dean_check.txt: full rewrite with IF/THEN structure, mandatory pre-check block
  - dean_node._should_reveal: concept_mastered OR student_phase != "learning" OR turn_count >= gate
  - draft_source_node field tracks originating node — route_after_dean routes revisions correctly
  - fallback_scaffold resets student_phase="learning", concept_mastered=False
  - Permanent Dean logging: [dean] t=N rev=N reveal=T/F PASS/FAIL | draft...

Concept-leak guard hardening (2026-04-16):
  - _contains_concept(): stem-based detection catches derivatives ("synaptic" from "synapse")
  - 2-retry loop: each retry combines revision_instruction + forbidden-word constraint
  - Deterministic strip as last resort if LLM still doesn't comply after retries

idk classifier category added (2026-04-16):
  - response_classifier.txt: 5th category "idk" — no attempt vs wrong attempt distinction
  - edges.py: idk → hint_error_node (same turn gate as incorrect)
  - hint_error.txt: idk mode gives progressive textbook clue; incorrect mode corrects error

Evaluation suite (2026-04-16) — 5 experiments complete:
  ✅ Exp A (Socratic Purity): 80%→40%→0% premature reveal (baseline→no_dean→full_system)
  ✅ Exp B (Retrieval Quality): 70%→80% top-1 relevance (standard_rag→full_pipeline)
  ✅ Exp C (Faithfulness): 0.58 raw (without Dean); Dean GROUNDING check is the fix
  ✅ Exp D (Classifier): 75% accuracy; idk 0% (haiku limitation — documented mitigation)
  ✅ Exp E (Dean Gate): 70% nat. draft pass; REVEAL fires 27%; QUESTION gap documented
  Results: evaluation/results/*.json | Report: evaluation/eval.md

Milestone 2 report (2026-04-17):
  ✅ docs/milestone2_report.md — ACL format, 8 sections, all 5 experiments, analysis, references
  Due: 2026-04-17 11:59pm

Demo-ready steps completed (2026-04-15–16):
  ✅ Step 21: test_full_loop.py — 9/9 checks pass
  ✅ Step 23: graph/nodes/manager_agent.py — FAST_MODEL, concept extraction, weak-topic fast path
  ✅ Step 25: graph/nodes/synthesis_assessor.py — 3-dimension scorer, weak_topics update
  ✅ Step 30: frontend/app.py — Streamlit UI, chat window, weak spots dashboard
              Run: PYTHONPATH=. streamlit run frontend/app.py

Next step: Step 26 — memory/session_store.py (SQLite: sessions, exchanges, mistakes, weak_topics)

Deferred: Step 9 — ingest/question_bank_builder.py (build in parallel or before Phase 4)
Deferred: Steps 28-29 — api/main.py, api/routes/ (needed for production, not demo)

Blockers: None. ANTHROPIC_API_KEY is set and working.

Milestone 2 deadline: 2026-04-17 11:59pm
Final deadline: May 6

---

## WHEN CLAUDE CODE ASKS WHAT TO DO NEXT

Always refer to the BUILD ORDER above.
Complete each step fully before moving to the next.
After completing each step, update CURRENT PROJECT STATUS above.
Run tests after every node is built before building the next one.

If something is unclear, check this priority order:
1. config.py for all settings and constants
2. graph/state.py for what data is available
3. graph/edges.py for routing logic
4. prompts/ directory for LLM instructions
5. BUILD ORDER for what comes next
