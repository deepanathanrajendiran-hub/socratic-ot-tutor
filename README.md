# Socratic OT Tutor

A Socratic AI Tutoring system for Occupational Therapy (OT) students at University at Buffalo.  
CSE 635 — NLP and Text Mining — under Professor Rohini Srihari.

---

## What It Does

Students ask anatomy and neuroscience questions. The system **never gives direct answers** in the first two turns. Instead it retrieves relevant textbook chunks and asks Socratic leading questions that guide the student to discover the answer themselves (**Tutor-not-Teller** philosophy).

After turn 2, if the student has made genuine attempts, the system may reveal the answer and immediately ask the student to apply it to a clinical OT scenario.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph |
| Primary LLM | Claude claude-sonnet-4-5 (Teacher, Dean, Synthesis) |
| Fast LLM | Claude claude-haiku-4-5 (Classifier, Manager, Router, CRAG) |
| Vision | GPT-4o (VLM node) |
| Embeddings | nomic-embed-text via Ollama (local) |
| Late Chunking | nomic-ai/nomic-embed-text-v1.5 via transformers |
| Vector DB | ChromaDB (local, single collection) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Memory | SQLite |
| API | FastAPI + Server-Sent Events |
| Frontend | Streamlit |
| PDF Parsing | PyMuPDF (fitz) |
| Evaluation | ragas, sentence-transformers |

---

## Project Structure

```
socratic-ot/
├── config.py                    # All constants — no hardcoding elsewhere
├── graph/
│   ├── state.py                 # GraphState TypedDict — central contract
│   ├── edges.py                 # ALL routing logic
│   ├── graph_builder.py         # Assembles the full LangGraph graph
│   └── nodes/                   # One node = one job
│       ├── teacher_socratic.py
│       ├── dean_node.py
│       ├── response_classifier.py
│       ├── manager_agent.py
│       ├── hint_error_node.py
│       ├── redirect_node.py
│       ├── explain_node.py
│       ├── step_advancer.py
│       ├── teach_node.py
│       ├── synthesis_assessor.py
│       ├── clinical_question_node.py
│       ├── mastery_choice_classifier.py
│       ├── topic_choice_node.py
│       └── topic_choice_classifier.py
├── retrieval/
│   ├── crag.py                  # Corrective RAG — single retrieval entry point
│   ├── ot_synonyms.py           # OT term expansion
│   └── turn_aware.py            # Concept-anchored turn query builder
├── ingest/
│   ├── parse_pdf.py             # OpenStax PDF → sections
│   ├── late_chunker.py          # v3 contextual late chunking
│   ├── vector_store.py          # ChromaDB abstraction
│   ├── reranker.py              # Cross-encoder reranking
│   ├── metadata_builder.py      # Diagram links + concept tags
│   └── run_ingest_pipeline.py   # Full ingest runner
├── prompts/                     # All LLM prompts as .txt files
├── evaluation/                  # Faithfulness, Socratic purity, classifier accuracy
├── frontend/
│   └── app.py                   # Streamlit UI
└── docs/
    ├── JOURNAL.md               # Decision log for every significant change
    └── milestone2_report.md     # ACL-format project report
```

---

## Setup

```bash
# 1. Clone and install
git clone https://github.com/deepanathanrajendiran-hub/socratic-ot-tutor.git
cd socratic-ot-tutor
pip install -r requirements.txt

# 2. Copy and fill in your API keys
cp .env.example .env
# Edit .env: add ANTHROPIC_API_KEY and OPENAI_API_KEY

# 3. Start Ollama and pull the embedding model
ollama pull nomic-embed-text

# 4. Run the ingest pipeline (requires OpenStax PDF in data/raw/textbooks/)
PYTHONPATH=. python3 ingest/run_ingest_pipeline.py

# 5. Launch the Streamlit UI
PYTHONPATH=. streamlit run frontend/app.py
```

---

## Absolute Rules (from CLAUDE.md)

1. Turn constraint lives in Python edges, **never** in a system prompt
2. Use LangGraph — never LangChain agents
3. No model names hardcoded anywhere except `config.py`
4. No OT-specific logic hardcoded in retrieval or routing
5. The Dean node runs on **every** teacher response before delivery
6. "I don't know" from a student before turn 2 gets **more scaffolding**, not the answer
7. Every node has **one job** — no node does retrieval AND generation AND evaluation
8. All prompts live in `prompts/` as `.txt` files, never inline in code

---

## Evaluation Results (Milestone 2)

| Metric | Score | Target |
|--------|-------|--------|
| Socratic Purity (full system) | 100% reveal-free | ✓ |
| Retrieval Top-1 Relevance | 80% | — |
| Faithfulness (8 scenarios) | 1.00 raw / 0.95 Dean | ✓ 0.85 |
| Faithfulness (16 scenarios) | 0.93 raw / 0.93 Dean | ✓ 0.85 |
| Classifier Accuracy | 75% | — |
| Dean Gate Pass Rate | 70% natural | — |

---

## Authors

Deepanat Hanrajendiran — University at Buffalo, CSE 635
