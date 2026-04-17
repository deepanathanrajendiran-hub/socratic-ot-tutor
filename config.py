import os
from dotenv import load_dotenv

load_dotenv(override=True)

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
CHUNK_OVERLAP    = 50
TOP_K_RETRIEVE   = 10
TOP_K_RERANK     =  3
WEAK_TOPIC_BOOST = 0.2   # added to cosine sim for known weak topics

WEAK_TOPIC_LOGIT_BOOST = 1.0  # added to cross-encoder logit for weak-topic chunks
                              # logit range ≈ -12 to +5; 1.0 ≈ equivalent effect
                              # to WEAK_TOPIC_BOOST = 0.2 on cosine distance

# ── Socratic Rules ────────────────────────────────────────────────────────────
SOCRATIC_TURN_GATE        = 2     # reveal allowed at turn >= this
DEAN_MAX_REVISIONS        = 2     # max Dean revision attempts
MAX_CHITCHAT_TURNS        = 2     # before forced topic transition
MAX_RESPONSE_SENTENCES    = 3     # before the guiding question
QUESTION_BANK_PER_CONCEPT = 5    # pre-generated questions per concept

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
OUT_OF_SCOPE_THRESHOLD   = -8.0  # cross-encoder logit below this → redirect
                                 # logit range ≈ -12 to +5; -8 = clearly off-topic

# ── v3: Evaluation targets ────────────────────────────────────────────────────
FAITHFULNESS_TARGET = 0.85


# ── Token budgets ──────────────────────────────────────────────────────────────
# Max tokens per LLM call — kept here so they can be tuned without touching
# call sites. eval values mirror the live graph node budgets.
TEACHER_MAX_TOKENS      = 400   # teacher / dean generation (Socratic response)
DEAN_MAX_TOKENS         = 300   # dean quality-gate JSON
FAITHFULNESS_MAX_TOKENS = 800   # haiku claim-checking JSON (eval only)
CRAG_EVAL_MAX_TOKENS    = 200   # CRAG evaluator JSON (live pipeline)

# ── Ingest constants ───────────────────────────────────────────────────────────
INGEST_BATCH_SIZE       = 100   # ChromaDB upsert batch size
OT_NEUROLOGY_CHAPTER    = 13    # OpenStax Anatomy chapter number for neurology
