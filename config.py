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

# ── LLM provider (anthropic | bedrock) ────────────────────────────────────────
# Switches between the direct Anthropic API and AWS Bedrock-hosted Claude.
# Bedrock requires `pip install "anthropic[bedrock]"` and AWS credentials.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()

# ── Models ────────────────────────────────────────────────────────────────────
# Model IDs differ between providers; PRIMARY_MODEL / FAST_MODEL resolve to
# the right ID for the active provider. If the env var is set but in the
# wrong format for the active provider, fall back to the provider default —
# this prevents stale .env values from breaking a provider switch.
_BEDROCK_DEFAULTS = {
    # us.* inference-profile IDs — required for on-demand invocation of Claude
    # 4.x on Bedrock in US regions. EU/APAC users override via PRIMARY_MODEL /
    # FAST_MODEL env vars (e.g. eu.anthropic.claude-sonnet-4-5-...).
    "PRIMARY_MODEL": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "FAST_MODEL":    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}
_ANTHROPIC_DEFAULTS = {
    "PRIMARY_MODEL": "claude-sonnet-4-5",
    "FAST_MODEL":    "claude-haiku-4-5",
}


def _resolve_model(name: str) -> str:
    val = os.getenv(name)
    defaults = _BEDROCK_DEFAULTS if LLM_PROVIDER == "bedrock" else _ANTHROPIC_DEFAULTS
    if not val:
        return defaults[name]
    # Bedrock IDs begin with "anthropic." (or a region/inference-profile prefix
    # like "us.anthropic."). Direct Anthropic API IDs don't. Reject mismatches.
    looks_like_bedrock = val.startswith("anthropic.") or ".anthropic." in val
    if LLM_PROVIDER == "bedrock" and not looks_like_bedrock:
        return defaults[name]
    if LLM_PROVIDER == "anthropic" and looks_like_bedrock:
        return defaults[name]
    return val


PRIMARY_MODEL = _resolve_model("PRIMARY_MODEL")
FAST_MODEL    = _resolve_model("FAST_MODEL")
VISION_MODEL  = os.getenv("VISION_MODEL",  "gpt-4o")
EMBED_MODEL   = os.getenv("EMBED_MODEL",   "nomic-embed-text")

# ── Embedding backend (transformers | ollama) ─────────────────────────────────
# Cloud Run has no ollama service; "transformers" runs nomic-embed-text-v1.5
# locally via the same library used by ingest/late_chunker.py. Dev-time can
# still use ollama for speed.
EMBED_BACKEND      = os.getenv("EMBED_BACKEND", "transformers").lower()
EMBED_TRANSFORMERS_MODEL = os.getenv("EMBED_TRANSFORMERS_MODEL",
                                      "nomic-ai/nomic-embed-text-v1.5")

# ── API Keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION are read by boto3
# directly when LLM_PROVIDER=bedrock — no need to surface them here.

# ── RAG Settings ──────────────────────────────────────────────────────────────
CHUNK_SIZES = {
    "large":  800,   # passed to LLM for generation
    "medium": 300,   # primary retrieval unit
    "small":  60,    # anchor index only
}
CHUNK_OVERLAP    = 50
TOP_K_RETRIEVE   = 15    # widened from 10 to compensate for dropping cosine-stage
                         # weak-topic boost (R-C6); reranker still picks top 3
TOP_K_RERANK     =  3
WEAK_TOPIC_BOOST = 0.2   # legacy — only used if cosine-stage boost is reintroduced

WEAK_TOPIC_LOGIT_BOOST = 1.0  # added to cross-encoder logit for weak-topic chunks
                              # logit range ≈ -12 to +5; 1.0 ≈ equivalent effect
                              # to WEAK_TOPIC_BOOST = 0.2 on cosine distance

# ── Socratic Rules ────────────────────────────────────────────────────────────
SOCRATIC_TURN_GATE        = 2     # reveal allowed at turn >= this
IDK_REVEAL_THRESHOLD      = 3     # consecutive 'idk' classifications before teach_node fires
                                  # counter resets to 0 on any non-idk classification
DEAN_MAX_REVISIONS        = 2     # max Dean revision attempts
MAX_CHITCHAT_TURNS        = 2     # before forced topic transition
MAX_RESPONSE_SENTENCES    = 3     # before the guiding question
QUESTION_BANK_PER_CONCEPT = 5    # pre-generated questions per concept

# ── Paths ─────────────────────────────────────────────────────────────────────
# Anchor every project path to this file's directory so the code works
# regardless of cwd. Phase 5 / Cloud Run will run from /app and may chdir;
# relative paths fail silently in that environment.
_BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(_BASE_DIR, "data")
RAW_DIR           = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR     = os.path.join(DATA_DIR, "processed")
CHROMA_DIR        = os.path.join(PROCESSED_DIR, "chroma_db")
CHUNKS_DIR        = os.path.join(PROCESSED_DIR, "chunks")
QUESTION_BANK_DIR = os.path.join(PROCESSED_DIR, "question_bank")
PROMPTS_DIR       = os.path.join(_BASE_DIR, "prompts")
DIAGRAMS_DIR      = os.path.join(RAW_DIR, "diagrams")
SESSIONS_DB_PATH  = os.getenv("SESSIONS_DB_PATH", os.path.join(DATA_DIR, "sessions.db"))

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
TEACHER_MAX_TOKENS         = int(os.getenv("TEACHER_MAX_TOKENS",         "600"))
DEAN_MAX_TOKENS            = int(os.getenv("DEAN_MAX_TOKENS",            "300"))
CLASSIFIER_MAX_TOKENS      = int(os.getenv("CLASSIFIER_MAX_TOKENS",      "10"))
MANAGER_MAX_TOKENS         = int(os.getenv("MANAGER_MAX_TOKENS",         "150"))
EXPLAIN_MAX_TOKENS         = int(os.getenv("EXPLAIN_MAX_TOKENS",         "500"))
HINT_MAX_TOKENS            = int(os.getenv("HINT_MAX_TOKENS",            "500"))
TEACH_MAX_TOKENS           = int(os.getenv("TEACH_MAX_TOKENS",           "500"))
SYNTHESIS_MAX_TOKENS       = int(os.getenv("SYNTHESIS_MAX_TOKENS",       "400"))
CLINICAL_MAX_TOKENS        = int(os.getenv("CLINICAL_MAX_TOKENS",        "400"))
REDIRECT_MAX_TOKENS        = int(os.getenv("REDIRECT_MAX_TOKENS",        "300"))
STEP_ADVANCER_MAX_TOKENS   = int(os.getenv("STEP_ADVANCER_MAX_TOKENS",   "300"))
TOPIC_CHOICE_MAX_TOKENS    = int(os.getenv("TOPIC_CHOICE_MAX_TOKENS",    "300"))
FAITHFULNESS_MAX_TOKENS    = int(os.getenv("FAITHFULNESS_MAX_TOKENS",    "800"))   # eval
CRAG_EVAL_MAX_TOKENS       = int(os.getenv("CRAG_EVAL_MAX_TOKENS",       "200"))

# ── Ingest constants ───────────────────────────────────────────────────────────
INGEST_BATCH_SIZE       = 100   # ChromaDB upsert batch size
OT_NEUROLOGY_CHAPTER    = 13    # OpenStax Anatomy chapter number for neurology
