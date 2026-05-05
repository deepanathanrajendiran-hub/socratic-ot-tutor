import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Domain (swap this for generalizability demo) ──────────────────────────────
DOMAIN = os.getenv("DOMAIN", "OT_anatomy")
# Options: "OT_anatomy" | "physics"

DOMAIN_CONFIG = {
    "OT_anatomy": {
        # Case-sensitive — must match the literal collection name in
        # chroma.sqlite3 (capital "OT" reflects how the ingest pipeline
        # named it; lowercase would be a different collection).
        "collection_name": "OT_anatomy_chunks",
        "system_context": "Occupational Therapy anatomy and neuroscience education",
        "target_exam": "NBCOT certification",
        "textbook": "OpenStax Anatomy and Physiology 2e",
        # Per-domain prompt slots used by the manager_agent and rapport
        # prompts. They make those prompts subject-agnostic so the same
        # graph can tutor anatomy or physics by swapping config + chunks.
        "subject_noun":     "anatomy",
        "example_concepts": "synapse, ulnar nerve, cerebellum, action potential, reflex arc, gray matter, anterior horn, spinothalamic tract, motor neuron, carpal tunnel, rotator cuff, brachial plexus, median nerve, radial nerve, sciatic nerve, peroneal nerve, hippocampus, hypothalamus, basal ganglia, corpus callosum, motor cortex, sensory cortex, cranial nerve, vertebra, ligament, tendon, muscle fiber, joint capsule",
        "reject_examples":  "\"brain anatomy\", \"nerves\", \"the nervous system\", \"the body\", \"muscles in general\"",
        "rapport_examples": "a nerve, a structure, a pathway, a joint, a brain region",
        # Concept-name parts that are also generic anatomical vocabulary
        # (Dean PASSes them on their own). Concept-leak detection skips
        # individual matching on these so "ulnar nerve" doesn't mark every
        # mention of "nerve" as a leak.
        "generic_words": {
            "nerve", "nerves", "system", "tract", "cord", "horn", "arc", "loop",
            "fiber", "fibers", "fibre", "fibres",
            # "neuron" / "neurons" / "neural" are basic vocabulary in the
            # OT anatomy domain — many concepts contain them (motor neuron,
            # sensory neuron, interneuron, etc.). Treat them as generic so
            # that "Which neuron sends signals to muscles?" doesn't get
            # mis-classified as the student naming "motor neuron" upfront.
            "neuron", "neurons", "neural",
            "lateral", "medial", "anterior", "posterior",
            "proximal", "distal", "superior", "inferior",
            "deep", "superficial",
            # Multi-context anatomy nouns. "matter" appears in both gray
            # matter and white matter; "tissue", "region", "substance"
            # likewise appear across many concepts. The full phrase is the
            # discriminator — flagging the lone word fires false positives
            # (e.g. concept="gray matter", draft mentioning "white matter"
            # gets marked as a leak when it isn't).
            "matter", "tissue", "region", "substance",
        },
        # Stems that match too many unrelated English words ("spin" stems
        # from "spinal" but also "spinach"/"spinning"). Concept-leak detection
        # falls back to exact-phrase match for words landing on these stems.
        "stem_blacklist": {
            "spin", "head", "hand", "foot", "side", "moto", "memo", "info",
            "data", "form", "kind", "type", "make", "back", "body", "mind",
            # "medi" stems from "median" but ALSO matches "medical",
            # "medication", "medicine", "mediator", "medial" — refusing to
            # give medical advice or mentioning the medial side of an arm
            # would otherwise false-positive a "median nerve" leak. The
            # full-phrase "median nerve" is still caught at the top.
            "medi",
        },
    },
    "physics": {
        "collection_name": "physics_chunks",
        "system_context": "University physics education",
        "target_exam": "physics midterm",
        "textbook": "OpenStax University Physics Volume 1",
        "subject_noun":     "physics",
        "example_concepts": "Newton's second law, kinetic energy, conservation of momentum, projectile motion, angular velocity, work-energy theorem, friction, simple harmonic motion, electric field, magnetic flux",
        "reject_examples":  "\"mechanics\", \"motion\", \"energy in general\", \"forces\", \"the laws of physics\"",
        "rapport_examples": "a law, a quantity, a phenomenon, a process, an equation",
        # Generic physics vocabulary — appears in concept names but doesn't
        # uniquely identify the concept ("Newton's second law" has "law" as
        # the generic word; "kinetic energy" has "energy"; "magnetic field"
        # has "field").
        "generic_words": {
            "law", "laws", "force", "forces", "energy", "field", "fields",
            "motion", "wave", "waves", "particle", "particles", "system",
            "principle", "principles", "theorem", "constant", "equation",
            "vector", "scalar",
        },
        # English-noise stems shared with other domains — physics adds none
        # of its own beyond the universal-noise set, since technical-term
        # stems like "fiel" / "syst" are already filtered by generic_words.
        "stem_blacklist": {
            "data", "form", "kind", "type", "make", "back", "body", "mind",
        },
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
_VERTEX_DEFAULTS = {
    # Vertex AI uses @date suffix instead of -date. Region availability varies;
    # us-east5 has the broadest Anthropic model coverage as of 2026-05.
    "PRIMARY_MODEL": "claude-sonnet-4-5@20250929",
    "FAST_MODEL":    "claude-haiku-4-5@20251001",
}


def _resolve_model(name: str) -> str:
    val = os.getenv(name)
    if LLM_PROVIDER == "bedrock":
        defaults = _BEDROCK_DEFAULTS
    elif LLM_PROVIDER == "vertex":
        defaults = _VERTEX_DEFAULTS
    else:
        defaults = _ANTHROPIC_DEFAULTS
    if not val:
        return defaults[name]
    # Format guards — reject env values in the wrong format for the active
    # provider so a stale .env doesn't silently break a provider switch.
    looks_like_bedrock = val.startswith("anthropic.") or ".anthropic." in val
    looks_like_vertex  = "@" in val
    if LLM_PROVIDER == "bedrock" and not looks_like_bedrock:
        return defaults[name]
    if LLM_PROVIDER == "vertex" and not looks_like_vertex:
        return defaults[name]
    if LLM_PROVIDER == "anthropic" and (looks_like_bedrock or looks_like_vertex):
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
SOCRATIC_TURN_GATE        = 3     # reveal allowed at turn >= this — i.e.
                                  # student gets 3 wrong attempts past the
                                  # opener before teach_node fires. Matches
                                  # IDK_REVEAL_THRESHOLD so both ladders end
                                  # at the same depth (3 strikes).
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
# Override via env so a single image can serve different hosts/ports without
# code changes. Cloud Run injects PORT; honor it if set.
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))

# Comma-separated allowlist of HTTP origins for CORS. Used by api/main.py.
# Default covers local dev + the prod Vercel hostname; override in prod via
# env so frontend redeploys don't require backend code changes.
_DEFAULT_CORS = "http://localhost:3000,https://socratic-ot.vercel.app"
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", _DEFAULT_CORS).split(",") if o.strip()]
# Optional regex for preview deployments (e.g. socratic-ot-<hash>.vercel.app).
CORS_ORIGIN_REGEX = os.getenv("CORS_ORIGIN_REGEX",
                              r"https://socratic-ot[a-z0-9-]*\.vercel\.app")

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

# Skip CRAG when the locked concept hasn't changed since last retrieval.
# Saves 2-7 s on follow-up turns within the same Socratic loop. Set
# RETRIEVAL_CACHE_DISABLE=1 in the env to force fresh retrieval every
# turn (useful when debugging retrieval drift or evaluating turn-aware
# query facets).
RETRIEVAL_CACHE_DISABLE = bool(int(os.getenv("RETRIEVAL_CACHE_DISABLE", "0")))

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
# 800 tokens — the function-mode reveal emits <thinking> block + 3 content
# blocks (explanation, everyday example, OT connection) + A/B/C/D menu.
# 500 truncated mid-sentence before the menu, leaving students without their
# next-step choices. (2026-05-03 fix surfaced via FN2/FN10 mastery_choice_menu
# regression.)
TEACH_MAX_TOKENS           = int(os.getenv("TEACH_MAX_TOKENS",           "800"))
SYNTHESIS_MAX_TOKENS       = int(os.getenv("SYNTHESIS_MAX_TOKENS",       "400"))
CLINICAL_MAX_TOKENS        = int(os.getenv("CLINICAL_MAX_TOKENS",        "400"))
RAPPORT_MAX_TOKENS         = int(os.getenv("RAPPORT_MAX_TOKENS",         "256"))
VLM_MAX_TOKENS             = int(os.getenv("VLM_MAX_TOKENS",             "400"))
REDIRECT_MAX_TOKENS        = int(os.getenv("REDIRECT_MAX_TOKENS",        "300"))
STEP_ADVANCER_MAX_TOKENS   = int(os.getenv("STEP_ADVANCER_MAX_TOKENS",   "300"))
TOPIC_CHOICE_MAX_TOKENS    = int(os.getenv("TOPIC_CHOICE_MAX_TOKENS",    "300"))
FAITHFULNESS_MAX_TOKENS    = int(os.getenv("FAITHFULNESS_MAX_TOKENS",    "800"))   # eval
CRAG_EVAL_MAX_TOKENS       = int(os.getenv("CRAG_EVAL_MAX_TOKENS",       "200"))

# ── Ingest constants ───────────────────────────────────────────────────────────
INGEST_BATCH_SIZE       = 100   # ChromaDB upsert batch size
OT_NEUROLOGY_CHAPTER    = 13    # OpenStax Anatomy chapter number for neurology


# ── Cross-session memory layer (optional) ────────────────────────────────────
# SqliteSaver always handles per-session conversation state. When
# MEMORY_BACKEND=mem0, an additional cross-session memory layer runs
# alongside it: extracts facts from each turn and lets the rapport node
# pull them back on the next session ("last time you worked on...").
#
# Default "sqlite" = conversation state only, no cross-session layer.
# Set to "mem0" + provide MEM0_API_KEY to enable.
MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "sqlite").lower()
MEM0_API_KEY   = os.getenv("MEM0_API_KEY")
# How many relevant memories to inject into the rapport prompt per turn.
MEM0_TOP_K     = int(os.getenv("MEM0_TOP_K", "4"))


# ── Per-node model overrides ──────────────────────────────────────────────────
# Each Sonnet-tier node reads its model via `model_for(node_name)` so a
# deployment can dial individual nodes down to Haiku for cost savings
# without touching the global PRIMARY_MODEL. Useful for A/B tests and for
# the post-demo cost-optimization pass (e.g. Dean → Haiku, see
# evaluation/teacher_model_ab.py).
#
# Recognized env vars (all optional — empty = use PRIMARY_MODEL):
#   TEACHER_MODEL_OVERRIDE     - teacher_socratic
#   DEAN_MODEL_OVERRIDE        - dean_node
#   STUDY_MODEL_OVERRIDE       - study_node
#   CLINICAL_MODEL_OVERRIDE    - clinical_question_node
#   TEACH_MODEL_OVERRIDE       - teach_node (post-mastery reveal+explain)
#   EXPLAIN_MODEL_OVERRIDE     - explain_node
#   HINT_MODEL_OVERRIDE        - hint_error_node
#   REDIRECT_MODEL_OVERRIDE    - redirect_node
#   STEP_ADVANCER_MODEL_OVERRIDE
#   TOPIC_CHOICE_MODEL_OVERRIDE
#   SYNTHESIS_MODEL_OVERRIDE   - synthesis_assessor
#   VLM_MODEL_OVERRIDE         - vlm_node (vision)
def model_for(node: str) -> str:
    """Return the model id for a given node, honoring overrides.

    Pass any node name (lowercase). If an override env var is set and
    valid for the active provider, returns that; otherwise returns
    PRIMARY_MODEL. Same provider-format guard as `_resolve_model`.
    """
    env_var = f"{node.upper()}_MODEL_OVERRIDE"
    val = os.getenv(env_var)
    if not val:
        return PRIMARY_MODEL
    looks_like_bedrock = val.startswith("anthropic.") or ".anthropic." in val
    if LLM_PROVIDER == "bedrock" and not looks_like_bedrock:
        return PRIMARY_MODEL
    if LLM_PROVIDER == "anthropic" and looks_like_bedrock:
        return PRIMARY_MODEL
    return val
