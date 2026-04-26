# Socratic-OT Cloud Run image
#
# Build:   docker build -t socratic-ot:dev .
# Run:     docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... socratic-ot:dev
# Deploy:  gcloud run deploy socratic-ot --source .
#
# IMPORTANT: pre-downloads cross-encoder + nomic-embed-text-v1.5 weights into
# the image so the first request after a cold start does not pay a 15s model
# download penalty (which the frontend would interpret as a hung request).

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_OFFLINE=0 \
    EMBED_BACKEND=transformers \
    LLM_PROVIDER=anthropic

WORKDIR /app

# System deps for tokenizers + chromadb sqlite + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

# Python deps first so the image cache is reused across code-only changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download both retrieval models so the first /chat request does not
# pay the cold-start tax. Without this, Cloud Run's frontend timeout
# (~30s before the user sees a hang) fires before the model arrives.
RUN python -c "from sentence_transformers import CrossEncoder; \
               CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
               AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5', \
                                             trust_remote_code=True); \
               AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', \
                                          trust_remote_code=True)"

# Copy source — last so code changes don't bust the dep + model layers
COPY . /app

# Cloud Run injects $PORT (8080 by default); fall back for local docker run
ENV PORT=8000
EXPOSE 8000

# uvicorn workers=1 because LangGraph state lives in-process; SqliteSaver
# is process-shared and check_same_thread=False handles uvicorn's thread pool.
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1
