"""
retrieval/embedder.py — backend-agnostic query embedder.

Used by retrieval.crag._embed_query so the live retrieval pipeline runs
identically in:
  * dev (laptop):    EMBED_BACKEND=ollama   → fast, requires ollama service
  * production:      EMBED_BACKEND=transformers → no extra service, ships in
                                                  the Cloud Run container

Both backends produce 768-d vectors with the same nomic asymmetric prefix
("search_query: ") so the index built by ingest/late_chunker.py remains valid
across the swap.
"""
from typing import List

import config


# ── transformers backend (lazy-init module-level singletons) ─────────────────

_tokenizer = None
_model = None
_device = None


def _load_transformers():
    """Lazy-load nomic-embed-text-v1.5 once per process. ~310MB download on
    first call; subsequent calls are no-ops. Falls back MPS → CUDA → CPU.
    """
    global _tokenizer, _model, _device
    if _model is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModel
    name = config.EMBED_TRANSFORMERS_MODEL
    _tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    _model = AutoModel.from_pretrained(name, trust_remote_code=True)
    if torch.backends.mps.is_available():
        _device = "mps"
    elif torch.cuda.is_available():
        _device = "cuda"
    else:
        _device = "cpu"
    try:
        _model.to(_device)
    except RuntimeError:
        _device = "cpu"
        _model.to(_device)
    _model.eval()


def _embed_transformers(text: str) -> List[float]:
    import torch
    _load_transformers()
    prefixed = f"search_query: {text}"
    inputs = _tokenizer(prefixed, return_tensors="pt",
                        truncation=True, max_length=512)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        out = _model(**inputs)
    # Attention-masked mean pool over tokens
    last = out.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    summed = (last * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    pooled = (summed / counts).squeeze(0).cpu().tolist()
    return pooled


def _embed_ollama(text: str) -> List[float]:
    import ollama
    resp = ollama.embed(model=config.EMBED_MODEL,
                        input=f"search_query: {text}")
    embeddings = resp.get("embeddings") or [resp.get("embedding")]
    return embeddings[0]


# ── Public ───────────────────────────────────────────────────────────────────

def embed_query(text: str) -> List[float]:
    """Embed a query string into a 768-d vector. Backend chosen by
    config.EMBED_BACKEND.
    """
    backend = config.EMBED_BACKEND
    if backend == "transformers":
        return _embed_transformers(text)
    if backend == "ollama":
        return _embed_ollama(text)
    raise ValueError(
        f"Unknown EMBED_BACKEND={backend!r}; expected 'transformers' or 'ollama'."
    )
