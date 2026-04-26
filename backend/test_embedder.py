"""
test_embedder.py — verify the transformers backend produces sane embeddings.

Skips on first invocation if the model isn't already in the HF cache to
avoid a 310MB download in CI. Locally, run once to seed the cache.
"""
import os
import math


def _model_cached() -> bool:
    """Best-effort check: is nomic-embed-text-v1.5 in the HF cache?"""
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    needle = "nomic-embed-text-v1.5"
    if not os.path.isdir(home):
        return False
    for root, _, files in os.walk(home):
        if needle in root:
            return True
    return False


def test_embed_query_returns_768d_vector():
    if not _model_cached():
        print("  [skip] nomic model not in HF cache (run a CRAG query first to seed)")
        return
    from retrieval.embedder import embed_query
    vec = embed_query("ulnar nerve")
    assert isinstance(vec, list)
    assert len(vec) == 768, f"Expected 768-d, got {len(vec)}"
    assert all(isinstance(x, float) for x in vec)


def test_embed_query_is_deterministic():
    if not _model_cached():
        print("  [skip] nomic model not in HF cache")
        return
    from retrieval.embedder import embed_query
    a = embed_query("ulnar nerve")
    b = embed_query("ulnar nerve")
    # Cosine of identical inputs should be ~1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    cos = dot / (na * nb + 1e-9)
    assert cos > 0.9999, f"Same text → same embedding; cos={cos}"


def test_embed_query_distinguishes_text():
    if not _model_cached():
        print("  [skip] nomic model not in HF cache")
        return
    from retrieval.embedder import embed_query
    a = embed_query("ulnar nerve")
    b = embed_query("median nerve")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    cos = dot / (na * nb + 1e-9)
    assert cos < 0.999, "Different terms should yield different embeddings"


def test_unknown_backend_raises():
    """Misconfigured EMBED_BACKEND must fail loudly, not silently embed garbage."""
    from retrieval import embedder
    import config
    original = config.EMBED_BACKEND
    try:
        config.EMBED_BACKEND = "fakebackend"
        try:
            embedder.embed_query("test")
            assert False, "Expected ValueError for unknown backend"
        except ValueError as e:
            assert "fakebackend" in str(e)
    finally:
        config.EMBED_BACKEND = original


if __name__ == "__main__":
    test_unknown_backend_raises()
    test_embed_query_returns_768d_vector()
    test_embed_query_is_deterministic()
    test_embed_query_distinguishes_text()
    print("PASS: 4 tests (some may have skipped if model not cached)")
