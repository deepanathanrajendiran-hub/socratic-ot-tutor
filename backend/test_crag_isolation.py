"""
test_crag_isolation.py — pin CRAG decision branches without hitting the LLM
or ChromaDB. Mocks: _embed_query, _vector_search, _evaluate_retrieval, and
the reranker singleton.

Branches covered:
  1. CORRECT          → straight pass-through
  2. AMBIGUOUS        → triggers refinement; uses refined results when better
  3. INCORRECT        → out_of_scope=True, returns ([], [], log)
  4. parse_failed     → flag propagated through the log
  5. LOW_CONFIDENCE   → rerank score below OUT_OF_SCOPE_THRESHOLD → redirect
"""
from unittest.mock import patch, MagicMock

import config


def _fake_chunks(score: float = 5.0) -> list[dict]:
    """3 fake reranked-shape chunks with a controllable top score."""
    return [
        {"id": "c1", "text": "ulnar nerve text",
         "section_id": "sec1", "section_title": "Peripheral nerves",
         "full_section_text": "Full section text 1.",
         "rerank_score": score},
        {"id": "c2", "text": "elbow",
         "section_id": "sec2", "section_title": "Plexus",
         "full_section_text": "Full section text 2.",
         "rerank_score": score - 1},
        {"id": "c3", "text": "other",
         "section_id": "sec3", "section_title": "Other",
         "full_section_text": "Full section text 3.",
         "rerank_score": score - 2},
    ]


def _patch_pipeline(eval_result, rerank_chunks=None, eval_side_effect=None):
    """Returns a context manager-style list of patches as a tuple of patcher
    objects so each test can compose them. Uses MagicMock for reranker."""
    fake_reranker = MagicMock()
    fake_reranker.rerank_with_logging.return_value = (
        rerank_chunks if rerank_chunks is not None else _fake_chunks(),
        {"rerank_log": "ok"},
    )
    fake_vs = MagicMock()
    fake_vs.get_full_section.return_value = "Full section text"
    return fake_reranker, fake_vs, eval_result, eval_side_effect


# ── 1. CORRECT path ──────────────────────────────────────────────────────────

def test_correct_decision_passes_through():
    from retrieval import crag
    with patch.object(crag, "_embed_query", return_value=[0.1] * 768), \
         patch.object(crag, "_vector_search", return_value=_fake_chunks()), \
         patch.object(crag, "_evaluate_retrieval",
                      return_value={"score": 0.85, "decision": "CORRECT",
                                    "refinement_query": None, "reason": "",
                                    "parse_failed": False}), \
         patch.object(crag, "get_reranker") as gr, \
         patch.object(crag, "get_vs"):
        fake_reranker = MagicMock()
        fake_reranker.rerank_with_logging.return_value = (_fake_chunks(),
                                                          {"rerank_log": "ok"})
        gr.return_value = fake_reranker
        chunks, sections, log = crag.corrective_retrieve(query="ulnar nerve")
        assert log["crag_decision"] == "CORRECT"
        assert log.get("out_of_scope") is False
        assert len(chunks) == 3
        assert len(sections) == 3


# ── 2. AMBIGUOUS triggers refinement, uses better refined result ─────────────

def test_ambiguous_triggers_refinement_and_uses_better_score():
    from retrieval import crag
    eval_calls = MagicMock(side_effect=[
        {"score": 0.5, "decision": "AMBIGUOUS",
         "refinement_query": "ulnar nerve compression elbow",
         "reason": "broad topic", "parse_failed": False},
        {"score": 0.9, "decision": "CORRECT",
         "refinement_query": None, "reason": "", "parse_failed": False},
    ])
    with patch.object(crag, "_embed_query", return_value=[0.1] * 768), \
         patch.object(crag, "_vector_search", return_value=_fake_chunks()), \
         patch.object(crag, "_evaluate_retrieval", eval_calls), \
         patch.object(crag, "get_reranker") as gr, \
         patch.object(crag, "get_vs"):
        fake_reranker = MagicMock()
        fake_reranker.rerank_with_logging.return_value = (_fake_chunks(),
                                                          {"rerank_log": "ok"})
        gr.return_value = fake_reranker
        _, _, log = crag.corrective_retrieve(query="ulnar nerve")
        assert eval_calls.call_count == 2, "refinement should re-evaluate"
        assert "AMBIGUOUS→REFINED" in log["crag_decision"]
        assert log["refined"] is True


# ── 3. INCORRECT → out_of_scope, no chunks ───────────────────────────────────

def test_incorrect_returns_empty_with_redirect_signal():
    from retrieval import crag
    with patch.object(crag, "_embed_query", return_value=[0.1] * 768), \
         patch.object(crag, "_vector_search", return_value=_fake_chunks()), \
         patch.object(crag, "_evaluate_retrieval",
                      return_value={"score": 0.1, "decision": "INCORRECT",
                                    "refinement_query": None, "reason": "off-topic",
                                    "parse_failed": False}):
        chunks, sections, log = crag.corrective_retrieve(
            query="best pizza in Buffalo")
        assert chunks == []
        assert sections == []
        assert log.get("out_of_scope") is True
        assert log["crag_decision"] == "INCORRECT"


# ── 4. parse_failed flag propagates through to the log ───────────────────────

def test_parse_failure_flag_surfaces_in_log():
    from retrieval import crag
    with patch.object(crag, "_embed_query", return_value=[0.1] * 768), \
         patch.object(crag, "_vector_search", return_value=_fake_chunks()), \
         patch.object(crag, "_evaluate_retrieval",
                      return_value={"score": 0.85, "decision": "CORRECT",
                                    "refinement_query": None, "reason": "",
                                    "parse_failed": True}), \
         patch.object(crag, "get_reranker") as gr, \
         patch.object(crag, "get_vs"):
        fake_reranker = MagicMock()
        fake_reranker.rerank_with_logging.return_value = (_fake_chunks(),
                                                          {"rerank_log": "ok"})
        gr.return_value = fake_reranker
        _, _, log = crag.corrective_retrieve(query="ulnar nerve")
        assert log.get("parse_failed") is True


# ── 5. LOW_CONFIDENCE: rerank score below threshold → redirect ───────────────

def test_low_rerank_score_triggers_redirect():
    from retrieval import crag
    far_below = config.OUT_OF_SCOPE_THRESHOLD - 5.0
    bad_chunks = _fake_chunks(score=far_below)
    with patch.object(crag, "_embed_query", return_value=[0.1] * 768), \
         patch.object(crag, "_vector_search", return_value=_fake_chunks()), \
         patch.object(crag, "_evaluate_retrieval",
                      return_value={"score": 0.85, "decision": "CORRECT",
                                    "refinement_query": None, "reason": "",
                                    "parse_failed": False}), \
         patch.object(crag, "get_reranker") as gr, \
         patch.object(crag, "get_vs"):
        fake_reranker = MagicMock()
        fake_reranker.rerank_with_logging.return_value = (bad_chunks,
                                                          {"rerank_log": "low"})
        gr.return_value = fake_reranker
        chunks, sections, log = crag.corrective_retrieve(query="ulnar nerve")
        assert chunks == []
        assert sections == []
        assert log.get("out_of_scope") is True
        assert log["crag_decision"] == "LOW_CONFIDENCE"


def test_empty_rerank_result_triggers_redirect():
    from retrieval import crag
    with patch.object(crag, "_embed_query", return_value=[0.1] * 768), \
         patch.object(crag, "_vector_search", return_value=_fake_chunks()), \
         patch.object(crag, "_evaluate_retrieval",
                      return_value={"score": 0.85, "decision": "CORRECT",
                                    "refinement_query": None, "reason": "",
                                    "parse_failed": False}), \
         patch.object(crag, "get_reranker") as gr, \
         patch.object(crag, "get_vs"):
        fake_reranker = MagicMock()
        fake_reranker.rerank_with_logging.return_value = ([], {"rerank_log": "empty"})
        gr.return_value = fake_reranker
        chunks, _, log = crag.corrective_retrieve(query="ulnar nerve")
        assert chunks == []
        assert log["crag_decision"] == "LOW_CONFIDENCE"


if __name__ == "__main__":
    test_correct_decision_passes_through()
    test_ambiguous_triggers_refinement_and_uses_better_score()
    test_incorrect_returns_empty_with_redirect_signal()
    test_parse_failure_flag_surfaces_in_log()
    test_low_rerank_score_triggers_redirect()
    test_empty_rerank_result_triggers_redirect()
    print("PASS: all 6 tests")
