"""
test_faithfulness_aggregation.py — pin the two faithfulness aggregates.

A careful reviewer needs to see both:
  (a) claims_only_score   — sum(supported)/sum(claims), excludes zero-claim by 0/0
  (b) all_responses_score — per-response average; zero-claim scores 1.0 → inflates

Reporting only one is misleading; reporting both is honest.
"""
from evaluation.faithfulness import aggregate_results


def test_zero_claim_excluded_from_claims_only():
    rows = [
        {"total_claims": 0, "supported_claims": 0, "score": 1.0},
        {"total_claims": 2, "supported_claims": 2, "score": 1.0},
        {"total_claims": 2, "supported_claims": 1, "score": 0.5},
    ]
    out = aggregate_results(rows)
    assert out["zero_claim_count"] == 1
    assert out["claims_response_count"] == 2
    # claims_only: (2 + 1) / (2 + 2) = 0.75
    assert abs(out["claims_only_score"] - 0.75) < 0.01
    # all_responses: (1.0 + 1.0 + 0.5) / 3 ≈ 0.833 — INFLATED
    assert abs(out["all_responses_score"] - 0.833) < 0.01
    # claims-only must be <= all-responses when zero-claim responses score 1.0
    assert out["claims_only_score"] < out["all_responses_score"]


def test_no_zero_claim_aggregates_match():
    rows = [
        {"total_claims": 4, "supported_claims": 4, "score": 1.0},
        {"total_claims": 4, "supported_claims": 2, "score": 0.5},
    ]
    out = aggregate_results(rows)
    assert out["zero_claim_count"] == 0
    # both metrics: (4+2)/(4+4) = 0.75   AND  (1.0+0.5)/2 = 0.75
    assert abs(out["claims_only_score"] - 0.75) < 0.01
    assert abs(out["all_responses_score"] - 0.75) < 0.01


def test_all_zero_claim_collapses_safely():
    rows = [
        {"total_claims": 0, "supported_claims": 0, "score": 1.0},
        {"total_claims": 0, "supported_claims": 0, "score": 1.0},
    ]
    out = aggregate_results(rows)
    assert out["zero_claim_count"] == 2
    assert out["claims_response_count"] == 0
    # No claim responses → claims_only is 0.0 (defensible default for "undefined")
    assert out["claims_only_score"] == 0.0
    assert out["all_responses_score"] == 1.0


def test_parse_errors_excluded():
    """Responses with score=None (parse error) must not be counted at all."""
    rows = [
        {"total_claims": 2, "supported_claims": 2, "score": 1.0},
        {"total_claims": None, "supported_claims": None, "score": None},
        {"total_claims": 2, "supported_claims": 1, "score": 0.5},
    ]
    out = aggregate_results(rows)
    assert out["n_total"] == 2  # parse error excluded
    assert out["claims_response_count"] == 2
    assert abs(out["claims_only_score"] - 0.75) < 0.01


def test_alternate_key_names():
    """Helper accepts both the live (total_count/supported_count) and the
    canonical (total_claims/supported_claims) field names so the existing
    saved JSON shape works without reformatting."""
    rows = [
        {"total_count": 4, "supported_count": 3, "faithfulness_score": 0.75},
        {"total_count": 0, "supported_count": 0, "faithfulness_score": 1.0},
    ]
    out = aggregate_results(rows)
    assert out["zero_claim_count"] == 1
    assert abs(out["claims_only_score"] - 0.75) < 0.01


def test_empty_input():
    out = aggregate_results([])
    assert out["claims_only_score"] == 0.0
    assert out["all_responses_score"] == 0.0
    assert out["n_total"] == 0


if __name__ == "__main__":
    test_zero_claim_excluded_from_claims_only()
    test_no_zero_claim_aggregates_match()
    test_all_zero_claim_collapses_safely()
    test_parse_errors_excluded()
    test_alternate_key_names()
    test_empty_input()
    print("PASS: all 6 tests")
