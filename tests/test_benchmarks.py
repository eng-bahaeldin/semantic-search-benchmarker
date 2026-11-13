import pytest
from src.benchmarker import compute_hit_rate_at_k, compute_mrr_at_k

def test_metric_math_assertions() -> None:
    ranking = [False, True, False, True, False]
    assert compute_hit_rate_at_k(ranking, k=3) == 1.0
    assert compute_mrr_at_k(ranking, k=5) == pytest.approx(0.5)

    missed = [False, False, False]
    assert compute_hit_rate_at_k(missed, k=3) == 0.0
    assert compute_mrr_at_k(missed, k=3) == 0.0
