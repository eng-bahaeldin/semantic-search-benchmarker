from pathlib import Path

Path("tests").mkdir(parents=True, exist_ok=True)

Path("tests/test_retrieval.py").write_text("""from src.chunking import StructureAwareChunker
from src.indexer import BM25SparseIndex, DenseVectorIndex
from src.search import UnifiedSearchEngine

def test_exact_keyword_retrieval_bm25() -> None:
    chunks = StructureAwareChunker().chunk_corpus("corpus")
    engine = UnifiedSearchEngine(
        bm25_index=BM25SparseIndex(chunks),
        dense_index=DenseVectorIndex(chunks, collection_prefix="test_exact"),
    )
    bm25_results, _ = engine.search_bm25("maxUnavailable", top_k=3)
    assert len(bm25_results) >= 1
    assert bm25_results[0].rank == 1
""", encoding="utf-8")

Path("tests/test_benchmarks.py").write_text("""import pytest
from src.benchmarker import compute_hit_rate_at_k, compute_mrr_at_k

def test_metric_math_assertions() -> None:
    ranking = [False, True, False, True, False]
    assert compute_hit_rate_at_k(ranking, k=3) == 1.0
    assert compute_mrr_at_k(ranking, k=5) == pytest.approx(0.5)

    missed = [False, False, False]
    assert compute_hit_rate_at_k(missed, k=3) == 0.0
    assert compute_mrr_at_k(missed, k=3) == 0.0
""", encoding="utf-8")

print("SUCCESS: All test files written successfully!")
