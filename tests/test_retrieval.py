from src.chunking import StructureAwareChunker
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
