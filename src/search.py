"""src/search.py - Unified Search Interface & Standardized SearchResult Schema."""

import time
from typing import List, Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field

from src.chunking import DocumentChunk
from src.indexer import BM25SparseIndex, DenseVectorIndex, tokenize_for_bm25


class SearchResult(BaseModel):
    """Standardized search result returned by both Sparse (BM25) and Dense retrieval engines."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(..., ge=1)
    chunk: DocumentChunk
    score: float = Field(..., description="BM25 lexical score or Cosine Similarity in [0.0, 1.0]")
    engine: Literal["BM25", "DENSE"]
    matched_terms: List[str] = Field(default_factory=list)


class UnifiedSearchEngine:
    """Decoupled search facade supporting side-by-side BM25 and Dense Vector retrieval."""

    def __init__(
        self,
        bm25_index: BM25SparseIndex,
        dense_index: DenseVectorIndex,
    ) -> None:
        self.bm25_index = bm25_index
        self.dense_index = dense_index

    def _find_matched_terms(self, query: str, chunk: DocumentChunk) -> List[str]:
        q_tokens = set(tokenize_for_bm25(query))
        c_tokens = set(tokenize_for_bm25(f"{chunk.metadata.get('header_hierarchy', '')} {chunk.content}"))
        return sorted(q_tokens.intersection(c_tokens))

    def search_bm25(self, query: str, top_k: int = 5) -> Tuple[List[SearchResult], float]:
        """Executes BM25 keyword retrieval and returns (results, latency_ms)."""
        t0 = time.perf_counter()
        raw_hits = self.bm25_index.search(query, top_k=top_k)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        formatted = [
            SearchResult(
                rank=idx,
                chunk=chunk,
                score=score,
                engine="BM25",
                matched_terms=self._find_matched_terms(query, chunk),
            )
            for idx, (chunk, score) in enumerate(raw_hits, start=1)
        ]
        return formatted, latency_ms

    def search_dense(self, query: str, top_k: int = 5) -> Tuple[List[SearchResult], float]:
        """Executes Dense Vector retrieval and returns (results, latency_ms)."""
        t0 = time.perf_counter()
        raw_hits = self.dense_index.search(query, top_k=top_k)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        formatted = [
            SearchResult(
                rank=idx,
                chunk=chunk,
                score=score,
                engine="DENSE",
                matched_terms=self._find_matched_terms(query, chunk),
            )
            for idx, (chunk, score) in enumerate(raw_hits, start=1)
        ]
        return formatted, latency_ms
