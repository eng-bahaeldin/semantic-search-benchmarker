"""src/benchmarker.py - Information Retrieval Metrics (Hit Rate@K, Recall@K, MRR@K) & Benchmark Harness."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field

from src.chunking import DocumentChunk, FixedSizeChunker, StructureAwareChunker
from src.indexer import BM25SparseIndex, DenseVectorIndex, get_shared_encoder
from src.search import UnifiedSearchEngine


class EvalQuery(BaseModel):
    """Ground-truth benchmark query mapped to semantic sections and anchor spans."""

    query_id: str
    query_type: str
    query: str
    target_doc: str
    target_section: str
    target_spans: List[str] = Field(default_factory=list)


class BenchmarkSummaryRow(BaseModel):
    """Aggregated IR metrics across all evaluation queries for a single retrieval configuration."""

    configuration: str
    chunk_count: int
    hit_rate_at_3: float
    hit_rate_at_5: float
    recall_at_5: float
    mrr_at_5: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float


def compute_hit_rate_at_k(ranked_relevance: Sequence[bool], k: int) -> float:
    """Returns 1.0 if at least one relevant item appears within the top-K ranks, else 0.0."""
    if k <= 0 or not ranked_relevance:
        return 0.0
    return 1.0 if any(ranked_relevance[:k]) else 0.0


def compute_mrr_at_k(ranked_relevance: Sequence[bool], k: int) -> float:
    """
    Computes Reciprocal Rank (1 / rank) of the FIRST relevant item within top-K.
    Returns 0.0 if no relevant item is retrieved inside top-K (never divides by zero).
    """
    if k <= 0 or not ranked_relevance:
        return 0.0
    for idx, is_rel in enumerate(ranked_relevance[:k], start=1):
        if is_rel:
            return 1.0 / float(idx)
    return 0.0


def compute_recall_at_k(
    retrieved_unit_ids: Sequence[str],
    relevant_unit_ids: Sequence[str],
    k: int,
) -> float:
    """
    Computes Recall@K = |{Relevant Units} ∩ {Top-K Retrieved Units}| / |{Relevant Units}|.
    Deduplicates retrieved units so multiple chunks from the same ground-truth target never exceed 1.0.
    """
    unique_relevant = set(relevant_unit_ids)
    if k <= 0 or not unique_relevant:
        return 0.0
    top_k_hits = set(retrieved_unit_ids[:k]).intersection(unique_relevant)
    return len(top_k_hits) / float(len(unique_relevant))


def chunk_matches_ground_truth(chunk: DocumentChunk, eval_item: EvalQuery) -> bool:
    """
    Leak-free relevance predicate evaluating whether a retrieved chunk captures the target section
    or distinctive ground-truth technical spans inside the expected parent document.
    """
    if chunk.parent_doc != eval_item.target_doc:
        return False

    hierarchy = str(chunk.metadata.get("header_hierarchy", "")).lower()
    content_lower = chunk.content.lower()

    if eval_item.target_section.lower() in hierarchy:
        return True

    if eval_item.target_spans:
        matched_spans = sum(1 for span in eval_item.target_spans if span.lower() in content_lower)
        if matched_spans >= 1:
            return True

    return False


class RetrievalBenchmarker:
    """Benchmarks BM25 Baseline against Fixed-256, Fixed-1024, and Structure-Aware Dense Retrieval."""

    def __init__(
        self,
        corpus_dir: str = "corpus",
        eval_path: str = "data/eval_queries.json",
    ) -> None:
        self.corpus_dir = corpus_dir
        self.eval_queries = self._load_eval_queries(eval_path)

    def _load_eval_queries(self, eval_path: str) -> List[EvalQuery]:
        raw_list = json.loads(Path(eval_path).read_text(encoding="utf-8"))
        return [EvalQuery(**item) for item in raw_list]

    def _evaluate_strategy(
        self,
        config_name: str,
        chunk_count: int,
        search_fn: Callable[[str, int], Tuple[List[Any], float]],
    ) -> Tuple[BenchmarkSummaryRow, Dict[str, Dict[str, Any]]]:
        hr3_scores: List[float] = []
        hr5_scores: List[float] = []
        recall5_scores: List[float] = []
        mrr5_scores: List[float] = []
        latencies: List[float] = []
        per_query: Dict[str, Dict[str, Any]] = {}

        for q_item in self.eval_queries:
            results, latency_ms = search_fn(q_item.query, 5)
            latencies.append(latency_ms)

            relevance_flags = [chunk_matches_ground_truth(r.chunk, q_item) for r in results]
            retrieved_units = [
                f"{q_item.target_doc}::{q_item.target_section}" if is_rel else r.chunk.chunk_id
                for r, is_rel in zip(results, relevance_flags)
            ]
            target_unit = [f"{q_item.target_doc}::{q_item.target_section}"]

            hr3 = compute_hit_rate_at_k(relevance_flags, k=3)
            hr5 = compute_hit_rate_at_k(relevance_flags, k=5)
            rec5 = compute_recall_at_k(retrieved_units, target_unit, k=5)
            mrr5 = compute_mrr_at_k(relevance_flags, k=5)

            hr3_scores.append(hr3)
            hr5_scores.append(hr5)
            recall5_scores.append(rec5)
            mrr5_scores.append(mrr5)

            first_hit_rank = next((idx for idx, flag in enumerate(relevance_flags, start=1) if flag), None)
            per_query[q_item.query_id] = {
                "query_id": q_item.query_id,
                "query_type": q_item.query_type,
                "query": q_item.query,
                "target_doc": q_item.target_doc,
                "target_section": q_item.target_section,
                "hit_at_5": hr5 == 1.0,
                "mrr_at_5": round(mrr5, 4),
                "first_hit_rank": first_hit_rank,
                "top_chunk_id": results[0].chunk.chunk_id if results else "NONE",
                "top_score": results[0].score if results else 0.0,
            }

        summary = BenchmarkSummaryRow(
            configuration=config_name,
            chunk_count=chunk_count,
            hit_rate_at_3=round(float(np.mean(hr3_scores)), 4),
            hit_rate_at_5=round(float(np.mean(hr5_scores)), 4),
            recall_at_5=round(float(np.mean(recall5_scores)), 4),
            mrr_at_5=round(float(np.mean(mrr5_scores)), 4),
            avg_latency_ms=round(float(np.mean(latencies)), 2),
            p50_latency_ms=round(float(np.percentile(latencies, 50)), 2),
            p95_latency_ms=round(float(np.percentile(latencies, 95)), 2),
        )
        return summary, per_query

    def run_all_benchmarks(self) -> Tuple[List[BenchmarkSummaryRow], List[Dict[str, Any]]]:
        encoder = get_shared_encoder()

        chunks_256 = FixedSizeChunker(chunk_size=256, chunk_overlap=32).chunk_corpus(self.corpus_dir)
        chunks_1024 = FixedSizeChunker(chunk_size=1024, chunk_overlap=128).chunk_corpus(self.corpus_dir)
        chunks_struct = StructureAwareChunker().chunk_corpus(self.corpus_dir)

        bm25_struct = BM25SparseIndex(chunks_struct)
        dense_256 = DenseVectorIndex(chunks_256, collection_prefix="fixed256", encoder=encoder)
        dense_1024 = DenseVectorIndex(chunks_1024, collection_prefix="fixed1024", encoder=encoder)
        dense_struct = DenseVectorIndex(chunks_struct, collection_prefix="struct", encoder=encoder)

        engine_struct = UnifiedSearchEngine(bm25_index=bm25_struct, dense_index=dense_struct)
        engine_256 = UnifiedSearchEngine(bm25_index=bm25_struct, dense_index=dense_256)
        engine_1024 = UnifiedSearchEngine(bm25_index=bm25_struct, dense_index=dense_1024)

        configs = [
            ("1. BM25 Keyword (Baseline)", len(chunks_struct), engine_struct.search_bm25),
            ("2. Dense Vector (Fixed 256 Tokens)", len(chunks_256), engine_256.search_dense),
            ("3. Dense Vector (Fixed 1024 Tokens)", len(chunks_1024), engine_1024.search_dense),
            ("4. Dense Vector (Structure-Aware)", len(chunks_struct), engine_struct.search_dense),
        ]

        summaries: List[BenchmarkSummaryRow] = []
        per_config_details: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for label, c_count, fn in configs:
            row, details = self._evaluate_strategy(label, c_count, fn)
            summaries.append(row)
            per_config_details[label] = details

        bm25_map = per_config_details["1. BM25 Keyword (Baseline)"]
        dense_map = per_config_details["4. Dense Vector (Structure-Aware)"]

        failure_cases: List[Dict[str, Any]] = []
        for q_item in self.eval_queries:
            qid = q_item.query_id
            b_res = bm25_map[qid]
            d_res = dense_map[qid]
            if b_res["mrr_at_5"] != d_res["mrr_at_5"]:
                winner = "BM25 Keyword" if b_res["mrr_at_5"] > d_res["mrr_at_5"] else "Dense Vector (Structure-Aware)"
                reason = (
                    "Exact identifier / parameter token match boosted IDF term weight in BM25 above dense semantic averaging."
                    if winner == "BM25 Keyword"
                    else "Paraphrased conceptual phrasing lacked exact keyword overlap; bi-encoder cosine similarity captured semantic intent."
                )
                failure_cases.append(
                    {
                        "Query ID": qid,
                        "Type": q_item.query_type.upper(),
                        "Winner": winner,
                        "Query": q_item.query,
                        "BM25 Rank": b_res["first_hit_rank"] or "MISS",
                        "Dense Rank": d_res["first_hit_rank"] or "MISS",
                        "Diagnostic Explanation": reason,
                    }
                )

        return summaries, failure_cases
