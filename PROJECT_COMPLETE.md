# Semantic Search & IR Benchmarker — Complete Master Project Document


---

## File: `README.md`

```markdown
# semantic-search-benchmarker

```


---

## File: `pyproject.toml`

```toml
[project]
name = "semantic-search-benchmarker"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
authors = [
    { email = "bahaa.ahmed.eng@gmail.com" }
]
requires-python = ">=3.11"
dependencies = [
    "chromadb>=1.5.9",
    "numpy>=2.4.6",
    "openai>=3.50.0",
    "pandas>=3.1.1",
    "plotly>=7.1.0",
    "pydantic>=2.14.3",
    "pytest>=9.2.0",
    "pytest-asyncio>=1.5.0",
    "python-dotenv>=1.2.4",
    "rank-bm25>=0.2.2",
    "sentence-transformers>=6.3.0",
    "streamlit>=1.67.0",
    "tiktoken>=0.14.0",
]

[project.scripts]
semantic-search-benchmarker = "semantic_search_benchmarker:main"

[build-system]
requires = ["uv_build>=0.13.4,<0.14.0"]
build-backend = "uv_build"

```


---

## File: `corpus/kubernetes_api_spec.md`

```markdown
# Kubernetes API

## RollingUpdateDeployment
Default maxUnavailable is 25%.
```


---

## File: `corpus/postgres_connection_pooling.md`

```markdown
# PostgreSQL Pooling

## ECONNRESET
Connection dropped by idle timeout.
```


---

## File: `corpus/fastify_runtime_architecture.md`

```markdown
# Fastify Architecture

## Body Limit
Default bodyLimit is 1MB.
```


---

## File: `corpus/envoy_zero_trust_mesh.md`

```markdown
# Envoy Mesh

## Outlier Detection
Ejects on 5 consecutive 5xx errors.
```


---

## File: `data/eval_queries.json`

```json
[
  {
    "query_id": "Q01",
    "query_type": "lexical",
    "query": "What is the default value of maxUnavailable in RollingUpdateDeployment?",
    "target_doc": "kubernetes_api_spec.md",
    "target_section": "RollingUpdateDeployment",
    "target_spans": [
      "maxUnavailable",
      "25%"
    ]
  },
  {
    "query_id": "Q02",
    "query_type": "lexical",
    "query": "Error: ECONNRESET in pooled connection",
    "target_doc": "postgres_connection_pooling.md",
    "target_section": "ECONNRESET",
    "target_spans": [
      "ECONNRESET"
    ]
  }
]
```


---

## File: `src/chunking.py`

```python
"""src/chunking.py - Fixed-Token and Structure-Aware Markdown Chunking Engines."""

from pathlib import Path
import re
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field
import tiktoken


class DocumentChunk(BaseModel):
    """Standardized chunk representation emitted by all chunking strategies."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(..., min_length=1, description="Deterministic identifier for the chunk")
    content: str = Field(..., min_length=1, description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Enriched metadata including header_hierarchy, line_start, line_end, token_count",
    )
    parent_doc: str = Field(..., min_length=1, description="Source filename in corpus/")


class FixedSizeChunker:
    """Splits text into fixed tiktoken windows with exact token overlap."""

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def _compute_line_range(self, full_text: str, char_start: int, char_end: int) -> tuple[int, int]:
        line_start = full_text.count("\n", 0, max(char_start, 0)) + 1
        line_end = full_text.count("\n", 0, max(char_end, 0)) + 1
        return line_start, max(line_start, line_end)

    def _active_headers_at_offset(self, full_text: str, char_offset: int) -> str:
        prefix = full_text[:char_offset]
        in_fence = False
        h1 = h2 = h3 = ""
        for raw_line in prefix.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if stripped.startswith("# "):
                h1, h2, h3 = stripped[2:].strip(), "", ""
            elif stripped.startswith("## "):
                h2, h3 = stripped[3:].strip(), ""
            elif stripped.startswith("### "):
                h3 = stripped[4:].strip()
        parts = [p for p in (h1, h2, h3) if p]
        return " > ".join(parts) if parts else "Document Root"

    def chunk_text(self, text: str, parent_doc: str) -> List[DocumentChunk]:
        cleaned = text.strip()
        if not cleaned:
            return []

        tokens = self._encoding.encode(cleaned)
        total_tokens = len(tokens)
        step = self.chunk_size - self.chunk_overlap

        chunks: List[DocumentChunk] = []
        cursor = 0
        idx = 0

        while cursor < total_tokens:
            end_idx = min(cursor + self.chunk_size, total_tokens)
            window_tokens = tokens[cursor:end_idx]
            window_text = self._encoding.decode(window_tokens)

            char_start = cleaned.find(window_text[:40]) if len(window_text) >= 40 else 0
            char_start = max(char_start, 0)
            char_end = min(char_start + len(window_text), len(cleaned))
            line_start, line_end = self._compute_line_range(cleaned, char_start, char_end)
            header_hierarchy = self._active_headers_at_offset(cleaned, char_start)

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{parent_doc}::fixed_{self.chunk_size}::chunk-{idx + 1}",
                    content=window_text,
                    parent_doc=parent_doc,
                    metadata={
                        "strategy": f"fixed_{self.chunk_size}",
                        "chunk_index": idx + 1,
                        "token_count": len(window_tokens),
                        "chunk_overlap": self.chunk_overlap,
                        "header_hierarchy": header_hierarchy,
                        "line_start": line_start,
                        "line_end": line_end,
                    },
                )
            )
            idx += 1
            if end_idx == total_tokens:
                break
            cursor += step

        return chunks

    def chunk_corpus(self, corpus_dir: str = "corpus") -> List[DocumentChunk]:
        corpus_path = Path(corpus_dir)
        all_chunks: List[DocumentChunk] = []
        for file_path in sorted(corpus_path.glob("*.md")):
            text = file_path.read_text(encoding="utf-8")
            all_chunks.extend(self.chunk_text(text, parent_doc=file_path.name))
        return all_chunks


class StructureAwareChunker:
    """Splits Markdown on header boundaries (#, ##, ###) while keeping fenced code blocks intact."""

    _HEADER_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$")

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    def chunk_text(self, text: str, parent_doc: str) -> List[DocumentChunk]:
        lines = text.splitlines()
        if not any(line.strip() for line in lines):
            return []

        chunks: List[DocumentChunk] = []
        h1 = h2 = h3 = ""
        in_code_fence = False

        current_lines: List[str] = []
        section_start_line = 1
        section_idx = 0

        def _current_hierarchy() -> str:
            parts = [p for p in (h1, h2, h3) if p]
            return " > ".join(parts) if parts else "Document Root"

        def _flush(end_line: int) -> None:
            nonlocal section_idx, current_lines
            body = "\n".join(current_lines).strip()
            if not body:
                current_lines = []
                return
            section_idx += 1
            hierarchy = _current_hierarchy()
            token_count = len(self._encoding.encode(body))
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{parent_doc}::structural::sec-{section_idx}",
                    content=body,
                    parent_doc=parent_doc,
                    metadata={
                        "strategy": "structure_aware",
                        "section_index": section_idx,
                        "header_hierarchy": hierarchy,
                        "section_title": h3 or h2 or h1 or "Document Root",
                        "token_count": token_count,
                        "line_start": section_start_line,
                        "line_end": max(section_start_line, end_line),
                    },
                )
            )
            current_lines = []

        for line_num, raw_line in enumerate(lines, start=1):
            stripped = raw_line.strip()

            if stripped.startswith("```"):
                in_code_fence = not in_code_fence
                if not current_lines:
                    section_start_line = line_num
                current_lines.append(raw_line)
                continue

            if not in_code_fence:
                header_match = self._HEADER_PATTERN.match(stripped)
                if header_match:
                    _flush(line_num - 1)
                    level = len(header_match.group(1))
                    title = header_match.group(2).strip()
                    if level == 1:
                        h1, h2, h3 = title, "", ""
                    elif level == 2:
                        h2, h3 = title, ""
                    else:
                        h3 = title
                    section_start_line = line_num
                    current_lines.append(raw_line)
                    continue

            if not current_lines:
                section_start_line = line_num
            current_lines.append(raw_line)

        _flush(len(lines))
        return chunks

    def chunk_corpus(self, corpus_dir: str = "corpus") -> List[DocumentChunk]:
        corpus_path = Path(corpus_dir)
        all_chunks: List[DocumentChunk] = []
        for file_path in sorted(corpus_path.glob("*.md")):
            text = file_path.read_text(encoding="utf-8")
            all_chunks.extend(self.chunk_text(text, parent_doc=file_path.name))
        return all_chunks

```


---

## File: `src/indexer.py`

```python
"""src/indexer.py - Dual Dense (ChromaDB + L2-Normalized Embeddings) and Sparse (BM25) Indices."""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import re
from typing import List, Optional, Tuple
import uuid

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.chunking import DocumentChunk

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
_SHARED_ENCODER: Optional[SentenceTransformer] = None


def get_shared_encoder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Loads SentenceTransformer once from local cache first to eliminate redundant model init."""
    global _SHARED_ENCODER
    if _SHARED_ENCODER is None:
        try:
            _SHARED_ENCODER = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            _SHARED_ENCODER = SentenceTransformer(model_name)
    return _SHARED_ENCODER


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Symmetric tokenizer applied identically during corpus indexing and query execution.
    Lowercases text and preserves alphanumeric + underscore identifiers (e.g. CUSTOM_CONFIG_FLAG_XYZ).
    """
    return _TOKEN_PATTERN.findall((text or "").lower())


class BM25SparseIndex:
    """Lexical keyword index backed by rank_bm25.BM25Okapi with symmetric tokenization."""

    def __init__(self, chunks: List[DocumentChunk]) -> None:
        self.chunks = chunks
        self._tokenized_corpus = [
            tokenize_for_bm25(f"{c.metadata.get('header_hierarchy', '')} {c.content}")
            for c in chunks
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Returns top-K (DocumentChunk, bm25_score) pairs sorted descending."""
        if not self.chunks or self._bm25 is None:
            return []
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Tuple[DocumentChunk, float]] = []
        for idx in top_indices:
            score_val = float(scores[int(idx)])
            if score_val > 0.0:
                results.append((self.chunks[int(idx)], round(score_val, 4)))
        return results


class DenseVectorIndex:
    """In-memory ChromaDB index storing unit-length L2-normalized embeddings in cosine space."""

    def __init__(
        self,
        chunks: List[DocumentChunk],
        collection_prefix: str = "bench",
        model_name: str = "all-MiniLM-L6-v2",
        encoder: Optional[SentenceTransformer] = None,
    ) -> None:
        self.chunks = chunks
        self._chunk_lookup = {c.chunk_id: c for c in chunks}
        self._encoder = encoder or get_shared_encoder(model_name)

        self._client = chromadb.EphemeralClient()
        unique_name = f"{collection_prefix}_{uuid.uuid4().hex[:8]}"
        self._collection = self._client.create_collection(
            name=unique_name,
            metadata={"hnsw:space": "cosine"},
        )

        if chunks:
            texts = [f"{c.metadata.get('header_hierarchy', '')}\n{c.content}" for c in chunks]
            embeddings = self._encoder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._collection.add(
                ids=[c.chunk_id for c in chunks],
                embeddings=embeddings.tolist(),
                documents=[c.content for c in chunks],
            )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Queries ChromaDB using a unit-normalized query vector and converts cosine distance (1 - cos)
        back into true cosine similarity in [0.0, 1.0].
        """
        if not self.chunks or not query.strip():
            return []

        query_vec = self._encoder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        n_results = min(top_k, len(self.chunks))
        raw = self._collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=n_results,
            include=["distances"],
        )

        ids_list = raw.get("ids", [[]])[0]
        distances_list = raw.get("distances", [[]])[0]

        results: List[Tuple[DocumentChunk, float]] = []
        for chunk_id, distance in zip(ids_list, distances_list):
            cosine_sim = max(0.0, min(1.0, round(1.0 - float(distance), 4)))
            chunk_obj = self._chunk_lookup.get(chunk_id)
            if chunk_obj is not None:
                results.append((chunk_obj, cosine_sim))

        results.sort(key=lambda item: item[1], reverse=True)
        return results

```


---

## File: `src/search.py`

```python
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

```


---

## File: `src/benchmarker.py`

```python
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

```


---

## File: `src/app.py`

```python
"""src/app.py - SearchWorkbench: Interactive Search Comparator & IR Benchmark Dashboard."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from src.benchmarker import RetrievalBenchmarker
from src.chunking import FixedSizeChunker, StructureAwareChunker
from src.indexer import BM25SparseIndex, DenseVectorIndex, get_shared_encoder
from src.search import UnifiedSearchEngine

st.set_page_config(
    page_title="SearchWorkbench | Semantic Search & IR Benchmarker",
    layout="wide",
)


@st.cache_resource(show_spinner="Indexing technical documentation corpus across chunking strategies...")
def load_search_engines() -> dict[str, UnifiedSearchEngine]:
    encoder = get_shared_encoder("all-MiniLM-L6-v2")

    chunks_struct = StructureAwareChunker().chunk_corpus("corpus")
    chunks_256 = FixedSizeChunker(chunk_size=256, chunk_overlap=32).chunk_corpus("corpus")
    chunks_1024 = FixedSizeChunker(chunk_size=1024, chunk_overlap=128).chunk_corpus("corpus")

    return {
        "Structure-Aware (Markdown Headers)": UnifiedSearchEngine(
            bm25_index=BM25SparseIndex(chunks_struct),
            dense_index=DenseVectorIndex(chunks_struct, collection_prefix="ui_struct", encoder=encoder),
        ),
        "Fixed-Size (256 Tokens / 32 Overlap)": UnifiedSearchEngine(
            bm25_index=BM25SparseIndex(chunks_256),
            dense_index=DenseVectorIndex(chunks_256, collection_prefix="ui_256", encoder=encoder),
        ),
        "Fixed-Size (1024 Tokens / 128 Overlap)": UnifiedSearchEngine(
            bm25_index=BM25SparseIndex(chunks_1024),
            dense_index=DenseVectorIndex(chunks_1024, collection_prefix="ui_1024", encoder=encoder),
        ),
    }


engines = load_search_engines()

st.title("SearchWorkbench: Semantic Search Engine & Empirical IR Benchmarker")
st.caption(
    "Comparing Sparse Lexical Retrieval (BM25Okapi) vs. Dense Vector Search (all-MiniLM-L6-v2 + ChromaDB) across Fixed & Structure-Aware Chunking"
)

tab_compare, tab_benchmark = st.tabs(
    ["1. Live Search Comparator (BM25 vs. Dense)", "2. Benchmark & Evaluation Dashboard"]
)

with tab_compare:
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        selected_strategy = st.selectbox(
            "Chunking & Indexing Strategy:",
            options=list(engines.keys()),
            index=0,
        )
    with ctrl_col2:
        top_k = st.slider("Top-K Results (K):", min_value=1, max_value=10, value=5)

    st.markdown("**Quick Comparison Presets:**")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    preset_q = ""
    with p_col1:
        if st.button("Lexical: maxUnavailable Default", width="stretch"):
            preset_q = "What is the default value of maxUnavailable in RollingUpdateDeployment?"
    with p_col2:
        if st.button("Lexical: ECONNRESET Pool Error", width="stretch"):
            preset_q = "Error: ECONNRESET in pooled connection"
    with p_col3:
        if st.button("Conceptual: Zero-Downtime Rollout", width="stretch"):
            preset_q = "How do I ensure zero-downtime rolling upgrades during heavy traffic?"
    with p_col4:
        if st.button("Conceptual: Socket Starvation", width="stretch"):
            preset_q = "Preventing network socket starvation in client database pools"

    with st.form("live_search_form", clear_on_submit=False):
        query_input = st.text_input(
            "Enter a technical search query (exact identifier or conceptual question):",
            value=preset_q,
            placeholder="e.g., CUSTOM_CONFIG_FLAG_XYZ or How does Envoy rotate mTLS certificates?",
        )
        run_search = st.form_submit_button("Compare BM25 vs. Dense Retrieval", type="primary")

    if run_search and query_input.strip():
        active_engine = engines[selected_strategy]
        bm25_hits, bm25_ms = active_engine.search_bm25(query_input, top_k=top_k)
        dense_hits, dense_ms = active_engine.search_dense(query_input, top_k=top_k)

        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("Sparse Lexical Search (BM25Okapi)")
            st.metric("Query Latency (ms)", f"{bm25_ms:.2f} ms", delta=f"{len(bm25_hits)} hits")
            if not bm25_hits:
                st.warning("Zero lexical token matches found in corpus.")
            for hit in bm25_hits:
                with st.container(border=True):
                    st.markdown(
                        f"**Rank #{hit.rank}** | BM25 Score: **`{hit.score:.4f}`** | `{hit.chunk.parent_doc}`"
                    )
                    st.caption(
                        f"Section: {hit.chunk.metadata.get('header_hierarchy')} | "
                        f"Tokens: {hit.chunk.metadata.get('token_count')} | "
                        f"Matched Terms: `{ ', '.join(hit.matched_terms) or 'none' }`"
                    )
                    st.markdown(hit.chunk.content[:420] + ("..." if len(hit.chunk.content) > 420 else ""))

        with right_col:
            st.subheader("Dense Vector Search (Cosine Similarity)")
            st.metric("Query Latency (ms)", f"{dense_ms:.2f} ms", delta=f"{len(dense_hits)} hits")
            for hit in dense_hits:
                with st.container(border=True):
                    st.markdown(
                        f"**Rank #{hit.rank}** | Cosine Similarity: **`{hit.score:.4f}`** | `{hit.chunk.parent_doc}`"
                    )
                    st.caption(
                        f"Section: {hit.chunk.metadata.get('header_hierarchy')} | "
                        f"Tokens: {hit.chunk.metadata.get('token_count')} | "
                        f"Lexical Overlap: `{ ', '.join(hit.matched_terms) or 'semantic match only' }`"
                    )
                    st.markdown(hit.chunk.content[:420] + ("..." if len(hit.chunk.content) > 420 else ""))

with tab_benchmark:
    st.markdown(
        "Evaluates **labeled technical queries** (`data/eval_queries.json`) "
        "across **BM25 Baseline**, **Dense Fixed-256**, **Dense Fixed-1024**, and **Dense Structure-Aware**."
    )

    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = None

    if st.button("Run Benchmark Suite (Ground-Truth Queries)", type="primary"):
        with st.spinner("Running 4-way empirical IR benchmark across Hit Rate@K, Recall@K, MRR@K, and Latency..."):
            benchmarker = RetrievalBenchmarker(corpus_dir="corpus", eval_path="data/eval_queries.json")
            summary_rows, failure_cases = benchmarker.run_all_benchmarks()
            st.session_state.benchmark_results = (summary_rows, failure_cases)

    if st.session_state.benchmark_results is not None:
        summary_rows, failure_cases = st.session_state.benchmark_results
        df = pd.DataFrame([row.model_dump() for row in summary_rows])

        st.subheader("Empirical Retrieval Metrics Comparison")
        display_df = df.rename(
            columns={
                "configuration": "Configuration",
                "chunk_count": "Total Chunks",
                "hit_rate_at_3": "Hit Rate@3",
                "hit_rate_at_5": "Hit Rate@5",
                "recall_at_5": "Recall@5",
                "mrr_at_5": "MRR@5",
                "avg_latency_ms": "Avg Latency (ms)",
                "p50_latency_ms": "p50 Latency (ms)",
                "p95_latency_ms": "p95 Latency (ms)",
            }
        )
        st.dataframe(display_df, width="stretch", hide_index=True)

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("#### Ranking Quality: Hit Rate@5 vs. MRR@5")
            quality_chart = display_df.set_index("Configuration")[["Hit Rate@3", "Hit Rate@5", "MRR@5"]]
            st.bar_chart(quality_chart)

        with chart_col2:
            st.markdown("#### Query Execution Latency: p50 vs. p95 (ms)")
            latency_chart = display_df.set_index("Configuration")[["p50 Latency (ms)", "p95 Latency (ms)"]]
            st.bar_chart(latency_chart)

        st.divider()
        st.subheader("Head-to-Head Failure & Trade-Off Analysis (BM25 vs. Structure-Aware Dense)")
        if failure_cases:
            st.dataframe(pd.DataFrame(failure_cases), width="stretch", hide_index=True)
        else:
            st.success("Both engines tied across every query in the evaluation dataset.")

```


---

## File: `tests/test_chunking.py`

```python

```


---

## File: `tests/test_retrieval.py`

```python
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

```


---

## File: `tests/test_benchmarks.py`

```python
import pytest
from src.benchmarker import compute_hit_rate_at_k, compute_mrr_at_k

def test_metric_math_assertions() -> None:
    ranking = [False, True, False, True, False]
    assert compute_hit_rate_at_k(ranking, k=3) == 1.0
    assert compute_mrr_at_k(ranking, k=5) == pytest.approx(0.5)

    missed = [False, False, False]
    assert compute_hit_rate_at_k(missed, k=3) == 0.0
    assert compute_mrr_at_k(missed, k=3) == 0.0

```
