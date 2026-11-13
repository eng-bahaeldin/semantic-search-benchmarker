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
