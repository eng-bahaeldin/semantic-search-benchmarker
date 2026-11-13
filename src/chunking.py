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
