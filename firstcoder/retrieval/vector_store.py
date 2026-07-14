"""Vector-store protocol and deterministic in-memory fake."""

from __future__ import annotations

import math
from typing import Any, Protocol

from firstcoder.retrieval.models import CodeChunk, CodeSearchHit


class VectorStore(Protocol):
    dimension: int

    def upsert(self, chunks: list[CodeChunk], vectors: list[list[float]]) -> int:
        ...

    def delete(self, repo_id: str, chunk_ids: list[str]) -> int:
        ...

    def delete_repo(self, repo_id: str) -> int:
        ...

    def list_chunk_hashes(self, repo_id: str) -> dict[str, str]:
        ...

    def search(
        self,
        *,
        repo_id: str,
        query_vector: list[float],
        top_k: int,
        path_prefix: str | None,
        include_tests: bool,
    ) -> list[CodeSearchHit]:
        ...

    def get_index_metadata(self, repo_id: str) -> dict[str, Any] | None:
        ...

    def set_index_metadata(self, repo_id: str, metadata: dict[str, Any]) -> None:
        ...


class FakeVectorStore:
    def __init__(self, *, dimension: int) -> None:
        self.dimension = dimension
        self._points: dict[tuple[str, str], tuple[CodeChunk, list[float]]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def upsert(self, chunks: list[CodeChunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal lengths")
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._validate_dimension(vector)
            self._points[(chunk.repo_id, chunk.chunk_id)] = (chunk, list(vector))
        return len(chunks)

    def delete(self, repo_id: str, chunk_ids: list[str]) -> int:
        deleted = 0
        for chunk_id in chunk_ids:
            if self._points.pop((repo_id, chunk_id), None) is not None:
                deleted += 1
        return deleted

    def delete_repo(self, repo_id: str) -> int:
        chunk_ids = [chunk_id for stored_repo, chunk_id in self._points if stored_repo == repo_id]
        return self.delete(repo_id, chunk_ids)

    def list_chunk_hashes(self, repo_id: str) -> dict[str, str]:
        return {
            chunk_id: chunk.content_hash
            for (stored_repo, chunk_id), (chunk, _) in self._points.items()
            if stored_repo == repo_id
        }

    def search(
        self,
        *,
        repo_id: str,
        query_vector: list[float],
        top_k: int,
        path_prefix: str | None,
        include_tests: bool,
    ) -> list[CodeSearchHit]:
        self._validate_dimension(query_vector)
        candidates: list[CodeSearchHit] = []
        for (stored_repo, _), (chunk, vector) in self._points.items():
            if stored_repo != repo_id:
                continue
            if path_prefix and not chunk.path.startswith(path_prefix):
                continue
            if not include_tests and chunk.is_test:
                continue
            candidates.append(_hit(chunk, _cosine(query_vector, vector)))
        return sorted(candidates, key=lambda hit: (-hit.score, hit.path, hit.start_line))[:top_k]

    def get_index_metadata(self, repo_id: str) -> dict[str, Any] | None:
        metadata = self._metadata.get(repo_id)
        return dict(metadata) if metadata is not None else None

    def set_index_metadata(self, repo_id: str, metadata: dict[str, Any]) -> None:
        self._metadata[repo_id] = dict(metadata)

    def _validate_dimension(self, vector: list[float]) -> None:
        if len(vector) != self.dimension:
            raise ValueError(f"vector dimension {len(vector)} does not match store dimension {self.dimension}")


def _hit(chunk: CodeChunk, score: float, *, preview_chars: int = 600) -> CodeSearchHit:
    return CodeSearchHit(
        chunk_id=chunk.chunk_id,
        repo_id=chunk.repo_id,
        path=chunk.path,
        symbol=chunk.symbol,
        kind=chunk.kind,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        score=float(score),
        bounded_preview=chunk.bounded_content[:preview_chars],
        content_hash=chunk.content_hash,
        is_test=chunk.is_test,
    )


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
