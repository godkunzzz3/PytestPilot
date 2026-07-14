"""Data models shared by local code indexing and semantic search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CodeChunk:
    chunk_id: str
    repo_id: str
    path: str
    language: str
    kind: str
    symbol: str
    start_line: int
    end_line: int
    is_test: bool
    content_hash: str
    bounded_content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CodeChunk":
        return cls(
            chunk_id=str(value["chunk_id"]),
            repo_id=str(value["repo_id"]),
            path=str(value["path"]),
            language=str(value.get("language") or "python"),
            kind=str(value.get("kind") or "unknown"),
            symbol=str(value.get("symbol") or "<module>"),
            start_line=int(value.get("start_line") or 1),
            end_line=int(value.get("end_line") or 1),
            is_test=bool(value.get("is_test")),
            content_hash=str(value.get("content_hash") or ""),
            bounded_content=str(value.get("bounded_content") or ""),
        )


@dataclass(frozen=True, slots=True)
class CodeSearchHit:
    chunk_id: str
    repo_id: str
    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    score: float
    bounded_preview: str
    content_hash: str
    is_test: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CodeIndexReport:
    repo_id: str
    scanned_file_count: int
    chunk_count: int
    upserted_count: int
    unchanged_count: int
    deleted_count: int
    elapsed_seconds: float
    model_name: str
    dimension: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RetrievalUnavailableError(RuntimeError):
    """The optional semantic retrieval backend cannot serve a request."""


class RebuildRequiredError(RuntimeError):
    """Stored index metadata is incompatible with the configured embedder."""
