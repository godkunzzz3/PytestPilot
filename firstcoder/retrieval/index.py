"""Incremental repository-level Python code indexing."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from firstcoder.retrieval.chunking import chunk_python_file, iter_python_files
from firstcoder.retrieval.embedding import EmbeddingProvider
from firstcoder.retrieval.models import CodeIndexReport, RebuildRequiredError
from firstcoder.retrieval.vector_store import VectorStore


class CodeIndexer:
    def __init__(
        self,
        root: str | Path,
        *,
        embedder: EmbeddingProvider,
        store: VectorStore,
        repo_id: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.root = Path(root).resolve()
        self.embedder = embedder
        self.store = store
        self.repo_id = repo_id or repository_id(self.root)
        self.batch_size = max(1, batch_size)

    def build(self, *, rebuild: bool = False) -> CodeIndexReport:
        started = time.perf_counter()
        existing_metadata = self.store.get_index_metadata(self.repo_id)
        expected_metadata = {"model_name": self.embedder.model_name, "dimension": self.embedder.dimension}
        if existing_metadata and not _compatible(existing_metadata, expected_metadata):
            if not rebuild:
                raise RebuildRequiredError(
                    "embedding model or dimension changed; rebuild the repository index"
                )
        if self.store.dimension != self.embedder.dimension:
            raise RebuildRequiredError("embedding dimension does not match vector store dimension")
        if rebuild:
            self.store.delete_repo(self.repo_id)

        files = iter_python_files(self.root)
        chunks = [
            chunk
            for path in files
            for chunk in chunk_python_file(self.root, path, repo_id=self.repo_id)
        ]
        existing_hashes = {} if rebuild else self.store.list_chunk_hashes(self.repo_id)
        current_ids = {chunk.chunk_id for chunk in chunks}
        changed = [chunk for chunk in chunks if existing_hashes.get(chunk.chunk_id) != chunk.content_hash]
        unchanged_count = len(chunks) - len(changed)
        upserted = 0
        for offset in range(0, len(changed), self.batch_size):
            batch = changed[offset : offset + self.batch_size]
            vectors = self.embedder.embed_documents([chunk.bounded_content for chunk in batch])
            upserted += self.store.upsert(batch, vectors)
        stale_ids = sorted(set(existing_hashes) - current_ids)
        deleted = self.store.delete(self.repo_id, stale_ids)
        metadata: dict[str, Any] = {
            **expected_metadata,
            "repo_id": self.repo_id,
            "chunk_count": len(chunks),
        }
        self.store.set_index_metadata(self.repo_id, metadata)
        return CodeIndexReport(
            repo_id=self.repo_id,
            scanned_file_count=len(files),
            chunk_count=len(chunks),
            upserted_count=upserted,
            unchanged_count=unchanged_count,
            deleted_count=deleted,
            elapsed_seconds=round(time.perf_counter() - started, 6),
            model_name=self.embedder.model_name,
            dimension=self.embedder.dimension,
        )

    def status(self) -> dict[str, Any]:
        metadata = self.store.get_index_metadata(self.repo_id) or {}
        return {
            "repo_id": self.repo_id,
            "indexed_chunk_count": len(self.store.list_chunk_hashes(self.repo_id)),
            "model_name": metadata.get("model_name"),
            "dimension": metadata.get("dimension"),
            "rebuild_required": bool(metadata) and not _compatible(
                metadata,
                {"model_name": self.embedder.model_name, "dimension": self.embedder.dimension},
            ),
        }


def repository_id(root: str | Path) -> str:
    return hashlib.sha256(str(Path(root).resolve()).encode("utf-8")).hexdigest()[:20]


def _compatible(stored: dict[str, Any], expected: dict[str, Any]) -> bool:
    return stored.get("model_name") == expected["model_name"] and stored.get("dimension") == expected["dimension"]
