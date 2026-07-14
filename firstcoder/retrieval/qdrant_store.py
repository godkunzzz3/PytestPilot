"""Qdrant local persistent implementation of the vector-store protocol."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from firstcoder.retrieval.models import CodeChunk, CodeSearchHit, RetrievalUnavailableError


class QdrantLocalVectorStore:
    def __init__(
        self,
        path: str | Path,
        *,
        dimension: int,
        model_name: str,
        collection_name: str = "firstcoder_code",
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.model_name = model_name
        self.collection_name = collection_name
        self._metadata_path = self.path / "index_metadata.json"
        try:
            self.client = QdrantClient(path=str(self.path / "storage"))
            self._ensure_collection()
        except Exception as exc:  # noqa: BLE001
            raise RetrievalUnavailableError(f"could not open Qdrant local index: {exc}") from exc

    def upsert(self, chunks: list[CodeChunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal lengths")
        for vector in vectors:
            self._validate_dimension(vector)
        if not chunks:
            return 0
        points = [
            models.PointStruct(id=_point_id(chunk.repo_id, chunk.chunk_id), vector=vector, payload=chunk.to_dict())
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalUnavailableError(f"Qdrant upsert failed: {exc}") from exc
        return len(points)

    def delete(self, repo_id: str, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[_point_id(repo_id, chunk_id) for chunk_id in chunk_ids]),
                wait=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise RetrievalUnavailableError(f"Qdrant delete failed: {exc}") from exc
        return len(chunk_ids)

    def delete_repo(self, repo_id: str) -> int:
        chunk_ids = list(self.list_chunk_hashes(repo_id))
        return self.delete(repo_id, chunk_ids)

    def list_chunk_hashes(self, repo_id: str) -> dict[str, str]:
        result: dict[str, str] = {}
        offset: Any | None = None
        try:
            while True:
                records, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=_repo_filter(repo_id),
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for record in records:
                    payload = dict(record.payload or {})
                    result[str(payload.get("chunk_id") or "")] = str(payload.get("content_hash") or "")
                if offset is None:
                    break
        except Exception as exc:  # noqa: BLE001
            raise RetrievalUnavailableError(f"Qdrant scroll failed: {exc}") from exc
        return {key: value for key, value in result.items() if key}

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
        must = [models.FieldCondition(key="repo_id", match=models.MatchValue(value=repo_id))]
        if not include_tests:
            must.append(models.FieldCondition(key="is_test", match=models.MatchValue(value=False)))
        try:
            points = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=models.Filter(must=must),
                limit=max(100, top_k * 20),
                with_payload=True,
            ).points
        except Exception as exc:  # noqa: BLE001
            raise RetrievalUnavailableError(f"Qdrant query failed: {exc}") from exc
        hits: list[CodeSearchHit] = []
        for point in points:
            payload = dict(point.payload or {})
            chunk = CodeChunk.from_dict(payload)
            if path_prefix and not chunk.path.startswith(path_prefix):
                continue
            hits.append(
                CodeSearchHit(
                    chunk_id=chunk.chunk_id,
                    repo_id=chunk.repo_id,
                    path=chunk.path,
                    symbol=chunk.symbol,
                    kind=chunk.kind,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    score=float(point.score),
                    bounded_preview=chunk.bounded_content[:600],
                    content_hash=chunk.content_hash,
                    is_test=chunk.is_test,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def get_index_metadata(self, repo_id: str) -> dict[str, Any] | None:
        all_metadata = self._read_metadata()
        value = all_metadata.get(repo_id)
        return dict(value) if isinstance(value, dict) else None

    def set_index_metadata(self, repo_id: str, metadata: dict[str, Any]) -> None:
        all_metadata = self._read_metadata()
        all_metadata[repo_id] = dict(metadata)
        self._metadata_path.write_text(
            json.dumps(all_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def close(self) -> None:
        self.client.close()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE),
            )
            return
        vectors = self.client.get_collection(self.collection_name).config.params.vectors
        stored_dimension = int(getattr(vectors, "size", 0))
        if stored_dimension != self.dimension:
            self.client.close()
            raise ValueError(
                f"Qdrant collection dimension {stored_dimension} does not match configured {self.dimension}"
            )

    def _read_metadata(self) -> dict[str, Any]:
        if not self._metadata_path.exists():
            return {}
        value = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, dict) else {}

    def _validate_dimension(self, vector: list[float]) -> None:
        if len(vector) != self.dimension:
            raise ValueError(f"vector dimension {len(vector)} does not match store dimension {self.dimension}")


def _point_id(repo_id: str, chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"firstcoder:{repo_id}:{chunk_id}"))


def _repo_filter(repo_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="repo_id", match=models.MatchValue(value=repo_id))]
    )
