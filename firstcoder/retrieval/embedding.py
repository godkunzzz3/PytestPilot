"""Replaceable local embedding providers."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Protocol


class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class FakeEmbeddingProvider:
    """Deterministic token-hash embeddings for offline unit tests."""

    def __init__(self, *, dimension: int = 32, model_name: str = "fake-token-hash") -> None:
        self.dimension = dimension
        self.model_name = model_name
        self.document_batch_count = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if texts:
            self.document_batch_count += 1
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in re.findall(r"[A-Za-z0-9_]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class FastEmbedProvider:
    model_name = "BAAI/bge-small-en-v1.5"
    dimension = 384

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        batch_size: int = 16,
        local_files_only: bool = False,
    ) -> None:
        from fastembed import TextEmbedding

        self.batch_size = batch_size
        self._model = TextEmbedding(
            model_name=self.model_name,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            providers=["CPUExecutionProvider"],
            cuda=False,
            local_files_only=local_files_only,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()
