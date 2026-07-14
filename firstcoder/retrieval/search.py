"""Semantic code-candidate search independent of AgentLoop."""

from __future__ import annotations

from firstcoder.retrieval.embedding import EmbeddingProvider
from firstcoder.retrieval.models import CodeSearchHit
from firstcoder.retrieval.vector_store import VectorStore


class SemanticCodeSearch:
    def __init__(self, *, embedder: EmbeddingProvider, store: VectorStore, repo_id: str) -> None:
        self.embedder = embedder
        self.store = store
        self.repo_id = repo_id

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        path_prefix: str | None = None,
        include_tests: bool = True,
    ) -> list[CodeSearchHit]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        vector = self.embedder.embed_query(query)
        return self.store.search(
            repo_id=self.repo_id,
            query_vector=vector,
            top_k=top_k,
            path_prefix=path_prefix,
            include_tests=include_tests,
        )
