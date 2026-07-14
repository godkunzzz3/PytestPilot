"""Local Python code indexing and semantic retrieval."""

from firstcoder.retrieval.chunking import chunk_python_file, iter_python_files
from firstcoder.retrieval.embedding import EmbeddingProvider, FakeEmbeddingProvider, FastEmbedProvider
from firstcoder.retrieval.index import CodeIndexer, repository_id
from firstcoder.retrieval.models import (
    CodeChunk,
    CodeIndexReport,
    CodeSearchHit,
    RebuildRequiredError,
    RetrievalUnavailableError,
)
from firstcoder.retrieval.qdrant_store import QdrantLocalVectorStore
from firstcoder.retrieval.search import SemanticCodeSearch
from firstcoder.retrieval.vector_store import FakeVectorStore, VectorStore

__all__ = [
    "CodeChunk",
    "CodeIndexReport",
    "CodeIndexer",
    "CodeSearchHit",
    "EmbeddingProvider",
    "FakeEmbeddingProvider",
    "FakeVectorStore",
    "FastEmbedProvider",
    "QdrantLocalVectorStore",
    "RebuildRequiredError",
    "RetrievalUnavailableError",
    "SemanticCodeSearch",
    "VectorStore",
    "chunk_python_file",
    "iter_python_files",
    "repository_id",
]
