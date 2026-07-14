from pathlib import Path

from firstcoder.retrieval import (
    CodeIndexer,
    FakeEmbeddingProvider,
    FakeVectorStore,
    RetrievalUnavailableError,
    SemanticCodeSearch,
)
from firstcoder.tools.code_search import create_code_search_tool


def _search(tmp_path: Path) -> SemanticCodeSearch:
    (tmp_path / "service.py").write_text(
        "def invalidate_user_cache():\n    return 'remove stale cached profile'\n",
        encoding="utf-8",
    )
    embedder = FakeEmbeddingProvider(dimension=16)
    store = FakeVectorStore(dimension=16)
    CodeIndexer(tmp_path, embedder=embedder, store=store, repo_id="repo-a").build()
    return SemanticCodeSearch(embedder=embedder, store=store, repo_id="repo-a")


def test_code_search_returns_bounded_structured_candidates(tmp_path: Path) -> None:
    tool = create_code_search_tool(tmp_path, search=_search(tmp_path))

    result = tool.executor(query="stale cached profile", top_k=5, path_prefix="service", include_tests=False)

    assert result.ok is True
    assert result.data["candidate_only"] is True
    assert result.data["requires_source_read"] is True
    assert result.data["hits"][0]["path"] == "service.py"
    assert result.data["hits"][0]["symbol"] in {"<module>", "invalidate_user_cache"}
    assert "content_hash" in result.data["hits"][0]
    assert len(result.content) <= 5_000


def test_code_search_validates_top_k_and_path_prefix(tmp_path: Path) -> None:
    tool = create_code_search_tool(tmp_path, search=_search(tmp_path))

    assert tool.executor(query="x", top_k=0).ok is False
    assert tool.executor(query="x", top_k=21).ok is False
    assert tool.executor(query="x", path_prefix="../outside").ok is False
    assert tool.executor(query="x", path_prefix="/absolute").ok is False


def test_code_search_returns_empty_hits(tmp_path: Path) -> None:
    embedder = FakeEmbeddingProvider(dimension=8)
    search = SemanticCodeSearch(embedder=embedder, store=FakeVectorStore(dimension=8), repo_id="repo-a")
    result = create_code_search_tool(tmp_path, search=search).executor(query="nothing")

    assert result.ok is True
    assert result.data["hits"] == []


def test_code_search_reports_structured_unavailable_without_breaking_other_tools(tmp_path: Path) -> None:
    class UnavailableSearch:
        def search(self, *args, **kwargs):
            raise RetrievalUnavailableError("qdrant is unavailable")

    result = create_code_search_tool(tmp_path, search=UnavailableSearch()).executor(query="failure")

    assert result.ok is False
    assert result.data["unavailable"] is True
    assert result.data["fallback_tools"] == ["grep", "glob", "view", "read_multi"]
