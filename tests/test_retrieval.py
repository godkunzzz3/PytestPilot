import json
import os
from pathlib import Path

import pytest

from firstcoder.retrieval import (
    CodeIndexer,
    FastEmbedProvider,
    FakeEmbeddingProvider,
    FakeVectorStore,
    QdrantLocalVectorStore,
    RebuildRequiredError,
    SemanticCodeSearch,
    chunk_python_file,
    iter_python_files,
)


def test_ast_chunker_covers_module_class_method_function_async_and_pytest(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        """\
VALUE = 1

class Service:
    def run(self):
        return VALUE

def helper():
    return VALUE

async def fetch():
    return VALUE

def test_helper():
    assert helper() == 1
""",
        encoding="utf-8",
    )

    chunks = chunk_python_file(tmp_path, source, repo_id="repo-a")
    kinds_symbols = {(chunk.kind, chunk.symbol) for chunk in chunks}

    assert ("module", "<module>") in kinds_symbols
    assert ("class", "Service") in kinds_symbols
    assert ("method", "Service.run") in kinds_symbols
    assert ("function", "helper") in kinds_symbols
    assert ("async_function", "fetch") in kinds_symbols
    assert ("test_function", "test_helper") in kinds_symbols
    assert all(chunk.language == "python" for chunk in chunks)
    assert all(chunk.path == "sample.py" for chunk in chunks)
    assert all(chunk.start_line <= chunk.end_line for chunk in chunks)
    json.dumps([chunk.to_dict() for chunk in chunks])


def test_syntax_error_uses_line_window_fallback(tmp_path: Path) -> None:
    source = tmp_path / "broken.py"
    source.write_text("def broken(:\n    pass\n", encoding="utf-8")

    chunks = chunk_python_file(tmp_path, source, repo_id="repo-a", window_lines=1)

    assert [chunk.kind for chunk in chunks] == ["fallback", "fallback"]
    assert [chunk.start_line for chunk in chunks] == [1, 2]


def test_chunk_id_is_stable_while_content_hash_tracks_changes(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    first = chunk_python_file(tmp_path, source, repo_id="repo-a")
    source.write_text("def value():\n    return 2\n", encoding="utf-8")
    second = chunk_python_file(tmp_path, source, repo_id="repo-a")

    first_function = next(chunk for chunk in first if chunk.symbol == "value")
    second_function = next(chunk for chunk in second if chunk.symbol == "value")
    assert first_function.chunk_id == second_function.chunk_id
    assert first_function.content_hash != second_function.content_hash


def test_python_file_scan_ignores_generated_and_environment_directories(tmp_path: Path) -> None:
    for relative in ["src/good.py", ".venv/bad.py", "build/bad.py", "runs/bad.py", "__pycache__/bad.py"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("VALUE = 1\n", encoding="utf-8")

    assert [path.relative_to(tmp_path).as_posix() for path in iter_python_files(tmp_path)] == ["src/good.py"]


def test_index_is_incremental_updates_changed_files_and_deletes_stale_chunks(tmp_path: Path) -> None:
    source = tmp_path / "service.py"
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    embedder = FakeEmbeddingProvider(dimension=16)
    store = FakeVectorStore(dimension=16)
    indexer = CodeIndexer(tmp_path, embedder=embedder, store=store, repo_id="repo-a")

    first = indexer.build()
    second = indexer.build()
    source.write_text("def value():\n    return 2\n", encoding="utf-8")
    third = indexer.build()
    source.unlink()
    fourth = indexer.build()

    assert first.upserted_count == first.chunk_count > 0
    assert second.upserted_count == 0
    assert second.unchanged_count == first.chunk_count
    assert third.upserted_count > 0
    assert fourth.chunk_count == 0
    assert fourth.deleted_count == third.chunk_count
    assert embedder.document_batch_count == 2


def test_repo_isolation_filters_and_search_json_serialization(tmp_path: Path) -> None:
    repo_a = tmp_path / "a"
    repo_b = tmp_path / "b"
    repo_a.mkdir()
    repo_b.mkdir()
    (repo_a / "service.py").write_text("def invalidate_cache():\n    return 'cache invalidation'\n", encoding="utf-8")
    (repo_a / "test_service.py").write_text("def test_cache():\n    assert True\n", encoding="utf-8")
    (repo_b / "other.py").write_text("def invalidate_cache():\n    return 'other repository'\n", encoding="utf-8")
    embedder = FakeEmbeddingProvider(dimension=32)
    store = FakeVectorStore(dimension=32)
    CodeIndexer(repo_a, embedder=embedder, store=store, repo_id="repo-a").build()
    CodeIndexer(repo_b, embedder=embedder, store=store, repo_id="repo-b").build()
    search = SemanticCodeSearch(embedder=embedder, store=store, repo_id="repo-a")

    hits = search.search("cache invalidation", top_k=5, path_prefix="service", include_tests=False)

    assert hits
    assert all(hit.repo_id == "repo-a" for hit in hits)
    assert all(hit.path.startswith("service") for hit in hits)
    assert all(not hit.is_test for hit in hits)
    assert len(hits[0].bounded_preview) <= 600
    json.dumps([hit.to_dict() for hit in hits])


def test_index_requires_rebuild_when_model_or_dimension_changes(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    store = FakeVectorStore(dimension=16)
    CodeIndexer(
        tmp_path,
        embedder=FakeEmbeddingProvider(dimension=16, model_name="fake-a"),
        store=store,
        repo_id="repo-a",
    ).build()

    with pytest.raises(RebuildRequiredError):
        CodeIndexer(
            tmp_path,
            embedder=FakeEmbeddingProvider(dimension=16, model_name="fake-b"),
            store=store,
            repo_id="repo-a",
        ).build()

    rebuilt = CodeIndexer(
        tmp_path,
        embedder=FakeEmbeddingProvider(dimension=16, model_name="fake-b"),
        store=store,
        repo_id="repo-a",
    ).build(rebuild=True)
    assert rebuilt.model_name == "fake-b"


def test_qdrant_local_store_persists_and_keeps_repositories_isolated(tmp_path: Path) -> None:
    db = tmp_path / "qdrant"
    embedder = FakeEmbeddingProvider(dimension=16)
    store = QdrantLocalVectorStore(db, dimension=16, model_name=embedder.model_name)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text("def target():\n    return 'semantic target'\n", encoding="utf-8")
    CodeIndexer(repo, embedder=embedder, store=store, repo_id="repo-a").build()
    store.close()

    reopened = QdrantLocalVectorStore(db, dimension=16, model_name=embedder.model_name)
    hits = SemanticCodeSearch(embedder=embedder, store=reopened, repo_id="repo-a").search("semantic target")
    assert hits
    assert reopened.list_chunk_hashes("repo-b") == {}
    reopened.close()


@pytest.mark.skipif(os.environ.get("FIRSTCODER_RUN_FASTEMBED_TESTS") != "1", reason="cached model integration")
def test_fastembed_cached_model_returns_384_dimensions() -> None:
    embedder = FastEmbedProvider(
        cache_dir=Path.home() / "Library" / "Caches" / "firstcoder" / "fastembed",
        local_files_only=True,
    )

    documents = embedder.embed_documents(["parse pytest assertion failure"])
    query = embedder.embed_query("locate failing source")

    assert len(documents) == 1
    assert len(documents[0]) == 384
    assert len(query) == 384
