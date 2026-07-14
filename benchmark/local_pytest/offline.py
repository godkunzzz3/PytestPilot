"""Offline deterministic/vector retrieval comparison for local pytest tasks."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.local_pytest.runner import LocalPytestTask
from firstcoder.ci import parse_pytest_output
from firstcoder.retrieval import CodeIndexer, EmbeddingProvider, SemanticCodeSearch, VectorStore


def evaluate_retrieval_candidates(
    repo: str | Path,
    *,
    task: LocalPytestTask,
    pytest_output: str,
    embedder: EmbeddingProvider,
    store: VectorStore,
) -> dict[str, Any]:
    root = Path(repo).resolve()
    report = parse_pytest_output(pytest_output, command=task.test_command, exit_code=1)
    deterministic = _deterministic_candidates(report)
    indexer = CodeIndexer(root, embedder=embedder, store=store, repo_id=task.id)
    index_report = indexer.build(rebuild=True)
    query = task.semantic_query or _failure_query(report) or task.problem_statement
    started = time.perf_counter()
    hits = SemanticCodeSearch(embedder=embedder, store=store, repo_id=task.id).search(
        query, top_k=5, include_tests=False
    )
    query_seconds = round(time.perf_counter() - started, 6)
    vector_paths = _unique([hit.path for hit in hits])[:5]
    relevant = set(task.relevant_files)
    fingerprint = report.failures[0].fingerprint if report.failures else None
    baseline_top5 = [
        {
            **candidate,
            "hit": candidate["path"] in relevant,
        }
        for candidate in deterministic[:5]
    ]
    vector_top5 = [
        {
            "path": hit.path,
            "symbol": hit.symbol,
            "score": round(hit.score, 6),
            "start_line": hit.start_line,
            "end_line": hit.end_line,
            "hit": hit.path in relevant,
        }
        for hit in hits[:5]
    ]
    return {
        "task_id": task.id,
        "query": query,
        "expected_relevant_paths": list(task.relevant_files),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "initial_failure_fingerprint": fingerprint,
        "baseline": {
            "mode": "baseline",
            "candidates": [item["path"] for item in baseline_top5],
            "top_5": baseline_top5,
            "relevant_file_hit_at_5": any(item["hit"] for item in baseline_top5),
        },
        "vector": {
            "mode": "vector",
            "candidates": vector_paths,
            "top_5": vector_top5,
            "relevant_file_hit_at_5": bool(relevant.intersection(vector_paths)),
            "index_report": index_report.to_dict(),
            "query_latency_seconds": query_seconds,
            "hit_paths": [hit.path for hit in hits],
            "scores": [round(hit.score, 6) for hit in hits],
        },
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "vector_store": type(store).__name__,
        "qdrant_mode": "local-persistent" if type(store).__name__ == "QdrantLocalVectorStore" else "in-memory-fake",
        "indexed_chunk_count": index_report.chunk_count,
        "index_latency_seconds": index_report.elapsed_seconds,
    }


def _deterministic_candidates(report: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for failure in report.failures:
        candidates.extend(
            {
                "path": location.path,
                "symbol": location.function,
                "score": None,
                "start_line": location.line,
                "end_line": location.line,
            }
            for location in failure.source_locations
        )
        if "::" in failure.node_id:
            candidates.append(
                {
                    "path": failure.node_id.split("::", 1)[0],
                    "symbol": failure.node_id.rsplit("::", 1)[-1],
                    "score": None,
                    "start_line": None,
                    "end_line": None,
                }
            )
    seen: set[str] = set()
    return [item for item in candidates if not (item["path"] in seen or seen.add(item["path"]))]


def _failure_query(report: Any) -> str:
    return " ".join(failure.message for failure in report.failures if failure.message).strip()


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
