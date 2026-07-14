"""Offline deterministic/vector retrieval comparison for local pytest tasks."""

from __future__ import annotations

import time
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
    return {
        "initial_failure_fingerprint": fingerprint,
        "baseline": {
            "mode": "baseline",
            "candidates": deterministic[:5],
            "relevant_file_hit_at_5": bool(relevant.intersection(deterministic[:5])),
        },
        "vector": {
            "mode": "vector",
            "candidates": vector_paths,
            "relevant_file_hit_at_5": bool(relevant.intersection(vector_paths)),
            "index_report": index_report.to_dict(),
            "query_latency_seconds": query_seconds,
            "hit_paths": [hit.path for hit in hits],
            "scores": [round(hit.score, 6) for hit in hits],
        },
    }


def _deterministic_candidates(report: Any) -> list[str]:
    paths: list[str] = []
    for failure in report.failures:
        paths.extend(location.path for location in failure.source_locations)
        if "::" in failure.node_id:
            paths.append(failure.node_id.split("::", 1)[0])
    return _unique(paths)


def _failure_query(report: Any) -> str:
    return " ".join(failure.message for failure in report.failures if failure.message).strip()


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
