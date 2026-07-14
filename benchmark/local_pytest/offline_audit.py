"""Regenerate raw Top-5 retrieval evidence without creating a Chat Provider."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.local_pytest.offline import evaluate_retrieval_candidates
from benchmark.local_pytest.runner import load_tasks_jsonl, materialize_task_repo, run_pytest
from firstcoder.retrieval import FastEmbedProvider, QdrantLocalVectorStore


TASK_IDS = ("service_repository_contract", "parser_dispatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate offline retrieval Top-5 audit evidence.")
    parser.add_argument("--tasks", default="benchmark/local_pytest/tasks.sample.jsonl")
    parser.add_argument("--out", default="runs/audit-hardening/offline-retrieval-results.json")
    parser.add_argument("--workdir", default="runs/audit-hardening/offline-work")
    parser.add_argument("--index-dir", default="runs/audit-hardening/offline-qdrant")
    args = parser.parse_args(argv)

    tasks = {task.id: task for task in load_tasks_jsonl(args.tasks)}
    cache_dir = Path(
        os.environ.get(
            "FIRSTCODER_FASTEMBED_CACHE",
            Path.home() / "Library" / "Caches" / "firstcoder" / "fastembed",
        )
    )
    embedder = FastEmbedProvider(cache_dir=cache_dir, local_files_only=True)
    rows = []
    for task_id in TASK_IDS:
        task = tasks[task_id]
        repo = materialize_task_repo(task, args.workdir, force=True)
        initial = run_pytest(repo, task.test_command)
        store = QdrantLocalVectorStore(
            Path(args.index_dir) / task_id,
            dimension=embedder.dimension,
            model_name=embedder.model_name,
        )
        try:
            rows.append(
                evaluate_retrieval_candidates(
                    repo,
                    task=task,
                    pytest_output=initial.output,
                    embedder=embedder,
                    store=store,
                )
            )
        finally:
            store.close()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote offline retrieval audit: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
