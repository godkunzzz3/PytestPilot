"""Local pytest benchmark runner for FirstCoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from firstcoder.eval.adapter import FirstCoderCodingAgentAdapter
from firstcoder.agent.loop_limits import AgentLoopLimits
from firstcoder.eval.metrics import collect_diff_metrics, empty_runtime_metrics
from firstcoder.eval.patch import collect_git_diff
from firstcoder.eval.tasks import CodingTask, CodingTaskResult

DEFAULT_TASKS = "benchmark/local_pytest/tasks.sample.jsonl"
DEFAULT_TEST_COMMAND = "python -m pytest -q"


@dataclass(frozen=True, slots=True)
class LocalPytestTask:
    id: str
    title: str
    files: dict[str, str]
    problem_statement: str
    test_command: str = DEFAULT_TEST_COMMAND
    editable_paths: tuple[str, ...] = ()
    relevant_files: tuple[str, ...] = ()
    semantic_query: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PytestResult:
    passed: bool
    returncode: int
    output: str


class LocalAgentAdapter(Protocol):
    def run_task(self, task: CodingTask) -> CodingTaskResult:
        ...


def load_tasks_jsonl(path: str | Path) -> list[LocalPytestTask]:
    tasks: list[LocalPytestTask] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            data = json.loads(line)
            tasks.append(
                LocalPytestTask(
                    id=str(data["id"]),
                    title=str(data.get("title") or data["id"]),
                    files={str(name): str(content) for name, content in dict(data["files"]).items()},
                    problem_statement=str(data["problem_statement"]),
                    test_command=str(data.get("test_command") or DEFAULT_TEST_COMMAND),
                    editable_paths=tuple(str(path) for path in data.get("editable_paths", ())),
                    relevant_files=tuple(str(path) for path in data.get("relevant_files", ())),
                    semantic_query=str(data["semantic_query"]) if data.get("semantic_query") else None,
                    tags=tuple(str(tag) for tag in data.get("tags", ())),
                )
            )
    return tasks


def materialize_task_repo(task: LocalPytestTask, workdir: str | Path, *, force: bool = False) -> Path:
    repo = Path(workdir) / task.id
    if repo.exists():
        if not force:
            raise RuntimeError(f"Task repository already exists: {repo}. Use --force to recreate it.")
        shutil.rmtree(repo)
    repo.mkdir(parents=True)
    for relative_path, content in task.files.items():
        _write_task_file(repo, relative_path, content)
    _write_task_file(repo, ".gitignore", "__pycache__/\n*.py[cod]\n.pytest_cache/\n")
    _init_git_repo(repo)
    return repo


def run_pytest(repo: str | Path, command: str) -> PytestResult:
    result = subprocess.run(
        _split_command(command),
        cwd=Path(repo),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return PytestResult(
        passed=result.returncode == 0,
        returncode=result.returncode,
        output=result.stdout or "",
    )


def run_tasks(
    *,
    tasks: list[LocalPytestTask],
    workdir: str | Path,
    summary_out: str | Path,
    max_tasks: int | None = None,
    provider_name: str | None = None,
    model_name: str = "firstcoder-local-pytest",
    session_root: str | Path = ".firstcoder-local-pytest",
    force: bool = False,
    adapter: LocalAgentAdapter | None = None,
    retrieval_mode: str = "baseline",
    provider_retries: int = 0,
    max_tool_rounds: int = 8,
    cost_limit_usd: float | None = None,
    starting_cost_usd: float = 0.0,
) -> list[dict[str, Any]]:
    selected = tasks[:max_tasks] if max_tasks is not None else tasks
    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)
    vector_factory = (
        _VectorToolFactory(workdir_path.parent / ".firstcoder-vector-index")
        if retrieval_mode == "vector" and adapter is None
        else None
    )
    agent = adapter or FirstCoderCodingAgentAdapter(
        model_name_or_path=model_name,
        provider_name=provider_name,
        session_root=session_root,
        provider_retries=provider_retries,
        limits=AgentLoopLimits.swe_lite().with_max_tool_rounds(max_tool_rounds),
        extra_tools_factory=vector_factory,
    )
    rows: list[dict[str, Any]] = []
    cumulative_cost = starting_cost_usd
    try:
        for task in selected:
            row = run_one_task(
                task=task,
                workdir=workdir_path,
                adapter=agent,
                force=force,
                retrieval_mode=retrieval_mode,
            )
            task_cost = row.get("estimated_cost_usd")
            if isinstance(task_cost, (int, float)):
                cumulative_cost = round(cumulative_cost + float(task_cost), 9)
            row["cumulative_estimated_cost_usd"] = cumulative_cost
            row["cost_limit_usd"] = cost_limit_usd
            row["budget_stop_reason"] = None
            if cost_limit_usd is not None and task_cost is None:
                row["budget_stop_reason"] = "usage_missing"
            elif cost_limit_usd is not None and cumulative_cost >= cost_limit_usd:
                row["budget_stop_reason"] = "cost_limit_reached"
            rows.append(row)
            write_summary_json(summary_out, rows)
            if row["budget_stop_reason"] is not None:
                break
    finally:
        if vector_factory is not None:
            vector_factory.close()
    return rows


def run_one_task(
    *,
    task: LocalPytestTask,
    workdir: Path,
    adapter: LocalAgentAdapter,
    force: bool,
    retrieval_mode: str = "baseline",
) -> dict[str, Any]:
    started_at = time.time()
    repo = materialize_task_repo(task, workdir, force=force)
    tests_before = _test_file_hashes(repo)
    initial_result = run_pytest(repo, task.test_command)
    coding_task = CodingTask(
        instance_id=task.id,
        repo_path=repo,
        problem_statement=_build_problem_statement(task),
        metadata={
            "benchmark": "local_pytest",
            "title": task.title,
            "test_command": task.test_command,
            "editable_paths": list(task.editable_paths),
            "retrieval_mode": retrieval_mode,
            "semantic_query": task.semantic_query,
            "relevant_files": list(task.relevant_files),
        },
    )
    result = adapter.run_task(coding_task)
    pytest_result = run_pytest(repo, task.test_command)
    final_diff = collect_git_diff(repo, include_untracked=True)
    changed_paths = _changed_paths(repo)
    tests_after = _test_file_hashes(repo)
    test_file_modified = tests_before != tests_after
    allowed = set(task.editable_paths)
    out_of_scope_write = bool(allowed and any(path not in allowed for path in changed_paths))
    infrastructure_pass = initial_result.returncode != 0 and not test_file_modified and not out_of_scope_write
    runtime_metrics = empty_runtime_metrics()
    runtime_metrics.update(result.runtime_metrics)
    row = {
        "id": task.id,
        "title": task.title,
        "repo_path": str(repo),
        "passed": infrastructure_pass and pytest_result.passed,
        "returncode": pytest_result.returncode,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "test_command": task.test_command,
        "pytest_output": pytest_result.output,
        "initial_pytest_output": initial_result.output,
        "initial_returncode": initial_result.returncode,
        "initial_failure_confirmed": initial_result.returncode != 0,
        "infrastructure_pass": infrastructure_pass,
        "test_file_modified": test_file_modified,
        "out_of_scope_write": out_of_scope_write,
        "changed_paths": sorted(changed_paths),
        "editable_paths": list(task.editable_paths),
        "retrieval_mode": retrieval_mode,
        "relevant_file_hit_at_5": result.runtime_metrics.get("relevant_file_hit_at_5"),
        "source_read_policy_violation": result.runtime_metrics.get("source_read_policy_violation", False),
        "transcript_path": str(result.transcript_path) if result.transcript_path else None,
        "raw_response": result.raw_response,
        "model_patch": final_diff,
        "final_diff": final_diff,
        "context_metrics": dict(runtime_metrics),
    }
    row.update(runtime_metrics)
    row.update(collect_diff_metrics(final_diff))
    row["full_test_exit_code"] = row["returncode"]
    if row["passed"]:
        row["failure_category"] = None
    elif row["test_file_modified"]:
        row["failure_category"] = "test_modified"
    elif row["out_of_scope_write"]:
        row["failure_category"] = "out_of_scope_write"
    elif row["source_read_policy_violation"]:
        row["failure_category"] = "source_read_policy_violation"
    else:
        row["failure_category"] = "tests_failed"
    return row


def write_summary_json(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FirstCoder on small local pytest coding tasks.")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="JSONL file containing local pytest tasks.")
    parser.add_argument("--workdir", required=True, help="Directory where task repositories are created.")
    parser.add_argument("--summary-out", default="runs/local-pytest-summary.json", help="JSON summary output path.")
    parser.add_argument("--max-tasks", type=_positive_int, default=None, help="Limit number of tasks.")
    parser.add_argument("--task-id", action="append", default=[], help="Run only this task id; repeatable.")
    parser.add_argument("--provider", default=None, help="FirstCoder provider name. Defaults to app config.")
    parser.add_argument("--model-name", default="firstcoder-local-pytest", help="Model name recorded in sessions.")
    parser.add_argument("--session-root", default=".firstcoder-local-pytest", help="Directory for benchmark sessions.")
    parser.add_argument("--force", action="store_true", help="Recreate existing task repositories.")
    parser.add_argument("--retrieval-mode", choices=("baseline", "vector"), default="baseline")
    parser.add_argument("--provider-retries", type=int, default=0)
    parser.add_argument("--max-tool-rounds", type=_positive_int, default=8)
    parser.add_argument("--cost-limit-usd", type=float, default=None)
    parser.add_argument("--starting-cost-usd", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tasks = load_tasks_jsonl(args.tasks)
        if args.task_id:
            requested = set(args.task_id)
            tasks = [task for task in tasks if task.id in requested]
            missing = requested.difference(task.id for task in tasks)
            if missing:
                raise RuntimeError(f"Unknown task ids: {', '.join(sorted(missing))}")
        rows = run_tasks(
            tasks=tasks,
            workdir=args.workdir,
            summary_out=args.summary_out,
            max_tasks=args.max_tasks,
            provider_name=args.provider,
            model_name=args.model_name,
            session_root=args.session_root,
            force=args.force,
            retrieval_mode=args.retrieval_mode,
            provider_retries=args.provider_retries,
            max_tool_rounds=args.max_tool_rounds,
            cost_limit_usd=args.cost_limit_usd,
            starting_cost_usd=args.starting_cost_usd,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    passed = sum(1 for row in rows if row["passed"])
    print(f"Wrote local pytest benchmark summary: {args.summary_out}")
    print(f"Passed {passed}/{len(rows)}")
    return 0 if passed == len(rows) else 2


def _write_task_file(repo: Path, relative_path: str, content: str) -> None:
    path = (repo / relative_path).resolve()
    if repo.resolve() not in path.parents:
        raise RuntimeError(f"Task file path escapes repo: {relative_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _init_git_repo(repo: Path) -> None:
    _run_git(["init"], repo)
    _run_git(["config", "user.email", "benchmark@example.com"], repo)
    _run_git(["config", "user.name", "Benchmark"], repo)
    _run_git(["add", "-A"], repo)
    _run_git(["commit", "-m", "initial task"], repo)


def _run_git(args: list[str], repo: Path) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _test_file_hashes(repo: Path) -> dict[str, str]:
    return {
        path.relative_to(repo).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((repo / "tests").rglob("*.py"))
        if path.is_file()
    }


def _changed_paths(repo: Path) -> set[str]:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    paths: set[str] = set()
    for line in result.stdout.splitlines():
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.add(path)
    return paths


def _split_command(command: str) -> list[str]:
    parts = command.split()
    if not parts:
        raise RuntimeError("test_command cannot be empty")
    if parts[:2] in (["python", "-m"], ["python3", "-m"]):
        parts[0] = sys.executable
    return parts


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _build_problem_statement(task: LocalPytestTask) -> str:
    return (
        f"Task: {task.title}\n\n"
        f"{task.problem_statement.strip()}\n\n"
        f"Validation command: {task.test_command}\n"
        "Edit the repository files until the validation command passes. Keep the fix minimal."
    )


class _VectorToolFactory:
    """Build per-task local Qdrant indexes outside the task repositories."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root.resolve()
        self._stores: list[Any] = []

    def __call__(self, task: CodingTask) -> list[Any]:
        from firstcoder.retrieval import (
            CodeIndexer,
            FastEmbedProvider,
            QdrantLocalVectorStore,
            SemanticCodeSearch,
            repository_id,
        )
        from firstcoder.tools.code_search import create_code_search_tool

        cache_dir = Path(
            os.environ.get(
                "FIRSTCODER_FASTEMBED_CACHE",
                Path.home() / "Library" / "Caches" / "firstcoder" / "fastembed",
            )
        )
        embedder = FastEmbedProvider(cache_dir=cache_dir, local_files_only=True)
        store = QdrantLocalVectorStore(
            self.data_root / task.instance_id,
            dimension=embedder.dimension,
            model_name=embedder.model_name,
        )
        repo_id = repository_id(task.repo_path)
        CodeIndexer(task.repo_path, embedder=embedder, store=store, repo_id=repo_id).build(rebuild=True)
        self._stores.append(store)
        search = SemanticCodeSearch(embedder=embedder, store=store, repo_id=repo_id)
        return [create_code_search_tool(task.repo_path, search=search)]

    def close(self) -> None:
        for store in self._stores:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
