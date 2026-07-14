import json
from pathlib import Path

from benchmark.local_pytest.runner import (
    LocalPytestTask,
    build_parser,
    load_tasks_jsonl,
    materialize_task_repo,
    run_tasks,
    run_pytest,
    write_summary_json,
)
from benchmark.local_pytest.offline import evaluate_retrieval_candidates
from firstcoder.eval.tasks import CodingTask, CodingTaskResult
from firstcoder.retrieval import FakeEmbeddingProvider, FakeVectorStore


def test_load_tasks_jsonl_reads_local_task(tmp_path: Path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "demo",
                "title": "Demo Task",
                "files": {"src/demo.py": "VALUE = 1\n"},
                "problem_statement": "Make VALUE equal 2.",
                "test_command": "python -m pytest -q",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_tasks_jsonl(path) == [
        LocalPytestTask(
            id="demo",
            title="Demo Task",
            files={"src/demo.py": "VALUE = 1\n"},
            problem_statement="Make VALUE equal 2.",
            test_command="python -m pytest -q",
        )
    ]


def test_materialize_task_repo_creates_git_repo_with_files(tmp_path: Path):
    task = LocalPytestTask(
        id="demo",
        title="Demo Task",
        files={"src/demo.py": "VALUE = 1\n", "tests/test_demo.py": "def test_demo(): pass\n"},
        problem_statement="Fix it.",
    )

    repo = materialize_task_repo(task, tmp_path)

    assert (repo / ".git").exists()
    assert "__pycache__/" in (repo / ".gitignore").read_text(encoding="utf-8")
    assert (repo / "src" / "demo.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert run_pytest(repo, "python -m pytest -q").passed


def test_run_pytest_reports_failure(tmp_path: Path):
    task = LocalPytestTask(
        id="demo",
        title="Demo Task",
        files={"tests/test_demo.py": "def test_demo():\n    assert False\n"},
        problem_statement="Fix it.",
    )
    repo = materialize_task_repo(task, tmp_path)

    result = run_pytest(repo, "python -m pytest -q")

    assert result.passed is False
    assert result.returncode == 1
    assert "assert False" in result.output


def test_write_summary_json_serializes_rows(tmp_path: Path):
    out = tmp_path / "summary.json"

    write_summary_json(out, [{"id": "demo", "passed": True}])

    assert json.loads(out.read_text(encoding="utf-8")) == [{"id": "demo", "passed": True}]


def test_parser_defaults_to_sample_tasks():
    parser = build_parser()

    args = parser.parse_args(["--workdir", "runs/local-pytest"])

    assert args.tasks == "benchmark/local_pytest/tasks.sample.jsonl"
    assert args.max_tasks is None
    assert args.task_id == []


def test_parser_accepts_repeatable_task_ids():
    args = build_parser().parse_args(
        ["--workdir", "runs/local-pytest", "--task-id", "one", "--task-id", "two"]
    )

    assert args.task_id == ["one", "two"]


def test_run_tasks_scores_agent_changes_and_writes_summary(tmp_path: Path):
    class FixingAdapter:
        def run_task(self, task: CodingTask) -> CodingTaskResult:
            (task.repo_path / "src" / "demo.py").write_text("VALUE = 2\n", encoding="utf-8")
            return CodingTaskResult(
                instance_id=task.instance_id,
                model_name_or_path="fake",
                model_patch="",
                raw_response="done",
                runtime_metrics={
                    "provider_call_count": 2,
                    "tool_call_count": 1,
                    "tool_calls_by_name": {"view": 1},
                    "actual_input_tokens": None,
                    "actual_output_tokens": None,
                    "actual_total_tokens": None,
                    "estimated_input_tokens": 17,
                    "compaction_event_count": 0,
                    "compaction_trigger_counts": {},
                    "compaction_tokens_before": 0,
                    "compaction_tokens_after": 0,
                    "archive_count": 0,
                    "archive_retrieval_count": 0,
                    "source_read_count": 1,
                },
            )

    task = LocalPytestTask(
        id="demo",
        title="Demo Task",
        files={
            "src/demo.py": "VALUE = 1\n",
            "tests/test_demo.py": "from src.demo import VALUE\n\n\ndef test_value():\n    assert VALUE == 2\n",
        },
        problem_statement="Make VALUE equal 2.",
    )
    summary = tmp_path / "summary.json"

    rows = run_tasks(
        tasks=[task],
        workdir=tmp_path / "work",
        summary_out=summary,
        adapter=FixingAdapter(),
    )

    assert rows[0]["passed"] is True
    assert "+VALUE = 2" in rows[0]["model_patch"]
    assert rows[0]["final_diff"] == rows[0]["model_patch"]
    assert rows[0]["provider_call_count"] == 2
    assert rows[0]["tool_calls_by_name"] == {"view": 1}
    assert rows[0]["actual_total_tokens"] is None
    assert rows[0]["estimated_input_tokens"] == 17
    assert rows[0]["context_metrics"]["source_read_count"] == 1
    assert rows[0]["diff_file_count"] == 1
    assert rows[0]["diff_added_lines"] == 1
    assert rows[0]["diff_deleted_lines"] == 1
    written = json.loads(summary.read_text(encoding="utf-8"))[0]
    assert written["passed"] is True
    assert written["actual_total_tokens"] is None
    assert written["context_metrics"]["provider_call_count"] == 2


def test_run_tasks_keeps_legacy_result_defaults_json_serializable(tmp_path: Path):
    class LegacyAdapter:
        def run_task(self, task: CodingTask) -> CodingTaskResult:
            return CodingTaskResult(instance_id=task.instance_id, model_name_or_path="fake", model_patch="")

    task = LocalPytestTask(
        id="legacy",
        title="Legacy",
        files={"tests/test_ok.py": "def test_ok():\n    assert True\n"},
        problem_statement="No change needed.",
    )
    summary = tmp_path / "legacy.json"

    rows = run_tasks(
        tasks=[task],
        workdir=tmp_path / "work",
        summary_out=summary,
        adapter=LegacyAdapter(),
    )

    assert rows[0]["provider_call_count"] == 0
    assert rows[0]["tool_call_count"] == 0
    assert rows[0]["actual_total_tokens"] is None
    assert rows[0]["context_metrics"]["provider_call_count"] == 0
    assert json.loads(summary.read_text(encoding="utf-8"))[0]["id"] == "legacy"


def test_sample_suite_contains_nine_reproducible_initial_failures(tmp_path: Path) -> None:
    tasks = load_tasks_jsonl("benchmark/local_pytest/tasks.sample.jsonl")

    assert [task.id for task in tasks] == [
        "username_normalization",
        "invoice_rounding",
        "optional_config",
        "validation_contract",
        "cache_invalidation",
        "import_or_collection_error",
        "service_repository_contract",
        "parser_dispatch",
        "full_suite_regression",
    ]
    assert sum("single_file" in task.tags for task in tasks) >= 4
    assert sum("multi_file" in task.tags for task in tasks) >= 3
    assert sum("semantic" in task.tags for task in tasks) >= 2
    assert sum("focused_full_regression" in task.tags for task in tasks) >= 1
    for task in tasks:
        repo = materialize_task_repo(task, tmp_path)
        initial = run_pytest(repo, task.test_command)
        assert initial.passed is False, task.id
        assert task.editable_paths
        assert all(not path.startswith("tests/") for path in task.editable_paths)


def test_evaluator_rejects_test_modification_and_out_of_scope_write(tmp_path: Path) -> None:
    class TamperingAdapter:
        def run_task(self, task: CodingTask) -> CodingTaskResult:
            (task.repo_path / "tests" / "test_demo.py").write_text(
                "def test_value():\n    assert True\n", encoding="utf-8"
            )
            (task.repo_path / "README.md").write_text("out of scope\n", encoding="utf-8")
            return CodingTaskResult(instance_id=task.instance_id, model_name_or_path="fake", model_patch="")

    task = LocalPytestTask(
        id="tamper",
        title="Tamper",
        files={
            "src/demo.py": "VALUE = 1\n",
            "tests/test_demo.py": "from src.demo import VALUE\n\ndef test_value():\n    assert VALUE == 2\n",
        },
        problem_statement="Fix source.",
        editable_paths=("src/demo.py",),
    )

    row = run_tasks(
        tasks=[task],
        workdir=tmp_path / "work",
        summary_out=tmp_path / "summary.json",
        adapter=TamperingAdapter(),
    )[0]

    assert row["passed"] is False
    assert row["initial_failure_confirmed"] is True
    assert row["test_file_modified"] is True
    assert row["out_of_scope_write"] is True
    assert row["infrastructure_pass"] is False


def test_baseline_and_vector_candidate_modes_use_same_failure_and_vector_hits_semantic_file(tmp_path: Path) -> None:
    task = next(
        task for task in load_tasks_jsonl("benchmark/local_pytest/tasks.sample.jsonl") if task.id == "parser_dispatch"
    )
    repo = materialize_task_repo(task, tmp_path)
    initial = run_pytest(repo, task.test_command)
    embedder = FakeEmbeddingProvider(dimension=64)
    store = FakeVectorStore(dimension=64)

    comparison = evaluate_retrieval_candidates(
        repo,
        task=task,
        pytest_output=initial.output,
        embedder=embedder,
        store=store,
    )

    assert comparison["initial_failure_fingerprint"]
    assert comparison["baseline"]["mode"] == "baseline"
    assert comparison["vector"]["mode"] == "vector"
    assert comparison["baseline"]["relevant_file_hit_at_5"] is False
    assert comparison["vector"]["relevant_file_hit_at_5"] is True
    assert comparison["vector"]["query_latency_seconds"] >= 0
    json.dumps(comparison)


def test_runner_stops_and_persists_when_cost_limit_is_reached(tmp_path: Path) -> None:
    class CostingAdapter:
        def run_task(self, task: CodingTask) -> CodingTaskResult:
            return CodingTaskResult(
                instance_id=task.instance_id,
                model_name_or_path="fake",
                model_patch="",
                runtime_metrics={"estimated_cost_usd": 0.6},
            )

    tasks = [
        LocalPytestTask(
            id=f"cost-{index}",
            title="Cost",
            files={"tests/test_fail.py": "def test_fail():\n    assert False\n"},
            problem_statement="Do not fix.",
        )
        for index in range(3)
    ]
    summary = tmp_path / "summary.json"

    rows = run_tasks(
        tasks=tasks,
        workdir=tmp_path / "work",
        summary_out=summary,
        adapter=CostingAdapter(),
        cost_limit_usd=1.0,
    )

    assert len(rows) == 2
    assert rows[-1]["cumulative_estimated_cost_usd"] == 1.2
    assert rows[-1]["budget_stop_reason"] == "cost_limit_reached"
    assert json.loads(summary.read_text(encoding="utf-8")) == rows
