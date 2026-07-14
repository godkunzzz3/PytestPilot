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
from firstcoder.eval.tasks import CodingTask, CodingTaskResult


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
