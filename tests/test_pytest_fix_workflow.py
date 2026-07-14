import json
import subprocess
from pathlib import Path

from firstcoder.eval.adapter import FirstCoderCodingAgentAdapter
from firstcoder.eval.tasks import CodingTask, CodingTaskResult
from firstcoder.providers.base import ChatProvider
from firstcoder.providers.types import ChatRequest, ChatResponse, ToolCall
from firstcoder.retrieval import RetrievalUnavailableError
from firstcoder.workflows.pytest_fix import PytestFixWorkflow


class NeverAdapter:
    def run_task(self, task: CodingTask) -> CodingTaskResult:
        raise AssertionError("agent must not run when baseline already passes")


class ScriptedAdapter:
    def __init__(self, actions):
        self.actions = list(actions)
        self.tasks: list[CodingTask] = []

    def run_task(self, task: CodingTask) -> CodingTaskResult:
        self.tasks.append(task)
        action = self.actions.pop(0)
        action(task.repo_path)
        transcript = task.repo_path.parent / f"{task.instance_id}.jsonl"
        _write_read_then_mutation_transcript(transcript)
        return CodingTaskResult(
            instance_id=task.instance_id,
            model_name_or_path="fake",
            model_patch="",
            transcript_path=transcript,
            runtime_metrics={
                "provider_call_count": 1,
                "tool_call_count": 2,
                "tool_calls_by_name": {"edit": 1, "view": 1},
                "actual_input_tokens": None,
                "actual_output_tokens": None,
                "actual_total_tokens": None,
                "estimated_input_tokens": 10,
                "source_read_count": 1,
            },
        )


class FakeRepairProvider(ChatProvider):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-repair"

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.calls == 1:
            return ChatResponse(
                provider=self.name,
                model=self.model,
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(id="read", name="view", arguments={"path": "src/value.py"})],
            )
        if self.calls == 2:
            return ChatResponse(
                provider=self.name,
                model=self.model,
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="edit",
                        name="edit",
                        arguments={"path": "src/value.py", "old": "VALUE = 1", "new": "VALUE = 2"},
                    )
                ],
            )
        return ChatResponse(provider=self.name, model=self.model, content="done", finish_reason="stop")


def _write_project(root: Path, *, value: int = 1, expected: int = 2, hidden_failure: bool = False) -> None:
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "value.py").write_text(f"VALUE = {value}\n", encoding="utf-8")
    (root / "tests" / "test_value.py").write_text(
        f"from src.value import VALUE\n\ndef test_value():\n    assert VALUE == {expected}\n",
        encoding="utf-8",
    )
    if hidden_failure:
        (root / "tests" / "test_z_hidden.py").write_text("def test_hidden():\n    assert False\n", encoding="utf-8")


def _init_git(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=root, check=True, capture_output=True)


def _fix(root: Path) -> None:
    (root / "src" / "value.py").write_text("VALUE = 2\n", encoding="utf-8")


def _noop(root: Path) -> None:
    return None


def _write_read_then_mutation_transcript(path: Path, *, read_first: bool = True) -> None:
    read = {
        "type": "tool_result",
        "payload": {"parts": [{"kind": "tool_result", "metadata": {"tool_name": "view", "ok": True, "data": {}}}]},
    }
    mutation = {
        "type": "tool_result",
        "payload": {"parts": [{"kind": "tool_result", "metadata": {"tool_name": "edit", "ok": True, "data": {}}}]},
    }
    events = [read, mutation] if read_first else [mutation, read]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def test_baseline_passes_without_calling_agent(tmp_path: Path) -> None:
    _write_project(tmp_path, value=2)
    result = PytestFixWorkflow(tmp_path, adapter=NeverAdapter()).run(test_command="python -m pytest -q")

    assert result.status == "passed"
    assert result.exit_code == 0
    assert result.attempts == []


def test_fake_provider_e2e_reads_source_edits_and_passes_focused_and_full(tmp_path: Path) -> None:
    _write_project(tmp_path)
    _init_git(tmp_path)
    provider = FakeRepairProvider()
    adapter = FirstCoderCodingAgentAdapter(
        provider_factory=lambda provider_name: provider,
        provider_retries=0,
        session_root=tmp_path.parent / "sessions",
    )

    result = PytestFixWorkflow(tmp_path, adapter=adapter).run(test_command="python -m pytest -q")

    assert result.status == "passed"
    assert result.exit_code == 0
    assert len(result.attempts) == 1
    assert result.attempts[0].focused_result.passed is True
    assert result.attempts[0].full_result is not None
    assert result.attempts[0].full_result.passed is True
    assert result.attempts[0].source_read_policy_violation is False
    assert provider.calls == 3
    assert "+VALUE = 2" in result.final_diff


def test_second_attempt_can_succeed(tmp_path: Path) -> None:
    _write_project(tmp_path)
    adapter = ScriptedAdapter([_noop, _fix])

    result = PytestFixWorkflow(tmp_path, adapter=adapter, max_attempts=2).run(test_command="python -m pytest -q")

    assert result.status == "passed"
    assert len(result.attempts) == 2
    assert result.attempts[0].focused_result.passed is False
    assert result.attempts[1].full_result is not None and result.attempts[1].full_result.passed


def test_attempt_limit_returns_two_when_tests_still_fail(tmp_path: Path) -> None:
    _write_project(tmp_path)
    result = PytestFixWorkflow(tmp_path, adapter=ScriptedAdapter([_noop, _noop]), max_attempts=2).run(
        test_command="python -m pytest -q"
    )

    assert result.status == "failed"
    assert result.exit_code == 2
    assert len(result.attempts) == 2


def test_focused_pass_but_full_suite_failure_is_not_success(tmp_path: Path) -> None:
    _write_project(tmp_path, hidden_failure=True)
    result = PytestFixWorkflow(tmp_path, adapter=ScriptedAdapter([_fix, _noop]), max_attempts=2).run(
        test_command="python -m pytest -q"
    )

    assert result.status == "failed"
    assert result.attempts[0].focused_result.passed is True
    assert result.attempts[0].full_result is not None
    assert result.attempts[0].full_result.passed is False


def test_qdrant_unavailable_falls_back_to_deterministic_candidates(tmp_path: Path) -> None:
    _write_project(tmp_path)

    class UnavailableSearch:
        def search(self, *args, **kwargs):
            raise RetrievalUnavailableError("offline")

    adapter = ScriptedAdapter([_fix])
    result = PytestFixWorkflow(tmp_path, adapter=adapter, semantic_search=UnavailableSearch()).run(
        test_command="python -m pytest -q"
    )

    assert result.status == "passed"
    assert result.attempts[0].semantic_candidates == []
    assert "src/value.py" in result.attempts[0].deterministic_candidates


def test_mutation_without_prior_source_read_records_policy_violation(tmp_path: Path) -> None:
    _write_project(tmp_path)

    class ViolatingAdapter(ScriptedAdapter):
        def run_task(self, task: CodingTask) -> CodingTaskResult:
            _fix(task.repo_path)
            transcript = task.repo_path.parent / "violation.jsonl"
            _write_read_then_mutation_transcript(transcript, read_first=False)
            return CodingTaskResult(
                instance_id=task.instance_id,
                model_name_or_path="fake",
                model_patch="",
                transcript_path=transcript,
            )

    result = PytestFixWorkflow(tmp_path, adapter=ViolatingAdapter([])).run(test_command="python -m pytest -q")

    assert result.status == "passed"
    assert result.attempts[0].source_read_policy_violation is True


def test_result_json_artifact_is_serializable(tmp_path: Path) -> None:
    _write_project(tmp_path)
    out = tmp_path.parent / "result.json"

    result = PytestFixWorkflow(tmp_path, adapter=ScriptedAdapter([_fix])).run(
        test_command="python -m pytest -q",
        json_out=out,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == result.to_dict()
    assert payload["metrics"]["provider_call_count"] == 1
    assert payload["metrics"]["actual_total_tokens"] is None
