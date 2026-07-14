import json
from pathlib import Path
import subprocess

import pytest

from firstcoder.agent.loop_limits import AgentLoopLimits
from firstcoder.eval.adapter import FirstCoderCodingAgentAdapter
from firstcoder.eval.adapter import RetryableBenchmarkProvider
from firstcoder.eval.metrics import collect_benchmark_policy_metrics, collect_context_metrics
from firstcoder.providers.base import ChatProvider
from firstcoder.eval.tasks import CodingTask
from firstcoder.context.events import SessionEvent
from firstcoder.context.store import JsonlSessionStore
from firstcoder.providers.errors import ProviderError, ProviderErrorKind
from firstcoder.providers.types import ChatRequest, ToolCall
from firstcoder.providers.types import ChatResponse


class FakeLoop:
    def __init__(self):
        self.messages: list[str] = []

    def run_user_turn(self, content: str) -> ChatResponse:
        self.messages.append(content)
        return ChatResponse(
            provider="fake",
            model="fake-model",
            content="done",
            finish_reason="stop",
        )


class FileWritingLoop:
    def __init__(self, repo: Path):
        self.repo = repo

    def run_user_turn(self, content: str) -> ChatResponse:
        (self.repo / "new_module.py").write_text("NEW_VALUE = 3\n", encoding="utf-8")
        return ChatResponse(
            provider="fake",
            model="fake-model",
            content="done",
            finish_reason="stop",
        )


class FakeProvider(ChatProvider):
    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(provider=self.name, model=self.model, content="done", finish_reason="stop")


class PatchProvider(ChatProvider):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.calls == 1:
            return ChatResponse(
                provider=self.name,
                model=self.model,
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_patch",
                        name="apply_patch",
                        arguments={
                            "patch": (
                                "*** Begin Patch\n"
                                "*** Add File: fixed.py\n"
                                "+VALUE = 42\n"
                                "*** End Patch"
                            )
                        },
                    )
                ],
            )
        return ChatResponse(provider=self.name, model=self.model, content="done", finish_reason="stop")


class FlakyProvider(ChatProvider):
    def __init__(self, failures: int, *, kind: ProviderErrorKind = ProviderErrorKind.SERVER_ERROR) -> None:
        self.failures = failures
        self.kind = kind
        self.calls = 0

    @property
    def name(self) -> str:
        return "flaky"

    @property
    def model(self) -> str:
        return "flaky-model"

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderError(self.kind, "temporary provider failure")
        return ChatResponse(provider=self.name, model=self.model, content="done", finish_reason="stop")


def test_benchmark_policy_metrics_track_vector_hit_and_mutation_before_read(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    events = [
        {
            "type": "tool_result",
            "payload": {
                "parts": [
                    {
                        "kind": "tool_result",
                        "metadata": {
                            "tool_name": "code_search",
                            "ok": True,
                            "data": {"hits": [{"path": "src/target.py"}]},
                        },
                    }
                ]
            },
        },
        {
            "type": "tool_result",
            "payload": {
                "parts": [{"kind": "tool_result", "metadata": {"tool_name": "edit", "ok": True}}]
            },
        },
    ]
    transcript.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

    metrics = collect_benchmark_policy_metrics(
        transcript_path=transcript, relevant_files=["src/target.py"]
    )

    assert metrics == {
        "source_read_policy_violation": True,
        "relevant_file_hit_at_5": True,
        "unread_edited_paths": [],
        "stale_read_paths": [],
        "new_files": [],
        "out_of_scope_paths": [],
        "code_search_call_count": 1,
        "retrieval_candidate_read_count": 0,
        "retrieval_policy_violation": False,
    }


def test_benchmark_policy_is_path_aware_and_requires_vector_hit_read(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"

    def event(tool_name: str, data: dict[str, object]) -> dict[str, object]:
        return {
            "type": "tool_result",
            "payload": {
                "parts": [
                    {
                        "kind": "tool_result",
                        "metadata": {"tool_name": tool_name, "ok": True, "data": data},
                    }
                ]
            },
        }

    events = [
        event("code_search", {"hits": [{"path": "./SRC/target.py"}]}),
        event("read_multi", {"files": [{"path": "src/target.py"}, {"path": "src/helper.py"}]}),
        event("apply_patch", {"changed_files": ["src/target.py", "src/unread.py", "src/new.py"]}),
    ]
    transcript.write_text("\n".join(json.dumps(item) for item in events), encoding="utf-8")

    metrics = collect_benchmark_policy_metrics(
        transcript_path=transcript,
        relevant_files=["src/target.py"],
        existing_paths=["src/target.py", "src/helper.py", "src/unread.py"],
        editable_paths=["src/target.py", "src/helper.py", "src/unread.py", "src/new.py"],
        retrieval_required=True,
        retrieval_mode="vector",
    )

    assert metrics["source_read_policy_violation"] is True
    assert metrics["unread_edited_paths"] == ["src/unread.py"]
    assert metrics["new_files"] == ["src/new.py"]
    assert metrics["out_of_scope_paths"] == []
    assert metrics["code_search_call_count"] == 1
    assert metrics["retrieval_candidate_read_count"] == 1
    assert metrics["retrieval_policy_violation"] is False


def test_vector_retrieval_required_fails_if_search_or_hit_read_is_late(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    events = [
        {
            "type": "tool_result",
            "payload": {"parts": [{"kind": "tool_result", "metadata": {
                "tool_name": "edit", "ok": True, "data": {"path": "src/target.py"}
            }}]},
        },
        {
            "type": "tool_result",
            "payload": {"parts": [{"kind": "tool_result", "metadata": {
                "tool_name": "code_search", "ok": True, "data": {"hits": [{"path": "src/target.py"}]}
            }}]},
        },
    ]
    transcript.write_text("\n".join(json.dumps(item) for item in events), encoding="utf-8")

    metrics = collect_benchmark_policy_metrics(
        transcript_path=transcript,
        relevant_files=["src/target.py"],
        existing_paths=["src/target.py"],
        editable_paths=["src/target.py"],
        retrieval_required=True,
        retrieval_mode="vector",
    )

    assert metrics["retrieval_policy_violation"] is True
    assert metrics["retrieval_candidate_read_count"] == 0


def init_repo(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, text=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    (repo / "README.md").write_text("benchmark repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, text=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, text=True, capture_output=True)


def test_adapter_builds_benchmark_prompt(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    loop = FakeLoop()
    adapter = FirstCoderCodingAgentAdapter(
        model_name_or_path="firstcoder-test",
        loop_factory=lambda task, session_root: loop,
        session_root=tmp_path / "sessions",
    )
    task = CodingTask(
        instance_id="sympy__sympy-20590",
        repo_path=repo,
        problem_statement="Fix the issue.",
        base_commit="abc123",
    )

    result = adapter.run_task(task)

    assert result.instance_id == "sympy__sympy-20590"
    assert result.model_name_or_path == "firstcoder-test"
    assert "Fix the issue." in loop.messages[0]
    assert "Return by editing files" in loop.messages[0]
    assert result.raw_response == "done"


def test_default_loop_factory_keeps_session_outside_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    session_root = tmp_path / "sessions"
    adapter = FirstCoderCodingAgentAdapter(
        session_root=session_root,
        provider_factory=lambda provider_name: FakeProvider(),
    )
    task = CodingTask(
        instance_id="sympy__sympy-20590",
        repo_path=repo,
        problem_statement="Fix the issue.",
    )

    loop = adapter._create_loop(task, session_root / task.instance_id)

    assert loop.session.store.root == session_root / task.instance_id
    assert repo not in loop.session.store.root.parents
    assert loop.session.mode == "bypass"
    assert "write" in loop.session.tool_registry.names()
    assert "ask_user" not in loop.session.tool_registry.names()
    assert loop.session.skill_catalog.skills == []
    assert loop.limits == AgentLoopLimits.swe_lite()
    message_id = loop.session.append_user_message("benchmark task")
    result = loop.session.tool_registry.execute(
        "task_boundary",
        {"decision": "new", "basis_message_id": message_id},
    )
    assert result.ok is True
    assert result.data["required_stable_count"] == 1
    assert result.data["confirmed_change"] is True


def test_default_loop_factory_uses_custom_limits(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    limits = AgentLoopLimits.swe_lite().with_max_tool_rounds(123)
    adapter = FirstCoderCodingAgentAdapter(
        session_root=tmp_path / "sessions",
        limits=limits,
        provider_factory=lambda provider_name: FakeProvider(),
    )
    task = CodingTask(
        instance_id="sympy__sympy-20590",
        repo_path=repo,
        problem_statement="Fix the issue.",
    )

    loop = adapter._create_loop(task, tmp_path / "sessions" / task.instance_id)

    assert loop.limits == limits


def test_default_loop_factory_registers_task_specific_extra_tools(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    seen: list[str] = []
    adapter = FirstCoderCodingAgentAdapter(
        session_root=tmp_path / "sessions",
        provider_factory=lambda provider_name: FakeProvider(),
        extra_tools_factory=lambda task: seen.append(task.instance_id) or [],
    )
    task = CodingTask(instance_id="task-tools", repo_path=repo, problem_statement="Fix it.")

    adapter._create_loop(task, tmp_path / "sessions" / task.instance_id)

    assert seen == ["task-tools"]


def test_default_loop_factory_wraps_provider_with_benchmark_retries(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    adapter = FirstCoderCodingAgentAdapter(
        session_root=tmp_path / "sessions",
        provider_factory=lambda provider_name: FakeProvider(),
    )
    task = CodingTask(
        instance_id="sympy__sympy-20590",
        repo_path=repo,
        problem_statement="Fix the issue.",
    )

    loop = adapter._create_loop(task, tmp_path / "sessions" / task.instance_id)

    assert isinstance(loop.provider, RetryableBenchmarkProvider)


def test_default_loop_factory_does_not_wrap_provider_when_retries_are_zero(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    provider = FakeProvider()
    adapter = FirstCoderCodingAgentAdapter(
        session_root=tmp_path / "sessions",
        provider_retries=0,
        provider_factory=lambda provider_name: provider,
    )
    task = CodingTask(instance_id="no-retry", repo_path=repo, problem_statement="Fix it.")

    loop = adapter._create_loop(task, tmp_path / "sessions" / task.instance_id)

    assert loop.provider is provider


def test_zero_retry_benchmark_calls_failing_provider_once(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    provider = FlakyProvider(failures=2)
    adapter = FirstCoderCodingAgentAdapter(
        session_root=tmp_path / "sessions",
        provider_retries=0,
        provider_factory=lambda provider_name: provider,
    )
    task = CodingTask(instance_id="no-hidden-retry", repo_path=repo, problem_statement="Fix it.")
    loop = adapter._create_loop(task, tmp_path / "sessions" / task.instance_id)

    with pytest.raises(ProviderError):
        loop.provider.complete(ChatRequest(messages=[]))

    assert provider.calls == 1


def test_benchmark_project_skill_allowlist_excludes_unrelated_skills_and_records_loaded_id(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    for name in ("pytest-ci-repair", "nature-paper2ppt"):
        skill_dir = repo / ".agents" / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name} workflow\n---\n# {name}\n",
            encoding="utf-8",
        )
    adapter = FirstCoderCodingAgentAdapter(
        session_root=tmp_path / "sessions",
        provider_retries=0,
        provider_factory=lambda provider_name: FakeProvider(),
        benchmark_skill_allowlist=("pytest-ci-repair",),
    )
    task = CodingTask(
        instance_id="skill-audit",
        repo_path=repo,
        problem_statement="Use pytest-ci-repair to inspect this pytest task.",
    )

    result = adapter.run_task(task)

    assert result.runtime_metrics["loaded_skill_ids"] == ["pytest-ci-repair"]
    transcript = result.transcript_path.read_text(encoding="utf-8")
    assert "nature-paper2ppt" not in transcript


def test_benchmark_prompt_is_identical_across_retrieval_modes(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    baseline_loop = FakeLoop()
    vector_loop = FakeLoop()
    common = {
        "retrieval_required": True,
        "existing_paths": ["src/service.py"],
        "editable_paths": ["src/service.py"],
    }
    baseline_task = CodingTask(
        instance_id="paired",
        repo_path=repo,
        problem_statement="Fix it.",
        metadata={**common, "retrieval_mode": "baseline"},
    )
    vector_task = CodingTask(
        instance_id="paired",
        repo_path=repo,
        problem_statement="Fix it.",
        metadata={**common, "retrieval_mode": "vector"},
    )

    FirstCoderCodingAgentAdapter(
        loop_factory=lambda task, root: baseline_loop,
        session_root=tmp_path / "baseline-sessions",
    ).run_task(baseline_task)
    FirstCoderCodingAgentAdapter(
        loop_factory=lambda task, root: vector_loop,
        session_root=tmp_path / "vector-sessions",
    ).run_task(vector_task)

    assert baseline_loop.messages[0].encode() == vector_loop.messages[0].encode()


def test_retryable_benchmark_provider_retries_transient_errors() -> None:
    provider = FlakyProvider(failures=2)
    delays: list[float] = []
    retrying = RetryableBenchmarkProvider(provider, max_retries=3, initial_delay_seconds=0.5, sleep=delays.append)

    response = retrying.complete(ChatRequest(messages=[]))

    assert response.content == "done"
    assert provider.calls == 3
    assert retrying.call_count == 3
    assert delays == [0.5, 1.0]


def test_retryable_benchmark_provider_does_not_retry_non_retryable_errors() -> None:
    provider = FlakyProvider(failures=1, kind=ProviderErrorKind.AUTH_ERROR)
    delays: list[float] = []
    retrying = RetryableBenchmarkProvider(provider, max_retries=3, initial_delay_seconds=0.5, sleep=delays.append)

    with pytest.raises(ProviderError):
        retrying.complete(ChatRequest(messages=[]))

    assert provider.calls == 1
    assert delays == []


def test_default_loop_factory_auto_allows_repo_writes_for_benchmarks(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    adapter = FirstCoderCodingAgentAdapter(
        session_root=tmp_path / "sessions",
        provider_factory=lambda provider_name: PatchProvider(),
    )
    task = CodingTask(
        instance_id="sympy__sympy-20590",
        repo_path=repo,
        problem_statement="Create the fix.",
    )

    result = adapter.run_task(task)

    assert result.raw_response == "done"
    assert "diff --git a/fixed.py b/fixed.py" in result.model_patch
    assert "+VALUE = 42" in result.model_patch
    assert result.runtime_metrics["provider_call_count"] == 2
    assert result.runtime_metrics["tool_call_count"] == 1
    assert result.runtime_metrics["tool_calls_by_name"] == {"apply_patch": 1}
    assert result.runtime_metrics["actual_total_tokens"] is None


def test_relative_session_root_is_resolved_outside_repo(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    loop = FakeLoop()
    seen_session_roots: list[Path] = []

    def make_loop(task: CodingTask, session_root: Path) -> FakeLoop:
        seen_session_roots.append(session_root)
        return loop

    monkeypatch.chdir(repo)
    adapter = FirstCoderCodingAgentAdapter(loop_factory=make_loop)
    task = CodingTask(
        instance_id="sympy__sympy-20590",
        repo_path=repo,
        problem_statement="Fix the issue.",
    )

    adapter.run_task(task)

    assert seen_session_roots == [tmp_path / ".firstcoder-eval" / "sympy__sympy-20590"]
    assert repo not in seen_session_roots[0].parents


def test_session_root_inside_repo_is_rejected(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    adapter = FirstCoderCodingAgentAdapter(session_root="repo/.firstcoder-eval")
    task = CodingTask(
        instance_id="sympy__sympy-20590",
        repo_path=repo,
        problem_statement="Fix the issue.",
    )

    with pytest.raises(ValueError, match="outside the task repository"):
        adapter.run_task(task)


def test_instance_id_is_sanitized_for_session_directory(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    loop = FakeLoop()
    seen_session_roots: list[Path] = []

    def make_loop(task: CodingTask, session_root: Path) -> FakeLoop:
        seen_session_roots.append(session_root)
        return loop

    adapter = FirstCoderCodingAgentAdapter(
        session_root=tmp_path / "sessions",
        loop_factory=make_loop,
    )
    task = CodingTask(
        instance_id="../sympy/sympy-20590",
        repo_path=repo,
        problem_statement="Fix the issue.",
    )

    result = adapter.run_task(task)

    assert result.instance_id == "../sympy/sympy-20590"
    expected_root = tmp_path / "sessions" / "__sympy_sympy-20590"
    assert seen_session_roots == [expected_root]
    assert result.transcript_path == expected_root / "sessions" / "__sympy_sympy-20590.jsonl"
    assert expected_root in result.transcript_path.parents


def test_default_loop_factory_sanitizes_internal_session_id(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    adapter = FirstCoderCodingAgentAdapter(
        session_root=tmp_path / "sessions",
        provider_factory=lambda provider_name: FakeProvider(),
    )
    task = CodingTask(
        instance_id="../sympy/sympy-20590",
        repo_path=repo,
        problem_statement="Fix the issue.",
    )

    loop = adapter._create_loop(task, tmp_path / "sessions" / "__sympy_sympy-20590")

    assert loop.session.session_id == "__sympy_sympy-20590"
    assert loop.session.store._session_path(loop.session.session_id) == (
        tmp_path / "sessions" / "__sympy_sympy-20590" / "sessions" / "__sympy_sympy-20590.jsonl"
    )


def test_adapter_patch_includes_untracked_files(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    init_repo(repo)
    adapter = FirstCoderCodingAgentAdapter(
        session_root=tmp_path / "sessions",
        loop_factory=lambda task, session_root: FileWritingLoop(repo),
    )
    task = CodingTask(
        instance_id="sympy__sympy-20590",
        repo_path=repo,
        problem_statement="Fix the issue.",
    )

    result = adapter.run_task(task)

    assert "diff --git a/new_module.py b/new_module.py" in result.model_patch
    assert "+NEW_VALUE = 3" in result.model_patch


def test_collect_context_metrics_uses_append_only_session_facts(tmp_path: Path) -> None:
    store = JsonlSessionStore(tmp_path)
    session_id = "sess_metrics"

    def append(event_type: str, payload: dict[str, object]) -> None:
        store.append_event(
            SessionEvent(
                id=f"event_{len(store.list_events(session_id))}",
                session_id=session_id,
                type=event_type,
                payload=payload,
            )
        )

    append(
        "user_message",
        {"message_id": "user_1", "parts": [{"id": "part_user", "kind": "text", "content": "fix it"}]},
    )
    append(
        "assistant_message",
        {
            "message_id": "assistant_1",
            "metadata": {"usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}},
            "parts": [
                {
                    "id": "part_call_view",
                    "kind": "tool_call",
                    "content": "",
                    "metadata": {"tool_call_id": "call_view", "tool_name": "view"},
                }
            ],
        },
    )
    append(
        "tool_result",
        {
            "message_id": "tool_1",
            "parts": [
                {
                    "id": "part_result_view",
                    "kind": "tool_result",
                    "content": "source text",
                    "metadata": {"tool_call_id": "call_view", "tool_name": "view", "ok": True, "data": {}},
                }
            ],
        },
    )
    append(
        "assistant_message",
        {
            "message_id": "assistant_2",
            "metadata": {"usage": {"input_tokens": 20, "output_tokens": 3, "total_tokens": 23}},
            "parts": [],
        },
    )
    append(
        "tool_result",
        {
            "message_id": "tool_denied",
            "parts": [
                {
                    "id": "part_denied",
                    "kind": "tool_result",
                    "content": "denied",
                    "metadata": {
                        "tool_call_id": "call_edit",
                        "tool_name": "edit",
                        "ok": False,
                        "data": {"request_type": "permission_denied"},
                    },
                }
            ],
        },
    )
    append(
        "tool_result",
        {
            "message_id": "tool_archive",
            "parts": [
                {
                    "id": "part_archive",
                    "kind": "tool_result",
                    "content": "archive text",
                    "metadata": {
                        "tool_call_id": "call_archive",
                        "tool_name": "retrieve_archive",
                        "ok": True,
                        "data": {},
                    },
                }
            ],
        },
    )
    append(
        "assistant_message",
        {
            "message_id": "assistant_3",
            "metadata": {"usage": {"input_tokens": 30, "output_tokens": 4, "total_tokens": 34}},
            "parts": [{"id": "part_text", "kind": "text", "content": "done"}],
        },
    )
    append(
        "compaction_completed",
        {
            "trigger": "auto",
            "before_tokens": 100,
            "after_tokens": 60,
            "event": {
                "replacements": [
                    {"replacement_part": {"metadata": {"archive_id": "archive_1"}}}
                ]
            },
        },
    )

    metrics = collect_context_metrics(
        transcript_path=tmp_path / "sessions" / f"{session_id}.jsonl",
        provider_call_count=3,
    )

    assert metrics["provider_call_count"] == 3
    assert metrics["tool_call_count"] == 2
    assert metrics["tool_calls_by_name"] == {"retrieve_archive": 1, "view": 1}
    assert metrics["actual_input_tokens"] == 60
    assert metrics["actual_output_tokens"] == 9
    assert metrics["actual_total_tokens"] == 69
    assert metrics["estimated_input_tokens"] > 0
    assert metrics["source_read_count"] == 1
    assert metrics["compaction_event_count"] == 1
    assert metrics["compaction_trigger_counts"] == {"auto": 1}
    assert metrics["compaction_tokens_before"] == 100
    assert metrics["compaction_tokens_after"] == 60
    assert metrics["archive_count"] == 1
    assert metrics["archive_retrieval_count"] == 1
    json.dumps(metrics)


def test_collect_context_metrics_keeps_missing_actual_usage_null(tmp_path: Path) -> None:
    store = JsonlSessionStore(tmp_path)
    store.append_event(
        SessionEvent(
            id="event_1",
            session_id="sess_missing_usage",
            type="assistant_message",
            payload={"message_id": "assistant_1", "metadata": {"usage": None}, "parts": []},
        )
    )

    metrics = collect_context_metrics(
        transcript_path=tmp_path / "sessions" / "sess_missing_usage.jsonl",
        provider_call_count=1,
    )

    assert metrics["actual_input_tokens"] is None
    assert metrics["actual_output_tokens"] is None
    assert metrics["actual_total_tokens"] is None
    assert metrics["estimated_input_tokens"] == 0


def test_collect_context_metrics_excludes_local_tool_limit_response_from_usage_coverage(tmp_path: Path) -> None:
    store = JsonlSessionStore(tmp_path)
    session_id = "sess_tool_limit"
    store.append_event(
        SessionEvent(
            id="provider_response",
            session_id=session_id,
            type="assistant_message",
            payload={
                "message_id": "assistant_provider",
                "metadata": {
                    "finish_reason": "tool_calls",
                    "usage": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
                },
                "parts": [],
            },
        )
    )
    store.append_event(
        SessionEvent(
            id="local_stop",
            session_id=session_id,
            type="assistant_message",
            payload={
                "message_id": "assistant_local",
                "metadata": {"finish_reason": "tool_round_limit", "usage": None},
                "parts": [{"kind": "text", "content": "stopped"}],
            },
        )
    )

    metrics = collect_context_metrics(
        transcript_path=tmp_path / "sessions" / f"{session_id}.jsonl",
        provider_call_count=1,
    )

    assert metrics["provider_call_count"] == 1
    assert metrics["prompt_tokens"] == 100
    assert metrics["completion_tokens"] == 10
    assert metrics["estimated_cost_usd"] == 0.0000168
