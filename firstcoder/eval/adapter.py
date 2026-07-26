"""Coding-agent adapters used by benchmark runners."""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Callable, Protocol

from firstcoder.agent.loop import AgentLoop
from firstcoder.agent.loop_limits import AgentLoopLimits
from firstcoder.agent.session import AgentSession
from firstcoder.context.store import JsonlSessionStore
from firstcoder.eval.patch import collect_git_diff
from firstcoder.eval.costs import BudgetedProvider, RequestBoundaryBudget
from firstcoder.eval.metrics import collect_benchmark_policy_metrics, collect_context_metrics
from firstcoder.eval.tasks import CodingTask, CodingTaskResult
from firstcoder.permissions.grants import PermissionGrantStore
from firstcoder.permissions.manager import PermissionManager
from firstcoder.permissions.policy import DefaultPermissionPolicy
from firstcoder.permissions.types import PermissionAction, PermissionDecision, PermissionDecisionKind, PermissionMode
from firstcoder.providers.base import ChatProvider
from firstcoder.providers.errors import ProviderError
from firstcoder.providers.factory import create_provider
from firstcoder.providers.types import ChatRequest, ChatResponse, ChatStreamEvent, ToolChoiceFunction
from firstcoder.tools.builtin import create_builtin_registry
from firstcoder.tools.fresh_source import FreshSourceGuard
from firstcoder.tools.types import Tool
from firstcoder.utils.sandbox_access import SandboxAccess


class CodingAgentAdapter(Protocol):
    def run_task(self, task: CodingTask) -> CodingTaskResult:
        ...


LoopFactory = Callable[[CodingTask, Path], AgentLoop]
ProviderFactory = Callable[[str | None], ChatProvider]
ExtraToolsFactory = Callable[[CodingTask], list[Tool]]
_UNSAFE_SESSION_DIR_CHARS = re.compile(r"[/\\:]")


class FirstCoderCodingAgentAdapter:
    """Runs FirstCoder against one repository-level coding task."""

    def __init__(
        self,
        *,
        model_name_or_path: str = "firstcoder",
        provider_name: str | None = None,
        session_root: str | Path = ".firstcoder-eval",
        limits: AgentLoopLimits | None = None,
        provider_retries: int = 3,
        provider_retry_initial_delay_seconds: float = 2.0,
        loop_factory: LoopFactory | None = None,
        provider_factory: ProviderFactory = create_provider,
        extra_tools: list[Tool] | None = None,
        extra_tools_factory: ExtraToolsFactory | None = None,
        request_budget: RequestBoundaryBudget | None = None,
        benchmark_skill_allowlist: tuple[str, ...] = (),
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.provider_name = provider_name
        self.session_root = Path(session_root)
        self.limits = limits
        self.provider_retries = provider_retries
        self.provider_retry_initial_delay_seconds = provider_retry_initial_delay_seconds
        self.loop_factory = loop_factory or self._create_loop
        self.provider_factory = provider_factory
        self.extra_tools = list(extra_tools or [])
        self.extra_tools_factory = extra_tools_factory
        self.request_budget = request_budget
        self.benchmark_skill_allowlist = benchmark_skill_allowlist

    def run_task(self, task: CodingTask) -> CodingTaskResult:
        session_root = self._session_root_for_task(task)
        session_root.mkdir(parents=True, exist_ok=True)
        loop = self.loop_factory(task, session_root)
        response = loop.run_user_turn(_build_task_prompt(task))
        transcript_path = session_root / "sessions" / f"{_session_dir_name(task.instance_id)}.jsonl"
        model_patch = collect_git_diff(task.repo_path, include_untracked=True)
        runtime_metrics = collect_context_metrics(
            transcript_path=transcript_path,
            provider_call_count=_provider_call_count(loop),
        )
        runtime_metrics.update(
            collect_benchmark_policy_metrics(
                transcript_path=transcript_path,
                relevant_files=task.metadata.get("relevant_files") or [],
                existing_paths=task.metadata.get("existing_paths") or [],
                editable_paths=task.metadata.get("editable_paths") or [],
                retrieval_required=bool(task.metadata.get("retrieval_required")),
                retrieval_mode=str(task.metadata.get("retrieval_mode") or "baseline"),
            )
        )
        return CodingTaskResult(
            instance_id=task.instance_id,
            model_name_or_path=self.model_name_or_path,
            model_patch=model_patch,
            transcript_path=transcript_path,
            raw_response=response.content,
            runtime_metrics=runtime_metrics,
        )

    def _session_root_for_task(self, task: CodingTask) -> Path:
        root = self.session_root
        if not root.is_absolute():
            root = task.repo_path.resolve().parent / root
        session_root = (root / _session_dir_name(task.instance_id)).resolve()
        repo = task.repo_path.resolve()
        if session_root == repo or repo in session_root.parents:
            raise ValueError("Benchmark session_root must resolve outside the task repository.")
        return session_root

    def _create_loop(self, task: CodingTask, session_root: Path) -> AgentLoop:
        sandbox_access = SandboxAccess()
        fresh_source_guard = None
        if task.metadata.get("enforce_fresh_source_guard"):
            fresh_source_guard = FreshSourceGuard(
                task.repo_path,
                editable_paths=task.metadata.get("editable_paths") or (),
            )
        registry = create_builtin_registry(
            task.repo_path,
            include_mutation_tools=True,
            include_execution_tools=True,
            include_network_tools=False,
            include_interactive_tools=False,
            access=sandbox_access,
            fresh_source_guard=fresh_source_guard,
            execution_backend=task.metadata.get("execution_backend"),
            resource_limits=task.metadata.get("resource_limits"),
        )
        task_tools = [*self.extra_tools]
        if self.extra_tools_factory is not None:
            task_tools.extend(self.extra_tools_factory(task))
        for tool in task_tools:
            registry.register(tool)
        permission_manager = PermissionManager(
            policy=BenchmarkPermissionPolicy(task.repo_path),
            grants=PermissionGrantStore(),
            mode=PermissionMode.BYPASS,
        )
        store = JsonlSessionStore(session_root)
        tools = registry.tools()
        session = AgentSession.from_project(
            store=store,
            session_id=_session_dir_name(task.instance_id),
            project_root=task.repo_path,
            tools=tools,
            permission_manager=permission_manager,
            sandbox_access=sandbox_access,
            load_skills=bool(self.benchmark_skill_allowlist),
            skill_allowlist=self.benchmark_skill_allowlist,
        )
        return AgentLoop(
            session=session,
            provider=self._create_provider(task, available_tools=set(registry.names())),
            tools=tools,
            limits=self.limits or AgentLoopLimits.swe_lite(),
        )

    def _create_provider(
        self,
        task: CodingTask | None = None,
        *,
        available_tools: set[str] | None = None,
    ) -> ChatProvider:
        provider = self.provider_factory(self.provider_name)
        if self.request_budget is not None:
            provider = BudgetedProvider(provider, self.request_budget)
        if (
            task is not None
            and task.metadata.get("retrieval_required")
            and task.metadata.get("retrieval_mode") == "vector"
            and "code_search" in (available_tools or set())
        ):
            provider = RequiredFirstToolProvider(provider, tool_name="code_search")
        if self.provider_retries <= 0:
            return provider
        return RetryableBenchmarkProvider(
            provider,
            max_retries=self.provider_retries,
            initial_delay_seconds=self.provider_retry_initial_delay_seconds,
        )


class RetryableBenchmarkProvider(ChatProvider):
    """Retry transient provider failures during non-interactive benchmark runs."""

    def __init__(
        self,
        provider: ChatProvider,
        *,
        max_retries: int,
        initial_delay_seconds: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider
        self.max_retries = max(0, max_retries)
        self.initial_delay_seconds = max(0.0, initial_delay_seconds)
        self.sleep = sleep
        self.call_count = 0

    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def model(self) -> str:
        return self.provider.model

    def complete(self, request: ChatRequest) -> ChatResponse:
        attempt = 0
        while True:
            try:
                self.call_count += 1
                return self.provider.complete(request)
            except ProviderError as exc:
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                delay = self.initial_delay_seconds * (2**attempt)
                if delay > 0:
                    self.sleep(delay)
                attempt += 1


class RequiredFirstToolProvider(ChatProvider):
    """Generic benchmark decorator forcing one configured tool on the first request only."""

    def __init__(self, provider: ChatProvider, *, tool_name: str) -> None:
        self.provider = provider
        self.tool_name = tool_name
        self.call_count = 0
        self._required_tool_sent = False

    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def capabilities(self):
        return getattr(self.provider, "capabilities", None)

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.call_count += 1
        return self.provider.complete(self._request(request))

    async def astream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        self.call_count += 1
        async for event in self.provider.astream(self._request(request)):
            yield event

    def _request(self, request: ChatRequest) -> ChatRequest:
        if self._required_tool_sent:
            return request
        if not any(tool.name == self.tool_name for tool in request.tools):
            return request
        self._required_tool_sent = True
        return replace(request, tool_choice=ToolChoiceFunction(self.tool_name))


class BenchmarkPermissionPolicy(DefaultPermissionPolicy):
    """Non-interactive benchmark policy for repo-local edits."""

    def decide(self, request, *, mode: PermissionMode) -> PermissionDecision:
        if request.action == PermissionAction.EXECUTE_SHELL:
            command = request.target.strip()
            if self._request_cwd_inside_root(request) and (
                command == "python -m pytest"
                or command.startswith("python -m pytest ")
                or command == "python3 -m pytest"
                or command.startswith("python3 -m pytest ")
            ):
                return PermissionDecision(
                    kind=PermissionDecisionKind.ALLOW,
                    reason="Benchmarks allow local pytest validation inside the task repository.",
                )
        if request.action == PermissionAction.WRITE_PATH:
            target = self._resolve_path(request.target, cwd=request.cwd)
            if self._is_inside_project(target) and not self._is_sensitive_path(target):
                return PermissionDecision(
                    kind=PermissionDecisionKind.ALLOW,
                    reason="Benchmarks allow non-sensitive writes inside the task repository.",
                )
        return super().decide(request, mode=mode)


def _build_task_prompt(task: CodingTask) -> str:
    base_commit = task.base_commit or "unknown"
    return (
        "You are running inside a SWE-bench style benchmark task.\n"
        f"Instance: {task.instance_id}\n"
        f"Base commit: {base_commit}\n\n"
        f"Working directory: {task.repo_path.resolve()}\n"
        f"Retrieval required: {'yes' if task.metadata.get('retrieval_required') else 'no'}\n\n"
        "Problem statement:\n"
        f"{task.problem_statement.strip()}\n\n"
        "Return by editing files in the repository. Do not write a final patch manually. "
        "Use tests when useful, keep changes minimal, and leave the repository with the fix applied. "
        "Never assume the repository is /workspace; tool paths and cwd are relative to the actual project root. "
        "Use the diagnostics tool for pytest so it runs with FirstCoder's active Python environment. "
        "Start with stack-trace paths, the failing test module, exact symbols, and grep. "
        "Before changing an existing file, read that exact file with view or read_multi. "
        "When a source read returns a read_token, pass that token to edit/write/delete; "
        "for apply_patch pass a read_tokens mapping keyed by every existing path. "
        "When this task is marked retrieval_required and code_search is available, call code_search before the first "
        "mutation, then read at least one returned candidate with view or read_multi. If code_search is unavailable, "
        "continue with deterministic grep/glob/view tools without asking the user. Never access /workspace. "
        "Run pytest through diagnostics so it uses FirstCoder's current .venv interpreter, not a missing bare python."
    )


def _session_dir_name(instance_id: str) -> str:
    safe = _UNSAFE_SESSION_DIR_CHARS.sub("_", instance_id)
    while ".." in safe:
        safe = safe.replace("..", "__")
    while "___" in safe:
        safe = safe.replace("___", "__")
    return safe or "instance"


def _provider_call_count(loop: AgentLoop) -> int:
    provider = getattr(loop, "provider", None)
    attempt_count = getattr(provider, "call_count", None)
    if isinstance(attempt_count, int):
        return attempt_count
    count = getattr(loop, "provider_call_count", 0)
    return int(count) if isinstance(count, int) else 0
