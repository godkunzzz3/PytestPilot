"""Bounded pytest diagnosis/repair workflow around the existing Agent adapter."""

from __future__ import annotations

import json
import shlex
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Protocol

from firstcoder.ci import PytestRunReport, parse_pytest_output
from firstcoder.eval.metrics import collect_diff_metrics, empty_runtime_metrics
from firstcoder.eval.patch import collect_git_diff
from firstcoder.eval.tasks import CodingTask, CodingTaskResult
from firstcoder.execution import ExecutionBackend, LocalProcessBackend, ResourceLimits
from firstcoder.retrieval.models import RetrievalUnavailableError
from firstcoder.workflows.models import PytestCommandResult, PytestFixAttempt, PytestFixResult
from firstcoder.workflows.attempt_workspace import AttemptWorkspaceManager, apply_selected_files
from firstcoder.workflows.prompts import build_pytest_repair_prompt


_SOURCE_READ_TOOLS = {"view", "read_multi"}
_MUTATION_TOOLS = {"write", "edit", "delete", "apply_patch"}


class RepairAdapter(Protocol):
    def run_task(self, task: CodingTask) -> CodingTaskResult:
        ...


class SemanticSearchLike(Protocol):
    def search(self, query: str, *, top_k: int = 5, path_prefix: str | None = None, include_tests: bool = True):
        ...


class PytestCommandRunner(Protocol):
    def run(self, command: str, *, cwd: Path | None = None) -> PytestCommandResult:
        ...


class SubprocessPytestCommandRunner:
    """Compatibility wrapper around an explicit ExecutionBackend."""

    def __init__(
        self,
        root: str | Path,
        *,
        backend: ExecutionBackend | None = None,
        limits: ResourceLimits | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.backend = backend or LocalProcessBackend()
        self.limits = limits or ResourceLimits(timeout_seconds=timeout_seconds or 300.0)

    def run(self, command: str, *, cwd: Path | None = None) -> PytestCommandResult:
        started = time.perf_counter()
        args = shlex.split(command)
        if not args:
            raise ValueError("pytest command cannot be empty")
        if args[:2] in (["python", "-m"], ["python3", "-m"]):
            args[0] = sys.executable
        completed = self.backend.run(args, (cwd or self.root).resolve(), self.limits)
        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        if completed.error:
            output = "\n".join(part for part in (output, completed.error) if part)
        return PytestCommandResult(
            command=command,
            exit_code=completed.exit_code,
            output=output,
            elapsed_seconds=round(time.perf_counter() - started, 6),
        )


class PytestFixWorkflow:
    def __init__(
        self,
        project_root: str | Path,
        *,
        adapter: RepairAdapter,
        semantic_search: SemanticSearchLike | None = None,
        command_runner: PytestCommandRunner | None = None,
        execution_backend: ExecutionBackend | None = None,
        resource_limits: ResourceLimits | None = None,
        editable_paths: list[str] | tuple[str, ...] | None = None,
        max_attempts: int = 2,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.adapter = adapter
        self.semantic_search = semantic_search
        self.execution_backend = execution_backend or LocalProcessBackend()
        self.resource_limits = resource_limits or ResourceLimits()
        self.command_runner = command_runner or SubprocessPytestCommandRunner(
            self.project_root,
            backend=self.execution_backend,
            limits=self.resource_limits,
        )
        self.editable_paths = list(editable_paths) if editable_paths is not None else _default_editable_paths(
            self.project_root
        )
        self.max_attempts = min(2, max(1, max_attempts))

    def run(
        self,
        *,
        test_command: str,
        failure_log: str | None = None,
        json_out: str | Path | None = None,
    ) -> PytestFixResult:
        started = time.perf_counter()
        if failure_log is None:
            with AttemptWorkspaceManager(self.project_root) as baseline_workspaces:
                baseline_root = baseline_workspaces.create(0)
                baseline_command = self.command_runner.run(test_command, cwd=baseline_root)
        else:
            baseline_command = PytestCommandResult(
                command="<provided failure log>",
                exit_code=1,
                output=failure_log,
                elapsed_seconds=0.0,
            )
        baseline_report = _report(baseline_command)
        current_report = baseline_report
        attempts: list[PytestFixAttempt] = []
        if baseline_command.passed:
            result = self._result(
                status="passed",
                test_command=test_command,
                baseline=baseline_report,
                final=current_report,
                attempts=attempts,
                started=started,
            )
            _write_result(json_out, result)
            return result

        with AttemptWorkspaceManager(self.project_root) as workspaces:
            previous_evidence = ""
            for number in range(1, self.max_attempts + 1):
                attempt_root = workspaces.create(number)
                deterministic = _deterministic_candidates(attempt_root, current_report)
                semantic = self._semantic_candidates(current_report)
                focused_command = _focused_command(test_command, current_report)
                problem_statement = build_pytest_repair_prompt(
                    current_report,
                    deterministic_candidates=deterministic,
                    semantic_candidates=semantic,
                    focused_command=focused_command,
                    full_command=test_command,
                )
                if previous_evidence:
                    problem_statement += previous_evidence
                task = CodingTask(
                    instance_id=f"pytest-fix-attempt-{number}",
                    repo_path=attempt_root,
                    problem_statement=problem_statement,
                    base_commit=workspaces.base_commit,
                    metadata={
                        "workflow": "pytest_fix",
                        "attempt": number,
                        "test_command": test_command,
                        "existing_paths": self.editable_paths,
                        "editable_paths": self.editable_paths,
                        "enforce_fresh_source_guard": True,
                        "execution_backend": self.execution_backend,
                        "resource_limits": self.resource_limits,
                    },
                )
                agent_result = self.adapter.run_task(task)
                attempt_patch = collect_git_diff(attempt_root, include_untracked=True)
                transcript_path = _preserve_transcript(
                    agent_result.transcript_path,
                    project_root=self.project_root,
                    attempt_number=number,
                )
                violation = _source_read_policy_violation(
                    transcript_path,
                    mutation_observed=bool(attempt_patch),
                )
                focused = self.command_runner.run(focused_command, cwd=attempt_root)
                full = self.command_runner.run(test_command, cwd=attempt_root) if focused.passed else None
                validation = full or focused
                current_report = _report(validation)
                selected = full is not None and full.passed
                introduced = _introduced_failures(baseline_report, current_report)
                attempts.append(
                    PytestFixAttempt(
                        number=number,
                        base_commit=workspaces.base_commit,
                        attempt_patch=attempt_patch,
                        deterministic_candidates=deterministic,
                        semantic_candidates=semantic,
                        transcript_path=str(transcript_path) if transcript_path else None,
                        focused_result=focused,
                        full_result=full,
                        source_read_policy_violation=violation,
                        introduced_failures=introduced,
                        selected=selected,
                        runtime_metrics=dict(agent_result.runtime_metrics),
                    )
                )
                if selected:
                    apply_selected_files(
                        attempt_root,
                        self.project_root,
                        editable_paths=self.editable_paths,
                    )
                    result = self._result(
                        status="passed",
                        test_command=test_command,
                        baseline=baseline_report,
                        final=current_report,
                        attempts=attempts,
                        started=started,
                    )
                    _write_result(json_out, result)
                    return result
                previous_evidence = _previous_attempt_evidence(
                    attempt_patch=attempt_patch,
                    validation=current_report,
                )

        result = self._result(
            status="failed",
            test_command=test_command,
            baseline=baseline_report,
            final=current_report,
            attempts=attempts,
            started=started,
        )
        _write_result(json_out, result)
        return result

    def _semantic_candidates(self, report: PytestRunReport) -> list[dict[str, object]]:
        if self.semantic_search is None or not report.failures:
            return []
        query = " ".join(
            filter(
                None,
                [
                    report.failures[0].node_id,
                    report.failures[0].exception_type,
                    report.failures[0].message,
                ],
            )
        )
        try:
            return [hit.to_dict() for hit in self.semantic_search.search(query, top_k=5, include_tests=False)]
        except (RetrievalUnavailableError, ValueError, OSError):
            return []

    def _result(
        self,
        *,
        status: str,
        test_command: str,
        baseline: PytestRunReport,
        final: PytestRunReport,
        attempts: list[PytestFixAttempt],
        started: float,
    ) -> PytestFixResult:
        final_diff = collect_git_diff(self.project_root, include_untracked=True)
        metrics = _aggregate_metrics(attempts)
        metrics.update(collect_diff_metrics(final_diff))
        metrics["source_read_policy_violation"] = any(
            attempt.source_read_policy_violation for attempt in attempts
        )
        return PytestFixResult(
            status=status,
            exit_code=0 if status == "passed" else 2,
            test_command=test_command,
            baseline_report=baseline,
            final_report=final,
            attempts=attempts,
            final_diff=final_diff,
            elapsed_seconds=round(time.perf_counter() - started, 6),
            metrics=metrics,
        )


def _report(result: PytestCommandResult) -> PytestRunReport:
    return parse_pytest_output(
        result.output,
        command=result.command,
        exit_code=result.exit_code,
        duration=result.elapsed_seconds,
    )


def _focused_command(command: str, report: PytestRunReport) -> str:
    if not report.failures:
        return command
    node_id = report.failures[0].node_id
    if node_id.startswith(("ERROR ", "FAILED ")) or " " in node_id:
        return command
    return f"{command} {shlex.quote(node_id)}"


def _deterministic_candidates(root: Path, report: PytestRunReport) -> list[str]:
    root = root.resolve()
    candidates: list[str] = []
    for failure in report.failures:
        for location in failure.source_locations:
            _append_repo_path(root, location.path, candidates)
        node_path = failure.node_id.split("::", 1)[0]
        _append_repo_path(root, node_path, candidates)
        test_name = Path(node_path).name
        if not test_name.startswith("test_"):
            for location in failure.source_locations:
                location_name = Path(location.path).name
                if location_name.startswith("test_"):
                    test_name = location_name
                    break
        if test_name.startswith("test_"):
            target_name = test_name.removeprefix("test_")
            for path in sorted(root.rglob(target_name)):
                if path.is_file() and "tests" not in path.relative_to(root).parts:
                    _append_repo_path(root, str(path), candidates)
    return candidates


def _append_repo_path(root: Path, value: str, candidates: list[str]) -> None:
    raw = Path(value.replace("\\", "/"))
    resolved = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if root != resolved and root not in resolved.parents:
        return
    if not resolved.exists():
        return
    relative = resolved.relative_to(root).as_posix()
    if relative not in candidates:
        candidates.append(relative)


def _source_read_policy_violation(
    transcript_path: str | Path | None,
    *,
    mutation_observed: bool = False,
) -> bool:
    if transcript_path is None or not Path(transcript_path).is_file():
        return mutation_observed
    read_seen = False
    mutation_seen = False
    for line in Path(transcript_path).read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "tool_result":
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        parts = payload.get("parts") if isinstance(payload.get("parts"), list) else []
        for part in parts:
            if not isinstance(part, dict):
                continue
            metadata = part.get("metadata") if isinstance(part.get("metadata"), dict) else {}
            if metadata.get("ok") is False:
                continue
            tool_name = str(metadata.get("tool_name") or "")
            if tool_name in _SOURCE_READ_TOOLS:
                read_seen = True
            elif tool_name in _MUTATION_TOOLS:
                mutation_seen = True
                if not read_seen:
                    return True
    return mutation_seen and not read_seen


def _aggregate_metrics(attempts: list[PytestFixAttempt]) -> dict[str, object]:
    metrics = empty_runtime_metrics()
    tool_counts: Counter[str] = Counter()
    actual_fields = ("actual_input_tokens", "actual_output_tokens", "actual_total_tokens")
    actual_complete = bool(attempts)
    for attempt in attempts:
        runtime = attempt.runtime_metrics
        metrics["provider_call_count"] += int(runtime.get("provider_call_count") or 0)
        metrics["tool_call_count"] += int(runtime.get("tool_call_count") or 0)
        metrics["estimated_input_tokens"] += int(runtime.get("estimated_input_tokens") or 0)
        metrics["source_read_count"] += int(runtime.get("source_read_count") or 0)
        metrics["compaction_event_count"] += int(runtime.get("compaction_event_count") or 0)
        metrics["archive_count"] += int(runtime.get("archive_count") or 0)
        metrics["archive_retrieval_count"] += int(runtime.get("archive_retrieval_count") or 0)
        tool_counts.update(runtime.get("tool_calls_by_name") or {})
        actual_complete = actual_complete and all(isinstance(runtime.get(field), int) for field in actual_fields)
    metrics["tool_calls_by_name"] = dict(sorted(tool_counts.items()))
    if actual_complete:
        for field in actual_fields:
            metrics[field] = sum(int(attempt.runtime_metrics[field]) for attempt in attempts)
    return metrics


def _write_result(path: str | Path | None, result: PytestFixResult) -> None:
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _default_editable_paths(root: Path) -> list[str]:
    ignored_parts = {".git", ".firstcoder", ".venv", "__pycache__", ".pytest_cache", "tests", "test"}
    return [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix == ".py"
        and not ignored_parts.intersection(path.relative_to(root).parts)
    ]


def _preserve_transcript(
    transcript_path: str | Path | None,
    *,
    project_root: Path,
    attempt_number: int,
) -> Path | None:
    if transcript_path is None:
        return None
    source = Path(transcript_path)
    if not source.is_file():
        return source
    try:
        source.relative_to(project_root)
        return source
    except ValueError:
        destination = project_root / ".firstcoder" / "pytest-fix-transcripts" / f"attempt-{attempt_number}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return destination


def _introduced_failures(baseline: PytestRunReport, validation: PytestRunReport) -> list[str]:
    baseline_fingerprints = {failure.fingerprint for failure in baseline.failures}
    return [
        failure.fingerprint
        for failure in validation.failures
        if failure.fingerprint and failure.fingerprint not in baseline_fingerprints
    ]


def _previous_attempt_evidence(*, attempt_patch: str, validation: PytestRunReport) -> str:
    bounded_patch = attempt_patch[-12_000:]
    bounded_output = validation.bounded_raw_output[-8_000:]
    return (
        "\n\nPrevious attempt evidence (the current worktree was reset to the original baseline; "
        "the patch below is evidence only and is not implicitly applied):\n"
        f"Patch:\n{bounded_patch or '<empty>'}\n"
        f"Validation exit code: {validation.exit_code}\n"
        f"Validation output:\n{bounded_output}\n"
    )
