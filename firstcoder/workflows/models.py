"""Serializable models for the pytest repair workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from firstcoder.ci.models import PytestRunReport


@dataclass(frozen=True, slots=True)
class PytestCommandResult:
    command: str
    exit_code: int
    output: str
    elapsed_seconds: float

    @property
    def passed(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "passed": self.passed,
            "output": self.output,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True, slots=True)
class PytestFixAttempt:
    number: int
    base_commit: str | None
    attempt_patch: str
    deterministic_candidates: list[str]
    semantic_candidates: list[dict[str, Any]]
    transcript_path: str | None
    focused_result: PytestCommandResult
    full_result: PytestCommandResult | None
    source_read_policy_violation: bool
    introduced_failures: list[str] = field(default_factory=list)
    selected: bool = False
    runtime_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "base_commit": self.base_commit,
            "attempt_patch": self.attempt_patch,
            "deterministic_candidates": self.deterministic_candidates,
            "semantic_candidates": self.semantic_candidates,
            "transcript_path": self.transcript_path,
            "focused_result": self.focused_result.to_dict(),
            "full_result": self.full_result.to_dict() if self.full_result is not None else None,
            "source_read_policy_violation": self.source_read_policy_violation,
            "introduced_failures": self.introduced_failures,
            "selected": self.selected,
            "runtime_metrics": self.runtime_metrics,
        }


@dataclass(frozen=True, slots=True)
class PytestFixResult:
    status: str
    exit_code: int
    test_command: str
    baseline_report: PytestRunReport
    final_report: PytestRunReport
    attempts: list[PytestFixAttempt]
    final_diff: str
    elapsed_seconds: float
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "test_command": self.test_command,
            "baseline_report": self.baseline_report.to_dict(),
            "final_report": self.final_report.to_dict(),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "final_diff": self.final_diff,
            "elapsed_seconds": self.elapsed_seconds,
            "metrics": self.metrics,
        }
