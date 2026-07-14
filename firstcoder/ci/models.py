"""JSON-compatible pytest failure evidence models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceLocation:
    path: str
    line: int
    function: str | None = None
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PytestFailure:
    node_id: str
    phase: str
    failure_kind: str
    exception_type: str | None
    message: str
    source_locations: list[SourceLocation] = field(default_factory=list)
    expected: str | None = None
    actual: str | None = None
    fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PytestRunReport:
    command: str
    exit_code: int
    status: str
    passed_count: int
    failed_count: int
    error_count: int
    skipped_count: int
    failures: list[PytestFailure] = field(default_factory=list)
    duration: float | None = None
    truncated: bool = False
    bounded_raw_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
