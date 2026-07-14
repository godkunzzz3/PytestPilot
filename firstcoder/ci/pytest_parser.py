"""Best-effort parser for human-readable pytest output."""

from __future__ import annotations

import re
from dataclasses import dataclass

from firstcoder.ci.fingerprint import failure_fingerprint
from firstcoder.ci.models import PytestFailure, PytestRunReport, SourceLocation
from firstcoder.ci.normalize import clean_pytest_text, normalize_path


_SUMMARY_FAILURE_RE = re.compile(r"^(FAILED|ERROR)\s+([^\s]+)(?:\s+-\s+(.*))?$", re.MULTILINE)
_COUNT_RE = re.compile(r"\b(\d+)\s+(passed|failed|errors?|skipped)\b", re.IGNORECASE)
_DURATION_RE = re.compile(r"\bin\s+(\d+(?:\.\d+)?)s\b", re.IGNORECASE)
_HEADER_RE = re.compile(r"^_{2,}\s*(.*?)\s*_{2,}\s*$", re.MULTILINE)
_EXCEPTION_RE = re.compile(
    r"^E\s+(?P<exception>[A-Za-z_][\w.]*(?:Error|Exception)):\s*(?P<message>.*)$",
    re.MULTILINE,
)
_SHORT_EXCEPTION_RE = re.compile(r"^(?P<exception>[A-Za-z_][\w.]*(?:Error|Exception)):\s*(?P<message>.*)$")
_SOURCE_RE = re.compile(
    r"^(?:E\s+)?(?P<path>(?:[A-Za-z]:[\\/]|/)?[^:\n]*?\.py):(?P<line>\d+)"
    r"(?::\s+in\s+(?P<function>[^\s:]+))?\s*(?::\s*[A-Za-z_][\w.]+)?$"
)
_EXPLICIT_EXPECTED_RE = re.compile(r"\bExpected\s*:\s*(.+?)(?:\s*$)", re.IGNORECASE | re.MULTILINE)
_EXPLICIT_ACTUAL_RE = re.compile(r"\bActual\s*:\s*(.+?)(?:\s*$)", re.IGNORECASE | re.MULTILINE)
_ASSERT_EQUAL_RE = re.compile(r"\bassert\s+(.+?)\s*==\s*(.+?)\s*$")
_SECTION_LABELS = {"FAILURES", "ERRORS", "SHORT TEST SUMMARY INFO", "WARNINGS SUMMARY"}


@dataclass(frozen=True, slots=True)
class _Candidate:
    result: str
    node_id: str
    summary_message: str


@dataclass(frozen=True, slots=True)
class _DetailBlock:
    label: str
    text: str


def parse_pytest_output(
    output: str,
    *,
    command: str = "",
    exit_code: int = 1,
    duration: float | None = None,
    max_raw_chars: int = 20_000,
) -> PytestRunReport:
    """Parse pytest console output without assuming one exact traceback style."""

    cleaned = clean_pytest_text(output)
    counts = _parse_counts(cleaned)
    resolved_duration = duration if duration is not None else _parse_duration(cleaned)
    candidates = _summary_candidates(cleaned)
    blocks = _detail_blocks(cleaned)
    if not candidates:
        candidates = _candidates_from_blocks(blocks)
    failures = [_build_failure(candidate, blocks, cleaned) for candidate in candidates]

    failed_count = counts["failed"] or sum(failure.phase == "call" for failure in failures)
    error_count = counts["error"] or sum(failure.phase != "call" for failure in failures)
    status = _status(
        cleaned,
        exit_code=exit_code,
        failed_count=failed_count,
        error_count=error_count,
        passed_count=counts["passed"],
    )
    bounded, truncated = _bounded_output(cleaned, max_raw_chars=max_raw_chars)
    return PytestRunReport(
        command=command,
        exit_code=exit_code,
        status=status,
        passed_count=counts["passed"],
        failed_count=failed_count,
        error_count=error_count,
        skipped_count=counts["skipped"],
        failures=failures,
        duration=resolved_duration,
        truncated=truncated,
        bounded_raw_output=bounded,
    )


def _parse_counts(text: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    for match in _COUNT_RE.finditer(text):
        label = match.group(2).lower()
        key = "error" if label.startswith("error") else label
        counts[key] = max(counts[key], int(match.group(1)))
    return counts


def _parse_duration(text: str) -> float | None:
    matches = list(_DURATION_RE.finditer(text))
    return float(matches[-1].group(1)) if matches else None


def _summary_candidates(text: str) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    seen: set[tuple[str, str]] = set()
    for match in _SUMMARY_FAILURE_RE.finditer(text):
        candidate = _Candidate(
            result=match.group(1),
            node_id=normalize_path(match.group(2)),
            summary_message=(match.group(3) or "").strip(),
        )
        key = (candidate.result, candidate.node_id)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return candidates


def _detail_blocks(text: str) -> list[_DetailBlock]:
    headers = list(_HEADER_RE.finditer(text))
    blocks: list[_DetailBlock] = []
    for index, match in enumerate(headers):
        label = " ".join(match.group(1).split())
        if not label or label.upper() in _SECTION_LABELS:
            continue
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        blocks.append(_DetailBlock(label=label, text=text[match.end() : end].strip()))
    return blocks


def _candidates_from_blocks(blocks: list[_DetailBlock]) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for block in blocks:
        lowered = block.label.lower()
        if not any(marker in lowered for marker in ("test_", "collecting", "setup of", "teardown of")):
            continue
        node_id = block.label
        result = "ERROR" if any(marker in lowered for marker in ("error", "setup", "teardown", "collecting")) else "FAILED"
        candidates.append(_Candidate(result=result, node_id=node_id, summary_message=""))
    return candidates


def _build_failure(candidate: _Candidate, blocks: list[_DetailBlock], full_text: str) -> PytestFailure:
    block = _matching_block(candidate, blocks)
    detail = block.text if block is not None else candidate.summary_message
    phase = _phase(candidate, block)
    exception_type, exception_message = _exception(detail, candidate.summary_message)
    message = exception_message or candidate.summary_message or _last_error_line(detail) or "unparsed pytest failure"
    if exception_type is None and (candidate.result == "FAILED" or "assert" in message.lower()):
        exception_type = "AssertionError"
    locations = _source_locations(detail)
    expected, actual = _expected_actual(detail, candidate.summary_message)
    kind = _failure_kind(phase, exception_type, message, full_text)
    fingerprint = failure_fingerprint(
        node_id=candidate.node_id,
        phase=phase,
        exception_type=exception_type,
        message=message,
        source_path=locations[0].path if locations else None,
    )
    return PytestFailure(
        node_id=candidate.node_id,
        phase=phase,
        failure_kind=kind,
        exception_type=exception_type,
        message=message,
        source_locations=locations,
        expected=expected,
        actual=actual,
        fingerprint=fingerprint,
    )


def _matching_block(candidate: _Candidate, blocks: list[_DetailBlock]) -> _DetailBlock | None:
    leaf = candidate.node_id.split("::")[-1]
    leaf_base = leaf.split("[")[0]
    path = candidate.node_id.split("::")[0]
    for block in blocks:
        label = block.label.replace("\\", "/")
        if leaf in label:
            return block
        if "collecting" in label.lower() and path in label:
            return block
    for block in blocks:
        if leaf_base in block.label.replace("\\", "/"):
            return block
    return None


def _phase(candidate: _Candidate, block: _DetailBlock | None) -> str:
    label = block.label.lower() if block is not None else ""
    message = candidate.summary_message.lower()
    if "setup" in label or "[setup]" in message:
        return "setup"
    if "teardown" in label or "[teardown]" in message:
        return "teardown"
    if "collecting" in label or "collection" in message or "importerror while importing" in message:
        return "collection"
    if candidate.result == "ERROR" and "::" not in candidate.node_id:
        return "collection"
    return "call"


def _exception(detail: str, summary: str) -> tuple[str | None, str]:
    matches = list(_EXCEPTION_RE.finditer(detail))
    if matches:
        match = matches[-1]
        return match.group("exception").split(".")[-1], match.group("message").strip()
    short = _SHORT_EXCEPTION_RE.match(summary.strip())
    if short:
        return short.group("exception").split(".")[-1], short.group("message").strip()
    if "ImportError while importing" in detail:
        return "ImportError", "ImportError while importing test module"
    return None, ""


def _last_error_line(detail: str) -> str:
    lines = [line.removeprefix("E").strip() for line in detail.splitlines() if line.lstrip().startswith("E")]
    return lines[-1] if lines else ""


def _source_locations(detail: str) -> list[SourceLocation]:
    locations: list[SourceLocation] = []
    seen: set[tuple[str, int, str | None]] = set()
    for raw in detail.splitlines():
        match = _SOURCE_RE.match(raw.strip())
        if match is None:
            continue
        path = normalize_path(match.group("path"))
        function = match.group("function")
        key = (path, int(match.group("line")), function)
        if key in seen:
            continue
        seen.add(key)
        locations.append(SourceLocation(path=path, line=key[1], function=function, raw=raw.strip()))
    return locations


def _expected_actual(detail: str, summary: str) -> tuple[str | None, str | None]:
    combined = f"{detail}\n{summary}"
    expected_match = _EXPLICIT_EXPECTED_RE.search(combined)
    actual_match = _EXPLICIT_ACTUAL_RE.search(combined)
    if expected_match or actual_match:
        expected = expected_match.group(1).strip() if expected_match else None
        actual = actual_match.group(1).strip() if actual_match else None
        return expected, actual
    for line in reversed(combined.splitlines()):
        match = _ASSERT_EQUAL_RE.search(line)
        if match:
            return match.group(2).strip(), match.group(1).strip()
    return None, None


def _failure_kind(phase: str, exception_type: str | None, message: str, full_text: str) -> str:
    if phase == "collection" and (exception_type == "ImportError" or "ImportError" in full_text):
        return "import_error"
    if phase == "collection":
        return "collection_error"
    if exception_type == "AssertionError" or message.lstrip().startswith("assert"):
        return "assertion"
    if phase in {"setup", "teardown"}:
        return "fixture_error"
    if exception_type:
        return "exception"
    return "unknown"


def _status(text: str, *, exit_code: int, failed_count: int, error_count: int, passed_count: int) -> str:
    if "no tests ran" in text.lower() or exit_code == 5:
        return "no_tests"
    if failed_count:
        return "failed"
    if error_count:
        return "error"
    if exit_code == 0 or passed_count:
        return "passed"
    return "unknown"


def _bounded_output(text: str, *, max_raw_chars: int) -> tuple[str, bool]:
    limit = max(0, max_raw_chars)
    if len(text) <= limit:
        return text, False
    marker = "\n...[pytest output truncated]...\n"
    if limit <= len(marker):
        return text[:limit], True
    available = limit - len(marker)
    head = available // 2
    tail = available - head
    return f"{text[:head]}{marker}{text[-tail:]}", True
