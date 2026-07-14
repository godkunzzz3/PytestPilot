"""Deterministic benchmark metrics collected from append-only session facts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from firstcoder.context.token_budget import estimate_text_tokens


_MESSAGE_EVENT_TYPES = {"user_message", "assistant_message", "tool_result"}
_COMPACTION_EVENT_TYPES = {"compaction_completed", "llm_compaction_completed"}
_SOURCE_READ_TOOLS = {"view", "read_multi"}
_NON_EXECUTION_REQUEST_TYPES = {"permission_confirmation", "permission_denied"}


def empty_runtime_metrics() -> dict[str, Any]:
    """Return the stable JSON-compatible metrics schema for an empty run."""

    return {
        "provider_call_count": 0,
        "tool_call_count": 0,
        "tool_calls_by_name": {},
        "actual_input_tokens": None,
        "actual_output_tokens": None,
        "actual_total_tokens": None,
        "estimated_input_tokens": 0,
        "compaction_event_count": 0,
        "compaction_trigger_counts": {},
        "compaction_tokens_before": 0,
        "compaction_tokens_after": 0,
        "archive_count": 0,
        "archive_retrieval_count": 0,
        "source_read_count": 0,
    }


def collect_context_metrics(
    *,
    transcript_path: str | Path | None,
    provider_call_count: int,
) -> dict[str, Any]:
    """Aggregate one benchmark turn without mutating or compacting its Session.

    Tool totals come from persisted tool results, not provider-visible projections. Permission
    confirmations, denials, and calls skipped while waiting for user input are protocol facts but
    were not executed, so they are excluded. Actual usage is reported only when every recorded
    Provider attempt has a complete usage record.
    """

    metrics = empty_runtime_metrics()
    events = _read_jsonl(transcript_path)
    assistant_events = [event for event in events if event.get("type") == "assistant_message"]
    resolved_provider_calls = max(int(provider_call_count), len(assistant_events))
    metrics["provider_call_count"] = resolved_provider_calls

    tool_counts: Counter[str] = Counter()
    source_reads = 0
    archive_retrievals = 0
    for event in events:
        if event.get("type") != "tool_result":
            continue
        for part in _event_parts(event):
            metadata = _mapping(part.get("metadata"))
            if str(part.get("kind") or "") != "tool_result" or not _was_executed(metadata):
                continue
            tool_name = str(metadata.get("tool_name") or "unknown")
            tool_counts[tool_name] += 1
            if tool_name in _SOURCE_READ_TOOLS:
                source_reads += 1
            if tool_name == "retrieve_archive":
                archive_retrievals += 1

    metrics["tool_call_count"] = sum(tool_counts.values())
    metrics["tool_calls_by_name"] = dict(sorted(tool_counts.items()))
    metrics["source_read_count"] = source_reads
    metrics["archive_retrieval_count"] = archive_retrievals

    usages = [_mapping(_event_metadata(event).get("usage")) for event in assistant_events]
    if _complete_usage_coverage(usages, resolved_provider_calls):
        metrics["actual_input_tokens"] = sum(int(usage["input_tokens"]) for usage in usages)
        metrics["actual_output_tokens"] = sum(int(usage["output_tokens"]) for usage in usages)
        metrics["actual_total_tokens"] = sum(int(usage["total_tokens"]) for usage in usages)

    metrics["estimated_input_tokens"] = _estimate_provider_input_tokens(events)

    compaction_events = [event for event in events if event.get("type") in _COMPACTION_EVENT_TYPES]
    triggers: Counter[str] = Counter()
    before_tokens = 0
    after_tokens = 0
    for event in compaction_events:
        payload = _mapping(event.get("payload"))
        triggers[str(payload.get("trigger") or "unknown")] += 1
        before_tokens += _optional_nonnegative_int(payload.get("before_tokens"))
        after_tokens += _optional_nonnegative_int(payload.get("after_tokens"))
    metrics["compaction_event_count"] = len(compaction_events)
    metrics["compaction_trigger_counts"] = dict(sorted(triggers.items()))
    metrics["compaction_tokens_before"] = before_tokens
    metrics["compaction_tokens_after"] = after_tokens
    metrics["archive_count"] = len(_collect_archive_ids(events))
    return metrics


def collect_diff_metrics(diff: str) -> dict[str, int]:
    """Count changed files and textual additions/deletions in a unified Git diff."""

    files: set[str] = set()
    added = 0
    deleted = 0
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split(" ", 3)
            if len(parts) >= 4:
                files.add(parts[3].removeprefix("b/"))
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1
    return {
        "diff_file_count": len(files),
        "diff_added_lines": added,
        "diff_deleted_lines": deleted,
    }


def _read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    transcript = Path(path)
    if not transcript.is_file():
        return []
    events: list[dict[str, Any]] = []
    with transcript.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
    return events


def _event_parts(event: dict[str, Any]) -> list[dict[str, Any]]:
    parts = _mapping(event.get("payload")).get("parts")
    if not isinstance(parts, list):
        return []
    return [part for part in parts if isinstance(part, dict)]


def _event_metadata(event: dict[str, Any]) -> dict[str, Any]:
    return _mapping(_mapping(event.get("payload")).get("metadata"))


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _was_executed(metadata: dict[str, Any]) -> bool:
    data = _mapping(metadata.get("data"))
    if data.get("requires_user_input") or data.get("skipped_due_to_user_input"):
        return False
    return str(data.get("request_type") or "") not in _NON_EXECUTION_REQUEST_TYPES


def _complete_usage_coverage(usages: list[dict[str, Any]], provider_calls: int) -> bool:
    if provider_calls <= 0 or len(usages) != provider_calls:
        return False
    return all(
        isinstance(usage.get(field), int) and int(usage[field]) >= 0
        for usage in usages
        for field in ("input_tokens", "output_tokens", "total_tokens")
    )


def _estimate_provider_input_tokens(events: list[dict[str, Any]]) -> int:
    cumulative_history_tokens = 0
    estimated = 0
    for event in events:
        event_type = str(event.get("type") or "")
        if event_type not in _MESSAGE_EVENT_TYPES:
            continue
        if event_type == "assistant_message":
            estimated += cumulative_history_tokens
        cumulative_history_tokens += sum(
            estimate_text_tokens(str(part.get("content") or "")) for part in _event_parts(event)
        )
    return estimated


def _optional_nonnegative_int(value: object) -> int:
    if not isinstance(value, int):
        return 0
    return max(0, value)


def _collect_archive_ids(events: list[dict[str, Any]]) -> set[str]:
    archive_ids: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            archive_id = value.get("archive_id")
            if archive_id:
                archive_ids.add(str(archive_id))
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    for event in events:
        visit(event.get("payload"))
    return archive_ids
