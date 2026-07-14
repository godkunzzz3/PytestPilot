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
_MUTATION_TOOLS = {"edit", "write", "delete", "apply_patch"}
_LOCAL_ASSISTANT_FINISH_REASONS = {"tool_round_limit", "provider_call_limit", "waiting_for_user_input"}


def empty_runtime_metrics() -> dict[str, Any]:
    """Return the stable JSON-compatible metrics schema for an empty run."""

    return {
        "provider_call_count": 0,
        "tool_call_count": 0,
        "tool_calls_by_name": {},
        "actual_input_tokens": None,
        "actual_output_tokens": None,
        "actual_total_tokens": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "prompt_cache_hit_tokens": None,
        "prompt_cache_miss_tokens": None,
        "estimated_cost_usd": None,
        "estimated_input_tokens": 0,
        "compaction_event_count": 0,
        "compaction_trigger_counts": {},
        "compaction_tokens_before": 0,
        "compaction_tokens_after": 0,
        "archive_count": 0,
        "archive_retrieval_count": 0,
        "source_read_count": 0,
        "loaded_skill_ids": [],
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
    assistant_events = [
        event
        for event in events
        if event.get("type") == "assistant_message"
        and _event_metadata(event).get("finish_reason") not in _LOCAL_ASSISTANT_FINISH_REASONS
    ]
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
        metrics["prompt_tokens"] = metrics["actual_input_tokens"]
        metrics["completion_tokens"] = metrics["actual_output_tokens"]
        metrics["total_tokens"] = metrics["actual_total_tokens"]
        metrics["estimated_cost_usd"] = round(
            metrics["prompt_tokens"] * 0.14 / 1_000_000
            + metrics["completion_tokens"] * 0.28 / 1_000_000,
            9,
        )
    if usages and all(isinstance(usage.get("prompt_cache_hit_tokens"), int) for usage in usages):
        metrics["prompt_cache_hit_tokens"] = sum(int(usage["prompt_cache_hit_tokens"]) for usage in usages)
    if usages and all(isinstance(usage.get("prompt_cache_miss_tokens"), int) for usage in usages):
        metrics["prompt_cache_miss_tokens"] = sum(int(usage["prompt_cache_miss_tokens"]) for usage in usages)

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
    metrics["loaded_skill_ids"] = sorted(
        {
            str(_mapping(event.get("payload")).get("skill_name"))
            for event in events
            if event.get("type") == "skill_loaded" and _mapping(event.get("payload")).get("skill_name")
        }
    )
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


def collect_benchmark_policy_metrics(
    *,
    transcript_path: str | Path | None,
    relevant_files: list[str] | tuple[str, ...],
    existing_paths: list[str] | tuple[str, ...] = (),
    editable_paths: list[str] | tuple[str, ...] = (),
    retrieval_required: bool = False,
    retrieval_mode: str = "baseline",
) -> dict[str, Any]:
    """Audit exact source reads and retrieval order from append-only tool results."""

    events = _read_jsonl(transcript_path)
    read_paths: dict[str, str] = {}
    mutated_paths: dict[str, str] = {}
    unread_paths: dict[str, str] = {}
    candidates: dict[str, str] = {}
    candidate_reads: set[str] = set()
    mutation_seen = False
    first_mutation_seen = False
    code_search_seen = False
    code_search_before_mutation = False
    relevant = {_normalize_policy_path(path) for path in relevant_files}
    existing = {_normalize_policy_path(path): _display_policy_path(path) for path in existing_paths}
    editable = {_normalize_policy_path(path): _display_policy_path(path) for path in editable_paths}
    hit = False
    code_search_calls = 0
    for event in events:
        if event.get("type") != "tool_result":
            continue
        for part in _event_parts(event):
            metadata = _mapping(part.get("metadata"))
            if metadata.get("ok") is False:
                continue
            tool_name = str(metadata.get("tool_name") or "")
            data = _mapping(metadata.get("data"))
            if tool_name in _SOURCE_READ_TOOLS:
                for path in _read_paths(tool_name, data):
                    normalized = _normalize_policy_path(path)
                    read_paths.setdefault(normalized, _display_policy_path(path))
                    if code_search_before_mutation and not first_mutation_seen and normalized in candidates:
                        candidate_reads.add(normalized)
            elif tool_name in _MUTATION_TOOLS:
                mutation_seen = True
                first_mutation_seen = True
                for path in _mutation_paths(tool_name, data):
                    normalized = _normalize_policy_path(path)
                    display = _display_policy_path(path)
                    mutated_paths.setdefault(normalized, display)
                    if normalized in existing and normalized not in read_paths:
                        unread_paths.setdefault(normalized, existing[normalized])
            elif tool_name == "code_search":
                code_search_seen = True
                code_search_calls += 1
                if not first_mutation_seen:
                    code_search_before_mutation = True
                hits = data.get("hits")
                if isinstance(hits, list):
                    paths = {
                        _normalize_policy_path(str(item.get("path")))
                        for item in hits[:5]
                        if isinstance(item, dict) and item.get("path")
                    }
                    for item in hits[:5]:
                        if isinstance(item, dict) and item.get("path"):
                            path = str(item["path"])
                            candidates.setdefault(_normalize_policy_path(path), _display_policy_path(path))
                    hit = hit or bool(relevant.intersection(paths))
    new_files = {
        normalized: display
        for normalized, display in mutated_paths.items()
        if existing and normalized not in existing
    }
    out_of_scope = {
        normalized: display
        for normalized, display in mutated_paths.items()
        if editable and normalized not in editable
    }
    legacy_unscoped_violation = mutation_seen and not existing and not read_paths
    retrieval_violation = bool(
        retrieval_required
        and retrieval_mode == "vector"
        and (not code_search_before_mutation or not candidate_reads)
    )
    return {
        "source_read_policy_violation": bool(unread_paths) or legacy_unscoped_violation,
        "relevant_file_hit_at_5": hit if code_search_seen else None,
        "unread_edited_paths": sorted(unread_paths.values(), key=str.casefold),
        "stale_read_paths": [],
        "new_files": sorted(new_files.values(), key=str.casefold),
        "out_of_scope_paths": sorted(out_of_scope.values(), key=str.casefold),
        "code_search_call_count": code_search_calls,
        "retrieval_candidate_read_count": len(candidate_reads),
        "retrieval_policy_violation": retrieval_violation,
    }


def _read_paths(tool_name: str, data: dict[str, Any]) -> list[str]:
    if tool_name == "view":
        return [str(data["path"])] if data.get("path") else []
    files = data.get("files")
    if not isinstance(files, list):
        return []
    return [str(item["path"]) for item in files if isinstance(item, dict) and item.get("path")]


def _mutation_paths(tool_name: str, data: dict[str, Any]) -> list[str]:
    if tool_name == "apply_patch":
        changed = data.get("changed_files")
        return [str(path) for path in changed] if isinstance(changed, list) else []
    return [str(data["path"])] if data.get("path") else []


def _display_policy_path(path: str) -> str:
    return str(path).replace("\\", "/").removeprefix("./")


def _normalize_policy_path(path: str) -> str:
    return _display_policy_path(path).casefold()


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
