"""Repair-task prompt construction kept outside AgentLoop."""

from __future__ import annotations

import json

from firstcoder.ci.models import PytestRunReport


def build_pytest_repair_prompt(
    report: PytestRunReport,
    *,
    deterministic_candidates: list[str],
    semantic_candidates: list[dict[str, object]],
    focused_command: str,
    full_command: str,
) -> str:
    evidence = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    deterministic = "\n".join(f"- {path}" for path in deterministic_candidates) or "- none"
    semantic = json.dumps(semantic_candidates, ensure_ascii=False, indent=2) if semantic_candidates else "[]"
    return (
        "Fix the Python/pytest CI failure using the smallest correct source change.\n\n"
        "Structured pytest evidence:\n"
        f"{evidence}\n\n"
        "Deterministic file candidates (higher priority):\n"
        f"{deterministic}\n\n"
        "Semantic candidates (candidate-only; never source truth):\n"
        f"{semantic}\n\n"
        "Before any write/edit/delete/apply_patch, call view or read_multi on the current source file. "
        "Do not edit from a vector-store preview. Do not modify tests. Keep the fix minimal.\n"
        f"Run the focused validation when useful: {focused_command}\n"
        f"The workflow will decide success only with the full command: {full_command}"
    )
