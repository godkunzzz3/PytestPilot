"""Run the audited, interleaved DeepSeek retrieval experiment.

This module is intentionally opt-in. Importing it, printing its plan, and running its unit tests
never create a Provider. The normal execution path must be launched from a login shell that has
DEEPSEEK_API_KEY and every request is guarded by one shared RequestBoundaryBudget.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.local_pytest.runner import (
    _VectorToolFactory,
    load_tasks_jsonl,
    run_one_task,
    write_summary_json,
)
from firstcoder.agent.loop_limits import AgentLoopLimits
from firstcoder.eval.adapter import FirstCoderCodingAgentAdapter
from firstcoder.eval.costs import (
    BudgetedProvider,
    RequestBoundaryBudget,
    RequestBudgetExceeded,
    estimate_deepseek_cost,
)
from firstcoder.providers.factory import create_provider
from firstcoder.providers.openai_compatible import OpenAICompatibleProvider
from firstcoder.providers.types import ChatMessage, ChatRequest


MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
THINKING = {"thinking": {"type": "disabled"}}
TASK_IDS = ("service_repository_contract", "parser_dispatch")
MAX_OUTPUT_TOKENS = 4096
MAX_TOOL_ROUNDS = 8
FIRSTCODER_RETRY = 0
MAX_ATTEMPTS = 1
TEMPERATURE = 0.0


def paired_plan() -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for run_index in range(1, 4):
        for task_id in TASK_IDS:
            for retrieval_mode in ("baseline", "vector"):
                plan.append(
                    {
                        "task_id": task_id,
                        "run_index": run_index,
                        "retrieval_mode": retrieval_mode,
                    }
                )
    return plan


def provider_audit_config(provider: OpenAICompatibleProvider, *, budget_limit_usd: float) -> dict[str, Any]:
    params = provider._build_completion_params(
        ChatRequest(messages=[ChatMessage(role="user", content="configuration audit")])
    )
    return {
        "provider": provider.name,
        "base_url": provider.base_url,
        "model": provider.model,
        "thinking": params.get("extra_body"),
        "temperature": params.get("temperature"),
        "max_tokens": params.get("max_tokens"),
        "sdk_max_retries": provider.sdk_max_retries,
        "firstcoder_retry": FIRSTCODER_RETRY,
        "max_attempts": MAX_ATTEMPTS,
        "max_tool_rounds": MAX_TOOL_ROUNDS,
        "concurrency": 1,
        "budget_limit_usd": budget_limit_usd,
        "budget_safety_factor": 1.20,
    }


def validate_provider(provider: OpenAICompatibleProvider) -> None:
    config = provider_audit_config(provider, budget_limit_usd=0.25)
    expected = {
        "provider": "deepseek",
        "base_url": BASE_URL,
        "model": MODEL,
        "thinking": THINKING,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "sdk_max_retries": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"DeepSeek experiment configuration mismatch: {mismatches}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the budgeted DeepSeek paired retrieval audit.")
    parser.add_argument("--out-dir", default="runs/audit-hardening")
    parser.add_argument("--tasks", default="benchmark/local_pytest/tasks.sample.jsonl")
    parser.add_argument("--budget-limit-usd", type=float, default=0.25)
    parser.add_argument("--starting-cost-usd", type=float, default=0.0)
    parser.add_argument("--rerun-policy-violations-from", default=None)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args(argv)
    if args.print_plan:
        print(json.dumps(paired_plan(), ensure_ascii=False, indent=2))
        return 0

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_payload: dict[str, Any] | None = None
    selected_plan = paired_plan()
    starting_request_count = 0
    if args.rerun_policy_violations_from:
        source_payload = json.loads(Path(args.rerun_policy_violations_from).read_text(encoding="utf-8"))
        selected_plan = [
            {
                "task_id": row["task_id"],
                "run_index": row["run_index"],
                "retrieval_mode": row["retrieval_mode"],
                "remediation_rerun": True,
            }
            for row in source_payload.get("runs", [])
            if row.get("retrieval_policy_violation")
        ]
        starting_request_count = int(source_payload.get("request_budget", {}).get("request_count") or 0)
        if not selected_plan:
            raise RuntimeError("source experiment has no retrieval policy violations to rerun")
    result_path = out_dir / (
        "paired-remediation-results.json" if source_payload is not None else "paired-experiment-results.json"
    )
    raw_provider = create_provider("deepseek")
    if not isinstance(raw_provider, OpenAICompatibleProvider):
        raise RuntimeError("deepseek must use OpenAICompatibleProvider")
    validate_provider(raw_provider)
    budget = RequestBoundaryBudget(
        limit_usd=args.budget_limit_usd,
        safety_factor=1.20,
        default_max_output_tokens=MAX_OUTPUT_TOKENS,
        committed_cost_usd=args.starting_cost_usd,
    )
    payload: dict[str, Any] = {
        "completed": False,
        "budget_exhausted": False,
        "provider_config": provider_audit_config(raw_provider, budget_limit_usd=args.budget_limit_usd),
        "plan": selected_plan,
        "smoke": source_payload.get("smoke") if source_payload is not None else None,
        "runs": [],
        "request_budget": budget.snapshot(),
        "starting_cost_usd": args.starting_cost_usd,
        "starting_request_count": starting_request_count,
        "source_experiment": str(Path(args.rerun_policy_violations_from).resolve()) if source_payload else None,
    }
    _save(result_path, payload, budget)

    try:
        if source_payload is None:
            smoke_provider = BudgetedProvider(raw_provider, budget)
            smoke_response = smoke_provider.complete(
                ChatRequest(
                    messages=[ChatMessage(role="user", content="Reply with exactly AUDIT_SMOKE_OK.")],
                    temperature=TEMPERATURE,
                    max_tokens=64,
                )
            )
            payload["smoke"] = {
                "passed": bool(smoke_response.content.strip()),
                "actual_model": smoke_response.model,
                "finish_reason": smoke_response.finish_reason,
                "request_boundary_reserved": budget.snapshot()["request_count"] == 1,
                **estimate_deepseek_cost(smoke_response.usage),
                "sdk_max_retries": raw_provider.sdk_max_retries,
                "thinking": raw_provider.extra_body,
                "temperature": raw_provider.default_temperature,
            }
            _save(result_path, payload, budget)
            if smoke_response.usage is None:
                raise RuntimeError("smoke usage missing")

        tasks = {task.id: task for task in load_tasks_jsonl(args.tasks)}
        missing = set(TASK_IDS).difference(tasks)
        if missing:
            raise RuntimeError(f"missing paired tasks: {', '.join(sorted(missing))}")

        for step, item in enumerate(selected_plan, start=1):
            task = tasks[item["task_id"]]
            mode = str(item["retrieval_mode"])
            run_index = int(item["run_index"])
            vector_factory = _VectorToolFactory(out_dir / "qdrant" / f"step-{step:02d}") if mode == "vector" else None
            try:
                adapter = FirstCoderCodingAgentAdapter(
                    model_name_or_path=MODEL,
                    provider_name="deepseek",
                    session_root=out_dir / "sessions" / f"step-{step:02d}-{task.id}-{mode}-run-{run_index}",
                    provider_retries=FIRSTCODER_RETRY,
                    limits=AgentLoopLimits.swe_lite().with_max_tool_rounds(MAX_TOOL_ROUNDS),
                    extra_tools_factory=vector_factory,
                    request_budget=budget,
                )
                row = run_one_task(
                    task=task,
                    workdir=out_dir / "work",
                    adapter=adapter,
                    force=True,
                    retrieval_mode=mode,
                )
            finally:
                if vector_factory is not None:
                    vector_factory.close()
            row.update(
                {
                    "task_id": task.id,
                    "run_index": run_index,
                    "retrieval_mode": mode,
                    "retrieval_required": task.retrieval_required,
                    "edited_paths": row.get("changed_paths", []),
                    "cumulative_new_cost_usd": budget.committed_cost_usd,
                    "sdk_max_retries": raw_provider.sdk_max_retries,
                    "firstcoder_retry": FIRSTCODER_RETRY,
                    "max_attempts": MAX_ATTEMPTS,
                    "thinking": raw_provider.extra_body,
                    "temperature": raw_provider.default_temperature,
                    "remediation_rerun": bool(item.get("remediation_rerun")),
                }
            )
            payload["runs"].append(row)
            _save(result_path, payload, budget)
    except RequestBudgetExceeded as exc:
        payload["budget_exhausted"] = True
        payload["stop_reason"] = exc.details
        _save(result_path, payload, budget)
        return 3
    except Exception as exc:
        payload["stop_reason"] = {
            "type": type(exc).__name__,
            "message": str(exc)[:1000],
        }
        _save(result_path, payload, budget)
        raise

    payload["completed"] = len(payload["runs"]) == len(payload["plan"])
    _save(result_path, payload, budget)
    write_summary_json(out_dir / "paired-run-summary.json", payload["runs"])
    print(f"Wrote paired experiment results: {result_path}")
    print(f"Completed {len(payload['runs'])}/{len(payload['plan'])} tasks")
    print(f"Conservative committed cost: ${budget.committed_cost_usd:.9f}")
    return 0 if payload["completed"] else 2


def _save(path: Path, payload: dict[str, Any], budget: RequestBoundaryBudget) -> None:
    payload["request_budget"] = budget.snapshot()
    payload["cumulative_request_count"] = int(payload.get("starting_request_count") or 0) + int(
        payload["request_budget"]["request_count"]
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
