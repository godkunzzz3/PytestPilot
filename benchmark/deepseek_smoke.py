"""Budgeted real DeepSeek compatibility smoke test.

Run only from a shell that already contains DEEPSEEK_API_KEY. The result artifact never stores
credentials or raw SDK request/response objects.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from firstcoder.eval.costs import BudgetedProvider, RequestBoundaryBudget, estimate_deepseek_cost
from firstcoder.providers.base import ChatProvider
from firstcoder.providers.factory import create_provider
from firstcoder.providers.openai_compatible import OpenAICompatibleProvider
from firstcoder.providers.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ToolCall,
    ToolChoiceFunction,
    ToolDefinition,
)


MODEL = "deepseek-v4-flash"
THINKING_DISABLED = {"thinking": {"type": "disabled"}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cost-limit-usd", type=float, default=1.0)
    parser.add_argument("--starting-cost-usd", type=float, default=0.0)
    args = parser.parse_args(argv)
    out = Path(args.out)
    budget = RequestBoundaryBudget(
        limit_usd=args.cost_limit_usd,
        committed_cost_usd=args.starting_cost_usd,
        default_max_output_tokens=4096,
    )
    results: list[dict[str, Any]] = []

    raw_provider = create_provider("deepseek")
    if not isinstance(raw_provider, OpenAICompatibleProvider):
        raise RuntimeError("deepseek must use the OpenAI-compatible provider")
    checks = _provider_checks(raw_provider)
    provider = BudgetedProvider(raw_provider, budget)
    _save(out, checks=checks, results=results, budget=budget)

    try:
        response = provider.complete(_text_request("Reply with exactly DEEPSEEK_SMOKE_OK."))
        _record(results, budget, "non_stream_text", response, content_nonempty=bool(response.content.strip()))
        _save(out, checks=checks, results=results, budget=budget)
        _require_budget(budget)

        stream_response = asyncio.run(_stream_text(provider))
        _record(
            results,
            budget,
            "stream_text",
            stream_response,
            content_nonempty=bool(stream_response.content.strip()),
        )
        _save(out, checks=checks, results=results, budget=budget)
        _require_budget(budget)

        first_tool = provider.complete(_tool_request("Call read_value once with name='alpha'.", "read_value"))
        _record(results, budget, "single_readonly_tool_call", first_tool, **_tool_checks(first_tool, "read_value"))
        _save(out, checks=checks, results=results, budget=budget)
        _require_tool(first_tool, "read_value")
        _require_budget(budget)

        filled_messages = [
            ChatMessage(role="user", content="Call read_value once, then report its returned value."),
            _assistant_message(first_tool),
            ChatMessage(role="tool", content='{"value": 42}', tool_call_id=first_tool.tool_calls[0].id),
        ]
        filled = provider.complete(ChatRequest(messages=filled_messages, tool_choice="none", temperature=0))
        _record(
            results,
            budget,
            "tool_result_round_trip",
            filled,
            content_nonempty=bool(filled.content.strip()),
            matching_tool_call_id=True,
        )
        _save(out, checks=checks, results=results, budget=budget)
        _require_budget(budget)

        multi_messages = [ChatMessage(role="user", content="Call read_value exactly once now with name='alpha'.")]
        multi_one = provider.complete(
            ChatRequest(
                messages=multi_messages,
                tools=_tools("read_value"),
                tool_choice=ToolChoiceFunction("read_value"),
                temperature=0,
            )
        )
        _record(results, budget, "multi_tool_round_1", multi_one, **_tool_checks(multi_one, "read_value"))
        _save(out, checks=checks, results=results, budget=budget)
        _require_tool(multi_one, "read_value")
        _require_budget(budget)
        multi_messages.extend(
            [
                _assistant_message(multi_one),
                ChatMessage(role="tool", content='{"value": 42}', tool_call_id=multi_one.tool_calls[0].id),
                ChatMessage(role="user", content="Now call read_status exactly once with name='service'."),
            ]
        )
        multi_two = provider.complete(
            ChatRequest(
                messages=multi_messages,
                tools=_tools("read_status"),
                tool_choice=ToolChoiceFunction("read_status"),
                temperature=0,
            )
        )
        _record(results, budget, "multi_tool_round_2", multi_two, **_tool_checks(multi_two, "read_status"))
        _save(out, checks=checks, results=results, budget=budget)
        _require_tool(multi_two, "read_status")
        _require_budget(budget)
        multi_messages.extend(
            [
                _assistant_message(multi_two),
                ChatMessage(role="tool", content='{"status": "ready"}', tool_call_id=multi_two.tool_calls[0].id),
            ]
        )
        multi_final = provider.complete(ChatRequest(messages=multi_messages, tool_choice="none", temperature=0))
        _record(
            results,
            budget,
            "multi_tool_final",
            multi_final,
            content_nonempty=bool(multi_final.content.strip()),
            tool_call_count=0,
        )
        _save(out, checks=checks, results=results, budget=budget, completed=True)
    except Exception as exc:
        _save(
            out,
            checks=checks,
            results=results,
            budget=budget,
            completed=False,
            error_type=type(exc).__name__,
            error=str(exc)[:1000],
        )
        raise

    print(json.dumps(_summary(checks, results, budget), ensure_ascii=False, indent=2))
    return 0


def _provider_checks(provider: OpenAICompatibleProvider) -> dict[str, Any]:
    params = provider._build_completion_params(_text_request("configuration check"))
    return {
        "provider": provider.name,
        "configured_model": provider.model,
        "base_url": provider.base_url,
        "thinking_disabled": params.get("extra_body") == THINKING_DISABLED,
        "max_output_tokens": params.get("max_tokens"),
        "temperature": params.get("temperature"),
        "sdk_max_retries": provider.sdk_max_retries,
        "api_key_recorded": False,
        "reasoning_content_round_trip_implemented": False,
    }


def _text_request(content: str) -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role="user", content=content)], temperature=0)


def _tool_request(content: str, name: str) -> ChatRequest:
    return ChatRequest(
        messages=[ChatMessage(role="user", content=content)],
        tools=_tools(name),
        tool_choice=ToolChoiceFunction(name),
        temperature=0,
    )


def _tools(only: str | None = None) -> list[ToolDefinition]:
    string_parameter = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    tools = [
        ToolDefinition(name="read_value", description="Return a fixed test value.", parameters=string_parameter),
        ToolDefinition(name="read_status", description="Return a fixed readiness status.", parameters=string_parameter),
    ]
    return [tool for tool in tools if only is None or tool.name == only]


async def _stream_text(provider: ChatProvider) -> ChatResponse:
    completed: ChatResponse | None = None
    async for event in provider.astream(_text_request("Reply with exactly STREAM_OK.")):
        if event.kind == "message_completed":
            completed = event.response
    if completed is None:
        raise RuntimeError("stream ended without message_completed")
    return completed


def _assistant_message(response: ChatResponse) -> ChatMessage:
    return ChatMessage(role="assistant", content=response.content, tool_calls=response.tool_calls)


def _tool_checks(response: ChatResponse, expected_name: str) -> dict[str, Any]:
    call = response.tool_calls[0] if len(response.tool_calls) == 1 else None
    return {
        "tool_call_count": len(response.tool_calls),
        "tool_name_correct": call is not None and call.name == expected_name,
        "arguments_are_object": call is not None and isinstance(call.arguments, dict),
        "tool_call_id_preserved": bool(call and call.id),
    }


def _require_tool(response: ChatResponse, expected_name: str) -> ToolCall:
    if len(response.tool_calls) != 1 or response.tool_calls[0].name != expected_name:
        raise RuntimeError(f"expected exactly one {expected_name} tool call")
    return response.tool_calls[0]


def _record(
    results: list[dict[str, Any]],
    budget: RequestBoundaryBudget,
    name: str,
    response: ChatResponse,
    **checks: Any,
) -> None:
    cost = estimate_deepseek_cost(response.usage)
    cost["cumulative_estimated_cost_usd"] = budget.committed_cost_usd
    results.append(
        {
            "name": name,
            "passed": all(value is not False for value in checks.values()),
            "actual_model": response.model,
            "finish_reason": response.finish_reason,
            "usage_present": response.usage is not None,
            **checks,
            **cost,
        }
    )


def _require_budget(budget: RequestBoundaryBudget) -> None:
    snapshot = budget.snapshot()
    if snapshot["usage_unknown"] or snapshot["committed_cost_usd"] >= snapshot["limit_usd"]:
        raise RuntimeError(f"request budget stopped: {snapshot['stop_reason'] or 'cost_limit_reached'}")


def _save(
    path: Path,
    *,
    checks: dict[str, Any],
    results: list[dict[str, Any]],
    budget: RequestBoundaryBudget,
    completed: bool = False,
    **extra: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed": completed,
        "provider_checks": checks,
        "results": results,
        "request_budget": budget.snapshot(),
        **extra,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summary(checks: dict[str, Any], results: list[dict[str, Any]], budget: RequestBoundaryBudget) -> dict[str, Any]:
    return {
        "provider_checks": checks,
        "smoke_passed": all(row["passed"] and row["usage_present"] for row in results),
        "request_count": len(results),
        "cumulative_estimated_cost_usd": budget.committed_cost_usd,
    }


if __name__ == "__main__":
    raise SystemExit(main())
