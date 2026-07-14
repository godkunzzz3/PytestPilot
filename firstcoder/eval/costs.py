"""Conservative token-cost accounting and hard budget tracking."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from firstcoder.context.token_budget import estimate_text_tokens
from firstcoder.providers.base import ChatProvider
from firstcoder.providers.types import ChatRequest, ChatResponse, ChatStreamEvent, TokenUsage


DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION = 0.14
DEEPSEEK_OUTPUT_USD_PER_MILLION = 0.28


def estimate_deepseek_cost(usage: TokenUsage | None) -> dict[str, Any]:
    """Estimate cost with every prompt token charged at the cache-miss rate."""

    prompt_tokens = usage.input_tokens if usage is not None else None
    completion_tokens = usage.output_tokens if usage is not None else None
    total_tokens = usage.total_tokens if usage is not None else None
    cost = None
    if prompt_tokens is not None and completion_tokens is not None:
        cost = round(
            prompt_tokens * DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION / 1_000_000
            + completion_tokens * DEEPSEEK_OUTPUT_USD_PER_MILLION / 1_000_000,
            9,
        )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_cache_hit_tokens": usage.prompt_cache_hit_tokens if usage is not None else None,
        "prompt_cache_miss_tokens": usage.prompt_cache_miss_tokens if usage is not None else None,
        "estimated_cost_usd": cost,
    }


@dataclass(slots=True)
class CostBudget:
    limit_usd: float = 1.0
    cumulative_estimated_cost_usd: float = 0.0

    @property
    def can_continue(self) -> bool:
        return self.cumulative_estimated_cost_usd < self.limit_usd

    def record(self, usage: TokenUsage | None) -> dict[str, Any]:
        result = estimate_deepseek_cost(usage)
        cost = result["estimated_cost_usd"]
        if cost is not None:
            self.cumulative_estimated_cost_usd = round(self.cumulative_estimated_cost_usd + cost, 9)
        result["cumulative_estimated_cost_usd"] = self.cumulative_estimated_cost_usd
        result["cost_limit_usd"] = self.limit_usd
        result["budget_exhausted"] = not self.can_continue
        return result


class RequestBudgetExceeded(RuntimeError):
    """A structured refusal raised before an HTTP request is sent."""

    def __init__(self, details: dict[str, Any]) -> None:
        self.details = details
        super().__init__(
            f"request budget stopped: {details.get('reason', 'unknown')} "
            f"(committed={details.get('committed_cost_usd')}, "
            f"reservation={details.get('reservation_cost_usd')}, limit={details.get('limit_usd')})"
        )


@dataclass(frozen=True, slots=True)
class RequestReservation:
    estimated_input_tokens: int
    reserved_input_tokens: int
    reserved_output_tokens: int
    reservation_cost_usd: float


@dataclass(slots=True)
class RequestBoundaryBudget:
    """Reserve conservatively before every provider request, then commit actual usage.

    The ledger deliberately stores only counts and prices. Prompt text, request bodies, headers,
    and credentials never enter budget events or snapshots.
    """

    limit_usd: float = 0.25
    safety_factor: float = 1.20
    default_max_output_tokens: int = 4096
    committed_cost_usd: float = 0.0
    usage_unknown: bool = False
    stop_reason: str | None = None
    _lock: threading.Lock = field(init=False, repr=False)
    _events: list[dict[str, Any]] = field(init=False, repr=False, default_factory=list)
    _request_count: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        if self.limit_usd <= 0:
            raise ValueError("limit_usd must be greater than zero")
        if self.safety_factor < 1:
            raise ValueError("safety_factor must be at least 1")
        if self.default_max_output_tokens <= 0:
            raise ValueError("default_max_output_tokens must be greater than zero")
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._request_count = 0

    def reserve(self, request: ChatRequest) -> RequestReservation:
        estimated = _estimate_request_tokens(request)
        reserved_input = math.ceil(estimated * self.safety_factor)
        reserved_output = request.max_tokens or self.default_max_output_tokens
        reservation_cost = _token_cost(reserved_input, reserved_output)
        with self._lock:
            reason = self.stop_reason
            if self.usage_unknown:
                reason = "usage_unknown"
            elif self.committed_cost_usd + reservation_cost > self.limit_usd:
                reason = "reservation_exceeds_limit"
            if reason is not None:
                self.stop_reason = reason
                details = self._denial_details(reason, reservation_cost)
                self._events.append({"kind": "request_denied", **details})
                raise RequestBudgetExceeded(details)
            self._request_count += 1
            self._events.append(
                {
                    "kind": "request_reserved",
                    "request_number": self._request_count,
                    "estimated_input_tokens": estimated,
                    "reserved_input_tokens": reserved_input,
                    "reserved_output_tokens": reserved_output,
                    "reservation_cost_usd": reservation_cost,
                }
            )
        return RequestReservation(estimated, reserved_input, reserved_output, reservation_cost)

    def commit(self, reservation: RequestReservation, usage: TokenUsage | None) -> dict[str, Any]:
        if not _usage_is_complete(usage):
            self.mark_usage_unknown("usage_missing")
            return {**estimate_deepseek_cost(usage), **self.snapshot()}
        assert usage is not None
        cost = _token_cost(int(usage.input_tokens or 0), int(usage.output_tokens or 0))
        with self._lock:
            self.committed_cost_usd = round(self.committed_cost_usd + cost, 9)
            self._events.append(
                {
                    "kind": "request_committed",
                    "request_number": self._request_count,
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "prompt_cache_hit_tokens": usage.prompt_cache_hit_tokens,
                    "prompt_cache_miss_tokens": usage.prompt_cache_miss_tokens,
                    "estimated_cost_usd": cost,
                    "committed_cost_usd": self.committed_cost_usd,
                }
            )
        return {**estimate_deepseek_cost(usage), **self.snapshot()}

    def mark_usage_unknown(self, reason: str) -> None:
        with self._lock:
            self.usage_unknown = True
            self.stop_reason = reason
            self._events.append(
                {
                    "kind": "usage_unknown",
                    "request_number": self._request_count,
                    "reason": reason,
                    "committed_cost_usd": self.committed_cost_usd,
                }
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "limit_usd": self.limit_usd,
                "safety_factor": self.safety_factor,
                "default_max_output_tokens": self.default_max_output_tokens,
                "committed_cost_usd": self.committed_cost_usd,
                "request_count": self._request_count,
                "usage_unknown": self.usage_unknown,
                "stop_reason": self.stop_reason,
                "events": [dict(event) for event in self._events],
            }

    def _denial_details(self, reason: str, reservation_cost: float) -> dict[str, Any]:
        return {
            "reason": reason,
            "committed_cost_usd": self.committed_cost_usd,
            "reservation_cost_usd": reservation_cost,
            "limit_usd": self.limit_usd,
            "usage_unknown": self.usage_unknown,
        }


class BudgetedProvider(ChatProvider):
    """Provider decorator enforcing a shared request-boundary budget ledger."""

    def __init__(self, provider: ChatProvider, budget: RequestBoundaryBudget) -> None:
        self.provider = provider
        self.budget = budget
        self.call_count = 0

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
        reservation = self.budget.reserve(request)
        self.call_count += 1
        try:
            response = self.provider.complete(request)
        except Exception:
            self.budget.mark_usage_unknown("provider_error_after_send")
            raise
        self.budget.commit(reservation, response.usage)
        return response

    async def astream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        reservation = self.budget.reserve(request)
        self.call_count += 1
        committed = False
        try:
            async for event in self.provider.astream(request):
                if event.kind == "message_completed" and event.response is not None and not committed:
                    self.budget.commit(reservation, event.response.usage)
                    committed = True
                yield event
        except Exception:
            if not committed:
                self.budget.mark_usage_unknown("provider_error_after_send")
            raise
        if not committed:
            self.budget.mark_usage_unknown("stream_usage_missing")


def _usage_is_complete(usage: TokenUsage | None) -> bool:
    return bool(
        usage is not None
        and isinstance(usage.input_tokens, int)
        and isinstance(usage.output_tokens, int)
        and isinstance(usage.total_tokens, int)
    )


def _token_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        input_tokens * DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION / 1_000_000
        + output_tokens * DEEPSEEK_OUTPUT_USD_PER_MILLION / 1_000_000,
        9,
    )


def _estimate_request_tokens(request: ChatRequest) -> int:
    payload = {
        "messages": [
            {
                "role": message.role,
                "content": message.content,
                "name": message.name,
                "tool_call_id": message.tool_call_id,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in message.tool_calls
                ],
            }
            for message in request.messages
        ],
        "tools": [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in request.tools
        ],
        "tool_choice": str(request.tool_choice),
        "extra_body": request.extra_body,
    }
    return estimate_text_tokens(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
