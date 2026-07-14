import asyncio
import json

import pytest

from firstcoder.eval.costs import (
    BudgetedProvider,
    CostBudget,
    RequestBoundaryBudget,
    RequestBudgetExceeded,
    estimate_deepseek_cost,
)
from firstcoder.providers.base import ChatProvider
from firstcoder.providers.types import ChatMessage, ChatRequest, ChatResponse, ChatStreamEvent, TokenUsage


class _FakeProvider(ChatProvider):
    def __init__(self, responses: list[ChatResponse], *, fail: Exception | None = None) -> None:
        self.responses = list(responses)
        self.fail = fail
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return self.responses.pop(0)


class _FakeStreamingProvider(_FakeProvider):
    async def astream(self, request: ChatRequest):
        self.calls += 1
        response = self.responses.pop(0)
        yield ChatStreamEvent(kind="message_started")
        yield ChatStreamEvent(kind="text_delta", text=response.content)
        yield ChatStreamEvent(kind="message_completed", response=response)


def _request(*, max_tokens: int = 100) -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role="user", content="small prompt")], max_tokens=max_tokens)


def _response(usage: TokenUsage | None) -> ChatResponse:
    return ChatResponse(provider="fake", model="fake-model", content="ok", usage=usage)


def test_deepseek_cost_uses_cache_miss_price_for_all_prompt_tokens() -> None:
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=500_000,
        total_tokens=1_500_000,
        prompt_cache_hit_tokens=900_000,
        prompt_cache_miss_tokens=100_000,
    )

    result = estimate_deepseek_cost(usage)

    assert result["prompt_tokens"] == 1_000_000
    assert result["completion_tokens"] == 500_000
    assert result["prompt_cache_hit_tokens"] == 900_000
    assert result["prompt_cache_miss_tokens"] == 100_000
    assert result["estimated_cost_usd"] == 0.28
    json.dumps(result)


def test_cost_budget_stops_at_limit_and_preserves_missing_usage() -> None:
    budget = CostBudget(limit_usd=1.0)

    first = budget.record(TokenUsage(input_tokens=1_000_000, output_tokens=0, total_tokens=1_000_000))
    missing = budget.record(None)

    assert first["cumulative_estimated_cost_usd"] == 0.14
    assert first["budget_exhausted"] is False
    assert missing["prompt_tokens"] is None
    assert missing["estimated_cost_usd"] is None
    assert missing["cumulative_estimated_cost_usd"] == 0.14

    final = budget.record(TokenUsage(input_tokens=0, output_tokens=4_000_000, total_tokens=4_000_000))

    assert final["budget_exhausted"] is True
    assert budget.can_continue is False
    json.dumps(final)


def test_request_boundary_denies_before_provider_call() -> None:
    provider = _FakeProvider([_response(TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2))])
    budget = RequestBoundaryBudget(limit_usd=0.000001, safety_factor=1.2, default_max_output_tokens=4096)
    guarded = BudgetedProvider(provider, budget)

    with pytest.raises(RequestBudgetExceeded) as captured:
        guarded.complete(_request(max_tokens=4096))

    assert provider.calls == 0
    assert captured.value.details["reason"] == "reservation_exceeds_limit"
    assert budget.snapshot()["request_count"] == 0


def test_request_boundary_commits_actual_usage_conservatively() -> None:
    usage = TokenUsage(
        input_tokens=1_000,
        output_tokens=500,
        total_tokens=1_500,
        prompt_cache_hit_tokens=900,
        prompt_cache_miss_tokens=100,
    )
    provider = _FakeProvider([_response(usage)])
    budget = RequestBoundaryBudget(limit_usd=0.25, default_max_output_tokens=4096)
    guarded = BudgetedProvider(provider, budget)

    response = guarded.complete(_request())

    assert response.usage is usage
    assert guarded.call_count == 1
    assert budget.snapshot()["committed_cost_usd"] == 0.00028
    assert budget.snapshot()["usage_unknown"] is False


def test_missing_usage_stops_all_future_requests() -> None:
    provider = _FakeProvider([_response(None), _response(TokenUsage(1, 1, 2))])
    budget = RequestBoundaryBudget(limit_usd=0.25)
    guarded = BudgetedProvider(provider, budget)

    assert guarded.complete(_request()).usage is None
    with pytest.raises(RequestBudgetExceeded) as captured:
        guarded.complete(_request())

    assert provider.calls == 1
    assert captured.value.details["reason"] == "usage_unknown"


def test_provider_error_after_send_marks_usage_unknown() -> None:
    provider = _FakeProvider([], fail=RuntimeError("network failed"))
    budget = RequestBoundaryBudget(limit_usd=0.25)
    guarded = BudgetedProvider(provider, budget)

    with pytest.raises(RuntimeError, match="network failed"):
        guarded.complete(_request())

    assert budget.snapshot()["usage_unknown"] is True
    assert budget.snapshot()["stop_reason"] == "provider_error_after_send"


def test_streaming_usage_is_committed_once() -> None:
    usage = TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120)
    provider = _FakeStreamingProvider([_response(usage)])
    budget = RequestBoundaryBudget(limit_usd=0.25)
    guarded = BudgetedProvider(provider, budget)

    async def consume() -> list[ChatStreamEvent]:
        return [event async for event in guarded.astream(_request())]

    events = asyncio.run(consume())

    assert events[-1].kind == "message_completed"
    assert budget.snapshot()["request_count"] == 1
    assert budget.snapshot()["committed_cost_usd"] == 0.0000196
