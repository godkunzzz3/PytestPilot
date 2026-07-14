import json

from firstcoder.eval.costs import CostBudget, estimate_deepseek_cost
from firstcoder.providers.types import TokenUsage


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
