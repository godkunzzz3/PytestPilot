"""Conservative token-cost accounting and hard budget tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from firstcoder.providers.types import TokenUsage


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
