import json

from benchmark.deepseek_paired import paired_plan, provider_audit_config, validate_provider
from firstcoder.providers.openai_compatible import OpenAICompatibleProvider


class _UnusedClient:
    pass


def _provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="deepseek",
        model="deepseek-v4-flash",
        api_key="must-not-appear",
        base_url="https://api.deepseek.com",
        extra_body={"thinking": {"type": "disabled"}},
        default_max_tokens=4096,
        default_temperature=0,
        sdk_max_retries=0,
        client=_UnusedClient(),
    )


def test_paired_plan_is_interleaved_and_has_twelve_runs() -> None:
    plan = paired_plan()

    assert len(plan) == 12
    assert [(row["task_id"], row["retrieval_mode"], row["run_index"]) for row in plan[:4]] == [
        ("service_repository_contract", "baseline", 1),
        ("service_repository_contract", "vector", 1),
        ("parser_dispatch", "baseline", 1),
        ("parser_dispatch", "vector", 1),
    ]
    assert plan[-1] == {"task_id": "parser_dispatch", "run_index": 3, "retrieval_mode": "vector"}


def test_provider_audit_config_is_valid_and_contains_no_api_key() -> None:
    provider = _provider()

    validate_provider(provider)
    payload = json.dumps(provider_audit_config(provider, budget_limit_usd=0.25))

    assert "must-not-appear" not in payload
    assert "api_key" not in payload.lower()
    assert '"sdk_max_retries": 0' in payload
