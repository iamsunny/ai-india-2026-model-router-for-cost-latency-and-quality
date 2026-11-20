import pytest

from router.catalog import TIERS, UnknownModelError, load_catalog
from router.cost import usage_cost


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def test_every_tier_has_both_providers_and_a_price(catalog):
    for tier in TIERS:
        for provider in ("openai", "anthropic"):
            spec = catalog.tier_spec(tier, provider)
            assert spec in catalog.models


def test_openai_tiers_are_the_ones_the_plan_names(catalog):
    assert catalog.tier_spec("small") == "openai:gpt-5.4-nano"
    assert catalog.tier_spec("mid") == "openai:gpt-5.4-mini"
    assert catalog.tier_spec("large") == "openai:gpt-5.4"


def test_unknown_tier_is_an_error(catalog):
    with pytest.raises(UnknownModelError):
        catalog.tier_spec("huge")


def test_dated_model_names_find_the_right_price(catalog):
    assert catalog.price_for("gpt-5.4-mini-2026-03-17").name == "gpt-5.4-mini"
    assert catalog.price_for("gpt-5.4-2026-03-05").name == "gpt-5.4"      # not gpt-5.4-mini or nano
    assert catalog.price_for("gpt-5.4-nano").name == "gpt-5.4-nano"
    assert catalog.price_for("claude-haiku-5-5-20261007").name == "claude-haiku-5-5"


def test_unknown_model_name_is_an_error_not_a_zero_price(catalog):
    with pytest.raises(UnknownModelError):
        catalog.price_for("scripted")


def usage(model, tokens_in, tokens_out, **details):
    u = {"input_tokens": tokens_in, "output_tokens": tokens_out}
    if details:
        u["input_token_details"] = details
    return {model: u}


@pytest.mark.parametrize("tokens_in, tokens_out, expected", [
    (677, 27, 0.000629),     # the three questions in the cost table of Agents Part 11
    (315, 69, 0.000547),
    (897, 126, 0.001240),
])
def test_cost_matches_agents_part_11(catalog, tokens_in, tokens_out, expected):
    cost = usage_cost(usage("gpt-5.4-mini", tokens_in, tokens_out), catalog)["cost_usd"]
    assert round(cost, 6) == expected


def test_cached_tokens_are_priced_at_the_cached_rate(catalog):
    # 1000 input tokens, 600 of them read from the cache, on gpt-5.4-mini
    cost = usage_cost(usage("gpt-5.4-mini", 1000, 100, cache_read=600), catalog)
    expected = (400 * 0.75 + 600 * 0.075 + 100 * 4.50) / 1e6
    assert cost["cost_usd"] == pytest.approx(expected)
    assert cost["cached_tokens"] == 600


def test_claude_cache_writes_have_their_own_price(catalog):
    cost = usage_cost(usage("claude-haiku-5-5", 1000, 50, cache_read=300, cache_creation=200), catalog)
    expected = (500 * 0.10 + 300 * 0.01 + 200 * 0.125 + 50 * 0.50) / 1e6
    assert cost["cost_usd"] == pytest.approx(expected)


def test_a_conversation_that_used_two_models_is_priced_per_model(catalog):
    both = {**usage("gpt-5.4-nano", 500, 20), **usage("gpt-5.4", 800, 100)}
    cost = usage_cost(both, catalog)
    expected = (500 * 0.20 + 20 * 1.25 + 800 * 2.50 + 100 * 15.00) / 1e6
    assert cost["cost_usd"] == pytest.approx(expected)
    assert set(cost["by_model"]) == {"openai:gpt-5.4-nano", "openai:gpt-5.4"}
    assert cost["input_tokens"] == 1300 and cost["output_tokens"] == 120
