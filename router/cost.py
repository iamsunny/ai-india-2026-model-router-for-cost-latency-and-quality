"""Turn token counts into money.

This is the Part 11 cost report from the Building AI Agents series, with one change: every
model has its own price, because a routed conversation can use more than one model.
"""
from __future__ import annotations

from .catalog import Catalog, load_catalog


def usage_cost(usage: dict, catalog: Catalog | None = None) -> dict:
    """Price a conversation.

    `usage` is what LangChain's usage callback collects: one entry per model name, each with
    input_tokens, output_tokens and, when the provider reports it, input_token_details with
    cache_read and cache_creation. input_tokens already includes the cached tokens.
    """
    catalog = catalog or load_catalog()
    total = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0, "by_model": {}}
    for model_name, u in usage.items():
        price = catalog.price_for(model_name)
        tokens_in = u.get("input_tokens", 0)
        tokens_out = u.get("output_tokens", 0)
        details = u.get("input_token_details", {}) or {}
        cached = details.get("cache_read", 0)
        written = details.get("cache_creation", 0)
        fresh = tokens_in - cached - written
        cost = (fresh * price.input
                + cached * price.cached_input
                + written * (price.cache_write if price.cache_write is not None else price.input)
                + tokens_out * price.output) / 1e6
        total["input_tokens"] += tokens_in
        total["output_tokens"] += tokens_out
        total["cached_tokens"] += cached
        total["cost_usd"] += cost
        total["by_model"][price.spec] = {"input_tokens": tokens_in, "output_tokens": tokens_out,
                                         "cached_tokens": cached, "cost_usd": cost}
    return total
