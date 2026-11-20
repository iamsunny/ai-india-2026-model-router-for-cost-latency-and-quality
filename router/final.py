"""The final setup, in one place: a router (rules or classifier), the cascade, and the telemetry log.

Part 6 and the drift check both build their gateway here, so they measure the same thing.
"""
from __future__ import annotations

from router.catalog import load_catalog
from router.classifier import DEFAULT_THRESHOLD, ClassifierRouter, LLMClassifier
from router.gateway import Gateway
from router.resilience import CascadeExecutor, EscalationBudget, resilient_factory
from router.rules import RulesRouter, load_tiers


def build_final_gateway(strategy: str = "rules", second_provider: bool = False, tiers: dict | None = None,
                        threshold: float = DEFAULT_THRESHOLD, brain: str = "openai:gpt-5.4-nano", classify=None,
                        sink=None, model_factory=resilient_factory, catalog=None) -> Gateway:
    """strategy is "rules" or "classifier". `classify` lets a test or an offline run supply the classifier."""
    catalog = catalog or load_catalog()
    tiers = tiers if tiers is not None else load_tiers()[0]
    if strategy == "classifier":
        router = ClassifierRouter(classify or LLMClassifier(brain), tiers, threshold=threshold,
                                  name="classifier router")
    elif strategy == "rules":
        router = RulesRouter(tiers)
    else:
        raise ValueError(f"strategy must be 'rules' or 'classifier', not {strategy!r}")
    executor = CascadeExecutor(second_provider=second_provider, budget=EscalationBudget(max_rate=0.25, window=100))
    return Gateway(router, executor=executor, catalog=catalog, model_factory=model_factory, sink=sink,
                   name=f"final setup ({strategy} + cascade)")
