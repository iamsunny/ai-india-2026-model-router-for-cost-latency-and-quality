"""Part 6: is the router still as good as the day you measured it?

Prices, models and questions all change. Three checks:

- Quality gate: run the held-out set again. A drop of 2 cases or more (the noise rule) fails.
- Cost gate: cost per conversation up by more than 20 percent fails.
- Escalation gate: escalation rate up by more than 10 points fails. This one moves first, because a cheap
  model that quietly gets worse fails the cascade check before users complain.
Plus two housekeeping warnings: prices not checked for 30 days, and a model with no price in models.yaml.
"""
from __future__ import annotations

from datetime import date

from evals.report import NOISE_CASES
from router.catalog import Catalog, UnknownModelError

MAX_COST_INCREASE = 0.20
MAX_ESCALATION_INCREASE = 0.10
MAX_PRICE_AGE_DAYS = 30


def gate(current: dict, baseline: dict, runs_per_case: int = 3) -> list[str]:
    """Return the reasons the router fails the gate, or an empty list. Both dicts have passed, runs,
    cost_per_conversation and escalation_rate."""
    failures = []
    lost = baseline["passed"] / baseline["runs"] - current["passed"] / current["runs"]
    if lost * current["runs"] >= NOISE_CASES * runs_per_case:
        failures.append(f"quality dropped: {current['passed']} of {current['runs']} passed, "
                        f"was {baseline['passed']} of {baseline['runs']}")
    base_cost = baseline["cost_per_conversation"]
    if base_cost and current["cost_per_conversation"] > base_cost * (1 + MAX_COST_INCREASE):
        failures.append(f"cost per conversation rose from ${base_cost:.5f} to ${current['cost_per_conversation']:.5f}")
    if current["escalation_rate"] - baseline["escalation_rate"] > MAX_ESCALATION_INCREASE:
        failures.append(f"escalation rate rose from {baseline['escalation_rate']:.0%} to {current['escalation_rate']:.0%}")
    return failures


def price_warnings(catalog: Catalog, records: list[dict], today: date | None = None) -> list[str]:
    today = today or date.today()
    out = []
    age = (today - date.fromisoformat(catalog.checked)).days
    if age > MAX_PRICE_AGE_DAYS:
        out.append(f"prices in models.yaml were checked {age} days ago. Check the vendors' pricing pages.")
    for spec in sorted({m for r in records for m in r["models"]}):
        try:
            catalog.price_for(spec)
        except UnknownModelError:
            out.append(f"{spec} has no price in models.yaml, so its cost is wrong")
    return out
