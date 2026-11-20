"""Estimate a routing choice from the Part 2 runs, with no new model calls.

Part 2 ran every held-out case on every tier. If a router sends case X to the small tier, the
result for X is what the small tier did in Part 2. Add the router's own overhead and you have an
honest estimate. It lets Part 4 sweep a threshold without paying for every setting. The final
choice is still run for real through the gateway before it goes on the scoreboard.
"""
from __future__ import annotations

from evals.report import stats


def replay(chosen: dict[str, str], baseline: list[dict], tier_labels: dict[str, str],
           overhead: dict[str, dict] | None = None) -> dict:
    """chosen: case id -> tier. baseline: Part 2 runs. tier_labels: tier -> config label.
    overhead: case id -> {"cost_usd", "seconds"} the router spent on that case (per run).
    Cases with no Part 2 runs are skipped and counted in 'skipped'."""
    overhead = overhead or {}
    by_key: dict = {}
    for r in baseline:
        by_key.setdefault((r["config"], r["id"]), []).append(r)
    rows, skipped = [], 0
    for case_id, tier in chosen.items():
        runs = by_key.get((tier_labels[tier], case_id))
        if not runs:
            skipped += 1
            continue
        extra = overhead.get(case_id, {})
        for r in runs:
            rows.append({**r, "cost_usd": r["cost_usd"] + extra.get("cost_usd", 0.0),
                         "seconds": r["seconds"] + extra.get("seconds", 0.0)})
    out = stats(rows)
    out["skipped"] = skipped
    return out
