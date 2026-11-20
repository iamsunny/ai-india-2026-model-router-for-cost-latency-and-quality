"""Part 6: read the gateway's request log and answer the questions that matter.

The gateway writes one record per request (route, models tried, tokens, cost, time, escalation, error).
Nothing here calls a model. If a route is not logged, its saving is a guess.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def load_records(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. 0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def summarize(records: list[dict]) -> dict:
    n = len(records)
    cost = sum(r["cost_usd"] for r in records)
    secs = [r["seconds"] for r in records]
    tokens_in = sum(r["input_tokens"] for r in records)
    return {"requests": n, "cost": cost, "cost_per_request": cost / n if n else 0.0,
            "per_100k": cost / n * 100_000 if n else 0.0,
            "escalation_rate": sum(1 for r in records if r["escalated"]) / n if n else 0.0,
            "error_rate": sum(1 for r in records if r["error"]) / n if n else 0.0,
            "router_cost": sum(r.get("router_cost_usd", 0.0) for r in records),
            "p50": percentile(secs, 50), "p95": percentile(secs, 95),
            "input_tokens": tokens_in, "cached_tokens": sum(r["cached_tokens"] for r in records)}


def by_key(records: list[dict], key: str) -> dict[str, dict]:
    groups: dict = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def escalation_trend(records: list[dict], window: int = 50) -> tuple[float, float] | None:
    """(rate in the latest `window` requests, rate in the `window` before). A rising rate is an early warning:
    the cheap tier is failing more checks. None if there are not two full windows."""
    if len(records) < 2 * window:
        return None
    rate = lambda rows: sum(1 for r in rows if r["escalated"]) / len(rows)  # noqa: E731
    return rate(records[-window:]), rate(records[-2 * window:-window])


def format_report(records: list[dict]) -> str:
    s = summarize(records)
    lines = [f"{s['requests']} requests   ${s['cost_per_request']:.5f} per request   about ${s['per_100k']:,.0f} per 100,000",
             f"escalated {s['escalation_rate']:.0%}   errors {s['error_rate']:.0%}   "
             f"latency p50 {s['p50']:.1f}s p95 {s['p95']:.1f}s   router overhead ${s['router_cost']:.5f} in total",
             f"cached tokens: {s['cached_tokens']} of {s['input_tokens']} input tokens"]
    for key in ("tier", "rule"):
        lines += ["", f"{key:<24} {'requests':>8} {'$/request':>11} {'share of cost':>14} {'escalated':>10} {'p95 s':>7}"]
        for k, g in by_key(records, key).items():
            lines.append(f"{k:<24} {g['requests']:>8} {g['cost_per_request']:>11.5f} "
                         f"{(g['cost'] / s['cost'] if s['cost'] else 0):>14.0%} {g['escalation_rate']:>10.0%} {g['p95']:>7.1f}")
    return "\n".join(lines)
