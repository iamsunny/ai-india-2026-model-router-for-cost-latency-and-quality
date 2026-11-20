"""Turn run results into the tables the Part 2 article shows.

Three ideas matter here:

- Counts, not bare percentages. A held-out half has about 6 cases per task type, so one flipped
  case moves a rate by about 17 points. We print "16 of 18" and treat gaps under 2 cases as noise.
- Cost per correct answer = the money spent on ALL runs (right and wrong) divided by the number of
  right ones. A cheap model that is often wrong is not cheap.
- A tier "passes" a task type when its pass rate reaches the target (default 90 percent).
"""
from __future__ import annotations

from collections import defaultdict

NOISE_CASES = 2          # a gap smaller than this many cases is treated as noise
DEFAULT_TARGET = 0.90    # pass rate a tier needs to count as good enough


def _group(results: list[dict], *keys: str) -> dict:
    groups: dict = defaultdict(list)
    for r in results:
        groups[tuple(r[k] for k in keys)].append(r)
    return groups


def stats(rows: list[dict]) -> dict:
    """Totals for a list of runs."""
    runs = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    cost = sum(r["cost_usd"] for r in rows)
    return {"runs": runs, "passed": passed, "cost": cost,
            "cost_per_conversation": cost / runs if runs else 0.0,
            "cost_per_correct": cost / passed if passed else None,
            "seconds": sum(r["seconds"] for r in rows) / runs if runs else 0.0,
            "errors": sum(1 for r in rows if r.get("error"))}


def by_config(results: list[dict]) -> dict[str, dict]:
    return {k[0]: stats(v) for k, v in _group(results, "config").items()}


def by_config_and_type(results: list[dict]) -> dict[tuple[str, str], dict]:
    return {k: stats(v) for k, v in _group(results, "config", "task_type").items()}


def runs_per_case(results: list[dict]) -> int:
    """How many times each case was run on each config (3 in the article)."""
    counts = defaultdict(int)
    for r in results:
        counts[(r["config"], r["id"])] += 1
    return max(counts.values()) if counts else 1


def gap_is_noise(passed_a: int, passed_b: int, per_case_runs: int) -> bool:
    """True when two pass counts differ by fewer than NOISE_CASES cases."""
    return abs(passed_a - passed_b) < NOISE_CASES * per_case_runs


def pass_rate(s: dict) -> float:
    return s["passed"] / s["runs"] if s["runs"] else 0.0


def count_text(s: dict) -> str:
    return f"{s['passed']} of {s['runs']}"


def money(x: float | None) -> str:
    return "n/a" if x is None else f"${x:.5f}"


def break_even(results: list[dict], cheap_to_costly: list[str], target: float = DEFAULT_TARGET) -> dict[str, str | None]:
    """For each task type, the cheapest config (first in the list) whose pass rate reaches the target."""
    table = by_config_and_type(results)
    types = sorted({t for _, t in table})
    out: dict[str, str | None] = {}
    for t in types:
        out[t] = next((c for c in cheap_to_costly
                       if (c, t) in table and pass_rate(table[(c, t)]) >= target), None)
    return out


def best_effort(results: list[dict], effort_labels: list[str]) -> str | None:
    """The best single-model effort setting: the highest pass count, and among gaps that are
    noise, the lowest cost per correct answer. A router has to beat this one, not just the default."""
    per = by_config(results)
    labels = [c for c in effort_labels if c in per]
    if not labels:
        return None
    k = runs_per_case(results)
    top = max(labels, key=lambda c: per[c]["passed"])
    close = [c for c in labels if gap_is_noise(per[c]["passed"], per[top]["passed"], k)]
    return min(close, key=lambda c: per[c]["cost_per_correct"] if per[c]["cost_per_correct"] is not None else 1e9)


def format_config_table(results: list[dict], labels: list[str]) -> str:
    per = by_config(results)
    lines = [f"{'config':<30} {'passed':>11} {'$/conversation':>15} {'$/correct':>11} {'seconds':>8}"]
    for c in labels:
        if c in per:
            s = per[c]
            lines.append(f"{c:<30} {count_text(s):>11} {money(s['cost_per_conversation']):>15} "
                         f"{money(s['cost_per_correct']):>11} {s['seconds']:>8.1f}")
    return "\n".join(lines)


def format_type_table(results: list[dict], labels: list[str], types: list[str]) -> str:
    table = by_config_and_type(results)
    head = f"{'task type':<11}" + "".join(f"{c[:22]:>26}" for c in labels)
    lines = [head]
    for t in types:
        cells = []
        for c in labels:
            s = table.get((c, t))
            cells.append(f"{count_text(s)}  {money(s['cost_per_correct'])}" if s else "-")
        lines.append(f"{t:<11}" + "".join(f"{x:>26}" for x in cells))
    return "\n".join(lines)


TIER_RANK = {"small": 0, "mid": 1, "large": 2}


def misroutes(router_results: list[dict], baseline_results: list[dict], mid_label: str, small_label: str) -> list[dict]:
    """Cases where the router's choice looks wrong, judged against the Part 2 runs of the same cases.

    under-served  the router failed most runs, but the mid model passed most. It saved money and lost the answer.
    over-served   the router sent it above the small tier, but the small model passed most runs.
                  The answer is fine and the saving was left on the table.
    """
    def majority(rows):
        return bool(rows) and sum(1 for r in rows if r["passed"]) * 2 >= len(rows)

    base = _group(baseline_results, "config", "id")
    out = []
    for (case_id,), rows in _group(router_results, "id").items():
        router_ok = majority(rows)
        mid_ok = majority(base.get((mid_label, case_id), []))
        small_ok = majority(base.get((small_label, case_id), []))
        tier = max((r["tier"] for r in rows), key=lambda t: TIER_RANK[t])
        rule = rows[0]["rule"]
        passed = sum(1 for r in rows if r["passed"])
        if not router_ok and mid_ok:
            out.append({"id": case_id, "kind": "under-served", "tier": tier, "rule": rule,
                        "passed": passed, "runs": len(rows)})
        elif tier != "small" and small_ok and router_ok:
            out.append({"id": case_id, "kind": "over-served", "tier": tier, "rule": rule,
                        "passed": passed, "runs": len(rows)})
    return out
