"""Part 6 check: replay conversations through the final setup and read the before and after table.

    python -m scripts.check_p6_report                # real run: needs the earlier parts' results in results/
    python -m scripts.check_p6_report --quick        # 4 conversations, 5 held-out cases, 1 run
    python -m scripts.check_p6_report --strategy classifier
    python -m scripts.check_p6_report --offline      # scripted models, no key, shows the format only

"Before" is everything on the mid model, the Part 11 baseline. "After" is the final setup: router, cascade,
budget, all behind the gateway, every request logged to results/routes.jsonl.
"""
import argparse
import json
from datetime import date
from pathlib import Path

from evals import oracle, report, scoreboard
from evals.cases import load_cases, load_conversations, load_recurring
from evals.routed import run_conversation, run_routed, run_suite_routed
from evals.runner import Config
from router import drift, telemetry
from router.catalog import load_catalog
from router.final import build_final_gateway
from router.gateway import Gateway, JsonlSink, ListSink, single_call
from router.rules import load_tiers
from router.types import Route

RESULTS = Path(__file__).resolve().parent.parent / "results"


def before_after(rows: list[tuple[str, dict, dict]]) -> str:
    lines = [f"{'':<34} {'before: all mid':>16} {'after: router':>14} {'change':>8}  route"]
    for name, before, after in rows:
        change = (after["cost_usd"] - before["cost_usd"]) / before["cost_usd"] if before["cost_usd"] else 0.0
        lines.append(f"{name:<34} ${before['cost_usd']:>14.6f} ${after['cost_usd']:>12.6f} {change:>+8.0%}  "
                     f"{after['tier']}{' (escalated)' if after['escalated'] else ''}")
    return "\n".join(lines)


def main(offline: bool = False, quick: bool = False, strategy: str = "rules", second_provider: bool = False,
         runs: int = 3, out_dir: Path = RESULTS, threshold: float | None = None) -> dict:
    catalog = load_catalog()
    tiers, source = load_tiers(out_dir / "break_even.json")
    cases_by_id = {c["id"]: c for c in load_cases()}
    conversations = load_conversations()[:4] if quick else load_conversations()
    holdout = load_cases("holdout")
    holdout = holdout[:5] if quick else holdout
    runs = 1 if quick else runs
    out_dir.mkdir(parents=True, exist_ok=True)
    routes_path = out_dir / ("routes_offline.jsonl" if offline else "routes.jsonl")
    routes_path.unlink(missing_ok=True)

    all_cases = load_cases() + [{**q, "key": q["id"]} for q in load_recurring()]
    model_for_case = None
    classify = None
    if offline:
        print("OFFLINE DEMO: the models are scripted and always right. Tokens, cost and time are not real.\n")
        oracle.use_empty_handbook()
        model_for_case = lambda spec, case: oracle.oracle_factory(Config(spec), case)  # noqa: E731
        classify = oracle.oracle_classify(all_cases)
    print(f"Tiers per task type come from {source}. Strategy: {strategy} + cascade.\n")

    sink = JsonlSink(routes_path)
    extra = {} if threshold is None else {"threshold": threshold}
    final = build_final_gateway(strategy, second_provider, tiers, classify=classify, sink=sink, catalog=catalog, **extra)
    baseline = Gateway(lambda m, h: Route("mid", "all-mid"), executor=single_call, catalog=catalog, sink=ListSink(),
                       name="all on mid")

    print(f"Replaying {len(conversations)} conversations and the recurring questions through both setups...", flush=True)
    before_turns, after_turns = [], []
    for conv in conversations:
        before_turns += run_conversation(conv, cases_by_id, baseline, model_for_case)
        after_turns += run_conversation(conv, cases_by_id, final, model_for_case)
    recurring = load_recurring()
    before_rec = [run_routed(q, baseline, 0, model_for_case) for q in recurring]
    after_rec = [run_routed(q, final, 0, model_for_case) for q in recurring]

    records = telemetry.load_records(routes_path)
    print("\n== What the telemetry says about the replay ==")
    print(telemetry.format_report(records))
    s = telemetry.summarize(records)
    longest = max((r["input_tokens"] for r in records), default=0)
    if s["cached_tokens"] == 0:
        print(f"\nCaching: no cached tokens in this replay. The longest request had {longest} input tokens, "
              "which may be under the minimum size a provider will cache. Caches are also per model, so routing splits them.")
    else:
        print(f"\nCaching: {s['cached_tokens']} of {s['input_tokens']} input tokens were served from a cache.")

    print("\n== Before and after, same questions ==")
    rows = [(f"{q['id']} ({q['task_type']})", b, a) for q, b, a in zip(recurring, before_rec, after_rec)]
    tot = lambda turns: {"cost_usd": sum(t["cost_usd"] for t in turns), "tier": "mixed",  # noqa: E731
                         "escalated": False}
    rows.append((f"all {len(after_turns)} replayed turns", tot(before_turns), tot(after_turns)))
    print(before_after(rows))
    n = len(after_turns)
    if n:
        b, a = tot(before_turns)["cost_usd"] / n, tot(after_turns)["cost_usd"] / n
        print(f"per turn: ${b:.6f} before, ${a:.6f} after. Per 100,000 turns: ${b * 1e5:,.0f} before, ${a * 1e5:,.0f} after.")
    passed = lambda turns: sum(1 for t in turns if t["passed"])  # noqa: E731
    print(f"quality on the replayed turns (cases scored as in Part 2): before {passed(before_turns)} of {n}, "
          f"after {passed(after_turns)} of {n}. This is not the held-out measure. See the scoreboard below.")

    print(f"\nRunning {len(holdout)} held-out cases x {runs} run(s) for the scoreboard...", flush=True)
    final.sink = ListSink()
    results = run_suite_routed(final, holdout, runs=runs, model_for_case=model_for_case)
    st = report.stats(results)
    esc = sum(1 for r in results if r["escalated"]) / len(results)
    baseline_row = {"passed": st["passed"], "runs": st["runs"], "cost_per_conversation": st["cost_per_conversation"],
                    "escalation_rate": esc, "checked": catalog.checked, "strategy": strategy,
                    "threshold": threshold, "saved": date.today().isoformat()}
    if not offline and not quick:
        scoreboard.record("final setup", 6, results, path=out_dir / "scoreboard.json")
        (out_dir / "final_baseline.json").write_text(json.dumps(baseline_row, indent=2), encoding="utf-8")
        print("\n== The scoreboard (held-out split) ==")
        print(scoreboard.format_board(out_dir / "scoreboard.json"))
        print("\nSaved results/final_baseline.json. Run python -m scripts.drift_check on a schedule to compare against it.")
    else:
        print(f"final setup: passed {report.count_text(st)}, ${st['cost_per_conversation']:.5f} per conversation, "
              f"escalated {esc:.0%} (not saved: offline or quick run)")

    print("\n== Housekeeping ==")
    warnings = drift.price_warnings(catalog, records)
    print("\n".join(warnings) if warnings else f"prices were checked {catalog.checked}: fine")
    trend = telemetry.escalation_trend(records)
    print("escalation trend: " + (f"latest 50 requests {trend[0]:.0%}, the 50 before {trend[1]:.0%}" if trend
                                  else "not enough requests yet for two windows of 50"))
    return {"records": records, "before": before_turns, "after": after_turns, "baseline_row": baseline_row}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--strategy", choices=("rules", "classifier"), default="rules")
    parser.add_argument("--second-provider", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=None,
                        help="classifier confidence threshold (default 0.7; Part 4 found 0.0 cheapest)")
    a = parser.parse_args()
    main(offline=a.offline, quick=a.quick, strategy=a.strategy, second_provider=a.second_provider, runs=a.runs,
         threshold=a.threshold)
