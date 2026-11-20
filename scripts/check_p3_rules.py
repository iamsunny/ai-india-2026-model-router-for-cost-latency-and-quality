"""Part 3 check: how does a few-rules router do on the held-out questions?

    python -m scripts.check_p3_rules                # real run on the held-out split (run Part 2 first)
    python -m scripts.check_p3_rules --quick        # 1 case per task type, 1 run
    python -m scripts.check_p3_rules --offline      # scripted model, no key, shows the format only

The first part of the output needs no model at all: deciding a route is free with rules.
It reads results/break_even.json and results/p2_runs.jsonl from Part 2, if they exist.
"""
import argparse
import json
from pathlib import Path

from evals import oracle, report, scoreboard
from evals.cases import TASK_TYPES, load_cases, load_recurring
from evals.routed import run_suite_routed
from evals.runner import Config
from router.catalog import load_catalog
from router.gateway import Gateway
from router.rules import RulesRouter, load_tiers

RESULTS = Path(__file__).resolve().parent.parent / "results"


def route_share(router: RulesRouter, cases: list[dict]) -> dict:
    """No model needed: which tier does each question get, and does the rule's task type match the label?"""
    share = {t: 0 for t in ("small", "mid", "large")}
    agree = 0
    grid = {t: {tier: 0 for tier in share} for t in TASK_TYPES}
    for c in cases:
        route = router(c["question"])
        _, guess = router.detect(c["question"])
        share[route.tier] += 1
        grid[c["task_type"]][route.tier] += 1
        agree += guess == c["task_type"]
    return {"share": share, "agree": agree, "total": len(cases), "grid": grid}


def pick_quick(cases: list[dict]) -> list[dict]:
    return [next(c for c in cases if c["task_type"] == t) for t in TASK_TYPES]


def main(offline: bool = False, quick: bool = False, runs: int = 3, split: str = "holdout",
         out_dir: Path = RESULTS, board_path: Path | None = None) -> dict:
    catalog = load_catalog()
    tiers, source = load_tiers(out_dir / "break_even.json")
    router = RulesRouter(tiers)
    board_path = board_path or out_dir / "scoreboard.json"
    if offline:
        print("OFFLINE DEMO: the model is scripted and always right, so passes, tokens, cost and time are not real.\n")
    print(f"Tiers per task type come from {source}:")
    for t, tier in tiers.items():
        print(f"  {t:<10} {tier}")

    print("\n== How the rules read the questions (no model, no cost) ==")
    for name in ("tune", "holdout"):
        r = route_share(router, load_cases(name))
        s = r["share"]
        print(f"{name:<8} rules named the right task type for {r['agree']} of {r['total']} questions. "
              f"Routed small {s['small']}, mid {s['mid']}, large {s['large']}.")
    print("The rules were written from the tune split. The gap between those two lines is the lesson of this part.")
    print("\nRoute for each task type on the held-out split (rows: labelled type, columns: tier chosen)")
    grid = route_share(router, load_cases("holdout"))["grid"]
    print(f"{'task type':<11}{'small':>7}{'mid':>6}{'large':>7}")
    for t in TASK_TYPES:
        print(f"{t:<11}{grid[t]['small']:>7}{grid[t]['mid']:>6}{grid[t]['large']:>7}")
    print("\nThe recurring questions")
    for q in load_recurring():
        route = router(q["question"])
        print(f"  {q['id']:<15} -> {route.tier:<6} ({route.rule})")

    cases = load_cases(split)
    cases = pick_quick(cases) if quick else cases
    runs = 1 if quick else runs
    if offline:
        oracle.use_empty_handbook()
    gateway = Gateway(router, catalog=catalog, name="rules router")
    print(f"\nRunning {len(cases)} cases x {runs} run(s) through the router...", flush=True)
    results = run_suite_routed(gateway, cases, runs=runs,
                               model_for_case=(lambda spec, case: oracle.oracle_factory(Config(spec), case))
                               if offline else None)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "p3_runs_offline.jsonl" if offline else "p3_runs.jsonl"
    (out_dir / name).write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")

    s = report.stats(results)
    print("\n== The rules router ==")
    print(f"passed {report.count_text(s)}   ${s['cost_per_conversation']:.5f} per conversation   "
          f"${s['cost_per_correct']:.5f} per correct answer   {s['seconds']:.1f}s" if s["passed"] else
          f"passed {report.count_text(s)}   ${s['cost_per_conversation']:.5f} per conversation   no correct answers")
    shares = {tier: sum(1 for r in results if r["tier"] == tier) for tier in ("small", "mid", "large")}
    print("runs routed: " + ", ".join(f"{k} {v}" for k, v in shares.items()))
    print("by task type: " + ", ".join(
        f"{t} {report.count_text(v)}" for (c, t), v in sorted(report.by_config_and_type(results).items())))

    baseline_path = out_dir / "p2_runs.jsonl"
    found = []
    if baseline_path.exists() and split == "holdout" and not offline:
        baseline = [json.loads(line) for line in baseline_path.read_text(encoding="utf-8").splitlines() if line]
        small = Config(catalog.tier_spec("small")).label
        mid = Config(catalog.tier_spec("mid")).label
        found = report.misroutes(results, baseline, mid, small)
        print("\n== Misroutes, judged against your Part 2 runs ==")
        if not found:
            print("none")
        for m in found:
            q = next(c["question"] for c in cases if c["id"] == m["id"])
            print(f"  {m['kind']:<12} {m['id']:<18} {m['tier']:<6} {m['rule']:<18} {m['passed']} of {m['runs']}  {q[:70]}")
    elif not offline:
        print("\nNo Part 2 runs found, so no misroute list. Run python -m scripts.check_p2_eval first.")

    if split == "holdout" and not offline:
        scoreboard.record("rules router", 3, results, path=board_path)
        print("\n== Scoreboard so far (held-out split) ==")
        print(scoreboard.format_board(board_path))
    return {"results": results, "misroutes": found, "stats": s}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--split", default="holdout", choices=("holdout", "tune"))
    a = parser.parse_args()
    main(offline=a.offline, quick=a.quick, runs=a.runs, split=a.split)
