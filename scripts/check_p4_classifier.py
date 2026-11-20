"""Part 4 check: can a small model read the question better than the rules, and does it pay for itself?

    python -m scripts.check_p4_classifier                       # gpt-5.4-nano as the router brain
    python -m scripts.check_p4_classifier --brains openai:gpt-5.4-nano anthropic:claude-haiku-5-5
    python -m scripts.check_p4_classifier --quick               # a few questions, a few cents
    python -m scripts.check_p4_classifier --offline             # scripted classifier, no key

Needs results/p2_runs.jsonl from Part 2 for the threshold sweep. The sweep is an estimate built
from those runs plus the measured classifier cost. The threshold it recommends is then run for
real through the gateway, and that run goes on the scoreboard.
The Claude brain is optional. It needs ANTHROPIC_API_KEY and pip install -r requirements-claude.txt.
"""
import argparse
import json
from pathlib import Path

from evals import oracle, report, scoreboard
from evals.cases import TASK_TYPES, load_cases, load_recurring
from evals.replay import replay
from evals.routed import run_suite_routed
from evals.runner import Config, run_suite
from router.catalog import TIERS, load_catalog
from router.classifier import ClassifierRouter, LLMClassifier, tier_for
from router.cost import usage_cost
from router.gateway import Gateway
from router.rules import RISK_RULES, check_risk, load_tiers

RESULTS = Path(__file__).resolve().parent.parent / "results"
THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
DEFAULT_BRAIN = "openai:gpt-5.4-nano"


def short(spec: str) -> str:
    return spec.split(":", 1)[-1]


def classify_all(classify, cases: list[dict], catalog) -> dict[str, dict]:
    """Classify every case once. Returns id -> {"c": Classification, "cost_usd", "seconds"}."""
    out = {}
    for case in cases:
        c = classify(case["question"], [])
        out[case.get("key", case["id"])] = {"c": c, "cost_usd": usage_cost(c.usage, catalog)["cost_usd"] if c.usage else 0.0,
                           "seconds": c.seconds, "task_type": case["task_type"], "split": case.get("split")}
    return out


def accuracy(labelled: dict[str, dict], split: str | None = None) -> tuple[int, int]:
    rows = [v for v in labelled.values() if split is None or v["split"] == split]
    return sum(1 for v in rows if v["c"].valid and v["c"].task_type == v["task_type"]), len(rows)


def decide(case: dict, labelled: dict, tiers: dict, threshold: float) -> tuple[str, float, float]:
    """The tier the classifier router would pick, and its overhead (cost, seconds). Risk rules skip the classifier."""
    if check_risk(case["question"], [], RISK_RULES):
        return "large", 0.0, 0.0
    v = labelled[case["id"]]
    return tier_for(v["c"], tiers, threshold)[0], v["cost_usd"], v["seconds"]


def sweep(cases, labelled, tiers, baseline, tier_labels, k):
    rows = []
    for th in THRESHOLDS:
        chosen, overhead = {}, {}
        for case in cases:
            tier, cost, secs = decide(case, labelled, tiers, th)
            chosen[case["id"]] = tier
            overhead[case["id"]] = {"cost_usd": cost, "seconds": secs}
        s = replay(chosen, baseline, tier_labels, overhead)
        s["threshold"] = th
        s["share"] = {t: sum(1 for x in chosen.values() if x == t) for t in TIERS}
        rows.append(s)
    top = max(r["passed"] for r in rows)
    ok = [r for r in rows if report.gap_is_noise(r["passed"], top, k)]
    # Cheapest per correct answer; on a tie, the higher (safer) threshold.
    best = min(ok, key=lambda r: (round(r["cost_per_correct"], 8) if r["cost_per_correct"] is not None else 1e9,
                                  -r["threshold"]))
    return rows, best["threshold"]


def main(offline: bool = False, quick: bool = False, brains: list[str] | None = None, threshold: float | None = None,
         runs: int = 3, out_dir: Path = RESULTS) -> dict:
    catalog = load_catalog()
    tiers, source = load_tiers(out_dir / "break_even.json")
    brains = brains or [DEFAULT_BRAIN]
    recurring = [{**q, "key": f"recurring:{q['id']}"} for q in load_recurring()]   # ids can clash with eval cases
    all_cases = load_cases() + recurring
    holdout = load_cases("holdout")
    if quick:
        holdout = [next(c for c in holdout if c["task_type"] == t) for t in TASK_TYPES]
        runs = 1
    scored_cases = holdout
    classify_cases = holdout + recurring if quick else all_cases
    tier_cfg = {t: Config(catalog.tier_spec(t)).label for t in TIERS}
    if offline:
        print("OFFLINE DEMO: the classifier and the model are scripted and always right. Numbers are not real.\n")
        oracle.use_empty_handbook()
    print(f"Tiers per task type come from {source}.\n")

    # Baseline for the estimate: Part 2 runs (offline: a scripted stand-in).
    path = out_dir / "p2_runs.jsonl"
    if offline:
        baseline = run_suite([Config(catalog.tier_spec(t)) for t in TIERS], scored_cases, runs=1,
                             factory=oracle.oracle_factory, catalog=catalog)
    elif path.exists():
        baseline = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    else:
        baseline = []
        print("No results/p2_runs.jsonl found. Run Part 2 first. The sweep and break-even are skipped.\n")
    k = report.runs_per_case(baseline) if baseline else 1

    summary = {}
    for brain in brains:
        print(f"== Router brain: {brain} ==")
        classify = (oracle.oracle_classify(all_cases) if offline else LLMClassifier(brain))
        labelled = classify_all(classify, classify_cases, catalog)
        invalid = sum(1 for v in labelled.values() if not v["c"].valid)
        for split in ("tune", "holdout"):
            ok, n = accuracy(labelled, split)
            if n:
                print(f"  named the right task type for {ok} of {n} {split} questions")
        print(f"  invalid replies: {invalid} of {len(labelled)} (these fall back to the mid tier)")
        reached = [labelled[c["id"]] for c in classify_cases
                   if c.get("split") and not check_risk(c["question"], [], RISK_RULES)]
        avg_cost = sum(v["cost_usd"] for v in reached) / max(1, len(reached))
        avg_secs = sum(v["seconds"] for v in reached) / max(1, len(reached))
        print(f"  overhead per request that reaches the classifier: ${avg_cost:.6f} and {avg_secs:.2f}s "
              f"(risk questions skip it)")
        print("  recurring questions:")
        for q in load_recurring():
            v = labelled[f"recurring:{q['id']}"]
            tier, rule = tier_for(v["c"], tiers, threshold if threshold is not None else 0.7)
            print(f"    {q['id']:<15} -> {tier:<6} ({rule}, says {v['c'].task_type}, confidence {v['c'].confidence})")
        entry = {"labelled": labelled, "avg_cost": avg_cost, "avg_seconds": avg_secs}
        if baseline:
            per = report.by_config(baseline)
            mid, small = per.get(tier_cfg["mid"]), per.get(tier_cfg["small"])
            if mid and small and mid["cost_per_conversation"] > small["cost_per_conversation"]:
                gain = mid["cost_per_conversation"] - small["cost_per_conversation"]
                print(f"  break-even: moving a request from mid to small saves about ${gain:.6f}, so the classifier "
                      f"must move {avg_cost / gain:.0%} of requests down to cover its own cost")
            rows, best = sweep(scored_cases, labelled, tiers, baseline, tier_cfg, k)
            print("  threshold sweep, estimated from your Part 2 runs plus this classifier's cost:")
            print(f"    {'threshold':>9} {'small/mid/large':>16} {'passed':>11} {'$/conversation':>15} {'$/correct':>11}")
            for r in rows:
                sh = r["share"]
                print(f"    {r['threshold']:>9} {sh['small']:>5}/{sh['mid']}/{sh['large']:<7} {report.count_text(r):>11} "
                      f"{report.money(r['cost_per_conversation']):>15} {report.money(r['cost_per_correct']):>11}")
            print(f"  recommended threshold: {best} (cheapest per correct answer among settings within noise of the best)")
            entry["best"] = best
        summary[brain] = entry
        print()

    first = brains[0]
    chosen = threshold if threshold is not None else summary[first].get("best", 0.7)
    classify = oracle.oracle_classify(all_cases) if offline else LLMClassifier(first)
    router = ClassifierRouter(classify, tiers, threshold=chosen, name=f"classifier router ({short(first)}, {chosen})")
    gateway = Gateway(router, catalog=catalog)
    print(f"Running {len(scored_cases)} held-out cases x {runs} run(s) for real through the gateway, threshold {chosen}...",
          flush=True)
    results = run_suite_routed(gateway, scored_cases, runs=runs,
                               model_for_case=(lambda spec, case: oracle.oracle_factory(Config(spec), case))
                               if offline else None)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ("p4_runs_offline.jsonl" if offline else "p4_runs.jsonl")).write_text(
        "\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")
    s = report.stats(results)
    over = sum(r["router_cost_usd"] for r in results) / max(1, len(results))
    print(f"\n== The classifier router ==\npassed {report.count_text(s)}   ${s['cost_per_conversation']:.5f} per conversation "
          f"(of which the router itself ${over:.6f})   {s['seconds']:.1f}s")
    if not offline and not quick:
        scoreboard.record(router.name, 4, results, path=out_dir / "scoreboard.json")
        print("\n== Scoreboard so far (held-out split) ==")
        print(scoreboard.format_board(out_dir / "scoreboard.json"))
    return {"results": results, "summary": summary, "threshold": chosen}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--brains", nargs="*")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--runs", type=int, default=3)
    a = parser.parse_args()
    main(offline=a.offline, quick=a.quick, brains=a.brains, threshold=a.threshold, runs=a.runs)
