"""Run the held-out set again and compare with the numbers saved by check_p6_report. Meant for a schedule.

    python -m scripts.drift_check             # exit code 0 = pass, 1 = fail
    python -m scripts.drift_check --offline   # needs a baseline saved by a live check_p6_report run

A failing run means one of: the model behind a tier changed, a price changed, or the questions changed.
Look at the failure line, then re-measure with Parts 2 and 4 before touching the router.
"""
import argparse
import json
import sys
from pathlib import Path

from evals import oracle, report
from evals.cases import load_cases
from evals.routed import run_suite_routed
from evals.runner import Config
from router import drift
from router.catalog import load_catalog
from router.final import build_final_gateway
from router.gateway import ListSink
from router.rules import load_tiers

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main(offline: bool = False, runs: int = 3, out_dir: Path = RESULTS, baseline: dict | None = None) -> int:
    catalog = load_catalog()
    base_path = out_dir / "final_baseline.json"
    if baseline is None:
        if not base_path.exists():
            print("No results/final_baseline.json. Run python -m scripts.check_p6_report first.")
            return 2
        baseline = json.loads(base_path.read_text(encoding="utf-8"))
    tiers, _ = load_tiers(out_dir / "break_even.json")
    model_for_case = classify = None
    if offline:
        print("OFFLINE DEMO: scripted models, so the comparison is not real.\n")
        oracle.use_empty_handbook()
        model_for_case = lambda spec, case: oracle.oracle_factory(Config(spec), case)  # noqa: E731
        classify = oracle.oracle_classify(load_cases())
    sink = ListSink()
    extra = {} if baseline.get("threshold") is None else {"threshold": baseline["threshold"]}
    gateway = build_final_gateway(baseline["strategy"], tiers=tiers, classify=classify, sink=sink, catalog=catalog, **extra)
    cases = load_cases("holdout")
    results = run_suite_routed(gateway, cases, runs=runs, model_for_case=model_for_case)
    s = report.stats(results)
    current = {"passed": s["passed"], "runs": s["runs"], "cost_per_conversation": s["cost_per_conversation"],
               "escalation_rate": sum(1 for r in results if r["escalated"]) / len(results)}
    print(f"baseline ({baseline['saved']}): {baseline['passed']} of {baseline['runs']} passed, "
          f"${baseline['cost_per_conversation']:.5f} per conversation, escalated {baseline['escalation_rate']:.0%}")
    print(f"now:                {s['passed']} of {s['runs']} passed, ${s['cost_per_conversation']:.5f} per conversation, "
          f"escalated {current['escalation_rate']:.0%}")
    failures = drift.gate(current, baseline, runs_per_case=runs)
    notes = drift.price_warnings(catalog, sink.records)
    for line in notes:
        print("warning:", line)
    if failures:
        print("\nFAIL")
        for f in failures:
            print(" -", f)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    a = parser.parse_args()
    sys.exit(main(offline=a.offline, runs=a.runs))
