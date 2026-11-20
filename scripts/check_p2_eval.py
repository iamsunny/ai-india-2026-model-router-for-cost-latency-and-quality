"""Part 2 check: is the small model good enough? Run the eval set on every tier and read it per task type.

    python -m scripts.check_p2_eval                  # real run: three OpenAI tiers + effort sweep, held-out split
    python -m scripts.check_p2_eval --quick          # 1 case per task type, 1 run: a few cents, checks your setup
    python -m scripts.check_p2_eval --no-sweep       # skip the effort sweep on the mid model
    python -m scripts.check_p2_eval --offline        # scripted model, no key, shows the output format only
    python -m scripts.check_p2_eval --sweep-only     # re-run only effort low, medium, high and merge them into results/p2_runs.jsonl

A full real run is 32 held-out cases x 3 runs x 7 configs = 672 runs. They run one after another
because the fake HRMS is shared, so allow 30 to 60 minutes. Raw runs go to results/p2_runs.jsonl.
Build the handbook index first (python -m helpdesk.ingest).
"""
import argparse
import json
from pathlib import Path

from evals import oracle, report, scoreboard
from evals.cases import TASK_TYPES, load_cases
from evals.runner import Config, default_factory, run_suite
from router.catalog import TIERS, load_catalog

EFFORTS = ("none", "low", "medium", "high")
RESULTS = Path(__file__).resolve().parent.parent / "results"


def pick_cases(split: str, quick: bool) -> list[dict]:
    cases = load_cases(split)
    if not quick:
        return cases
    return [next(c for c in cases if c["task_type"] == t) for t in TASK_TYPES]


def main(offline: bool = False, quick: bool = False, sweep: bool = True, runs: int = 3,
         split: str = "holdout", target: float = report.DEFAULT_TARGET, out_dir: Path = RESULTS,
         sweep_only: bool = False) -> dict:
    catalog = load_catalog()
    tiers = [Config(catalog.tier_spec(t)) for t in TIERS]
    mid = catalog.tier_spec("mid")
    efforts = [Config(mid, e) for e in EFFORTS] if sweep else []
    cases = pick_cases(split, quick)
    runs = 1 if quick else runs
    factory = oracle.oracle_factory if offline else default_factory
    if offline:
        oracle.use_empty_handbook()
        print("OFFLINE DEMO: the model is scripted and always right, so passes, tokens, cost and time are not real.\n")
    if split != "holdout":
        print("Warning: this is the tuning split. The scoreboard only takes held-out runs, so it will not be updated.\n")
    configs = tiers + efforts
    kept: list[dict] = []
    if sweep_only:
        # Keep the tier runs and the effort-none runs. Re-run effort low, medium and high.
        redo = [c for c in efforts if c.effort != "none"]
        old = [json.loads(line) for line in (out_dir / "p2_runs.jsonl").read_text(encoding="utf-8").splitlines() if line]
        redo_labels = {c.label for c in redo}
        kept = [r for r in old if r["config"] not in redo_labels]
        run_configs = redo
    else:
        run_configs = configs
    print(f"Prices from models.yaml, checked {catalog.checked}")
    print(f"{len(cases)} cases x {runs} run(s) x {len(run_configs)} configs = {len(cases) * runs * len(run_configs)} runs\n")
    results = kept + run_suite(run_configs, cases, runs=runs, factory=factory, catalog=catalog,
                               progress=lambda m: print("running", m, flush=True))

    out_dir.mkdir(parents=True, exist_ok=True)
    name = "p2_runs_offline.jsonl" if offline else "p2_runs.jsonl"
    (out_dir / name).write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")

    tier_labels = [c.label for c in tiers]
    effort_labels = [c.label for c in efforts]
    types = [t for t in TASK_TYPES if any(r["task_type"] == t for r in results)]
    k = report.runs_per_case(results)

    print("\n== Each tier, all task types ==")
    print(report.format_config_table(results, tier_labels))
    print("\n== Each tier, by task type: passed, and cost per correct answer ==")
    print(report.format_type_table(results, tier_labels, types))
    errors = sum(1 for r in results if r.get("error"))
    if errors:
        print(f"\n{errors} runs ended in an API error and count as failed. See results/{name} for the messages.")
    best = None
    if sweep:
        print("\n== Effort sweep on the mid model ==")
        print(report.format_config_table(results, effort_labels))
        best = report.best_effort(results, effort_labels)
        print(f"\nBest single-model effort setting: {best}")
        print("A router has to beat this setting, not only the default.")
    table = report.break_even(results, tier_labels, target)
    print(f"\n== Break-even: the cheapest tier that passes at {target:.0%} or better ==")
    for t in types:
        print(f"  {t:<10} {table[t] or 'no tier reaches the target'}")
    print(f"\nNoise rule: each case ran {k} time(s), so a gap under {report.NOISE_CASES * k} runs "
          f"({report.NOISE_CASES} cases) is noise.")

    if split == "holdout" and not offline:
        tier_of = dict(zip(tier_labels, TIERS))
        (out_dir / "break_even.json").write_text(
            json.dumps({t: tier_of.get(label) for t, label in table.items()}, indent=2), encoding="utf-8")
        print("\nSaved results/break_even.json. The Part 3 rules router reads its tiers from it.")
        by = lambda label: [r for r in results if r["config"] == label]  # noqa: E731
        scoreboard.record("all on large", 2, by(tier_labels[2]))
        scoreboard.record("all on mid (baseline)", 2, by(tier_labels[1]))
        if best:
            board = scoreboard.load()
            for key in [k_ for k_ in board if k_.startswith("best effort:")]:
                del board[key]   # drop a stale row, for example from before a sweep re-run
            scoreboard.PATH.write_text(json.dumps(board, indent=2), encoding="utf-8")
            scoreboard.record(f"best effort: {best}", 2, by(best))
        print("\n== Scoreboard so far (held-out split) ==")
        print(scoreboard.format_board())
    return {"results": results, "best_effort": best, "break_even": table}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="scripted model, no API key")
    parser.add_argument("--quick", action="store_true", help="one case per task type, one run")
    parser.add_argument("--no-sweep", action="store_true", help="skip the effort sweep")
    parser.add_argument("--sweep-only", action="store_true", help="re-run only effort low, medium and high, keep the other runs")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--split", default="holdout", choices=("holdout", "tune"))
    parser.add_argument("--target", type=float, default=report.DEFAULT_TARGET)
    a = parser.parse_args()
    main(offline=a.offline, quick=a.quick, sweep=not a.no_sweep, runs=a.runs, split=a.split, target=a.target,
         sweep_only=a.sweep_only)
