import pytest

from evals import report, scoreboard
from evals.cases import load_cases
from evals.runner import Config, run_case, run_suite
from helpdesk.scripted import ScriptedModel, ai
from scripts import check_p2_eval


def r(config, task_type, passed, cost=0.001, split="holdout", i="c", run=0, seconds=1.0, error=None):
    return {"config": config, "task_type": task_type, "passed": passed, "cost_usd": cost, "seconds": seconds,
            "split": split, "id": i, "run": run, "error": error}


def test_cost_per_correct_counts_the_money_spent_on_wrong_answers():
    rows = [r("m", "lookup", True, 0.001), r("m", "lookup", False, 0.001), r("m", "lookup", False, 0.001)]
    s = report.stats(rows)
    assert s["passed"] == 1 and s["runs"] == 3
    assert s["cost_per_correct"] == pytest.approx(0.003)      # all three runs paid for one right answer
    assert report.stats([r("m", "lookup", False)])["cost_per_correct"] is None
    assert report.count_text(s) == "1 of 3"


def test_gap_under_two_cases_is_noise():
    assert report.gap_is_noise(16, 17, per_case_runs=3)       # 1 run apart
    assert report.gap_is_noise(12, 17, per_case_runs=3)       # 5 runs apart, still under 2 cases (6 runs)
    assert not report.gap_is_noise(10, 17, per_case_runs=3)


def test_break_even_picks_the_cheapest_tier_that_reaches_the_target():
    rows = ([r("small", "lookup", True, i=f"a{n}") for n in range(9)] + [r("small", "lookup", False, i="a9")]
            + [r("small", "reasoning", n < 5, i=f"b{n}") for n in range(10)]
            + [r("mid", "reasoning", True, i=f"b{n}") for n in range(10)]
            + [r("mid", "lookup", True, i=f"a{n}") for n in range(10)])
    table = report.break_even(rows, ["small", "mid"], target=0.9)
    assert table == {"lookup": "small", "reasoning": "mid"}
    assert report.break_even(rows[:10], ["small"], target=0.99) == {"lookup": None}


def test_best_effort_prefers_the_cheaper_setting_when_the_gap_is_noise():
    rows = []
    for n in range(18):
        rows.append(r("low", "policy", n < 16, cost=0.001, i=f"q{n // 3}", run=n % 3))
        rows.append(r("high", "policy", n < 17, cost=0.004, i=f"q{n // 3}", run=n % 3))
    assert report.best_effort(rows, ["low", "high"]) == "low"   # 16 vs 17 of 18 is noise, low is cheaper
    far = [x for x in rows if x["config"] == "high"] + [r("low", "policy", False, i=f"q{n}") for n in range(18)]
    assert report.best_effort(far, ["low", "high"]) == "high"


def test_scoreboard_takes_held_out_runs_only(tmp_path):
    path = tmp_path / "board.json"
    row = scoreboard.record("all on mid", 2, [r("m", "lookup", True)], path=path)
    assert row["passed"] == 1
    assert "all on mid" in scoreboard.format_board(path)
    with pytest.raises(ValueError, match="held-out"):
        scoreboard.record("peeked", 2, [r("m", "lookup", True, split="tune")], path=path)


def test_a_model_error_is_a_failed_run_not_a_crash():
    case = load_cases("holdout")[0]

    def broken(config, case):
        raise TimeoutError("the model timed out")

    out = run_case(case, Config("openai:gpt-5.4-mini"), broken)
    assert out["passed"] is False and "TimeoutError" in out["error"] and out["cost_usd"] == 0


def test_a_wrong_answer_fails_and_a_right_one_passes():
    case = next(c for c in load_cases("holdout") if c["id"] == "leave-sick")
    call = ai("", ("get_leave_balance", {"employee_id": "E1042"}), model_name="gpt-5.4-nano", tokens=(400, 10))

    def factory(wrong):
        text = "You have 9 sick days." if wrong else "You have 6 sick days."
        return lambda config, c: ScriptedModel(script=[call, ai(text, model_name="gpt-5.4-nano", tokens=(300, 12))], seen=[])

    bad = run_case(case, Config("openai:gpt-5.4-nano"), factory(True))
    good = run_case(case, Config("openai:gpt-5.4-nano"), factory(False))
    assert (bad["passed"], good["passed"]) == (False, True)
    assert good["cost_usd"] == pytest.approx((700 * 0.20 + 22 * 1.25) / 1e6)


def test_run_suite_runs_every_case_the_requested_number_of_times():
    from evals import oracle
    oracle.use_empty_handbook()
    cases = load_cases("holdout")[:3]
    out = run_suite([Config("openai:gpt-5.4-nano")], cases, runs=2, factory=oracle.oracle_factory)
    assert len(out) == 6 and all(x["passed"] for x in out)


def test_oracle_passes_every_case_so_the_wiring_is_sound():
    from evals import oracle
    oracle.use_empty_handbook()
    out = run_suite([Config("openai:gpt-5.4-mini")], load_cases(), runs=1, factory=oracle.oracle_factory)
    failed = [x["id"] for x in out if not x["passed"]]
    assert failed == [], failed


def test_offline_check_prints_the_tables_and_writes_runs(tmp_path, capsys):
    out = check_p2_eval.main(offline=True, quick=True, out_dir=tmp_path)
    text = capsys.readouterr().out
    for part in ("OFFLINE DEMO", "Each tier, by task type", "Effort sweep", "Best single-model effort setting",
                 "Break-even", "Noise rule"):
        assert part in text
    assert (tmp_path / "p2_runs_offline.jsonl").exists()
    assert "Scoreboard" not in text            # offline numbers never reach the scoreboard
    assert set(out["break_even"]) == {"lookup", "policy", "reasoning", "action", "sensitive"}


def test_effort_low_medium_high_use_the_responses_api(monkeypatch):
    """OpenAI refuses function tools with reasoning on Chat Completions, so those efforts need the Responses API."""
    from evals import runner
    seen = []
    monkeypatch.setattr(runner, "get_model", lambda spec, **kw: seen.append(kw) or object())
    for effort in ("none", "low", "medium", "high"):
        runner.default_factory(runner.Config("openai:gpt-5.4-mini", effort))
    assert seen[0] == {"reasoning_effort": "none"}
    for kw, effort in zip(seen[1:], ("low", "medium", "high")):
        assert kw == {"use_responses_api": True, "reasoning": {"effort": effort}}


def test_sweep_only_keeps_tier_runs_and_replaces_failed_effort_runs(tmp_path, monkeypatch):
    import json
    from evals import oracle, scoreboard
    from scripts import check_p2_eval as p2
    monkeypatch.setattr(scoreboard, "PATH", tmp_path / "scoreboard.json")
    oracle.use_empty_handbook()
    first = p2.main(offline=True, quick=True, out_dir=tmp_path)
    runs = tmp_path / "p2_runs.jsonl"
    runs.write_text((tmp_path / "p2_runs_offline.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    rows = [json.loads(line) for line in runs.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        if r["effort"] in ("low", "medium", "high"):
            r["error"], r["passed"] = "boom", False
    runs.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    p2.RESULTS = tmp_path
    out = p2.main(offline=True, quick=True, sweep_only=True, out_dir=tmp_path)
    got = out["results"]
    assert len(got) == len(first["results"])
    assert not any(r["error"] for r in got)
