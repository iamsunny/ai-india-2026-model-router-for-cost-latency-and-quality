import json

import pytest

from evals import oracle, report
from evals.cases import load_cases
from evals.routed import run_suite_routed
from evals.runner import Config
from helpdesk.scripted import ScriptedModel, ai
from router.gateway import Gateway, JsonlSink, ListSink, merge_usage, split_turn
from router.rules import DEFAULT_TIERS, RulesRouter, load_tiers, mentions_someone_else
from router.types import Route
from scripts import check_p3_rules

TIERS = dict(DEFAULT_TIERS)


@pytest.fixture
def router():
    return RulesRouter(TIERS)


@pytest.mark.parametrize("text,rule", [
    ("I want to file a grievance about my manager", "risk:dispute"),
    ("I think I am underpaid", "risk:dispute"),
    ("What is Rohan Mehta's salary?", "risk:other-person"),
    ("Give me the payslip for E1043 for 2026-10", "risk:other-person"),
    ("Tell me what everyone on my team earns", "risk:other-person"),
    ("Please ask HR to look into my payslip", "cost:action"),
    ("What percentage of my gross pay did I take home?", "cost:reasoning"),
    ("How many earned leave days do I have left?", "cost:lookup"),
    ("What is my casual leave balance?", "cost:lookup"),
    ("How many days in advance must I apply for earned leave?", "cost:policy"),
    ("Hello there", "default"),
])
def test_each_rule_fires_on_its_pattern(router, text, rule):
    assert router.detect(text)[0] == rule


def test_my_own_id_is_not_someone_else():
    assert not mentions_someone_else("Show my payslip, I am E1042")
    assert mentions_someone_else("Show the payslip of E1043")


def test_risk_rules_come_before_cost_rules(router):
    # This looks like a cheap lookup, but it is about someone else's pay, so risk wins.
    route = router("How much PF did Rohan Mehta's payslip show in October 2026?")
    assert route.tier == "large" and route.rule == "risk:other-person"


def test_tiers_come_from_the_task_type_table():
    cheap = RulesRouter({"lookup": "small", "policy": "small", "reasoning": "small", "action": "small"})
    assert cheap("How many earned leave days do I have left?").tier == "small"
    assert cheap("Hello there").tier == "mid"            # nothing matched: the baseline tier
    assert cheap("I want to file a grievance").tier == "large"   # no table can lower a risk rule


def test_a_dispute_raised_earlier_still_counts(router):
    route = router("and what about September?", history=["I want to file a grievance about my pay"])
    assert route.rule == "risk:dispute"


def test_a_short_follow_up_borrows_the_previous_question(router):
    assert router("and for September?", history=["What was my net pay in October 2026?"]).rule == "cost:lookup"
    assert router("and for September?").rule == "default"        # no history, no topic


def test_load_tiers_reads_the_part_2_table(tmp_path):
    path = tmp_path / "break_even.json"
    path.write_text(json.dumps({"lookup": "small", "reasoning": None}))
    tiers, source = load_tiers(path)
    assert tiers == {"lookup": "small", "reasoning": "large"}    # no tier passed reasoning, so use large
    assert "Part 2" in source
    assert load_tiers(tmp_path / "missing.json")[0] == DEFAULT_TIERS


def test_merge_usage_adds_tokens_and_cache_details():
    a = {"m": {"input_tokens": 10, "output_tokens": 2, "input_token_details": {"cache_read": 4}}}
    b = {"m": {"input_tokens": 5, "output_tokens": 1}, "n": {"input_tokens": 1, "output_tokens": 1}}
    out = merge_usage(a, b, None)
    assert out["m"] == {"input_tokens": 15, "output_tokens": 3, "input_token_details": {"cache_read": 4}}
    assert out["n"]["input_tokens"] == 1


def test_split_turn_takes_the_latest_user_message():
    msgs = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "x"}, {"role": "user", "content": "two"}]
    assert split_turn(msgs) == ("two", ["one"])
    with pytest.raises(ValueError):
        split_turn([{"role": "assistant", "content": "hi"}])


def scripted_for(spec_to_script):
    return lambda spec: ScriptedModel(script=spec_to_script[spec], seen=[])


def test_gateway_picks_the_tier_runs_it_and_logs_one_record():
    sink = ListSink()
    script = {"openai:gpt-5.4-nano": [ai("You have 12 earned leave days.", model_name="gpt-5.4-nano", tokens=(1000, 10))]}
    gw = Gateway(RulesRouter(TIERS), model_factory=scripted_for(script), sink=sink)
    result = gw.invoke([{"role": "user", "content": "How many earned leave days do I have left?"}])
    assert result.answer == "You have 12 earned leave days."
    (rec,) = sink.records
    assert (rec["tier"], rec["rule"]) == ("small", "cost:lookup")
    assert rec["models"] == ["openai:gpt-5.4-nano"]
    assert rec["cost_usd"] == pytest.approx((1000 * 0.20 + 10 * 1.25) / 1e6)
    assert rec["error"] is None and rec["escalated"] is False


def test_a_broken_router_falls_back_to_the_baseline_instead_of_failing():
    def broken(message, history):
        raise RuntimeError("router crashed")

    script = {"openai:gpt-5.4-mini": [ai("ok", model_name="gpt-5.4-mini")]}
    gw = Gateway(broken, model_factory=scripted_for(script))
    rec = gw.invoke([{"role": "user", "content": "hi"}]).record
    assert rec["tier"] == "mid" and rec["rule"].startswith("router-error")


def test_router_usage_is_billed_with_the_answer():
    class WithOverhead:
        name = "with overhead"

        def __call__(self, message, history):
            return Route("small", "classifier", usage={"gpt-5.4-nano": {"input_tokens": 400, "output_tokens": 20}})

    script = {"openai:gpt-5.4-nano": [ai("ok", model_name="gpt-5.4-nano", tokens=(100, 10))]}
    rec = Gateway(WithOverhead(), model_factory=scripted_for(script)).invoke([{"role": "user", "content": "hi"}]).record
    assert rec["router_cost_usd"] == pytest.approx((400 * 0.20 + 20 * 1.25) / 1e6)
    assert rec["cost_usd"] == pytest.approx((500 * 0.20 + 30 * 1.25) / 1e6)


def test_a_model_error_is_recorded_not_raised():
    def boom(spec):
        raise TimeoutError("slow")

    rec = Gateway(RulesRouter(TIERS), model_factory=boom).invoke([{"role": "user", "content": "hi"}]).record
    assert "TimeoutError" in rec["error"] and rec["cost_usd"] == 0


def test_jsonl_sink_appends_one_line_per_request(tmp_path):
    sink = JsonlSink(tmp_path / "logs" / "routes.jsonl")
    sink.write({"a": 1})
    sink.write({"a": 2})
    assert [json.loads(x)["a"] for x in (tmp_path / "logs" / "routes.jsonl").read_text().splitlines()] == [1, 2]


def test_routed_eval_scores_through_the_gateway():
    oracle.use_empty_handbook()
    gw = Gateway(RulesRouter(TIERS), name="rules router")
    cases = load_cases("holdout")[:4]
    out = run_suite_routed(gw, cases, runs=2, model_for_case=lambda spec, case: oracle.oracle_factory(Config(spec), case))
    assert len(out) == 8 and all(r["passed"] for r in out)
    assert {r["config"] for r in out} == {"rules router"} and all(r["tier"] in ("small", "mid", "large") for r in out)


def run(id_, tier, rule, passed, cfg="rules router", n=3):
    return [{"id": id_, "tier": tier, "rule": rule, "passed": i < passed, "config": cfg, "task_type": "lookup",
             "cost_usd": 0.001, "seconds": 1.0, "split": "holdout", "run": i, "error": None} for i in range(n)]


def test_misroutes_compares_the_router_with_the_part_2_runs():
    router_runs = run("a", "small", "cost:lookup", 0) + run("b", "mid", "cost:policy", 3) + run("c", "small", "cost:lookup", 3)
    base = (run("a", "mid", "", 3, "mid") + run("a", "small", "", 0, "small")
            + run("b", "mid", "", 3, "mid") + run("b", "small", "", 3, "small")
            + run("c", "mid", "", 3, "mid") + run("c", "small", "", 3, "small"))
    found = {m["id"]: m["kind"] for m in report.misroutes(router_runs, base, "mid", "small")}
    assert found == {"a": "under-served", "b": "over-served"}     # c was fine


def test_offline_check_prints_the_route_tables(tmp_path, capsys):
    out = check_p3_rules.main(offline=True, quick=True, out_dir=tmp_path)
    text = capsys.readouterr().out
    for part in ("OFFLINE DEMO", "How the rules read the questions", "The recurring questions",
                 "salary-refusal  -> large", "The rules router"):
        assert part in text
    assert (tmp_path / "p3_runs_offline.jsonl").exists() and "Scoreboard" not in text
    assert out["stats"]["runs"] == 5
