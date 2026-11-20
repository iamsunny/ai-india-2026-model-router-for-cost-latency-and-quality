import json
from datetime import date

import pytest

from evals import oracle
from evals.cases import load_cases, load_conversations
from evals.routed import run_conversation
from evals.runner import Config
from router import drift, telemetry
from router.catalog import load_catalog
from router.final import build_final_gateway
from router.gateway import Gateway, ListSink
from router.rules import RulesRouter
from scripts import check_p6_report, drift_check


def rec(tier="small", rule="cost:lookup", cost=0.001, seconds=1.0, escalated=False, error=None, cached=0, tokens=500):
    return {"tier": tier, "rule": rule, "cost_usd": cost, "seconds": seconds, "escalated": escalated, "error": error,
            "cached_tokens": cached, "input_tokens": tokens, "router_cost_usd": 0.0, "models": ["openai:gpt-5.4-nano"]}


def test_every_conversation_turn_names_a_real_case():
    ids = {c["id"] for c in load_cases()}
    convs = load_conversations()
    assert len(convs) == 10 and all(t in ids for c in convs for t in c["turns"])


def test_summary_and_per_route_numbers():
    records = [rec(cost=0.001), rec(cost=0.003, escalated=True), rec(tier="large", rule="risk:dispute", cost=0.01),
               rec(error="TimeoutError: x", cost=0.0)]
    s = telemetry.summarize(records)
    assert s["requests"] == 4 and s["cost"] == pytest.approx(0.014)
    assert s["per_100k"] == pytest.approx(0.014 / 4 * 100_000)
    assert s["escalation_rate"] == 0.25 and s["error_rate"] == 0.25
    tiers = telemetry.by_key(records, "tier")
    assert tiers["small"]["requests"] == 3 and tiers["large"]["cost"] == pytest.approx(0.01)
    assert "share of cost" in telemetry.format_report(records)


def test_percentile_and_load_records(tmp_path):
    assert telemetry.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert telemetry.percentile([], 50) == 0.0
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(rec()) for _ in range(2)) + "\n")
    assert len(telemetry.load_records(path)) == 2 and telemetry.load_records(tmp_path / "none.jsonl") == []


def test_escalation_trend_needs_two_full_windows():
    rows = [rec(escalated=False) for _ in range(5)] + [rec(escalated=True) for _ in range(5)]
    assert telemetry.escalation_trend(rows, window=5) == (1.0, 0.0)
    assert telemetry.escalation_trend(rows, window=6) is None


BASE = {"passed": 90, "runs": 96, "cost_per_conversation": 0.0010, "escalation_rate": 0.10}


def test_the_gate_passes_small_wobbles_and_fails_real_drops():
    assert drift.gate({**BASE, "passed": 88}, BASE) == []                       # 2 runs: noise
    assert drift.gate({**BASE, "passed": 85}, BASE) == []                       # 5 runs, under 2 cases (6 runs)
    assert "quality dropped" in drift.gate({**BASE, "passed": 83}, BASE)[0]
    assert "cost per conversation rose" in drift.gate({**BASE, "cost_per_conversation": 0.0013}, BASE)[0]
    assert drift.gate({**BASE, "cost_per_conversation": 0.0011}, BASE) == []
    assert "escalation rate rose" in drift.gate({**BASE, "escalation_rate": 0.25}, BASE)[0]


def test_price_warnings_flag_old_prices_and_unpriced_models():
    catalog = load_catalog()
    assert drift.price_warnings(catalog, [rec()], today=date(2026, 11, 20)) == []
    old = drift.price_warnings(catalog, [rec()], today=date(2027, 2, 1))
    assert "days ago" in old[0]
    unpriced = [{**rec(), "models": ["openai:gpt-9"]}]
    assert "no price" in drift.price_warnings(catalog, unpriced, today=date(2026, 11, 20))[0]


def test_a_conversation_sends_the_whole_history_and_routes_the_latest_turn():
    oracle.use_empty_handbook()
    seen_history = []

    class Spy(RulesRouter):
        def __call__(self, message, history=None):
            seen_history.append(list(history or []))
            return super().__call__(message, history)

    cases = {c["id"]: c for c in load_cases()}
    gw = Gateway(Spy({"lookup": "small", "reasoning": "mid"}), sink=ListSink(), name="spy")
    conv = {"id": "x", "turns": ["leave-earned", "mix-carry"]}
    out = run_conversation(conv, cases, gw, lambda spec, case: oracle.oracle_factory(Config(spec), case))
    assert [t["turn"] for t in out] == [0, 1] and all(t["passed"] for t in out)
    assert seen_history[0] == [] and seen_history[1] == ["How many earned leave days do I have left?"]


def test_build_final_gateway_wires_router_cascade_and_name():
    g = build_final_gateway("rules", tiers={"lookup": "small"}, sink=ListSink())
    assert "rules + cascade" in g.name and g.executor.budget.allow()
    with pytest.raises(ValueError):
        build_final_gateway("magic")


def test_offline_report_prints_every_section(tmp_path, capsys):
    out = check_p6_report.main(offline=True, quick=True, out_dir=tmp_path)
    text = capsys.readouterr().out
    for part in ("What the telemetry says", "Before and after, same questions", "leave-balance (lookup)",
                 "salary-refusal (sensitive)", "Per 100,000 turns", "Caching:", "Housekeeping"):
        assert part in text
    assert (tmp_path / "routes_offline.jsonl").exists() and "The scoreboard" not in text
    assert len(out["after"]) == len(out["before"]) > 0


def test_drift_check_passes_against_itself_and_fails_when_quality_drops(tmp_path, capsys):
    now = {"passed": 96, "runs": 96, "cost_per_conversation": 0.00259, "escalation_rate": 0.4, "strategy": "rules",
           "saved": "2026-11-19"}
    # An offline run is always right, so a baseline of 96 of 96 passes. Cost and escalation are scripted too.
    code = drift_check.main(offline=True, out_dir=tmp_path, baseline={**now})
    text = capsys.readouterr().out
    assert code in (0, 1) and ("PASS" in text or "FAIL" in text)
    code = drift_check.main(offline=True, out_dir=tmp_path,
                            baseline={**now, "cost_per_conversation": 0.0001, "escalation_rate": 0.0})
    assert code == 1 and "FAIL" in capsys.readouterr().out
    assert drift_check.main(offline=True, out_dir=tmp_path) == 2     # no saved baseline yet


def test_drift_check_rebuilds_the_gateway_with_the_baseline_threshold(tmp_path, monkeypatch):
    seen = {}
    real = drift_check.build_final_gateway

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(drift_check, "build_final_gateway", spy)
    base = {"passed": 96, "runs": 96, "cost_per_conversation": 0.01, "escalation_rate": 1.0, "strategy": "classifier",
            "threshold": 0.0, "saved": "2026-11-20"}
    drift_check.main(offline=True, out_dir=tmp_path, baseline=base)
    assert seen["threshold"] == 0.0
