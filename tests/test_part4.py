
import pytest

from evals.replay import replay
from helpdesk.scripted import ScriptedModel, ai
from router.classifier import (Classification, ClassifierRouter, LLMClassifier, parse_output, tier_for, up)
from router.gateway import Gateway
from scripts import check_p4_classifier

TIERS = {"lookup": "small", "action": "small", "policy": "mid", "reasoning": "mid"}


def good(task="lookup", difficulty="easy", confidence=0.9):
    return Classification(True, task, difficulty, confidence)


def test_parse_accepts_plain_and_fenced_json_and_nothing_looser():
    text = '{"task_type": "policy", "difficulty": "easy", "confidence": 0.8}'
    assert parse_output(text).task_type == "policy"
    assert parse_output(f"```json\n{text}\n```").confidence == 0.8
    for bad in ('{"task_type": "banter", "difficulty": "easy", "confidence": 0.8}',
                '{"task_type": "policy", "difficulty": "easy", "confidence": 1.7}',
                '{"task_type": "policy"}', "I think this is a policy question.", ""):
        with pytest.raises(ValueError):
            parse_output(bad)


def test_up_moves_one_tier_and_stops_at_large():
    assert [up(t) for t in ("small", "mid", "large")] == ["mid", "large", "large"]


def test_tier_for_follows_the_task_type_table():
    assert tier_for(good("lookup"), TIERS, 0.7) == ("small", "classifier:lookup")
    assert tier_for(good("policy"), TIERS, 0.7) == ("mid", "classifier:policy")


def test_hard_questions_and_low_confidence_each_move_up_one_tier():
    assert tier_for(good("lookup", "hard"), TIERS, 0.7) == ("mid", "classifier:lookup+hard")
    assert tier_for(good("lookup", confidence=0.6), TIERS, 0.7) == ("mid", "classifier:lookup+low-confidence")
    assert tier_for(good("lookup", "hard", 0.6), TIERS, 0.7)[0] == "large"
    assert tier_for(good("lookup", confidence=0.6), TIERS, 0.5)[0] == "small"   # the threshold is the dial


def test_sensitive_is_always_large_and_invalid_falls_back_to_mid():
    assert tier_for(good("sensitive", confidence=0.99), TIERS, 0.0)[0] == "large"
    assert tier_for(Classification(False), TIERS, 0.7) == ("mid", "classifier:invalid")


def scripted_classifier(reply, name="gpt-5.4-nano", tokens=(420, 25)):
    factory = lambda spec: ScriptedModel(script=[ai(reply, model_name=name, tokens=tokens)], seen=[])  # noqa: E731
    return LLMClassifier("openai:gpt-5.4-nano", factory)


def test_llm_classifier_parses_a_reply_and_reports_its_own_usage():
    c = scripted_classifier('{"task_type": "lookup", "difficulty": "easy", "confidence": 0.95}')("How many sick days do I have?")
    assert c.valid and c.task_type == "lookup" and c.usage["gpt-5.4-nano"]["input_tokens"] == 420


def test_llm_classifier_never_raises_on_bad_output_or_model_errors():
    c = scripted_classifier("Sure! This looks like a lookup.")("hi")
    assert not c.valid and "invalid output" in c.error and c.usage      # the tokens were still spent

    def boom(spec):
        raise TimeoutError("slow")

    c = LLMClassifier("openai:gpt-5.4-nano", boom)("hi")
    assert not c.valid and "TimeoutError" in c.error


def test_a_short_follow_up_sends_the_earlier_question_too():
    seen = {}

    def factory(spec):
        model = ScriptedModel(script=[ai('{"task_type": "lookup", "difficulty": "easy", "confidence": 0.9}')], seen=[])
        seen["model"] = model
        return model

    LLMClassifier("openai:gpt-5.4-nano", factory)("and for September?", ["What was my net pay in October 2026?"])
    sent = seen["model"].seen[0][-1].content
    assert "net pay in October" in sent and "and for September?" in sent


def test_risk_questions_skip_the_classifier_and_its_cost():
    calls = []

    def classify(message, history):
        calls.append(message)
        return good()

    router = ClassifierRouter(classify, TIERS)
    route = router("What is Rohan Mehta's salary?")
    assert (route.tier, route.rule, route.usage) == ("large", "risk:other-person", {}) and calls == []
    assert router("How many sick days do I have?").tier == "small" and len(calls) == 1


def test_gateway_bills_the_classifier_with_the_answer():
    classify = scripted_classifier('{"task_type": "lookup", "difficulty": "easy", "confidence": 0.95}')
    script = {"openai:gpt-5.4-nano": [ai("6 sick days.", model_name="gpt-5.4-nano", tokens=(1000, 10))]}
    gw = Gateway(ClassifierRouter(classify, TIERS), model_factory=lambda spec: ScriptedModel(script=script[spec], seen=[]))
    rec = gw.invoke([{"role": "user", "content": "How many sick days do I have?"}]).record
    assert rec["router_cost_usd"] == pytest.approx((420 * 0.20 + 25 * 1.25) / 1e6)
    assert rec["cost_usd"] == pytest.approx((1420 * 0.20 + 35 * 1.25) / 1e6)


def row(config, id_, passed, cost, seconds=1.0):
    return {"config": config, "id": id_, "passed": passed, "cost_usd": cost, "seconds": seconds, "run": 0}


def test_replay_uses_the_part_2_run_for_the_chosen_tier_and_adds_overhead():
    base = [row("nano", "a", True, 0.001), row("mini", "a", True, 0.004), row("nano", "b", False, 0.001),
            row("mini", "b", True, 0.004)]
    labels = {"small": "nano", "mid": "mini"}
    out = replay({"a": "small", "b": "mid"}, base, labels, {"a": {"cost_usd": 0.0005, "seconds": 0.5}})
    assert out["passed"] == 2 and out["runs"] == 2
    assert out["cost"] == pytest.approx(0.001 + 0.0005 + 0.004)
    assert replay({"zzz": "small"}, base, labels)["skipped"] == 1


def test_offline_check_prints_the_sweep_and_the_break_even(tmp_path, capsys):
    out = check_p4_classifier.main(offline=True, quick=True, out_dir=tmp_path)
    text = capsys.readouterr().out
    for part in ("Router brain", "invalid replies", "overhead per request", "break-even", "threshold sweep",
                 "recommended threshold", "The classifier router", "recurring questions"):
        assert part in text
    assert "Scoreboard" not in text and (tmp_path / "p4_runs_offline.jsonl").exists()
    assert out["threshold"] in check_p4_classifier.THRESHOLDS
