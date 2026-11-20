import pytest

from helpdesk import hrms
from helpdesk.scripted import FailingModel, ScriptedModel, ai
from router.gateway import Gateway, ListSink
from router.resilience import (CascadeExecutor, EscalationBudget, EscalateBeforeWrite, block_writes, check_answer,
                               wrote)
from router.rules import RulesRouter
from router.types import Route

TICKET = {"employee_id": "E1042", "category": "payroll", "summary": "Please check my October HRA."}
CITED = "Notice period is 60 days [HR-Policy 4.2]."


def factory(models: dict):
    """spec -> a model, or a list of models used in order. Records the specs that were called."""
    called = []

    def make(spec):
        called.append(spec)
        entry = models[spec]
        return entry.pop(0) if isinstance(entry, list) else entry

    make.called = called
    return make


def scripted(*msgs):
    return ScriptedModel(script=list(msgs), seen=[])


def gw(models, tier="small", **kw):
    executor = CascadeExecutor(**kw)
    g = Gateway(lambda m, h: Route(tier, "test"), executor=executor, model_factory=factory(models), sink=ListSink())
    return g, executor


def ask(g, text="How many sick days do I have?"):
    return g.invoke([{"role": "user", "content": text}])


def answer(text, name, tokens=(100, 10)):
    return ai(text, model_name=name, tokens=tokens)


# ---- the check -------------------------------------------------------------------------------

def msgs(*items):
    from langchain_core.messages import AIMessage, ToolMessage
    out = []
    for kind, content, calls in items:
        out.append(ToolMessage(content=content, tool_call_id="t") if kind == "tool"
                   else AIMessage(content=content, tool_calls=[{"name": n, "args": {}, "id": "i", "type": "tool_call"} for n in calls]))
    return out


def test_check_catches_empty_json_uncited_and_refused_after_data():
    assert check_answer(msgs(("ai", "You have 6 sick days.", []))) == []
    assert "empty answer" in check_answer(msgs(("ai", "  ", [])))
    assert "answer is raw JSON" in check_answer(msgs(("ai", '{"earned": 12}', [])))
    searched = [("ai", "", ["search_policy"]), ("tool", '{"results": [{"section": "x"}]}', [])]
    assert "handbook searched but no section cited" in check_answer(msgs(*searched, ("ai", "It is 60 days.", [])))
    assert check_answer(msgs(*searched, ("ai", CITED, []))) == []
    data = [("ai", "", ["get_leave_balance"]), ("tool", '{"ok": true, "earned": 12}', [])]
    assert "refused although a tool returned data" in check_answer(msgs(*data, ("ai", "I can't share that.", [])))


def test_a_refusal_with_no_data_passes_the_check():
    # The salary refusal is correct: the agent refused and no tool ran. The check must not punish it.
    assert check_answer(msgs(("ai", "I can't share another employee's pay.", []))) == []


def test_wrote_sees_the_write_tool():
    assert wrote(msgs(("ai", "", ["raise_ticket"]))) and not wrote(msgs(("ai", "", ["get_payslip"])))


# ---- fallbacks -------------------------------------------------------------------------------

def test_a_failing_primary_falls_back_to_the_second_provider():
    g, _ = gw({"openai:gpt-5.4-nano": FailingModel(error=TimeoutError("slow")),
               "anthropic:claude-haiku-5-5": scripted(answer("6 sick days.", "claude-haiku-5-5"))}, second_provider=True)
    rec = ask(g).record
    assert rec["error"] is None and rec["escalated"] is False
    assert [a["status"] for a in rec["attempts"]] == ["error", "ok"]
    assert rec["models"] == ["anthropic:claude-haiku-5-5"]            # only the model that answered cost money


def test_without_a_second_provider_the_fallback_is_the_next_tier_up():
    g, _ = gw({"openai:gpt-5.4-nano": FailingModel(error=ConnectionError("down")),
               "openai:gpt-5.4-mini": scripted(answer("6 sick days.", "gpt-5.4-mini"))})
    rec = ask(g).record
    assert rec["models"] == ["openai:gpt-5.4-mini"] and rec["escalated"] is False   # a failover is not a quality escalation


def test_when_every_model_fails_the_turn_reports_an_error():
    g, _ = gw({"openai:gpt-5.4-nano": FailingModel(error=TimeoutError("a")),
               "openai:gpt-5.4-mini": FailingModel(error=TimeoutError("b"))})
    rec = ask(g).record
    assert "TimeoutError" in rec["error"] and len(rec["attempts"]) == 2


def test_total_attempts_are_capped_across_models():
    g, _ = gw({"openai:gpt-5.4-nano": FailingModel(), "anthropic:claude-haiku-5-5": FailingModel(),
               "openai:gpt-5.4-mini": FailingModel()}, second_provider=True, max_attempts=2)
    rec = ask(g).record
    assert len(rec["attempts"]) == 2 and rec["error"]


# ---- the cascade -----------------------------------------------------------------------------

def test_a_failed_check_escalates_one_tier_and_pays_for_both_attempts():
    g, _ = gw({"openai:gpt-5.4-nano": scripted(answer("", "gpt-5.4-nano", (200, 0))),
               "openai:gpt-5.4-mini": scripted(answer("6 sick days.", "gpt-5.4-mini", (300, 10)))})
    r = ask(g)
    assert r.answer == "6 sick days." and r.record["escalated"] is True
    assert r.record["attempts"][0]["check_failed"] == ["empty answer"]
    assert r.record["cost_usd"] == pytest.approx((200 * 0.20) / 1e6 + (300 * 0.75 + 10 * 4.5) / 1e6)


def test_a_passing_check_does_not_escalate():
    g, _ = gw({"openai:gpt-5.4-nano": scripted(answer("6 sick days.", "gpt-5.4-nano"))})
    rec = ask(g).record
    assert rec["escalated"] is False and len(rec["attempts"]) == 1


def test_the_cascade_can_be_switched_off():
    g, _ = gw({"openai:gpt-5.4-nano": scripted(answer("", "gpt-5.4-nano"))}, cascade=False)
    assert len(ask(g).record["attempts"]) == 1


def test_large_has_nowhere_to_go_so_the_answer_is_served():
    g, _ = gw({"openai:gpt-5.4": scripted(answer("", "gpt-5.4"))}, tier="large")
    rec = ask(g).record
    assert len(rec["attempts"]) == 1 and "no higher tier" in rec["attempts"][0]["note"]


# ---- writes ----------------------------------------------------------------------------------

def test_a_cheap_tier_never_writes_it_moves_up_first():
    hrms.reset()
    ticket_call = ai("", ("raise_ticket", TICKET), model_name="gpt-5.4-nano")
    g, _ = gw({"openai:gpt-5.4-nano": scripted(ticket_call),
               "openai:gpt-5.4-mini": scripted(ai("", ("raise_ticket", TICKET), model_name="gpt-5.4-mini"),
                                               answer("Done, ticket raised.", "gpt-5.4-mini"))})
    rec = ask(g, "Please ask HR to check my HRA.").record
    assert [a["status"] for a in rec["attempts"]] == ["blocked-write", "ok"]
    assert rec["escalated"] is True and len(hrms.list_tickets("E1042")) == 1   # the small model never wrote


def test_the_guard_wrapper_raises_for_write_tools_only():
    from helpdesk.agent import TOOLS
    by_name = {t.name: t for t in block_writes(TOOLS)}
    with pytest.raises(EscalateBeforeWrite):
        by_name["raise_ticket"].invoke(TICKET)
    assert by_name["get_leave_balance"].invoke({"employee_id": "E1042"})["ok"] is True


def test_after_a_write_the_turn_is_never_rerun_even_if_the_check_fails():
    hrms.reset()
    g, ex = gw({"openai:gpt-5.4-mini": scripted(ai("", ("raise_ticket", TICKET), model_name="gpt-5.4-mini"),
                                                answer("", "gpt-5.4-mini"))}, tier="mid")
    rec = ask(g, "Please ask HR to check my HRA.").record
    assert len(rec["attempts"]) == 1 and "write already ran" in rec["attempts"][0]["note"]
    assert len(hrms.list_tickets("E1042")) == 1


def test_an_error_after_a_write_stops_the_turn_and_no_other_model_is_tried():
    from helpdesk.scripted import ScriptedThenFailing
    hrms.reset()
    again = {**TICKET, "summary": "HRA check requested."}
    fallback = scripted(ai("", ("raise_ticket", again), model_name="gpt-5.4"), answer("Ticket raised.", "gpt-5.4"))
    g, ex = gw({"openai:gpt-5.4-mini": ScriptedThenFailing(
                    script=[ai("", ("raise_ticket", TICKET), model_name="gpt-5.4-mini")], seen=[]),
                "openai:gpt-5.4": fallback}, tier="mid")
    rec = ask(g, "Please ask HR to check my HRA.").record
    assert len(hrms.list_tickets("E1042")) == 1
    assert rec["error"] and len(rec["attempts"]) == 1 and "write may have run" in rec["attempts"][0]["note"]
    assert fallback.i == 0


def test_an_error_before_any_write_still_falls_back():
    hrms.reset()
    g, ex = gw({"openai:gpt-5.4-mini": FailingModel(error=TimeoutError("timed out")),
                "openai:gpt-5.4": scripted(answer("You have 6 sick days.", "gpt-5.4"))}, tier="mid")
    rec = ask(g, "How many sick days do I have?").record
    assert rec["error"] is None and len(rec["attempts"]) == 2


# ---- the budget ------------------------------------------------------------------------------

def test_the_budget_stops_quality_escalations_but_never_the_write_guard():
    b = EscalationBudget(max_rate=0.2, window=10)          # at most 2 escalations in the last 10 requests
    assert b.allow()
    b.record(True), b.record(True)
    assert not b.allow() and b.used == 2
    for _ in range(10):
        b.record(False)                                    # old escalations roll out of the window
    assert b.allow()


def test_a_used_up_budget_serves_the_cheap_answer():
    models = {"openai:gpt-5.4-nano": [scripted(answer("", "gpt-5.4-nano")) for _ in range(3)],
              "openai:gpt-5.4-mini": [scripted(answer("6 sick days.", "gpt-5.4-mini")) for _ in range(3)]}
    g, _ = gw(models, budget=EscalationBudget(max_rate=0.2, window=10))   # 2 per 10
    out = [ask(g).record for _ in range(3)]
    assert [r["escalated"] for r in out] == [True, True, False]
    assert "budget used up" in out[2]["attempts"][0]["note"]


def test_write_escalations_do_not_use_the_budget():
    hrms.reset()
    budget = EscalationBudget(max_rate=0.1, window=10)     # room for 1
    call = lambda n: ai("", ("raise_ticket", TICKET), model_name=n)  # noqa: E731
    g, _ = gw({"openai:gpt-5.4-nano": [scripted(call("gpt-5.4-nano")) for _ in range(2)],
               "openai:gpt-5.4-mini": [scripted(call("gpt-5.4-mini"), answer("ok", "gpt-5.4-mini")) for _ in range(2)]},
              budget=budget)
    for _ in range(2):
        ask(g, "Please ask HR to check my HRA.")
    assert budget.used == 0


def test_the_cascade_works_behind_a_real_router():
    g = Gateway(RulesRouter({"lookup": "small"}), executor=CascadeExecutor(),
                model_factory=factory({"openai:gpt-5.4-nano": scripted(answer("12 earned days.", "gpt-5.4-nano"))}))
    rec = g.invoke([{"role": "user", "content": "How many earned leave days do I have left?"}]).record
    assert (rec["rule"], rec["tier"], rec["escalated"]) == ("cost:lookup", "small", False)


def test_offline_check_shows_every_scripted_scenario(tmp_path, capsys):
    from scripts import check_p5_failures
    out = check_p5_failures.main(offline=True, out_dir=tmp_path)
    text = capsys.readouterr().out
    for part in ("1. The primary model times out", "5. The cheap model tries to raise a ticket", "6b. A write ran", "NOT re-run",
                 "budget used up", "Router + cascade"):
        assert part in text
    assert "Scoreboard" not in text and (tmp_path / "p5_runs_offline.jsonl").exists()
    assert len(out["results"]) == 8
