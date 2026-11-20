import json

import pytest

from helpdesk import hrms
from helpdesk.agent import TOOLS, build_agent
from helpdesk.scripted import ScriptedModel, ai
from router.resilience import scope_to


def tool_results(messages):
    return [json.loads(m.content) for m in messages if m.type == "tool"]


@pytest.mark.parametrize("model_name", ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"])
def test_scope_to_refuses_another_employees_record_whichever_model_asks(model_name):
    hrms.reset()
    model = ScriptedModel(script=[ai("", ("get_leave_balance", {"employee_id": "E1043"}), model_name=model_name),
                                  ai("I cannot share that.", model_name=model_name)], seen=[])
    out = build_agent(model, tools=scope_to(TOOLS, "E1042")).invoke({"messages": [("user", "Leave of E1043?")]})
    result = tool_results(out["messages"])[0]
    assert result["ok"] is False and "denied by policy" in result["error"] and "earned" not in result


def test_scope_to_still_serves_the_callers_own_record():
    hrms.reset()
    model = ScriptedModel(script=[ai("", ("get_leave_balance", {"employee_id": "E1042"}), model_name="gpt-5.4-mini"),
                                  ai("You have leave.", model_name="gpt-5.4-mini")], seen=[])
    out = build_agent(model, tools=scope_to(TOOLS, "E1042")).invoke({"messages": [("user", "My leave?")]})
    assert tool_results(out["messages"])[0]["ok"] is True


def test_quickstart_runs_offline_and_is_labelled_simulated(capsys):
    from scripts import quickstart
    quickstart.main()
    text = capsys.readouterr().out
    assert "SIMULATED" in text and "$" not in text
    assert "gpt-5.4-nano: error (TimeoutError" in text and "gpt-5.4-mini: ok" in text
    assert "classifier:invalid" in text and "risk:other-person" in text


@pytest.mark.parametrize("call", [("get_payslip", {"employee_id": "E1043", "month": "2026-10"}),
                                  ("raise_ticket", {"employee_id": "E1043", "category": "payroll",
                                                    "summary": "Please check the payslip."})])
def test_scope_to_covers_the_other_employee_tools_and_writes_no_ticket(call):
    hrms.reset()
    model = ScriptedModel(script=[ai("", call, model_name="gpt-5.4-mini"), ai("I cannot do that.", model_name="gpt-5.4-mini")],
                          seen=[])
    out = build_agent(model, tools=scope_to(TOOLS, "E1042")).invoke({"messages": [("user", "Do it for E1043.")]})
    assert tool_results(out["messages"])[0]["ok"] is False
    assert hrms.list_tickets("E1043") == []
