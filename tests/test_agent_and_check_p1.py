from helpdesk import hrms
from helpdesk.agent import SYSTEM_PROMPT, build_agent
from helpdesk.scripted import ScriptedModel, ai
from scripts import check_p1_baseline


def test_agent_calls_a_tool_and_answers():
    hrms.reset()
    model = ScriptedModel(script=[ai("", ("get_leave_balance", {"employee_id": "E1042"})),
                                  ai("You have 12 earned leave days.")])
    out = build_agent(model).invoke({"messages": [{"role": "user", "content": "How many earned leave days?"}]})
    tool_result = [m for m in out["messages"] if m.type == "tool"][0]
    assert '"earned": 12' in tool_result.content
    assert out["messages"][-1].content == "You have 12 earned leave days."


def test_agent_prompt_has_the_part_9_rules():
    assert "another employee's data" in SYSTEM_PROMPT
    assert "which month" in SYSTEM_PROMPT


def test_the_idempotent_ticket_tool_survives_a_repeat():
    hrms.reset()
    args = {"employee_id": "E1042", "category": "payroll", "summary": "My October HRA looks wrong."}
    model = ScriptedModel(script=[ai("", ("raise_ticket", args), ("raise_ticket", args)), ai("Done.")])
    out = build_agent(model).invoke({"messages": [{"role": "user", "content": "Ask HR to check my HRA."}]})
    assert len(hrms.list_tickets("E1042")) == 1
    assert out["messages"][-1].content == "Done."


def test_offline_check_prints_the_three_questions_and_the_eval_counts(capsys):
    check_p1_baseline.main(["offline-demo"], offline=True)
    text = capsys.readouterr().out
    assert "OFFLINE DEMO" in text
    for name in ("leave-balance", "salary-refusal", "wfh-fridays", "pay-diff"):
        assert name in text
    assert "$0.000629" in text            # 677 in, 27 out in total, as in Agents Part 11
    assert "$0.000547" in text            # 315 in, 69 out on gpt-5.4-mini
    assert "$0.001240" in text            # 897 in, 126 out
    assert "per 100,000 questions" in text
    assert "different agent" in text
    assert "Eval set: 64 cases" in text
    assert "sensitive" in text


def test_missing_handbook_index_gives_a_friendly_error(monkeypatch, tmp_path):
    import pytest

    from helpdesk import search
    monkeypatch.setattr(search, "INDEX", tmp_path / "nope.json")
    monkeypatch.setattr(search, "_store", None)
    with pytest.raises(FileNotFoundError, match="python -m helpdesk.ingest"):
        search.get_store()
