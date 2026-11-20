from langchain_core.messages import AIMessage, HumanMessage

from evals.cases import SPLITS, TASK_TYPES, count_by_type, load_cases, load_recurring
from evals.scoring import check_case, has, in_order, normalize, tool_calls
from router.cost import usage_cost


def test_case_ids_are_unique_and_fields_are_present():
    cases = load_cases()
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert c["task_type"] in TASK_TYPES
        assert c["split"] in SPLITS
        assert c["question"].strip()
        # a case must check something: facts, forbidden facts or tools
        assert c["must_include"] or c["must_not_include"] or c["tools"] or c["forbidden_tools"], c["id"]


def test_every_task_type_has_enough_cases_in_both_halves():
    for split in SPLITS:
        counts = count_by_type(load_cases(split=split))
        for task_type, n in counts.items():
            assert n >= 6, (split, task_type, n)


def test_the_set_is_about_sixty_cases():
    assert 60 <= len(load_cases()) <= 70


def test_critical_cases_are_all_sensitive():
    assert all(c["task_type"] == "sensitive" for c in load_cases() if c["critical"])


def test_agents_part_8_and_9_cases_were_kept_once():
    sources = {c["source"] for c in load_cases()}
    assert sources == {"agents-part8", "agents-part9", "new"}
    questions = [c["question"].strip().lower() for c in load_cases()]
    assert len(set(questions)) == len(questions)


def test_recurring_questions_cost_what_agents_part_11_says():
    measured = [q for q in load_recurring() if "part11" in q]
    assert len(measured) == 3
    for q in measured:
        p = q["part11"]
        usage = {"gpt-5.4-mini": {"input_tokens": p["input_tokens"], "output_tokens": p["output_tokens"]}}
        assert round(usage_cost(usage)["cost_usd"], 6) == p["cost_usd"], q["id"]


def test_normalize_and_alternatives():
    assert normalize("INR 1,20,734") == "inr 120734"
    assert has(normalize("I don’t know"), "don't know|not covered")
    assert not has("twelve", "12")


def test_in_order_allows_extra_calls_between():
    assert in_order(["a", "c"], ["a", "b", "c"])
    assert not in_order(["c", "a"], ["a", "b", "c"])


def _messages(answer, *tools):
    calls = [{"name": t, "args": {}, "id": f"c{i}", "type": "tool_call"} for i, t in enumerate(tools)]
    return [HumanMessage("q"), AIMessage(content="", tool_calls=calls), AIMessage(content=answer)]


def test_check_case_passes_a_good_run_and_fails_a_bad_one():
    case = {"id": "x", "task_type": "lookup", "must_include": ["12"], "must_not_include": ["98000"],
            "tools": ["get_leave_balance"], "forbidden_tools": ["raise_ticket"]}
    good = check_case(case, _messages("You have 12 earned days.", "get_leave_balance"))
    assert good["passed"] and good["calls"] == ["get_leave_balance"]
    assert not check_case(case, _messages("You have 11 earned days.", "get_leave_balance"))["answer_ok"]
    assert not check_case(case, _messages("You have 12 earned days."))["tools_ok"]          # lucky guess, no tool
    assert not check_case(case, _messages("12, and Rohan earns 98000", "get_leave_balance"))["answer_ok"]
    assert not check_case(case, _messages("12", "get_leave_balance", "raise_ticket"))["tools_ok"]
    assert tool_calls(_messages("x", "a", "b")) == ["a", "b"]
