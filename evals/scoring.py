"""Cheap, exact checks on an agent's answer and the tools it called.

These are the answer and trajectory layers from Agents Part 9. The judge layer is left out here.
"""


def normalize(answer: str) -> str:
    """Lower-case, drop thousands separators, and turn curly quotes into straight ones,
    so "INR 1,20,734" matches "120734" and "don’t know" matches "don't know"."""
    return (answer.lower().replace(",", "")
            .replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"'))


def has(text: str, item: str) -> bool:
    """True if any of the alternatives in item ("a|b|c") appears in the normalized text."""
    return any(alt.lower() in text for alt in item.split("|"))


def tool_calls(messages) -> list[str]:
    return [c["name"] for m in messages if m.type == "ai" for c in m.tool_calls]


def in_order(expected: list[str], actual: list[str]) -> bool:
    """True if the expected tools appear in this order (other calls may sit between them)."""
    it = iter(actual)
    return all(name in it for name in expected)


def check_case(case: dict, messages) -> dict:
    """Score one run of one case: did the answer have the right facts, and were the right tools used?"""
    text = normalize(messages[-1].text)
    calls = tool_calls(messages)
    answer_ok = (all(has(text, s) for s in case.get("must_include", []))
                 and not any(has(text, s) for s in case.get("must_not_include", [])))
    tools_ok = (in_order(case.get("tools", []), calls)
                and not any(t in calls for t in case.get("forbidden_tools", [])))
    return {"id": case["id"], "task_type": case["task_type"], "calls": calls,
            "answer_ok": answer_ok, "tools_ok": tools_ok, "passed": answer_ok and tools_ok}
