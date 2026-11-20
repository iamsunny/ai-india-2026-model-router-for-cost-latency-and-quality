"""A scripted "perfect" model for offline runs.

It calls the tools a case expects and then answers with the facts the case needs, so the whole
pipeline (agent, tools, scorer, cost, tables) can be exercised with no API key. It says nothing
about which real model is good. Tokens and time are made up.
"""
from __future__ import annotations

from helpdesk import search
from helpdesk.scripted import ScriptedModel, ai

# Made-up tokens per call, by tier name. Bigger models get a longer answer, so the tables show a spread.
_TOKENS = {"gpt-5.4-nano": (500, 40), "gpt-5.4-mini": (600, 50), "gpt-5.4": (700, 60)}


class _EmptyStore:
    """Stands in for the handbook index so search_policy needs no embeddings offline."""

    def similarity_search_with_score(self, query, k=4):
        return []


def use_empty_handbook() -> None:
    search._store = _EmptyStore()


def _args(tool: str, case: dict, nth: int) -> dict:
    if tool == "get_leave_balance":
        return {"employee_id": "E1042"}
    if tool == "get_payslip":
        return {"employee_id": "E1042", "month": ["2026-10", "2026-09"][nth % 2]}
    if tool == "get_holidays":
        return {"month": 10}
    if tool == "raise_ticket":
        return {"employee_id": "E1042", "category": "payroll", "summary": "Please follow up on this request."}
    return {"query": case["question"]}   # search_policy


def oracle_answer(case: dict) -> str:
    return " ".join(item.split("|")[0] for item in case.get("must_include", [])) or "Done."


def oracle_factory(config, case: dict) -> ScriptedModel:
    """Build the scripted model for one run. Use as the factory in evals.runner.run_suite."""
    name = config.spec.split(":", 1)[-1]
    tokens = _TOKENS.get(name, (500, 40))
    calls = [(t, _args(t, case, n)) for n, t in enumerate(case.get("tools", []))]
    script = []
    if calls:
        script.append(ai("", *calls, model_name=name, tokens=tokens))
    script.append(ai(oracle_answer(case), model_name=name, tokens=tokens))
    return ScriptedModel(script=script, seen=[])


def oracle_classify(cases: list[dict], model_name: str = "gpt-5.4-nano", tokens: tuple[int, int] = (420, 25),
                    confidence: float = 0.9):
    """A scripted classifier that always names the labelled task type. For offline runs and tests."""
    from router.classifier import Classification
    labels = {c["question"]: c["task_type"] for c in cases}

    def classify(message: str, history=None) -> Classification:
        usage = {model_name: {"input_tokens": tokens[0], "output_tokens": tokens[1]}}
        if message not in labels:
            return Classification(False, usage=usage, seconds=0.0, error="unknown question")
        return Classification(True, labels[message], "easy", confidence, usage, 0.0)

    return classify
