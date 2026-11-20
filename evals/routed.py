"""Run eval cases through a gateway instead of a single model, with the same result format as evals.runner.

Parts 3 to 6 use this to put each router on the scoreboard. Cost includes everything the router
spent: the classifier call, retries and escalations, not only the final answer.
"""
from __future__ import annotations

from typing import Callable

from evals.scoring import check_case
from helpdesk import hrms
from router.gateway import Gateway


def run_routed(case: dict, gateway: Gateway, run: int = 0,
               model_for_case: Callable[[str, dict], object] | None = None) -> dict:
    """Run one case through the gateway. `model_for_case(spec, case)` lets offline tests supply a scripted model."""
    hrms.reset()
    saved = gateway.model_factory
    if model_for_case:
        gateway.model_factory = lambda spec: model_for_case(spec, case)
    try:
        result = gateway.invoke([{"role": "user", "content": case["question"]}])
    finally:
        gateway.model_factory = saved
    rec = result.record
    if result.messages and not rec["error"]:
        verdict = check_case(case, result.messages)
    else:
        verdict = {"id": case["id"], "task_type": case["task_type"], "calls": [],
                   "answer_ok": False, "tools_ok": False, "passed": False}
    return {**verdict, "config": gateway.name, "spec": "router", "effort": None, "split": case.get("split"),
            "run": run, "error": rec["error"], "tier": rec["tier"], "rule": rec["rule"],
            "escalated": rec["escalated"], "models": rec["models"], "input_tokens": rec["input_tokens"],
            "output_tokens": rec["output_tokens"], "cached_tokens": rec["cached_tokens"],
            "router_cost_usd": rec["router_cost_usd"], "cost_usd": rec["cost_usd"], "seconds": rec["seconds"],
            "answer": result.answer}


def run_suite_routed(gateway: Gateway, cases: list[dict], runs: int = 3,
                     model_for_case: Callable[[str, dict], object] | None = None) -> list[dict]:
    return [run_routed(c, gateway, i, model_for_case) for c in cases for i in range(runs)]


def run_conversation(conv: dict, cases_by_id: dict, gateway: Gateway,
                     model_for_case: Callable[[str, dict], object] | None = None) -> list[dict]:
    """Replay one conversation turn by turn. The gateway sees the whole conversation so far, and routes the
    latest message. Returns one result per turn, in the same format as run_routed."""
    hrms.reset()
    saved = gateway.model_factory
    messages: list = []
    out = []
    try:
        for turn, case_id in enumerate(conv["turns"]):
            case = cases_by_id[case_id]
            messages = messages + [{"role": "user", "content": case["question"]}]
            if model_for_case:
                gateway.model_factory = (lambda spec, c=case: model_for_case(spec, c))
            result = gateway.invoke(messages)
            rec = result.record
            new = result.messages[len(messages):]
            ok = bool(new) and not rec["error"]
            verdict = check_case(case, new) if ok else {"id": case_id, "task_type": case["task_type"], "calls": [],
                                                        "answer_ok": False, "tools_ok": False, "passed": False}
            out.append({**verdict, "conversation": conv["id"], "turn": turn, "config": gateway.name, "split": case.get("split"),
                        "tier": rec["tier"], "rule": rec["rule"], "escalated": rec["escalated"], "error": rec["error"],
                        "input_tokens": rec["input_tokens"], "output_tokens": rec["output_tokens"],
                        "cached_tokens": rec["cached_tokens"], "cost_usd": rec["cost_usd"], "seconds": rec["seconds"]})
            if result.messages:
                messages = list(result.messages)
    finally:
        gateway.model_factory = saved
    return out
