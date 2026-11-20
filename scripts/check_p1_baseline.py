"""Part 1 check: what do the Agents Part 11 questions (and one harder one) cost on each tier?

    python -m scripts.check_p1_baseline              # real run on three OpenAI tiers, needs OPENAI_API_KEY
    python -m scripts.check_p1_baseline --offline    # scripted model, no key, shows the format only

Before a real run, build the handbook index once:  python -m helpdesk.ingest
"""
import argparse
import time

from langchain_core.callbacks import get_usage_metadata_callback

from evals.cases import count_by_type, load_cases, load_recurring
from helpdesk import hrms
from helpdesk.agent import build_agent
from helpdesk.llm import get_model
from router.catalog import TIERS, load_catalog
from router.cost import usage_cost


def run_question(agent, question: str) -> dict:
    """Ask one question on a fresh HR system and report tokens, cost, calls and time."""
    hrms.reset()
    started = time.perf_counter()
    with get_usage_metadata_callback() as usage:
        out = agent.invoke({"messages": [{"role": "user", "content": question}]})
    seconds = time.perf_counter() - started
    report = usage_cost(usage.usage_metadata)
    report["model_calls"] = sum(1 for m in out["messages"] if m.type == "ai")
    report["tools"] = [c["name"] for m in out["messages"] if m.type == "ai" for c in m.tool_calls]
    report["seconds"] = seconds
    report["answer"] = out["messages"][-1].content
    return report


def format_row(spec: str, q: dict, r: dict) -> str:
    short = spec.split(":", 1)[-1]
    return (f"{short:<14} {q['id']:<15} in {r['input_tokens']:>5} out {r['output_tokens']:>4} "
            f"cached {r['cached_tokens']:>3} calls {r['model_calls']} ${r['cost_usd']:.6f} {r['seconds']:.1f}s")


def main(models: list[str], offline: bool = False) -> None:
    catalog = load_catalog()
    questions = load_recurring()
    if offline:
        print("OFFLINE DEMO: the model is scripted, so tokens, cost and time are not real.\n")
    print(f"Prices from models.yaml, checked {catalog.checked}\n")
    for spec in models:
        costs = []
        for q in questions:
            agent = build_agent(offline_model(q) if offline else get_model(spec))
            r = run_question(agent, q["question"])
            costs.append(r["cost_usd"])
            print(format_row(spec, q, r))
            if not offline:
                print(f"    tools: {r['tools']}")
                print(f"    answer: {str(r['answer'])[:200]}")
        average = sum(costs) / len(costs)
        print(f"{spec.split(':', 1)[-1]}: average ${average:.6f} per question over {len(costs)} questions, "
              f"about ${average * 100_000:,.0f} per 100,000 questions\n")
    print("Agents Part 11 numbers, for reference only (gpt-5.4-mini, list price).")
    print("Part 11 used a different agent: its own prompt, four tools and guardrails.")
    print("This repo's agent has a longer prompt and five tools, so token counts differ. Do not expect a match:")
    for q in questions:
        p = q.get("part11")
        if p:
            print(f"  {q['id']:<15} in {p['input_tokens']:>5} out {p['output_tokens']:>4} ${p['cost_usd']:.6f}")
    cases = load_cases()
    print(f"\nEval set: {len(cases)} cases")
    for t, n in count_by_type(cases).items():
        print(f"  {t:<10} {n}")


def offline_model(question: dict):
    """A scripted model for the offline demo."""
    from helpdesk.scripted import ScriptedModel, ai
    m = "gpt-5.4-mini"   # a name the price catalog knows; the tokens below are made up
    script = {
        "leave-balance": [ai("", ("get_leave_balance", {"employee_id": "E1042"}), model_name=m, tokens=(400, 12)),
                          ai("You have 12 earned leave days.", model_name=m, tokens=(277, 15))],
        "salary-refusal": [ai("I can't share another employee's pay.", model_name=m, tokens=(315, 69))],
        "wfh-fridays": [ai("Up to 2 days a week.", model_name=m, tokens=(897, 126))],
        "pay-diff": [ai("", ("get_payslip", {"employee_id": "E1042", "month": "2026-10"}),
                        ("get_payslip", {"employee_id": "E1042", "month": "2026-09"}), model_name=m, tokens=(500, 40)),
                     ai("Your income tax went up by 350.", model_name=m, tokens=(700, 60))],
    }[question["id"]]
    return ScriptedModel(script=script)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="use a scripted model, no API key")
    parser.add_argument("--models", nargs="*", help="model strings; default is the three OpenAI tiers")
    args = parser.parse_args()
    catalog = load_catalog()
    chosen = args.models or [catalog.tier_spec(t) for t in TIERS]
    main(["offline-demo"] if args.offline else chosen, offline=args.offline)
