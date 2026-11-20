"""Part 5 check: what happens when a call fails, and what does the safety net cost?

    python -m scripts.check_p5_failures               # scripted failures + a forced failure + a real cascade run
    python -m scripts.check_p5_failures --offline     # scripted failures only, no key
    python -m scripts.check_p5_failures --second-provider   # also use Claude as the fallback (needs ANTHROPIC_API_KEY)
    python -m scripts.check_p5_failures --strategy classifier   # put the cascade behind the Part 4 router

Section 1 needs no key: scripted models fail on command, and the output shows each attempt.
Section 2 breaks the primary model on purpose for one real question.
Section 3 runs the held-out set through router + cascade and puts a row on the scoreboard.
A second provider is a second processor of your HR data. Check that this is acceptable before using it.
"""
import argparse
import json
from pathlib import Path

from evals import oracle, report, scoreboard
from evals.cases import load_cases, load_recurring
from evals.routed import run_suite_routed
from evals.runner import Config
from helpdesk import hrms
from helpdesk.scripted import FailingModel, ScriptedModel, ScriptedThenFailing, ai
from router.catalog import load_catalog
from router.classifier import ClassifierRouter, LLMClassifier
from router.gateway import Gateway
from router.resilience import CascadeExecutor, EscalationBudget, resilient_factory
from router.rules import RulesRouter, load_tiers
from router.types import Route

RESULTS = Path(__file__).resolve().parent.parent / "results"
TICKET = {"employee_id": "E1042", "category": "payroll", "summary": "Please check my October HRA."}


def trail(record: dict) -> str:
    """One line: what each attempt did."""
    parts = []
    for a in record["attempts"]:
        what = a["status"] + (f" ({a['error'].split(':')[0]})" if a.get("error") else "")
        if a.get("check_failed"):
            what += f", check failed: {'; '.join(a['check_failed'])}"
        if a.get("note"):
            what += f", {a['note']}"
        parts.append(f"{a['spec'].split(':', 1)[-1]}: {what}")
    return "  ->  ".join(parts)


def scripted_scenarios() -> None:
    """Failure injection with scripted models. Prints what the safety net did in each case."""
    def run(name, models, tier="small", question="How many sick days do I have?", **kw):
        hrms.reset()
        factory = lambda spec: models[spec].pop(0) if isinstance(models[spec], list) else models[spec]  # noqa: E731
        g = Gateway(lambda m, h: Route(tier, "scripted"), executor=CascadeExecutor(**kw), model_factory=factory)
        rec = g.invoke([{"role": "user", "content": question}]).record
        print(f"\n{name}\n  {trail(rec)}")
        print(f"  escalated: {rec['escalated']}   error: {rec['error']}   tickets written: {len(hrms.list_tickets('E1042'))}")

    ok = lambda text, name: ai(text, model_name=name)  # noqa: E731
    nano, mini, haiku = "gpt-5.4-nano", "gpt-5.4-mini", "claude-haiku-5-5"
    scripted = lambda *m: ScriptedModel(script=list(m), seen=[])  # noqa: E731
    run("1. The primary model times out, and a second provider answers",
        {f"openai:{nano}": FailingModel(error=TimeoutError("timed out")),
         f"anthropic:{haiku}": scripted(ok("You have 6 sick days.", haiku))}, second_provider=True)
    run("2. The primary model times out and there is no second provider: the next tier up answers",
        {f"openai:{nano}": FailingModel(error=TimeoutError("timed out")),
         f"openai:{mini}": scripted(ok("You have 6 sick days.", mini))})
    run("3. Every model fails: the turn reports an error and stops at the attempt limit",
        {f"openai:{nano}": FailingModel(), f"openai:{mini}": FailingModel()}, max_attempts=2)
    run("4. The cheap model returns an empty answer: the cascade moves up one tier",
        {f"openai:{nano}": scripted(ok("", nano)), f"openai:{mini}": scripted(ok("You have 6 sick days.", mini))})
    run("5. The cheap model tries to raise a ticket: it moves up BEFORE the write runs",
        {f"openai:{nano}": scripted(ai("", ("raise_ticket", TICKET), model_name=nano)),
         f"openai:{mini}": scripted(ai("", ("raise_ticket", TICKET), model_name=mini), ok("Ticket raised.", mini))},
        question="Please ask HR to check my HRA.")
    run("6. A write already ran and the check fails: the turn is NOT re-run",
        {f"openai:{mini}": scripted(ai("", ("raise_ticket", TICKET), model_name=mini), ok("", mini))},
        tier="mid", question="Please ask HR to check my HRA.")
    again = {**TICKET, "summary": "HRA check requested."}
    run("6b. A write ran, then the model errors: no other model is tried, the turn stops with an error",
        {f"openai:{mini}": ScriptedThenFailing(script=[ai("", ("raise_ticket", TICKET), model_name=mini)], seen=[],
                                               error=TimeoutError("timed out")),
         "openai:gpt-5.4": scripted(ai("", ("raise_ticket", again), model_name="gpt-5.4"), ok("Ticket raised.", "gpt-5.4"))},
        tier="mid", question="Please ask HR to check my HRA.")
    models = {f"openai:{nano}": [scripted(ok("", nano)) for _ in range(3)],
              f"openai:{mini}": [scripted(ok("6 sick days.", mini)) for _ in range(3)]}
    print("\n7. The escalation budget: 2 escalations allowed per 10 requests, and the check fails on every request")
    ex = CascadeExecutor(budget=EscalationBudget(max_rate=0.2, window=10))
    g = Gateway(lambda m, h: Route("small", "scripted"), executor=ex,
                model_factory=lambda spec: models[spec].pop(0))
    for i in range(3):
        rec = g.invoke([{"role": "user", "content": "How many sick days do I have?"}]).record
        print(f"  request {i + 1}: escalated {rec['escalated']}. {trail(rec)}")


def main(offline: bool = False, second_provider: bool = False, strategy: str = "rules", runs: int = 3,
         break_question: str = "leave-balance", out_dir: Path = RESULTS) -> dict:
    catalog = load_catalog()
    print("== 1. Scripted failures (no key needed) ==")
    scripted_scenarios()

    tiers, source = load_tiers(out_dir / "break_even.json")
    cases = load_cases("holdout")
    if offline:
        print("\nOFFLINE DEMO: the models are scripted and always right, so sections 2 and 3 show format only.")
        oracle.use_empty_handbook()
    router = (ClassifierRouter(LLMClassifier("openai:gpt-5.4-nano"), tiers, name="classifier router")
              if strategy == "classifier" and not offline else RulesRouter(tiers))

    print("\n== 2. A forced failure on one real question ==")
    q = next(x for x in load_recurring() if x["id"] == break_question)
    primary = catalog.tier_spec(router(q["question"]).tier)
    state = {"broken": False}

    def break_primary_once(spec):
        if spec == primary and not state["broken"]:
            state["broken"] = True
            return FailingModel(error=TimeoutError("injected failure"))
        return resilient_factory(spec)

    forced = Gateway(router, executor=CascadeExecutor(second_provider=second_provider), catalog=catalog,
                     model_factory=break_primary_once, name="forced failure")
    if offline:
        print(f"(offline: skipped. A real run breaks {primary} once for '{q['question']}')")
    else:
        hrms.reset()
        res = forced.invoke([{"role": "user", "content": q["question"]}])
        print(f"question: {q['question']}\nbroke: {primary}\n{trail(res.record)}\n"
              f"cost ${res.record['cost_usd']:.6f}   answer: {res.answer[:150]}")

    print("\n== 3. Router + cascade on the held-out questions ==")
    budget = EscalationBudget(max_rate=0.25, window=100)
    executor = CascadeExecutor(second_provider=second_provider, budget=budget)
    name = f"{strategy} router + cascade"
    gateway = Gateway(router, executor=executor, catalog=catalog,
                      model_factory=resilient_factory, name=name)
    run_cases = cases if not offline else cases[:8]
    results = run_suite_routed(gateway, run_cases, runs=1 if offline else runs,
                               model_for_case=(lambda spec, case: oracle.oracle_factory(Config(spec), case))
                               if offline else None)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ("p5_runs_offline.jsonl" if offline else "p5_runs.jsonl")).write_text(
        "\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")
    s = report.stats(results)
    esc = [r for r in results if r["escalated"]]
    print(f"passed {report.count_text(s)}   ${s['cost_per_conversation']:.5f} per conversation   {s['seconds']:.1f}s")
    print(f"escalated {len(esc)} of {len(results)} runs; "
          f"{sum(1 for r in esc if r['passed'])} of those passed. Budget used: {budget.used} of the last "
          f"{len(budget.recent)} requests.")
    if not offline:
        scoreboard.record(name, 5, results, path=out_dir / "scoreboard.json")
        print("\n== Scoreboard so far (held-out split) ==")
        print(scoreboard.format_board(out_dir / "scoreboard.json"))
    return {"results": results, "budget_used": budget.used}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--second-provider", action="store_true")
    parser.add_argument("--strategy", choices=("rules", "classifier"), default="rules")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--break-question", default="leave-balance")
    a = parser.parse_args()
    main(offline=a.offline, second_provider=a.second_provider, strategy=a.strategy, runs=a.runs,
         break_question=a.break_question)
