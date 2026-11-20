"""Quickstart: rules routes, a canned classification and a forced fallback. No API key, no network.

    python -m scripts.quickstart

SIMULATED. Every model here is scripted (helpdesk/scripted.py). The router code is the real code.
The replies, token counts and timings are not, so this prints no costs and no pass counts.
"""
from evals import oracle
from evals.cases import load_recurring
from helpdesk import hrms
from helpdesk.scripted import FailingModel, ScriptedModel, ai
from router.classifier import ClassifierRouter, LLMClassifier
from router.gateway import Gateway
from router.resilience import CascadeExecutor
from router.rules import RulesRouter, load_tiers
from router.types import Route

NANO, MINI = "openai:gpt-5.4-nano", "openai:gpt-5.4-mini"
CANNED = '{"task_type": "policy", "difficulty": "easy", "confidence": 0.55}'


def scripted(*replies):
    return ScriptedModel(script=list(replies), seen=[])


def main() -> None:
    print("SIMULATED: scripted models, no API key, no network. Routing code is real; model replies are canned.\n")
    tiers, _ = load_tiers()          # built-in defaults when Part 2 has not run
    oracle.use_empty_handbook()      # so no embedding call is needed

    print("1. Rules route (real: plain Python, no model involved)")
    rules = RulesRouter(tiers)
    for item in load_recurring():
        r = rules(item["question"])
        print(f"   {item['question']!r} -> {r.tier} ({r.rule})")

    print("\n2. Classification (SIMULATED reply; real parsing, validation and threshold code)")
    print(f"   canned classifier reply: {CANNED}")
    q = "Can I work from home on Fridays?"
    for threshold in (0.0, 0.7):
        classify = LLMClassifier(NANO, model_factory=lambda spec: scripted(ai(CANNED, model_name="gpt-5.4-nano")))
        r = ClassifierRouter(classify, tiers, threshold=threshold)(q)
        print(f"   threshold {threshold}: {q!r} -> {r.tier} ({r.rule})")
    bad = LLMClassifier(NANO, model_factory=lambda spec: scripted(ai("I think this is a policy question")))
    r = ClassifierRouter(bad, tiers)(q)
    print(f"   reply that is not JSON -> {r.tier} ({r.rule})")

    print("\n3. Forced fallback (SIMULATED failure; real cascade code)")
    hrms.reset()
    models = {NANO: FailingModel(error=TimeoutError("injected timeout")),
              MINI: scripted(ai("", ("get_leave_balance", {"employee_id": "E1042"}), model_name="gpt-5.4-mini"),
                             ai("You have 6 sick days.", model_name="gpt-5.4-mini"))}
    gateway = Gateway(lambda m, h: Route("small", "forced"), executor=CascadeExecutor(), model_factory=models.__getitem__)
    res = gateway.invoke([{"role": "user", "content": "How many sick days do I have?"}])
    for a in res.record["attempts"]:
        print(f"   {a['spec']}: {a['status']}" + (f" ({a['error']})" if a.get("error") else ""))
    print(f"   answer (scripted text, after a real call to the fake HRMS tool): {res.answer}")
    print("\nNothing above measures a model. For measured numbers, run the check scripts with a key.")


if __name__ == "__main__":
    main()
