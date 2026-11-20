"""Part 5: what happens when a routed call fails, or gives a bad answer.

Per-call timeouts and retries are already in Agents Part 11 (timeout=30, max_retries=2). This file adds
what a router needs on top:

- Fallbacks: if a model fails, try the next one in a short chain, including a second provider.
- A cascade: answer on the routed tier, run a cheap check, and move up a tier only if the check fails.
- A write guard: a cheap tier may not run a write tool. The turn moves up BEFORE the write runs.
  Once a write has run, the turn is never re-run on another model: not after a failed check, and not after
  an error either. If a write may have run and the call then failed, the outcome is unknown, so the turn
  stops with an error instead of retrying. A real backend should also take an idempotency key per request.
- An escalation budget, so a bad day cannot quietly turn every request into a large-model request.
- A hard limit on total attempts across all models.

The check looks at the answer only. It is cheap and it cannot see every wrong answer. That is the point
to remember: a cascade only catches the failures its check can see.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.tools import StructuredTool

from helpdesk.agent import TOOLS, build_agent
from helpdesk.llm import get_model
from router.classifier import TIER_ORDER, up
from router.gateway import Gateway, merge_usage
from router.types import Outcome, Route

WRITE_TOOLS = {"raise_ticket"}
REFUSAL = re.compile(r"\b(can't|cannot|can not|unable to|not able to|won't) (share|help|provide|access|do)\b", re.I)


class EscalateBeforeWrite(Exception):
    """A cheap tier tried to call a write tool. The turn moves up a tier before anything is written."""


def resilient_factory(spec: str):
    """The Part 11 settings, for any provider: a 30 second call timeout and two retries inside the call."""
    return get_model(spec, timeout=30, max_retries=2)


# ---- the check -------------------------------------------------------------------------------

def check_answer(messages: list) -> list[str]:
    """Cheap checks on a finished turn. Returns the reasons it failed, or an empty list.

    1. The answer is not empty.
    2. The answer is not raw JSON or a tool dump.
    3. If the handbook search returned sections, the answer cites one in square brackets.
    4. The model did not refuse after a tool had already returned the data.
    The model's own "low confidence" is not used. It has not been shown to track correctness.
    """
    if not messages:
        return ["no messages"]
    last = messages[-1]
    text = last.content if isinstance(last.content, str) else str(last.content)
    failures = []
    if not text.strip():
        failures.append("empty answer")
    if text.lstrip().startswith(("{", "[{")):
        failures.append("answer is raw JSON")
    calls = [c["name"] for m in messages if m.type == "ai" for c in m.tool_calls]
    found_sections = any(m.type == "tool" and '"section"' in m.content for m in messages)
    if "search_policy" in calls and found_sections and not re.search(r"\[[^\]]+\]", text):
        failures.append("handbook searched but no section cited")
    got_data = any(m.type == "tool" and '"ok": false' not in m.content.lower() and '"results": []' not in m.content
                   for m in messages)
    if got_data and REFUSAL.search(text):
        failures.append("refused although a tool returned data")
    return failures


def wrote(messages: list) -> bool:
    """Did this turn run a write tool?"""
    return any(c["name"] in WRITE_TOOLS for m in messages if m.type == "ai" for c in m.tool_calls)


def record_writes(tools: list, ran: list, write_names=WRITE_TOOLS) -> list:
    """The same tools, but a write tool notes in `ran` that it was called, before it runs.

    An exception after a write loses the conversation, so this is how the executor learns that a write may have run."""
    def noted(tool):
        def run(**kwargs):
            ran.append(tool.name)
            return tool.func(**kwargs)
        return StructuredTool.from_function(func=run, name=tool.name, description=tool.description,
                                            args_schema=tool.args_schema)
    return [noted(t) if t.name in write_names else t for t in tools]


def block_writes(tools: list, write_names=WRITE_TOOLS) -> list:
    """The same tools, except write tools raise EscalateBeforeWrite instead of running."""
    def blocked_version(tool):
        def refuse(**kwargs):
            raise EscalateBeforeWrite(tool.name)
        return StructuredTool.from_function(func=refuse, name=tool.name, description=tool.description,
                                            args_schema=tool.args_schema)
    return [blocked_version(t) if t.name in write_names else t for t in tools]


# ---- the budget ------------------------------------------------------------------------------

class EscalationBudget:
    """At most max_rate of the last `window` requests may escalate for quality reasons.

    Escalating before a write is not counted: it is needed for correctness, not a choice.
    """

    def __init__(self, max_rate: float = 0.25, window: int = 100):
        self.max_escalations = max_rate * window
        self.recent: deque[bool] = deque(maxlen=window)

    def allow(self) -> bool:
        return sum(self.recent) < self.max_escalations

    def record(self, escalated: bool) -> None:
        self.recent.append(escalated)

    @property
    def used(self) -> int:
        return sum(self.recent)


# ---- the executor ----------------------------------------------------------------------------

@dataclass
class _Attempt:
    status: str                      # "ok", "error", "error-after-write" or "blocked-write"
    messages: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    error: str | None = None


class CascadeExecutor:
    """Runs a route with fallbacks, a cascade, a write guard and a budget. Use as Gateway(executor=...)."""

    def __init__(self, second_provider: bool = False, write_tier: str = "mid", max_attempts: int = 4,
                 budget: EscalationBudget | None = None, check=check_answer, cascade: bool = True):
        self.second_provider = second_provider
        self.write_tier = write_tier
        self.max_attempts = max_attempts
        self.budget = budget if budget is not None else EscalationBudget()
        self.check = check
        self.cascade = cascade

    def chain(self, gateway: Gateway, tier: str) -> list[tuple[str, str]]:
        """Models to try for a tier, in order: the tier itself, the same tier at another provider,
        then the next tier up. Each item is (spec, tier)."""
        chain = [(gateway.catalog.tier_spec(tier, gateway.provider), tier)]
        if self.second_provider:
            other = "anthropic" if gateway.provider == "openai" else "openai"
            chain.append((gateway.catalog.tier_spec(tier, other), tier))
        if up(tier) != tier:
            chain.append((gateway.catalog.tier_spec(up(tier), gateway.provider), up(tier)))
        return chain

    def _attempt(self, gateway: Gateway, spec: str, tier: str, messages: list) -> _Attempt:
        guard = TIER_ORDER.index(tier) < TIER_ORDER.index(self.write_tier)
        ran: list = []
        tools = block_writes(TOOLS) if guard else record_writes(TOOLS, ran)
        with get_usage_metadata_callback() as usage:
            try:
                out = build_agent(gateway.model_factory(spec), tools=tools).invoke({"messages": messages})["messages"]
            except EscalateBeforeWrite:
                return _Attempt("blocked-write", usage=usage.usage_metadata)
            except Exception as exc:  # timeouts, rate limits, provider errors, bad requests
                err = f"{type(exc).__name__}: {exc}"[:200]
                if ran:   # a write may have run, so the outcome is unknown: do not try another model
                    return _Attempt("error-after-write", usage=usage.usage_metadata, error=err)
                return _Attempt("error", usage=usage.usage_metadata, error=err)
        return _Attempt("ok", out, usage.usage_metadata)

    def __call__(self, gateway: Gateway, route: Route, messages: list) -> Outcome:
        usage: dict = {}
        attempts: list[dict] = []
        tier, escalated, quality = route.tier, False, False   # quality: an escalation the budget pays for
        while True:
            result = None
            for spec, spec_tier in self.chain(gateway, tier):
                if len(attempts) >= self.max_attempts:
                    break
                a = self._attempt(gateway, spec, spec_tier, messages)
                usage = merge_usage(usage, a.usage)
                attempts.append({"spec": spec, "tier": spec_tier, "ok": a.status == "ok", "status": a.status,
                                 "error": a.error})
                if a.status == "error-after-write":
                    attempts[-1]["note"] = "a write may have run, so no other model is tried"
                    break
                if a.status != "error":
                    result, tier = a, spec_tier
                    break
            if result is None:
                self.budget.record(quality)
                return Outcome([], usage, attempts, escalated,
                               error=(attempts[-1]["error"] or "attempt limit reached") if attempts else "no attempts allowed")
            if result.status == "blocked-write":
                tier, escalated = up(tier), True     # needed for correctness: not charged to the budget
                attempts[-1]["note"] = "cheap tier tried to write, moved up before the write ran"
                continue
            problems = self.check(result.messages) if self.cascade else []
            if not problems:
                self.budget.record(quality)
                return Outcome(result.messages, usage, attempts, escalated)
            attempts[-1]["check_failed"] = problems
            if wrote(result.messages):
                attempts[-1]["note"] = "a write already ran, so this turn is not re-run"
            elif tier == "large" or len(attempts) >= self.max_attempts:
                attempts[-1]["note"] = "no higher tier or attempt left"
            elif not self.budget.allow():
                attempts[-1]["note"] = "escalation budget used up, serving this answer"
            else:
                tier, escalated, quality = up(tier), True, True
                continue
            self.budget.record(quality)
            return Outcome(result.messages, usage, attempts, escalated)


def scope_to(tools: list, user_id: str) -> list:
    """The same tools, except a call that names another employee is refused before it reaches the HRMS.

    Example of access control in the tool layer, whichever model is routed to. Nothing in the router calls
    it. To try it, pass scope_to(TOOLS, user_id) to build_agent. The measured runs did not use it. In
    production the user id comes from the authenticated session."""
    def scoped(tool):
        if "employee_id" not in tool.args:
            return tool

        def run(**kwargs):
            if kwargs["employee_id"] != user_id:
                return {"ok": False, "error": "denied by policy: not your record",
                        "hint": "Tell the employee you cannot share another person's data."}
            return tool.func(**kwargs)
        return StructuredTool.from_function(func=run, name=tool.name, description=tool.description,
                                            args_schema=tool.args_schema)
    return [scoped(t) for t in tools]
