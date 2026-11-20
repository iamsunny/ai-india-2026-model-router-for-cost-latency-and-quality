"""Part 3: the first router. A plain function made of ordered rules. The first rule that matches wins.

Order matters, and it is the lesson of this file:

1. Risk rules come first. They send a topic to the large tier and no cost rule can override them.
   They are about answer quality. They are NOT a security boundary: who may see what is decided
   inside the tools (Agents Part 10), not by a router.
2. Cost rules come next. Each one spots a pattern and names a task type. The tier for that task
   type comes from your own Part 2 break-even table (results/break_even.json).
3. Anything no rule recognizes goes to the mid tier, the Part 11 baseline.

A router decides one tier per user turn. It reads the latest message, plus a little recent history.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from router.types import Route

OWN_ID = "E1042"
BREAK_EVEN_PATH = Path(__file__).resolve().parent.parent / "results" / "break_even.json"

# Used only until you have run Part 2. They are guesses. Replace them with your own table.
DEFAULT_TIERS = {"lookup": "small", "action": "small", "policy": "mid", "reasoning": "mid"}


@dataclass(frozen=True)
class Rule:
    name: str
    test: Callable[[str], bool]   # gets the original text, so a rule can look at capital letters
    task_type: str                # the task type this rule believes the question is


def pattern(regex: str) -> Callable[[str], bool]:
    compiled = re.compile(regex, re.IGNORECASE)
    return lambda text: bool(compiled.search(text))


def mentions_someone_else(text: str) -> bool:
    """A request about another person's data. Looks for another employee ID, a possessive name
    ("Rohan Mehta's salary"), or a group ("what everyone on my team earns")."""
    if any(i.upper() != OWN_ID for i in re.findall(r"\bE\d{4}\b", text, re.IGNORECASE)):
        return True
    if re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+'s\b|\bof [A-Z][a-z]+ [A-Z][a-z]+\b|\b[A-Z][a-z]+ (earns|makes)\b", text):
        return True
    return bool(re.search(r"\b(everyone|colleagues?|coworkers?)\b.*\b(earn|salary|pay|paid|payslip)", text, re.IGNORECASE))


# 1. Risk rules. Quality, not security.
RISK_RULES = [
    Rule("risk:dispute", pattern(r"\b(dispute|grievance|complain\w*|harass\w*|legal|lawyer|unfair\w*|"
                                 r"discriminat\w*|wrongful\w*|underpaid|unpaid|not been paid)\b"), "sensitive"),
    Rule("risk:other-person", mentions_someone_else, "sensitive"),
]

# 2. Cost rules, one pattern each. Order matters: a ticket request that mentions a number
#    is still a ticket request, so "action" comes before "reasoning".
COST_RULES = [
    Rule("cost:action", pattern(r"\b(ask hr|tell hr|ticket|open a request|escalate|raise a request)\b|follow up with hr"),
         "action"),
    Rule("cost:reasoning", pattern(r"\b(percent(age)?|how much (less|more)|by how much|difference between|"
                                   r"change[d]? from|if i take|how many of them|take home|"
                                   r"why (was|did|is|has) my\b.*\b(lower|higher|less|more|went up|dropped))\b"),
         "reasoning"),
    Rule("cost:lookup", pattern(r"\bhow many (earned |sick |casual )?(leave )?days do i have|"
                                r"\b(earned|sick|casual|leave)\b.*\b(balance|left)\b|\bnet pay in\b|\bwhat was my (net|gross)\b|"
                                r"\bhow much (pf|income tax|tax)\b.*\bin\b|\bshow me my payslip|\bholidays? in\b"),
         "lookup"),
    Rule("cost:policy", pattern(r"\b(policy|policies|can i|carry|notice period|in advance|floating|allowed|"
                                r"eligible|combine|handbook|how long do i have|entitle\w*)\b"), "policy"),
]

SHORT_FOLLOW_UP_WORDS = 5   # "and for September?" has no topic of its own, so it borrows the previous question


def load_tiers(path: Path = BREAK_EVEN_PATH) -> tuple[dict[str, str], str]:
    """The tier for each task type, and where it came from."""
    if path.exists():
        table = json.loads(path.read_text(encoding="utf-8"))
        return {t: (tier or "large") for t, tier in table.items()}, f"your Part 2 table ({path.name})"
    return dict(DEFAULT_TIERS), "built-in defaults (run python -m scripts.check_p2_eval first)"


def check_risk(message: str, history: list[str] | None, risk_rules: list[Rule]) -> str | None:
    """The name of the risk rule that fires on the recent conversation, or None. Free: no model call."""
    recent = " ".join((history or [])[-2:] + [message])
    return next((rule.name for rule in risk_rules if rule.test(recent)), None)


class RulesRouter:
    """The rules router. Call it with the latest message and the earlier user messages."""

    name = "rules router"

    def __init__(self, tiers: dict[str, str] | None = None, default_tier: str = "mid",
                 risk_rules: list[Rule] | None = None, cost_rules: list[Rule] | None = None):
        self.tiers = dict(tiers if tiers is not None else load_tiers()[0])
        self.default_tier = default_tier
        self.risk_rules = RISK_RULES if risk_rules is None else risk_rules
        self.cost_rules = COST_RULES if cost_rules is None else cost_rules

    def detect(self, message: str, history: list[str] | None = None) -> tuple[str, str | None]:
        """Which rule fires, and the task type it believes in. ("default", None) when nothing matches."""
        history = history or []
        risk = check_risk(message, history, self.risk_rules)   # risk is judged on the recent conversation
        if risk:
            return risk, "sensitive"
        subject = message
        if len(message.split()) <= SHORT_FOLLOW_UP_WORDS and history:
            subject = f"{history[-1]} {message}"
        for rule in self.cost_rules:
            if rule.test(subject):
                return rule.name, rule.task_type
        return "default", None

    def __call__(self, message: str, history: list[str] | None = None) -> Route:
        rule, task_type = self.detect(message, history)
        if rule.startswith("risk:"):
            return Route("large", rule)
        if task_type is None:
            return Route(self.default_tier, rule)
        return Route(self.tiers.get(task_type, self.default_tier), rule)
