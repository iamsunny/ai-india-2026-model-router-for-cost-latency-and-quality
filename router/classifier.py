"""Part 4: let a small model read the question and name its task type, difficulty and confidence.

The classifier is one extra model call before the real one, so it costs tokens and time on every
request. Three guards keep it honest:

- Risk rules still run first, in front of the classifier. A model should not decide what a policy
  already decided, and a risk question never pays for a classifier call.
- The output format is strict. Anything that does not validate falls back to the mid tier.
- A confidence threshold on the model's self-reported confidence (not a calibrated probability).
  Below it, the route moves up one tier. At 0.0 it never does.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

from langchain_core.callbacks import get_usage_metadata_callback
from pydantic import BaseModel, Field, ValidationError

from helpdesk.llm import get_model
from router.rules import RISK_RULES, Rule, check_risk, load_tiers
from router.types import Route

TIER_ORDER = ("small", "mid", "large")
DEFAULT_THRESHOLD = 0.7
SHORT_FOLLOW_UP_WORDS = 5

CLASSIFIER_PROMPT = """You sort questions sent to an HR helpdesk. Reply with JSON only, no other text:
{"task_type": "...", "difficulty": "...", "confidence": 0.0}

task_type:
- lookup: read one fact from the employee's own records (leave balance, a payslip figure, a holiday date)
- policy: a rule from the company handbook
- reasoning: needs a calculation or comparing several numbers
- action: asks HR to follow up, raise a ticket or change something
- sensitive: about another person's data, a pay dispute, a complaint or a legal matter

difficulty: easy, medium or hard.
confidence: your confidence in task_type, from 0 to 1."""


class ClassifierOutput(BaseModel):
    task_type: Literal["lookup", "policy", "reasoning", "action", "sensitive"]
    difficulty: Literal["easy", "medium", "hard"]
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass
class Classification:
    valid: bool
    task_type: str | None = None
    difficulty: str | None = None
    confidence: float = 0.0
    usage: dict = field(default_factory=dict)   # what the classifier call itself cost, by model name
    seconds: float = 0.0
    error: str | None = None


def up(tier: str) -> str:
    """One tier more capable, capped at large."""
    return TIER_ORDER[min(TIER_ORDER.index(tier) + 1, len(TIER_ORDER) - 1)]


def tier_for(c: Classification, tiers: dict[str, str], threshold: float, default_tier: str = "mid") -> tuple[str, str]:
    """Turn a classification into (tier, rule name). Pure, so a threshold sweep needs no model calls."""
    if not c.valid:
        return default_tier, "classifier:invalid"
    if c.task_type == "sensitive":
        return "large", "classifier:sensitive"
    tier = tiers.get(c.task_type, default_tier)
    rule = f"classifier:{c.task_type}"
    if c.difficulty == "hard":
        tier, rule = up(tier), rule + "+hard"
    if c.confidence < threshold:
        tier, rule = up(tier), rule + "+low-confidence"
    return tier, rule


def parse_output(text: str) -> ClassifierOutput:
    """Parse the model's reply. Accepts a JSON object, with or without a code fence, and nothing looser."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return ClassifierOutput.model_validate(json.loads(cleaned))


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


class LLMClassifier:
    """Asks a small model to classify one question. Returns a Classification, never raises."""

    def __init__(self, spec: str, model_factory: Callable[[str], object] = get_model):
        self.spec = spec
        self.model_factory = model_factory

    def __call__(self, message: str, history: list[str] | None = None) -> Classification:
        history = history or []
        question = message
        if len(message.split()) <= SHORT_FOLLOW_UP_WORDS and history:
            question = f"Earlier question: {history[-1]}\nFollow-up: {message}"
        started = time.perf_counter()
        usage: dict = {}
        try:
            with get_usage_metadata_callback() as cb:
                reply = self.model_factory(self.spec).invoke(
                    [{"role": "system", "content": CLASSIFIER_PROMPT}, {"role": "user", "content": question}])
            usage = cb.usage_metadata
            out = parse_output(_text(reply.content))
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            return Classification(False, usage=usage, seconds=time.perf_counter() - started,
                                  error=f"invalid output: {type(exc).__name__}")
        except Exception as exc:  # a timeout or rate limit: the router falls back, the user is not affected
            return Classification(False, usage=usage, seconds=time.perf_counter() - started,
                                  error=f"{type(exc).__name__}: {exc}"[:200])
        return Classification(True, out.task_type, out.difficulty, out.confidence, usage,
                              time.perf_counter() - started)


class ClassifierRouter:
    name = "classifier router"

    def __init__(self, classify: Callable[[str, list[str]], Classification], tiers: dict[str, str] | None = None,
                 threshold: float = DEFAULT_THRESHOLD, default_tier: str = "mid",
                 risk_rules: list[Rule] | None = None, name: str | None = None):
        self.classify = classify
        self.tiers = dict(tiers if tiers is not None else load_tiers()[0])
        self.threshold = threshold
        self.default_tier = default_tier
        self.risk_rules = RISK_RULES if risk_rules is None else risk_rules
        self.name = name or self.name

    def __call__(self, message: str, history: list[str] | None = None) -> Route:
        risk = check_risk(message, history, self.risk_rules)
        if risk:
            return Route("large", risk)            # no classifier call, so no overhead
        c = self.classify(message, history or [])
        tier, rule = tier_for(c, self.tiers, self.threshold, self.default_tier)
        return Route(tier, rule, usage=c.usage, seconds=c.seconds)
