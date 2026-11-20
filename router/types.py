"""The small data types every router part shares."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Route:
    """A routing decision for one user turn.

    tier     small, mid or large (the names in models.yaml)
    rule     which rule or step made the choice, so every decision can be explained later
    usage    tokens the router itself spent deciding (empty for rules, filled by a classifier), by model name
    seconds  time the router itself took
    """
    tier: str
    rule: str
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class Outcome:
    """What running a route produced. A cascade (Part 5) can try several models, so this keeps them all."""
    messages: list
    usage: dict = field(default_factory=dict)      # tokens by model name, across every attempt
    attempts: list = field(default_factory=list)   # one {"spec", "ok", "error"} per model tried
    escalated: bool = False
    error: str | None = None
