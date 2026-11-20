"""The gateway: the one function the agent calls. It picks the model, runs it and logs what happened.

Parts 3 to 6 plug into it:

- strategy  decides the tier for one user turn. Part 3: the rules router. Part 4: the classifier.
- executor  runs the route. Part 3: one model call (single_call). Part 5: retries, fallbacks, cascade.
- sink      receives one record per request. Part 6 reads these records to build the report.

The agent never chooses a model itself. That keeps routing out of the agent code.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from langchain_core.callbacks import get_usage_metadata_callback

from helpdesk.agent import build_agent
from helpdesk.llm import get_model
from router.catalog import Catalog, load_catalog
from router.cost import usage_cost
from router.types import Outcome, Route

ROUTER_ERROR_TIER = "mid"   # if a router itself breaks, fall back to the baseline instead of failing the user


class ListSink:
    """Keeps records in memory. Handy in tests."""

    def __init__(self):
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


class JsonlSink:
    """Appends one JSON line per request."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def merge_usage(*usages: dict) -> dict:
    """Add up usage dicts (tokens by model name), including the cached-token details."""
    total: dict = {}
    for usage in usages:
        for model, u in (usage or {}).items():
            t = total.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "input_token_details": {}})
            t["input_tokens"] += u.get("input_tokens", 0)
            t["output_tokens"] += u.get("output_tokens", 0)
            for key, n in (u.get("input_token_details") or {}).items():
                t["input_token_details"][key] = t["input_token_details"].get(key, 0) + n
    return total


def split_turn(messages: list) -> tuple[str, list[str]]:
    """The latest user message, and the user messages before it."""
    texts = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", None)
        if role in ("user", "human"):
            content = m["content"] if isinstance(m, dict) else m.content
            texts.append(content if isinstance(content, str) else str(content))
    if not texts:
        raise ValueError("The conversation has no user message to route")
    return texts[-1], texts[:-1]


def single_call(gateway: "Gateway", route: Route, messages: list) -> Outcome:
    """Run the route's tier once. Part 5 replaces this with an executor that can retry and escalate."""
    spec = gateway.catalog.tier_spec(route.tier, gateway.provider)
    error = None
    out_messages: list = []
    with get_usage_metadata_callback() as usage:
        try:
            out_messages = build_agent(gateway.model_factory(spec)).invoke({"messages": messages})["messages"]
        except Exception as exc:  # the caller sees the error in the record
            error = f"{type(exc).__name__}: {exc}"[:300]
    return Outcome(messages=out_messages, usage=usage.usage_metadata,
                   attempts=[{"spec": spec, "ok": error is None, "error": error}], error=error)


class Result:
    def __init__(self, messages: list, record: dict):
        self.messages = messages
        self.record = record

    @property
    def answer(self) -> str:
        return str(self.messages[-1].content) if self.messages else ""


class Gateway:
    def __init__(self, strategy: Callable[[str, list[str]], Route], executor: Callable = single_call,
                 catalog: Catalog | None = None, provider: str = "openai",
                 model_factory: Callable[[str], object] = get_model, sink=None, name: str | None = None):
        self.strategy = strategy
        self.executor = executor
        self.catalog = catalog or load_catalog()
        self.provider = provider
        self.model_factory = model_factory
        self.sink = sink if sink is not None else ListSink()
        self.name = name or getattr(strategy, "name", "gateway")

    def decide(self, messages: list) -> Route:
        latest, history = split_turn(messages)
        started = time.perf_counter()
        try:
            route = self.strategy(latest, history)
        except Exception as exc:  # a broken router must never take the helpdesk down
            return Route(ROUTER_ERROR_TIER, f"router-error:{type(exc).__name__}",
                         seconds=time.perf_counter() - started)
        if route.seconds == 0.0:
            route = Route(route.tier, route.rule, route.usage, time.perf_counter() - started)
        return route

    def invoke(self, messages: list) -> Result:
        """Route one user turn, run it, log it. `messages` is the whole conversation so far."""
        started = time.perf_counter()
        route = self.decide(messages)
        outcome = self.executor(self, route, messages)
        usage = merge_usage(route.usage, outcome.usage)
        cost = usage_cost(usage, self.catalog)
        tools = [c["name"] for m in outcome.messages if m.type == "ai" for c in m.tool_calls]
        record = {"ts": time.time(), "question": split_turn(messages)[0], "gateway": self.name,
                  "tier": route.tier, "rule": route.rule, "router_seconds": route.seconds,
                  "router_cost_usd": usage_cost(route.usage, self.catalog)["cost_usd"] if route.usage else 0.0,
                  "models": list(cost["by_model"]), "attempts": outcome.attempts,
                  "escalated": outcome.escalated, "tools": tools,
                  "input_tokens": cost["input_tokens"], "output_tokens": cost["output_tokens"],
                  "cached_tokens": cost["cached_tokens"], "cost_usd": cost["cost_usd"],
                  "seconds": time.perf_counter() - started, "error": outcome.error}
        self.sink.write(record)
        return Result(outcome.messages, record)
