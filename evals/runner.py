"""Run eval cases on a model and score them.

One run = one question, one model, one fresh HR system. The result keeps what the series needs
later: did it pass, what did it cost, how long did it take. Part 2 uses this to compare tiers and
effort settings. Parts 3 to 6 reuse it to score routers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from langchain_core.callbacks import get_usage_metadata_callback

from evals.scoring import check_case
from helpdesk import hrms
from helpdesk.agent import build_agent
from helpdesk.llm import get_model
from router.catalog import Catalog, load_catalog
from router.cost import usage_cost


@dataclass(frozen=True)
class Config:
    """One thing to test: a model, and optionally a reasoning effort (none, low, medium, high)."""
    spec: str
    effort: str | None = None

    @property
    def label(self) -> str:
        name = self.spec.split(":", 1)[-1]
        return f"{name} (effort {self.effort})" if self.effort else name


ModelFactory = Callable[[Config, dict], object]


def default_factory(config: Config, case: dict | None = None):
    """A real chat model (the case is ignored; offline factories use it).

    Effort "none" works on the Chat Completions API. Effort low, medium and high do not: OpenAI refuses function
    tools with reasoning there ("use /v1/responses"). So those three go through the Responses API.
    """
    if config.effort in ("low", "medium", "high"):
        return get_model(config.spec, use_responses_api=True, reasoning={"effort": config.effort})
    kwargs = {"reasoning_effort": config.effort} if config.effort else {}
    return get_model(config.spec, **kwargs)


def run_case(case: dict, config: Config, factory: ModelFactory = default_factory,
             catalog: Catalog | None = None, run: int = 0) -> dict:
    """Run one case once. A model error counts as a failed run, so one bad call cannot stop a long sweep."""
    catalog = catalog or load_catalog()
    hrms.reset()
    started = time.perf_counter()
    error = None
    messages = []
    usage_by_model: dict = {}
    try:
        agent = build_agent(factory(config, case))
        with get_usage_metadata_callback() as usage:
            out = agent.invoke({"messages": [{"role": "user", "content": case["question"]}]})
        messages = out["messages"]
        usage_by_model = usage.usage_metadata
    except Exception as exc:  # a timeout or rate limit is a failed run, not a crashed sweep
        error = f"{type(exc).__name__}: {exc}"[:300]
    seconds = time.perf_counter() - started
    cost = usage_cost(usage_by_model, catalog)
    if error:
        verdict = {"id": case["id"], "task_type": case["task_type"], "calls": [],
                   "answer_ok": False, "tools_ok": False, "passed": False}
    else:
        verdict = check_case(case, messages)
    return {**verdict, "config": config.label, "spec": config.spec, "effort": config.effort,
            "split": case.get("split"), "run": run, "error": error,
            "input_tokens": cost["input_tokens"], "output_tokens": cost["output_tokens"],
            "cached_tokens": cost["cached_tokens"], "cost_usd": cost["cost_usd"], "seconds": seconds,
            "answer": messages[-1].text if messages else ""}


def run_suite(configs: list[Config], cases: list[dict], runs: int = 3,
              factory: ModelFactory = default_factory, catalog: Catalog | None = None,
              progress: Callable[[str], None] | None = None) -> list[dict]:
    """Run every case `runs` times on every config. Returns one result per run."""
    catalog = catalog or load_catalog()
    results = []
    for config in configs:
        if progress:
            progress(f"{config.label}: {len(cases) * runs} runs")
        for case in cases:
            for i in range(runs):
                results.append(run_case(case, config, factory, catalog, run=i))
    return results
