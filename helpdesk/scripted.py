"""A scripted chat model so tests and offline demos run with no API key.

Each call to the model returns the next AIMessage from the script. That lets the
tests check our wiring (tools, state, middleware) without a real LLM.
"""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def ai(content: str = "", *calls: tuple[str, dict], model_name: str = "scripted",
       tokens: tuple[int, int] = (100, 20)) -> AIMessage:
    """Build an AIMessage. ai("text") answers; ai("", ("tool", {...})) calls a tool.

    model_name is what the usage callback reports, so cost code can price it. tokens is (input, output).
    """
    return AIMessage(content=content, tool_calls=[
        {"name": n, "args": a, "id": f"call_{n}_{i}", "type": "tool_call"} for i, (n, a) in enumerate(calls)],
        usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": sum(tokens)},
        response_metadata={"model_name": model_name})


class ScriptedModel(BaseChatModel):
    script: list[AIMessage]
    seen: list = []
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        if self.i >= len(self.script):
            raise AssertionError("The scripted model ran out of replies")
        msg = self.script[self.i]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


class FailingModel(BaseChatModel):
    """A model that fails on command, for failure-injection tests: it raises `error` on every call."""
    error: Any = None

    @property
    def _llm_type(self) -> str:
        return "failing"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FailingModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise self.error or TimeoutError("the model timed out")


class ScriptedThenFailing(ScriptedModel):
    """Replays its script, then raises `error` on every later call. Used to test a failure that follows a write."""
    error: Any = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self.i >= len(self.script):
            raise self.error or TimeoutError("the model timed out")
        return super()._generate(messages, stop, run_manager, **kwargs)
