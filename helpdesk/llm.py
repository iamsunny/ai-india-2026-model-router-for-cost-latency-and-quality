"""One place to choose the model, so the router can pick a different one per request.

Copied from Agents Part 2, with one change: get_model() can take the model string directly.
"""
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

DEFAULT_MODEL = "openai:gpt-5.4-mini"


def get_model(spec: str | None = None, **kwargs):
    """Return a chat model.

    `spec` is a LangChain model string such as "openai:gpt-5.4-nano" or "anthropic:claude-haiku-5-5".
    Without it, the MODEL value in .env is used, and then the Part 11 baseline, gpt-5.4-mini.
    """
    return init_chat_model(spec or os.getenv("MODEL", DEFAULT_MODEL), **kwargs)
