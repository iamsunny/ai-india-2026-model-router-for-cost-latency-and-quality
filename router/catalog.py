"""The price catalog: every model the router can use, read from models.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CATALOG_PATH = Path(__file__).resolve().parent.parent / "models.yaml"
TIERS = ("small", "mid", "large")


class UnknownModelError(KeyError):
    """A model name that has no price in models.yaml."""


@dataclass(frozen=True)
class ModelPrice:
    spec: str                      # what LangChain wants, for example "openai:gpt-5.4-mini"
    input: float                   # USD per 1M input tokens that are not cached
    cached_input: float            # USD per 1M input tokens read from the cache
    output: float                  # USD per 1M output tokens
    cache_write: float | None = None   # USD per 1M tokens written to a cache; None means normal input price

    @property
    def provider(self) -> str:
        return self.spec.split(":", 1)[0]

    @property
    def name(self) -> str:
        return self.spec.split(":", 1)[1]


@dataclass(frozen=True)
class Catalog:
    models: dict[str, ModelPrice]
    tiers: dict[str, dict[str, str]]
    checked: str

    def tier_spec(self, tier: str, provider: str = "openai") -> str:
        """The model string for a tier, for example tier_spec("small") is "openai:gpt-5.4-nano"."""
        try:
            return self.tiers[tier][provider]
        except KeyError:
            raise UnknownModelError(f"No {provider!r} model for tier {tier!r} in models.yaml") from None

    def price_for(self, model_name: str) -> ModelPrice:
        """Find the price for a model name as a provider reports it.

        Providers often add a date to the name, such as "gpt-5.4-mini-2026-03-17". We match the
        longest catalog name that the reported name starts with, so "gpt-5.4-mini-2026-03-17" finds
        "gpt-5.4-mini" and not "gpt-5.4".
        """
        wanted = model_name.split(":", 1)[-1]
        best = None
        for price in self.models.values():
            if wanted == price.name or wanted.startswith(price.name + "-"):
                if best is None or len(price.name) > len(best.name):
                    best = price
        if best is None:
            raise UnknownModelError(f"No price for model {model_name!r}. Add it to models.yaml.")
        return best


def load_catalog(path: Path = CATALOG_PATH) -> Catalog:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    models = {spec: ModelPrice(spec=spec, **fields) for spec, fields in raw["models"].items()}
    tiers = raw["tiers"]
    for tier, by_provider in tiers.items():
        for provider, spec in by_provider.items():
            if spec not in models:
                raise ValueError(f"Tier {tier!r} ({provider}) points to {spec!r}, which has no price")
    return Catalog(models=models, tiers=tiers, checked=str(raw["checked"]))
