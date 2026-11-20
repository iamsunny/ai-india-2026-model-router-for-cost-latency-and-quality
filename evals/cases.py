"""Load the eval cases.

evals/cases.jsonl is the question set every part of the series reuses. It starts from 18
questions of Agents Part 8 and 16 cases of Agents Part 9 (duplicates removed), plus 30 new cases so
every task type has enough questions. Each case has a task type and a split:

- tune: use these to write rules and prompts and to pick thresholds.
- holdout: use these only to report results, so the numbers are not fitted to the router.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_TYPES = ("lookup", "policy", "reasoning", "action", "sensitive")
SPLITS = ("tune", "holdout")


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_cases(split: str | None = None, task_type: str | None = None) -> list[dict]:
    cases = _read(HERE / "cases.jsonl")
    if split:
        cases = [c for c in cases if c["split"] == split]
    if task_type:
        cases = [c for c in cases if c["task_type"] == task_type]
    return cases


def load_recurring() -> list[dict]:
    """The four questions from the cost table at the end of Agents Part 11."""
    return _read(HERE / "recurring.jsonl")


def count_by_type(cases: list[dict]) -> dict[str, int]:
    return {t: sum(1 for c in cases if c["task_type"] == t) for t in TASK_TYPES}


def load_conversations() -> list[dict]:
    """Sample conversations for the Part 6 replay. Each turn names a case, so quality can be checked per turn."""
    return _read(HERE / "conversations.jsonl")
