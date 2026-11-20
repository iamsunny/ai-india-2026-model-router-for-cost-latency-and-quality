"""The scoreboard: one table that grows in every part of the series.

Every row is measured on the held-out split only. The tuning split is for writing rules and prompts,
so a number from it would flatter the router. record() refuses anything else.
"""
from __future__ import annotations

import json
from pathlib import Path

from evals.report import count_text, money, stats

PATH = Path(__file__).resolve().parent.parent / "results" / "scoreboard.json"


def load(path: Path = PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def record(name: str, part: int, results: list[dict], path: Path = PATH) -> dict:
    """Add or replace a row. `results` are runs from the held-out split."""
    splits = {r["split"] for r in results}
    if splits != {"holdout"}:
        raise ValueError(f"The scoreboard takes held-out runs only, got splits {sorted(map(str, splits))}")
    s = stats(results)
    row = {"part": part, "runs": s["runs"], "passed": s["passed"],
           "cost_per_conversation": s["cost_per_conversation"], "cost_per_correct": s["cost_per_correct"],
           "seconds": s["seconds"]}
    board = load(path)
    board[name] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(board, indent=2), encoding="utf-8")
    return row


def format_board(path: Path = PATH) -> str:
    board = load(path)
    lines = [f"{'row':<32} {'part':>4} {'passed':>11} {'$/conversation':>15} {'$/correct':>11} {'seconds':>8}"]
    for name, r in board.items():
        lines.append(f"{name:<32} {r['part']:>4} {count_text(r):>11} {money(r['cost_per_conversation']):>15} "
                     f"{money(r['cost_per_correct']):>11} {r['seconds']:>8.1f}")
    return "\n".join(lines)
