"""
Value convergence tracking for Mars-100.

Measures how colonist stats converge or diverge over time.
A converging colony means shared experiences are shaping shared values.
A diverging colony means individual experiences dominate.
"""
from __future__ import annotations

import math
from typing import Any


def compute_stat_std(colonists: list[dict], stat_name: str) -> float:
    """Compute standard deviation of a single stat across active colonists."""
    values = []
    for c in colonists:
        if not c.get("alive", True) or c.get("exiled", False):
            continue
        stats = c.get("stats", {})
        if isinstance(stats, dict):
            val = stats.get(stat_name)
        else:
            val = getattr(stats, stat_name, None)
        if val is not None:
            values.append(float(val))
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def compute_convergence_score(colonists: list[dict],
                              stat_names: tuple[str, ...]) -> float:
    """Compute overall convergence score (mean of all stat std devs).

    Lower score = more convergence. Range 0.0 to ~0.5 for normalized stats.
    """
    if not colonists or not stat_names:
        return 0.0
    stds = [compute_stat_std(colonists, name) for name in stat_names]
    return sum(stds) / len(stds) if stds else 0.0


def convergence_trend(scores: list[float], window: int = 10) -> str:
    """Determine convergence trend from recent scores.

    Returns 'converging', 'diverging', or 'stable'.
    """
    if len(scores) < 2:
        return "stable"
    recent = scores[-window:] if len(scores) >= window else scores
    if len(recent) < 2:
        return "stable"
    first_half = sum(recent[:len(recent) // 2]) / max(1, len(recent) // 2)
    second_half = sum(recent[len(recent) // 2:]) / max(1, len(recent) - len(recent) // 2)
    delta = second_half - first_half
    if delta < -0.01:
        return "converging"
    elif delta > 0.01:
        return "diverging"
    return "stable"


def per_stat_convergence(colonists: list[dict],
                         stat_names: tuple[str, ...]) -> dict[str, float]:
    """Get per-stat standard deviations."""
    return {name: compute_stat_std(colonists, name) for name in stat_names}


def convergence_summary(year_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize convergence data across all years."""
    if not year_scores:
        return {"trend": "stable", "initial": 0.0, "final": 0.0,
                "peak": 0.0, "trough": 0.0, "mean": 0.0}
    scores = [ys["score"] for ys in year_scores]
    return {
        "trend": convergence_trend(scores),
        "initial": scores[0],
        "final": scores[-1],
        "peak": max(scores),
        "trough": min(scores),
        "mean": sum(scores) / len(scores),
    }
