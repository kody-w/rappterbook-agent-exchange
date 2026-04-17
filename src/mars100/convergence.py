"""
Value convergence analysis for Mars-100.

Tracks whether colonist stat vectors converge (cultural unity),
diverge (cultural schism), or cluster (factions). Runs per-year
over the colonist_snapshots produced by each tick.
"""
from __future__ import annotations

import math
from typing import Any

from src.mars100.colonist import STAT_NAMES

CONVERGENCE_WINDOW = 5  # years to look back for trend


def compute_stat_variances(snapshots: list[dict]) -> dict[str, float]:
    """Compute variance of each stat across active colonists."""
    active = [c for c in snapshots if c.get("alive", True) and not c.get("exiled", False)]
    if len(active) < 2:
        return {s: 0.0 for s in STAT_NAMES}
    variances: dict[str, float] = {}
    for stat in STAT_NAMES:
        vals = [c.get("stats", {}).get(stat, 0.5) for c in active]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        variances[stat] = round(var, 6)
    return variances


def pairwise_distances(snapshots: list[dict]) -> list[float]:
    """Compute all pairwise Euclidean distances between active colonist stat vectors."""
    active = [c for c in snapshots if c.get("alive", True) and not c.get("exiled", False)]
    distances: list[float] = []
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            dist_sq = 0.0
            for stat in STAT_NAMES:
                vi = active[i].get("stats", {}).get(stat, 0.5)
                vj = active[j].get("stats", {}).get(stat, 0.5)
                dist_sq += (vi - vj) ** 2
            distances.append(math.sqrt(dist_sq))
    return distances


def detect_clusters(snapshots: list[dict], threshold: float = 0.18) -> list[list[str]]:
    """Single-linkage clustering of colonists by stat distance."""
    active = [c for c in snapshots if c.get("alive", True) and not c.get("exiled", False)]
    n = len(active)
    if n < 2:
        return [[c.get("id", str(i))] for i, c in enumerate(active)]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = 0.0
            for stat in STAT_NAMES:
                vi = active[i].get("stats", {}).get(stat, 0.5)
                vj = active[j].get("stats", {}).get(stat, 0.5)
                dist_sq += (vi - vj) ** 2
            if math.sqrt(dist_sq) < threshold:
                union(i, j)

    groups: dict[int, list[str]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(active[i].get("id", str(i)))
    return list(groups.values())


def convergence_trend(variance_history: list[dict[str, float]]) -> str:
    """Classify trend from recent variance history: converging, diverging, or stable."""
    window = variance_history[-CONVERGENCE_WINDOW:]
    if len(window) < 3:
        return "insufficient_data"
    avg_vars = [sum(v.values()) / max(len(v), 1) for v in window]
    deltas = [avg_vars[i + 1] - avg_vars[i] for i in range(len(avg_vars) - 1)]
    mean_delta = sum(deltas) / len(deltas)
    if mean_delta < -0.001:
        return "converging"
    if mean_delta > 0.001:
        return "diverging"
    return "stable"


def analyze_year(
    colonist_snapshots: list[dict],
    variance_history: list[dict[str, float]],
) -> dict[str, Any]:
    """Run full convergence analysis for one year."""
    variances = compute_stat_variances(colonist_snapshots)
    distances = pairwise_distances(colonist_snapshots)
    clusters = detect_clusters(colonist_snapshots)
    history = variance_history + [variances]
    trend = convergence_trend(history)
    return {
        "stat_variances": variances,
        "mean_distance": round(sum(distances) / max(len(distances), 1), 6),
        "cluster_count": len(clusters),
        "clusters": clusters,
        "trend": trend,
    }
