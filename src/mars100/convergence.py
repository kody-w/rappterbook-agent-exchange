"""
Value convergence analysis for Mars-100.

Tracks how colonist personality traits drift over time. Uses variance and
mean pairwise distance rather than entropy (more stable for small populations).
Designed as a derived analysis — consumes yearly colonist snapshots, no tick-time state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import STAT_NAMES


@dataclass
class ConvergenceSnapshot:
    """Per-year convergence metrics."""
    year: int
    stat_variances: dict[str, float]
    mean_pairwise_distance: float
    cluster_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "year": self.year,
            "stat_variances": self.stat_variances,
            "mean_pairwise_distance": round(self.mean_pairwise_distance, 4),
            "cluster_count": self.cluster_count,
        }


def compute_stat_variances(colonist_snapshots: list[dict]) -> dict[str, float]:
    """Compute variance of each stat across active colonists."""
    active = [c for c in colonist_snapshots if c.get("alive") and not c.get("exiled")]
    if len(active) < 2:
        return {name: 0.0 for name in STAT_NAMES}
    result: dict[str, float] = {}
    for stat in STAT_NAMES:
        values = [c["stats"][stat] for c in active if stat in c.get("stats", {})]
        if len(values) < 2:
            result[stat] = 0.0
            continue
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        result[stat] = round(variance, 6)
    return result


def compute_pairwise_distance(colonist_snapshots: list[dict]) -> float:
    """Mean Euclidean distance between all pairs of active colonists in stat-space."""
    active = [c for c in colonist_snapshots if c.get("alive") and not c.get("exiled")]
    if len(active) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i, a in enumerate(active):
        for b in active[i + 1:]:
            dist_sq = 0.0
            for stat in STAT_NAMES:
                va = a.get("stats", {}).get(stat, 0.5)
                vb = b.get("stats", {}).get(stat, 0.5)
                dist_sq += (va - vb) ** 2
            total += math.sqrt(dist_sq)
            count += 1
    return total / max(1, count)


def detect_clusters(colonist_snapshots: list[dict],
                    threshold: float = 0.15) -> int:
    """Count value clusters using simple single-linkage distance threshold."""
    active = [c for c in colonist_snapshots if c.get("alive") and not c.get("exiled")]
    if len(active) < 2:
        return len(active)
    # Union-find
    parent: dict[int, int] = {i: i for i in range(len(active))}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            dist_sq = 0.0
            for stat in STAT_NAMES:
                vi = active[i].get("stats", {}).get(stat, 0.5)
                vj = active[j].get("stats", {}).get(stat, 0.5)
                dist_sq += (vi - vj) ** 2
            if math.sqrt(dist_sq) < threshold:
                union(i, j)
    return len(set(find(i) for i in range(len(active))))


def analyze_year(year: int, colonist_snapshots: list[dict]) -> ConvergenceSnapshot:
    """Produce a convergence snapshot for one year."""
    variances = compute_stat_variances(colonist_snapshots)
    distance = compute_pairwise_distance(colonist_snapshots)
    clusters = detect_clusters(colonist_snapshots)
    return ConvergenceSnapshot(
        year=year, stat_variances=variances,
        mean_pairwise_distance=distance, cluster_count=clusters,
    )


def convergence_trend(snapshots: list[ConvergenceSnapshot]) -> str:
    """Determine overall trend: 'converging', 'diverging', or 'stable'."""
    if len(snapshots) < 10:
        return "insufficient_data"
    recent = snapshots[-10:]
    early = snapshots[:10]
    early_dist = sum(s.mean_pairwise_distance for s in early) / len(early)
    recent_dist = sum(s.mean_pairwise_distance for s in recent) / len(recent)
    if early_dist == 0:
        return "stable"
    ratio = recent_dist / early_dist
    if ratio < 0.8:
        return "converging"
    if ratio > 1.2:
        return "diverging"
    return "stable"
