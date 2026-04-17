"""
Cultural factions for Mars-100.

When colonist value clusters form, they become political blocs.
Factions emerge from convergence analysis and influence governance
voting, resource allocation preferences, and social dynamics.

Factions are DETECTED, not assigned — they emerge from stat similarity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.mars100.colonist import STAT_NAMES


@dataclass
class Faction:
    """A detected cultural faction (value cluster as political bloc)."""
    faction_id: str
    member_ids: list[str]
    centroid: dict[str, float]
    dominant_stat: str
    cohesion: float  # 0-1: how tight the cluster is
    year_formed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "faction_id": self.faction_id,
            "member_ids": self.member_ids,
            "centroid": {k: round(v, 4) for k, v in self.centroid.items()},
            "dominant_stat": self.dominant_stat,
            "cohesion": round(self.cohesion, 4),
            "year_formed": self.year_formed,
            "size": len(self.member_ids),
        }


def _compute_centroid(members: list[dict]) -> dict[str, float]:
    """Mean stat vector of a group of colonists."""
    if not members:
        return {s: 0.5 for s in STAT_NAMES}
    centroid: dict[str, float] = {}
    for stat in STAT_NAMES:
        vals = [m.get("stats", {}).get(stat, 0.5) for m in members]
        centroid[stat] = sum(vals) / len(vals)
    return centroid


def _compute_cohesion(members: list[dict], centroid: dict[str, float]) -> float:
    """Internal cohesion: 1 - normalized mean distance to centroid."""
    if len(members) < 2:
        return 1.0
    total = 0.0
    for m in members:
        dist_sq = 0.0
        for stat in STAT_NAMES:
            v = m.get("stats", {}).get(stat, 0.5)
            dist_sq += (v - centroid[stat]) ** 2
        total += math.sqrt(dist_sq)
    mean_dist = total / len(members)
    max_possible = math.sqrt(len(STAT_NAMES))  # max possible distance in unit hypercube
    return max(0.0, 1.0 - mean_dist / max_possible)


def detect_factions(
    colonist_snapshots: list[dict],
    year: int,
    threshold: float = 0.18,
    min_size: int = 2,
) -> list[Faction]:
    """Detect cultural factions from colonist value clusters.

    Uses single-linkage clustering: colonists within `threshold` Euclidean
    distance of each other form a group. Groups with >= min_size members
    become factions.
    """
    active = [c for c in colonist_snapshots
              if c.get("alive", True) and not c.get("exiled", False)]
    if len(active) < min_size:
        return []

    n = len(active)
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

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    factions: list[Faction] = []
    for idx, (_, indices) in enumerate(sorted(groups.items())):
        if len(indices) < min_size:
            continue
        members = [active[i] for i in indices]
        member_ids = [m.get("id", str(i)) for m in members]
        centroid = _compute_centroid(members)
        dominant = max(STAT_NAMES, key=lambda s: centroid[s])
        cohesion = _compute_cohesion(members, centroid)
        factions.append(Faction(
            faction_id=f"faction-y{year}-{idx}",
            member_ids=member_ids,
            centroid=centroid,
            dominant_stat=dominant,
            cohesion=cohesion,
            year_formed=year,
        ))

    return factions


def faction_vote_modifier(
    colonist_id: str,
    proposer_id: str,
    factions: list[Faction],
) -> float:
    """Compute vote bias from faction alignment.

    Same faction → in-group bonus. Different factions → out-group penalty.
    No faction membership → neutral.
    """
    voter_faction = None
    proposer_faction = None
    for f in factions:
        if colonist_id in f.member_ids:
            voter_faction = f.faction_id
        if proposer_id in f.member_ids:
            proposer_faction = f.faction_id

    if voter_faction is None or proposer_faction is None:
        return 0.0
    if voter_faction == proposer_faction:
        return 0.15
    return -0.10


def summarize_factions(factions: list[Faction]) -> dict[str, Any]:
    """Generate a summary of detected factions for the year record."""
    if not factions:
        return {"count": 0, "factions": [], "dominant_stat_distribution": {}}

    stat_dist: dict[str, int] = {}
    for f in factions:
        stat_dist[f.dominant_stat] = stat_dist.get(f.dominant_stat, 0) + 1

    return {
        "count": len(factions),
        "factions": [f.to_dict() for f in factions],
        "total_members": sum(len(f.member_ids) for f in factions),
        "dominant_stat_distribution": stat_dist,
        "avg_cohesion": round(
            sum(f.cohesion for f in factions) / len(factions), 4
        ),
    }
