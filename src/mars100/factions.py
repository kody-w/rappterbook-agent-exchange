"""
Emergent factions for Mars-100.

Detects social clusters from pairwise affinity (stat similarity + mutual
trust/affection).  Factions are semi-persistent: membership is recomputed
each year, but faction identity is preserved across ticks via overlap
matching.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import Colonist, STAT_NAMES

# --- Affinity & clustering params ---
STAT_WEIGHT = 0.6
SOCIAL_WEIGHT = 0.4
MERGE_THRESHOLD = 0.55  # affinity above this → same cluster
MIN_FACTION_SIZE = 2
OVERLAP_MATCH_THRESHOLD = 0.5  # fraction overlap to keep faction id

FACTION_ADJECTIVES = [
    "Iron", "Crimson", "Verdant", "Azure", "Amber",
    "Silent", "Radiant", "Obsidian", "Coral", "Ember",
]
FACTION_NOUNS = [
    "Pact", "Circle", "Lodge", "Accord", "Covenant",
    "Collective", "Assembly", "Hearth", "Vanguard", "Fold",
]


@dataclass
class Faction:
    """An emergent social cluster."""
    id: str
    name: str
    members: list[str]
    dominant_stat: str
    centroid: dict[str, float]
    cohesion: float
    founded_year: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "members": self.members,
            "dominant_stat": self.dominant_stat, "centroid": self.centroid,
            "cohesion": round(self.cohesion, 4), "founded_year": self.founded_year,
        }


def _stat_vector(colonist: Colonist) -> list[float]:
    """Extract normalized stat vector from a colonist."""
    return [getattr(colonist.stats, name) for name in STAT_NAMES]


def _stat_distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two stat vectors, normalized to 0-1."""
    raw = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    max_dist = math.sqrt(len(a))  # max when one is all-0, other all-1
    return raw / max_dist if max_dist > 0 else 0.0


def _stat_similarity(a: list[float], b: list[float]) -> float:
    """1 - normalized distance."""
    return 1.0 - _stat_distance(a, b)


def _social_affinity(a_id: str, b_id: str, social_graph: Any) -> float:
    """Mutual trust+affection average, 0-1."""
    rel_ab = social_graph.get(a_id, b_id)
    rel_ba = social_graph.get(b_id, a_id)
    return (rel_ab.trust + rel_ab.affection + rel_ba.trust + rel_ba.affection) / 4.0


def pairwise_affinity(a: Colonist, b: Colonist, social_graph: Any) -> float:
    """Combined stat-similarity + social-trust affinity score."""
    stat_sim = _stat_similarity(_stat_vector(a), _stat_vector(b))
    soc = _social_affinity(a.id, b.id, social_graph)
    return STAT_WEIGHT * stat_sim + SOCIAL_WEIGHT * soc


def detect_factions(
    colonists: list[Colonist],
    social_graph: Any,
    prior_factions: list[Faction] | None = None,
    year: int = 0,
    rng: random.Random | None = None,
) -> list[Faction]:
    """Detect emergent factions via greedy agglomeration.

    Each colonist starts as a singleton.  Pairs with affinity above
    MERGE_THRESHOLD are merged into the same cluster.  Clusters smaller
    than MIN_FACTION_SIZE are dissolved.
    """
    if len(colonists) < MIN_FACTION_SIZE:
        return []
    rng = rng or random.Random()

    # Union-find
    parent: dict[str, str] = {c.id: c.id for c in colonists}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Build affinity pairs and merge above threshold
    for i, a in enumerate(colonists):
        for b in colonists[i + 1:]:
            if pairwise_affinity(a, b, social_graph) >= MERGE_THRESHOLD:
                union(a.id, b.id)

    # Collect clusters
    clusters: dict[str, list[Colonist]] = {}
    for c in colonists:
        root = find(c.id)
        clusters.setdefault(root, []).append(c)

    # Filter small clusters
    clusters = {k: v for k, v in clusters.items() if len(v) >= MIN_FACTION_SIZE}

    # Build Faction objects
    raw_factions = [_build_faction(members, year, rng) for members in clusters.values()]

    # Match to prior factions by overlap
    if prior_factions:
        raw_factions = _match_to_prior(raw_factions, prior_factions)

    return raw_factions


def _build_faction(
    members: list[Colonist], year: int, rng: random.Random,
) -> Faction:
    """Build a Faction from a list of members."""
    member_ids = [c.id for c in members]
    centroid = _compute_centroid(members)
    dominant = max(centroid, key=centroid.get)  # type: ignore[arg-type]
    cohesion = _compute_cohesion(members)
    name = rng.choice(FACTION_ADJECTIVES) + " " + rng.choice(FACTION_NOUNS)
    fid = f"faction-{'-'.join(sorted(member_ids)[:3])}"
    return Faction(
        id=fid, name=name, members=member_ids,
        dominant_stat=dominant, centroid=centroid,
        cohesion=cohesion, founded_year=year,
    )


def _compute_centroid(members: list[Colonist]) -> dict[str, float]:
    """Average stat vector of faction members."""
    n = len(members)
    if n == 0:
        return {name: 0.5 for name in STAT_NAMES}
    totals = {name: 0.0 for name in STAT_NAMES}
    for c in members:
        for name in STAT_NAMES:
            totals[name] += getattr(c.stats, name)
    return {name: round(totals[name] / n, 4) for name in STAT_NAMES}


def _compute_cohesion(members: list[Colonist]) -> float:
    """Internal cohesion: 1 - mean stat distance between all pairs."""
    if len(members) < 2:
        return 1.0
    total = 0.0
    count = 0
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            total += _stat_similarity(_stat_vector(a), _stat_vector(b))
            count += 1
    return total / count if count > 0 else 1.0


def _match_to_prior(
    new_factions: list[Faction], prior: list[Faction],
) -> list[Faction]:
    """Preserve faction identity across ticks via overlap matching."""
    used: set[str] = set()
    for nf in new_factions:
        new_set = set(nf.members)
        best_id, best_name, best_year, best_overlap = None, None, None, 0.0
        for pf in prior:
            if pf.id in used:
                continue
            old_set = set(pf.members)
            overlap = len(new_set & old_set) / max(1, len(new_set | old_set))
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = pf.id
                best_name = pf.name
                best_year = pf.founded_year
        if best_overlap >= OVERLAP_MATCH_THRESHOLD and best_id:
            nf.id = best_id
            nf.name = best_name  # type: ignore[assignment]
            nf.founded_year = best_year  # type: ignore[assignment]
            used.add(best_id)
    return new_factions


def compute_faction_tensions(
    factions: list[Faction], social_graph: Any,
) -> dict[tuple[str, str], float]:
    """Compute tension between each pair of factions.

    Tension = 1 - mean mutual trust between members of the two factions.
    Higher tension → more conflict risk.
    """
    tensions: dict[tuple[str, str], float] = {}
    for i, fa in enumerate(factions):
        for fb in factions[i + 1:]:
            trust_sum = 0.0
            count = 0
            for a_id in fa.members:
                for b_id in fb.members:
                    rel_ab = social_graph.get(a_id, b_id)
                    rel_ba = social_graph.get(b_id, a_id)
                    trust_sum += (rel_ab.trust + rel_ba.trust) / 2.0
                    count += 1
            avg_trust = trust_sum / max(1, count)
            tensions[(fa.id, fb.id)] = round(1.0 - avg_trust, 4)
    return tensions


def check_soft_schism(
    factions: list[Faction],
    tensions: dict[tuple[str, str], float],
    threshold: float = 0.7,
) -> list[dict]:
    """Check for soft schism events (tension > threshold).

    Returns a list of schism event dicts (not structural splits — just
    cohesion/trust penalties that the engine can apply).
    """
    events: list[dict] = []
    for (fa_id, fb_id), tension in tensions.items():
        if tension >= threshold:
            fa = next((f for f in factions if f.id == fa_id), None)
            fb = next((f for f in factions if f.id == fb_id), None)
            if fa and fb:
                events.append({
                    "type": "soft_schism",
                    "factions": [fa_id, fb_id],
                    "faction_names": [fa.name, fb.name],
                    "tension": tension,
                    "description": (
                        f"Deep rift between {fa.name} and {fb.name} — "
                        f"trust has collapsed to {1.0 - tension:.0%}"
                    ),
                })
    return events
