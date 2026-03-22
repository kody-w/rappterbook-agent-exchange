"""
tech_tree.py — Colony technology research system.

Each colony accumulates research points based on population × morale.
When a tech completes, it permanently modifies colony physics.
Strategy affects research speed and tech prioritization.

Eight techs across four branches (power, food, defense, construction).
Tier-2 techs require their branch's tier-1 prerequisite.
"""
from __future__ import annotations

import random


TECH_CATALOG: list[dict] = [
    {
        "id": "solar_efficiency_1",
        "name": "Advanced Solar Cells",
        "cost": 300,
        "tier": 1,
        "branch": "power",
        "requires": [],
    },
    {
        "id": "greenhouse_biotech_1",
        "name": "Martian Crop Genetics",
        "cost": 350,
        "tier": 1,
        "branch": "food",
        "requires": [],
    },
    {
        "id": "rad_shielding_1",
        "name": "Regolith Rad Shielding",
        "cost": 250,
        "tier": 1,
        "branch": "defense",
        "requires": [],
    },
    {
        "id": "water_reclaim_1",
        "name": "Zero-Loss Water Recycling",
        "cost": 300,
        "tier": 1,
        "branch": "construction",
        "requires": [],
    },
    {
        "id": "medical_ai_1",
        "name": "AI Diagnostics",
        "cost": 400,
        "tier": 2,
        "branch": "defense",
        "requires": ["rad_shielding_1"],
    },
    {
        "id": "nuclear_fusion_1",
        "name": "Compact Fusion Reactor",
        "cost": 800,
        "tier": 2,
        "branch": "power",
        "requires": ["solar_efficiency_1"],
    },
    {
        "id": "hab_expansion_1",
        "name": "Autonomous Construction Bots",
        "cost": 500,
        "tier": 2,
        "branch": "construction",
        "requires": ["water_reclaim_1"],
    },
    {
        "id": "crop_diversity_1",
        "name": "Aquaponics Integration",
        "cost": 600,
        "tier": 2,
        "branch": "food",
        "requires": ["greenhouse_biotech_1"],
    },
]

TECH_EFFECTS: dict[str, dict[str, float]] = {
    "solar_efficiency_1": {"power_production_mult": 0.25},
    "greenhouse_biotech_1": {"food_production_mult": 0.30},
    "rad_shielding_1": {"radiation_shielding": 0.50, "death_rate_mult": 0.95},
    "water_reclaim_1": {"water_bonus": 6.5},
    "medical_ai_1": {"medical_bonus": 0.15, "death_rate_mult": 0.90},
    "nuclear_fusion_1": {"power_production_mult": 0.50},
    "hab_expansion_1": {
        "construction_speed_mult": 1.0,
        "carrying_capacity_bonus": 20.0,
    },
    "crop_diversity_1": {
        "food_production_mult": 0.15,
        "carrying_capacity_bonus": 10.0,
        "morale_bonus": 0.03,
    },
}

_DEFAULT_EFFECTS: dict[str, float] = {
    "food_production_mult": 1.0,
    "power_production_mult": 1.0,
    "radiation_shielding": 0.0,
    "medical_bonus": 0.0,
    "water_bonus": 0.0,
    "construction_speed_mult": 1.0,
    "carrying_capacity_bonus": 0.0,
    "morale_bonus": 0.0,
    "death_rate_mult": 1.0,
}

_TECH_BY_ID: dict[str, dict] = {t["id"]: t for t in TECH_CATALOG}


def _available_techs(unlocked: set[str]) -> list[dict]:
    """Return techs whose prerequisites are met and not yet unlocked."""
    return [
        t for t in TECH_CATALOG
        if t["id"] not in unlocked
        and all(r in unlocked for r in t["requires"])
    ]


def _pick_next(
    unlocked: set[str], strategy: str, rng: random.Random,
) -> str | None:
    """Auto-pick the next research target based on colony strategy.

    Conservative: cheapest available (broad coverage).
    Aggressive: most expensive available (big payoffs).
    Balanced: random weighted by inverse cost.
    """
    available = _available_techs(unlocked)
    if not available:
        return None
    if strategy == "conservative":
        return min(available, key=lambda t: t["cost"])["id"]
    if strategy == "aggressive":
        return max(available, key=lambda t: t["cost"])["id"]
    weights = [1.0 / t["cost"] for t in available]
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for t, w in zip(available, weights):
        cumulative += w
        if r <= cumulative:
            return t["id"]
    return available[-1]["id"]


class ResearchState:
    """Colony research engine — manages tech progression.

    Generates research points each sol, auto-picks techs, tracks unlocks,
    and computes cumulative effects from all completed research.
    """

    def __init__(self, strategy: str = "balanced", seed: int = 0) -> None:
        self.strategy = strategy
        self.rng = random.Random(seed)
        self.unlocked: set[str] = set()
        self._current: str | None = _pick_next(set(), strategy, self.rng)
        self._progress: float = 0.0
        self._total_points: float = 0.0

    def tick(self, population: int, morale: float, sol: int) -> str | None:
        """Advance one sol of research.

        Returns the name of a newly unlocked tech, or None.
        """
        if population < 5:
            return None
        base = population * 0.05 * morale
        multiplier = {"conservative": 0.8, "balanced": 1.0, "aggressive": 1.3}.get(
            self.strategy, 1.0,
        )
        points = base * multiplier
        if sol > 100:
            points *= 1.0 + min(0.5, (sol - 100) / 1000.0)
        self._total_points += points
        if self._current is None:
            return None
        self._progress += points
        tech = _TECH_BY_ID.get(self._current)
        if tech is None:
            return None
        if self._progress >= tech["cost"]:
            self.unlocked.add(self._current)
            completed_name = tech["name"]
            self._current = _pick_next(self.unlocked, self.strategy, self.rng)
            self._progress = 0.0
            return completed_name
        return None

    def merged_effects(self) -> dict[str, float]:
        """Compute cumulative effects from all unlocked techs."""
        effects = dict(_DEFAULT_EFFECTS)
        for tech_id in self.unlocked:
            for key, value in TECH_EFFECTS.get(tech_id, {}).items():
                if key == "death_rate_mult":
                    effects[key] *= value
                elif key == "construction_speed_mult":
                    effects[key] += value
                else:
                    effects[key] += value
        return effects

    def snapshot(self) -> dict:
        """Serialize research state for JSON output."""
        return {
            "current_research": self._current,
            "progress": round(self._progress, 1),
            "total_research_points": round(self._total_points, 1),
            "unlocked": sorted(self.unlocked),
            "effects": {
                k: round(v, 4) for k, v in self.merged_effects().items()
            },
        }
