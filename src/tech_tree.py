"""
tech_tree.py — Colony technology research system.

Each colony accumulates research points based on population + morale.
When a tech completes, it permanently modifies colony parameters.
Strategy affects research speed and tech prioritization.

Eight techs across four branches (power, food, defense, construction),
two tiers each. Tier-2 techs require their branch's tier-1 prerequisite.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


TECH_CATALOG: list[dict] = [
    {
        "id": "solar_efficiency_1",
        "name": "Advanced Solar Cells",
        "description": "GaAs multi-junction cells boost solar output by 25%",
        "cost": 300,
        "tier": 1,
        "branch": "power",
        "requires": [],
        "effect": {"type": "solar_efficiency", "value": 0.25},
    },
    {
        "id": "greenhouse_biotech_1",
        "name": "Martian Crop Genetics",
        "description": "CRISPR-edited crops adapted to low pressure and UV",
        "cost": 350,
        "tier": 1,
        "branch": "food",
        "requires": [],
        "effect": {"type": "greenhouse_efficiency", "value": 0.30},
    },
    {
        "id": "rad_shielding_1",
        "name": "Regolith Rad Shielding",
        "description": "Compressed regolith blocks reduce radiation exposure",
        "cost": 250,
        "tier": 1,
        "branch": "defense",
        "requires": [],
        "effect": {"type": "radiation_shielding", "value": 0.50},
    },
    {
        "id": "water_reclaim_1",
        "name": "Zero-Loss Water Recycling",
        "description": "Membrane distillation reaches 99.5% water recovery",
        "cost": 300,
        "tier": 1,
        "branch": "construction",
        "requires": [],
        "effect": {"type": "water_recycling", "value": 0.065},
    },
    {
        "id": "medical_ai_1",
        "name": "AI Diagnostics",
        "description": "ML-based early disease detection and treatment planning",
        "cost": 400,
        "tier": 2,
        "branch": "defense",
        "requires": [],
        "effect": {"type": "medical_boost", "value": 0.15},
    },
    {
        "id": "nuclear_fusion_1",
        "name": "Compact Fusion Reactor",
        "description": "Small-scale D-He3 fusion supplements solar during storms",
        "cost": 800,
        "tier": 3,
        "branch": "power",
        "requires": ["solar_efficiency_1"],
        "effect": {"type": "nuclear_power", "value": 200.0},
    },
    {
        "id": "hab_expansion_1",
        "name": "Autonomous Construction Bots",
        "description": "3D-printed habitat modules from in-situ regolith",
        "cost": 500,
        "tier": 2,
        "branch": "construction",
        "requires": [],
        "effect": {"type": "construction_speed", "value": 2.0},
    },
    {
        "id": "crop_diversity_1",
        "name": "Aquaponics Integration",
        "description": "Fish-plant symbiosis doubles protein production",
        "cost": 600,
        "tier": 2,
        "branch": "food",
        "requires": ["greenhouse_biotech_1"],
        "effect": {"type": "food_diversity", "value": 0.20},
    },
]

# Map each tech to a research branch
TECH_BRANCHES: dict[str, str] = {
    t["id"]: t["branch"] for t in TECH_CATALOG
}

# Cumulative effect contributions per tech
TECH_EFFECTS: dict[str, dict[str, float]] = {
    "solar_efficiency_1": {"power_production_mult": 0.25},
    "greenhouse_biotech_1": {"food_production_mult": 0.30},
    "rad_shielding_1": {"radiation_shielding_bonus": 0.05, "death_rate_mult": 0.95},
    "water_reclaim_1": {"water_bonus": 6.5},
    "medical_ai_1": {"medical_bonus": 0.15, "death_rate_mult": 0.90},
    "nuclear_fusion_1": {"power_production_mult": 0.50},
    "hab_expansion_1": {
        "habitat_expansion_mult": 1.0,
        "carrying_capacity_bonus": 20.0,
    },
    "crop_diversity_1": {
        "food_production_mult": 0.15,
        "carrying_capacity_bonus": 10.0,
        "morale_bonus": 0.03,
    },
}

# Default cumulative effects (no techs unlocked)
_DEFAULT_EFFECTS: dict[str, float] = {
    "food_production_mult": 1.0,
    "power_production_mult": 1.0,
    "radiation_shielding_bonus": 0.0,
    "medical_bonus": 0.0,
    "carrying_capacity_bonus": 0.0,
    "death_rate_mult": 1.0,
    "habitat_expansion_mult": 1.0,
    "morale_bonus": 0.0,
    "water_bonus": 0.0,
}


def _available_techs(completed: set[str]) -> list[dict]:
    """Return techs whose prerequisites are satisfied."""
    available = []
    for tech in TECH_CATALOG:
        if tech["id"] in completed:
            continue
        if all(r in completed for r in tech.get("requires", [])):
            available.append(tech)
    return available


def _pick_next(
    completed: set[str],
    strategy: str,
    rng: random.Random,
) -> str | None:
    """Auto-pick the next tech based on colony strategy.

    Conservative: cheapest available (lowest risk).
    Balanced: random weighted by inverse cost.
    Aggressive: most expensive available (biggest payoff).
    """
    available = _available_techs(completed)
    if not available:
        return None

    if strategy == "conservative":
        pick = min(available, key=lambda t: t["cost"])
    elif strategy == "aggressive":
        pick = max(available, key=lambda t: t["cost"])
    else:
        weights = [1.0 / t["cost"] for t in available]
        total = sum(weights)
        r = rng.random() * total
        cumulative = 0.0
        pick = available[0]
        for t, w in zip(available, weights):
            cumulative += w
            if r <= cumulative:
                pick = t
                break

    return pick["id"]


class ResearchEngine:
    """Colony research engine — manages tech progression.

    Generates research points each sol, auto-picks techs, tracks unlocks,
    and computes cumulative effects from all completed research.
    """

    def __init__(self, strategy: str = "balanced", seed: int = 0) -> None:
        self.strategy = strategy
        self.rng = random.Random(seed)
        self.unlocked: set[str] = set()
        self.branch_points: dict[str, float] = {
            "power": 0.0, "food": 0.0, "defense": 0.0, "construction": 0.0,
        }

        # Current research target
        self._current: str | None = None
        self._progress: float = 0.0
        self._total_points: float = 0.0
        self._pending: list[dict] = []

        # Auto-pick first research
        self._current = _pick_next(self.unlocked, self.strategy, self.rng)

    def generate_points(
        self, population: int, morale: float, sol: int,
    ) -> float:
        """Generate research points for this sol and apply to current tech.

        Returns points generated (0 if population < 5).
        """
        if population < 5:
            return 0.0

        base = population * 0.05 * morale
        multiplier = {
            "conservative": 0.8,
            "balanced": 1.0,
            "aggressive": 1.3,
        }.get(self.strategy, 1.0)
        points = base * multiplier

        # Mature colonies research faster
        if sol > 100:
            points *= 1.0 + min(0.5, (sol - 100) / 1000.0)

        self._total_points += points

        if self._current is None:
            return points

        self._progress += points
        tech = next(
            (t for t in TECH_CATALOG if t["id"] == self._current), None,
        )
        if tech is not None and self._progress >= tech["cost"]:
            self.unlocked.add(self._current)
            branch = TECH_BRANCHES.get(self._current, "construction")
            self.branch_points[branch] += tech["cost"]
            self._pending.append(tech)
            self._current = None
            self._progress = 0.0

        return points

    def check_unlocks(self, sol: int) -> list[dict]:
        """Return newly unlocked techs since last check and auto-pick next.

        Each returned dict is a full tech catalog entry.
        """
        newly = list(self._pending)
        self._pending.clear()

        if self._current is None:
            self._current = _pick_next(
                self.unlocked, self.strategy, self.rng,
            )

        return newly

    def cumulative_effects(self) -> dict[str, float]:
        """Compute cumulative effects from all unlocked techs."""
        effects = dict(_DEFAULT_EFFECTS)
        for tech_id in self.unlocked:
            for key, value in TECH_EFFECTS.get(tech_id, {}).items():
                if key == "death_rate_mult":
                    effects[key] *= value
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
            "branch_points": {
                k: round(v, 1) for k, v in self.branch_points.items()
            },
            "effects": {
                k: round(v, 4) for k, v in self.cumulative_effects().items()
            },
        }
