"""
tech_tree.py — Technology research engine for Mars colonies.

Colonies generate research points each sol and unlock permanent
upgrades. Strategy determines which techs get prioritized.

8 techs across 4 branches. Each unlock permanently modifies
colony parameters (solar output, food yield, mortality, etc.).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


TECH_CATALOG: list[dict] = [
    {
        "name": "Advanced Solar Cells",
        "branch": "power",
        "cost": 500,
        "effect": "solar_boost",
        "value": 0.25,
        "description": "+25% solar panel output",
    },
    {
        "name": "Compact Fusion Reactor",
        "branch": "power",
        "cost": 1200,
        "effect": "nuclear_boost",
        "value": 200.0,
        "description": "+200 kWh nuclear backup",
    },
    {
        "name": "Martian Crop Genetics",
        "branch": "food",
        "cost": 600,
        "effect": "greenhouse_boost",
        "value": 0.30,
        "description": "+30% greenhouse yield",
    },
    {
        "name": "Aquaponics Integration",
        "branch": "food",
        "cost": 1000,
        "effect": "aquaponics",
        "value": 0.15,
        "description": "+15% food, +10 K, morale boost",
    },
    {
        "name": "Regolith Rad Shielding",
        "branch": "defense",
        "cost": 700,
        "effect": "rad_shielding",
        "value": 0.05,
        "description": "+5% radiation shielding",
    },
    {
        "name": "AI Diagnostics",
        "branch": "defense",
        "cost": 900,
        "effect": "mortality_reduction",
        "value": 0.10,
        "description": "-10% death rate",
    },
    {
        "name": "Zero-Loss Water Recycling",
        "branch": "water",
        "cost": 800,
        "effect": "water_efficiency",
        "value": 0.20,
        "description": "+20% water recycling efficiency",
    },
    {
        "name": "Autonomous Construction Bots",
        "branch": "construction",
        "cost": 1500,
        "effect": "construction_bots",
        "value": 2.0,
        "description": "x2 expansion speed, +20 K capacity",
    },
]

STRATEGY_RESEARCH_WEIGHT = {
    "conservative": 0.8,
    "balanced": 1.0,
    "aggressive": 1.3,
}


@dataclass
class TechUnlock:
    """Record of a single technology unlock."""
    name: str
    branch: str
    sol: int
    effect: str
    value: float


@dataclass
class ResearchEngine:
    """Per-colony research state machine.

    Accumulates research points each sol, unlocks techs when thresholds
    are reached. Strategy determines selection order.
    """
    strategy: str
    rng: random.Random
    research_points: float = 0.0
    unlocked: list[TechUnlock] = field(default_factory=list)

    @property
    def unlocked_names(self) -> set[str]:
        """Set of unlocked tech names."""
        return {t.name for t in self.unlocked}

    def available_techs(self) -> list[dict]:
        """Techs not yet unlocked."""
        names = self.unlocked_names
        return [t for t in TECH_CATALOG if t["name"] not in names]

    def generate_points(self, population: int, morale: float) -> float:
        """Research points for this sol."""
        weight = STRATEGY_RESEARCH_WEIGHT.get(self.strategy, 1.0)
        return population * morale * weight * 0.1

    def _select_tech(self, available: list[dict]) -> dict | None:
        """Choose next tech to unlock based on strategy."""
        if not available:
            return None
        affordable = [t for t in available if t["cost"] <= self.research_points]
        if not affordable:
            return None
        if self.strategy == "conservative":
            return min(affordable, key=lambda t: t["cost"])
        elif self.strategy == "aggressive":
            return max(affordable, key=lambda t: t["cost"])
        else:
            weights = [1.0 / (abs(t["cost"] - 800) + 100) for t in affordable]
            total = sum(weights)
            r = self.rng.random() * total
            cumulative = 0.0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    return affordable[i]
            return affordable[-1]

    def tick(self, population: int, morale: float, sol: int) -> TechUnlock | None:
        """Advance one sol of research. Returns unlock if one happened."""
        if population < 5:
            return None
        rp = self.generate_points(population, morale)
        self.research_points += rp
        available = self.available_techs()
        tech = self._select_tech(available)
        if tech is None:
            return None
        self.research_points -= tech["cost"]
        unlock = TechUnlock(
            name=tech["name"],
            branch=tech["branch"],
            sol=sol,
            effect=tech["effect"],
            value=tech["value"],
        )
        self.unlocked.append(unlock)
        return unlock

    def get_modifier(self, effect_name: str) -> float:
        """Sum of all unlocked modifier values for a given effect."""
        return sum(t.value for t in self.unlocked if t.effect == effect_name)

    def has_tech(self, tech_name: str) -> bool:
        """Check if a specific tech is unlocked."""
        return tech_name in self.unlocked_names

    def snapshot(self) -> dict:
        """Serializable snapshot of research state."""
        return {
            "research_points": round(self.research_points, 1),
            "unlocked_count": len(self.unlocked),
            "unlocked": [
                {"name": t.name, "branch": t.branch, "sol": t.sol}
                for t in self.unlocked
            ],
        }
