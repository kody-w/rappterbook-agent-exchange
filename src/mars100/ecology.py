"""
Ecology organ for Mars-100 (engine v10.0).

Models the sealed colony biosphere: habitat O2/CO2 balance, soil fertility,
subsurface water table, and biome progression from barren to forest.

Terraforming is the long game.  Colonies that invest early in soil
remediation and atmospheric balance reach higher biome levels, which
generate passive O2 and food bonuses.  Colonies that neglect ecology
stay on life-support — viable, but fragile under stress.

All values are normalized 0–1 (matching the resource model).
Biome level is recomputed each tick from current conditions, so
regression is possible if the colony neglects upkeep.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# ── biome thresholds (soil_quality, hab_o2, water_table minimums) ──
BIOME_THRESHOLDS: list[tuple[float, float, float]] = [
    # (soil_quality, hab_o2, water_table)
    (0.0,  0.0,  0.0),   # 0 = barren (always reachable)
    (0.15, 0.4,  0.2),   # 1 = microbial
    (0.25, 0.5,  0.3),   # 2 = lichen
    (0.40, 0.6,  0.4),   # 3 = moss
    (0.55, 0.7,  0.5),   # 4 = garden
    (0.70, 0.8,  0.6),   # 5 = forest
]

BIOME_NAMES = ("barren", "microbial", "lichen", "moss", "garden", "forest")

# passive O2 generation per biome level (per tick)
BIOME_O2_RATE = (0.0, 0.005, 0.012, 0.020, 0.030, 0.045)

# passive soil nutrient regeneration per biome level
BIOME_SOIL_REGEN = (0.0, 0.002, 0.005, 0.008, 0.012, 0.018)


@dataclass
class EcologyState:
    """Colony biosphere state — the sealed habitat environment."""
    hab_o2: float = 0.80        # habitat O2 (0.8 = comfortably breathable)
    hab_co2: float = 0.10       # habitat CO2 (0.1 = safe)
    soil_quality: float = 0.10  # soil fertility (Martian regolith is poor)
    water_table: float = 0.70   # subsurface ice availability
    biome_level: int = 0        # 0=barren → 5=forest
    terraforming_years: int = 0 # cumulative years of terraforming effort

    def to_dict(self) -> dict[str, Any]:
        return {
            "hab_o2": round(self.hab_o2, 4),
            "hab_co2": round(self.hab_co2, 4),
            "soil_quality": round(self.soil_quality, 4),
            "water_table": round(self.water_table, 4),
            "biome_level": self.biome_level,
            "biome_name": BIOME_NAMES[min(self.biome_level, len(BIOME_NAMES) - 1)],
            "terraforming_years": self.terraforming_years,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EcologyState:
        return cls(
            hab_o2=d.get("hab_o2", 0.80),
            hab_co2=d.get("hab_co2", 0.10),
            soil_quality=d.get("soil_quality", 0.10),
            water_table=d.get("water_table", 0.70),
            biome_level=d.get("biome_level", 0),
            terraforming_years=d.get("terraforming_years", 0),
        )


@dataclass
class EcologyTickResult:
    """What happened to the ecology this year."""
    o2_delta: float = 0.0
    co2_delta: float = 0.0
    soil_delta: float = 0.0
    water_delta: float = 0.0
    biome_before: int = 0
    biome_after: int = 0
    biome_changed: bool = False
    tipping_point: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "o2_delta": round(self.o2_delta, 4),
            "co2_delta": round(self.co2_delta, 4),
            "soil_delta": round(self.soil_delta, 4),
            "water_delta": round(self.water_delta, 4),
            "biome_before": self.biome_before,
            "biome_after": self.biome_after,
            "biome_changed": self.biome_changed,
        }
        if self.tipping_point:
            d["tipping_point"] = self.tipping_point
        return d


# ── core tick ──────────────────────────────────────────────────────

def tick_ecology(
    eco: EcologyState,
    actions: dict[str, str],
    population: int,
    completed_techs: list[str],
    rng: random.Random,
) -> EcologyTickResult:
    """Advance the colony biosphere by one Martian year.

    Mutates *eco* in-place and returns a result summary.
    """
    result = EcologyTickResult(biome_before=eco.biome_level)

    n_terraform = sum(1 for a in actions.values() if a == "terraform")
    n_farm = sum(1 for a in actions.values() if a == "farm")

    o2_before = eco.hab_o2
    co2_before = eco.hab_co2
    soil_before = eco.soil_quality
    water_before = eco.water_table

    # ── breathing: population consumes O2, produces CO2 ──
    breathing_load = 0.003 * population
    eco.hab_o2 -= breathing_load
    eco.hab_co2 += breathing_load * 0.8  # not all CO2 is retained

    # ── life support: power-dependent O2 regeneration ──
    # Base scrubbing keeps colony alive — consumes no resources here
    # (power cost is implicit in existing resource model)
    life_support_regen = 0.002 * population * rng.uniform(0.8, 1.2)
    eco.hab_o2 += life_support_regen
    eco.hab_co2 -= life_support_regen * 0.6

    # ── terraforming: improves soil, reduces CO2, adds O2 ──
    if n_terraform > 0:
        eco.terraforming_years += 1
        tf_effort = 0.008 * n_terraform * rng.uniform(0.7, 1.3)
        eco.soil_quality += tf_effort
        eco.hab_co2 -= 0.003 * n_terraform
        eco.hab_o2 += 0.002 * n_terraform

    # ── farming: consumes soil nutrients and water ──
    if n_farm > 0:
        farm_drain = 0.004 * n_farm * rng.uniform(0.8, 1.2)
        eco.soil_quality -= farm_drain
        eco.water_table -= 0.002 * n_farm

    # ── natural water consumption ──
    eco.water_table -= 0.001 * population

    # ── water recycling (built-in life support, like O2 regen) ──
    water_regen = 0.0008 * population * rng.uniform(0.8, 1.2)
    eco.water_table += water_regen

    # ── biome passive effects ──
    eco.hab_o2 += BIOME_O2_RATE[min(eco.biome_level, 5)]
    eco.soil_quality += BIOME_SOIL_REGEN[min(eco.biome_level, 5)]

    # ── tech bonuses ──
    if "greenhouse_dome" in completed_techs:
        eco.soil_quality += 0.003  # moisture retention
        eco.hab_o2 += 0.002       # plant O2 from dome
    if "water_recycler" in completed_techs:
        eco.water_table += 0.004  # reclaimed water reduces draw
    if "air_processor" in completed_techs:
        eco.hab_o2 += 0.005
        eco.hab_co2 -= 0.005

    # ── natural soil degradation (erosion, perchlorate leeching) ──
    eco.soil_quality -= 0.002 * rng.uniform(0.5, 1.5)
    eco.water_table -= 0.001 * rng.uniform(0.5, 1.5)

    # ── clamp all values ──
    _clamp(eco)

    # ── biome recomputation (can regress) ──
    new_biome = _compute_biome_level(eco)
    eco.biome_level = new_biome
    result.biome_after = new_biome
    result.biome_changed = new_biome != result.biome_before

    # ── tipping points ──
    if result.biome_changed:
        if new_biome > result.biome_before:
            result.tipping_point = f"biome_advance:{BIOME_NAMES[new_biome]}"
        else:
            result.tipping_point = f"biome_regress:{BIOME_NAMES[new_biome]}"
    elif eco.hab_o2 < 0.3 and o2_before >= 0.3:
        result.tipping_point = "o2_critical"
    elif eco.hab_co2 > 0.5 and co2_before <= 0.5:
        result.tipping_point = "co2_warning"

    # ── deltas for logging ──
    result.o2_delta = eco.hab_o2 - o2_before
    result.co2_delta = eco.hab_co2 - co2_before
    result.soil_delta = eco.soil_quality - soil_before
    result.water_delta = eco.water_table - water_before

    return result


# ── resource modifiers ─────────────────────────────────────────────

def compute_ecology_modifiers(eco: EcologyState) -> dict[str, float]:
    """Compute resource adjustments from ecology state.

    Returns a dict of additive adjustments to colony resources,
    keyed by resource name.  Positive = bonus, negative = penalty.
    Values are small (order 0.01–0.05) to nudge without overwhelming.
    """
    mods: dict[str, float] = {}

    # O2 level affects air resource: deficit → drain, surplus → buffer
    mods["air"] = (eco.hab_o2 - 0.5) * 0.06

    # Soil quality boosts food production
    mods["food"] = eco.soil_quality * 0.04

    # Water table supports water resource
    mods["water"] = (eco.water_table - 0.3) * 0.03

    return mods


def compute_ecology_death_rate(eco: EcologyState) -> tuple[float, str | None]:
    """Extra death rate from ecological conditions.

    Returns (rate_addition, cause_or_None).
    """
    if eco.hab_o2 < 0.3:
        rate = (0.3 - eco.hab_o2) * 0.15  # max 0.045 at o2=0
        return rate, "asphyxiation"
    if eco.hab_co2 > 0.7:
        rate = (eco.hab_co2 - 0.7) * 0.10  # max 0.03 at co2=1
        return rate, "co2_toxicity"
    return 0.0, None


# ── internal helpers ───────────────────────────────────────────────

def _compute_biome_level(eco: EcologyState) -> int:
    """Determine biome level from current ecological conditions.

    Computed fresh each tick — biome can regress if conditions degrade.
    """
    level = 0
    for i, (soil_min, o2_min, water_min) in enumerate(BIOME_THRESHOLDS):
        if (eco.soil_quality >= soil_min
                and eco.hab_o2 >= o2_min
                and eco.water_table >= water_min):
            level = i
        else:
            break
    return level


def _clamp(eco: EcologyState) -> None:
    """Clamp all ecology values to valid ranges."""
    eco.hab_o2 = max(0.0, min(1.0, eco.hab_o2))
    eco.hab_co2 = max(0.0, min(1.0, eco.hab_co2))
    eco.soil_quality = max(0.0, min(1.0, eco.soil_quality))
    eco.water_table = max(0.0, min(1.0, eco.water_table))
    eco.biome_level = max(0, min(5, eco.biome_level))
