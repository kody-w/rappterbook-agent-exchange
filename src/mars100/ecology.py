"""
Ecology organ for Mars-100 colony simulation (engine v10.0).

Models the Martian biosphere: atmosphere composition, soil remediation,
water cycle, and flora coverage.  Terraforming is a slow, costly process
that takes decades to produce meaningful returns.

Design constraints (from rubber-duck review):
  - O2 + CO2 constrained to sum <= 1.0 (remainder is N2/Ar)
  - terraforming_progress is DERIVED, never set independently
  - biome_level is DERIVED from terraforming_progress thresholds
  - downstream effects use maintenance/spoilage modifiers only (one-year lag)
  - ecological costs (power, water upkeep) prevent free-lunch dynamics
  - benefits ramp slowly; storms damage flora proportional to coverage
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# -- biome thresholds --------------------------------------------------------

BIOME_THRESHOLDS: list[tuple[float, str]] = [
    (0.85, "earthlike"),
    (0.70, "forest"),
    (0.50, "garden"),
    (0.30, "greenhouse"),
    (0.15, "pioneer"),
    (0.00, "barren"),
]


def biome_from_progress(progress: float) -> str:
    """Determine biome level from terraforming progress."""
    for threshold, name in BIOME_THRESHOLDS:
        if progress >= threshold:
            return name
    return "barren"


# -- atmosphere --------------------------------------------------------------

@dataclass
class Atmosphere:
    """Martian atmosphere composition (fractions 0-1).

    Real Mars: ~95% CO2, ~2.7% N2, ~0.13% O2, ~600 Pa.
    O2 + CO2 are tracked; remainder is implicitly N2/Ar.
    Constraint: o2 + co2 <= 1.0 always.
    """
    o2: float = 0.002
    co2: float = 0.95
    pressure: float = 0.01  # normalized: 0 = vacuum, 1 = Earth sea level

    def clamp(self) -> None:
        self.o2 = max(0.0, min(1.0, self.o2))
        self.co2 = max(0.0, min(1.0, self.co2))
        self.pressure = max(0.0, min(1.0, self.pressure))
        total = self.o2 + self.co2
        if total > 1.0:
            scale = 1.0 / total
            self.o2 *= scale
            self.co2 *= scale

    def to_dict(self) -> dict[str, float]:
        return {"o2": round(self.o2, 6), "co2": round(self.co2, 6),
                "pressure": round(self.pressure, 6)}

    @classmethod
    def from_dict(cls, d: dict) -> Atmosphere:
        return cls(o2=d.get("o2", 0.002), co2=d.get("co2", 0.95),
                   pressure=d.get("pressure", 0.01))


# -- soil --------------------------------------------------------------------

@dataclass
class SoilState:
    """Martian soil (regolith) state.

    Perchlorates start high (~0.8) and decrease through active remediation.
    Nutrients start near zero and build slowly with composting/amendments.
    """
    nutrient_level: float = 0.02
    perchlorate: float = 0.80

    def clamp(self) -> None:
        self.nutrient_level = max(0.0, min(1.0, self.nutrient_level))
        self.perchlorate = max(0.0, min(1.0, self.perchlorate))

    def to_dict(self) -> dict[str, float]:
        return {"nutrient_level": round(self.nutrient_level, 6),
                "perchlorate": round(self.perchlorate, 6)}

    @classmethod
    def from_dict(cls, d: dict) -> SoilState:
        return cls(nutrient_level=d.get("nutrient_level", 0.02),
                   perchlorate=d.get("perchlorate", 0.80))


# -- water cycle -------------------------------------------------------------

@dataclass
class WaterCycle:
    """Water cycle model.

    ice_reserves: remaining accessible ice (depletes with mining).
    recycling_efficiency: closed-loop recovery (improves with tech).
    aquifer_discovered: one-time event unlocking deeper water.
    """
    ice_reserves: float = 0.60
    recycling_efficiency: float = 0.30
    aquifer_discovered: bool = False

    def clamp(self) -> None:
        self.ice_reserves = max(0.0, min(1.0, self.ice_reserves))
        self.recycling_efficiency = max(0.0, min(1.0, self.recycling_efficiency))

    def to_dict(self) -> dict:
        return {"ice_reserves": round(self.ice_reserves, 6),
                "recycling_efficiency": round(self.recycling_efficiency, 6),
                "aquifer_discovered": self.aquifer_discovered}

    @classmethod
    def from_dict(cls, d: dict) -> WaterCycle:
        return cls(ice_reserves=d.get("ice_reserves", 0.60),
                   recycling_efficiency=d.get("recycling_efficiency", 0.30),
                   aquifer_discovered=d.get("aquifer_discovered", False))


# -- flora -------------------------------------------------------------------

@dataclass
class Flora:
    """Surface and enclosed vegetation state.

    coverage: fraction of usable land with living plants (0-1).
    biodiversity: species diversity index (0-1).
    crop_health: health of food-producing crops (0-1).
    """
    coverage: float = 0.0
    biodiversity: float = 0.0
    crop_health: float = 0.30

    def clamp(self) -> None:
        self.coverage = max(0.0, min(1.0, self.coverage))
        self.biodiversity = max(0.0, min(1.0, self.biodiversity))
        self.crop_health = max(0.0, min(1.0, self.crop_health))

    def to_dict(self) -> dict[str, float]:
        return {"coverage": round(self.coverage, 6),
                "biodiversity": round(self.biodiversity, 6),
                "crop_health": round(self.crop_health, 6)}

    @classmethod
    def from_dict(cls, d: dict) -> Flora:
        return cls(coverage=d.get("coverage", 0.0),
                   biodiversity=d.get("biodiversity", 0.0),
                   crop_health=d.get("crop_health", 0.30))


# -- biosphere (composite) --------------------------------------------------

@dataclass
class Biosphere:
    """Complete biosphere state. terraforming_progress and biome are derived."""
    atmosphere: Atmosphere = field(default_factory=Atmosphere)
    soil: SoilState = field(default_factory=SoilState)
    water_cycle: WaterCycle = field(default_factory=WaterCycle)
    flora: Flora = field(default_factory=Flora)

    @property
    def terraforming_progress(self) -> float:
        """Derived 0-1 score summarising overall biosphere health."""
        atmo_score = (self.atmosphere.o2 * 0.6
                      + self.atmosphere.pressure * 0.4)
        soil_score = (self.soil.nutrient_level * 0.6
                      + (1.0 - self.soil.perchlorate) * 0.4)
        water_score = (self.water_cycle.recycling_efficiency * 0.5
                       + self.water_cycle.ice_reserves * 0.3
                       + (0.2 if self.water_cycle.aquifer_discovered else 0.0))
        flora_score = (self.flora.coverage * 0.5
                       + self.flora.biodiversity * 0.3
                       + self.flora.crop_health * 0.2)
        return (atmo_score * 0.25 + soil_score * 0.25
                + water_score * 0.25 + flora_score * 0.25)

    @property
    def biome(self) -> str:
        """Derived biome level from terraforming progress thresholds."""
        return biome_from_progress(self.terraforming_progress)

    def clamp(self) -> None:
        self.atmosphere.clamp()
        self.soil.clamp()
        self.water_cycle.clamp()
        self.flora.clamp()

    def to_dict(self) -> dict:
        return {
            "atmosphere": self.atmosphere.to_dict(),
            "soil": self.soil.to_dict(),
            "water_cycle": self.water_cycle.to_dict(),
            "flora": self.flora.to_dict(),
            "terraforming_progress": round(self.terraforming_progress, 6),
            "biome": self.biome,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Biosphere:
        return cls(
            atmosphere=Atmosphere.from_dict(d.get("atmosphere", {})),
            soil=SoilState.from_dict(d.get("soil", {})),
            water_cycle=WaterCycle.from_dict(d.get("water_cycle", {})),
            flora=Flora.from_dict(d.get("flora", {})),
        )


# -- tick result -------------------------------------------------------------

@dataclass
class EcologyResult:
    """Result of one year's ecological tick."""
    biome_before: str
    biome_after: str
    terraforming_before: float
    terraforming_after: float
    tipping_point_triggered: bool = False
    flora_damage: float = 0.0
    aquifer_event: bool = False

    def to_dict(self) -> dict:
        return {
            "biome_before": self.biome_before,
            "biome_after": self.biome_after,
            "terraforming_before": round(self.terraforming_before, 6),
            "terraforming_after": round(self.terraforming_after, 6),
            "tipping_point_triggered": self.tipping_point_triggered,
            "flora_damage": round(self.flora_damage, 6),
            "aquifer_event": self.aquifer_event,
        }


# -- ecological constants ----------------------------------------------------

TERRAFORM_O2_GAIN = 0.003
TERRAFORM_CO2_LOSS = 0.004
TERRAFORM_PERCHLORATE_LOSS = 0.005
TERRAFORM_PRESSURE_GAIN = 0.001

FARM_COVERAGE_GAIN = 0.008
FARM_CROP_HEALTH_GAIN = 0.010
FARM_NUTRIENT_GAIN = 0.003

FLORA_DECAY_RATE = 0.02
CROP_DECAY_RATE = 0.03
NUTRIENT_DECAY_RATE = 0.005
ICE_DEPLETION_RATE = 0.003

FLORA_TIPPING_THRESHOLD = 0.50
FLORA_TIPPING_O2_BONUS = 0.005
FLORA_TIPPING_NUTRIENT_BONUS = 0.004

STORM_FLORA_DAMAGE_FRACTION = 0.15
STORM_CROP_DAMAGE_FRACTION = 0.10

TECH_ECOLOGY_EFFECTS: dict[str, dict[str, float]] = {
    "greenhouse_dome": {"crop_health_bonus": 0.01, "coverage_bonus": 0.005},
    "water_recycler": {"recycling_bonus": 0.01, "ice_conservation": 0.002},
}

BASE_ECOLOGY_UPKEEP: dict[str, float] = {
    "power": 0.005,
    "water": 0.003,
}


# -- tick function -----------------------------------------------------------

def tick_ecology(
    biosphere: Biosphere,
    year: int,
    action_counts: dict[str, int],
    event_names: list[str],
    event_severities: list[float],
    infra_completed: list[str],
    rng: random.Random,
) -> EcologyResult:
    """Advance the biosphere by one Martian year."""
    biome_before = biosphere.biome
    tf_before = biosphere.terraforming_progress
    result = EcologyResult(
        biome_before=biome_before,
        biome_after=biome_before,
        terraforming_before=tf_before,
        terraforming_after=tf_before,
    )

    # --- terraforming actions ---
    n_terraform = action_counts.get("terraform", 0)
    if n_terraform > 0:
        biosphere.atmosphere.o2 += TERRAFORM_O2_GAIN * n_terraform
        biosphere.atmosphere.co2 -= TERRAFORM_CO2_LOSS * n_terraform
        biosphere.atmosphere.pressure += TERRAFORM_PRESSURE_GAIN * n_terraform
        biosphere.soil.perchlorate -= TERRAFORM_PERCHLORATE_LOSS * n_terraform

    # --- farming actions ---
    n_farm = action_counts.get("farm", 0)
    if n_farm > 0:
        soil_quality = biosphere.soil.nutrient_level * (1.0 - biosphere.soil.perchlorate)
        effective_gain = max(0.1, soil_quality)
        biosphere.flora.coverage += FARM_COVERAGE_GAIN * n_farm * effective_gain
        biosphere.flora.crop_health += FARM_CROP_HEALTH_GAIN * n_farm * effective_gain
        biosphere.soil.nutrient_level += FARM_NUTRIENT_GAIN * n_farm

    # --- infrastructure tech bonuses ---
    for tech_id in infra_completed:
        bonuses = TECH_ECOLOGY_EFFECTS.get(tech_id)
        if bonuses:
            biosphere.flora.crop_health += bonuses.get("crop_health_bonus", 0.0)
            biosphere.flora.coverage += bonuses.get("coverage_bonus", 0.0)
            biosphere.water_cycle.recycling_efficiency += bonuses.get(
                "recycling_bonus", 0.0)
            biosphere.water_cycle.ice_reserves += bonuses.get(
                "ice_conservation", 0.0)

    # --- natural decay ---
    biosphere.flora.coverage -= FLORA_DECAY_RATE
    biosphere.flora.crop_health -= CROP_DECAY_RATE
    biosphere.soil.nutrient_level -= NUTRIENT_DECAY_RATE
    biosphere.water_cycle.ice_reserves -= ICE_DEPLETION_RATE

    # --- event impacts ---
    for ename, esev in zip(event_names, event_severities):
        if ename in ("dust_storm", "solar_flare"):
            damage = STORM_FLORA_DAMAGE_FRACTION * esev * biosphere.flora.coverage
            biosphere.flora.coverage -= damage
            crop_damage = STORM_CROP_DAMAGE_FRACTION * esev * biosphere.flora.crop_health
            biosphere.flora.crop_health -= crop_damage
            result.flora_damage += damage + crop_damage
        elif ename == "ice_volcano":
            biosphere.water_cycle.ice_reserves += 0.05 * (1.0 + esev)
        elif ename == "resource_strike":
            biosphere.soil.nutrient_level += 0.02 * (1.0 + esev)

    # --- aquifer discovery (rare, one-time) ---
    if not biosphere.water_cycle.aquifer_discovered and year >= 15:
        discovery_chance = 0.02 + action_counts.get("explore", 0) * 0.01
        if rng.random() < discovery_chance:
            biosphere.water_cycle.aquifer_discovered = True
            biosphere.water_cycle.ice_reserves += 0.15
            result.aquifer_event = True

    # --- flora tipping point (self-reinforcing photosynthesis) ---
    if biosphere.flora.coverage >= FLORA_TIPPING_THRESHOLD:
        biosphere.atmosphere.o2 += FLORA_TIPPING_O2_BONUS
        biosphere.soil.nutrient_level += FLORA_TIPPING_NUTRIENT_BONUS
        result.tipping_point_triggered = True

    # --- biodiversity slowly follows coverage ---
    if biosphere.flora.coverage > 0.1:
        bio_target = biosphere.flora.coverage * 0.6
        biosphere.flora.biodiversity += (bio_target - biosphere.flora.biodiversity) * 0.1
    else:
        biosphere.flora.biodiversity *= 0.95

    # --- stochastic micro-variation ---
    biosphere.atmosphere.o2 += rng.gauss(0, 0.0005)
    biosphere.soil.nutrient_level += rng.gauss(0, 0.001)
    biosphere.flora.coverage += rng.gauss(0, 0.001)

    # --- final clamp ---
    biosphere.clamp()

    result.biome_after = biosphere.biome
    result.terraforming_after = biosphere.terraforming_progress
    return result


# -- downstream modifiers (applied with one-year lag) -----------------------

def compute_ecology_modifiers(biosphere: Biosphere) -> dict[str, float]:
    """Compute resource modifiers from current biosphere state.

    Returns maintenance/spoilage multipliers compatible with the existing
    tick_resources() infra_modifiers interface. Values < 1.0 are beneficial.
    """
    mods: dict[str, float] = {}

    crop_benefit = biosphere.flora.crop_health * 0.3
    mods["food_spoilage_mult"] = max(0.3, 1.0 - crop_benefit)

    water_benefit = biosphere.water_cycle.recycling_efficiency * 0.25
    mods["water_maintenance_mult"] = max(0.4, 1.0 - water_benefit)

    o2_benefit = min(biosphere.atmosphere.o2, 0.21) * 0.5
    mods["air_spoilage_mult"] = max(0.5, 1.0 - o2_benefit)

    if biosphere.soil.perchlorate > 0.5:
        excess = biosphere.soil.perchlorate - 0.5
        mods["death_rate_ecology_mult"] = 1.0 + excess * 0.4
    else:
        mods["death_rate_ecology_mult"] = 1.0

    return mods


def compute_ecology_upkeep(biosphere: Biosphere) -> dict[str, float]:
    """Compute per-year resource costs for maintaining the biosphere.

    Higher flora coverage and better recycling require more power/water.
    """
    coverage_factor = 1.0 + biosphere.flora.coverage * 2.0
    return {
        "power": BASE_ECOLOGY_UPKEEP["power"] * coverage_factor,
        "water": BASE_ECOLOGY_UPKEEP["water"] * coverage_factor,
    }
