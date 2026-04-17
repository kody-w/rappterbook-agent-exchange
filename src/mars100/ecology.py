"""
Ecology organ for Mars-100.

Models a living Mars biosphere that evolves each year in response to
colony terraforming and farming actions.  Produces feedback loops with
resources (food, water, air) and psychology (stress reduction from
nature exposure).

Components:
  Atmosphere  - CO2 to O2 conversion, pressure build-up
  SoilState   - perchlorate remediation, organic content
  WaterCycle  - ice mining yield, aquifer recharge
  Flora       - crop yield, wild plant coverage
  Fauna       - introduced species population (insects, worms, fish)

All functions are pure (no I/O, no globals).  The Biosphere dataclass
is the composite state object passed through the frame loop.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Atmosphere
MARS_BASE_CO2 = 0.953
O2_CONVERSION_RATE = 0.003
PRESSURE_BUILD_RATE = 0.001
DUST_STORM_PRESSURE_LOSS = 0.02

# Soil
PERCHLORATE_INITIAL = 0.8
REMEDIATION_RATE = 0.01
ORGANIC_GROWTH_RATE = 0.005
PERCHLORATE_SAFE_THRESHOLD = 0.3

# Water
ICE_MINING_BASE = 0.02
ICE_MINING_BONUS = 0.005
AQUIFER_RECHARGE_RATE = 0.002
WATER_CYCLE_THRESHOLD = 0.15

# Flora
CROP_GROWTH_RATE = 0.008
WILD_PLANT_RATE = 0.003
FLORA_MIN_SOIL_ORGANIC = 0.05
FLORA_MIN_PERCHLORATE = 0.5
FLORA_OUTDOOR_THRESHOLD = 0.3
FLORA_DECLINE_RATE = 0.005

# Fauna
FAUNA_GROWTH_RATE = 0.004
FAUNA_MIN_FLORA = 0.1
FAUNA_DECLINE_RATE = 0.003

# Biosphere composite weights
BIOSPHERE_WEIGHTS = {
    "atmosphere": 0.20,
    "soil": 0.20,
    "water": 0.20,
    "flora": 0.25,
    "fauna": 0.15,
}

# Resource bonuses
FOOD_BONUS_PER_FLORA = 0.03
WATER_BONUS_PER_AQUIFER = 0.02
AIR_BONUS_PER_O2 = 0.01

# Psychology bonus
PSYCH_STRESS_REDUCTION_MAX = 0.05


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Atmosphere:
    """Mars atmospheric state."""
    co2_fraction: float = MARS_BASE_CO2
    o2_fraction: float = 0.001
    pressure: float = 0.006

    def to_dict(self) -> dict[str, float]:
        return {
            "co2_fraction": round(self.co2_fraction, 6),
            "o2_fraction": round(self.o2_fraction, 6),
            "pressure": round(self.pressure, 6),
        }

    def health_score(self) -> float:
        """0-1 score: how Earth-like is the atmosphere?"""
        o2_score = min(1.0, self.o2_fraction / 0.21)
        pressure_score = min(1.0, self.pressure / 101.3)
        return 0.6 * o2_score + 0.4 * pressure_score


@dataclass
class SoilState:
    """Mars soil / regolith state."""
    perchlorate_level: float = PERCHLORATE_INITIAL
    organic_content: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "perchlorate_level": round(self.perchlorate_level, 6),
            "organic_content": round(self.organic_content, 6),
        }

    def health_score(self) -> float:
        """0-1 score: how suitable is soil for growing?"""
        perch_score = max(0.0, 1.0 - self.perchlorate_level)
        organic_score = min(1.0, self.organic_content / 0.3)
        return 0.5 * perch_score + 0.5 * organic_score


@dataclass
class WaterCycle:
    """Mars water cycle state."""
    ice_reserves: float = 0.5
    aquifer_level: float = 0.0
    surface_water: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "ice_reserves": round(self.ice_reserves, 6),
            "aquifer_level": round(self.aquifer_level, 6),
            "surface_water": round(self.surface_water, 6),
        }

    def health_score(self) -> float:
        """0-1 score: how accessible is water?"""
        return min(1.0, 0.3 * self.ice_reserves + 0.4 * self.aquifer_level
                   + 0.3 * self.surface_water)


@dataclass
class Flora:
    """Plant life on Mars."""
    crop_yield: float = 0.0
    wild_coverage: float = 0.0
    biodiversity: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "crop_yield": round(self.crop_yield, 6),
            "wild_coverage": round(self.wild_coverage, 6),
            "biodiversity": round(self.biodiversity, 6),
        }

    def health_score(self) -> float:
        """0-1 score: how lush is plant life?"""
        return min(1.0, 0.4 * self.crop_yield + 0.35 * self.wild_coverage
                   + 0.25 * self.biodiversity)


@dataclass
class Fauna:
    """Animal life on Mars (introduced species)."""
    population: float = 0.0
    diversity: float = 0.0
    ecosystem_stability: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "population": round(self.population, 6),
            "diversity": round(self.diversity, 6),
            "ecosystem_stability": round(self.ecosystem_stability, 6),
        }

    def health_score(self) -> float:
        """0-1 score: how established is animal life?"""
        return min(1.0, 0.4 * self.population + 0.3 * self.diversity
                   + 0.3 * self.ecosystem_stability)


@dataclass
class Biosphere:
    """Composite Mars biosphere state."""
    atmosphere: Atmosphere = field(default_factory=Atmosphere)
    soil: SoilState = field(default_factory=SoilState)
    water: WaterCycle = field(default_factory=WaterCycle)
    flora: Flora = field(default_factory=Flora)
    fauna: Fauna = field(default_factory=Fauna)

    def biosphere_index(self) -> float:
        """Weighted composite health score (0-1)."""
        scores = {
            "atmosphere": self.atmosphere.health_score(),
            "soil": self.soil.health_score(),
            "water": self.water.health_score(),
            "flora": self.flora.health_score(),
            "fauna": self.fauna.health_score(),
        }
        return sum(BIOSPHERE_WEIGHTS[k] * v for k, v in scores.items())

    def to_dict(self) -> dict[str, Any]:
        return {
            "atmosphere": self.atmosphere.to_dict(),
            "soil": self.soil.to_dict(),
            "water": self.water.to_dict(),
            "flora": self.flora.to_dict(),
            "fauna": self.fauna.to_dict(),
            "biosphere_index": round(self.biosphere_index(), 6),
        }


@dataclass
class EcologyTickResult:
    """Result of one year's ecology evolution."""
    year: int
    biosphere_before: dict[str, Any]
    biosphere_after: dict[str, Any]
    biosphere_index: float
    events: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "year": self.year,
            "biosphere_before": self.biosphere_before,
            "biosphere_after": self.biosphere_after,
            "biosphere_index": round(self.biosphere_index, 6),
            "events": self.events,
        }


# ---------------------------------------------------------------------------
# Pure tick functions
# ---------------------------------------------------------------------------

def tick_atmosphere(atmo: Atmosphere, terraforming_effort: int,
                    has_dust_storm: bool, rng: random.Random) -> list[str]:
    """Evolve atmosphere for one year. Returns list of events."""
    events: list[str] = []

    o2_gain = O2_CONVERSION_RATE * max(1, terraforming_effort)
    noise = rng.gauss(0, 0.0005)
    atmo.o2_fraction = max(0.0, min(1.0, atmo.o2_fraction + o2_gain + noise))
    atmo.co2_fraction = max(0.0, atmo.co2_fraction - o2_gain * 0.8)

    pressure_gain = PRESSURE_BUILD_RATE * max(1, terraforming_effort) + rng.gauss(0, 0.0002)
    if has_dust_storm:
        pressure_gain -= DUST_STORM_PRESSURE_LOSS
        events.append("dust_storm_pressure_loss")
    atmo.pressure = max(0.001, atmo.pressure + pressure_gain)

    if atmo.o2_fraction > 0.05:
        events.append("breathable_trace_o2")
    if atmo.pressure > 1.0:
        events.append("pressure_milestone_1kpa")

    return events


def tick_soil(soil: SoilState, farming_effort: int,
              has_flora: bool, rng: random.Random) -> list[str]:
    """Evolve soil for one year."""
    events: list[str] = []

    remediation = REMEDIATION_RATE * max(1, farming_effort)
    noise = rng.gauss(0, 0.002)
    soil.perchlorate_level = max(0.0, soil.perchlorate_level - remediation + noise)

    organic_gain = ORGANIC_GROWTH_RATE * max(1, farming_effort)
    if has_flora:
        organic_gain *= 1.5
    soil.organic_content = min(1.0, max(0.0, soil.organic_content + organic_gain + rng.gauss(0, 0.001)))

    if soil.perchlorate_level < PERCHLORATE_SAFE_THRESHOLD:
        events.append("perchlorate_safe_for_crops")
    if soil.organic_content > 0.1:
        events.append("organic_soil_milestone")

    return events


def tick_water_cycle(water: WaterCycle, terraforming_effort: int,
                     pressure: float, rng: random.Random) -> list[str]:
    """Evolve water cycle for one year."""
    events: list[str] = []

    ice_extracted = ICE_MINING_BASE + ICE_MINING_BONUS * terraforming_effort
    water.ice_reserves = max(0.0, water.ice_reserves - ice_extracted * 0.1)

    recharge = AQUIFER_RECHARGE_RATE + ice_extracted * 0.5
    water.aquifer_level = min(1.0, max(0.0, water.aquifer_level + recharge + rng.gauss(0, 0.001)))

    if pressure > WATER_CYCLE_THRESHOLD:
        surface_gain = 0.005 * (pressure / WATER_CYCLE_THRESHOLD)
        water.surface_water = min(1.0, water.surface_water + surface_gain)
        events.append("surface_water_forming")
    else:
        water.surface_water = max(0.0, water.surface_water - 0.002)

    return events


def tick_flora(flora: Flora, soil: SoilState, water: WaterCycle,
               pressure: float, farming_effort: int,
               rng: random.Random) -> list[str]:
    """Evolve plant life for one year."""
    events: list[str] = []

    if farming_effort > 0 or flora.crop_yield > 0:
        soil_mult = max(0.3, 1.0 - soil.perchlorate_level)
        crop_gain = CROP_GROWTH_RATE * max(1, farming_effort) * soil_mult
        water_mult = min(1.0, water.aquifer_level * 3)
        crop_gain *= max(0.2, water_mult)
        flora.crop_yield = min(1.0, max(0.0, flora.crop_yield + crop_gain + rng.gauss(0, 0.001)))

    if (soil.perchlorate_level < FLORA_MIN_PERCHLORATE
            and soil.organic_content > FLORA_MIN_SOIL_ORGANIC):
        wild_gain = WILD_PLANT_RATE
        if soil.perchlorate_level < FLORA_OUTDOOR_THRESHOLD and pressure > 0.01:
            wild_gain *= 2.0
            events.append("outdoor_plants_spreading")
        flora.wild_coverage = min(1.0, max(0.0, flora.wild_coverage + wild_gain + rng.gauss(0, 0.0005)))
    elif flora.wild_coverage > 0:
        flora.wild_coverage = max(0.0, flora.wild_coverage - FLORA_DECLINE_RATE)
        events.append("wild_flora_declining")

    total_flora = flora.crop_yield + flora.wild_coverage
    target_biodiv = min(1.0, total_flora * 0.5)
    flora.biodiversity += (target_biodiv - flora.biodiversity) * 0.1

    if flora.crop_yield > 0.3:
        events.append("substantial_crop_production")

    return events


def tick_fauna(fauna: Fauna, flora: Flora, rng: random.Random) -> list[str]:
    """Evolve animal life for one year."""
    events: list[str] = []

    flora_total = flora.crop_yield + flora.wild_coverage
    if flora_total > FAUNA_MIN_FLORA:
        capacity = min(1.0, flora_total * 0.5)
        growth = FAUNA_GROWTH_RATE * (1 - fauna.population / max(0.01, capacity))
        fauna.population = min(1.0, max(0.0, fauna.population + growth + rng.gauss(0, 0.001)))

        target_div = min(1.0, fauna.population * 0.6)
        fauna.diversity += (target_div - fauna.diversity) * 0.1

        stability_gain = 0.002 * fauna.population * fauna.diversity
        fauna.ecosystem_stability = min(1.0, fauna.ecosystem_stability + stability_gain)

        if fauna.population > 0.1:
            events.append("fauna_established")
    elif fauna.population > 0:
        fauna.population = max(0.0, fauna.population - FAUNA_DECLINE_RATE)
        fauna.diversity = max(0.0, fauna.diversity - FAUNA_DECLINE_RATE * 0.5)
        events.append("fauna_declining")

    return events


def tick_ecology(
    biosphere: Biosphere,
    year: int,
    terraforming_effort: int,
    colony_size: int,
    has_dust_storm: bool = False,
    has_solar_event: bool = False,
    rng: random.Random | None = None,
) -> EcologyTickResult:
    """Evolve the full biosphere for one year.

    Args:
        biosphere: Current biosphere state (mutated in-place).
        year: Current simulation year.
        terraforming_effort: Number of colonists doing terraform/farm.
        colony_size: Number of active colonists.
        has_dust_storm: Whether a dust storm occurred this year.
        has_solar_event: Whether a solar flare occurred.
        rng: Random number generator (ecology-dedicated stream).

    Returns:
        EcologyTickResult with before/after snapshots and events.
    """
    if rng is None:
        rng = random.Random(year)

    before = biosphere.to_dict()
    all_events: list[str] = []

    atmo_events = tick_atmosphere(
        biosphere.atmosphere, terraforming_effort, has_dust_storm, rng)
    all_events.extend(atmo_events)

    farming_effort = max(0, terraforming_effort)
    has_any_flora = (biosphere.flora.crop_yield > 0.01
                     or biosphere.flora.wild_coverage > 0.01)
    soil_events = tick_soil(biosphere.soil, farming_effort, has_any_flora, rng)
    all_events.extend(soil_events)

    water_events = tick_water_cycle(
        biosphere.water, terraforming_effort, biosphere.atmosphere.pressure, rng)
    all_events.extend(water_events)

    flora_events = tick_flora(
        biosphere.flora, biosphere.soil, biosphere.water,
        biosphere.atmosphere.pressure, farming_effort, rng)
    all_events.extend(flora_events)

    fauna_events = tick_fauna(biosphere.fauna, biosphere.flora, rng)
    all_events.extend(fauna_events)

    if has_solar_event:
        damage = rng.uniform(0.02, 0.08)
        biosphere.flora.wild_coverage = max(
            0.0, biosphere.flora.wild_coverage - damage)
        biosphere.fauna.population = max(
            0.0, biosphere.fauna.population - damage * 0.5)
        all_events.append("solar_radiation_damage")

    after = biosphere.to_dict()

    return EcologyTickResult(
        year=year,
        biosphere_before=before,
        biosphere_after=after,
        biosphere_index=biosphere.biosphere_index(),
        events=all_events,
    )


# ---------------------------------------------------------------------------
# Bonus computation (for engine integration)
# ---------------------------------------------------------------------------

def compute_ecology_resource_bonus(biosphere: Biosphere) -> dict[str, float]:
    """Compute resource bonuses from the biosphere.

    Returns dict with keys: food, water, air_maintenance_reduction.
    These are additive deltas applied directly to resources.
    Uses LAST year's biosphere (one-year lag in engine).
    """
    food_bonus = biosphere.flora.health_score() * FOOD_BONUS_PER_FLORA
    water_bonus = biosphere.water.aquifer_level * WATER_BONUS_PER_AQUIFER
    air_reduction = biosphere.atmosphere.o2_fraction * AIR_BONUS_PER_O2

    return {
        "food": food_bonus,
        "water": water_bonus,
        "air_maintenance_reduction": air_reduction,
    }


def compute_ecology_psych_bonus(biosphere: Biosphere) -> float:
    """Compute stress reduction from nature exposure.

    Returns a float [0, PSYCH_STRESS_REDUCTION_MAX] to subtract from
    colonist stress.  Uses LAST year's biosphere.
    """
    bio_idx = biosphere.biosphere_index()
    return min(PSYCH_STRESS_REDUCTION_MAX, bio_idx * 0.15)
