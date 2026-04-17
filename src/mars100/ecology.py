"""
Ecology organ for Mars-100 (engine v10.0).

Models Mars terraforming: atmosphere, soil, flora -> biosphere.
One-year lag: LAST year's biosphere drives THIS year's resource bonuses.
Active decline: without continued effort, ecology deteriorates toward Mars baseline.

RNG offset: seed + 11213
"""
from __future__ import annotations

import random as _random_module
from dataclasses import dataclass, field
from typing import Any

BIOME_NAMES = ("barren", "lichen", "moss", "grassland", "shrubland", "forest")
BIOME_UP_THRESHOLDS = (0.0, 0.08, 0.18, 0.32, 0.50, 0.70)
BIOME_DOWN_THRESHOLDS = (0.0, 0.05, 0.14, 0.27, 0.44, 0.63)
assert len(BIOME_NAMES) == len(BIOME_UP_THRESHOLDS) == len(BIOME_DOWN_THRESHOLDS)

PRESSURE_GATE_KPA = 5.0
TEMPERATURE_GATE_C = -40.0
PERCHLORATE_GATE = 0.3

MAX_FOOD_BONUS = 0.015
MAX_AIR_BONUS = 0.010
MAX_WATER_BONUS = 0.008
MAX_NATURE_STRESS_REDUCTION = 0.05

MARS_BASELINE_PRESSURE_KPA = 0.6
TERRAFORM_PRESSURE_RATE = 0.012
TERRAFORM_O2_RATE = 0.003
FLORA_O2_RATE = 0.002
GREENHOUSE_TECH_TEMP_BONUS = 0.15
ATMO_GREENHOUSE_COEFF = 0.05
OUTDOOR_PLANT_TEMP_COEFF = 0.5
FARM_PERCHLORATE_RATE = 0.008
FARM_ORGANIC_RATE = 0.004
FLORA_ORGANIC_RATE = 0.002
TERRAFORM_MOISTURE_RATE = 0.003
FARM_MOISTURE_RATE = 0.002
PRESSURE_MOISTURE_BONUS = 0.005
MOISTURE_DRY_FACTOR = 0.98
FARM_GREENHOUSE_RATE = 0.006
GREENHOUSE_DECAY_ACTIVE = 0.001
GREENHOUSE_DECAY_IDLE = 0.005
OUTDOOR_BASE_RATE = 0.003
OUTDOOR_FROM_GREENHOUSE = 0.002
OUTDOOR_DIE_FACTOR = 0.95
OUTDOOR_DIE_FLAT = 0.001
BLOOM_PROBABILITY = 0.02
BLOOM_MIN_INDEX = 0.15
BLOOM_BONUS = 1.1
CONTAM_PROBABILITY = 0.02
CONTAM_MIN_PERCHLORATE = 0.5
CONTAM_ORGANIC_FACTOR = 0.9


@dataclass
class EcologyState:
    """Mars biosphere state."""
    pressure_kpa: float = 0.6
    o2_kpa: float = 0.001
    temperature_c: float = -60.0
    perchlorate: float = 0.8
    soil_organic: float = 0.0
    soil_moisture: float = 0.05
    greenhouse_crops: float = 0.0
    outdoor_plants: float = 0.0
    biome_level: int = 0
    biosphere_index: float = 0.0
    biome_unlocks: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pressure_kpa": round(self.pressure_kpa, 4),
            "o2_kpa": round(self.o2_kpa, 4),
            "temperature_c": round(self.temperature_c, 2),
            "perchlorate": round(self.perchlorate, 4),
            "soil_organic": round(self.soil_organic, 4),
            "soil_moisture": round(self.soil_moisture, 4),
            "greenhouse_crops": round(self.greenhouse_crops, 4),
            "outdoor_plants": round(self.outdoor_plants, 4),
            "biome_level": self.biome_level,
            "biosphere_index": round(self.biosphere_index, 4),
            "biome_unlocks": list(self.biome_unlocks),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EcologyState":
        return cls(
            pressure_kpa=d.get("pressure_kpa", 0.6),
            o2_kpa=d.get("o2_kpa", 0.001),
            temperature_c=d.get("temperature_c", -60.0),
            perchlorate=d.get("perchlorate", 0.8),
            soil_organic=d.get("soil_organic", 0.0),
            soil_moisture=d.get("soil_moisture", 0.05),
            greenhouse_crops=d.get("greenhouse_crops", 0.0),
            outdoor_plants=d.get("outdoor_plants", 0.0),
            biome_level=d.get("biome_level", 0),
            biosphere_index=d.get("biosphere_index", 0.0),
            biome_unlocks=list(d.get("biome_unlocks", [])),
        )

    def clamp(self) -> None:
        """Enforce physical bounds on all fields."""
        self.pressure_kpa = max(0.0, self.pressure_kpa)
        self.o2_kpa = max(0.0, min(self.pressure_kpa, self.o2_kpa))
        self.temperature_c = max(-80.0, min(30.0, self.temperature_c))
        self.perchlorate = max(0.0, min(1.0, self.perchlorate))
        self.soil_organic = max(0.0, min(1.0, self.soil_organic))
        self.soil_moisture = max(0.0, min(1.0, self.soil_moisture))
        self.greenhouse_crops = max(0.0, min(1.0, self.greenhouse_crops))
        self.outdoor_plants = max(0.0, min(1.0, self.outdoor_plants))
        self.biome_level = max(0, min(len(BIOME_NAMES) - 1, self.biome_level))
        self.biosphere_index = max(0.0, min(1.0, self.biosphere_index))

    def health(self) -> float:
        """Alias for biosphere_index for engine compatibility."""
        return self.biosphere_index


@dataclass
class EcologyYearContext:
    """Input context for one ecology tick."""
    year: int
    terraform_count: int
    farm_count: int
    research_count: int
    population: int
    infrastructure_completed: list


@dataclass
class EcologyTickResult:
    """Output of one ecology tick."""
    resource_bonuses: dict = field(default_factory=dict)
    nature_stress_reduction: float = 0.0
    biome_transition: dict | None = None
    tipping_event: str | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "resource_bonuses": self.resource_bonuses,
            "nature_stress_reduction": round(self.nature_stress_reduction, 4),
        }
        if self.biome_transition:
            d["biome_transition"] = self.biome_transition
        if self.tipping_event:
            d["tipping_event"] = self.tipping_event
        return d


def compute_atmosphere_score(eco: EcologyState) -> float:
    """Normalize atmosphere state into [0, 1] score."""
    pressure_score = min(1.0, eco.pressure_kpa / 50.0)
    o2_score = min(1.0, eco.o2_kpa / 16.0)
    temp_score = max(0.0, min(1.0, (eco.temperature_c + 60.0) / 80.0))
    return pressure_score * 0.3 + o2_score * 0.4 + temp_score * 0.3


def compute_soil_score(eco: EcologyState) -> float:
    """Normalize soil health into [0, 1] score."""
    perc_score = 1.0 - eco.perchlorate
    return perc_score * 0.4 + eco.soil_organic * 0.35 + eco.soil_moisture * 0.25


def compute_flora_score(eco: EcologyState) -> float:
    """Normalize flora coverage into [0, 1] score."""
    return eco.greenhouse_crops * 0.6 + eco.outdoor_plants * 0.4


def compute_biosphere_index(eco: EcologyState) -> float:
    """Compute weighted composite biosphere health [0-1]."""
    atmo = compute_atmosphere_score(eco)
    soil = compute_soil_score(eco)
    flora = compute_flora_score(eco)
    return atmo * 0.3 + soil * 0.3 + flora * 0.4


def outdoor_habitable(eco: EcologyState) -> bool:
    """Check if outdoor conditions support plant life."""
    return (eco.pressure_kpa >= PRESSURE_GATE_KPA
            and eco.temperature_c >= TEMPERATURE_GATE_C
            and eco.perchlorate < PERCHLORATE_GATE)


def has_greenhouse_tech(infra_completed: list) -> bool:
    """Check if any greenhouse technology is completed."""
    return any("greenhouse" in str(t).lower() for t in infra_completed)


def compute_ecology_bonuses(eco: EcologyState) -> dict:
    """Compute resource bonuses from ecology state."""
    food = (eco.greenhouse_crops * MAX_FOOD_BONUS
            + eco.outdoor_plants * MAX_FOOD_BONUS * 0.5)
    air = eco.outdoor_plants * MAX_AIR_BONUS + eco.o2_kpa * 0.001
    water = eco.soil_moisture * MAX_WATER_BONUS
    return {"food": round(food, 6), "air": round(air, 6), "water": round(water, 6)}


def compute_resource_modifiers(eco: EcologyState) -> dict[str, float]:
    """Compute resource spoilage/maintenance multipliers from ecology.

    Returns multipliers close to 1.0 that reduce spoilage as ecology improves.
    Applied to resource drain in the engine (one-year lag).
    """
    h = eco.biosphere_index
    return {
        "food_spoilage_mult": max(0.5, 1.0 - h * 0.3),
        "air_maintenance_mult": max(0.5, 1.0 - h * 0.2),
        "water_maintenance_mult": max(0.5, 1.0 - h * 0.15),
    }


def compute_nature_stress_reduction(eco: EcologyState) -> float:
    """Compute stress reduction from nature exposure."""
    return eco.biosphere_index * MAX_NATURE_STRESS_REDUCTION


def update_biome_level(eco: EcologyState) -> dict | None:
    """Update biome level with hysteresis. Returns transition info or None."""
    old_level = eco.biome_level
    new_level = old_level
    # Check promotion
    for level in range(old_level + 1, len(BIOME_UP_THRESHOLDS)):
        if eco.biosphere_index >= BIOME_UP_THRESHOLDS[level]:
            new_level = level
        else:
            break
    # Check demotion (only if no promotion happened)
    if new_level == old_level and old_level > 0:
        if eco.biosphere_index < BIOME_DOWN_THRESHOLDS[old_level]:
            new_level = old_level - 1
    if new_level == old_level:
        return None
    first_time = new_level > old_level and new_level not in eco.biome_unlocks
    eco.biome_level = new_level
    if first_time:
        eco.biome_unlocks.append(new_level)
    return {
        "from_level": old_level, "to_level": new_level,
        "from_name": BIOME_NAMES[old_level], "to_name": BIOME_NAMES[new_level],
        "direction": "up" if new_level > old_level else "down",
        "first_time": first_time,
    }


def tick_ecology(
    eco: EcologyState,
    ctx: EcologyYearContext,
    rng: _random_module.Random,
) -> EcologyTickResult:
    """Advance ecology by one year. Mutates eco in place.

    Resource bonuses are computed from state BEFORE mutation (1-year lag).
    """
    result = EcologyTickResult()
    # Bonuses from LAST year's state (before we mutate)
    result.resource_bonuses = compute_ecology_bonuses(eco)
    result.nature_stress_reduction = compute_nature_stress_reduction(eco)

    # --- Atmosphere ---
    pressure_gain = ctx.terraform_count * TERRAFORM_PRESSURE_RATE
    pressure_gain *= max(0.5, 1.0 + rng.gauss(0, 0.1))
    eco.pressure_kpa += pressure_gain

    o2_gain = (ctx.terraform_count * TERRAFORM_O2_RATE
               + (eco.greenhouse_crops + eco.outdoor_plants) * FLORA_O2_RATE)
    eco.o2_kpa += o2_gain

    temp_gain = 0.0
    if eco.pressure_kpa > MARS_BASELINE_PRESSURE_KPA:
        temp_gain += ((eco.pressure_kpa - MARS_BASELINE_PRESSURE_KPA)
                      * ATMO_GREENHOUSE_COEFF)
    if has_greenhouse_tech(ctx.infrastructure_completed):
        temp_gain += GREENHOUSE_TECH_TEMP_BONUS
    temp_gain += eco.outdoor_plants * OUTDOOR_PLANT_TEMP_COEFF
    eco.temperature_c += temp_gain

    # --- Soil ---
    eco.perchlorate = max(
        0.0, eco.perchlorate - ctx.farm_count * FARM_PERCHLORATE_RATE
    )
    organic_gain = (ctx.farm_count * FARM_ORGANIC_RATE
                    + eco.greenhouse_crops * FLORA_ORGANIC_RATE)
    eco.soil_organic = min(1.0, eco.soil_organic + organic_gain)

    moisture_gain = (ctx.terraform_count * TERRAFORM_MOISTURE_RATE
                     + ctx.farm_count * FARM_MOISTURE_RATE)
    if eco.pressure_kpa >= PRESSURE_GATE_KPA:
        moisture_gain += PRESSURE_MOISTURE_BONUS
    eco.soil_moisture = min(
        1.0, eco.soil_moisture * MOISTURE_DRY_FACTOR + moisture_gain
    )

    # --- Flora ---
    soil_quality = compute_soil_score(eco)
    greenhouse_growth = (ctx.farm_count * FARM_GREENHOUSE_RATE
                         * (0.3 + 0.7 * soil_quality))
    greenhouse_decay = (GREENHOUSE_DECAY_IDLE if ctx.farm_count == 0
                        else GREENHOUSE_DECAY_ACTIVE)
    eco.greenhouse_crops = max(
        0.0, min(1.0, eco.greenhouse_crops + greenhouse_growth - greenhouse_decay)
    )

    if outdoor_habitable(eco):
        outdoor_growth = (OUTDOOR_BASE_RATE * soil_quality
                          + eco.greenhouse_crops * OUTDOOR_FROM_GREENHOUSE)
        eco.outdoor_plants = min(1.0, eco.outdoor_plants + outdoor_growth)
    else:
        eco.outdoor_plants = max(
            0.0, eco.outdoor_plants * OUTDOOR_DIE_FACTOR - OUTDOOR_DIE_FLAT
        )

    # --- Stochastic events ---
    roll = rng.random()
    if roll < BLOOM_PROBABILITY and eco.biosphere_index >= BLOOM_MIN_INDEX:
        eco.greenhouse_crops = min(1.0, eco.greenhouse_crops * BLOOM_BONUS)
        result.tipping_event = "ecological_bloom"
    elif (roll < BLOOM_PROBABILITY + CONTAM_PROBABILITY
          and eco.perchlorate >= CONTAM_MIN_PERCHLORATE):
        eco.soil_organic *= CONTAM_ORGANIC_FACTOR
        result.tipping_event = "perchlorate_contamination"

    # --- Composite index + biome update ---
    eco.biosphere_index = compute_biosphere_index(eco)
    transition = update_biome_level(eco)
    if transition:
        result.biome_transition = transition
    eco.clamp()
    return result
