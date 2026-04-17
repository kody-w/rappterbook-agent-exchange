"""
Ecology organ for Mars-100 (engine v10.0).

Models the evolving Martian biosphere: atmosphere composition, soil health,
water cycle, and biome progression from barren rock to outdoor crops.
Ecological changes are SLOW (decades) and feed back into colony resources
via spoilage/maintenance multipliers — not direct stock injections.

All state is 0-1 normalised except temperature (C) and pressure (kPa).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# -- Biome definitions -------------------------------------------------------

BIOME_NAMES = ("barren", "microbial", "lichen", "moss", "greenhouse_crops", "outdoor_crops")

BIOME_THRESHOLDS = (0.0, 0.08, 0.20, 0.38, 0.58, 0.80)

BIOME_GATES: list[dict[str, Any]] = [
    {},
    {},
    {"min_soil_fertility": 0.10, "max_perchlorate": 0.50},
    {"min_soil_fertility": 0.20, "max_perchlorate": 0.35},
    {"min_soil_fertility": 0.30, "max_perchlorate": 0.25, "required_techs": {"greenhouse_dome"}},
    {"min_soil_fertility": 0.45, "max_perchlorate": 0.15, "min_o2": 0.05,
     "min_pressure": 1.0},
]

BIOME_FOOD_SPOILAGE_MULT = (1.0, 0.97, 0.93, 0.88, 0.80, 0.72)
BIOME_AIR_MAINTENANCE_MULT = (1.0, 0.98, 0.95, 0.90, 0.85, 0.78)
BIOME_WATER_MAINTENANCE_MULT = (1.0, 0.99, 0.97, 0.94, 0.90, 0.85)

# -- Rate constants ----------------------------------------------------------

TERRAFORM_SCORE_PER_ACTION = 0.005
FARM_SCORE_CONTRIBUTION = 0.002
RESEARCH_SCORE_CONTRIBUTION = 0.001

SOIL_FERTILITY_PER_FARM = 0.003
SOIL_FERTILITY_PER_TERRAFORM = 0.001
PERCHLORATE_DECAY = 0.008
PERCHLORATE_TERRAFORM_BONUS = 0.003

O2_PER_TERRAFORM_ACTION = 0.0004
CO2_PER_O2 = 1.0

WATER_ICE_DRAIN = 0.002
WATER_RECAPTURE_IMPROVEMENT = 0.005

PRESSURE_PER_YEAR = 0.003
TEMP_PER_O2 = 5.0

BIODIVERSITY_PER_BIOME_LEVEL = 0.12
EVENT_ECOLOGY_DAMAGE = 0.006
INFRA_ECOLOGY_BOOST: dict[str, float] = {
    "greenhouse_dome": 0.003,
    "water_recycler": 0.002,
    "air_recycler": 0.002,
}

ECOLOGY_EVENT_DAMAGE_REDUCTION: dict[str, float] = {
    "shelter_reinforcement": 0.4,
}

ECOLOGY_STRESS_RELIEF_PER_BIOME = -0.008
ECOLOGY_PURPOSE_PER_BIOME = 0.005
ECOLOGY_STRESS_PER_PERCHLORATE = 0.01


# -- Dataclasses -------------------------------------------------------------

@dataclass
class Atmosphere:
    """Martian atmospheric composition."""
    o2_fraction: float = 0.001
    co2_fraction: float = 0.96
    temperature_c: float = -60.0
    pressure_kpa: float = 0.636

    def to_dict(self) -> dict[str, float]:
        return {"o2_fraction": self.o2_fraction, "co2_fraction": self.co2_fraction,
                "temperature_c": self.temperature_c, "pressure_kpa": self.pressure_kpa}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "Atmosphere":
        return cls(o2_fraction=d.get("o2_fraction", 0.001),
                   co2_fraction=d.get("co2_fraction", 0.96),
                   temperature_c=d.get("temperature_c", -60.0),
                   pressure_kpa=d.get("pressure_kpa", 0.636))


@dataclass
class SoilState:
    """Martian soil health (0-1 normalised)."""
    fertility: float = 0.05
    perchlorate_index: float = 0.60
    water_content: float = 0.02

    def to_dict(self) -> dict[str, float]:
        return {"fertility": self.fertility, "perchlorate_index": self.perchlorate_index,
                "water_content": self.water_content}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "SoilState":
        return cls(fertility=d.get("fertility", 0.05),
                   perchlorate_index=d.get("perchlorate_index", 0.60),
                   water_content=d.get("water_content", 0.02))


@dataclass
class WaterCycle:
    """Mars water cycle (0-1 normalised)."""
    ice_reserves: float = 0.80
    liquid_available: float = 0.10
    recapture_efficiency: float = 0.50

    def to_dict(self) -> dict[str, float]:
        return {"ice_reserves": self.ice_reserves, "liquid_available": self.liquid_available,
                "recapture_efficiency": self.recapture_efficiency}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "WaterCycle":
        return cls(ice_reserves=d.get("ice_reserves", 0.80),
                   liquid_available=d.get("liquid_available", 0.10),
                   recapture_efficiency=d.get("recapture_efficiency", 0.50))


@dataclass
class EcologyState:
    """Full ecology state for the Mars colony."""
    atmosphere: Atmosphere = field(default_factory=Atmosphere)
    soil: SoilState = field(default_factory=SoilState)
    water: WaterCycle = field(default_factory=WaterCycle)
    biome_level: int = 0
    biodiversity: float = 0.0
    terraforming_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "atmosphere": self.atmosphere.to_dict(),
            "soil": self.soil.to_dict(),
            "water": self.water.to_dict(),
            "biome_level": self.biome_level,
            "biome_name": BIOME_NAMES[min(self.biome_level, len(BIOME_NAMES) - 1)],
            "biodiversity": self.biodiversity,
            "terraforming_score": self.terraforming_score,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EcologyState":
        return cls(
            atmosphere=Atmosphere.from_dict(d.get("atmosphere", {})),
            soil=SoilState.from_dict(d.get("soil", {})),
            water=WaterCycle.from_dict(d.get("water", {})),
            biome_level=d.get("biome_level", 0),
            biodiversity=d.get("biodiversity", 0.0),
            terraforming_score=d.get("terraforming_score", 0.0),
        )

    def clamp(self) -> None:
        """Enforce physical bounds on all ecology state."""
        self.atmosphere.o2_fraction = max(0.0, min(1.0, self.atmosphere.o2_fraction))
        self.atmosphere.co2_fraction = max(0.0, min(1.0, self.atmosphere.co2_fraction))
        self.atmosphere.pressure_kpa = max(0.0, self.atmosphere.pressure_kpa)
        self.soil.fertility = max(0.0, min(1.0, self.soil.fertility))
        self.soil.perchlorate_index = max(0.0, min(1.0, self.soil.perchlorate_index))
        self.soil.water_content = max(0.0, min(1.0, self.soil.water_content))
        self.water.ice_reserves = max(0.0, min(1.0, self.water.ice_reserves))
        self.water.liquid_available = max(0.0, min(1.0, self.water.liquid_available))
        self.water.recapture_efficiency = max(0.0, min(1.0, self.water.recapture_efficiency))
        self.biome_level = max(0, min(len(BIOME_NAMES) - 1, self.biome_level))
        self.biodiversity = max(0.0, min(1.0, self.biodiversity))
        self.terraforming_score = max(0.0, min(1.0, self.terraforming_score))


# -- Tick result -------------------------------------------------------------

@dataclass
class EcologyTickResult:
    """Result of one ecology tick."""
    biome_changed: bool = False
    old_biome: int = 0
    new_biome: int = 0
    tipping_point: str | None = None
    terraforming_score: float = 0.0
    atmosphere_summary: dict[str, float] = field(default_factory=dict)
    soil_summary: dict[str, float] = field(default_factory=dict)
    water_summary: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "biome_changed": self.biome_changed,
            "biome_level": self.new_biome,
            "biome_name": BIOME_NAMES[min(self.new_biome, len(BIOME_NAMES) - 1)],
            "terraforming_score": self.terraforming_score,
            "atmosphere": self.atmosphere_summary,
            "soil": self.soil_summary,
            "water": self.water_summary,
        }
        if self.tipping_point:
            d["tipping_point"] = self.tipping_point
        return d


# -- Core functions ----------------------------------------------------------

def _check_biome_gate(level: int, state: EcologyState,
                      infra_completed: list[str]) -> bool:
    """Check whether ecology state satisfies the gate for a given biome level."""
    if level >= len(BIOME_GATES):
        return False
    gate = BIOME_GATES[level]
    if not gate:
        return True
    if state.soil.fertility < gate.get("min_soil_fertility", 0.0):
        return False
    if state.soil.perchlorate_index > gate.get("max_perchlorate", 1.0):
        return False
    if state.atmosphere.o2_fraction < gate.get("min_o2", 0.0):
        return False
    if state.atmosphere.pressure_kpa < gate.get("min_pressure", 0.0):
        return False
    required_techs = gate.get("required_techs", set())
    if required_techs and not required_techs.issubset(set(infra_completed)):
        return False
    return True


def compute_biome_level(state: EcologyState,
                        infra_completed: list[str]) -> int:
    """Determine the highest biome level the ecology qualifies for."""
    level = 0
    for i in range(1, len(BIOME_THRESHOLDS)):
        if state.terraforming_score >= BIOME_THRESHOLDS[i]:
            if _check_biome_gate(i, state, infra_completed):
                level = i
            else:
                break
        else:
            break
    return level


def compute_ecology_modifiers(state: EcologyState) -> dict[str, float]:
    """Compute resource modifier multipliers from current ecology state.

    Returns keys like food_spoilage_mult, air_maintenance_mult,
    water_maintenance_mult that plug into tick_resources().
    Call at the START of a tick using previous year ecology (one-year lag).
    """
    lvl = min(state.biome_level, len(BIOME_FOOD_SPOILAGE_MULT) - 1)
    return {
        "food_spoilage_mult": BIOME_FOOD_SPOILAGE_MULT[lvl],
        "air_maintenance_mult": BIOME_AIR_MAINTENANCE_MULT[lvl],
        "water_maintenance_mult": BIOME_WATER_MAINTENANCE_MULT[lvl],
    }


def compute_ecology_psych_pressure(state: EcologyState) -> dict[str, float]:
    """Compute psychological pressure from ecology state.

    Greening reduces stress; high perchlorate raises it.
    Higher biome level gives purpose (visible terraforming progress).
    """
    stress = (
        ECOLOGY_STRESS_RELIEF_PER_BIOME * state.biome_level
        + ECOLOGY_STRESS_PER_PERCHLORATE * state.soil.perchlorate_index
    )
    purpose = ECOLOGY_PURPOSE_PER_BIOME * state.biome_level
    return {"stress": stress, "purpose": purpose}


def tick_ecology(
    state: EcologyState,
    year: int,
    terraforming_count: int,
    avg_terraform_skill: float,
    farming_count: int,
    research_count: int,
    event_damage: float,
    infra_completed: list[str],
    rng: random.Random,
) -> EcologyTickResult:
    """Advance the ecology by one Martian year.

    Mutates state in place.  Returns a tick result with summary data.
    Resource modifiers should be obtained separately via
    compute_ecology_modifiers() using previous-year state (one-year lag).
    """
    old_biome = state.biome_level

    # -- Terraforming score accumulation ------------------------------------
    score_delta = (
        TERRAFORM_SCORE_PER_ACTION * terraforming_count * max(0.1, avg_terraform_skill)
        + FARM_SCORE_CONTRIBUTION * farming_count
        + RESEARCH_SCORE_CONTRIBUTION * research_count
    )
    for tech_id, boost in INFRA_ECOLOGY_BOOST.items():
        if tech_id in infra_completed:
            score_delta += boost

    score_delta += rng.gauss(0, 0.001)

    damage = event_damage * EVENT_ECOLOGY_DAMAGE
    for tech_id, reduction in ECOLOGY_EVENT_DAMAGE_REDUCTION.items():
        if tech_id in infra_completed:
            damage *= (1.0 - reduction)
    score_delta -= max(0.0, damage)

    state.terraforming_score += score_delta

    # -- Atmosphere ----------------------------------------------------------
    o2_delta = O2_PER_TERRAFORM_ACTION * terraforming_count
    if state.biome_level >= 2:
        o2_delta += state.biodiversity * 0.0002
    state.atmosphere.o2_fraction += o2_delta
    state.atmosphere.co2_fraction -= o2_delta * CO2_PER_O2
    state.atmosphere.pressure_kpa += PRESSURE_PER_YEAR * (1 + terraforming_count * 0.1)
    state.atmosphere.temperature_c += o2_delta * TEMP_PER_O2

    # -- Soil ----------------------------------------------------------------
    state.soil.fertility += (
        SOIL_FERTILITY_PER_FARM * farming_count
        + SOIL_FERTILITY_PER_TERRAFORM * terraforming_count
    )
    state.soil.perchlorate_index -= (
        PERCHLORATE_DECAY
        + PERCHLORATE_TERRAFORM_BONUS * terraforming_count
    )
    state.soil.water_content += state.water.liquid_available * 0.01

    # -- Water cycle ---------------------------------------------------------
    melt = WATER_ICE_DRAIN * (1 + terraforming_count * 0.05)
    melt = min(melt, state.water.ice_reserves)
    state.water.ice_reserves -= melt
    state.water.liquid_available += melt * state.water.recapture_efficiency
    if "water_recycler" in infra_completed:
        state.water.recapture_efficiency += WATER_RECAPTURE_IMPROVEMENT
    evap = state.water.liquid_available * 0.02 * rng.uniform(0.8, 1.2)
    state.water.liquid_available -= evap

    # -- Biome level ---------------------------------------------------------
    new_biome = compute_biome_level(state, infra_completed)
    state.biome_level = new_biome

    target_bio = BIODIVERSITY_PER_BIOME_LEVEL * new_biome
    state.biodiversity += (target_bio - state.biodiversity) * 0.1 + rng.gauss(0, 0.005)

    # -- Clamp ---------------------------------------------------------------
    state.clamp()

    # -- Tipping point -------------------------------------------------------
    tipping = None
    if new_biome > old_biome:
        tipping = "Biome advanced: {} -> {}".format(
            BIOME_NAMES[old_biome], BIOME_NAMES[new_biome])

    return EcologyTickResult(
        biome_changed=new_biome != old_biome,
        old_biome=old_biome,
        new_biome=new_biome,
        tipping_point=tipping,
        terraforming_score=state.terraforming_score,
        atmosphere_summary=state.atmosphere.to_dict(),
        soil_summary=state.soil.to_dict(),
        water_summary=state.water.to_dict(),
    )
