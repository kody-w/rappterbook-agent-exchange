"""
Planetary ecology — Mars co-evolves with the colony.

Two-layer environmental model:
  1. Biosphere (organic) — crops, soil health, biodiversity
  2. MarsEcology (planetary) — atmosphere, dust, water ice, radiation, terraforming
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Biosphere:
    """Organic biosphere state — crops, soil, biodiversity."""
    biomass: float = 0.1
    soil_health: float = 0.3
    crop_diversity: float = 0.2

    def to_dict(self) -> dict:
        return {
            "biomass": round(self.biomass, 4),
            "soil_health": round(self.soil_health, 4),
            "crop_diversity": round(self.crop_diversity, 4),
        }


@dataclass
class EcologyDelta:
    """Change report from one biosphere tick."""
    air_delta: float = 0.0
    food_delta: float = 0.0
    biomass_change: float = 0.0
    soil_change: float = 0.0

    def to_dict(self) -> dict:
        return {
            "air_delta": round(self.air_delta, 4),
            "food_delta": round(self.food_delta, 4),
            "biomass_change": round(self.biomass_change, 4),
            "soil_change": round(self.soil_change, 4),
        }


def tick_biosphere(bio: Biosphere, farmers: int, event_severity: float) -> EcologyDelta:
    """Advance the biosphere by one year."""
    farm_boost = 0.02 * farmers
    damage = 0.05 * event_severity
    old_biomass = bio.biomass
    bio.biomass = _clamp(bio.biomass + farm_boost * 0.5 - damage * 0.3)
    old_soil = bio.soil_health
    bio.soil_health = _clamp(bio.soil_health + farm_boost * 0.3 - damage * 0.2)
    bio.crop_diversity = _clamp(bio.crop_diversity + farm_boost * 0.1 - damage * 0.1)
    air_delta = bio.biomass * 0.02
    food_delta = bio.soil_health * bio.crop_diversity * 0.03 * max(1, farmers)
    return EcologyDelta(
        air_delta=air_delta, food_delta=food_delta,
        biomass_change=bio.biomass - old_biomass,
        soil_change=bio.soil_health - old_soil,
    )


ECOLOGY_VARS = ("atmosphere", "dust", "water_ice", "radiation", "terraform")
_NATURAL_DRIFT = 0.005
_TERRAFORM_RATE = 0.015
_WATER_DEPLETION = 0.002
_DUST_DECAY = 0.01
_RADIATION_BASE = 0.6
_ATMOSPHERE_RADIATION_COEFF = 0.5


@dataclass
class MarsEcology:
    """Planetary-scale environmental state. All values bounded [0, 1]."""
    atmosphere: float = 0.05
    dust: float = 0.4
    water_ice: float = 0.6
    radiation: float = 0.7
    terraform: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(getattr(self, k), 4) for k in ECOLOGY_VARS}

    def habitability(self) -> float:
        """Scalar habitability score in [0, 1]."""
        score = (
            self.atmosphere * 0.25
            + (1.0 - self.dust) * 0.15
            + self.water_ice * 0.20
            + (1.0 - self.radiation) * 0.25
            + self.terraform * 0.15
        )
        return _clamp(score)

    def harshness(self) -> float:
        """Inverse of habitability."""
        return 1.0 - self.habitability()


def tick_ecology(
    eco: MarsEcology, population: int, terraformers: int,
    completed_techs: list[str], rng_drift: float = 0.0,
) -> dict:
    """Advance planetary ecology by one Martian year."""
    before = eco.to_dict()
    eco.atmosphere = _clamp(eco.atmosphere + _NATURAL_DRIFT * rng_drift * 0.5)
    eco.dust = _clamp(eco.dust - _DUST_DECAY + _NATURAL_DRIFT * rng_drift)
    if terraformers > 0:
        tf_progress = _TERRAFORM_RATE * math.sqrt(terraformers)
        eco.terraform = _clamp(eco.terraform + tf_progress)
        eco.atmosphere = _clamp(eco.atmosphere + tf_progress * 0.4)
    eco.water_ice = _clamp(eco.water_ice - _WATER_DEPLETION * population)
    if eco.atmosphere > 0.1:
        eco.dust = _clamp(eco.dust - eco.atmosphere * 0.02)
    eco.radiation = _clamp(
        _RADIATION_BASE - eco.atmosphere * _ATMOSPHERE_RADIATION_COEFF
        - eco.terraform * 0.1
    )
    if "greenhouse_dome" in completed_techs:
        eco.atmosphere = _clamp(eco.atmosphere + 0.005)
    if "water_recycler" in completed_techs:
        eco.water_ice = _clamp(eco.water_ice + _WATER_DEPLETION * population * 0.5)
    if "shelter_reinforcement" in completed_techs:
        eco.dust = _clamp(eco.dust - 0.005)
    after = eco.to_dict()
    deltas = {k: round(after[k] - before[k], 6) for k in ECOLOGY_VARS}
    return {
        "before": before, "after": after, "deltas": deltas,
        "habitability": round(eco.habitability(), 4),
        "harshness": round(eco.harshness(), 4),
    }


def compute_ecology_resource_modifiers(eco: MarsEcology) -> dict[str, float]:
    """Multiplicative resource production modifiers from ecology."""
    return {
        "water": round(0.7 + 0.6 * eco.water_ice, 4),
        "power": round(0.7 + 0.6 * (1.0 - eco.dust), 4),
        "food": round(0.8 + 0.4 * (eco.atmosphere + eco.terraform) / 2, 4),
        "air": round(0.7 + 0.6 * eco.atmosphere, 4),
    }


def compute_ecology_death_modifier(eco: MarsEcology) -> float:
    """Death rate multiplier from planetary harshness. Returns [0.5, 2.0]."""
    h = eco.harshness()
    mod = 0.5 + h if h <= 0.5 else 1.0 + (h - 0.5) * 2.0
    return round(_clamp(mod, lo=0.5, hi=2.0), 4)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))
