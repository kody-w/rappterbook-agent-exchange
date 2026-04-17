"""
Mars-100 biosphere ecology organ.

Models biomass growth, soil health, crop diversity, and their effects on
colony air and food resources. The biosphere is a living system that decays
under stress and can be devastated by blight when crop diversity is too low.

Integrates with the infrastructure tech tree (greenhouse_dome, research_lab)
and with colonist farming actions.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# --- Tuning constants (module-level for test reference) ---

BIOMASS_GROWTH_BASE = 0.02
BIOMASS_DECAY_STRESS = 0.03
SOIL_REGEN_RATE = 0.005
SOIL_DEGRADE_RATE = 0.01
PHOTOSYNTHESIS_RATE = 0.04
CROP_YIELD_RATE = 0.03
BLIGHT_THRESHOLD = 0.25
BLIGHT_BIOMASS_LOSS = 0.4
BLIGHT_PROBABILITY = 0.3
GREENHOUSE_GROWTH_MULT = 1.4
GREENHOUSE_FOOD_MULT = 1.3
RESEARCH_LAB_DIVERSITY_BOOST = 0.02
MARS_AMBIENT_LIGHT = 0.43
FARMING_INTENSITY_SCALE = 0.1
DIVERSITY_DECAY_RATE = 0.01
DIVERSITY_FARMING_PENALTY = 0.005
MIN_WATER_FOR_GROWTH = 0.1
MIN_POWER_FOR_LIGHTS = 0.15


@dataclass
class Biosphere:
    """Living biosphere state for the Mars colony."""
    biomass: float = 0.1
    soil_health: float = 0.5
    crop_diversity: float = 0.5

    def to_dict(self) -> dict:
        return {
            "biomass": round(self.biomass, 4),
            "soil_health": round(self.soil_health, 4),
            "crop_diversity": round(self.crop_diversity, 4),
        }

    def clone(self) -> Biosphere:
        return Biosphere(
            biomass=self.biomass,
            soil_health=self.soil_health,
            crop_diversity=self.crop_diversity,
        )


@dataclass
class EcologyDelta:
    """Result of one ecology tick — deltas to fold into resource pipeline."""
    air_delta: float = 0.0
    food_delta: float = 0.0
    blight_occurred: bool = False
    biomass_before: float = 0.0
    biomass_after: float = 0.0

    def to_dict(self) -> dict:
        return {
            "air_delta": round(self.air_delta, 6),
            "food_delta": round(self.food_delta, 6),
            "blight_occurred": self.blight_occurred,
            "biomass_before": round(self.biomass_before, 4),
            "biomass_after": round(self.biomass_after, 4),
        }


def _light_factor(power_level: float) -> float:
    """Compute effective light for photosynthesis (Mars ambient + grow lights)."""
    grow_light = max(0.0, power_level - MIN_POWER_FOR_LIGHTS) * 0.5
    return min(1.0, MARS_AMBIENT_LIGHT + grow_light)


def _farming_intensity(farmer_count: int, active_count: int,
                       avg_hydroponics: float) -> float:
    """How much farming effort the colony is putting in (0-1)."""
    if active_count == 0:
        return 0.0
    ratio = farmer_count / active_count
    skill_factor = 0.5 + 0.5 * avg_hydroponics
    return min(1.0, ratio * skill_factor * (1.0 / FARMING_INTENSITY_SCALE))


def tick_biosphere(
    bio: Biosphere,
    *,
    water_level: float,
    power_level: float,
    farmer_count: int,
    active_count: int,
    avg_hydroponics: float,
    has_greenhouse: bool = False,
    has_research_lab: bool = False,
    rng: random.Random | None = None,
) -> EcologyDelta:
    """Advance the biosphere one Martian year. Mutates bio in-place, returns delta."""
    rng = rng or random.Random()
    biomass_before = bio.biomass

    light = _light_factor(power_level)
    farming = _farming_intensity(farmer_count, active_count, avg_hydroponics)

    # --- Biomass growth / decay ---
    water_ok = water_level >= MIN_WATER_FOR_GROWTH
    growth_mult = GREENHOUSE_GROWTH_MULT if has_greenhouse else 1.0

    if water_ok and farming > 0:
        growth = BIOMASS_GROWTH_BASE * light * bio.soil_health * farming * growth_mult
        bio.biomass = min(1.0, bio.biomass + growth)
    else:
        stress = BIOMASS_DECAY_STRESS
        if not water_ok:
            stress *= 1.5
        if farming == 0:
            stress *= 1.2
        bio.biomass = max(0.0, bio.biomass - stress)

    # --- Blight check ---
    blight = False
    if bio.crop_diversity < BLIGHT_THRESHOLD and bio.biomass > 0:
        if rng.random() < BLIGHT_PROBABILITY:
            bio.biomass = max(0.0, bio.biomass * (1.0 - BLIGHT_BIOMASS_LOSS))
            blight = True

    # --- Soil dynamics ---
    if farming > 0:
        bio.soil_health = max(0.0, bio.soil_health - SOIL_DEGRADE_RATE * farming)
    bio.soil_health = min(1.0, bio.soil_health + SOIL_REGEN_RATE * (1.0 - farming))

    # --- Crop diversity ---
    if farmer_count > 0:
        bio.crop_diversity = max(0.0,
            bio.crop_diversity - DIVERSITY_FARMING_PENALTY * farmer_count)
    if has_research_lab:
        bio.crop_diversity = min(1.0,
            bio.crop_diversity + RESEARCH_LAB_DIVERSITY_BOOST)
    bio.crop_diversity = max(0.0,
        bio.crop_diversity - DIVERSITY_DECAY_RATE)

    # --- Output deltas ---
    air_delta = PHOTOSYNTHESIS_RATE * bio.biomass * bio.soil_health * light
    food_mult = GREENHOUSE_FOOD_MULT if has_greenhouse else 1.0
    food_delta = CROP_YIELD_RATE * bio.biomass * bio.soil_health * food_mult

    # Clamp all state to [0, 1]
    bio.biomass = max(0.0, min(1.0, bio.biomass))
    bio.soil_health = max(0.0, min(1.0, bio.soil_health))
    bio.crop_diversity = max(0.0, min(1.0, bio.crop_diversity))

    return EcologyDelta(
        air_delta=air_delta,
        food_delta=food_delta,
        blight_occurred=blight,
        biomass_before=biomass_before,
        biomass_after=bio.biomass,
    )
