"""
Health organ for Mars-100 colony simulation (engine v9.0).

Tracks per-colonist physical health: fitness, cumulative radiation,
chronic conditions.  Aging is the primary new mortality driver --
colonists who survive resource scarcity may still die of old age,
radiation sickness, or accumulated chronic illness.

Phase 1 scope (v9.0):
  - HealthState per colonist (fitness, radiation, chronic_conditions)
  - tick_health(): update health from year context
  - Aging with proper biological-age tracking
  - Cumulative radiation from Mars background + solar flare events
  - Chronic conditions: bone_loss, radiation_damage (probabilistic)
  - health_death_modifier(): multiplier on base death rate
  - health_death_cause(): returns dominant health hazard for cause-of-death

Deferred (v9.1+):
  - Injury subsystem (acquisition paths, healing)
  - Epidemic/infection spread
  - Genetic predisposition for children
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

# -- constants ---------------------------------------------------------------

# Mars year ≈ 1.88 Earth years
MARS_YEAR_IN_EARTH_YEARS = 1.88

# Founders arrive at 25-35 Earth years old
FOUNDER_AGE_RANGE = (25, 35)
# Immigrants arrive at 25-45 Earth years old
IMMIGRANT_AGE_RANGE = (25, 45)

# Aging: fitness declines slowly at first, accelerates after 60 Earth years
AGING_ONSET_EARTH_YEARS = 45
AGING_RATE_YOUNG = 0.002      # per Mars year before onset
AGING_RATE_OLD = 0.008         # per Mars year after onset
AGING_ACCELERATION_AGE = 70    # Earth years; rate doubles again

# Radiation
RADIATION_BACKGROUND = 0.006   # per Mars year (Mars surface ~0.67 mSv/day)
SOLAR_FLARE_RADIATION = 0.06   # per solar flare event
SHELTER_RADIATION_MULT = 0.5   # shelter_reinforcement halves radiation intake

# Chronic conditions
BONE_LOSS_ONSET_YEAR = 15      # Mars years in colony before bone loss risk
BONE_LOSS_PROBABILITY = 0.08   # per year after onset
BONE_LOSS_FITNESS_PENALTY = 0.008  # per year if present

RADIATION_DAMAGE_THRESHOLD = 0.40  # radiation level to trigger condition
RADIATION_DAMAGE_PROBABILITY = 0.12  # per year above threshold
RADIATION_DAMAGE_FITNESS_PENALTY = 0.012  # per year if present

# Medicine interaction
MEDICINE_HEALING_FACTOR = 0.6  # high medicine slows chronic condition effects
MED_BAY_CONDITION_FACTOR = 0.5 # med_bay halves condition progression

# Death thresholds
FITNESS_CRITICAL = 0.10        # below this, health-related death likely
OLD_AGE_MIN_EARTH_YEARS = 70   # minimum age for old_age death cause
RADIATION_LETHAL = 0.85        # above this, radiation_sickness death possible


# -- data classes ------------------------------------------------------------

@dataclass
class HealthState:
    """Per-colonist physical health state.

    fitness:            0 = dead, 1 = peak health
    radiation:          0 = no exposure, 1 = lethal dose
    chronic_conditions: list of active condition IDs
    initial_age:        Earth years at colony arrival
    """
    fitness: float = 0.85
    radiation: float = 0.0
    chronic_conditions: list[str] = field(default_factory=list)
    initial_age: float = 30.0

    def biological_age(self, birth_year: int, current_year: int) -> float:
        """Compute current biological age in Earth years."""
        mars_years_in_colony = max(0, current_year - birth_year)
        return self.initial_age + mars_years_in_colony * MARS_YEAR_IN_EARTH_YEARS

    def to_dict(self) -> dict:
        return {
            "fitness": round(self.fitness, 4),
            "radiation": round(self.radiation, 4),
            "chronic_conditions": list(self.chronic_conditions),
            "initial_age": round(self.initial_age, 1),
        }

    @classmethod
    def from_dict(cls, d: dict) -> HealthState:
        return cls(
            fitness=d.get("fitness", 0.85),
            radiation=d.get("radiation", 0.0),
            chronic_conditions=list(d.get("chronic_conditions", [])),
            initial_age=d.get("initial_age", 30.0),
        )


@dataclass
class ColonistHealthContext:
    """Per-colonist context needed by tick_health."""
    colonist_id: str
    birth_year: int
    has_solar_flare: bool
    medicine_level: float
    has_med_bay: bool
    has_shelter: bool
    mars_years_in_colony: int


@dataclass
class HealthTickResult:
    """Result of one year's health tick across all colonists."""
    new_conditions: list[dict]
    fitness_changes: dict[str, float]
    radiation_changes: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "new_conditions": self.new_conditions,
            "fitness_changes": {k: round(v, 4) for k, v in self.fitness_changes.items()},
            "radiation_changes": {k: round(v, 4) for k, v in self.radiation_changes.items()},
        }


# -- helpers -----------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def compute_aging_delta(health: HealthState, birth_year: int,
                        current_year: int) -> float:
    """Compute fitness loss from aging this year.

    Fitness declines slowly before onset age, faster after, and
    accelerates again past acceleration age.
    """
    age = health.biological_age(birth_year, current_year)
    if age < AGING_ONSET_EARTH_YEARS:
        return AGING_RATE_YOUNG
    elif age < AGING_ACCELERATION_AGE:
        return AGING_RATE_OLD
    else:
        # Quadratic acceleration past 70 Earth years
        extra_years = age - AGING_ACCELERATION_AGE
        return AGING_RATE_OLD + 0.0003 * extra_years


def compute_radiation_delta(ctx: ColonistHealthContext) -> float:
    """Compute radiation increase for this year."""
    rad = RADIATION_BACKGROUND
    if ctx.has_solar_flare:
        rad += SOLAR_FLARE_RADIATION
    if ctx.has_shelter:
        rad *= SHELTER_RADIATION_MULT
    return rad


def check_new_conditions(health: HealthState, ctx: ColonistHealthContext,
                         rng: random.Random) -> list[str]:
    """Check for new chronic conditions developing this year."""
    new: list[str] = []

    # Bone loss: develops after extended time in Mars gravity
    if ("bone_loss" not in health.chronic_conditions
            and ctx.mars_years_in_colony >= BONE_LOSS_ONSET_YEAR
            and rng.random() < BONE_LOSS_PROBABILITY):
        new.append("bone_loss")

    # Radiation damage: develops when cumulative radiation is high
    if ("radiation_damage" not in health.chronic_conditions
            and health.radiation >= RADIATION_DAMAGE_THRESHOLD
            and rng.random() < RADIATION_DAMAGE_PROBABILITY):
        new.append("radiation_damage")

    return new


def compute_condition_penalty(health: HealthState, medicine_level: float,
                              has_med_bay: bool) -> float:
    """Compute total fitness penalty from chronic conditions."""
    penalty = 0.0
    for cond in health.chronic_conditions:
        if cond == "bone_loss":
            penalty += BONE_LOSS_FITNESS_PENALTY
        elif cond == "radiation_damage":
            penalty += RADIATION_DAMAGE_FITNESS_PENALTY

    # Medicine and med_bay mitigate condition effects
    if medicine_level > 0.3:
        penalty *= MEDICINE_HEALING_FACTOR + (1 - MEDICINE_HEALING_FACTOR) * (1 - medicine_level)
    if has_med_bay:
        penalty *= MED_BAY_CONDITION_FACTOR

    return penalty


def health_death_modifier(health: HealthState) -> float:
    """Return death-rate multiplier based on health state.

    Returns 1.0 for healthy colonists, >1.0 for unhealthy ones.
    """
    modifier = 1.0

    # Low fitness increases death rate
    if health.fitness < 0.3:
        modifier += (0.3 - health.fitness) * 5.0  # up to +1.5 at fitness=0

    # High radiation increases death rate
    if health.radiation > 0.5:
        modifier += (health.radiation - 0.5) * 3.0  # up to +1.5 at radiation=1.0

    # Chronic conditions add small constant increase
    modifier += len(health.chronic_conditions) * 0.15

    return modifier


def health_death_cause(health: HealthState, birth_year: int,
                       current_year: int) -> str | None:
    """Return the dominant health-related death cause, or None if healthy.

    Used by engine's hazard-bucket system to assign cause of death.
    """
    age = health.biological_age(birth_year, current_year)

    # Build hazard scores
    hazards: dict[str, float] = {}

    if age >= OLD_AGE_MIN_EARTH_YEARS and health.fitness < 0.2:
        hazards["old_age"] = (age - OLD_AGE_MIN_EARTH_YEARS) * 0.05 + (0.2 - health.fitness) * 3.0

    if health.radiation >= RADIATION_LETHAL:
        hazards["radiation_sickness"] = (health.radiation - RADIATION_LETHAL) * 8.0

    if "radiation_damage" in health.chronic_conditions and health.fitness < 0.15:
        hazards["chronic_illness"] = (0.15 - health.fitness) * 5.0

    if "bone_loss" in health.chronic_conditions and health.fitness < 0.15:
        hazards["chronic_illness"] = hazards.get("chronic_illness", 0) + 1.0

    if not hazards:
        return None

    return max(hazards, key=hazards.get)  # type: ignore[arg-type]


# -- main tick ---------------------------------------------------------------

def tick_health(
    health_map: dict[str, HealthState],
    contexts: list[ColonistHealthContext],
    year: int,
    rng: random.Random,
) -> HealthTickResult:
    """Advance health state for all active colonists by one Mars year.

    Creates HealthState entries for new colonists not yet in the map.
    """
    new_conditions: list[dict] = []
    fitness_changes: dict[str, float] = {}
    radiation_changes: dict[str, float] = {}

    for ctx in contexts:
        cid = ctx.colonist_id

        # Initialize health state for new colonists
        if cid not in health_map:
            health_map[cid] = HealthState(
                fitness=0.85 + rng.gauss(0, 0.03),
                radiation=rng.uniform(0.0, 0.02),
                initial_age=rng.uniform(25, 35),
            )

        health = health_map[cid]
        fitness_before = health.fitness
        radiation_before = health.radiation

        # 1. Aging
        aging_loss = compute_aging_delta(health, ctx.birth_year, year)
        health.fitness = _clamp(health.fitness - aging_loss)

        # 2. Radiation accumulation
        rad_delta = compute_radiation_delta(ctx)
        health.radiation = _clamp(health.radiation + rad_delta)

        # 3. Check for new chronic conditions
        acquired = check_new_conditions(health, ctx, rng)
        for cond in acquired:
            health.chronic_conditions.append(cond)
            new_conditions.append({
                "colonist_id": cid, "condition": cond, "year": year,
            })

        # 4. Apply chronic condition penalties
        condition_penalty = compute_condition_penalty(
            health, ctx.medicine_level, ctx.has_med_bay)
        health.fitness = _clamp(health.fitness - condition_penalty)

        # 5. Natural recovery when medicine is available
        if ctx.medicine_level > 0.5 and health.fitness < 0.7:
            recovery = 0.01 * ctx.medicine_level
            if ctx.has_med_bay:
                recovery *= 1.5
            health.fitness = _clamp(health.fitness + recovery)

        fitness_changes[cid] = health.fitness - fitness_before
        radiation_changes[cid] = health.radiation - radiation_before

    return HealthTickResult(
        new_conditions=new_conditions,
        fitness_changes=fitness_changes,
        radiation_changes=radiation_changes,
    )
