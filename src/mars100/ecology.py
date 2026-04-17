"""
Ecology organ for Mars-100 colony simulation (engine v9.0).

Models the colony's local environment as a living system that responds
to colonist actions.  NOT planetary-scale terraforming — this is the
immediate biosphere the colony inhabits: greenhouse health, soil
remediation, water recycling, atmospheric pockets inside habs.

Key dynamics:
  - Greenhouse biome health (productivity feedback)
  - Regolith remediation (soil detox over decades)
  - Closed-loop water resilience (recycling vs ice mining)
  - Local hab atmosphere quality (O2 pockets)
  - Milestones: edge-triggered, fire once

Ecology modifiers take effect NEXT tick (no same-tick feedback loops).
All state values have physical-unit semantics and enforced bounds.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# -- constants ---------------------------------------------------------------

# Soil remediation: perchlorate removal per terraforming action
REMEDIATION_PER_TERRAFORM = 0.008
REMEDIATION_PER_FARM = 0.003
REMEDIATION_NATURAL_DECAY = 0.001

# Greenhouse biome
BIOME_GROWTH_BASE = 0.005
BIOME_FARM_BONUS = 0.012
BIOME_NEGLECT_DECAY = 0.008
BIOME_MAX = 1.0

# Water cycle
ICE_MINING_DEPLETION = 0.006
ICE_RECYCLING_FLOOR = 0.10
ICE_DEEP_DRILL_UNLOCK = 0.20
ICE_RECOVERY_FROM_IMPORT = 0.02

# Hab atmosphere (local O2 quality in pressurized areas)
O2_BASE = 0.50
O2_FARM_BONUS = 0.008
O2_BIOME_BONUS_FACTOR = 0.01
O2_POPULATION_DRAIN = 0.003
O2_MIN = 0.10
O2_MAX = 1.0

# Temperature stability (hab internal)
TEMP_STABILITY_BASE = 0.70
TEMP_POWER_FACTOR = 0.15
TEMP_POP_DRAIN = 0.005
TEMP_MIN = 0.20
TEMP_MAX = 1.0

# Modifier caps (prevent runaway)
MAX_FOOD_BONUS = 0.30
MAX_WATER_PENALTY = 0.25
MAX_EVENT_DAMAGE_REDUCTION = 0.40
MAX_DEATH_RATE_REDUCTION = 0.20

# Milestone thresholds
MILESTONE_ALGAE_BLOOM = 0.40
MILESTONE_STABLE_BIOME = 0.60
MILESTONE_WATER_INDEPENDENCE = 0.80
MILESTONE_BREATHABLE_HAB = 0.85


# -- data classes ------------------------------------------------------------

@dataclass
class EcologyState:
    """Colony ecology state.  All values 0.0-1.0 unless noted.

    soil_toxicity:      1.0 = pristine Mars regolith (toxic), 0.0 = remediated
    biome_health:       0.0 = barren, 1.0 = thriving greenhouse ecosystem
    water_ice_reserve:  1.0 = untouched subsurface ice, 0.0 = depleted
    water_recycling:    0.0 = no recycling, 1.0 = perfect closed loop
    hab_o2_quality:     0.0 = dangerously low, 1.0 = fresh-air quality
    temp_stability:     0.0 = extreme fluctuations, 1.0 = perfectly regulated
    milestones:         list of edge-triggered milestone names (fire once)
    """
    soil_toxicity: float = 0.80
    biome_health: float = 0.10
    water_ice_reserve: float = 1.0
    water_recycling: float = 0.20
    hab_o2_quality: float = 0.50
    temp_stability: float = 0.70
    milestones: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "soil_toxicity": round(self.soil_toxicity, 4),
            "biome_health": round(self.biome_health, 4),
            "water_ice_reserve": round(self.water_ice_reserve, 4),
            "water_recycling": round(self.water_recycling, 4),
            "hab_o2_quality": round(self.hab_o2_quality, 4),
            "temp_stability": round(self.temp_stability, 4),
            "milestones": list(self.milestones),
        }

    @classmethod
    def from_dict(cls, d: dict) -> EcologyState:
        """Reconstruct from a serialized dict."""
        return cls(
            soil_toxicity=d.get("soil_toxicity", 0.80),
            biome_health=d.get("biome_health", 0.10),
            water_ice_reserve=d.get("water_ice_reserve", 1.0),
            water_recycling=d.get("water_recycling", 0.20),
            hab_o2_quality=d.get("hab_o2_quality", 0.50),
            temp_stability=d.get("temp_stability", 0.70),
            milestones=list(d.get("milestones", [])),
        )

    def clamp(self) -> None:
        """Enforce physical bounds on all state variables."""
        self.soil_toxicity = max(0.0, min(1.0, self.soil_toxicity))
        self.biome_health = max(0.0, min(BIOME_MAX, self.biome_health))
        self.water_ice_reserve = max(0.0, min(1.0, self.water_ice_reserve))
        self.water_recycling = max(0.0, min(1.0, self.water_recycling))
        self.hab_o2_quality = max(O2_MIN, min(O2_MAX, self.hab_o2_quality))
        self.temp_stability = max(TEMP_MIN, min(TEMP_MAX, self.temp_stability))

    def terraforming_score(self) -> float:
        """Composite progress score (0-1).  Derived, not stored."""
        return (
            (1.0 - self.soil_toxicity) * 0.25
            + self.biome_health * 0.25
            + self.water_recycling * 0.20
            + self.hab_o2_quality * 0.15
            + self.temp_stability * 0.15
        )


@dataclass
class EcologyTickResult:
    """Result of one year's ecology evolution."""
    soil_delta: float = 0.0
    biome_delta: float = 0.0
    ice_delta: float = 0.0
    recycling_delta: float = 0.0
    o2_delta: float = 0.0
    temp_delta: float = 0.0
    new_milestones: list[str] = field(default_factory=list)
    terraforming_score: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "soil_delta": round(self.soil_delta, 4),
            "biome_delta": round(self.biome_delta, 4),
            "ice_delta": round(self.ice_delta, 4),
            "recycling_delta": round(self.recycling_delta, 4),
            "o2_delta": round(self.o2_delta, 4),
            "temp_delta": round(self.temp_delta, 4),
            "new_milestones": list(self.new_milestones),
            "terraforming_score": round(self.terraforming_score, 4),
        }


# -- tick function -----------------------------------------------------------

def tick_ecology(
    state: EcologyState,
    action_counts: dict[str, int],
    population: int,
    resource_power: float,
    infra_completed: list[str],
    earth_ship_this_year: bool,
    rng: random.Random,
) -> EcologyTickResult:
    """Advance colony ecology by one Martian year.

    Args:
        state: mutable ecology state (modified in place)
        action_counts: maps action name -> count of colonists doing it
        population: number of living colonists
        resource_power: current power resource level (0-1)
        infra_completed: list of completed infrastructure tech IDs
        earth_ship_this_year: whether an Earth supply ship arrived
        rng: dedicated ecology RNG stream

    Returns:
        EcologyTickResult with deltas and new milestones.
    """
    result = EcologyTickResult()
    old = state.to_dict()

    # --- soil remediation ---
    terraform_count = action_counts.get("terraform", 0)
    farm_count = action_counts.get("farm", 0)
    soil_improvement = (
        terraform_count * REMEDIATION_PER_TERRAFORM
        + farm_count * REMEDIATION_PER_FARM
        + REMEDIATION_NATURAL_DECAY
        + rng.gauss(0, 0.002)
    )
    if "perchlorate_scrubber" in infra_completed:
        soil_improvement *= 1.5
    state.soil_toxicity -= soil_improvement

    # --- greenhouse biome ---
    biome_growth = BIOME_GROWTH_BASE + farm_count * BIOME_FARM_BONUS
    if "greenhouse_dome" in infra_completed:
        biome_growth *= 1.4
    toxicity_factor = max(0.3, 1.0 - state.soil_toxicity)
    biome_growth *= toxicity_factor
    if farm_count == 0:
        biome_growth -= BIOME_NEGLECT_DECAY
    biome_growth += rng.gauss(0, 0.003)
    state.biome_health += biome_growth

    # --- water ice reserve ---
    miners = terraform_count
    ice_loss = miners * ICE_MINING_DEPLETION + rng.gauss(0, 0.002)
    if "water_recycler" in infra_completed:
        ice_loss *= 0.5
        state.water_recycling = min(1.0, state.water_recycling + 0.015)
    if ("deep_drill" in infra_completed
            and state.water_ice_reserve < ICE_DEEP_DRILL_UNLOCK):
        ice_loss -= 0.01
    if earth_ship_this_year:
        ice_loss -= ICE_RECOVERY_FROM_IMPORT
    if (state.water_ice_reserve - ice_loss < ICE_RECYCLING_FLOOR
            and state.water_recycling > 0.5):
        ice_loss = max(0.0, state.water_ice_reserve - ICE_RECYCLING_FLOOR)
    state.water_ice_reserve -= ice_loss

    # Water recycling improves slowly with research
    research_count = action_counts.get("research", 0)
    state.water_recycling += research_count * 0.005 + rng.gauss(0, 0.002)

    # --- hab O2 quality ---
    o2_gain = farm_count * O2_FARM_BONUS + state.biome_health * O2_BIOME_BONUS_FACTOR
    o2_loss = population * O2_POPULATION_DRAIN
    if "o2_generator" in infra_completed:
        o2_gain += 0.01
    state.hab_o2_quality += o2_gain - o2_loss + rng.gauss(0, 0.003)

    # --- temperature stability ---
    power_bonus = resource_power * TEMP_POWER_FACTOR
    pop_drain = population * TEMP_POP_DRAIN
    temp_change = power_bonus - pop_drain + rng.gauss(0, 0.005)
    if "thermal_regulator" in infra_completed:
        temp_change += 0.02
    state.temp_stability += temp_change

    # --- clamp all values ---
    state.clamp()

    # --- check milestones (edge-triggered: fire once only) ---
    _check_milestones(state, result)

    # --- compute deltas ---
    now = state.to_dict()
    result.soil_delta = now["soil_toxicity"] - old["soil_toxicity"]
    result.biome_delta = now["biome_health"] - old["biome_health"]
    result.ice_delta = now["water_ice_reserve"] - old["water_ice_reserve"]
    result.recycling_delta = now["water_recycling"] - old["water_recycling"]
    result.o2_delta = now["hab_o2_quality"] - old["hab_o2_quality"]
    result.temp_delta = now["temp_stability"] - old["temp_stability"]
    result.terraforming_score = state.terraforming_score()

    return result


def _check_milestones(state: EcologyState, result: EcologyTickResult) -> None:
    """Check for milestone triggers.  Each fires exactly once."""
    candidates = [
        ("first_algae_bloom", state.soil_toxicity < MILESTONE_ALGAE_BLOOM),
        ("stable_biome", state.biome_health > MILESTONE_STABLE_BIOME),
        ("water_independence", state.water_recycling > MILESTONE_WATER_INDEPENDENCE),
        ("breathable_hab", state.hab_o2_quality > MILESTONE_BREATHABLE_HAB),
    ]
    for name, condition in candidates:
        if condition and name not in state.milestones:
            state.milestones.append(name)
            result.new_milestones.append(name)


# -- modifier function (feeds into NEXT tick's resource computation) ---------

def compute_ecology_modifiers(state: EcologyState) -> dict[str, float]:
    """Compute resource/event modifiers from ecology state.

    Returns dict compatible with the infra_modifiers pattern:
      food_production_mult, water_production_mult,
      event_damage_mult, death_rate_ecology_mult.

    These are applied in the NEXT tick (no same-tick feedback).
    """
    mods: dict[str, float] = {}

    # Biome health -> food production bonus (capped)
    food_bonus = min(MAX_FOOD_BONUS, state.biome_health * 0.35)
    mods["food_production_mult"] = 1.0 + food_bonus

    # Water recycling + ice -> water penalty when ice is low
    if state.water_ice_reserve < 0.3:
        penalty = min(MAX_WATER_PENALTY,
                      (0.3 - state.water_ice_reserve) * 0.5)
        penalty *= max(0.2, 1.0 - state.water_recycling)
        mods["water_production_mult"] = 1.0 - penalty
    else:
        mods["water_production_mult"] = 1.0

    # Temp stability + O2 -> reduced event damage
    stability_factor = (state.temp_stability + state.hab_o2_quality) / 2.0
    damage_reduction = min(MAX_EVENT_DAMAGE_REDUCTION,
                           stability_factor * 0.3)
    mods["event_damage_mult"] = 1.0 - damage_reduction

    # Overall ecology -> slight death rate improvement
    score = state.terraforming_score()
    death_reduction = min(MAX_DEATH_RATE_REDUCTION, score * 0.2)
    mods["death_rate_ecology_mult"] = 1.0 - death_reduction

    return mods


def compute_ecology_psych_effects(
    state: EcologyState,
    new_milestones: list[str],
) -> dict[str, float]:
    """Compute psychological effects from ecology state.

    Returns modifiers for the psychology organ:
      purpose_boost: positive when milestones achieved
      stress_from_ice: stress increase from ice depletion worry
      loneliness_reduction: thriving biome reduces isolation
    """
    effects: dict[str, float] = {}

    effects["purpose_boost"] = len(new_milestones) * 0.08

    if state.water_ice_reserve < 0.25:
        effects["stress_from_ice"] = (0.25 - state.water_ice_reserve) * 0.15
    else:
        effects["stress_from_ice"] = 0.0

    if state.biome_health > 0.4:
        effects["loneliness_reduction"] = (state.biome_health - 0.4) * 0.05
    else:
        effects["loneliness_reduction"] = 0.0

    return effects
