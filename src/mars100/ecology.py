"""
Ecology organ for Mars-100 colony simulation (engine v9.0).

Tracks planetary-scale terraforming progress: atmospheric composition,
soil fertility, water cycle, and biodiversity.  Colonist actions accumulate
ecological change over decades; tipping-point thresholds fire *once* when
crossed, creating regime shifts.

Feedback to resources is **delayed by one tick** — this year's ecology state
produces modifiers consumed by *next* year's resource tick, preventing
double-counting with the immediate skill bonuses that terraform/farm already
provide.

Key physical constraint: gas fractions always sum to 1.0 via an implicit
"other gases" (N₂ + Ar) remainder.

Phase 1 scope:
  - EcologyState: atmosphere, soil, water cycle, biodiversity
  - tick_ecology(): update ecology from actions and events
  - Tipping points: one-time milestone flags
  - compute_ecology_modifiers(): delayed resource modifiers
  - Defer: weather patterns, microclimate zones (v10+)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# -- constants ---------------------------------------------------------------

# Mars starting conditions (real values, fraction of Earth)
MARS_PRESSURE_ATM = 0.006          # ~0.6% of Earth
MARS_TEMP_C = -60.0
MARS_O2_FRACTION = 0.0013
MARS_CO2_FRACTION = 0.9532
MARS_OTHER_FRACTION = 1.0 - MARS_O2_FRACTION - MARS_CO2_FRACTION  # ~0.0455

# Per-year contribution rates from colonist actions
TERRAFORM_PRESSURE_PER_COLONIST = 0.00008   # very slow
TERRAFORM_O2_PER_COLONIST = 0.00003
FARM_SOIL_PER_COLONIST = 0.006
FARM_BIODIVERSITY_PER_COLONIST = 0.003
RESEARCH_PRESSURE_BONUS = 0.00002
RESEARCH_SOIL_BONUS = 0.002

# Temperature coupling: pressure raises temp (greenhouse effect)
TEMP_PER_PRESSURE_UNIT = 800.0    # 1 atm of CO₂ ≈ +800°C (Venus-like upper bound)
TEMP_FLOOR = -60.0
TEMP_CEILING = 30.0               # we cap at Earth-like

# Tipping-point thresholds
LICHEN_PRESSURE_THRESHOLD = 0.01          # outdoor lichen survival
LICHEN_TEMP_THRESHOLD = -40.0
WATER_CYCLE_TEMP_THRESHOLD = -30.0        # ice melt begins
MASK_BREATHING_PARTIAL_PRESSURE = 0.005   # O₂ partial pressure for mask breathing
OUTDOOR_FARMING_SOIL_THRESHOLD = 0.5
OUTDOOR_FARMING_TEMP_THRESHOLD = -10.0

# Modifier strengths (applied to NEXT year's resource tick)
MAX_AIR_MODIFIER = 0.15           # at most +15% air production
MAX_FOOD_MODIFIER = 0.20          # at most +20% food production
MAX_WATER_MODIFIER = 0.10         # at most +10% water production

# Dust storm intensity modifier: early warming can increase storms
DUST_STORM_TEMP_RANGE = (-50.0, -20.0)    # storms peak in this range


# -- data classes ------------------------------------------------------------

@dataclass
class EcologyMilestones:
    """One-time tipping-point flags (fire once, never reset)."""
    lichen_viable: bool = False
    water_cycle_active: bool = False
    mask_breathing: bool = False
    outdoor_farming: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "lichen_viable": self.lichen_viable,
            "water_cycle_active": self.water_cycle_active,
            "mask_breathing": self.mask_breathing,
            "outdoor_farming": self.outdoor_farming,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EcologyMilestones:
        return cls(
            lichen_viable=d.get("lichen_viable", False),
            water_cycle_active=d.get("water_cycle_active", False),
            mask_breathing=d.get("mask_breathing", False),
            outdoor_farming=d.get("outdoor_farming", False),
        )


@dataclass
class EcologyState:
    """Planetary ecology state — accumulates across years."""
    atmosphere_pressure: float = MARS_PRESSURE_ATM
    o2_fraction: float = MARS_O2_FRACTION
    co2_fraction: float = MARS_CO2_FRACTION
    temperature_avg: float = MARS_TEMP_C
    soil_fertility: float = 0.0
    water_cycle_strength: float = 0.0
    biodiversity_index: float = 0.0
    outdoor_coverage: float = 0.0
    milestones: EcologyMilestones = field(default_factory=EcologyMilestones)
    milestone_events: list[dict] = field(default_factory=list)

    @property
    def other_fraction(self) -> float:
        """N₂ + Ar + trace gases (implicit remainder)."""
        return max(0.0, 1.0 - self.o2_fraction - self.co2_fraction)

    @property
    def o2_partial_pressure(self) -> float:
        """Partial pressure of O₂ in atm."""
        return self.atmosphere_pressure * self.o2_fraction

    def to_dict(self) -> dict[str, Any]:
        return {
            "atmosphere_pressure": round(self.atmosphere_pressure, 6),
            "o2_fraction": round(self.o2_fraction, 6),
            "co2_fraction": round(self.co2_fraction, 6),
            "other_fraction": round(self.other_fraction, 6),
            "o2_partial_pressure": round(self.o2_partial_pressure, 6),
            "temperature_avg": round(self.temperature_avg, 2),
            "soil_fertility": round(self.soil_fertility, 4),
            "water_cycle_strength": round(self.water_cycle_strength, 4),
            "biodiversity_index": round(self.biodiversity_index, 4),
            "outdoor_coverage": round(self.outdoor_coverage, 4),
            "milestones": self.milestones.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> EcologyState:
        ms = EcologyMilestones.from_dict(d.get("milestones", {}))
        return cls(
            atmosphere_pressure=d.get("atmosphere_pressure", MARS_PRESSURE_ATM),
            o2_fraction=d.get("o2_fraction", MARS_O2_FRACTION),
            co2_fraction=d.get("co2_fraction", MARS_CO2_FRACTION),
            temperature_avg=d.get("temperature_avg", MARS_TEMP_C),
            soil_fertility=d.get("soil_fertility", 0.0),
            water_cycle_strength=d.get("water_cycle_strength", 0.0),
            biodiversity_index=d.get("biodiversity_index", 0.0),
            outdoor_coverage=d.get("outdoor_coverage", 0.0),
            milestones=ms,
        )


@dataclass
class EcologyTickResult:
    """Result of one year's ecology update."""
    pressure_delta: float = 0.0
    o2_delta: float = 0.0
    temp_delta: float = 0.0
    soil_delta: float = 0.0
    water_cycle_delta: float = 0.0
    biodiversity_delta: float = 0.0
    outdoor_coverage_delta: float = 0.0
    new_milestones: list[str] = field(default_factory=list)
    modifiers: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pressure_delta": round(self.pressure_delta, 6),
            "o2_delta": round(self.o2_delta, 6),
            "temp_delta": round(self.temp_delta, 2),
            "soil_delta": round(self.soil_delta, 4),
            "water_cycle_delta": round(self.water_cycle_delta, 4),
            "biodiversity_delta": round(self.biodiversity_delta, 4),
            "outdoor_coverage_delta": round(self.outdoor_coverage_delta, 4),
            "new_milestones": self.new_milestones,
            "modifiers": {k: round(v, 4) for k, v in self.modifiers.items()},
        }


# -- pure computation --------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def compute_terraforming_contribution(
    action_counts: dict[str, int],
    infra_completed: list[str],
) -> dict[str, float]:
    """Compute per-year ecological contributions from colonist actions.

    Returns deltas for pressure, o2_fraction, soil_fertility, biodiversity.
    """
    n_terraform = action_counts.get("terraform", 0)
    n_farm = action_counts.get("farm", 0)
    n_research = action_counts.get("research", 0)

    # Infrastructure tech bonuses
    has_greenhouse_upgrade = "advanced_greenhouse" in infra_completed
    has_atmo_processor = "atmospheric_processor" in infra_completed

    pressure_delta = (
        n_terraform * TERRAFORM_PRESSURE_PER_COLONIST
        + n_research * RESEARCH_PRESSURE_BONUS
    )
    if has_atmo_processor:
        pressure_delta *= 1.5

    o2_delta = n_terraform * TERRAFORM_O2_PER_COLONIST
    if has_atmo_processor:
        o2_delta *= 1.3

    soil_delta = (
        n_farm * FARM_SOIL_PER_COLONIST
        + n_research * RESEARCH_SOIL_BONUS
    )

    biodiversity_delta = n_farm * FARM_BIODIVERSITY_PER_COLONIST
    if has_greenhouse_upgrade:
        biodiversity_delta *= 1.4

    return {
        "pressure": pressure_delta,
        "o2": o2_delta,
        "soil": soil_delta,
        "biodiversity": biodiversity_delta,
    }


def compute_event_ecology_effects(
    events: list[dict],
) -> dict[str, float]:
    """Compute ecology effects from environmental events."""
    effects: dict[str, float] = {
        "soil": 0.0,
        "outdoor_coverage": 0.0,
        "biodiversity": 0.0,
    }
    for ev in events:
        name = ev.get("name", "")
        severity = ev.get("severity", 0.0)
        if name == "dust_storm":
            effects["outdoor_coverage"] -= severity * 0.02
            effects["soil"] -= severity * 0.002
        elif name == "resource_strike":
            effects["soil"] += 0.005
        elif name == "breakthrough":
            effects["biodiversity"] += 0.005
        elif name == "ice_volcano":
            effects["soil"] += severity * 0.003
        elif name == "cave_discovery":
            effects["biodiversity"] += 0.003
    return effects


def check_milestones(
    ecology: EcologyState,
    year: int,
) -> list[str]:
    """Check for newly crossed tipping-point thresholds.

    Returns list of milestone names that just fired this year.
    Each milestone fires exactly once.
    """
    fired: list[str] = []

    if (not ecology.milestones.lichen_viable
            and ecology.atmosphere_pressure >= LICHEN_PRESSURE_THRESHOLD
            and ecology.temperature_avg >= LICHEN_TEMP_THRESHOLD):
        ecology.milestones.lichen_viable = True
        fired.append("lichen_viable")
        ecology.milestone_events.append({
            "milestone": "lichen_viable", "year": year,
            "description": "Atmospheric conditions now support outdoor lichen growth.",
        })

    if (not ecology.milestones.water_cycle_active
            and ecology.temperature_avg >= WATER_CYCLE_TEMP_THRESHOLD):
        ecology.milestones.water_cycle_active = True
        fired.append("water_cycle_active")
        ecology.milestone_events.append({
            "milestone": "water_cycle_active", "year": year,
            "description": "Polar ice caps begin sublimating — a primitive water cycle emerges.",
        })

    if (not ecology.milestones.mask_breathing
            and ecology.o2_partial_pressure >= MASK_BREATHING_PARTIAL_PRESSURE):
        ecology.milestones.mask_breathing = True
        fired.append("mask_breathing")
        ecology.milestone_events.append({
            "milestone": "mask_breathing", "year": year,
            "description": "O₂ partial pressure sufficient for outdoor activity with breathing masks.",
        })

    if (not ecology.milestones.outdoor_farming
            and ecology.soil_fertility >= OUTDOOR_FARMING_SOIL_THRESHOLD
            and ecology.temperature_avg >= OUTDOOR_FARMING_TEMP_THRESHOLD):
        ecology.milestones.outdoor_farming = True
        fired.append("outdoor_farming")
        ecology.milestone_events.append({
            "milestone": "outdoor_farming", "year": year,
            "description": "Soil fertility and temperature support open-air agriculture.",
        })

    return fired


def compute_temperature(pressure: float, base_temp: float = MARS_TEMP_C) -> float:
    """Derive temperature from atmospheric pressure (greenhouse effect).

    Simple model: more atmosphere → warmer, with diminishing returns.
    """
    # Pressure increase above Mars baseline drives warming
    pressure_increase = max(0.0, pressure - MARS_PRESSURE_ATM)
    warming = TEMP_PER_PRESSURE_UNIT * pressure_increase
    # Diminishing returns: square root scaling prevents runaway
    if warming > 1.0:
        warming = warming ** 0.5
    return max(TEMP_FLOOR, min(TEMP_CEILING, base_temp + warming))


def compute_dust_storm_modifier(temperature: float) -> float:
    """Compute dust storm intensity modifier based on temperature.

    Early warming increases dust storm activity (more convection).
    Once past -20°C, storms diminish as atmosphere stabilizes.
    Returns a multiplier >= 0 applied to dust storm severity.
    """
    lo, hi = DUST_STORM_TEMP_RANGE
    if temperature < lo:
        return 1.0  # baseline cold Mars
    if temperature > hi:
        # Past peak: storms diminish
        decay = (temperature - hi) / 30.0
        return max(0.5, 1.3 - decay)
    # In the warming band: storms intensify
    t = (temperature - lo) / (hi - lo)
    return 1.0 + 0.3 * t


def compute_ecology_modifiers(ecology: EcologyState) -> dict[str, float]:
    """Produce resource modifiers from current ecology state.

    These are applied to the NEXT year's resource tick (delayed feedback).
    Returns modifier dict compatible with tick_resources infra_modifiers.
    """
    mods: dict[str, float] = {}

    # Air production bonus from atmosphere/biodiversity
    air_bonus = min(MAX_AIR_MODIFIER,
                    ecology.biodiversity_index * 0.1
                    + ecology.outdoor_coverage * 0.05)
    if air_bonus > 0.001:
        mods["air_ecology_bonus"] = air_bonus

    # Food production bonus from soil and water cycle
    food_bonus = min(MAX_FOOD_MODIFIER,
                     ecology.soil_fertility * 0.15
                     + ecology.water_cycle_strength * 0.05)
    if ecology.milestones.outdoor_farming:
        food_bonus = min(MAX_FOOD_MODIFIER, food_bonus + 0.05)
    if food_bonus > 0.001:
        mods["food_ecology_bonus"] = food_bonus

    # Water bonus from water cycle
    water_bonus = min(MAX_WATER_MODIFIER,
                      ecology.water_cycle_strength * 0.10)
    if water_bonus > 0.001:
        mods["water_ecology_bonus"] = water_bonus

    return mods


def _update_gas_fractions(
    ecology: EcologyState,
    o2_delta: float,
) -> None:
    """Update O₂ and CO₂ fractions maintaining the sum-to-1.0 invariant.

    When O₂ increases, CO₂ decreases (plants convert CO₂ → O₂).
    The "other gases" fraction stays approximately constant.
    """
    other = ecology.other_fraction
    new_o2 = _clamp(ecology.o2_fraction + o2_delta, 0.0, 1.0 - other - 0.001)
    new_co2 = max(0.001, 1.0 - new_o2 - other)
    ecology.o2_fraction = new_o2
    ecology.co2_fraction = new_co2


# -- main tick ---------------------------------------------------------------

def tick_ecology(
    ecology: EcologyState,
    action_counts: dict[str, int],
    events: list[dict],
    infra_completed: list[str],
    year: int,
) -> EcologyTickResult:
    """Run one year of ecological updates.  Mutates ecology in place.

    Returns result with deltas and delayed resource modifiers.
    """
    result = EcologyTickResult()

    # 1. Terraforming contributions from colonist actions
    contributions = compute_terraforming_contribution(action_counts, infra_completed)

    # 2. Event effects on ecology
    event_effects = compute_event_ecology_effects(events)

    # 3. Apply atmospheric changes
    pressure_delta = contributions["pressure"]
    ecology.atmosphere_pressure += pressure_delta
    ecology.atmosphere_pressure = max(MARS_PRESSURE_ATM,
                                       min(2.0, ecology.atmosphere_pressure))
    result.pressure_delta = pressure_delta

    # 4. Update gas fractions (O₂ up → CO₂ down, sum = 1.0)
    o2_delta = contributions["o2"]
    _update_gas_fractions(ecology, o2_delta)
    result.o2_delta = o2_delta

    # 5. Temperature derived from pressure
    old_temp = ecology.temperature_avg
    ecology.temperature_avg = compute_temperature(ecology.atmosphere_pressure)
    result.temp_delta = ecology.temperature_avg - old_temp

    # 6. Soil fertility
    soil_delta = contributions["soil"] + event_effects.get("soil", 0.0)
    ecology.soil_fertility = _clamp(ecology.soil_fertility + soil_delta)
    result.soil_delta = soil_delta

    # 7. Water cycle (driven by temperature milestones)
    if ecology.milestones.water_cycle_active:
        # Active water cycle: slowly strengthens
        wc_delta = 0.005 + ecology.soil_fertility * 0.002
    else:
        wc_delta = 0.0
    ecology.water_cycle_strength = _clamp(ecology.water_cycle_strength + wc_delta)
    result.water_cycle_delta = wc_delta

    # 8. Biodiversity
    bio_delta = (contributions["biodiversity"]
                 + event_effects.get("biodiversity", 0.0))
    if ecology.milestones.lichen_viable:
        bio_delta += ecology.outdoor_coverage * 0.002
    ecology.biodiversity_index = _clamp(ecology.biodiversity_index + bio_delta)
    result.biodiversity_delta = bio_delta

    # 9. Outdoor coverage (lichen/moss)
    coverage_delta = event_effects.get("outdoor_coverage", 0.0)
    if ecology.milestones.lichen_viable:
        coverage_delta += 0.003 + ecology.soil_fertility * 0.002
    ecology.outdoor_coverage = _clamp(ecology.outdoor_coverage + coverage_delta)
    result.outdoor_coverage_delta = coverage_delta

    # 10. Check milestones (one-time only)
    new_milestones = check_milestones(ecology, year)
    result.new_milestones = new_milestones

    # 11. Compute delayed resource modifiers for NEXT year
    result.modifiers = compute_ecology_modifiers(ecology)

    return result
