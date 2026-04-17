"""
Ecology organ for Mars-100 — planetary biosphere dynamics (engine v10.0).

Models the Martian biosphere as a modifier layer on colony resources.
Terraforming actions improve soil, reduce perchlorate toxicity, and enable
flora growth.  Flora produces continuous resource bonuses (air, food).
Water table dynamics create a third resource feedback loop.

All values bounded [0, 1].  Effects are continuous — no hard thresholds
or permanent bonuses.  Self-sustaining status is reversible.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


DUST_EVENTS = frozenset({"dust_storm", "mega_storm", "solar_storm"})
WATER_EVENTS = frozenset({"ice_strike", "resource_strike"})
DAMAGE_EVENTS = frozenset({"equipment_failure", "habitat_breach"})


@dataclass
class Biosphere:
    """Planetary ecology state for Mars.

    Tracks soil, flora, water, and atmosphere as the colony terraforms.
    All fields normalised 0.0-1.0.
    """
    soil_fertility: float = 0.05
    perchlorate_level: float = 0.80
    flora_coverage: float = 0.0
    water_table: float = 0.10
    atmosphere_density: float = 0.01
    self_sustaining_year: int | None = None

    def health(self) -> float:
        """Overall biosphere health 0.0-1.0."""
        return (self.soil_fertility * 0.25
                + self.flora_coverage * 0.30
                + self.water_table * 0.20
                + (1.0 - self.perchlorate_level) * 0.15
                + self.atmosphere_density * 0.10)

    def resource_bonuses(self) -> dict[str, float]:
        """Continuous resource production bonuses from the biosphere."""
        usable_soil = self.soil_fertility * max(0.0, 1.0 - self.perchlorate_level)
        return {
            "air": self.flora_coverage * usable_soil * 0.04,
            "food": self.flora_coverage * usable_soil * 0.03,
            "water": self.water_table * 0.015,
        }

    def is_self_sustaining(self) -> bool:
        """Whether flora growth naturally exceeds baseline attrition."""
        if self.flora_coverage < 0.05:
            return False
        growth_potential = (self.soil_fertility * self.water_table
                           * max(0.0, 1.0 - self.perchlorate_level) * 0.15)
        base_attrition = 0.005
        return growth_potential > base_attrition

    def clamp(self) -> None:
        """Ensure all fields in [0, 1]."""
        self.soil_fertility = _clamp(self.soil_fertility)
        self.perchlorate_level = _clamp(self.perchlorate_level)
        self.flora_coverage = _clamp(self.flora_coverage)
        self.water_table = _clamp(self.water_table)
        self.atmosphere_density = _clamp(self.atmosphere_density)

    def to_dict(self) -> dict:
        """Serialise to JSON-safe dict."""
        return {
            "soil_fertility": round(self.soil_fertility, 6),
            "perchlorate_level": round(self.perchlorate_level, 6),
            "flora_coverage": round(self.flora_coverage, 6),
            "water_table": round(self.water_table, 6),
            "atmosphere_density": round(self.atmosphere_density, 6),
            "health": round(self.health(), 6),
            "self_sustaining": self.is_self_sustaining(),
            "self_sustaining_year": self.self_sustaining_year,
            "resource_bonuses": {k: round(v, 6)
                                 for k, v in self.resource_bonuses().items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> Biosphere:
        """Deserialise from dict."""
        if not d:
            return cls()
        return cls(
            soil_fertility=d.get("soil_fertility", 0.05),
            perchlorate_level=d.get("perchlorate_level", 0.80),
            flora_coverage=d.get("flora_coverage", 0.0),
            water_table=d.get("water_table", 0.10),
            atmosphere_density=d.get("atmosphere_density", 0.01),
            self_sustaining_year=d.get("self_sustaining_year"),
        )


@dataclass
class EcologyTickResult:
    """Result of one year of biosphere evolution."""
    soil_delta: float = 0.0
    perchlorate_delta: float = 0.0
    flora_delta: float = 0.0
    water_delta: float = 0.0
    atmosphere_delta: float = 0.0
    became_self_sustaining: bool = False
    lost_sustaining: bool = False
    resource_bonuses: dict[str, float] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to JSON-safe dict."""
        return {
            "soil_delta": round(self.soil_delta, 6),
            "perchlorate_delta": round(self.perchlorate_delta, 6),
            "flora_delta": round(self.flora_delta, 6),
            "water_delta": round(self.water_delta, 6),
            "atmosphere_delta": round(self.atmosphere_delta, 6),
            "became_self_sustaining": self.became_self_sustaining,
            "lost_sustaining": self.lost_sustaining,
            "resource_bonuses": {k: round(v, 6)
                                 for k, v in self.resource_bonuses.items()},
            "events": self.events,
        }


def tick_ecology(
    bio: Biosphere,
    year: int,
    terraformers: int,
    avg_terraform_skill: float,
    farmers: int,
    population: int,
    event_type: str,
    event_severity: float,
    infra_completed: list[str],
    rng: random.Random,
) -> EcologyTickResult:
    """Advance the biosphere by one Martian year.

    Mutates *bio* in place and returns a result describing what changed.
    All delta computations use pre-mutation values to avoid order dependence.
    """
    result = EcologyTickResult()
    was_sustaining = bio.self_sustaining_year is not None

    soil_0 = bio.soil_fertility
    perc_0 = bio.perchlorate_level
    flora_0 = bio.flora_coverage
    water_0 = bio.water_table
    atmo_0 = bio.atmosphere_density

    # --- 1. Perchlorate remediation ---
    perc_reduction = terraformers * avg_terraform_skill * 0.008
    perc_reduction += rng.gauss(0, 0.002)
    bio.perchlorate_level = _clamp(perc_0 - perc_reduction)

    # --- 2. Soil fertility ---
    usable_fraction = max(0.0, 1.0 - perc_0)
    fertility_gain = (terraformers * avg_terraform_skill * 0.005
                      * usable_fraction)
    fertility_gain += farmers * 0.002
    fertility_gain += rng.gauss(0, 0.001)
    bio.soil_fertility = _clamp(soil_0 + fertility_gain)

    # --- 3. Water table ---
    water_consumption = population * 0.001
    water_mining = 0.008
    if "water_recycler" in infra_completed:
        water_mining += 0.004
        result.events.append("water_recycler active")
    water_noise = rng.gauss(0, 0.002)
    water_delta = water_mining - water_consumption + water_noise
    if event_type in WATER_EVENTS:
        water_delta += event_severity * 0.03
        result.events.append(f"water event: +{event_severity * 0.03:.3f}")
    bio.water_table = _clamp(water_0 + water_delta)

    # --- 4. Flora growth ---
    can_grow = (soil_0 > 0.10 and water_0 > 0.05 and perc_0 < 0.70)
    if can_grow:
        growth_rate = (soil_0 * water_0
                       * max(0.0, 1.0 - perc_0) * 0.15)
        if "greenhouse_dome" in infra_completed:
            growth_rate *= 1.5
            result.events.append("greenhouse_dome boost")
        # Sigmoid damping near saturation
        growth_rate *= (1.0 - flora_0)
    else:
        growth_rate = 0.0

    base_attrition = population * 0.0005
    storm_damage = 0.0
    if event_type in DUST_EVENTS:
        storm_damage = event_severity * flora_0 * 0.15
        result.events.append(f"dust damage: -{storm_damage:.4f}")
    if event_type in DAMAGE_EVENTS:
        storm_damage += event_severity * flora_0 * 0.05

    flora_delta = growth_rate - base_attrition - storm_damage
    bio.flora_coverage = _clamp(flora_0 + flora_delta)

    # --- 5. Atmosphere (very slow) ---
    co2_fixation = flora_0 * 0.0005
    atmosphere_delta = co2_fixation + rng.gauss(0, 0.0001)
    bio.atmosphere_density = _clamp(atmo_0 + atmosphere_delta)

    # --- 6. Self-sustaining check (reversible) ---
    is_sustaining = bio.is_self_sustaining()
    if is_sustaining and not was_sustaining:
        bio.self_sustaining_year = year
        result.became_self_sustaining = True
        result.events.append(f"biosphere self-sustaining in year {year}")
    elif not is_sustaining and was_sustaining:
        bio.self_sustaining_year = None
        result.lost_sustaining = True
        result.events.append("biosphere lost self-sustaining status")

    # --- 7. Record deltas ---
    result.resource_bonuses = bio.resource_bonuses()
    result.soil_delta = bio.soil_fertility - soil_0
    result.perchlorate_delta = bio.perchlorate_level - perc_0
    result.flora_delta = bio.flora_coverage - flora_0
    result.water_delta = bio.water_table - water_0
    result.atmosphere_delta = bio.atmosphere_density - atmo_0

    bio.clamp()
    return result
