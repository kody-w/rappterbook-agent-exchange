"""
Ecology organ for Mars-100 colony simulation (engine v9.0).

Models the local biosphere around the colony habitat: atmospheric
composition, soil fertility, temperature regulation, and water
availability.  Scope is *local controlled biosphere* (dome-adjacent
terraforming), not planetary engineering.

Key dynamics:
  - Terraforming effort each year nudges atmospheric/soil/water state
  - Infrastructure (greenhouse, soil amendment) amplifies change rate
  - Ecological events (microbial bloom, contamination, frost shock)
  - Tipping points unlock narrative biome stages (irreversible milestones)
  - Continuous state drives modifiers for resources and psychology
  - 1-year lag: this year's effort → next year's environment

Engine v9.0.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# -- constants ---------------------------------------------------------------

TERRAFORM_EFFORT_BASE = 0.004
RESEARCH_EFFORT_BASE = 0.002

ATMOSPHERE_DECAY = 0.001
TEMPERATURE_DECAY = 0.002
SOIL_DECAY = 0.0005
WATER_DECAY = 0.001

BIOME_THRESHOLDS: list[dict[str, float]] = [
    {},
    {"soil_fertility": 0.10, "temperature": 0.08},
    {"o2_fraction": 0.04, "soil_fertility": 0.18, "temperature": 0.12},
    {"o2_fraction": 0.08, "soil_fertility": 0.28, "water_coverage": 0.04},
    {"o2_fraction": 0.13, "soil_fertility": 0.40, "temperature": 0.25,
     "water_coverage": 0.10},
    {"o2_fraction": 0.17, "soil_fertility": 0.55, "water_coverage": 0.18,
     "temperature": 0.35},
]

BIOME_NAMES = ["barren", "microbes", "lichen", "moss", "grassland", "forest"]

TIPPING_POINTS: dict[str, dict[str, float]] = {
    "first_microbes": {"soil_fertility": 0.10, "temperature": 0.08},
    "lichen_age":     {"o2_fraction": 0.04, "soil_fertility": 0.18},
    "water_cycle":    {"water_coverage": 0.06, "temperature": 0.15},
    "moss_era":       {"o2_fraction": 0.08, "soil_fertility": 0.28},
    "breathable_air": {"atmosphere_pressure": 0.40, "o2_fraction": 0.16},
    "open_sky":       {"atmosphere_pressure": 0.50, "o2_fraction": 0.19,
                       "temperature": 0.30},
}

ECOLOGY_EVENTS = [
    {"name": "microbial_bloom", "prob": 0.08, "min_biome": 1,
     "effects": {"soil_fertility": 0.02, "o2_fraction": 0.005}},
    {"name": "greenhouse_spike", "prob": 0.06, "min_biome": 0,
     "effects": {"temperature": 0.015, "atmosphere_pressure": 0.005}},
    {"name": "soil_contamination", "prob": 0.05, "min_biome": 1,
     "effects": {"soil_fertility": -0.03}},
    {"name": "frost_shock", "prob": 0.07, "min_biome": 0,
     "effects": {"temperature": -0.02, "soil_fertility": -0.01}},
    {"name": "aquifer_discovery", "prob": 0.04, "min_biome": 0,
     "effects": {"water_coverage": 0.03}},
    {"name": "algae_die_off", "prob": 0.04, "min_biome": 2,
     "effects": {"o2_fraction": -0.01, "soil_fertility": -0.01}},
    {"name": "symbiotic_emergence", "prob": 0.03, "min_biome": 2,
     "effects": {"soil_fertility": 0.025, "o2_fraction": 0.008}},
]


# -- data classes ------------------------------------------------------------

@dataclass
class Biosphere:
    """Local biosphere state around the Mars colony.

    All values normalized 0.0-1.0 where 1.0 = Earth-like conditions.
    """
    atmosphere_pressure: float = 0.006
    o2_fraction: float = 0.001
    temperature: float = 0.0
    soil_fertility: float = 0.01
    water_coverage: float = 0.0
    biome_level: int = 0
    tipping_points_hit: list[str] = field(default_factory=list)

    def breathable_quality(self) -> float:
        """Effective breathable air quality: pressure * O2 fraction."""
        return self.atmosphere_pressure * self.o2_fraction

    def habitability_score(self) -> float:
        """Overall habitability 0-1, weighted average of all axes."""
        return (self.atmosphere_pressure * 0.15
                + self.o2_fraction * 0.20
                + self.temperature * 0.20
                + self.soil_fertility * 0.25
                + self.water_coverage * 0.20)

    def biome_name(self) -> str:
        """Human-readable name for current biome level."""
        if 0 <= self.biome_level < len(BIOME_NAMES):
            return BIOME_NAMES[self.biome_level]
        return "unknown"

    def clamp(self) -> None:
        """Clamp all values to valid range."""
        self.atmosphere_pressure = max(0.0, min(1.0, self.atmosphere_pressure))
        self.o2_fraction = max(0.0, min(1.0, self.o2_fraction))
        self.temperature = max(0.0, min(1.0, self.temperature))
        self.soil_fertility = max(0.0, min(1.0, self.soil_fertility))
        self.water_coverage = max(0.0, min(1.0, self.water_coverage))
        self.biome_level = max(0, min(5, self.biome_level))

    def to_dict(self) -> dict[str, Any]:
        return {
            "atmosphere_pressure": round(self.atmosphere_pressure, 6),
            "o2_fraction": round(self.o2_fraction, 6),
            "temperature": round(self.temperature, 6),
            "soil_fertility": round(self.soil_fertility, 6),
            "water_coverage": round(self.water_coverage, 6),
            "biome_level": self.biome_level,
            "biome_name": self.biome_name(),
            "breathable_quality": round(self.breathable_quality(), 6),
            "habitability_score": round(self.habitability_score(), 6),
            "tipping_points_hit": list(self.tipping_points_hit),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Biosphere:
        if not d:
            return cls()
        bio = cls(
            atmosphere_pressure=d.get("atmosphere_pressure", 0.006),
            o2_fraction=d.get("o2_fraction", 0.001),
            temperature=d.get("temperature", 0.0),
            soil_fertility=d.get("soil_fertility", 0.01),
            water_coverage=d.get("water_coverage", 0.0),
            biome_level=d.get("biome_level", 0),
            tipping_points_hit=list(d.get("tipping_points_hit", [])),
        )
        bio.clamp()
        return bio


@dataclass
class EcologyEvent:
    """An ecological event that occurred this year."""
    name: str
    year: int
    effects: dict[str, float]

    def to_dict(self) -> dict:
        return {"name": self.name, "year": self.year,
                "effects": {k: round(v, 6) for k, v in self.effects.items()}}


@dataclass
class EcologyTickResult:
    """Result of one year of ecological evolution."""
    year: int
    biosphere_snapshot: dict
    events: list[dict]
    new_tipping_points: list[str]
    biome_level: int
    biome_name: str
    habitability_score: float
    terraform_effort: float

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "biosphere": self.biosphere_snapshot,
            "events": self.events,
            "new_tipping_points": self.new_tipping_points,
            "biome_level": self.biome_level,
            "biome_name": self.biome_name,
            "habitability_score": round(self.habitability_score, 6),
            "terraform_effort": round(self.terraform_effort, 6),
        }


# -- pure helpers ------------------------------------------------------------

def _compute_biome_level(bio: Biosphere) -> int:
    """Compute current biome level from continuous state."""
    for level in range(len(BIOME_THRESHOLDS) - 1, 0, -1):
        reqs = BIOME_THRESHOLDS[level]
        if all(getattr(bio, attr) >= val for attr, val in reqs.items()):
            return level
    return 0


def _check_tipping_points(bio: Biosphere) -> list[str]:
    """Check for newly hit tipping points. Returns list of new ones."""
    new_points: list[str] = []
    for name, reqs in TIPPING_POINTS.items():
        if name in bio.tipping_points_hit:
            continue
        if all(getattr(bio, attr) >= val for attr, val in reqs.items()):
            bio.tipping_points_hit.append(name)
            new_points.append(name)
    return new_points


def _generate_ecology_events(
    bio: Biosphere, year: int, rng: random.Random,
) -> list[EcologyEvent]:
    """Generate random ecological events for this year."""
    events: list[EcologyEvent] = []
    for template in ECOLOGY_EVENTS:
        if bio.biome_level < template["min_biome"]:
            continue
        if rng.random() < template["prob"]:
            events.append(EcologyEvent(
                name=template["name"], year=year,
                effects=dict(template["effects"])))
    return events


def _apply_biome_feedback(bio: Biosphere) -> dict[str, float]:
    """Living biomes provide self-reinforcing ecological feedback."""
    if bio.biome_level <= 0:
        return {}
    factor = bio.biome_level * 0.001
    return {
        "o2_fraction": factor * 0.8,
        "soil_fertility": factor * 0.5,
        "water_coverage": factor * 0.3,
    }


# -- modifier computation (consumed by engine) --------------------------------

def compute_ecology_modifiers(bio: Biosphere) -> dict[str, float]:
    """Compute resource/psychology modifiers from current biosphere state.

    All modifiers are multiplicative (1.0 = no change).
    Uses continuous state for smooth transitions.
    """
    breathable = bio.breathable_quality()
    return {
        "food_production_mult": 1.0 + bio.soil_fertility * 0.25,
        "air_maintenance_mult": max(0.5, 1.0 - breathable * 2.0),
        "medicine_production_mult": 1.0 + bio.soil_fertility * 0.2,
        "power_consumption_mult": max(0.7, 1.0 - bio.temperature * 0.3),
        "morale_ecology_bonus": bio.biome_level * 0.02,
        "habitability": bio.habitability_score(),
    }


# -- main tick ---------------------------------------------------------------

def tick_ecology(
    bio: Biosphere,
    terraform_count: int,
    research_count: int,
    population: int,
    infra_completed: list[str],
    year: int,
    rng: random.Random,
) -> EcologyTickResult:
    """Run one year of ecological evolution. Mutates bio in place."""
    # 1. Compute terraforming effort
    effort = (terraform_count * TERRAFORM_EFFORT_BASE
              + research_count * RESEARCH_EFFORT_BASE)
    if "greenhouse" in infra_completed:
        effort *= 1.3
    if "soil_processor" in infra_completed:
        effort *= 1.2

    industrial_co2 = population * 0.0003

    # 2. Apply effort to biosphere axes
    bio.atmosphere_pressure += effort * 0.15 + industrial_co2
    bio.o2_fraction += effort * 0.20
    bio.temperature += effort * 0.08 + bio.atmosphere_pressure * 0.002
    soil_boost = effort * 0.25
    if "soil_processor" in infra_completed:
        soil_boost *= 1.4
    bio.soil_fertility += soil_boost
    water_gain = effort * 0.10 + bio.temperature * 0.001
    if "water_extractor" in infra_completed:
        water_gain *= 1.5
    bio.water_coverage += water_gain

    # 3. Natural decay
    bio.atmosphere_pressure -= ATMOSPHERE_DECAY
    bio.temperature -= TEMPERATURE_DECAY
    bio.soil_fertility -= SOIL_DECAY
    bio.water_coverage -= WATER_DECAY

    # 4. Biome self-reinforcement
    feedback = _apply_biome_feedback(bio)
    for attr, delta in feedback.items():
        setattr(bio, attr, getattr(bio, attr) + delta)

    # 5. Random ecological events
    eco_events = _generate_ecology_events(bio, year, rng)
    for event in eco_events:
        for attr, delta in event.effects.items():
            if hasattr(bio, attr):
                setattr(bio, attr, getattr(bio, attr) + delta)

    # 6. Clamp
    bio.clamp()

    # 7. Check tipping points and biome level
    new_tipping = _check_tipping_points(bio)
    bio.biome_level = _compute_biome_level(bio)

    return EcologyTickResult(
        year=year, biosphere_snapshot=bio.to_dict(),
        events=[e.to_dict() for e in eco_events],
        new_tipping_points=new_tipping, biome_level=bio.biome_level,
        biome_name=bio.biome_name(),
        habitability_score=bio.habitability_score(),
        terraform_effort=effort,
    )
