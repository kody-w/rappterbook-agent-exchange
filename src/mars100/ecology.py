"""
Ecology organ for Mars-100 colony simulation (engine v9.0).

Tracks long-term planetary transformation: atmosphere, soil, water table,
radiation.  These change VERY slowly (~0.001/year per terraforming action)
so that 100-year arcs show gradual terraforming progress.

Four state variables (stored):
  atmosphere_pressure — 0.01 (thin CO2) → max ~0.15 (breathable-ish)
  soil_fertility      — 0.02 (barren regolith) → max ~0.30 (amended)
  water_table         — 0.10 (subsurface ice) → varies (extraction depletes)
  radiation_level     — 0.90 (no magnetosphere) → min ~0.50 (atmo shielding)

Two derived metrics (computed, not stored):
  biodiversity  — threshold-gated from atmosphere + soil
  habitability  — weighted composite of all four state vars

Ecology modifies colony BASE_PRODUCTION multipliers, creating feedback loops:
  better soil → more food → more population → more terraforming → better soil
  over-extraction → water table drops → food crisis → population decline

Milestones (threshold crossings) are fed into the culture organ as memories.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# -- constants ---------------------------------------------------------------

# Per-action ecology deltas (very slow — compounding over decades matters)
TERRAFORM_ATMOSPHERE_DELTA = 0.0012
TERRAFORM_SOIL_DELTA = 0.0005
FARM_SOIL_DELTA = 0.0008
EXPLORE_WATER_DELTA = 0.0004
RESEARCH_RADIATION_DELTA = 0.0003

# Natural processes
ATMOSPHERE_NATURAL_LOSS = 0.0002   # Mars bleeds atmosphere without magnetosphere
SOIL_NATURAL_DECAY = 0.0001       # Perchlorate re-contamination
WATER_NATURAL_RECHARGE = 0.0001   # Slow subsurface ice melting
RADIATION_ATMOSPHERE_FACTOR = 0.15 # Atmosphere shields radiation (radiation -= atmo * factor)

# Extraction costs to water table
WATER_EXTRACTION_PER_CAPITA = 0.0008  # Each colonist depletes water table slightly

# Biodiversity thresholds
MICROBE_THRESHOLD = {"atmosphere_pressure": 0.04, "soil_fertility": 0.08}
LICHEN_THRESHOLD = {"atmosphere_pressure": 0.07, "soil_fertility": 0.12}
PLANT_THRESHOLD = {"atmosphere_pressure": 0.10, "soil_fertility": 0.18}

# Infrastructure effects on ecology
GREENHOUSE_SOIL_BONUS = 0.0004       # greenhouse_dome accelerates soil improvement
WATER_RECYCLER_EXTRACTION_REDUCTION = 0.5  # halves water table depletion

# Production modifier ranges (ecology state → resource production multiplier)
MAX_FOOD_BONUS = 0.30        # +30% food production at max soil fertility
MAX_WATER_BONUS = 0.20       # +20% water production at high water table
MAX_POWER_PENALTY = -0.10    # dust from low atmosphere reduces solar
MAX_MEDICINE_BONUS = 0.15    # biodiversity enables medicinal compounds

# Milestones
MILESTONES = [
    {"id": "first_microbes", "label": "First Martian microbes established",
     "condition": {"atmosphere_pressure": 0.04, "soil_fertility": 0.08}},
    {"id": "lichen_bloom", "label": "Lichen bloom on exposed regolith",
     "condition": {"atmosphere_pressure": 0.07, "soil_fertility": 0.12}},
    {"id": "first_plants", "label": "First outdoor plants survive a full sol",
     "condition": {"atmosphere_pressure": 0.10, "soil_fertility": 0.18}},
    {"id": "water_accessible", "label": "Surface water detected in low basins",
     "condition": {"water_table": 0.20}},
    {"id": "radiation_safe_outdoors", "label": "Radiation drops below outdoor safety threshold",
     "condition": {"radiation_level_max": 0.65}},
    {"id": "atmosphere_thickening", "label": "Atmospheric pressure visibly rising — sky color shifting",
     "condition": {"atmosphere_pressure": 0.06}},
]


# -- data classes ------------------------------------------------------------

@dataclass
class EcologyState:
    """Long-term planetary ecology.  All values 0.0–1.0."""
    atmosphere_pressure: float = 0.01
    soil_fertility: float = 0.02
    water_table: float = 0.10
    radiation_level: float = 0.90
    milestones_achieved: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "atmosphere_pressure": round(self.atmosphere_pressure, 6),
            "soil_fertility": round(self.soil_fertility, 6),
            "water_table": round(self.water_table, 6),
            "radiation_level": round(self.radiation_level, 6),
            "biodiversity": round(self.biodiversity, 6),
            "habitability": round(self.habitability, 6),
            "milestones_achieved": list(self.milestones_achieved),
        }

    @classmethod
    def from_dict(cls, d: dict) -> EcologyState:
        return cls(
            atmosphere_pressure=d.get("atmosphere_pressure", 0.01),
            soil_fertility=d.get("soil_fertility", 0.02),
            water_table=d.get("water_table", 0.10),
            radiation_level=d.get("radiation_level", 0.90),
            milestones_achieved=list(d.get("milestones_achieved", [])),
        )

    @property
    def biodiversity(self) -> float:
        """Derived: threshold-gated from atmosphere and soil."""
        if (self.atmosphere_pressure < MICROBE_THRESHOLD["atmosphere_pressure"]
                or self.soil_fertility < MICROBE_THRESHOLD["soil_fertility"]):
            return 0.0
        # Linear ramp above microbe threshold
        atmo_excess = self.atmosphere_pressure - MICROBE_THRESHOLD["atmosphere_pressure"]
        soil_excess = self.soil_fertility - MICROBE_THRESHOLD["soil_fertility"]
        raw = (atmo_excess * 0.5 + soil_excess * 0.5) * 3.0
        return _clamp(raw)

    @property
    def habitability(self) -> float:
        """Derived: weighted composite of all state variables."""
        atmo_score = self.atmosphere_pressure * 2.0   # double-weighted
        soil_score = self.soil_fertility
        water_score = self.water_table
        rad_score = 1.0 - self.radiation_level        # lower radiation = better
        raw = (atmo_score * 0.35 + soil_score * 0.25
               + water_score * 0.20 + rad_score * 0.20)
        return _clamp(raw)


@dataclass
class EcologyTickResult:
    """Result of one year's ecology update."""
    state_before: dict = field(default_factory=dict)
    state_after: dict = field(default_factory=dict)
    new_milestones: list[str] = field(default_factory=list)
    habitability: float = 0.0
    biodiversity: float = 0.0
    resource_modifiers: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "state_before": self.state_before,
            "state_after": self.state_after,
            "new_milestones": self.new_milestones,
            "habitability": round(self.habitability, 6),
            "biodiversity": round(self.biodiversity, 6),
            "resource_modifiers": {k: round(v, 6) for k, v in self.resource_modifiers.items()},
        }


# -- pure helpers ------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def compute_ecology_production_modifiers(state: EcologyState) -> dict[str, float]:
    """Compute resource BASE_PRODUCTION multipliers from ecology state.

    Returns a dict of multipliers (1.0 = no change, 1.3 = +30%).
    Applied to colony.tick_resources via infra_modifiers style.
    """
    soil_bonus = state.soil_fertility * MAX_FOOD_BONUS / 0.30  # linear 0→MAX
    water_bonus = max(0.0, state.water_table - 0.05) * MAX_WATER_BONUS / 0.15
    # Low atmosphere = dust on solar panels
    atmo_penalty = max(MAX_POWER_PENALTY, -0.10 * (1.0 - state.atmosphere_pressure * 5))
    med_bonus = state.biodiversity * MAX_MEDICINE_BONUS / 0.15 if state.biodiversity > 0 else 0.0

    return {
        "food_production_mult": 1.0 + _clamp(soil_bonus, 0.0, MAX_FOOD_BONUS),
        "water_production_mult": 1.0 + _clamp(water_bonus, 0.0, MAX_WATER_BONUS),
        "power_production_mult": 1.0 + _clamp(atmo_penalty, MAX_POWER_PENALTY, 0.0),
        "medicine_production_mult": 1.0 + _clamp(med_bonus, 0.0, MAX_MEDICINE_BONUS),
    }


def check_milestones(state: EcologyState) -> list[str]:
    """Check which new milestones have been achieved."""
    new: list[str] = []
    for ms in MILESTONES:
        if ms["id"] in state.milestones_achieved:
            continue
        passed = True
        for key, threshold in ms["condition"].items():
            if key == "radiation_level_max":
                if state.radiation_level > threshold:
                    passed = False
            else:
                if getattr(state, key, 0.0) < threshold:
                    passed = False
        if passed:
            new.append(ms["id"])
    return new


def get_milestone_label(milestone_id: str) -> str:
    """Get the human-readable label for a milestone."""
    for ms in MILESTONES:
        if ms["id"] == milestone_id:
            return ms["label"]
    return milestone_id


# -- main tick ---------------------------------------------------------------

def tick_ecology(
    state: EcologyState,
    actions: dict[str, str],
    population: int,
    infra_completed: list[str],
    year: int,
    rng: random.Random,
) -> EcologyTickResult:
    """Advance ecology by one Martian year.  Mutates state in place."""
    before = state.to_dict()

    # Count relevant actions
    terraform_count = sum(1 for a in actions.values() if a == "terraform")
    farm_count = sum(1 for a in actions.values() if a == "farm")
    explore_count = sum(1 for a in actions.values() if a == "explore")
    research_count = sum(1 for a in actions.values() if a == "research")

    # --- atmosphere ---
    atmo_gain = terraform_count * TERRAFORM_ATMOSPHERE_DELTA
    atmo_loss = ATMOSPHERE_NATURAL_LOSS
    # Diminishing returns at higher pressures
    atmo_gain *= max(0.1, 1.0 - state.atmosphere_pressure * 3.0)
    noise = rng.gauss(0, 0.0002)
    state.atmosphere_pressure = _clamp(
        state.atmosphere_pressure + atmo_gain - atmo_loss + noise)

    # --- soil fertility ---
    soil_gain = (terraform_count * TERRAFORM_SOIL_DELTA
                 + farm_count * FARM_SOIL_DELTA)
    if "greenhouse_dome" in infra_completed:
        soil_gain += GREENHOUSE_SOIL_BONUS
    soil_loss = SOIL_NATURAL_DECAY
    noise = rng.gauss(0, 0.0001)
    state.soil_fertility = _clamp(
        state.soil_fertility + soil_gain - soil_loss + noise)

    # --- water table ---
    water_recharge = WATER_NATURAL_RECHARGE + explore_count * EXPLORE_WATER_DELTA
    water_extraction = population * WATER_EXTRACTION_PER_CAPITA
    if "water_recycler" in infra_completed:
        water_extraction *= WATER_RECYCLER_EXTRACTION_REDUCTION
    noise = rng.gauss(0, 0.0003)
    state.water_table = _clamp(
        state.water_table + water_recharge - water_extraction + noise)

    # --- radiation ---
    # Atmosphere provides natural shielding; also decays slowly with research
    atmo_shielding = state.atmosphere_pressure * RADIATION_ATMOSPHERE_FACTOR
    research_effect = research_count * RESEARCH_RADIATION_DELTA
    # Radiation can't increase (no new radiation sources on Mars)
    rad_delta = -(atmo_shielding * 0.01 + research_effect)
    noise = rng.gauss(0, 0.0001)
    state.radiation_level = _clamp(state.radiation_level + rad_delta + noise)

    # --- milestones ---
    new_milestones = check_milestones(state)
    for ms_id in new_milestones:
        state.milestones_achieved.append(ms_id)

    # --- compute derived values and modifiers ---
    resource_mods = compute_ecology_production_modifiers(state)

    return EcologyTickResult(
        state_before=before,
        state_after=state.to_dict(),
        new_milestones=new_milestones,
        habitability=state.habitability,
        biodiversity=state.biodiversity,
        resource_modifiers=resource_mods,
    )
