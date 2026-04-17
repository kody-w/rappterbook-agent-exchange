"""
Medicine / health organ for Mars-100 colony simulation (engine v9.0).

Tracks per-colonist health: vitality (age-decaying baseline), injury load,
disease burden, and cumulative radiation exposure.  Colony-level medical
infrastructure determines treatment capacity per year.

Phase 1 scope:
  - HealthState per colonist (vitality, injury, disease, radiation)
  - MedicalState for colony (capacity, epidemic tracking)
  - tick_medicine(): age decay, injuries, epidemics, treatment
  - New death causes: old_age, radiation_sickness, epidemic, untreated_injury
  - ONE downstream hook: health-specific death checks in engine._check_death
  - Defer: productivity effects, psychology interaction, medical economics (v10+)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# -- constants ---------------------------------------------------------------

BASE_RADIATION_PER_YEAR = 0.005
RADIATION_DUST_STORM_BONUS = 0.008
RADIATION_SOLAR_FLARE_BONUS = 0.012

EPIDEMIC_BASE_PROB = 0.03
EPIDEMIC_CROWDING_THRESHOLD = 15
EPIDEMIC_DURATION_MIN = 2
EPIDEMIC_DURATION_MAX = 5
EPIDEMIC_DISEASE_PER_YEAR = 0.12

TREATMENT_INJURY_FACTOR = 0.70   # multiply injury_load by this
TREATMENT_DISEASE_FACTOR = 0.60  # multiply disease_load by this

BASE_MEDICAL_CAPACITY = 2
MED_BAY_CAPACITY_BONUS = 3

OLD_AGE_VITALITY_THRESHOLD = 0.08
OLD_AGE_MIN_BIO_AGE = 40
RADIATION_DEATH_THRESHOLD = 0.90
EPIDEMIC_DEATH_THRESHOLD = 0.80
INJURY_DEATH_THRESHOLD = 0.90

FOUNDER_AGE_OFFSET = 30
IMMIGRANT_AGE_OFFSET = 28
CHILD_AGE_OFFSET = 0


# -- data classes ------------------------------------------------------------

@dataclass
class HealthState:
    """Per-colonist health state.

    vitality:    1.0 = peak health, 0.0 = death's door.  Decays with age.
    injury_load: 0.0 = uninjured, 1.0 = critical injuries.
    disease_load: 0.0 = healthy, 1.0 = terminal disease.
    radiation:   0.0 = unexposed, 1.0 = lethal accumulation.  Never decreases.
    age_offset:  biological age at sim entry (founders ~30, immigrants ~28, children 0).
    """
    vitality: float = 1.0
    injury_load: float = 0.0
    disease_load: float = 0.0
    radiation: float = 0.0
    age_offset: int = FOUNDER_AGE_OFFSET

    def biological_age(self, current_year: int, birth_year: int) -> int:
        """Compute biological age from sim year and birth year."""
        return self.age_offset + (current_year - birth_year)

    def to_dict(self) -> dict:
        return {
            "vitality": round(self.vitality, 4),
            "injury_load": round(self.injury_load, 4),
            "disease_load": round(self.disease_load, 4),
            "radiation": round(self.radiation, 4),
            "age_offset": self.age_offset,
        }

    @classmethod
    def from_dict(cls, d: dict) -> HealthState:
        return cls(
            vitality=d.get("vitality", 1.0),
            injury_load=d.get("injury_load", 0.0),
            disease_load=d.get("disease_load", 0.0),
            radiation=d.get("radiation", 0.0),
            age_offset=d.get("age_offset", FOUNDER_AGE_OFFSET),
        )


@dataclass
class MedicalState:
    """Colony-wide medical infrastructure state."""
    medical_capacity: int = BASE_MEDICAL_CAPACITY
    epidemic_active: bool = False
    epidemic_year_started: int = -1
    epidemic_severity: float = 0.0
    epidemic_duration: int = 0
    treatments_given: int = 0

    def to_dict(self) -> dict:
        return {
            "medical_capacity": self.medical_capacity,
            "epidemic_active": self.epidemic_active,
            "epidemic_year_started": self.epidemic_year_started,
            "epidemic_severity": round(self.epidemic_severity, 4),
            "epidemic_duration": self.epidemic_duration,
            "treatments_given": self.treatments_given,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MedicalState:
        return cls(
            medical_capacity=d.get("medical_capacity", BASE_MEDICAL_CAPACITY),
            epidemic_active=d.get("epidemic_active", False),
            epidemic_year_started=d.get("epidemic_year_started", -1),
            epidemic_severity=d.get("epidemic_severity", 0.0),
            epidemic_duration=d.get("epidemic_duration", 0),
            treatments_given=d.get("treatments_given", 0),
        )


@dataclass
class ColonistHealthContext:
    """Per-colonist context for one medicine tick."""
    colonist_id: str
    birth_year: int
    action: str
    event_severity: float
    event_name: str
    has_med_bay: bool
    population: int
    food_level: float
    water_level: float
    medicine_level: float


@dataclass
class MedicineTickResult:
    """Result of one medicine tick across the colony."""
    treatments: list[str] = field(default_factory=list)
    epidemic_started: bool = False
    epidemic_ended: bool = False
    epidemic_active: bool = False
    avg_vitality: float = 1.0
    avg_radiation: float = 0.0
    injuries_this_year: int = 0

    def to_dict(self) -> dict:
        return {
            "treatments": list(self.treatments),
            "epidemic_started": self.epidemic_started,
            "epidemic_ended": self.epidemic_ended,
            "epidemic_active": self.epidemic_active,
            "avg_vitality": round(self.avg_vitality, 4),
            "avg_radiation": round(self.avg_radiation, 4),
            "injuries_this_year": self.injuries_this_year,
        }


# -- age decay ---------------------------------------------------------------

def _age_decay_rate(bio_age: int) -> float:
    """Piecewise vitality decay rate per year based on biological age.

    Young (<20):    0.001  (negligible)
    Adult (20-50):  0.002 .. 0.005  (gradual)
    Elder (50-70):  0.005 .. 0.013  (accelerating)
    Ancient (70+):  0.013 + 0.001 per year over 70  (steep)
    """
    if bio_age < 20:
        return 0.001
    if bio_age < 50:
        return 0.002 + (bio_age - 20) * 0.0001
    if bio_age < 70:
        return 0.005 + (bio_age - 50) * 0.0004
    return 0.013 + (bio_age - 70) * 0.001


def _compute_vitality(base_vitality: float, injury_load: float,
                      disease_load: float, radiation: float) -> float:
    """Derive effective vitality from health factors.

    Each stressor independently reduces vitality from its base (age-decayed)
    value.  The formula is multiplicative so stressors compound.
    """
    factor = ((1.0 - injury_load * 0.3)
              * (1.0 - disease_load * 0.4)
              * (1.0 - radiation * 0.2))
    return max(0.0, min(1.0, base_vitality * max(0.0, factor)))


# -- epidemic logic ----------------------------------------------------------

def _check_epidemic_start(med_state: MedicalState, population: int,
                          food: float, water: float, medicine: float,
                          rng: random.Random) -> bool:
    """Check whether an epidemic begins this year.

    Triggers require crowding AND resource scarcity (food, water, or medicine).
    """
    if med_state.epidemic_active:
        return False
    if population < EPIDEMIC_CROWDING_THRESHOLD:
        return False
    scarcity = (food < 0.3) or (water < 0.3) or (medicine < 0.3)
    if not scarcity:
        return False
    prob = EPIDEMIC_BASE_PROB + (population - EPIDEMIC_CROWDING_THRESHOLD) * 0.005
    prob += max(0.0, 0.3 - food) * 0.1
    prob += max(0.0, 0.3 - water) * 0.1
    prob += max(0.0, 0.3 - medicine) * 0.15
    return rng.random() < min(prob, 0.40)


def _tick_epidemic(med_state: MedicalState, year: int,
                   rng: random.Random) -> tuple[bool, bool]:
    """Advance epidemic state. Returns (started, ended)."""
    started = False
    ended = False
    if not med_state.epidemic_active:
        return started, ended
    years_active = year - med_state.epidemic_year_started
    if years_active >= med_state.epidemic_duration:
        med_state.epidemic_active = False
        med_state.epidemic_severity = 0.0
        ended = True
    return started, ended


# -- main tick ---------------------------------------------------------------

def tick_medicine(health_map: dict[str, HealthState],
                  med_state: MedicalState,
                  contexts: list[ColonistHealthContext],
                  year: int,
                  rng: random.Random) -> MedicineTickResult:
    """Advance the colony's medical state by one year.

    Steps:
      1. Update medical capacity from infrastructure.
      2. Age decay for each colonist.
      3. Apply event injuries.
      4. Accumulate radiation.
      5. Check / advance epidemic.
      6. Apply epidemic disease load.
      7. Treat the sickest colonists (up to capacity).
      8. Recompute vitality.
    """
    result = MedicineTickResult()
    if not contexts:
        return result

    # 1. Medical capacity: base + med_bay bonus if built
    has_med_bay = any(c.has_med_bay for c in contexts)
    med_state.medical_capacity = BASE_MEDICAL_CAPACITY + (
        MED_BAY_CAPACITY_BONUS if has_med_bay else 0)

    # Ensure every colonist has a HealthState
    for ctx in contexts:
        if ctx.colonist_id not in health_map:
            # Determine age offset from birth_year
            if ctx.birth_year == 0:
                offset = FOUNDER_AGE_OFFSET
            elif ctx.birth_year < year - 5:
                offset = IMMIGRANT_AGE_OFFSET
            else:
                offset = CHILD_AGE_OFFSET
            health_map[ctx.colonist_id] = HealthState(age_offset=offset)

    # 2-4. Per-colonist updates
    for ctx in contexts:
        hs = health_map[ctx.colonist_id]
        bio_age = hs.biological_age(year, ctx.birth_year)

        # 2. Age decay
        decay = _age_decay_rate(bio_age)
        hs.vitality = max(0.0, hs.vitality - decay)

        # 3. Event injuries
        if ctx.event_severity > 0.3:
            injury_chance = ctx.event_severity * 0.4
            if rng.random() < injury_chance:
                injury_amount = ctx.event_severity * rng.uniform(0.02, 0.08)
                hs.injury_load = min(1.0, hs.injury_load + injury_amount)
                result.injuries_this_year += 1

        # Action-based injury risk (explore, sabotage are dangerous)
        if ctx.action == "explore" and rng.random() < 0.08:
            hs.injury_load = min(1.0, hs.injury_load + rng.uniform(0.01, 0.04))
            result.injuries_this_year += 1
        elif ctx.action == "sabotage" and rng.random() < 0.12:
            hs.injury_load = min(1.0, hs.injury_load + rng.uniform(0.02, 0.06))
            result.injuries_this_year += 1

        # 4. Radiation accumulation (never decreases)
        rad_bonus = 0.0
        if ctx.event_name == "dust_storm":
            rad_bonus = RADIATION_DUST_STORM_BONUS
        elif ctx.event_name in ("solar_flare", "cosmic_event"):
            rad_bonus = RADIATION_SOLAR_FLARE_BONUS
        hs.radiation = min(1.0, hs.radiation + BASE_RADIATION_PER_YEAR + rad_bonus)

    # 5. Epidemic check / advance
    sample_ctx = contexts[0]
    if not med_state.epidemic_active:
        if _check_epidemic_start(
                med_state, sample_ctx.population,
                sample_ctx.food_level, sample_ctx.water_level,
                sample_ctx.medicine_level, rng):
            med_state.epidemic_active = True
            med_state.epidemic_year_started = year
            med_state.epidemic_severity = rng.uniform(0.3, 0.8)
            med_state.epidemic_duration = rng.randint(
                EPIDEMIC_DURATION_MIN, EPIDEMIC_DURATION_MAX)
            result.epidemic_started = True

    started, ended = _tick_epidemic(med_state, year, rng)
    if ended:
        result.epidemic_ended = True
    result.epidemic_active = med_state.epidemic_active

    # 6. Epidemic disease load
    if med_state.epidemic_active:
        for ctx in contexts:
            hs = health_map[ctx.colonist_id]
            dose = EPIDEMIC_DISEASE_PER_YEAR * med_state.epidemic_severity
            hs.disease_load = min(1.0, hs.disease_load + dose)

    # Natural disease recovery (slow, even without treatment)
    for ctx in contexts:
        hs = health_map[ctx.colonist_id]
        if not med_state.epidemic_active:
            hs.disease_load = max(0.0, hs.disease_load - 0.03)

    # 7. Medical treatment: prioritize sickest colonists
    treatable = sorted(
        contexts,
        key=lambda c: health_map[c.colonist_id].vitality)
    treated_count = 0
    for ctx in treatable:
        if treated_count >= med_state.medical_capacity:
            break
        hs = health_map[ctx.colonist_id]
        needs_treatment = hs.injury_load > 0.05 or hs.disease_load > 0.05
        if not needs_treatment:
            continue
        # Treatment effectiveness decreases with age
        bio_age = hs.biological_age(year, ctx.birth_year)
        age_penalty = max(0.5, 1.0 - max(0, bio_age - 50) * 0.01)
        hs.injury_load *= TREATMENT_INJURY_FACTOR * (1.0 / age_penalty)
        hs.injury_load = min(1.0, max(0.0, hs.injury_load))
        hs.disease_load *= TREATMENT_DISEASE_FACTOR * (1.0 / age_penalty)
        hs.disease_load = min(1.0, max(0.0, hs.disease_load))
        treated_count += 1
        med_state.treatments_given += 1
        result.treatments.append(ctx.colonist_id)

    # 8. Recompute vitality incorporating all stressors
    vitalities = []
    radiations = []
    for ctx in contexts:
        hs = health_map[ctx.colonist_id]
        # Base vitality already decayed by age; now apply stressor penalties
        hs.vitality = _compute_vitality(
            hs.vitality, hs.injury_load, hs.disease_load, hs.radiation)
        vitalities.append(hs.vitality)
        radiations.append(hs.radiation)

    result.avg_vitality = sum(vitalities) / len(vitalities)
    result.avg_radiation = sum(radiations) / len(radiations)
    return result


# -- death cause checks (called from engine._check_death) -------------------

def check_health_death(health: HealthState, current_year: int,
                       birth_year: int) -> str | None:
    """Check whether a colonist dies from a health-specific cause.

    Returns a cause string or None.  These are checked BEFORE the generic
    resource-based death roll in engine._check_death, so health deaths
    take priority when conditions are met.
    """
    bio_age = health.biological_age(current_year, birth_year)

    # Old age: very low vitality + advanced age
    if health.vitality < OLD_AGE_VITALITY_THRESHOLD and bio_age >= OLD_AGE_MIN_BIO_AGE:
        return "old_age"

    # Radiation sickness: cumulative exposure past threshold
    if health.radiation >= RADIATION_DEATH_THRESHOLD:
        return "radiation_sickness"

    # Untreated injury: critical injury load
    if health.injury_load >= INJURY_DEATH_THRESHOLD:
        return "untreated_injury"

    # Epidemic death: high disease during active epidemic
    if health.disease_load >= EPIDEMIC_DEATH_THRESHOLD:
        return "epidemic"

    return None
