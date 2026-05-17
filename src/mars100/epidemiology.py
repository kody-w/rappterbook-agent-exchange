"""
Epidemiology organ for Mars-100 (engine v12.0).

Models disease outbreaks as colony-scale crises with biological realism:

  * Per-colonist SEIR state (Susceptible / Exposed / Infected / Recovered).
  * Multiple disease strains can coexist; a colonist tracks one active
    infection at a time but accumulates strain-specific immunity.
  * Spread is driven by the social graph: each infected colonist exposes
    high-trust contacts each year, with an effective R0 modulated by
    crowding, ecology pollution, and infrastructure.
  * Diseases emerge stochastically; emergence pressure rises with low
    medicine, immigration flux, and biosphere collapse.
  * Recovered colonists are immune for a strain-specific duration,
    after which they slowly re-enter the susceptible pool (drift).
  * Disease deaths are surfaced as a separate mortality channel so the
    engine's _check_death can prefer "plague" over generic causes.
  * Active outbreaks raise stress, push behavior toward rest, and
    erode faction cohesion.
  * When prevalence crosses an alarm threshold, the organ flags a
    quarantine_needed crisis that governance can consume.

RNG offset: seed + 13367
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S = "susceptible"
E = "exposed"
I = "infected"
R = "recovered"

VALID_STATUSES = (S, E, I, R)

DEFAULT_BASE_R0 = 1.8
EXPOSURE_TRUST_THRESHOLD = 0.25
MAX_DAILY_CONTACTS = 8
INCUBATION_PROGRESS = 0.85

DISEASE_DEATH_BASE = 0.04
DISEASE_DEATH_FRAILTY = 0.10
MAX_DISEASE_DEATH_RATE = 0.55

BASE_RECOVERY_RATE = 0.55
IMMUNITY_DECAY_PER_YEAR = 0.08

EMERGENCE_BASE_PROB = 0.012
EMERGENCE_MAX_PROB = 0.20
EMERGENCE_COOLDOWN_YEARS = 3

QUARANTINE_PREVALENCE = 0.20
PANDEMIC_PREVALENCE = 0.40

MAX_COHESION_PENALTY = 0.20
MAX_STRESS_BUMP = 0.18

MED_BAY_R0_MULT = 0.65
MED_BAY_MORTALITY_MULT = 0.55
AIR_RECYCLER_R0_MULT = 0.85
RESEARCH_LAB_R0_MULT = 0.90
GREENHOUSE_RECOVERY_BONUS = 0.05

_STRAIN_PREFIXES = [
    "Red", "Dust", "Olympus", "Tharsis", "Iron", "Cold",
    "Hollow", "Silent", "Bright", "Deep", "Crater", "Polar",
]
_STRAIN_SUFFIXES = [
    "Lung", "Fever", "Cough", "Shakes", "Bloom", "Sweat",
    "Hush", "Marrow", "Vertigo", "Ague", "Rot", "Chill",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Disease:
    id: str
    name: str
    origin_year: int
    r0: float = DEFAULT_BASE_R0
    mortality: float = DISEASE_DEATH_BASE
    duration: float = 1.0
    immunity_duration: float = 6.0
    vector: str = "airborne"
    extinct: bool = False
    total_infections: int = 0
    total_deaths: int = 0
    total_recoveries: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "origin_year": self.origin_year,
            "r0": round(self.r0, 4), "mortality": round(self.mortality, 4),
            "duration": round(self.duration, 4),
            "immunity_duration": round(self.immunity_duration, 4),
            "vector": self.vector, "extinct": self.extinct,
            "total_infections": self.total_infections,
            "total_deaths": self.total_deaths,
            "total_recoveries": self.total_recoveries,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Disease":
        return cls(
            id=d["id"], name=d["name"], origin_year=int(d["origin_year"]),
            r0=float(d.get("r0", DEFAULT_BASE_R0)),
            mortality=float(d.get("mortality", DISEASE_DEATH_BASE)),
            duration=float(d.get("duration", 1.0)),
            immunity_duration=float(d.get("immunity_duration", 6.0)),
            vector=d.get("vector", "airborne"),
            extinct=bool(d.get("extinct", False)),
            total_infections=int(d.get("total_infections", 0)),
            total_deaths=int(d.get("total_deaths", 0)),
            total_recoveries=int(d.get("total_recoveries", 0)),
        )


@dataclass
class HealthRecord:
    status: str = S
    active_strain: str | None = None
    years_in_status: int = 0
    immunities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "active_strain": self.active_strain,
            "years_in_status": self.years_in_status,
            "immunities": {k: round(v, 4) for k, v in self.immunities.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HealthRecord":
        return cls(
            status=d.get("status", S),
            active_strain=d.get("active_strain"),
            years_in_status=int(d.get("years_in_status", 0)),
            immunities={k: float(v) for k, v in d.get("immunities", {}).items()},
        )


@dataclass
class EpidemiologyState:
    diseases: dict[str, Disease] = field(default_factory=dict)
    health: dict[str, HealthRecord] = field(default_factory=dict)
    last_emergence_year: int = -10_000
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "diseases": {did: d.to_dict() for did, d in self.diseases.items()},
            "health": {cid: h.to_dict() for cid, h in self.health.items()},
            "last_emergence_year": self.last_emergence_year,
            "history": list(self.history[-200:]),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EpidemiologyState":
        return cls(
            diseases={did: Disease.from_dict(v)
                      for did, v in d.get("diseases", {}).items()},
            health={cid: HealthRecord.from_dict(v)
                    for cid, v in d.get("health", {}).items()},
            last_emergence_year=int(d.get("last_emergence_year", -10_000)),
            history=list(d.get("history", [])),
        )


@dataclass
class EpidemiologyYearContext:
    year: int
    active_colonists: list[dict]
    actions: dict[str, str]
    social_get: Callable[[str, str], Any]
    medicine_level: float
    population: int
    infrastructure_completed: list[str]
    ecology_pollution: float = 0.0
    immigrants_this_year: int = 0
    quarantine_active: bool = False


@dataclass
class InfectionEvent:
    colonist_id: str
    strain_id: str
    year: int
    via: str = "transmission"


@dataclass
class EpidemiologyTickResult:
    year: int
    new_strains: list[dict] = field(default_factory=list)
    new_infections: list[InfectionEvent] = field(default_factory=list)
    recoveries: list[str] = field(default_factory=list)
    deaths: list[dict] = field(default_factory=list)
    prevalence: float = 0.0
    susceptible_fraction: float = 1.0
    recovered_fraction: float = 0.0
    active_outbreaks: list[str] = field(default_factory=list)
    quarantine_needed: bool = False
    pandemic: bool = False
    cohesion_penalty: float = 0.0
    stress_bump: float = 0.0
    effective_r0_by_strain: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "new_strains": list(self.new_strains),
            "new_infections": [
                {"colonist_id": e.colonist_id, "strain_id": e.strain_id,
                 "year": e.year, "via": e.via}
                for e in self.new_infections],
            "recoveries": list(self.recoveries),
            "deaths": list(self.deaths),
            "prevalence": round(self.prevalence, 4),
            "susceptible_fraction": round(self.susceptible_fraction, 4),
            "recovered_fraction": round(self.recovered_fraction, 4),
            "active_outbreaks": list(self.active_outbreaks),
            "quarantine_needed": self.quarantine_needed,
            "pandemic": self.pandemic,
            "cohesion_penalty": round(self.cohesion_penalty, 4),
            "stress_bump": round(self.stress_bump, 4),
            "effective_r0_by_strain": {
                k: round(v, 4) for k, v in self.effective_r0_by_strain.items()},
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_record(state: EpidemiologyState, cid: str) -> HealthRecord:
    rec = state.health.get(cid)
    if rec is None:
        rec = HealthRecord()
        state.health[cid] = rec
    return rec


def _strain_name(rng) -> str:
    return f"{rng.choice(_STRAIN_PREFIXES)} {rng.choice(_STRAIN_SUFFIXES)}"


def _frailty(colonist: dict) -> float:
    stats = colonist.get("stats", {}) or {}
    age = float(colonist.get("age", 30))
    resolve = float(stats.get("resolve", 0.5))
    faith = float(stats.get("faith", 0.5))
    age_frailty = max(0.0, min(1.0, (age - 40.0) / 50.0))
    constitution = max(0.0, min(1.0, (resolve + faith) / 2.0))
    return max(0.0, min(1.0, 0.65 * age_frailty + 0.35 * (1.0 - constitution)))


def _infrastructure_r0_mult(completed: list[str]) -> float:
    mult = 1.0
    if "med_bay" in completed:
        mult *= MED_BAY_R0_MULT
    if "air_recycler" in completed:
        mult *= AIR_RECYCLER_R0_MULT
    if "research_lab" in completed:
        mult *= RESEARCH_LAB_R0_MULT
    return mult


def _infrastructure_mortality_mult(completed: list[str]) -> float:
    mult = 1.0
    if "med_bay" in completed:
        mult *= MED_BAY_MORTALITY_MULT
    return mult


def _crowding_factor(population: int) -> float:
    if population <= 1:
        return 0.5
    return 1.0 + math.log10(population) * 0.25


def compute_emergence_probability(
        medicine_level: float, ecology_pollution: float,
        immigrants: int, years_since_last: int) -> float:
    if years_since_last < EMERGENCE_COOLDOWN_YEARS:
        return 0.0
    prob = EMERGENCE_BASE_PROB
    prob += max(0.0, 0.5 - medicine_level) * 0.06
    prob += max(0.0, ecology_pollution) * 0.05
    prob += min(immigrants, 8) * 0.008
    return max(0.0, min(EMERGENCE_MAX_PROB, prob))


def _spawn_strain(state: EpidemiologyState, year: int, rng,
                  medicine_level: float, pollution: float) -> Disease:
    sid = f"strain-{year}-{rng.randint(1000, 9999)}"
    vector = rng.choices(
        ["airborne", "contact", "environmental"],
        weights=[0.55, 0.25, 0.20], k=1)[0]
    severity = 0.6 + rng.random() * 0.8
    severity += max(0.0, pollution - 0.3) * 0.6
    severity += max(0.0, 0.4 - medicine_level) * 0.5
    r0 = max(0.7, min(3.5, DEFAULT_BASE_R0 * (0.7 + 0.6 * rng.random())
                      + 0.4 * (severity - 1.0)))
    mortality = max(0.005, min(0.45,
                               DISEASE_DEATH_BASE * (0.5 + 1.4 * rng.random()) * severity))
    duration = max(0.3, 0.7 + rng.random() * 1.8)
    immunity = max(1.0, 3.0 + rng.random() * 8.0)
    name = _strain_name(rng)
    disease = Disease(
        id=sid, name=name, origin_year=year,
        r0=r0, mortality=mortality, duration=duration,
        immunity_duration=immunity, vector=vector)
    state.diseases[sid] = disease
    state.last_emergence_year = year
    state.history.append({
        "year": year, "event": "strain_emerged",
        "strain_id": sid, "name": name, "r0": round(r0, 3),
        "mortality": round(mortality, 3), "vector": vector,
    })
    return disease


def _seed_patient_zero(state: EpidemiologyState, disease: Disease,
                       active_ids: list[str], rng) -> str | None:
    candidates = [
        cid for cid in active_ids
        if _ensure_record(state, cid).status in (S, R)
    ]
    if not candidates:
        return None
    patient = rng.choice(candidates)
    rec = state.health[patient]
    rec.status = I
    rec.active_strain = disease.id
    rec.years_in_status = 0
    disease.total_infections += 1
    return patient


def _decay_immunities(rec: HealthRecord) -> None:
    if not rec.immunities:
        return
    expired = []
    for strain_id in list(rec.immunities):
        rec.immunities[strain_id] -= IMMUNITY_DECAY_PER_YEAR
        if rec.immunities[strain_id] <= 0:
            expired.append(strain_id)
    for sid in expired:
        rec.immunities.pop(sid, None)


def _is_immune(rec: HealthRecord, strain_id: str) -> bool:
    return rec.immunities.get(strain_id, 0.0) > 0.0


def _eligible_for_infection(rec: HealthRecord, strain_id: str) -> bool:
    if rec.status in (E, I):
        return False
    if _is_immune(rec, strain_id):
        return False
    return True


def compute_effective_r0(disease: Disease, ctx: EpidemiologyYearContext) -> float:
    r0 = disease.r0
    r0 *= _infrastructure_r0_mult(ctx.infrastructure_completed)
    r0 *= _crowding_factor(ctx.population)
    if disease.vector == "airborne" and "air_recycler" in ctx.infrastructure_completed:
        r0 *= 0.90
    if disease.vector == "environmental":
        r0 *= (1.0 + 0.4 * max(0.0, ctx.ecology_pollution - 0.2))
    rest_count = sum(1 for a in ctx.actions.values() if a == "rest")
    mediate_count = sum(1 for a in ctx.actions.values() if a == "mediate")
    if ctx.population > 0:
        rest_frac = rest_count / ctx.population
        med_frac = mediate_count / ctx.population
        r0 *= max(0.5, 1.0 - 0.4 * rest_frac - 0.2 * med_frac)
    r0 *= max(0.6, 1.0 - 0.4 * max(0.0, ctx.medicine_level - 0.5))
    if ctx.quarantine_active:
        r0 *= 0.45
    return max(0.05, r0)


# ---------------------------------------------------------------------------
# Tick
# ---------------------------------------------------------------------------

def tick_epidemiology(state: EpidemiologyState,
                      ctx: EpidemiologyYearContext,
                      rng) -> EpidemiologyTickResult:
    result = EpidemiologyTickResult(year=ctx.year)
    active_ids = [c["id"] for c in ctx.active_colonists if c.get("id")]
    pop = max(1, len(active_ids))

    alive_set = set(active_ids)
    for stale in [cid for cid in state.health if cid not in alive_set]:
        state.health.pop(stale, None)
    for cid in active_ids:
        _ensure_record(state, cid)

    # 1. Decay immunities, advance E -> I.
    for cid in active_ids:
        rec = state.health[cid]
        _decay_immunities(rec)
        if rec.status == E:
            rec.years_in_status += 1
            if rng.random() < INCUBATION_PROGRESS:
                rec.status = I
                rec.years_in_status = 0

    # 2. Spread from infected.
    infected_pairs: list[tuple[str, str]] = []
    for cid in active_ids:
        rec = state.health[cid]
        if rec.status != I or rec.active_strain is None:
            continue
        disease = state.diseases.get(rec.active_strain)
        if disease is None or disease.extinct:
            continue
        eff_r0 = compute_effective_r0(disease, ctx)
        result.effective_r0_by_strain[disease.id] = eff_r0
        contacts = []
        for other in active_ids:
            if other == cid:
                continue
            rel = ctx.social_get(cid, other)
            trust = getattr(rel, "trust", None)
            if trust is None:
                continue
            if trust >= EXPOSURE_TRUST_THRESHOLD:
                contacts.append((other, trust))
        contacts.sort(key=lambda x: x[1], reverse=True)
        contacts = contacts[:MAX_DAILY_CONTACTS]
        if not contacts:
            continue
        per_contact_p = min(0.75, eff_r0 / max(1, len(contacts)))
        for target, trust in contacts:
            t_rec = state.health[target]
            if not _eligible_for_infection(t_rec, disease.id):
                continue
            if rng.random() < per_contact_p * (0.5 + 0.5 * trust):
                infected_pairs.append((target, disease.id))

    seen: set[str] = set()
    for target, strain_id in infected_pairs:
        if target in seen:
            continue
        seen.add(target)
        rec = state.health[target]
        if rec.status in (E, I):
            continue
        if _is_immune(rec, strain_id):
            continue
        rec.status = E
        rec.active_strain = strain_id
        rec.years_in_status = 0
        disease = state.diseases.get(strain_id)
        if disease:
            disease.total_infections += 1
        result.new_infections.append(InfectionEvent(
            colonist_id=target, strain_id=strain_id, year=ctx.year))

    # 3. Resolve infected: recovery or death.
    mortality_mult = _infrastructure_mortality_mult(ctx.infrastructure_completed)
    recovery_bonus = (GREENHOUSE_RECOVERY_BONUS
                      if "greenhouse_dome" in ctx.infrastructure_completed else 0.0)
    colonist_by_id = {c["id"]: c for c in ctx.active_colonists}
    for cid in list(active_ids):
        rec = state.health[cid]
        if rec.status != I or rec.active_strain is None:
            continue
        disease = state.diseases.get(rec.active_strain)
        if disease is None:
            _record_recovery(state, cid, rec, disease, result)
            continue
        rec.years_in_status += 1
        colonist = colonist_by_id.get(cid, {})
        frailty = _frailty(colonist)
        death_p = disease.mortality * mortality_mult
        death_p += DISEASE_DEATH_FRAILTY * frailty
        if ctx.medicine_level < 0.3:
            death_p *= 1.4
        death_p = max(0.0, min(MAX_DISEASE_DEATH_RATE, death_p))
        if rng.random() < death_p:
            disease.total_deaths += 1
            result.deaths.append({
                "colonist_id": cid, "strain_id": disease.id,
                "strain_name": disease.name, "year": ctx.year,
                "cause": f"plague ({disease.name})",
            })
            rec.status = R
            rec.active_strain = None
            rec.years_in_status = 0
            rec.immunities[disease.id] = disease.immunity_duration
            continue
        recov_p = min(0.95, BASE_RECOVERY_RATE + recovery_bonus
                      + 0.10 * rec.years_in_status)
        if rec.years_in_status >= max(1, math.ceil(disease.duration)):
            recov_p = max(recov_p, 0.85)
        if rng.random() < recov_p:
            _record_recovery(state, cid, rec, disease, result)

    # 4. Emergence.
    yrs_since = ctx.year - state.last_emergence_year
    p_emerge = compute_emergence_probability(
        ctx.medicine_level, ctx.ecology_pollution,
        ctx.immigrants_this_year, yrs_since)
    if rng.random() < p_emerge:
        disease = _spawn_strain(
            state, ctx.year, rng, ctx.medicine_level, ctx.ecology_pollution)
        patient = _seed_patient_zero(state, disease, active_ids, rng)
        if patient is not None:
            result.new_infections.append(InfectionEvent(
                colonist_id=patient, strain_id=disease.id,
                year=ctx.year, via="emergence"))
        result.new_strains.append(disease.to_dict())

    # 5. Extinct strains.
    active_strain_ids = {
        rec.active_strain for rec in state.health.values()
        if rec.active_strain is not None
    }
    for disease in state.diseases.values():
        if disease.extinct:
            continue
        if disease.id in active_strain_ids:
            continue
        if disease.origin_year < ctx.year - 1:
            disease.extinct = True
            state.history.append({
                "year": ctx.year, "event": "strain_extinct",
                "strain_id": disease.id, "name": disease.name,
                "total_infections": disease.total_infections,
                "total_deaths": disease.total_deaths,
            })

    # 6. Summary.
    infected_now = sum(1 for r in state.health.values() if r.status == I)
    susceptible_now = sum(1 for r in state.health.values() if r.status == S)
    recovered_now = sum(1 for r in state.health.values() if r.status == R)
    result.prevalence = infected_now / pop
    result.susceptible_fraction = susceptible_now / pop
    result.recovered_fraction = recovered_now / pop
    result.active_outbreaks = sorted(active_strain_ids)
    result.quarantine_needed = result.prevalence >= QUARANTINE_PREVALENCE
    result.pandemic = result.prevalence >= PANDEMIC_PREVALENCE
    result.cohesion_penalty = min(MAX_COHESION_PENALTY, result.prevalence * 0.55)
    result.stress_bump = min(MAX_STRESS_BUMP, result.prevalence * 0.50)

    if result.pandemic:
        state.history.append({
            "year": ctx.year, "event": "pandemic_declared",
            "prevalence": round(result.prevalence, 3),
            "strains": list(result.active_outbreaks),
        })
    return result


def _record_recovery(state: EpidemiologyState, cid: str,
                     rec: HealthRecord, disease: Disease | None,
                     result: EpidemiologyTickResult) -> None:
    rec.status = R
    if disease is not None:
        rec.immunities[disease.id] = disease.immunity_duration
        disease.total_recoveries += 1
    rec.active_strain = None
    rec.years_in_status = 0
    result.recoveries.append(cid)


# ---------------------------------------------------------------------------
# Public hooks for other organs
# ---------------------------------------------------------------------------

def colonist_is_infected(state: EpidemiologyState, cid: str) -> bool:
    rec = state.health.get(cid)
    return bool(rec and rec.status == I)


def colonist_strain(state: EpidemiologyState, cid: str) -> str | None:
    rec = state.health.get(cid)
    return rec.active_strain if rec else None


def colonist_health_summary(state: EpidemiologyState, cid: str) -> dict:
    rec = state.health.get(cid)
    if rec is None:
        return {"status": S, "active_strain": None,
                "immunities": {}, "years_in_status": 0}
    return rec.to_dict()


def behavior_action_bias(state: EpidemiologyState) -> dict[str, dict[str, float]]:
    bias: dict[str, dict[str, float]] = {}
    for cid, rec in state.health.items():
        if rec.status == I:
            bias[cid] = {"rest": 0.35, "sabotage": -0.15,
                         "explore": -0.20, "terraform": -0.10}
        elif rec.status == E:
            bias[cid] = {"rest": 0.10}
    return bias


def diplomacy_cohesion_penalty(result: EpidemiologyTickResult) -> float:
    return result.cohesion_penalty


def stress_bump(result: EpidemiologyTickResult) -> float:
    return result.stress_bump
