"""
Medicine organ for Mars-100 colony simulation (engine v9.0).

Tracks per-colonist health: immune strength, active diseases, aging.
Diseases spread through the social graph (contagion). Epidemics
emerge when 3+ colonists share a condition.

Integration hooks:
  - tick_medicine(): called after psychology, before death check
  - disease_death_check(): independent hazard (additive, not replacing)
  - med_bay tech reduces disease severity by 30% (via infra effects)
  - Stress (from psychology) suppresses immune strength

Design:
  - HealthState is PERSISTENT across years (not re-created)
  - Dedicated RNG stream (seed + 10007) — does not shift other organs
  - Contagion uses top-3 social contacts, not all trust > threshold
  - Epidemic threshold: 3+ colonists with same disease -> spread doubles
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# -- constants ---------------------------------------------------------------

AGING_BASE_RATE = 0.003
AGING_ACCELERATION_AGE = 50
AGING_ACCELERATION_MULT = 2.5

IMMUNE_RECOVERY_RATE = 0.04
IMMUNE_STRESS_THRESHOLD = 0.7
IMMUNE_STRESS_PENALTY = 0.15

CONTAGION_TOP_N = 3
CONTAGION_BASE_PROB = 0.15
EPIDEMIC_THRESHOLD = 3
EPIDEMIC_SPREAD_BONUS = 0.20

DISEASE_NATURAL_RECOVERY = 0.08
DISEASE_REST_BONUS = 0.06
DISEASE_MEDIATE_BONUS = 0.03
DISEASE_RESEARCH_COLONY_BONUS = 0.01

MED_BAY_SEVERITY_MULT = 0.7

DISEASE_DEATH_HEALTH_THRESHOLD = 0.15
DISEASE_DEATH_BASE_PROB = 0.08
DISEASE_DEATH_LOW_MEDICINE_MULT = 2.0

CHRONIC_PROBABILITY = 0.10
CHRONIC_MIN_SEVERITY = 0.15


# -- disease types -----------------------------------------------------------

DISEASE_TYPES: dict[str, dict[str, Any]] = {
    "respiratory": {
        "base_severity": 0.30,
        "base_duration": 3,
        "triggers": ["dust_storm", "epidemic"],
    },
    "radiation_sickness": {
        "base_severity": 0.40,
        "base_duration": 2,
        "triggers": ["solar_flare"],
    },
    "infection": {
        "base_severity": 0.35,
        "base_duration": 4,
        "triggers": ["epidemic", "equipment_failure"],
    },
    "chronic_fatigue": {
        "base_severity": 0.20,
        "base_duration": 5,
        "triggers": [],
    },
    "injury": {
        "base_severity": 0.25,
        "base_duration": 2,
        "triggers": ["dust_storm", "equipment_failure", "ice_volcano"],
    },
}


# -- data classes ------------------------------------------------------------

@dataclass
class Disease:
    """An active disease condition on a colonist."""
    disease_type: str
    severity: float
    duration_remaining: int
    chronic: bool = False

    def to_dict(self) -> dict:
        return {
            "type": self.disease_type,
            "severity": round(self.severity, 4),
            "duration_remaining": self.duration_remaining,
            "chronic": self.chronic,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Disease":
        return cls(
            disease_type=d.get("type", "infection"),
            severity=d.get("severity", 0.3),
            duration_remaining=d.get("duration_remaining", 2),
            chronic=d.get("chronic", False),
        )


@dataclass
class HealthState:
    """Per-colonist persistent health state."""
    health: float = 1.0
    immune_strength: float = 0.7
    conditions: list[Disease] = field(default_factory=list)
    medical_history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "health": round(self.health, 4),
            "immune_strength": round(self.immune_strength, 4),
            "conditions": [c.to_dict() for c in self.conditions],
            "condition_count": len(self.conditions),
            "medical_history_count": len(self.medical_history),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HealthState":
        return cls(
            health=d.get("health", 1.0),
            immune_strength=d.get("immune_strength", 0.7),
            conditions=[Disease.from_dict(c) for c in d.get("conditions", [])],
            medical_history=d.get("medical_history", []),
        )


@dataclass
class ColonistMedContext:
    """Year context for one colonist's medical tick."""
    colonist_id: str
    age: int
    action: str
    stress: float
    event_names: list[str]
    top_contacts: list[str]
    medicine_resource: float
    has_med_bay: bool
    researchers: int


@dataclass
class MedicineTickResult:
    """Summary of one year's medical outcomes."""
    new_diseases: list[dict]
    recovered: list[dict]
    chronic_developed: list[dict]
    epidemics: list[str]
    avg_health: float
    avg_immune: float
    total_conditions: int

    def to_dict(self) -> dict:
        return {
            "new_diseases": self.new_diseases,
            "recovered": self.recovered,
            "chronic_developed": self.chronic_developed,
            "epidemics": self.epidemics,
            "avg_health": round(self.avg_health, 4),
            "avg_immune": round(self.avg_immune, 4),
            "total_conditions": self.total_conditions,
        }


# -- core functions ----------------------------------------------------------

def _apply_aging(health_state: HealthState, age: int) -> float:
    """Apply age-related health decline. Returns health delta (negative)."""
    rate = AGING_BASE_RATE
    if age > AGING_ACCELERATION_AGE:
        excess = age - AGING_ACCELERATION_AGE
        rate += AGING_BASE_RATE * (AGING_ACCELERATION_MULT - 1.0) * min(excess / 30.0, 1.0)
    health_state.health = max(0.0, health_state.health - rate)
    return -rate


def _apply_stress_immunity(health_state: HealthState, stress: float) -> None:
    """High stress suppresses immune strength."""
    if stress > IMMUNE_STRESS_THRESHOLD:
        penalty = IMMUNE_STRESS_PENALTY * (stress - IMMUNE_STRESS_THRESHOLD) / (1.0 - IMMUNE_STRESS_THRESHOLD)
        health_state.immune_strength = max(0.0, health_state.immune_strength - penalty)
    else:
        health_state.immune_strength = min(1.0, health_state.immune_strength + IMMUNE_RECOVERY_RATE * 0.5)


def _trigger_diseases_from_events(
    health_state: HealthState,
    event_names: list[str],
    immune: float,
    rng: random.Random,
) -> list[Disease]:
    """Check if events trigger new diseases."""
    new_diseases: list[Disease] = []
    for dtype, spec in DISEASE_TYPES.items():
        if any(ev in spec["triggers"] for ev in event_names):
            resist = immune * 0.5 + rng.random() * 0.5
            if resist < 0.5:
                already_has = any(c.disease_type == dtype for c in health_state.conditions)
                if not already_has:
                    severity = spec["base_severity"] * (1.0 - immune * 0.3) + rng.gauss(0, 0.05)
                    severity = max(0.05, min(1.0, severity))
                    new_diseases.append(Disease(
                        disease_type=dtype,
                        severity=severity,
                        duration_remaining=spec["base_duration"],
                    ))
    return new_diseases


def _trigger_stress_fatigue(
    health_state: HealthState,
    stress: float,
    rng: random.Random,
) -> "Disease | None":
    """Sustained high stress can cause chronic fatigue."""
    if stress > 0.8 and rng.random() < 0.12:
        already_has = any(c.disease_type == "chronic_fatigue" for c in health_state.conditions)
        if not already_has:
            return Disease(
                disease_type="chronic_fatigue",
                severity=DISEASE_TYPES["chronic_fatigue"]["base_severity"],
                duration_remaining=DISEASE_TYPES["chronic_fatigue"]["base_duration"],
            )
    return None


def _spread_contagion(
    health_map: dict[str, HealthState],
    contexts: list[ColonistMedContext],
    disease_counts: dict[str, int],
    rng: random.Random,
) -> list[tuple[str, Disease]]:
    """Spread diseases through social contacts."""
    new_infections: list[tuple[str, Disease]] = []
    for ctx in contexts:
        hs = health_map.get(ctx.colonist_id)
        if not hs or not hs.conditions:
            continue
        for contact_id in ctx.top_contacts:
            contact_hs = health_map.get(contact_id)
            if contact_hs is None:
                continue
            for disease in hs.conditions:
                if disease.disease_type in ("injury", "chronic_fatigue"):
                    continue
                already_has = any(c.disease_type == disease.disease_type
                                  for c in contact_hs.conditions)
                if already_has:
                    continue
                spread_prob = CONTAGION_BASE_PROB * disease.severity
                if disease_counts.get(disease.disease_type, 0) >= EPIDEMIC_THRESHOLD:
                    spread_prob += EPIDEMIC_SPREAD_BONUS
                spread_prob *= (1.0 - contact_hs.immune_strength * 0.5)
                if rng.random() < spread_prob:
                    spec = DISEASE_TYPES.get(disease.disease_type, {})
                    new_sev = disease.severity * 0.7 + rng.gauss(0, 0.05)
                    new_sev = max(0.05, min(1.0, new_sev))
                    new_infections.append((contact_id, Disease(
                        disease_type=disease.disease_type,
                        severity=new_sev,
                        duration_remaining=spec.get("base_duration", 3),
                    )))
    return new_infections


def _resolve_conditions(
    health_state: HealthState,
    action: str,
    has_med_bay: bool,
    colony_research_bonus: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Tick down conditions, apply treatment."""
    recovered: list[dict] = []
    chronic_developed: list[dict] = []
    surviving: list[Disease] = []

    for disease in health_state.conditions:
        recovery = DISEASE_NATURAL_RECOVERY
        if action == "rest":
            recovery += DISEASE_REST_BONUS
        if action == "mediate":
            recovery += DISEASE_MEDIATE_BONUS
        recovery += colony_research_bonus
        if has_med_bay:
            recovery *= (1.0 / MED_BAY_SEVERITY_MULT)

        disease.severity = max(0.0, disease.severity - recovery)
        if has_med_bay:
            disease.severity *= MED_BAY_SEVERITY_MULT

        disease.duration_remaining -= 1

        if disease.severity < 0.05 or (disease.duration_remaining <= 0 and not disease.chronic):
            if (disease.duration_remaining <= 0 and not disease.chronic
                    and rng.random() < CHRONIC_PROBABILITY):
                disease.chronic = True
                disease.severity = max(disease.severity, CHRONIC_MIN_SEVERITY)
                disease.duration_remaining = 99
                chronic_developed.append({"colonist_id": "pending", "disease": disease.disease_type})
                surviving.append(disease)
            else:
                recovered.append({"colonist_id": "pending", "disease": disease.disease_type})
                health_state.medical_history.append(disease.disease_type)
        else:
            surviving.append(disease)

    health_state.conditions = surviving
    return recovered, chronic_developed


def _update_health_from_conditions(health_state: HealthState) -> None:
    """Reduce health based on total disease burden."""
    if not health_state.conditions:
        health_state.health = min(1.0, health_state.health + 0.02)
        return
    total_burden = sum(d.severity for d in health_state.conditions)
    health_state.health = max(0.0, health_state.health - total_burden * 0.08)


def disease_death_check(
    health_state: HealthState,
    medicine_resource: float,
    rng: random.Random,
) -> "str | None":
    """Independent disease death hazard."""
    if health_state.health >= DISEASE_DEATH_HEALTH_THRESHOLD:
        return None
    if not health_state.conditions:
        return None
    prob = DISEASE_DEATH_BASE_PROB * (DISEASE_DEATH_HEALTH_THRESHOLD - health_state.health)
    if medicine_resource < 0.15:
        prob *= DISEASE_DEATH_LOW_MEDICINE_MULT
    if rng.random() < prob:
        worst = max(health_state.conditions, key=lambda d: d.severity)
        cause_map = {
            "respiratory": "respiratory failure",
            "radiation_sickness": "radiation poisoning",
            "infection": "sepsis",
            "chronic_fatigue": "organ failure",
            "injury": "complications from injury",
        }
        return cause_map.get(worst.disease_type, "disease")
    return None


def get_top_contacts(
    colonist_id: str,
    active_ids: list[str],
    social_edges: dict[str, dict[str, Any]],
    n: int = CONTAGION_TOP_N,
) -> list[str]:
    """Get top-N social contacts by trust score."""
    if colonist_id not in social_edges:
        return []
    contacts = []
    for other_id in active_ids:
        if other_id == colonist_id:
            continue
        rel = social_edges.get(colonist_id, {}).get(other_id)
        if rel is None:
            continue
        trust = getattr(rel, 'trust', 0.5) if hasattr(rel, 'trust') else rel.get('trust', 0.5)
        contacts.append((other_id, trust))
    contacts.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in contacts[:n]]


def tick_medicine(
    health_map: dict[str, HealthState],
    contexts: list[ColonistMedContext],
    year: int,
    rng: random.Random,
) -> MedicineTickResult:
    """Advance the medicine organ by one year."""
    all_new_diseases: list[dict] = []
    all_recovered: list[dict] = []
    all_chronic: list[dict] = []

    for ctx in contexts:
        if ctx.colonist_id not in health_map:
            health_map[ctx.colonist_id] = HealthState()

    for ctx in contexts:
        hs = health_map[ctx.colonist_id]
        _apply_aging(hs, ctx.age)
        _apply_stress_immunity(hs, ctx.stress)

    for ctx in contexts:
        hs = health_map[ctx.colonist_id]
        new_from_events = _trigger_diseases_from_events(
            hs, ctx.event_names, hs.immune_strength, rng)
        fatigue = _trigger_stress_fatigue(hs, ctx.stress, rng)
        if fatigue:
            new_from_events.append(fatigue)
        for d in new_from_events:
            hs.conditions.append(d)
            all_new_diseases.append({
                "colonist_id": ctx.colonist_id,
                "disease": d.disease_type,
                "severity": round(d.severity, 4),
            })

    disease_counts: dict[str, int] = {}
    for ctx in contexts:
        hs = health_map.get(ctx.colonist_id)
        if hs:
            for d in hs.conditions:
                disease_counts[d.disease_type] = disease_counts.get(d.disease_type, 0) + 1

    epidemics = [dtype for dtype, count in disease_counts.items()
                 if count >= EPIDEMIC_THRESHOLD]

    new_infections = _spread_contagion(health_map, contexts, disease_counts, rng)
    seen: set[tuple[str, str]] = set()
    for cid, disease in new_infections:
        key = (cid, disease.disease_type)
        if key in seen:
            continue
        seen.add(key)
        hs = health_map.get(cid)
        if hs and not any(c.disease_type == disease.disease_type for c in hs.conditions):
            hs.conditions.append(disease)
            all_new_diseases.append({
                "colonist_id": cid, "disease": disease.disease_type,
                "severity": round(disease.severity, 4), "source": "contagion",
            })

    research_bonus = sum(1 for ctx in contexts if ctx.action == "research") * DISEASE_RESEARCH_COLONY_BONUS
    for ctx in contexts:
        hs = health_map[ctx.colonist_id]
        rec, chron = _resolve_conditions(hs, ctx.action, ctx.has_med_bay, research_bonus, rng)
        for r in rec:
            r["colonist_id"] = ctx.colonist_id
        for c in chron:
            c["colonist_id"] = ctx.colonist_id
        all_recovered.extend(rec)
        all_chronic.extend(chron)

    for ctx in contexts:
        hs = health_map[ctx.colonist_id]
        _update_health_from_conditions(hs)
        hs.health = max(0.0, min(1.0, hs.health))
        hs.immune_strength = max(0.0, min(1.0, hs.immune_strength))

    all_health = [health_map[ctx.colonist_id].health for ctx in contexts]
    all_immune = [health_map[ctx.colonist_id].immune_strength for ctx in contexts]
    total_cond = sum(len(health_map[ctx.colonist_id].conditions) for ctx in contexts)

    return MedicineTickResult(
        new_diseases=all_new_diseases, recovered=all_recovered,
        chronic_developed=all_chronic, epidemics=epidemics,
        avg_health=sum(all_health) / max(1, len(all_health)),
        avg_immune=sum(all_immune) / max(1, len(all_immune)),
        total_conditions=total_cond,
    )
