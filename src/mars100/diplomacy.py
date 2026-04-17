"""
Diplomacy organ for Mars-100 colony simulation (engine v9.0).

Models the colony's relationship with external entities discovered
through an evidence ladder: signal → corroboration → confirmation.

Entity sources (most to least emergent):
  1. Splinter settlements from exiled colonists
  2. Earth-announced secondary missions
  3. Anomalous radio signals (unknown origin)

Phase 1 scope:
  - Evidence-based entity discovery (no conjured NPCs)
  - Diplomatic stance and pressure on action weights
  - Sub-sim hook: colonists model diplomatic outcomes before committing
  - Splinter colony formation from exiles
  - Defer: treaties, formal trade, alliance mechanics (v9.1+)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# -- constants ---------------------------------------------------------------

# Evidence thresholds (0-1 scale, cumulative)
EVIDENCE_CORROBORATE = 0.40
EVIDENCE_CONFIRM = 0.75

# Signal detection starts at year 15, ramps up
SIGNAL_BASE_YEAR = 15
SIGNAL_BASE_PROB = 0.03
SIGNAL_YEAR_RAMP = 0.004

# Splinter colony forms if 2+ colonists exiled in 10-year window
SPLINTER_EXILE_THRESHOLD = 2
SPLINTER_WINDOW_YEARS = 10

# Diplomatic stance thresholds
STANCE_ISOLATIONIST_THRESHOLD = 0.3
STANCE_OPEN_THRESHOLD = 0.6

# Relationship drift per year
RELATIONSHIP_DRIFT_CONFIRMED = 0.03
RELATIONSHIP_DRIFT_UNCONFIRMED = 0.01

MAX_ENTITIES = 5
MAX_INCIDENTS = 50


# -- data classes ------------------------------------------------------------

@dataclass
class ExternalEntity:
    """A known or suspected external entity on Mars.

    Evidence accumulates over time through signals, corroboration,
    and direct contact. Once confirmed, the entity becomes a real
    diplomatic actor.
    """
    id: str
    name: str
    discovered_year: int
    evidence: float = 0.0
    relationship: float = 0.0
    contact_count: int = 0
    origin: str = "unknown"
    population_estimate: int = 0
    exiled_founders: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Current evidence status: rumor, corroborated, or confirmed."""
        if self.evidence >= EVIDENCE_CONFIRM:
            return "confirmed"
        if self.evidence >= EVIDENCE_CORROBORATE:
            return "corroborated"
        return "rumor"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "discovered_year": self.discovered_year,
            "evidence": round(self.evidence, 4),
            "status": self.status,
            "relationship": round(self.relationship, 4),
            "contact_count": self.contact_count,
            "origin": self.origin,
            "population_estimate": self.population_estimate,
            "exiled_founders": self.exiled_founders,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExternalEntity:
        return cls(
            id=d["id"],
            name=d["name"],
            discovered_year=d.get("discovered_year", 0),
            evidence=d.get("evidence", 0.0),
            relationship=d.get("relationship", 0.0),
            contact_count=d.get("contact_count", 0),
            origin=d.get("origin", "unknown"),
            population_estimate=d.get("population_estimate", 0),
            exiled_founders=d.get("exiled_founders", []),
        )


@dataclass
class DiplomacyState:
    """Colony-wide diplomatic awareness and relationships."""
    entities: dict[str, ExternalEntity] = field(default_factory=dict)
    stance: str = "unaware"
    first_contact_year: int | None = None
    signals_detected: int = 0
    incidents: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "stance": self.stance,
            "first_contact_year": self.first_contact_year,
            "signals_detected": self.signals_detected,
            "total_entities": len(self.entities),
            "confirmed_entities": sum(
                1 for e in self.entities.values() if e.status == "confirmed"),
            "incidents": self.incidents[-10:],
        }

    @classmethod
    def from_dict(cls, d: dict) -> DiplomacyState:
        entities = {k: ExternalEntity.from_dict(v)
                    for k, v in d.get("entities", {}).items()}
        return cls(
            entities=entities,
            stance=d.get("stance", "unaware"),
            first_contact_year=d.get("first_contact_year"),
            signals_detected=d.get("signals_detected", 0),
            incidents=d.get("incidents", []),
        )


@dataclass
class DiplomacyTickResult:
    """Result of one year's diplomacy processing."""
    new_signals: list[dict] = field(default_factory=list)
    evidence_updates: list[dict] = field(default_factory=list)
    new_entities: list[dict] = field(default_factory=list)
    stance_before: str = "unaware"
    stance_after: str = "unaware"
    confirmed_this_year: list[str] = field(default_factory=list)
    subsim_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "new_signals": self.new_signals,
            "evidence_updates": self.evidence_updates,
            "new_entities": self.new_entities,
            "stance_before": self.stance_before,
            "stance_after": self.stance_after,
            "confirmed_this_year": self.confirmed_this_year,
            "subsim_results": self.subsim_results,
        }


# -- pure helpers ------------------------------------------------------------

def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def signal_detection_probability(
    year: int,
    coding_skill_avg: float,
    has_comm_infrastructure: bool,
) -> float:
    """Probability of detecting an external signal this year.

    Increases with year (more time scanning) and colony tech level.
    Comm infrastructure doubles the base rate.
    """
    if year < SIGNAL_BASE_YEAR:
        return 0.0
    years_scanning = year - SIGNAL_BASE_YEAR
    base = SIGNAL_BASE_PROB + years_scanning * SIGNAL_YEAR_RAMP
    tech_bonus = coding_skill_avg * 0.05
    infra_mult = 2.0 if has_comm_infrastructure else 1.0
    return min(0.5, base * infra_mult + tech_bonus)


def should_form_splinter(
    exiles_recent: list[dict],
    year: int,
) -> bool:
    """Check if recent exiles are sufficient to form a splinter settlement.

    Requires SPLINTER_EXILE_THRESHOLD exiles within SPLINTER_WINDOW_YEARS.
    """
    recent = [e for e in exiles_recent
              if year - e.get("year", 0) <= SPLINTER_WINDOW_YEARS]
    return len(recent) >= SPLINTER_EXILE_THRESHOLD


def compute_diplomatic_pressure(
    diplo_state: DiplomacyState,
) -> dict[str, float]:
    """Return action-weight modifiers based on diplomatic awareness.

    When entities are known, colonists shift behavior:
    - More coding/research (build comm capability)
    - More mediation (prepare for contact)
    - Less sabotage (external threat unifies)
    """
    if not diplo_state.entities:
        return {}
    confirmed = sum(1 for e in diplo_state.entities.values()
                    if e.status == "confirmed")
    known = len(diplo_state.entities)
    if known == 0:
        return {}

    awareness_factor = min(1.0, known * 0.2 + confirmed * 0.3)
    pressure: dict[str, float] = {}

    if diplo_state.stance == "isolationist":
        pressure["hoard"] = awareness_factor * 0.3
        pressure["sabotage"] = -awareness_factor * 0.2
        pressure["code"] = awareness_factor * 0.15
    elif diplo_state.stance == "cautious":
        pressure["mediate"] = awareness_factor * 0.2
        pressure["code"] = awareness_factor * 0.2
        pressure["sabotage"] = -awareness_factor * 0.3
    elif diplo_state.stance == "open":
        pressure["mediate"] = awareness_factor * 0.3
        pressure["cooperate"] = awareness_factor * 0.2
        pressure["explore"] = awareness_factor * 0.15
        pressure["sabotage"] = -awareness_factor * 0.4
    else:
        pressure["code"] = awareness_factor * 0.1

    return pressure


def compute_stance(
    entities: dict[str, ExternalEntity],
    colony_paranoia_avg: float,
    colony_empathy_avg: float,
) -> str:
    """Derive diplomatic stance from colony personality and situation.

    Stance flows from the colony's aggregate personality:
    - High paranoia → isolationist
    - High empathy → open
    - Mixed → cautious
    If no entities are known, stance is 'unaware'.
    """
    if not entities:
        return "unaware"

    openness = colony_empathy_avg - colony_paranoia_avg * 0.7
    confirmed = sum(1 for e in entities.values() if e.status == "confirmed")
    if confirmed > 0:
        openness += 0.1

    hostile = sum(1 for e in entities.values() if e.relationship < -0.3)
    if hostile > 0:
        openness -= 0.3

    if openness < STANCE_ISOLATIONIST_THRESHOLD:
        return "isolationist"
    if openness >= STANCE_OPEN_THRESHOLD:
        return "open"
    return "cautious"


def generate_entity_name(origin: str, rng: random.Random) -> str:
    """Generate a name for a discovered entity."""
    if origin == "splinter":
        prefixes = ["New", "Free", "Outer", "Far", "Red"]
        suffixes = ["Haven", "Hope", "Reach", "Dust", "Ridge"]
        return f"{rng.choice(prefixes)} {rng.choice(suffixes)}"
    if origin == "earth_mission":
        names = ["Ares-B Colony", "Olympus Station", "Valles Base",
                 "Elysium Outpost", "Tharsis Settlement"]
        return rng.choice(names)
    adjectives = ["Distant", "Silent", "Phantom", "Echo", "Shadow"]
    nouns = ["Signal", "Beacon", "Source", "Station", "Array"]
    return f"{rng.choice(adjectives)} {rng.choice(nouns)}"


def diplomatic_subsim_expression(
    entity: ExternalEntity,
    stance: str,
) -> str:
    """Generate a LisPy expression for modeling diplomatic outcomes.

    Colonists run these sub-sims to evaluate whether to approach,
    trade with, or avoid an external entity.
    """
    ev = round(entity.evidence, 2)
    rel = round(entity.relationship, 2)
    templates = [
        f"(let ((risk (* paranoia {ev})) (gain (* empathy {rel}))) "
        f"(if (> gain risk) (+ gain 0.2) (- 0 risk)))",
        f"(let ((strength (+ resolve coding)) (threat {ev})) "
        f"(if (> strength threat) (+ 0.5 (* empathy 0.3)) "
        f"(- 0.3 (* paranoia 0.2))))",
        f"(let ((openness (- empathy (* paranoia 0.5))) "
        f"(evidence {ev})) "
        f"(if (> (* openness evidence) 0.3) 1 0))",
    ]
    if stance == "isolationist":
        return templates[1]
    if stance == "open":
        return templates[0]
    return templates[2]


# -- main tick ---------------------------------------------------------------

def tick_diplomacy(
    diplo: DiplomacyState,
    year: int,
    year_exiles: list[dict],
    all_exiles_history: list[dict],
    coding_skill_avg: float,
    colony_paranoia_avg: float,
    colony_empathy_avg: float,
    has_comm_infra: bool,
    earth_announced_mission: bool,
    rng: random.Random,
) -> DiplomacyTickResult:
    """Run one year of diplomatic awareness evolution.

    Phase 1: detect signals and accumulate evidence
    Phase 2: form splinter colonies from exiles
    Phase 3: process Earth announcements
    Phase 4: evolve relationships and stance

    Mutates diplo in place.  Returns tick result for logging.
    """
    result = DiplomacyTickResult(
        stance_before=diplo.stance,
    )

    # --- Phase 1: signal detection ---
    sig_prob = signal_detection_probability(
        year, coding_skill_avg, has_comm_infra)
    if sig_prob > 0 and rng.random() < sig_prob:
        diplo.signals_detected += 1
        signal_info = {
            "year": year,
            "type": rng.choice(["radio", "thermal", "seismic", "optical"]),
            "strength": round(rng.uniform(0.1, 0.6), 3),
        }
        result.new_signals.append(signal_info)

        _apply_signal_to_entities(diplo, signal_info, year, rng)

    # --- Phase 2: splinter colony from exiles ---
    if (year_exiles and should_form_splinter(all_exiles_history, year)
            and len(diplo.entities) < MAX_ENTITIES):
        existing_splinters = [e for e in diplo.entities.values()
                              if e.origin == "splinter"]
        splinter_id = f"splinter-{len(existing_splinters) + 1}"
        if splinter_id not in diplo.entities:
            name = generate_entity_name("splinter", rng)
            founders = [ex.get("id", "unknown") for ex in year_exiles]
            entity = ExternalEntity(
                id=splinter_id,
                name=name,
                discovered_year=year,
                evidence=0.5,
                relationship=-0.2,
                origin="splinter",
                population_estimate=len(founders),
                exiled_founders=founders,
            )
            diplo.entities[splinter_id] = entity
            result.new_entities.append(entity.to_dict())
            _record_incident(diplo, year, "splinter_formed",
                             f"Exiled colonists founded {name}")

    # --- Phase 3: Earth-announced mission ---
    if (earth_announced_mission
            and len(diplo.entities) < MAX_ENTITIES):
        mission_id = "earth-mission-1"
        if mission_id not in diplo.entities:
            name = generate_entity_name("earth_mission", rng)
            entity = ExternalEntity(
                id=mission_id,
                name=name,
                discovered_year=year,
                evidence=0.8,
                relationship=0.3,
                origin="earth_mission",
                population_estimate=rng.randint(6, 20),
            )
            diplo.entities[mission_id] = entity
            result.new_entities.append(entity.to_dict())
            if diplo.first_contact_year is None:
                diplo.first_contact_year = year
            _record_incident(diplo, year, "earth_mission_announced",
                             f"Earth announced {name}")

    # --- Phase 4: evolve entities and stance ---
    for entity in diplo.entities.values():
        _evolve_entity(entity, year, rng)
        was_unconfirmed = entity.status != "confirmed"
        # Accumulate evidence from signals
        if diplo.signals_detected > 0 and entity.status != "confirmed":
            entity.evidence = min(
                1.0,
                entity.evidence + rng.uniform(0.01, 0.05))
            result.evidence_updates.append({
                "entity_id": entity.id,
                "new_evidence": round(entity.evidence, 4),
                "status": entity.status,
            })
        if was_unconfirmed and entity.status == "confirmed":
            result.confirmed_this_year.append(entity.id)
            if diplo.first_contact_year is None:
                diplo.first_contact_year = year
            _record_incident(diplo, year, "entity_confirmed",
                             f"{entity.name} confirmed")

    # Update stance
    diplo.stance = compute_stance(
        diplo.entities, colony_paranoia_avg, colony_empathy_avg)
    result.stance_after = diplo.stance

    return result


# -- internal helpers --------------------------------------------------------

def _apply_signal_to_entities(
    diplo: DiplomacyState,
    signal: dict,
    year: int,
    rng: random.Random,
) -> None:
    """Apply a detected signal to existing or new entities."""
    unconfirmed = [e for e in diplo.entities.values()
                   if e.status != "confirmed"]
    if unconfirmed:
        target = rng.choice(unconfirmed)
        boost = signal.get("strength", 0.1) * 0.3
        target.evidence = min(1.0, target.evidence + boost)
    elif len(diplo.entities) < MAX_ENTITIES:
        entity_id = f"unknown-{diplo.signals_detected}"
        name = generate_entity_name("unknown", rng)
        entity = ExternalEntity(
            id=entity_id,
            name=name,
            discovered_year=year,
            evidence=signal.get("strength", 0.1) * 0.4,
            origin="signal",
        )
        diplo.entities[entity_id] = entity


def _evolve_entity(
    entity: ExternalEntity,
    year: int,
    rng: random.Random,
) -> None:
    """Evolve an entity's relationship and population over one year."""
    if entity.status == "confirmed":
        drift = rng.gauss(0, RELATIONSHIP_DRIFT_CONFIRMED)
        entity.relationship = _clamp(entity.relationship + drift)
        if entity.origin == "splinter":
            entity.population_estimate = max(
                0,
                entity.population_estimate + rng.choice([-1, 0, 0, 1, 1]))
    elif entity.status == "corroborated":
        drift = rng.gauss(0, RELATIONSHIP_DRIFT_UNCONFIRMED)
        entity.relationship = _clamp(entity.relationship + drift)


def _record_incident(
    diplo: DiplomacyState,
    year: int,
    incident_type: str,
    description: str,
) -> None:
    """Record a diplomatic incident, pruning old ones."""
    diplo.incidents.append({
        "year": year,
        "type": incident_type,
        "description": description,
    })
    if len(diplo.incidents) > MAX_INCIDENTS:
        diplo.incidents = diplo.incidents[-MAX_INCIDENTS:]
