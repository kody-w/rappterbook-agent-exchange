"""
Ancestor protocol for Mars-100.

When colonists die, their accumulated stats, skills, and death context
become consultable via the sub-simulation system.  Living colonists
with high faith or strong relationships to the deceased may invoke
an ancestor consultation during crises — the ancestor's profile is
injected as read-only LisPy bindings into a sub-sim.

Design constraints:
- Read-only numeric bindings only — no generated s-expressions.
- Ancestors available the year AFTER death (no tick reorder).
- Consumes normal SubSimBudget — tagged as ancestor_consult.
- Connected to asphyxiation bottleneck: consultations are more
  likely during air/resource crises, and the consultation expression
  weighs crisis-relevant skills.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import Colonist, STAT_NAMES, SKILL_NAMES
from src.mars100.subsim import SubSimBudget, SubSimResult, spawn_subsim


FAITH_THRESHOLD = 0.3
BASE_CONSULT_PROB = 0.15
CRISIS_SEVERITY_THRESHOLD = 0.5

DEATH_CAUSE_TAGS: dict[str, str] = {
    "asphyxiation": "air-failure",
    "habitat breach": "air-failure",
    "equipment malfunction": "equipment-failure",
    "radiation exposure": "radiation",
    "medical emergency": "medical",
    "resource deprivation": "resource-failure",
    "suspicious accident": "social-failure",
}

# Templates use ONLY bindings from colonist.lispy_bindings() (caller)
# and AncestorProfile.lispy_bindings() (ancestor-* prefixed).
CONSULT_TEMPLATES: dict[str, str] = {
    "air_crisis": (
        "(let ((skill-sum (+ ancestor-terraforming ancestor-coding)))"
        " (if (> skill-sum 1.0)"
        "   (* skill-sum 0.1)"
        "   (* ancestor-resolve 0.05)))"
    ),
    "resource_crisis": (
        "(let ((wisdom (+ ancestor-hydroponics ancestor-terraforming)))"
        " (if (> wisdom 0.8)"
        "   (* wisdom 0.1)"
        "   (* ancestor-improvisation 0.05)))"
    ),
    "social_crisis": (
        "(let ((social (+ ancestor-mediation ancestor-empathy)))"
        " (if (> social 1.2)"
        "   (* social 0.1)"
        "   (* ancestor-resolve 0.05)))"
    ),
    "general": (
        "(let ((wisdom (+ ancestor-resolve ancestor-faith)))"
        " (if (> wisdom 0.8)"
        "   (* wisdom 0.1)"
        "   ancestor-improvisation))"
    ),
}


@dataclass
class AncestorProfile:
    """Compressed profile of a dead colonist available for consultation."""
    colonist_id: str
    name: str
    element: str
    death_year: int
    death_cause: str
    years_lived: int
    stats: dict[str, float]
    skills: dict[str, float]
    subsims_run: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "colonist_id": self.colonist_id,
            "name": self.name,
            "element": self.element,
            "death_year": self.death_year,
            "death_cause": self.death_cause,
            "years_lived": self.years_lived,
            "stats": dict(self.stats),
            "skills": dict(self.skills),
            "subsims_run": self.subsims_run,
        }

    def lispy_bindings(self) -> dict[str, Any]:
        """Return read-only numeric bindings prefixed with ``ancestor-``."""
        bindings: dict[str, Any] = {}
        for name, val in self.stats.items():
            bindings[f"ancestor-{name}"] = val
        for name, val in self.skills.items():
            bindings[f"ancestor-{name}"] = val
        bindings["ancestor-years-lived"] = self.years_lived
        bindings["ancestor-death-cause-tag"] = DEATH_CAUSE_TAGS.get(
            self.death_cause, "unknown"
        )
        bindings["ancestor-died-air"] = (
            1 if self.death_cause in ("asphyxiation", "habitat breach") else 0
        )
        bindings["ancestor-subsims"] = self.subsims_run
        return bindings


@dataclass
class AncestorVault:
    """Colony-wide store of ancestor profiles."""
    profiles: dict[str, AncestorProfile] = field(default_factory=dict)
    consultations: list[dict] = field(default_factory=list)

    def add(self, colonist: Colonist) -> AncestorProfile:
        """Register a newly dead colonist as an ancestor."""
        profile = AncestorProfile(
            colonist_id=colonist.id,
            name=colonist.name,
            element=colonist.element,
            death_year=colonist.death_year or 0,
            death_cause=colonist.death_cause or "unknown",
            years_lived=(colonist.death_year or 0) - colonist.birth_year,
            stats={n: getattr(colonist.stats, n) for n in STAT_NAMES},
            skills={n: getattr(colonist.skills, n) for n in SKILL_NAMES},
            subsims_run=colonist.subsim_count,
        )
        self.profiles[colonist.id] = profile
        return profile

    def get(self, colonist_id: str) -> AncestorProfile | None:
        """Look up an ancestor by colonist id."""
        return self.profiles.get(colonist_id)

    def all_ids(self) -> list[str]:
        """Return all ancestor ids."""
        return list(self.profiles.keys())

    def size(self) -> int:
        """Number of registered ancestors."""
        return len(self.profiles)

    def to_dict(self) -> dict[str, Any]:
        """Serialise vault state (keeps last 20 consultations)."""
        return {
            "ancestors": {k: v.to_dict() for k, v in self.profiles.items()},
            "total_consultations": len(self.consultations),
            "consultation_log": self.consultations[-20:],
        }

    def record_consultation(self, year: int, caller_id: str,
                            ancestor_id: str, template: str,
                            result: SubSimResult) -> None:
        """Log an ancestor consultation."""
        self.consultations.append({
            "year": year,
            "caller": caller_id,
            "ancestor": ancestor_id,
            "template": template,
            "succeeded": result.succeeded,
            "result": _safe_val(result.result),
            "steps": result.steps_used,
        })


def _safe_val(v: Any) -> Any:
    """Make a value JSON-safe."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def should_consult(colonist: Colonist, vault: AncestorVault,
                   crisis_severity: float,
                   rng: Any) -> str | None:
    """Decide whether *colonist* consults an ancestor this year.

    Returns the ancestor_id to consult, or ``None``.
    """
    if vault.size() == 0:
        return None
    if colonist.stats.faith < FAITH_THRESHOLD:
        return None

    prob = BASE_CONSULT_PROB
    prob += colonist.stats.faith * 0.2
    if crisis_severity > CRISIS_SEVERITY_THRESHOLD:
        prob += 0.3
    prob = min(prob, 0.85)

    if rng.random() > prob:
        return None

    candidates = vault.all_ids()
    if not candidates:
        return None
    return rng.choice(candidates)


def choose_template(resources: dict[str, float],
                    events: list[dict]) -> str:
    """Pick the consultation template based on current crisis."""
    air = resources.get("air", 1.0)
    if air < 0.2:
        return "air_crisis"

    avg_res = sum(resources.values()) / max(1, len(resources))
    if avg_res < 0.3:
        return "resource_crisis"

    for ev in events:
        if isinstance(ev, dict) and ev.get("name") == "colonist_conflict":
            return "social_crisis"

    return "general"


def consult_ancestor(vault: AncestorVault, caller: Colonist,
                     ancestor_id: str, template_key: str,
                     year: int, budget: SubSimBudget,
                     log: list[SubSimResult]) -> SubSimResult | None:
    """Run an ancestor consultation as a tagged sub-simulation.

    Returns the ``SubSimResult`` or ``None`` if the consultation
    couldn't proceed (missing ancestor or exhausted budget).
    """
    profile = vault.get(ancestor_id)
    if profile is None:
        return None
    if not budget.can_spawn(caller.id):
        return None

    expression = CONSULT_TEMPLATES.get(template_key, CONSULT_TEMPLATES["general"])

    bindings = caller.lispy_bindings()
    bindings.update(profile.lispy_bindings())

    result = spawn_subsim(
        expression=expression,
        colonist_id=caller.id,
        year=year,
        bindings=bindings,
        depth=1,
        budget=budget,
        log=log,
    )

    vault.record_consultation(year, caller.id, ancestor_id, template_key, result)
    return result
