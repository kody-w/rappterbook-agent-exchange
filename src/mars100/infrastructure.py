"""
Infrastructure system for Mars-100.

Colonists build permanent improvements that modify the resource model.
Each tech has resource costs, build time, prerequisites, effects, and
ongoing operating costs. At most one project active at a time.

Effects target resilience (spoilage reduction, death rate modifiers,
event damage mitigation) rather than raw production to avoid saturating
the 0.0-1.0 resource bounds.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TechSpec:
    """Blueprint for a buildable technology."""
    id: str
    name: str
    description: str
    resource_cost: dict[str, float]
    build_time: int  # years of sufficient worker commitment
    workers_needed: int  # minimum colonists choosing "research" per year
    prereqs: list[str]
    effects: dict[str, float]
    operating_cost: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "resource_cost": self.resource_cost, "build_time": self.build_time,
            "workers_needed": self.workers_needed, "prereqs": self.prereqs,
            "effects": self.effects, "operating_cost": self.operating_cost,
        }


TECH_TREE: list[TechSpec] = [
    TechSpec(
        id="greenhouse_dome",
        name="Greenhouse Dome",
        description="Sealed pressurized growing chamber reduces crop spoilage and boosts yield.",
        resource_cost={"power": 0.08, "food": 0.04},
        build_time=3, workers_needed=2, prereqs=[],
        effects={"food_spoilage_mult": 0.5, "food_production_mult": 1.10},
        operating_cost={"power": 0.005},
    ),
    TechSpec(
        id="water_recycler",
        name="Water Recycler",
        description="Closed-loop water purification dramatically reduces waste.",
        resource_cost={"power": 0.06, "medicine": 0.03},
        build_time=3, workers_needed=2, prereqs=[],
        effects={"water_spoilage_mult": 0.4, "water_consumption_mult": 0.85},
        operating_cost={"power": 0.003},
    ),
    TechSpec(
        id="power_grid",
        name="Power Grid Upgrade",
        description="Redundant wiring and smart load-balancing cuts maintenance overhead.",
        resource_cost={"food": 0.05, "water": 0.03},
        build_time=4, workers_needed=2, prereqs=[],
        effects={"power_maintenance_mult": 0.6},
        operating_cost={"food": 0.002},
    ),
    TechSpec(
        id="med_bay",
        name="Med Bay",
        description="Surgical suite with diagnostic AI lowers mortality and stretches medicine stocks.",
        resource_cost={"power": 0.08, "food": 0.06, "water": 0.04},
        build_time=5, workers_needed=3, prereqs=["greenhouse_dome"],
        effects={"death_rate_mult": 0.70, "medicine_production_mult": 1.15},
        operating_cost={"power": 0.004, "medicine": 0.002},
    ),
    TechSpec(
        id="shelter_reinforcement",
        name="Shelter Reinforcement",
        description="Regolith shielding and blast doors reduce damage from dust storms and solar flares.",
        resource_cost={"power": 0.07, "food": 0.04, "water": 0.03},
        build_time=4, workers_needed=3, prereqs=["power_grid"],
        effects={"event_damage_mult": 0.6},
        operating_cost={"power": 0.003},
    ),
    TechSpec(
        id="research_lab",
        name="Research Lab",
        description="Dedicated computation and experimentation facility expands sub-simulation capacity.",
        resource_cost={"power": 0.10, "food": 0.05, "medicine": 0.03},
        build_time=6, workers_needed=3, prereqs=["med_bay", "shelter_reinforcement"],
        effects={"subsim_budget_mult": 1.5},
        operating_cost={"power": 0.005, "food": 0.002},
    ),
]

TECH_BY_ID: dict[str, TechSpec] = {t.id: t for t in TECH_TREE}

MAX_STALL_YEARS = 3


@dataclass
class ActiveProject:
    """A technology currently under construction."""
    tech_id: str
    progress: int = 0  # years of sufficient worker commitment
    stall_years: int = 0  # consecutive years without enough workers

    def to_dict(self) -> dict:
        return {"tech_id": self.tech_id, "progress": self.progress,
                "stall_years": self.stall_years}

    @classmethod
    def from_dict(cls, d: dict) -> ActiveProject:
        return cls(tech_id=d["tech_id"], progress=d.get("progress", 0),
                   stall_years=d.get("stall_years", 0))


@dataclass
class InfrastructureState:
    """Colony infrastructure state."""
    completed: list[str] = field(default_factory=list)
    project: ActiveProject | None = None
    history: list[dict] = field(default_factory=list)
    abandoned: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "completed": self.completed,
            "project": self.project.to_dict() if self.project else None,
            "history": self.history,
            "abandoned": self.abandoned,
        }

    @classmethod
    def from_dict(cls, d: dict) -> InfrastructureState:
        project = ActiveProject.from_dict(d["project"]) if d.get("project") else None
        return cls(completed=d.get("completed", []), project=project,
                   history=d.get("history", []), abandoned=d.get("abandoned", []))


def available_techs(state: InfrastructureState) -> list[TechSpec]:
    """Return techs that are buildable: not completed, not in-progress, prereqs met."""
    in_progress = state.project.tech_id if state.project else None
    result: list[TechSpec] = []
    for tech in TECH_TREE:
        if tech.id in state.completed:
            continue
        if tech.id == in_progress:
            continue
        if all(p in state.completed for p in tech.prereqs):
            result.append(tech)
    return result


def can_afford(tech: TechSpec, resources_dict: dict[str, float]) -> bool:
    """Check if colony can afford the upfront resource cost."""
    for resource, cost in tech.resource_cost.items():
        if resources_dict.get(resource, 0.0) < cost:
            return False
    return True


def choose_project(state: InfrastructureState, resources_dict: dict[str, float],
                   active_colonists: list[Any], rng: random.Random) -> TechSpec | None:
    """Choose the next infrastructure project via colonist preference voting.

    Each active colonist votes for the tech most aligned with their strongest
    skill.  Ties broken randomly.  Returns None if no affordable tech or
    already building.
    """
    if state.project is not None:
        return None
    candidates = [t for t in available_techs(state) if can_afford(t, resources_dict)]
    if not candidates:
        return None

    skill_tech_affinity: dict[str, list[str]] = {
        "hydroponics": ["greenhouse_dome"],
        "terraforming": ["shelter_reinforcement", "water_recycler"],
        "coding": ["research_lab", "power_grid"],
        "mediation": ["med_bay"],
        "prayer": ["shelter_reinforcement"],
        "sabotage": [],
    }

    scores: dict[str, float] = {t.id: 0.0 for t in candidates}
    candidate_ids = {t.id for t in candidates}
    for colonist in active_colonists:
        best_skill = colonist.skills.best_skill()
        preferred = skill_tech_affinity.get(best_skill, [])
        for tid in preferred:
            if tid in scores:
                scores[tid] += 1.0
        for tid in candidate_ids:
            scores[tid] += rng.uniform(0, 0.3)

    best_id = max(scores, key=lambda k: scores[k])
    return TECH_BY_ID[best_id]


def start_project(state: InfrastructureState, tech: TechSpec,
                  resources: Any, year: int) -> None:
    """Begin construction of a technology, deducting upfront costs."""
    from src.mars100.colony import RESOURCE_NAMES
    for resource, cost in tech.resource_cost.items():
        if resource in RESOURCE_NAMES:
            current = getattr(resources, resource)
            setattr(resources, resource, max(0.0, current - cost))
    state.project = ActiveProject(tech_id=tech.id)
    state.history.append({"year": year, "event": "started", "tech_id": tech.id})


def tick_infrastructure(state: InfrastructureState, researcher_count: int,
                        skill_avg: float, year: int) -> dict | None:
    """Advance active project by one year. Returns completion event or None.

    Args:
        state: current infrastructure state
        researcher_count: number of colonists who chose "research" this year
        skill_avg: average relevant skill level of researchers (0.0-1.0)
        year: current simulation year

    Returns:
        Dict describing completion event, or None.
    """
    if state.project is None:
        return None

    tech = TECH_BY_ID.get(state.project.tech_id)
    if tech is None:
        state.project = None
        return None

    if researcher_count >= tech.workers_needed:
        bonus = 1 + skill_avg * 0.3
        state.project.progress += max(1, int(bonus))
        state.project.stall_years = 0
    else:
        state.project.stall_years += 1

    if state.project.stall_years >= MAX_STALL_YEARS:
        abandoned_id = state.project.tech_id
        state.history.append({"year": year, "event": "abandoned", "tech_id": abandoned_id})
        state.abandoned.append(abandoned_id)
        state.project = None
        return {"event": "abandoned", "tech_id": abandoned_id, "year": year}

    if state.project.progress >= tech.build_time:
        completed_id = state.project.tech_id
        state.completed.append(completed_id)
        state.history.append({"year": year, "event": "completed", "tech_id": completed_id})
        state.project = None
        return {"event": "completed", "tech_id": completed_id, "year": year,
                "effects": tech.effects}

    return None


def compute_resource_modifiers(completed_techs: list[str]) -> dict[str, float]:
    """Compute aggregate resource modifiers from all completed technologies.

    Returns a dict of modifier keys to their cumulative multiplier values.
    Multiplicative stacking: if two techs both modify "food_spoilage_mult"
    by 0.5 and 0.8, the result is 0.4.
    """
    modifiers: dict[str, float] = {}
    for tid in completed_techs:
        tech = TECH_BY_ID.get(tid)
        if tech is None:
            continue
        for key, value in tech.effects.items():
            if key in modifiers:
                modifiers[key] *= value
            else:
                modifiers[key] = value
    return modifiers


def compute_operating_costs(completed_techs: list[str]) -> dict[str, float]:
    """Sum up per-year operating costs of all completed technologies."""
    costs: dict[str, float] = {}
    for tid in completed_techs:
        tech = TECH_BY_ID.get(tid)
        if tech is None:
            continue
        for resource, amount in tech.operating_cost.items():
            costs[resource] = costs.get(resource, 0.0) + amount
    return costs


def validate_tech_tree() -> list[str]:
    """Validate the tech tree for structural issues. Returns list of errors."""
    errors: list[str] = []
    ids = {t.id for t in TECH_TREE}
    for tech in TECH_TREE:
        for prereq in tech.prereqs:
            if prereq not in ids:
                errors.append(f"{tech.id}: unknown prereq '{prereq}'")
            if prereq == tech.id:
                errors.append(f"{tech.id}: self-referential prereq")
        for resource, cost in tech.resource_cost.items():
            if cost < 0:
                errors.append(f"{tech.id}: negative cost for {resource}")
        if tech.build_time < 1:
            errors.append(f"{tech.id}: build_time must be >= 1")
        if tech.workers_needed < 1:
            errors.append(f"{tech.id}: workers_needed must be >= 1")

    # Check for cycles in prerequisites
    visited: set[str] = set()
    def _has_cycle(tid: str, path: set[str]) -> bool:
        if tid in path:
            return True
        if tid in visited:
            return False
        path.add(tid)
        tech = TECH_BY_ID.get(tid)
        if tech:
            for prereq in tech.prereqs:
                if _has_cycle(prereq, path):
                    return True
        path.remove(tid)
        visited.add(tid)
        return False

    for tech in TECH_TREE:
        if _has_cycle(tech.id, set()):
            errors.append(f"{tech.id}: prerequisite cycle detected")
    return errors
