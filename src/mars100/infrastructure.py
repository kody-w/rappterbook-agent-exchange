"""
Infrastructure / tech tree for the Mars-100 colony.

Persistent colony improvements built over multiple years.  Each tech
modifies resource modifiers (spoilage, maintenance, death rate) rather
than directly boosting raw production -- keeps values inside the 0-1 range.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

MAX_STALL_YEARS = 3


@dataclass(frozen=True)
class TechSpec:
    """Blueprint for a colony technology."""
    id: str
    name: str
    description: str
    build_time: int
    workers_needed: int
    resource_cost: dict
    effects: dict
    operating_cost: dict

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name,
            "description": self.description,
            "build_time": self.build_time,
            "workers_needed": self.workers_needed,
            "resource_cost": dict(self.resource_cost),
            "effects": dict(self.effects),
            "operating_cost": dict(self.operating_cost),
        }


TECH_TREE = [
    TechSpec(
        id="greenhouse_dome", name="Greenhouse Dome",
        description="Pressurised growing dome -- cuts food spoilage in half.",
        build_time=4, workers_needed=2,
        resource_cost={"power": 0.15, "air": 0.05},
        effects={"food_spoilage_mult": 0.5},
        operating_cost={"power": 0.005},
    ),
    TechSpec(
        id="water_recycler", name="Water Recycler",
        description="Closed-loop water recovery -- halves water maintenance.",
        build_time=3, workers_needed=2,
        resource_cost={"power": 0.12, "water": 0.05},
        effects={"water_maintenance_mult": 0.5},
        operating_cost={"power": 0.003},
    ),
    TechSpec(
        id="power_grid", name="Power Grid Upgrade",
        description="Distributed solar + battery -- reduces power spoilage.",
        build_time=5, workers_needed=3,
        resource_cost={"power": 0.12, "air": 0.08},
        effects={"power_spoilage_mult": 0.6},
        operating_cost={"power": 0.005},
    ),
    TechSpec(
        id="med_bay", name="Medical Bay",
        description="Proper med bay -- reduces death rate by 40 pct.",
        build_time=4, workers_needed=2,
        resource_cost={"power": 0.10, "food": 0.05, "medicine": 0.05},
        effects={"death_rate_mult": 0.6},
        operating_cost={"power": 0.004, "food": 0.003},
    ),
    TechSpec(
        id="shelter_reinforcement", name="Shelter Reinforcement",
        description="Reinforced habitats -- event damage reduced by 30 pct.",
        build_time=3, workers_needed=2,
        resource_cost={"power": 0.10, "air": 0.05},
        effects={"event_damage_mult": 0.7},
        operating_cost={"power": 0.003},
    ),
    TechSpec(
        id="research_lab", name="Research Lab",
        description="Dedicated lab -- boosts sub-sim budget by 50 pct.",
        build_time=6, workers_needed=3,
        resource_cost={"power": 0.18, "air": 0.08},
        effects={"subsim_budget_mult": 1.5},
        operating_cost={"power": 0.006},
    ),
    TechSpec(
        id="air_recycler", name="Air Recycler",
        description="Closed-loop atmospheric processor -- cuts air spoilage by 60 pct and maintenance by 40 pct.",
        build_time=5, workers_needed=2,
        resource_cost={"power": 0.15, "water": 0.05},
        effects={"air_spoilage_mult": 0.4, "air_maintenance_mult": 0.6},
        operating_cost={"power": 0.005},
    ),
]

TECH_BY_ID = {t.id: t for t in TECH_TREE}


@dataclass
class ActiveProject:
    """A tech currently under construction."""
    tech_id: str
    progress: int = 0
    stall_years: int = 0

    def to_dict(self) -> dict:
        return {"tech_id": self.tech_id, "progress": self.progress,
                "stall_years": self.stall_years}


@dataclass
class InfrastructureState:
    """Colony-wide infrastructure status."""
    completed: list = field(default_factory=list)
    project: ActiveProject = None
    abandoned: list = field(default_factory=list)
    history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "completed": list(self.completed),
            "active_project": self.project.to_dict() if self.project else None,
            "abandoned": list(self.abandoned),
            "history": list(self.history),
        }


def available_techs(state):
    """Techs not yet completed or abandoned."""
    done = set(state.completed) | set(state.abandoned)
    return [t for t in TECH_TREE if t.id not in done]


def can_afford(tech, resources_dict):
    """Check if the colony can afford a tech's upfront cost."""
    for res, cost in tech.resource_cost.items():
        if resources_dict.get(res, 0.0) < cost:
            return False
    return True


def choose_project(state, resources_dict, active_colonists, rng):
    """Choose the next project via colonist preference voting."""
    if state.project is not None:
        return None
    candidates = [t for t in available_techs(state) if can_afford(t, resources_dict)]
    if not candidates:
        return None

    skill_tech_affinity = {
        "hydroponics": ["greenhouse_dome"],
        "terraforming": ["shelter_reinforcement", "water_recycler"],
        "coding": ["research_lab", "power_grid"],
        "mediation": ["med_bay"],
        "prayer": ["shelter_reinforcement"],
        "sabotage": [],
    }

    scores = {t.id: 0.0 for t in candidates}
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


def start_project(state, tech, resources, year):
    """Begin construction, deducting upfront cost from Resources object."""
    from src.mars100.colony import RESOURCE_NAMES
    for resource, cost in tech.resource_cost.items():
        if resource in RESOURCE_NAMES:
            current = getattr(resources, resource)
            setattr(resources, resource, max(0.0, current - cost))
    state.project = ActiveProject(tech_id=tech.id)
    state.history.append({"year": year, "event": "started", "tech_id": tech.id})


def tick_infrastructure(state, researcher_count, skill_avg, year):
    """Advance active project by one year. Returns event dict or None."""
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


def compute_resource_modifiers(completed_techs):
    """Multiplicatively combine effects of all completed techs."""
    mods = {}
    for tid in completed_techs:
        tech = TECH_BY_ID.get(tid)
        if tech is None:
            continue
        for key, val in tech.effects.items():
            mods[key] = mods.get(key, 1.0) * val
    return mods


def compute_operating_costs(completed_techs):
    """Sum operating costs of all completed techs."""
    costs = {}
    for tid in completed_techs:
        tech = TECH_BY_ID.get(tid)
        if tech is None:
            continue
        for res, cost in tech.operating_cost.items():
            costs[res] = costs.get(res, 0.0) + cost
    return costs


def validate_tech_tree():
    """Smoke-check the tech tree for consistency."""
    errors = []
    ids = set()
    for tech in TECH_TREE:
        if tech.id in ids:
            errors.append(f"Duplicate tech id: {tech.id}")
        ids.add(tech.id)
        if tech.build_time < 1:
            errors.append(f"{tech.id}: build_time must be >= 1")
        if tech.workers_needed < 1:
            errors.append(f"{tech.id}: workers_needed must be >= 1")
        for res, cost in tech.resource_cost.items():
            if cost < 0 or cost > 1:
                errors.append(f"{tech.id}: resource_cost[{res}]={cost} out of [0,1]")
        for res, cost in tech.operating_cost.items():
            if cost < 0:
                errors.append(f"{tech.id}: operating_cost[{res}]={cost} negative")
    return errors
