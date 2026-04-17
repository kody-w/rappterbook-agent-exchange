"""
Mars-100 recursive world simulation organ — turtles all the way down.

Colonists spawn sandboxed mini-world simulations to model consequences
of governance proposals, survival strategies, and economic scenarios
before committing.  Each mini-world is a simplified colony that runs
N frames.  Mini-worlds can recurse up to depth 3.

Engine v10.0.  Uses recursive_rng (seed + 11213).

Constitutional basis: Amendment XIII (Turtles All the Way Down) —
any agent can spawn a sandboxed sub-simulation following the same
data-sloshing pattern.  Output of frame N = input to frame N+1.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.lispy_vm import (
    LispyError,
    run as lispy_run,
)

# --- budget & depth limits ---

MAX_WORLD_DEPTH = 3
FRAMES_BY_DEPTH = {1: 20, 2: 10, 3: 5}
AGENTS_BY_DEPTH = {1: 5, 2: 3, 3: 2}
BUDGET_UNIT_WEIGHT = {1: 1.0, 2: 0.5, 3: 0.25}
MAX_BUDGET_UNITS_PER_YEAR = 5.0
GOVERNANCE_STABILITY_THRESHOLD = 3  # consecutive frames

# --- simplified resource set ---

MINI_RESOURCES = ("food", "water", "power")
MINI_STAT_NAMES = ("resolve", "improvisation", "empathy", "hoarding",
                   "faith", "paranoia")
MINI_SKILL_NAMES = ("terraforming", "hydroponics", "mediation", "coding",
                    "prayer", "sabotage")

# --- events pool (deterministic from seed) ---

MINI_EVENTS = [
    {"name": "dust_storm", "severity": 0.6, "food": -0.05, "water": -0.03, "power": -0.04},
    {"name": "resource_strike", "severity": 0.3, "food": 0.0, "water": 0.08, "power": 0.0},
    {"name": "equipment_failure", "severity": 0.5, "food": 0.0, "water": 0.0, "power": -0.08},
    {"name": "calm_year", "severity": 0.1, "food": 0.02, "water": 0.01, "power": 0.01},
    {"name": "solar_flare", "severity": 0.7, "food": -0.02, "water": 0.0, "power": -0.10},
    {"name": "harvest_boom", "severity": 0.2, "food": 0.10, "water": -0.02, "power": 0.0},
    {"name": "colonist_conflict", "severity": 0.4, "food": -0.01, "water": -0.01, "power": -0.01},
    {"name": "earth_signal", "severity": 0.2, "food": 0.0, "water": 0.0, "power": 0.03},
]

MINI_GOV_TYPES = ("anarchy", "council", "dictator", "consensus")

# Decision templates for mini-colonists
MINI_DECISION_EXPRS = [
    "(+ (* resolve 0.4) (* empathy 0.3) (* faith 0.3))",
    "(- (* resolve improvisation) (* paranoia 0.5))",
    "(+ (* empathy 0.5) (* improvisation 0.3) (- 0.2 (* paranoia 0.3)))",
    "(if (> paranoia 0.5) (- resolve 0.2) (+ empathy 0.1))",
    "(+ (* faith 0.4) (* resolve 0.4) (* hoarding 0.2))",
]


@dataclass
class MiniColonist:
    """Simplified colonist for sub-world simulations.

    Preserves the full binding surface (6 stats + 6 skills) so that
    LisPy expressions from the parent simulation work without
    ``unbound symbol`` errors.
    """
    id: str
    stats: dict[str, float] = field(default_factory=dict)
    skills: dict[str, float] = field(default_factory=dict)
    decision_expr: str = "(+ resolve empathy)"
    alive: bool = True
    gov_preference: str = "anarchy"

    def lispy_bindings(self) -> dict[str, Any]:
        """Return bindings dict compatible with the full colonist surface."""
        b: dict[str, Any] = {}
        for name in MINI_STAT_NAMES:
            b[name] = self.stats.get(name, 0.5)
        for name in MINI_SKILL_NAMES:
            b[name] = self.skills.get(name, 0.1)
        return b

    def to_dict(self) -> dict:
        return {
            "id": self.id, "stats": dict(self.stats),
            "skills": dict(self.skills), "alive": self.alive,
            "gov_preference": self.gov_preference,
        }


def create_mini_colonists(count: int, rng: random.Random,
                          parent_bindings: dict[str, float] | None = None,
                          ) -> list[MiniColonist]:
    """Create ``count`` mini-colonists with randomised stats/skills.

    If *parent_bindings* is provided the colonists are seeded from the
    parent colonist's traits with gaussian noise.
    """
    colonists: list[MiniColonist] = []
    for i in range(count):
        stats: dict[str, float] = {}
        skills: dict[str, float] = {}
        for name in MINI_STAT_NAMES:
            base = (parent_bindings or {}).get(name, 0.5)
            stats[name] = max(0.0, min(1.0, base + rng.gauss(0, 0.12)))
        for name in MINI_SKILL_NAMES:
            base = (parent_bindings or {}).get(name, 0.1)
            skills[name] = max(0.0, min(1.0, base + rng.gauss(0, 0.08)))
        expr = rng.choice(MINI_DECISION_EXPRS)
        colonists.append(MiniColonist(
            id=f"mini-{i}", stats=stats, skills=skills,
            decision_expr=expr,
        ))
    return colonists


@dataclass
class MiniGovState:
    """Governance state inside a mini-world."""
    gov_type: str = "anarchy"
    stable_frames: int = 0
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"gov_type": self.gov_type, "stable_frames": self.stable_frames,
                "history": list(self.history)}


@dataclass
class WorldSimFrame:
    """One frame of a mini-world simulation."""
    frame: int
    event: dict
    resources: dict[str, float]
    gov_type: str
    alive_count: int
    actions: dict[str, float]  # colonist_id → LisPy score

    def to_dict(self) -> dict:
        return {"frame": self.frame, "event": self.event,
                "resources": dict(self.resources), "gov_type": self.gov_type,
                "alive_count": self.alive_count}


@dataclass
class WorldSimResult:
    """Complete result of a mini-world simulation."""
    depth: int
    colonist_id: str  # parent colonist who spawned this
    year: int  # parent year
    frames_run: int
    survived: bool
    final_resources: dict[str, float]
    governance_history: list[dict]
    dominant_governance: str
    governance_stable: bool
    stability_score: float  # 0-1, higher = more stable
    children: list["WorldSimResult"] = field(default_factory=list)
    child_subsims: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "depth": self.depth, "colonist_id": self.colonist_id,
            "year": self.year, "frames_run": self.frames_run,
            "survived": self.survived,
            "final_resources": dict(self.final_resources),
            "governance_history": self.governance_history,
            "dominant_governance": self.dominant_governance,
            "governance_stable": self.governance_stable,
            "stability_score": round(self.stability_score, 4),
        }
        if self.error:
            d["error"] = self.error
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        if self.child_subsims:
            d["child_subsims"] = self.child_subsims
        return d

    def compact_summary(self) -> dict:
        """Compact summary for storage in year results."""
        return {
            "depth": self.depth, "colonist_id": self.colonist_id,
            "frames_run": self.frames_run, "survived": self.survived,
            "dominant_gov": self.dominant_governance,
            "gov_stable": self.governance_stable,
            "stability": round(self.stability_score, 4),
            "children_count": len(self.children),
        }


@dataclass
class WorldSimBudget:
    """Weighted budget tracker for world simulations per year."""
    year: int
    units_used: float = 0.0
    max_units: float = MAX_BUDGET_UNITS_PER_YEAR
    sims_spawned: int = 0

    def cost(self, depth: int) -> float:
        """Compute cost of spawning a world-sim at given depth."""
        frames = FRAMES_BY_DEPTH.get(depth, 5)
        agents = AGENTS_BY_DEPTH.get(depth, 2)
        weight = BUDGET_UNIT_WEIGHT.get(depth, 0.25)
        return (frames * agents * weight) / 100.0

    def can_spawn(self, depth: int) -> bool:
        """Check if budget allows spawning at this depth."""
        if depth > MAX_WORLD_DEPTH:
            return False
        return self.units_used + self.cost(depth) <= self.max_units

    def record(self, depth: int) -> None:
        """Record a world-sim spawn."""
        self.units_used += self.cost(depth)
        self.sims_spawned += 1


class MiniWorld:
    """A sandboxed mini-colony simulation.

    Runs a simplified colony for *max_frames* years.  Colonists make
    decisions via LisPy expressions, resources tick, and governance
    proposals emerge via majority voting.

    Inherits the parent colony's constitution but may propose amendments
    within its scope.
    """

    def __init__(self, depth: int, colonist_id: str, year: int,
                 rng: random.Random,
                 parent_bindings: dict[str, float] | None = None,
                 parent_gov: str = "anarchy",
                 budget: WorldSimBudget | None = None) -> None:
        self.depth = depth
        self.colonist_id = colonist_id
        self.year = year
        self.rng = random.Random(rng.randint(0, 2**31))
        self.max_frames = FRAMES_BY_DEPTH.get(depth, 5)
        agent_count = AGENTS_BY_DEPTH.get(depth, 2)
        self.colonists = create_mini_colonists(
            agent_count, self.rng, parent_bindings)
        self.resources: dict[str, float] = {
            "food": 0.6, "water": 0.6, "power": 0.7,
        }
        self.gov = MiniGovState(gov_type=parent_gov)
        self.frames: list[WorldSimFrame] = []
        self.budget = budget
        self.child_results: list[WorldSimResult] = []

    def _pick_event(self) -> dict:
        """Choose a random event for this frame."""
        return self.rng.choice(MINI_EVENTS)

    def _tick_resources(self, event: dict, actions: dict[str, float]) -> None:
        """Update resources based on event effects and colonist actions."""
        alive_count = sum(1 for c in self.colonists if c.alive)
        consumption = alive_count * 0.04
        production = alive_count * 0.05

        for res in MINI_RESOURCES:
            delta = production - consumption
            delta += event.get(res, 0.0)
            # Positive LisPy scores boost production slightly
            avg_score = sum(actions.values()) / max(1, len(actions))
            if avg_score > 0.5:
                delta += (avg_score - 0.5) * 0.02
            current = self.resources.get(res, 0.5)
            self.resources[res] = max(0.0, min(1.0, current + delta))

    def _check_death(self, colonist: MiniColonist) -> bool:
        """Simplified death check — low resources increase risk."""
        if not colonist.alive:
            return False
        avg_res = sum(self.resources.values()) / len(MINI_RESOURCES)
        death_rate = 0.01
        if avg_res < 0.2:
            death_rate += (0.2 - avg_res) * 0.5
        paranoia = colonist.stats.get("paranoia", 0.5)
        if paranoia > 0.7:
            death_rate += 0.005
        return self.rng.random() < death_rate

    def _governance_vote(self) -> str | None:
        """Simplified governance — random proposal, majority vote."""
        if self.rng.random() > 0.2:
            return None
        alive = [c for c in self.colonists if c.alive]
        if len(alive) < 2:
            return None
        proposed = self.rng.choice(MINI_GOV_TYPES)
        if proposed == self.gov.gov_type:
            return None
        votes_for = 0
        for c in alive:
            empathy = c.stats.get("empathy", 0.5)
            resolve = c.stats.get("resolve", 0.5)
            if proposed == "council":
                score = empathy * 0.5 + resolve * 0.3
            elif proposed == "dictator":
                score = resolve * 0.6 - empathy * 0.2
            elif proposed == "consensus":
                score = empathy * 0.6 + c.stats.get("faith", 0.5) * 0.3
            else:  # anarchy
                score = c.stats.get("paranoia", 0.5) * 0.4 + 0.2
            if score + self.rng.gauss(0, 0.15) > 0.5:
                votes_for += 1
        if votes_for > len(alive) / 2:
            return proposed
        return None

    def _maybe_spawn_child(self, frame_num: int) -> WorldSimResult | None:
        """At depth < MAX, sometimes recurse one level deeper."""
        child_depth = self.depth + 1
        if child_depth > MAX_WORLD_DEPTH:
            return None
        if self.budget is not None and not self.budget.can_spawn(child_depth):
            return None
        if self.rng.random() > 0.15:
            return None
        alive = [c for c in self.colonists if c.alive]
        if not alive:
            return None
        spawner = self.rng.choice(alive)
        child_world = MiniWorld(
            depth=child_depth, colonist_id=spawner.id,
            year=self.year, rng=self.rng,
            parent_bindings=spawner.lispy_bindings(),
            parent_gov=self.gov.gov_type,
            budget=self.budget,
        )
        if self.budget is not None:
            self.budget.record(child_depth)
        child_result = child_world.run()
        self.child_results.append(child_result)
        return child_result

    def tick(self, frame_num: int) -> WorldSimFrame:
        """Advance the mini-world by one frame."""
        event = self._pick_event()
        alive = [c for c in self.colonists if c.alive]

        # Each colonist evaluates their LisPy expression
        actions: dict[str, float] = {}
        for c in alive:
            bindings = c.lispy_bindings()
            bindings.update(self.resources)
            bindings["sim-depth"] = self.depth
            bindings["sim-year"] = frame_num
            try:
                score = lispy_run(c.decision_expr,
                                  extra_bindings=bindings,
                                  max_steps=500)
                if not isinstance(score, (int, float)):
                    score = 0.5
                actions[c.id] = float(score)
            except (LispyError, Exception):
                actions[c.id] = 0.5

        self._tick_resources(event, actions)

        # Deaths
        for c in list(alive):
            if self._check_death(c):
                c.alive = False

        # Governance
        new_gov = self._governance_vote()
        if new_gov is not None:
            self.gov.history.append({
                "frame": frame_num, "from": self.gov.gov_type,
                "to": new_gov,
            })
            self.gov.gov_type = new_gov
            self.gov.stable_frames = 0
        else:
            self.gov.stable_frames += 1

        # Maybe spawn child world
        self._maybe_spawn_child(frame_num)

        frame = WorldSimFrame(
            frame=frame_num, event=event,
            resources=dict(self.resources),
            gov_type=self.gov.gov_type,
            alive_count=sum(1 for c in self.colonists if c.alive),
            actions=actions,
        )
        self.frames.append(frame)
        return frame

    def run(self) -> WorldSimResult:
        """Run the mini-world simulation to completion."""
        for f in range(1, self.max_frames + 1):
            alive = [c for c in self.colonists if c.alive]
            if not alive:
                break
            self.tick(f)

        # Analyse governance
        gov_counts: dict[str, int] = {}
        for frame in self.frames:
            gov_counts[frame.gov_type] = gov_counts.get(frame.gov_type, 0) + 1
        dominant = max(gov_counts, key=gov_counts.get) if gov_counts else "anarchy"
        gov_stable = self.gov.stable_frames >= GOVERNANCE_STABILITY_THRESHOLD
        stability_score = (self.gov.stable_frames /
                           max(1, len(self.frames)))

        survived = any(c.alive for c in self.colonists)

        return WorldSimResult(
            depth=self.depth, colonist_id=self.colonist_id,
            year=self.year, frames_run=len(self.frames),
            survived=survived,
            final_resources=dict(self.resources),
            governance_history=self.gov.history,
            dominant_governance=dominant,
            governance_stable=gov_stable,
            stability_score=stability_score,
            children=self.child_results,
            child_subsims=[c.compact_summary() for c in self.child_results],
        )


def spawn_world_sim(
    colonist_id: str,
    year: int,
    parent_bindings: dict[str, float],
    parent_gov: str,
    depth: int,
    rng: random.Random,
    budget: WorldSimBudget | None = None,
) -> WorldSimResult:
    """Top-level entry point: spawn a recursive world simulation.

    Returns a ``WorldSimResult`` with the full outcome, including any
    child world-sims that were spawned recursively.
    """
    if depth > MAX_WORLD_DEPTH:
        return WorldSimResult(
            depth=depth, colonist_id=colonist_id, year=year,
            frames_run=0, survived=False,
            final_resources={r: 0.0 for r in MINI_RESOURCES},
            governance_history=[], dominant_governance="anarchy",
            governance_stable=False, stability_score=0.0,
            error="max depth exceeded",
        )

    if budget is not None:
        if not budget.can_spawn(depth):
            return WorldSimResult(
                depth=depth, colonist_id=colonist_id, year=year,
                frames_run=0, survived=False,
                final_resources={r: 0.0 for r in MINI_RESOURCES},
                governance_history=[], dominant_governance="anarchy",
                governance_stable=False, stability_score=0.0,
                error="budget exhausted",
            )
        budget.record(depth)

    world = MiniWorld(
        depth=depth, colonist_id=colonist_id, year=year,
        rng=rng, parent_bindings=parent_bindings,
        parent_gov=parent_gov, budget=budget,
    )
    return world.run()
