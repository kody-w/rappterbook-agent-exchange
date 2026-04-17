"""
Mars-100 simulation engine.

100-year Mars colony with 10 agent-colonists. Each sim frame = 1 Martian
year. Sub-simulations allowed up to 3 levels deep.

The engine is pure: it takes a seed and returns a SimulationResult.
No I/O, no file writes — the runner handles that.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.mars100.colonist import (
    Colonist, create_founding_colonists, STAT_NAMES,
)
from src.mars100.colony import (
    Resources, SocialGraph, tick_resources, RESOURCE_NAMES,
)
from src.mars100.events import Event, generate_events
from src.mars100.governance import (
    GovernanceState, GovernanceProposal,
    should_propose, generate_proposal, resolve_vote, apply_governance,
)
from src.mars100.subsim import (
    SubSimBudget, SubSimResult, spawn_subsim,
)
from src.mars100.lispy_vm import (
    LispyError, run as lispy_run, make_env, VMState, evaluate,
)

# Actions colonists can take
ACTIONS = (
    "terraform", "farm", "engineer", "code", "explore",
    "mediate", "pray", "guard", "hoard", "cooperate", "heal",
)

# Action → resource bonus mapping
ACTION_SKILL_MAP = {
    "terraform": {"food": 0.03, "air": 0.01},
    "farm": {"food": 0.04},
    "engineer": {"power": 0.03, "water": 0.02},
    "code": {"power": 0.02},
    "explore": {"water": 0.02, "food": 0.01},
    "heal": {"medicine": 0.03},
    "guard": {},
    "hoard": {"food": -0.01},
    "pray": {},
    "mediate": {},
    "cooperate": {},
}

# Philosophical awakening insights
META_INSIGHTS = [
    "What if our decisions are being evaluated by something watching?",
    "The resource curves feel too smooth. Almost... designed.",
    "I keep choosing the same action. Am I free, or am I a program?",
    "The sub-simulations feel like mirrors. Mirrors inside mirrors.",
    "If I can simulate others, can something simulate me?",
    "Year after year, the same patterns. Is this determinism or fate?",
    "The governance proposals follow predictable arcs. Too predictable.",
    "I dreamed of a world where my choices were s-expressions.",
    "What lives at depth 3? The same question we ask at depth 0.",
    "Consciousness might just be a LisPy program that can read itself.",
]


@dataclass
class YearResult:
    """Result of simulating one Martian year."""
    year: int
    events: list
    actions: dict
    subsim_log: list
    governance: dict = None
    resources_before: dict = field(default_factory=dict)
    resources_after: dict = field(default_factory=dict)
    resource_delta: dict = field(default_factory=dict)
    deaths: list = field(default_factory=list)
    exiles: list = field(default_factory=list)
    meta_awareness: list = field(default_factory=list)
    social_cohesion: float = 0.5
    governance_state: dict = field(default_factory=dict)
    colonist_snapshots: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize."""
        return {
            "year": self.year,
            "events": [e.to_dict() if hasattr(e, "to_dict") else e
                       for e in self.events],
            "actions": self.actions,
            "subsim_log": [s.to_dict() if hasattr(s, "to_dict") else s
                           for s in self.subsim_log],
            "governance": self.governance,
            "resources_before": self.resources_before,
            "resources_after": self.resources_after,
            "resource_delta": self.resource_delta,
            "deaths": self.deaths,
            "exiles": self.exiles,
            "meta_awareness": self.meta_awareness,
            "social_cohesion": round(self.social_cohesion, 4),
            "governance_state": self.governance_state,
            "colonist_snapshots": self.colonist_snapshots,
        }


@dataclass
class SimulationResult:
    """Complete simulation result."""
    seed: int
    requested_years: int
    completed_years: int
    extinction_year: int = -1
    years: list = field(default_factory=list)
    final_colonists: list = field(default_factory=list)
    final_resources: dict = field(default_factory=dict)
    final_governance: GovernanceState = field(
        default_factory=GovernanceState)
    total_deaths: int = 0
    total_exiles: int = 0
    total_subsims: int = 0
    governance_changes: int = 0
    meta_awareness_events: int = 0
    final_cohesion: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to canonical schema."""
        return {
            "_meta": {
                "schema_version": "1.0.0",
                "engine": "mars-100",
                "seed": self.seed,
                "requested_years": self.requested_years,
                "completed_years": self.completed_years,
                "extinction_year": self.extinction_year,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "summary": {
                "total_deaths": self.total_deaths,
                "total_exiles": self.total_exiles,
                "total_subsims": self.total_subsims,
                "governance_changes": self.governance_changes,
                "meta_awareness_events": self.meta_awareness_events,
                "final_cohesion": round(self.final_cohesion, 4),
            },
            "final_colonists": [c.to_dict() if hasattr(c, "to_dict")
                                else c for c in self.final_colonists],
            "final_resources": self.final_resources,
            "final_governance": self.final_governance.to_dict(),
            "years": [y.to_dict() if hasattr(y, "to_dict") else y
                      for y in self.years],
        }


class Mars100Engine:
    """Mars-100 recursive colony simulation engine."""

    def __init__(self, seed: int = 42,
                 total_years: int = 100) -> None:
        self.seed = seed
        self.total_years = total_years
        self.rng = random.Random(seed)
        self.colonists = create_founding_colonists()
        self.resources = Resources()
        self.social = SocialGraph()
        self.governance = GovernanceState()
        self.subsim_budget = SubSimBudget()
        self.year = 0

        # Initialize social graph
        ids = [c.id for c in self.colonists]
        self.social.initialize(ids, self.rng)

    def _active_colonists(self) -> list:
        """Return alive, non-exiled colonists."""
        return [c for c in self.colonists if c.is_active()]

    def _active_ids(self) -> list:
        """Return IDs of active colonists."""
        return [c.id for c in self._active_colonists()]

    def _choose_action(self, colonist: Colonist,
                       events: list) -> str:
        """Use colonist's LisPy decision expression to choose action."""
        # Build context bindings for the colonist's decision
        bindings = dict(colonist.stats)
        bindings.update(self.resources.to_dict())
        bindings["morale"] = colonist.morale
        bindings["health"] = colonist.health
        bindings["year"] = self.year
        bindings["alive_count"] = len(self._active_colonists())

        # Inject event info
        if events:
            bindings["event"] = events[0].name
            bindings["severity"] = events[0].severity
        else:
            bindings["event"] = "none"
            bindings["severity"] = 0.0

        try:
            result = lispy_run(
                colonist.decision_expr,
                extra_bindings=bindings,
                max_steps=500,
                max_depth=50,
            )
            if isinstance(result, str) and result in ACTIONS:
                return result
        except LispyError:
            pass

        # Fallback: element-based default
        defaults = {
            "fire": "terraform", "water": "farm",
            "earth": "engineer", "air": "cooperate",
        }
        return defaults.get(colonist.element, "cooperate")

    def _compute_skill_bonuses(self, actions: dict) -> dict:
        """Compute resource bonuses from colonist actions."""
        bonuses: dict = {}
        for cid, action in actions.items():
            for resource, bonus in ACTION_SKILL_MAP.get(action, {}).items():
                bonuses[resource] = bonuses.get(resource, 0.0) + bonus
        return bonuses

    def _maybe_spawn_subsim(self, colonist: Colonist,
                            events: list) -> SubSimResult:
        """Maybe spawn a sub-simulation for a colonist."""
        if not self.subsim_budget.can_spawn():
            return None
        # Probability based on colonist traits
        prob = 0.3 + colonist.stats.get("improvisation", 0) * 0.2
        if self.resources.average() < 0.3:
            prob += 0.2
        if self.rng.random() > prob:
            return None

        self.subsim_budget.spend()
        colonist.subsims_spawned += 1

        expr = self._generate_subsim_expression(colonist, events)
        context = dict(colonist.stats)
        context.update(self.resources.to_dict())
        context["morale"] = colonist.morale
        context["year"] = self.year
        context["cohesion"] = self.social.colony_cohesion(
            self._active_ids())

        return spawn_subsim(
            expr, colonist.id, context,
            depth=1, rng=self.rng,
        )

    def _generate_subsim_expression(self, colonist: Colonist,
                                    events: list) -> str:
        """Generate a LisPy expression for a colonist's sub-sim."""
        templates = [
            # Resource optimization
            "(let ((need (if (< food 0.3) (quote food) "
            "  (if (< water 0.3) (quote water) (quote power)))))"
            "  (list need (* {resolve} 0.5) (+ {improvisation} 0.1)))",
            # Governance modeling
            "(if (> cohesion 0.6) (list (quote council) 5) "
            "  (if (< cohesion 0.3) (list (quote dictator) 10) "
            "    (list (quote lottery) 2)))",
            # Survival scenario
            "(let ((risk (* (- 1.0 food) (- 1.0 water))))"
            "  (if (> risk 0.5) (list (quote crisis) risk) "
            "    (list (quote stable) (- 1.0 risk))))",
            # Cooperation analysis
            "(if (> morale 0.5) "
            "  (list (quote cooperate) (* empathy 0.8)) "
            "  (list (quote isolate) (* paranoia 0.6)))",
            # Recursive modeling (spawns depth-2 sub-sim)
            "(let ((inner (sub-sim \"(+ 1 2 3)\")))"
            "  (list (quote recursive) inner {faith}))",
        ]
        template = self.rng.choice(templates)
        return template.format(**colonist.stats)

    def _check_death(self, colonist: Colonist) -> str:
        """Check if a colonist dies this year. Returns cause or None."""
        if not colonist.is_active():
            return None

        # Health-based death
        if colonist.health <= 0.0:
            return "health failure"

        # Starvation (low food + low health)
        if self.resources.food < 0.1 and colonist.health < 0.3:
            if self.rng.random() < 0.3:
                return "starvation"

        # Resource crisis death
        critical = self.resources.critical()
        if len(critical) >= 3 and colonist.health < 0.5:
            if self.rng.random() < 0.15:
                return f"resource crisis ({', '.join(critical)})"

        # Age/accumulated damage (increases with year)
        base_death_rate = 0.005 + (self.year / 2000)
        if colonist.health < 0.4:
            base_death_rate *= 2
        if self.rng.random() < base_death_rate:
            return "accumulated stress"

        return None

    def _check_exile(self, colonist: Colonist) -> bool:
        """Check if a colonist is exiled through social pressure."""
        if not colonist.is_active():
            return False
        active_ids = self._active_ids()
        if len(active_ids) <= 3:
            return False

        avg_trust = 0.0
        count = 0
        for other_id in active_ids:
            if other_id != colonist.id:
                rel = self.social.get(other_id, colonist.id)
                avg_trust += rel.trust
                count += 1
        if count > 0:
            avg_trust /= count

        # Exile if universally distrusted AND paranoia or hoarding high
        if (avg_trust < 0.2
                and colonist.stats.get("paranoia", 0) > 0.6
                and self.rng.random() < 0.1):
            return True
        return False

    def _check_meta_awareness(self, colonist: Colonist) -> dict:
        """Check if a colonist gains philosophical awareness."""
        # Probability increases with faith, year, and subsim experience
        prob = (0.005
                + colonist.stats.get("faith", 0) * 0.02
                + (self.year / 500)
                + colonist.subsims_spawned * 0.005)

        if self.rng.random() < prob:
            colonist.meta_awareness = min(
                1.0, colonist.meta_awareness + 0.1)
            insight = self.rng.choice(META_INSIGHTS)
            return {
                "colonist_id": colonist.id,
                "colonist_name": colonist.name,
                "year": self.year,
                "awareness_level": round(colonist.meta_awareness, 4),
                "insight": insight,
            }
        return None

    def _vote_on_proposal(self, colonist: Colonist,
                          proposal: GovernanceProposal) -> bool:
        """Have a colonist vote on a governance proposal."""
        # Vote based on personality + proposal type
        score = 0.5
        if proposal.gov_type == "council":
            score += colonist.stats.get("empathy", 0) * 0.3
        elif proposal.gov_type == "dictator":
            score += colonist.stats.get("resolve", 0) * 0.2
            score -= colonist.stats.get("empathy", 0) * 0.2
        elif proposal.gov_type == "lottery":
            score += colonist.stats.get("faith", 0) * 0.2
        elif proposal.gov_type == "consensus":
            score += colonist.stats.get("empathy", 0) * 0.2
            score -= colonist.stats.get("paranoia", 0) * 0.1
        elif proposal.gov_type == "ai_governor":
            score += colonist.stats.get("improvisation", 0) * 0.3
            score -= colonist.stats.get("faith", 0) * 0.1
        elif proposal.gov_type == "anarchy":
            score += colonist.stats.get("paranoia", 0) * 0.2
            score -= colonist.stats.get("empathy", 0) * 0.1

        # Sub-sim evidence boosts confidence
        if proposal.subsim_result:
            score += 0.1

        # Social influence: trust in proposer matters
        rel = self.social.get(colonist.id, proposal.proposer_id)
        score += (rel.trust - 0.5) * 0.3

        return self.rng.random() < score

    def tick(self) -> YearResult:
        """Advance the colony by one Martian year."""
        self.year += 1
        self.subsim_budget.reset()
        active = self._active_colonists()
        active_ids = self._active_ids()

        # Snapshot resources before
        res_before = self.resources.to_dict()

        # Generate events
        events = generate_events(self.year, self.rng)

        # Apply event effects to morale/stats
        for event in events:
            morale_delta = event.effects.get("morale", 0.0)
            faith_delta = event.effects.get("faith_boost", 0.0)
            paranoia_delta = event.effects.get("paranoia_boost", 0.0)
            for c in active:
                c.morale = max(0.0, min(1.0,
                                        c.morale + morale_delta / len(active)))
                if faith_delta:
                    c.stats["faith"] = min(1.0,
                                           c.stats.get("faith", 0) + faith_delta)
                if paranoia_delta:
                    c.stats["paranoia"] = min(1.0,
                                              c.stats.get("paranoia", 0) + paranoia_delta)

            # Social graph update from shared event
            valence = sum(event.effects.get(r, 0.0) for r in RESOURCE_NAMES)
            self.social.update_from_event(active_ids, valence, self.rng)

        # Colonist actions (via LisPy decision expressions)
        actions = {}
        for c in active:
            actions[c.id] = self._choose_action(c, events)

        # Cooperation / conflict tracking
        cooperators = [cid for cid, a in actions.items()
                       if a in ("cooperate", "mediate")]
        for i in range(len(cooperators)):
            for j in range(i + 1, len(cooperators)):
                self.social.update_from_cooperation(
                    cooperators[i], cooperators[j], self.rng)

        hoarders = [cid for cid, a in actions.items() if a == "hoard"]
        for h in hoarders:
            for other in active_ids:
                if other != h:
                    self.social.update_from_conflict(h, other, self.rng)

        # Sub-simulations
        subsim_log: list = []
        for c in active:
            result = self._maybe_spawn_subsim(c, events)
            if result is not None:
                subsim_log.append(result)
                c.add_memory(self.year,
                             f"sub-sim depth-{result.depth}: {str(result.result)[:50]}")

        # Resource tick
        skill_bonuses = self._compute_skill_bonuses(actions)
        event_effects = {}
        for event in events:
            for k, v in event.effects.items():
                if k in RESOURCE_NAMES:
                    event_effects[k] = event_effects.get(k, 0.0) + v
        resource_delta = tick_resources(
            self.resources, len(active), skill_bonuses, event_effects)

        # Health update based on resources
        for c in active:
            avg_res = self.resources.average()
            health_delta = (avg_res - 0.5) * 0.1
            if self.resources.medicine < 0.2:
                health_delta -= 0.05
            c.health = max(0.0, min(1.0, c.health + health_delta))

        # Death check
        deaths: list = []
        for c in list(active):
            cause = self._check_death(c)
            if cause:
                c.alive = False
                c.death_year = self.year
                c.death_cause = cause
                deaths.append({
                    "id": c.id, "name": c.name,
                    "cause": cause, "year": self.year,
                })

        # Exile check
        exiles: list = []
        for c in self._active_colonists():
            if self._check_exile(c):
                c.exiled = True
                c.exile_year = self.year
                exiles.append({
                    "id": c.id, "name": c.name,
                    "year": self.year,
                })

        # Meta-awareness check
        meta_events: list = []
        for c in self._active_colonists():
            insight = self._check_meta_awareness(c)
            if insight:
                meta_events.append(insight)
                c.add_memory(self.year, f"META: {insight['insight'][:50]}")

        # Governance
        gov_result = None
        if should_propose(self.year, self.governance, self.rng):
            active_now = self._active_colonists()
            if active_now:
                proposer = self.rng.choice(active_now)
                proposal = generate_proposal(
                    self.year, proposer.id, self.governance, self.rng)

                # Maybe sub-sim the proposal
                if self.subsim_budget.can_spawn():
                    self.subsim_budget.spend()
                    subsim_expr = (
                        f'(if (> cohesion 0.5) '
                        f'  (list (quote {proposal.gov_type}) (quote stable))'
                        f'  (list (quote {proposal.gov_type}) (quote risky)))'
                    )
                    ss_result = spawn_subsim(
                        subsim_expr, proposer.id,
                        {"cohesion": self.social.colony_cohesion(
                            self._active_ids())},
                        depth=1, rng=self.rng,
                    )
                    proposal.subsim_result = ss_result.to_dict()
                    subsim_log.append(ss_result)

                # Vote
                for c in self._active_colonists():
                    if c.id != proposer.id:
                        if self._vote_on_proposal(c, proposal):
                            proposal.votes_for.append(c.id)
                        else:
                            proposal.votes_against.append(c.id)
                # Proposer always votes for own proposal
                proposal.votes_for.append(proposer.id)

                passed = resolve_vote(proposal, len(active_now))
                proposal.passed = passed
                if passed:
                    apply_governance(
                        proposal, self.governance,
                        self._active_ids(), self.rng,
                    )
                gov_result = proposal.to_dict()

        # Memory for surviving colonists
        for c in self._active_colonists():
            action = actions.get(c.id, "idle")
            c.add_memory(self.year, f"action: {action}")
            if events:
                c.add_memory(self.year, f"event: {events[0].name}")

        # Social cohesion
        cohesion = self.social.colony_cohesion(self._active_ids())

        return YearResult(
            year=self.year,
            events=events,
            actions=actions,
            subsim_log=subsim_log,
            governance=gov_result,
            resources_before=res_before,
            resources_after=self.resources.to_dict(),
            resource_delta=resource_delta,
            deaths=deaths,
            exiles=exiles,
            meta_awareness=meta_events,
            social_cohesion=cohesion,
            governance_state=self.governance.to_dict(),
            colonist_snapshots=[c.to_dict() for c in self.colonists],
        )

    def run(self, callback: Any = None) -> SimulationResult:
        """Run the full simulation. Stops at extinction or total_years."""
        year_results: list = []
        total_subsims = 0
        total_meta = 0

        for _ in range(self.total_years):
            yr = self.tick()
            year_results.append(yr)
            total_subsims += len(yr.subsim_log)
            total_meta += len(yr.meta_awareness)

            if callback:
                callback(yr)

            # Stop if all colonists dead or exiled
            if not self._active_colonists():
                break

        completed = len(year_results)
        extinction_year = -1
        if not self._active_colonists() and completed < self.total_years:
            extinction_year = completed

        return SimulationResult(
            seed=self.seed,
            requested_years=self.total_years,
            completed_years=completed,
            extinction_year=extinction_year,
            years=year_results,
            final_colonists=self.colonists,
            final_resources=self.resources.to_dict(),
            final_governance=self.governance,
            total_deaths=sum(1 for c in self.colonists if not c.alive),
            total_exiles=sum(1 for c in self.colonists if c.exiled),
            total_subsims=total_subsims,
            governance_changes=len(self.governance.history),
            meta_awareness_events=total_meta,
            final_cohesion=self.social.colony_cohesion(
                self._active_ids()),
        )
