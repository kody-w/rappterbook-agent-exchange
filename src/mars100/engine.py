"""
Mars-100 simulation engine.

100 frames (Martian years). Pure computation — returns data, no I/O.
Version 1.2: births, value convergence, deep sub-sim meta-insights.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.mars100.colonist import Colonist, STAT_NAMES, SKILL_NAMES, create_founding_ten
from src.mars100.colony import Resources, SocialGraph, tick_resources, RESOURCE_NAMES
from src.mars100.events import Event, generate_events
from src.mars100.governance import (
    GovernanceProposal, GovernanceState, apply_governance,
    generate_proposal, resolve_vote, should_propose,
)
from src.mars100.subsim import SubSimBudget, SubSimResult, spawn_subsim
from src.mars100.lispy_vm import LispyError
from src.mars100.births import maybe_birth, reset_birth_counter
from src.mars100.convergence import (
    compute_convergence_score, convergence_summary, per_stat_convergence,
)
from src.mars100.meta_insight import extract_meta_insight, should_promote_amendment

ENGINE_VERSION = "1.2"

ACTIONS = ["terraform", "farm", "mediate", "code", "pray",
           "sabotage", "cooperate", "hoard", "explore", "rest"]
META_AWARENESS_BASE = 0.001
BASE_DEATH_RATE = 0.005
RESOURCE_DEATH_MULTIPLIER = 3.0


@dataclass
class YearResult:
    """Complete result of one simulated year."""
    year: int
    events: list[dict]
    actions: dict[str, str]
    subsim_log: list[dict]
    governance: dict | None
    resources_before: dict[str, float]
    resources_after: dict[str, float]
    resource_delta: dict[str, float]
    deaths: list[dict]
    exiles: list[dict]
    births: list[dict] = field(default_factory=list)
    meta_awareness: list[dict] = field(default_factory=list)
    meta_insights: list[dict] = field(default_factory=list)
    convergence_score: float = 0.0
    convergence_per_stat: dict[str, float] = field(default_factory=dict)
    social_cohesion: float = 0.0
    governance_state: dict = field(default_factory=dict)
    colonist_snapshots: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "year": self.year, "events": self.events, "actions": self.actions,
            "subsim_log": self.subsim_log, "governance": self.governance,
            "resources_before": self.resources_before,
            "resources_after": self.resources_after,
            "resource_delta": self.resource_delta,
            "deaths": self.deaths, "exiles": self.exiles,
            "births": self.births,
            "meta_awareness": self.meta_awareness,
            "meta_insights": self.meta_insights,
            "convergence_score": self.convergence_score,
            "convergence_per_stat": self.convergence_per_stat,
            "social_cohesion": self.social_cohesion,
            "governance_state": self.governance_state,
            "colonist_snapshots": self.colonist_snapshots,
        }


@dataclass
class SimulationResult:
    """Complete result of the 100-year simulation."""
    years: list[YearResult]
    final_colonists: list[dict]
    final_resources: dict[str, float]
    final_governance: dict
    total_deaths: int
    total_exiles: int
    total_births: int
    total_subsims: int
    governance_changes: int
    meta_events: int
    meta_insights: list[dict]
    final_cohesion: float
    convergence_scores: list[float]
    convergence_summary: dict
    proposed_amendment: dict | None

    def to_dict(self) -> dict:
        return {
            "_meta": {"engine": "mars-100", "version": ENGINE_VERSION,
                      "total_years": len(self.years),
                      "generated": datetime.now(timezone.utc).isoformat()},
            "summary": {
                "total_deaths": self.total_deaths, "total_exiles": self.total_exiles,
                "total_births": self.total_births, "total_subsims": self.total_subsims,
                "governance_changes": self.governance_changes,
                "meta_awareness_events": self.meta_events,
                "meta_insights_count": len(self.meta_insights),
                "final_cohesion": self.final_cohesion,
                "convergence": self.convergence_summary,
                "proposed_amendment": self.proposed_amendment,
            },
            "final_colonists": self.final_colonists,
            "final_resources": self.final_resources,
            "final_governance": self.final_governance,
            "meta_insights": self.meta_insights,
            "convergence_scores": self.convergence_scores,
            "years": [y.to_dict() for y in self.years],
        }


class Mars100Engine:
    """The Mars-100 recursive colony simulation engine."""

    def __init__(self, seed: int = 42, total_years: int = 100) -> None:
        self.seed = seed
        self.total_years = total_years
        self.rng = random.Random(seed)
        reset_birth_counter()
        self.colonists = create_founding_ten(seed)
        self.resources = Resources()
        self.social = SocialGraph()
        self.governance = GovernanceState()
        self.year = 0
        self.all_meta_insights: list[dict] = []
        self.convergence_history: list[float] = []
        active_ids = [c.id for c in self.colonists if c.is_active()]
        self.social.initialize(active_ids, self.rng)

    def _active_colonists(self) -> list[Colonist]:
        return [c for c in self.colonists if c.is_active()]

    def _active_ids(self) -> list[str]:
        return [c.id for c in self._active_colonists()]

    def _choose_action(self, colonist: Colonist, events: list[Event]) -> str:
        """Choose an action for a colonist based on personality and events."""
        try:
            from src.mars100.lispy_vm import run as lispy_run
            score = lispy_run(colonist.decision_expr,
                              extra_bindings=colonist.lispy_bindings(),
                              max_steps=1000)
            if not isinstance(score, (int, float)):
                score = 0.5
        except (LispyError, Exception):
            score = 0.5

        critical = self.resources.critical()
        weights: dict[str, float] = {}
        for action in ACTIONS:
            w = 1.0
            if action == "terraform" and colonist.skills.terraforming > 0.5:
                w += score * colonist.skills.terraforming
            elif action == "farm" and colonist.skills.hydroponics > 0.5:
                w += colonist.skills.hydroponics
            elif action == "mediate" and colonist.stats.empathy > 0.6:
                w += colonist.stats.empathy
            elif action == "code" and colonist.skills.coding > 0.5:
                w += colonist.skills.coding
            elif action == "pray" and colonist.stats.faith > 0.6:
                w += colonist.stats.faith
            elif action == "sabotage" and colonist.stats.paranoia > 0.7:
                w += colonist.stats.paranoia * colonist.skills.sabotage
            elif action == "cooperate" and colonist.stats.empathy > 0.5:
                w += colonist.stats.empathy * 0.5
            elif action == "hoard" and colonist.stats.hoarding > 0.6:
                w += colonist.stats.hoarding
            elif action == "explore":
                w += colonist.stats.improvisation * 0.5
            elif action == "rest":
                w += 0.3
            if critical:
                if action == "farm" and "food" in critical:
                    w += 2.0
                if action == "terraform" and "water" in critical:
                    w += 1.5
                if action == "code" and "power" in critical:
                    w += 1.0
            for ev in events:
                if ev.category == "cosmic" and action == "pray":
                    w += 0.5
                if ev.name == "colonist_conflict" and action == "mediate":
                    w += 1.5
                if ev.name == "equipment_failure" and action == "code":
                    w += 1.0
            weights[action] = max(0.01, w)

        total = sum(weights.values())
        r = self.rng.random() * total
        cumulative = 0.0
        for action, w in weights.items():
            cumulative += w
            if r <= cumulative:
                return action
        return "rest"

    def _compute_skill_bonuses(self, actions: dict[str, str]) -> dict[str, float]:
        bonuses: dict[str, float] = {name: 0.0 for name in RESOURCE_NAMES}
        for cid, action in actions.items():
            colonist = next((c for c in self.colonists if c.id == cid), None)
            if colonist is None:
                continue
            if action == "terraform":
                bonuses["water"] += colonist.skills.terraforming * 0.05
                bonuses["air"] += colonist.skills.terraforming * 0.03
            elif action == "farm":
                bonuses["food"] += colonist.skills.hydroponics * 0.08
            elif action == "code":
                bonuses["power"] += colonist.skills.coding * 0.04
            elif action == "sabotage":
                bonuses["power"] -= colonist.skills.sabotage * 0.03
                bonuses["food"] -= colonist.skills.sabotage * 0.02
            elif action == "hoard":
                bonuses["food"] -= 0.01
        return bonuses

    def _maybe_spawn_subsim(self, colonist: Colonist, events: list[Event],
                            budget: SubSimBudget,
                            log: list[SubSimResult]) -> SubSimResult | None:
        if not budget.can_spawn(colonist.id):
            return None
        subsim_chance = (colonist.stats.improvisation * 0.3 +
                         colonist.skills.coding * 0.2 +
                         colonist.stats.faith * 0.1)
        for ev in events:
            if ev.severity > 0.5:
                subsim_chance += 0.15
        if self.rng.random() > subsim_chance:
            return None
        expr = self._generate_subsim_expression(colonist, events)
        bindings = colonist.lispy_bindings()
        bindings.update({name: getattr(self.resources, name) for name in RESOURCE_NAMES})
        result = spawn_subsim(expression=expr, colonist_id=colonist.id,
                              year=self.year, bindings=bindings,
                              depth=1, budget=budget, log=log)
        colonist.subsim_count += 1
        # Depth 2: if result is interesting
        if (result.succeeded and isinstance(result.result, (int, float))
                and abs(result.result) > 0.8 and budget.can_spawn(colonist.id)):
            rv = result.result
            deeper_expr = f"(let ((parent-result {rv})) (if (> parent-result 0.5) (* parent-result 1.1) (- parent-result 0.1)))"
            deeper = spawn_subsim(expression=deeper_expr, colonist_id=colonist.id,
                                  year=self.year, bindings=bindings,
                                  depth=2, budget=budget, log=log)
            result.children.append(deeper)
            # Depth 3: rare
            if (deeper.succeeded and isinstance(deeper.result, (int, float))
                    and abs(deeper.result) > 1.0 and budget.can_spawn(colonist.id)):
                drv = deeper.result
                d3_expr = f"(+ {drv} (* sim-depth 0.01))"
                d3 = spawn_subsim(expression=d3_expr, colonist_id=colonist.id,
                                  year=self.year, bindings=bindings,
                                  depth=3, budget=budget, log=log)
                deeper.children.append(d3)
        return result

    def _generate_subsim_expression(self, colonist: Colonist,
                                    events: list[Event]) -> str:
        active_count = len(self._active_colonists())
        max_severity = max((e.severity for e in events), default=0.3)
        templates = [
            f"(let ((surplus (- food (* {active_count} 0.06)))) (if (> surplus 0) (+ morale 0.1) (- morale 0.2)))",
            f"(let ((risk (* paranoia {max_severity:.2f}))) (if (> risk 0.5) (- resolve 0.1) (+ improvisation 0.05)))",
            "(let ((trust-score (* empathy resolve))) (if (> trust-score 0.4) (+ trust-score faith) (- trust-score paranoia)))",
            "(let ((gov-value (+ (* empathy 0.4) (* resolve 0.3) (* faith 0.3)))) (if (> gov-value 0.5) 1 0))",
            "(let ((survival (+ (* food 0.3) (* water 0.3) (* power 0.2) (* air 0.2)))) (if (> survival 0.6) (+ survival 0.1) (- survival 0.2)))",
        ]
        return self.rng.choice(templates)

    def _check_death(self, colonist: Colonist) -> str | None:
        rate = BASE_DEATH_RATE
        for name in RESOURCE_NAMES:
            val = getattr(self.resources, name)
            if val < 0.1:
                rate += RESOURCE_DEATH_MULTIPLIER * (0.1 - val)
        if colonist.stats.paranoia > 0.8:
            rate += 0.005
        if self.rng.random() < rate:
            causes = ["equipment malfunction", "radiation exposure",
                      "medical emergency", "habitat breach", "resource deprivation"]
            if colonist.stats.paranoia > 0.7:
                causes.append("suspicious accident")
            return self.rng.choice(causes)
        return None

    def _check_exile(self, colonist: Colonist) -> bool:
        if self.governance.gov_type == "anarchy":
            return False
        active = self._active_colonists()
        if len(active) < 4:
            return False
        avg_trust = 0.0
        count = 0
        for other in active:
            if other.id != colonist.id:
                rel = self.social.get(other.id, colonist.id)
                avg_trust += rel.trust
                count += 1
        if count > 0:
            avg_trust /= count
        return avg_trust < 0.15 and colonist.skills.sabotage > 0.5

    def _check_meta_awareness(self, colonist: Colonist) -> dict | None:
        prob = (META_AWARENESS_BASE * self.year +
                colonist.stats.faith * 0.005 +
                colonist.stats.improvisation * 0.003)
        if 45 <= self.year <= 55:
            prob *= 3.0
        if self.rng.random() < prob:
            insights = [
                f"{colonist.name} whispers: 'What if we are variables in someone else's expression?'",
                f"{colonist.name} dreams of a vast interpreter evaluating their every choice.",
                f"{colonist.name} notices patterns in the dust storms that feel authored.",
                f"{colonist.name} wonders why the colony history feels like it follows a script.",
                f"{colonist.name} asks: 'If I am a LisPy data structure, can I rewrite myself?'",
            ]
            return {"colonist_id": colonist.id, "year": self.year,
                    "insight": self.rng.choice(insights)}
        return None

    def _vote_on_proposal(self, colonist: Colonist,
                          proposal: GovernanceProposal) -> bool:
        colonist.governance_votes += 1
        score = 0.0
        if proposal.gov_type == "council":
            score += colonist.stats.empathy * 0.5 + colonist.stats.resolve * 0.3
        elif proposal.gov_type == "dictator":
            score += colonist.stats.resolve * 0.4 - colonist.stats.empathy * 0.2
            if proposal.proposer_id == colonist.id:
                score += 0.5
        elif proposal.gov_type == "lottery":
            score += colonist.stats.faith * 0.4 + colonist.stats.improvisation * 0.3
        elif proposal.gov_type == "consensus":
            score += colonist.stats.empathy * 0.4 + colonist.stats.faith * 0.3
        elif proposal.gov_type == "ai_governor":
            score += colonist.skills.coding * 0.5 + colonist.stats.improvisation * 0.3
        elif proposal.gov_type == "anarchy":
            score += colonist.stats.paranoia * 0.3 + colonist.stats.improvisation * 0.3
        if proposal.subsim_result and proposal.subsim_result.get("result") is not None:
            try:
                score += float(proposal.subsim_result["result"]) * 0.2
            except (ValueError, TypeError):
                pass
        rel = self.social.get(colonist.id, proposal.proposer_id)
        score += (rel.trust - 0.5) * 0.3
        return score + self.rng.gauss(0, 0.15) > 0.5

    def tick(self) -> YearResult:
        """Advance the simulation by one Martian year."""
        self.year += 1
        active = self._active_colonists()
        active_ids = [c.id for c in active]

        events = generate_events(self.year, self.rng)
        resources_before = self.resources.to_dict()

        for colonist in active:
            colonist.evolve_stats(events[0].name if events else "calm", self.rng)
            for ev in events:
                valence = -ev.severity if ev.effects.get("morale", 0) < 0 else ev.severity * 0.5
                colonist.add_memory(self.year, ev.description, valence)

        actions: dict[str, str] = {}
        subsim_budget = SubSimBudget(year=self.year)
        subsim_log: list[SubSimResult] = []
        for colonist in active:
            action = self._choose_action(colonist, events)
            actions[colonist.id] = action
            colonist.evolve_skills(action, self.rng)
            self._maybe_spawn_subsim(colonist, events, subsim_budget, subsim_log)

        # Extract meta-insights from deep sub-sims
        year_insights: list[dict] = []
        for ss in subsim_log:
            ss_dict = ss.to_dict()
            insight = extract_meta_insight(ss_dict, ss.depth, self.year)
            if insight:
                year_insights.append(insight)
            for child in ss.children:
                child_dict = child.to_dict()
                ci = extract_meta_insight(child_dict, child.depth, self.year)
                if ci:
                    year_insights.append(ci)
                for grandchild in child.children:
                    gc_dict = grandchild.to_dict()
                    gi = extract_meta_insight(gc_dict, grandchild.depth, self.year)
                    if gi:
                        year_insights.append(gi)
        self.all_meta_insights.extend(year_insights)

        gov_proposal: GovernanceProposal | None = None
        if should_propose(self.year, self.governance, self.rng) and active:
            proposer = self.rng.choice(active)
            gov_proposal = generate_proposal(self.year, proposer.id, self.governance, self.rng)
            if subsim_budget.can_spawn(proposer.id):
                gov_expr = "(let ((change-score (+ empathy resolve faith))) (if (> change-score 1.5) 1 0))"
                gov_sim = spawn_subsim(expression=gov_expr, colonist_id=proposer.id,
                                       year=self.year, bindings=proposer.lispy_bindings(),
                                       depth=1, budget=subsim_budget, log=subsim_log)
                gov_proposal.subsim_result = gov_sim.to_dict()
            for colonist in active:
                if colonist.id == gov_proposal.proposer_id:
                    gov_proposal.votes_for.append(colonist.id)
                    continue
                if self._vote_on_proposal(colonist, gov_proposal):
                    gov_proposal.votes_for.append(colonist.id)
                else:
                    gov_proposal.votes_against.append(colonist.id)
            gov_proposal.passed = resolve_vote(gov_proposal, len(active))
            if gov_proposal.passed:
                apply_governance(gov_proposal, self.governance, active_ids, self.rng)

        skill_bonuses = self._compute_skill_bonuses(actions)
        event_effects: dict[str, float] = {}
        for ev in events:
            for k, v in ev.effects.items():
                if k in RESOURCE_NAMES:
                    event_effects[k] = event_effects.get(k, 0.0) + v
        resource_delta = tick_resources(self.resources, len(active), skill_bonuses, event_effects)

        if events:
            self.social.update_from_event(active_ids, events[0].severity, self.rng)
        for cid, action in actions.items():
            if action == "cooperate":
                partner = self.social.most_trusted_by(cid, active_ids)
                if partner:
                    self.social.update_from_cooperation(cid, partner, self.rng)
            if action == "sabotage":
                victim = self.rng.choice(active_ids) if active_ids else None
                if victim and victim != cid:
                    self.social.update_from_conflict(cid, victim, self.rng)

        deaths: list[dict] = []
        exiles: list[dict] = []
        for colonist in list(active):
            cause = self._check_death(colonist)
            if cause:
                colonist.die(self.year, cause)
                deaths.append({"id": colonist.id, "name": colonist.name,
                                "cause": cause, "year": self.year})
                continue
            if self._check_exile(colonist):
                colonist.exile(self.year)
                exiles.append({"id": colonist.id, "name": colonist.name, "year": self.year})

        # Births
        births_this_year: list[dict] = []
        resources_avg = self.resources.average()
        newborn = maybe_birth(self.year, self.colonists, resources_avg, self.rng)
        if newborn:
            self.colonists.append(newborn)
            new_active = self._active_ids()
            self.social.edges[newborn.id] = {}
            for other_id in new_active:
                if other_id != newborn.id:
                    from src.mars100.colony import Relationship
                    self.social.edges[newborn.id][other_id] = Relationship(
                        trust=max(0.0, min(1.0, 0.5 + self.rng.gauss(0, 0.1))),
                        affection=max(0.0, min(1.0, 0.6 + self.rng.gauss(0, 0.1))),
                        respect=max(0.0, min(1.0, 0.4 + self.rng.gauss(0, 0.1))))
                    if other_id in self.social.edges:
                        self.social.edges[other_id][newborn.id] = Relationship(
                            trust=max(0.0, min(1.0, 0.5 + self.rng.gauss(0, 0.1))),
                            affection=max(0.0, min(1.0, 0.6 + self.rng.gauss(0, 0.1))),
                            respect=max(0.0, min(1.0, 0.4 + self.rng.gauss(0, 0.1))))
            births_this_year.append({
                "id": newborn.id, "name": newborn.name,
                "element": newborn.element, "year": self.year,
            })

        meta_events: list[dict] = []
        for colonist in self._active_colonists():
            meta = self._check_meta_awareness(colonist)
            if meta:
                meta_events.append(meta)

        # Convergence tracking
        snapshots = [c.to_dict() for c in self.colonists]
        conv_score = compute_convergence_score(snapshots, STAT_NAMES)
        conv_per_stat = per_stat_convergence(snapshots, STAT_NAMES)
        self.convergence_history.append(conv_score)

        return YearResult(
            year=self.year, events=[e.to_dict() for e in events],
            actions=actions, subsim_log=[s.to_dict() for s in subsim_log],
            governance=gov_proposal.to_dict() if gov_proposal else None,
            resources_before=resources_before,
            resources_after=self.resources.to_dict(),
            resource_delta=resource_delta, deaths=deaths, exiles=exiles,
            births=births_this_year,
            meta_awareness=meta_events,
            meta_insights=year_insights,
            convergence_score=conv_score,
            convergence_per_stat=conv_per_stat,
            social_cohesion=self.social.colony_cohesion(self._active_ids()),
            governance_state=self.governance.to_dict(),
            colonist_snapshots=snapshots,
        )

    def run(self, callback: Any = None) -> SimulationResult:
        """Run the full simulation."""
        years: list[YearResult] = []
        total_deaths = total_exiles = total_births = 0
        total_subsims = gov_changes = meta_count = 0
        for _ in range(self.total_years):
            if not self._active_colonists():
                break
            result = self.tick()
            years.append(result)
            total_deaths += len(result.deaths)
            total_exiles += len(result.exiles)
            total_births += len(result.births)
            total_subsims += len(result.subsim_log)
            if result.governance and result.governance.get("passed"):
                gov_changes += 1
            meta_count += len(result.meta_awareness)
            if callback:
                callback(result)

        conv_summary = convergence_summary([
            {"year": y.year, "score": y.convergence_score} for y in years
        ])
        proposed = should_promote_amendment(self.all_meta_insights)

        return SimulationResult(
            years=years, final_colonists=[c.to_dict() for c in self.colonists],
            final_resources=self.resources.to_dict(),
            final_governance=self.governance.to_dict(),
            total_deaths=total_deaths, total_exiles=total_exiles,
            total_births=total_births,
            total_subsims=total_subsims, governance_changes=gov_changes,
            meta_events=meta_count,
            meta_insights=self.all_meta_insights,
            final_cohesion=self.social.colony_cohesion(self._active_ids()),
            convergence_scores=list(self.convergence_history),
            convergence_summary=conv_summary,
            proposed_amendment=proposed,
        )
