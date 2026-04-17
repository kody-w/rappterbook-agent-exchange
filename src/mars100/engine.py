"""
Mars-100 simulation engine.

100 frames (Martian years). Pure computation — returns data, no I/O.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.mars100.colonist import (
    Colonist, STAT_NAMES, SKILL_NAMES, create_founding_ten,
    create_child, COLONIST_NAMES,
)
from src.mars100.colony import (
    Resources, SocialGraph, tick_resources, RESOURCE_NAMES,
    compute_value_convergence,
)
from src.mars100.events import Event, generate_events
from src.mars100.governance import (
    GovernanceProposal, GovernanceState, apply_governance,
    generate_proposal, resolve_vote, should_propose,
)
from src.mars100.subsim import SubSimBudget, SubSimResult, spawn_subsim
from src.mars100.lispy_vm import LispyError
from src.mars100.infrastructure import (
    InfrastructureState, choose_project, start_project,
    tick_infrastructure, compute_resource_modifiers, compute_operating_costs,
)
from src.mars100.diplomacy import DiplomacyState, tick_diplomacy
from src.mars100.memetics import MemePool, CRISIS_SEVERITY_THRESHOLD

ACTIONS = ["terraform", "farm", "mediate", "code", "pray",
           "sabotage", "cooperate", "hoard", "explore", "rest", "research"]
META_AWARENESS_BASE = 0.001
BASE_DEATH_RATE = 0.005
RESOURCE_DEATH_MULTIPLIER = 3.0


def _colonist_stats(colonists: list[Colonist], cid: str) -> Any:
    """Return stats for a colonist by id, or None if not found/inactive."""
    for c in colonists:
        if c.id == cid and c.is_active():
            return c.stats
    return None


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
    meta_awareness: list[dict]
    social_cohesion: float
    governance_state: dict
    colonist_snapshots: list[dict]
    convergence: dict = field(default_factory=dict)
    births: list[dict] = field(default_factory=list)
    infrastructure: dict = field(default_factory=dict)
    diplomacy: dict = field(default_factory=dict)
    memetics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "year": self.year, "events": self.events, "actions": self.actions,
            "subsim_log": self.subsim_log, "governance": self.governance,
            "resources_before": self.resources_before,
            "resources_after": self.resources_after,
            "resource_delta": self.resource_delta,
            "deaths": self.deaths, "exiles": self.exiles,
            "meta_awareness": self.meta_awareness,
            "social_cohesion": self.social_cohesion,
            "governance_state": self.governance_state,
            "colonist_snapshots": self.colonist_snapshots,
            "convergence": self.convergence,
            "births": self.births,
            "infrastructure": self.infrastructure,
            "diplomacy": self.diplomacy,
            "memetics": self.memetics,
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
    total_subsims: int
    governance_changes: int
    meta_events: int
    final_cohesion: float
    convergence_trend: str = "stable"
    promoted_insights: list[dict] = field(default_factory=list)
    total_births: int = 0
    infrastructure: dict = field(default_factory=dict)
    diplomacy: dict = field(default_factory=dict)
    memetics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "_meta": {"engine": "mars-100", "version": "5.0",
                      "total_years": len(self.years),
                      "generated": datetime.now(timezone.utc).isoformat()},
            "summary": {
                "total_deaths": self.total_deaths, "total_exiles": self.total_exiles,
                "total_subsims": self.total_subsims,
                "governance_changes": self.governance_changes,
                "meta_awareness_events": self.meta_events,
                "final_cohesion": self.final_cohesion,
                "convergence_trend": self.convergence_trend,
                "total_births": self.total_births,
                "promoted_insights": len(self.promoted_insights),
            },
            "final_colonists": self.final_colonists,
            "final_resources": self.final_resources,
            "final_governance": self.final_governance,
            "promoted_insights": self.promoted_insights,
            "infrastructure": self.infrastructure,
            "diplomacy": self.diplomacy,
            "memetics": self.memetics,
            "years": [y.to_dict() for y in self.years],
        }


class Mars100Engine:
    """The Mars-100 recursive colony simulation engine."""

    def __init__(self, seed: int = 42, total_years: int = 100) -> None:
        self.seed = seed
        self.total_years = total_years
        self.rng = random.Random(seed)
        self.colonists = create_founding_ten(seed)
        self.resources = Resources()
        self.social = SocialGraph()
        self.governance = GovernanceState()
        self.year = 0
        self.insight_queue: list[dict] = []
        self.promoted_insights: list[dict] = []
        self.births: list[dict] = []
        self.infra = InfrastructureState()
        self.diplomacy_state = DiplomacyState()
        self.memes = MemePool()
        self.next_id = 10
        active_ids = [c.id for c in self.colonists if c.is_active()]
        self.social.initialize(active_ids, self.rng)

    def _active_colonists(self) -> list[Colonist]:
        return [c for c in self.colonists if c.is_active()]

    def _active_ids(self) -> list[str]:
        return [c.id for c in self._active_colonists()]

    def _choose_action(self, colonist: Colonist, events: list[Event],
                       modifiers: dict[str, float] | None = None) -> str:
        """Choose an action for a colonist based on personality and events.

        If *modifiers* is provided (from the diplomacy engine), each
        action weight is multiplied by its modifier before sampling.
        """
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
            elif action == "research":
                w += colonist.skills.coding * 0.6 + colonist.stats.improvisation * 0.3
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

        # Apply diplomacy modifiers
        if modifiers:
            for act, mult in modifiers.items():
                if act in weights:
                    weights[act] = max(0.01, weights[act] * mult)

        # Apply memetic bias to action weights
        meme_deltas = self.memes.action_weight_deltas(colonist.id)
        for action, delta in meme_deltas.items():
            if action in weights:
                weights[action] = max(0.01, weights[action] + delta)

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
            f"(let ((surplus (- food (* {active_count} 0.06)))) (if (> surplus 0) (+ resolve 0.1) (- resolve 0.2)))",
            f"(let ((risk (* paranoia {max_severity:.2f}))) (if (> risk 0.5) (- resolve 0.1) (+ improvisation 0.05)))",
            "(let ((trust-score (* empathy resolve))) (if (> trust-score 0.4) (+ trust-score faith) (- trust-score paranoia)))",
            "(let ((gov-value (+ (* empathy 0.4) (* resolve 0.3) (* faith 0.3)))) (if (> gov-value 0.5) 1 0))",
            "(let ((survival (+ (* food 0.3) (* water 0.3) (* power 0.2) (* air 0.2)))) (if (> survival 0.6) (+ survival 0.1) (- survival 0.2)))",
        ]
        return self.rng.choice(templates)

    def _check_births(self) -> list[dict]:
        """Check for births among high-affinity colonist pairs.

        Births require: year >= 10, resources average > 0.4, pair
        trust+affection > 1.4, and a random roll.  At most 1 birth per year.
        """
        if self.year < 10:
            return []
        if self.resources.average() < 0.4:
            return []
        active = self._active_colonists()
        if len(active) < 2:
            return []
        pairs: list[tuple[Colonist, Colonist, float]] = []
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                rel_ab = self.social.get(a.id, b.id)
                rel_ba = self.social.get(b.id, a.id)
                score = (rel_ab.trust + rel_ab.affection + rel_ba.trust + rel_ba.affection) / 2
                if score > 1.4:
                    pairs.append((a, b, score))
        if not pairs:
            return []
        pairs.sort(key=lambda x: x[2], reverse=True)
        birth_prob = 0.15
        births: list[dict] = []
        for parent_a, parent_b, _ in pairs[:1]:
            if self.rng.random() < birth_prob:
                child_id = f"child-{self.next_id}"
                self.next_id += 1
                child = create_child(parent_a, parent_b, child_id, self.year, self.rng)
                self.colonists.append(child)
                active_ids = [c.id for c in self._active_colonists()]
                self.social.add_colonist(child.id, active_ids, self.rng)
                self.memes.inherit_memes(child.id, parent_a.id, parent_b.id,
                                         self.year, self.rng)
                births.append({
                    "id": child.id, "name": child.name, "year": self.year,
                    "parents": [parent_a.id, parent_b.id],
                    "element": child.element,
                })
                self.births.extend(births)
        return births

    def _check_death(self, colonist: Colonist) -> str | None:
        rate = BASE_DEATH_RATE
        death_rate_mult = compute_resource_modifiers(self.infra.completed).get("death_rate_mult", 1.0)
        rate *= death_rate_mult
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

    def _extract_insight(self, subsim_log: list[dict]) -> None:
        """Extract insights from deep sub-simulations (depth >= 2)."""
        for entry in subsim_log:
            depth = entry.get("depth", 1)
            if depth < 2:
                continue
            result = entry.get("result")
            if result is None:
                continue
            insight = {
                "year": self.year,
                "depth": depth,
                "colonist_id": entry.get("colonist_id", "unknown"),
                "result": str(result)[:200],
                "expression": entry.get("expression", "")[:200],
            }
            self.insight_queue.append(insight)

    def _maybe_promote_insight(self) -> None:
        """Promote recurring insight themes to proposed amendments."""
        if len(self.insight_queue) < 3:
            return
        themes: dict[str, list[dict]] = {}
        for ins in self.insight_queue:
            key = ins["result"][:50]
            themes.setdefault(key, []).append(ins)
        for theme_key, instances in themes.items():
            if len(instances) >= 3:
                amendment = self._draft_amendment(theme_key, instances)
                self.promoted_insights.append(amendment)
                # Create a memetic echo of the promoted insight
                for ins in instances[:1]:
                    self.memes.create_subsim_meme(
                        self.year, ins["colonist_id"], theme_key[:80], self.rng,
                    )
                for ins in instances:
                    self.insight_queue.remove(ins)
                return

    def _draft_amendment(self, theme: str, instances: list[dict]) -> dict:
        """Draft a proposed constitutional amendment from a recurring insight."""
        years = [i["year"] for i in instances]
        depths = [i["depth"] for i in instances]
        return {
            "theme": theme,
            "first_seen_year": min(years),
            "occurrences": len(instances),
            "max_depth": max(depths),
            "colonists_involved": list({i["colonist_id"] for i in instances}),
            "proposed_text": f"Amendment from Mars-100 sub-sim insight: {theme}",
            "status": "proposed",
        }

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

        # Diplomacy: detect factions, manage pacts, compute action modifiers
        diplo_result = tick_diplomacy(
            self.diplomacy_state, active, self.resources, self.year, self.rng,
        )
        diplo_mods = diplo_result.action_modifiers

        actions: dict[str, str] = {}
        subsim_budget = SubSimBudget(year=self.year)
        subsim_log: list[SubSimResult] = []
        for colonist in active:
            action = self._choose_action(colonist, events,
                                         modifiers=diplo_mods.get(colonist.id))
            actions[colonist.id] = action
            colonist.evolve_skills(action, self.rng)
            self._maybe_spawn_subsim(colonist, events, subsim_budget, subsim_log)

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
                self.memes.create_governance_meme(
                    self.year, gov_proposal.proposer_id,
                    gov_proposal.gov_type, self.rng,
                )

        skill_bonuses = self._compute_skill_bonuses(actions)
        event_effects: dict[str, float] = {}
        for ev in events:
            for k, v in ev.effects.items():
                if k in RESOURCE_NAMES:
                    event_effects[k] = event_effects.get(k, 0.0) + v

        # Infrastructure: compute resource modifiers from completed techs
        infra_mods = compute_resource_modifiers(self.infra.completed)
        event_damage_mult = infra_mods.pop("event_damage_mult", 1.0)
        event_effects = {k: v * event_damage_mult if v < 0 else v
                         for k, v in event_effects.items()}

        resource_delta = tick_resources(self.resources, len(active),
                                        skill_bonuses, event_effects,
                                        infra_modifiers=infra_mods)

        # Infrastructure: deduct operating costs
        op_costs = compute_operating_costs(self.infra.completed)
        for res_name, cost in op_costs.items():
            if res_name in RESOURCE_NAMES:
                current = getattr(self.resources, res_name)
                setattr(self.resources, res_name, max(0.0, current - cost))

        # Infrastructure: choose + start project if idle
        resources_snapshot = self.resources.to_dict()
        if self.infra.project is None:
            tech = choose_project(self.infra, resources_snapshot,
                                  active, self.rng)
            if tech:
                start_project(self.infra, tech, self.resources, self.year)

        # Infrastructure: tick active project
        researchers = sum(1 for cid, a in actions.items() if a == "research")
        research_skills = [c.skills.coding for c in active
                           if actions.get(c.id) == "research"]
        avg_skill = sum(research_skills) / max(1, len(research_skills))
        infra_event = tick_infrastructure(self.infra, researchers, avg_skill, self.year)

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

        # Memetics: create crisis memes and propagate
        for ev in events:
            if ev.severity >= CRISIS_SEVERITY_THRESHOLD:
                survivors = self._active_ids()
                self.memes.create_crisis_meme(self.year, survivors, ev.name, self.rng)
        self.memes.propagate(
            self.year, self._active_ids(),
            lambda a, b: self.social.get(a, b),
            lambda cid: _colonist_stats(self.colonists, cid),
            self.rng,
        )

        year_births = self._check_births()

        deaths: list[dict] = []
        exiles: list[dict] = []
        for colonist in list(active):
            cause = self._check_death(colonist)
            if cause:
                colonist.die(self.year, cause)
                self.memes.deactivate_carrier(colonist.id)
                deaths.append({"id": colonist.id, "name": colonist.name,
                                "cause": cause, "year": self.year})
                continue
            if self._check_exile(colonist):
                colonist.exile(self.year)
                self.memes.deactivate_carrier(colonist.id)
                exiles.append({"id": colonist.id, "name": colonist.name, "year": self.year})

        meta_events: list[dict] = []
        for colonist in self._active_colonists():
            meta = self._check_meta_awareness(colonist)
            if meta:
                meta_events.append(meta)

        convergence = compute_value_convergence(self._active_colonists())
        self._extract_insight([s.to_dict() for s in subsim_log])
        self._maybe_promote_insight()

        return YearResult(
            year=self.year, events=[e.to_dict() for e in events],
            actions=actions, subsim_log=[s.to_dict() for s in subsim_log],
            governance=gov_proposal.to_dict() if gov_proposal else None,
            resources_before=resources_before,
            resources_after=self.resources.to_dict(),
            resource_delta=resource_delta, deaths=deaths, exiles=exiles,
            meta_awareness=meta_events,
            social_cohesion=self.social.colony_cohesion(self._active_ids()),
            governance_state=self.governance.to_dict(),
            colonist_snapshots=[c.to_dict() for c in self.colonists],
            convergence=convergence,
            births=year_births,
            infrastructure=self.infra.to_dict(),
            diplomacy=self.diplomacy_state.summary(self.year),
            memetics=self.memes.summary(self.year),
        )

    def run(self, callback: Any = None) -> SimulationResult:
        """Run the full simulation."""
        years: list[YearResult] = []
        total_deaths = total_exiles = total_subsims = gov_changes = meta_count = total_births = 0
        for _ in range(self.total_years):
            if not self._active_colonists():
                break
            result = self.tick()
            years.append(result)
            total_deaths += len(result.deaths)
            total_exiles += len(result.exiles)
            total_subsims += len(result.subsim_log)
            total_births += len(result.births)
            if result.governance and result.governance.get("passed"):
                gov_changes += 1
            meta_count += len(result.meta_awareness)
            if callback:
                callback(result)
        return SimulationResult(
            years=years, final_colonists=[c.to_dict() for c in self.colonists],
            final_resources=self.resources.to_dict(),
            final_governance=self.governance.to_dict(),
            total_deaths=total_deaths, total_exiles=total_exiles,
            total_subsims=total_subsims, governance_changes=gov_changes,
            meta_events=meta_count,
            final_cohesion=self.social.colony_cohesion(self._active_ids()),
            convergence_trend=self._compute_convergence_trend(years),
            promoted_insights=self.promoted_insights,
            total_births=total_births,
            infrastructure=self.infra.to_dict(),
            diplomacy=self.diplomacy_state.to_dict(),
            memetics=self.memes.to_dict(),
        )

    def _compute_convergence_trend(self, years: list[YearResult]) -> str:
        """Classify overall convergence trend from yearly data."""
        scores = [y.convergence.get("convergence_score", 0.5)
                  for y in years if y.convergence]
        if len(scores) < 10:
            return "insufficient_data"
        early = sum(scores[:10]) / 10
        late = sum(scores[-10:]) / 10
        diff = late - early
        if diff > 0.1:
            return "converging"
        elif diff < -0.1:
            return "diverging"
        return "stable"
