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
from src.mars100.culture import (
    CulturalMemory, YearContext as CultureYearContext,
    evolve_culture, compute_cultural_pressure,
)
from src.mars100.earth import (
    EarthState, tick_earth, compute_maintenance_modifier,
    check_independence_conditions, declare_independence,
)
from src.mars100.economics import (
    EconomicState, tick_economics, compute_economic_pressure,
    liquidate_estate, endow_immigrant as econ_endow_immigrant,
)
from src.mars100.psychology import (
    PsychState, ColonistPsychContext, tick_psychology,
    death_rate_modifier,
)
from src.mars100.behavior import (
    BehaviorProfile, BehaviorTickResult, ContagionDelta,
    compute_action_perturbation, compute_social_contagion,
    update_learned_preferences,
)
from src.mars100.ecology import (
    EcologyState, EcologyYearContext, EcologyTickResult,
    tick_ecology, compute_resource_modifiers as compute_ecology_modifiers,
    compute_nature_stress_reduction,
)
from src.mars100.diplomacy import (
    DiplomacyState, DiplomacyTickResult,
    tick_diplomacy, compute_bloc_pressure, compute_faction_vote_bias,
)
from src.mars100.comm_channels import (
    CommChannelsState, tick_comm_channels, compute_revival_pressure,
)
from src.mars100.rumors import (
    RumorsState, tick_rumors, build_channel_lookup,
)
from src.mars100.colonist import create_immigrant

ACTIONS = ["terraform", "farm", "mediate", "code", "pray",
           "sabotage", "cooperate", "hoard", "explore", "rest", "research"]
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
    meta_awareness: list[dict]
    social_cohesion: float
    governance_state: dict
    colonist_snapshots: list[dict]
    convergence: dict = field(default_factory=dict)
    births: list[dict] = field(default_factory=list)
    infrastructure: dict = field(default_factory=dict)
    culture: dict = field(default_factory=dict)
    earth: dict = field(default_factory=dict)
    earth_events: list[dict] = field(default_factory=list)
    diplomacy: dict = field(default_factory=dict)
    immigrants: list[dict] = field(default_factory=list)
    economics: dict = field(default_factory=dict)
    psychology: dict = field(default_factory=dict)
    behavior: dict = field(default_factory=dict)
    ecology: dict = field(default_factory=dict)
    comm_channels: dict = field(default_factory=dict)
    rumors: dict = field(default_factory=dict)

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
            "culture": self.culture,
            "earth": self.earth,
            "earth_events": self.earth_events,
            "diplomacy": self.diplomacy,
            "immigrants": self.immigrants,
            "economics": self.economics,
            "psychology": self.psychology,
            "behavior": self.behavior,
            "ecology": self.ecology,
            "comm_channels": self.comm_channels,
            "rumors": self.rumors,
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
    final_culture: dict = field(default_factory=dict)
    final_earth: dict = field(default_factory=dict)
    total_immigrants: int = 0
    total_ships: int = 0
    final_economics: dict = field(default_factory=dict)
    final_psychology: dict = field(default_factory=dict)
    total_crises: int = 0
    final_behavior: dict = field(default_factory=dict)
    final_ecology: dict = field(default_factory=dict)
    final_diplomacy: dict = field(default_factory=dict)
    total_factions_formed: int = 0
    total_schisms: int = 0
    final_comm_channels: dict = field(default_factory=dict)
    total_comm_flatlines: int = 0
    total_comm_revival_prompts: int = 0
    total_bridge_prompts: int = 0
    total_bridge_revivals: int = 0

    def to_dict(self) -> dict:
        return {
            "_meta": {"engine": "mars-100", "version": "12.1",
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
                "total_immigrants": self.total_immigrants,
                "total_ships": self.total_ships,
                "promoted_insights": len(self.promoted_insights),
                "total_crises": self.total_crises,
                "total_factions_formed": self.total_factions_formed,
                "total_schisms": self.total_schisms,
                "total_comm_flatlines": self.total_comm_flatlines,
                "total_comm_revival_prompts": self.total_comm_revival_prompts,
                "total_bridge_prompts": self.total_bridge_prompts,
                "total_bridge_revivals": self.total_bridge_revivals,
            },
            "final_colonists": self.final_colonists,
            "final_resources": self.final_resources,
            "final_governance": self.final_governance,
            "promoted_insights": self.promoted_insights,
            "infrastructure": self.infrastructure,
            "final_culture": self.final_culture,
            "final_earth": self.final_earth,
            "final_economics": self.final_economics,
            "final_psychology": self.final_psychology,
            "final_behavior": self.final_behavior,
            "final_ecology": self.final_ecology,
            "final_diplomacy": self.final_diplomacy,
            "final_comm_channels": self.final_comm_channels,
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
        self.culture = CulturalMemory()
        self.culture_rng = random.Random(seed + 7919)
        self.earth = EarthState()
        self.earth_rng = random.Random(seed + 6151)
        self.economics = EconomicState()
        self.econ_rng = random.Random(seed + 8191)
        self.psych_map: dict[str, PsychState] = {}
        self.psych_rng = random.Random(seed + 9049)
        self.behavior_map: dict[str, BehaviorProfile] = {}
        self.ecology = EcologyState()
        self.ecology_rng = random.Random(seed + 11213)
        self.diplo = DiplomacyState()
        self.diplo_rng = random.Random(seed + 12553)
        self.comm_channels = CommChannelsState()
        self.comm_rng = random.Random(seed + 14401)
        self.rumors = RumorsState()
        self.rumors_rng = random.Random(seed + 15619)
        self.pending_ecology_mods: dict[str, float] = {}
        self.next_id = 10
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

        # External pressure: cultural + economic + behavioral (combined clamp)
        cultural_pressure = compute_cultural_pressure(self.culture)
        econ_pressure = compute_economic_pressure(colonist)
        psych = self.psych_map.get(colonist.id)
        profile = self.behavior_map.get(colonist.id, BehaviorProfile())
        behavior_pressure = compute_action_perturbation(
            psych.stress if psych else 0.0,
            psych.morale if psych else 0.5,
            psych.purpose if psych else 0.5,
            profile, ACTIONS,
        ) if psych else {}
        diplo_pressure = compute_bloc_pressure(self.diplo, colonist.id, ACTIONS)
        for act in ACTIONS:
            combined = (cultural_pressure.get(act, 0.0)
                        + econ_pressure.get(act, 0.0)
                        + behavior_pressure.get(act, 0.0)
                        + diplo_pressure.get(act, 0.0))
            if act in weights:
                weights[act] = max(0.01, weights[act] + combined)

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
                births.append({
                    "id": child.id, "name": child.name, "year": self.year,
                    "parents": [parent_a.id, parent_b.id],
                    "element": child.element,
                })
                self.births.extend(births)
        return births

    def _check_death(self, colonist: Colonist) -> str | None:
        rate = BASE_DEATH_RATE
        # Psychology: low morale increases death rate
        psych = self.psych_map.get(colonist.id)
        if psych:
            rate *= death_rate_modifier(psych.morale)
        death_rate_mult = compute_resource_modifiers(self.infra.completed).get("death_rate_mult", 1.0)
        rate *= death_rate_mult
        critical_resources: list[str] = []
        for name in RESOURCE_NAMES:
            val = getattr(self.resources, name)
            if val < 0.1:
                rate += RESOURCE_DEATH_MULTIPLIER * (0.1 - val)
                critical_resources.append(name)
        if colonist.stats.paranoia > 0.8:
            rate += 0.005
        if self.rng.random() < rate:
            resource_causes = {
                "air": "asphyxiation", "food": "starvation",
                "water": "dehydration", "power": "hypothermia",
                "medicine": "untreated illness",
            }
            if critical_resources:
                weights = [(r, 0.1 - getattr(self.resources, r))
                           for r in critical_resources]
                total_w = sum(w for _, w in weights)
                roll = self.rng.random() * total_w
                cumul = 0.0
                for res, w in weights:
                    cumul += w
                    if roll <= cumul:
                        return resource_causes.get(res, "resource deprivation")
                return resource_causes.get(critical_resources[0],
                                           "resource deprivation")
            causes = ["equipment malfunction", "radiation exposure",
                      "medical emergency", "habitat breach"]
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
        # Faction loyalty bias
        score += compute_faction_vote_bias(
            self.diplo, colonist.id, proposal.proposer_id)
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

        actions: dict[str, str] = {}
        subsim_budget = SubSimBudget(year=self.year)
        subsim_log: list[SubSimResult] = []
        for colonist in active:
            action = self._choose_action(colonist, events)
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

        # Ecology: merge lagged ecology modifiers (one-year lag)
        for k, v in self.pending_ecology_mods.items():
            infra_mods[k] = infra_mods.get(k, 1.0) * v

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

        # --- economics: labor income → trade → taxation → inequality ---
        econ_result = tick_economics(
            colonists=self.colonists, actions=actions,
            gov_type=self.governance.gov_type, social=self.social,
            resources=self.resources, economic_state=self.economics,
            year=self.year, rng=self.econ_rng)

        # Economic revolt: force governance re-election next year
        if self.economics.pending_revolt:
            self.economics.pending_revolt = False
            if active:
                proposer = self.econ_rng.choice(active)
                revolt_proposal = generate_proposal(
                    self.year, proposer.id, self.governance, self.econ_rng)
                revolt_proposal.subsim_result = {"revolt": True}
                for c in active:
                    if c.id == revolt_proposal.proposer_id:
                        revolt_proposal.votes_for.append(c.id)
                    elif self._vote_on_proposal(c, revolt_proposal):
                        revolt_proposal.votes_for.append(c.id)
                    else:
                        revolt_proposal.votes_against.append(c.id)
                revolt_proposal.passed = resolve_vote(revolt_proposal, len(active))
                if revolt_proposal.passed:
                    apply_governance(revolt_proposal, self.governance,
                                     active_ids, self.econ_rng)

        # --- psychology tick (dedicated RNG stream) ---
        event_sev = events[0].severity if events else 0.0
        res_avg = self.resources.average()
        earth_contacted = (hasattr(self.earth, 'last_contact_year')
                           and self.earth.last_contact_year == self.year)
        infra_just_completed = bool(
            infra_event and (
                infra_event.get("completed", False)
                if isinstance(infra_event, dict)
                else getattr(infra_event, 'completed', False)))
        psych_contexts: list[ColonistPsychContext] = []
        for c in active:
            n_connections = sum(
                1 for oid in active_ids
                if oid != c.id and self.social.get(c.id, oid).trust > 0.3)
            trusts = [self.social.get(c.id, oid).trust
                      for oid in active_ids if oid != c.id]
            avg_trust = sum(trusts) / max(1, len(trusts))
            participated_gov = (gov_proposal is not None
                                and c.id in (gov_proposal.votes_for
                                             + gov_proposal.votes_against))
            ran_subsim = any(s.colonist_id == c.id for s in subsim_log)
            psych_contexts.append(ColonistPsychContext(
                colonist_id=c.id, action=actions.get(c.id, "rest"),
                event_severity=event_sev, resource_avg=res_avg,
                social_connections=n_connections, avg_trust=avg_trust,
                earth_contact=earth_contacted,
                infra_completed=infra_just_completed,
                gov_participated=participated_gov,
                subsim_ran=ran_subsim,
                resolve=c.stats.resolve, empathy=c.stats.empathy,
                faith=c.stats.faith, paranoia=c.stats.paranoia,
            ))
        psych_result = tick_psychology(
            self.psych_map, psych_contexts, self.year, self.psych_rng)

        # --- behavior: social contagion (frozen snapshot, simultaneous) ---
        frozen_psych = {cid: self.psych_map[cid].to_dict()
                        for cid in active_ids if cid in self.psych_map}
        contagion_deltas: list[ContagionDelta] = []
        for c in active:
            trust_pairs = [
                (oid, self.social.get(c.id, oid).trust)
                for oid in active_ids if oid != c.id
            ]
            cd = compute_social_contagion(c.id, frozen_psych, trust_pairs)
            contagion_deltas.append(cd)
        for cd in contagion_deltas:
            ps = self.psych_map.get(cd.colonist_id)
            if ps is not None:
                ps.stress = max(0.0, min(1.0, ps.stress + cd.stress_delta))
                ps.loneliness = max(0.0, min(1.0,
                    ps.loneliness + cd.loneliness_delta))
                ps.purpose = max(0.0, min(1.0,
                    ps.purpose + cd.purpose_delta))

        # --- behavior: update learned preferences from action outcomes ---
        behavior_result = BehaviorTickResult(
            contagion=[cd.to_dict() for cd in contagion_deltas])
        for cid, action in actions.items():
            profile = self.behavior_map.get(cid)
            if profile is None:
                profile = BehaviorProfile()
                self.behavior_map[cid] = profile
            updated = update_learned_preferences(
                profile, action, resource_delta)
            behavior_result.learned_updates[cid] = updated
            ps = self.psych_map.get(cid)
            if ps:
                perturb = compute_action_perturbation(
                    ps.stress, ps.morale, ps.purpose, profile, ACTIONS)
                behavior_result.perturbations[cid] = perturb

        # --- ecology: tick biosphere (dedicated RNG stream) ---
        eco_terraformers = sum(1 for a in actions.values() if a == "terraform")
        eco_farmers = sum(1 for a in actions.values() if a == "farm")
        eco_researchers = sum(1 for a in actions.values() if a == "research")
        eco_ctx = EcologyYearContext(
            year=self.year, terraform_count=eco_terraformers,
            farm_count=eco_farmers, research_count=eco_researchers,
            population=len(active),
            infrastructure_completed=self.infra.completed)
        ecology_result = tick_ecology(self.ecology, eco_ctx, self.ecology_rng)

        # Apply ecology resource bonuses (from LAST year's state)
        for res_name, bonus in ecology_result.resource_bonuses.items():
            if res_name in RESOURCE_NAMES and bonus > 0:
                current = getattr(self.resources, res_name, 0.0)
                setattr(self.resources, res_name, current + bonus)

        # Stage ecology modifiers for NEXT year (one-year lag)
        self.pending_ecology_mods = compute_ecology_modifiers(self.ecology)

        # Ecology -> psychology: nature exposure reduces stress
        nature_reduction = ecology_result.nature_stress_reduction
        if nature_reduction > 0:
            for ps in self.psych_map.values():
                ps.stress = max(0.0, ps.stress - nature_reduction)

        year_births = self._check_births()

        deaths: list[dict] = []
        exiles: list[dict] = []
        estates: list[dict] = []
        for colonist in list(active):
            cause = self._check_death(colonist)
            if cause:
                colonist.die(self.year, cause)
                estate = liquidate_estate(colonist, self.resources)
                deaths.append({"id": colonist.id, "name": colonist.name,
                                "cause": cause, "year": self.year})
                if estate:
                    estates.append({"id": colonist.id, "type": "death", "estate": estate})
                continue
            if self._check_exile(colonist):
                colonist.exile(self.year)
                estate = liquidate_estate(colonist, self.resources)
                exiles.append({"id": colonist.id, "name": colonist.name, "year": self.year})
                if estate:
                    estates.append({"id": colonist.id, "type": "exile", "estate": estate})

        econ_result.estates_liquidated = estates

        # --- cultural memory evolution ---
        death_records = [
            {"colonist_id": d.get("id", "unknown"),
             "dominant_trait": d.get("dominant_trait", "resolve")}
            for d in deaths]
        exile_records = [
            {"colonist_id": e.get("id", "unknown")}
            for e in exiles]
        gov_records = []
        if gov_proposal is not None:
            gov_records.append({"passed": gov_proposal.passed,
                                "title": gov_proposal.gov_type})
        action_counts = {}
        for act_name in actions.values():
            action_counts[act_name] = action_counts.get(act_name, 0) + 1
        culture_ctx = CultureYearContext(
            year=self.year,
            event_type=events[0].name if events else "none",
            event_severity=events[0].severity if events else 0.0,
            deaths=death_records, exiles=exile_records,
            governance_proposals=gov_records,
            subsim_count=len(subsim_log),
            action_counts=action_counts,
            resources=self.resources.to_dict(),
            colonists=[c.to_dict() for c in self._active_colonists()])
        evolve_culture(self.culture, culture_ctx, self.culture_rng)

        # --- Earth Protocol ---
        earth_result = tick_earth(
            self.earth, self.year,
            death_count=len(deaths),
            resource_avg=self.resources.average(),
            population=len(self._active_colonists()),
            rng=self.earth_rng)

        # Apply maintenance modifier to infrastructure operating costs
        maint_mod = earth_result.maintenance_modifier
        if maint_mod != 1.0:
            for res_name, cost in op_costs.items():
                if res_name in RESOURCE_NAMES:
                    current = getattr(self.resources, res_name)
                    adjustment = cost * (maint_mod - 1.0)
                    setattr(self.resources, res_name,
                            max(0.0, current - adjustment))

        # Check independence
        if check_independence_conditions(
                self.year, len(self._active_colonists()),
                self.resources.average(), self.earth):
            active_now = self._active_colonists()
            votes_for = sum(1 for c in active_now
                            if c.stats.resolve > 0.5 or c.stats.paranoia > 0.6)
            if votes_for > len(active_now) / 2:
                declare_independence(self.earth, self.year)
                earth_result.independence_declared = True

        # Process immigrants from Earth
        year_immigrants: list[dict] = []
        if not self.earth.independent:
            resources_avg = self.resources.average()
            if (self.earth.funding > 0.3
                    and resources_avg > 0.4
                    and getattr(self.resources, "air", 0) > 0.3
                    and len(self._active_colonists()) < 40
                    and self.earth_rng.random() < 0.15 * self.earth.funding):
                imm_id = f"imm-{self.next_id}"
                self.next_id += 1
                immigrant = create_immigrant(imm_id, self.year, self.earth_rng)
                econ_endow_immigrant(immigrant)
                self.colonists.append(immigrant)
                active_ids_now = [c.id for c in self._active_colonists()]
                self.social.add_colonist(immigrant.id, active_ids_now,
                                         self.earth_rng)
                year_immigrants.append({
                    "id": immigrant.id, "name": immigrant.name,
                    "year": self.year, "archetype": immigrant.archetype,
                    "element": immigrant.element,
                })

        # --- diplomacy: faction formation, alliances, tensions ---
        # Ticked LATE so faction membership reflects end-of-year population
        diplo_result = tick_diplomacy(
            state=self.diplo,
            active_colonists=[c.to_dict() for c in self._active_colonists()],
            social_get=self.social.get,
            actions=actions,
            year=self.year,
            rng=self.diplo_rng)

        # --- comm channels: per-pair communication health, flatline detection ---
        # Ticked AFTER diplomacy so faction membership feeds the contact signal
        active_ids_for_comm = [c.id for c in self._active_colonists()]
        faction_membership: dict[str, str] = {}
        for fac_id, faction in self.diplo.factions.items():
            for member_id in getattr(faction, "members", []):
                faction_membership[member_id] = fac_id
        comm_result = tick_comm_channels(
            state=self.comm_channels,
            active_ids=active_ids_for_comm,
            actions=actions,
            social_get=self.social.get,
            faction_membership=faction_membership,
            year=self.year,
            rng=self.comm_rng)

        # --- rumors: information flow over the comm graph (engine v13.0) ---
        # Vital channels carry; flatlined ones block. Rumors mutate over time.
        rumor_lookup = build_channel_lookup(self.comm_channels)
        rumor_result = tick_rumors(
            state=self.rumors,
            year=self.year,
            active_colonist_ids=active_ids_for_comm,
            channel_lookup=rumor_lookup,
            rng=self.rumors_rng,
        )

        meta_events: list[dict] = []
        for colonist in self._active_colonists():
            meta = self._check_meta_awareness(colonist)
            if meta:
                meta_events.append(meta)

        # Final clamp — multiple subsystems may have nudged resources
        self.resources.clamp()

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
            culture=self.culture.summary(),
            earth=self.earth.to_dict(),
            earth_events=[earth_result.to_dict()],
            diplomacy=diplo_result.to_dict(),
            immigrants=year_immigrants,
            economics=econ_result.to_dict(),
            psychology=psych_result.to_dict(),
            behavior=behavior_result.to_dict(),
            ecology=ecology_result.to_dict(),
            comm_channels=comm_result.to_dict(),
            rumors=rumor_result.to_dict(),
        )

    def run(self, callback: Any = None) -> SimulationResult:
        """Run the full simulation."""
        years: list[YearResult] = []
        total_deaths = total_exiles = total_subsims = gov_changes = meta_count = total_births = 0
        total_immigrants = 0
        total_factions_formed = total_schisms = 0
        total_comm_flatlines = total_comm_revival_prompts = 0
        for _ in range(self.total_years):
            if not self._active_colonists():
                break
            result = self.tick()
            years.append(result)
            total_deaths += len(result.deaths)
            total_exiles += len(result.exiles)
            total_subsims += len(result.subsim_log)
            total_births += len(result.births)
            total_immigrants += len(result.immigrants)
            if result.governance and result.governance.get("passed"):
                gov_changes += 1
            meta_count += len(result.meta_awareness)
            total_factions_formed += len(result.diplomacy.get("factions_formed", []))
            total_schisms += len(result.diplomacy.get("schisms", []))
            total_comm_flatlines += len(result.comm_channels.get("flatlined", []))
            total_comm_revival_prompts += len(
                result.comm_channels.get("revival_prompts", []))
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
            total_immigrants=total_immigrants,
            total_ships=self.earth.ships_launched,
            infrastructure=self.infra.to_dict(),
            final_culture=self.culture.to_dict(),
            final_earth=self.earth.to_dict(),
            final_economics=self.economics.to_dict(),
            final_psychology={cid: p.to_dict() for cid, p in self.psych_map.items()},
            final_behavior={cid: b.to_dict() for cid, b in self.behavior_map.items()},
            final_ecology=self.ecology.to_dict(),
            final_diplomacy=self.diplo.to_dict(),
            total_factions_formed=total_factions_formed,
            total_schisms=total_schisms,
            final_comm_channels=self.comm_channels.to_dict(),
            total_comm_flatlines=total_comm_flatlines,
            total_comm_revival_prompts=total_comm_revival_prompts,
            total_bridge_prompts=self.comm_channels.total_bridge_prompts,
            total_bridge_revivals=self.comm_channels.total_bridge_revivals,
            total_crises=sum(
                len(y.psychology.get("crises", []))
                for y in years if isinstance(y.psychology, dict)),
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