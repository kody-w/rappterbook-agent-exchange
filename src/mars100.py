"""
mars100.py — Mars-100 recursive colony simulation kernel.

10 colonists. 100 Mars years. Sub-simulations up to depth 3.
The frame loop is fractal: output of year N = input to year N+1.

Architecture:
  Each year follows: observe → propose intents → resolve conflicts → commit state.
  Colonists express decisions as LisPy s-expressions.
  Sub-sims are sandboxed LisPy evaluations with capped budgets.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.colonist import (
    Colonist, create_colony, STATS, SKILLS, MEMORY_CAP,
)
from src.governance import (
    Proposal, Constitution, GovernanceState,
    colonist_votes, compute_fitness, _gini,
)
from src.lispy import (
    Evaluator, Env, standard_env, to_sexpr, parse, run,
    LispyError, StepLimitError, DepthLimitError,
)


# ---------------------------------------------------------------------------
# Environmental events
# ---------------------------------------------------------------------------

EVENTS = [
    {"id": "dust_storm",      "weight": 15, "food": -0.05, "power": -0.10, "morale": -0.05, "description": "A planet-wide dust storm darkens the sky for months"},
    {"id": "resource_strike", "weight": 10, "food": 0.0, "water": 0.15, "morale": 0.05, "description": "Drilling crew hits a subsurface ice deposit"},
    {"id": "equipment_fail",  "weight": 12, "power": -0.08, "hab": -0.05, "morale": -0.08, "description": "Critical life support component fails"},
    {"id": "earth_contact",   "weight": 8,  "morale": 0.10, "description": "Transmission received from Earth — first in years"},
    {"id": "solar_flare",     "weight": 10, "power": -0.05, "health": -0.10, "morale": -0.05, "description": "Intense solar particle event forces shelter-in-place"},
    {"id": "aquifer_found",   "weight": 5,  "water": 0.20, "morale": 0.08, "description": "Deep radar reveals a massive subsurface aquifer"},
    {"id": "disease_outbreak","weight": 10, "health": -0.15, "morale": -0.10, "description": "Unknown pathogen spreads through the hab modules"},
    {"id": "meteor_impact",   "weight": 3,  "hab": -0.15, "morale": -0.12, "description": "Bolide strikes near the colony — crater visible from airlock"},
    {"id": "greenhouse_bloom","weight": 8,  "food": 0.12, "morale": 0.06, "description": "First Mars-native crop cycle exceeds all projections"},
    {"id": "comms_blackout",  "weight": 7,  "morale": -0.06, "description": "Solar conjunction blocks all Earth communication for weeks"},
    {"id": "alien_signal",    "weight": 2,  "morale": 0.0, "paranoia_all": 0.05, "description": "Repeating signal detected from Phobos — origin unknown"},
    {"id": "calm_year",       "weight": 20, "morale": 0.02, "description": "An uneventful year — the colony settles into routine"},
]


@dataclass
class Resources:
    """Colony shared resources — all normalized 0.0 to 2.0."""
    food: float = 1.0
    water: float = 1.0
    power: float = 1.0
    oxygen: float = 1.0
    hab_integrity: float = 1.0
    morale: float = 0.7

    def apply_event(self, event: dict) -> None:
        """Apply environmental event effects."""
        self.food = max(0.0, min(2.0, self.food + event.get("food", 0.0)))
        self.water = max(0.0, min(2.0, self.water + event.get("water", 0.0)))
        self.power = max(0.0, min(2.0, self.power + event.get("power", 0.0)))
        self.hab_integrity = max(0.0, min(2.0, self.hab_integrity + event.get("hab", 0.0)))
        self.morale = max(0.0, min(1.0, self.morale + event.get("morale", 0.0)))

    def consumption_tick(self, alive_count: int) -> None:
        """Per-year consumption by colonists."""
        drain = alive_count * 0.04  # moderate consumption
        self.food = max(0.0, self.food - drain)
        self.water = max(0.0, self.water - drain * 0.6)
        self.oxygen = max(0.0, self.oxygen - drain * 0.3)
        # Passive regeneration from life support systems
        self.food = min(2.0, self.food + 0.15)  # greenhouse baseline
        self.water = min(2.0, self.water + 0.12)  # water recycling
        self.power = min(2.0, self.power + 0.08)
        self.oxygen = min(2.0, self.oxygen + 0.10)

    def crisis_level(self) -> float:
        """0 = no crisis, 1 = total collapse. Min of critical resources."""
        critical = min(self.food, self.water, self.power, self.oxygen,
                       self.hab_integrity)
        return max(0.0, 1.0 - critical)

    def to_dict(self) -> dict:
        """JSON-safe serialization."""
        return {
            "food": round(self.food, 4),
            "water": round(self.water, 4),
            "power": round(self.power, 4),
            "oxygen": round(self.oxygen, 4),
            "hab_integrity": round(self.hab_integrity, 4),
            "morale": round(self.morale, 4),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Resources:
        """Deserialize with clamping."""
        return cls(
            food=max(0.0, min(2.0, data.get("food", 1.0))),
            water=max(0.0, min(2.0, data.get("water", 1.0))),
            power=max(0.0, min(2.0, data.get("power", 1.0))),
            oxygen=max(0.0, min(2.0, data.get("oxygen", 1.0))),
            hab_integrity=max(0.0, min(2.0, data.get("hab_integrity", 1.0))),
            morale=max(0.0, min(1.0, data.get("morale", 0.7))),
        )


# ---------------------------------------------------------------------------
# Intent resolution
# ---------------------------------------------------------------------------

ACTIONS = (
    "gather_food", "gather_water", "repair_hab", "generate_power",
    "terraform", "mediate", "propose", "explore", "pray", "sabotage",
    "run_subsim",
)


@dataclass
class Intent:
    """A colonist's intended action for this year."""
    colonist_id: str
    action: str
    target_id: str | None = None
    proposal: Proposal | None = None
    subsim_expr: str | None = None
    effectiveness: float = 0.0


def decide_action(colonist: Colonist, resources: Resources, year: int,
                   event: dict, governance: GovernanceState,
                   rng: random.Random) -> Intent:
    """Determine what action a colonist takes this year based on stats + situation."""
    crisis = resources.crisis_level()

    # Survival instinct: if resources critical, gather what's lowest
    if crisis > 0.6:
        lowest = min(
            ("gather_food", resources.food),
            ("gather_water", resources.water),
            ("generate_power", resources.power),
            ("repair_hab", resources.hab_integrity),
            key=lambda x: x[1],
        )
        action = lowest[0]
        eff = colonist.effectiveness() * (0.5 + rng.random() * 0.5)
        return Intent(colonist.id, action, effectiveness=eff)

    # Personality-driven action selection
    weights: dict[str, float] = {}
    weights["gather_food"] = (1.0 - resources.food) * 0.8 + colonist.stat("hoarding") * 0.3
    weights["gather_water"] = (1.0 - resources.water) * 0.7
    weights["repair_hab"] = (1.0 - resources.hab_integrity) * 0.9
    weights["generate_power"] = (1.0 - resources.power) * 0.6
    weights["terraform"] = colonist.skill("terraforming") * 0.5 * resources.food
    weights["mediate"] = colonist.stat("empathy") * 0.4 * crisis
    weights["explore"] = colonist.stat("improvisation") * 0.3 * (1.0 - crisis)
    weights["pray"] = colonist.stat("faith") * 0.3 * crisis
    weights["sabotage"] = colonist.skill("sabotage") * colonist.stat("paranoia") * 0.2

    # Governance proposals if enough stability
    if crisis < 0.4 and rng.random() < 0.15:
        weights["propose"] = 0.6

    # Sub-sim if high improvisation + coding skill
    if colonist.skill("coding") > 0.3 and colonist.stat("improvisation") > 0.4:
        if rng.random() < 0.08:
            weights["run_subsim"] = 0.5

    # Weighted random selection
    total = sum(weights.values())
    if total == 0:
        action = "explore"
    else:
        roll = rng.random() * total
        cumulative = 0.0
        action = "explore"
        for act, w in weights.items():
            cumulative += w
            if roll <= cumulative:
                action = act
                break

    eff = colonist.effectiveness() * (0.5 + rng.random() * 0.5)
    return Intent(colonist.id, action, effectiveness=eff)


def resolve_intents(intents: list[Intent], colonists: list[Colonist],
                    resources: Resources, year: int, governance: GovernanceState,
                    rng: random.Random) -> list[dict]:
    """Resolve all intents simultaneously, handling conflicts.

    Returns list of action outcomes.
    """
    outcomes: list[dict] = []
    resource_contributions: dict[str, float] = {
        "food": 0.0, "water": 0.0, "power": 0.0, "hab": 0.0,
    }

    for intent in intents:
        colonist = next((c for c in colonists if c.id == intent.colonist_id), None)
        if not colonist or not colonist.alive:
            continue

        outcome = _apply_intent(intent, colonist, colonists, resources,
                                 resource_contributions, year, governance, rng)
        outcomes.append(outcome)

    # Apply accumulated resource changes
    resources.food = max(0.0, min(2.0, resources.food + resource_contributions["food"]))
    resources.water = max(0.0, min(2.0, resources.water + resource_contributions["water"]))
    resources.power = max(0.0, min(2.0, resources.power + resource_contributions["power"]))
    resources.hab_integrity = max(0.0, min(2.0, resources.hab_integrity + resource_contributions["hab"]))

    return outcomes


def _apply_intent(intent: Intent, colonist: Colonist, colonists: list[Colonist],
                   resources: Resources, contributions: dict[str, float],
                   year: int, governance: GovernanceState,
                   rng: random.Random) -> dict:
    """Apply a single colonist's intent, return outcome dict."""
    eff = intent.effectiveness
    action = intent.action
    result = {"colonist": colonist.id, "action": action, "year": year}

    if action == "gather_food":
        gain = eff * 0.12 * (1.0 + colonist.skill("hydroponics") * 0.5)
        contributions["food"] += gain
        colonist.adjust_skill("hydroponics", 0.01)
        result["gain"] = round(gain, 4)

    elif action == "gather_water":
        gain = eff * 0.10
        contributions["water"] += gain
        result["gain"] = round(gain, 4)

    elif action == "repair_hab":
        gain = eff * 0.08
        contributions["hab"] += gain
        result["gain"] = round(gain, 4)

    elif action == "generate_power":
        gain = eff * 0.09
        contributions["power"] += gain
        result["gain"] = round(gain, 4)

    elif action == "terraform":
        progress = eff * 0.005 * (1.0 + colonist.skill("terraforming") * 0.5)
        colonist.adjust_skill("terraforming", 0.015)
        result["progress"] = round(progress, 6)

    elif action == "mediate":
        # Improve relationships between two lowest-trust pairs
        active = [c for c in colonists if c.alive and c.id != colonist.id]
        if len(active) >= 2:
            pair = rng.sample(active, 2)
            boost = colonist.skill("mediation") * 0.1 * eff
            pair[0].adjust_trust(pair[1].id, boost)
            pair[1].adjust_trust(pair[0].id, boost)
            colonist.adjust_skill("mediation", 0.01)
            result["mediated"] = [pair[0].id, pair[1].id]
            result["boost"] = round(boost, 4)

    elif action == "explore":
        discovery_roll = rng.random()
        if discovery_roll < colonist.stat("improvisation") * 0.2:
            # Found something useful
            resource_type = rng.choice(["food", "water", "power"])
            gain = 0.05 + rng.random() * 0.1
            contributions[resource_type if resource_type != "power" else "power"] += gain
            colonist.add_memory(year, f"discovered {resource_type} cache", 0.6)
            result["discovery"] = resource_type
            result["gain"] = round(gain, 4)

    elif action == "pray":
        morale_boost = colonist.stat("faith") * 0.03 * eff
        resources.morale = min(1.0, resources.morale + morale_boost)
        colonist.adjust_skill("prayer", 0.01)
        colonist.adjust_stat("faith", 0.005)
        result["morale_boost"] = round(morale_boost, 4)

    elif action == "sabotage":
        # Sabotage reduces a random resource, damages trust
        target_resource = rng.choice(["food", "water", "power"])
        damage = colonist.skill("sabotage") * 0.08 * eff
        contributions[target_resource if target_resource != "power" else "power"] -= damage
        # Trust damage if caught (paranoia makes them better at hiding)
        catch_chance = 1.0 - colonist.stat("paranoia") * 0.5
        if rng.random() < catch_chance:
            for c in colonists:
                if c.alive and c.id != colonist.id:
                    c.adjust_trust(colonist.id, -0.15)
            result["caught"] = True
            colonist.add_memory(year, "caught sabotaging", 0.9)
        else:
            result["caught"] = False
        result["damage"] = round(damage, 4)

    elif action == "propose":
        result["proposal"] = "generated"
        # Actual proposal creation happens in the governance phase

    elif action == "run_subsim":
        colonist.subsims_run += 1
        result["subsim"] = "queued"

    colonist.add_memory(year, f"action: {action}", 0.3)
    return result


# ---------------------------------------------------------------------------
# Year tick
# ---------------------------------------------------------------------------

@dataclass
class YearResult:
    """Complete output of one Mars year."""
    year: int
    event: dict
    intents: list[dict]
    outcomes: list[dict]
    proposals: list[dict]
    governance_label: str
    resources: dict
    alive_count: int
    dead_this_year: list[str]
    discoveries: list[str]
    subsim_log: list[dict]
    colonist_diaries: list[dict]
    meta_insight: str | None = None
    simulation_awareness: dict | None = None

    def to_dict(self) -> dict:
        """JSON-safe serialization."""
        d = {
            "year": self.year,
            "event": self.event,
            "outcomes": self.outcomes,
            "proposals": self.proposals,
            "governance_label": self.governance_label,
            "resources": self.resources,
            "alive_count": self.alive_count,
            "dead_this_year": self.dead_this_year,
            "subsim_log": self.subsim_log,
            "colonist_diaries": self.colonist_diaries,
        }
        if self.meta_insight:
            d["meta_insight"] = self.meta_insight
        if self.simulation_awareness:
            d["simulation_awareness"] = self.simulation_awareness
        return d


def pick_event(rng: random.Random) -> dict:
    """Weighted random event selection."""
    total = sum(e["weight"] for e in EVENTS)
    roll = rng.random() * total
    cumulative = 0.0
    for event in EVENTS:
        cumulative += event["weight"]
        if roll <= cumulative:
            return dict(event)
    return dict(EVENTS[-1])


def tick_year(year: int, colonists: list[Colonist], resources: Resources,
              governance: GovernanceState, rng: random.Random) -> YearResult:
    """Run one Mars year: event → intents → resolve → governance → update."""

    active = [c for c in colonists if c.alive]
    if not active:
        return YearResult(
            year=year, event={"id": "extinction", "description": "No colonists remain"},
            intents=[], outcomes=[], proposals=[], governance_label="collapsed",
            resources=resources.to_dict(), alive_count=0, dead_this_year=[],
            discoveries=[], subsim_log=[], colonist_diaries=[],
        )

    # 1. Environmental event
    event = pick_event(rng)
    resources.apply_event(event)

    # Apply paranoia boost from alien signal
    if event.get("paranoia_all"):
        for c in active:
            c.adjust_stat("paranoia", event["paranoia_all"])

    # 2. Resource consumption
    resources.consumption_tick(len(active))

    # 3. Each colonist decides intent
    intents: list[Intent] = []
    for c in active:
        intent = decide_action(c, resources, year, event, governance, rng)
        intents.append(intent)

    # 4. Resolve intents simultaneously
    outcomes = resolve_intents(intents, colonists, resources, year, governance, rng)

    # Initialize meta_insight early — may be set by sub-sims or simulation awareness
    meta_insight = None

    # 5. Governance phase: proposals and votes
    proposals_this_year: list[dict] = []
    proposers = [i for i in intents if i.action == "propose"]
    for intent in proposers[:governance.constitution.max_proposals_per_year]:
        proposer = next((c for c in colonists if c.id == intent.colonist_id), None)
        if not proposer:
            continue
        proposal = _generate_proposal(proposer, colonists, resources, year, governance, rng)
        if proposal:
            _run_vote(proposal, colonists, governance)
            _apply_proposal_outcome(proposal, colonists, resources, governance, year)
            proposals_this_year.append(proposal.to_dict())
            governance.proposals_history.append(proposal.to_dict())

    # 6. Sub-simulations
    subsim_log: list[dict] = []
    subsim_runners = [i for i in intents if i.action == "run_subsim"]
    for intent in subsim_runners[:2]:  # max 2 sub-sims per year
        colonist = next((c for c in colonists if c.id == intent.colonist_id), None)
        if colonist:
            subsim_result = _run_colonist_subsim(colonist, colonists, resources,
                                                   year, governance, rng)
            subsim_log.append(subsim_result)
            # Extract meta-insight from deep sub-sims
            if subsim_result.get("meta_insight") and meta_insight is None:
                meta_insight = subsim_result["meta_insight"]

    # Also run sub-sim for amendment proposals (evidence-gathering)
    for prop_dict in proposals_this_year:
        if prop_dict.get("kind") == "amendment":
            proposer = next((c for c in colonists if c.id == prop_dict.get("proposer_id") and c.alive), None)
            if proposer and len(subsim_log) < 4:
                subsim_result = _run_colonist_subsim(proposer, colonists, resources,
                                                       year, governance, rng)
                subsim_log.append(subsim_result)

    # 6b. Births — Mars-born colonists
    child = maybe_birth(year, colonists, resources, rng)
    if child:
        colonists.append(child)

    # 7. Mortality check
    dead_this_year: list[str] = []
    for c in active:
        if _check_death(c, resources, year, rng):
            dead_this_year.append(c.id)

    # 8. Relationship drift
    _drift_relationships(active, event, resources, rng)

    # 9. Leadership scoring
    _update_leadership(colonists, governance, year, rng)

    # 10. Power snapshot and governance inference
    governance.record_power_snapshot(year, colonists)
    gov_label = governance.infer_governance_type(year)
    governance.year_labels.append({"year": year, "label": gov_label})

    # 11. Meta-insight check (simulation awareness)
    sim_awareness = None
    for c in active:
        if c.alive and c.discovery_potential(year) > 0.75:
            if rng.random() < (c.discovery_potential(year) - 0.7) * 0.5:
                sim_awareness = {
                    "colonist": c.id,
                    "year": year,
                    "paranoia": round(c.stat("paranoia"), 3),
                    "insight": f"{c.name} questions the nature of their reality",
                }
                c.add_memory(year, "questioned if reality is a simulation", 1.0)
                c.adjust_stat("paranoia", 0.05)
                if year > 60 and c.subsims_run > 3 and meta_insight is None:
                    meta_insight = (
                        f"Year {year}: {c.name} proposes that recursive "
                        f"self-modeling reveals the colony itself may be a "
                        f"sub-simulation — governance should account for the "
                        f"possibility that their decisions are being observed"
                    )
                break

    # 12. Generate diary entries
    diaries = _generate_diaries(active, event, outcomes, year, governance, rng)

    # 13. Morale adjustment from governance and events
    if gov_label in ("democracy", "republic"):
        resources.morale = min(1.0, resources.morale + 0.01)
    elif gov_label in ("tyranny", "autocracy"):
        resources.morale = max(0.0, resources.morale - 0.02)

    alive_count = sum(1 for c in colonists if c.alive)

    return YearResult(
        year=year,
        event={"id": event["id"], "description": event["description"]},
        intents=[{"colonist": i.colonist_id, "action": i.action} for i in intents],
        outcomes=outcomes,
        proposals=proposals_this_year,
        governance_label=gov_label,
        resources=resources.to_dict(),
        alive_count=alive_count,
        dead_this_year=dead_this_year,
        discoveries=[o.get("discovery", "") for o in outcomes if o.get("discovery")],
        subsim_log=subsim_log,
        colonist_diaries=diaries,
        meta_insight=meta_insight,
        simulation_awareness=sim_awareness,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_proposal(proposer: Colonist, colonists: list[Colonist],
                        resources: Resources, year: int,
                        governance: GovernanceState,
                        rng: random.Random) -> Proposal | None:
    """Generate a contextually appropriate governance proposal."""
    kind_weights = {
        "leader": 0.3 if governance.leader_id is None else 0.1,
        "resource": 0.3 * (1.0 - resources.food),
        "policy": 0.2,
        "exile": 0.1 * proposer.stat("paranoia"),
        "amendment": 0.1 * max(0, year - 20) / 80,
    }

    total = sum(kind_weights.values())
    roll = rng.random() * total
    cumulative = 0.0
    kind = "policy"
    for k, w in kind_weights.items():
        cumulative += w
        if roll <= cumulative:
            kind = k
            break

    pid = f"p-{year}-{proposer.id[:4]}"
    proposer.proposals_made += 1

    if kind == "leader":
        return Proposal(
            id=pid, year=year, proposer_id=proposer.id, kind="leader",
            description=f"{proposer.name} proposes themselves as colony leader",
            target=proposer.id,
        )

    if kind == "resource":
        new_share = rng.uniform(0.5, 0.9)
        return Proposal(
            id=pid, year=year, proposer_id=proposer.id, kind="resource",
            description=f"Set resource sharing to {new_share:.0%}",
            parameter="resource_share_pct", value=new_share,
        )

    if kind == "exile":
        # Target the least-trusted colonist
        active = [c for c in colonists if c.alive and c.id != proposer.id]
        if not active:
            return None
        target = min(active, key=lambda c: proposer.trust(c.id))
        if proposer.trust(target.id) > -0.2:
            return None  # not distrustful enough
        return Proposal(
            id=pid, year=year, proposer_id=proposer.id, kind="exile",
            description=f"Exile {target.name} from the colony",
            target=target.id,
        )

    if kind == "amendment":
        param = rng.choice(["decision_threshold", "exile_threshold", "leader_term_years"])
        current = getattr(governance.constitution, param)
        delta = rng.choice([-0.05, 0.05]) if isinstance(current, float) else rng.choice([-1, 1])
        new_val = current + delta
        return Proposal(
            id=pid, year=year, proposer_id=proposer.id, kind="amendment",
            description=f"Amend {param} from {current} to {new_val}",
            parameter=param, value=new_val,
        )

    # Default policy proposal
    return Proposal(
        id=pid, year=year, proposer_id=proposer.id, kind="policy",
        description=f"{proposer.name} proposes a colony improvement initiative",
    )


def _run_vote(proposal: Proposal, colonists: list[Colonist],
               governance: GovernanceState) -> None:
    """Conduct a vote on a proposal."""
    threshold = governance.constitution.decision_threshold
    if proposal.kind == "exile":
        threshold = governance.constitution.exile_threshold

    for c in colonists:
        if not c.alive:
            continue
        vote = colonist_votes(c, proposal, governance.leader_id)
        if vote == "for":
            proposal.votes_for.append(c.id)
        elif vote == "against":
            proposal.votes_against.append(c.id)
        else:
            proposal.abstentions.append(c.id)

    proposal.outcome = proposal.vote_result(threshold)


def _apply_proposal_outcome(proposal: Proposal, colonists: list[Colonist],
                              resources: Resources, governance: GovernanceState,
                              year: int) -> None:
    """Apply the effects of a passed proposal."""
    if proposal.outcome != "passed":
        return

    proposer = next((c for c in colonists if c.id == proposal.proposer_id), None)
    if proposer:
        proposer.proposals_passed += 1
        proposer.leadership_score += 0.1

    if proposal.kind == "leader":
        governance.leader_id = proposal.target or proposal.proposer_id
        governance.leader_since = year

    elif proposal.kind == "resource" and proposal.parameter and proposal.value is not None:
        governance.constitution.resource_share_pct = max(0.0, min(1.0, proposal.value))

    elif proposal.kind == "exile" and proposal.target:
        target = next((c for c in colonists if c.id == proposal.target), None)
        if target and target.alive:
            target.die(year, "exiled from colony")
            target.times_exiled += 1
            governance.exile_log.append({"year": year, "colonist": target.id})

    elif proposal.kind == "amendment" and proposal.parameter and proposal.value is not None:
        param = proposal.parameter
        if hasattr(governance.constitution, param):
            setattr(governance.constitution, param, proposal.value)
            governance.constitution.amendments.append({
                "year": year,
                "parameter": param,
                "value": proposal.value,
                "proposed_by": proposal.proposer_id,
            })


def _run_colonist_subsim(colonist: Colonist, colonists: list[Colonist],
                           resources: Resources, year: int,
                           governance: GovernanceState,
                           rng: random.Random) -> dict:
    """Run a sub-simulation for a colonist to evaluate a scenario."""
    # Build a LisPy program that models a governance decision
    colony_state = {
        "year": year,
        "food": resources.food,
        "water": resources.water,
        "power": resources.power,
        "alive": sum(1 for c in colonists if c.alive),
        "morale": resources.morale,
        "crisis": resources.crisis_level(),
    }

    # The colonist constructs a simple LisPy model
    sexpr = f"""
    (let ((food {resources.food:.2f})
          (water {resources.water:.2f})
          (pop {sum(1 for c in colonists if c.alive)})
          (crisis {resources.crisis_level():.2f}))
      (if (> crisis 0.5)
        (list "ration" (* food 0.8) (* water 0.7))
        (if (> food 1.5)
          (list "expand" (+ pop 2) food)
          (list "maintain" pop food))))
    """

    evaluator = Evaluator(step_limit=2000, max_subsim_depth=3, year_budget=5000)
    env = standard_env()

    try:
        expr = parse(sexpr.strip())
        result = evaluator.eval(expr, env)
    except (LispyError, Exception) as e:
        result = f"subsim-error: {e}"

    log = {
        "colonist": colonist.id,
        "year": year,
        "depth": 1,
        "sexpr": sexpr.strip(),
        "result": result if isinstance(result, (str, list, int, float, bool)) else str(result),
        "steps_used": evaluator.steps,
        "nested_subsims": evaluator.subsim_log,
    }

    colonist.add_memory(year, f"ran sub-sim: {result}", 0.7)
    return log


def _check_death(colonist: Colonist, resources: Resources, year: int,
                  rng: random.Random) -> bool:
    """Check if a colonist dies this year. Returns True if dead."""
    if not colonist.alive:
        return False

    # Starvation
    if resources.food <= 0.05:
        if rng.random() < 0.3:
            colonist.die(year, "starvation")
            return True

    # Dehydration
    if resources.water <= 0.05:
        if rng.random() < 0.4:
            colonist.die(year, "dehydration")
            return True

    # Hab breach
    if resources.hab_integrity <= 0.1:
        if rng.random() < 0.2:
            colonist.die(year, "habitat breach")
            return True

    # Age/accident (very low base rate)
    years_on_mars = year - colonist.year_joined
    age_risk = 0.005 + years_on_mars * 0.001
    if rng.random() < age_risk:
        colonist.die(year, "accident" if rng.random() < 0.7 else "natural causes")
        return True

    return False


def _drift_relationships(active: list[Colonist], event: dict,
                           resources: Resources, rng: random.Random) -> None:
    """Relationships drift based on shared experience."""
    crisis = resources.crisis_level()

    for i, a in enumerate(active):
        for b in active[i + 1:]:
            # Shared crisis builds bonds (or destroys them for paranoid)
            if crisis > 0.4:
                if a.stat("empathy") > 0.5 and b.stat("empathy") > 0.5:
                    a.adjust_trust(b.id, 0.02)
                    b.adjust_trust(a.id, 0.02)
                elif a.stat("paranoia") > 0.6 or b.stat("paranoia") > 0.6:
                    a.adjust_trust(b.id, -0.01)
                    b.adjust_trust(a.id, -0.01)

            # Calm times: slow convergence
            if event["id"] == "calm_year":
                a.adjust_trust(b.id, 0.005)
                b.adjust_trust(a.id, 0.005)

            # Small random drift
            drift = rng.gauss(0, 0.01)
            a.adjust_trust(b.id, drift)
            b.adjust_trust(a.id, drift)


def _update_leadership(colonists: list[Colonist], governance: GovernanceState,
                        year: int, rng: random.Random) -> None:
    """Update leadership scores based on effectiveness and proposals."""
    active = [c for c in colonists if c.alive]
    for c in active:
        # Leadership decays slightly each year
        c.leadership_score *= 0.95
        # Boost from effectiveness and cooperation
        c.leadership_score += c.effectiveness() * 0.02
        c.leadership_score += c.cooperation_tendency() * 0.01
        # Passed proposals boost leadership
        c.leadership_score = max(0.0, min(5.0, c.leadership_score))

    # Re-election check
    if governance.leader_id:
        tenure = year - governance.leader_since
        if tenure >= governance.constitution.leader_term_years:
            # Automatic re-election — highest leadership score wins
            if active:
                new_leader = max(active, key=lambda c: c.leadership_score)
                governance.leader_id = new_leader.id
                governance.leader_since = year


def _generate_diaries(active: list[Colonist], event: dict,
                       outcomes: list[dict], year: int,
                       governance: GovernanceState,
                       rng: random.Random) -> list[dict]:
    """Generate diary entries from 3 colonists' perspectives."""
    if len(active) < 3:
        narrators = active[:]
    else:
        # Pick 3: the leader (if alive), most empathetic, most paranoid
        candidates = sorted(active, key=lambda c: c.stat("empathy"), reverse=True)
        narrators = [candidates[0]]  # most empathetic
        paranoid = max(active, key=lambda c: c.stat("paranoia"))
        if paranoid not in narrators:
            narrators.append(paranoid)
        # Third: leader or random
        if governance.leader_id:
            leader = next((c for c in active if c.id == governance.leader_id), None)
            if leader and leader not in narrators:
                narrators.append(leader)
        while len(narrators) < 3 and len(narrators) < len(active):
            pick = rng.choice(active)
            if pick not in narrators:
                narrators.append(pick)

    diaries: list[dict] = []
    for c in narrators[:3]:
        # Find this colonist's action outcome
        my_outcome = next((o for o in outcomes if o.get("colonist") == c.id), {})
        action_desc = my_outcome.get("action", "observed")

        mood = "determined" if c.stat("resolve") > 0.7 else \
               "anxious" if c.stat("paranoia") > 0.6 else \
               "hopeful" if c.stat("faith") > 0.6 else \
               "contemplative"

        entry = (
            f"Year {year} — {c.name} ({c.element})\n"
            f"Mood: {mood}. Event: {event['description']}.\n"
            f"Action: {action_desc}. Colony morale: {'high' if governance.constitution.resource_share_pct > 0.6 else 'strained'}.\n"
        )

        diaries.append({
            "colonist_id": c.id,
            "colonist_name": c.name,
            "element": c.element,
            "year": year,
            "mood": mood,
            "entry": entry,
        })

    return diaries
