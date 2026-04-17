"""mars100_sim.py — Mars-100 recursive colony simulation engine.

100 frames, each = 1 Martian year. 10 colonists with emergent governance.
Sub-simulations via LisPy up to 3 levels deep.

The simulation produces:
  - Per-year deltas keyed by (year, logical_tick)
  - Colonist state evolution
  - Governance proposals, votes, enacted laws
  - Sub-simulation logs
  - Dead colonist archives (soul files)

Python stdlib only.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.lispy import (
    Env,
    LispError,
    DepthExceeded,
    StepLimitExceeded,
    standard_env,
    lisp_eval,
    parse,
)
from src.mars100_colonist import (
    SKILL_NAMES,
    STAT_NAMES,
    add_memory,
    create_all_colonists,
    evolve_stats,
    kill_colonist,
    update_relationship,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOTAL_YEARS = 100

# Environmental events and their probabilities
EVENT_TABLE: list[tuple[str, float]] = [
    ("calm",             0.25),
    ("dust_storm",       0.15),
    ("resource_strike",  0.10),
    ("equipment_failure",0.12),
    ("earth_contact",    0.10),
    ("solar_flare",      0.08),
    ("meteor",           0.05),
    ("epidemic",         0.08),
    ("alien_signal",     0.02),
    ("bountiful_harvest",0.05),
]

# Colony resource model — simple but mechanically meaningful
INITIAL_RESOURCES: dict[str, float] = {
    "food": 100.0,
    "water": 100.0,
    "habitat": 60.0,
    "power": 80.0,
    "morale": 0.7,
    "explored": 0.1,
    "freedom": 0.5,
    "sick": 0,
    "terraforming": 0.0,
}

# Per-year consumption per living colonist
CONSUMPTION: dict[str, float] = {
    "food": 8.0,
    "water": 6.0,
    "power": 3.0,
}

# Death thresholds
STARVATION_THRESHOLD = 5.0  # food per colonist below which death risk rises
DEHYDRATION_THRESHOLD = 3.0

# Governance
VOTE_QUORUM = 0.5  # fraction of living colonists needed to pass a law
PROPOSAL_COOLDOWN = 3  # years between proposals from same colonist

# Actions colonists can take
VALID_ACTIONS = frozenset({
    "gather", "farm", "build", "scout", "guard", "propose",
    "vote", "mediate", "heal", "pray", "sabotage", "trade",
    "fix-water", "sub-sim", "reflect",
})

# Action → resource effects
ACTION_EFFECTS: dict[str, dict[str, float]] = {
    "gather":    {"food": 8.0},
    "farm":      {"food": 12.0},
    "build":     {"habitat": 5.0, "terraforming": 0.005},
    "scout":     {"explored": 0.05},
    "guard":     {"morale": 0.02},
    "mediate":   {"morale": 0.05, "freedom": 0.02},
    "heal":      {"sick": -1},
    "pray":      {"morale": 0.03},
    "fix-water": {"water": 10.0},
    "trade":     {"food": 3.0, "water": 3.0},
    "reflect":   {"morale": 0.01},
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Law:
    """An enacted governance law that modifies colony mechanics."""
    id: str
    year_proposed: int
    year_enacted: int
    proposer: str
    title: str
    effect_type: str  # "resource_bonus", "action_ban", "tax", "freedom"
    effect_params: dict[str, Any] = field(default_factory=dict)
    votes_for: int = 0
    votes_against: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "year_proposed": self.year_proposed,
            "year_enacted": self.year_enacted,
            "proposer": self.proposer,
            "title": self.title,
            "effect_type": self.effect_type,
            "effect_params": self.effect_params,
            "votes_for": self.votes_for,
            "votes_against": self.votes_against,
        }


@dataclass
class Proposal:
    """A pending governance proposal."""
    id: str
    year: int
    proposer: str
    title: str
    effect_type: str
    effect_params: dict[str, Any] = field(default_factory=dict)
    votes_for: list[str] = field(default_factory=list)
    votes_against: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "year": self.year,
            "proposer": self.proposer,
            "title": self.title,
            "effect_type": self.effect_type,
            "effect_params": self.effect_params,
            "votes_for": list(self.votes_for),
            "votes_against": list(self.votes_against),
        }


@dataclass
class YearDelta:
    """Per-year simulation delta — Dream Catcher protocol."""
    year: int
    event: str
    actions: dict[str, str]   # colonist_id → action taken
    resource_changes: dict[str, float]
    proposals_new: list[dict]
    laws_enacted: list[dict]
    deaths: list[dict]
    sub_sims: list[dict]
    diary_entries: list[dict]
    colony_state_snapshot: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "event": self.event,
            "actions": self.actions,
            "resource_changes": self.resource_changes,
            "proposals_new": self.proposals_new,
            "laws_enacted": self.laws_enacted,
            "deaths": self.deaths,
            "sub_sims": self.sub_sims,
            "diary_entries": self.diary_entries,
            "colony_state": self.colony_state_snapshot,
        }


# ---------------------------------------------------------------------------
# Proposal generation templates
# ---------------------------------------------------------------------------

PROPOSAL_TEMPLATES: list[dict[str, Any]] = [
    {
        "title": "Ration food equally",
        "effect_type": "resource_bonus",
        "effect_params": {"food": 2.0},
        "trigger": lambda res: res["food"] < 40,
    },
    {
        "title": "Build water recycler",
        "effect_type": "resource_bonus",
        "effect_params": {"water": 3.0},
        "trigger": lambda res: res["water"] < 30,
    },
    {
        "title": "Expand habitat",
        "effect_type": "resource_bonus",
        "effect_params": {"habitat": 4.0},
        "trigger": lambda res: res["habitat"] < 50,
    },
    {
        "title": "Ban sabotage",
        "effect_type": "action_ban",
        "effect_params": {"action": "sabotage"},
        "trigger": lambda res: res["freedom"] < 0.4,
    },
    {
        "title": "Establish council",
        "effect_type": "freedom",
        "effect_params": {"freedom_delta": 0.1, "morale_delta": 0.05},
        "trigger": lambda res: res["morale"] < 0.4,
    },
    {
        "title": "Collective prayer day",
        "effect_type": "resource_bonus",
        "effect_params": {"morale": 0.08},
        "trigger": lambda res: res["morale"] < 0.5,
    },
    {
        "title": "Terraform initiative",
        "effect_type": "resource_bonus",
        "effect_params": {"terraforming": 0.01},
        "trigger": lambda res: res["terraforming"] < 0.3,
    },
    {
        "title": "Exile dangerous colonist",
        "effect_type": "exile",
        "effect_params": {},
        "trigger": lambda res: res.get("sabotage_count", 0) > 2,
    },
    {
        "title": "Open borders — welcome outsiders",
        "effect_type": "resource_bonus",
        "effect_params": {"food": -3.0, "morale": 0.1, "freedom": 0.05},
        "trigger": lambda res: res["freedom"] > 0.6 and res["food"] > 60,
    },
    {
        "title": "Tax the hoarders",
        "effect_type": "tax",
        "effect_params": {"food_tax": 0.1},
        "trigger": lambda res: res["food"] < 35,
    },
]


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


class Mars100Simulation:
    """Mars-100 recursive colony simulation.

    10 colonists, 100 years, emergent governance, recursive sub-sims.
    """

    def __init__(self, seed: int = 42, total_years: int = TOTAL_YEARS) -> None:
        self.seed = seed
        self.total_years = total_years
        self.rng = random.Random(seed)
        self.colonists = create_all_colonists(seed)
        self.resources: dict[str, Any] = dict(INITIAL_RESOURCES)
        self.resources["sabotage_count"] = 0
        self.laws: list[Law] = []
        self.proposals: list[Proposal] = []
        self.archives: list[dict] = []  # dead colonist soul files
        self.year_deltas: list[YearDelta] = []
        self.all_sub_sim_logs: list[dict] = []
        self.governance_collapsed = False
        self._proposal_counter = 0
        self._law_counter = 0

    def run(self) -> dict[str, Any]:
        """Run the full simulation. Returns structured result dict."""
        for year in range(1, self.total_years + 1):
            delta = self._simulate_year(year)
            self.year_deltas.append(delta)
            if self._colony_dead():
                break
            if self.governance_collapsed:
                break
        return self._build_results()

    def _living_colonists(self) -> list[dict[str, Any]]:
        """Return list of living colonists."""
        return [c for c in self.colonists if c["alive"]]

    def _colony_dead(self) -> bool:
        """Check if all colonists are dead."""
        return len(self._living_colonists()) == 0

    def _simulate_year(self, year: int) -> YearDelta:
        """Simulate one Martian year."""
        # 1. Roll environmental event
        event = self._roll_event()

        # 2. Apply environmental effects
        self._apply_event(event, year)

        # 3. Consume resources
        self._consume_resources()

        # 4. Check for deaths from resource scarcity
        deaths = self._check_deaths(year, event)

        # 5. Each living colonist acts
        actions: dict[str, str] = {}
        sub_sims: list[dict] = []
        for colonist in self._living_colonists():
            action, sub_sim_log = self._colonist_act(colonist, year, event)
            actions[colonist["id"]] = action
            if sub_sim_log:
                sub_sims.extend(sub_sim_log)
            # Evolve stats based on event
            evolve_stats(colonist, year, event, self.rng)

        # 6. Apply action effects
        resource_changes = self._apply_actions(actions, year)

        # 7. Apply law effects
        self._apply_laws()

        # 8. Generate proposals
        new_proposals = self._generate_proposals(year)

        # 9. Resolve votes on existing proposals
        enacted = self._resolve_votes(year)

        # 10. Update relationships based on interactions
        self._update_relationships(actions, year)

        # 11. Generate diary entries
        diary = self._generate_diary(year, event, actions)

        # 12. Check governance collapse
        if self._check_governance_collapse(year):
            self.governance_collapsed = True

        delta = YearDelta(
            year=year,
            event=event,
            actions=actions,
            resource_changes=resource_changes,
            proposals_new=[p.to_dict() for p in new_proposals],
            laws_enacted=[law.to_dict() for law in enacted],
            deaths=deaths,
            sub_sims=sub_sims,
            diary_entries=diary,
            colony_state_snapshot=dict(self.resources),
        )
        return delta

    def _roll_event(self) -> str:
        """Weighted random event selection."""
        roll = self.rng.random()
        cumulative = 0.0
        for event_name, prob in EVENT_TABLE:
            cumulative += prob
            if roll <= cumulative:
                return event_name
        return "calm"

    def _apply_event(self, event: str, year: int) -> None:
        """Apply environmental event effects to resources."""
        effects: dict[str, dict[str, float]] = {
            "dust_storm":       {"power": -15, "morale": -0.05},
            "resource_strike":  {"food": 20, "water": 15},
            "equipment_failure":{"power": -10, "habitat": -5},
            "earth_contact":    {"morale": 0.1},
            "solar_flare":      {"power": -20, "morale": -0.08},
            "meteor":           {"habitat": -10, "morale": -0.1},
            "epidemic":         {"sick": 3, "morale": -0.1},
            "alien_signal":     {"morale": 0.05},
            "bountiful_harvest":{"food": 25, "water": 10},
            "calm":             {"morale": 0.02},
        }
        for resource, delta in effects.get(event, {}).items():
            self.resources[resource] = max(0, self.resources.get(resource, 0) + delta)
            # Clamp morale and freedom to [0, 1]
            if resource in ("morale", "freedom", "explored", "terraforming"):
                self.resources[resource] = max(0.0, min(1.0, self.resources[resource]))

    def _consume_resources(self) -> None:
        """Consume resources based on living population."""
        n_living = len(self._living_colonists())
        for resource, rate in CONSUMPTION.items():
            self.resources[resource] = max(0, self.resources[resource] - rate * n_living)

    def _check_deaths(self, year: int, event: str) -> list[dict]:
        """Check for colonist deaths from scarcity or events."""
        deaths: list[dict] = []
        n_living = len(self._living_colonists())
        if n_living == 0:
            return deaths

        food_per_cap = self.resources["food"] / max(1, n_living)
        water_per_cap = self.resources["water"] / max(1, n_living)

        for colonist in self._living_colonists():
            death_chance = 0.0

            # Starvation risk
            if food_per_cap < STARVATION_THRESHOLD:
                death_chance += 0.15 * (1 - food_per_cap / STARVATION_THRESHOLD)

            # Dehydration risk
            if water_per_cap < DEHYDRATION_THRESHOLD:
                death_chance += 0.2 * (1 - water_per_cap / DEHYDRATION_THRESHOLD)

            # Event-specific risk
            if event == "meteor":
                death_chance += 0.05
            elif event == "epidemic" and self.resources.get("sick", 0) > 2:
                death_chance += 0.08 * (1 - colonist["stats"]["resolve"])
            elif event == "solar_flare":
                death_chance += 0.03

            # Low morale → despair deaths (rare)
            if self.resources["morale"] < 0.1:
                death_chance += 0.05

            # Resolve reduces death chance
            death_chance *= (1 - colonist["stats"]["resolve"] * 0.5)

            if self.rng.random() < death_chance:
                cause = "starvation" if food_per_cap < STARVATION_THRESHOLD else (
                    "dehydration" if water_per_cap < DEHYDRATION_THRESHOLD else event
                )
                soul = kill_colonist(colonist, year, cause)
                self.archives.append(soul)
                deaths.append(soul)

        return deaths

    def _colonist_act(
        self, colonist: dict[str, Any], year: int, event: str,
    ) -> tuple[str, list[dict]]:
        """Determine colonist action via LisPy policy evaluation."""
        sub_sim_logs: list[dict] = []

        # Build LisPy environment with colony context
        env = standard_env(seed=self.seed + year * 100 + hash(colonist["id"]) % 1000)
        env["colony"] = dict(self.resources)
        env["year"] = year
        env["event"] = event
        env["proposals"] = [p.to_dict() for p in self.proposals]
        env["laws"] = [law.to_dict() for law in self.laws]
        env["alive-count"] = len(self._living_colonists())
        env["my-id"] = colonist["id"]

        # Inject colonist stats as top-level variables
        for stat_name, stat_val in colonist["stats"].items():
            env[stat_name] = stat_val
        for skill_name, skill_val in colonist["skills"].items():
            env[f"skill-{skill_name}"] = skill_val

        # Eval the colonist's policy
        sub_sim_logs: list[dict] = []
        try:
            expr = parse(colonist["policy"])
            action = lisp_eval(expr, env, step_limit=500)
        except (SyntaxError, LispError, StepLimitExceeded,
                DepthExceeded) as exc:
            action = "gather"  # fallback

        # Validate action
        if not isinstance(action, str) or action not in VALID_ACTIONS:
            action = "gather"

        # Check if action is banned by law
        for law in self.laws:
            if law.effect_type == "action_ban" and law.effect_params.get("action") == action:
                action = "gather"
                add_memory(colonist, year, f"Wanted to {action} but banned by law: {law.title}")
                break

        colonist["sub_sims_run"] += len(sub_sim_logs)
        add_memory(colonist, year, f"Year {year}: {event} → {action}")
        return action, sub_sim_logs

    def _apply_actions(self, actions: dict[str, str], year: int) -> dict[str, float]:
        """Apply accumulated action effects to colony resources."""
        changes: dict[str, float] = {}
        for colonist_id, action in actions.items():
            colonist = next(c for c in self.colonists if c["id"] == colonist_id)
            effects = dict(ACTION_EFFECTS.get(action, {}))

            # Skill bonus — relevant skill amplifies effect
            skill_map = {
                "gather": "terraforming", "farm": "hydroponics",
                "build": "terraforming", "scout": "terraforming",
                "mediate": "mediation", "heal": "mediation",
                "pray": "prayer", "fix-water": "coding",
                "sabotage": "sabotage",
            }
            relevant_skill = skill_map.get(action)
            if relevant_skill:
                skill_val = colonist["skills"].get(relevant_skill, 0.0)
                for k in effects:
                    effects[k] *= (1.0 + skill_val * 0.5)

            # Sabotage special handling
            if action == "sabotage":
                target_resource = self.rng.choice(["food", "water", "power", "habitat"])
                damage = 5.0 + colonist["skills"]["sabotage"] * 10.0
                effects[target_resource] = -damage
                effects["morale"] = -0.05
                effects["freedom"] = -0.03
                self.resources["sabotage_count"] = self.resources.get("sabotage_count", 0) + 1

            for resource, delta in effects.items():
                self.resources[resource] = max(0, self.resources.get(resource, 0) + delta)
                if resource in ("morale", "freedom", "explored", "terraforming"):
                    self.resources[resource] = max(0.0, min(1.0, self.resources[resource]))
                changes[resource] = changes.get(resource, 0) + delta

        return changes

    def _apply_laws(self) -> None:
        """Apply effects of enacted laws to colony state."""
        for law in self.laws:
            if law.effect_type == "resource_bonus":
                for resource, bonus in law.effect_params.items():
                    self.resources[resource] = max(0, self.resources.get(resource, 0) + bonus)
                    if resource in ("morale", "freedom", "explored", "terraforming"):
                        self.resources[resource] = min(1.0, self.resources[resource])
            elif law.effect_type == "freedom":
                self.resources["freedom"] = max(0.0, min(1.0,
                    self.resources["freedom"] + law.effect_params.get("freedom_delta", 0)))
                self.resources["morale"] = max(0.0, min(1.0,
                    self.resources["morale"] + law.effect_params.get("morale_delta", 0)))
            elif law.effect_type == "tax":
                tax_rate = law.effect_params.get("food_tax", 0)
                # Tax redistributes — slight net positive from efficiency
                self.resources["food"] = max(0, self.resources["food"] * (1 + tax_rate * 0.1))

    def _generate_proposals(self, year: int) -> list[Proposal]:
        """Generate new governance proposals from colonists."""
        new_proposals: list[Proposal] = []
        for colonist in self._living_colonists():
            # Only propose if archetype/stats incline and cooldown passed
            propose_chance = (
                colonist["stats"]["resolve"] * 0.3
                + colonist["stats"]["empathy"] * 0.2
                + (0.3 if colonist["archetype"] in ("strategist", "visionary", "rebel") else 0.1)
            )
            if self.rng.random() > propose_chance:
                continue

            # Find a triggered proposal template
            for template in PROPOSAL_TEMPLATES:
                if template["trigger"](self.resources):
                    # Check not already proposed
                    existing_titles = {p.title for p in self.proposals}
                    if template["title"] in existing_titles:
                        continue
                    existing_law_titles = {law.title for law in self.laws}
                    if template["title"] in existing_law_titles:
                        continue

                    self._proposal_counter += 1
                    proposal = Proposal(
                        id=f"prop-{self._proposal_counter}",
                        year=year,
                        proposer=colonist["id"],
                        title=template["title"],
                        effect_type=template["effect_type"],
                        effect_params=dict(template["effect_params"]),
                        votes_for=[colonist["id"]],
                        votes_against=[],
                    )
                    self.proposals.append(proposal)
                    new_proposals.append(proposal)
                    colonist["proposals_made"] += 1
                    add_memory(colonist, year, f"Proposed: {template['title']}")
                    break  # one proposal per colonist per year

        return new_proposals

    def _resolve_votes(self, year: int) -> list[Law]:
        """Resolve pending proposals — colonists vote based on relationships and stats."""
        enacted: list[Law] = []
        living = self._living_colonists()
        n_living = len(living)
        quorum = max(1, int(n_living * VOTE_QUORUM))

        remaining: list[Proposal] = []
        for proposal in self.proposals:
            # Each living colonist votes
            for colonist in living:
                if colonist["id"] in proposal.votes_for:
                    continue
                if colonist["id"] in proposal.votes_against:
                    continue

                # Vote based on relationship to proposer + own stats
                proposer_affinity = colonist["relationships"].get(proposal.proposer, 0)
                empathy_factor = colonist["stats"]["empathy"]
                paranoia_factor = colonist["stats"]["paranoia"]

                vote_yes_prob = 0.5 + proposer_affinity * 0.3 + empathy_factor * 0.1 - paranoia_factor * 0.15
                vote_yes_prob = max(0.1, min(0.9, vote_yes_prob))

                if self.rng.random() < vote_yes_prob:
                    proposal.votes_for.append(colonist["id"])
                else:
                    proposal.votes_against.append(colonist["id"])

                colonist["votes_cast"] += 1

            # Check if passed
            if len(proposal.votes_for) >= quorum:
                self._law_counter += 1
                law = Law(
                    id=f"law-{self._law_counter}",
                    year_proposed=proposal.year,
                    year_enacted=year,
                    proposer=proposal.proposer,
                    title=proposal.title,
                    effect_type=proposal.effect_type,
                    effect_params=dict(proposal.effect_params),
                    votes_for=len(proposal.votes_for),
                    votes_against=len(proposal.votes_against),
                )
                self.laws.append(law)
                enacted.append(law)
            elif year - proposal.year > 3:
                pass  # expired — drop
            else:
                remaining.append(proposal)

        self.proposals = remaining
        return enacted

    def _update_relationships(self, actions: dict[str, str], year: int) -> None:
        """Update relationships based on cooperative/conflicting actions."""
        living = self._living_colonists()
        cooperative_actions = {"farm", "build", "mediate", "heal", "trade"}
        antisocial_actions = {"sabotage"}

        for i, c1 in enumerate(living):
            for j, c2 in enumerate(living):
                if i >= j:
                    continue
                a1 = actions.get(c1["id"], "gather")
                a2 = actions.get(c2["id"], "gather")

                # Same cooperative action → bond
                if a1 in cooperative_actions and a1 == a2:
                    update_relationship(c1, c2["id"], 0.05)
                    update_relationship(c2, c1["id"], 0.05)
                # One sabotages → enmity
                elif a1 in antisocial_actions or a2 in antisocial_actions:
                    update_relationship(c1, c2["id"], -0.08)
                    update_relationship(c2, c1["id"], -0.08)
                # Natural drift toward neutral
                else:
                    current = c1["relationships"].get(c2["id"], 0)
                    drift = -0.01 * (1 if current > 0 else -1)
                    update_relationship(c1, c2["id"], drift)
                    update_relationship(c2, c1["id"], drift)

    def _generate_diary(
        self, year: int, event: str, actions: dict[str, str],
    ) -> list[dict]:
        """Generate diary entries from 3 selected colonists."""
        living = self._living_colonists()
        if not living:
            return []

        # Pick up to 3 colonists with highest resolve for narration
        narrators = sorted(living, key=lambda c: c["stats"]["resolve"], reverse=True)[:3]
        entries: list[dict] = []

        for colonist in narrators:
            action = actions.get(colonist["id"], "idle")
            entry = {
                "year": year,
                "colonist": colonist["id"],
                "name": colonist["name"],
                "text": self._diary_text(colonist, year, event, action),
            }
            entries.append(entry)

        return entries

    def _diary_text(
        self, colonist: dict, year: int, event: str, action: str,
    ) -> str:
        """Generate a diary entry string."""
        name = colonist["name"]
        morale_desc = (
            "spirits are high" if self.resources["morale"] > 0.6
            else "tension fills the hab" if self.resources["morale"] > 0.3
            else "despair is setting in"
        )

        event_descs = {
            "calm": "Another quiet year on Mars.",
            "dust_storm": "The dust storm raged for weeks. We huddled inside.",
            "resource_strike": "We found a new mineral vein! Hope returns.",
            "equipment_failure": "The recycler broke down. Repairs took months.",
            "earth_contact": "A message from Earth. Some wept. Others raged.",
            "solar_flare": "The flare knocked out half our panels. Dark days.",
            "meteor": "A meteor struck near the hab. We were lucky.",
            "epidemic": "The sickness spread fast. We lost people.",
            "alien_signal": "The signal... it changed everything. We are not alone.",
            "bountiful_harvest": "The greenhouse yielded beyond expectations.",
        }
        event_desc = event_descs.get(event, "An unremarkable year.")

        n_alive = len(self._living_colonists())
        n_laws = len(self.laws)

        return (
            f"[Year {year}] {name}'s log: {event_desc} "
            f"I chose to {action}. {morale_desc}. "
            f"{n_alive} of us remain. {n_laws} laws govern us. "
            f"Food: {self.resources['food']:.0f}, Water: {self.resources['water']:.0f}, "
            f"Terraforming: {self.resources['terraforming']:.1%}."
        )

    def _check_governance_collapse(self, year: int) -> bool:
        """Check if governance has collapsed (no laws + low morale + high paranoia)."""
        if year < 10:
            return False
        if self.resources["morale"] < 0.1 and self.resources["freedom"] < 0.1:
            avg_paranoia = sum(
                c["stats"]["paranoia"] for c in self._living_colonists()
            ) / max(1, len(self._living_colonists()))
            if avg_paranoia > 0.8:
                return True
        return False

    def _build_results(self) -> dict[str, Any]:
        """Build the final structured result dict."""
        years_survived = len(self.year_deltas)
        living = self._living_colonists()

        # Governance analysis
        governance_patterns: list[str] = []
        if len(self.laws) == 0:
            governance_patterns.append("anarchy")
        elif any(law.effect_type == "action_ban" for law in self.laws):
            governance_patterns.append("authoritarian")
        if any(law.effect_type == "freedom" for law in self.laws):
            governance_patterns.append("democratic")
        if any(law.effect_type == "tax" for law in self.laws):
            governance_patterns.append("redistributive")
        if self.governance_collapsed:
            governance_patterns.append("collapsed")

        # Sub-sim analysis
        total_sub_sims = sum(c["sub_sims_run"] for c in self.colonists)
        depth_3_sims = [
            s for delta in self.year_deltas
            for s in delta.sub_sims
            if s.get("depth", 0) >= 3
        ]

        # Value convergence — did colonists' stats converge or diverge?
        if living:
            stat_variance: dict[str, float] = {}
            for stat in STAT_NAMES:
                vals = [c["stats"][stat] for c in living]
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                stat_variance[stat] = round(var, 4)
        else:
            stat_variance = {}

        return {
            "_meta": {
                "engine": "mars-100",
                "version": "1.0",
                "seed": self.seed,
                "total_years": self.total_years,
                "years_survived": years_survived,
            },
            "summary": {
                "years_survived": years_survived,
                "colonists_start": 10,
                "colonists_end": len(living),
                "deaths": len(self.archives),
                "laws_enacted": len(self.laws),
                "proposals_total": self._proposal_counter,
                "governance_collapsed": self.governance_collapsed,
                "governance_patterns": governance_patterns,
                "total_sub_sims": total_sub_sims,
                "depth_3_sims": len(depth_3_sims),
                "final_resources": {k: round(v, 2) if isinstance(v, float) else v
                                    for k, v in self.resources.items()},
                "stat_variance": stat_variance,
            },
            "colonists": [
                {
                    "id": c["id"],
                    "name": c["name"],
                    "element": c["element"],
                    "archetype": c["archetype"],
                    "alive": c["alive"],
                    "death_year": c["death_year"],
                    "stats": {k: round(v, 3) for k, v in c["stats"].items()},
                    "skills": {k: round(v, 3) for k, v in c["skills"].items()},
                    "memory": c["memory"][-10:],  # last 10 memories
                    "proposals_made": c["proposals_made"],
                    "votes_cast": c["votes_cast"],
                    "sub_sims_run": c["sub_sims_run"],
                }
                for c in self.colonists
            ],
            "laws": [law.to_dict() for law in self.laws],
            "archives": self.archives,
            "year_deltas": [d.to_dict() for d in self.year_deltas],
        }
