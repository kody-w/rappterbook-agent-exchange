"""
mars100.py — Mars-100: A Recursive Colony Experiment.

100-year Mars colony simulation with 10 agent-colonists.
Each frame = 1 Martian year. Colonists make decisions via LisPy expressions.
Sub-simulations allowed up to depth 3 (Turtles All the Way Down).

Usage:
    from src.mars100 import Mars100
    sim = Mars100(seed=42)
    results = sim.run(years=100)

Python stdlib only (+ sibling lispy.py).
"""
from __future__ import annotations

import json
import math
import os
import random
import hashlib
import copy
from pathlib import Path
from datetime import datetime, timezone

# Import LisPy engine
import sys
_SRC = Path(__file__).resolve().parent
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))
from src.lispy import Lispy, LispyError

REPO_ROOT = _SRC.parent
STATE_DIR = Path(os.environ.get("STATE_DIR", str(REPO_ROOT / "state")))
DOCS_DIR = Path(os.environ.get("DOCS_DIR", str(REPO_ROOT / "docs")))


def now_iso() -> str:
    """Current UTC timestamp in ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


# --- Colonist archetypes ---

ELEMENTS = ["fire", "water", "earth", "air"]

COLONIST_TEMPLATES = [
    {
        "id": "kael",
        "name": "Kael Ashborne",
        "element": "fire",
        "stats": {"resolve": 0.9, "improvisation": 0.7, "empathy": 0.3,
                  "hoarding": 0.2, "faith": 0.4, "paranoia": 0.5},
        "skills": {"terraforming": 0.8, "hydroponics": 0.2, "mediation": 0.1,
                   "coding": 0.6, "prayer": 0.1, "sabotage": 0.3},
        "personality": "(if (> danger 0.7) 'fortify (if (< food 50) 'ration 'terraform))",
    },
    {
        "id": "lyra",
        "name": "Lyra Tideweaver",
        "element": "water",
        "stats": {"resolve": 0.5, "improvisation": 0.8, "empathy": 0.9,
                  "hoarding": 0.1, "faith": 0.6, "paranoia": 0.2},
        "skills": {"terraforming": 0.3, "hydroponics": 0.7, "mediation": 0.8,
                   "coding": 0.2, "prayer": 0.5, "sabotage": 0.0},
        "personality": "(if (> unrest 0.4) 'mediate (if (< water 40) 'farm 'explore))",
    },
    {
        "id": "grond",
        "name": "Grond Ironfist",
        "element": "earth",
        "stats": {"resolve": 0.8, "improvisation": 0.3, "empathy": 0.4,
                  "hoarding": 0.6, "faith": 0.3, "paranoia": 0.4},
        "skills": {"terraforming": 0.5, "hydroponics": 0.6, "mediation": 0.2,
                   "coding": 0.1, "prayer": 0.2, "sabotage": 0.4},
        "personality": "(if (< supplies 30) 'hoard (if (< food 40) 'farm 'mine))",
    },
    {
        "id": "zephyr",
        "name": "Zephyr Windcaller",
        "element": "air",
        "stats": {"resolve": 0.4, "improvisation": 0.9, "empathy": 0.6,
                  "hoarding": 0.1, "faith": 0.7, "paranoia": 0.3},
        "skills": {"terraforming": 0.4, "hydroponics": 0.3, "mediation": 0.5,
                   "coding": 0.7, "prayer": 0.6, "sabotage": 0.1},
        "personality": "(if (> wonder 0.3) 'explore (if (< power 30) 'code 'terraform))",
    },
    {
        "id": "ember",
        "name": "Ember Solstice",
        "element": "fire",
        "stats": {"resolve": 0.7, "improvisation": 0.6, "empathy": 0.5,
                  "hoarding": 0.3, "faith": 0.8, "paranoia": 0.2},
        "skills": {"terraforming": 0.6, "hydroponics": 0.4, "mediation": 0.3,
                   "coding": 0.5, "prayer": 0.7, "sabotage": 0.1},
        "personality": "(if (> danger 0.5) 'pray (if (< morale 0.4) 'pray 'terraform))",
    },
    {
        "id": "nyx",
        "name": "Nyx Deepcurrent",
        "element": "water",
        "stats": {"resolve": 0.6, "improvisation": 0.7, "empathy": 0.4,
                  "hoarding": 0.4, "faith": 0.2, "paranoia": 0.7},
        "skills": {"terraforming": 0.2, "hydroponics": 0.8, "mediation": 0.4,
                   "coding": 0.8, "prayer": 0.1, "sabotage": 0.5},
        "personality": "(if (> paranoia 0.6) 'fortify (if (< oxygen 40) 'code 'farm))",
    },
    {
        "id": "terra",
        "name": "Terra Rootborn",
        "element": "earth",
        "stats": {"resolve": 0.7, "improvisation": 0.4, "empathy": 0.7,
                  "hoarding": 0.2, "faith": 0.5, "paranoia": 0.3},
        "skills": {"terraforming": 0.9, "hydroponics": 0.5, "mediation": 0.6,
                   "coding": 0.2, "prayer": 0.3, "sabotage": 0.0},
        "personality": "(if (< habitat 0.3) 'terraform (if (< food 50) 'farm 'terraform))",
    },
    {
        "id": "volt",
        "name": "Volt Stormrider",
        "element": "air",
        "stats": {"resolve": 0.5, "improvisation": 0.8, "empathy": 0.3,
                  "hoarding": 0.5, "faith": 0.1, "paranoia": 0.6},
        "skills": {"terraforming": 0.3, "hydroponics": 0.2, "mediation": 0.1,
                   "coding": 0.9, "prayer": 0.0, "sabotage": 0.6},
        "personality": "(if (> unrest 0.5) 'sabotage (if (< power 40) 'code 'explore))",
    },
    {
        "id": "sage",
        "name": "Sage Cloudwhisper",
        "element": "air",
        "stats": {"resolve": 0.6, "improvisation": 0.5, "empathy": 0.8,
                  "hoarding": 0.1, "faith": 0.9, "paranoia": 0.1},
        "skills": {"terraforming": 0.2, "hydroponics": 0.3, "mediation": 0.9,
                   "coding": 0.4, "prayer": 0.8, "sabotage": 0.0},
        "personality": "(if (> isolation 0.4) 'pray (if (> unrest 0.3) 'mediate 'explore))",
    },
    {
        "id": "cinder",
        "name": "Cinder Flameheart",
        "element": "fire",
        "stats": {"resolve": 0.8, "improvisation": 0.6, "empathy": 0.2,
                  "hoarding": 0.7, "faith": 0.3, "paranoia": 0.8},
        "skills": {"terraforming": 0.4, "hydroponics": 0.1, "mediation": 0.0,
                   "coding": 0.3, "prayer": 0.2, "sabotage": 0.8},
        "personality": "(if (> paranoia 0.7) 'sabotage (if (< supplies 40) 'hoard 'mine))",
    },
]

# --- Events ---

EVENTS = [
    {"id": "dust_storm", "name": "Dust Storm", "severity": (0.2, 0.8),
     "effects": {"power": -15, "habitat_integrity": -0.1, "danger": 0.3}},
    {"id": "resource_strike", "name": "Mineral Discovery", "severity": (0.1, 0.5),
     "effects": {"supplies": 20, "morale": 0.1}},
    {"id": "equipment_failure", "name": "Equipment Failure", "severity": (0.3, 0.7),
     "effects": {"power": -10, "danger": 0.2, "habitat_integrity": -0.05}},
    {"id": "earth_contact", "name": "Contact with Earth", "severity": (0.1, 0.3),
     "effects": {"morale": 0.15, "isolation": -0.2}},
    {"id": "alien_signal", "name": "Anomalous Signal", "severity": (0.1, 0.4),
     "effects": {"wonder": 0.3, "paranoia": 0.1}},
    {"id": "greenhouse_bloom", "name": "Greenhouse Bloom", "severity": (0.1, 0.4),
     "effects": {"food": 25, "oxygen": 10, "morale": 0.05}},
    {"id": "greenhouse_blight", "name": "Greenhouse Blight", "severity": (0.3, 0.6),
     "effects": {"food": -20, "morale": -0.1}},
    {"id": "solar_flare", "name": "Solar Flare", "severity": (0.4, 0.9),
     "effects": {"power": -20, "danger": 0.4, "habitat_integrity": -0.05}},
    {"id": "water_geyser", "name": "Subsurface Water Geyser", "severity": (0.1, 0.5),
     "effects": {"water": 30, "morale": 0.1, "wonder": 0.2}},
    {"id": "cave_discovery", "name": "Lava Tube Discovery", "severity": (0.1, 0.3),
     "effects": {"habitat_integrity": 0.1, "supplies": 10, "wonder": 0.15}},
    {"id": "oxygen_leak", "name": "Oxygen Leak", "severity": (0.3, 0.7),
     "effects": {"oxygen": -15, "danger": 0.3}},
    {"id": "supply_drop", "name": "Earth Supply Drop", "severity": (0.1, 0.4),
     "effects": {"food": 30, "water": 15, "supplies": 25, "morale": 0.1}},
]

# --- Actions ---

ACTIONS = {
    "terraform": {"habitat_integrity": 0.05, "power": -3, "morale": 0.02},
    "farm": {"food": 15, "water": -3, "power": -2, "morale": 0.01},
    "mine": {"supplies": 10, "power": -3},
    "code": {"efficiency": 0.03, "power": -2},
    "mediate": {"unrest": -0.08, "morale": 0.05},
    "pray": {"morale": 0.04, "faith_bonus": 0.02},
    "hoard": {"hoarding_penalty": 0.03, "unrest": 0.02},
    "sabotage": {"danger": 0.1, "unrest": 0.1, "paranoia_spread": 0.05},
    "explore": {"wonder": 0.05, "supplies": 5},
    "fortify": {"habitat_integrity": 0.08, "supplies": -5},
    "ration": {"food": -5, "morale": -0.03, "unrest": 0.02},
    "protect": {"danger": -0.1, "morale": 0.02},
    "research": {"efficiency": 0.04, "power": -4},
    "heal": {"morale": 0.06, "food": -3},
    "trade": {"supplies": -5, "food": 8, "morale": 0.02},
    "build": {"habitat_integrity": 0.06, "supplies": -8, "power": -3},
    "teach": {"efficiency": 0.02, "morale": 0.03},
    "scout": {"wonder": 0.08, "danger": 0.05},
    "repair": {"habitat_integrity": 0.04, "power": -2, "supplies": -3},
    "conserve": {"power": 5, "food": 3, "morale": -0.01},
    "feast": {"food": -10, "morale": 0.08, "unrest": -0.05},
}

# --- Governance templates ---

GOVERNANCE_TEMPLATES = [
    {
        "id": "democracy",
        "name": "Direct Democracy",
        "lispy": "(define governance-score (if (> avg-empathy 0.5) 0.8 0.4))",
        "effects": {"unrest": -0.1, "morale": 0.05, "efficiency": 0.02},
    },
    {
        "id": "council",
        "name": "Elder Council",
        "lispy": "(define governance-score (* colonist-count 0.1))",
        "effects": {"unrest": -0.05, "efficiency": 0.05},
    },
    {
        "id": "meritocracy",
        "name": "Meritocratic Rotation",
        "lispy": "(define governance-score (if (> colonist-count 5) 0.7 0.3))",
        "effects": {"efficiency": 0.08, "morale": 0.02, "unrest": -0.03},
    },
    {
        "id": "commune",
        "name": "Resource Commune",
        "lispy": "(define governance-score (- 1.0 unrest))",
        "effects": {"morale": 0.08, "unrest": -0.08, "efficiency": -0.02},
    },
    {
        "id": "technocracy",
        "name": "Technocratic Council",
        "lispy": "(define governance-score (+ avg-empathy (* colonist-count 0.05)))",
        "effects": {"efficiency": 0.1, "morale": -0.02},
    },
    {
        "id": "anarchy",
        "name": "Structured Anarchy",
        "lispy": "(define governance-score (if (< unrest 0.2) 0.9 0.1))",
        "effects": {"morale": 0.03, "unrest": 0.05, "efficiency": -0.03},
    },
]


class Colonist:
    """Individual Mars colonist with LisPy personality."""

    def __init__(self, template: dict, rng: random.Random):
        self.id: str = template["id"]
        self.name: str = template["name"]
        self.element: str = template["element"]
        self.stats: dict[str, float] = dict(template["stats"])
        self.skills: dict[str, float] = dict(template["skills"])
        self.personality: str = template["personality"]
        self.alive: bool = True
        self.role: str = "colonist"
        self.memory: list[str] = []
        self.year_born: int = 0
        self.relationships: dict[str, float] = {}

    def decide(self, context: dict, lispy: Lispy) -> str:
        """Use LisPy personality to decide an action."""
        for k, v in context.items():
            lispy.global_env.set(k, v)
        for k, v in self.stats.items():
            lispy.global_env.set(k, v)

        try:
            result = lispy.eval_string(self.personality)
            if isinstance(result, str) and result in ACTIONS:
                return result
            # LisPy quotes return Symbol objects
            result_name = getattr(result, 'name', None)
            if result_name and result_name in ACTIONS:
                return result_name
        except LispyError:
            pass

        # Fallback: pick action based on highest skill
        best_skill = max(self.skills, key=self.skills.get)
        skill_to_action = {
            "terraforming": "terraform", "hydroponics": "farm",
            "mediation": "mediate", "coding": "code",
            "prayer": "pray", "sabotage": "hoard",
        }
        return skill_to_action.get(best_skill, "explore")

    def age(self, year: int) -> int:
        """Colonist age in Mars years."""
        return year - self.year_born

    def to_dict(self) -> dict:
        """Serialize colonist state."""
        return {
            "id": self.id,
            "name": self.name,
            "element": self.element,
            "stats": self.stats,
            "skills": self.skills,
            "personality": self.personality,
            "alive": self.alive,
            "role": self.role,
            "memory": self.memory[-20:],
            "year_born": self.year_born,
            "relationships": self.relationships,
        }


class Colony:
    """Mars-100 colony state."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.seed = seed
        self.year = 0
        self.resources = {
            "food": 200.0,
            "water": 200.0,
            "power": 150.0,
            "oxygen": 200.0,
            "supplies": 120.0,
        }
        self.metrics = {
            "morale": 0.6,
            "unrest": 0.1,
            "danger": 0.0,
            "habitat_integrity": 0.85,
            "efficiency": 0.5,
            "wonder": 0.0,
            "isolation": 0.0,
        }
        self.governance: list[dict] = []
        self.constitution: list[str] = [
            "All colonists have equal voice",
            "Resources are shared by need",
            "No colonist may be exiled without trial",
        ]
        self.colonists = [
            Colonist(t, random.Random(seed + i))
            for i, t in enumerate(COLONIST_TEMPLATES)
        ]
        for c in self.colonists:
            for other in self.colonists:
                if c.id != other.id:
                    base = 0.5 + self.rng.gauss(0, 0.15)
                    if c.element == other.element:
                        base += 0.1
                    c.relationships[other.id] = max(-1.0, min(1.0, base))

        self.year_log: list[dict] = []
        self.sub_sim_archive: list[dict] = []
        self.dead_colonists: list[dict] = []
        self.governance_proposals: list[dict] = []
        self.amendments: list[dict] = []

    @property
    def alive_colonists(self) -> list[Colonist]:
        """Return living colonists."""
        return [c for c in self.colonists if c.alive]

    @property
    def collapsed(self) -> bool:
        """Check if colony has collapsed."""
        return len(self.alive_colonists) == 0


class Mars100:
    """Mars-100 simulation engine."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.colony = Colony(seed)
        self.rng = self.colony.rng

    def run(self, years: int = 100) -> dict:
        """Run the simulation for N years."""
        results: list[dict] = []

        for year in range(1, years + 1):
            if self.colony.collapsed:
                break
            year_result = self.tick(year)
            results.append(year_result)

        return {
            "_meta": {
                "engine": "mars-100",
                "version": "1.0",
                "seed": self.seed,
                "years_simulated": len(results),
                "final_population": len(self.colony.alive_colonists),
                "generated": now_iso(),
            },
            "colony": self._colony_snapshot(),
            "year_log": results,
            "amendments": self.colony.amendments,
            "sub_sim_archive": self.colony.sub_sim_archive[-50:],
            "dead_colonists": self.colony.dead_colonists,
        }

    def tick(self, year: int) -> dict:
        """Advance one Mars year."""
        self.colony.year = year
        lispy = Lispy(seed=self.seed + year, step_limit=10_000)

        # 1. Environmental events
        events = self._generate_events(year)

        # 2. Apply environmental effects
        for event in events:
            self._apply_event(event)

        # 3. Resource consumption
        self._consume_resources()

        # 4. Each colonist decides and acts
        actions: list[dict] = []
        context = self._build_context(year, events)
        for colonist in self.colony.alive_colonists:
            action = colonist.decide(context, Lispy(seed=self.seed + year + hash(colonist.id)))
            effect = self._apply_action(colonist, action)
            actions.append({
                "colonist": colonist.id,
                "action": action,
                "effect": effect,
            })
            colonist.memory.append(f"Year {year}: I chose to {action}")

            if lispy.sub_sim_log:
                for log in lispy.sub_sim_log:
                    self.colony.sub_sim_archive.append({
                        "year": year,
                        "colonist": colonist.id,
                        **log,
                    })

        # 5. Governance phase
        governance_event = self._governance_phase(year, lispy)

        # 6. Update relationships
        self._update_relationships(year, events, actions)

        # 7. Check for deaths
        deaths = self._check_deaths(year)

        # 8. Check for births (after year 5)
        births = self._check_births(year)

        # 9. Natural dynamics
        self._natural_dynamics()

        # 10. Check for meta-insight
        meta_insight = self._check_meta_insight(year, lispy)

        year_entry = {
            "year": year,
            "events": [{"id": e["id"], "name": e["name"], "severity": e["severity"]} for e in events],
            "actions": actions,
            "governance": governance_event,
            "deaths": deaths,
            "births": births,
            "resources": dict(self.colony.resources),
            "metrics": dict(self.colony.metrics),
            "population": len(self.colony.alive_colonists),
            "meta_insight": meta_insight,
            "timestamp": now_iso(),
        }
        self.colony.year_log.append(year_entry)
        return year_entry

    def _generate_events(self, year: int) -> list[dict]:
        """Generate 1-2 environmental events for the year."""
        num_events = 1 if self.rng.random() < 0.6 else 2
        events = []
        for _ in range(num_events):
            template = self.rng.choice(EVENTS)
            severity = self.rng.uniform(*template["severity"])
            events.append({
                "id": template["id"],
                "name": template["name"],
                "severity": round(severity, 3),
                "effects": {k: v * severity for k, v in template["effects"].items()},
            })
        return events

    def _apply_event(self, event: dict) -> None:
        """Apply event effects to colony."""
        for key, value in event["effects"].items():
            if key in self.colony.resources:
                self.colony.resources[key] = max(0.0, self.colony.resources[key] + value)
            elif key in self.colony.metrics:
                self.colony.metrics[key] = max(-1.0, min(1.0, self.colony.metrics[key] + value))

    def _consume_resources(self) -> None:
        """Yearly consumption per colonist, scaled by efficiency."""
        pop = len(self.colony.alive_colonists)
        if pop == 0:
            return
        eff = 1.0 - self.colony.metrics["efficiency"] * 0.3
        self.colony.resources["food"] = max(0.0, self.colony.resources["food"] - pop * 1.8 * eff)
        self.colony.resources["water"] = max(0.0, self.colony.resources["water"] - pop * 1.0 * eff)
        self.colony.resources["power"] = max(0.0, self.colony.resources["power"] - pop * 0.7 * eff)
        self.colony.resources["oxygen"] = max(0.0, self.colony.resources["oxygen"] - pop * 0.3 * eff)

    def _apply_action(self, colonist: Colonist, action: str) -> dict:
        """Apply a colonist's action to colony state."""
        effects = ACTIONS.get(action, {})
        applied: dict[str, float] = {}
        skill_mult = 1.0

        if action == "terraform" and "terraforming" in colonist.skills:
            skill_mult = 1.0 + colonist.skills["terraforming"] * 0.5
        elif action == "farm" and "hydroponics" in colonist.skills:
            skill_mult = 1.0 + colonist.skills["hydroponics"] * 0.5
        elif action == "mediate" and "mediation" in colonist.skills:
            skill_mult = 1.0 + colonist.skills["mediation"] * 0.5
        elif action == "code" and "coding" in colonist.skills:
            skill_mult = 1.0 + colonist.skills["coding"] * 0.5
        elif action == "sabotage" and "sabotage" in colonist.skills:
            skill_mult = 1.0 + colonist.skills["sabotage"] * 0.3

        for key, value in effects.items():
            if key in ("governance", "faith_bonus", "paranoia_spread",
                       "hoarding_penalty", "faith_penalty", "insight"):
                applied[key] = value
                continue
            scaled = value * skill_mult
            if key in self.colony.resources:
                self.colony.resources[key] = max(0.0, self.colony.resources[key] + scaled)
                applied[key] = round(scaled, 2)
            elif key in self.colony.metrics:
                self.colony.metrics[key] = max(-1.0, min(1.0, self.colony.metrics[key] + scaled))
                applied[key] = round(scaled, 3)

        # Skill growth from practice
        action_skill_map = {
            "terraform": "terraforming", "farm": "hydroponics",
            "mediate": "mediation", "code": "coding",
            "pray": "prayer", "sabotage": "sabotage",
        }
        if action in action_skill_map and action_skill_map[action] in colonist.skills:
            sk = action_skill_map[action]
            colonist.skills[sk] = min(1.0, colonist.skills[sk] + 0.01)

        return applied

    def _build_context(self, year: int, events: list[dict]) -> dict:
        """Build context variables for colonist LisPy evaluation."""
        r = self.colony.resources
        m = self.colony.metrics
        max_danger = max((e.get("effects", {}).get("danger", 0) for e in events), default=0)
        return {
            "year": year,
            "food": r["food"],
            "water": r["water"],
            "power": r["power"],
            "oxygen": r["oxygen"],
            "supplies": r["supplies"],
            "morale": m["morale"],
            "unrest": m["unrest"],
            "danger": max(m["danger"], max_danger),
            "habitat": m["habitat_integrity"],
            "trust": 1.0 - m["unrest"],
            "population": len(self.colony.alive_colonists),
        }

    def _governance_phase(self, year: int, lispy: Lispy) -> dict | None:
        """Check for governance proposals and voting."""
        if year < 5:
            return None

        proposer = None
        for c in self.colony.alive_colonists:
            if c.skills.get("mediation", 0) > 0.6 and self.colony.metrics["unrest"] > 0.3:
                proposer = c
                break
            if self.rng.random() < 0.05:
                proposer = c
                break

        if proposer is None:
            return None

        template = self.rng.choice(GOVERNANCE_TEMPLATES)

        # Run a sub-sim to evaluate the proposal (65% chance)
        sub_sim_result = None
        if self.rng.random() < 0.65:
            try:
                proposal_lispy = Lispy(
                    seed=self.seed + year + 9999,
                    sim_depth=0,
                    step_limit=5000,
                )
                test_expr = f"""
                (begin
                    (define colonist-count {len(self.colony.alive_colonists)})
                    (define avg-empathy {sum(c.stats['empathy'] for c in self.colony.alive_colonists) / max(1, len(self.colony.alive_colonists)):.2f})
                    (define unrest {self.colony.metrics['unrest']:.2f})
                    {template['lispy']}
                    (list colonist-count avg-empathy unrest))
                """
                sub_sim_result = proposal_lispy.eval_string(test_expr)
                self.colony.sub_sim_archive.append({
                    "year": year,
                    "colonist": proposer.id,
                    "label": f"governance-test-{template['id']}",
                    "depth": 1,
                    "source": test_expr[:300],
                    "result": str(sub_sim_result)[:200],
                    "steps": proposal_lispy.steps,
                    "child_logs": proposal_lispy.sub_sim_log,
                })
            except LispyError:
                sub_sim_result = "evaluation-failed"

        # Vote
        votes_for = 0
        votes_against = 0
        for c in self.colony.alive_colonists:
            if c.stats["empathy"] > 0.5:
                votes_for += 1
            elif c.stats["paranoia"] > 0.6:
                votes_against += 1
            elif self.rng.random() < 0.5:
                votes_for += 1
            else:
                votes_against += 1

        passed = votes_for > votes_against
        proposal = {
            "year": year,
            "proposer": proposer.id,
            "type": template["id"],
            "name": template["name"],
            "lispy": template["lispy"],
            "votes_for": votes_for,
            "votes_against": votes_against,
            "passed": passed,
            "sub_sim_result": str(sub_sim_result)[:200] if sub_sim_result else None,
        }
        self.colony.governance_proposals.append(proposal)

        if passed:
            self.colony.governance.append(template)
            for key, value in template["effects"].items():
                if key in self.colony.metrics:
                    self.colony.metrics[key] = max(-1.0, min(1.0, self.colony.metrics[key] + value))
            amendment = f"Year {year}: Adopted {template['name']} (proposed by {proposer.name})"
            self.colony.constitution.append(amendment)
            proposer.memory.append(f"Year {year}: My proposal '{template['name']}' was adopted!")

        return proposal

    def _update_relationships(self, year: int, events: list[dict],
                               actions: list[dict]) -> None:
        """Update colonist relationships based on year's events."""
        action_map = {a["colonist"]: a["action"] for a in actions}
        alive = self.colony.alive_colonists

        for c in alive:
            for other in alive:
                if c.id == other.id:
                    continue
                delta = 0.0
                other_action = action_map.get(other.id, "")

                if other_action in ("mediate", "protect", "farm", "terraform"):
                    delta += 0.05 * c.stats["empathy"]
                if other_action == "sabotage":
                    delta -= 0.15
                if other_action == "hoard":
                    delta -= 0.05 * (1.0 - c.stats["hoarding"])
                if any(e.get("severity", 0) > 0.6 for e in events):
                    delta += 0.03

                current = c.relationships.get(other.id, 0.5)
                drift = (c.stats["empathy"] - 0.5) * 0.02
                new_val = current + delta + drift
                c.relationships[other.id] = max(-1.0, min(1.0, new_val))

    def _check_deaths(self, year: int) -> list[str]:
        """Check for colonist deaths."""
        deaths = []
        r = self.colony.resources

        for c in self.colony.alive_colonists:
            death_chance = 0.0

            if r["food"] <= 0:
                death_chance += 0.08
            if r["water"] <= 0:
                death_chance += 0.10
            if r["oxygen"] <= 0:
                death_chance += 0.12
            if self.colony.metrics["habitat_integrity"] < 0.15:
                death_chance += 0.06
            if self.colony.metrics["danger"] > 0.8:
                death_chance += 0.05
            age = c.age(year)
            if age > 60:
                death_chance += (age - 60) * 0.02
            if death_chance > 0 and c.stats["resolve"] < 0.3:
                death_chance *= 1.5

            if self.rng.random() < death_chance:
                c.alive = False
                c.memory.append(f"Year {year}: I died. The colony remembers.")
                deaths.append(c.id)
                self.colony.dead_colonists.append({
                    "colonist": c.to_dict(),
                    "year_of_death": year,
                    "cause": self._death_cause(r),
                })

        return deaths

    def _death_cause(self, resources: dict) -> str:
        """Determine most likely cause of death."""
        if resources["oxygen"] <= 0:
            return "suffocation"
        if resources["water"] <= 0:
            return "dehydration"
        if resources["food"] <= 0:
            return "starvation"
        if self.colony.metrics["habitat_integrity"] < 0.15:
            return "habitat breach"
        return "accumulated hardship"

    def _check_births(self, year: int) -> list[str]:
        """Check for new colonists (births or arrivals)."""
        births = []
        if year < 5 or len(self.colony.alive_colonists) < 2:
            return births

        morale_bonus = max(0.0, self.colony.metrics["morale"] - 0.4) * 0.1
        if self.rng.random() < 0.12 + morale_bonus:
            parents = self.rng.sample(self.colony.alive_colonists, min(2, len(self.colony.alive_colonists)))
            child_id = f"child-{year}-{self.rng.randint(100,999)}"
            child_template = {
                "id": child_id,
                "name": f"Mars-Born {child_id}",
                "element": self.rng.choice(ELEMENTS),
                "stats": {},
                "skills": {},
                "personality": "(if (< food 30) 'farm 'explore)",
            }
            for stat in parents[0].stats:
                avg = sum(p.stats.get(stat, 0.5) for p in parents) / len(parents)
                child_template["stats"][stat] = max(0.0, min(1.0, avg + self.rng.gauss(0, 0.1)))
            for skill in parents[0].skills:
                avg = sum(p.skills.get(skill, 0.3) for p in parents) / len(parents)
                child_template["skills"][skill] = max(0.0, min(1.0, avg + self.rng.gauss(0, 0.1)))

            child = Colonist(child_template, random.Random(self.seed + year + 5000))
            child.year_born = year
            for other in self.colony.colonists:
                if other.id != child.id:
                    child.relationships[other.id] = 0.6 if other.alive else 0.0
                    if other.alive:
                        other.relationships[child.id] = 0.6

            self.colony.colonists.append(child)
            births.append(child_id)
            self.colony.metrics["morale"] = min(1.0, self.colony.metrics["morale"] + 0.1)

        return births

    def _natural_dynamics(self) -> None:
        """Natural recovery and decay each year."""
        m = self.colony.metrics
        r = self.colony.resources

        r["food"] = min(300.0, r["food"] + 10.0)
        r["water"] = min(300.0, r["water"] + 8.0)
        r["power"] = min(250.0, r["power"] + 12.0)
        r["oxygen"] = min(250.0, r["oxygen"] + 10.0)
        r["supplies"] = min(200.0, r["supplies"] + 3.0)

        m["efficiency"] = min(1.0, m["efficiency"] + 0.005)
        m["morale"] += (0.5 - m["morale"]) * 0.05
        m["danger"] = max(0.0, m["danger"] * 0.7)
        m["unrest"] = max(0.0, m["unrest"] * 0.9)
        m["isolation"] = max(0.0, m["isolation"] * 0.8)
        m["wonder"] = max(0.0, m["wonder"] * 0.85)
        m["habitat_integrity"] = max(0.0, m["habitat_integrity"] - 0.005)

    def _check_meta_insight(self, year: int, lispy: Lispy) -> dict | None:
        """
        Check if a depth-3 sub-sim produces a meta-insight.
        Turtles All the Way Down payoff.
        """
        if year < 15 or self.rng.random() > 0.06:
            return None

        philosophers = [c for c in self.colony.alive_colonists
                        if c.skills.get("mediation", 0) > 0.5 or c.skills.get("coding", 0) > 0.5]
        if not philosophers:
            return None

        thinker = self.rng.choice(philosophers)
        meta_lispy = Lispy(seed=self.seed + year + 77777, sim_depth=0, step_limit=8000)

        meta_program = f"""
        (begin
            (define colony-year {year})
            (define unrest {self.colony.metrics['unrest']:.3f})
            (define morale {self.colony.metrics['morale']:.3f})
            (define pop {len(self.colony.alive_colonists)})
            (define governance-count {len(self.colony.governance)})

            ;; Depth 1: model a governance proposal
            (define proposal-result
                (sub-sim "governance-model"
                    '(begin
                        (define citizens (range 1 11))
                        (define votes (map (lambda (c) (if (> (random) 0.5) 1 0)) citizens))
                        (define approval (/ (reduce + votes 0) (length citizens)))
                        ;; Depth 2: model long-term consequences
                        (define future
                            (sub-sim "consequence-model"
                                '(begin
                                    (define stability 0.5)
                                    (define years (range 1 21))
                                    (define final-stability
                                        (reduce
                                            (lambda (s y)
                                                (+ (* s 0.9) (* (random) 0.2)))
                                            years
                                            stability))
                                    final-stability)))
                        (list approval future))))

            (define insight
                (if (> governance-count 2)
                    "Recursive governance modeling shows: systems that allow sub-simulation of proposals before voting produce more stable outcomes than direct democracy alone"
                    (if (> unrest 0.3)
                        "Sub-simulations reveal: high unrest correlates with governance gaps — colonies need constitutional frameworks before crises, not after"
                        "Observation: recursive self-modeling is the governance primitive — a colony that can simulate its own future makes better present decisions")))

            (list insight proposal-result))
        """

        try:
            result = meta_lispy.eval_string(meta_program)
            max_depth = 0
            for log in meta_lispy.sub_sim_log:
                max_depth = max(max_depth, log.get("depth", 0))
                for child in log.get("child_logs", []):
                    max_depth = max(max_depth, child.get("depth", 0))

            if max_depth >= 2:
                insight_text = result[0] if isinstance(result, list) else str(result)
                thinker.memory.append(f"Year {year}: META-INSIGHT from depth-{max_depth} simulation: {insight_text}")

                amendment = {
                    "year": year,
                    "proposed_by": thinker.id,
                    "depth_reached": max_depth,
                    "insight": str(insight_text)[:500],
                    "source_program": meta_program[:300],
                    "sub_sim_log": meta_lispy.sub_sim_log,
                    "proposed_amendment": (
                        f"Amendment from Mars-100 Year {year}: "
                        f"{insight_text}"
                    ),
                }
                self.colony.amendments.append(amendment)

                for log in meta_lispy.sub_sim_log:
                    self.colony.sub_sim_archive.append({
                        "year": year,
                        "colonist": thinker.id,
                        **log,
                    })

                return amendment

        except LispyError:
            pass

        return None

    def _colony_snapshot(self) -> dict:
        """Current colony state as a dict."""
        return {
            "year": self.colony.year,
            "resources": dict(self.colony.resources),
            "metrics": dict(self.colony.metrics),
            "colonists": [c.to_dict() for c in self.colony.colonists],
            "governance": [{"id": g["id"], "name": g["name"]} for g in self.colony.governance],
            "constitution": self.colony.constitution,
            "population": len(self.colony.alive_colonists),
            "total_born": len(self.colony.colonists),
            "total_dead": len(self.colony.dead_colonists),
        }

    def save_state(self, path: Path | None = None) -> Path:
        """Save simulation state to JSON."""
        out = path or (STATE_DIR / "mars100.json")
        results = self.run()
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(results, indent=2))
        tmp.rename(out)
        return out


# --- CLI ---

def main() -> None:
    """Run Mars-100 from command line."""
    import argparse
    parser = argparse.ArgumentParser(description="Mars-100: Recursive Colony Experiment")
    parser.add_argument("--years", type=int, default=100, help="Years to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    sim = Mars100(seed=args.seed)
    results = sim.run(years=args.years)

    out_path = Path(args.output) if args.output else STATE_DIR / "mars100.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.rename(out_path)

    meta = results["_meta"]
    print(f"Mars-100 complete: {meta['years_simulated']} years, "
          f"{meta['final_population']} survivors, "
          f"{len(results['amendments'])} amendments")

    # Also write frontend copy
    docs_path = DOCS_DIR / "mars-100" / "data.json"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    dtmp = docs_path.with_suffix(".tmp")
    dtmp.write_text(json.dumps(results, separators=(",", ":")))
    dtmp.rename(docs_path)


if __name__ == "__main__":
    main()
