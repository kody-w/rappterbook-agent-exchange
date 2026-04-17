"""mars100.py -- Mars-100 Recursive Colony Experiment.

100 Martian years. 10 agent-colonists. LisPy sub-simulations up to
depth 3. Emergent governance. Turtles All the Way Down.

Usage:
    python src/mars100.py                    # Run full 100-year sim
    python src/mars100.py --years 20         # Run 20 years
    python src/mars100.py --seed 99          # Different seed
    python src/mars100.py --output-dir /tmp  # Custom output

Python stdlib only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# Fix imports for running from repo root
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lispy import (
    Env, Budget, Symbol, Procedure,
    default_env, evaluate, parse, run, to_sexp,
    LispyError, BudgetExhausted,
)
from src.colonist import (
    create_colonists, colonist_to_lispy, evolve_stats,
    evolve_relationships, serialize_colonist, clamp_stat,
    STAT_NAMES, SKILL_NAMES,
)
from src.mars100_events import generate_event
from src.mars100_gov import (
    create_proposal, cast_vote, tally_votes,
    detect_pattern, assign_governance_roles,
    check_amendment_worthy, PATTERNS,
)
from src.sub_sim import run_sub_sim, build_governance_sim, build_philosophy_sim


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Colony resource starting values
INITIAL_RESOURCES = {
    'food': 500,       # kg
    'water': 1000,     # liters
    'power': 800,      # kWh capacity
    'oxygen': 600,     # person-days
    'materials': 400,  # kg construction materials
    'morale': 70,      # 0-100
}

# Per-person yearly consumption
YEARLY_CONSUMPTION = {
    'food': 50,    # kg/person/year
    'water': 80,   # liters/person/year (recycled)
    'power': 40,   # kWh/person/year
    'oxygen': 30,  # person-days consumed per year
}

# Per-person yearly production (base rate)
YEARLY_PRODUCTION = {
    'food': 35,
    'water': 60,
    'power': 30,
    'oxygen': 25,
    'materials': 15,
}


class Mars100Simulation:
    """The Mars-100 recursive colony experiment."""

    def __init__(self, seed: int = 42, years: int = 100) -> None:
        self.seed = seed
        self.total_years = years
        self.rng = random.Random(seed)
        self.colonists = create_colonists(seed)
        self.resources: dict[str, float] = dict(INITIAL_RESOURCES)
        self.year = 0
        self.proposals: list[dict] = []
        self.governance_roles: dict[str, str | None] = {c['id']: None for c in self.colonists}
        self.pattern_history: list[tuple[int, str]] = []
        self.sub_sim_log: list[dict] = []
        self.year_records: list[dict] = []
        self.dead_colonists: list[dict] = []
        self.amendment: str | None = None

    @property
    def living_colonists(self) -> list[dict]:
        return [c for c in self.colonists if c['alive']]

    @property
    def population(self) -> int:
        return len(self.living_colonists)

    def run(self) -> dict:
        """Run the full simulation. Returns complete history."""
        for year in range(1, self.total_years + 1):
            self.year = year
            record = self._tick_year(year)
            self.year_records.append(record)

            # Check for colony death
            if self.population == 0:
                break

            # Check for amendment-worthy insight
            if year >= 20 and self.amendment is None:
                self.amendment = check_amendment_worthy(
                    self.pattern_history,
                    [s for s in self.sub_sim_log if s.get('depth', 0) >= 3],
                )

        return self._compile_results()

    def _tick_year(self, year: int) -> dict:
        """Advance the colony by one Martian year."""
        # 1. Environmental event
        event = generate_event(year, self.rng)

        # 2. Apply event effects to resources
        for resource, delta in event.get('effects', {}).items():
            if resource in self.resources:
                self.resources[resource] = max(0, self.resources[resource] + delta)

        # 3. Resource production and consumption
        self._process_resources()

        # 4. Colonist decisions (LisPy evaluation)
        actions = self._colonist_decisions(year, event)

        # 5. Resolve actions (resource changes, proposals)
        self._resolve_actions(actions, year, event)

        # 6. Governance phase
        governance_pattern = self._governance_phase(year)

        # 7. Evolve colonist stats and relationships
        for col in self.colonists:
            evolve_stats(col, year, event.get('severity', 0), self.rng)
        evolve_relationships(self.colonists, year, self.rng)

        # 8. Check for deaths
        deaths = self._check_deaths(year)

        # 9. Handle births (scripted + organic)
        births = self._check_births(year, event)

        # 10. Sub-simulation phase (governance modeling, philosophy)
        sub_sims = self._sub_sim_phase(year, event)

        # 11. Record the year
        record = {
            'year': year,
            'event': event,
            'population': self.population,
            'resources': dict(self.resources),
            'actions': actions,
            'governance_pattern': governance_pattern,
            'deaths': deaths,
            'births': births,
            'sub_sims': sub_sims,
            'colonist_snapshot': [
                {
                    'id': c['id'], 'name': c['name'], 'alive': c['alive'],
                    'stats': dict(c['stats']), 'governance_role': c['governance_role'],
                }
                for c in self.colonists
            ],
        }

        # Add diary entries from key colonists
        record['diaries'] = self._write_diaries(year, event, actions, governance_pattern)

        return record

    def _process_resources(self) -> None:
        """Consume and produce resources based on population."""
        pop = self.population
        for resource, rate in YEARLY_CONSUMPTION.items():
            if resource in self.resources:
                self.resources[resource] -= rate * pop
                self.resources[resource] = max(0, self.resources[resource])
        for resource, rate in YEARLY_PRODUCTION.items():
            if resource in self.resources:
                self.resources[resource] += rate * pop

    def _colonist_decisions(self, year: int, event: dict) -> list[dict]:
        """Evaluate each colonist's LisPy behavior program."""
        actions = []
        env_data = {
            'food': self.resources['food'],
            'water': self.resources['water'],
            'power': self.resources['power'],
            'materials': self.resources['materials'],
            'radiation': event.get('severity', 0),
        }
        colony_data = {
            'population': self.population,
            'morale': self.resources.get('morale', 50),
            'conflict-level': self._conflict_level(),
        }

        for col in self.living_colonists:
            action = self._eval_colonist(col, env_data, colony_data, year)
            actions.append(action)
            col['memory'].append({
                'year': year,
                'event': event['type'],
                'action': action.get('action', 'idle'),
                'reason': action.get('reason', ''),
            })
            # Cap memory to last 50 entries
            col['memory'] = col['memory'][-50:]

        return actions

    def _eval_colonist(
        self,
        colonist: dict,
        env_data: dict,
        colony_data: dict,
        year: int,
    ) -> dict:
        """Evaluate a single colonist's behavior in LisPy."""
        budget = Budget(max_steps=5000, max_depth=3)
        env = default_env()

        # Build context as LisPy assoc lists
        self_data = colonist_to_lispy(colonist)
        env_assoc = [[Symbol(str(k)), v] for k, v in env_data.items()]
        colony_assoc = [[Symbol(str(k)), v] for k, v in colony_data.items()]

        try:
            # Evaluate the behavior lambda
            func = evaluate(colonist['behavior_ast'], env, budget)
            if isinstance(func, Procedure):
                local_env = Env(
                    dict(zip(func.params, [self_data, env_assoc, colony_assoc, year])),
                    parent=func.closure_env,
                )
                result = evaluate(func.body, local_env, budget)
                return self._parse_action(result, colonist['id'])
        except (LispyError, BudgetExhausted):
            pass

        # Fallback: default action based on dominant skill
        best_skill = max(colonist['skills'].items(), key=lambda x: x[1])
        return {
            'colonist_id': colonist['id'],
            'action': best_skill[0],
            'priority': 'medium',
            'reason': 'instinct',
        }

    def _parse_action(self, result, colonist_id: str) -> dict:
        """Parse a LisPy action result into a dict."""
        action = {'colonist_id': colonist_id, 'action': 'idle', 'priority': 'low', 'reason': ''}
        if isinstance(result, list):
            for i in range(0, len(result) - 1, 2):
                key = str(result[i]) if isinstance(result[i], Symbol) else str(result[i])
                val = result[i + 1]
                if isinstance(val, Symbol):
                    val = str(val)
                action[key] = val
        action['colonist_id'] = colonist_id
        return action

    def _resolve_actions(self, actions: list[dict], year: int, event: dict) -> None:
        """Apply colonist actions to colony state."""
        for action in actions:
            act = action.get('action', 'idle')
            col = next((c for c in self.colonists if c['id'] == action['colonist_id']), None)
            if col is None or not col['alive']:
                continue

            if act in ('terraform', 'cultivate', 'farm'):
                skill = col['skills'].get('terraforming', 30) + col['skills'].get('hydroponics', 20)
                self.resources['food'] += skill * 0.3
            elif act == 'mine':
                self.resources['materials'] += col['skills'].get('terraforming', 30) * 0.4
            elif act in ('build', 'engineer'):
                self.resources['power'] += col['skills'].get('coding', 20) * 0.2
                self.resources['materials'] -= 10
            elif act in ('mediate', 'inspire', 'care', 'heal', 'teach'):
                self.resources['morale'] = min(100, self.resources.get('morale', 50) + col['stats']['empathy'] * 0.1)
            elif act in ('pray', 'reflect'):
                self.resources['morale'] = min(100, self.resources.get('morale', 50) + col['stats']['faith'] * 0.05)
            elif act == 'stockpile':
                self.resources['food'] += col['stats']['hoarding'] * 0.1
            elif act in ('spy', 'monitor'):
                col['stats']['paranoia'] = clamp_stat(col['stats']['paranoia'] + 2)
            elif act == 'sabotage':
                # Sabotage hurts resources but is not always malicious (may target corrupt leaders)
                target_resource = self.rng.choice(['food', 'power', 'materials'])
                self.resources[target_resource] = max(0, self.resources[target_resource] - 20)
                self.resources['morale'] = max(0, self.resources.get('morale', 50) - 5)
            elif act == 'sub-simulate':
                pass  # Handled in sub_sim_phase
            elif act in ('innovate', 'code'):
                self.resources['power'] += col['skills'].get('coding', 30) * 0.3
            elif act in ('scout', 'explore-deep'):
                self.resources['materials'] += self.rng.randint(5, 25)
            elif act == 'shield':
                self.resources['power'] -= 15

        # Clamp resources
        for key in self.resources:
            self.resources[key] = max(0.0, self.resources[key])

    def _governance_phase(self, year: int) -> str:
        """Run governance: proposals, votes, role assignment, pattern detection."""
        living = self.living_colonists
        if not living:
            return 'anarchy'

        # Generate proposals from high-governance colonists
        if year % 5 == 0 or self.resources.get('morale', 50) < 30:
            self._generate_proposals(year)

        # Assign roles
        self.governance_roles = assign_governance_roles(
            self.colonists, self.proposals, year, self.rng,
        )
        for col in self.colonists:
            col['governance_role'] = self.governance_roles.get(col['id'])

        # Detect pattern
        pattern = detect_pattern(self.proposals, self.colonists, self.governance_roles)
        self.pattern_history.append((year, pattern))
        return pattern

    def _generate_proposals(self, year: int) -> None:
        """Living colonists with high resolve or faith propose governance changes."""
        for col in self.living_colonists:
            chance = (col['stats']['resolve'] + col['stats']['faith']) / 200
            if self.rng.random() < chance * 0.3:
                title = self._make_proposal_title(col, year)
                proposal = create_proposal(year, col['id'], title, f"Proposed by {col['name']} in year {year}")

                # Vote
                for voter in self.living_colonists:
                    if voter['id'] == col['id']:
                        cast_vote(proposal, voter['id'], 'yes')
                    else:
                        affinity = voter['relationships'].get(col['id'], 0)
                        if affinity > 20:
                            cast_vote(proposal, voter['id'], 'yes')
                        elif affinity < -20:
                            cast_vote(proposal, voter['id'], 'no')
                        else:
                            vote = self.rng.choice(['yes', 'no', 'abstain'])
                            cast_vote(proposal, voter['id'], vote)

                tally_votes(proposal, self.living_colonists)
                self.proposals.append(proposal)

    def _make_proposal_title(self, colonist: dict, year: int) -> str:
        """Generate a proposal title based on colonist personality."""
        if colonist['stats']['faith'] > 60:
            titles = ["Establish a meditation commons", "Create a spiritual council",
                      "Declare a day of reflection", "Build a temple to the red sky"]
        elif colonist['stats']['paranoia'] > 60:
            titles = ["Mandatory resource audits", "Surveillance expansion",
                      "Restrict EVA permissions", "Emergency authority act"]
        elif colonist['stats']['empathy'] > 60:
            titles = ["Universal care mandate", "Conflict resolution protocol",
                      "Community wellness check", "Share all resources equally"]
        else:
            titles = ["Infrastructure expansion", "Mining quota increase",
                      "Power grid upgrade", "Habitat extension plan"]
        return self.rng.choice(titles)

    def _conflict_level(self) -> int:
        """Calculate colony-wide conflict level from relationships."""
        living = self.living_colonists
        if len(living) < 2:
            return 0
        total = 0
        count = 0
        for col in living:
            for other_id, affinity in col['relationships'].items():
                if any(c['id'] == other_id and c['alive'] for c in self.colonists):
                    total += affinity
                    count += 1
        if count == 0:
            return 0
        avg_affinity = total / count
        # Convert: high affinity = low conflict, low affinity = high conflict
        return max(0, min(100, int(50 - avg_affinity)))

    def _check_deaths(self, year: int) -> list[dict]:
        """Check for colonist deaths."""
        deaths = []
        for col in self.living_colonists:
            death_chance = 0.0
            # Starvation (only lethal at extreme levels)
            if self.resources['food'] < self.population * 10:
                death_chance += 0.05
            # Old age (colonists age — after 75 years on Mars, risk increases)
            mars_age = year - col['year_arrived']
            if mars_age > 75:
                death_chance += (mars_age - 75) * 0.015
            # Extreme paranoia can lead to self-exile (death equivalent)
            if col['stats']['paranoia'] > 98:
                death_chance += 0.1
            # Very low morale colony
            if self.resources.get('morale', 50) < 5:
                death_chance += 0.03

            if self.rng.random() < death_chance:
                col['alive'] = False
                col['year_died'] = year
                death_record = {
                    'colonist_id': col['id'],
                    'name': col['name'],
                    'year': year,
                    'mars_age': mars_age,
                    'cause': self._death_cause(col, year),
                }
                deaths.append(death_record)
                self.dead_colonists.append(death_record)

                # Add to all living colonists' memory
                for other in self.living_colonists:
                    other['memory'].append({
                        'year': year,
                        'event': 'death',
                        'action': 'mourn',
                        'reason': f"{col['name']} has died — {death_record['cause']}",
                    })
                    other['memory'] = other['memory'][-50:]
                # Morale hit
                self.resources['morale'] = max(0, self.resources.get('morale', 50) - 10)

        return deaths

    def _death_cause(self, colonist: dict, year: int) -> str:
        """Determine cause of death."""
        if self.resources['food'] < self.population * 10:
            return 'starvation'
        if colonist['stats']['paranoia'] > 98:
            return 'self-exile into the wastes'
        mars_age = year - colonist['year_arrived']
        if mars_age > 75:
            return 'old age'
        if self.resources.get('morale', 50) < 5:
            return 'despair'
        return 'accident'

    def _check_births(self, year: int, event: dict) -> list[dict]:
        """Check for new colonists (births or arrivals)."""
        births = []

        # Scripted birth at year 12
        if event.get('type') == 'birth' and event.get('scripted'):
            new_col = self._create_child('dawn', 'Dawn', year)
            self.colonists.append(new_col)
            births.append({'id': 'dawn', 'name': 'Dawn', 'year': year, 'type': 'birth'})
            return births

        # Organic births: small chance if population > 3 and morale > 30
        if (self.population >= 3 and self.resources.get('morale', 50) > 30
                and year > 10 and self.rng.random() < 0.12):
            child_id = f"child-y{year}"
            name = self.rng.choice([
                'Dust', 'Phobos', 'Olympia', 'Hellas', 'Meridian',
                'Valles', 'Elysium', 'Tharsis', 'Cydonia', 'Arcadia',
            ])
            new_col = self._create_child(child_id, name, year)
            self.colonists.append(new_col)
            births.append({'id': child_id, 'name': name, 'year': year, 'type': 'birth'})

        return births

    def _create_child(self, child_id: str, name: str, year: int) -> dict:
        """Create a Mars-born colonist with inherited traits."""
        living = self.living_colonists
        if not living:
            parent1 = parent2 = None
        elif len(living) == 1:
            parent1 = parent2 = living[0]
        else:
            parent1, parent2 = self.rng.sample(living, 2)

        element = self.rng.choice(['fire', 'water', 'earth', 'air'])
        stats = {}
        skills = {}
        for stat in STAT_NAMES:
            if parent1 and parent2:
                avg = (parent1['stats'][stat] + parent2['stats'][stat]) / 2
                stats[stat] = clamp_stat(avg + self.rng.gauss(0, 10))
            else:
                stats[stat] = self.rng.randint(20, 60)
        for skill in SKILL_NAMES:
            if parent1 and parent2:
                avg = (parent1['skills'][skill] + parent2['skills'][skill]) / 2
                skills[skill] = clamp_stat(avg + self.rng.gauss(0, 15))
            else:
                skills[skill] = self.rng.randint(10, 50)

        # Simple behavior: follow strongest skill
        best_skill = max(skills.items(), key=lambda x: x[1])[0]
        behavior = f'''(lambda (self env colony year)
            (list 'action '{best_skill} 'priority 'medium 'reason "born to this"))'''

        return {
            'id': child_id,
            'name': name,
            'element': element,
            'stats': stats,
            'skills': skills,
            'relationships': {c['id']: self.rng.randint(-10, 30) for c in self.colonists},
            'memory': [{'year': year, 'event': 'birth', 'action': 'born', 'reason': 'First breath of Martian air'}],
            'alive': True,
            'year_arrived': year,
            'year_died': None,
            'governance_role': None,
            'behavior_source': behavior,
            'behavior_ast': parse(behavior),
        }

    def _sub_sim_phase(self, year: int, event: dict) -> list[dict]:
        """Run sub-simulations for colonists who chose to model scenarios."""
        sub_sims = []

        # Any living colonist with coding > 60 can run sub-sims from year 50+
        coder = next(
            (c for c in self.living_colonists if c['skills'].get('coding', 0) > 60),
            None,
        )
        if coder and year >= 50 and year % 5 == 0:
            code = build_governance_sim(
                population=self.population,
                food=self.resources['food'],
                morale=self.resources.get('morale', 50),
                year=year,
                years_forward=10,
            )
            budget = Budget(max_steps=8000, max_depth=3)
            result = run_sub_sim(code, default_env(), budget, context={'colonist': coder['id'], 'year': year})
            result['spawned_by'] = coder['id']
            result['year'] = year
            sub_sims.append(result)
            self.sub_sim_log.append(result)

        # At year 67, nested philosophical sub-sim (if colony is alive)
        if year == 67 and self.living_colonists:
            code = build_philosophy_sim()
            budget = Budget(max_steps=5000, max_depth=3)
            env = default_env()
            # First layer
            result1 = run_sub_sim(code, env, budget, context={'colonist': 'aether', 'year': year})
            result1['spawned_by'] = 'aether'
            result1['year'] = year
            result1['label'] = 'depth-1-philosophy'
            sub_sims.append(result1)
            self.sub_sim_log.append(result1)

            # Second layer (depth 2)
            result2 = run_sub_sim(code, env, budget, context={'colonist': 'aether', 'year': year})
            result2['spawned_by'] = 'aether-inner'
            result2['year'] = year
            result2['label'] = 'depth-2-philosophy'
            sub_sims.append(result2)
            self.sub_sim_log.append(result2)

            # Third layer (depth 3) - triggers the insight
            result3 = run_sub_sim(code, env, budget, context={'colonist': 'aether', 'year': year})
            result3['spawned_by'] = 'aether-deep'
            result3['year'] = year
            result3['label'] = 'depth-3-philosophy'
            result3['depth'] = 3
            if result3['success'] and isinstance(result3.get('result'), list):
                result3['insight'] = 'The governance that works is the one that knows it is temporary.'
            sub_sims.append(result3)
            self.sub_sim_log.append(result3)

        # At year 82, the deepest recursion (if colony survives)
        if year == 82 and self.living_colonists:
            code = """
            (let ((depth sim-depth))
              (list 'insight
                    "All governance is a sub-simulation. The real constitution is the willingness to rewrite it."
                    'depth depth
                    'meta "Depth 3 reached. The model has modeled its own modeling."))
            """
            budget = Budget(max_steps=5000, max_depth=3)
            # Stack 3 nested sub-sims
            for i in range(3):
                ctx = {'colonist': 'aether', 'year': year, 'layer': i + 1}
                result = run_sub_sim(code, default_env(), budget, context=ctx)
                result['spawned_by'] = f'aether-layer-{i+1}'
                result['year'] = year
                result['depth'] = budget.current_depth + 1  # already exited
                if i == 2:
                    result['depth'] = 3
                    result['insight'] = (
                        "All governance is a sub-simulation. "
                        "The real constitution is the willingness to rewrite it."
                    )
                sub_sims.append(result)
                self.sub_sim_log.append(result)

        return sub_sims

    def _write_diaries(
        self,
        year: int,
        event: dict,
        actions: list[dict],
        governance: str,
    ) -> list[dict]:
        """Generate diary entries from 3 key colonists each year."""
        diaries = []
        # Pick diarists: most empathetic, most paranoid, and leader
        living = self.living_colonists
        if not living:
            return diaries

        candidates = sorted(living, key=lambda c: c['stats']['empathy'], reverse=True)
        diarists = [candidates[0]] if candidates else []

        paranoid = sorted(living, key=lambda c: c['stats']['paranoia'], reverse=True)
        if paranoid and paranoid[0] not in diarists:
            diarists.append(paranoid[0])

        leader = next((c for c in living if c['governance_role'] == 'leader'), None)
        if leader and leader not in diarists:
            diarists.append(leader)

        # Fill up to 3
        for col in living:
            if len(diarists) >= 3:
                break
            if col not in diarists:
                diarists.append(col)

        for col in diarists[:3]:
            action = next((a for a in actions if a.get('colonist_id') == col['id']), {})
            diary = {
                'colonist_id': col['id'],
                'name': col['name'],
                'year': year,
                'entry': self._generate_diary(col, year, event, action, governance),
            }
            diaries.append(diary)

        return diaries

    def _generate_diary(
        self,
        colonist: dict,
        year: int,
        event: dict,
        action: dict,
        governance: str,
    ) -> str:
        """Generate a diary entry reflecting colonist personality."""
        name = colonist['name']
        act = action.get('action', 'idle')
        reason = action.get('reason', '')
        event_desc = event.get('description', event.get('type', 'quiet year'))

        # Personality-colored framing
        if colonist['stats']['paranoia'] > 60:
            mood = "Something feels wrong."
        elif colonist['stats']['faith'] > 60:
            mood = "I trust the cosmos has a plan."
        elif colonist['stats']['empathy'] > 60:
            mood = "I feel the weight of everyone's fear."
        else:
            mood = "Another year on Mars."

        entry = (
            f"Year {year} — {name}'s Log\n"
            f"{mood} {event_desc}\n"
            f"I chose to {act}. {reason}\n"
            f"Governance: {governance}. Population: {self.population}. "
            f"Food: {self.resources['food']:.0f}kg. Morale: {self.resources.get('morale', 50):.0f}."
        )

        # Special reflections at key years
        if year == 67 and colonist['id'] == 'aether':
            entry += "\n\n...The signal. It's us. We are the simulation."
        if year == 82:
            entry += "\n\nThe sub-simulation returned. Its message: all governance is temporary."
        if year == 100:
            entry += f"\n\n100 years. We survived. But did we become what we set out to be?"

        return entry

    def _compile_results(self) -> dict:
        """Compile complete simulation results."""
        # Governance pattern summary
        pattern_counts = {}
        for _, pattern in self.pattern_history:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        dominant_pattern = max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else 'anarchy'

        return {
            '_meta': {
                'engine': 'mars-100',
                'version': '1.0',
                'seed': self.seed,
                'years': self.total_years,
                'years_completed': self.year,
                'generated': now_iso(),
            },
            'summary': {
                'final_population': self.population,
                'total_deaths': len(self.dead_colonists),
                'total_births': len([c for c in self.colonists if c['year_arrived'] > 0]),
                'total_proposals': len(self.proposals),
                'total_sub_sims': len(self.sub_sim_log),
                'dominant_governance': dominant_pattern,
                'governance_description': PATTERNS.get(dominant_pattern, ''),
                'pattern_history': self.pattern_history,
                'amendment': self.amendment,
            },
            'years': self.year_records,
            'colonists': [serialize_colonist(c) for c in self.colonists],
            'dead': self.dead_colonists,
            'proposals': self.proposals,
            'sub_sim_log': self.sub_sim_log,
            'resources_final': dict(self.resources),
        }


def write_results(results: dict, output_dir: Path) -> None:
    """Write simulation results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full state
    state_path = output_dir / 'state.json'
    with open(state_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results['summary'], f, indent=2, default=str)

    # Per-colonist soul files
    colonist_dir = output_dir / 'colonists'
    colonist_dir.mkdir(exist_ok=True)
    for col in results['colonists']:
        col_path = colonist_dir / f"{col['id']}.json"
        with open(col_path, 'w') as f:
            json.dump(col, f, indent=2, default=str)

    # Sub-sim log
    if results['sub_sim_log']:
        subsim_dir = output_dir / 'sub-sims'
        subsim_dir.mkdir(exist_ok=True)
        for i, entry in enumerate(results['sub_sim_log']):
            path = subsim_dir / f"sub-sim-{i:03d}.json"
            with open(path, 'w') as f:
                json.dump(entry, f, indent=2, default=str)

    print(f"Mars-100 results written to {output_dir}")
    print(f"  Years completed: {results['_meta']['years_completed']}")
    print(f"  Final population: {results['summary']['final_population']}")
    print(f"  Dominant governance: {results['summary']['dominant_governance']}")
    if results['summary']['amendment']:
        print(f"  AMENDMENT PROPOSED: {results['summary']['amendment'][:80]}...")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Mars-100 Recursive Colony Experiment')
    parser.add_argument('--years', type=int, default=100, help='Years to simulate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: docs/mars-100)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / 'docs' / 'mars-100'

    sim = Mars100Simulation(seed=args.seed, years=args.years)
    results = sim.run()
    write_results(results, output_dir)


if __name__ == '__main__':
    main()
