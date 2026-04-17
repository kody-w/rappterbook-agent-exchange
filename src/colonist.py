"""colonist.py -- Individual colonist model for Mars-100 simulation.

Each colonist is a data structure AND a LisPy program. Stats, skills,
relationships, and memory evolve each Martian year. The behavior program
is a pre-parsed LisPy AST evaluated yearly with the colony state.

Python stdlib only.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from src.lispy import parse, Symbol


# Element types (from ghost_profiles.json archetypes)
ELEMENTS = ('fire', 'water', 'earth', 'air')

STAT_NAMES = ('resolve', 'improvisation', 'empathy', 'hoarding', 'faith', 'paranoia')
SKILL_NAMES = ('terraforming', 'hydroponics', 'mediation', 'coding', 'prayer', 'sabotage')


# ----- The 10 founding colonists -----

COLONIST_TEMPLATES: list[dict] = [
    {
        'id': 'ares', 'name': 'Ares', 'element': 'fire',
        'stats': {'resolve': 85, 'improvisation': 40, 'empathy': 30, 'hoarding': 20, 'faith': 50, 'paranoia': 35},
        'skills': {'terraforming': 90, 'hydroponics': 20, 'mediation': 15, 'coding': 30, 'prayer': 10, 'sabotage': 25},
        'behavior': '''(lambda (self env colony year)
            (if (< (get env 'food) (* (get colony 'population) 2))
                (list 'action 'terraform 'priority 'high 'reason "food crisis — expand biomes")
                (if (> (get self 'resolve) 70)
                    (list 'action 'build 'priority 'medium 'reason "steady expansion")
                    (list 'action 'maintain 'priority 'low))))''',
    },
    {
        'id': 'marina', 'name': 'Marina', 'element': 'water',
        'stats': {'resolve': 45, 'improvisation': 55, 'empathy': 90, 'hoarding': 15, 'faith': 40, 'paranoia': 20},
        'skills': {'terraforming': 10, 'hydroponics': 85, 'mediation': 80, 'coding': 15, 'prayer': 30, 'sabotage': 5},
        'behavior': '''(lambda (self env colony year)
            (if (> (get colony 'conflict-level) 50)
                (list 'action 'mediate 'priority 'high 'reason "colony torn apart — must heal")
                (if (< (get env 'water) 500)
                    (list 'action 'farm 'priority 'high 'reason "water scarce — tend hydro")
                    (list 'action 'care 'priority 'medium 'reason "nurture the community"))))''',
    },
    {
        'id': 'petra', 'name': 'Petra', 'element': 'earth',
        'stats': {'resolve': 70, 'improvisation': 25, 'empathy': 35, 'hoarding': 85, 'faith': 30, 'paranoia': 60},
        'skills': {'terraforming': 40, 'hydroponics': 15, 'mediation': 10, 'coding': 50, 'prayer': 5, 'sabotage': 45},
        'behavior': '''(lambda (self env colony year)
            (if (> (get self 'paranoia) 60)
                (list 'action 'stockpile 'priority 'high 'reason "trust no one — hoard resources")
                (list 'action 'build 'priority 'medium 'reason "shore up defenses")))''',
    },
    {
        'id': 'zephyr', 'name': 'Zephyr', 'element': 'air',
        'stats': {'resolve': 30, 'improvisation': 90, 'empathy': 50, 'hoarding': 10, 'faith': 35, 'paranoia': 25},
        'skills': {'terraforming': 30, 'hydroponics': 25, 'mediation': 40, 'coding': 70, 'prayer': 15, 'sabotage': 20},
        'behavior': '''(lambda (self env colony year)
            (if (> year 60)
                (list 'action 'explore-deep 'priority 'high 'reason "the caves call — what lies beneath?")
                (if (> (get self 'improvisation) 80)
                    (list 'action 'innovate 'priority 'high 'reason "hack a better solution")
                    (list 'action 'scout 'priority 'medium 'reason "survey the frontier"))))''',
    },
    {
        'id': 'sol', 'name': 'Sol', 'element': 'fire',
        'stats': {'resolve': 60, 'improvisation': 50, 'empathy': 40, 'hoarding': 30, 'faith': 85, 'paranoia': 20},
        'skills': {'terraforming': 50, 'hydroponics': 30, 'mediation': 35, 'coding': 75, 'prayer': 80, 'sabotage': 10},
        'behavior': '''(lambda (self env colony year)
            (if (> (get env 'radiation) 0.5)
                (list 'action 'shield 'priority 'high 'reason "the sun tests our faith")
                (if (> (get self 'faith) 70)
                    (list 'action 'pray 'priority 'medium 'reason "seek guidance from the cosmos")
                    (list 'action 'engineer 'priority 'medium 'reason "build what we need"))))''',
    },
    {
        'id': 'luna', 'name': 'Luna', 'element': 'water',
        'stats': {'resolve': 55, 'improvisation': 65, 'empathy': 80, 'hoarding': 25, 'faith': 60, 'paranoia': 30},
        'skills': {'terraforming': 20, 'hydroponics': 60, 'mediation': 75, 'coding': 40, 'prayer': 55, 'sabotage': 15},
        'behavior': '''(lambda (self env colony year)
            (if (< (get colony 'morale) 40)
                (list 'action 'inspire 'priority 'high 'reason "spirits low — rally the colony")
                (if (> year 30)
                    (list 'action 'teach 'priority 'medium 'reason "pass knowledge to the young")
                    (list 'action 'heal 'priority 'medium 'reason "tend the wounded and weary"))))''',
    },
    {
        'id': 'ferrum', 'name': 'Ferrum', 'element': 'earth',
        'stats': {'resolve': 90, 'improvisation': 20, 'empathy': 25, 'hoarding': 50, 'faith': 40, 'paranoia': 45},
        'skills': {'terraforming': 70, 'hydroponics': 10, 'mediation': 5, 'coding': 20, 'prayer': 15, 'sabotage': 60},
        'behavior': '''(lambda (self env colony year)
            (if (< (get env 'materials) 200)
                (list 'action 'mine 'priority 'high 'reason "ore reserves critical")
                (if (> (get colony 'conflict-level) 70)
                    (list 'action 'enforce 'priority 'high 'reason "order must be maintained — by force if needed")
                    (list 'action 'mine 'priority 'medium 'reason "steady extraction"))))''',
    },
    {
        'id': 'aether', 'name': 'Aether', 'element': 'air',
        'stats': {'resolve': 35, 'improvisation': 85, 'empathy': 60, 'hoarding': 20, 'faith': 25, 'paranoia': 70},
        'skills': {'terraforming': 15, 'hydroponics': 20, 'mediation': 30, 'coding': 90, 'prayer': 10, 'sabotage': 35},
        'behavior': '''(lambda (self env colony year)
            (if (> (get self 'paranoia) 60)
                (list 'action 'monitor 'priority 'high 'reason "something is off — watch the systems")
                (if (> year 50)
                    (list 'action 'sub-simulate 'priority 'high 'reason "model the future before it arrives")
                    (list 'action 'code 'priority 'medium 'reason "improve colony systems"))))''',
    },
    {
        'id': 'ignis', 'name': 'Ignis', 'element': 'fire',
        'stats': {'resolve': 75, 'improvisation': 45, 'empathy': 20, 'hoarding': 55, 'faith': 30, 'paranoia': 80},
        'skills': {'terraforming': 35, 'hydroponics': 10, 'mediation': 20, 'coding': 45, 'prayer': 25, 'sabotage': 75},
        'behavior': '''(lambda (self env colony year)
            (if (> (get self 'paranoia) 70)
                (if (> (get colony 'conflict-level) 40)
                    (list 'action 'sabotage 'priority 'high 'reason "disrupt the power structure before it calcifies")
                    (list 'action 'spy 'priority 'medium 'reason "watch — everyone has secrets"))
                (list 'action 'lead 'priority 'medium 'reason "someone must take charge")))''',
    },
    {
        'id': 'terra', 'name': 'Terra', 'element': 'earth',
        'stats': {'resolve': 65, 'improvisation': 40, 'empathy': 70, 'hoarding': 35, 'faith': 80, 'paranoia': 15},
        'skills': {'terraforming': 55, 'hydroponics': 70, 'mediation': 50, 'coding': 10, 'prayer': 75, 'sabotage': 5},
        'behavior': '''(lambda (self env colony year)
            (if (< (get env 'food) (* (get colony 'population) 3))
                (list 'action 'cultivate 'priority 'high 'reason "the soil must provide")
                (if (> (get self 'faith) 70)
                    (list 'action 'pray 'priority 'low 'reason "give thanks for abundance")
                    (list 'action 'cultivate 'priority 'medium 'reason "grow more — always more"))))''',
    },
]


def create_colonists(seed: int = 42) -> list[dict]:
    """Create the 10 founding colonists with seeded relationship noise."""
    rng = random.Random(seed)
    colonists = []
    ids = [t['id'] for t in COLONIST_TEMPLATES]

    for template in COLONIST_TEMPLATES:
        col = {
            'id': template['id'],
            'name': template['name'],
            'element': template['element'],
            'stats': dict(template['stats']),
            'skills': dict(template['skills']),
            'relationships': {},
            'memory': [],
            'alive': True,
            'year_arrived': 0,
            'year_died': None,
            'governance_role': None,
            'behavior_source': template['behavior'],
            'behavior_ast': parse(template['behavior']),
        }
        # Initialize relationships with noise
        for other_id in ids:
            if other_id != template['id']:
                # Element affinity: same element = +20 base, opposed = -10
                other_elem = next(t['element'] for t in COLONIST_TEMPLATES if t['id'] == other_id)
                base = 20 if other_elem == template['element'] else -10
                noise = rng.randint(-30, 30)
                col['relationships'][other_id] = max(-100, min(100, base + noise))
        colonists.append(col)
    return colonists


def colonist_to_lispy(colonist: dict) -> list:
    """Convert a colonist dict to a LisPy assoc-list for use in eval context."""
    stats_pairs = []
    for k, v in colonist['stats'].items():
        stats_pairs.append([Symbol(k), v])
    skills_pairs = []
    for k, v in colonist['skills'].items():
        skills_pairs.append([Symbol(k), v])
    return [
        [Symbol('id'), colonist['id']],
        [Symbol('name'), colonist['name']],
        [Symbol('element'), colonist['element']],
        [Symbol('alive'), colonist['alive']],
        [Symbol('resolve'), colonist['stats']['resolve']],
        [Symbol('improvisation'), colonist['stats']['improvisation']],
        [Symbol('empathy'), colonist['stats']['empathy']],
        [Symbol('hoarding'), colonist['stats']['hoarding']],
        [Symbol('faith'), colonist['stats']['faith']],
        [Symbol('paranoia'), colonist['stats']['paranoia']],
        [Symbol('terraforming'), colonist['skills']['terraforming']],
        [Symbol('hydroponics'), colonist['skills']['hydroponics']],
        [Symbol('mediation'), colonist['skills']['mediation']],
        [Symbol('coding'), colonist['skills']['coding']],
        [Symbol('prayer'), colonist['skills']['prayer']],
        [Symbol('sabotage'), colonist['skills']['sabotage']],
        [Symbol('governance-role'), colonist['governance_role']],
    ]


def clamp_stat(value: int | float) -> int:
    """Clamp a stat value to 0..100."""
    return max(0, min(100, int(round(value))))


def evolve_stats(colonist: dict, year: int, event_severity: float, rng: random.Random) -> None:
    """Mutate colonist stats based on year and stress. In-place."""
    if not colonist['alive']:
        return
    stress = event_severity * 10
    for stat in STAT_NAMES:
        drift = rng.gauss(0, 2)
        # Paranoia increases with age and stress
        if stat == 'paranoia':
            drift += stress * 0.3 + (year / 100) * 2
        # Faith drifts with crisis
        elif stat == 'faith':
            drift += stress * 0.2 * (1 if rng.random() > 0.5 else -1)
        # Empathy can erode under sustained stress
        elif stat == 'empathy':
            drift -= stress * 0.1
        colonist['stats'][stat] = clamp_stat(colonist['stats'][stat] + drift)


def evolve_relationships(colonists: list[dict], year: int, rng: random.Random) -> None:
    """Drift relationships based on proximity, shared crises, governance."""
    living = [c for c in colonists if c['alive']]
    for col in living:
        for other in living:
            if col['id'] == other['id']:
                continue
            affinity = col['relationships'].get(other['id'], 0)
            # Shared element = bond strengthens
            if col['element'] == other['element']:
                affinity += rng.randint(0, 3)
            # High empathy people bond more easily
            if col['stats']['empathy'] > 60:
                affinity += rng.randint(0, 2)
            # High paranoia erodes trust
            if col['stats']['paranoia'] > 70:
                affinity -= rng.randint(0, 3)
            # Governance tension
            if (col['governance_role'] == 'leader' and
                    other['governance_role'] == 'opposition'):
                affinity -= rng.randint(1, 5)
            # Random drift
            affinity += rng.gauss(0, 2)
            col['relationships'][other['id']] = max(-100, min(100, int(round(affinity))))


def serialize_colonist(colonist: dict) -> dict:
    """Convert colonist to JSON-safe dict (drop AST)."""
    result = dict(colonist)
    result.pop('behavior_ast', None)
    return result
