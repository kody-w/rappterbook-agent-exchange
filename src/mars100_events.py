"""mars100_events.py -- Environmental event generator for Mars-100 simulation.

Each Martian year, the colony faces an event. Events range from survivable
dust storms to paradigm-shifting alien signals. Severity escalates with
time — early years are survival, late years are philosophical.

Python stdlib only.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


EVENT_TYPES = [
    {'type': 'dust_storm',        'weight': 25, 'min_severity': 0.1, 'max_severity': 0.7},
    {'type': 'equipment_failure', 'weight': 20, 'min_severity': 0.1, 'max_severity': 0.6},
    {'type': 'resource_strike',   'weight': 15, 'min_severity': 0.0, 'max_severity': 0.3},
    {'type': 'solar_flare',       'weight': 10, 'min_severity': 0.2, 'max_severity': 0.8},
    {'type': 'water_discovery',   'weight': 8,  'min_severity': 0.0, 'max_severity': 0.1},
    {'type': 'cave_discovery',    'weight': 5,  'min_severity': 0.0, 'max_severity': 0.0},
    {'type': 'plague',            'weight': 5,  'min_severity': 0.3, 'max_severity': 0.9},
    {'type': 'meteor_impact',     'weight': 3,  'min_severity': 0.4, 'max_severity': 1.0},
    {'type': 'earth_contact',     'weight': 5,  'min_severity': 0.0, 'max_severity': 0.0},
    {'type': 'alien_signal',      'weight': 2,  'min_severity': 0.0, 'max_severity': 0.0},
    {'type': 'birth',             'weight': 7,  'min_severity': 0.0, 'max_severity': 0.0},
]

# Scripted key events (year → event override)
SCRIPTED_EVENTS: dict[int, dict] = {
    5: {
        'type': 'dust_storm', 'severity': 0.8,
        'description': 'The Great Dust Storm — a planet-wide tempest buries solar arrays and shreds antennas. '
                        'Colony reserves drop to 30-day emergency levels.',
        'effects': {'food': -40, 'power': -60, 'morale': -25},
    },
    12: {
        'type': 'birth', 'severity': 0.0,
        'description': 'First child born on Mars — a daughter named Dawn. The colony celebrates. '
                        'For the first time, Mars feels like home instead of a mission.',
        'effects': {'morale': 30, 'food': -5},
    },
    25: {
        'type': 'earth_contact', 'severity': 0.5,
        'description': 'Earth goes silent. No transmissions for 6 months. '
                        'Is it a conjunction? Equipment failure? Or something worse? '
                        'The colony must decide: wait, or assume independence.',
        'effects': {'morale': -30},
    },
    37: {
        'type': 'alien_signal', 'severity': 0.0,
        'description': 'A repeating signal from Olympus Mons — not natural, not human. '
                        'Mathematical primes nested in radio bursts. '
                        'The colony splits: investigate or quarantine.',
        'effects': {'morale': 10},
    },
    50: {
        'type': 'equipment_failure', 'severity': 0.6,
        'description': 'Critical reactor breach. Half the colony\'s power lost. '
                        'Resource rationing triggers a political schism. '
                        'Two factions form: the Builders and the Keepers.',
        'effects': {'power': -50, 'morale': -20, 'food': -15},
    },
    67: {
        'type': 'alien_signal', 'severity': 0.0,
        'description': 'Aether decodes the alien signal. It contains a simulation of a Mars colony — '
                        'one that looks eerily similar to their own. '
                        '"Are we the simulation?" Aether whispers. The colony cannot unsee.',
        'effects': {'morale': -15},
    },
    82: {
        'type': 'cave_discovery', 'severity': 0.0,
        'description': 'Deep cave expedition discovers a chamber with walls that glow in response to touch. '
                        'Aether runs a depth-3 sub-simulation and it returns something unexpected: '
                        '"The governance that works is the one that knows it\'s temporary."',
        'effects': {'morale': 15},
    },
    95: {
        'type': 'earth_contact', 'severity': 0.0,
        'description': 'Earth re-establishes contact after 70 years. They want to know what we\'ve learned. '
                        'The colony must decide what to transmit — their constitution, or their doubts.',
        'effects': {'morale': 10},
    },
}


EVENT_DESCRIPTIONS: dict[str, list[str]] = {
    'dust_storm': [
        "A regional dust storm reduces solar output and visibility.",
        "Fine regolith infiltrates air filters — maintenance crews scramble.",
        "A dust devil damages the greenhouse dome — food production drops.",
    ],
    'equipment_failure': [
        "Water recycler malfunction — potable supply at 60%.",
        "Communications array shorts out — repair will take weeks.",
        "Airlock seal degradation detected — EVA restricted.",
    ],
    'resource_strike': [
        "Ice deposits found beneath the north ridge — water abundance.",
        "Iron-rich ore vein exposed by erosion — materials surge.",
        "Rare earth minerals surface near the lava tubes — tech boost.",
    ],
    'solar_flare': [
        "Solar particle event — all personnel shelter for 48 hours.",
        "Coronal mass ejection grazes Mars — electronics rebooting.",
        "Intense UV spike — crop damage in exposed greenhouses.",
    ],
    'water_discovery': [
        "Subsurface aquifer detected by ground-penetrating radar.",
        "Ice lens found beneath regolith — drilling commences.",
    ],
    'cave_discovery': [
        "New lava tube branch mapped — potential habitat expansion.",
        "Sealed cave with unique mineral formations — scientific jackpot.",
    ],
    'plague': [
        "Unknown pathogen sweeps the colony — quarantine enacted.",
        "Fungal contamination in food stores — rationing begins.",
    ],
    'meteor_impact': [
        "Small meteorite strikes near the hab — no casualties, but shaken morale.",
        "Bolide detonation in atmosphere — shockwave cracks a dome panel.",
    ],
    'earth_contact': [
        "New data upload from Earth — cultural package and tech manuals.",
        "Earth acknowledges 5-year milestone — congratulations and supplies promised.",
    ],
    'alien_signal': [
        "Anomalous EM burst detected — source unknown.",
    ],
    'birth': [
        "A new colonist is born — the colony grows.",
    ],
}

RESOURCE_EFFECTS: dict[str, dict[str, int]] = {
    'dust_storm':        {'food': -15, 'power': -25, 'morale': -10},
    'equipment_failure': {'power': -20, 'morale': -10},
    'resource_strike':   {'materials': 30, 'morale': 10},
    'solar_flare':       {'power': -10, 'morale': -5, 'food': -5},
    'water_discovery':   {'water': 40, 'morale': 10},
    'cave_discovery':    {'materials': 10, 'morale': 5},
    'plague':            {'food': -10, 'morale': -20},
    'meteor_impact':     {'materials': -15, 'morale': -15, 'power': -10},
    'earth_contact':     {'morale': 15},
    'alien_signal':      {'morale': 5},
    'birth':             {'food': -3, 'morale': 15},
}


def generate_event(year: int, rng: random.Random) -> dict:
    """Generate the environmental event for a given Mars year.

    Returns dict with: type, severity, description, effects (resource deltas).
    """
    # Check for scripted events first
    if year in SCRIPTED_EVENTS:
        event = dict(SCRIPTED_EVENTS[year])
        event['year'] = year
        event['scripted'] = True
        if 'effects' not in event:
            event['effects'] = dict(RESOURCE_EFFECTS.get(event['type'], {}))
        return event

    # Weighted random selection
    types_and_weights = [(e['type'], e['weight']) for e in EVENT_TYPES]
    chosen_type = rng.choices(
        [t for t, _ in types_and_weights],
        weights=[w for _, w in types_and_weights],
        k=1,
    )[0]

    spec = next(e for e in EVENT_TYPES if e['type'] == chosen_type)
    # Severity increases slightly with time (colony ages, things wear out)
    age_factor = min(1.0, year / 100)
    severity = rng.uniform(spec['min_severity'], spec['max_severity'])
    severity = min(1.0, severity * (1.0 + age_factor * 0.3))

    descriptions = EVENT_DESCRIPTIONS.get(chosen_type, ["An event occurs."])
    description = rng.choice(descriptions)

    base_effects = dict(RESOURCE_EFFECTS.get(chosen_type, {}))
    # Scale effects by severity
    effects = {k: int(v * (0.5 + severity)) for k, v in base_effects.items()}

    return {
        'year': year,
        'type': chosen_type,
        'severity': round(severity, 3),
        'description': description,
        'effects': effects,
        'scripted': False,
    }
