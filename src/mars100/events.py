"""
Environmental events for Mars-100.

Each Martian year, one or more events occur affecting resources, morale, relationships.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class Event:
    """A single environmental event."""
    name: str
    category: str
    severity: float
    description: str
    effects: dict[str, float]

    def to_dict(self) -> dict:
        return {"name": self.name, "category": self.category,
                "severity": self.severity, "description": self.description,
                "effects": self.effects}


EVENT_TEMPLATES: list[dict[str, Any]] = [
    {"name": "dust_storm", "category": "environment", "severity_range": (0.3, 0.8),
     "descriptions": ["A massive dust storm engulfs the colony for weeks.",
                       "Fine Martian regolith clogs air filters and reduces solar output.",
                       "The worst dust storm in decades buries the southern airlock."],
     "effects": {"power": -0.15, "air": -0.05, "morale": -0.1}},
    {"name": "resource_strike", "category": "environment", "severity_range": (0.0, 0.3),
     "descriptions": ["Ice deposits discovered beneath the eastern ridge.",
                       "A rich vein of iron oxide uncovered during excavation.",
                       "Subsurface water table detected by ground-penetrating radar."],
     "effects": {"water": 0.2, "food": 0.1, "morale": 0.15}},
    {"name": "equipment_failure", "category": "environment", "severity_range": (0.4, 0.9),
     "descriptions": ["The primary water recycler fails catastrophically.",
                       "Greenhouse dome seal breach \u2014 crops exposed to vacuum.",
                       "Nuclear reactor scram \u2014 emergency power only for 30 sols."],
     "effects": {"water": -0.1, "power": -0.2, "food": -0.05}},
    {"name": "earth_contact", "category": "social", "severity_range": (0.0, 0.2),
     "descriptions": ["Message from Earth: supply ship confirmed for next launch window.",
                       "Video letters from family arrive after 22-minute delay.",
                       "Earth sends updated colony charter with governance suggestions."],
     "effects": {"morale": 0.2, "food": 0.05}},
    {"name": "alien_signal", "category": "cosmic", "severity_range": (0.1, 0.5),
     "descriptions": ["Anomalous radio signal detected from Phobos.",
                       "Strange crystalline structures found in deep drill core.",
                       "Repeating pattern in cosmic ray data \u2014 statistical fluke or message?"],
     "effects": {"morale": 0.1, "faith_boost": 0.05, "paranoia_boost": 0.1}},
    {"name": "solar_flare", "category": "cosmic", "severity_range": (0.5, 1.0),
     "descriptions": ["Coronal mass ejection detected \u2014 48 hours to shelter.",
                       "Severe radiation event \u2014 all colonists confined to shielded core.",
                       "Electronics damaged by electromagnetic pulse from solar event."],
     "effects": {"power": -0.1, "medicine": -0.1, "morale": -0.15}},
    {"name": "ice_volcano", "category": "environment", "severity_range": (0.2, 0.6),
     "descriptions": ["Cryovolcanic eruption 200km north \u2014 ground tremors felt.",
                       "Ice geyser breaks through permafrost near the colony perimeter."],
     "effects": {"water": 0.15, "air": 0.05, "morale": -0.05}},
    {"name": "phobos_transit", "category": "cosmic", "severity_range": (0.0, 0.1),
     "descriptions": ["Phobos eclipses the sun \u2014 30 seconds of twilight at noon.",
                       "Double shadow: Phobos and Deimos transit simultaneously."],
     "effects": {"morale": 0.05, "faith_boost": 0.03}},
    {"name": "cave_discovery", "category": "environment", "severity_range": (0.0, 0.3),
     "descriptions": ["Lava tube entrance found \u2014 potential expansion space.",
                       "Underground cavern with stable temperature discovered."],
     "effects": {"morale": 0.15, "food": 0.05}},
    {"name": "colonist_conflict", "category": "social", "severity_range": (0.3, 0.7),
     "descriptions": ["A bitter dispute over food rationing escalates to shouting.",
                       "Accusations of resource hoarding fracture the colony trust."],
     "effects": {"morale": -0.2, "food": -0.03}},
    {"name": "breakthrough", "category": "discovery", "severity_range": (0.0, 0.2),
     "descriptions": ["New hydroponics technique doubles crop yield per unit water.",
                       "Coding breakthrough: automated drone maintenance saves labor."],
     "effects": {"food": 0.15, "morale": 0.1, "power": 0.05}},
    {"name": "epidemic", "category": "social", "severity_range": (0.5, 0.9),
     "descriptions": ["Unknown pathogen spreads through the water system.",
                       "Regolith dust causes chronic respiratory inflammation."],
     "effects": {"medicine": -0.3, "morale": -0.25, "food": -0.05}},
]


def generate_events(year: int, rng: random.Random,
                    event_count: int = 0) -> list[Event]:
    """Generate 1-3 events for a given year."""
    if event_count <= 0:
        base = 1 + year // 25
        event_count = min(3, max(1, base + (1 if rng.random() < 0.3 else 0)))
    chosen = rng.sample(EVENT_TEMPLATES, min(event_count, len(EVENT_TEMPLATES)))
    events: list[Event] = []
    for tmpl in chosen:
        sev_lo, sev_hi = tmpl["severity_range"]
        severity = rng.uniform(sev_lo, sev_hi)
        desc = rng.choice(tmpl["descriptions"])
        effects = {k: v * (0.5 + severity) for k, v in tmpl["effects"].items()}
        events.append(Event(name=tmpl["name"], category=tmpl["category"],
                           severity=severity, description=desc, effects=effects))
    return events
