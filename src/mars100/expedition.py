"""
Expedition & Discovery engine for Mars-100.

Colonists who choose the 'explore' action may form expeditions to discover
sites on Mars.  Expeditions add geography to the simulation — the colony's
known territory expands over 100 years.

Rules:
  - An expedition forms when 2+ colonists choose 'explore' in the same year.
  - Expedition members do NOT contribute normal skill bonuses (they're away).
  - Success depends on team stats, environmental conditions, and luck.
  - Discovered sites provide diminishing resource bonuses each year.
  - Failed expeditions can kill members (distinct cause: 'expedition').
  - Sites are unique and monotonically accumulate.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import Colonist, STAT_NAMES

# ── Mars geography templates ─────────────────────────────────────────────

SITE_TYPES: list[dict[str, Any]] = [
    {
        "type": "ice_field", "label": "Ice Field",
        "bonus": {"water": 0.03}, "meta_boost": 0.0,
        "descriptions": [
            "Vast sub-surface ice deposits glinting beneath regolith.",
            "Glacier remnant trapped in a polar crater — ancient water.",
        ],
    },
    {
        "type": "lava_tube", "label": "Lava Tube",
        "bonus": {"air": 0.02}, "meta_boost": 0.0,
        "descriptions": [
            "Collapsed skylight reveals a pressure-tight lava tube.",
            "A winding cavern shielded from radiation — potential habitat.",
        ],
    },
    {
        "type": "mineral_vein", "label": "Mineral Vein",
        "bonus": {"power": 0.02, "medicine": 0.01}, "meta_boost": 0.0,
        "descriptions": [
            "Iron-rich outcrop with rare-earth trace minerals.",
            "Crystalline deposits embedded in basalt — useful for electronics.",
        ],
    },
    {
        "type": "ancient_anomaly", "label": "Ancient Anomaly",
        "bonus": {}, "meta_boost": 0.02,
        "descriptions": [
            "Geometric patterns etched in stone — natural or not?",
            "A subsurface void emitting faint electromagnetic pulses.",
        ],
    },
    {
        "type": "fertile_crater", "label": "Fertile Crater",
        "bonus": {"food": 0.03}, "meta_boost": 0.0,
        "descriptions": [
            "Clay-rich soil in a sheltered crater — nearly arable.",
            "Nitrogen-bearing regolith layer perfect for amended farming.",
        ],
    },
    {
        "type": "geothermal_vent", "label": "Geothermal Vent",
        "bonus": {"power": 0.03}, "meta_boost": 0.0,
        "descriptions": [
            "Steam plume rising from a fissure — geothermal energy source.",
            "Volcanic heat seep with stable 40°C surface temperature.",
        ],
    },
]

SITE_NAMES_POOL = [
    "Olympus Shelf", "Valles Edge", "Hellas Rim", "Elysium Hollow",
    "Tharsis Notch", "Arcadia Flats", "Syrtis Ridge", "Isidis Basin",
    "Chryse Ford", "Amazonis Reach", "Noctis Fissure", "Gale Saddle",
    "Jezero Shelf", "Utopia Scarp", "Meridiani Bluff", "Arabia Steppe",
    "Noachis Plateau", "Cimmeria Drift", "Hesperia Gulch", "Tempe Rise",
]

# ── Expedition parameters ────────────────────────────────────────────────

MIN_TEAM_SIZE = 2
EXPEDITION_BASE_SUCCESS = 0.55
EXPEDITION_DEATH_RATE = 0.08
EXPEDITION_FOOD_COST = 0.04
EXPEDITION_POWER_COST = 0.02
MAX_SITES = 15
BONUS_DECAY_RATE = 0.85  # each site's bonus decays annually


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class Site:
    """A discovered location on Mars."""
    id: str
    name: str
    site_type: str
    label: str
    description: str
    discovered_year: int
    discovered_by: list[str]
    lat: float
    lon: float
    bonus: dict[str, float]
    meta_boost: float
    years_active: int = 0

    def current_bonus(self) -> dict[str, float]:
        """Compute diminishing bonus based on years since discovery."""
        decay = BONUS_DECAY_RATE ** self.years_active
        return {k: v * decay for k, v in self.bonus.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "type": self.site_type,
            "label": self.label, "description": self.description,
            "discovered_year": self.discovered_year,
            "discovered_by": self.discovered_by,
            "lat": round(self.lat, 2), "lon": round(self.lon, 2),
            "bonus": self.bonus, "meta_boost": self.meta_boost,
            "years_active": self.years_active,
            "current_bonus": {k: round(v, 4) for k, v in self.current_bonus().items()},
        }


@dataclass
class ExpeditionResult:
    """Outcome of one expedition."""
    year: int
    team: list[str]
    success: bool
    site: dict | None
    deaths: list[dict]
    bonds: list[tuple[str, str]]
    narrative: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "year": self.year, "team": self.team, "success": self.success,
            "site": self.site, "deaths": self.deaths,
            "bonds": [(a, b) for a, b in self.bonds],
            "narrative": self.narrative,
        }


# ── Core functions ───────────────────────────────────────────────────────

def can_form_expedition(explorers: list[Colonist],
                        food: float, power: float) -> bool:
    """Check whether conditions allow an expedition this year."""
    if len(explorers) < MIN_TEAM_SIZE:
        return False
    if food < EXPEDITION_FOOD_COST * 2:
        return False
    if power < EXPEDITION_POWER_COST * 2:
        return False
    return True


def select_team(explorers: list[Colonist], rng: random.Random,
                max_size: int = 4) -> list[Colonist]:
    """Select the best available team from colonists who chose 'explore'.

    Prefers colonists with high resolve + improvisation.
    """
    scored = [(c, c.stats.resolve + c.stats.improvisation) for c in explorers]
    scored.sort(key=lambda x: x[1], reverse=True)
    size = min(max_size, len(scored))
    size = max(MIN_TEAM_SIZE, rng.randint(MIN_TEAM_SIZE, size))
    return [c for c, _ in scored[:size]]


def compute_success_probability(team: list[Colonist],
                                event_severity: float,
                                num_known_sites: int) -> float:
    """Calculate expedition success probability.

    Higher with: good team stats, calm conditions, fewer existing sites
    (diminishing returns on easy finds).
    """
    avg_resolve = sum(c.stats.resolve for c in team) / len(team)
    avg_improv = sum(c.stats.improvisation for c in team) / len(team)
    team_bonus = (avg_resolve * 0.4 + avg_improv * 0.4) * 0.3

    severity_penalty = event_severity * 0.2

    # Harder to find new sites as map fills
    saturation_penalty = num_known_sites * 0.03

    prob = EXPEDITION_BASE_SUCCESS + team_bonus - severity_penalty - saturation_penalty
    return max(0.05, min(0.95, prob))


def generate_site(year: int, team: list[Colonist],
                  known_sites: list[Site],
                  rng: random.Random) -> Site:
    """Generate a newly discovered site.

    Avoids duplicate types/names from existing sites.
    """
    used_types = {s.site_type for s in known_sites}
    used_names = {s.name for s in known_sites}

    available_types = [t for t in SITE_TYPES if t["type"] not in used_types]
    if not available_types:
        available_types = list(SITE_TYPES)

    template = rng.choice(available_types)

    available_names = [n for n in SITE_NAMES_POOL if n not in used_names]
    if not available_names:
        available_names = [f"Site-{year}-{rng.randint(100, 999)}"]
    name = rng.choice(available_names)

    site_id = f"site-{year}-{template['type']}"
    lat = rng.uniform(-60.0, 60.0)
    lon = rng.uniform(-180.0, 180.0)

    return Site(
        id=site_id, name=name, site_type=template["type"],
        label=template["label"],
        description=rng.choice(template["descriptions"]),
        discovered_year=year,
        discovered_by=[c.id for c in team],
        lat=lat, lon=lon,
        bonus=dict(template["bonus"]),
        meta_boost=template["meta_boost"],
    )


def run_expedition(year: int, team: list[Colonist],
                   known_sites: list[Site],
                   event_severity: float,
                   rng: random.Random) -> ExpeditionResult:
    """Execute an expedition and return the result.

    On success: discover a site, bond the team.
    On failure: possible deaths, some bonding from shared hardship.
    """
    success_prob = compute_success_probability(team, event_severity,
                                               len(known_sites))
    success = rng.random() < success_prob

    deaths: list[dict] = []
    site_dict: dict | None = None
    bonds: list[tuple[str, str]] = []
    narrative_parts: list[str] = []

    team_names = [c.name for c in team]
    narrative_parts.append(
        f"Expedition launched: {', '.join(team_names)} venture into the Martian wastes."
    )

    if success:
        site = generate_site(year, team, known_sites, rng)
        known_sites.append(site)
        site_dict = site.to_dict()
        narrative_parts.append(
            f"Discovery! {site.label} found at {site.name}: {site.description}"
        )
    else:
        narrative_parts.append(
            "The expedition returns empty-handed — harsh terrain and worse luck."
        )

    # Death check — expedition deaths are riskier than colony
    for member in team:
        death_rate = EXPEDITION_DEATH_RATE
        if not success:
            death_rate *= 1.5
        if event_severity > 0.5:
            death_rate *= 1.3
        if rng.random() < death_rate:
            cause = rng.choice([
                "lost in dust storm during expedition",
                "equipment failure on expedition",
                "fall into crevasse during expedition",
                "radiation exposure during expedition",
                "oxygen depletion during expedition",
            ])
            deaths.append({"id": member.id, "name": member.name,
                           "cause": cause, "year": year})
            member.die(year, cause)
            narrative_parts.append(
                f"LOSS: {member.name} did not return. Cause: {cause}."
            )

    # Shared hardship creates bonds between surviving members
    surviving = [c for c in team if c.is_active()]
    for i, a in enumerate(surviving):
        for b in surviving[i + 1:]:
            bonds.append((a.id, b.id))

    if bonds and not deaths:
        narrative_parts.append("The team returns changed — forged by shared hardship.")
    elif bonds and deaths:
        narrative_parts.append("Survivors carry the weight of loss, bound together.")

    return ExpeditionResult(
        year=year, team=[c.id for c in team], success=success,
        site=site_dict, deaths=deaths, bonds=bonds,
        narrative=" ".join(narrative_parts),
    )


def compute_site_bonuses(sites: list[Site]) -> dict[str, float]:
    """Sum up diminishing resource bonuses from all discovered sites."""
    totals: dict[str, float] = {}
    for site in sites:
        for resource, bonus in site.current_bonus().items():
            totals[resource] = totals.get(resource, 0.0) + bonus
    return totals


def compute_meta_boost(sites: list[Site]) -> float:
    """Sum meta-awareness boost from anomaly-type sites."""
    total = 0.0
    for site in sites:
        decay = BONUS_DECAY_RATE ** site.years_active
        total += site.meta_boost * decay
    return total


def age_sites(sites: list[Site]) -> None:
    """Increment years_active for all known sites (call once per year)."""
    for site in sites:
        site.years_active += 1
