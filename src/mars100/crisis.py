"""
Crisis Protocol — the colony's immune system.

Detects resource crises, biases colonist actions toward survival work,
applies consumption rationing, and harvests wisdom from dead colonists
as legacy warnings. Purely additive — all existing behavior preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CRISIS_THRESHOLD = 0.15
MAX_PREPAREDNESS = 2.0
TRUST_BOOST_PER_YEAR = 0.02
RATIONING_MULT = 0.75
RESOURCE_RESPONSE_MAP: dict[str, list[tuple[str, float]]] = {
    "food":     [("farm", 3.0), ("cooperate", 1.0)],
    "water":    [("terraform", 2.5), ("explore", 1.0)],
    "power":    [("code", 2.0), ("research", 1.5)],
    "air":      [("terraform", 3.0), ("code", 1.5)],
    "medicine": [("pray", 1.5), ("mediate", 1.0), ("cooperate", 1.5)],
}


@dataclass
class CrisisEpisode:
    """One continuous stretch of a resource being critical."""
    resource: str
    start_year: int
    end_year: int | None = None
    peak_deficit: float = 0.0
    deaths_during: int = 0

    def is_active(self) -> bool:
        """Return True if this episode hasn\'t ended."""
        return self.end_year is None

    def duration(self) -> int:
        """Return years this episode has lasted so far."""
        end = self.end_year if self.end_year is not None else self.start_year
        return max(1, end - self.start_year + 1)

    def to_dict(self) -> dict:
        return {
            "resource": self.resource, "start_year": self.start_year,
            "end_year": self.end_year,
            "peak_deficit": round(self.peak_deficit, 4),
            "deaths_during": self.deaths_during,
            "duration": self.duration(),
        }


@dataclass
class LegacyWarning:
    """Wisdom harvested from a dead colonist about a crisis."""
    colonist_id: str
    colonist_name: str
    year: int
    cause: str
    warning: str

    def to_dict(self) -> dict:
        return {
            "colonist_id": self.colonist_id, "colonist_name": self.colonist_name,
            "year": self.year, "cause": self.cause, "warning": self.warning,
        }


@dataclass
class CrisisState:
    """Tracks all crisis episodes, preparedness, and legacy warnings."""
    episodes: list[CrisisEpisode] = field(default_factory=list)
    active: list[str] = field(default_factory=list)
    preparedness: float = 1.0
    legacy_warnings: list[LegacyWarning] = field(default_factory=list)
    total_crisis_years: int = 0

    def to_dict(self) -> dict:
        return {
            "episodes": [e.to_dict() for e in self.episodes],
            "active": list(self.active),
            "preparedness": round(self.preparedness, 4),
            "legacy_warnings": [w.to_dict() for w in self.legacy_warnings],
            "total_crisis_years": self.total_crisis_years,
            "total_episodes": len(self.episodes),
            "active_episodes": sum(1 for e in self.episodes if e.is_active()),
        }


def detect_crises(resources: object) -> list[str]:
    """Return list of resource names currently below crisis threshold."""
    from src.mars100.colony import RESOURCE_NAMES
    critical: list[str] = []
    for name in RESOURCE_NAMES:
        level = getattr(resources, name, 1.0)
        if level < CRISIS_THRESHOLD:
            critical.append(name)
    return critical


def update_crisis_state(state: CrisisState, crises: list[str],
                        year: int) -> None:
    """Update crisis state: start new episodes, end resolved ones."""
    previously_active = set(state.active)
    currently_critical = set(crises)
    for resource in currently_critical - previously_active:
        state.episodes.append(CrisisEpisode(resource=resource, start_year=year))
        state.preparedness = min(MAX_PREPAREDNESS, state.preparedness + 0.1)
    for resource in previously_active - currently_critical:
        for ep in reversed(state.episodes):
            if ep.resource == resource and ep.is_active():
                ep.end_year = year
                break
    state.active = list(currently_critical)
    if currently_critical:
        state.total_crisis_years += 1


def compute_consumption_modifier(state: CrisisState) -> float:
    """Return consumption multiplier (< 1.0 means rationing)."""
    if not state.active:
        return 1.0
    return RATIONING_MULT


def crisis_action_weights(state: CrisisState) -> dict[str, float]:
    """Compute action weight bonuses based on active crises."""
    bonuses: dict[str, float] = {}
    if not state.active:
        return bonuses
    for resource in state.active:
        for action, weight in RESOURCE_RESPONSE_MAP.get(resource, []):
            bonuses[action] = bonuses.get(action, 0.0) + weight
    for action in bonuses:
        bonuses[action] *= state.preparedness
    return bonuses


def compute_trust_boost(state: CrisisState) -> float:
    """Return per-pair trust boost during active crisis."""
    if not state.active:
        return 0.0
    return TRUST_BOOST_PER_YEAR * len(state.active)


def harvest_legacy(colonist_dict: dict, state: CrisisState,
                   year: int, cause: str) -> None:
    """Extract a legacy warning from a dead colonist."""
    cid = colonist_dict.get("id", "unknown")
    name = colonist_dict.get("name", "Unknown")
    cause_resource = _cause_to_resource(cause)
    for ep in state.episodes:
        if ep.is_active() and ep.resource == cause_resource:
            ep.deaths_during += 1
    memories = colonist_dict.get("memories", [])
    strongest = max(memories, key=lambda m: abs(m.get("emotional_valence", 0)),
                    default=None) if memories else None
    if strongest:
        event_text = strongest["event"][:60]
        memory_note = f" Last thought: '{event_text}'"
    else:
        memory_note = ""
    warning = f"{name} died of {cause} in year {year}.{memory_note}"
    state.legacy_warnings.append(LegacyWarning(
        colonist_id=cid, colonist_name=name,
        year=year, cause=cause, warning=warning,
    ))


def format_crisis_year_data(state: CrisisState, consumption_mult: float,
                            min_level: float) -> dict:
    """Format crisis data for inclusion in YearResult."""
    return {
        "active_crises": list(state.active),
        "consumption_mult": round(consumption_mult, 4),
        "min_resource_level": round(min_level, 4),
        "preparedness": round(state.preparedness, 4),
        "total_episodes": len(state.episodes),
        "active_episodes": sum(1 for e in state.episodes if e.is_active()),
        "legacy_count": len(state.legacy_warnings),
        "total_crisis_years": state.total_crisis_years,
    }


def _cause_to_resource(cause: str) -> str:
    """Map a death cause string to the most likely resource."""
    cause_lower = cause.lower()
    mapping = {
        "asphyxia": "air", "suffoc": "air", "air": "air",
        "starv": "food", "hunger": "food", "food": "food",
        "dehydr": "water", "thirst": "water", "water": "water",
        "power": "power", "freez": "power", "cold": "power",
        "disease": "medicine", "illness": "medicine", "medic": "medicine",
    }
    for keyword, resource in mapping.items():
        if keyword in cause_lower:
            return resource
    return "unknown"
