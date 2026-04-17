"""
Oral history engine for Mars-100.

Memories propagate through the social graph, distort per-carrier
(telephone game), and mythify when spread widely enough.
Myths influence colonist behavior via action weight modifiers.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


# --- Theme / stance taxonomy ---

EVENT_TO_THEME: dict[str, str] = {
    "dust_storm": "storm",
    "solar_flare": "storm",
    "meteor_shower": "storm",
    "resource_strike": "breakthrough",
    "tech_breakthrough": "breakthrough",
    "alien_signal": "breakthrough",
    "equipment_failure": "failure",
    "medical_emergency": "failure",
    "habitat_breach": "failure",
    "colonist_conflict": "conflict",
    "earth_contact": "origin",
    "supply_pod": "origin",
    "calm": "origin",
}

STANCES = ("cautionary", "heroic", "aspirational")

# Maps myth theme → action weight bonuses
MYTH_ACTION_MODIFIERS: dict[str, dict[str, float]] = {
    "storm": {"terraform": 0.3, "pray": 0.2, "cooperate": 0.15},
    "failure": {"code": 0.25, "cooperate": 0.2, "hoard": 0.1},
    "conflict": {"mediate": 0.3, "pray": 0.15, "sabotage": -0.1},
    "breakthrough": {"explore": 0.3, "code": 0.2, "terraform": 0.15},
    "origin": {"pray": 0.2, "cooperate": 0.15, "farm": 0.1},
}

# --- Data classes ---


@dataclass
class MemoryVariant:
    """One colonist's version of a shared memory."""
    colonist_id: str
    emotional_weight: float
    heard_from: str  # "direct" or colonist_id
    fidelity: float  # 1.0 = perfect, degrades on each retelling
    heard_year: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "colonist_id": self.colonist_id,
            "emotional_weight": round(self.emotional_weight, 4),
            "heard_from": self.heard_from,
            "fidelity": round(self.fidelity, 4),
            "heard_year": self.heard_year,
        }


@dataclass
class SharedMemory:
    """A colony-wide memory with per-carrier variants."""
    memory_id: str  # e.g. "dust_storm_y5"
    event_name: str
    origin_year: int
    description: str
    variants: list[MemoryVariant] = field(default_factory=list)
    is_myth: bool = False
    myth_year: int | None = None
    theme: str = "origin"
    stance: str = "cautionary"
    salience: float = 1.0
    times_shared: int = 0

    def carrier_ids(self) -> list[str]:
        """Return sorted list of carrier IDs."""
        return sorted(v.colonist_id for v in self.variants)

    def has_carrier(self, colonist_id: str) -> bool:
        return any(v.colonist_id == colonist_id for v in self.variants)

    def get_variant(self, colonist_id: str) -> MemoryVariant | None:
        for v in self.variants:
            if v.colonist_id == colonist_id:
                return v
        return None

    def remove_carrier(self, colonist_id: str) -> None:
        self.variants = [v for v in self.variants if v.colonist_id != colonist_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "event_name": self.event_name,
            "origin_year": self.origin_year,
            "description": self.description,
            "variants": [v.to_dict() for v in self.variants],
            "is_myth": self.is_myth,
            "myth_year": self.myth_year,
            "theme": self.theme,
            "stance": self.stance,
            "salience": round(self.salience, 4),
            "times_shared": self.times_shared,
            "carrier_count": len(self.variants),
        }


@dataclass
class OralHistory:
    """Colony-wide oral history pool."""
    memories: list[SharedMemory] = field(default_factory=list)

    def get_memory(self, memory_id: str) -> SharedMemory | None:
        for m in self.memories:
            if m.memory_id == memory_id:
                return m
        return None

    def myths(self) -> list[SharedMemory]:
        return [m for m in self.memories if m.is_myth]

    def carrier_myths(self, colonist_id: str) -> list[SharedMemory]:
        return [m for m in self.memories if m.is_myth and m.has_carrier(colonist_id)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "memories": [m.to_dict() for m in self.memories],
            "total_memories": len(self.memories),
            "total_myths": len(self.myths()),
        }


# --- Core functions ---


def witness_event(
    history: OralHistory,
    event_name: str,
    year: int,
    description: str,
    witness_ids: list[str],
    severity: float,
    rng: random.Random,
) -> SharedMemory | None:
    """All active colonists witness an event, creating a shared memory.

    Only events with severity >= 0.3 create oral memories.
    Returns the created SharedMemory or None.
    """
    if severity < 0.3:
        return None
    memory_id = f"{event_name}_y{year}"
    if history.get_memory(memory_id) is not None:
        return None

    theme = EVENT_TO_THEME.get(event_name, "origin")
    stance = rng.choice(STANCES)

    mem = SharedMemory(
        memory_id=memory_id,
        event_name=event_name,
        origin_year=year,
        description=description,
        theme=theme,
        stance=stance,
        salience=min(1.0, severity * 1.2),
    )

    for cid in sorted(witness_ids):
        weight = severity * (0.8 + rng.random() * 0.4)
        if rng.random() < 0.3:
            weight = -weight
        mem.variants.append(MemoryVariant(
            colonist_id=cid,
            emotional_weight=max(-1.0, min(1.0, weight)),
            heard_from="direct",
            fidelity=1.0,
            heard_year=year,
        ))

    history.memories.append(mem)
    return mem


def share_memory(
    history: OralHistory,
    speaker_id: str,
    listener_id: str,
    year: int,
    trust: float,
    rng: random.Random,
) -> list[dict]:
    """Speaker shares memories with listener along a cooperation edge.

    Higher trust = more sharing. Returns log of what was shared.
    """
    log: list[dict] = []
    share_prob = 0.3 + trust * 0.4

    for mem in history.memories:
        speaker_var = mem.get_variant(speaker_id)
        if speaker_var is None:
            continue
        if mem.has_carrier(listener_id):
            continue
        if rng.random() > share_prob:
            continue

        # Telephone game: fidelity degrades, emotion drifts
        new_fidelity = speaker_var.fidelity * (0.7 + trust * 0.2)
        drift = rng.gauss(0, 0.15 * (1.0 - new_fidelity))
        new_weight = max(-1.0, min(1.0, speaker_var.emotional_weight + drift))

        mem.variants.append(MemoryVariant(
            colonist_id=listener_id,
            emotional_weight=new_weight,
            heard_from=speaker_id,
            fidelity=max(0.0, min(1.0, new_fidelity)),
            heard_year=year,
        ))
        mem.times_shared += 1
        log.append({
            "memory_id": mem.memory_id,
            "speaker": speaker_id,
            "listener": listener_id,
            "fidelity": round(new_fidelity, 4),
        })

    return log


def check_mythification(
    history: OralHistory,
    active_count: int,
    year: int,
) -> list[dict]:
    """Check if any shared memories have mythified.

    Myth threshold: >= ceil(active * 0.35) carriers,
    times_shared >= 5, salience > 0.5.
    Returns log of newly mythified memories.
    """
    threshold = math.ceil(active_count * 0.35) if active_count > 0 else 1
    log: list[dict] = []

    for mem in history.memories:
        if mem.is_myth:
            continue
        if len(mem.variants) < threshold:
            continue
        if mem.times_shared < 5:
            continue
        if mem.salience <= 0.5:
            continue

        mem.is_myth = True
        mem.myth_year = year
        # Mythification amplifies salience
        mem.salience = min(1.0, mem.salience * 1.3)
        log.append({
            "memory_id": mem.memory_id,
            "theme": mem.theme,
            "stance": mem.stance,
            "carriers": len(mem.variants),
            "year": year,
        })

    return log


def decay_salience(
    history: OralHistory,
    year: int,
    rng: random.Random,
) -> None:
    """Decay salience of all memories. Myths decay slower.

    Memories with zero carriers are pruned.
    """
    to_remove: list[str] = []
    for mem in history.memories:
        if not mem.variants:
            to_remove.append(mem.memory_id)
            continue
        rate = 0.02 if mem.is_myth else 0.08
        mem.salience = max(0.0, mem.salience - rate)
        if mem.salience <= 0.0 and not mem.is_myth:
            to_remove.append(mem.memory_id)

    history.memories = [m for m in history.memories if m.memory_id not in to_remove]


def on_death(history: OralHistory, colonist_id: str) -> None:
    """Remove a dead/exiled colonist from all memory carrier lists."""
    for mem in history.memories:
        mem.remove_carrier(colonist_id)


def on_birth(
    history: OralHistory,
    child_id: str,
    active_ids: list[str],
) -> None:
    """Newborn inherits myths from the colony (cultural transmission).

    Child gets variants for up to 3 most salient myths,
    with reduced fidelity (0.5).
    """
    myths = sorted(history.myths(), key=lambda m: m.salience, reverse=True)
    for myth in myths[:3]:
        if myth.has_carrier(child_id):
            continue
        # Inherit average emotional weight from existing carriers
        if not myth.variants:
            continue
        avg_weight = sum(v.emotional_weight for v in myth.variants) / len(myth.variants)
        myth.variants.append(MemoryVariant(
            colonist_id=child_id,
            emotional_weight=avg_weight,
            heard_from="cultural",
            fidelity=0.5,
            heard_year=myth.origin_year,
        ))


def action_modifiers(
    history: OralHistory,
    colonist_id: str,
) -> dict[str, float]:
    """Return action weight bonuses from myths this colonist carries.

    Only myths where colonist is a carrier have influence.
    Scale = abs(emotional_weight) * salience.
    """
    mods: dict[str, float] = {}
    for myth in history.carrier_myths(colonist_id):
        var = myth.get_variant(colonist_id)
        if var is None:
            continue
        scale = abs(var.emotional_weight) * myth.salience
        theme_mods = MYTH_ACTION_MODIFIERS.get(myth.theme, {})
        for action, bonus in theme_mods.items():
            mods[action] = mods.get(action, 0.0) + bonus * scale
    return mods


def subsim_bindings(
    history: OralHistory,
    colonist_id: str,
) -> dict[str, float]:
    """Return LisPy bindings for myths this colonist carries.

    Provides myth-count and myth-salience for sub-sim expressions.
    """
    myths = history.carrier_myths(colonist_id)
    if not myths:
        return {"myth-count": 0, "myth-salience": 0.0}
    total_salience = sum(m.salience for m in myths)
    return {
        "myth-count": len(myths),
        "myth-salience": round(total_salience / len(myths), 4),
    }


def tick_year(
    history: OralHistory,
    year: int,
    events: list[dict],
    active_ids: list[str],
    cooperation_pairs: list[tuple[str, str]],
    trust_fn: Any,
    rng: random.Random,
) -> dict[str, Any]:
    """Run one year of oral history. Returns year log.

    Args:
        history: the OralHistory state
        year: current simulation year
        events: list of event dicts with 'name', 'severity', 'description'
        active_ids: list of active colonist IDs
        cooperation_pairs: list of (speaker_id, listener_id)
        trust_fn: callable(from_id, to_id) -> float
        rng: seeded random
    """
    # 1. Witness new events
    witnessed: list[str] = []
    for ev in events:
        mem = witness_event(
            history,
            event_name=ev.get("name", "unknown"),
            year=year,
            description=ev.get("description", ""),
            witness_ids=active_ids,
            severity=ev.get("severity", 0.0),
            rng=rng,
        )
        if mem is not None:
            witnessed.append(mem.memory_id)

    # 2. Share memories along cooperation edges
    sharing_log: list[dict] = []
    for speaker_id, listener_id in cooperation_pairs:
        if speaker_id not in active_ids or listener_id not in active_ids:
            continue
        trust = trust_fn(speaker_id, listener_id)
        entries = share_memory(history, speaker_id, listener_id, year, trust, rng)
        sharing_log.extend(entries)

    # 3. Check mythification
    myth_log = check_mythification(history, len(active_ids), year)

    # 4. Decay salience + prune
    decay_salience(history, year, rng)

    return {
        "year": year,
        "witnessed": witnessed,
        "sharing": sharing_log,
        "new_myths": myth_log,
        "total_memories": len(history.memories),
        "total_myths": len(history.myths()),
    }
