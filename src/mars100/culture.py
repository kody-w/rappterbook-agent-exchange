"""
Cultural DNA engine for Mars-100.

Memes — stories, innovations, taboos, rituals, songs — spread through the
social graph, mutate during transmission, and causally influence colonist
actions and governance votes.  Culture is the organism's collective memory.

Single source of truth: each Meme tracks its own believer set.  Colonists
carry no duplicate meme list — query ``memes_known_by(colonist_id)`` instead.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

MEME_TYPES = ("story", "innovation", "taboo", "ritual", "song")

MAX_MEMES_PER_COLONIST = 7
COLONY_MEME_CAP_FACTOR = 3  # colony_cap = active_count * factor
TRANSMISSION_BASE_PROB = 0.25
ACCURACY_DECAY = 0.05  # per transmission
FORGET_THRESHOLD = 0  # believer count at which meme is pruned
INNOVATION_BOOST = 0.04  # resource production bonus per innovation
TABOO_PENALTY = 0.3  # action weight multiplier for taboo-ed actions
RITUAL_BOND = 0.02  # trust boost between ritual participants
SONG_MORALE = 0.01  # flat morale-like resource boost per active song


@dataclass
class Meme:
    """A cultural unit that spreads through the colony."""
    id: str
    meme_type: str
    content: str
    origin_colonist: str
    origin_year: int
    accuracy: float  # 0.0-1.0, degrades with transmission
    believers: list[str]  # sorted colonist IDs (source of truth)
    influence: dict[str, float]  # maps action or resource name to modifier
    generation: int = 0  # how many times re-transmitted

    def to_dict(self) -> dict:
        return {
            "id": self.id, "meme_type": self.meme_type, "content": self.content,
            "origin_colonist": self.origin_colonist, "origin_year": self.origin_year,
            "accuracy": round(self.accuracy, 4),
            "believers": sorted(self.believers),
            "influence": {k: round(v, 4) for k, v in self.influence.items()},
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Meme:
        return cls(
            id=d["id"], meme_type=d["meme_type"], content=d["content"],
            origin_colonist=d["origin_colonist"], origin_year=d["origin_year"],
            accuracy=d.get("accuracy", 1.0),
            believers=sorted(d.get("believers", [])),
            influence=d.get("influence", {}),
            generation=d.get("generation", 0),
        )


@dataclass
class InteractionRecord:
    """Structured log of one colonist's action in a year."""
    colonist_id: str
    action: str
    partner_id: str | None  # for cooperate/conflict
    resource_contribution: dict[str, float]
    outcome: str  # "normal", "exceptional", "failure"
    adjacent_death: str | None  # id of colonist who died same year, if any

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "colonist_id": self.colonist_id, "action": self.action,
            "resource_contribution": {k: round(v, 4) for k, v in self.resource_contribution.items()},
            "outcome": self.outcome,
        }
        if self.partner_id:
            d["partner_id"] = self.partner_id
        if self.adjacent_death:
            d["adjacent_death"] = self.adjacent_death
        return d


@dataclass
class CulturalMemory:
    """Colony-wide cultural state — all active memes and history."""
    memes: dict[str, Meme] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    next_id: int = 0

    def to_dict(self) -> dict:
        return {
            "memes": {k: v.to_dict() for k, v in self.memes.items()},
            "history": self.history[-200:],  # cap history
            "next_id": self.next_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CulturalMemory:
        memes = {k: Meme.from_dict(v) for k, v in d.get("memes", {}).items()}
        return cls(memes=memes, history=d.get("history", []),
                   next_id=d.get("next_id", 0))

    def _gen_id(self) -> str:
        mid = f"meme-{self.next_id}"
        self.next_id += 1
        return mid

    def memes_known_by(self, colonist_id: str) -> list[Meme]:
        """All memes a colonist currently believes."""
        return [m for m in self.memes.values() if colonist_id in m.believers]

    def active_meme_count(self) -> int:
        return len(self.memes)


def create_meme_from_event(
    culture: CulturalMemory,
    colonist_id: str,
    year: int,
    event_name: str,
    event_severity: float,
    rng: random.Random,
) -> Meme | None:
    """Create a meme from a significant event.  High-severity events are more likely."""
    if rng.random() > event_severity * 0.6 + 0.1:
        return None

    meme_type = _infer_type_from_event(event_name, event_severity, rng)
    content = _generate_content(meme_type, event_name, year, colonist_id)
    influence = _compute_influence(meme_type, event_name, event_severity)

    meme = Meme(
        id=culture._gen_id(), meme_type=meme_type, content=content,
        origin_colonist=colonist_id, origin_year=year,
        accuracy=1.0, believers=[colonist_id], influence=influence,
    )
    culture.memes[meme.id] = meme
    culture.history.append({
        "year": year, "event": "created", "meme_id": meme.id,
        "type": meme_type, "colonist": colonist_id,
    })
    return meme


def create_meme_from_interaction(
    culture: CulturalMemory,
    interaction: InteractionRecord,
    year: int,
    rng: random.Random,
) -> Meme | None:
    """Create an innovation or ritual from an exceptional interaction."""
    if interaction.outcome == "exceptional" and rng.random() < 0.4:
        meme = Meme(
            id=culture._gen_id(), meme_type="innovation",
            content=f"Year {year}: {interaction.colonist_id} discovered improved {interaction.action}",
            origin_colonist=interaction.colonist_id, origin_year=year,
            accuracy=1.0, believers=[interaction.colonist_id],
            influence={interaction.action: INNOVATION_BOOST},
        )
        culture.memes[meme.id] = meme
        culture.history.append({
            "year": year, "event": "created", "meme_id": meme.id,
            "type": "innovation", "colonist": interaction.colonist_id,
        })
        return meme

    if interaction.action == "cooperate" and interaction.partner_id and rng.random() < 0.15:
        believers = sorted([interaction.colonist_id, interaction.partner_id])
        meme = Meme(
            id=culture._gen_id(), meme_type="ritual",
            content=f"Year {year}: cooperation ritual between {interaction.colonist_id} and {interaction.partner_id}",
            origin_colonist=interaction.colonist_id, origin_year=year,
            accuracy=1.0, believers=believers,
            influence={"cooperate": 0.05, "trust_boost": RITUAL_BOND},
        )
        culture.memes[meme.id] = meme
        culture.history.append({
            "year": year, "event": "created", "meme_id": meme.id,
            "type": "ritual", "colonist": interaction.colonist_id,
        })
        return meme

    if interaction.adjacent_death and rng.random() < 0.5:
        meme = Meme(
            id=culture._gen_id(), meme_type="taboo",
            content=f"Year {year}: after {interaction.adjacent_death}'s death, avoid {interaction.action}",
            origin_colonist=interaction.colonist_id, origin_year=year,
            accuracy=1.0, believers=[interaction.colonist_id],
            influence={interaction.action: -TABOO_PENALTY},
        )
        culture.memes[meme.id] = meme
        culture.history.append({
            "year": year, "event": "created", "meme_id": meme.id,
            "type": "taboo", "colonist": interaction.colonist_id,
        })
        return meme

    return None


def transmit_memes(
    culture: CulturalMemory,
    active_colonists: list[Any],
    social_graph: Any,
    year: int,
    rng: random.Random,
) -> list[dict]:
    """Spread memes through the social graph.  Trust increases transmission probability."""
    transmissions: list[dict] = []
    active_ids = [c.id for c in active_colonists]

    for meme in list(culture.memes.values()):
        for believer_id in list(meme.believers):
            if believer_id not in active_ids:
                continue
            for candidate in active_colonists:
                if candidate.id == believer_id or candidate.id in meme.believers:
                    continue
                if len(culture.memes_known_by(candidate.id)) >= MAX_MEMES_PER_COLONIST:
                    # Displace weakest meme if new one is more salient
                    weakest = _weakest_meme(culture, candidate.id)
                    if weakest is None:
                        continue
                    if meme.accuracy < weakest.accuracy:
                        continue
                    # Displace
                    weakest.believers = [b for b in weakest.believers if b != candidate.id]

                rel = social_graph.get(believer_id, candidate.id)
                prob = TRANSMISSION_BASE_PROB * rel.trust * meme.accuracy
                if meme.meme_type == "innovation":
                    prob *= 1.3  # innovations spread faster
                if meme.meme_type == "taboo":
                    prob *= 1.2  # fear spreads fast

                if rng.random() < prob:
                    meme.believers = sorted(set(meme.believers) | {candidate.id})
                    meme.accuracy = max(0.1, meme.accuracy - ACCURACY_DECAY)
                    meme.generation += 1
                    transmissions.append({
                        "year": year, "event": "transmitted", "meme_id": meme.id,
                        "from": believer_id, "to": candidate.id,
                        "accuracy": round(meme.accuracy, 4),
                    })
    culture.history.extend(transmissions)
    return transmissions


def transmit_to_child(
    culture: CulturalMemory,
    parent_ids: list[str],
    child_id: str,
    rng: random.Random,
) -> list[str]:
    """Vertical transmission: parents pass top memes to a newborn."""
    inherited: list[str] = []
    parent_memes: list[Meme] = []
    for pid in parent_ids:
        parent_memes.extend(culture.memes_known_by(pid))
    # Deduplicate
    seen: set[str] = set()
    unique: list[Meme] = []
    for m in parent_memes:
        if m.id not in seen:
            seen.add(m.id)
            unique.append(m)
    # Sort by accuracy descending, take up to 3
    unique.sort(key=lambda m: m.accuracy, reverse=True)
    for meme in unique[:3]:
        if rng.random() < 0.7:
            meme.believers = sorted(set(meme.believers) | {child_id})
            inherited.append(meme.id)
    return inherited


def prune_dead_colonist(culture: CulturalMemory, colonist_id: str) -> None:
    """Remove a dead/exiled colonist from all meme believer lists."""
    for meme in list(culture.memes.values()):
        meme.believers = [b for b in meme.believers if b != colonist_id]


def forget_memes(culture: CulturalMemory, year: int) -> list[str]:
    """Remove memes that have lost all believers."""
    forgotten: list[str] = []
    for mid, meme in list(culture.memes.items()):
        if len(meme.believers) <= FORGET_THRESHOLD:
            forgotten.append(mid)
            culture.history.append({
                "year": year, "event": "forgotten", "meme_id": mid,
                "type": meme.meme_type,
            })
            del culture.memes[mid]
    return forgotten


def enforce_colony_cap(culture: CulturalMemory, active_count: int) -> list[str]:
    """Prune lowest-salience memes if colony exceeds cap."""
    cap = max(10, active_count * COLONY_MEME_CAP_FACTOR)
    if len(culture.memes) <= cap:
        return []
    # Score memes: believers * accuracy
    scored = sorted(
        culture.memes.items(),
        key=lambda kv: len(kv[1].believers) * kv[1].accuracy,
    )
    pruned: list[str] = []
    while len(culture.memes) > cap and scored:
        mid, _ = scored.pop(0)
        del culture.memes[mid]
        pruned.append(mid)
    return pruned


def cultural_action_modifiers(culture: CulturalMemory,
                              colonist_id: str) -> dict[str, float]:
    """Compute action weight modifiers from a colonist's known memes.

    Returns a dict mapping action names to additive weight adjustments.
    Modifiers are subtle — they bias ties, not overwrite personality.
    """
    modifiers: dict[str, float] = {}
    for meme in culture.memes_known_by(colonist_id):
        for key, val in meme.influence.items():
            if key in ("trust_boost",):
                continue  # non-action modifiers handled elsewhere
            modifiers[key] = modifiers.get(key, 0.0) + val * meme.accuracy
    return modifiers


def cultural_vote_modifier(culture: CulturalMemory,
                           colonist_id: str,
                           proposal_type: str) -> float:
    """Compute vote score modifier from cultural memes.

    Returns a small float that biases governance votes.
    """
    modifier = 0.0
    for meme in culture.memes_known_by(colonist_id):
        if meme.meme_type == "story":
            modifier += 0.02 * meme.accuracy  # stories make colonists slightly more conservative
        elif meme.meme_type == "innovation":
            modifier += 0.03 * meme.accuracy  # innovations bias toward progressive governance
        elif meme.meme_type == "taboo":
            modifier -= 0.02 * meme.accuracy  # taboos bias against change
        elif meme.meme_type == "ritual":
            if proposal_type == "consensus":
                modifier += 0.04 * meme.accuracy  # rituals favor consensus governance
    return modifier


def cultural_resource_bonuses(culture: CulturalMemory,
                              active_ids: list[str]) -> dict[str, float]:
    """Compute colony-wide resource bonuses from innovations and songs."""
    bonuses: dict[str, float] = {}
    for meme in culture.memes.values():
        if meme.meme_type == "innovation":
            active_believers = [b for b in meme.believers if b in active_ids]
            for key, val in meme.influence.items():
                if val > 0:
                    bonuses[key] = bonuses.get(key, 0.0) + val * len(active_believers) * meme.accuracy * 0.1
        elif meme.meme_type == "song":
            active_believers = [b for b in meme.believers if b in active_ids]
            if active_believers:
                bonuses["medicine"] = bonuses.get("medicine", 0.0) + SONG_MORALE * len(active_believers) * meme.accuracy
    return bonuses


def tick_culture(
    culture: CulturalMemory,
    year: int,
    active_colonists: list[Any],
    social_graph: Any,
    events: list[Any],
    interactions: list[InteractionRecord],
    deaths: list[dict],
    rng: random.Random,
) -> dict:
    """Advance culture by one year.  Returns summary dict."""
    created: list[dict] = []
    active_ids = [c.id for c in active_colonists]

    # 1. Create memes from events
    for ev in events:
        if not hasattr(ev, 'severity'):
            continue
        creator = rng.choice(active_colonists) if active_colonists else None
        if creator:
            meme = create_meme_from_event(culture, creator.id, year,
                                          ev.name if hasattr(ev, 'name') else str(ev),
                                          ev.severity, rng)
            if meme:
                created.append(meme.to_dict())

    # 2. Create memes from interactions
    death_ids = {d.get("id", "") for d in deaths}
    for inter in interactions:
        if death_ids:
            inter.adjacent_death = inter.adjacent_death or rng.choice(list(death_ids)) if rng.random() < 0.3 else inter.adjacent_death
        meme = create_meme_from_interaction(culture, inter, year, rng)
        if meme:
            created.append(meme.to_dict())

    # 3. Songs from high-faith colonists
    for c in active_colonists:
        if c.stats.faith > 0.7 and rng.random() < 0.08:
            meme = Meme(
                id=culture._gen_id(), meme_type="song",
                content=f"Year {year}: {c.name}'s hymn to the red sky",
                origin_colonist=c.id, origin_year=year,
                accuracy=1.0, believers=[c.id],
                influence={"medicine": SONG_MORALE},
            )
            culture.memes[meme.id] = meme
            culture.history.append({
                "year": year, "event": "created", "meme_id": meme.id,
                "type": "song", "colonist": c.id,
            })
            created.append(meme.to_dict())

    # 4. Transmit existing memes
    transmissions = transmit_memes(culture, active_colonists, social_graph, year, rng)

    # 5. Prune dead colonists from memes
    for d in deaths:
        prune_dead_colonist(culture, d.get("id", ""))

    # 6. Forget memes with no believers
    forgotten = forget_memes(culture, year)

    # 7. Enforce colony cap
    pruned = enforce_colony_cap(culture, len(active_colonists))

    return {
        "year": year,
        "memes_created": len(created),
        "transmissions": len(transmissions),
        "memes_forgotten": len(forgotten),
        "memes_pruned": len(pruned),
        "active_memes": culture.active_meme_count(),
        "created_details": created[:5],  # cap detail output
    }


# --- Internal helpers ---

def _infer_type_from_event(event_name: str, severity: float,
                           rng: random.Random) -> str:
    """Infer meme type from event characteristics."""
    if severity > 0.7:
        return rng.choice(["taboo", "story"])
    if "discovery" in event_name or "resource" in event_name:
        return "innovation"
    if severity < 0.3:
        return "song"
    return "story"


def _generate_content(meme_type: str, event_name: str, year: int,
                      colonist_id: str) -> str:
    """Generate human-readable meme content."""
    templates = {
        "story": f"Year {year}: The tale of {event_name}, as told by {colonist_id}",
        "innovation": f"Year {year}: {colonist_id}'s discovery during {event_name}",
        "taboo": f"Year {year}: Never repeat what happened during {event_name}",
        "ritual": f"Year {year}: The ceremony born from {event_name}",
        "song": f"Year {year}: A song inspired by {event_name}",
    }
    return templates.get(meme_type, f"Year {year}: {event_name}")


def _compute_influence(meme_type: str, event_name: str,
                       severity: float) -> dict[str, float]:
    """Compute influence modifiers for a meme."""
    if meme_type == "taboo":
        # Taboos discourage related actions
        action_map = {
            "dust_storm": "explore",
            "equipment_failure": "code",
            "solar_flare": "terraform",
            "colonist_conflict": "sabotage",
        }
        action = action_map.get(event_name, "explore")
        return {action: -TABOO_PENALTY * severity}
    elif meme_type == "innovation":
        return {"terraform": INNOVATION_BOOST, "farm": INNOVATION_BOOST}
    elif meme_type == "song":
        return {"medicine": SONG_MORALE}
    elif meme_type == "story":
        return {"pray": 0.02, "mediate": 0.02}  # stories encourage reflection
    elif meme_type == "ritual":
        return {"cooperate": 0.05, "trust_boost": RITUAL_BOND}
    return {}


def _weakest_meme(culture: CulturalMemory, colonist_id: str) -> Meme | None:
    """Find the weakest meme known by a colonist (lowest accuracy * believer count)."""
    known = culture.memes_known_by(colonist_id)
    if not known:
        return None
    return min(known, key=lambda m: m.accuracy * len(m.believers))
