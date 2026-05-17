"""
Channels organ for Mars-100 (engine v12.0).

Models the colony's emergent **discourse channels** — semi-persistent
discussion fora that form around factions, actions, and shared topics.

Each tick a channel either GROWS (members post, vitality rises) or
DECAYS (silence, vitality falls). Channels with zero activity for
FLATLINE_YEARS are flagged as **flatlined**. The archivist
(this organ) then emits **revival prompts** — proposals nudging a
plausible leader to restart conversation.

This mirrors the seed mission from the parent network:
> "monitor state/channels.json for dead channels (0 posts in 10+ frames)
>  and generate revival prompts for the autonomy loop"

Translated to the colony: dead discourse channels are habitat dark
spots. Reviving them restores cultural memory and faction cohesion.

One-year lag: channels tick AFTER diplomacy so faction membership
matches end-of-year survivors. Output feeds into NEXT year's
colonist action choices (via revival_prompts).

RNG offset: seed + 13841
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many silent years before a channel is flatlined.
FLATLINE_YEARS = 10

# Vitality dynamics — bounded in [0.0, 1.0].
VITALITY_MAX = 1.0
VITALITY_MIN = 0.0
VITALITY_BIRTH = 0.55
VITALITY_DECAY_PER_SILENT_YEAR = 0.07
VITALITY_GAIN_PER_POST = 0.04
VITALITY_GAIN_PER_MEMBER = 0.005
DEAD_VITALITY_THRESHOLD = 0.10

# Minimum members for a channel to be considered alive at all.
MIN_MEMBERS = 2

# Topic channels emerge when this many colonists share an action in one year.
TOPIC_EMERGENCE_THRESHOLD = 3

# Revival prompts — caps keep the autonomy loop polite.
MAX_REVIVAL_PROMPTS_PER_YEAR = 5
REVIVAL_COOLDOWN_YEARS = 3  # don't re-prompt the same channel too often

# Auto-archive after this many years of being flatlined with no revival.
ARCHIVE_AFTER_FLATLINE_YEARS = 25

# Channel kinds.
KIND_FACTION = "faction"
KIND_TOPIC = "topic"

# Map of action -> human-readable topic name for emergent channels.
ACTION_TOPICS = {
    "terraform": "Terraform Talk",
    "farm": "The Greenhouse",
    "mediate": "Peace Circle",
    "code": "The Forge",
    "pray": "Vespers",
    "sabotage": "The Shadow Net",
    "cooperate": "Mutual Aid",
    "hoard": "Stockpile",
    "explore": "Far Horizons",
    "rest": "The Hearth",
    "research": "Lab Notes",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Channel:
    """A discourse channel — like a subrappter for the colony."""
    id: str
    name: str
    kind: str                     # KIND_FACTION | KIND_TOPIC
    faction_id: str | None
    topic: str | None
    members: list[str]
    founded_year: int
    last_active_year: int
    vitality: float = VITALITY_BIRTH
    total_posts: int = 0
    posts_last_year: int = 0
    flatlined: bool = False
    flatlined_since: int | None = None
    last_revival_year: int | None = None
    revival_attempts: int = 0
    archived: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "faction_id": self.faction_id,
            "topic": self.topic,
            "members": list(self.members),
            "founded_year": self.founded_year,
            "last_active_year": self.last_active_year,
            "vitality": round(self.vitality, 4),
            "total_posts": self.total_posts,
            "posts_last_year": self.posts_last_year,
            "flatlined": self.flatlined,
            "flatlined_since": self.flatlined_since,
            "last_revival_year": self.last_revival_year,
            "revival_attempts": self.revival_attempts,
            "archived": self.archived,
        }


@dataclass
class ChannelsState:
    """Persistent state of the discourse organ."""
    channels: dict[str, Channel] = field(default_factory=dict)
    next_channel_seq: int = 1

    def to_dict(self) -> dict:
        return {
            "channels": {cid: c.to_dict() for cid, c in self.channels.items()},
            "next_channel_seq": self.next_channel_seq,
        }


@dataclass
class ChannelHealthResult:
    """What happened to discourse in one year."""
    year: int
    channels_created: list[dict] = field(default_factory=list)
    channels_flatlined: list[dict] = field(default_factory=list)
    channels_revived: list[dict] = field(default_factory=list)
    channels_archived: list[dict] = field(default_factory=list)
    revival_prompts: list[dict] = field(default_factory=list)
    vitals: list[dict] = field(default_factory=list)

    @property
    def alive_count(self) -> int:
        return sum(1 for v in self.vitals if not v.get("flatlined") and not v.get("archived"))

    @property
    def flatlined_count(self) -> int:
        return sum(1 for v in self.vitals if v.get("flatlined") and not v.get("archived"))

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "channels_created": list(self.channels_created),
            "channels_flatlined": list(self.channels_flatlined),
            "channels_revived": list(self.channels_revived),
            "channels_archived": list(self.channels_archived),
            "revival_prompts": list(self.revival_prompts),
            "alive_count": self.alive_count,
            "flatlined_count": self.flatlined_count,
            "vitals": list(self.vitals),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = VITALITY_MIN, hi: float = VITALITY_MAX) -> float:
    return max(lo, min(hi, x))


def _next_id(state: ChannelsState, prefix: str) -> str:
    cid = f"{prefix}-{state.next_channel_seq:04d}"
    state.next_channel_seq += 1
    return cid


def _make_faction_channel(state: ChannelsState, faction: dict, year: int) -> Channel:
    cid = _next_id(state, "ch-fx")
    name = f"#{faction.get('name', 'faction').lower().replace(' ', '-')}"
    return Channel(
        id=cid,
        name=name,
        kind=KIND_FACTION,
        faction_id=faction.get("id"),
        topic=faction.get("ideology"),
        members=list(faction.get("members", [])),
        founded_year=year,
        last_active_year=year,
    )


def _make_topic_channel(state: ChannelsState, action: str,
                        actors: list[str], year: int) -> Channel:
    cid = _next_id(state, "ch-tp")
    label = ACTION_TOPICS.get(action, action.title())
    return Channel(
        id=cid,
        name=f"#{action}",
        kind=KIND_TOPIC,
        faction_id=None,
        topic=label,
        members=list(actors),
        founded_year=year,
        last_active_year=year,
    )


def _existing_faction_channel(state: ChannelsState, faction_id: str) -> Channel | None:
    for ch in state.channels.values():
        if ch.kind == KIND_FACTION and ch.faction_id == faction_id and not ch.archived:
            return ch
    return None


def _existing_topic_channel(state: ChannelsState, action: str) -> Channel | None:
    name = f"#{action}"
    for ch in state.channels.values():
        if ch.kind == KIND_TOPIC and ch.name == name and not ch.archived:
            return ch
    return None


def _pick_leader(channel: Channel, active_ids: set,
                 social_get: Callable | None) -> str | None:
    """Pick the most-connected living member to nudge for revival."""
    eligible = [m for m in channel.members if m in active_ids]
    if not eligible:
        return None
    if social_get is None:
        return eligible[0]
    best_id, best_score = eligible[0], -math.inf
    for cid in eligible:
        score = 0.0
        for other in eligible:
            if other == cid:
                continue
            try:
                score += float(social_get(cid, other) or 0.0)
            except Exception:
                pass
        if score > best_score:
            best_id, best_score = cid, score
    return best_id


def _revival_prompt(channel: Channel, leader_id: str | None, year: int) -> dict:
    label = channel.topic or channel.name
    if leader_id:
        text = (f"The archivist flags {channel.name} as flatlined. "
                f"{leader_id}, you carry the channel — speak on {label}.")
    else:
        text = (f"The archivist flags {channel.name} as flatlined. "
                f"No living member remains; the channel awaits adoption.")
    return {
        "channel_id": channel.id,
        "channel_name": channel.name,
        "leader_id": leader_id,
        "year": year,
        "kind": channel.kind,
        "vitality": round(channel.vitality, 4),
        "silent_years": year - channel.last_active_year,
        "topic": channel.topic,
        "prompt": text,
    }


# ---------------------------------------------------------------------------
# Tick
# ---------------------------------------------------------------------------

def tick_channels(
    state: ChannelsState,
    factions: list,
    actions: dict,
    active_colonist_ids: list,
    year: int,
    social_get: Callable | None,
    rng,
) -> ChannelHealthResult:
    """Advance the discourse organ by one year.

    Args:
        state: persistent ChannelsState (mutated in place)
        factions: list of faction dicts from diplomacy state
        actions: {colonist_id: action_name} for this year
        active_colonist_ids: list of currently-living colonist ids
        year: current simulation year
        social_get: callable(a, b) -> trust score in [-1, 1], or None
        rng: dedicated random.Random instance for this organ

    Returns:
        ChannelHealthResult summarising what changed.
    """
    result = ChannelHealthResult(year=year)
    active_set = set(active_colonist_ids)

    # 1) Faction-channel mirroring: every faction has exactly one channel.
    for fac in factions:
        fid = fac.get("id")
        if not fid:
            continue
        existing = _existing_faction_channel(state, fid)
        members = [m for m in fac.get("members", []) if m in active_set]
        if existing is None:
            if len(members) < MIN_MEMBERS:
                continue
            ch = _make_faction_channel(state, fac, year)
            ch.members = members
            state.channels[ch.id] = ch
            result.channels_created.append({
                "id": ch.id, "name": ch.name, "kind": ch.kind,
                "faction_id": fid, "founded_year": year,
            })
        else:
            existing.members = members

    # 2) Topic-channel emergence: cluster actions.
    action_actors: dict = {}
    for cid, act in (actions or {}).items():
        if cid not in active_set or not act:
            continue
        action_actors.setdefault(act, []).append(cid)

    for act, actors in action_actors.items():
        if len(actors) < TOPIC_EMERGENCE_THRESHOLD:
            continue
        existing = _existing_topic_channel(state, act)
        if existing is None:
            ch = _make_topic_channel(state, act, actors, year)
            state.channels[ch.id] = ch
            result.channels_created.append({
                "id": ch.id, "name": ch.name, "kind": ch.kind,
                "topic": ch.topic, "founded_year": year,
            })
        else:
            roster = [m for m in existing.members if m in active_set]
            for a in actors:
                if a not in roster:
                    roster.append(a)
            existing.members = roster

    # 3) Activity accounting.
    for ch in state.channels.values():
        if ch.archived:
            continue
        live_members = [m for m in ch.members if m in active_set]
        ch.members = live_members

        posts = 0
        if ch.kind == KIND_FACTION:
            fac = next((f for f in factions if f.get("id") == ch.faction_id), None)
            cohesion = float(fac.get("cohesion", 0.0)) if fac else 0.0
            if fac and len(live_members) >= MIN_MEMBERS:
                base = int(round(len(live_members) * (0.4 + cohesion)))
                posts = max(0, base + rng.randint(-1, 2))
        else:  # KIND_TOPIC
            actors_this_year = action_actors.get(ch.name.lstrip("#"), [])
            doer_count = sum(1 for m in live_members if m in actors_this_year)
            if doer_count >= 1:
                posts = doer_count + rng.randint(0, 1)

        ch.posts_last_year = posts
        if posts > 0:
            ch.total_posts += posts
            ch.last_active_year = year
            gain = (posts * VITALITY_GAIN_PER_POST
                    + len(live_members) * VITALITY_GAIN_PER_MEMBER)
            ch.vitality = _clamp(ch.vitality + gain)
            if ch.flatlined:
                ch.flatlined = False
                ch.flatlined_since = None
                ch.revival_attempts = 0
                result.channels_revived.append({
                    "id": ch.id, "name": ch.name, "year": year,
                    "posts": posts, "vitality": round(ch.vitality, 4),
                })
        else:
            ch.vitality = _clamp(ch.vitality - VITALITY_DECAY_PER_SILENT_YEAR)

    # 4) Flatline detection.
    for ch in state.channels.values():
        if ch.archived or ch.flatlined:
            continue
        silent_years = year - ch.last_active_year
        too_quiet = silent_years >= FLATLINE_YEARS
        too_weak = ch.vitality <= DEAD_VITALITY_THRESHOLD
        too_small = len(ch.members) < MIN_MEMBERS
        if too_quiet or too_weak or too_small:
            ch.flatlined = True
            ch.flatlined_since = year
            result.channels_flatlined.append({
                "id": ch.id, "name": ch.name, "year": year,
                "silent_years": silent_years,
                "vitality": round(ch.vitality, 4),
                "reason": ("too_quiet" if too_quiet
                           else "too_weak" if too_weak
                           else "too_small"),
            })

    # 5) Revival prompts — sorted by severity.
    candidates: list = []
    for ch in state.channels.values():
        if not ch.flatlined or ch.archived:
            continue
        last = ch.last_revival_year
        if last is not None and (year - last) < REVIVAL_COOLDOWN_YEARS:
            continue
        if not any(m in active_set for m in ch.members):
            continue
        candidates.append(ch)

    candidates.sort(
        key=lambda c: (year - c.last_active_year, -c.vitality),
        reverse=True,
    )

    for ch in candidates[:MAX_REVIVAL_PROMPTS_PER_YEAR]:
        leader = _pick_leader(ch, active_set, social_get)
        prompt = _revival_prompt(ch, leader, year)
        result.revival_prompts.append(prompt)
        ch.last_revival_year = year
        ch.revival_attempts += 1

    # 6) Auto-archive long-dead channels.
    for ch in state.channels.values():
        if ch.archived or not ch.flatlined:
            continue
        if ch.flatlined_since is None:
            continue
        if (year - ch.flatlined_since) >= ARCHIVE_AFTER_FLATLINE_YEARS:
            ch.archived = True
            result.channels_archived.append({
                "id": ch.id, "name": ch.name, "year": year,
                "flatlined_since": ch.flatlined_since,
                "total_posts": ch.total_posts,
            })

    # 7) Per-channel vitals snapshot.
    for ch in state.channels.values():
        result.vitals.append(ch.to_dict())

    return result


# ---------------------------------------------------------------------------
# Pressure on actions
# ---------------------------------------------------------------------------

def compute_revival_pressure(prompts: list) -> dict:
    """Translate revival prompts into per-colonist action nudges.

    Returns: {colonist_id: {action: bonus}} where bonus is in [0, 0.15].
    The autonomy loop applies these as soft pressure on _choose_action().
    """
    out: dict = {}
    for p in prompts:
        leader = p.get("leader_id")
        if not leader:
            continue
        kind = p.get("kind")
        if kind == KIND_TOPIC:
            action = (p.get("channel_name") or "").lstrip("#")
            if action:
                out.setdefault(leader, {})[action] = 0.10
        else:  # KIND_FACTION
            slot = out.setdefault(leader, {})
            slot["mediate"] = max(slot.get("mediate", 0.0), 0.08)
            slot["cooperate"] = max(slot.get("cooperate", 0.0), 0.05)
    return out
