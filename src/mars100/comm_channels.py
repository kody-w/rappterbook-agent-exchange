"""
Comm channels organ (engine v12.0).

Tracks the *communication channel* between every pair of active colonists —
who's still talking, who's drifted into silence, who's flatlined, who needs
a revival prompt.

Adapted from the Rappterbook channel_health concept: a channel that hasn't
had a "post" in N+ frames is dead and needs revival. Here a "post" is any
inferred interaction in a given year (cooperate/mediate action, mutual high
trust, shared faction membership).

Pure functions + plain dataclasses. No I/O. Deterministic given a seed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

# ----- tuning constants ----------------------------------------------------

FLATLINE_SILENCE_YEARS = 10
"""Years of zero contact before a channel is officially flatlined.
Matches the Rappterbook archivist's '0 posts in 10+ frames' rule."""

FADING_SILENCE_YEARS = 5
"""Years of zero contact before a channel is 'fading' (early warning)."""

REVIVED_GRACE_YEARS = 3
"""How long a channel keeps the 'revived' badge after its last flatline."""

MAX_REVIVAL_PROMPTS_PER_TICK = 6
"""Cap per year so we don't drown the action chooser."""

EFFICACY_WINDOW_YEARS = 5
"""How many years after a prompt fires we still credit a revival to it.
Beyond this window, a revival counts as 'organic' (no prompt influence).
This closes the feedback loop on whether revival prompts actually work."""

STRONG_TRUST_THRESHOLD = 0.55
"""Trust at/above this counts as a 'passive' contact signal."""

VITAL_VITALITY = 0.55
"""Score above this = 'vital'."""

DORMANT_VITALITY = 0.18
"""Score below this (with no recent contact) = 'dormant'."""

STRONG_CONTACT_ACTIONS = ("cooperate", "mediate")

STATUS_VITAL = "vital"
STATUS_FADING = "fading"
STATUS_FLATLINED = "flatlined"
STATUS_REVIVED = "revived"
STATUS_DORMANT = "dormant"
STATUS_INACTIVE = "inactive"  # one or both colonists gone

REVIVAL_BLUEPRINTS = (
    "{a} and {b} haven't spoken in {n} years — propose a shared task.",
    "Reach out to {b}: the {a}<->{b} channel has flatlined ({n}y silent).",
    "{a}: bring {b} into your next cooperate action — bridge the silence.",
    "Old bond fading: {a} and {b} were close once. {n} years of silence.",
    "Mediator opening — repair {a}<->{b}, the colony's longest dead channel.",
)

# ----- data ----------------------------------------------------------------

def compute_efficacy_rate(state: "CommChannelsState") -> float:
    """Fraction of revivals that occurred within EFFICACY_WINDOW_YEARS of a prompt.

    Returns 0.0 when there have been no revivals yet (nothing to credit).
    1.0 means every revival followed a recent prompt; 0.0 means revivals
    are purely organic. This is the feedback signal the autonomy loop
    uses to know whether revival prompts are pulling their weight.
    """
    total = state.total_prompted_revivals + state.total_organic_revivals
    if total == 0:
        return 0.0
    return round(state.total_prompted_revivals / total, 4)


def pair_key(a: str, b: str) -> tuple[str, str]:
    """Canonical (sorted) pair key — channels are undirected."""
    if a == b:
        raise ValueError("self-channel is meaningless")
    return (a, b) if a < b else (b, a)


@dataclass
class Channel:
    """Per-pair communication state."""
    a: str
    b: str
    born_year: int
    last_contact_year: int
    total_contacts: int = 0
    strong_contacts: int = 0
    silence_streak: int = 0
    vitality: float = 1.0
    status: str = STATUS_VITAL
    flatline_count: int = 0
    revival_count: int = 0
    last_flatline_year: int = -999
    last_revival_year: int = -999
    last_prompted_year: int = -999
    prompt_count: int = 0
    prompted_revivals: int = 0
    organic_revivals: int = 0

    def to_dict(self) -> dict:
        return {
            "a": self.a, "b": self.b,
            "born_year": self.born_year,
            "last_contact_year": self.last_contact_year,
            "total_contacts": self.total_contacts,
            "strong_contacts": self.strong_contacts,
            "silence_streak": self.silence_streak,
            "vitality": round(self.vitality, 4),
            "status": self.status,
            "flatline_count": self.flatline_count,
            "revival_count": self.revival_count,
            "last_flatline_year": self.last_flatline_year,
            "last_revival_year": self.last_revival_year,
            "last_prompted_year": self.last_prompted_year,
            "prompt_count": self.prompt_count,
            "prompted_revivals": self.prompted_revivals,
            "organic_revivals": self.organic_revivals,
        }


@dataclass
class RevivalPrompt:
    """A nudge for the action chooser — try to talk to X."""
    target_a: str
    target_b: str
    text: str
    silence_years: int
    suggested_action: str
    year: int

    def to_dict(self) -> dict:
        return {
            "target_a": self.target_a, "target_b": self.target_b,
            "text": self.text, "silence_years": self.silence_years,
            "suggested_action": self.suggested_action, "year": self.year,
        }


@dataclass
class CommChannelsState:
    """Persistent comm-channel state, lives on the engine."""
    channels: dict = field(default_factory=dict)  # tuple[str,str] -> Channel
    revival_log: list = field(default_factory=list)
    flatline_log: list = field(default_factory=list)
    total_prompts_fired: int = 0
    total_prompted_revivals: int = 0
    total_organic_revivals: int = 0

    def to_dict(self) -> dict:
        return {
            "channels": {f"{k[0]}|{k[1]}": v.to_dict()
                          for k, v in self.channels.items()},
            "revival_log": list(self.revival_log[-50:]),
            "flatline_log": list(self.flatline_log[-50:]),
            "total_prompts_fired": self.total_prompts_fired,
            "total_prompted_revivals": self.total_prompted_revivals,
            "total_organic_revivals": self.total_organic_revivals,
            "prompt_efficacy_rate": compute_efficacy_rate(self),
        }


@dataclass
class CommChannelsTickResult:
    """Per-year output."""
    year: int
    new_channels: list
    flatlined: list
    revived: list
    fading: list
    revival_prompts: list
    summary: dict
    health_score: float
    dead_channel_names: list

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "new_channels": list(self.new_channels),
            "flatlined": list(self.flatlined),
            "revived": list(self.revived),
            "fading": list(self.fading),
            "revival_prompts": [p.to_dict() if hasattr(p, "to_dict") else p
                                 for p in self.revival_prompts],
            "summary": dict(self.summary),
            "health_score": round(self.health_score, 4),
            "dead_channel_names": list(self.dead_channel_names),
        }


# ----- pure helpers --------------------------------------------------------

def compute_vitality(channel: Channel, current_year: int) -> float:
    """Vitality = recency * frequency + bonuses, clamped [0,1].

    - Recency decays exponentially (silence streak / 10 = half-life-ish).
    - Frequency rewards total contacts (log scale, saturates around 20).
    - Strong-ratio adds up to +0.15 for high cooperate/mediate share.
    - Fresh channels (<= 3 years) get a +0.10 nursery bonus.
    """
    recency = math.exp(-channel.silence_streak / 10.0)
    freq = math.log1p(channel.total_contacts) / math.log1p(20.0)
    if channel.total_contacts > 0:
        strong_ratio = channel.strong_contacts / channel.total_contacts
    else:
        strong_ratio = 0.0
    age = current_year - channel.born_year
    nursery = 0.10 if age <= 3 else 0.0
    score = 0.6 * recency + 0.3 * freq + 0.15 * strong_ratio + nursery
    return max(0.0, min(1.0, score))


def classify_status(channel: Channel, both_active: bool,
                     current_year: int) -> str:
    """Pick the channel's status label from its current vitals."""
    if not both_active:
        return STATUS_INACTIVE
    # 'Revived' badge takes priority for a few years after revival
    if (channel.last_revival_year >= 0
            and current_year - channel.last_revival_year < REVIVED_GRACE_YEARS
            and channel.silence_streak < FADING_SILENCE_YEARS):
        return STATUS_REVIVED
    if channel.silence_streak >= FLATLINE_SILENCE_YEARS:
        return STATUS_FLATLINED
    if channel.silence_streak >= FADING_SILENCE_YEARS:
        return STATUS_FADING
    if channel.vitality >= VITAL_VITALITY:
        return STATUS_VITAL
    if channel.vitality < DORMANT_VITALITY:
        return STATUS_DORMANT
    return STATUS_VITAL


def infer_contacts(
    active_ids: list,
    actions: dict,
    social_get: Callable,
    faction_membership: dict,
    year: int,
) -> dict:
    """Return {(a,b): (strength, is_strong)} for every pair with contact.

    Strength is 1 or 2; is_strong means at least one cooperate/mediate
    action linked the pair.

    Signals (additive, capped at 2):
      - Both colonists used a STRONG action (cooperate/mediate): +2
      - Either colonist used a STRONG action:                   +1
      - max(trust(a,b), trust(b,a)) >= STRONG_TRUST_THRESHOLD:  +1
      - Same faction (sampled on even years):                   +1
    """
    contacts: dict = {}
    n = len(active_ids)
    for i in range(n):
        a = active_ids[i]
        a_strong = actions.get(a) in STRONG_CONTACT_ACTIONS
        for j in range(i + 1, n):
            b = active_ids[j]
            b_strong = actions.get(b) in STRONG_CONTACT_ACTIONS
            strength = 0
            is_strong = False
            if a_strong and b_strong:
                strength += 2
                is_strong = True
            elif a_strong or b_strong:
                strength += 1
                is_strong = True
            try:
                rel_ab = social_get(a, b)
                rel_ba = social_get(b, a)
                t = max(getattr(rel_ab, "trust", 0.0),
                        getattr(rel_ba, "trust", 0.0))
            except Exception:
                t = 0.0
            if t >= STRONG_TRUST_THRESHOLD:
                strength += 1
            if year % 2 == 0:
                fa = faction_membership.get(a)
                fb = faction_membership.get(b)
                if fa is not None and fa == fb:
                    strength += 1
            if strength > 0:
                contacts[pair_key(a, b)] = (min(strength, 2), is_strong)
    return contacts


def generate_revival_prompt(channel: Channel, year: int,
                             rng) -> RevivalPrompt:
    """Make a deterministic revival prompt for a flatlined channel."""
    blueprint = REVIVAL_BLUEPRINTS[rng.randrange(len(REVIVAL_BLUEPRINTS))]
    text = blueprint.format(a=channel.a, b=channel.b,
                             n=channel.silence_streak)
    suggested = "cooperate" if channel.silence_streak < 15 else "mediate"
    return RevivalPrompt(
        target_a=channel.a, target_b=channel.b, text=text,
        silence_years=channel.silence_streak,
        suggested_action=suggested, year=year)


def compute_revival_pressure(state: CommChannelsState,
                              actions_pool: list) -> dict:
    """Per-action nudge for the action chooser, summed across all prompts.

    Returns {action_name: pressure}. Pressure is small (a few hundredths).
    """
    pressure = {a: 0.0 for a in actions_pool}
    if not state.revival_log:
        return pressure
    recent = state.revival_log[-MAX_REVIVAL_PROMPTS_PER_TICK:]
    for prompt in recent:
        act = prompt["suggested_action"] if isinstance(prompt, dict) \
            else prompt.suggested_action
        if act in pressure:
            pressure[act] += 0.04
    return pressure


def compute_colony_comm_health(state: CommChannelsState,
                                active_ids: list) -> float:
    """Mean vitality across active-pair channels, in [0,1]."""
    if not active_ids:
        return 1.0
    active_set = set(active_ids)
    vitalities = [ch.vitality for k, ch in state.channels.items()
                   if k[0] in active_set and k[1] in active_set]
    if not vitalities:
        return 1.0
    return sum(vitalities) / len(vitalities)


# ----- main tick -----------------------------------------------------------

def tick_comm_channels(
    state: CommChannelsState,
    active_ids: list,
    actions: dict,
    social_get: Callable,
    faction_membership: dict,
    year: int,
    rng,
) -> CommChannelsTickResult:
    """Mutate state by one year. Returns per-year report."""
    active_set = set(active_ids)
    contacts = infer_contacts(active_ids, actions, social_get,
                               faction_membership, year)

    new_channels: list = []
    flatlined: list = []
    revived: list = []
    fading: list = []
    dead_names: list = []

    # 1. Ensure every active pair has a channel
    for i, a in enumerate(active_ids):
        for b in active_ids[i + 1:]:
            key = pair_key(a, b)
            if key not in state.channels:
                state.channels[key] = Channel(
                    a=key[0], b=key[1], born_year=year,
                    last_contact_year=year,
                    total_contacts=0, strong_contacts=0,
                    silence_streak=0, vitality=1.0, status=STATUS_VITAL)
                new_channels.append("{}|{}".format(key[0], key[1]))

    # 2. Update each channel
    for key, ch in state.channels.items():
        both_active = (key[0] in active_set) and (key[1] in active_set)
        prior_status = ch.status

        if both_active and key in contacts:
            strength, is_strong = contacts[key]
            ch.total_contacts += 1
            if is_strong:
                ch.strong_contacts += 1
            was_silent = ch.silence_streak
            ch.last_contact_year = year
            ch.silence_streak = 0
            # Revival check: came back from flatline?
            if was_silent >= FLATLINE_SILENCE_YEARS:
                ch.revival_count += 1
                ch.last_revival_year = year
                # Attribute revival to a prior prompt if one fired recently.
                prompted = (ch.last_prompted_year >= 0 and
                            (year - ch.last_prompted_year)
                            <= EFFICACY_WINDOW_YEARS)
                if prompted:
                    ch.prompted_revivals += 1
                    state.total_prompted_revivals += 1
                else:
                    ch.organic_revivals += 1
                    state.total_organic_revivals += 1
                revived.append("{}|{}".format(key[0], key[1]))
        else:
            # No contact this year — bump silence if both still active
            if both_active:
                ch.silence_streak += 1

        ch.vitality = compute_vitality(ch, year)
        ch.status = classify_status(ch, both_active, year)

        # Newly fading?
        if (both_active and ch.status == STATUS_FADING
                and prior_status not in (STATUS_FADING, STATUS_FLATLINED)):
            fading.append("{}|{}".format(key[0], key[1]))

        # Newly flatlined?
        if (both_active and ch.status == STATUS_FLATLINED
                and prior_status != STATUS_FLATLINED):
            ch.flatline_count += 1
            ch.last_flatline_year = year
            flatlined.append("{}|{}".format(key[0], key[1]))
            state.flatline_log.append({
                "pair": "{}|{}".format(key[0], key[1]),
                "year": year, "silence_years": ch.silence_streak,
            })

        if ch.status == STATUS_FLATLINED:
            dead_names.append("{}|{}".format(key[0], key[1]))

    # 3. Generate revival prompts for the most urgent flatlines
    flatlined_channels = [state.channels[k] for k in state.channels
                           if state.channels[k].status == STATUS_FLATLINED
                           and k[0] in active_set and k[1] in active_set]
    flatlined_channels.sort(
        key=lambda c: (-c.silence_streak, c.a, c.b))
    revival_prompts: list = []
    for ch in flatlined_channels[:MAX_REVIVAL_PROMPTS_PER_TICK]:
        prompt = generate_revival_prompt(ch, year, rng)
        revival_prompts.append(prompt)
        state.revival_log.append(prompt.to_dict())
        # Mark the channel so we can credit any near-future revival.
        ch.last_prompted_year = year
        ch.prompt_count += 1
        state.total_prompts_fired += 1
    state.revival_log = state.revival_log[-200:]
    state.flatline_log = state.flatline_log[-200:]

    # 4. Summary
    summary: dict = {}
    for ch in state.channels.values():
        summary[ch.status] = summary.get(ch.status, 0) + 1
    health = compute_colony_comm_health(state, active_ids)

    return CommChannelsTickResult(
        year=year, new_channels=new_channels, flatlined=flatlined,
        revived=revived, fading=fading,
        revival_prompts=revival_prompts, summary=summary,
        health_score=health, dead_channel_names=dead_names)
