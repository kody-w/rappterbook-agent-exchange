"""
Vitals organ for Mars-100 (engine v12.0).

Homeostasis monitor. Reads ALL subsystem state each year and tracks
per-channel vitality — births, governance, factions, economics, ecology,
psychology, infrastructure, culture, earth comms. Flags channels that
have flatlined (no detectable change for N years) and emits revival
prompts the engine can surface to the next tick.

The archivist analogue: think of each subsystem as a channel in
``state/channels.json``. When a channel goes silent for 10+ frames, the
vitals organ raises a revival flag so the autonomy loop can nudge
colonists toward action there.

The organ NEVER mutates other subsystems. It only OBSERVES and reports.
Revival prompts are advisory metadata for the autonomy loop and the
narrator — they describe what's dying, not what to do about it. The
engine consumes them via ``compute_action_nudges`` to bias next year's
action selection toward dark channels (gentle nudge, capped).

RNG offset: seed + 13499 (kept for parity with sibling organs even
though vitals is fully deterministic given input state).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLATLINE_THRESHOLD = 10
DYING_THRESHOLD = 0.25
HEALTHY_THRESHOLD = 0.60
PULSE_HALF_LIFE = 6.0
MAX_ACTION_NUDGE = 0.12

CHANNEL_NAMES = (
    "resources",
    "births",
    "governance",
    "infrastructure",
    "culture",
    "economics",
    "diplomacy",
    "ecology",
    "psychology",
    "earth",
    "social",
)

_CHANNEL_TO_ACTION_HINTS: dict[str, tuple[str, ...]] = {
    "resources":      ("farm", "terraform"),
    "births":         ("cooperate", "mediate"),
    "governance":     ("mediate", "cooperate"),
    "infrastructure": ("code", "terraform"),
    "culture":        ("pray", "mediate"),
    "economics":      ("cooperate", "farm"),
    "diplomacy":      ("mediate", "cooperate"),
    "ecology":        ("terraform", "farm"),
    "psychology":     ("rest", "mediate"),
    "earth":          ("code", "explore"),
    "social":         ("cooperate", "mediate"),
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ChannelVitals:
    """Vitals for one tracked channel."""
    name: str
    pulse: float = 0.0
    silent_years: int = 0
    last_signal_year: int | None = None
    total_signals: int = 0
    revivals_emitted: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pulse": round(self.pulse, 4),
            "silent_years": self.silent_years,
            "last_signal_year": self.last_signal_year,
            "total_signals": self.total_signals,
            "revivals_emitted": self.revivals_emitted,
            "status": self.status(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ChannelVitals:
        return cls(
            name=d["name"],
            pulse=d.get("pulse", 0.0),
            silent_years=d.get("silent_years", 0),
            last_signal_year=d.get("last_signal_year"),
            total_signals=d.get("total_signals", 0),
            revivals_emitted=d.get("revivals_emitted", 0),
        )

    def status(self) -> str:
        if self.silent_years >= FLATLINE_THRESHOLD:
            return "flatlined"
        if self.pulse < DYING_THRESHOLD:
            return "dying"
        if self.pulse >= HEALTHY_THRESHOLD:
            return "healthy"
        return "stable"


@dataclass
class RevivalPrompt:
    """Advisory directive emitted when a channel goes dark."""
    channel: str
    year: int
    silent_years: int
    pulse: float
    severity: str
    suggestion: str

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "year": self.year,
            "silent_years": self.silent_years,
            "pulse": round(self.pulse, 4),
            "severity": self.severity,
            "suggestion": self.suggestion,
        }


@dataclass
class VitalsState:
    """Colony-wide homeostasis state."""
    channels: dict[str, ChannelVitals] = field(default_factory=dict)
    prompt_history: list[dict] = field(default_factory=list)
    last_overall: float = 0.0

    def __post_init__(self) -> None:
        for name in CHANNEL_NAMES:
            self.channels.setdefault(name, ChannelVitals(name=name))

    def to_dict(self) -> dict:
        return {
            "channels": {n: c.to_dict() for n, c in self.channels.items()},
            "overall_vitality": round(self.last_overall, 4),
            "flatlined_count": sum(
                1 for c in self.channels.values()
                if c.silent_years >= FLATLINE_THRESHOLD),
            "dying_count": sum(
                1 for c in self.channels.values()
                if c.pulse < DYING_THRESHOLD
                and c.silent_years < FLATLINE_THRESHOLD),
        }

    @classmethod
    def from_dict(cls, d: dict) -> VitalsState:
        state = cls()
        for name, cd in d.get("channels", {}).items():
            state.channels[name] = ChannelVitals.from_dict(cd)
        state.last_overall = d.get("overall_vitality", 0.0)
        return state


@dataclass
class VitalsTickResult:
    """Result of one year of vitals monitoring."""
    year: int
    pulses: dict[str, float] = field(default_factory=dict)
    revivals: list[dict] = field(default_factory=list)
    overall_vitality: float = 0.0
    flatlined: list[str] = field(default_factory=list)
    dying: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "pulses": {k: round(v, 4) for k, v in self.pulses.items()},
            "revivals": self.revivals,
            "overall_vitality": round(self.overall_vitality, 4),
            "flatlined": list(self.flatlined),
            "dying": list(self.dying),
        }


# ---------------------------------------------------------------------------
# Signal extraction — one detector per channel
# ---------------------------------------------------------------------------

def _signal_resources(year_result: dict) -> float:
    delta = year_result.get("resource_delta", {}) or {}
    total = sum(abs(v) for v in delta.values() if isinstance(v, (int, float)))
    return min(1.0, total / 10.0)


def _signal_births(year_result: dict) -> float:
    n = len(year_result.get("births", []) or [])
    n += len(year_result.get("immigrants", []) or [])
    return min(1.0, n / 2.0)


def _signal_governance(year_result: dict) -> float:
    return 1.0 if year_result.get("governance") else 0.0


def _signal_infrastructure(year_result: dict) -> float:
    infra = year_result.get("infrastructure", {}) or {}
    active = infra.get("active_projects", []) or []
    completed_this_year = infra.get("completed_this_year", 0) or 0
    if completed_this_year:
        return 1.0
    return min(1.0, 0.3 * len(active))


def _signal_culture(year_result: dict) -> float:
    culture = year_result.get("culture", {}) or {}
    new_traditions = culture.get("new_traditions", 0) or 0
    new_martyrs = culture.get("new_martyrs", 0) or 0
    new_taboos = culture.get("new_taboos", 0) or 0
    raw = new_traditions + new_martyrs + new_taboos
    return min(1.0, raw / 2.0)


def _signal_economics(year_result: dict) -> float:
    econ = year_result.get("economics", {}) or {}
    trades = len(econ.get("trades", []) or [])
    labor = econ.get("total_labor_income", 0) or 0
    if trades:
        return min(1.0, 0.4 + 0.15 * trades)
    if labor > 0:
        return 0.35
    return 0.0


def _signal_diplomacy(year_result: dict) -> float:
    diplo = year_result.get("diplomacy", {}) or {}
    formed = len(diplo.get("factions_formed", []) or [])
    dissolved = len(diplo.get("factions_dissolved", []) or [])
    alliances_f = len(diplo.get("alliances_formed", []) or [])
    alliances_b = len(diplo.get("alliances_broken", []) or [])
    schisms = len(diplo.get("schisms", []) or [])
    leader_ch = len(diplo.get("leader_changes", []) or [])
    raw = formed + dissolved + alliances_f + alliances_b + schisms + leader_ch
    if raw:
        return min(1.0, 0.5 + 0.1 * raw)
    if diplo.get("faction_count", 0):
        return 0.15
    return 0.0


def _signal_ecology(year_result: dict,
                     prev_biosphere: float | None) -> tuple[float, float]:
    """Return (signal, current_biosphere). First call has no baseline -> 0."""
    eco = year_result.get("ecology", {}) or {}
    biosphere = eco.get("biosphere_index", 0.0) or 0.0
    if prev_biosphere is None:
        return (0.0, biosphere)
    delta = abs(biosphere - prev_biosphere)
    return (min(1.0, delta * 20.0), biosphere)


def _signal_psychology(year_result: dict) -> float:
    psy = year_result.get("psychology", {}) or {}
    crises = len(psy.get("crises", []) or [])
    if crises:
        return min(1.0, 0.5 + 0.2 * crises)
    morale_change = abs(psy.get("morale_delta", 0.0) or 0.0)
    return min(1.0, morale_change * 4.0)


def _signal_earth(year_result: dict) -> float:
    earth_events = year_result.get("earth_events", []) or []
    if not earth_events:
        return 0.0
    raw = 0.0
    for ev in earth_events:
        if not isinstance(ev, dict):
            continue
        if ev.get("ship_arrived") or ev.get("ship_launched"):
            raw += 0.6
        msgs = ev.get("messages", []) or []
        raw += 0.2 * len(msgs)
    return min(1.0, raw)


def _signal_social(year_result: dict,
                    prev_cohesion: float | None) -> tuple[float, float]:
    """Return (signal, current_cohesion). First call has no baseline -> 0."""
    cohesion = year_result.get("social_cohesion", 0.0) or 0.0
    if prev_cohesion is None:
        return (0.0, cohesion)
    delta = abs(cohesion - prev_cohesion)
    return (min(1.0, delta * 5.0), cohesion)


# ---------------------------------------------------------------------------
# Revival prompt templates
# ---------------------------------------------------------------------------

_SUGGESTIONS: dict[str, str] = {
    "resources":      "no measurable resource flux — stockpiles stagnant; spur production or trade",
    "births":         "no new colonists arriving by birth or immigration; reopen Earth lanes or encourage pairings",
    "governance":     "no proposals raised; surface a governance prompt or schedule an assembly",
    "infrastructure": "no project progress; queue a build the colony can afford",
    "culture":        "no traditions, martyrs, or taboos forming; commemorate a recent event",
    "economics":      "no trade and no labor income; restart the market or seed a small grant",
    "diplomacy":      "no faction activity; nudge a high-cohesion cluster toward formation",
    "ecology":        "biosphere index frozen; tend a greenhouse or rotate biomes",
    "psychology":     "no crises and no morale movement; check on the bottom quartile",
    "earth":          "Earth has gone silent; signal home or prepare an independence vote",
    "social":         "cohesion has plateaued; pair newcomers with elders",
}


def _build_revival(channel: ChannelVitals, year: int) -> RevivalPrompt:
    severity = "critical" if channel.silent_years >= 2 * FLATLINE_THRESHOLD else "warn"
    return RevivalPrompt(
        channel=channel.name,
        year=year,
        silent_years=channel.silent_years,
        pulse=channel.pulse,
        severity=severity,
        suggestion=_SUGGESTIONS.get(channel.name, "channel dark — investigate"),
    )


# ---------------------------------------------------------------------------
# Tick
# ---------------------------------------------------------------------------

def _decay_factor() -> float:
    """Per-year multiplicative decay derived from PULSE_HALF_LIFE."""
    return 0.5 ** (1.0 / PULSE_HALF_LIFE)


def tick_vitals(state: VitalsState,
                year: int,
                year_result: dict,
                prev_ecology_biosphere: float | None = None,
                prev_social_cohesion: float | None = None) -> VitalsTickResult:
    """Run one year of vitals monitoring.

    Pure observation. Does not mutate ``year_result`` or any other subsystem.
    Mutates ``state`` (channel pulses, silent counters, prompt history).
    """
    decay = _decay_factor()
    pulses: dict[str, float] = {}

    eco_signal, _new_biosphere = _signal_ecology(year_result, prev_ecology_biosphere)
    soc_signal, _new_cohesion = _signal_social(year_result, prev_social_cohesion)
    raw_signals: dict[str, float] = {
        "resources":      _signal_resources(year_result),
        "births":         _signal_births(year_result),
        "governance":     _signal_governance(year_result),
        "infrastructure": _signal_infrastructure(year_result),
        "culture":        _signal_culture(year_result),
        "economics":      _signal_economics(year_result),
        "diplomacy":      _signal_diplomacy(year_result),
        "ecology":        eco_signal,
        "psychology":     _signal_psychology(year_result),
        "earth":          _signal_earth(year_result),
        "social":         soc_signal,
    }

    revivals_this_year: list[RevivalPrompt] = []

    for name in CHANNEL_NAMES:
        ch = state.channels[name]
        signal = raw_signals.get(name, 0.0)
        ch.pulse = min(1.0, ch.pulse * decay + signal * (1.0 - decay))
        if signal > 0.05:
            ch.silent_years = 0
            ch.last_signal_year = year
            ch.total_signals += 1
        else:
            ch.silent_years += 1
        pulses[name] = ch.pulse

        crossed = ch.silent_years == FLATLINE_THRESHOLD
        chronic = (ch.silent_years > FLATLINE_THRESHOLD
                   and (ch.silent_years - FLATLINE_THRESHOLD)
                       % FLATLINE_THRESHOLD == 0)
        if crossed or chronic:
            prompt = _build_revival(ch, year)
            ch.revivals_emitted += 1
            revivals_this_year.append(prompt)
            state.prompt_history.append(prompt.to_dict())

    overall = sum(pulses.values()) / max(1, len(pulses))
    state.last_overall = overall

    flatlined = [n for n in CHANNEL_NAMES
                 if state.channels[n].silent_years >= FLATLINE_THRESHOLD]
    dying = [n for n in CHANNEL_NAMES
             if n not in flatlined
             and state.channels[n].pulse < DYING_THRESHOLD]

    return VitalsTickResult(
        year=year,
        pulses=pulses,
        revivals=[p.to_dict() for p in revivals_this_year],
        overall_vitality=overall,
        flatlined=flatlined,
        dying=dying,
    )


# ---------------------------------------------------------------------------
# Action nudges — closes the feedback loop into the next tick
# ---------------------------------------------------------------------------

def compute_action_nudges(state: VitalsState,
                          action_names: list[str]) -> dict[str, float]:
    """Map flatlined/dying channels to action weight nudges.

    Returns a dict ``action -> nudge`` in ``[0, MAX_ACTION_NUDGE]`` that the
    engine adds to next year's action scores. Healthy channels contribute
    nothing. This is intentionally gentle: vitals NUDGES, never decides.
    """
    nudges: dict[str, float] = {a: 0.0 for a in action_names}
    for name, ch in state.channels.items():
        if ch.silent_years >= FLATLINE_THRESHOLD:
            weight = MAX_ACTION_NUDGE
        elif ch.pulse < DYING_THRESHOLD:
            weight = MAX_ACTION_NUDGE * (1.0 - ch.pulse / DYING_THRESHOLD)
        else:
            continue
        for action in _CHANNEL_TO_ACTION_HINTS.get(name, ()):
            if action in nudges:
                nudges[action] = min(MAX_ACTION_NUDGE,
                                     nudges[action] + weight * 0.5)
    return nudges


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

def colony_health_report(state: VitalsState) -> dict[str, Any]:
    """Snapshot suitable for the narrator / dashboard."""
    channels = {n: c.to_dict() for n, c in state.channels.items()}
    flatlined = [n for n, c in state.channels.items()
                 if c.silent_years >= FLATLINE_THRESHOLD]
    dying = [n for n, c in state.channels.items()
             if c.pulse < DYING_THRESHOLD
             and c.silent_years < FLATLINE_THRESHOLD]
    healthy = [n for n, c in state.channels.items()
               if c.pulse >= HEALTHY_THRESHOLD]
    return {
        "overall_vitality": round(state.last_overall, 4),
        "channels": channels,
        "flatlined": flatlined,
        "dying": dying,
        "healthy": healthy,
        "total_revivals_emitted": sum(
            c.revivals_emitted for c in state.channels.values()),
        "recent_prompts": state.prompt_history[-10:],
    }
