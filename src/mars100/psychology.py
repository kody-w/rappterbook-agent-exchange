"""
Psychology organ for Mars-100 colony simulation (engine v8.0).

Models persistent mental/emotional states per colonist:
  - stress: accumulates from crises, resource scarcity, events
  - morale: boosted by cooperation, success, good resources
  - grief: spikes when close friends die or are exiled, decays over time

Key dynamics:
  - Stress modulated by personality (paranoia amplifies, resolve dampens)
  - Morale modulated by faith (buffers lows) and empathy (amplifies highs)
  - Breakdowns occur when stress exceeds threshold — colonist loses action
  - Grief proportional to trust with lost colonist
  - Bonding events when colonists help each other through breakdowns

Integration:
  - Pre-action: stress/morale updates, breakdown checks, action modifiers
  - Post-death: grief processing, bonding events
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# -- constants ---------------------------------------------------------------

GRIEF_DECAY_RATE = 0.15          # grief lost per year
STRESS_DECAY_RATE = 0.08         # natural stress relief per year
MORALE_REVERSION_RATE = 0.1      # pull toward 0.5 per year
BREAKDOWN_THRESHOLD = 0.85       # stress above this risks breakdown
BREAKDOWN_SCALE = 2.0            # prob = (stress - threshold) * scale
GRIEF_FACTOR = 0.4               # trust * factor = grief delta
MAX_BONDS = 5                    # cap bonds per colonist
MIN_PRODUCTIVITY = 0.5           # floor for productivity multiplier
MAX_PRODUCTIVITY = 1.2           # ceiling for productivity multiplier


@dataclass
class PsychState:
    """Per-colonist psychological state."""
    stress: float = 0.2
    morale: float = 0.7
    grief: float = 0.0
    bonds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "stress": round(self.stress, 4),
            "morale": round(self.morale, 4),
            "grief": round(self.grief, 4),
            "bonds": {k: round(v, 4) for k, v in self.bonds.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PsychState:
        """Deserialize from dict."""
        return cls(
            stress=d.get("stress", 0.2),
            morale=d.get("morale", 0.7),
            grief=d.get("grief", 0.0),
            bonds=dict(d.get("bonds", {})),
        )


@dataclass
class PsychTickResult:
    """Result of pre-action psychology tick."""
    breakdowns: list[dict] = field(default_factory=list)
    avg_stress: float = 0.0
    avg_morale: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "breakdowns": self.breakdowns,
            "avg_stress": round(self.avg_stress, 4),
            "avg_morale": round(self.avg_morale, 4),
        }


@dataclass
class PsychPostResult:
    """Result of post-death psychology tick."""
    grief_events: list[dict] = field(default_factory=list)
    bonding_events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grief_events": self.grief_events,
            "bonding_events": self.bonding_events,
        }


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def _update_stress(state: PsychState, event_severity: float,
                   resource_avg: float, paranoia: float,
                   resolve: float, rng: random.Random) -> None:
    """Update stress for one colonist.

    Paranoia amplifies stress gain; resolve dampens it.
    """
    # Natural decay
    state.stress -= STRESS_DECAY_RATE

    # Event stress (paranoia amplifies)
    paranoia_mult = 1.0 + paranoia * 0.5
    state.stress += event_severity * 0.3 * paranoia_mult

    # Resource scarcity stress
    scarcity = max(0.0, 1.0 - resource_avg)
    state.stress += scarcity * 0.15

    # Resolve dampens accumulated stress
    resolve_dampen = resolve * 0.05
    state.stress -= resolve_dampen

    # Small random noise
    state.stress += rng.gauss(0, 0.02)

    state.stress = _clamp(state.stress)


def _update_morale(state: PsychState, resource_avg: float,
                   faith: float, empathy: float) -> None:
    """Update morale for one colonist.

    Faith buffers low morale; empathy amplifies social morale.
    """
    # Revert toward baseline
    state.morale += (0.5 - state.morale) * MORALE_REVERSION_RATE

    # Resource effects
    if resource_avg > 0.6:
        bonus = 0.05 * (1.0 + empathy * 0.3)
        state.morale += bonus
    elif resource_avg < 0.3:
        penalty = 0.1
        faith_buffer = faith * 0.04
        state.morale -= max(0.0, penalty - faith_buffer)

    # Grief suppresses morale
    state.morale -= state.grief * 0.2

    state.morale = _clamp(state.morale)


def _check_breakdown(state: PsychState, rng: random.Random) -> bool:
    """Check if a colonist breaks down from stress."""
    if state.stress <= BREAKDOWN_THRESHOLD:
        return False
    prob = (state.stress - BREAKDOWN_THRESHOLD) * BREAKDOWN_SCALE
    return rng.random() < prob


def tick_psychology_pre(
    states: dict[str, PsychState],
    active_ids: list[str],
    events: list[Any],
    resources: Any,
    year: int,
    rng: random.Random,
    colonist_stats: dict[str, dict[str, float]] | None = None,
) -> PsychTickResult:
    """Pre-action psychology tick: update stress/morale, check breakdowns.

    Args:
        states: Mutable per-colonist psychology states.
        active_ids: IDs of active colonists (will be sorted internally).
        events: List of Event objects with .severity.
        resources: Resources object with .average().
        year: Current year.
        rng: Dedicated psychology RNG.
        colonist_stats: Optional dict mapping colonist_id to stats dict
            with keys 'paranoia', 'resolve', 'faith', 'empathy'.

    Returns:
        PsychTickResult with breakdown list and averages.
    """
    if colonist_stats is None:
        colonist_stats = {}

    event_severity = max((ev.severity for ev in events), default=0.0)
    resource_avg = resources.average() if hasattr(resources, 'average') else 0.5

    breakdowns: list[dict] = []
    total_stress = 0.0
    total_morale = 0.0

    for cid in sorted(active_ids):
        if cid not in states:
            states[cid] = PsychState()
        state = states[cid]

        stats = colonist_stats.get(cid, {})
        paranoia = stats.get("paranoia", 0.5)
        resolve = stats.get("resolve", 0.5)
        faith = stats.get("faith", 0.5)
        empathy = stats.get("empathy", 0.5)

        # Grief decay
        state.grief = _clamp(state.grief - GRIEF_DECAY_RATE)

        _update_stress(state, event_severity, resource_avg,
                       paranoia, resolve, rng)
        _update_morale(state, resource_avg, faith, empathy)

        if _check_breakdown(state, rng):
            breakdowns.append({
                "colonist_id": cid,
                "stress_level": round(state.stress, 4),
                "year": year,
            })
            # Breakdown partially relieves stress
            state.stress = _clamp(state.stress - 0.2)

        total_stress += state.stress
        total_morale += state.morale

    n = max(1, len(active_ids))
    return PsychTickResult(
        breakdowns=breakdowns,
        avg_stress=total_stress / n,
        avg_morale=total_morale / n,
    )


def tick_psychology_post(
    states: dict[str, PsychState],
    deaths: list[dict],
    exiles: list[dict],
    breakdowns: list[dict],
    social_get: Any,
    active_ids: list[str],
    rng: random.Random,
) -> PsychPostResult:
    """Post-death psychology tick: process grief and bonding.

    Args:
        states: Mutable per-colonist psychology states.
        deaths: List of death dicts with 'id' key.
        exiles: List of exile dicts with 'id' key.
        breakdowns: List of breakdown dicts with 'colonist_id' key.
        social_get: Callable(from_id, to_id) -> Relationship with .trust.
        active_ids: IDs of colonists still active AFTER deaths/exiles.
        rng: Dedicated psychology RNG.

    Returns:
        PsychPostResult with grief and bonding events.
    """
    result = PsychPostResult()

    lost_ids = [d["id"] for d in deaths] + [e["id"] for e in exiles]

    # Process grief for each loss
    for lost_id in lost_ids:
        for cid in sorted(active_ids):
            if cid == lost_id:
                continue
            state = states.get(cid)
            if state is None:
                continue
            try:
                rel = social_get(cid, lost_id)
                trust = getattr(rel, 'trust', 0.5)
            except (KeyError, AttributeError):
                trust = 0.3

            grief_delta = trust * GRIEF_FACTOR
            if grief_delta > 0.01:
                state.grief = _clamp(state.grief + grief_delta)
                state.stress = _clamp(state.stress + grief_delta * 0.3)
                result.grief_events.append({
                    "colonist_id": cid,
                    "source_id": lost_id,
                    "intensity": round(grief_delta, 4),
                })

    # Bonding from helping broken-down colonists
    broken_ids = {b["colonist_id"] for b in breakdowns}
    available_helpers = [cid for cid in sorted(active_ids)
                         if cid not in broken_ids]

    for b in breakdowns:
        bid = b["colonist_id"]
        if bid not in states or not available_helpers:
            continue

        # Find most trusted available helper
        best_helper = None
        best_trust = -1.0
        for hid in available_helpers:
            try:
                rel = social_get(bid, hid)
                t = getattr(rel, 'trust', 0.5)
            except (KeyError, AttributeError):
                t = 0.3
            if t > best_trust:
                best_trust = t
                best_helper = hid

        if best_helper is None:
            continue

        bond_delta = 0.1 + rng.gauss(0, 0.02)
        bond_delta = max(0.01, bond_delta)

        # Update bonds for both parties
        b_state = states[bid]
        h_state = states.get(best_helper)

        b_state.bonds[best_helper] = _clamp(
            b_state.bonds.get(best_helper, 0.0) + bond_delta)
        if h_state is not None:
            h_state.bonds[bid] = _clamp(
                h_state.bonds.get(bid, 0.0) + bond_delta)
            h_state.morale = _clamp(h_state.morale + 0.05)

        result.bonding_events.append({
            "helper_id": best_helper,
            "helped_id": bid,
            "bond_delta": round(bond_delta, 4),
        })

    # Prune bonds to MAX_BONDS per colonist (keep strongest, break ties by id)
    for cid in sorted(active_ids):
        state = states.get(cid)
        if state is None or len(state.bonds) <= MAX_BONDS:
            continue
        sorted_bonds = sorted(state.bonds.items(),
                              key=lambda x: (-x[1], x[0]))
        state.bonds = dict(sorted_bonds[:MAX_BONDS])

    # Clean up psych state for dead/exiled colonists
    for lost_id in lost_ids:
        states.pop(lost_id, None)

    return result


def compute_action_modifiers(state: PsychState) -> dict[str, float]:
    """Compute action weight modifiers from mental state.

    High stress → more pray/rest, less cooperate/explore.
    Low morale → more rest/hoard, less cooperate/code.
    High grief → more pray/mediate, less sabotage.

    Returns dict mapping action name → weight delta.
    """
    s = state.stress
    m = state.morale
    g = state.grief

    return {
        "pray": s * 0.4 + g * 0.3,
        "rest": s * 0.3 + (1.0 - m) * 0.3,
        "mediate": g * 0.2 + m * 0.1,
        "cooperate": -s * 0.3 + m * 0.2,
        "explore": -s * 0.2 + m * 0.15,
        "code": m * 0.2 - s * 0.1,
        "farm": (1.0 - m) * 0.1,
        "hoard": (1.0 - m) * 0.2 + s * 0.1,
        "sabotage": s * 0.2 - g * 0.3,
        "terraform": -s * 0.1,
        "research": m * 0.15 - s * 0.15,
    }


def compute_productivity(state: PsychState) -> float:
    """Compute skill/action effectiveness multiplier from mental state.

    Morale drives productivity; stress and grief reduce it.
    Returns value in [MIN_PRODUCTIVITY, MAX_PRODUCTIVITY].
    """
    base = 0.5 + state.morale * 0.7
    stress_penalty = 1.0 - state.stress * 0.3
    grief_penalty = 1.0 - state.grief * 0.2
    raw = base * stress_penalty * grief_penalty
    return _clamp(raw, MIN_PRODUCTIVITY, MAX_PRODUCTIVITY)


def initialize_state() -> PsychState:
    """Create default psychology state for a new colonist."""
    return PsychState()


def initialize_child_state() -> PsychState:
    """Create psychology state for a colony-born child."""
    return PsychState(stress=0.1, morale=0.8)


def initialize_immigrant_state() -> PsychState:
    """Create psychology state for an immigrant (higher stress from journey)."""
    return PsychState(stress=0.4, morale=0.5)
