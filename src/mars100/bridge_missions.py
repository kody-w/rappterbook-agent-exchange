"""
Bridge missions organ (engine v12.1).

Listens to comm_channels output. When a channel between (A, B) flatlines,
this organ recruits a third colonist C — one who still trusts both A and B —
and dispatches them as a **bridger**: their next 'mediate' action is biased
toward repairing the broken pair.

Revival prompts go from "nudge text in a log" to "named mission with a chosen
human, a deadline, and a resolved outcome (succeeded / failed / expired)".

Pure functions + dataclasses. No I/O. Deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from src.mars100.comm_channels import pair_key

MISSION_MAX_AGE_YEARS = 6
MIN_BRIDGER_TRUST = 0.35
MAX_NEW_MISSIONS_PER_TICK = 4
BRIDGER_ACTION_PRESSURE = 0.08

STATUS_ACTIVE = "active"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_EXPIRED = "expired"


@dataclass
class BridgeMission:
    a: str
    b: str
    bridger: str
    born_year: int
    age: int = 0
    status: str = STATUS_ACTIVE
    resolved_year: int = -1

    def to_dict(self) -> dict:
        return {
            "a": self.a, "b": self.b, "bridger": self.bridger,
            "born_year": self.born_year, "age": self.age,
            "status": self.status, "resolved_year": self.resolved_year,
        }


@dataclass
class BridgeMissionsState:
    missions: dict = field(default_factory=dict)
    ledger: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "missions": {"{}|{}".format(k[0], k[1]): m.to_dict()
                          for k, m in self.missions.items()},
            "ledger": list(self.ledger[-50:]),
        }


@dataclass
class BridgeMissionsTickResult:
    year: int
    spawned: list
    succeeded: list
    failed: list
    expired: list
    active_count: int

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "spawned": list(self.spawned),
            "succeeded": list(self.succeeded),
            "failed": list(self.failed),
            "expired": list(self.expired),
            "active_count": self.active_count,
        }


def _trust(social_get: Callable, x: str, y: str) -> float:
    try:
        rel_xy = social_get(x, y)
        rel_yx = social_get(y, x)
    except Exception:
        return 0.0
    return max(getattr(rel_xy, "trust", 0.0),
               getattr(rel_yx, "trust", 0.0))


def select_bridger(a, b, candidates, social_get):
    """Pick the candidate with highest min(trust_to_a, trust_to_b).

    Both endpoints must be trusted above MIN_BRIDGER_TRUST. Returns
    (id, score) or (None, 0.0) if no candidate qualifies.
    """
    best = (None, 0.0)
    for c in candidates:
        if c == a or c == b:
            continue
        t_ca = _trust(social_get, c, a)
        t_cb = _trust(social_get, c, b)
        if t_ca < MIN_BRIDGER_TRUST or t_cb < MIN_BRIDGER_TRUST:
            continue
        score = min(t_ca, t_cb)
        if score > best[1]:
            best = (c, score)
    return best


def compute_bridger_pressure(state, actions_pool):
    """{action: pressure} aggregated across active missions. Only 'mediate'
    receives pressure. Other actions return 0.0 for safe merging."""
    pressure = {a: 0.0 for a in actions_pool}
    if "mediate" not in pressure:
        return pressure
    pressure["mediate"] += BRIDGER_ACTION_PRESSURE * sum(
        1 for m in state.missions.values() if m.status == STATUS_ACTIVE)
    return pressure


def per_colonist_pressure(state):
    """{colonist_id: extra mediate-bias} — used by the chooser to bias
    SPECIFICALLY the bridgers' choice, not everybody's."""
    out: dict = {}
    for m in state.missions.values():
        if m.status != STATUS_ACTIVE:
            continue
        out[m.bridger] = out.get(m.bridger, 0.0) + BRIDGER_ACTION_PRESSURE
    return out


def tick_bridge_missions(state, *, newly_flatlined_keys, revived_keys,
                          active_ids, actions, social_get, year):
    """Advance the bridge-missions organ by one year.

    Mutates `state`. Returns BridgeMissionsTickResult.
    Resolution order: age → succeeded (revived OR bridger mediated) →
    expired (someone left) → failed (aged out) → spawn (capped).
    """
    active_set = set(active_ids)
    spawned: list = []
    succeeded: list = []
    failed: list = []
    expired: list = []
    revived_set = {pair_key(*k) for k in revived_keys}

    to_drop: list = []
    for key, m in state.missions.items():
        if m.status != STATUS_ACTIVE:
            to_drop.append(key)
            continue
        m.age += 1
        if key in revived_set or actions.get(m.bridger) == "mediate":
            m.status = STATUS_SUCCEEDED
            m.resolved_year = year
            succeeded.append(_label(m))
            state.ledger.append({"pair": _label(m), "bridger": m.bridger,
                                   "year": year, "status": STATUS_SUCCEEDED})
            to_drop.append(key)
            continue
        if (m.a not in active_set or m.b not in active_set
                or m.bridger not in active_set):
            m.status = STATUS_EXPIRED
            m.resolved_year = year
            expired.append(_label(m))
            state.ledger.append({"pair": _label(m), "bridger": m.bridger,
                                   "year": year, "status": STATUS_EXPIRED})
            to_drop.append(key)
            continue
        if m.age >= MISSION_MAX_AGE_YEARS:
            m.status = STATUS_FAILED
            m.resolved_year = year
            failed.append(_label(m))
            state.ledger.append({"pair": _label(m), "bridger": m.bridger,
                                   "year": year, "status": STATUS_FAILED})
            to_drop.append(key)

    for k in to_drop:
        state.missions.pop(k, None)

    new_pairs = sorted({pair_key(*k) for k in newly_flatlined_keys})
    for pk in new_pairs[:MAX_NEW_MISSIONS_PER_TICK]:
        if pk in state.missions:
            continue
        a, b = pk
        if a not in active_set or b not in active_set:
            continue
        bridger, _score = select_bridger(a, b, active_ids, social_get)
        if bridger is None:
            continue
        mission = BridgeMission(a=a, b=b, bridger=bridger,
                                  born_year=year, age=0,
                                  status=STATUS_ACTIVE)
        state.missions[pk] = mission
        spawned.append(_label(mission))

    state.ledger = state.ledger[-200:]
    active_count = sum(1 for m in state.missions.values()
                        if m.status == STATUS_ACTIVE)
    return BridgeMissionsTickResult(
        year=year, spawned=spawned, succeeded=succeeded,
        failed=failed, expired=expired, active_count=active_count)


def _label(m: BridgeMission) -> str:
    return "{}|{}<-{}".format(m.a, m.b, m.bridger)


def parse_pair_label(label: str):
    """'a|b' -> ('a', 'b'). Inverse of the '{}|{}'.format pattern used by
    comm_channels' flatlined/revived/fading lists."""
    a, _, b = label.partition("|")
    return (a, b)
