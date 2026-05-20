"""Tests for the bridge_missions organ."""
from __future__ import annotations

import pytest

from src.mars100.bridge_missions import (
    BridgeMission, BridgeMissionsState,
    select_bridger, compute_bridger_pressure, per_colonist_pressure,
    tick_bridge_missions, parse_pair_label,
    MISSION_MAX_AGE_YEARS, MIN_BRIDGER_TRUST, MAX_NEW_MISSIONS_PER_TICK,
    BRIDGER_ACTION_PRESSURE,
    STATUS_ACTIVE, STATUS_SUCCEEDED, STATUS_FAILED, STATUS_EXPIRED,
)


class _FakeRel:
    def __init__(self, trust=0.5):
        self.trust = trust


def _social_get(trust_map):
    def get(x, y):
        return _FakeRel(trust=trust_map.get((x, y), 0.0))
    return get


def test_parse_pair_label():
    assert parse_pair_label("alice|bob") == ("alice", "bob")


def test_select_bridger_picks_highest_min_trust():
    sg = _social_get({
        ("c1", "a"): 0.9, ("c1", "b"): 0.4,
        ("c2", "a"): 0.6, ("c2", "b"): 0.7,
        ("c3", "a"): 0.99, ("c3", "b"): 0.1,
    })
    pick, score = select_bridger("a", "b", ["c1", "c2", "c3"], sg)
    assert pick == "c2"
    assert score == pytest.approx(0.6)


def test_select_bridger_rejects_endpoints():
    sg = _social_get({("a", "a"): 1.0, ("a", "b"): 1.0})
    pick, _ = select_bridger("a", "b", ["a", "b"], sg)
    assert pick is None


def test_select_bridger_requires_threshold_on_both_sides():
    sg = _social_get({("c", "a"): MIN_BRIDGER_TRUST - 0.01, ("c", "b"): 0.9})
    pick, _ = select_bridger("a", "b", ["c"], sg)
    assert pick is None


def test_select_bridger_no_candidates_returns_none():
    sg = _social_get({})
    pick, score = select_bridger("a", "b", [], sg)
    assert pick is None and score == 0.0


def test_compute_bridger_pressure_only_mediate():
    state = BridgeMissionsState()
    state.missions[("a", "b")] = BridgeMission(a="a", b="b", bridger="c", born_year=0)
    state.missions[("a", "c")] = BridgeMission(a="a", b="c", bridger="d", born_year=0)
    p = compute_bridger_pressure(state, ["mediate", "cooperate", "rest"])
    assert p["mediate"] == pytest.approx(2 * BRIDGER_ACTION_PRESSURE)
    assert p["cooperate"] == 0.0


def test_compute_bridger_pressure_no_mediate_in_pool():
    state = BridgeMissionsState()
    state.missions[("a", "b")] = BridgeMission(a="a", b="b", bridger="c", born_year=0)
    assert compute_bridger_pressure(state, ["rest"]) == {"rest": 0.0}


def test_compute_bridger_pressure_skips_resolved_missions():
    state = BridgeMissionsState()
    state.missions[("a", "b")] = BridgeMission(
        a="a", b="b", bridger="c", born_year=0, status=STATUS_SUCCEEDED)
    assert compute_bridger_pressure(state, ["mediate"])["mediate"] == 0.0


def test_per_colonist_pressure_aggregates_by_bridger():
    state = BridgeMissionsState()
    state.missions[("a", "b")] = BridgeMission(a="a", b="b", bridger="c", born_year=0)
    state.missions[("a", "d")] = BridgeMission(a="a", b="d", bridger="c", born_year=0)
    state.missions[("e", "f")] = BridgeMission(a="e", b="f", bridger="g", born_year=0)
    pressures = per_colonist_pressure(state)
    assert pressures["c"] == pytest.approx(2 * BRIDGER_ACTION_PRESSURE)
    assert pressures["g"] == pytest.approx(BRIDGER_ACTION_PRESSURE)


def _common_sg():
    return _social_get({
        ("c", "a"): 0.8, ("c", "b"): 0.7,
        ("d", "a"): 0.6, ("d", "b"): 0.6,
    })


def test_tick_spawns_mission_for_newly_flatlined_pair():
    state = BridgeMissionsState()
    result = tick_bridge_missions(
        state, newly_flatlined_keys=[("a", "b")], revived_keys=[],
        active_ids=["a", "b", "c", "d"],
        actions={"a": "rest", "b": "rest", "c": "rest", "d": "rest"},
        social_get=_common_sg(), year=10)
    assert result.active_count == 1
    assert len(result.spawned) == 1
    m = state.missions[("a", "b")]
    assert m.bridger == "c"
    assert m.status == STATUS_ACTIVE
    assert m.born_year == 10


def test_tick_idempotent_when_pair_already_has_mission():
    state = BridgeMissionsState()
    state.missions[("a", "b")] = BridgeMission(a="a", b="b", bridger="c", born_year=5)
    result = tick_bridge_missions(
        state, newly_flatlined_keys=[("a", "b")], revived_keys=[],
        active_ids=["a", "b", "c"], actions={"c": "rest"},
        social_get=_common_sg(), year=6)
    assert result.spawned == []
    assert state.missions[("a", "b")].bridger == "c"


def test_tick_resolves_succeeded_on_revival():
    state = BridgeMissionsState()
    state.missions[("a", "b")] = BridgeMission(a="a", b="b", bridger="c", born_year=0)
    result = tick_bridge_missions(
        state, newly_flatlined_keys=[], revived_keys=[("a", "b")],
        active_ids=["a", "b", "c"], actions={"c": "rest"},
        social_get=_common_sg(), year=3)
    assert len(result.succeeded) == 1
    assert ("a", "b") not in state.missions
    assert state.ledger[-1]["status"] == STATUS_SUCCEEDED


def test_tick_resolves_succeeded_when_bridger_mediates():
    state = BridgeMissionsState()
    state.missions[("a", "b")] = BridgeMission(a="a", b="b", bridger="c", born_year=0)
    result = tick_bridge_missions(
        state, newly_flatlined_keys=[], revived_keys=[],
        active_ids=["a", "b", "c"], actions={"c": "mediate"},
        social_get=_common_sg(), year=1)
    assert len(result.succeeded) == 1


def test_tick_expires_when_bridger_leaves():
    state = BridgeMissionsState()
    state.missions[("a", "b")] = BridgeMission(a="a", b="b", bridger="c", born_year=0)
    result = tick_bridge_missions(
        state, newly_flatlined_keys=[], revived_keys=[],
        active_ids=["a", "b"], actions={},
        social_get=_common_sg(), year=2)
    assert len(result.expired) == 1


def test_tick_fails_when_age_exceeds_max():
    state = BridgeMissionsState()
    state.missions[("a", "b")] = BridgeMission(
        a="a", b="b", bridger="c", born_year=0, age=MISSION_MAX_AGE_YEARS - 1)
    result = tick_bridge_missions(
        state, newly_flatlined_keys=[], revived_keys=[],
        active_ids=["a", "b", "c"], actions={"c": "rest"},
        social_get=_common_sg(), year=99)
    assert len(result.failed) == 1


def test_tick_caps_new_missions_per_tick():
    state = BridgeMissionsState()
    pairs = [("a", "b"), ("a", "c"), ("b", "c"),
              ("d", "e"), ("d", "f"), ("e", "f")]
    trust_map = {(x, y): 0.9 for x in "abcdef" for y in "abcdef" if x != y}
    sg = _social_get(trust_map)
    result = tick_bridge_missions(
        state, newly_flatlined_keys=pairs, revived_keys=[],
        active_ids=list("abcdef"),
        actions={k: "rest" for k in "abcdef"},
        social_get=sg, year=20)
    assert len(result.spawned) <= MAX_NEW_MISSIONS_PER_TICK


def test_tick_no_mission_when_no_qualifying_bridger():
    state = BridgeMissionsState()
    sg = _social_get({})
    result = tick_bridge_missions(
        state, newly_flatlined_keys=[("a", "b")], revived_keys=[],
        active_ids=["a", "b", "c", "d"], actions={},
        social_get=sg, year=0)
    assert result.spawned == []


def test_tick_resolved_year_is_set():
    state = BridgeMissionsState()
    state.missions[("a", "b")] = BridgeMission(a="a", b="b", bridger="c", born_year=0)
    tick_bridge_missions(
        state, newly_flatlined_keys=[], revived_keys=[("a", "b")],
        active_ids=["a", "b", "c"], actions={},
        social_get=_common_sg(), year=7)
    assert state.ledger[-1]["year"] == 7


def test_state_to_dict_roundtrip_shape():
    state = BridgeMissionsState()
    state.missions[("a", "b")] = BridgeMission(a="a", b="b", bridger="c", born_year=0)
    state.ledger.append({"pair": "a|b<-c", "bridger": "c",
                          "year": 1, "status": STATUS_SUCCEEDED})
    d = state.to_dict()
    assert "a|b" in d["missions"]
    assert d["missions"]["a|b"]["bridger"] == "c"


def test_smoke_ten_ticks_no_crash():
    """10-tick smoke: state stays consistent, succeeded outcome is reached."""
    state = BridgeMissionsState()
    active = ["a", "b", "c", "d", "e"]
    sg = _social_get({(x, y): 0.7 for x in active for y in active if x != y})
    for year in range(10):
        flat = [("a", "b")] if year == 0 else []
        revived = [("a", "b")] if year == 4 else []
        result = tick_bridge_missions(
            state, newly_flatlined_keys=flat, revived_keys=revived,
            active_ids=active, actions={"c": "rest"},
            social_get=sg, year=year)
        assert result.active_count >= 0
    assert state.missions == {}
    assert any(e["status"] == STATUS_SUCCEEDED for e in state.ledger)
