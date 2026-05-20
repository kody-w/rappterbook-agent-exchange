"""Tests for rumor diffusion organ (engine v13.0)."""
from __future__ import annotations
import random
import pytest

from src.mars100.rumors import (
    DECAY_YEARS, MAX_RUMORS,
    Rumor, RumorsState, RumorsTickResult,
    build_channel_lookup, compute_fragmentation,
    tick_rumors, _mutate_text, _pick_seed,
)


def _complete_lookup(ids, vitality=0.8, status="vital"):
    lookup = {}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            key = (a, b) if a < b else (b, a)
            lookup[key] = (vitality, status)
    return lookup


def test_mutate_text_swaps_known_phrase():
    rng = random.Random(1)
    out = _mutate_text("the reactor's neutron count is creeping", rng)
    assert "spiking" in out


def test_mutate_text_fallback():
    rng = random.Random(2)
    assert _mutate_text("nothing interesting", rng).endswith("(or so they say)")


def test_pick_seed_uses_provided_ids():
    rng = random.Random(3)
    ids = ["alpha", "beta", "gamma"]
    origin, text = _pick_seed(rng, ids)
    assert origin in ids and text


def test_origination_requires_active_colonists():
    state = RumorsState()
    rng = random.Random(0)
    r = tick_rumors(state, 1, [], {}, rng, origin_rate=1.0)
    assert r.born == [] and r.active_count == 0


def test_high_origin_rate_creates_rumors():
    state = RumorsState()
    ids = [f"c{i}" for i in range(5)]
    lookup = _complete_lookup(ids)
    rng = random.Random(42)
    total = 0
    for y in range(1, 11):
        total += len(tick_rumors(state, y, ids, lookup, rng, origin_rate=1.0).born)
    assert total >= 1


def test_rumors_spread_along_vital_channels():
    state = RumorsState()
    ids = [f"c{i}" for i in range(8)]
    lookup = _complete_lookup(ids, vitality=0.95)
    rng = random.Random(7)
    for y in range(1, 6):
        tick_rumors(state, y, ids, lookup, rng, origin_rate=1.0, pass_rate=0.9)
    biggest = max((len(r.carriers) for r in state.rumors.values()), default=0)
    assert biggest >= 2


def test_flatlined_channels_block_transmission():
    state = RumorsState()
    ids = [f"c{i}" for i in range(6)]
    lookup = _complete_lookup(ids, vitality=0.0, status="flatlined")
    rng = random.Random(11)
    blocks = 0
    for y in range(1, 6):
        blocks += tick_rumors(state, y, ids, lookup, rng,
                                origin_rate=1.0, pass_rate=1.0).cross_channel_blocks
    for r in state.rumors.values():
        assert len(r.carriers) == 1
    assert blocks > 0


def test_inactive_channels_also_block():
    state = RumorsState()
    ids = ["a", "b", "c"]
    lookup = _complete_lookup(ids, vitality=0.9, status="inactive")
    rng = random.Random(13)
    for y in range(1, 4):
        tick_rumors(state, y, ids, lookup, rng, origin_rate=1.0, pass_rate=1.0)
    for r in state.rumors.values():
        assert len(r.carriers) == 1


def test_rumor_dies_after_decay_years():
    state = RumorsState()
    rng = random.Random(0)
    rumor = Rumor(id="r1", text="t", origin_id="a", origin_year=1,
                   born_year=1, last_growth_year=1)
    rumor.carriers.add("a")
    state.rumors["r1"] = rumor
    state.next_id = 2
    for y in range(2, 2 + DECAY_YEARS + 1):
        tick_rumors(state, y, ["a", "b"], {}, rng, origin_rate=0.0)
    assert "r1" not in state.rumors
    assert any(a["id"] == "r1" for a in state.archive)


def test_carriers_pruned_when_inactive():
    state = RumorsState()
    rumor = Rumor(id="r1", text="t", origin_id="a", origin_year=1,
                   born_year=1, last_growth_year=1)
    rumor.carriers = {"a", "b", "c"}
    state.rumors["r1"] = rumor
    state.next_id = 2
    rng = random.Random(0)
    tick_rumors(state, 2, ["a", "b"], {}, rng, origin_rate=0.0)
    assert state.rumors["r1"].carriers == {"a", "b"}


def test_fragmentation_zero_when_consensus():
    state = RumorsState()
    for i in range(3):
        r = Rumor(id=f"r{i}", text="t", origin_id="a", origin_year=1,
                   born_year=1, last_growth_year=1)
        r.carriers = {"a", "b", "c"}
        state.rumors[r.id] = r
    assert compute_fragmentation(state.rumors, {"a", "b", "c"}) == 0.0


def test_fragmentation_one_when_disjoint():
    state = RumorsState()
    r1 = Rumor(id="r1", text="t", origin_id="a", origin_year=1,
                born_year=1, last_growth_year=1)
    r1.carriers = {"a", "b"}
    r2 = Rumor(id="r2", text="t2", origin_id="c", origin_year=1,
                born_year=1, last_growth_year=1)
    r2.carriers = {"c", "d"}
    state.rumors["r1"] = r1
    state.rumors["r2"] = r2
    assert compute_fragmentation(state.rumors, {"a", "b", "c", "d"}) == 1.0


def test_fragmentation_in_bounds():
    state = RumorsState()
    ids = [f"c{i}" for i in range(6)]
    lookup = _complete_lookup(ids, vitality=0.5)
    rng = random.Random(99)
    for y in range(1, 12):
        r = tick_rumors(state, y, ids, lookup, rng, origin_rate=0.5, pass_rate=0.4)
        assert 0.0 <= r.fragmentation <= 1.0


def test_deterministic():
    ids = [f"c{i}" for i in range(5)]
    lookup = _complete_lookup(ids, vitality=0.7)

    def run():
        state = RumorsState()
        rng = random.Random(2025)
        last = None
        for y in range(1, 16):
            last = tick_rumors(state, y, ids, lookup, rng,
                                origin_rate=0.6, pass_rate=0.5)
        return state.to_dict(), last.to_dict()

    a, al = run()
    b, bl = run()
    assert a == b and al == bl


def test_rumor_cap_respected():
    state = RumorsState()
    ids = [f"c{i}" for i in range(10)]
    lookup = _complete_lookup(ids, vitality=0.9)
    rng = random.Random(5)
    for y in range(1, 60):
        tick_rumors(state, y, ids, lookup, rng, origin_rate=1.0, pass_rate=0.7)
    assert len(state.rumors) <= MAX_RUMORS


def test_to_dict_json_serialisable():
    import json
    state = RumorsState()
    ids = ["a", "b", "c"]
    lookup = _complete_lookup(ids)
    rng = random.Random(0)
    for y in range(1, 6):
        tick_rumors(state, y, ids, lookup, rng, origin_rate=1.0)
    parsed = json.loads(json.dumps(state.to_dict()))
    assert "rumors" in parsed and "next_id" in parsed


def test_build_channel_lookup_shape():
    from src.mars100.comm_channels import CommChannelsState, Channel as CC
    cs = CommChannelsState()
    ch = CC(a="x", b="y", born_year=0, last_contact_year=0)
    ch.vitality = 0.42
    ch.status = "vital"
    cs.channels[("x", "y")] = ch
    lookup = build_channel_lookup(cs)
    assert lookup[("x", "y")] == (0.42, "vital")


def test_smoke_10_years():
    state = RumorsState()
    ids = [f"c{i}" for i in range(6)]
    lookup = _complete_lookup(ids)
    rng = random.Random(0)
    for y in range(1, 11):
        r = tick_rumors(state, y, ids, lookup, rng)
        assert isinstance(r, RumorsTickResult)
        assert r.transmissions >= 0
        assert r.cross_channel_blocks >= 0
        assert 0.0 <= r.fragmentation <= 1.0
