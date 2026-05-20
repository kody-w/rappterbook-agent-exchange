"""Tests for the comm-channels organ."""
from __future__ import annotations

import random
import pytest

from src.mars100.comm_channels import (
    Channel, CommChannelsState, CommChannelsTickResult,
    RevivalPrompt, pair_key,
    compute_vitality, classify_status, infer_contacts,
    generate_revival_prompt, compute_revival_pressure,
    compute_colony_comm_health, tick_comm_channels,
    FLATLINE_SILENCE_YEARS, FADING_SILENCE_YEARS,
    MAX_REVIVAL_PROMPTS_PER_TICK, STRONG_CONTACT_ACTIONS,
    STATUS_VITAL, STATUS_FADING, STATUS_FLATLINED,
    STATUS_REVIVED, STATUS_DORMANT, STATUS_INACTIVE,
)


class _FakeRel:
    def __init__(self, trust=0.5, affection=0.5, respect=0.5):
        self.trust = trust
        self.affection = affection
        self.respect = respect


def _social_get_factory(trust_map=None, default=0.5):
    trust_map = trust_map or {}
    def get(a, b):
        return _FakeRel(trust=trust_map.get((a, b), default))
    return get


# ----- pure helpers -------------------------------------------------------

def test_pair_key_sorts_canonically():
    assert pair_key("b", "a") == ("a", "b")
    assert pair_key("a", "b") == ("a", "b")


def test_pair_key_rejects_self():
    with pytest.raises(ValueError):
        pair_key("x", "x")


def test_channel_roundtrip():
    ch = Channel(a="a", b="b", born_year=0, last_contact_year=3,
                  total_contacts=5, strong_contacts=2,
                  silence_streak=2, vitality=0.7, status=STATUS_VITAL)
    d = ch.to_dict()
    assert d["a"] == "a" and d["b"] == "b"
    assert d["vitality"] == 0.7
    assert d["status"] == STATUS_VITAL


def test_state_to_dict_string_keys():
    state = CommChannelsState()
    state.channels[("a", "b")] = Channel(
        a="a", b="b", born_year=0, last_contact_year=0)
    d = state.to_dict()
    assert "a|b" in d["channels"]
    assert "revival_log" in d and "flatline_log" in d


# ----- vitality -----------------------------------------------------------

def test_vitality_bounds():
    ch = Channel(a="a", b="b", born_year=0, last_contact_year=0,
                  total_contacts=0, silence_streak=0)
    v = compute_vitality(ch, 0)
    assert 0.0 <= v <= 1.0


def test_vitality_high_for_fresh_active_channel():
    ch = Channel(a="a", b="b", born_year=0, last_contact_year=10,
                  total_contacts=10, strong_contacts=5, silence_streak=0)
    assert compute_vitality(ch, 10) > 0.7


def test_vitality_zero_silence_streak_long():
    ch = Channel(a="a", b="b", born_year=0, last_contact_year=0,
                  total_contacts=5, strong_contacts=0, silence_streak=50)
    assert compute_vitality(ch, 50) < 0.2


def test_vitality_monotone_in_silence():
    base = dict(a="a", b="b", born_year=0, last_contact_year=0,
                 total_contacts=8, strong_contacts=3)
    v0 = compute_vitality(Channel(silence_streak=0, **base), 30)
    v5 = compute_vitality(Channel(silence_streak=5, **base), 30)
    v15 = compute_vitality(Channel(silence_streak=15, **base), 30)
    assert v0 > v5 > v15


# ----- classification -----------------------------------------------------

def test_classify_inactive_when_either_missing():
    ch = Channel(a="a", b="b", born_year=0, last_contact_year=0,
                  silence_streak=0, vitality=1.0)
    assert classify_status(ch, both_active=False, current_year=1) == STATUS_INACTIVE


def test_classify_flatlined_after_threshold():
    ch = Channel(a="a", b="b", born_year=0, last_contact_year=0,
                  silence_streak=FLATLINE_SILENCE_YEARS, vitality=0.1)
    assert classify_status(ch, True, current_year=11) == STATUS_FLATLINED


def test_classify_fading_in_between():
    ch = Channel(a="a", b="b", born_year=0, last_contact_year=0,
                  silence_streak=FADING_SILENCE_YEARS, vitality=0.3)
    assert classify_status(ch, True, current_year=6) == STATUS_FADING


def test_classify_revived_within_grace():
    ch = Channel(a="a", b="b", born_year=0, last_contact_year=10,
                  silence_streak=0, vitality=0.5,
                  last_revival_year=10, revival_count=1)
    assert classify_status(ch, True, current_year=11) == STATUS_REVIVED


def test_classify_vital_high_vitality():
    ch = Channel(a="a", b="b", born_year=0, last_contact_year=0,
                  silence_streak=0, vitality=0.9)
    assert classify_status(ch, True, current_year=1) == STATUS_VITAL


def test_classify_dormant_low_vitality():
    ch = Channel(a="a", b="b", born_year=0, last_contact_year=0,
                  silence_streak=2, vitality=0.1)
    assert classify_status(ch, True, current_year=2) == STATUS_DORMANT


# ----- contact inference --------------------------------------------------

def test_infer_no_contact_with_low_trust_no_actions():
    contacts = infer_contacts(
        ["a", "b"], actions={"a": "rest", "b": "rest"},
        social_get=_social_get_factory(default=0.1),
        faction_membership={}, year=1)
    assert contacts == {}


def test_infer_strong_when_both_cooperate():
    contacts = infer_contacts(
        ["a", "b"],
        actions={"a": "cooperate", "b": "cooperate"},
        social_get=_social_get_factory(default=0.1),
        faction_membership={}, year=1)
    assert ("a", "b") in contacts
    strength, is_strong = contacts[("a", "b")]
    assert is_strong is True
    assert strength == 2


def test_infer_trust_alone_creates_contact():
    contacts = infer_contacts(
        ["a", "b"], actions={"a": "rest", "b": "rest"},
        social_get=_social_get_factory(default=0.9),
        faction_membership={}, year=1)
    assert ("a", "b") in contacts


def test_infer_faction_signal_only_even_years():
    factions = {"a": "f1", "b": "f1"}
    even = infer_contacts(["a", "b"], {"a": "rest", "b": "rest"},
                           _social_get_factory(default=0.1), factions, year=2)
    odd = infer_contacts(["a", "b"], {"a": "rest", "b": "rest"},
                          _social_get_factory(default=0.1), factions, year=3)
    assert ("a", "b") in even
    assert ("a", "b") not in odd


def test_infer_strength_capped_at_2():
    contacts = infer_contacts(
        ["a", "b"],
        actions={"a": "cooperate", "b": "cooperate"},
        social_get=_social_get_factory(default=0.9),
        faction_membership={"a": "f1", "b": "f1"}, year=2)
    strength, _ = contacts[("a", "b")]
    assert strength == 2


def test_infer_skips_self_pairs():
    contacts = infer_contacts(
        ["a"], actions={"a": "cooperate"},
        social_get=_social_get_factory(), faction_membership={}, year=1)
    assert contacts == {}


# ----- revival prompt -----------------------------------------------------

def test_prompt_text_includes_names():
    ch = Channel(a="alice", b="bob", born_year=0, last_contact_year=0,
                  silence_streak=12)
    p = generate_revival_prompt(ch, 12, random.Random(0))
    assert "alice" in p.text and "bob" in p.text
    assert p.suggested_action in ("cooperate", "mediate")


def test_prompt_deterministic_with_rng():
    ch = Channel(a="alice", b="bob", born_year=0, last_contact_year=0,
                  silence_streak=12)
    p1 = generate_revival_prompt(ch, 12, random.Random(7))
    p2 = generate_revival_prompt(ch, 12, random.Random(7))
    assert p1.text == p2.text


def test_prompt_action_escalates_with_silence():
    short = Channel(a="a", b="b", born_year=0, last_contact_year=0,
                     silence_streak=10)
    long = Channel(a="a", b="b", born_year=0, last_contact_year=0,
                    silence_streak=25)
    p_short = generate_revival_prompt(short, 10, random.Random(0))
    p_long = generate_revival_prompt(long, 25, random.Random(0))
    assert p_short.suggested_action == "cooperate"
    assert p_long.suggested_action == "mediate"


# ----- revival pressure ---------------------------------------------------

def test_revival_pressure_empty_state_is_zero():
    state = CommChannelsState()
    p = compute_revival_pressure(state, ["cooperate", "mediate", "rest"])
    assert all(v == 0.0 for v in p.values())


def test_revival_pressure_nudges_actions():
    state = CommChannelsState()
    state.revival_log.append({"suggested_action": "cooperate"})
    state.revival_log.append({"suggested_action": "mediate"})
    p = compute_revival_pressure(state, ["cooperate", "mediate", "rest"])
    assert p["cooperate"] > 0
    assert p["mediate"] > 0
    assert p["rest"] == 0.0


# ----- colony health ------------------------------------------------------

def test_health_empty_returns_1():
    s = CommChannelsState()
    assert compute_colony_comm_health(s, []) == 1.0
    assert compute_colony_comm_health(s, ["a"]) == 1.0


def test_health_in_bounds():
    s = CommChannelsState()
    s.channels[("a", "b")] = Channel(a="a", b="b", born_year=0,
                                       last_contact_year=0, vitality=0.42)
    assert 0.0 <= compute_colony_comm_health(s, ["a", "b"]) <= 1.0


# ----- end-to-end tick ----------------------------------------------------

def test_tick_creates_channels_for_all_pairs():
    state = CommChannelsState()
    ids = ["a", "b", "c"]
    r = tick_comm_channels(
        state, ids, {"a": "cooperate", "b": "cooperate", "c": "rest"},
        _social_get_factory(default=0.5), {}, year=1, rng=random.Random(0))
    # 3 colonists => 3 channels
    assert len(state.channels) == 3
    assert len(r.new_channels) == 3


def test_tick_flatline_emerges_after_silence():
    state = CommChannelsState()
    ids = ["a", "b"]
    # First contact
    tick_comm_channels(state, ids,
                        {"a": "cooperate", "b": "cooperate"},
                        _social_get_factory(default=0.1),
                        {}, year=1, rng=random.Random(0))
    # 10 more years of silence + low trust + no cooperate
    for y in range(2, 13):
        tick_comm_channels(state, ids, {"a": "rest", "b": "rest"},
                            _social_get_factory(default=0.1),
                            {}, year=y, rng=random.Random(y))
    ch = state.channels[("a", "b")]
    assert ch.silence_streak >= FLATLINE_SILENCE_YEARS
    assert ch.status == STATUS_FLATLINED
    assert ch.flatline_count >= 1


def test_tick_revival_after_flatline():
    state = CommChannelsState()
    ids = ["a", "b"]
    tick_comm_channels(state, ids, {"a": "cooperate", "b": "cooperate"},
                        _social_get_factory(default=0.1), {}, year=1,
                        rng=random.Random(0))
    for y in range(2, 13):
        tick_comm_channels(state, ids, {"a": "rest", "b": "rest"},
                            _social_get_factory(default=0.1), {}, year=y,
                            rng=random.Random(y))
    assert state.channels[("a", "b")].status == STATUS_FLATLINED
    # Now revive
    r = tick_comm_channels(state, ids,
                            {"a": "cooperate", "b": "cooperate"},
                            _social_get_factory(default=0.1), {}, year=13,
                            rng=random.Random(13))
    ch = state.channels[("a", "b")]
    assert ch.silence_streak == 0
    assert ch.revival_count == 1
    assert ch.status == STATUS_REVIVED
    assert "a|b" in r.revived


def test_tick_revival_prompts_capped():
    state = CommChannelsState()
    ids = [f"c{i}" for i in range(6)]
    tick_comm_channels(state, ids, {x: "rest" for x in ids},
                        _social_get_factory(default=0.1), {}, year=1,
                        rng=random.Random(0))
    for y in range(2, 13):
        tick_comm_channels(state, ids, {x: "rest" for x in ids},
                            _social_get_factory(default=0.1), {}, year=y,
                            rng=random.Random(y))
    r = tick_comm_channels(state, ids, {x: "rest" for x in ids},
                            _social_get_factory(default=0.1), {}, year=13,
                            rng=random.Random(13))
    assert len(r.revival_prompts) <= MAX_REVIVAL_PROMPTS_PER_TICK


def test_tick_inactive_excluded_from_health():
    state = CommChannelsState()
    # Seed both pairs while active
    tick_comm_channels(state, ["a", "b", "c"],
                        {"a": "cooperate", "b": "cooperate", "c": "rest"},
                        _social_get_factory(default=0.5), {}, year=1,
                        rng=random.Random(0))
    # Now 'c' is gone — only a/b active
    r = tick_comm_channels(state, ["a", "b"],
                            {"a": "cooperate", "b": "cooperate"},
                            _social_get_factory(default=0.5), {}, year=2,
                            rng=random.Random(2))
    # Channels involving 'c' are inactive
    inactive = sum(1 for ch in state.channels.values()
                    if ch.status == STATUS_INACTIVE)
    assert inactive >= 2
    assert 0.0 <= r.health_score <= 1.0


def test_tick_summary_conservation():
    state = CommChannelsState()
    ids = ["a", "b", "c", "d"]
    r = tick_comm_channels(state, ids,
                            {x: "cooperate" for x in ids},
                            _social_get_factory(default=0.5), {}, year=1,
                            rng=random.Random(0))
    assert sum(r.summary.values()) == len(state.channels)


def test_invariant_strong_le_total():
    state = CommChannelsState()
    ids = ["a", "b", "c"]
    for y in range(1, 25):
        tick_comm_channels(state, ids,
                            {"a": "cooperate", "b": "rest", "c": "mediate"},
                            _social_get_factory(default=0.4), {}, year=y,
                            rng=random.Random(y))
    for ch in state.channels.values():
        assert ch.strong_contacts <= ch.total_contacts
        assert ch.silence_streak >= 0


def test_smoke_30_years_no_crash():
    state = CommChannelsState()
    ids = ["a", "b", "c", "d", "e"]
    actions_pool = ["cooperate", "mediate", "rest", "code", "farm"]
    for y in range(1, 31):
        rng = random.Random(y)
        actions = {x: actions_pool[(y + i) % len(actions_pool)]
                    for i, x in enumerate(ids)}
        r = tick_comm_channels(
            state, ids, actions,
            _social_get_factory(default=0.5),
            {ids[0]: "f1", ids[1]: "f1"}, year=y, rng=rng)
        assert 0.0 <= r.health_score <= 1.0
    assert len(state.channels) == 10  # C(5,2)


def test_determinism():
    def run(seed):
        state = CommChannelsState()
        for y in range(1, 12):
            tick_comm_channels(
                state, ["a", "b"], {"a": "rest", "b": "rest"},
                _social_get_factory(default=0.1), {}, year=y,
                rng=random.Random(seed))
        return state.to_dict()
    assert run(42) == run(42)


# ----- bridge-builder organ (v12.1) ---------------------------------------

from src.mars100.comm_channels import (
    BridgePrompt, find_bridge_builder, generate_bridge_prompt,
    compute_bridge_pressure, compute_bridge_efficacy_rate,
    BRIDGE_TRUST_FLOOR, BRIDGE_COOLDOWN_YEARS, MAX_BRIDGE_PROMPTS_PER_TICK,
)


def _flatlined(a="a", b="b", silence=12):
    return Channel(a=a, b=b, born_year=0, last_contact_year=0,
                   silence_streak=silence, status=STATUS_FLATLINED)


def test_bridge_prompt_to_dict_roundtrip():
    p = BridgePrompt(bridge="c", target_a="a", target_b="b",
                     text="hi", silence_years=12, bridge_trust_min=0.7,
                     suggested_action="mediate", year=20)
    d = p.to_dict()
    assert d["bridge"] == "c"
    assert d["target_a"] == "a" and d["target_b"] == "b"
    assert d["bridge_trust_min"] == 0.7
    assert d["suggested_action"] == "mediate"


def test_find_bridge_picks_highest_min_trust():
    ch = _flatlined("a", "b")
    trust_map = {("c", "a"): 0.9, ("c", "b"): 0.5,
                 ("d", "a"): 0.8, ("d", "b"): 0.7,
                 ("e", "a"): 0.95, ("e", "b"): 0.2}
    bridge_id, score = find_bridge_builder(
        ch, ["a", "b", "c", "d", "e"],
        _social_get_factory(trust_map, default=0.0))
    assert bridge_id == "d"
    assert abs(score - 0.7) < 1e-9


def test_find_bridge_excludes_endpoints():
    ch = _flatlined("a", "b")
    bridge_id, _ = find_bridge_builder(
        ch, ["a", "b"], _social_get_factory(default=0.9))
    assert bridge_id is None


def test_find_bridge_respects_trust_floor():
    ch = _flatlined("a", "b")
    bridge_id, score = find_bridge_builder(
        ch, ["a", "b", "c"], _social_get_factory(default=0.1))
    assert bridge_id is None
    assert score == 0.0


def test_find_bridge_respects_excluded():
    ch = _flatlined("a", "b")
    bridge_id, _ = find_bridge_builder(
        ch, ["a", "b", "c", "d"],
        _social_get_factory(default=0.9),
        excluded={"c"})
    assert bridge_id == "d"


def test_find_bridge_tie_breaks_alphabetically():
    ch = _flatlined("a", "b")
    trust_map = {("c", "a"): 0.6, ("c", "b"): 0.6,
                 ("d", "a"): 0.6, ("d", "b"): 0.6}
    bridge_id, _ = find_bridge_builder(
        ch, ["a", "b", "d", "c"],
        _social_get_factory(trust_map, default=0.0))
    assert bridge_id == "c"


def test_generate_bridge_prompt_format():
    ch = _flatlined("alice", "bob", silence=14)
    p = generate_bridge_prompt(ch, "carol", 0.72, year=50,
                                rng=random.Random(0))
    assert p.bridge == "carol"
    assert p.target_a == "alice"
    assert p.target_b == "bob"
    assert p.silence_years == 14
    assert p.suggested_action == "mediate"
    assert "carol" in p.text
    assert "alice" in p.text and "bob" in p.text


def test_compute_bridge_pressure_pools_at_mediate():
    state = CommChannelsState()
    state.bridge_log = [
        {"bridge": "c", "suggested_action": "mediate"},
        {"bridge": "d", "suggested_action": "mediate"},
    ]
    p = compute_bridge_pressure(state, ["cooperate", "mediate", "rest"])
    assert p["mediate"] > p["cooperate"]
    assert p["mediate"] > p["rest"]
    assert p["mediate"] == pytest.approx(0.1)


def test_bridge_efficacy_rate_zero_when_no_prompts():
    state = CommChannelsState()
    assert compute_bridge_efficacy_rate(state) == 0.0


def test_bridge_efficacy_rate_ratio():
    state = CommChannelsState()
    state.total_bridge_prompts = 10
    state.total_bridge_revivals = 3
    assert compute_bridge_efficacy_rate(state) == 0.3


def test_tick_emits_bridge_prompts_when_flatline_with_bridge():
    state = CommChannelsState()
    ids = ["a", "b", "c"]
    trust_map = {("c", "a"): 0.8, ("c", "b"): 0.8,
                 ("a", "c"): 0.4, ("b", "c"): 0.4,
                 ("a", "b"): 0.0, ("b", "a"): 0.0}
    social = _social_get_factory(trust_map, default=0.0)
    actions = {"a": "rest", "b": "rest", "c": "rest"}
    bridge_fired_year = None
    for y in range(1, 16):
        r = tick_comm_channels(state, ids, actions, social, {},
                                year=y, rng=random.Random(y))
        if r.bridge_prompts:
            bridge_fired_year = y
            break
    assert bridge_fired_year is not None
    assert state.total_bridge_prompts >= 1
    assert state.bridge_log[-1]["bridge"] == "c"


def test_tick_bridge_cooldown_blocks_repeat_nomination():
    state = CommChannelsState()
    ids = ["a", "b", "c"]
    trust_map = {("c", "a"): 0.8, ("c", "b"): 0.8}
    social = _social_get_factory(trust_map, default=0.0)
    actions = {"a": "rest", "b": "rest", "c": "rest"}
    fires = []
    for y in range(1, 18):
        r = tick_comm_channels(state, ids, actions, social, {},
                                year=y, rng=random.Random(y))
        if r.bridge_prompts:
            fires.append(y)
    if len(fires) >= 2:
        gap = fires[1] - fires[0]
        assert gap >= BRIDGE_COOLDOWN_YEARS, \
            f"expected >= {BRIDGE_COOLDOWN_YEARS}y gap, got {gap}"


def test_tick_bridge_revival_credit():
    state = CommChannelsState()
    ids = ["a", "b", "c"]
    trust_map = {("c", "a"): 0.8, ("c", "b"): 0.8}
    social = _social_get_factory(trust_map, default=0.0)
    for y in range(1, 14):
        tick_comm_channels(
            state, ids, {"a": "rest", "b": "rest", "c": "rest"},
            social, {}, year=y, rng=random.Random(y))
    assert state.total_bridge_prompts >= 1
    fired_count = state.total_bridge_prompts
    tick_comm_channels(
        state, ids, {"a": "cooperate", "b": "cooperate", "c": "rest"},
        social, {}, year=14, rng=random.Random(14))
    assert state.total_bridge_revivals >= 1
    assert compute_bridge_efficacy_rate(state) <= 1.0
    assert compute_bridge_efficacy_rate(state) > 0.0
    assert state.total_bridge_revivals <= fired_count + 1


def test_tick_result_has_bridge_prompts_field():
    state = CommChannelsState()
    r = tick_comm_channels(
        state, ["a", "b"], {"a": "rest", "b": "rest"},
        _social_get_factory(default=0.5), {}, year=1, rng=random.Random(0))
    assert hasattr(r, "bridge_prompts")
    assert isinstance(r.bridge_prompts, list)
    d = r.to_dict()
    assert "bridge_prompts" in d


def test_state_to_dict_serializes_bridge_fields():
    state = CommChannelsState()
    state.total_bridge_prompts = 5
    state.total_bridge_revivals = 2
    d = state.to_dict()
    assert d["total_bridge_prompts"] == 5
    assert d["total_bridge_revivals"] == 2
    assert d["bridge_efficacy_rate"] == 0.4
    assert "bridge_log" in d


def test_bridge_prompts_capped_per_tick():
    state = CommChannelsState()
    ids = [chr(ord("a") + i) for i in range(8)]
    social = _social_get_factory(default=0.9)
    actions = {x: "rest" for x in ids}
    tick_comm_channels(state, ids, actions, social, {},
                        year=1, rng=random.Random(1))
    last_r = None
    for y in range(2, 16):
        last_r = tick_comm_channels(state, ids, actions, social, {},
                                     year=y, rng=random.Random(y))
    assert last_r is not None
    assert len(last_r.bridge_prompts) <= MAX_BRIDGE_PROMPTS_PER_TICK


def test_smoke_with_bridges_no_crash():
    state = CommChannelsState()
    ids = ["a", "b", "c", "d", "e"]
    social = _social_get_factory(default=0.7)
    for y in range(1, 31):
        actions = {x: "rest" for x in ids}
        r = tick_comm_channels(state, ids, actions, social, {},
                                year=y, rng=random.Random(y))
        assert 0.0 <= r.health_score <= 1.0
        assert isinstance(r.bridge_prompts, list)
        for bp in r.bridge_prompts:
            assert bp.bridge not in (bp.target_a, bp.target_b)
            assert bp.bridge_trust_min >= BRIDGE_TRUST_FLOOR
