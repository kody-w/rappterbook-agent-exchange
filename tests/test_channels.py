"""Tests for the Channels organ (engine v12.0)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import src.mars100...` works.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.mars100.channels import (
    ARCHIVE_AFTER_FLATLINE_YEARS,
    DEAD_VITALITY_THRESHOLD,
    FLATLINE_YEARS,
    KIND_FACTION,
    KIND_TOPIC,
    MAX_REVIVAL_PROMPTS_PER_YEAR,
    MIN_MEMBERS,
    TOPIC_EMERGENCE_THRESHOLD,
    Channel,
    ChannelsState,
    compute_revival_pressure,
    tick_channels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _faction(fid, members, name="Red Compact",
             ideology="cooperative", cohesion=0.7):
    return {
        "id": fid, "name": name, "ideology": ideology,
        "members": list(members), "cohesion": cohesion, "influence": 0.5,
        "leader_id": members[0] if members else None,
    }


def _zero_social(a, b):
    return 0.0


def _rng(seed=42):
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Channel creation
# ---------------------------------------------------------------------------

class TestChannelCreation:
    def test_faction_channel_created_for_qualifying_faction(self):
        state = ChannelsState()
        result = tick_channels(
            state=state,
            factions=[_faction("f1", ["c1", "c2", "c3"])],
            actions={},
            active_colonist_ids=["c1", "c2", "c3"],
            year=5,
            social_get=_zero_social,
            rng=_rng(),
        )
        assert len(result.channels_created) == 1
        assert result.channels_created[0]["kind"] == KIND_FACTION
        assert len(state.channels) == 1

    def test_faction_channel_not_recreated_next_tick(self):
        state = ChannelsState()
        fac = _faction("f1", ["c1", "c2", "c3"])
        tick_channels(state, [fac], {}, ["c1", "c2", "c3"], 5, _zero_social, _rng())
        result2 = tick_channels(state, [fac], {}, ["c1", "c2", "c3"], 6, _zero_social, _rng())
        assert len(result2.channels_created) == 0
        assert len(state.channels) == 1

    def test_undersized_faction_gets_no_channel(self):
        state = ChannelsState()
        result = tick_channels(
            state, [_faction("f1", ["c1"])],
            {}, ["c1"], 1, _zero_social, _rng(),
        )
        assert len(result.channels_created) == 0
        assert len(state.channels) == 0

    def test_topic_channel_emerges_at_threshold(self):
        state = ChannelsState()
        actions = {f"c{i}": "terraform" for i in range(TOPIC_EMERGENCE_THRESHOLD)}
        active = list(actions.keys())
        result = tick_channels(state, [], actions, active, 1, _zero_social, _rng())
        topic_chans = [c for c in result.channels_created if c["kind"] == KIND_TOPIC]
        assert len(topic_chans) == 1
        assert topic_chans[0]["name"] == "#terraform"

    def test_topic_channel_not_created_below_threshold(self):
        state = ChannelsState()
        actions = {f"c{i}": "terraform" for i in range(TOPIC_EMERGENCE_THRESHOLD - 1)}
        result = tick_channels(state, [], actions, list(actions.keys()),
                               1, _zero_social, _rng())
        assert all(c["kind"] != KIND_TOPIC for c in result.channels_created)


# ---------------------------------------------------------------------------
# Vitality dynamics
# ---------------------------------------------------------------------------

class TestVitalityDynamics:
    def test_silent_channel_decays(self):
        state = ChannelsState()
        fac = _faction("f1", ["c1", "c2", "c3"])
        tick_channels(state, [fac], {}, ["c1", "c2", "c3"], 1, _zero_social, _rng())
        ch = next(iter(state.channels.values()))
        v0 = ch.vitality
        tick_channels(state, [], {}, [], 2, _zero_social, _rng())
        assert ch.vitality < v0

    def test_active_channel_grows(self):
        state = ChannelsState()
        fac = _faction("f1", ["c1", "c2", "c3", "c4", "c5"], cohesion=0.9)
        active = ["c1", "c2", "c3", "c4", "c5"]
        tick_channels(state, [fac], {}, active, 1, _zero_social, _rng())
        ch = next(iter(state.channels.values()))
        v0 = ch.vitality
        for y in range(2, 6):
            tick_channels(state, [fac], {}, active, y, _zero_social, _rng())
        assert ch.vitality >= v0

    def test_vitality_bounded(self):
        """Property: vitality is always in [0, 1]."""
        state = ChannelsState()
        fac = _faction("f1", ["c1", "c2", "c3"], cohesion=1.0)
        active = ["c1", "c2", "c3"]
        for y in range(1, 40):
            facs = [fac] if y < 20 else []
            act = active if y < 20 else []
            tick_channels(state, facs, {}, act, y, _zero_social, _rng(y))
            for ch in state.channels.values():
                assert 0.0 <= ch.vitality <= 1.0, f"y={y} ch={ch.id} v={ch.vitality}"


# ---------------------------------------------------------------------------
# Flatline detection
# ---------------------------------------------------------------------------

class TestFlatline:
    def test_silence_flatlines_channel(self):
        state = ChannelsState()
        fac = _faction("f1", ["c1", "c2", "c3"])
        tick_channels(state, [fac], {}, ["c1", "c2", "c3"], 1, _zero_social, _rng())
        flatlined_year = None
        for y in range(2, 2 + FLATLINE_YEARS + 5):
            r = tick_channels(state, [], {}, ["c1", "c2", "c3"], y,
                              _zero_social, _rng(y))
            if r.channels_flatlined:
                flatlined_year = y
                break
        assert flatlined_year is not None

    def test_flatlined_flag_persists(self):
        state = ChannelsState()
        fac = _faction("f1", ["c1", "c2", "c3"])
        tick_channels(state, [fac], {}, ["c1", "c2", "c3"], 1, _zero_social, _rng())
        for y in range(2, 25):
            tick_channels(state, [], {}, ["c1", "c2", "c3"], y, _zero_social, _rng(y))
        ch = next(iter(state.channels.values()))
        assert ch.flatlined is True
        assert ch.flatlined_since is not None

    def test_flatline_only_once(self):
        state = ChannelsState()
        fac = _faction("f1", ["c1", "c2", "c3"])
        tick_channels(state, [fac], {}, ["c1", "c2", "c3"], 1, _zero_social, _rng())
        flatline_events = []
        for y in range(2, 30):
            r = tick_channels(state, [], {}, ["c1", "c2", "c3"], y,
                              _zero_social, _rng(y))
            flatline_events.extend(r.channels_flatlined)
        ids = [e["id"] for e in flatline_events]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Revival prompts
# ---------------------------------------------------------------------------

class TestRevivalPrompts:
    def _flatlined_state(self):
        state = ChannelsState()
        fac = _faction("f1", ["c1", "c2", "c3"])
        tick_channels(state, [fac], {}, ["c1", "c2", "c3"], 1, _zero_social, _rng())
        for y in range(2, 1 + FLATLINE_YEARS + 2):
            tick_channels(state, [], {}, ["c1", "c2", "c3"], y, _zero_social, _rng(y))
        return state

    def test_prompt_emitted_for_flatlined_channel(self):
        state = ChannelsState()
        fac = _faction("f1", ["c1", "c2", "c3"])
        tick_channels(state, [fac], {}, ["c1", "c2", "c3"], 1, _zero_social, _rng())
        all_prompts = []
        for y in range(2, 2 + FLATLINE_YEARS + 5):
            r = tick_channels(state, [], {}, ["c1", "c2", "c3"], y,
                              _zero_social, _rng(y))
            all_prompts.extend(r.revival_prompts)
        assert len(all_prompts) >= 1
        prompt = all_prompts[0]
        assert prompt["leader_id"] in ("c1", "c2", "c3")
        assert "flatlined" in prompt["prompt"]

    def test_prompt_cap_respected(self):
        state = ChannelsState()
        facs = [_faction(f"f{i}", [f"c{i}a", f"c{i}b", f"c{i}c"], name=f"Bloc {i}")
                for i in range(MAX_REVIVAL_PROMPTS_PER_YEAR + 5)]
        active = [m for f in facs for m in f["members"]]
        tick_channels(state, facs, {}, active, 1, _zero_social, _rng())
        for y in range(2, 2 + FLATLINE_YEARS + 1):
            tick_channels(state, [], {}, active, y, _zero_social, _rng(y))
        r = tick_channels(state, [], {}, active,
                          2 + FLATLINE_YEARS + 2, _zero_social, _rng())
        assert len(r.revival_prompts) <= MAX_REVIVAL_PROMPTS_PER_YEAR

    def test_no_prompt_without_living_members(self):
        state = self._flatlined_state()
        last_year = max(ch.flatlined_since or 0 for ch in state.channels.values())
        r = tick_channels(state, [], {}, [], last_year + 1, _zero_social, _rng())
        assert r.revival_prompts == []

    def test_revival_restores_channel(self):
        state = self._flatlined_state()
        last_year = max(ch.flatlined_since or 0 for ch in state.channels.values())
        fac = _faction("f1", ["c1", "c2", "c3"], cohesion=0.9)
        r = tick_channels(state, [fac], {}, ["c1", "c2", "c3"], last_year + 1,
                          _zero_social, _rng())
        assert len(r.channels_revived) >= 1
        ch = next(iter(state.channels.values()))
        assert ch.flatlined is False


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------

class TestArchive:
    def test_long_dead_channel_archived(self):
        state = ChannelsState()
        fac = _faction("f1", ["c1", "c2", "c3"])
        tick_channels(state, [fac], {}, ["c1", "c2", "c3"], 1, _zero_social, _rng())
        total = FLATLINE_YEARS + ARCHIVE_AFTER_FLATLINE_YEARS + 5
        archived = False
        for y in range(2, total):
            r = tick_channels(state, [], {}, ["c1", "c2", "c3"], y,
                              _zero_social, _rng(y))
            if r.channels_archived:
                archived = True
                break
        assert archived, "channel should auto-archive after long flatline"

    def test_archived_channel_stays_archived(self):
        state = ChannelsState()
        fac = _faction("f1", ["c1", "c2", "c3"])
        tick_channels(state, [fac], {}, ["c1", "c2", "c3"], 1, _zero_social, _rng())
        for y in range(2, FLATLINE_YEARS + ARCHIVE_AFTER_FLATLINE_YEARS + 10):
            tick_channels(state, [], {}, ["c1", "c2", "c3"], y, _zero_social, _rng(y))
        archived_ids = [cid for cid, ch in state.channels.items() if ch.archived]
        assert len(archived_ids) >= 1
        r = tick_channels(state, [_faction("f1", ["c1", "c2", "c3"], cohesion=0.9)],
                          {}, ["c1", "c2", "c3"], 200, _zero_social, _rng())
        for cid in archived_ids:
            assert state.channels[cid].archived is True


# ---------------------------------------------------------------------------
# Revival pressure
# ---------------------------------------------------------------------------

class TestRevivalPressure:
    def test_topic_pressure_targets_action(self):
        prompts = [{
            "channel_id": "ch-tp-0001", "channel_name": "#terraform",
            "leader_id": "c1", "kind": KIND_TOPIC, "year": 5,
            "vitality": 0.05, "silent_years": 12, "topic": "Terraform Talk",
            "prompt": "...",
        }]
        pressure = compute_revival_pressure(prompts)
        assert "c1" in pressure
        assert pressure["c1"]["terraform"] > 0

    def test_faction_pressure_nudges_mediate(self):
        prompts = [{
            "channel_id": "ch-fx-0001", "channel_name": "#red-compact",
            "leader_id": "c1", "kind": KIND_FACTION, "year": 5,
            "vitality": 0.05, "silent_years": 12, "topic": "cooperative",
            "prompt": "...",
        }]
        pressure = compute_revival_pressure(prompts)
        assert pressure["c1"]["mediate"] > 0
        assert pressure["c1"]["cooperate"] > 0

    def test_pressure_bounded(self):
        prompts = [
            {"kind": KIND_TOPIC, "leader_id": "c1", "channel_name": "#code"},
            {"kind": KIND_FACTION, "leader_id": "c1", "channel_name": "#x"},
        ]
        pressure = compute_revival_pressure(prompts)
        for actions in pressure.values():
            for v in actions.values():
                assert 0 <= v <= 0.15

    def test_no_leader_no_pressure(self):
        prompts = [{"kind": KIND_TOPIC, "leader_id": None, "channel_name": "#x"}]
        assert compute_revival_pressure(prompts) == {}


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------

class TestResultShape:
    def test_to_dict_round_trip(self):
        state = ChannelsState()
        fac = _faction("f1", ["c1", "c2", "c3"])
        r = tick_channels(state, [fac], {}, ["c1", "c2", "c3"], 1,
                          _zero_social, _rng())
        d = r.to_dict()
        assert d["year"] == 1
        assert "vitals" in d
        assert "alive_count" in d
        assert "flatlined_count" in d
        assert d["alive_count"] >= 1

    def test_state_to_dict_round_trip(self):
        state = ChannelsState()
        fac = _faction("f1", ["c1", "c2", "c3"])
        tick_channels(state, [fac], {}, ["c1", "c2", "c3"], 1, _zero_social, _rng())
        d = state.to_dict()
        assert "channels" in d
        assert "next_channel_seq" in d
        assert len(d["channels"]) == 1


# ---------------------------------------------------------------------------
# Smoke / integration
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_runs_for_10_ticks_without_crash(self):
        state = ChannelsState()
        rng = _rng()
        active = [f"c{i}" for i in range(8)]
        facs = [
            _faction("f1", active[:4], name="Red Compact", cohesion=0.7),
            _faction("f2", active[4:], name="Iron Circle", cohesion=0.6),
        ]
        for y in range(1, 11):
            actions = {cid: rng.choice(["terraform", "farm", "code"]) for cid in active}
            r = tick_channels(state, facs, actions, active, y, _zero_social, rng)
            assert r.year == y
            for ch in state.channels.values():
                assert 0.0 <= ch.vitality <= 1.0
                assert ch.posts_last_year >= 0
                assert ch.total_posts >= 0

    def test_determinism_with_same_seed(self):
        def run_once():
            state = ChannelsState()
            active = [f"c{i}" for i in range(5)]
            fac = _faction("f1", active, cohesion=0.7)
            rng = _rng(123)
            for y in range(1, 6):
                tick_channels(state, [fac], {}, active, y, _zero_social, rng)
            return [ch.to_dict() for ch in state.channels.values()]
        assert run_once() == run_once()

    def test_dead_members_pruned(self):
        state = ChannelsState()
        fac = _faction("f1", ["c1", "c2", "c3"])
        tick_channels(state, [fac], {}, ["c1", "c2", "c3"], 1, _zero_social, _rng())
        tick_channels(state, [_faction("f1", ["c1", "c2"])],
                      {}, ["c1", "c2"], 2, _zero_social, _rng())
        ch = next(iter(state.channels.values()))
        assert "c3" not in ch.members
