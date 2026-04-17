"""Tests for the Mars-100 psychology organ."""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.mars100.psychology import (
    PsychState,
    PsychTickResult,
    PsychPostResult,
    tick_psychology_pre,
    tick_psychology_post,
    compute_action_modifiers,
    compute_productivity,
    initialize_state,
    initialize_child_state,
    initialize_immigrant_state,
    GRIEF_DECAY_RATE,
    STRESS_DECAY_RATE,
    BREAKDOWN_THRESHOLD,
    MAX_BONDS,
    MIN_PRODUCTIVITY,
    MAX_PRODUCTIVITY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeEvent:
    severity: float = 0.5
    name: str = "dust_storm"


@dataclass
class FakeResources:
    food: float = 0.5
    water: float = 0.5
    power: float = 0.5
    air: float = 0.5
    medicine: float = 0.5

    def average(self) -> float:
        vals = [self.food, self.water, self.power, self.air, self.medicine]
        return sum(vals) / len(vals)


@dataclass
class FakeRelationship:
    trust: float = 0.5


def make_social_get(trust_map: dict[tuple[str, str], float] | None = None):
    """Create a social_get callable for testing."""
    default_trust = 0.5
    def social_get(from_id: str, to_id: str) -> FakeRelationship:
        if trust_map:
            return FakeRelationship(trust=trust_map.get((from_id, to_id), default_trust))
        return FakeRelationship(trust=default_trust)
    return social_get


# ---------------------------------------------------------------------------
# PsychState
# ---------------------------------------------------------------------------

class TestPsychState:
    def test_default(self):
        s = PsychState()
        assert s.stress == 0.2
        assert s.morale == 0.7
        assert s.grief == 0.0
        assert s.bonds == {}

    def test_to_dict(self):
        s = PsychState(stress=0.5, morale=0.6, grief=0.3, bonds={"a": 0.1})
        d = s.to_dict()
        assert d["stress"] == 0.5
        assert d["morale"] == 0.6
        assert d["grief"] == 0.3
        assert d["bonds"] == {"a": 0.1}

    def test_from_dict(self):
        d = {"stress": 0.4, "morale": 0.8, "grief": 0.1, "bonds": {"b": 0.2}}
        s = PsychState.from_dict(d)
        assert s.stress == 0.4
        assert s.morale == 0.8
        assert s.bonds == {"b": 0.2}

    def test_from_dict_defaults(self):
        s = PsychState.from_dict({})
        assert s.stress == 0.2
        assert s.morale == 0.7


class TestTickResults:
    def test_pre_result_to_dict(self):
        r = PsychTickResult(breakdowns=[{"colonist_id": "c1"}],
                            avg_stress=0.3, avg_morale=0.6)
        d = r.to_dict()
        assert len(d["breakdowns"]) == 1
        assert d["avg_stress"] == 0.3

    def test_post_result_to_dict(self):
        r = PsychPostResult(grief_events=[{"colonist_id": "c1"}],
                            bonding_events=[])
        d = r.to_dict()
        assert len(d["grief_events"]) == 1
        assert d["bonding_events"] == []


# ---------------------------------------------------------------------------
# tick_psychology_pre
# ---------------------------------------------------------------------------

class TestTickPsychologyPre:
    def test_basic_tick(self):
        states: dict[str, PsychState] = {}
        ids = ["c1", "c2", "c3"]
        events = [FakeEvent(severity=0.3)]
        res = FakeResources()
        rng = random.Random(42)

        result = tick_psychology_pre(states, ids, events, res, year=1, rng=rng)

        assert len(states) == 3
        assert isinstance(result, PsychTickResult)
        assert result.avg_stress >= 0
        assert result.avg_morale >= 0

    def test_creates_missing_states(self):
        states: dict[str, PsychState] = {"c1": PsychState()}
        ids = ["c1", "c2"]
        result = tick_psychology_pre(states, ids, [FakeEvent()],
                                     FakeResources(), 1, random.Random(42))
        assert "c2" in states

    def test_high_severity_increases_stress(self):
        states = {"c1": PsychState(stress=0.3)}
        tick_psychology_pre(states, ["c1"], [FakeEvent(severity=0.9)],
                           FakeResources(), 1, random.Random(42))
        assert states["c1"].stress > 0.3

    def test_good_resources_boost_morale(self):
        states = {"c1": PsychState(morale=0.5)}
        res = FakeResources(food=0.8, water=0.8, power=0.8, air=0.8, medicine=0.8)
        tick_psychology_pre(states, ["c1"], [], res, 1, random.Random(42))
        assert states["c1"].morale > 0.5

    def test_bad_resources_reduce_morale(self):
        states = {"c1": PsychState(morale=0.7)}
        res = FakeResources(food=0.1, water=0.1, power=0.1, air=0.1, medicine=0.1)
        tick_psychology_pre(states, ["c1"], [FakeEvent(severity=0.0)],
                           res, 1, random.Random(42))
        assert states["c1"].morale < 0.7

    def test_grief_decays(self):
        states = {"c1": PsychState(grief=0.5)}
        tick_psychology_pre(states, ["c1"], [], FakeResources(),
                           1, random.Random(42))
        assert states["c1"].grief < 0.5

    def test_breakdown_at_high_stress(self):
        states = {"c1": PsychState(stress=0.95)}
        rng = random.Random(1)
        # Run multiple times to get at least one breakdown
        breakdowns_found = False
        for seed in range(100):
            states["c1"].stress = 0.95
            result = tick_psychology_pre(
                states, ["c1"], [FakeEvent(severity=0.8)],
                FakeResources(food=0.1, water=0.1, power=0.1, air=0.1, medicine=0.1),
                1, random.Random(seed))
            if result.breakdowns:
                breakdowns_found = True
                break
        assert breakdowns_found, "Expected at least one breakdown in 100 tries"

    def test_no_breakdown_at_low_stress(self):
        states = {"c1": PsychState(stress=0.1)}
        result = tick_psychology_pre(
            states, ["c1"], [], FakeResources(), 1, random.Random(42))
        assert len(result.breakdowns) == 0

    def test_paranoia_amplifies_stress(self):
        # Compare high paranoia vs low paranoia
        s_high = {"c1": PsychState(stress=0.3)}
        s_low = {"c1": PsychState(stress=0.3)}
        stats_high = {"c1": {"paranoia": 0.9, "resolve": 0.5, "faith": 0.5, "empathy": 0.5}}
        stats_low = {"c1": {"paranoia": 0.1, "resolve": 0.5, "faith": 0.5, "empathy": 0.5}}

        tick_psychology_pre(s_high, ["c1"], [FakeEvent(severity=0.5)],
                           FakeResources(), 1, random.Random(42),
                           colonist_stats=stats_high)
        tick_psychology_pre(s_low, ["c1"], [FakeEvent(severity=0.5)],
                           FakeResources(), 1, random.Random(42),
                           colonist_stats=stats_low)
        assert s_high["c1"].stress > s_low["c1"].stress

    def test_resolve_dampens_stress(self):
        s_high = {"c1": PsychState(stress=0.5)}
        s_low = {"c1": PsychState(stress=0.5)}
        stats_high = {"c1": {"paranoia": 0.5, "resolve": 0.9, "faith": 0.5, "empathy": 0.5}}
        stats_low = {"c1": {"paranoia": 0.5, "resolve": 0.1, "faith": 0.5, "empathy": 0.5}}

        tick_psychology_pre(s_high, ["c1"], [FakeEvent(severity=0.3)],
                           FakeResources(), 1, random.Random(42),
                           colonist_stats=stats_high)
        tick_psychology_pre(s_low, ["c1"], [FakeEvent(severity=0.3)],
                           FakeResources(), 1, random.Random(42),
                           colonist_stats=stats_low)
        assert s_high["c1"].stress < s_low["c1"].stress

    def test_stress_clamped_0_1(self):
        states = {"c1": PsychState(stress=0.99)}
        tick_psychology_pre(states, ["c1"], [FakeEvent(severity=1.0)],
                           FakeResources(food=0.0, water=0.0, power=0.0,
                                         air=0.0, medicine=0.0),
                           1, random.Random(42))
        assert 0 <= states["c1"].stress <= 1.0

    def test_morale_clamped_0_1(self):
        states = {"c1": PsychState(morale=0.99)}
        res = FakeResources(food=1.0, water=1.0, power=1.0, air=1.0, medicine=1.0)
        tick_psychology_pre(states, ["c1"], [], res, 1, random.Random(42))
        assert 0 <= states["c1"].morale <= 1.0

    def test_deterministic_with_same_rng(self):
        def run_tick(seed):
            states = {"c1": PsychState(), "c2": PsychState()}
            return tick_psychology_pre(
                states, ["c1", "c2"], [FakeEvent()],
                FakeResources(), 1, random.Random(seed))

        r1 = run_tick(42)
        r2 = run_tick(42)
        assert r1.avg_stress == r2.avg_stress
        assert r1.avg_morale == r2.avg_morale

    def test_sorted_ids_determinism(self):
        """Processing order shouldn't depend on input order."""
        def run_tick(ids):
            states = {cid: PsychState() for cid in ids}
            return tick_psychology_pre(
                states, ids, [FakeEvent()],
                FakeResources(), 1, random.Random(42))

        r1 = run_tick(["c1", "c2", "c3"])
        r2 = run_tick(["c3", "c1", "c2"])
        assert r1.avg_stress == r2.avg_stress

    def test_empty_active_ids(self):
        states: dict[str, PsychState] = {}
        result = tick_psychology_pre(states, [], [], FakeResources(),
                                     1, random.Random(42))
        assert result.breakdowns == []
        assert result.avg_stress == 0.0


# ---------------------------------------------------------------------------
# tick_psychology_post
# ---------------------------------------------------------------------------

class TestTickPsychologyPost:
    def test_grief_from_death(self):
        states = {"c1": PsychState(), "c2": PsychState()}
        deaths = [{"id": "c3"}]  # c3 died
        social = make_social_get({("c1", "c3"): 0.8, ("c2", "c3"): 0.3})

        result = tick_psychology_post(
            states, deaths, [], [], social, ["c1", "c2"],
            random.Random(42))

        assert len(result.grief_events) > 0
        # c1 had higher trust with c3, should have more grief
        c1_grief = states["c1"].grief
        c2_grief = states["c2"].grief
        assert c1_grief > c2_grief

    def test_grief_from_exile(self):
        states = {"c1": PsychState()}
        exiles = [{"id": "c2"}]
        social = make_social_get({("c1", "c2"): 0.7})

        result = tick_psychology_post(
            states, [], exiles, [], social, ["c1"],
            random.Random(42))

        assert states["c1"].grief > 0

    def test_bonding_from_breakdown(self):
        states = {"c1": PsychState(), "c2": PsychState()}
        breakdowns = [{"colonist_id": "c1"}]
        social = make_social_get({("c1", "c2"): 0.7})

        result = tick_psychology_post(
            states, [], [], breakdowns, social,
            ["c1", "c2"], random.Random(42))

        assert len(result.bonding_events) > 0
        assert "c2" in states["c1"].bonds
        assert "c1" in states["c2"].bonds

    def test_helper_morale_boost(self):
        states = {"c1": PsychState(), "c2": PsychState(morale=0.5)}
        breakdowns = [{"colonist_id": "c1"}]
        social = make_social_get()

        tick_psychology_post(states, [], [], breakdowns, social,
                            ["c1", "c2"], random.Random(42))

        # c2 helped c1, should get morale boost
        assert states["c2"].morale > 0.5

    def test_cleans_up_dead_states(self):
        states = {"c1": PsychState(), "c2": PsychState(), "c3": PsychState()}
        deaths = [{"id": "c3"}]

        tick_psychology_post(states, deaths, [], [], make_social_get(),
                            ["c1", "c2"], random.Random(42))

        assert "c3" not in states
        assert "c1" in states
        assert "c2" in states

    def test_no_deaths_no_grief(self):
        states = {"c1": PsychState(), "c2": PsychState()}
        result = tick_psychology_post(
            states, [], [], [], make_social_get(),
            ["c1", "c2"], random.Random(42))
        assert result.grief_events == []

    def test_bonds_capped_at_max(self):
        # Create state with MAX_BONDS + 2 bonds already
        bonds = {f"x{i}": 0.5 for i in range(MAX_BONDS + 2)}
        states = {"c1": PsychState(bonds=bonds), "c2": PsychState()}
        breakdowns = [{"colonist_id": "c1"}]
        social = make_social_get()

        tick_psychology_post(states, [], [], breakdowns, social,
                            ["c1", "c2"], random.Random(42))

        assert len(states["c1"].bonds) <= MAX_BONDS

    def test_grief_increases_stress(self):
        states = {"c1": PsychState(stress=0.2)}
        deaths = [{"id": "c2"}]
        social = make_social_get({("c1", "c2"): 0.9})

        tick_psychology_post(states, deaths, [], [], social,
                            ["c1"], random.Random(42))

        assert states["c1"].stress > 0.2

    def test_empty_inputs(self):
        states: dict[str, PsychState] = {}
        result = tick_psychology_post(
            states, [], [], [], make_social_get(), [],
            random.Random(42))
        assert result.grief_events == []
        assert result.bonding_events == []


# ---------------------------------------------------------------------------
# compute_action_modifiers
# ---------------------------------------------------------------------------

class TestActionModifiers:
    def test_high_stress_promotes_rest_pray(self):
        mods = compute_action_modifiers(PsychState(stress=0.9, morale=0.5))
        assert mods["pray"] > 0
        assert mods["rest"] > 0

    def test_high_stress_reduces_cooperate(self):
        mods = compute_action_modifiers(PsychState(stress=0.9, morale=0.5))
        assert mods["cooperate"] < 0

    def test_high_morale_promotes_cooperate(self):
        mods = compute_action_modifiers(PsychState(stress=0.1, morale=0.9))
        assert mods["cooperate"] > 0

    def test_grief_promotes_mediate(self):
        mods = compute_action_modifiers(PsychState(grief=0.8))
        assert mods["mediate"] > 0

    def test_grief_reduces_sabotage(self):
        mods = compute_action_modifiers(PsychState(grief=0.8, stress=0.1))
        assert mods["sabotage"] < 0

    def test_all_actions_present(self):
        mods = compute_action_modifiers(PsychState())
        for action in ["pray", "rest", "mediate", "cooperate", "explore",
                        "code", "farm", "hoard", "sabotage", "terraform",
                        "research"]:
            assert action in mods


# ---------------------------------------------------------------------------
# compute_productivity
# ---------------------------------------------------------------------------

class TestProductivity:
    def test_high_morale_high_productivity(self):
        p = compute_productivity(PsychState(stress=0.0, morale=1.0, grief=0.0))
        assert p > 1.0

    def test_low_morale_low_productivity(self):
        p = compute_productivity(PsychState(stress=0.5, morale=0.2, grief=0.3))
        assert p < 0.8

    def test_minimum_floor(self):
        p = compute_productivity(PsychState(stress=1.0, morale=0.0, grief=1.0))
        assert p >= MIN_PRODUCTIVITY

    def test_maximum_ceiling(self):
        p = compute_productivity(PsychState(stress=0.0, morale=1.0, grief=0.0))
        assert p <= MAX_PRODUCTIVITY

    def test_default_state_reasonable(self):
        p = compute_productivity(PsychState())
        assert 0.8 < p < 1.1

    def test_bounded_across_extremes(self):
        """Property: productivity always in [MIN, MAX] for any valid state."""
        rng = random.Random(42)
        for _ in range(100):
            state = PsychState(
                stress=rng.random(),
                morale=rng.random(),
                grief=rng.random(),
            )
            p = compute_productivity(state)
            assert MIN_PRODUCTIVITY <= p <= MAX_PRODUCTIVITY


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_default_state(self):
        s = initialize_state()
        assert s.stress == 0.2
        assert s.morale == 0.7

    def test_child_state(self):
        s = initialize_child_state()
        assert s.stress < initialize_state().stress
        assert s.morale > initialize_state().morale

    def test_immigrant_state(self):
        s = initialize_immigrant_state()
        assert s.stress > initialize_state().stress
        assert s.morale < initialize_state().morale


# ---------------------------------------------------------------------------
# Multi-year smoke test
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_10_year_pre_tick(self):
        """Smoke test: 10 years of pre-ticks don't crash or produce NaN."""
        states: dict[str, PsychState] = {}
        ids = [f"c{i}" for i in range(10)]
        rng = random.Random(42)

        for year in range(1, 11):
            events = [FakeEvent(severity=rng.random())]
            res = FakeResources(
                food=rng.random(), water=rng.random(),
                power=rng.random(), air=rng.random(),
                medicine=rng.random())
            result = tick_psychology_pre(states, ids, events, res, year, rng)

            assert not any(
                v != v  # NaN check
                for cid in ids
                for v in [states[cid].stress, states[cid].morale, states[cid].grief]
            )
            assert 0 <= result.avg_stress <= 1
            assert 0 <= result.avg_morale <= 1

    def test_full_lifecycle(self):
        """Smoke test: pre-tick → deaths → post-tick lifecycle."""
        states = {f"c{i}": PsychState() for i in range(5)}
        ids = [f"c{i}" for i in range(5)]
        rng = random.Random(42)

        # Pre-tick
        pre = tick_psychology_pre(
            states, ids, [FakeEvent(severity=0.6)],
            FakeResources(), 1, rng)

        # Simulate a death
        deaths = [{"id": "c4"}]
        active_after = ["c0", "c1", "c2", "c3"]

        # Post-tick
        post = tick_psychology_post(
            states, deaths, [], pre.breakdowns,
            make_social_get(), active_after, rng)

        assert "c4" not in states
        assert len(states) == 4
        # At least some grief should have been processed
        total_grief = sum(s.grief for s in states.values())
        assert total_grief > 0


class TestEngineIntegration:
    """Verify psychology organ is wired into the engine."""

    def test_year_result_has_psychology(self):
        from src.mars100.engine import Mars100Engine
        eng = Mars100Engine(seed=42)
        yr = eng.tick()
        assert "breakdowns" in yr.psychology
        assert "avg_stress" in yr.psychology
        assert "avg_morale" in yr.psychology

    def test_simulation_result_has_psychology(self):
        from src.mars100.engine import Mars100Engine
        eng = Mars100Engine(seed=42)
        result = eng.run()
        d = result.to_dict()
        assert "final_psychology" in d
        assert "total_breakdowns" in d["summary"]
        assert isinstance(result.total_breakdowns, int)

    def test_breakdown_forces_rest(self):
        """A broken-down colonist should get action='breakdown'."""
        from src.mars100.engine import Mars100Engine
        found = False
        for seed in range(100):
            eng = Mars100Engine(seed=seed)
            result = eng.run()
            for yr in result.years:
                if yr.psychology.get("breakdowns"):
                    for bd in yr.psychology["breakdowns"]:
                        cid = bd["colonist_id"]
                        if cid in yr.actions:
                            assert yr.actions[cid] == "breakdown"
                            found = True
                            break
                if found:
                    break
            if found:
                break

    def test_psych_states_initialized_for_all(self):
        from src.mars100.engine import Mars100Engine
        eng = Mars100Engine(seed=42)
        active = [c.id for c in eng.colonists if c.alive]
        for cid in active:
            assert cid in eng.psych_states

    def test_version_8_0(self):
        from src.mars100.engine import Mars100Engine
        eng = Mars100Engine(seed=42)
        result = eng.run()
        d = result.to_dict()
        assert d["_meta"]["version"] == "8.0"
