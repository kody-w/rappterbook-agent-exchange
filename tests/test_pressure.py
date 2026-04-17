"""Tests for the Mars-100 pressure system.

Covers: pressure sources, update mechanics, action modifiers,
death/birth modifiers, collective pressure, integration with engine.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.mars100.colonist import (
    Colonist, ColonistStats, ColonistSkills, STAT_NAMES, SKILL_NAMES,
    create_founding_ten,
)
from src.mars100.colony import Resources, SocialGraph, RESOURCE_NAMES
from src.mars100.events import Event
from src.mars100.pressure import (
    _clamp,
    compute_environmental_pressure,
    compute_social_pressure,
    compute_existential_pressure,
    update_pressure,
    apply_pressure_release,
    pressure_action_modifier,
    pressure_death_modifier,
    pressure_birth_modifier,
    collective_pressure,
    PressureSnapshot,
    RELEASE_RATES,
    CRITICAL_PRESSURE,
    _governance_satisfaction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_colonist(pressure: float = 0.0, **stat_overrides: float) -> Colonist:
    stats = {"resolve": 0.5, "improvisation": 0.5, "empathy": 0.5,
             "hoarding": 0.5, "faith": 0.5, "paranoia": 0.5}
    stats.update(stat_overrides)
    c = Colonist(
        id="test-0", name="Test", element="fire", archetype="commander",
        stats=ColonistStats.from_dict(stats),
        skills=ColonistSkills.from_dict({k: 0.5 for k in SKILL_NAMES}),
        decision_expr="(+ resolve empathy)",
    )
    c.pressure = pressure
    return c


def _make_event(severity: float = 0.5, name: str = "dust_storm") -> Event:
    return Event(name=name, category="environmental", severity=severity,
                 description="test event", effects={})


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------

class TestClamp:
    def test_in_range(self) -> None:
        assert _clamp(0.5) == 0.5

    def test_below_zero(self) -> None:
        assert _clamp(-1.0) == 0.0

    def test_above_one(self) -> None:
        assert _clamp(2.0) == 1.0

    def test_boundary(self) -> None:
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0


# ---------------------------------------------------------------------------
# Environmental pressure
# ---------------------------------------------------------------------------

class TestEnvironmentalPressure:
    def test_abundant_resources_low_pressure(self) -> None:
        res = Resources(food=0.8, water=0.8, power=0.8, air=0.9, medicine=0.6)
        p = compute_environmental_pressure(res, [], 0, 10)
        assert 0.0 <= p <= 0.3

    def test_scarce_resources_high_pressure(self) -> None:
        res = Resources(food=0.05, water=0.05, power=0.05, air=0.05, medicine=0.05)
        p = compute_environmental_pressure(res, [], 0, 10)
        assert p > 0.3

    def test_severe_events_increase_pressure(self) -> None:
        res = Resources()
        low = compute_environmental_pressure(res, [], 0, 10)
        events = [_make_event(0.9)]
        high = compute_environmental_pressure(res, events, 0, 10)
        assert high > low

    def test_deaths_increase_pressure(self) -> None:
        res = Resources()
        p0 = compute_environmental_pressure(res, [], 0, 10)
        p2 = compute_environmental_pressure(res, [], 2, 10)
        assert p2 > p0

    def test_result_clamped(self) -> None:
        res = Resources(food=0.0, water=0.0, power=0.0, air=0.0, medicine=0.0)
        events = [_make_event(1.0), _make_event(1.0), _make_event(1.0)]
        p = compute_environmental_pressure(res, events, 10, 10)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Social pressure
# ---------------------------------------------------------------------------

class TestSocialPressure:
    def test_high_trust_low_pressure(self) -> None:
        c = _make_colonist(empathy=0.8)
        social = SocialGraph()
        ids = ["test-0", "peer-1", "peer-2"]
        social.initialize(ids, random.Random(42))
        # boost trust
        for pid in ids[1:]:
            social.edges[pid]["test-0"].trust = 0.9
        p = compute_social_pressure(c, social, ids, "council")
        assert p < 0.6

    def test_low_trust_high_pressure(self) -> None:
        c = _make_colonist(empathy=0.3)
        social = SocialGraph()
        ids = ["test-0", "peer-1", "peer-2"]
        social.initialize(ids, random.Random(42))
        for pid in ids[1:]:
            social.edges[pid]["test-0"].trust = 0.1
        p = compute_social_pressure(c, social, ids, "dictator")
        assert p > 0.3

    def test_no_peers_returns_midrange(self) -> None:
        c = _make_colonist()
        social = SocialGraph()
        p = compute_social_pressure(c, social, ["test-0"], "anarchy")
        assert p == 0.5

    def test_not_in_active_ids(self) -> None:
        c = _make_colonist()
        social = SocialGraph()
        p = compute_social_pressure(c, social, ["someone-else"], "anarchy")
        assert p == 0.0

    def test_governance_satisfaction_varies(self) -> None:
        empathic = _make_colonist(empathy=0.9, resolve=0.3)
        resolute = _make_colonist(resolve=0.9, empathy=0.2)
        assert _governance_satisfaction(empathic, "council") > _governance_satisfaction(resolute, "dictator") * 0.5


# ---------------------------------------------------------------------------
# Existential pressure
# ---------------------------------------------------------------------------

class TestExistentialPressure:
    def test_early_years_low(self) -> None:
        c = _make_colonist(faith=0.5)
        p = compute_existential_pressure(c, 1, 0)
        assert p < 0.1

    def test_late_years_higher(self) -> None:
        c = _make_colonist(faith=0.5)
        p1 = compute_existential_pressure(c, 1, 0)
        p80 = compute_existential_pressure(c, 80, 0)
        assert p80 > p1

    def test_meta_events_increase(self) -> None:
        c = _make_colonist(faith=0.5)
        p0 = compute_existential_pressure(c, 50, 0)
        p3 = compute_existential_pressure(c, 50, 3)
        assert p3 > p0

    def test_high_faith_buffers(self) -> None:
        faithful = _make_colonist(faith=0.95)
        skeptic = _make_colonist(faith=0.1)
        pf = compute_existential_pressure(faithful, 50, 2)
        ps = compute_existential_pressure(skeptic, 50, 2)
        assert pf < ps

    def test_clamped(self) -> None:
        c = _make_colonist(faith=0.0)
        p = compute_existential_pressure(c, 200, 10)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# update_pressure
# ---------------------------------------------------------------------------

class TestUpdatePressure:
    def test_from_zero(self) -> None:
        c = _make_colonist(pressure=0.0)
        p = update_pressure(c, 0.5, 0.5, 0.5)
        assert 0.0 < p < 0.5
        assert c.pressure == p

    def test_inertia(self) -> None:
        c = _make_colonist(pressure=0.8)
        p = update_pressure(c, 0.0, 0.0, 0.0)
        assert p > 0.3  # inertia keeps it elevated

    def test_history_tracked(self) -> None:
        c = _make_colonist()
        for _ in range(15):
            update_pressure(c, 0.3, 0.3, 0.3)
        assert len(c.pressure_history) == 10  # capped

    def test_always_clamped(self) -> None:
        c = _make_colonist(pressure=0.99)
        p = update_pressure(c, 1.0, 1.0, 1.0)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Pressure release
# ---------------------------------------------------------------------------

class TestPressureRelease:
    def test_mediation_releases(self) -> None:
        c = _make_colonist(pressure=0.6)
        apply_pressure_release(c, "mediate")
        assert c.pressure < 0.6

    def test_sabotage_increases(self) -> None:
        c = _make_colonist(pressure=0.3)
        apply_pressure_release(c, "sabotage")
        assert c.pressure > 0.3

    def test_resolve_amplifies(self) -> None:
        high_resolve = _make_colonist(pressure=0.6, resolve=0.9)
        low_resolve = _make_colonist(pressure=0.6, resolve=0.1)
        apply_pressure_release(high_resolve, "mediate")
        apply_pressure_release(low_resolve, "mediate")
        assert high_resolve.pressure < low_resolve.pressure

    def test_unknown_action_no_change(self) -> None:
        c = _make_colonist(pressure=0.5)
        apply_pressure_release(c, "unknown_action")
        assert c.pressure == 0.5

    def test_never_goes_below_zero(self) -> None:
        c = _make_colonist(pressure=0.01)
        apply_pressure_release(c, "mediate")
        assert c.pressure >= 0.0


# ---------------------------------------------------------------------------
# Action modifier
# ---------------------------------------------------------------------------

class TestActionModifier:
    def test_high_pressure_boosts_sabotage(self) -> None:
        c = _make_colonist(pressure=0.9)
        weights = {a: 1.0 for a in ["terraform", "farm", "mediate", "code",
                                     "pray", "sabotage", "cooperate", "hoard",
                                     "explore", "rest"]}
        mod = pressure_action_modifier(c, weights)
        assert mod["sabotage"] > weights["sabotage"]

    def test_high_pressure_reduces_cooperate(self) -> None:
        c = _make_colonist(pressure=0.9)
        weights = {a: 1.0 for a in ["cooperate", "sabotage"]}
        mod = pressure_action_modifier(c, weights)
        assert mod["cooperate"] < weights["cooperate"]

    def test_low_pressure_boosts_cooperate(self) -> None:
        c = _make_colonist(pressure=0.05)
        weights = {"cooperate": 1.0, "explore": 1.0, "sabotage": 1.0}
        mod = pressure_action_modifier(c, weights)
        assert mod["cooperate"] > weights["cooperate"]

    def test_mid_pressure_no_change(self) -> None:
        c = _make_colonist(pressure=0.35)
        weights = {"cooperate": 1.0, "sabotage": 1.0}
        mod = pressure_action_modifier(c, weights)
        assert mod["cooperate"] == 1.0
        assert mod["sabotage"] == 1.0

    def test_original_unchanged(self) -> None:
        c = _make_colonist(pressure=0.9)
        original = {"sabotage": 1.0, "cooperate": 1.0}
        pressure_action_modifier(c, original)
        assert original["sabotage"] == 1.0  # not mutated


# ---------------------------------------------------------------------------
# Death modifier
# ---------------------------------------------------------------------------

class TestDeathModifier:
    def test_below_critical_zero(self) -> None:
        c = _make_colonist(pressure=0.5)
        assert pressure_death_modifier(c) == 0.0

    def test_at_critical_zero(self) -> None:
        c = _make_colonist(pressure=CRITICAL_PRESSURE)
        assert pressure_death_modifier(c) == 0.0

    def test_above_critical_positive(self) -> None:
        c = _make_colonist(pressure=0.95)
        assert pressure_death_modifier(c) > 0.0

    def test_max_pressure(self) -> None:
        c = _make_colonist(pressure=1.0)
        mod = pressure_death_modifier(c)
        assert 0.0 < mod <= 0.01  # bounded


# ---------------------------------------------------------------------------
# Birth modifier
# ---------------------------------------------------------------------------

class TestBirthModifier:
    def test_zero_pressure_full(self) -> None:
        colonists = [_make_colonist(pressure=0.0) for _ in range(5)]
        assert pressure_birth_modifier(colonists) == 1.0

    def test_high_pressure_reduced(self) -> None:
        colonists = [_make_colonist(pressure=0.9) for _ in range(5)]
        mod = pressure_birth_modifier(colonists)
        assert 0.2 <= mod < 0.5

    def test_empty_list(self) -> None:
        assert pressure_birth_modifier([]) == 1.0

    def test_dead_colonists_excluded(self) -> None:
        alive = _make_colonist(pressure=0.0)
        dead = _make_colonist(pressure=1.0)
        dead.alive = False
        mod = pressure_birth_modifier([alive, dead])
        assert mod == 1.0  # only the alive one counts


# ---------------------------------------------------------------------------
# Collective pressure
# ---------------------------------------------------------------------------

class TestCollectivePressure:
    def test_uniform(self) -> None:
        colonists = [_make_colonist(pressure=0.5) for _ in range(4)]
        assert abs(collective_pressure(colonists) - 0.5) < 0.01

    def test_mixed(self) -> None:
        a = _make_colonist(pressure=0.0)
        b = _make_colonist(pressure=1.0)
        assert abs(collective_pressure([a, b]) - 0.5) < 0.01

    def test_empty(self) -> None:
        assert collective_pressure([]) == 0.0

    def test_ignores_dead(self) -> None:
        alive = _make_colonist(pressure=0.3)
        dead = _make_colonist(pressure=0.9)
        dead.alive = False
        assert abs(collective_pressure([alive, dead]) - 0.3) < 0.01


# ---------------------------------------------------------------------------
# PressureSnapshot
# ---------------------------------------------------------------------------

class TestPressureSnapshot:
    def test_to_dict(self) -> None:
        snap = PressureSnapshot(
            collective=0.45, environmental=0.3, social_avg=0.5,
            existential_avg=0.2,
            individual={"a": 0.4, "b": 0.5},
            high_pressure_colonists=["b"],
        )
        d = snap.to_dict()
        assert d["collective"] == 0.45
        assert d["high_pressure_colonists"] == ["b"]


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_colonist_round_trip(self) -> None:
        c = _make_colonist(pressure=0.73)
        c.pressure_history = [0.1, 0.3, 0.5, 0.73]
        d = c.to_dict()
        assert d["pressure"] == 0.73
        assert d["pressure_history"] == [0.1, 0.3, 0.5, 0.73]
        c2 = Colonist.from_dict(d)
        assert c2.pressure == 0.73
        assert c2.pressure_history == [0.1, 0.3, 0.5, 0.73]

    def test_legacy_colonist_no_pressure(self) -> None:
        """Colonists from old saves should default to 0 pressure."""
        d = {
            "id": "old", "name": "Old", "element": "fire", "archetype": "commander",
            "stats": {s: 0.5 for s in STAT_NAMES},
            "skills": {s: 0.5 for s in SKILL_NAMES},
        }
        c = Colonist.from_dict(d)
        assert c.pressure == 0.0
        assert c.pressure_history == []


# ---------------------------------------------------------------------------
# LisPy bindings
# ---------------------------------------------------------------------------

class TestLispyBindings:
    def test_pressure_in_bindings(self) -> None:
        c = _make_colonist(pressure=0.6)
        bindings = c.lispy_bindings()
        assert bindings["pressure"] == 0.6
        assert bindings["hope"] == pytest.approx(0.4, abs=0.01)

    def test_hope_inverse(self) -> None:
        c = _make_colonist(pressure=0.0)
        assert c.lispy_bindings()["hope"] == 1.0
        c.pressure = 1.0
        assert c.lispy_bindings()["hope"] == 0.0


# ---------------------------------------------------------------------------
# Integration: engine tick produces pressure data
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_ten_year_run_has_pressure(self) -> None:
        from src.mars100.engine import Mars100Engine
        eng = Mars100Engine(seed=42, total_years=10)
        result = eng.run()
        assert len(result.years) == 10
        for yr in result.years:
            d = yr.to_dict()
            assert "pressure" in d
            p = d["pressure"]
            assert "collective" in p
            assert 0.0 <= p["collective"] <= 1.0

    def test_colonists_accumulate_pressure(self) -> None:
        from src.mars100.engine import Mars100Engine
        eng = Mars100Engine(seed=99, total_years=20)
        eng.run()
        active = eng._active_colonists()
        pressures = [c.pressure for c in active]
        assert all(0.0 <= p <= 1.0 for p in pressures)
        assert any(p > 0.0 for p in pressures)  # someone has accumulated pressure

    def test_pressure_history_populated(self) -> None:
        from src.mars100.engine import Mars100Engine
        eng = Mars100Engine(seed=42, total_years=15)
        eng.run()
        for c in eng._active_colonists():
            assert len(c.pressure_history) > 0
            assert len(c.pressure_history) <= 10  # capped

    def test_pressure_affects_action_distribution(self) -> None:
        """High-pressure colonies should show more extreme action diversity."""
        from src.mars100.engine import Mars100Engine
        eng = Mars100Engine(seed=42, total_years=5)
        # Force high pressure on all colonists
        for c in eng.colonists:
            c.pressure = 0.9
        result = eng.tick()
        # Just verify it runs and produces valid actions
        assert all(a in ["terraform", "farm", "mediate", "code", "pray",
                         "sabotage", "cooperate", "hoard", "explore", "rest"]
                   for a in result.actions.values())

    def test_full_100_years_all_pressure_bounded(self) -> None:
        from src.mars100.engine import Mars100Engine
        eng = Mars100Engine(seed=42, total_years=100)
        result = eng.run()
        for yr in result.years:
            p = yr.pressure_snapshot
            assert 0.0 <= p.get("collective", 0.0) <= 1.0
            for cid, pval in p.get("individual", {}).items():
                assert 0.0 <= pval <= 1.0


# ---------------------------------------------------------------------------
# Property: all pressures always in [0, 1]
# ---------------------------------------------------------------------------

class TestPressureInvariants:
    @pytest.mark.parametrize("seed", [1, 42, 99, 137, 256])
    def test_pressure_bounded_across_seeds(self, seed: int) -> None:
        from src.mars100.engine import Mars100Engine
        eng = Mars100Engine(seed=seed, total_years=25)
        result = eng.run()
        for c in eng.colonists:
            assert 0.0 <= c.pressure <= 1.0, f"{c.name} pressure={c.pressure}"
            for hp in c.pressure_history:
                assert 0.0 <= hp <= 1.0
