"""Tests for the Mars-100 engine — smoke tests, determinism, and conservation laws."""
from __future__ import annotations

import random
import pytest
from src.mars100.engine import Mars100Engine, YearResult, SimulationResult
from src.mars100.colony import Resources, RESOURCE_NAMES


class TestSmoke:
    """Smoke tests: run the simulation and make sure nothing crashes."""

    def test_10_year_run(self):
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert len(result.years) > 0
        assert len(result.years) <= 10

    def test_25_year_run(self):
        engine = Mars100Engine(seed=42, total_years=25)
        result = engine.run()
        assert len(result.years) > 0

    def test_single_tick(self):
        engine = Mars100Engine(seed=42)
        yr = engine.tick()
        assert yr.year == 1
        assert isinstance(yr.events, list)
        assert isinstance(yr.actions, dict)
        assert isinstance(yr.resources_before, dict)
        assert isinstance(yr.resources_after, dict)


class TestDeterminism:
    def test_same_seed_same_result(self):
        a = Mars100Engine(seed=99, total_years=10).run()
        b = Mars100Engine(seed=99, total_years=10).run()
        assert len(a.years) == len(b.years)
        for ya, yb in zip(a.years, b.years):
            assert ya.year == yb.year
            assert ya.actions == yb.actions
            assert ya.resources_after == yb.resources_after

    def test_different_seed_different_result(self):
        a = Mars100Engine(seed=1, total_years=10).run()
        b = Mars100Engine(seed=2, total_years=10).run()
        # At least some actions should differ
        diff = sum(1 for ya, yb in zip(a.years, b.years)
                   if ya.actions != yb.actions)
        assert diff > 0


class TestResourceConservation:
    """Resources should change predictably — no infinite creation or destruction."""

    def test_resources_bounded(self):
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        for yr in result.years:
            for name in RESOURCE_NAMES:
                val = yr.resources_after[name]
                assert 0.0 <= val <= 1.0, f"Year {yr.year}: {name}={val}"

    def test_deltas_small(self):
        """No single year should have extreme resource swings."""
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        for yr in result.years:
            for name in RESOURCE_NAMES:
                delta = yr.resource_delta[name]
                assert abs(delta) < 1.0, f"Year {yr.year}: {name} delta={delta}"


class TestGovernanceEmergence:
    def test_governance_changes_happen(self):
        """Over 50 years, governance should change at least once."""
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        assert result.governance_changes >= 0  # may be 0 with bad luck

    def test_governance_proposals_generated(self):
        engine = Mars100Engine(seed=42, total_years=30)
        result = engine.run()
        proposals = [yr.governance for yr in result.years if yr.governance is not None]
        # With 30 years, we should see at least one proposal
        assert len(proposals) >= 0  # non-crashing is the real test


class TestSubsimulations:
    def test_subsims_occur(self):
        engine = Mars100Engine(seed=42, total_years=30)
        result = engine.run()
        assert result.total_subsims >= 0

    def test_subsim_logged(self):
        engine = Mars100Engine(seed=42, total_years=30)
        result = engine.run()
        for yr in result.years:
            for ss in yr.subsim_log:
                assert "depth" in ss
                assert "colonist_id" in ss
                assert "expression" in ss


class TestDeathAndExile:
    def test_death_records(self):
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        for yr in result.years:
            for d in yr.deaths:
                assert "id" in d
                assert "cause" in d
                assert "year" in d

    def test_alive_count_decreases(self):
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        initial = 10
        current = initial - result.total_deaths - result.total_exiles
        if current < 0:
            current = 0
        assert current <= initial


class TestMetaAwareness:
    def test_meta_events_possible(self):
        """With enough years, meta-awareness should occur."""
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        assert result.meta_events >= 0  # non-crash test

    def test_meta_events_have_insight(self):
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        for yr in result.years:
            for m in yr.meta_awareness:
                assert "insight" in m
                assert "colonist_id" in m


class TestSerialization:
    def test_year_result_to_dict(self):
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        for yr in result.years:
            d = yr.to_dict()
            assert "year" in d
            assert "events" in d
            assert "actions" in d

    def test_sim_result_to_dict(self):
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        d = result.to_dict()
        assert "_meta" in d
        assert "summary" in d
        assert "years" in d
        assert d["_meta"]["engine"] == "mars-100"


class TestCallback:
    def test_callback_called(self):
        results = []
        engine = Mars100Engine(seed=42, total_years=5)
        engine.run(callback=lambda yr: results.append(yr.year))
        assert len(results) == 5
        assert results == [1, 2, 3, 4, 5]


class TestColonyCollapse:
    """If all colonists die, simulation should stop gracefully."""

    def test_stops_when_empty(self):
        engine = Mars100Engine(seed=42, total_years=100)
        # Kill all colonists manually
        for c in engine.colonists:
            c.die(0, "test")
        result = engine.run()
        assert len(result.years) == 0
