"""Tests for the Mars-100 simulation engine (src/mars100_sim.py).

Covers: full sim smoke test, year simulation, resource consumption,
governance (proposals, voting, laws), death mechanics, sub-sim spawning,
determinism, and invariant checks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.mars100_sim import (
    CONSUMPTION,
    EVENT_TABLE,
    INITIAL_RESOURCES,
    TOTAL_YEARS,
    VALID_ACTIONS,
    Law,
    Mars100Simulation,
    Proposal,
    YearDelta,
)


# ---------------------------------------------------------------------------
# Smoke test — run 10 years without crash
# ---------------------------------------------------------------------------


class TestSmokeTest:
    def test_run_10_years(self) -> None:
        sim = Mars100Simulation(seed=42, total_years=10)
        results = sim.run()
        assert results["_meta"]["engine"] == "mars-100"
        assert results["_meta"]["years_survived"] <= 10
        assert results["summary"]["colonists_start"] == 10

    def test_run_100_years(self) -> None:
        sim = Mars100Simulation(seed=42, total_years=100)
        results = sim.run()
        assert results["_meta"]["years_survived"] <= 100
        assert results["_meta"]["years_survived"] > 0

    def test_results_serializable(self) -> None:
        """Results must be JSON-serializable."""
        sim = Mars100Simulation(seed=42, total_years=10)
        results = sim.run()
        serialized = json.dumps(results)
        assert len(serialized) > 0
        roundtrip = json.loads(serialized)
        assert roundtrip["_meta"]["engine"] == "mars-100"

    @pytest.mark.parametrize("seed", range(5))
    def test_multiple_seeds_no_crash(self, seed: int) -> None:
        sim = Mars100Simulation(seed=seed, total_years=20)
        results = sim.run()
        assert results["_meta"]["years_survived"] > 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_result(self) -> None:
        sim1 = Mars100Simulation(seed=42, total_years=20)
        sim2 = Mars100Simulation(seed=42, total_years=20)
        r1 = sim1.run()
        r2 = sim2.run()
        assert r1["summary"]["years_survived"] == r2["summary"]["years_survived"]
        assert r1["summary"]["deaths"] == r2["summary"]["deaths"]
        assert r1["summary"]["laws_enacted"] == r2["summary"]["laws_enacted"]

    def test_different_seeds_differ(self) -> None:
        sim1 = Mars100Simulation(seed=42, total_years=50)
        sim2 = Mars100Simulation(seed=99, total_years=50)
        r1 = sim1.run()
        r2 = sim2.run()
        # At least something should differ
        assert (r1["summary"]["deaths"] != r2["summary"]["deaths"]
                or r1["summary"]["laws_enacted"] != r2["summary"]["laws_enacted"]
                or r1["summary"]["years_survived"] != r2["summary"]["years_survived"])


# ---------------------------------------------------------------------------
# Resource mechanics
# ---------------------------------------------------------------------------


class TestResources:
    def test_initial_resources(self) -> None:
        sim = Mars100Simulation(seed=42)
        for key, val in INITIAL_RESOURCES.items():
            assert sim.resources[key] == val

    def test_resources_non_negative(self) -> None:
        """All resources stay >= 0 throughout simulation."""
        sim = Mars100Simulation(seed=42, total_years=50)
        results = sim.run()
        for delta in results["year_deltas"]:
            for key, val in delta["colony_state"].items():
                if isinstance(val, (int, float)):
                    assert val >= 0, f"Year {delta['year']}: {key} = {val}"

    def test_morale_bounded(self) -> None:
        """Morale stays in [0, 1]."""
        sim = Mars100Simulation(seed=42, total_years=50)
        results = sim.run()
        for delta in results["year_deltas"]:
            morale = delta["colony_state"].get("morale", 0)
            assert 0.0 <= morale <= 1.0, f"Year {delta['year']}: morale = {morale}"

    def test_consumption_reduces_food(self) -> None:
        """Food should decrease after consumption step."""
        sim = Mars100Simulation(seed=42, total_years=1)
        initial_food = sim.resources["food"]
        sim.run()
        # Even with gathering, 10 colonists consuming 8 food/year each
        # means net change is detectable
        assert sim.resources["food"] != initial_food


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class TestEvents:
    def test_event_probabilities_sum_to_one(self) -> None:
        total = sum(prob for _, prob in EVENT_TABLE)
        assert abs(total - 1.0) < 0.01

    def test_roll_event_returns_valid(self) -> None:
        sim = Mars100Simulation(seed=42)
        valid = {name for name, _ in EVENT_TABLE}
        for _ in range(100):
            event = sim._roll_event()
            assert event in valid


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------


class TestGovernance:
    def test_proposals_emerge(self) -> None:
        """Over 50 years, at least one proposal should emerge."""
        sim = Mars100Simulation(seed=42, total_years=50)
        results = sim.run()
        assert results["summary"]["proposals_total"] > 0

    def test_laws_enacted(self) -> None:
        """Over 50 years, at least one law should be enacted."""
        sim = Mars100Simulation(seed=42, total_years=50)
        results = sim.run()
        assert results["summary"]["laws_enacted"] > 0

    def test_law_has_required_fields(self) -> None:
        sim = Mars100Simulation(seed=42, total_years=50)
        results = sim.run()
        if results["laws"]:
            law = results["laws"][0]
            for field in ("id", "year_proposed", "year_enacted", "proposer",
                          "title", "effect_type"):
                assert field in law

    def test_governance_patterns_present(self) -> None:
        sim = Mars100Simulation(seed=42, total_years=50)
        results = sim.run()
        patterns = results["summary"]["governance_patterns"]
        assert isinstance(patterns, list)
        assert len(patterns) > 0


# ---------------------------------------------------------------------------
# Death mechanics
# ---------------------------------------------------------------------------


class TestDeaths:
    def test_dead_colonist_archived(self) -> None:
        """Run long enough that some colonists die; check archives."""
        sim = Mars100Simulation(seed=7, total_years=100)
        results = sim.run()
        if results["summary"]["deaths"] > 0:
            assert len(results["archives"]) > 0
            soul = results["archives"][0]
            assert "epitaph" in soul
            assert soul["death_year"] is not None

    def test_dead_colonist_not_alive(self) -> None:
        sim = Mars100Simulation(seed=7, total_years=100)
        results = sim.run()
        for c in results["colonists"]:
            if c["death_year"] is not None:
                assert c["alive"] is False


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class TestActions:
    def test_all_actions_valid(self) -> None:
        sim = Mars100Simulation(seed=42, total_years=20)
        results = sim.run()
        for delta in results["year_deltas"]:
            for colonist_id, action in delta["actions"].items():
                assert action in VALID_ACTIONS, f"{colonist_id} took invalid action: {action}"


# ---------------------------------------------------------------------------
# Year deltas
# ---------------------------------------------------------------------------


class TestYearDeltas:
    def test_deltas_sequential(self) -> None:
        sim = Mars100Simulation(seed=42, total_years=20)
        results = sim.run()
        years = [d["year"] for d in results["year_deltas"]]
        assert years == list(range(1, len(years) + 1))

    def test_deltas_have_diary(self) -> None:
        sim = Mars100Simulation(seed=42, total_years=5)
        results = sim.run()
        for delta in results["year_deltas"]:
            assert "diary_entries" in delta
            # Should have up to 3 narrators
            assert len(delta["diary_entries"]) <= 3

    def test_delta_has_colony_state(self) -> None:
        sim = Mars100Simulation(seed=42, total_years=5)
        results = sim.run()
        for delta in results["year_deltas"]:
            assert "colony_state" in delta
            assert "food" in delta["colony_state"]
            assert "morale" in delta["colony_state"]


# ---------------------------------------------------------------------------
# Colonist summary
# ---------------------------------------------------------------------------


class TestColonistSummary:
    def test_all_10_colonists_in_results(self) -> None:
        sim = Mars100Simulation(seed=42, total_years=10)
        results = sim.run()
        assert len(results["colonists"]) == 10

    def test_colonist_has_stats_and_skills(self) -> None:
        sim = Mars100Simulation(seed=42, total_years=10)
        results = sim.run()
        for c in results["colonists"]:
            assert "stats" in c
            assert "skills" in c
            assert len(c["stats"]) == 6
            assert len(c["skills"]) == 6

    def test_colonist_memory_capped(self) -> None:
        """Memory in results is capped to last 10."""
        sim = Mars100Simulation(seed=42, total_years=100)
        results = sim.run()
        for c in results["colonists"]:
            assert len(c["memory"]) <= 10


# ---------------------------------------------------------------------------
# Invariant: stat values always in [0, 1]
# ---------------------------------------------------------------------------


class TestInvariants:
    @pytest.mark.parametrize("seed", range(5))
    def test_final_stats_bounded(self, seed: int) -> None:
        sim = Mars100Simulation(seed=seed, total_years=50)
        results = sim.run()
        for c in results["colonists"]:
            for stat_name, val in c["stats"].items():
                assert 0.0 <= val <= 1.0, f"{c['id']}.{stat_name} = {val}"
            for skill_name, val in c["skills"].items():
                assert 0.0 <= val <= 1.0, f"{c['id']}.{skill_name} = {val}"

    @pytest.mark.parametrize("seed", range(3))
    def test_freedom_bounded(self, seed: int) -> None:
        sim = Mars100Simulation(seed=seed, total_years=50)
        results = sim.run()
        for delta in results["year_deltas"]:
            freedom = delta["colony_state"].get("freedom", 0)
            assert 0.0 <= freedom <= 1.0
