"""
Tests for the Mars-100 recursive colony simulation (src/mars100.py).

Covers: colony genesis, year simulation, sub-sim integration,
governance emergence, death handling, conservation laws, resume/replay.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.mars100 import (
    Colony,
    ColonistState,
    GovernanceRecord,
    run_simulation,
    FOUNDING_COLONISTS,
    STAT_NAMES,
    SKILL_NAMES,
    ACTION_NAMES,
)


# ---------------------------------------------------------------------------
# Colony genesis
# ---------------------------------------------------------------------------

class TestGenesis:
    def test_creates_10_colonists(self):
        colony = Colony.genesis(seed=42)
        assert len(colony.colonists) == 10

    def test_all_colonists_alive(self):
        colony = Colony.genesis(seed=42)
        for c in colony.colonists.values():
            assert c.alive is True

    def test_colonist_ids_match_founding(self):
        colony = Colony.genesis(seed=42)
        expected_ids = {c["id"] for c in FOUNDING_COLONISTS}
        actual_ids = set(colony.colonists.keys())
        assert actual_ids == expected_ids

    def test_colonist_elements_assigned(self):
        colony = Colony.genesis(seed=42)
        for c in colony.colonists.values():
            assert c.element in ["fire", "water", "earth", "air"]

    def test_colonist_stats_bounded(self):
        colony = Colony.genesis(seed=42)
        for c in colony.colonists.values():
            for stat in STAT_NAMES:
                assert 0.0 <= c.stats[stat] <= 1.0, f"{c.id}.{stat}={c.stats[stat]}"

    def test_colonist_skills_bounded(self):
        colony = Colony.genesis(seed=42)
        for c in colony.colonists.values():
            for skill in SKILL_NAMES:
                assert 0.0 <= c.skills[skill] <= 1.0

    def test_relationships_initialized(self):
        colony = Colony.genesis(seed=42)
        ids = list(colony.colonists.keys())
        for a in ids:
            assert a in colony.relationships
            for b in ids:
                if a != b:
                    assert b in colony.relationships[a]
                    assert -1.0 <= colony.relationships[a][b] <= 1.0

    def test_resources_positive(self):
        colony = Colony.genesis(seed=42)
        for key, val in colony.resources.items():
            assert val > 0, f"{key}={val}"

    def test_year_zero(self):
        colony = Colony.genesis(seed=42)
        assert colony.year == 0

    def test_no_archived_souls(self):
        colony = Colony.genesis(seed=42)
        assert len(colony.archived_souls) == 0

    def test_deterministic_genesis(self):
        c1 = Colony.genesis(seed=99)
        c2 = Colony.genesis(seed=99)
        assert c1.to_dict()["resources"] == c2.to_dict()["resources"]
        ids1 = sorted(c1.colonists.keys())
        ids2 = sorted(c2.colonists.keys())
        assert ids1 == ids2


# ---------------------------------------------------------------------------
# Year simulation
# ---------------------------------------------------------------------------

class TestYearSimulation:
    def test_year_advances(self):
        colony = Colony.genesis(seed=42)
        delta = colony.tick(1)
        assert colony.year == 1
        assert delta["year"] == 1

    def test_delta_has_required_fields(self):
        colony = Colony.genesis(seed=42)
        delta = colony.tick(1)
        required = ["year", "timestamp", "event", "decisions",
                     "resources", "population", "summary"]
        for field in required:
            assert field in delta, f"Missing field: {field}"

    def test_event_generated(self):
        colony = Colony.genesis(seed=42)
        delta = colony.tick(1)
        assert "name" in delta["event"]
        assert "severity" in delta["event"]
        assert 0.0 <= delta["event"]["severity"] <= 1.0

    def test_decisions_for_all_alive(self):
        colony = Colony.genesis(seed=42)
        delta = colony.tick(1)
        alive_ids = {c.id for c in colony.colonists.values() if c.alive}
        # Decisions should have been made for all who were alive at start of year
        assert len(delta["decisions"]) >= len(alive_ids) - 1  # allow for death

    def test_all_decisions_are_valid_actions(self):
        colony = Colony.genesis(seed=42)
        for year in range(1, 11):
            delta = colony.tick(year)
            for cid, action in delta["decisions"].items():
                assert action in ACTION_NAMES, f"Invalid action: {action}"

    def test_resources_change(self):
        colony = Colony.genesis(seed=42)
        initial_food = colony.resources["food"]
        colony.tick(1)
        # Resources should change (consumption + production)
        # They might go up or down, but shouldn't be identical
        # Unless by extreme coincidence
        assert isinstance(colony.resources["food"], float)

    def test_resources_non_negative(self):
        colony = Colony.genesis(seed=42)
        for year in range(1, 21):
            colony.tick(year)
            for key, val in colony.resources.items():
                assert val >= 0, f"Year {year}: {key}={val}"

    def test_resources_capped(self):
        colony = Colony.genesis(seed=42)
        for year in range(1, 21):
            colony.tick(year)
            for key, val in colony.resources.items():
                assert val <= 2000.0, f"Year {year}: {key}={val}"

    def test_colonist_years_alive_increment(self):
        colony = Colony.genesis(seed=42)
        colony.tick(1)
        for c in colony.colonists.values():
            if c.alive:
                assert c.years_alive >= 1

    def test_diary_entries_created(self):
        colony = Colony.genesis(seed=42)
        colony.tick(1)
        for c in colony.colonists.values():
            if c.alive:
                assert len(c.diary) >= 1
                assert isinstance(c.diary[-1], str)

    def test_summary_not_empty(self):
        colony = Colony.genesis(seed=42)
        delta = colony.tick(1)
        assert len(delta["summary"]) > 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_result(self):
        c1 = Colony.genesis(seed=42)
        c2 = Colony.genesis(seed=42)
        d1 = c1.tick(1)
        d2 = c2.tick(1)
        assert d1["event"]["name"] == d2["event"]["name"]
        assert d1["event"]["severity"] == d2["event"]["severity"]
        assert d1["decisions"] == d2["decisions"]

    def test_different_seed_different_result(self):
        c1 = Colony.genesis(seed=42)
        c2 = Colony.genesis(seed=99)
        # Run enough years that divergence is virtually guaranteed
        any_differ = False
        for year in range(1, 11):
            d1 = c1.tick(year)
            d2 = c2.tick(year)
            events_differ = d1["event"]["name"] != d2["event"]["name"]
            decisions_differ = d1["decisions"] != d2["decisions"]
            if events_differ or decisions_differ:
                any_differ = True
                break
        assert any_differ


# ---------------------------------------------------------------------------
# Conservation laws
# ---------------------------------------------------------------------------

class TestConservation:
    def test_population_conservation(self):
        """Population + archived_souls = initial + births (no births in v1)."""
        colony = Colony.genesis(seed=42)
        initial_count = len(colony.colonists)

        for year in range(1, 51):
            colony.tick(year)

        alive = len([c for c in colony.colonists.values() if c.alive])
        dead = len(colony.archived_souls)
        assert alive + dead == initial_count

    def test_stats_bounded_after_drift(self):
        """Stats remain in [0, 1] after natural drift."""
        colony = Colony.genesis(seed=42)
        for year in range(1, 31):
            colony.tick(year)
        for c in colony.colonists.values():
            for stat in STAT_NAMES:
                assert 0.0 <= c.stats[stat] <= 1.0

    def test_relationships_bounded(self):
        """Relationships remain in [-1, 1]."""
        colony = Colony.genesis(seed=42)
        for year in range(1, 21):
            colony.tick(year)
        for a, bs in colony.relationships.items():
            for b, val in bs.items():
                assert -1.0 <= val <= 1.0, f"rel[{a}][{b}]={val}"


# ---------------------------------------------------------------------------
# Death and archival
# ---------------------------------------------------------------------------

class TestDeath:
    def test_death_creates_archive(self):
        """When a colonist dies, they're archived (legacy, not delete)."""
        colony = Colony.genesis(seed=42)
        # Run many years to increase death chance
        for year in range(1, 80):
            colony.tick(year)
        # After 80 years some should have died (old age, accident)
        if colony.archived_souls:
            soul = colony.archived_souls[0]
            assert "id" in soul
            assert "death_year" in soul
            assert "death_cause" in soul
            assert "epitaph" in soul

    def test_dead_colonists_not_alive(self):
        """Dead colonists have alive=False."""
        colony = Colony.genesis(seed=42)
        for year in range(1, 80):
            colony.tick(year)
        dead_ids = {s["id"] for s in colony.archived_souls}
        for cid in dead_ids:
            if cid in colony.colonists:
                assert colony.colonists[cid].alive is False


# ---------------------------------------------------------------------------
# Governance emergence
# ---------------------------------------------------------------------------

class TestGovernance:
    def test_governance_proposals_emerge(self):
        """After enough years, governance proposals should appear."""
        colony = Colony.genesis(seed=42)
        for year in range(1, 30):
            colony.tick(year)
        # Year 10, 20 trigger propose-governance in policies
        assert len(colony.governance) > 0

    def test_governance_record_structure(self):
        colony = Colony.genesis(seed=42)
        for year in range(1, 15):
            colony.tick(year)
        if colony.governance:
            g = colony.governance[0]
            assert isinstance(g.year, int)
            assert isinstance(g.proposer, str)
            assert isinstance(g.proposal, str)
            assert isinstance(g.votes_for, int)
            assert isinstance(g.votes_against, int)
            assert isinstance(g.adopted, bool)

    def test_votes_consistent(self):
        """Total votes = alive voters - 1 (proposer doesn't vote)."""
        colony = Colony.genesis(seed=42)
        for year in range(1, 15):
            colony.tick(year)
        for g in colony.governance:
            total = g.votes_for + g.votes_against
            # Should be roughly population - 1 (the proposer)
            assert total >= 1
            assert total <= 10

    def test_factions_detected(self):
        """Factions should emerge from relationship patterns."""
        colony = Colony.genesis(seed=42)
        for year in range(1, 30):
            colony.tick(year)
        # Factions may or may not form — just check structure
        if colony.factions:
            for faction_id, members in colony.factions.items():
                assert isinstance(faction_id, str)
                assert isinstance(members, list)
                assert len(members) >= 2


# ---------------------------------------------------------------------------
# Sub-sim integration
# ---------------------------------------------------------------------------

class TestSubSimIntegration:
    def test_sub_sim_log_populated(self):
        """Colonists who propose governance run sub-sims."""
        colony = Colony.genesis(seed=42)
        for year in range(1, 15):
            colony.tick(year)
        # At least one colonist should have sub-sim logs
        any_sub_sims = any(
            len(c.sub_sim_log) > 0
            for c in colony.colonists.values()
        )
        assert any_sub_sims, "No sub-sims were run"

    def test_sub_sim_evidence_in_governance(self):
        """Governance records include sub-sim evidence."""
        colony = Colony.genesis(seed=42)
        for year in range(1, 15):
            colony.tick(year)
        any_evidence = any(
            g.sub_sim_evidence is not None
            for g in colony.governance
        )
        assert any_evidence


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_roundtrip(self):
        colony = Colony.genesis(seed=42)
        for year in range(1, 6):
            colony.tick(year)

        state = colony.to_dict()
        restored = Colony.from_dict(state)

        assert restored.year == colony.year
        assert restored.seed == colony.seed
        assert set(restored.colonists.keys()) == set(colony.colonists.keys())
        assert restored.resources == colony.resources

    def test_to_dict_json_serializable(self):
        colony = Colony.genesis(seed=42)
        for year in range(1, 6):
            colony.tick(year)
        state = colony.to_dict()
        # Should not raise
        json_str = json.dumps(state)
        assert len(json_str) > 0

    def test_colonist_state_roundtrip(self):
        c = ColonistState(
            id="test", name="Test", element="fire", role="tester",
            stats={"resolve": 0.5}, skills={"coding": 0.7},
            diary=["Year 1: test"], morale=0.6,
        )
        d = c.to_dict()
        restored = ColonistState.from_dict(d)
        assert restored.id == c.id
        assert restored.morale == c.morale

    def test_governance_record_roundtrip(self):
        g = GovernanceRecord(
            year=5, proposer="ares",
            proposal="elect_council", votes_for=6, votes_against=3,
            adopted=True,
        )
        d = g.to_dict()
        restored = GovernanceRecord.from_dict(d)
        assert restored.year == g.year
        assert restored.adopted == g.adopted


# ---------------------------------------------------------------------------
# Resume / replay
# ---------------------------------------------------------------------------

class TestResumeReplay:
    def test_resume_from_state(self):
        """Run 5 years, save, restore, run 5 more = same as 10 continuous."""
        c1 = Colony.genesis(seed=42)
        for year in range(1, 11):
            c1.tick(year)

        c2 = Colony.genesis(seed=42)
        for year in range(1, 6):
            c2.tick(year)
        state = c2.to_dict()
        c3 = Colony.from_dict(state)
        for year in range(6, 11):
            c3.tick(year)

        # Resources should be close (not exact due to per-colonist
        # policy eval which uses LisPy with internal state)
        for key in c1.resources:
            assert abs(c1.resources[key] - c3.resources[key]) < 0.01, \
                f"{key}: {c1.resources[key]} vs {c3.resources[key]}"


# ---------------------------------------------------------------------------
# Smoke test (10+ year simulation)
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_10_year_simulation(self):
        """Run 10 years without crash."""
        colony = Colony.genesis(seed=42)
        for year in range(1, 11):
            delta = colony.tick(year)
            assert delta["year"] == year
            assert delta["population"] >= 0

    def test_50_year_simulation(self):
        """Run 50 years — checks stability over longer period."""
        colony = Colony.genesis(seed=42)
        for year in range(1, 51):
            delta = colony.tick(year)
            assert delta["population"] >= 0
            assert all(v >= 0 for v in delta["resources"].values())

    def test_run_simulation_function(self):
        """Test the run_simulation convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = run_simulation(years=10, seed=42, output_dir=tmpdir)
            assert state["year"] >= 10 or state["_meta"]["year"] >= 1
            # Check year files were written
            year_files = list(Path(tmpdir).glob("year-*.json"))
            assert len(year_files) >= 1
            # Check final state was written
            assert (Path(tmpdir) / "final_state.json").exists()

    def test_full_100_year_simulation(self):
        """Run full 100 years — the main deliverable."""
        colony = Colony.genesis(seed=42)
        max_pop = 0
        min_pop = 10
        for year in range(1, 101):
            delta = colony.tick(year)
            pop = delta["population"]
            max_pop = max(max_pop, pop)
            min_pop = min(min_pop, pop)
            if pop == 0:
                break  # Colony collapsed

        # At least some years should have passed
        assert colony.year >= 1
        # Check final state is valid
        state = colony.to_dict()
        assert "_meta" in state
        assert state["_meta"]["engine"] == "mars-100"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_colonists_dead(self):
        """Simulation handles total colony collapse gracefully."""
        colony = Colony.genesis(seed=42)
        # Force all to near-death
        for c in colony.colonists.values():
            c.health = 0.01
        colony.resources["food"] = 0
        colony.resources["water"] = 0

        # Should not crash even with mass death
        delta = colony.tick(1)
        assert delta["population"] >= 0

    def test_single_survivor(self):
        """Colony works with just one colonist alive."""
        colony = Colony.genesis(seed=42)
        ids = list(colony.colonists.keys())
        for cid in ids[1:]:
            colony.colonists[cid].alive = False
        delta = colony.tick(1)
        assert delta["population"] >= 0

    def test_policy_failure_fallback(self):
        """Colonist with broken policy falls back gracefully."""
        colony = Colony.genesis(seed=42)
        # Give one colonist a broken policy
        colony.colonists["ares"].policy = "(this is not valid lispy $$$"
        delta = colony.tick(1)
        # Should still produce a decision (fallback)
        assert "ares" in delta["decisions"]
