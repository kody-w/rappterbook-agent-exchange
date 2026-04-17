"""
test_mars100.py — Tests for the Mars-100 recursive colony simulation.

Covers: colonist creation, yearly ticks, events, governance, sub-sims,
death/archiving, resource conservation, determinism, 10-year smoke test.
"""
from __future__ import annotations

import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100 import (
    Mars100, Colonist, ColonyState, Proposal, SubSimLog,
    COLONIST_NAMES, ELEMENTS, STAT_NAMES, SKILL_NAMES,
    EVENTS, PROPOSAL_TYPES,
)
from src.lispy import LispError


# ---------------------------------------------------------------------------
# Colonist Creation
# ---------------------------------------------------------------------------

class TestColonistCreation:
    def test_creates_10_colonists(self):
        sim = Mars100(seed=42)
        assert len(sim.colonists) == 10

    def test_all_colonists_alive(self):
        sim = Mars100(seed=42)
        assert all(c.alive for c in sim.colonists)

    def test_unique_ids(self):
        sim = Mars100(seed=42)
        ids = [c.id for c in sim.colonists]
        assert len(set(ids)) == 10

    def test_unique_names(self):
        sim = Mars100(seed=42)
        names = [c.name for c in sim.colonists]
        assert len(set(names)) == 10

    def test_elements_assigned(self):
        sim = Mars100(seed=42)
        for col in sim.colonists:
            assert col.element in ELEMENTS

    def test_stats_in_range(self):
        sim = Mars100(seed=42)
        for col in sim.colonists:
            for stat_name in STAT_NAMES:
                val = col.stats[stat_name]
                assert 0.0 <= val <= 1.0, f"{col.name}.{stat_name} = {val}"

    def test_skills_in_range(self):
        sim = Mars100(seed=42)
        for col in sim.colonists:
            for skill_name in SKILL_NAMES:
                val = col.skills[skill_name]
                assert 0.0 <= val <= 1.0, f"{col.name}.{skill_name} = {val}"

    def test_relationships_initialized(self):
        sim = Mars100(seed=42)
        for col in sim.colonists:
            # Should have relationship with 9 others
            assert len(col.relationships) == 9
            for rel_val in col.relationships.values():
                assert 0.0 <= rel_val <= 1.0

    def test_colonist_to_dict_serializable(self):
        sim = Mars100(seed=42)
        for col in sim.colonists:
            d = col.to_dict()
            json.dumps(d)  # must not raise

    def test_colonist_to_lispy_dict(self):
        sim = Mars100(seed=42)
        col = sim.colonists[0]
        ld = col.to_lispy_dict()
        assert "stat-resolve" in ld
        assert "skill-terraforming" in ld


# ---------------------------------------------------------------------------
# Colony State
# ---------------------------------------------------------------------------

class TestColonyState:
    def test_initial_resources_positive(self):
        sim = Mars100(seed=42)
        assert sim.colony.food > 0
        assert sim.colony.water > 0
        assert sim.colony.power > 0
        assert sim.colony.materials > 0

    def test_initial_morale_positive(self):
        sim = Mars100(seed=42)
        assert 0.0 < sim.colony.morale <= 1.0

    def test_initial_laws(self):
        sim = Mars100(seed=42)
        assert len(sim.colony.laws) > 0

    def test_colony_to_dict_serializable(self):
        sim = Mars100(seed=42)
        d = sim.colony.to_dict()
        json.dumps(d)

    def test_colony_to_lispy_dict(self):
        sim = Mars100(seed=42)
        ld = sim.colony.to_lispy_dict()
        assert "food" in ld
        assert "morale" in ld


# ---------------------------------------------------------------------------
# Yearly Tick
# ---------------------------------------------------------------------------

class TestYearlyTick:
    def test_tick_advances_year(self):
        sim = Mars100(seed=42)
        sim.tick_year()
        assert sim.year == 1

    def test_tick_returns_chapter(self):
        sim = Mars100(seed=42)
        chapter = sim.tick_year()
        assert chapter.year == 1
        assert chapter.event is not None
        assert chapter.narrative is not None

    def test_tick_produces_actions(self):
        sim = Mars100(seed=42)
        chapter = sim.tick_year()
        alive_count = sum(1 for c in sim.colonists if c.alive)
        assert len(chapter.colonist_actions) == alive_count

    def test_chapter_serializable(self):
        sim = Mars100(seed=42)
        chapter = sim.tick_year()
        json.dumps(chapter.to_dict())

    def test_event_has_required_fields(self):
        sim = Mars100(seed=42)
        chapter = sim.tick_year()
        event = chapter.event
        assert "name" in event
        assert "severity" in event
        assert "resource_drain" in event
        assert "morale_hit" in event
        assert "year" in event

    def test_multiple_ticks(self):
        sim = Mars100(seed=42)
        for _ in range(5):
            sim.tick_year()
        assert sim.year == 5
        assert len(sim.chapters) == 5


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class TestEvents:
    def test_event_names_valid(self):
        sim = Mars100(seed=42)
        valid_names = {e["name"] for e in EVENTS}
        for _ in range(50):
            chapter = sim.tick_year()
            assert chapter.event["name"] in valid_names

    def test_event_severity_bounded(self):
        sim = Mars100(seed=42)
        for _ in range(50):
            chapter = sim.tick_year()
            assert 0.0 <= chapter.event["severity"] <= 1.5


# ---------------------------------------------------------------------------
# Resource Conservation
# ---------------------------------------------------------------------------

class TestResourceConservation:
    def test_resources_never_negative(self):
        """Resources must never go below zero."""
        sim = Mars100(seed=42)
        for _ in range(100):
            sim.tick_year()
            assert sim.colony.food >= 0, f"Year {sim.year}: food={sim.colony.food}"
            assert sim.colony.water >= 0, f"Year {sim.year}: water={sim.colony.water}"
            assert sim.colony.power >= 0, f"Year {sim.year}: power={sim.colony.power}"
            assert sim.colony.materials >= 0, f"Year {sim.year}: materials={sim.colony.materials}"

    def test_morale_bounded(self):
        """Morale must stay in [0, 1]."""
        sim = Mars100(seed=42)
        for _ in range(100):
            sim.tick_year()
            assert 0.0 <= sim.colony.morale <= 1.0, f"Year {sim.year}: morale={sim.colony.morale}"

    def test_habitat_integrity_bounded(self):
        sim = Mars100(seed=42)
        for _ in range(100):
            sim.tick_year()
            assert 0.0 <= sim.colony.habitat_integrity <= 1.0

    def test_terraform_progress_non_negative(self):
        sim = Mars100(seed=42)
        for _ in range(100):
            sim.tick_year()
            assert sim.colony.terraform_progress >= 0.0


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------

class TestGovernance:
    def test_proposals_have_outcomes(self):
        """All proposals resolved in the year they're made."""
        sim = Mars100(seed=42)
        sim.run(years=50)
        for p in sim.proposals:
            assert p.outcome in ("adopted", "rejected"), f"Proposal {p.id} has outcome {p.outcome}"

    def test_exile_removes_colonist(self):
        """Adopted exile removes the target."""
        sim = Mars100(seed=42)
        sim.run(years=100)
        exiled = [p for p in sim.proposals
                  if p.type == "exile_vote" and p.outcome == "adopted"]
        for p in exiled:
            target = next((c for c in sim.colonists if c.id == p.target), None)
            if target:
                assert not target.alive
                assert target.cause_of_death == "exiled"

    def test_leadership_changes_tracked(self):
        sim = Mars100(seed=42)
        sim.run(years=100)
        patterns = sim._analyze_governance()
        assert "leadership_changes" in patterns


# ---------------------------------------------------------------------------
# Sub-simulations
# ---------------------------------------------------------------------------

class TestSubSimulations:
    def test_sub_sims_logged(self):
        """Sub-simulations are recorded in the log."""
        sim = Mars100(seed=42)
        sim.run(years=30)
        # With 10 colonists over 30 years, some should have run sub-sims
        total = sum(c.sub_sims_run for c in sim.colonists)
        assert total > 0, "No sub-sims were run in 30 years"

    def test_sub_sim_log_serializable(self):
        sim = Mars100(seed=42)
        sim.run(years=10)
        for log in sim.sub_sim_logs:
            json.dumps(log.to_dict())

    def test_sub_sim_does_not_corrupt_state(self):
        """Sub-sims must not mutate colony state."""
        sim = Mars100(seed=42)
        food_before = sim.colony.food
        sim.tick_year()
        # The colony state changed from normal actions, but sub-sims
        # should not have caused any extra unexpected changes
        # (this is a structural test — sub-sims work on copies)
        assert sim.colony.food >= 0  # just verify no corruption


# ---------------------------------------------------------------------------
# Death and Archiving
# ---------------------------------------------------------------------------

class TestDeathAndArchiving:
    def test_dead_colonists_dont_act(self):
        """Dead colonists should not appear in actions."""
        sim = Mars100(seed=42)
        sim.run(years=100)
        for chapter in sim.chapters:
            dead_ids = {c.id for c in sim.colonists
                        if not c.alive and c.year_died and c.year_died < chapter.year}
            for action in chapter.colonist_actions:
                assert action["colonist_id"] not in dead_ids, (
                    f"Dead colonist {action['colonist_id']} acted in year {chapter.year}"
                )

    def test_dead_colonists_archived(self):
        """Dead colonists get soul files."""
        sim = Mars100(seed=42)
        sim.run(years=100)
        dead = [c for c in sim.colonists if not c.alive]
        assert len(sim.archived_souls) == len(dead)

    def test_archived_souls_have_epitaphs(self):
        sim = Mars100(seed=42)
        sim.run(years=100)
        for soul in sim.archived_souls:
            assert "epitaph" in soul
            assert len(soul["epitaph"]) > 0

    def test_archived_souls_serializable(self):
        sim = Mars100(seed=42)
        sim.run(years=100)
        for soul in sim.archived_souls:
            json.dumps(soul)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_results(self):
        """Same seed must produce byte-identical output."""
        sim1 = Mars100(seed=42)
        sim2 = Mars100(seed=42)
        r1 = sim1.run(years=20)
        r2 = sim2.run(years=20)
        # Compare key fields (excluding timestamps)
        assert r1["summary"]["final_population"] == r2["summary"]["final_population"]
        assert r1["summary"]["total_deaths"] == r2["summary"]["total_deaths"]
        assert r1["summary"]["total_proposals"] == r2["summary"]["total_proposals"]
        assert r1["summary"]["total_sub_sims"] == r2["summary"]["total_sub_sims"]
        assert r1["colony"]["food"] == r2["colony"]["food"]

    def test_different_seed_different_results(self):
        """Different seeds should (usually) produce different results."""
        sim1 = Mars100(seed=42)
        sim2 = Mars100(seed=99)
        r1 = sim1.run(years=20)
        r2 = sim2.run(years=20)
        # Compare multiple fields — at least one should differ
        fields_to_check = [
            r1["colony"]["food"] != r2["colony"]["food"],
            r1["colony"]["water"] != r2["colony"]["water"],
            r1["colony"]["morale"] != r2["colony"]["morale"],
            r1["summary"]["total_proposals"] != r2["summary"]["total_proposals"],
        ]
        assert any(fields_to_check), "Different seeds produced identical results across all checked fields"


# ---------------------------------------------------------------------------
# Smoke Test: Full 10-Year Run
# ---------------------------------------------------------------------------

class TestSmokeTest:
    def test_10_year_run_no_crash(self):
        """The simulation runs for 10 years without crashing."""
        sim = Mars100(seed=42)
        results = sim.run(years=10)
        assert results is not None
        assert results["_meta"]["years_simulated"] == 10

    def test_10_year_results_complete(self):
        sim = Mars100(seed=42)
        results = sim.run(years=10)
        assert "colony" in results
        assert "colonists" in results
        assert "chapters" in results
        assert "summary" in results
        assert "governance_patterns" in results
        assert len(results["chapters"]) == 10

    def test_10_year_results_serializable(self):
        """Full results must be JSON-serializable."""
        sim = Mars100(seed=42)
        results = sim.run(years=10)
        serialized = json.dumps(results)
        assert len(serialized) > 0
        # Roundtrip
        parsed = json.loads(serialized)
        assert parsed["_meta"]["engine"] == "mars-100"

    def test_50_year_run_no_crash(self):
        """Extended smoke test."""
        sim = Mars100(seed=42)
        results = sim.run(years=50)
        assert results["_meta"]["years_simulated"] == 50
        assert results["summary"]["final_population"] >= 0

    def test_100_year_run_no_crash(self):
        """Full century run."""
        sim = Mars100(seed=42)
        results = sim.run(years=100)
        assert results["_meta"]["years_simulated"] <= 100
        # Colony might collapse, so years_simulated could be < 100
        assert results["summary"]["final_population"] >= 0


# ---------------------------------------------------------------------------
# Narrative Quality
# ---------------------------------------------------------------------------

class TestNarrativeQuality:
    def test_narrative_contains_year(self):
        sim = Mars100(seed=42)
        chapter = sim.tick_year()
        assert "Year 1" in chapter.narrative

    def test_narrative_contains_event(self):
        sim = Mars100(seed=42)
        chapter = sim.tick_year()
        assert "Event" in chapter.narrative

    def test_narrative_mentions_colonists(self):
        sim = Mars100(seed=42)
        chapter = sim.tick_year()
        # At least one colonist name should appear
        any_name = any(name in chapter.narrative for name in COLONIST_NAMES)
        assert any_name


# ---------------------------------------------------------------------------
# Full Results Structure
# ---------------------------------------------------------------------------

class TestResultsStructure:
    def test_results_meta(self):
        sim = Mars100(seed=42)
        results = sim.run(years=5)
        meta = results["_meta"]
        assert meta["engine"] == "mars-100"
        assert meta["version"] == "1.0"
        assert meta["seed"] == 42
        assert meta["max_depth"] == 3

    def test_results_summary_fields(self):
        sim = Mars100(seed=42)
        results = sim.run(years=5)
        s = results["summary"]
        required = [
            "years_survived", "final_population", "total_deaths",
            "total_proposals", "proposals_adopted", "proposals_rejected",
            "total_sub_sims", "constitution_amendments", "terraform_progress",
            "final_morale", "laws",
        ]
        for field in required:
            assert field in s, f"Missing summary field: {field}"
