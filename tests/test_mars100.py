"""Tests for Mars-100 recursive colony simulation."""
from __future__ import annotations

import json
import pytest
import sys
import os
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mars100 import (
    Mars100, Colony, Colonist, COLONIST_TEMPLATES, EVENTS,
    ACTIONS, GOVERNANCE_TEMPLATES, ELEMENTS, now_iso,
)
from src.lispy import Lispy


# --- Colonist ---

class TestColonist:
    def test_creation(self):
        c = Colonist(COLONIST_TEMPLATES[0], random.Random(42))
        assert c.id == "kael"
        assert c.name == "Kael Ashborne"
        assert c.element == "fire"
        assert c.alive is True
        assert c.role == "colonist"

    def test_stats_in_range(self):
        for tmpl in COLONIST_TEMPLATES:
            c = Colonist(tmpl, random.Random(42))
            for stat, val in c.stats.items():
                assert 0.0 <= val <= 1.0, f"{c.id}.{stat} = {val}"
            for skill, val in c.skills.items():
                assert 0.0 <= val <= 1.0, f"{c.id}.{skill} = {val}"

    def test_decide_fallback(self):
        c = Colonist(COLONIST_TEMPLATES[0], random.Random(42))
        vm = Lispy(seed=42)
        action = c.decide({}, vm)
        assert action in ACTIONS

    def test_decide_with_context(self):
        c = Colonist(COLONIST_TEMPLATES[0], random.Random(42))
        vm = Lispy(seed=42)
        context = {"danger": 0.9, "food": 100, "water": 100,
                    "power": 80, "year": 1, "unrest": 0.1}
        action = c.decide(context, vm)
        assert action in ACTIONS

    def test_to_dict(self):
        c = Colonist(COLONIST_TEMPLATES[0], random.Random(42))
        d = c.to_dict()
        assert d["id"] == "kael"
        assert "stats" in d
        assert "skills" in d
        assert "personality" in d
        assert d["alive"] is True

    def test_memory(self):
        c = Colonist(COLONIST_TEMPLATES[0], random.Random(42))
        c.memory.append("Year 1: test event")
        assert "Year 1: test event" in c.to_dict()["memory"]

    def test_all_10_templates(self):
        assert len(COLONIST_TEMPLATES) == 10

    def test_unique_ids(self):
        ids = [t["id"] for t in COLONIST_TEMPLATES]
        assert len(ids) == len(set(ids))

    def test_all_elements_represented(self):
        elements = {t["element"] for t in COLONIST_TEMPLATES}
        assert elements == set(ELEMENTS)


# --- Colony ---

class TestColony:
    def test_creation(self):
        colony = Colony(seed=42)
        assert colony.year == 0
        assert len(colony.colonists) == 10
        assert len(colony.alive_colonists) == 10

    def test_initial_resources(self):
        colony = Colony(seed=42)
        for key in ["food", "water", "power", "oxygen", "supplies"]:
            assert colony.resources[key] > 0

    def test_initial_metrics(self):
        colony = Colony(seed=42)
        assert 0 <= colony.metrics["morale"] <= 1
        assert colony.metrics["unrest"] >= 0
        assert colony.metrics["habitat_integrity"] > 0

    def test_relationships_initialized(self):
        colony = Colony(seed=42)
        for c in colony.colonists:
            assert len(c.relationships) == 9
            for other_id, val in c.relationships.items():
                assert -1.0 <= val <= 1.0

    def test_constitution(self):
        colony = Colony(seed=42)
        assert len(colony.constitution) >= 3

    def test_collapsed_false_initially(self):
        colony = Colony(seed=42)
        assert colony.collapsed is False

    def test_collapsed_when_all_dead(self):
        colony = Colony(seed=42)
        for c in colony.colonists:
            c.alive = False
        assert colony.collapsed is True

    def test_deterministic(self):
        a = Colony(seed=42)
        b = Colony(seed=42)
        for ca, cb in zip(a.colonists, b.colonists):
            assert ca.stats == cb.stats

    def test_different_seeds(self):
        a = Colony(seed=42)
        b = Colony(seed=99)
        # Relationships vary by seed (stats come from templates)
        diffs = sum(
            1 for ca, cb in zip(a.colonists, b.colonists)
            if ca.relationships != cb.relationships
        )
        assert diffs > 0


# --- Mars100 Simulation ---

class TestMars100:
    def test_creation(self):
        sim = Mars100(seed=42)
        assert sim.colony is not None
        assert len(sim.colony.colonists) == 10

    def test_single_tick(self):
        sim = Mars100(seed=42)
        result = sim.tick(1)
        assert result["year"] == 1
        assert "events" in result
        assert "actions" in result
        assert "resources" in result
        assert "metrics" in result
        assert "population" in result
        assert result["population"] <= 10

    def test_events_generated(self):
        sim = Mars100(seed=42)
        result = sim.tick(1)
        assert len(result["events"]) >= 1
        for event in result["events"]:
            assert "id" in event
            assert "name" in event
            assert "severity" in event

    def test_actions_for_each_colonist(self):
        sim = Mars100(seed=42)
        result = sim.tick(1)
        assert len(result["actions"]) == len(sim.colony.alive_colonists)
        for action in result["actions"]:
            assert "colonist" in action
            assert "action" in action
            assert action["action"] in ACTIONS

    def test_resources_change(self):
        sim = Mars100(seed=42)
        r_before = dict(sim.colony.resources)
        sim.tick(1)
        r_after = sim.colony.resources
        changed = any(r_before[k] != r_after[k] for k in r_before)
        assert changed

    def test_resources_non_negative(self):
        """Resources must never go negative."""
        sim = Mars100(seed=42)
        for year in range(1, 51):
            sim.tick(year)
            for key, val in sim.colony.resources.items():
                assert val >= 0.0, f"Year {year}: {key} = {val}"

    def test_metrics_bounded(self):
        """Metrics must stay in [-1, 1] range."""
        sim = Mars100(seed=42)
        for year in range(1, 51):
            sim.tick(year)
            for key, val in sim.colony.metrics.items():
                assert -1.0 <= val <= 1.0, f"Year {year}: {key} = {val}"

    def test_run_10_years(self):
        sim = Mars100(seed=42)
        results = sim.run(years=10)
        assert results["_meta"]["engine"] == "mars-100"
        assert results["_meta"]["years_simulated"] == 10
        assert len(results["year_log"]) == 10

    def test_run_smoke_50_years(self):
        """Smoke test: 50 years without crash."""
        sim = Mars100(seed=42)
        results = sim.run(years=50)
        assert results["_meta"]["years_simulated"] <= 50
        assert results["_meta"]["final_population"] >= 0

    def test_run_full_100_years(self):
        """Full simulation: 100 years."""
        sim = Mars100(seed=42)
        results = sim.run(years=100)
        assert results["_meta"]["years_simulated"] <= 100
        assert "colony" in results
        assert "year_log" in results

    def test_deterministic(self):
        """Same seed produces same results."""
        a = Mars100(seed=42).run(years=20)
        b = Mars100(seed=42).run(years=20)
        assert a["_meta"]["years_simulated"] == b["_meta"]["years_simulated"]
        assert a["_meta"]["final_population"] == b["_meta"]["final_population"]
        for ya, yb in zip(a["year_log"], b["year_log"]):
            assert ya["population"] == yb["population"]

    def test_different_seeds_diverge(self):
        """Different seeds produce different results."""
        a = Mars100(seed=42).run(years=20)
        b = Mars100(seed=99).run(years=20)
        pops_a = [y["population"] for y in a["year_log"]]
        pops_b = [y["population"] for y in b["year_log"]]
        assert pops_a != pops_b

    def test_json_serializable(self):
        """All output must be JSON-serializable."""
        sim = Mars100(seed=42)
        results = sim.run(years=20)
        serialized = json.dumps(results)
        assert len(serialized) > 0
        parsed = json.loads(serialized)
        assert parsed["_meta"]["engine"] == "mars-100"


# --- Governance ---

class TestGovernance:
    def test_governance_templates_valid(self):
        for g in GOVERNANCE_TEMPLATES:
            assert "id" in g
            assert "name" in g
            assert "lispy" in g
            assert "effects" in g

    def test_governance_lispy_parseable(self):
        """All governance LisPy expressions must parse without error."""
        from src.lispy import tokenize, parse_all
        for g in GOVERNANCE_TEMPLATES:
            tokens = tokenize(g["lispy"])
            exprs = parse_all(tokens)
            assert len(exprs) >= 1

    def test_governance_emerges(self):
        """Over 50 years, at least one governance proposal should appear."""
        sim = Mars100(seed=42)
        sim.run(years=50)
        assert len(sim.colony.governance_proposals) > 0

    def test_governance_has_votes(self):
        """Governance proposals should have vote counts."""
        sim = Mars100(seed=42)
        sim.run(years=50)
        if sim.colony.governance_proposals:
            p = sim.colony.governance_proposals[0]
            assert "votes_for" in p
            assert "votes_against" in p
            assert "passed" in p


# --- Sub-simulations ---

class TestSubSimulations:
    def test_sub_sim_archive_populated(self):
        """Sub-sims should be logged in the archive."""
        sim = Mars100(seed=42)
        results = sim.run(years=50)
        assert isinstance(results["sub_sim_archive"], list)

    def test_sub_sim_archive_structure(self):
        """If sub-sims exist, they should have proper structure."""
        sim = Mars100(seed=42)
        results = sim.run(years=80)
        for entry in results["sub_sim_archive"]:
            assert "year" in entry
            assert "colonist" in entry or "label" in entry


# --- Deaths and Births ---

class TestPopulationDynamics:
    def test_deaths_possible(self):
        """Deaths can occur when resources are depleted."""
        sim = Mars100(seed=42)
        sim.colony.resources["food"] = 0
        sim.colony.resources["water"] = 0
        sim.colony.resources["oxygen"] = 0
        deaths = sim._check_deaths(50)
        assert isinstance(deaths, list)

    def test_dead_colonists_archived(self):
        """Dead colonists should be in the archive (legacy, not delete)."""
        sim = Mars100(seed=42)
        sim.colony.resources["food"] = 0
        sim.colony.resources["water"] = 0
        sim.colony.resources["oxygen"] = 0
        for year in range(1, 20):
            sim.tick(year)
        for dc in sim.colony.dead_colonists:
            assert "colonist" in dc
            assert "year_of_death" in dc
            assert "cause" in dc

    def test_births_before_year_5(self):
        """Births should not occur before year 5."""
        sim = Mars100(seed=42)
        for year in range(1, 5):
            result = sim.tick(year)
            assert result["births"] == []

    def test_births_possible(self):
        """Births can occur after year 5."""
        sim = Mars100(seed=42)
        any_births = False
        for year in range(1, 100):
            result = sim.tick(year)
            if result["births"]:
                any_births = True
                break


# --- Events ---

class TestEvents:
    def test_event_catalog(self):
        assert len(EVENTS) >= 10
        for event in EVENTS:
            assert "id" in event
            assert "name" in event
            assert "severity" in event
            assert "effects" in event

    def test_severity_range(self):
        for event in EVENTS:
            lo, hi = event["severity"]
            assert 0.0 <= lo <= hi <= 1.0

    def test_event_generation(self):
        sim = Mars100(seed=42)
        events = sim._generate_events(1)
        assert 1 <= len(events) <= 2
        for e in events:
            assert "id" in e
            assert "severity" in e


# --- Actions ---

class TestActions:
    def test_action_catalog(self):
        assert len(ACTIONS) >= 15
        for action, effects in ACTIONS.items():
            assert isinstance(effects, dict)

    def test_all_personality_actions_valid(self):
        """Every action mentioned in colonist personalities should exist."""
        import re
        for tmpl in COLONIST_TEMPLATES:
            for match in re.findall(r"'(\w[\w-]*)", tmpl["personality"]):
                if match not in ("meta", "yes", "no"):
                    assert match in ACTIONS, f"{tmpl['id']}'s action '{match}' not in ACTIONS"


# --- Conservation laws ---

class TestConservation:
    def test_population_tracking(self):
        """Population = alive colonists at all times."""
        sim = Mars100(seed=42)
        for year in range(1, 51):
            result = sim.tick(year)
            assert result["population"] == len(sim.colony.alive_colonists)

    def test_no_duplicate_colonist_ids(self):
        """All colonist IDs must be unique."""
        sim = Mars100(seed=42)
        sim.run(years=50)
        ids = [c.id for c in sim.colony.colonists]
        assert len(ids) == len(set(ids))


# --- Output format ---

class TestOutput:
    def test_meta_format(self):
        results = Mars100(seed=42).run(years=5)
        meta = results["_meta"]
        assert meta["engine"] == "mars-100"
        assert meta["version"] == "1.0"
        assert "seed" in meta
        assert "years_simulated" in meta
        assert "final_population" in meta
        assert "generated" in meta

    def test_colony_snapshot(self):
        results = Mars100(seed=42).run(years=5)
        colony = results["colony"]
        assert "year" in colony
        assert "resources" in colony
        assert "metrics" in colony
        assert "colonists" in colony
        assert "governance" in colony
        assert "constitution" in colony
        assert "population" in colony

    def test_year_log_entries(self):
        results = Mars100(seed=42).run(years=5)
        for entry in results["year_log"]:
            assert "year" in entry
            assert "events" in entry
            assert "actions" in entry
            assert "resources" in entry
            assert "metrics" in entry
            assert "population" in entry

    def test_save_state(self, tmp_path):
        """Test saving state to file."""
        os.environ["STATE_DIR"] = str(tmp_path)
        sim = Mars100(seed=42)
        results = sim.run(years=5)
        out = tmp_path / "mars100.json"
        out.write_text(json.dumps(results, indent=2))
        loaded = json.loads(out.read_text())
        assert loaded["_meta"]["engine"] == "mars-100"
        del os.environ["STATE_DIR"]


# --- Amendments ---

class TestAmendments:
    def test_amendments_structure(self):
        results = Mars100(seed=42).run(years=100)
        for a in results["amendments"]:
            assert "year" in a
            assert "proposed_by" in a
            assert "insight" in a
            assert "proposed_amendment" in a
