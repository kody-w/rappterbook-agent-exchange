"""Tests for the Mars-100 recursive colony simulation."""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars100 import (
    COLONIST_NAMES,
    ELEMENTS,
    EVENTS,
    GOVERNANCE_TYPES,
    INITIAL_RESOURCES,
    SKILL_NAMES,
    STAT_NAMES,
    apply_event_effects,
    check_births,
    check_meta_awareness,
    colonist_to_env,
    consume_resources,
    create_colony,
    create_colonist,
    evolve_relationships,
    generate_action_expr,
    generate_subsim_expr,
    init_relationships,
    pick_event,
    process_action,
    resolve_proposals,
    run_simulation,
    tick_year,
)
from src.lispy import Budget, DepthLimitExceeded, LispError, make_env, run as lispy_run, run_in_env


# ---------------------------------------------------------------------------
# Colony creation
# ---------------------------------------------------------------------------

class TestColonyCreation:
    def test_creates_colony(self):
        colony = create_colony(seed=42)
        assert colony["year"] == 0
        assert len(colony["colonists"]) == 10
        assert colony["resources"]["food"] > 0

    def test_colonists_have_required_fields(self):
        colony = create_colony()
        for c in colony["colonists"]:
            assert "id" in c
            assert "name" in c
            assert c["element"] in ELEMENTS
            assert all(s in c["stats"] for s in STAT_NAMES)
            assert all(s in c["skills"] for s in SKILL_NAMES)
            assert c["alive"] is True
            assert isinstance(c["relationships"], dict)

    def test_relationships_initialized(self):
        colony = create_colony()
        for c in colony["colonists"]:
            assert len(c["relationships"]) == 9

    def test_deterministic_creation(self):
        a = create_colony(seed=42)
        b = create_colony(seed=42)
        assert a["colonists"][0]["name"] == b["colonists"][0]["name"]
        assert a["colonists"][0]["stats"] == b["colonists"][0]["stats"]

    def test_different_seeds_differ(self):
        a = create_colony(seed=42)
        b = create_colony(seed=99)
        assert a["colonists"][0]["stats"] != b["colonists"][0]["stats"]

    def test_element_affinity_boosts(self):
        rng = random.Random(100)
        for _ in range(50):
            c = create_colonist(0, rng)
            if c["element"] == "fire":
                assert c["stats"]["resolve"] >= 35
            elif c["element"] == "water":
                assert c["stats"]["empathy"] >= 35

    def test_custom_colony_size(self):
        colony = create_colony(n_colonists=5)
        assert len(colony["colonists"]) == 5

    def test_meta_fields(self):
        colony = create_colony()
        assert colony["_meta"]["engine"] == "mars-100"
        assert colony["governance"]["system"] == "direct_democracy"


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class TestEvents:
    def test_pick_event_valid(self):
        rng = random.Random(42)
        for year in range(1, 50):
            event = pick_event(year, rng)
            assert "id" in event
            assert "severity" in event

    def test_supply_ship_only_every_5_years(self):
        rng = random.Random(42)
        for year in range(1, 100):
            event = pick_event(year, rng)
            if event["id"] == "supply_ship":
                assert year % 5 == 0

    def test_event_effects_bounded(self):
        colony = create_colony()
        rng = random.Random(42)
        for event in EVENTS:
            test_colony = copy.deepcopy(colony)
            apply_event_effects(test_colony, event, rng)
            for key, val in test_colony["resources"].items():
                if isinstance(val, (int, float)):
                    assert val >= 0, f"{key} negative after {event['id']}"


# ---------------------------------------------------------------------------
# Colonist actions via LisPy
# ---------------------------------------------------------------------------

class TestColonistActions:
    def test_action_expr_parses(self):
        colony = create_colony()
        for c in colony["colonists"]:
            for event in EVENTS[:3]:
                expr = generate_action_expr(c, event, 10)
                env, ctx = make_env()
                colonist_to_env(c, colony, event, env)
                result = run_in_env(expr, env, ctx)
                assert result is not None

    def test_action_processing(self):
        colony = create_colony()
        rng = random.Random(42)
        c = colony["colonists"][0]
        narrative = process_action(c, {"type": "work", "skill": "hydroponics"}, colony, rng)
        assert c["name"] in narrative
        narrative = process_action(c, None, colony, rng)
        assert "idle" in narrative.lower()

    def test_propose_creates_proposal(self):
        colony = create_colony()
        c = colony["colonists"][0]
        rng = random.Random(42)
        process_action(c, {"type": "propose", "governance_type": "leadership_election", "detail": "I should lead"}, colony, rng)
        assert len(colony["proposals_pending"]) == 1

    def test_mediate_improves_relationships(self):
        colony = create_colony()
        rng = random.Random(42)
        c = colony["colonists"][0]
        id_a = colony["colonists"][1]["id"]
        id_b = colony["colonists"][2]["id"]
        rel_before = colony["colonists"][1]["relationships"].get(str(id_b), 0)
        process_action(c, {"type": "mediate", "between": [id_a, id_b]}, colony, rng)
        rel_after = colony["colonists"][1]["relationships"].get(str(id_b), 0)
        assert rel_after >= rel_before


# ---------------------------------------------------------------------------
# Sub-simulation
# ---------------------------------------------------------------------------

class TestSubSim:
    def test_subsim_expr_valid(self):
        colony = create_colony()
        c = colony["colonists"][0]
        event = EVENTS[0]
        for gtype in GOVERNANCE_TYPES[:4]:
            proposal = {"governance_type": gtype}
            expr = generate_subsim_expr(c, proposal)
            env, ctx = make_env()
            colonist_to_env(c, colony, event, env)
            result = run_in_env(expr, env, ctx)
            assert result is not None
            assert isinstance(result, list)

    def test_subsim_isolation(self):
        colony = create_colony()
        original_food = colony["resources"]["food"]
        c = colony["colonists"][0]
        event = EVENTS[0]
        expr = '(sub-sim (begin (set! colony-food 999999) colony-food))'
        env, ctx = make_env()
        colonist_to_env(c, colony, event, env)
        run_in_env(expr, env, ctx)
        assert colony["resources"]["food"] == original_food


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------

class TestGovernance:
    def test_proposal_resolution(self):
        colony = create_colony()
        rng = random.Random(42)
        colony["proposals_pending"].append({
            "id": 0, "year": 1, "proposer": 0,
            "governance_type": "research_directive", "detail": "fund coding",
            "votes_for": [0], "votes_against": [], "resolved": False,
        })
        effects = resolve_proposals(colony, rng)
        assert len(effects) > 0

    def test_dead_colonists_dont_vote(self):
        colony = create_colony()
        rng = random.Random(42)
        for c in colony["colonists"][:5]:
            c["alive"] = False
        colony["proposals_pending"].append({
            "id": 0, "year": 1, "proposer": 5,
            "governance_type": "leadership_election", "detail": "",
            "votes_for": [5], "votes_against": [], "resolved": False,
        })
        effects = resolve_proposals(colony, rng)
        assert len(effects) > 0


# ---------------------------------------------------------------------------
# Resources and survival
# ---------------------------------------------------------------------------

class TestResources:
    def test_consumption_changes_resources(self):
        colony = create_colony()
        food_before = colony["resources"]["food"]
        consume_resources(colony)
        assert colony["resources"]["food"] != food_before

    def test_resources_never_negative(self):
        colony = create_colony()
        colony["resources"]["food"] = 0
        colony["resources"]["oxygen"] = 0
        consume_resources(colony)
        assert colony["resources"]["food"] >= 0
        assert colony["resources"]["oxygen"] >= 0

    def test_starvation_death(self):
        colony = create_colony()
        colony["resources"]["food"] = 0
        colony["resources"]["oxygen"] = 0
        effects = consume_resources(colony)
        deaths = [e for e in effects if "dies" in e.lower()]
        assert len(deaths) > 0

    def test_death_creates_soul(self):
        colony = create_colony()
        colony["resources"]["food"] = 0
        consume_resources(colony)
        if colony["dead_souls"]:
            soul = colony["dead_souls"][0]
            assert soul["alive"] is False
            assert soul["cause_of_death"] is not None

    def test_extinction_handled(self):
        colony = create_colony()
        for c in colony["colonists"]:
            c["alive"] = False
        effects = consume_resources(colony)
        assert "extinct" in effects[0].lower()


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

class TestPopulation:
    def test_births_require_minimum_pop(self):
        colony = create_colony()
        for c in colony["colonists"][1:]:
            c["alive"] = False
        rng = random.Random(42)
        for _ in range(100):
            check_births(colony, rng)
        assert len(colony["colonists"]) == 10  # no new

    def test_births_add_relationships(self):
        colony = create_colony()
        colony["resources"]["morale"] = 100
        rng = random.Random(42)
        for _ in range(200):
            check_births(colony, rng)
        newborns = [c for c in colony["colonists"] if c["year_born"] > 0]
        if newborns:
            assert len(newborns[0]["relationships"]) > 0


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

class TestRelationships:
    def test_relationships_change(self):
        colony = create_colony()
        rng = random.Random(42)
        before = copy.deepcopy(colony["colonists"][0]["relationships"])
        evolve_relationships(colony, rng)
        assert colony["colonists"][0]["relationships"] != before

    def test_relationships_bounded(self):
        colony = create_colony()
        rng = random.Random(42)
        for _ in range(100):
            evolve_relationships(colony, rng)
        for c in colony["colonists"]:
            if c["alive"]:
                for val in c["relationships"].values():
                    assert -100 <= val <= 100


# ---------------------------------------------------------------------------
# Meta-awareness
# ---------------------------------------------------------------------------

class TestMetaAwareness:
    def test_no_awareness_early(self):
        colony = create_colony()
        assert check_meta_awareness(colony, 5) is None

    def test_awareness_possible_late(self):
        colony = create_colony()
        c = colony["colonists"][0]
        c["stats"]["improvisation"] = 90
        c["stats"]["faith"] = 90
        c["sub_sims_run"] = 10
        result = check_meta_awareness(colony, 80)
        assert result is not None


# ---------------------------------------------------------------------------
# Full simulation smoke tests
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_smoke_10_years(self):
        result = run_simulation(years=10, seed=42)
        assert result["summary"]["years_survived"] == 10
        assert result["summary"]["final_population"] > 0

    def test_smoke_50_years(self):
        result = run_simulation(years=50, seed=42)
        assert result["summary"]["years_survived"] > 0

    def test_deterministic_replay(self):
        a = run_simulation(years=20, seed=42)
        b = run_simulation(years=20, seed=42)
        assert a["summary"]["years_survived"] == b["summary"]["years_survived"]
        assert a["summary"]["final_population"] == b["summary"]["final_population"]
        assert a["summary"]["population_curve"] == b["summary"]["population_curve"]

    def test_different_seeds_diverge(self):
        a = run_simulation(years=30, seed=42)
        b = run_simulation(years=30, seed=99)
        assert a["summary"]["population_curve"] != b["summary"]["population_curve"]

    def test_sub_sims_generated(self):
        result = run_simulation(years=30, seed=42)
        assert result["summary"]["total_sub_simulations"] > 0

    def test_diary_entries_generated(self):
        result = run_simulation(years=10, seed=42)
        for delta in result["deltas"]:
            if delta["population"] > 0:
                assert len(delta["diary_entries"]) > 0

    def test_population_curve_valid(self):
        result = run_simulation(years=30, seed=42)
        for pop in result["summary"]["population_curve"]:
            assert isinstance(pop, int)
            assert pop >= 0

    def test_morale_bounded(self):
        result = run_simulation(years=50, seed=42)
        for m in result["summary"]["morale_curve"]:
            assert 0 <= m <= 100

    def test_governance_evolves(self):
        result = run_simulation(years=50, seed=42)
        gov_events = sum(len(d["governance_results"]) for d in result["deltas"])
        assert gov_events > 0

    def test_resources_bounded(self):
        result = run_simulation(years=50, seed=42)
        for delta in result["deltas"]:
            for key, val in delta["resources_snapshot"].items():
                if isinstance(val, (int, float)):
                    assert val >= 0, f"Resource {key} negative in year {delta['year']}"

    def test_history_length_matches_years(self):
        result = run_simulation(years=20, seed=42)
        assert len(result["deltas"]) == result["summary"]["years_survived"]
