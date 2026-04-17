"""Tests for the Mars-100 recursive colony simulation."""
from __future__ import annotations
import copy, random, sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars100 import (
    COLONIST_NAMES, ELEMENTS, EVENTS, GOVERNANCE_TYPES,
    INITIAL_RESOURCES, SKILL_NAMES, STAT_NAMES,
    apply_event_effects, check_births, check_meta_awareness,
    colonist_to_env, consume_resources, create_colony,
    create_colonist, evolve_relationships, generate_action_expr,
    generate_subsim_expr, init_relationships, pick_event,
    process_action, resolve_proposals, run_simulation, tick_year,
)
from src.lispy import Budget, DepthLimitExceeded, LispError, make_env, run as lispy_run, run_in_env

class TestColonyCreation:
    def test_creates_colony(self):
        colony = create_colony(seed=42)
        assert colony["year"] == 0
        assert len(colony["colonists"]) == 10
        assert colony["resources"]["food"] > 0
    def test_colonists_have_required_fields(self):
        colony = create_colony()
        for c in colony["colonists"]:
            assert "id" in c and "name" in c
            assert c["element"] in ELEMENTS
            assert all(s in c["stats"] for s in STAT_NAMES)
            assert all(s in c["skills"] for s in SKILL_NAMES)
            assert c["alive"] is True
    def test_relationships_initialized(self):
        colony = create_colony()
        for c in colony["colonists"]:
            assert len(c["relationships"]) == 9
    def test_deterministic_creation(self):
        a, b = create_colony(seed=42), create_colony(seed=42)
        assert a["colonists"][0]["name"] == b["colonists"][0]["name"]
        assert a["colonists"][0]["stats"] == b["colonists"][0]["stats"]
    def test_different_seeds_differ(self):
        a, b = create_colony(seed=42), create_colony(seed=99)
        assert a["colonists"][0]["stats"] != b["colonists"][0]["stats"]
    def test_element_affinity_boosts(self):
        rng = random.Random(100)
        for _ in range(50):
            c = create_colonist(0, rng)
            if c["element"] == "fire": assert c["stats"]["resolve"] >= 35
            elif c["element"] == "water": assert c["stats"]["empathy"] >= 35
    def test_custom_colony_size(self):
        assert len(create_colony(n_colonists=5)["colonists"]) == 5
    def test_meta_fields(self):
        colony = create_colony()
        assert colony["_meta"]["engine"] == "mars-100"
        assert colony["governance"]["system"] == "direct_democracy"

class TestEvents:
    def test_pick_event_valid(self):
        rng = random.Random(42)
        for year in range(1, 50):
            event = pick_event(year, rng)
            assert "id" in event and "severity" in event
    def test_supply_ship_only_every_5_years(self):
        rng = random.Random(42)
        for year in range(1, 100):
            event = pick_event(year, rng)
            if event["id"] == "supply_ship": assert year % 5 == 0
    def test_event_effects_bounded(self):
        colony = create_colony()
        rng = random.Random(42)
        for event in EVENTS:
            tc = copy.deepcopy(colony)
            apply_event_effects(tc, event, rng)
            for key, val in tc["resources"].items():
                if isinstance(val, (int, float)): assert val >= 0

class TestColonistActions:
    def test_action_expr_parses(self):
        colony = create_colony()
        for c in colony["colonists"]:
            for event in EVENTS[:3]:
                expr = generate_action_expr(c, event, 10)
                env, ctx = make_env(seed=42)
                colonist_to_env(c, colony, event, env)
                result = run_in_env(expr, env, ctx)
    def test_action_processing(self):
        colony = create_colony()
        rng = random.Random(42)
        c = colony["colonists"][0]
        assert c["name"] in process_action(c, {"type": "work", "skill": "hydroponics"}, colony, rng)
        assert "idle" in process_action(c, None, colony, rng).lower()
    def test_propose_creates_proposal(self):
        colony = create_colony()
        c = colony["colonists"][0]
        process_action(c, {"type": "propose", "governance_type": "leadership_election", "detail": "I should lead"}, colony, random.Random(42))
        assert len(colony["proposals_pending"]) == 1
    def test_mediate_improves_relationships(self):
        colony = create_colony()
        rng = random.Random(42)
        c = colony["colonists"][0]
        id_a, id_b = colony["colonists"][1]["id"], colony["colonists"][2]["id"]
        rel_before = colony["colonists"][1]["relationships"].get(str(id_b), 0)
        process_action(c, {"type": "mediate", "between": [id_a, id_b]}, colony, rng)
        assert colony["colonists"][1]["relationships"].get(str(id_b), 0) >= rel_before

class TestSubSim:
    def test_subsim_expr_valid(self):
        colony = create_colony()
        c = colony["colonists"][0]
        event = EVENTS[0]
        for gtype in GOVERNANCE_TYPES[:4]:
            expr = generate_subsim_expr(c, {"governance_type": gtype})
            env, ctx = make_env(seed=42)
            colonist_to_env(c, colony, event, env)
            result = run_in_env(expr, env, ctx)
            assert result is not None and isinstance(result, list)
    def test_subsim_isolation(self):
        colony = create_colony()
        original_food = colony["resources"]["food"]
        c, event = colony["colonists"][0], EVENTS[0]
        env, ctx = make_env(seed=42)
        colonist_to_env(c, colony, event, env)
        run_in_env('(sub-sim (begin (set! colony-food 999999) colony-food))', env, ctx)
        assert colony["resources"]["food"] == original_food

class TestGovernance:
    def test_proposal_resolution(self):
        colony = create_colony()
        colony["proposals_pending"].append({
            "id": 0, "year": 1, "proposer": 0, "governance_type": "research_directive",
            "detail": "fund coding", "votes_for": [0], "votes_against": [], "resolved": False,
        })
        assert len(resolve_proposals(colony, random.Random(42))) > 0
    def test_dead_colonists_dont_vote(self):
        colony = create_colony()
        for c in colony["colonists"][:5]: c["alive"] = False
        colony["proposals_pending"].append({
            "id": 0, "year": 1, "proposer": 5, "governance_type": "leadership_election",
            "detail": "", "votes_for": [5], "votes_against": [], "resolved": False,
        })
        assert len(resolve_proposals(colony, random.Random(42))) > 0

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
        assert colony["resources"]["food"] >= 0 and colony["resources"]["oxygen"] >= 0
    def test_starvation_death(self):
        colony = create_colony()
        colony["resources"]["food"] = 0
        colony["resources"]["oxygen"] = 0
        effects = consume_resources(colony)
        assert any("dies" in e.lower() for e in effects)
    def test_death_creates_soul(self):
        colony = create_colony()
        colony["resources"]["food"] = 0
        consume_resources(colony)
        if colony["dead_souls"]:
            assert colony["dead_souls"][0]["alive"] is False
    def test_extinction_handled(self):
        colony = create_colony()
        for c in colony["colonists"]: c["alive"] = False
        assert "extinct" in consume_resources(colony)[0].lower()

class TestPopulation:
    def test_births_require_minimum_pop(self):
        colony = create_colony()
        for c in colony["colonists"][1:]: c["alive"] = False
        for _ in range(100): check_births(colony, random.Random(42))
        assert len(colony["colonists"]) == 10
    def test_births_add_relationships(self):
        colony = create_colony()
        colony["resources"]["morale"] = 100
        rng = random.Random(42)
        for _ in range(200): check_births(colony, rng)
        newborns = [c for c in colony["colonists"] if c["year_born"] > 0]
        if newborns: assert len(newborns[0]["relationships"]) > 0

class TestRelationships:
    def test_relationships_change(self):
        colony = create_colony()
        before = copy.deepcopy(colony["colonists"][0]["relationships"])
        evolve_relationships(colony, random.Random(42))
        assert colony["colonists"][0]["relationships"] != before
    def test_relationships_bounded(self):
        colony = create_colony()
        rng = random.Random(42)
        for _ in range(100): evolve_relationships(colony, rng)
        for c in colony["colonists"]:
            if c["alive"]:
                for val in c["relationships"].values(): assert -100 <= val <= 100

class TestMetaAwareness:
    def test_no_awareness_early(self):
        assert check_meta_awareness(create_colony(), 5) is None
    def test_awareness_possible_late(self):
        colony = create_colony()
        c = colony["colonists"][0]
        c["stats"]["improvisation"] = 90
        c["stats"]["faith"] = 90
        c["sub_sims_run"] = 10
        assert check_meta_awareness(colony, 80) is not None

class TestSimulation:
    def test_smoke_10_years(self):
        r = run_simulation(years=10, seed=42)
        assert r["summary"]["years_survived"] == 10
        assert r["summary"]["final_population"] > 0
    def test_smoke_50_years(self):
        assert run_simulation(years=50, seed=42)["summary"]["years_survived"] > 0
    def test_deterministic_replay(self):
        a, b = run_simulation(years=20, seed=42), run_simulation(years=20, seed=42)
        assert a["summary"]["years_survived"] == b["summary"]["years_survived"]
        assert a["summary"]["population_curve"] == b["summary"]["population_curve"]
    def test_different_seeds_diverge(self):
        a, b = run_simulation(years=30, seed=42), run_simulation(years=30, seed=99)
        assert a["summary"]["population_curve"] != b["summary"]["population_curve"]
    def test_sub_sims_generated(self):
        assert run_simulation(years=30, seed=42)["summary"]["total_sub_simulations"] > 0
    def test_diary_entries_generated(self):
        r = run_simulation(years=10, seed=42)
        for d in r["deltas"]:
            if d["population"] > 0: assert len(d["diary_entries"]) > 0
    def test_population_curve_valid(self):
        for pop in run_simulation(years=30, seed=42)["summary"]["population_curve"]:
            assert isinstance(pop, int) and pop >= 0
    def test_morale_bounded(self):
        for m in run_simulation(years=50, seed=42)["summary"]["morale_curve"]:
            assert 0 <= m <= 100
    def test_governance_evolves(self):
        assert sum(len(d["governance_results"]) for d in run_simulation(years=50, seed=42)["deltas"]) > 0
    def test_resources_bounded(self):
        for d in run_simulation(years=50, seed=42)["deltas"]:
            for key, val in d["resources_snapshot"].items():
                if isinstance(val, (int, float)): assert val >= 0
    def test_history_length_matches_years(self):
        r = run_simulation(years=20, seed=42)
        assert len(r["deltas"]) == r["summary"]["years_survived"]


# ----------- New tests for evolution features -----------

from src.mars100 import compute_value_convergence, spawn_deep_subsim


class TestValueConvergence:
    """Value convergence tracking — measures stat divergence across the population."""

    def test_convergence_returns_dict(self):
        colony = create_colony(seed=42)
        result = compute_value_convergence(colony)
        assert isinstance(result, dict)
        assert "per_stat" in result and "score" in result and "n" in result

    def test_convergence_per_stat_keys(self):
        colony = create_colony(seed=42)
        result = compute_value_convergence(colony)
        for stat in STAT_NAMES:
            assert stat in result["per_stat"]
            assert isinstance(result["per_stat"][stat], float)
            assert result["per_stat"][stat] >= 0

    def test_convergence_score_non_negative(self):
        colony = create_colony(seed=42)
        result = compute_value_convergence(colony)
        assert result["score"] >= 0.0

    def test_convergence_with_identical_colonists(self):
        colony = create_colony(seed=42)
        # Make all colonists identical
        template = colony["colonists"][0]["stats"].copy()
        for c in colony["colonists"]:
            c["stats"] = template.copy()
        result = compute_value_convergence(colony)
        assert result["score"] == 0.0

    def test_convergence_with_one_alive(self):
        colony = create_colony(seed=42)
        for c in colony["colonists"][1:]:
            c["alive"] = False
        result = compute_value_convergence(colony)
        assert result["score"] == 0.0
        assert result["n"] == 1

    def test_convergence_in_delta(self):
        colony = create_colony(seed=42)
        delta = tick_year(colony, 1, 42)
        assert "value_convergence" in delta
        assert delta["value_convergence"]["n"] >= 1

    def test_convergence_in_summary(self):
        r = run_simulation(years=10, seed=42)
        assert "convergence_curve" in r["summary"]
        assert len(r["summary"]["convergence_curve"]) == r["summary"]["years_survived"]
        for score in r["summary"]["convergence_curve"]:
            assert isinstance(score, (int, float)) and score >= 0

    def test_convergence_curve_bounded(self):
        """Convergence score should be bounded — max std dev is 50 for 0-100 stats."""
        r = run_simulation(years=50, seed=42)
        for score in r["summary"]["convergence_curve"]:
            assert score < 60, f"Convergence score {score} seems unreasonably high"


class TestDeepSubSims:
    """Deep sub-simulations (depth 2-3) for skilled colonists."""

    def test_deep_subsim_returns_list(self):
        colony = create_colony(seed=42)
        event = EVENTS[0]
        result = spawn_deep_subsim(colony["colonists"][0], colony, event, 1, 42)
        assert isinstance(result, list)

    def test_deep_subsim_low_skill_returns_empty(self):
        colony = create_colony(seed=42)
        c = colony["colonists"][0]
        c["skills"]["coding"] = 5
        c["stats"]["improvisation"] = 5
        result = spawn_deep_subsim(c, colony, EVENTS[0], 1, 42)
        assert result == []

    def test_deep_subsim_high_skill_returns_entries(self):
        colony = create_colony(seed=42)
        c = colony["colonists"][0]
        c["skills"]["coding"] = 70
        c["stats"]["improvisation"] = 60
        c["sub_sims_run"] = 0
        result = spawn_deep_subsim(c, colony, EVENTS[0], 50, 42)
        assert len(result) >= 1
        assert result[0]["depth"] == 2
        assert c["sub_sims_run"] >= 1

    def test_depth_3_requires_veteran(self):
        colony = create_colony(seed=42)
        c = colony["colonists"][0]
        c["skills"]["coding"] = 70
        c["stats"]["improvisation"] = 60
        c["stats"]["faith"] = 60
        c["year_born"] = 0
        c["sub_sims_run"] = 0
        # Year 10 — not veteran enough (need 20+ years alive)
        result = spawn_deep_subsim(c, colony, EVENTS[0], 10, 42)
        depth_3 = [s for s in result if s.get("depth") == 3]
        assert len(depth_3) == 0

    def test_depth_3_veteran_colonist(self):
        colony = create_colony(seed=42)
        c = colony["colonists"][0]
        c["skills"]["coding"] = 70
        c["stats"]["improvisation"] = 60
        c["stats"]["faith"] = 60
        c["stats"]["resolve"] = 80
        c["stats"]["empathy"] = 50
        c["year_born"] = 0
        c["sub_sims_run"] = 0
        result = spawn_deep_subsim(c, colony, EVENTS[0], 50, 42)
        depth_3 = [s for s in result if s.get("depth") == 3]
        assert len(depth_3) == 1
        assert "meta_insight" in depth_3[0]

    def test_deep_sims_in_tick(self):
        """Deep sims should appear in tick deltas after year 1."""
        colony = create_colony(seed=42)
        # Boost several colonists to trigger deep sims
        for c in colony["colonists"][:3]:
            c["skills"]["coding"] = 65
            c["stats"]["improvisation"] = 55
        delta = tick_year(colony, 5, 42)
        deep = [s for s in delta["sub_sims"] if s.get("depth", 1) >= 2]
        assert len(deep) >= 1, "Expected at least one deep sub-sim with boosted colonists"

    def test_deep_sims_in_summary(self):
        r = run_simulation(years=10, seed=42)
        assert "deep_sub_simulations" in r["summary"]
        assert isinstance(r["summary"]["deep_sub_simulations"], int)

    def test_meta_insights_in_summary(self):
        r = run_simulation(years=10, seed=42)
        assert "depth_3_meta_insights" in r["summary"]
        assert isinstance(r["summary"]["depth_3_meta_insights"], list)


class TestMetaInsights:
    """Meta-insight extraction from depth-3 sub-simulations."""

    def test_extract_meta_insight_from_valid(self):
        from src.mars100.sim import _extract_meta_insight
        result = ["depth-3", 3, 5000.0, "Governance works best when every voice is heard."]
        assert _extract_meta_insight(result) is not None

    def test_extract_meta_insight_short_string(self):
        from src.mars100.sim import _extract_meta_insight
        result = ["depth-3", 3, 5000.0, "short"]
        assert _extract_meta_insight(result) is None

    def test_extract_meta_insight_missing(self):
        from src.mars100.sim import _extract_meta_insight
        assert _extract_meta_insight(["depth-3", 3]) is None
        assert _extract_meta_insight("not a list") is None
        assert _extract_meta_insight(42) is None

    def test_full_sim_meta_insights(self):
        """Run a full sim with boosted colonists and verify meta-insights emerge."""
        colony = create_colony(seed=42)
        for c in colony["colonists"][:3]:
            c["skills"]["coding"] = 70
            c["stats"]["improvisation"] = 60
            c["stats"]["faith"] = 60
            c["stats"]["resolve"] = 80
            c["stats"]["empathy"] = 50
        deltas = []
        for year in range(1, 60):
            delta = tick_year(colony, year, 42)
            deltas.append(delta)
        all_d3 = [s for d in deltas for s in d["sub_sims"]
                   if s.get("depth") == 3 and s.get("meta_insight")]
        # With boosted colonists over 60 years, we should see meta-insights
        assert len(all_d3) >= 1, "Expected at least 1 depth-3 meta-insight over 60 years"
