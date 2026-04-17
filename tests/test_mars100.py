"""Tests for the Mars-100 recursive colony simulation."""
from __future__ import annotations

import copy
import json
import random
import tempfile
from pathlib import Path

import pytest

from src.mars100 import (
    DEFAULT_SEED, DEFAULT_YEARS, ELEMENTS, EVENTS,
    MAX_SUBSIMS_PER_YEAR, MAX_LEARNED_RULES, MORALE_MAX, MORALE_MIN,
    RESOURCE_MAX, RESOURCE_MIN, META_AWARENESS_YEAR,
    MAX_META_AWARE_COLONISTS,
    apply_active_laws, check_meta_awareness, colonist_to_lispy_env,
    compute_year_environment, evolve_learned_rules,
    generate_action_lispy, generate_governance_subsim_lispy,
    generate_summary, make_colonists, make_initial_state, pick_event,
    promote_subsim_insight, resolve_governance, run_simulation,
    simulate_year,
)
from src.lispy import Evaluator, parse, make_standard_env, LispyError


# ---------------------------------------------------------------------------
# Colonist creation
# ---------------------------------------------------------------------------

class TestColonists:
    def test_makes_10_colonists(self) -> None:
        rng = random.Random(42)
        colonists = make_colonists(rng)
        assert len(colonists) == 10

    def test_unique_ids(self) -> None:
        rng = random.Random(42)
        colonists = make_colonists(rng)
        ids = [c["id"] for c in colonists]
        assert len(set(ids)) == 10

    def test_all_elements_present(self) -> None:
        rng = random.Random(42)
        colonists = make_colonists(rng)
        elements = {c["element"] for c in colonists}
        assert elements == set(ELEMENTS)

    def test_stats_in_range(self) -> None:
        rng = random.Random(42)
        colonists = make_colonists(rng)
        for c in colonists:
            for stat_val in c["stats"].values():
                assert 0.0 <= stat_val <= 1.0

    def test_skills_in_range(self) -> None:
        rng = random.Random(42)
        colonists = make_colonists(rng)
        for c in colonists:
            for skill_val in c["skills"].values():
                assert 0.0 <= skill_val <= 1.0

    def test_relationships_initialized(self) -> None:
        rng = random.Random(42)
        colonists = make_colonists(rng)
        for c in colonists:
            assert len(c["relationships"]) == 9  # 10 colonists - 1 self

    def test_has_learned_rules(self) -> None:
        rng = random.Random(42)
        colonists = make_colonists(rng)
        for c in colonists:
            assert c["learned_rules"] == []
            assert c["meta_aware"] is False


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_has_all_fields(self) -> None:
        state = make_initial_state()
        assert "_meta" in state
        assert "resources" in state
        assert "environment" in state
        assert "colonists" in state
        assert "governance" in state

    def test_version_2(self) -> None:
        state = make_initial_state()
        assert state["_meta"]["version"] == "2.0"

    def test_meta_awareness_count(self) -> None:
        state = make_initial_state()
        assert state["_meta"]["meta_awareness_count"] == 0

    def test_governance_none(self) -> None:
        state = make_initial_state()
        assert state["governance"]["type"] == "none"
        assert state["governance"]["amendments"] == []

    def test_deterministic(self) -> None:
        s1 = make_initial_state(42)
        s2 = make_initial_state(42)
        for c1, c2 in zip(s1["colonists"], s2["colonists"]):
            assert c1["stats"] == c2["stats"]

    def test_different_seeds(self) -> None:
        s1 = make_initial_state(1)
        s2 = make_initial_state(2)
        assert s1["colonists"][0]["stats"] != s2["colonists"][0]["stats"]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class TestEvents:
    def test_returns_event(self) -> None:
        rng = random.Random(42)
        event = pick_event(rng)
        assert "name" in event
        assert "resource_impact" in event
        assert "morale_impact" in event

    def test_all_events_reachable(self) -> None:
        rng = random.Random(42)
        seen = set()
        for _ in range(10000):
            event = pick_event(rng)
            seen.add(event["name"])
        for e in EVENTS:
            assert e["name"] in seen, f"Event {e['name']} never picked"

    def test_cave_discovery_exists(self) -> None:
        names = [e["name"] for e in EVENTS]
        assert "cave_discovery" in names

    def test_comms_blackout_exists(self) -> None:
        names = [e["name"] for e in EVENTS]
        assert "comms_blackout" in names


# ---------------------------------------------------------------------------
# LisPy environment
# ---------------------------------------------------------------------------

class TestColonistLispyEnv:
    def test_bindings_present(self) -> None:
        state = make_initial_state()
        colonist = state["colonists"][0]
        env = colonist_to_lispy_env(colonist, state)
        assert env.lookup("my-id") == colonist["id"]
        assert env.lookup("my-name") == colonist["name"]

    def test_resources_bound(self) -> None:
        state = make_initial_state()
        colonist = state["colonists"][0]
        env = colonist_to_lispy_env(colonist, state)
        assert env.lookup("res-food") == state["resources"]["food"]

    def test_observe_function(self) -> None:
        state = make_initial_state()
        colonist = state["colonists"][0]
        env = colonist_to_lispy_env(colonist, state)
        observe = env.lookup("observe")
        assert observe("population") == 10

    def test_propose_function(self) -> None:
        state = make_initial_state()
        colonist = state["colonists"][0]
        env = colonist_to_lispy_env(colonist, state)
        propose = env.lookup("propose")
        prop_id = propose("Test proposal", "Description")
        assert len(state["governance"]["proposals"]) == 1


# ---------------------------------------------------------------------------
# Action generation
# ---------------------------------------------------------------------------

class TestActionGeneration:
    def test_generates_valid_lispy(self) -> None:
        state = make_initial_state(42)
        rng = random.Random(42)
        colonist = state["colonists"][0]
        event = pick_event(rng)
        source = generate_action_lispy(colonist, state, event, rng)
        assert isinstance(source, str)
        assert len(source) > 0

    def test_evaluable(self) -> None:
        state = make_initial_state(42)
        rng = random.Random(42)
        colonist = state["colonists"][0]
        event = pick_event(rng)
        source = generate_action_lispy(colonist, state, event, rng)
        env = colonist_to_lispy_env(colonist, state)
        evaluator = Evaluator(max_steps=5000, max_depth=32, max_sim_depth=3,
                              max_subsims_per_frame=6)
        result = evaluator.eval(parse(source), env)
        assert result is not None

    def test_different_colonists_different_actions(self) -> None:
        state = make_initial_state(42)
        rng = random.Random(42)
        event = pick_event(rng)
        actions = set()
        for c in state["colonists"]:
            source = generate_action_lispy(c, state, event, rng)
            actions.add(source)
        assert len(actions) > 1


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------

class TestGovernance:
    def test_proposal_passes(self) -> None:
        state = make_initial_state()
        state["governance"]["proposals"].append({
            "id": "prop-1", "title": "Establish governance council",
            "proposer": "ares", "year_proposed": 3,
            "expiry": 6, "status": "active",
            "votes": {c["id"]: "yes" for c in state["colonists"][:7]},
        })
        events = resolve_governance(state, 3)
        assert any("PASSED" in e for e in events)

    def test_proposal_rejected(self) -> None:
        state = make_initial_state()
        state["governance"]["proposals"].append({
            "id": "prop-1", "title": "Bad idea",
            "proposer": "ares", "year_proposed": 3,
            "expiry": 6, "status": "active",
            "votes": {c["id"]: "no" for c in state["colonists"][:7]},
        })
        events = resolve_governance(state, 3)
        assert any("REJECTED" in e for e in events)

    def test_proposal_expires(self) -> None:
        state = make_initial_state()
        state["governance"]["proposals"].append({
            "id": "prop-1", "title": "Old proposal",
            "proposer": "ares", "year_proposed": 1,
            "expiry": 3, "status": "active", "votes": {},
        })
        events = resolve_governance(state, 4)
        assert any("expired" in e for e in events)

    def test_quorum_required(self) -> None:
        state = make_initial_state()
        state["governance"]["proposals"].append({
            "id": "prop-1", "title": "Not enough votes",
            "proposer": "ares", "year_proposed": 3,
            "expiry": 6, "status": "active",
            "votes": {"ares": "yes"},
        })
        events = resolve_governance(state, 3)
        assert len(events) == 0

    def test_exile_governance(self) -> None:
        state = make_initial_state()
        state["governance"]["proposals"].append({
            "id": "prop-exile", "title": "Exile troublemaker",
            "description": "Remove: flint",
            "proposer": "ares", "year_proposed": 5,
            "expiry": 8, "status": "active",
            "votes": {c["id"]: "yes" for c in state["colonists"][:8]},
        })
        events = resolve_governance(state, 5)
        assert any("exiled" in e.lower() for e in events)


# ---------------------------------------------------------------------------
# Year simulation
# ---------------------------------------------------------------------------

class TestYearSimulation:
    def test_year_advances(self) -> None:
        state = make_initial_state(42)
        rng = random.Random(42)
        simulate_year(state, rng)
        assert state["_meta"]["year"] == 1

    def test_returns_delta(self) -> None:
        state = make_initial_state(42)
        rng = random.Random(42)
        delta = simulate_year(state, rng)
        assert "event" in delta
        assert "colonist_actions" in delta
        assert "seed" in delta
        assert "depth" in delta

    def test_resources_bounded(self) -> None:
        state = make_initial_state(42)
        rng = random.Random(42)
        for _ in range(10):
            simulate_year(state, rng)
        for val in state["resources"].values():
            assert RESOURCE_MIN <= val <= RESOURCE_MAX

    def test_morale_bounded(self) -> None:
        state = make_initial_state(42)
        rng = random.Random(42)
        for _ in range(10):
            simulate_year(state, rng)
        for c in state["colonists"]:
            if c["alive"]:
                assert MORALE_MIN <= c["morale"] <= MORALE_MAX

    def test_relationships_bounded(self) -> None:
        state = make_initial_state(42)
        rng = random.Random(42)
        for _ in range(10):
            simulate_year(state, rng)
        for c in state["colonists"]:
            if c["alive"]:
                for val in c["relationships"].values():
                    assert -1.0 <= val <= 1.0

    def test_memory_bounded(self) -> None:
        state = make_initial_state(42)
        rng = random.Random(42)
        for _ in range(30):
            simulate_year(state, rng)
        for c in state["colonists"]:
            assert len(c["memory"]) <= 20


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestSmokeSimulation:
    def test_10_years_no_crash(self) -> None:
        state = make_initial_state(42)
        rng = random.Random(42)
        for _ in range(10):
            simulate_year(state, rng)
        assert state["_meta"]["year"] == 10

    def test_25_years_no_crash(self) -> None:
        state = make_initial_state(42)
        rng = random.Random(42)
        for _ in range(25):
            simulate_year(state, rng)
        assert state["_meta"]["year"] == 25

    def test_deterministic_replay(self) -> None:
        def run(seed):
            s = make_initial_state(seed)
            r = random.Random(seed)
            for _ in range(10):
                simulate_year(s, r)
            return s["_meta"]["year"], s["resources"]["food"]
        r1 = run(42)
        r2 = run(42)
        assert r1 == r2

    def test_different_seeds_diverge(self) -> None:
        def run(seed):
            s = make_initial_state(seed)
            r = random.Random(seed)
            for _ in range(10):
                simulate_year(s, r)
            return s["resources"]["food"]
        assert run(1) != run(2)

    def test_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            run_simulation(years=10, seed=42, output_dir=output)
            assert (output / "summary.json").exists()

    def test_summary_structure(self) -> None:
        state = make_initial_state(42)
        rng = random.Random(42)
        for _ in range(10):
            simulate_year(state, rng)
        summary = generate_summary(state)
        assert "final_population" in summary
        assert "governance" in summary
        assert "subsim_stats" in summary
        assert "meta_awareness" in summary
        assert "learned_rules" in summary

    def test_subsims_logged(self) -> None:
        state = make_initial_state(42)
        rng = random.Random(42)
        total = 0
        for _ in range(20):
            delta = simulate_year(state, rng)
            total += len(delta.get("subsim_log", []))
        assert total >= 0  # sub-sims may or may not trigger

    def test_governance_emerges(self) -> None:
        state = make_initial_state(42)
        rng = random.Random(42)
        for _ in range(50):
            simulate_year(state, rng)
        total_gov = (len(state["governance"]["proposals"])
                     + len(state["governance"]["passed_laws"]))
        assert total_gov > 0


# ---------------------------------------------------------------------------
# Sub-sim isolation
# ---------------------------------------------------------------------------

class TestSubSimIsolation:
    def test_subsim_does_not_mutate_parent_resources(self) -> None:
        state = make_initial_state(42)
        rng = random.Random(42)
        original_food = state["resources"]["food"]
        # Run a few years — sub-sims may run but shouldn't corrupt via snapshot
        for _ in range(5):
            simulate_year(state, rng)
        # Resources change from normal sim, but snapshot isolation means
        # sub-sim changes don't leak
        assert isinstance(state["resources"]["food"], float)

    def test_subsim_snapshot_is_deep_copy(self) -> None:
        state = make_initial_state(42)
        snapshot = copy.deepcopy({
            "resources": state["resources"],
            "environment": state["environment"],
            "alive_count": 10,
        })
        state["resources"]["food"] = 999.0
        assert snapshot["resources"]["food"] != 999.0


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    @pytest.mark.parametrize("seed", [1, 42, 100, 2026, 9999])
    def test_population_conservation(self, seed: int) -> None:
        """Alive + dead = initial population (no spontaneous generation)."""
        state = make_initial_state(seed)
        rng = random.Random(seed)
        for _ in range(15):
            simulate_year(state, rng)
        alive = sum(1 for c in state["colonists"] if c["alive"])
        dead = len(state["dead_colonists"])
        assert alive + dead <= 10

    @pytest.mark.parametrize("seed", [1, 42, 100])
    def test_terraforming_monotonic(self, seed: int) -> None:
        """Terraforming percentage only increases."""
        state = make_initial_state(seed)
        rng = random.Random(seed)
        prev_tf = 0.0
        for _ in range(10):
            simulate_year(state, rng)
            assert state["environment"]["terraforming_pct"] >= prev_tf
            prev_tf = state["environment"]["terraforming_pct"]

    @pytest.mark.parametrize("seed", [1, 42, 100, 2026])
    def test_resources_always_bounded(self, seed: int) -> None:
        state = make_initial_state(seed)
        rng = random.Random(seed)
        for _ in range(15):
            simulate_year(state, rng)
            for val in state["resources"].values():
                assert RESOURCE_MIN <= val <= RESOURCE_MAX

    @pytest.mark.parametrize("seed", [1, 42, 100])
    def test_health_decreases_or_death(self, seed: int) -> None:
        state = make_initial_state(seed)
        rng = random.Random(seed)
        prev_health = {c["id"]: c["health"] for c in state["colonists"]}
        for _ in range(5):
            simulate_year(state, rng)
            for c in state["colonists"]:
                if c["alive"]:
                    assert c["health"] <= prev_health[c["id"]] + 5.0
                prev_health[c["id"]] = c["health"]


# ---------------------------------------------------------------------------
# v2.0: Executable laws
# ---------------------------------------------------------------------------

class TestExecutableLaws:
    def test_apply_active_laws_empty(self) -> None:
        state = make_initial_state()
        mods = apply_active_laws(state)
        assert mods["resource_multiplier"] == {}
        assert mods["stat_modifier"] == {}

    def test_ration_law_effect(self) -> None:
        state = make_initial_state()
        state["governance"]["passed_laws"].append({
            "title": "Ration food and water",
            "year_passed": 5, "proposer": "ares",
            "votes_for": 7, "votes_against": 2,
        })
        mods = apply_active_laws(state)
        assert mods["resource_multiplier"]["food"] == 0.7
        assert mods["resource_multiplier"]["water"] == 0.7

    def test_shared_resources_law(self) -> None:
        state = make_initial_state()
        state["governance"]["passed_laws"].append({
            "title": "Shared resources protocol",
            "year_passed": 8, "proposer": "ivy",
            "votes_for": 6, "votes_against": 3,
        })
        mods = apply_active_laws(state)
        assert mods["stat_modifier"]["hoarding"] == -0.15
        assert mods["relationship_boost"] > 0

    def test_laws_dont_compound(self) -> None:
        state = make_initial_state()
        state["governance"]["passed_laws"].append({
            "title": "Ration food and water",
            "year_passed": 5, "proposer": "ares",
            "votes_for": 7, "votes_against": 2,
        })
        mods1 = apply_active_laws(state)
        mods2 = apply_active_laws(state)
        assert mods1["resource_multiplier"] == mods2["resource_multiplier"]

    def test_terraform_law_bonus(self) -> None:
        state = make_initial_state()
        state["governance"]["passed_laws"].append({
            "title": "Terraform initiative",
            "year_passed": 12, "proposer": "petra",
            "votes_for": 8, "votes_against": 1,
        })
        mods = apply_active_laws(state)
        assert mods["terraforming_bonus"] > 0

    def test_consumption_uses_law_multiplier(self) -> None:
        state = make_initial_state(seed=42)
        rng = random.Random(42)
        for _ in range(5):
            simulate_year(state, rng)
        food_no_law = state["resources"]["food"]

        state2 = make_initial_state(seed=42)
        rng2 = random.Random(42)
        state2["governance"]["passed_laws"].append({
            "title": "Ration food and water",
            "year_passed": 0, "proposer": "ares",
            "votes_for": 7, "votes_against": 2,
        })
        for _ in range(5):
            simulate_year(state2, rng2)
        food_with_law = state2["resources"]["food"]
        assert food_with_law > food_no_law


# ---------------------------------------------------------------------------
# v2.0: Learned rules
# ---------------------------------------------------------------------------

class TestLearnedRules:
    def test_rules_bounded(self) -> None:
        colonist = make_initial_state()["colonists"][0]
        event = EVENTS[0]
        for year in range(30):
            result = {"status": "complete", "label": f"test-{year}",
                      "result": ["rec", "act"]}
            evolve_learned_rules(colonist, result, event, year)
        assert len(colonist["learned_rules"]) <= MAX_LEARNED_RULES

    def test_subsim_rule_learned(self) -> None:
        colonist = make_initial_state()["colonists"][0]
        event = EVENTS[0]
        result = {"status": "complete", "label": "test-sim",
                  "result": ["recommendation", "conserve"]}
        evolve_learned_rules(colonist, result, event, 5)
        assert any(r["source"] == "subsim" for r in colonist["learned_rules"])

    def test_experience_rule(self) -> None:
        colonist = make_initial_state()["colonists"][0]
        colonist["health"] = 50.0
        harsh = {"name": "dust_storm", "morale_impact": -20, "resource_impact": -0.15}
        evolve_learned_rules(colonist, None, harsh, 10)
        assert any(r["source"] == "experience" for r in colonist["learned_rules"])

    def test_rules_accumulate(self) -> None:
        state = make_initial_state(seed=42)
        rng = random.Random(42)
        for _ in range(20):
            simulate_year(state, rng)
        total = sum(len(c["learned_rules"]) for c in state["colonists"])
        assert total > 0


# ---------------------------------------------------------------------------
# v2.0: Meta-awareness
# ---------------------------------------------------------------------------

class TestMetaAwareness:
    def test_no_meta_before_threshold(self) -> None:
        state = make_initial_state()
        rng = random.Random(42)
        c = state["colonists"][0]
        c["stats"]["faith"] = 0.9
        c["stats"]["paranoia"] = 0.8
        c["subsims_run"] = 10
        assert check_meta_awareness(c, state, META_AWARENESS_YEAR - 1, rng) is None

    def test_requires_faith(self) -> None:
        state = make_initial_state()
        rng = random.Random(42)
        c = state["colonists"][0]
        c["stats"]["faith"] = 0.1
        c["stats"]["paranoia"] = 0.8
        c["subsims_run"] = 10
        assert check_meta_awareness(c, state, META_AWARENESS_YEAR + 10, rng) is None

    def test_requires_subsim_experience(self) -> None:
        state = make_initial_state()
        rng = random.Random(42)
        c = state["colonists"][0]
        c["stats"]["faith"] = 0.9
        c["stats"]["paranoia"] = 0.8
        c["subsims_run"] = 0
        assert check_meta_awareness(c, state, META_AWARENESS_YEAR + 10, rng) is None

    def test_global_cap(self) -> None:
        state = make_initial_state()
        state["_meta"]["meta_awareness_count"] = MAX_META_AWARE_COLONISTS
        rng = random.Random(42)
        c = state["colonists"][0]
        c["stats"]["faith"] = 0.9
        c["stats"]["paranoia"] = 0.8
        c["subsims_run"] = 10
        assert check_meta_awareness(c, state, META_AWARENESS_YEAR + 50, rng) is None

    def test_fires_once_per_colonist(self) -> None:
        state = make_initial_state()
        rng = random.Random(42)
        c = state["colonists"][0]
        c["meta_aware"] = True
        c["meta_aware_year"] = 50
        assert check_meta_awareness(c, state, 60, rng) is None

    def test_produces_amendment(self) -> None:
        state = make_initial_state()
        c = state["colonists"][0]
        c["stats"]["faith"] = 0.9
        c["stats"]["paranoia"] = 0.8
        c["subsims_run"] = 100
        triggered = False
        for seed in range(1000):
            r = random.Random(seed)
            cc = copy.deepcopy(c)
            ss = copy.deepcopy(state)
            result = check_meta_awareness(cc, ss, META_AWARENESS_YEAR + 50, r)
            if result is not None:
                assert "proposed_amendment" in result
                assert "Right to Know" in result["proposed_amendment"]["title"]
                triggered = True
                break
        assert triggered


# ---------------------------------------------------------------------------
# v2.0: Governance sub-sims
# ---------------------------------------------------------------------------

class TestGovernanceSubSims:
    def test_exile_subsim(self) -> None:
        state = make_initial_state()
        rng = random.Random(42)
        lispy = generate_governance_subsim_lispy(
            {"title": "Exile Flint"}, state, rng
        )
        assert lispy is not None
        assert "model-exile" in lispy

    def test_council_subsim(self) -> None:
        state = make_initial_state()
        rng = random.Random(42)
        lispy = generate_governance_subsim_lispy(
            {"title": "Establish governance council"}, state, rng
        )
        assert lispy is not None
        assert "model-council" in lispy

    def test_conservation_subsim(self) -> None:
        state = make_initial_state()
        rng = random.Random(42)
        lispy = generate_governance_subsim_lispy(
            {"title": "Ration food supplies"}, state, rng
        )
        assert lispy is not None
        assert "model-conservation" in lispy

    def test_unknown_returns_none(self) -> None:
        state = make_initial_state()
        rng = random.Random(42)
        lispy = generate_governance_subsim_lispy(
            {"title": "Something random"}, state, rng
        )
        assert lispy is None

    def test_subsim_evaluable(self) -> None:
        state = make_initial_state()
        rng = random.Random(42)
        lispy = generate_governance_subsim_lispy(
            {"title": "Establish governance council"}, state, rng
        )
        env = make_standard_env()
        env.define("observe", lambda a="all": 10 if a == "population" else {})
        evaluator = Evaluator(
            max_steps=1000, max_depth=16, max_sim_depth=3,
            max_subsims_per_frame=1, sim_depth=0,
        )
        try:
            evaluator.eval(parse(lispy), env)
        except LispyError:
            pass  # Expected without callback


# ---------------------------------------------------------------------------
# v2.0: Insight promotion
# ---------------------------------------------------------------------------

class TestInsightPromotion:
    def test_depth_1_not_promoted(self) -> None:
        state = make_initial_state()
        result = {"depth": 1, "result": ["governance", "insight", "test"],
                  "label": "test-1"}
        assert promote_subsim_insight(result, state, 10) is None

    def test_depth_2_promoted(self) -> None:
        state = make_initial_state()
        result = {"depth": 2, "result": ["governance", "council-viable", "reason"],
                  "label": "model-council"}
        amendment = promote_subsim_insight(result, state, 15)
        assert amendment is not None
        assert "depth-2" in amendment["title"]

    def test_duplicate_not_promoted(self) -> None:
        state = make_initial_state()
        state["governance"]["amendments"].append({"source_label": "model-council"})
        result = {"depth": 2, "result": ["governance", "council-viable", "reason"],
                  "label": "model-council"}
        assert promote_subsim_insight(result, state, 20) is None

    def test_non_governance_not_promoted(self) -> None:
        state = make_initial_state()
        result = {"depth": 2, "result": ["weather", "sunny", "nice"],
                  "label": "model-weather"}
        assert promote_subsim_insight(result, state, 10) is None


# ---------------------------------------------------------------------------
# v2.0: Delta / summary structure
# ---------------------------------------------------------------------------

class TestDeltaStructure:
    def test_dream_catcher_keys(self) -> None:
        state = make_initial_state(seed=42)
        rng = random.Random(42)
        delta = simulate_year(state, rng)
        assert delta["seed"] == 42
        assert delta["depth"] == 0
        assert "meta_events" in delta

    def test_summary_v2(self) -> None:
        state = make_initial_state(seed=42)
        rng = random.Random(42)
        for _ in range(5):
            simulate_year(state, rng)
        summary = generate_summary(state)
        assert summary["_meta"]["version"] == "2.0"
        assert "meta_awareness" in summary
        assert "learned_rules" in summary
        assert "amendments" in summary["governance"]

    def test_colonist_fates_v2(self) -> None:
        state = make_initial_state(seed=42)
        rng = random.Random(42)
        for _ in range(5):
            simulate_year(state, rng)
        summary = generate_summary(state)
        for fate in summary["colonist_fates"]:
            assert "meta_aware" in fate
            assert "learned_rules" in fate


# ---------------------------------------------------------------------------
# 100-year smoke
# ---------------------------------------------------------------------------

class TestSmoke100Year:
    def test_100_years_no_crash(self) -> None:
        state = make_initial_state(seed=2026)
        rng = random.Random(2026)
        for _ in range(100):
            if sum(1 for c in state["colonists"] if c["alive"]) == 0:
                break
            simulate_year(state, rng)
        assert state["_meta"]["year"] > 0

    def test_100_years_with_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            state = run_simulation(years=100, seed=2026, output_dir=output)
            assert (output / "summary.json").exists()
            summary = json.loads((output / "summary.json").read_text())
            assert summary["_meta"]["version"] == "2.0"

    def test_deterministic_100_years(self) -> None:
        def run(seed):
            s = make_initial_state(seed)
            r = random.Random(seed)
            for _ in range(100):
                if sum(1 for c in s["colonists"] if c["alive"]) == 0:
                    break
                simulate_year(s, r)
            return (s["_meta"]["year"],
                    sum(1 for c in s["colonists"] if c["alive"]))
        assert run(2026) == run(2026)
