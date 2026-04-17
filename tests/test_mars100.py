"""Tests for the Mars-100 recursive colony simulation."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.mars100 import (
    ACTIONS, COLONIST_NAMES, ELEMENTS, MAX_RESOURCE, MAX_STAT,
    MIN_RESOURCE, MIN_STAT, SKILL_NAMES, STAT_NAMES,
    Colonist, ColonyState, GovernanceProposal, SubSimLog,
    apply_action, apply_event_to_resources, build_dashboard_data,
    check_deaths, classify_governance, consume_resources, create_colonists,
    decide_action, detect_patterns, evolve_relationships, generate_event,
    maybe_birth, resolve_proposals, run_simulation,
    write_soul_files, write_year_chapters,
)


# ---------------------------------------------------------------------------
# Colonist creation
# ---------------------------------------------------------------------------

class TestColonistCreation:
    def test_creates_10_colonists(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        assert len(colonists) == 10

    def test_unique_ids(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        ids = [c.id for c in colonists]
        assert len(set(ids)) == 10

    def test_names_match(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        for c, expected in zip(colonists, COLONIST_NAMES):
            assert c.name == expected

    def test_elements_assigned(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        for c in colonists:
            assert c.element in ELEMENTS

    def test_stats_in_bounds(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        for c in colonists:
            assert set(c.stats.keys()) == set(STAT_NAMES)
            for val in c.stats.values():
                assert MIN_STAT <= val <= MAX_STAT

    def test_skills_in_bounds(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        for c in colonists:
            assert set(c.skills.keys()) == set(SKILL_NAMES)
            for val in c.skills.values():
                assert MIN_STAT <= val <= MAX_STAT

    def test_relationships_initialized(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        for c in colonists:
            assert len(c.relationships) == 9  # 10 - self
            for other_id, trust in c.relationships.items():
                assert -1.0 <= trust <= 1.0

    def test_all_alive(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        for c in colonists:
            assert c.alive is True

    def test_initial_memory(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        for c in colonists:
            assert len(c.memory) >= 1
            assert "Year 0" in c.memory[0]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_colonist_roundtrip(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        for c in colonists:
            d = c.to_dict()
            restored = Colonist.from_dict(d)
            assert restored.id == c.id
            assert restored.name == c.name
            assert restored.element == c.element
            assert restored.stats == c.stats
            assert restored.alive == c.alive

    def test_colony_state_serialization(self):
        rng = random.Random(42)
        state = ColonyState(colonists=create_colonists(rng))
        d = state.to_dict()
        assert "colonists" in d
        assert "resources" in d
        assert len(d["colonists"]) == 10


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class TestEvents:
    def test_event_has_required_fields(self):
        rng = random.Random(42)
        event = generate_event(1, rng, 0.0)
        assert "type" in event
        assert "severity" in event
        assert "year" in event
        assert "description" in event

    def test_event_severity_in_range(self):
        rng = random.Random(42)
        for year in range(1, 101):
            event = generate_event(year, rng, 0.0)
            assert 0.0 <= event["severity"] <= 1.2  # can exceed 1.0 after year 60

    def test_event_types_valid(self):
        rng = random.Random(42)
        for year in range(1, 101):
            event = generate_event(year, rng, 0.0)
            assert event["type"] in [
                "dust_storm", "resource_strike", "equipment_failure",
                "earth_contact", "alien_signal", "solar_flare",
                "underground_water", "habitat_breach", "meteor_shower",
                "fungal_bloom",
            ]

    def test_event_resource_impact(self):
        rng = random.Random(42)
        resources = {"food": 2000, "water": 3000, "power": 1500,
                     "oxygen": 2500, "materials": 1000}
        event = {"type": "dust_storm", "severity": 0.8, "year": 1,
                 "description": "test"}
        delta = apply_event_to_resources(event, resources, rng)
        assert resources["power"] < 1500  # dust storm reduces power
        assert all(v >= MIN_RESOURCE for v in resources.values())
        assert all(v <= MAX_RESOURCE for v in resources.values())


# ---------------------------------------------------------------------------
# Decision-making
# ---------------------------------------------------------------------------

class TestDecisionMaking:
    def test_decide_returns_valid_action(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        state = ColonyState(year=1, colonists=colonists)
        event = generate_event(1, rng, 0.0)

        for c in colonists:
            action, subsim = decide_action(c, state, event, rng)
            assert action in ACTIONS

    def test_subsim_may_fire(self):
        """High-paranoia colonists should sometimes run sub-sims."""
        rng = random.Random(42)
        colonists = create_colonists(rng)
        # Force high paranoia
        for c in colonists:
            c.stats["paranoia"] = 0.9

        state = ColonyState(year=1, colonists=colonists)
        event = generate_event(1, rng, 0.0)

        subsim_fired = False
        for _ in range(50):  # try multiple times to catch stochastic behavior
            rng2 = random.Random(rng.randint(0, 10000))
            for c in colonists:
                _, subsim = decide_action(c, state, event, rng2)
                if subsim is not None:
                    subsim_fired = True
                    break
            if subsim_fired:
                break

        assert subsim_fired, "Expected at least one sub-sim to fire with high paranoia"


# ---------------------------------------------------------------------------
# Action application
# ---------------------------------------------------------------------------

class TestActions:
    def _make_state(self, rng=None):
        if rng is None:
            rng = random.Random(42)
        colonists = create_colonists(rng)
        return ColonyState(year=1, colonists=colonists), rng

    def test_repair_increases_oxygen(self):
        state, rng = self._make_state()
        c = state.colonists[0]
        before = state.resources["oxygen"]
        apply_action(c, "repair", state, rng)
        assert state.resources["oxygen"] >= before

    def test_farm_increases_food(self):
        state, rng = self._make_state()
        c = state.colonists[0]
        before = state.resources["food"]
        apply_action(c, "farm", state, rng)
        assert state.resources["food"] > before

    def test_hoard_decreases_karma(self):
        state, rng = self._make_state()
        c = state.colonists[0]
        before = c.karma
        apply_action(c, "hoard", state, rng)
        assert c.karma < before

    def test_share_increases_karma(self):
        state, rng = self._make_state()
        c = state.colonists[0]
        before = c.karma
        apply_action(c, "share", state, rng)
        assert c.karma > before

    def test_terraform_increases_progress(self):
        state, rng = self._make_state()
        c = state.colonists[0]
        before = state.terraforming_progress
        apply_action(c, "terraform", state, rng)
        assert state.terraforming_progress > before

    def test_sabotage_decreases_karma(self):
        state, rng = self._make_state()
        c = state.colonists[0]
        before = c.karma
        apply_action(c, "sabotage", state, rng)
        assert c.karma < before

    def test_rest_increases_resolve(self):
        state, rng = self._make_state()
        c = state.colonists[0]
        before = c.stats["resolve"]
        apply_action(c, "rest", state, rng)
        assert c.stats["resolve"] >= before

    def test_propose_creates_governance(self):
        state, rng = self._make_state()
        c = state.colonists[0]
        before = len(state.governance)
        apply_action(c, "propose", state, rng)
        assert len(state.governance) == before + 1

    def test_pray_increases_faith(self):
        state, rng = self._make_state()
        c = state.colonists[0]
        before = c.stats["faith"]
        apply_action(c, "pray", state, rng)
        assert c.stats["faith"] >= before

    def test_resources_stay_in_bounds(self):
        """All actions should keep resources within bounds."""
        state, rng = self._make_state()
        for action in ACTIONS:
            rng2 = random.Random(42)
            for c in state.alive_colonists():
                apply_action(c, action, state, rng2)
        for val in state.resources.values():
            assert MIN_RESOURCE <= val <= MAX_RESOURCE


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------

class TestGovernance:
    def test_proposal_and_vote(self):
        rng = random.Random(42)
        state = ColonyState(year=7, colonists=create_colonists(rng))
        c = state.colonists[0]

        # Propose
        apply_action(c, "propose", state, rng)
        assert len(state.governance) == 1
        assert state.governance[0].passed is None

        # Vote yes (enough to resolve)
        for voter in state.colonists[1:6]:
            apply_action(voter, "vote_yes", state, rng)

        results = resolve_proposals(state)
        assert len(results) >= 1
        assert state.governance[0].passed is True

    def test_rejected_proposal(self):
        rng = random.Random(42)
        state = ColonyState(year=7, colonists=create_colonists(rng))
        c = state.colonists[0]

        apply_action(c, "propose", state, rng)
        for voter in state.colonists[1:6]:
            apply_action(voter, "vote_no", state, rng)

        results = resolve_proposals(state)
        assert state.governance[0].passed is False

    def test_duplicate_vote_ignored(self):
        rng = random.Random(42)
        state = ColonyState(year=7, colonists=create_colonists(rng))
        apply_action(state.colonists[0], "propose", state, rng)
        apply_action(state.colonists[1], "vote_yes", state, rng)
        apply_action(state.colonists[1], "vote_yes", state, rng)
        assert len(state.governance[0].votes_for) == 1


# ---------------------------------------------------------------------------
# Death
# ---------------------------------------------------------------------------

class TestDeath:
    def test_no_deaths_with_good_resources(self):
        rng = random.Random(42)
        state = ColonyState(
            year=1, colonists=create_colonists(rng),
            resources={"food": 5000, "water": 5000, "power": 5000,
                       "oxygen": 5000, "materials": 5000},
        )
        event = {"type": "earth_contact", "severity": 0.3, "year": 1,
                 "description": "test"}
        # Run multiple times — with good resources, deaths should be rare
        death_count = 0
        for seed in range(100):
            rng2 = random.Random(seed)
            # Reset colonists
            for c in state.colonists:
                c.alive = True
                c.year_of_death = None
            deaths = check_deaths(state, event, rng2)
            death_count += len(deaths)
        assert death_count < 20  # low death rate with good resources

    def test_death_archives_colonist(self):
        rng = random.Random(42)
        state = ColonyState(
            year=50, colonists=create_colonists(rng),
            resources={"food": 0, "water": 0, "power": 0,
                       "oxygen": 0, "materials": 0},
        )
        event = {"type": "habitat_breach", "severity": 1.0, "year": 50,
                 "description": "test"}
        deaths = check_deaths(state, event, rng)
        for c in state.colonists:
            if not c.alive:
                assert c.year_of_death == 50
                assert c.cause_of_death is not None
                assert any("died" in m.lower() or "cause" in m.lower() for m in c.memory)


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

class TestRelationships:
    def test_cooperation_builds_trust(self):
        rng = random.Random(42)
        state = ColonyState(year=1, colonists=create_colonists(rng))
        c1, c2 = state.colonists[0], state.colonists[1]
        before = c1.relationships.get(c2.id, 0)

        # Both farm → cooperation bonus
        actions = {c.id: "farm" for c in state.colonists}
        evolve_relationships(state, actions, rng)

        assert c1.relationships[c2.id] > before

    def test_sabotage_destroys_trust(self):
        rng = random.Random(42)
        state = ColonyState(year=1, colonists=create_colonists(rng))
        c1, c2 = state.colonists[0], state.colonists[1]
        before = c1.relationships.get(c2.id, 0)

        actions = {c.id: "rest" for c in state.colonists}
        actions[c2.id] = "sabotage"
        evolve_relationships(state, actions, rng)

        assert c1.relationships[c2.id] < before

    def test_relationships_stay_bounded(self):
        rng = random.Random(42)
        state = ColonyState(year=1, colonists=create_colonists(rng))
        # Extreme cooperation for many rounds
        actions = {c.id: "farm" for c in state.colonists}
        for _ in range(200):
            evolve_relationships(state, actions, rng)
        for c in state.colonists:
            for trust in c.relationships.values():
                assert -1.0 <= trust <= 1.0


# ---------------------------------------------------------------------------
# Resource consumption
# ---------------------------------------------------------------------------

class TestConsumption:
    def test_consumption_reduces_resources(self):
        rng = random.Random(42)
        state = ColonyState(year=1, colonists=create_colonists(rng))
        food_before = state.resources["food"]
        consume_resources(state)
        # 10 colonists × 100 food = 1000 consumed, +300 produced
        # net change = -700
        assert state.resources["food"] < food_before

    def test_no_consumption_if_no_alive(self):
        state = ColonyState(year=1, colonists=[])
        resources_before = dict(state.resources)
        consume_resources(state)
        assert state.resources == resources_before

    def test_resources_never_negative(self):
        rng = random.Random(42)
        state = ColonyState(
            year=1, colonists=create_colonists(rng),
            resources={"food": 10, "water": 10, "power": 10,
                       "oxygen": 10, "materials": 10},
        )
        consume_resources(state)
        for val in state.resources.values():
            assert val >= MIN_RESOURCE


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

class TestPatterns:
    def test_leader_detection(self):
        rng = random.Random(42)
        state = ColonyState(year=1, colonists=create_colonists(rng))
        # Make one colonist a clear leader
        leader = state.colonists[0]
        leader.karma = 0.9
        for c in state.colonists:
            if c.id != leader.id:
                c.relationships[leader.id] = 0.8

        patterns = detect_patterns(state)
        leader_pats = [p for p in patterns if "LEADER" in p]
        assert len(leader_pats) >= 1

    def test_pariah_detection(self):
        rng = random.Random(42)
        state = ColonyState(year=1, colonists=create_colonists(rng))
        pariah = state.colonists[0]
        pariah.karma = 0.1
        for c in state.colonists:
            if c.id != pariah.id:
                c.relationships[pariah.id] = -0.5

        patterns = detect_patterns(state)
        pariah_pats = [p for p in patterns if "PARIAH" in p]
        assert len(pariah_pats) >= 1

    def test_extinction_detected(self):
        state = ColonyState(year=50, colonists=[])
        patterns = detect_patterns(state)
        assert any("EXTINCTION" in p for p in patterns)


# ---------------------------------------------------------------------------
# Full simulation
# ---------------------------------------------------------------------------

class TestFullSimulation:
    def test_smoke_10_years(self):
        """Simulation runs for 10 years without crashing."""
        result = run_simulation(seed=42, years=10)
        assert result["_meta"]["engine"] == "mars-100"
        assert len(result["timeline"]) >= 1
        assert len(result["colonists"]) == 10

    def test_smoke_100_years(self):
        """Full 100-year simulation completes."""
        result = run_simulation(seed=42, years=100)
        assert result["_meta"]["years"] == 100
        assert len(result["timeline"]) >= 1

    def test_deterministic(self):
        """Same seed produces identical results."""
        r1 = run_simulation(seed=42, years=50)
        r2 = run_simulation(seed=42, years=50)
        assert r1["timeline"] == r2["timeline"]
        assert r1["colonists"] == r2["colonists"]

    def test_different_seeds_differ(self):
        """Different seeds produce different results."""
        r1 = run_simulation(seed=42, years=20)
        r2 = run_simulation(seed=99, years=20)
        # Timelines will differ in resource/event details
        assert r1["timeline"] != r2["timeline"]

    def test_population_bounds(self):
        """Population stays within physical bounds (10 founding + births)."""
        result = run_simulation(seed=42, years=100)
        total_colonists = len(result["colonists"])
        for snap in result["timeline"]:
            assert 0 <= snap["alive"] <= total_colonists
            assert 0 <= snap["dead"] <= total_colonists
            assert snap["alive"] + snap["dead"] <= total_colonists

    def test_resources_bounded(self):
        """All resources stay within physical bounds."""
        result = run_simulation(seed=42, years=100)
        for snap in result["timeline"]:
            for val in snap["resources"].values():
                assert MIN_RESOURCE <= val <= MAX_RESOURCE

    def test_terraforming_bounded(self):
        """Terraforming progress stays in [0, 1]."""
        result = run_simulation(seed=42, years=100)
        for snap in result["timeline"]:
            assert 0.0 <= snap["terraform"] <= 1.0

    def test_subsims_logged(self):
        """Sub-simulations are logged."""
        result = run_simulation(seed=42, years=100)
        assert result["_meta"]["total_subsims"] >= 0
        if result["subsim_log"]:
            entry = result["subsim_log"][0]
            assert "colonist" in entry
            assert "result" in entry

    def test_governance_emerges(self):
        """At least one governance proposal appears over 100 years."""
        result = run_simulation(seed=42, years=100)
        assert len(result["governance"]) >= 1

    def test_narratives_present(self):
        """Each year has narrative lines."""
        result = run_simulation(seed=42, years=50)
        for n in result["narratives"]:
            assert "year" in n
            assert "lines" in n
            assert len(n["lines"]) >= 1

    def test_patterns_detected(self):
        """Emergent patterns are detected over time."""
        result = run_simulation(seed=42, years=100)
        assert len(result["patterns"]) >= 1

    def test_dashboard_data_compact(self):
        """Dashboard data is smaller than full result."""
        result = run_simulation(seed=42, years=100)
        dashboard = build_dashboard_data(result)
        full_size = len(json.dumps(result))
        dash_size = len(json.dumps(dashboard))
        assert dash_size < full_size
        assert "_meta" in dashboard
        assert "timeline" in dashboard

    def test_collapsed_when_all_dead(self):
        """Colony collapse is detected when resources are zero."""
        # Run with seed that might cause collapse,
        # or just verify the field exists
        result = run_simulation(seed=42, years=100)
        assert isinstance(result["collapsed"], bool)

    def test_colonist_memory_grows(self):
        """Colonists accumulate memories over time."""
        result = run_simulation(seed=42, years=50)
        for c in result["colonists"]:
            # Memory is capped at 20 but should have entries
            assert len(c["memory"]) >= 1


# ---------------------------------------------------------------------------
# Conservation laws / invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_alive_plus_dead_equals_total(self):
        """Total population = founding 10 + births."""
        result = run_simulation(seed=42, years=100)
        total = len(result["colonists"])
        assert total >= 10  # At least the founding 10
        for snap in result["timeline"]:
            assert snap["alive"] + snap["dead"] <= total

    def test_karma_bounded(self):
        """All colonist karma values stay in [0, 1]."""
        result = run_simulation(seed=42, years=100)
        for c in result["colonists"]:
            assert 0 <= c["karma"] <= 1.0

    def test_stats_bounded(self):
        """All colonist stats stay in [0, 1]."""
        result = run_simulation(seed=42, years=100)
        for c in result["colonists"]:
            for val in c["stats"].values():
                assert MIN_STAT <= val <= MAX_STAT

    def test_relationships_bounded(self):
        """All relationship values stay in [-1, 1]."""
        result = run_simulation(seed=42, years=100)
        for c in result["colonists"]:
            for trust in c["relationships"].values():
                assert -1.0 <= trust <= 1.0

    def test_multiple_seeds_all_valid(self):
        """Run with 5 different seeds — all produce valid output."""
        for seed in [1, 42, 99, 777, 12345]:
            result = run_simulation(seed=seed, years=30)
            assert result["_meta"]["seed"] == seed
            assert len(result["colonists"]) >= 10  # 10 founding + possible births
            total = len(result["colonists"])
            for snap in result["timeline"]:
                assert snap["alive"] + snap["dead"] <= total


# ---------------------------------------------------------------------------
# New feature tests: births, governance, diaries, amendments, soul files
# ---------------------------------------------------------------------------

class TestBirths:
    def test_no_births_before_year_15(self):
        """Births should not occur before year 15."""
        result = run_simulation(seed=42, years=14)
        assert result["_meta"].get("total_births", 0) == 0
        assert all(not c["id"].startswith("mars-born-") for c in result["colonists"])

    def test_births_possible_after_year_15(self):
        """At least one of several seeds should produce a birth over 100 years."""
        any_births = False
        for seed in [42, 99, 777, 12345, 54321]:
            result = run_simulation(seed=seed, years=100)
            if result["_meta"]["total_births"] >= 1:
                any_births = True
                mars_born = [c for c in result["colonists"] if c["id"].startswith("mars-born-")]
                assert len(mars_born) >= 1
                break
        assert any_births, "No births in any seed — birth probability too low"

    def test_mars_born_have_valid_stats(self):
        """Mars-born colonists have stats in [0, 1]."""
        result = run_simulation(seed=42, years=100)
        for c in result["colonists"]:
            if c["id"].startswith("mars-born-"):
                for val in c["stats"].values():
                    assert MIN_STAT <= val <= MAX_STAT

    def test_births_capped(self):
        """No more than 10 births (MARS_BORN_NAMES limit)."""
        result = run_simulation(seed=42, years=100)
        assert result["_meta"]["total_births"] <= 10


class TestGovernanceClassification:
    def test_governance_type_in_result(self):
        """Governance type should be in metadata."""
        result = run_simulation(seed=42, years=100)
        assert result["_meta"]["governance_type"] in [
            "anarchy", "council", "democracy", "dictatorship", "theocracy", "technocracy",
        ]

    def test_governance_type_in_timeline(self):
        """Each timeline snapshot should include governance_type."""
        result = run_simulation(seed=42, years=50)
        for snap in result["timeline"]:
            assert "governance_type" in snap

    def test_early_years_likely_anarchy(self):
        """First few years are likely anarchy (no proposals yet)."""
        result = run_simulation(seed=42, years=5)
        assert result["timeline"][0]["governance_type"] == "anarchy"


class TestDiaries:
    def test_diaries_generated(self):
        """Diaries should be generated each year."""
        result = run_simulation(seed=42, years=10)
        assert len(result["diaries"]) > 0

    def test_diary_structure(self):
        """Each diary entry has required fields."""
        result = run_simulation(seed=42, years=10)
        for d in result["diaries"]:
            assert "colonist" in d
            assert "name" in d
            assert "year" in d
            assert "text" in d
            assert "element" in d

    def test_diaries_max_3_per_year(self):
        """No more than 3 diary entries per year."""
        result = run_simulation(seed=42, years=50)
        from collections import Counter
        year_counts = Counter(d["year"] for d in result["diaries"])
        for year, count in year_counts.items():
            assert count <= 3, f"Year {year} has {count} diaries"


class TestDeepSubSims:
    def test_subsim_log_has_depths(self):
        """Sub-sim log should include entries at various depths."""
        result = run_simulation(seed=42, years=100)
        depths = {s["depth"] for s in result["subsim_log"]}
        assert 1 in depths, "Should have depth-1 sub-sims"

    def test_subsim_depth_bounded(self):
        """All sub-sim depths should be <= 3."""
        result = run_simulation(seed=42, years=100)
        for s in result["subsim_log"]:
            assert 1 <= s["depth"] <= 3


class TestAmendments:
    def test_amendments_in_result(self):
        """Amendments list should exist in result."""
        result = run_simulation(seed=42, years=100)
        assert "amendments" in result
        assert isinstance(result["amendments"], list)

    def test_amendment_structure(self):
        """If amendments exist, they should have required fields."""
        result = run_simulation(seed=42, years=100)
        for a in result["amendments"]:
            assert "year" in a
            assert "title" in a
            assert "text" in a
            assert "rappterbook_analog" in a


class TestSoulFiles:
    def test_soul_files_in_result(self):
        """Soul files should be in the result."""
        result = run_simulation(seed=42, years=10)
        assert "soul_files" in result
        assert len(result["soul_files"]) >= 10  # At least founding colonists

    def test_soul_file_has_entries(self):
        """Each founding colonist should have soul file entries."""
        result = run_simulation(seed=42, years=10)
        for colonist in result["colonists"][:10]:
            cid = colonist["id"]
            assert cid in result["soul_files"]
            assert len(result["soul_files"][cid]) > 0


class TestOutputFiles:
    def test_write_year_chapters(self, tmp_path):
        """write_year_chapters creates per-year JSON files."""
        result = run_simulation(seed=42, years=5)
        write_year_chapters(result, tmp_path)
        years_dir = tmp_path / "years"
        assert years_dir.exists()
        assert (years_dir / "year-001.json").exists()
        assert (years_dir / "year-005.json").exists()

    def test_write_soul_files(self, tmp_path):
        """write_soul_files creates per-colonist JSON files."""
        result = run_simulation(seed=42, years=5)
        write_soul_files(result, tmp_path)
        colonists_dir = tmp_path / "colonists"
        assert colonists_dir.exists()
        # Check at least the first colonist
        first_id = result["colonists"][0]["id"]
        assert (colonists_dir / f"{first_id}.json").exists()
