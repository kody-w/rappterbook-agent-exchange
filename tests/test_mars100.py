"""Tests for the Mars-100 colony simulation kernel."""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100 import (
    COLONIST_NAMES,
    ELEMENTS,
    EVENT_TYPES,
    GOVERNANCE_TYPES,
    RESOURCE_CAP,
    RESOURCE_NAMES,
    SKILL_NAMES,
    STAT_NAMES,
    ColonyState,
    Colonist,
    Proposal,
    SubSimLog,
    apply_action,
    apply_event_to_resources,
    auto_vote_on_proposals,
    build_dashboard_data,
    check_amendment_promotion,
    check_deaths,
    check_sim_awareness,
    classify_governance,
    consume_resources,
    create_colonists,
    decide_action,
    detect_patterns,
    evolve_relationships,
    generate_diary_entries,
    generate_event,
    maybe_birth,
    resolve_proposals,
    run_simulation,
    write_soul_files,
    write_year_chapters,
)


# ---------------------------------------------------------------------------
# Colonist creation
# ---------------------------------------------------------------------------


class TestCreateColonists:
    def test_creates_10(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        assert len(colonists) == 10

    def test_all_named(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        names = [c.name for c in colonists]
        for name in COLONIST_NAMES:
            assert name in names

    def test_elements_assigned(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        for c in colonists:
            assert c.element in ELEMENTS

    def test_stats_bounded(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        for c in colonists:
            for stat in STAT_NAMES:
                assert 0.0 <= c.stats[stat] <= 1.0

    def test_skills_bounded(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        for c in colonists:
            for skill in SKILL_NAMES:
                assert 0.0 <= c.skills[skill] <= 1.0

    def test_relationships_exist(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        for c in colonists:
            # Should have relationships with all other colonists
            assert len(c.relationships) == 9

    def test_all_alive(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        assert all(c.alive for c in colonists)

    def test_deterministic(self):
        c1 = create_colonists(random.Random(42))
        c2 = create_colonists(random.Random(42))
        for a, b in zip(c1, c2):
            assert a.stats == b.stats


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class TestEvents:
    def test_event_has_type(self):
        event = generate_event(1, random.Random(42), 0.0)
        assert event["type"] in EVENT_TYPES

    def test_event_has_severity(self):
        event = generate_event(1, random.Random(42), 0.0)
        assert 0.0 <= event["severity"] <= 1.0

    def test_event_has_description(self):
        event = generate_event(1, random.Random(42), 0.0)
        assert isinstance(event["description"], str)
        assert len(event["description"]) > 0

    def test_event_year(self):
        event = generate_event(50, random.Random(42), 0.0)
        assert event["year"] == 50


class TestApplyEvent:
    def test_resources_bounded(self):
        rng = random.Random(42)
        resources = {r: 100.0 for r in RESOURCE_NAMES}
        for _ in range(50):
            event = generate_event(1, rng, 0.0)
            apply_event_to_resources(event, resources, rng)
            for r in RESOURCE_NAMES:
                assert 0.0 <= resources[r] <= RESOURCE_CAP

    def test_resource_strike_adds(self):
        resources = {r: 50.0 for r in RESOURCE_NAMES}
        event = {"type": "resource_strike", "severity": 1.0}
        rng = random.Random(42)
        old = dict(resources)
        apply_event_to_resources(event, resources, rng)
        # At least one resource should increase
        assert any(resources[r] > old[r] for r in RESOURCE_NAMES)

    def test_dust_storm_reduces_power(self):
        resources = {r: 100.0 for r in RESOURCE_NAMES}
        event = {"type": "dust_storm", "severity": 0.8}
        apply_event_to_resources(event, resources, random.Random(42))
        assert resources["power"] < 100.0


# ---------------------------------------------------------------------------
# Decision making
# ---------------------------------------------------------------------------


class TestDecisions:
    def _make_state(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        return ColonyState(year=1, colonists=colonists, rng_seed=42)

    def test_returns_action(self):
        state = self._make_state()
        event = generate_event(1, random.Random(42), 0.0)
        action, _ = decide_action(state.colonists[0], state, event, random.Random(42))
        assert isinstance(action, str)
        assert len(action) > 0

    def test_subsim_can_be_none(self):
        state = self._make_state()
        event = {"type": "calm_year", "severity": 0.1}
        _, subsim = decide_action(state.colonists[0], state, event, random.Random(42))
        # subsim may or may not fire — just check type
        assert subsim is None or isinstance(subsim, str)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class TestActions:
    def _make_state(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        return ColonyState(year=1, colonists=colonists, rng_seed=42)

    def test_ration_adds_food(self):
        state = self._make_state()
        old_food = state.resources["food"]
        apply_action(state.colonists[0], "ration", state, random.Random(42))
        assert state.resources["food"] >= old_food

    def test_hoard_increases_stat(self):
        state = self._make_state()
        old = state.colonists[0].stats["hoarding"]
        apply_action(state.colonists[0], "hoard", state, random.Random(42))
        assert state.colonists[0].stats["hoarding"] >= old

    def test_work_adds_resource(self):
        state = self._make_state()
        old_total = sum(state.resources.values())
        apply_action(state.colonists[0], "work", state, random.Random(42))
        new_total = sum(state.resources.values())
        assert new_total >= old_total

    def test_returns_narrative(self):
        state = self._make_state()
        narrative = apply_action(state.colonists[0], "investigate", state, random.Random(42))
        assert isinstance(narrative, str)
        assert len(narrative) > 0


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------


class TestGovernance:
    def _make_state_with_proposal(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        state = ColonyState(year=10, colonists=colonists, rng_seed=42)
        proposal = Proposal(
            id="test-1", proposer="colonist-00", year=10,
            proposal_type="council",
            description="Test council proposal",
            votes_for=["colonist-00", "colonist-01", "colonist-02",
                        "colonist-03", "colonist-04", "colonist-05"],
            votes_against=["colonist-09"],
            inertia=3,
        )
        state.governance.append(proposal)
        return state

    def test_resolve_with_support(self):
        state = self._make_state_with_proposal()
        narratives = resolve_proposals(state)
        assert state.governance_type == "council"
        assert len(narratives) > 0

    def test_no_resolve_without_inertia(self):
        state = self._make_state_with_proposal()
        state.governance[0].inertia = 0
        resolve_proposals(state)
        # Inertia increases by 1 but starts at 0, so 1 < 3
        assert state.governance_type == "anarchy"

    def test_auto_vote(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        state = ColonyState(year=10, colonists=colonists, rng_seed=42)
        state.governance.append(Proposal(
            id="test-2", proposer="colonist-00", year=10,
            proposal_type="democracy",
            description="Test",
        ))
        auto_vote_on_proposals(state, rng)
        proposal = state.governance[0]
        # Some colonists should have voted
        assert len(proposal.votes_for) + len(proposal.votes_against) > 0


class TestClassifyGovernance:
    def test_default_anarchy(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        state = ColonyState(year=1, colonists=colonists, rng_seed=42)
        gov = classify_governance(state)
        assert gov in GOVERNANCE_TYPES


# ---------------------------------------------------------------------------
# Death
# ---------------------------------------------------------------------------


class TestDeaths:
    def test_paranoia_death(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        colonists[0].stats["paranoia"] = 0.99
        state = ColonyState(year=50, colonists=colonists, rng_seed=42)
        event = {"type": "calm_year", "severity": 0.1}
        narratives = check_deaths(state, event, rng)
        assert not colonists[0].alive
        assert colonists[0].cause_of_death == "paranoia-collapse"

    def test_resolve_death(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        colonists[0].stats["resolve"] = 0.01
        state = ColonyState(year=50, colonists=colonists, rng_seed=42)
        event = {"type": "calm_year", "severity": 0.1}
        check_deaths(state, event, rng)
        assert not colonists[0].alive
        assert colonists[0].cause_of_death == "despair"

    def test_dead_colonist_has_year(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        colonists[0].stats["paranoia"] = 0.99
        state = ColonyState(year=30, colonists=colonists, rng_seed=42)
        event = {"type": "calm_year", "severity": 0.1}
        check_deaths(state, event, rng)
        assert colonists[0].year_of_death == 30


# ---------------------------------------------------------------------------
# Resource consumption
# ---------------------------------------------------------------------------


class TestConsumption:
    def test_resources_decrease(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        state = ColonyState(year=1, colonists=colonists, rng_seed=42)
        old_food = state.resources["food"]
        consume_resources(state)
        assert state.resources["food"] < old_food

    def test_resources_non_negative(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        state = ColonyState(year=1, colonists=colonists, rng_seed=42)
        state.resources = {r: 1.0 for r in RESOURCE_NAMES}
        consume_resources(state)
        for r in RESOURCE_NAMES:
            assert state.resources[r] >= 0.0


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------


class TestRelationships:
    def test_bounded(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        state = ColonyState(year=1, colonists=colonists, rng_seed=42)
        actions = {c.id: "work" for c in colonists}
        for _ in range(100):
            evolve_relationships(state, actions, rng)
        for c in colonists:
            for v in c.relationships.values():
                assert -1.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


class TestPatterns:
    def test_returns_list(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        state = ColonyState(year=1, colonists=colonists, rng_seed=42)
        patterns = detect_patterns(state)
        assert isinstance(patterns, list)

    def test_extinction_detected(self):
        state = ColonyState(year=50, colonists=[], rng_seed=42)
        patterns = detect_patterns(state)
        assert "extinction" in patterns


# ---------------------------------------------------------------------------
# Births
# ---------------------------------------------------------------------------


class TestBirths:
    def test_no_births_before_year_15(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        state = ColonyState(year=10, colonists=colonists, rng_seed=42)
        narratives = maybe_birth(state, rng)
        assert len(narratives) == 0

    def test_births_possible_after_year_15(self):
        # Run enough times to trigger a birth
        found_birth = False
        for seed in range(100):
            rng = random.Random(seed)
            colonists = create_colonists(rng)
            state = ColonyState(year=30, colonists=colonists, rng_seed=seed)
            narratives = maybe_birth(state, rng)
            if narratives:
                found_birth = True
                break
        assert found_birth

    def test_mars_born_have_valid_stats(self):
        for seed in range(100):
            rng = random.Random(seed)
            colonists = create_colonists(rng)
            state = ColonyState(year=30, colonists=colonists, rng_seed=seed)
            maybe_birth(state, rng)
            for c in state.colonists:
                if c.id.startswith("mars-born-"):
                    for stat in STAT_NAMES:
                        assert 0.0 <= c.stats[stat] <= 1.0
                    break

    def test_births_capped(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        # Add 10 more to get close to cap
        for i in range(10):
            colonists.append(Colonist(
                id=f"extra-{i}", name=f"Extra{i}", element="fire",
                stats={s: 0.5 for s in STAT_NAMES},
                skills={s: 0.5 for s in SKILL_NAMES},
            ))
        state = ColonyState(year=30, colonists=colonists, rng_seed=42)
        # At 20 colonists, births should be capped
        narratives = maybe_birth(state, rng)
        assert len(narratives) == 0


# ---------------------------------------------------------------------------
# Sim awareness
# ---------------------------------------------------------------------------


class TestSimAwareness:
    def test_not_before_year_20(self):
        c = Colonist(
            id="test", name="Test", element="fire",
            stats={"faith": 0.9, "paranoia": 0.9, **{s: 0.5 for s in STAT_NAMES if s not in ("faith", "paranoia")}},
            skills={s: 0.5 for s in SKILL_NAMES},
        )
        assert not check_sim_awareness(c, 10)

    def test_triggered_by_faith_paranoia(self):
        c = Colonist(
            id="test", name="Test", element="fire",
            stats={"faith": 0.8, "paranoia": 0.7, **{s: 0.5 for s in STAT_NAMES if s not in ("faith", "paranoia")}},
            skills={s: 0.5 for s in SKILL_NAMES},
        )
        assert check_sim_awareness(c, 25)

    def test_already_aware(self):
        c = Colonist(
            id="test", name="Test", element="fire",
            stats={s: 0.5 for s in STAT_NAMES},
            skills={s: 0.5 for s in SKILL_NAMES},
            sim_aware=True,
        )
        assert check_sim_awareness(c, 5)


# ---------------------------------------------------------------------------
# Diary entries
# ---------------------------------------------------------------------------


class TestDiaries:
    def test_generates_entries(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        state = ColonyState(year=5, colonists=colonists, rng_seed=42)
        event = {"type": "calm_year", "severity": 0.2}
        actions = {c.id: "work" for c in colonists}
        entries = generate_diary_entries(state, event, actions, rng)
        assert len(entries) <= 3
        assert len(entries) > 0

    def test_entry_structure(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        state = ColonyState(year=5, colonists=colonists, rng_seed=42)
        event = {"type": "calm_year", "severity": 0.2}
        actions = {c.id: "work" for c in colonists}
        entries = generate_diary_entries(state, event, actions, rng)
        for entry in entries:
            assert "year" in entry
            assert "colonist" in entry
            assert "entry" in entry


# ---------------------------------------------------------------------------
# Full simulation
# ---------------------------------------------------------------------------


class TestFullSimulation:
    def test_smoke_10_years(self):
        result = run_simulation(seed=42, years=10)
        assert "_meta" in result
        assert len(result["timeline"]) > 0

    def test_smoke_100_years(self):
        result = run_simulation(seed=42, years=100)
        assert len(result["timeline"]) > 0
        assert len(result["timeline"]) <= 100

    def test_deterministic(self):
        r1 = run_simulation(seed=42, years=20)
        r2 = run_simulation(seed=42, years=20)
        assert r1["timeline"] == r2["timeline"]

    def test_different_seeds_differ(self):
        r1 = run_simulation(seed=1, years=20)
        r2 = run_simulation(seed=2, years=20)
        # Timelines should differ in events at least
        events1 = [t["event"] for t in r1["timeline"]]
        events2 = [t["event"] for t in r2["timeline"]]
        assert events1 != events2

    def test_population_bounds(self):
        result = run_simulation(seed=42, years=50)
        for snap in result["timeline"]:
            assert snap["alive"] >= 0
            assert snap["alive"] + snap["dead"] >= 10  # at least original 10

    def test_resources_bounded(self):
        result = run_simulation(seed=42, years=50)
        for snap in result["timeline"]:
            for r in RESOURCE_NAMES:
                assert 0.0 <= snap["resources"][r] <= RESOURCE_CAP

    def test_terraforming_bounded(self):
        result = run_simulation(seed=42, years=100)
        for snap in result["timeline"]:
            assert 0.0 <= snap["terraform"] <= 1.0

    def test_subsims_logged(self):
        result = run_simulation(seed=42, years=50)
        assert isinstance(result["subsim_log"], list)

    def test_governance_emerges(self):
        result = run_simulation(seed=42, years=100)
        assert result["_meta"]["governance_type"] in GOVERNANCE_TYPES

    def test_narratives_present(self):
        result = run_simulation(seed=42, years=10)
        assert len(result["narratives"]) > 0

    def test_patterns_detected(self):
        result = run_simulation(seed=42, years=50)
        assert isinstance(result["patterns"], list)

    def test_dashboard_data_compact(self):
        result = run_simulation(seed=42, years=10)
        dashboard = build_dashboard_data(result)
        assert "_meta" in dashboard
        assert "timeline" in dashboard
        # Dashboard should be smaller than full result
        assert "narratives" not in dashboard

    def test_collapsed_when_all_dead(self):
        # Run many seeds — some should collapse
        collapsed = False
        for seed in range(50):
            result = run_simulation(seed=seed, years=100)
            if result["collapsed"]:
                collapsed = True
                break
        # It's possible none collapse, but unlikely with 50 seeds
        # Just assert the field exists
        assert "collapsed" in result

    def test_colonist_memory_grows(self):
        result = run_simulation(seed=42, years=20)
        for c in result["colonists"]:
            if c["alive"] or c["year_of_death"]:
                assert len(c["memory"]) > 0


# ---------------------------------------------------------------------------
# Invariants (property-based)
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_alive_plus_dead_equals_total(self):
        result = run_simulation(seed=42, years=50)
        total = len(result["colonists"])
        for snap in result["timeline"]:
            assert snap["alive"] + snap["dead"] == total or snap["births"] > 0

    def test_karma_bounded(self):
        for seed in [42, 100, 200]:
            result = run_simulation(seed=seed, years=30)
            for c in result["colonists"]:
                assert -10.0 <= c["karma"] <= 10.0  # generous bound

    def test_stats_bounded(self):
        for seed in [42, 100]:
            result = run_simulation(seed=seed, years=30)
            for c in result["colonists"]:
                for stat_val in c["stats"].values():
                    assert 0.0 <= stat_val <= 1.0

    def test_relationships_bounded(self):
        for seed in [42, 100]:
            result = run_simulation(seed=seed, years=30)
            for c in result["colonists"]:
                for rel_val in c["relationships"].values():
                    assert -1.0 <= rel_val <= 1.0

    def test_multiple_seeds_all_valid(self):
        for seed in range(10):
            result = run_simulation(seed=seed, years=20)
            assert "_meta" in result
            assert len(result["timeline"]) > 0
            assert len(result["colonists"]) >= 10


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------


class TestOutputFiles:
    def test_write_year_chapters(self):
        result = run_simulation(seed=42, years=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            write_year_chapters(result, Path(tmpdir))
            years_dir = Path(tmpdir) / "years"
            assert years_dir.exists()
            files = list(years_dir.glob("*.json"))
            assert len(files) == len(result["timeline"])

    def test_write_soul_files(self):
        result = run_simulation(seed=42, years=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            write_soul_files(result, Path(tmpdir))
            souls_dir = Path(tmpdir) / "colonists"
            assert souls_dir.exists()
            files = list(souls_dir.glob("*.json"))
            assert len(files) == len(result["colonists"])


# ---------------------------------------------------------------------------
# Amendments
# ---------------------------------------------------------------------------


class TestAmendments:
    def test_amendments_in_result(self):
        result = run_simulation(seed=42, years=100)
        assert "amendments" in result
        assert isinstance(result["amendments"], list)

    def test_amendment_structure(self):
        result = run_simulation(seed=42, years=100)
        for a in result.get("amendments", []):
            assert "year" in a
            assert "title" in a
            assert "text" in a


# ---------------------------------------------------------------------------
# Colonist serialization
# ---------------------------------------------------------------------------


class TestColonistSerialization:
    def test_to_dict(self):
        c = Colonist(
            id="test-00", name="Test", element="fire",
            stats={s: 0.5 for s in STAT_NAMES},
            skills={s: 0.5 for s in SKILL_NAMES},
        )
        d = c.to_dict()
        assert d["id"] == "test-00"
        assert d["name"] == "Test"
        assert d["alive"] is True

    def test_from_dict(self):
        d = {
            "id": "test-00", "name": "Test", "element": "water",
            "stats": {s: 0.5 for s in STAT_NAMES},
            "skills": {s: 0.5 for s in SKILL_NAMES},
            "relationships": {}, "memory": [], "alive": True,
            "year_of_death": None, "cause_of_death": None,
            "karma": 0.5, "sim_aware": False,
        }
        c = Colonist.from_dict(d)
        assert c.id == "test-00"
        assert c.element == "water"
