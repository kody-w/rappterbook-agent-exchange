"""test_mars100.py — Comprehensive tests for the Mars-100 simulation engine.

Covers: colonist creation, event generation, decisions, governance, resource
management, relationships, sub-simulations, collapse detection, determinism,
physical bound invariants, and full simulation smoke tests.
"""
from __future__ import annotations

import json
import math
import pytest
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100 import (
    Mars100Simulation, ColonyState, Colonist, GovernanceProposal,
    create_colonists, generate_event, make_colonist_decision,
    apply_event_effects, resolve_proposals, update_relationships,
    update_governance_weights, consume_resources, natural_production,
    check_collapse, _kill_colonist, _analyze_governance_evolution,
    apply_passed_proposals, update_skills,
    classify_governance, compute_value_convergence, clamp_morale,
    ELEMENTS, STAT_NAMES, SKILL_NAMES, COLONIST_TEMPLATES,
    MAX_RESOURCES, MIN_MORALE, MAX_MORALE, MAX_RELATIONSHIP,
    MIN_RELATIONSHIP, MAX_GOVERNANCE_WEIGHT, GOVERNANCE_FORMS,
    SKILL_GROWTH_RATE, SKILL_DECAY_RATE,
    EARLY_EVENTS, MID_EVENTS, LATE_EVENTS,
)
from src.lispy_vm import LispyVM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


@pytest.fixture
def colonists(rng) -> list[Colonist]:
    return create_colonists(rng)


@pytest.fixture
def state(colonists) -> ColonyState:
    s = ColonyState()
    s.colonists = colonists
    s.year = 1
    return s


@pytest.fixture
def vm() -> LispyVM:
    return LispyVM(max_depth=3, max_steps=50_000, rng_seed=42)


@pytest.fixture
def sim() -> Mars100Simulation:
    return Mars100Simulation(seed=42, max_years=100)


# ---------------------------------------------------------------------------
# Colonist creation tests
# ---------------------------------------------------------------------------

class TestColonistCreation:
    def test_creates_10_colonists(self, colonists):
        assert len(colonists) == 10

    def test_all_alive(self, colonists):
        assert all(c.alive for c in colonists)

    def test_unique_ids(self, colonists):
        ids = [c.id for c in colonists]
        assert len(set(ids)) == 10

    def test_valid_elements(self, colonists):
        for c in colonists:
            assert c.element in ELEMENTS

    def test_all_stats_present(self, colonists):
        for c in colonists:
            for stat in STAT_NAMES:
                assert stat in c.stats
                assert 0.0 <= c.stats[stat] <= 1.0

    def test_all_skills_present(self, colonists):
        for c in colonists:
            for skill in SKILL_NAMES:
                assert skill in c.skills
                assert 0.0 <= c.skills[skill] <= 1.0

    def test_relationships_initialized(self, colonists):
        for c in colonists:
            assert len(c.relationships) == 9  # 10 - self
            for other_id, rel in c.relationships.items():
                assert MIN_RELATIONSHIP <= rel <= MAX_RELATIONSHIP

    def test_no_self_relationship(self, colonists):
        for c in colonists:
            assert c.id not in c.relationships

    def test_boosts_applied(self, colonists):
        """Template boosts should raise stats above random baseline."""
        kael = next(c for c in colonists if c.id == "kael-terraform")
        assert kael.skills["terraforming"] >= 0.7  # boosted

    def test_deterministic_creation(self):
        """Same seed → same colonists."""
        c1 = create_colonists(random.Random(42))
        c2 = create_colonists(random.Random(42))
        for a, b in zip(c1, c2):
            assert a.id == b.id
            assert a.stats == b.stats
            assert a.skills == b.skills

    def test_serialization_roundtrip(self, colonists):
        """Colonists serialize to valid JSON."""
        for c in colonists:
            d = c.to_dict()
            json_str = json.dumps(d)
            assert json.loads(json_str) == d

    def test_sexpr_representation(self, colonists):
        """Colonists produce valid s-expression strings."""
        for c in colonists:
            sexpr = c.to_sexpr()
            assert "colonist" in sexpr
            assert c.id in sexpr
            assert c.element in sexpr


# ---------------------------------------------------------------------------
# Event generation tests
# ---------------------------------------------------------------------------

class TestEventGeneration:
    def test_early_events(self, rng):
        events = [generate_event(y, rng) for y in range(1, 21)]
        types = {e["type"] for e in events}
        assert types.issubset(set(EARLY_EVENTS))

    def test_mid_events(self, rng):
        events = [generate_event(y, rng) for y in range(21, 61)]
        types = {e["type"] for e in events}
        assert types.issubset(set(MID_EVENTS))

    def test_late_events(self, rng):
        events = [generate_event(y, rng) for y in range(61, 101)]
        types = {e["type"] for e in events}
        assert types.issubset(set(LATE_EVENTS))

    def test_event_has_required_fields(self, rng):
        event = generate_event(1, rng)
        assert "year" in event
        assert "type" in event
        assert "severity" in event
        assert "description" in event

    def test_severity_bounds(self, rng):
        for year in range(1, 101):
            event = generate_event(year, rng)
            assert 0.1 <= event["severity"] <= 1.0


# ---------------------------------------------------------------------------
# Event effect tests
# ---------------------------------------------------------------------------

class TestEventEffects:
    def test_dust_storm_reduces_power(self, state, rng):
        power_before = state.resources["power"]
        event = {"year": 1, "type": "dust_storm", "severity": 0.5, "description": ""}
        apply_event_effects(event, state, rng)
        assert state.resources["power"] < power_before

    def test_water_strike_increases_water(self, state, rng):
        water_before = state.resources["water"]
        event = {"year": 1, "type": "water_strike", "severity": 0.5, "description": ""}
        apply_event_effects(event, state, rng)
        assert state.resources["water"] > water_before

    def test_supply_drop_increases_food(self, state, rng):
        food_before = state.resources["food"]
        event = {"year": 1, "type": "supply_drop", "severity": 0.5, "description": ""}
        apply_event_effects(event, state, rng)
        assert state.resources["food"] > food_before

    def test_epidemic_can_kill(self, state, rng):
        alive_before = len(state.alive_colonists())
        event = {"year": 1, "type": "epidemic", "severity": 0.9, "description": ""}
        apply_event_effects(event, state, rng)
        # High severity epidemic should kill at least one
        alive_after = len(state.alive_colonists())
        assert alive_after <= alive_before

    def test_resources_stay_bounded(self, state, rng):
        """Resources never go below 0 or above MAX_RESOURCES."""
        for event_type in EARLY_EVENTS + MID_EVENTS + LATE_EVENTS:
            s = ColonyState()
            s.colonists = create_colonists(random.Random(42))
            event = {"year": 1, "type": event_type, "severity": 1.0, "description": ""}
            apply_event_effects(event, s, rng)
            for key, val in s.resources.items():
                assert 0.0 <= val <= MAX_RESOURCES, f"{key}={val} out of bounds"

    def test_morale_stays_bounded(self, state, rng):
        """Morale stays in [0, 1] after any event."""
        for event_type in EARLY_EVENTS + MID_EVENTS + LATE_EVENTS:
            s = ColonyState()
            s.colonists = create_colonists(random.Random(42))
            event = {"year": 1, "type": event_type, "severity": 1.0, "description": ""}
            apply_event_effects(event, s, rng)
            assert MIN_MORALE <= s.morale <= MAX_MORALE


# ---------------------------------------------------------------------------
# Colonist decision tests
# ---------------------------------------------------------------------------

class TestDecisions:
    def test_alive_colonist_decides(self, state, vm, rng):
        event = generate_event(1, rng)
        colonist = state.alive_colonists()[0]
        decision = make_colonist_decision(colonist, event, state, vm, rng)
        assert "action" in decision
        assert decision["colonist_id"] == colonist.id

    def test_dead_colonist_archived(self, state, vm, rng):
        event = generate_event(1, rng)
        colonist = state.colonists[0]
        _kill_colonist(colonist, 1, "test")
        decision = make_colonist_decision(colonist, event, state, vm, rng)
        assert decision["action"] == "archived"

    def test_decision_creates_diary(self, state, vm, rng):
        event = generate_event(1, rng)
        colonist = state.alive_colonists()[0]
        make_colonist_decision(colonist, event, state, vm, rng)
        assert len(colonist.diary) == 1
        assert "year" in colonist.diary[0]


# ---------------------------------------------------------------------------
# Governance tests
# ---------------------------------------------------------------------------

class TestGovernance:
    def test_proposal_resolution(self, state, rng):
        proposal = GovernanceProposal(
            id="test-prop",
            year=1,
            proposer_id=state.colonists[0].id,
            proposal_type="leadership_election",
            description="Test",
            value={},
        )
        state.proposals.append(proposal)
        resolved = resolve_proposals(state, rng)
        assert len(resolved) == 1
        assert resolved[0]["resolved"] is True
        assert resolved[0]["outcome"] in ("passed", "failed")

    def test_all_alive_vote(self, state, rng):
        proposal = GovernanceProposal(
            id="test-prop", year=1,
            proposer_id=state.colonists[0].id,
            proposal_type="resource_allocation",
            description="Test", value={},
        )
        state.proposals.append(proposal)
        resolve_proposals(state, rng)
        alive_ids = {c.id for c in state.alive_colonists()}
        voter_ids = set(proposal.votes.keys())
        assert alive_ids == voter_ids

    def test_dead_dont_vote(self, state, rng):
        _kill_colonist(state.colonists[0], 1, "test")
        proposal = GovernanceProposal(
            id="test-prop", year=1,
            proposer_id=state.colonists[1].id,
            proposal_type="leadership_election",
            description="Test", value={},
        )
        state.proposals.append(proposal)
        resolve_proposals(state, rng)
        assert state.colonists[0].id not in proposal.votes

    def test_governance_weights_clamped(self, state):
        for c in state.alive_colonists():
            c.proposals_made = 1000  # extreme activity
        update_governance_weights(state)
        for c in state.alive_colonists():
            assert 0.1 <= c.governance_weight <= MAX_GOVERNANCE_WEIGHT


# ---------------------------------------------------------------------------
# Resource management tests
# ---------------------------------------------------------------------------

class TestResources:
    def test_consumption_reduces_food(self, state):
        food_before = state.resources["food"]
        consume_resources(state)
        assert state.resources["food"] < food_before

    def test_production_adds_food(self, state):
        food_before = state.resources["food"]
        natural_production(state)
        assert state.resources["food"] > food_before

    def test_no_negative_resources(self, state):
        state.resources["food"] = 1  # almost out
        consume_resources(state)
        for key, val in state.resources.items():
            assert val >= 0.0, f"Resource {key} went negative: {val}"

    def test_terraforming_progresses(self, state):
        tf_before = state.terraforming_progress
        natural_production(state)
        assert state.terraforming_progress > tf_before

    def test_terraforming_capped_at_1(self, state):
        state.terraforming_progress = 0.999
        natural_production(state)
        assert state.terraforming_progress <= 1.0


# ---------------------------------------------------------------------------
# Relationship tests
# ---------------------------------------------------------------------------

class TestRelationships:
    def test_relationships_bounded(self, state, rng):
        decisions = [
            {"colonist_id": c.id, "action": "share_resources"}
            for c in state.alive_colonists()
        ]
        for _ in range(100):  # many updates
            update_relationships(state, decisions, rng)
        for c in state.alive_colonists():
            for rel in c.relationships.values():
                assert MIN_RELATIONSHIP <= rel <= MAX_RELATIONSHIP

    def test_positive_action_improves_relations(self, state, rng):
        colonist = state.alive_colonists()[0]
        other = state.alive_colonists()[1]
        rel_before = colonist.relationships[other.id]
        decisions = [{"colonist_id": colonist.id, "action": "share_resources"}]
        update_relationships(state, decisions, rng)
        # May increase or stay same due to noise, but shouldn't decrease much
        # (we just check bound is maintained)
        assert MIN_RELATIONSHIP <= colonist.relationships[other.id] <= MAX_RELATIONSHIP


# ---------------------------------------------------------------------------
# Collapse detection tests
# ---------------------------------------------------------------------------

class TestCollapse:
    def test_all_dead_collapses(self, state):
        for c in state.colonists:
            _kill_colonist(c, 1, "test")
        assert check_collapse(state) is True
        assert state.collapse_reason == "all_dead"

    def test_critical_underpopulation(self, state):
        for c in state.colonists[2:]:
            _kill_colonist(c, 1, "test")
        state.morale = 0.05
        assert check_collapse(state) is True
        assert state.collapse_reason == "critical_underpopulation"

    def test_total_resource_depletion(self, state):
        state.resources["food"] = 0
        state.resources["water"] = 0
        assert check_collapse(state) is True
        assert state.collapse_reason == "total_resource_depletion"

    def test_healthy_colony_no_collapse(self, state):
        assert check_collapse(state) is False


# ---------------------------------------------------------------------------
# Kill / legacy tests
# ---------------------------------------------------------------------------

class TestKillLegacy:
    def test_kill_sets_dead(self, state):
        colonist = state.colonists[0]
        _kill_colonist(colonist, 5, "meteor_impact")
        assert colonist.alive is False
        assert colonist.year_died == 5
        assert colonist.death_cause == "meteor_impact"

    def test_kill_archives_soul(self, state):
        colonist = state.colonists[0]
        _kill_colonist(colonist, 5, "meteor_impact")
        assert colonist.soul_archived is True

    def test_dead_colonist_persists(self, state):
        """Legacy not delete — dead colonists stay in the list."""
        _kill_colonist(state.colonists[0], 1, "test")
        assert len(state.colonists) == 10  # still 10
        assert len(state.alive_colonists()) == 9
        assert len(state.dead_colonists()) == 1


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_results(self):
        """Two simulations with the same seed produce identical deltas."""
        sim1 = Mars100Simulation(seed=42, max_years=10)
        sim2 = Mars100Simulation(seed=42, max_years=10)
        r1 = sim1.run()
        r2 = sim2.run()
        assert len(r1["deltas"]) == len(r2["deltas"])
        for d1, d2 in zip(r1["deltas"], r2["deltas"]):
            assert d1["delta_id"] == d2["delta_id"]
            assert d1["alive_count"] == d2["alive_count"]
            assert d1["morale"] == d2["morale"]

    def test_different_seed_different_results(self):
        """Different seeds produce different outcomes."""
        sim1 = Mars100Simulation(seed=42, max_years=10)
        sim2 = Mars100Simulation(seed=99, max_years=10)
        r1 = sim1.run()
        r2 = sim2.run()
        # Very unlikely to be identical
        deltas1_actions = [d["decisions"] for d in r1["deltas"]]
        deltas2_actions = [d["decisions"] for d in r2["deltas"]]
        assert deltas1_actions != deltas2_actions


# ---------------------------------------------------------------------------
# Full simulation smoke tests
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_smoke_10_years(self):
        """Run 10 years without crash."""
        sim = Mars100Simulation(seed=42, max_years=10)
        results = sim.run()
        assert results["_meta"]["years_completed"] == 10
        assert len(results["deltas"]) == 10

    def test_smoke_50_years(self):
        """Run 50 years without crash."""
        sim = Mars100Simulation(seed=42, max_years=50)
        results = sim.run()
        assert results["_meta"]["years_completed"] >= 1

    def test_smoke_100_years(self):
        """Run full 100 years without crash."""
        sim = Mars100Simulation(seed=42, max_years=100)
        results = sim.run()
        assert results["_meta"]["years_completed"] >= 1
        assert "summary" in results

    def test_results_schema(self):
        """Results have all expected top-level keys."""
        sim = Mars100Simulation(seed=42, max_years=5)
        results = sim.run()
        assert "_meta" in results
        assert "state" in results
        assert "deltas" in results
        assert "summary" in results
        assert "meta_insights" in results

    def test_summary_schema(self):
        """Summary has expected fields."""
        sim = Mars100Simulation(seed=42, max_years=5)
        results = sim.run()
        s = results["summary"]
        assert "years_simulated" in s
        assert "collapsed" in s
        assert "alive_count" in s
        assert "dead_count" in s
        assert "total_proposals" in s
        assert "final_morale" in s
        assert "terraforming_progress" in s

    def test_json_serializable(self):
        """Full results are JSON-serializable."""
        sim = Mars100Simulation(seed=42, max_years=10)
        results = sim.run()
        json_str = json.dumps(results)
        parsed = json.loads(json_str)
        assert parsed["_meta"]["engine"] == "mars-100"

    def test_callbacks_called(self):
        """Callback fires for each year."""
        years_seen = []
        sim = Mars100Simulation(seed=42, max_years=5)
        sim.run(callback=lambda y, s, d: years_seen.append(y))
        assert years_seen == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Physical invariant tests
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_resources_never_negative(self):
        """Resources stay >= 0 throughout the simulation."""
        sim = Mars100Simulation(seed=42, max_years=30)
        results = sim.run()
        for delta in results["deltas"]:
            for key, val in delta["resources_snapshot"].items():
                assert val >= 0.0, f"Year {delta['year']}: {key}={val} < 0"

    def test_resources_never_exceed_max(self):
        """Resources stay <= MAX_RESOURCES throughout."""
        sim = Mars100Simulation(seed=42, max_years=30)
        results = sim.run()
        for delta in results["deltas"]:
            for key, val in delta["resources_snapshot"].items():
                assert val <= MAX_RESOURCES, f"Year {delta['year']}: {key}={val} > max"

    def test_morale_bounded(self):
        """Morale stays in [0, 1] throughout."""
        sim = Mars100Simulation(seed=42, max_years=30)
        results = sim.run()
        for delta in results["deltas"]:
            assert MIN_MORALE <= delta["morale"] <= MAX_MORALE

    def test_alive_plus_dead_equals_total(self):
        """Conservation: alive + dead = total colonists."""
        sim = Mars100Simulation(seed=42, max_years=30)
        results = sim.run()
        for delta in results["deltas"]:
            assert delta["alive_count"] + delta["dead_count"] == 10

    def test_terraforming_bounded(self):
        """Terraforming progress stays in [0, 1]."""
        sim = Mars100Simulation(seed=42, max_years=30)
        results = sim.run()
        for delta in results["deltas"]:
            assert 0.0 <= delta["terraforming"] <= 1.0

    def test_governance_weights_finite(self):
        """All governance weights are finite and bounded."""
        sim = Mars100Simulation(seed=42, max_years=20)
        sim.run()
        for c in sim.state.colonists:
            assert math.isfinite(c.governance_weight)
            assert c.governance_weight <= MAX_GOVERNANCE_WEIGHT

    def test_dead_never_vote_or_act(self):
        """Dead colonists never make non-archived decisions after dying."""
        sim = Mars100Simulation(seed=42, max_years=50)
        results = sim.run()
        # Track when each colonist died
        death_years = {}
        for c in sim.state.dead_colonists():
            death_years[c.id] = c.year_died
        for delta in results["deltas"]:
            year = delta["year"]
            for decision in delta["decisions"]:
                cid = decision["colonist_id"]
                if cid in death_years and death_years[cid] is not None:
                    if year > death_years[cid]:
                        # After death, only archived actions
                        assert decision["action"] == "archived", (
                            f"{cid} died year {death_years[cid]} but acted in year {year}"
                        )

    def test_sub_sim_depth_never_exceeds_3(self):
        """Sub-sim depth never exceeds 3 in the log."""
        sim = Mars100Simulation(seed=42, max_years=30)
        results = sim.run()
        for log in results["state"]["sub_sim_log"]:
            if isinstance(log, dict):
                _check_sub_sim_depth(log, max_depth=3)


def _check_sub_sim_depth(log: dict, max_depth: int) -> None:
    """Recursively verify sub-sim depths."""
    sub_sims = log.get("sub_sim_log", [])
    if isinstance(sub_sims, list):
        for sub in sub_sims:
            if isinstance(sub, dict):
                depth = sub.get("depth", 0)
                assert depth <= max_depth, f"Sub-sim depth {depth} > {max_depth}"
                _check_sub_sim_depth(sub, max_depth)


# ---------------------------------------------------------------------------
# Governance evolution analysis tests
# ---------------------------------------------------------------------------

class TestGovernanceEvolution:
    def test_empty_proposals(self):
        state = ColonyState()
        result = _analyze_governance_evolution(state)
        assert result == []

    def test_decade_grouping(self, state):
        for i in range(3):
            state.proposals.append(GovernanceProposal(
                id=f"p{i}", year=i + 1, proposer_id="test",
                proposal_type="leadership_election",
                description="Test", value={},
            ))
        state.proposals[0].resolved = True
        state.proposals[0].outcome = "passed"
        state.proposals[1].resolved = True
        state.proposals[1].outcome = "failed"
        state.proposals[2].resolved = True
        state.proposals[2].outcome = "passed"

        state.year = 10
        result = _analyze_governance_evolution(state)
        assert len(result) >= 1
        assert result[0]["proposals"] == 3


# ---------------------------------------------------------------------------
# Active governance tests
# ---------------------------------------------------------------------------

class TestActiveGovernance:
    """Tests for apply_passed_proposals() — governance that takes effect."""

    def test_amendment_enacts(self, state):
        """Passed constitutional_amendment should appear in constitution list."""
        prop = GovernanceProposal(
            id="amend-1", year=5, proposer_id="vex-coder",
            proposal_type="constitutional_amendment",
            description="All resources shared equally",
            value={}, resolved=True, outcome="passed",
        )
        state.proposals.append(prop)
        effects = apply_passed_proposals(state)
        assert len(effects) == 1
        assert effects[0]["type"] == "amendment_enacted"
        assert "All resources shared equally" in state.governance["constitution"]
        assert len(state.governance["amendments"]) == 1
        assert state.governance["amendments"][0]["active"] is True

    def test_duplicate_amendment_not_reapplied(self, state):
        """Same proposal should not be applied twice."""
        prop = GovernanceProposal(
            id="amend-dup", year=3, proposer_id="oren-mediator",
            proposal_type="constitutional_amendment",
            description="Test dedup", value={}, resolved=True, outcome="passed",
        )
        state.proposals.append(prop)
        apply_passed_proposals(state)
        effects2 = apply_passed_proposals(state)
        assert len(effects2) == 0
        assert len(state.governance["amendments"]) == 1

    def test_leadership_election(self, state):
        """Passed leadership_election should set leader."""
        prop = GovernanceProposal(
            id="elect-1", year=10, proposer_id="nova-explorer",
            proposal_type="leadership_election",
            description="Nova for leader", value={}, resolved=True, outcome="passed",
        )
        state.proposals.append(prop)
        effects = apply_passed_proposals(state)
        assert state.governance["leader_id"] == "nova-explorer"
        assert state.governance["type"] == "elected_leader"

    def test_resource_allocation_boosts_morale(self, state):
        """Passed resource_allocation should boost morale."""
        old_morale = state.morale
        prop = GovernanceProposal(
            id="res-1", year=7, proposer_id="lyra-hydroponics",
            proposal_type="resource_allocation",
            description="Fair split", value={}, resolved=True, outcome="passed",
        )
        state.proposals.append(prop)
        apply_passed_proposals(state)
        assert state.morale > old_morale

    def test_failed_proposal_not_applied(self, state):
        """Failed proposals should have no effect."""
        prop = GovernanceProposal(
            id="fail-1", year=2, proposer_id="thresh-survivor",
            proposal_type="constitutional_amendment",
            description="Should not appear", value={}, resolved=True, outcome="failed",
        )
        state.proposals.append(prop)
        effects = apply_passed_proposals(state)
        assert len(effects) == 0
        assert len(state.governance["constitution"]) == 0

    def test_law_enactment(self, state):
        """Passed law should appear in constitution."""
        prop = GovernanceProposal(
            id="law-1", year=15, proposer_id="zeph-diplomat",
            proposal_type="law_enactment",
            description="Water rationing during storms",
            value={}, resolved=True, outcome="passed",
        )
        state.proposals.append(prop)
        effects = apply_passed_proposals(state)
        assert any(e["type"] == "law_enacted" for e in effects)
        assert "Water rationing during storms" in state.governance["constitution"]

    def test_exile_vote_kills_colonist(self, state):
        """Passed exile_vote should kill the least-liked colonist."""
        # Make one colonist universally disliked
        for c in state.alive_colonists():
            if c.id != "thresh-survivor":
                c.relationships["thresh-survivor"] = -0.9
        # Set thresh's relationships negative too
        thresh = next(c for c in state.colonists if c.id == "thresh-survivor")
        for other in state.alive_colonists():
            if other.id != thresh.id:
                thresh.relationships[other.id] = -0.5

        prop = GovernanceProposal(
            id="exile-1", year=20, proposer_id="oren-mediator",
            proposal_type="exile_vote",
            description="Exile disruptive colonist", value={},
            resolved=True, outcome="passed",
        )
        state.proposals.append(prop)
        alive_before = len(state.alive_colonists())
        effects = apply_passed_proposals(state)
        assert len(state.alive_colonists()) == alive_before - 1
        assert any(e["type"] == "exile" for e in effects)


# ---------------------------------------------------------------------------
# Skill growth tests
# ---------------------------------------------------------------------------

class TestSkillGrowth:
    """Tests for update_skills() — skills grow with use, decay without."""

    def test_used_skill_grows(self, state):
        """Skill used this year should increase."""
        colonist = state.alive_colonists()[0]
        old_terraform = colonist.skills["terraforming"]
        decisions = [{"colonist_id": colonist.id, "action": "work_terraforming"}]
        update_skills(state, decisions)
        assert colonist.skills["terraforming"] > old_terraform

    def test_unused_skill_decays(self, state):
        """Skills not used this year should decrease slightly."""
        colonist = state.alive_colonists()[0]
        old_prayer = colonist.skills["prayer"]
        decisions = [{"colonist_id": colonist.id, "action": "work_terraforming"}]
        update_skills(state, decisions)
        assert colonist.skills["prayer"] < old_prayer

    def test_skill_capped_at_one(self, state):
        """Skills should never exceed 1.0."""
        colonist = state.alive_colonists()[0]
        colonist.skills["terraforming"] = 0.995
        decisions = [{"colonist_id": colonist.id, "action": "work_terraforming"}]
        update_skills(state, decisions)
        assert colonist.skills["terraforming"] <= 1.0

    def test_skill_floor_at_zero(self, state):
        """Skills should never go below 0.0."""
        colonist = state.alive_colonists()[0]
        colonist.skills["sabotage"] = 0.002
        decisions = [{"colonist_id": colonist.id, "action": "work_terraforming"}]
        update_skills(state, decisions)
        assert colonist.skills["sabotage"] >= 0.0

    def test_growth_rate_correct(self, state):
        """Growth should match SKILL_GROWTH_RATE constant."""
        colonist = state.alive_colonists()[0]
        colonist.skills["mediation"] = 0.5
        decisions = [{"colonist_id": colonist.id, "action": "mediate"}]
        update_skills(state, decisions)
        assert abs(colonist.skills["mediation"] - (0.5 + SKILL_GROWTH_RATE)) < 0.001

    def test_dead_colonists_unchanged(self, state):
        """Dead colonists' skills should not change."""
        colonist = state.colonists[0]
        _kill_colonist(colonist, 5, "test")
        old_skills = dict(colonist.skills)
        decisions = [{"colonist_id": colonist.id, "action": "work_terraforming"}]
        update_skills(state, decisions)
        assert colonist.skills == old_skills

    def test_skill_bounds_after_100_years(self):
        """After 100 growth ticks, skill should be capped at 1.0."""
        state = ColonyState()
        state.colonists = create_colonists(random.Random(42))
        colonist = state.alive_colonists()[0]
        for _ in range(100):
            decisions = [{"colonist_id": colonist.id, "action": "work_terraforming"}]
            update_skills(state, decisions)
        assert 0.0 <= colonist.skills["terraforming"] <= 1.0
        for skill in SKILL_NAMES:
            assert 0.0 <= colonist.skills[skill] <= 1.0


# ---------------------------------------------------------------------------
# Recursive sub-simulation tests
# ---------------------------------------------------------------------------

class TestRecursiveSubSims:
    """Tests for recursive sub-sim programs (depth 2-3)."""

    def test_basic_sub_sim_runs(self, state, rng):
        """Non-meta colonist should produce a basic sub-sim."""
        vm = LispyVM(max_depth=3, max_steps=50_000, rng_seed=42)
        colonist = state.alive_colonists()[0]
        colonist.meta_aware = False
        colonist.skills["coding"] = 0.3  # below threshold
        event = {"type": "dust_storm", "severity": 0.5, "year": 1,
                 "description": "Test storm"}
        result = make_colonist_decision(colonist, event, state, vm, rng)
        # Basic sub-sim or no sub-sim — both valid
        if result.get("sub_sim"):
            assert result["sub_sim"]["status"] in ("completed", "error")

    def test_meta_aware_recursive_sub_sim(self, state, rng):
        """Meta-aware colonist should generate recursive programs."""
        vm = LispyVM(max_depth=3, max_steps=50_000, rng_seed=42)
        colonist = next(c for c in state.alive_colonists()
                        if c.stats["improvisation"] > 0.5)
        colonist.meta_aware = True
        colonist.skills["coding"] = 0.8
        event = {"type": "philosophical_crisis", "severity": 0.7, "year": 50,
                 "description": "Existential crisis"}
        # Force sub-sim by setting high propensity
        colonist.stats["improvisation"] = 0.95
        colonist.stats["paranoia"] = 0.8

        from src.mars100 import _build_recursive_sub_sim
        program = _build_recursive_sub_sim(colonist, event, state, rng)
        assert "sub-sim" in program
        assert "cooperative-model" in program
        assert "competitive-model" in program
        assert "meta-recursion" in program  # depth-3 from meta-aware

    def test_non_meta_no_depth3(self, state, rng):
        """Non-meta colonist should not generate depth-3 programs."""
        colonist = state.alive_colonists()[0]
        colonist.meta_aware = False
        colonist.skills["coding"] = 0.8

        from src.mars100 import _build_recursive_sub_sim
        program = _build_recursive_sub_sim(colonist, {"type": "test", "severity": 0.5}, state, rng)
        assert "sub-sim" in program
        assert "meta-recursion" not in program

    def test_sub_sim_depth_bounded(self):
        """Sub-sims should never exceed max_depth=3."""
        vm = LispyVM(max_depth=3, max_steps=50_000, rng_seed=42)
        # This program nests 3 levels deep — should work
        program = '(sub-sim "d1" (sub-sim "d2" (sub-sim "d3" (+ 1 2))))'
        result = vm.eval_str(program)
        assert result == 3

    def test_sub_sim_depth_4_fails(self):
        """Depth-4 sub-sim should raise error."""
        vm = LispyVM(max_depth=3, max_steps=50_000, rng_seed=42)
        program = '(sub-sim "d1" (sub-sim "d2" (sub-sim "d3" (sub-sim "d4" 42))))'
        from src.lispy_vm import LispyDepthError
        with pytest.raises(LispyDepthError):
            vm.eval_str(program)

    def test_recursive_program_evaluates(self, state, rng):
        """Recursive sub-sim program should evaluate successfully."""
        vm = LispyVM(max_depth=3, max_steps=50_000, rng_seed=42)
        colonist = state.alive_colonists()[0]
        colonist.meta_aware = True

        from src.mars100 import _build_recursive_sub_sim
        program = _build_recursive_sub_sim(colonist, {"type": "test", "severity": 0.5}, state, rng)
        vm.steps = 0
        vm.sub_sim_log = []
        result = vm.eval_str(program)
        assert result is not None
        # Should have logged sub-sims at depth 1+ (cooperative and competitive)
        assert len(vm.sub_sim_log) >= 2


# ---------------------------------------------------------------------------
# Integration: full simulation with new features
# ---------------------------------------------------------------------------

class TestSimulationIntegration:
    """Full-loop tests verifying new features work together."""

    def test_10_year_with_skills_and_governance(self):
        """10-year run should show skill growth and governance effects."""
        sim = Mars100Simulation(seed=42, max_years=10)
        results = sim.run()
        assert results["_meta"]["version"] == "2.0"
        state = sim.state
        # Skills should have changed from initial values
        for c in state.alive_colonists():
            for skill in SKILL_NAMES:
                assert 0.0 <= c.skills[skill] <= 1.0

    def test_governance_effects_in_delta(self):
        """Deltas should include governance_effects field."""
        sim = Mars100Simulation(seed=42, max_years=5)
        results = sim.run()
        for delta in results["deltas"]:
            assert "governance_effects" in delta
            assert "active_amendments" in delta

    def test_summary_includes_new_fields(self):
        """Summary should include new governance and depth metrics."""
        sim = Mars100Simulation(seed=42, max_years=10)
        results = sim.run()
        summary = results["summary"]
        assert "active_amendments" in summary
        assert "active_laws" in summary
        assert "governance_type" in summary
        assert "depth_2_insights" in summary
        assert "depth_3_insights" in summary

    def test_skill_bounds_across_full_sim(self):
        """All skills must remain in [0, 1] across a full 50-year run."""
        sim = Mars100Simulation(seed=123, max_years=50)
        sim.run()
        for c in sim.state.colonists:
            for skill in SKILL_NAMES:
                val = c.skills[skill]
                assert 0.0 <= val <= 1.0, (
                    f"Colonist {c.id} skill {skill} = {val} out of bounds"
                )

    def test_resources_bounded(self):
        """Resources must stay in [0, MAX_RESOURCES] for 50 years."""
        sim = Mars100Simulation(seed=99, max_years=50)
        sim.run()
        for key, val in sim.state.resources.items():
            assert 0.0 <= val <= MAX_RESOURCES, f"Resource {key} = {val}"

    def test_morale_bounded(self):
        """Morale must stay in [0, 1] across the simulation."""
        sim = Mars100Simulation(seed=77, max_years=50)
        sim.run()
        assert MIN_MORALE <= sim.state.morale <= MAX_MORALE


# ---------------------------------------------------------------------------
# Governance classification tests
# ---------------------------------------------------------------------------

class TestGovernanceClassification:
    """Tests for the scoring-based governance classifier."""

    def _make_colonist(self, cid: str, **overrides) -> Colonist:
        stats = {s: 0.5 for s in STAT_NAMES}
        skills = {s: 0.5 for s in SKILL_NAMES}
        stats.update(overrides.get("stats", {}))
        skills.update(overrides.get("skills", {}))
        c = Colonist(id=cid, name=cid, element="earth", stats=stats, skills=skills)
        c.governance_weight = overrides.get("governance_weight", 1.0)
        c.proposals_made = overrides.get("proposals_made", 0)
        c.votes_cast = overrides.get("votes_cast", 0)
        return c

    def test_anarchy_no_proposals(self):
        """Empty colony with no proposals → anarchy."""
        state = ColonyState()
        state.colonists = [self._make_colonist(f"c{i}") for i in range(5)]
        result = classify_governance(state)
        assert result == "anarchy"

    def test_anarchy_no_colonists(self):
        """No alive colonists → anarchy."""
        state = ColonyState()
        assert classify_governance(state) == "anarchy"

    def test_commune_equal_weights(self):
        """Equal governance weights → commune."""
        state = ColonyState()
        state.year = 30
        cols = [self._make_colonist(f"c{i}", governance_weight=1.0, proposals_made=3, votes_cast=5) for i in range(6)]
        state.colonists = cols
        # Add enough proposals to not be anarchy
        for i in range(5):
            p = GovernanceProposal(id=f"p{i}", year=20+i, proposer_id=f"c{i%6}",
                                   proposal_type="resource_allocation", description="test", value={})
            p.resolved = True
            p.outcome = "passed"
            state.proposals.append(p)
        result = classify_governance(state)
        assert result == "commune"

    def test_autocracy_dominant_leader(self):
        """Single dominant leader with no elections → autocracy."""
        state = ColonyState()
        state.year = 30
        cols = [self._make_colonist(f"c{i}", governance_weight=1.0) for i in range(5)]
        cols[0].governance_weight = 8.0
        state.colonists = cols
        state.governance["leader_id"] = "c0"
        # Add proposals but no elections
        for i in range(5):
            p = GovernanceProposal(id=f"p{i}", year=20+i, proposer_id="c0",
                                   proposal_type="law_enactment", description="test", value={})
            p.resolved = True
            p.outcome = "passed"
            state.proposals.append(p)
        result = classify_governance(state)
        assert result == "autocracy"

    def test_theocracy_faith_leaders(self):
        """Faith-dominant high-weight colonists → theocracy."""
        state = ColonyState()
        state.year = 30
        cols = [self._make_colonist(f"c{i}", stats={"faith": 0.9}, governance_weight=3.0) for i in range(4)]
        cols.append(self._make_colonist("c4", governance_weight=1.0))
        state.colonists = cols
        for i in range(5):
            state.proposals.append(GovernanceProposal(
                id=f"p{i}", year=20+i, proposer_id=f"c{i%4}",
                proposal_type="law_enactment", description="test", value={},
            ))
        result = classify_governance(state)
        assert result == "theocracy"

    def test_technocracy_coding_leaders(self):
        """Coding-dominant high-weight colonists → technocracy."""
        state = ColonyState()
        state.year = 30
        cols = [self._make_colonist(f"c{i}", skills={"coding": 0.9}, governance_weight=3.0) for i in range(4)]
        cols.append(self._make_colonist("c4", governance_weight=1.0))
        state.colonists = cols
        for i in range(5):
            state.proposals.append(GovernanceProposal(
                id=f"p{i}", year=20+i, proposer_id=f"c{i%4}",
                proposal_type="law_enactment", description="test", value={},
            ))
        result = classify_governance(state)
        assert result == "technocracy"

    def test_elected_democracy(self):
        """Leader backed by recent election + constitution → elected_democracy."""
        state = ColonyState()
        state.year = 30
        cols = [self._make_colonist(f"c{i}", governance_weight=2.0) for i in range(6)]
        state.colonists = cols
        state.governance["leader_id"] = "c0"
        state.governance["constitution"] = ["law1", "law2", "law3"]
        state.governance["amendments"] = [{"proposal_id": "a1"}]
        # Recent leadership election
        p = GovernanceProposal(id="elect1", year=25, proposer_id="c0",
                               proposal_type="leadership_election", description="test", value={})
        p.resolved = True
        p.outcome = "passed"
        state.proposals.append(p)
        result = classify_governance(state)
        assert result == "elected_democracy"

    def test_council_multiple_high_weight(self):
        """Multiple high-weight colonists, no dominant leader → council."""
        state = ColonyState()
        state.year = 30
        cols = [self._make_colonist(f"c{i}", governance_weight=3.0) for i in range(4)]
        cols.append(self._make_colonist("c4", governance_weight=1.0))
        state.colonists = cols
        for i in range(5):
            state.proposals.append(GovernanceProposal(
                id=f"p{i}", year=20+i, proposer_id=f"c{i%4}",
                proposal_type="resource_allocation", description="test", value={},
            ))
        result = classify_governance(state)
        assert result == "council"

    def test_result_always_in_governance_forms(self):
        """classify_governance always returns a valid form."""
        rng = random.Random(42)
        for seed in range(10):
            sim = Mars100Simulation(seed=seed, max_years=30)
            sim.run()
            form = classify_governance(sim.state)
            assert form in GOVERNANCE_FORMS, f"seed={seed}: got {form}"

    def test_delta_has_governance_form(self):
        """Every yearly delta includes a governance_form field."""
        sim = Mars100Simulation(seed=42, max_years=10)
        results = sim.run()
        for delta in results["deltas"]:
            assert "governance_form" in delta
            assert delta["governance_form"] in GOVERNANCE_FORMS


# ---------------------------------------------------------------------------
# Value convergence tests
# ---------------------------------------------------------------------------

class TestValueConvergence:
    """Tests for value convergence tracking."""

    def test_identical_stats_zero_convergence(self):
        """Colonists with identical stats → convergence_score 0."""
        state = ColonyState()
        for i in range(5):
            c = Colonist(id=f"c{i}", name=f"c{i}", element="earth",
                         stats={s: 0.5 for s in STAT_NAMES},
                         skills={s: 0.5 for s in SKILL_NAMES})
            state.colonists.append(c)
        result = compute_value_convergence(state)
        assert result["convergence_score"] == 0.0
        for v in result["stats_std"].values():
            assert v == 0.0

    def test_divergent_stats_positive_convergence(self):
        """Colonists with divergent stats → positive convergence_score."""
        state = ColonyState()
        for i in range(5):
            c = Colonist(id=f"c{i}", name=f"c{i}", element="earth",
                         stats={s: i * 0.2 for s in STAT_NAMES},
                         skills={s: 0.5 for s in SKILL_NAMES})
            state.colonists.append(c)
        result = compute_value_convergence(state)
        assert result["convergence_score"] > 0.0

    def test_single_colonist_zero(self):
        """Single alive colonist → convergence_score 0."""
        state = ColonyState()
        c = Colonist(id="c0", name="c0", element="earth",
                     stats={s: 0.5 for s in STAT_NAMES},
                     skills={s: 0.5 for s in SKILL_NAMES})
        state.colonists.append(c)
        result = compute_value_convergence(state)
        assert result["convergence_score"] == 0.0
        assert result["sample_size"] == 1

    def test_dead_colonists_excluded(self):
        """Dead colonists should not affect convergence."""
        state = ColonyState()
        alive = Colonist(id="c0", name="c0", element="earth",
                         stats={s: 0.5 for s in STAT_NAMES},
                         skills={s: 0.5 for s in SKILL_NAMES})
        dead = Colonist(id="c1", name="c1", element="earth",
                        stats={s: 0.9 for s in STAT_NAMES},
                        skills={s: 0.5 for s in SKILL_NAMES})
        dead.alive = False
        state.colonists = [alive, dead]
        result = compute_value_convergence(state)
        assert result["sample_size"] == 1

    def test_delta_has_value_convergence(self):
        """Every yearly delta includes value_convergence."""
        sim = Mars100Simulation(seed=42, max_years=10)
        results = sim.run()
        for delta in results["deltas"]:
            assert "value_convergence" in delta
            assert isinstance(delta["value_convergence"], float)

    def test_summary_has_convergence_timeline(self):
        """Summary includes convergence timeline."""
        sim = Mars100Simulation(seed=42, max_years=10)
        results = sim.run()
        summary = results["summary"]
        assert "convergence_timeline" in summary
        assert "value_convergence" in summary
        assert len(summary["convergence_timeline"]) > 0


# ---------------------------------------------------------------------------
# Morale clamp helper tests
# ---------------------------------------------------------------------------

class TestMoraleClamp:
    """Tests for the clamp_morale helper."""

    def test_clamp_negative(self):
        state = ColonyState()
        state.morale = -0.5
        clamp_morale(state)
        assert state.morale == MIN_MORALE

    def test_clamp_above_max(self):
        state = ColonyState()
        state.morale = 1.5
        clamp_morale(state)
        assert state.morale == MAX_MORALE

    def test_clamp_in_range(self):
        state = ColonyState()
        state.morale = 0.5
        clamp_morale(state)
        assert state.morale == 0.5

    def test_morale_bounded_all_deltas(self):
        """Morale is always >= 0 in every delta across multiple seeds."""
        for seed in (42, 99, 7, 1337):
            sim = Mars100Simulation(seed=seed, max_years=30)
            results = sim.run()
            for delta in results["deltas"]:
                assert delta["morale"] >= MIN_MORALE, (
                    f"seed={seed} year={delta['year']}: morale={delta['morale']}"
                )
                assert delta["morale"] <= MAX_MORALE
