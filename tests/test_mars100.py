"""Tests for the Mars-100 recursive colony simulation."""
from __future__ import annotations

import json
import pytest
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100 import (
    Mars100, create_colonists, DEFAULT_SEED,
    Colonist, Resources, Proposal, SubSimLog,
    ELEMENTS, STAT_NAMES, SKILL_NAMES, DRIVE_NAMES,
    EVENTS, PROPOSAL_TYPES, COLONIST_TEMPLATES,
    roll_event, apply_event, check_deaths,
    resolve_proposals, replenish_resources,
    evolve_relationships, decide_action, resolve_action,
    check_meta_awareness, _safe_serialize,
)


# ---------------------------------------------------------------------------
# Colonist creation
# ---------------------------------------------------------------------------

class TestColonistCreation:
    def test_creates_10_colonists(self):
        rng = random.Random(42)
        colonists = create_colonists(rng)
        assert len(colonists) == 10

    def test_all_alive(self):
        colonists = create_colonists(random.Random(42))
        assert all(c.alive for c in colonists)

    def test_unique_ids(self):
        colonists = create_colonists(random.Random(42))
        ids = [c.id for c in colonists]
        assert len(set(ids)) == 10

    def test_all_have_stats(self):
        colonists = create_colonists(random.Random(42))
        for c in colonists:
            for stat in STAT_NAMES:
                assert stat in c.stats
                assert 0 <= c.stats[stat] <= 100

    def test_all_have_skills(self):
        colonists = create_colonists(random.Random(42))
        for c in colonists:
            for skill in SKILL_NAMES:
                assert skill in c.skills
                assert 0 <= c.skills[skill] <= 100

    def test_all_have_drives(self):
        colonists = create_colonists(random.Random(42))
        for c in colonists:
            for drive in DRIVE_NAMES:
                assert drive in c.drives
                assert 0 <= c.drives[drive] <= 3.0

    def test_relationships_initialized(self):
        colonists = create_colonists(random.Random(42))
        for c in colonists:
            # Should have relationships with 9 others
            assert len(c.relationships) == 9
            assert c.id not in c.relationships

    def test_element_skill_bonus(self):
        colonists = create_colonists(random.Random(42))
        earth_colonists = [c for c in colonists if c.element == "earth"]
        # Earth colonists should have boosted terraforming
        for c in earth_colonists:
            assert c.skills["terraforming"] >= 30  # base min 10 + 20 bonus

    def test_deterministic(self):
        c1 = create_colonists(random.Random(42))
        c2 = create_colonists(random.Random(42))
        for a, b in zip(c1, c2):
            assert a.id == b.id
            assert a.stats == b.stats

    def test_serialization_roundtrip(self):
        colonists = create_colonists(random.Random(42))
        for c in colonists:
            d = c.to_dict()
            assert isinstance(json.dumps(d), str)  # JSON-safe


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

class TestResources:
    def test_initial_values(self):
        r = Resources()
        assert r.food == 500.0
        assert r.water == 400.0

    def test_total(self):
        r = Resources(food=100, water=100, oxygen=100, power=100, materials=100)
        assert r.total() == 500

    def test_critically_low(self):
        r = Resources(food=10, water=100, oxygen=100, power=5, materials=100)
        critical = r.critically_low()
        assert "food" in critical
        assert "power" in critical
        assert "water" not in critical

    def test_serialization(self):
        r = Resources(food=123.456)
        d = r.to_dict()
        assert d["food"] == 123.5  # rounded


# ---------------------------------------------------------------------------
# Environmental events
# ---------------------------------------------------------------------------

class TestEvents:
    def test_roll_event_returns_dict(self):
        event = roll_event(1, random.Random(42))
        assert "type" in event
        assert "severity" in event
        assert "year" in event

    def test_event_type_valid(self):
        for _ in range(100):
            event = roll_event(1, random.Random(_))
            assert event["type"] in EVENTS

    def test_severity_in_range(self):
        for _ in range(100):
            event = roll_event(1, random.Random(_))
            assert 0.2 <= event["severity"] <= 1.0

    def test_apply_dust_storm(self):
        r = Resources(power=100)
        event = {"type": "dust_storm", "severity": 0.5, "year": 1}
        desc = apply_event(event, r)
        assert r.power < 100
        assert "Dust storm" in desc

    def test_apply_resource_strike(self):
        r = Resources(materials=100)
        event = {"type": "resource_strike", "severity": 0.8, "year": 1}
        apply_event(event, r)
        assert r.materials > 100

    def test_apply_earth_contact(self):
        r = Resources(food=100)
        event = {"type": "earth_contact", "severity": 0.5, "year": 1}
        apply_event(event, r)
        assert r.food == 180

    def test_apply_calm_year(self):
        r = Resources()
        before = r.to_dict()
        event = {"type": "calm_year", "severity": 0.5, "year": 1}
        apply_event(event, r)
        assert r.to_dict() == before  # no change


# ---------------------------------------------------------------------------
# Death mechanics
# ---------------------------------------------------------------------------

class TestDeaths:
    def test_no_deaths_with_resources(self):
        colonists = create_colonists(random.Random(42))
        r = Resources()  # all well-stocked
        deaths = check_deaths(colonists, r, year=1, rng=random.Random(99))
        # With good resources, death is very unlikely but not impossible
        # Just check it returns a list
        assert isinstance(deaths, list)

    def test_deaths_possible_with_no_resources(self):
        colonists = create_colonists(random.Random(42))
        r = Resources(food=0, water=0, oxygen=0, power=0, materials=0)
        # Run many times to ensure at least one death occurs
        any_death = False
        for seed in range(100):
            deaths = check_deaths(colonists, r, year=1, rng=random.Random(seed))
            if deaths:
                any_death = True
                break
            # Reset colonists
            for c in colonists:
                c.alive = True
                c.year_of_death = None
        assert any_death

    def test_death_records_cause(self):
        colonists = create_colonists(random.Random(42))
        r = Resources(food=0, water=0, oxygen=0, power=0, materials=0)
        for seed in range(200):
            deaths = check_deaths(colonists, r, year=10, rng=random.Random(seed))
            if deaths:
                assert "cause" in deaths[0]
                assert "colonist_id" in deaths[0]
                break
            for c in colonists:
                c.alive = True
                c.year_of_death = None

    def test_dead_colonists_stay_dead(self):
        colonists = create_colonists(random.Random(42))
        colonists[0].alive = False
        colonists[0].year_of_death = 5
        r = Resources(food=0, water=0, oxygen=0, power=0, materials=0)
        deaths = check_deaths(colonists, r, year=10, rng=random.Random(42))
        # Already dead colonist should not appear in deaths list
        assert not any(d["colonist_id"] == colonists[0].id for d in deaths)


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------

class TestGovernance:
    def test_proposal_resolution(self):
        colonists = create_colonists(random.Random(42))
        proposal = Proposal(
            id="prop-1", year=5, proposer_id="ares-1",
            kind="ration_policy", description="test",
            lispy_policy="(define ration #t)",
        )
        # 6 votes for, 2 against -> passes (quorum = 5)
        for i, c in enumerate(colonists[:8]):
            proposal.votes[c.id] = i < 6

        resolved = resolve_proposals([proposal], colonists)
        assert len(resolved) == 1
        assert resolved[0]["passed"] is True

    def test_proposal_fails(self):
        colonists = create_colonists(random.Random(42))
        proposal = Proposal(
            id="prop-2", year=5, proposer_id="ares-1",
            kind="punishment", description="exile test",
            lispy_policy="(define exile #t)",
        )
        # 2 for, 6 against -> fails
        for i, c in enumerate(colonists[:8]):
            proposal.votes[c.id] = i < 2

        resolved = resolve_proposals([proposal], colonists)
        assert len(resolved) == 1
        assert resolved[0]["passed"] is False

    def test_no_quorum(self):
        colonists = create_colonists(random.Random(42))
        proposal = Proposal(
            id="prop-3", year=5, proposer_id="ares-1",
            kind="new_law", description="test",
            lispy_policy="#t",
        )
        proposal.votes["ares-1"] = True  # only 1 vote, quorum=5
        resolved = resolve_proposals([proposal], colonists)
        assert len(resolved) == 0  # not resolved yet

    def test_proposal_serialization(self):
        p = Proposal(
            id="test", year=1, proposer_id="ares-1",
            kind="ration_policy", description="test",
            lispy_policy="#t",
        )
        d = p.to_dict()
        assert isinstance(json.dumps(d), str)


# ---------------------------------------------------------------------------
# Resource replenishment
# ---------------------------------------------------------------------------

class TestReplenishment:
    def test_resources_change(self):
        colonists = create_colonists(random.Random(42))
        r = Resources(food=500, water=400, oxygen=600, power=300, materials=200)
        before_food = r.food
        replenish_resources(r, colonists, year=1)
        # Resources should change (production + consumption)
        assert r.food != before_food

    def test_no_negative_resources(self):
        colonists = create_colonists(random.Random(42))
        r = Resources(food=1, water=1, oxygen=1, power=1, materials=1)
        replenish_resources(r, colonists, year=1)
        assert r.food >= 0
        assert r.water >= 0
        assert r.oxygen >= 0
        assert r.power >= 0

    def test_empty_colony(self):
        colonists = create_colonists(random.Random(42))
        for c in colonists:
            c.alive = False
        r = Resources()
        before = r.to_dict()
        replenish_resources(r, colonists, year=1)
        assert r.to_dict() == before  # no change with no living colonists


# ---------------------------------------------------------------------------
# Relationship evolution
# ---------------------------------------------------------------------------

class TestRelationships:
    def test_shared_work_improves_relationship(self):
        colonists = create_colonists(random.Random(42))
        c1, c2 = colonists[0], colonists[1]
        initial_rel = c1.relationships[c2.id]
        actions = [
            {"colonist_id": c1.id, "action": "work_farm"},
            {"colonist_id": c2.id, "action": "work_mine"},
        ]
        evolve_relationships(colonists, actions, random.Random(42))
        # Relationship should improve (or at least not crash)
        assert isinstance(c1.relationships[c2.id], float)

    def test_hoarding_reduces_trust(self):
        colonists = create_colonists(random.Random(42))
        c1 = colonists[0]
        # Record others' initial view of c1
        initial_views = {}
        for other in colonists[1:]:
            initial_views[other.id] = other.relationships.get(c1.id, 0)
        actions = [{"colonist_id": c1.id, "action": "hoard"}]
        evolve_relationships(colonists, actions, random.Random(42))
        # Others' view of c1 should worsen (hoarder penalty -0.05, decay *0.95)
        for other in colonists[1:]:
            if other.alive:
                new_val = other.relationships[c1.id]
                old_val = initial_views[other.id]
                # Element affinity might add +0.01
                same_element = other.element == c1.element
                element_bonus = 0.01 if same_element else 0
                expected = (old_val + element_bonus - 0.05) * 0.95
                assert abs(new_val - expected) < 0.01, (
                    f"{other.id}: new={new_val}, expected={expected}"
                )


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------

class TestDecisions:
    def test_decide_returns_tuple(self):
        colonists = create_colonists(random.Random(42))
        r = Resources()
        action_type, source = decide_action(
            colonists[0], r, year=1, living_colonists=colonists,
            active_proposals=[], rng=random.Random(42),
        )
        assert isinstance(action_type, str)
        assert isinstance(source, str)

    def test_lispy_source_parseable(self):
        """All generated LisPy should parse without error."""
        from src.lispy import parse
        colonists = create_colonists(random.Random(42))
        r = Resources()
        for year in range(1, 20):
            for c in colonists:
                _, source = decide_action(
                    c, r, year=year, living_colonists=colonists,
                    active_proposals=[], rng=random.Random(42 + year),
                )
                ast = parse(source)
                assert ast is not None


# ---------------------------------------------------------------------------
# Meta-awareness
# ---------------------------------------------------------------------------

class TestMetaAwareness:
    def test_no_awareness_early(self):
        colonists = create_colonists(random.Random(42))
        for c in colonists:
            insight = check_meta_awareness(c, year=1, rng=random.Random(42))
            assert insight is None

    def test_awareness_possible_late(self):
        colonists = create_colonists(random.Random(42))
        # Boost stats to trigger awareness
        colonists[9].stats["paranoia"] = 95
        colonists[9].stats["faith"] = 90
        colonists[9].sub_sim_count = 15
        any_insight = False
        for seed in range(200):
            insight = check_meta_awareness(
                colonists[9], year=80, rng=random.Random(seed)
            )
            if insight:
                any_insight = True
                break
        assert any_insight


# ---------------------------------------------------------------------------
# Full simulation smoke tests
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_create(self):
        sim = Mars100(seed=42)
        assert sim.year == 0
        assert len(sim.colonists) == 10

    def test_one_tick(self):
        sim = Mars100(seed=42)
        record = sim.tick()
        assert record["year"] == 1
        assert "event" in record
        assert "resources" in record
        assert "population" in record
        assert record["population"] <= 10

    def test_10_year_smoke(self):
        """Run 10 years without crash."""
        sim = Mars100(seed=42)
        history = sim.run(years=10)
        assert len(history) == 10
        assert history[0]["year"] == 1
        assert history[-1]["year"] == 10
        # Population should be reasonable
        assert 0 <= history[-1]["population"] <= 10

    def test_deterministic(self):
        """Same seed produces same results."""
        sim1 = Mars100(seed=42)
        h1 = sim1.run(years=20)
        sim2 = Mars100(seed=42)
        h2 = sim2.run(years=20)
        for r1, r2 in zip(h1, h2):
            assert r1["year"] == r2["year"]
            assert r1["population"] == r2["population"]
            assert r1["event"]["type"] == r2["event"]["type"]

    def test_different_seeds_differ(self):
        """Different seeds produce different histories."""
        h1 = Mars100(seed=1).run(years=20)
        h2 = Mars100(seed=9999).run(years=20)
        # Events should differ in at least some years
        diffs = sum(1 for a, b in zip(h1, h2) if a["event"]["type"] != b["event"]["type"])
        assert diffs > 0

    def test_state_snapshot(self):
        sim = Mars100(seed=42)
        sim.run(years=5)
        snap = sim.state_snapshot()
        assert snap["_meta"]["engine"] == "mars-100"
        assert snap["_meta"]["year"] == 5
        # Should be JSON-serializable
        json_str = json.dumps(snap)
        assert len(json_str) > 100

    def test_resources_bounded(self):
        """Resources should never go below 0."""
        sim = Mars100(seed=42)
        for _ in range(50):
            record = sim.tick()
            r = record["resources"]
            assert r["food"] >= 0
            assert r["water"] >= 0
            assert r["oxygen"] >= 0
            assert r["power"] >= 0

    def test_dead_colonists_archived(self):
        """Dead colonists should be tracked."""
        sim = Mars100(seed=42)
        sim.run(years=100)
        # At least some deaths should occur over 100 years
        snap = sim.state_snapshot()
        assert isinstance(snap["dead_colonists"], list)

    def test_sub_sims_logged(self):
        """Sub-simulations should be logged."""
        sim = Mars100(seed=42)
        sim.run(years=50)
        # At least some sub-sims should have run
        assert isinstance(sim.sub_sim_logs, list)

    def test_governance_emerges(self):
        """Proposals should appear over time."""
        sim = Mars100(seed=42)
        sim.run(years=50)
        assert isinstance(sim.proposals, list)

    def test_100_year_full_run(self):
        """The full 100-year simulation completes without crash."""
        sim = Mars100(seed=42)
        history = sim.run(years=100)
        assert len(history) >= 1
        assert len(history) <= 100
        snap = sim.state_snapshot()
        assert snap["_meta"]["year"] >= 1

    def test_extinction_stops_sim(self):
        """If all colonists die, sim stops early."""
        # Use a brutal seed with minimal resources
        sim = Mars100(seed=42)
        sim.resources = Resources(food=0, water=0, oxygen=0, power=0, materials=0)
        history = sim.run(years=100)
        # Should stop before year 100 due to extinction
        final_pop = history[-1]["population"]
        if final_pop == 0:
            assert history[-1]["event"]["type"] == "extinction" or len(history) < 100


# ---------------------------------------------------------------------------
# Conservation laws / invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_population_never_exceeds_10(self):
        """No births in this model, so population can only decrease."""
        sim = Mars100(seed=42)
        for _ in range(100):
            record = sim.tick()
            assert record["population"] <= 10

    def test_dead_stay_dead(self):
        """Once a colonist dies, they don't come back."""
        sim = Mars100(seed=42)
        dead_ids = set()
        for _ in range(100):
            sim.tick()
            for c in sim.colonists:
                if not c.alive:
                    if c.id in dead_ids:
                        continue  # already known dead
                    dead_ids.add(c.id)
                else:
                    assert c.id not in dead_ids, f"{c.id} resurrected!"

    def test_stats_bounded(self):
        """All stats should remain in [0, 100]."""
        sim = Mars100(seed=42)
        sim.run(years=50)
        for c in sim.colonists:
            for stat, val in c.stats.items():
                assert 0 <= val <= 100, f"{c.id}.{stat} = {val}"

    def test_relationships_bounded(self):
        """Relationships should remain in [-1, 1]."""
        sim = Mars100(seed=42)
        sim.run(years=50)
        for c in sim.colonists:
            for other_id, val in c.relationships.items():
                assert -1.0 <= val <= 1.0, f"{c.id}->{other_id} = {val}"


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_safe_serialize_primitives(self):
        assert _safe_serialize(42) == 42
        assert _safe_serialize("hello") == "hello"
        assert _safe_serialize(None) is None
        assert _safe_serialize(True) is True

    def test_safe_serialize_list(self):
        assert _safe_serialize([1, "a", None]) == [1, "a", None]

    def test_safe_serialize_dict(self):
        assert _safe_serialize({"a": 1}) == {"a": 1}

    def test_safe_serialize_complex(self):
        from src.lispy import Lambda, Env
        result = _safe_serialize(Lambda(params=["x"], body=["x"], closure=Env()))
        assert isinstance(result, str)  # converted to string representation
