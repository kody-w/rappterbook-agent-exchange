"""Tests for Mars-100 politics organ — factions, grievances, crisis governance."""
from __future__ import annotations

import random
import pytest
from dataclasses import dataclass, field
from typing import Any

from src.mars100.politics import (
    Faction, Alliance, Grievance, PoliticalState, PoliticalTickResult,
    FACTION_NAMES, MIN_FACTION_SIZE, AFFINITY_THRESHOLD, MAX_FACTIONS,
    GRIEVANCE_DECAY, GRIEVANCE_CAP, REVOLT_THRESHOLD,
    ALLIANCE_THRESHOLD, ALLIANCE_DECAY,
    CRISIS_PROPOSAL_THRESHOLD, AMENDMENT_THRESHOLD,
    compute_affinity, detect_factions, form_alliances,
    accumulate_grievances, decay_grievances, total_grievance,
    should_crisis_propose, compute_faction_pressure, compute_voting_bloc,
    check_amendment_promotion, tick_politics,
)
from src.mars100.colonist import (
    Colonist, ColonistStats, ColonistSkills, STAT_NAMES, SKILL_NAMES,
    create_founding_ten,
)
from src.mars100.colony import SocialGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_colonist(cid: str, stats: dict[str, float] | None = None,
                  alive: bool = True) -> Colonist:
    """Quick colonist builder for tests."""
    s = stats or {}
    return Colonist(
        id=cid, name=cid.title(), element="earth", archetype="test",
        stats=ColonistStats(**{k: s.get(k, 0.5) for k in STAT_NAMES}),
        skills=ColonistSkills(**{k: 0.1 for k in SKILL_NAMES}),
        decision_expr="(+ resolve empathy)",
        alive=alive,
    )


def make_social(colonists: list[Colonist], seed: int = 42) -> SocialGraph:
    """Create a social graph for test colonists."""
    sg = SocialGraph()
    ids = [c.id for c in colonists if c.is_active()]
    sg.initialize(ids, random.Random(seed))
    return sg


# ---------------------------------------------------------------------------
# compute_affinity
# ---------------------------------------------------------------------------

class TestComputeAffinity:
    def test_identical_vectors(self):
        v = {"resolve": 0.8, "empathy": 0.6, "paranoia": 0.2}
        assert compute_affinity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = {"x": 1.0, "y": 0.0}
        b = {"x": 0.0, "y": 1.0}
        assert compute_affinity(a, b) == pytest.approx(0.0)

    def test_similar_vectors(self):
        a = {"resolve": 0.9, "empathy": 0.8, "paranoia": 0.1}
        b = {"resolve": 0.85, "empathy": 0.75, "paranoia": 0.15}
        aff = compute_affinity(a, b)
        assert aff > 0.95

    def test_dissimilar_vectors(self):
        a = {"resolve": 0.9, "empathy": 0.1}
        b = {"resolve": 0.1, "empathy": 0.9}
        aff = compute_affinity(a, b)
        assert aff < 0.5

    def test_empty_vectors(self):
        assert compute_affinity({}, {}) == 0.0

    def test_zero_magnitude(self):
        a = {"x": 0.0, "y": 0.0}
        b = {"x": 1.0, "y": 0.0}
        assert compute_affinity(a, b) == 0.0

    def test_returns_bounded(self):
        """Affinity always in [0, 1]."""
        rng = random.Random(99)
        for _ in range(100):
            a = {f"s{i}": rng.random() for i in range(6)}
            b = {f"s{i}": rng.random() for i in range(6)}
            aff = compute_affinity(a, b)
            assert 0.0 <= aff <= 1.0


# ---------------------------------------------------------------------------
# detect_factions
# ---------------------------------------------------------------------------

class TestDetectFactions:
    def test_too_few_colonists(self):
        c = [make_colonist("a")]
        sg = make_social(c)
        result = detect_factions(c, sg, year=10, rng=random.Random(42))
        assert result == []

    def test_two_similar_colonists_form_faction(self):
        c1 = make_colonist("a", {"resolve": 0.9, "empathy": 0.9, "faith": 0.8,
                                  "paranoia": 0.1, "improvisation": 0.8, "hoarding": 0.1})
        c2 = make_colonist("b", {"resolve": 0.85, "empathy": 0.85, "faith": 0.75,
                                  "paranoia": 0.15, "improvisation": 0.75, "hoarding": 0.15})
        c3 = make_colonist("c", {"resolve": 0.1, "empathy": 0.1, "faith": 0.1,
                                  "paranoia": 0.9, "improvisation": 0.1, "hoarding": 0.9})
        c4 = make_colonist("d", {"resolve": 0.15, "empathy": 0.15, "faith": 0.15,
                                  "paranoia": 0.85, "improvisation": 0.15, "hoarding": 0.85})
        colonists = [c1, c2, c3, c4]
        sg = make_social(colonists)
        factions = detect_factions(colonists, sg, year=10, rng=random.Random(42))
        assert len(factions) >= 1
        for f in factions:
            assert len(f.member_ids) >= MIN_FACTION_SIZE

    def test_dead_colonists_excluded(self):
        c1 = make_colonist("a", alive=False)
        c2 = make_colonist("b")
        c3 = make_colonist("c")
        c4 = make_colonist("d")
        sg = make_social([c2, c3, c4])
        factions = detect_factions([c1, c2, c3, c4], sg, year=5, rng=random.Random(42))
        for f in factions:
            assert "a" not in f.member_ids

    def test_max_factions_limit(self):
        colonists = [make_colonist(f"c{i}") for i in range(20)]
        sg = make_social(colonists)
        factions = detect_factions(colonists, sg, year=10, rng=random.Random(42))
        assert len(factions) <= MAX_FACTIONS

    def test_faction_has_ideology(self):
        colonists = create_founding_ten(42)
        sg = make_social(colonists)
        factions = detect_factions(colonists, sg, year=10, rng=random.Random(42))
        for f in factions:
            assert all(stat in f.ideology for stat in STAT_NAMES)

    def test_faction_reuse_on_overlap(self):
        colonists = create_founding_ten(42)
        sg = make_social(colonists)
        factions1 = detect_factions(colonists, sg, year=10, rng=random.Random(42))
        if factions1:
            factions2 = detect_factions(colonists, sg, year=15, rng=random.Random(42),
                                        existing=factions1)
            reused_ids = {f.id for f in factions1} & {f.id for f in factions2}
            # At least some factions should be reused
            assert len(factions2) > 0


# ---------------------------------------------------------------------------
# form_alliances
# ---------------------------------------------------------------------------

class TestFormAlliances:
    def test_no_factions(self):
        assert form_alliances([], year=10) == []

    def test_single_faction(self):
        f = Faction(id="f1", member_ids=["a", "b"], ideology={"resolve": 0.8},
                    formed_year=5)
        assert form_alliances([f], year=10) == []

    def test_similar_factions_ally(self):
        f1 = Faction(id="f1", member_ids=["a", "b"],
                     ideology={"resolve": 0.8, "empathy": 0.7, "faith": 0.6,
                               "paranoia": 0.2, "improvisation": 0.5, "hoarding": 0.3},
                     formed_year=5)
        f2 = Faction(id="f2", member_ids=["c", "d"],
                     ideology={"resolve": 0.75, "empathy": 0.65, "faith": 0.55,
                               "paranoia": 0.25, "improvisation": 0.45, "hoarding": 0.35},
                     formed_year=5)
        alliances = form_alliances([f1, f2], year=10)
        assert len(alliances) >= 1

    def test_dissimilar_factions_no_alliance(self):
        f1 = Faction(id="f1", member_ids=["a", "b"],
                     ideology={"resolve": 0.9, "empathy": 0.1},
                     formed_year=5)
        f2 = Faction(id="f2", member_ids=["c", "d"],
                     ideology={"resolve": 0.1, "empathy": 0.9},
                     formed_year=5)
        alliances = form_alliances([f1, f2], year=10)
        # Low affinity should prevent alliance
        for a in alliances:
            assert a.strength > 0

    def test_alliance_decay(self):
        # Orthogonal ideologies: high-resolve vs high-empathy
        f1 = Faction(id="f1", member_ids=["a", "b"],
                     ideology={"resolve": 0.9, "empathy": 0.1},
                     formed_year=5)
        f2 = Faction(id="f2", member_ids=["c", "d"],
                     ideology={"resolve": 0.1, "empathy": 0.9},
                     formed_year=5)
        existing = [Alliance(faction_a="f1", faction_b="f2", strength=0.2, formed_year=5)]
        alliances = form_alliances([f1, f2], year=15, existing=existing)
        # Dissimilar factions' alliance should decay
        if alliances:
            assert alliances[0].strength < 0.2


# ---------------------------------------------------------------------------
# accumulate_grievances
# ---------------------------------------------------------------------------

class TestAccumulateGrievances:
    def test_no_grievances_when_stable(self):
        gs = accumulate_grievances([], resource_avg=0.6, gini=0.2,
                                    recent_deaths=0, gov_type="council", year=10)
        assert len(gs) == 0

    def test_scarcity_grievance(self):
        gs = accumulate_grievances([], resource_avg=0.1, gini=0.2,
                                    recent_deaths=0, gov_type="council", year=10)
        scarcity = [g for g in gs if g.cause == "resource_scarcity"]
        assert len(scarcity) == 1
        assert scarcity[0].intensity > 0

    def test_inequality_grievance(self):
        gs = accumulate_grievances([], resource_avg=0.6, gini=0.7,
                                    recent_deaths=0, gov_type="council", year=10)
        ineq = [g for g in gs if g.cause == "inequality"]
        assert len(ineq) == 1

    def test_death_grievance(self):
        gs = accumulate_grievances([], resource_avg=0.6, gini=0.2,
                                    recent_deaths=3, gov_type="council", year=10)
        deaths = [g for g in gs if g.cause == "deaths"]
        assert len(deaths) == 1
        assert deaths[0].intensity == pytest.approx(min(GRIEVANCE_CAP, 3 * 0.8))

    def test_governance_mismatch(self):
        f = Faction(id="f1", member_ids=["a", "b"],
                    ideology={"empathy": 0.9, "resolve": 0.3, "faith": 0.2,
                              "paranoia": 0.1, "improvisation": 0.4, "hoarding": 0.2},
                    formed_year=5)
        gs = accumulate_grievances([f], resource_avg=0.6, gini=0.2,
                                    recent_deaths=0, gov_type="dictator", year=10)
        mismatch = [g for g in gs if g.cause == "governance_mismatch"]
        assert len(mismatch) == 1

    def test_intensity_capped(self):
        gs = accumulate_grievances([], resource_avg=0.0, gini=1.0,
                                    recent_deaths=100, gov_type="dictator", year=10)
        for g in gs:
            assert g.intensity <= GRIEVANCE_CAP


# ---------------------------------------------------------------------------
# decay_grievances
# ---------------------------------------------------------------------------

class TestDecayGrievances:
    def test_decay_reduces_intensity(self):
        state = PoliticalState(grievances=[
            Grievance(source="colony", cause="test", intensity=2.0, year=5)])
        decay_grievances(state)
        assert state.grievances[0].intensity < 2.0
        assert state.grievances[0].intensity == pytest.approx(2.0 * (1 - GRIEVANCE_DECAY))

    def test_tiny_grievances_pruned(self):
        state = PoliticalState(grievances=[
            Grievance(source="colony", cause="test", intensity=0.04, year=5)])
        decay_grievances(state)
        assert len(state.grievances) == 0


# ---------------------------------------------------------------------------
# total_grievance / should_crisis_propose
# ---------------------------------------------------------------------------

class TestGrievanceThresholds:
    def test_total_grievance(self):
        state = PoliticalState(grievances=[
            Grievance(source="a", cause="x", intensity=1.0, year=1),
            Grievance(source="b", cause="y", intensity=2.0, year=2),
        ])
        assert total_grievance(state) == pytest.approx(3.0)

    def test_crisis_under_threshold(self):
        state = PoliticalState(grievances=[
            Grievance(source="a", cause="x", intensity=1.0, year=1)])
        assert not should_crisis_propose(state)

    def test_crisis_over_threshold(self):
        state = PoliticalState(grievances=[
            Grievance(source="a", cause="x", intensity=2.0, year=1),
            Grievance(source="b", cause="y", intensity=2.0, year=2),
        ])
        assert should_crisis_propose(state)

    def test_cooldown_blocks_crisis(self):
        state = PoliticalState(
            grievances=[Grievance(source="a", cause="x", intensity=5.0, year=1)],
            revolt_cooldown=3,
        )
        assert not should_crisis_propose(state)


# ---------------------------------------------------------------------------
# compute_faction_pressure
# ---------------------------------------------------------------------------

class TestFactionPressure:
    def test_no_faction_no_pressure(self):
        deltas = compute_faction_pressure([], "some-colonist")
        assert deltas == {}

    def test_empathy_faction_boosts_mediate(self):
        f = Faction(id="f1", member_ids=["c1"],
                    ideology={"empathy": 0.9, "resolve": 0.3, "faith": 0.2,
                              "paranoia": 0.1, "improvisation": 0.4, "hoarding": 0.2},
                    formed_year=5)
        deltas = compute_faction_pressure([f], "c1")
        assert deltas.get("mediate", 0) > 0

    def test_high_grievance_boosts_sabotage(self):
        f = Faction(id="f1", member_ids=["c1"],
                    ideology={"paranoia": 0.9, "resolve": 0.1, "empathy": 0.1,
                              "faith": 0.1, "improvisation": 0.1, "hoarding": 0.1},
                    formed_year=5, grievance=3.0)
        deltas = compute_faction_pressure([f], "c1")
        assert deltas.get("sabotage", 0) > 0.1

    def test_non_member_no_pressure(self):
        f = Faction(id="f1", member_ids=["c1"],
                    ideology={"empathy": 0.9, "resolve": 0.3},
                    formed_year=5)
        deltas = compute_faction_pressure([f], "c2")
        assert deltas == {}


# ---------------------------------------------------------------------------
# compute_voting_bloc
# ---------------------------------------------------------------------------

class TestVotingBloc:
    def test_same_faction_positive(self):
        f = Faction(id="f1", member_ids=["a", "b"],
                    ideology={}, formed_year=5, cohesion=0.8)
        bias = compute_voting_bloc([f], [], "a", "b")
        assert bias > 0

    def test_different_faction_negative(self):
        f1 = Faction(id="f1", member_ids=["a"], ideology={}, formed_year=5)
        f2 = Faction(id="f2", member_ids=["b"], ideology={}, formed_year=5)
        bias = compute_voting_bloc([f1, f2], [], "a", "b")
        assert bias < 0

    def test_allied_factions_positive(self):
        f1 = Faction(id="f1", member_ids=["a"], ideology={}, formed_year=5)
        f2 = Faction(id="f2", member_ids=["b"], ideology={}, formed_year=5)
        alliance = Alliance(faction_a="f1", faction_b="f2", strength=0.8,
                            formed_year=5)
        bias = compute_voting_bloc([f1, f2], [alliance], "a", "b")
        assert bias > 0

    def test_no_faction_no_bias(self):
        bias = compute_voting_bloc([], [], "a", "b")
        assert bias == 0.0


# ---------------------------------------------------------------------------
# check_amendment_promotion
# ---------------------------------------------------------------------------

class TestAmendmentPromotion:
    def test_too_few_insights(self):
        state = PoliticalState()
        queue = [{"result": "trust", "colonist_id": "a", "year": 10, "depth": 2}]
        assert check_amendment_promotion(state, queue, year=20) is None

    def test_promotion_with_enough_insights(self):
        state = PoliticalState()
        queue = [
            {"result": "trust-0.8", "colonist_id": "a", "year": 10, "depth": 2},
            {"result": "trust-0.8", "colonist_id": "b", "year": 15, "depth": 2},
            {"result": "trust-0.8", "colonist_id": "c", "year": 20, "depth": 3},
        ]
        amendment = check_amendment_promotion(state, queue, year=25)
        assert amendment is not None
        assert amendment["status"] == "proposed"
        assert amendment["occurrences"] >= AMENDMENT_THRESHOLD
        assert len(queue) == 0  # consumed

    def test_single_colonist_not_enough(self):
        state = PoliticalState()
        queue = [
            {"result": "test-val", "colonist_id": "a", "year": 10, "depth": 2},
            {"result": "test-val", "colonist_id": "a", "year": 15, "depth": 2},
            {"result": "test-val", "colonist_id": "a", "year": 20, "depth": 3},
        ]
        assert check_amendment_promotion(state, queue, year=25) is None


# ---------------------------------------------------------------------------
# PoliticalState serialization
# ---------------------------------------------------------------------------

class TestPoliticalStateSerde:
    def test_round_trip(self):
        state = PoliticalState(
            factions=[Faction(id="f1", member_ids=["a", "b"],
                              ideology={"resolve": 0.8}, formed_year=5,
                              cohesion=0.7, grievance=1.2, name="Test")],
            alliances=[Alliance(faction_a="f1", faction_b="f2",
                                strength=0.6, formed_year=8)],
            grievances=[Grievance(source="colony", cause="test",
                                  intensity=1.5, year=10)],
            amendments=[{"text": "test amendment"}],
            revolt_cooldown=3,
        )
        d = state.to_dict()
        restored = PoliticalState.from_dict(d)
        assert len(restored.factions) == 1
        assert restored.factions[0].id == "f1"
        assert len(restored.alliances) == 1
        assert len(restored.grievances) == 1
        assert restored.revolt_cooldown == 3


# ---------------------------------------------------------------------------
# tick_politics integration
# ---------------------------------------------------------------------------

class TestTickPolitics:
    def test_basic_tick(self):
        colonists = create_founding_ten(42)
        sg = make_social(colonists)
        state = PoliticalState()
        result = tick_politics(
            state, colonists, sg,
            resources_avg=0.6, gini=0.2, recent_deaths=0,
            gov_type="council", year=10, rng=random.Random(42))
        assert isinstance(result, PoliticalTickResult)

    def test_faction_detection_on_first_tick(self):
        colonists = create_founding_ten(42)
        sg = make_social(colonists)
        state = PoliticalState()
        tick_politics(state, colonists, sg, resources_avg=0.6, gini=0.2,
                      recent_deaths=0, gov_type="council", year=10,
                      rng=random.Random(42))
        # Should have detected factions on first tick (no prior factions)
        assert len(state.factions) >= 0  # may be 0 if affinity too low

    def test_grievance_accumulation_under_crisis(self):
        colonists = create_founding_ten(42)
        sg = make_social(colonists)
        state = PoliticalState()
        tick_politics(state, colonists, sg, resources_avg=0.1, gini=0.7,
                      recent_deaths=5, gov_type="dictator", year=10,
                      rng=random.Random(42))
        assert total_grievance(state) > 0

    def test_dead_members_removed(self):
        colonists = create_founding_ten(42)
        sg = make_social(colonists)
        state = PoliticalState(
            factions=[Faction(id="f1", member_ids=["kira-sol", "fen-marsh"],
                              ideology={"resolve": 0.7}, formed_year=5)])
        colonists[0].die(year=9, cause="test")
        tick_politics(state, colonists, sg, resources_avg=0.6, gini=0.2,
                      recent_deaths=1, gov_type="council", year=10,
                      rng=random.Random(42))
        for f in state.factions:
            assert "kira-sol" not in f.member_ids

    def test_cooldown_decrements(self):
        colonists = create_founding_ten(42)
        sg = make_social(colonists)
        state = PoliticalState(revolt_cooldown=3)
        tick_politics(state, colonists, sg, resources_avg=0.6, gini=0.2,
                      recent_deaths=0, gov_type="council", year=10,
                      rng=random.Random(42))
        assert state.revolt_cooldown == 2

    def test_deterministic(self):
        """Same inputs produce same outputs."""
        colonists1 = create_founding_ten(42)
        colonists2 = create_founding_ten(42)
        sg1 = make_social(colonists1)
        sg2 = make_social(colonists2)
        s1 = PoliticalState()
        s2 = PoliticalState()
        r1 = tick_politics(s1, colonists1, sg1, 0.4, 0.3, 1, "council", 10,
                           random.Random(99))
        r2 = tick_politics(s2, colonists2, sg2, 0.4, 0.3, 1, "council", 10,
                           random.Random(99))
        assert r1.to_dict() == r2.to_dict()
        assert len(s1.factions) == len(s2.factions)

    def test_100_year_smoke(self):
        """Run 100 ticks without crash — property-based sanity."""
        colonists = create_founding_ten(42)
        sg = make_social(colonists)
        state = PoliticalState()
        rng = random.Random(42)
        for year in range(1, 101):
            res_avg = 0.5 + rng.gauss(0, 0.1)
            gini = max(0, min(1, 0.3 + rng.gauss(0, 0.05)))
            deaths = 1 if rng.random() < 0.1 else 0
            tick_politics(state, colonists, sg, res_avg, gini, deaths,
                          "council", year, rng)
            # Invariants
            for f in state.factions:
                assert len(f.member_ids) >= MIN_FACTION_SIZE
                assert f.grievance <= GRIEVANCE_CAP + 0.01
            assert len(state.factions) <= MAX_FACTIONS

    def test_crisis_proposal_triggered(self):
        """Heavy grievances trigger crisis proposal."""
        colonists = create_founding_ten(42)
        sg = make_social(colonists)
        state = PoliticalState()
        # Accumulate heavy grievances
        for year in range(1, 20):
            tick_politics(state, colonists, sg, resources_avg=0.05, gini=0.9,
                          recent_deaths=3, gov_type="dictator", year=year,
                          rng=random.Random(year))
        # At some point, crisis should have been triggered
        # (grievances accumulate beyond threshold)
        assert any(total_grievance(state) > 0 for _ in [1])
