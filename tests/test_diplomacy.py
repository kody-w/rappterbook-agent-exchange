"""Tests for Mars-100 diplomacy organ (engine v11.0)."""
from __future__ import annotations

import math
import random

import pytest

from src.mars100.diplomacy import (
    Faction, Treaty, DiplomacyState, DiplomacyTickResult,
    stat_vector, stat_similarity, trust_density,
    detect_clusters, match_faction, determine_ideology,
    elect_leader, compute_faction_cohesion,
    propose_treaty, resolve_incidents,
    compute_bloc_vote_influence, compute_diplomacy_pressure,
    compute_exile_modifier, compute_loneliness_reduction,
    compute_trade_pact_bonus, compute_fragmentation,
    tick_diplomacy,
    MIN_FACTION_SIZE, MAX_FACTIONS, SIMILARITY_THRESHOLD,
    TRUST_DENSITY_THRESHOLD, BLOC_VOTE_CAP,
    NON_AGGRESSION_EXILE_REDUCTION, TREATY_TYPES,
    MIN_POPULATION_FOR_FACTIONS,
)
from src.mars100.colonist import (
    Colonist, ColonistStats, ColonistSkills, STAT_NAMES,
)
from src.mars100.colony import Relationship


# --- helpers ---

def make_colonist(cid, dominant="resolve", val=0.8, base=0.3):
    stats = {name: base for name in STAT_NAMES}
    stats[dominant] = val
    skills = {name: 0.1 for name in
              ("terraforming", "hydroponics", "mediation",
               "coding", "prayer", "sabotage")}
    return Colonist(
        id=cid, name=f"Test-{cid}", element="fire", archetype="test",
        stats=ColonistStats.from_dict(stats),
        skills=ColonistSkills.from_dict(skills),
        decision_expr="(+ resolve 0.1)",
    )


def make_social_edges(ids, base_trust=0.5, rng=None):
    if rng is None:
        rng = random.Random(1)
    edges = {}
    for a in ids:
        edges[a] = {}
        for b in ids:
            if a != b:
                edges[a][b] = Relationship(
                    trust=base_trust + rng.gauss(0, 0.01),
                    affection=0.5, respect=0.5)
    return edges


# ===== stat_vector / stat_similarity =====

class TestStatVector:
    def test_vector_length(self):
        c = make_colonist("c0")
        v = stat_vector(c)
        assert len(v) == len(STAT_NAMES)

    def test_vector_values(self):
        c = make_colonist("c0", "empathy", val=0.9, base=0.2)
        v = stat_vector(c)
        assert v[STAT_NAMES.index("empathy")] == pytest.approx(0.9, abs=0.01)


class TestStatSimilarity:
    def test_identical_colonists(self):
        c = make_colonist("c0")
        assert stat_similarity(c, c) == pytest.approx(1.0, abs=0.01)

    def test_similar_colonists(self):
        a = make_colonist("a", "resolve", 0.9, 0.3)
        b = make_colonist("b", "resolve", 0.85, 0.32)
        sim = stat_similarity(a, b)
        assert 0.9 < sim <= 1.0

    def test_different_colonists(self):
        a = make_colonist("a", "resolve", 0.95, 0.1)
        b = make_colonist("b", "empathy", 0.95, 0.1)
        sim = stat_similarity(a, b)
        assert sim < 0.9

    def test_similarity_bounded(self):
        rng = random.Random(99)
        for _ in range(20):
            a = make_colonist("a", rng.choice(list(STAT_NAMES)))
            b = make_colonist("b", rng.choice(list(STAT_NAMES)))
            sim = stat_similarity(a, b)
            assert 0.0 <= sim <= 1.0

    def test_zero_vector(self):
        c = make_colonist("c0", "resolve", 0.0, 0.0)
        assert stat_similarity(c, c) == 0.0

    def test_symmetry(self):
        rng = random.Random(123)
        for _ in range(50):
            stats_a = {s: rng.random() for s in STAT_NAMES}
            stats_b = {s: rng.random() for s in STAT_NAMES}
            a = Colonist(id="a", name="A", element="fire", archetype="t",
                         stats=ColonistStats.from_dict(stats_a),
                         skills=ColonistSkills.from_dict({s: 0 for s in
                            ("terraforming","hydroponics","mediation",
                             "coding","prayer","sabotage")}),
                         decision_expr="0")
            b = Colonist(id="b", name="B", element="water", archetype="t",
                         stats=ColonistStats.from_dict(stats_b),
                         skills=ColonistSkills.from_dict({s: 0 for s in
                            ("terraforming","hydroponics","mediation",
                             "coding","prayer","sabotage")}),
                         decision_expr="0")
            assert stat_similarity(a, b) == pytest.approx(
                stat_similarity(b, a), abs=1e-10)


# ===== trust_density =====

class TestTrustDensity:
    def test_empty_group(self):
        assert trust_density([], {}) == 0.0

    def test_single_member(self):
        assert trust_density(["a"], {"a": {}}) == 0.0

    def test_high_trust_group(self):
        ids = ["a", "b", "c"]
        edges = make_social_edges(ids, base_trust=0.8)
        td = trust_density(ids, edges)
        assert td > 0.7

    def test_low_trust_group(self):
        ids = ["a", "b", "c"]
        edges = make_social_edges(ids, base_trust=0.1)
        td = trust_density(ids, edges)
        assert td < 0.2


# ===== detect_clusters =====

class TestDetectClusters:
    def test_too_few_colonists(self):
        colonists = [make_colonist(f"c{i}") for i in range(2)]
        edges = make_social_edges([c.id for c in colonists], 0.8)
        clusters = detect_clusters(colonists, edges, random.Random(42))
        assert clusters == []

    def test_homogeneous_group_forms_cluster(self):
        colonists = [make_colonist(f"c{i}", "resolve", 0.8, 0.3)
                     for i in range(5)]
        ids = [c.id for c in colonists]
        edges = make_social_edges(ids, base_trust=0.6)
        clusters = detect_clusters(colonists, edges, random.Random(42))
        assert len(clusters) >= 1
        assert all(len(c) >= MIN_FACTION_SIZE for c in clusters)

    def test_max_factions_respected(self):
        colonists = []
        for i, stat in enumerate(list(STAT_NAMES)[:5]):
            for j in range(3):
                colonists.append(make_colonist(f"g{i}-{j}", stat, 0.95, 0.1))
        ids = [c.id for c in colonists]
        edges = make_social_edges(ids, base_trust=0.6)
        clusters = detect_clusters(colonists, edges, random.Random(42))
        assert len(clusters) <= MAX_FACTIONS

    def test_clusters_disjoint(self):
        colonists = [make_colonist(f"c{i}", "resolve" if i < 4 else "empathy",
                                   0.9, 0.2) for i in range(8)]
        ids = [c.id for c in colonists]
        edges = make_social_edges(ids, base_trust=0.6)
        clusters = detect_clusters(colonists, edges, random.Random(42))
        all_members = []
        for cl in clusters:
            all_members.extend(cl)
        assert len(all_members) == len(set(all_members))


# ===== match_faction =====

class TestMatchFaction:
    def test_match_high_overlap(self):
        faction = Faction(id="f0", name="Test", ideology="resolve",
                          member_ids=["a", "b", "c"], founded_year=1)
        matched = match_faction(["a", "b", "c", "d"], [faction])
        assert matched is faction

    def test_no_match_low_overlap(self):
        faction = Faction(id="f0", name="Test", ideology="resolve",
                          member_ids=["a", "b", "c"], founded_year=1)
        matched = match_faction(["x", "y", "z"], [faction])
        assert matched is None

    def test_skip_dissolved(self):
        faction = Faction(id="f0", name="Test", ideology="resolve",
                          member_ids=["a", "b", "c"], founded_year=1,
                          dissolved_year=5)
        matched = match_faction(["a", "b", "c"], [faction])
        assert matched is None


# ===== determine_ideology =====

class TestDetermineIdeology:
    def test_dominant_stat(self):
        colonists = [make_colonist(f"c{i}", "faith", 0.9, 0.2) for i in range(3)]
        ideology = determine_ideology(colonists, [c.id for c in colonists])
        assert ideology == "faith"

    def test_mixed_stats(self):
        colonists = [
            make_colonist("a", "resolve", 0.9, 0.1),
            make_colonist("b", "resolve", 0.8, 0.1),
            make_colonist("c", "empathy", 0.7, 0.1),
        ]
        ideology = determine_ideology(colonists, [c.id for c in colonists])
        assert ideology == "resolve"


# ===== elect_leader =====

class TestElectLeader:
    def test_empty(self):
        assert elect_leader([], {}, random.Random(1)) is None

    def test_highest_respect_wins(self):
        ids = ["a", "b", "c"]
        edges = {}
        for cid in ids:
            edges[cid] = {}
        for voter in ids:
            for candidate in ids:
                if voter != candidate:
                    respect = 0.9 if candidate == "b" else 0.3
                    edges[voter][candidate] = Relationship(
                        trust=0.5, affection=0.5, respect=respect)
        leader = elect_leader(ids, edges, random.Random(1))
        assert leader == "b"


# ===== Faction / Treaty dataclass =====

class TestFaction:
    def test_is_active(self):
        f = Faction(id="f0", name="Test", ideology="resolve", founded_year=1)
        assert f.is_active()
        f.dissolved_year = 5
        assert not f.is_active()

    def test_roundtrip(self):
        f = Faction(id="f0", name="Test", ideology="empathy",
                    member_ids=["a","b"], leader_id="a",
                    cohesion=0.7, founded_year=3)
        d = f.to_dict()
        f2 = Faction.from_dict(d)
        assert f2.id == f.id
        assert f2.member_ids == f.member_ids
        assert f2.cohesion == pytest.approx(f.cohesion, abs=0.001)


class TestTreaty:
    def test_pair_key_sorted(self):
        t = Treaty(id="t0", treaty_type="alliance",
                   faction_a="f1", faction_b="f0",
                   start_year=1, duration=10)
        assert t.pair_key() == ("f0", "f1")

    def test_roundtrip(self):
        t = Treaty(id="t0", treaty_type="trade_pact",
                   faction_a="f0", faction_b="f1",
                   start_year=5, duration=10)
        d = t.to_dict()
        t2 = Treaty.from_dict(d)
        assert t2.treaty_type == "trade_pact"
        assert t2.duration == 10


# ===== DiplomacyState =====

class TestDiplomacyState:
    def test_faction_of(self):
        f = Faction(id="f0", name="T", ideology="r",
                    member_ids=["a","b"], founded_year=1)
        state = DiplomacyState(factions=[f])
        assert state.faction_of("a") is f
        assert state.faction_of("z") is None

    def test_treaty_between(self):
        t = Treaty(id="t0", treaty_type="alliance",
                   faction_a="f0", faction_b="f1",
                   start_year=1, duration=10)
        state = DiplomacyState(treaties=[t])
        assert state.treaty_between("f0", "f1") is t
        assert state.treaty_between("f1", "f0") is t
        assert state.treaty_between("f0", "f2") is None

    def test_treaty_type_filter(self):
        t = Treaty(id="t0", treaty_type="alliance",
                   faction_a="f0", faction_b="f1",
                   start_year=1, duration=10)
        state = DiplomacyState(treaties=[t])
        assert state.treaty_between("f0", "f1", "alliance") is t
        assert state.treaty_between("f0", "f1", "trade_pact") is None

    def test_roundtrip(self):
        f = Faction(id="f0", name="T", ideology="r",
                    member_ids=["a"], founded_year=1)
        t = Treaty(id="t0", treaty_type="alliance",
                   faction_a="f0", faction_b="f1",
                   start_year=1, duration=10)
        state = DiplomacyState(factions=[f], treaties=[t],
                               incidents=[{"year":1}], next_faction_id=1)
        d = state.to_dict()
        s2 = DiplomacyState.from_dict(d)
        assert len(s2.factions) == 1
        assert len(s2.treaties) == 1
        assert s2.next_faction_id == 1


# ===== propose_treaty =====

class TestProposeTreaty:
    def test_low_cohesion_rejected(self):
        fa = Faction(id="f0", name="A", ideology="r",
                     member_ids=["a"], leader_id="a",
                     cohesion=0.1, founded_year=1)
        fb = Faction(id="f1", name="B", ideology="e",
                     member_ids=["b"], leader_id="b",
                     cohesion=0.8, founded_year=1)
        state = DiplomacyState()
        result = propose_treaty(fa, fb, state, 10, random.Random(1))
        assert result is None

    def test_no_leader_rejected(self):
        fa = Faction(id="f0", name="A", ideology="r",
                     member_ids=["a"], leader_id=None,
                     cohesion=0.8, founded_year=1)
        fb = Faction(id="f1", name="B", ideology="e",
                     member_ids=["b"], leader_id="b",
                     cohesion=0.8, founded_year=1)
        state = DiplomacyState()
        result = propose_treaty(fa, fb, state, 10, random.Random(1))
        assert result is None

    def test_valid_treaty_proposed(self):
        fa = Faction(id="f0", name="A", ideology="r",
                     member_ids=["a"], leader_id="a",
                     cohesion=0.7, founded_year=1)
        fb = Faction(id="f1", name="B", ideology="e",
                     member_ids=["b"], leader_id="b",
                     cohesion=0.7, founded_year=1)
        state = DiplomacyState()
        result = propose_treaty(fa, fb, state, 10, random.Random(1))
        assert result is not None
        assert result.treaty_type in TREATY_TYPES
        assert result.active

    def test_no_duplicate_treaty_type(self):
        fa = Faction(id="f0", name="A", ideology="r",
                     member_ids=["a"], leader_id="a",
                     cohesion=0.7, founded_year=1)
        fb = Faction(id="f1", name="B", ideology="e",
                     member_ids=["b"], leader_id="b",
                     cohesion=0.7, founded_year=1)
        existing = [
            Treaty(id=f"t{i}", treaty_type=tt,
                   faction_a="f0", faction_b="f1",
                   start_year=1, duration=10)
            for i, tt in enumerate(TREATY_TYPES)
        ]
        state = DiplomacyState(treaties=existing)
        result = propose_treaty(fa, fb, state, 10, random.Random(1))
        assert result is None


# ===== bloc vote / diplomacy pressure / exile modifier =====

class TestBlocVoteInfluence:
    def test_no_faction(self):
        assert compute_bloc_vote_influence("c0", "council", None, 0.5) == 0.0

    def test_bounded(self):
        f = Faction(id="f0", name="T", ideology="empathy",
                    cohesion=1.0, founded_year=1)
        rng = random.Random(77)
        for _ in range(100):
            leader_pref = rng.uniform(-2, 2)
            gov_type = rng.choice(["council","dictator","anarchy",
                                   "lottery","consensus","ai_governor"])
            influence = compute_bloc_vote_influence("c0", gov_type, f, leader_pref)
            assert -BLOC_VOTE_CAP <= influence <= BLOC_VOTE_CAP

    def test_ideology_alignment(self):
        f = Faction(id="f0", name="T", ideology="empathy",
                    cohesion=0.8, founded_year=1)
        consensus_inf = compute_bloc_vote_influence("c0", "consensus", f, 0.5)
        dictator_inf = compute_bloc_vote_influence("c0", "dictator", f, 0.5)
        assert consensus_inf > dictator_inf


class TestDiplomacyPressure:
    def test_no_faction(self):
        state = DiplomacyState()
        actions = ["cooperate", "sabotage", "rest"]
        pressures = compute_diplomacy_pressure(state, "c0", actions)
        assert all(v == 0.0 for v in pressures.values())

    def test_faction_boosts_cooperate(self):
        f = Faction(id="f0", name="T", ideology="resolve",
                    member_ids=["c0"], cohesion=0.8, founded_year=1)
        state = DiplomacyState(factions=[f])
        actions = ["cooperate", "sabotage", "rest"]
        pressures = compute_diplomacy_pressure(state, "c0", actions)
        assert pressures["cooperate"] > 0
        assert pressures["sabotage"] < 0


class TestExileModifier:
    def test_no_faction(self):
        state = DiplomacyState()
        assert compute_exile_modifier(state, "a", "b") == 1.0

    def test_same_faction(self):
        f = Faction(id="f0", name="T", ideology="r",
                    member_ids=["a","b"], founded_year=1)
        state = DiplomacyState(factions=[f])
        mod = compute_exile_modifier(state, "a", "b")
        assert mod < 1.0

    def test_non_aggression(self):
        f0 = Faction(id="f0", name="A", ideology="r",
                     member_ids=["a"], founded_year=1)
        f1 = Faction(id="f1", name="B", ideology="e",
                     member_ids=["b"], founded_year=1)
        t = Treaty(id="t0", treaty_type="non_aggression",
                   faction_a="f0", faction_b="f1",
                   start_year=1, duration=10)
        state = DiplomacyState(factions=[f0, f1], treaties=[t])
        mod = compute_exile_modifier(state, "a", "b")
        assert mod == pytest.approx(1.0 - NON_AGGRESSION_EXILE_REDUCTION)


# ===== loneliness / trade / fragmentation =====

class TestLonelinessReduction:
    def test_no_faction(self):
        state = DiplomacyState()
        assert compute_loneliness_reduction(state, "c0") == 0.0

    def test_larger_faction_more_reduction(self):
        f_small = Faction(id="f0", name="T", ideology="r",
                          member_ids=["a","b","c"], cohesion=0.8, founded_year=1)
        f_large = Faction(id="f1", name="U", ideology="e",
                          member_ids=["d","e","f","g","h"],
                          cohesion=0.8, founded_year=1)
        r_small = compute_loneliness_reduction(DiplomacyState(factions=[f_small]), "a")
        r_large = compute_loneliness_reduction(DiplomacyState(factions=[f_large]), "d")
        assert r_large > r_small

    def test_bounded(self):
        f = Faction(id="f0", name="T", ideology="r",
                    member_ids=[f"c{i}" for i in range(20)],
                    cohesion=1.0, founded_year=1)
        state = DiplomacyState(factions=[f])
        r = compute_loneliness_reduction(state, "c0")
        assert 0.0 <= r <= 0.06


class TestTradePactBonus:
    def test_no_treaties(self):
        assert compute_trade_pact_bonus(DiplomacyState()) == {}

    def test_with_trade_pacts(self):
        t = Treaty(id="t0", treaty_type="trade_pact",
                   faction_a="f0", faction_b="f1",
                   start_year=1, duration=10)
        bonuses = compute_trade_pact_bonus(DiplomacyState(treaties=[t]))
        assert bonuses.get("food", 0) > 0
        assert bonuses.get("water", 0) > 0


class TestFragmentation:
    def test_no_factions(self):
        assert compute_fragmentation(DiplomacyState(), 10) == 0.0

    def test_one_faction(self):
        f = Faction(id="f0", name="T", ideology="r",
                    member_ids=["a","b","c"], founded_year=1)
        assert compute_fragmentation(DiplomacyState(factions=[f]), 10) == 0.0

    def test_bounded(self):
        factions = [
            Faction(id=f"f{i}", name=f"F{i}", ideology="r",
                    member_ids=[f"g{i}-{j}" for j in range(5)],
                    founded_year=1)
            for i in range(4)
        ]
        frag = compute_fragmentation(DiplomacyState(factions=factions), 20)
        assert 0.0 <= frag <= 1.0


# ===== resolve_incidents =====

class TestResolveIncidents:
    def test_no_sabotage_no_guaranteed_incidents(self):
        factions = [Faction(id="f0", name="T", ideology="r",
                            member_ids=["a"], founded_year=1)]
        incidents = resolve_incidents(factions, [], {"a": "farm"}, 1,
                                      random.Random(42))
        assert isinstance(incidents, list)

    def test_sabotage_may_create_incident(self):
        f0 = Faction(id="f0", name="A", ideology="r",
                     member_ids=["a"], founded_year=1)
        f1 = Faction(id="f1", name="B", ideology="e",
                     member_ids=["b"], founded_year=1)
        found = False
        for seed in range(100):
            incidents = resolve_incidents(
                [f0, f1], [], {"a": "sabotage", "b": "farm"}, 1,
                random.Random(seed))
            if incidents:
                found = True
                break
        assert found


# ===== tick_diplomacy integration =====

class TestTickDiplomacy:
    def _make_population(self, n, dominant="resolve", trust=0.6):
        colonists = [make_colonist(f"c{i}", dominant, 0.8, 0.3)
                     for i in range(n)]
        ids = [c.id for c in colonists]
        edges = make_social_edges(ids, base_trust=trust)
        return colonists, edges

    def test_no_factions_small_population(self):
        colonists, edges = self._make_population(5)
        state = DiplomacyState()
        tick_diplomacy(state, colonists, edges, [], {}, 1, random.Random(42))
        assert len(state.active_factions()) == 0

    def test_faction_forms_large_population(self):
        colonists, edges = self._make_population(
            MIN_POPULATION_FOR_FACTIONS + 2, trust=0.6)
        state = DiplomacyState()
        result = tick_diplomacy(state, colonists, edges, [], {}, 1,
                                random.Random(42))
        assert len(result.factions_formed) >= 1

    def test_dead_colonists_pruned(self):
        colonists, edges = self._make_population(10, trust=0.6)
        state = DiplomacyState()
        tick_diplomacy(state, colonists, edges, [], {}, 1, random.Random(42))
        active = state.active_factions()
        if active:
            victim = active[0].member_ids[0]
            for c in colonists:
                if c.id == victim:
                    c.alive = False
            tick_diplomacy(state, colonists, edges, [], {}, 2, random.Random(43))
            for f in state.active_factions():
                assert victim not in f.member_ids

    def test_faction_dissolves_when_undersized(self):
        f = Faction(id="f0", name="T", ideology="r",
                    member_ids=["c0","c1"], leader_id="c0",
                    cohesion=0.5, founded_year=1)
        state = DiplomacyState(factions=[f], next_faction_id=1)
        colonists = [make_colonist("c0"), make_colonist("c1")]
        edges = make_social_edges(["c0","c1"])
        result = tick_diplomacy(state, colonists, edges, [], {}, 5,
                                random.Random(42))
        assert len(result.factions_dissolved) == 1
        assert not f.is_active()

    def test_treaties_expire(self):
        f0 = Faction(id="f0", name="A", ideology="r",
                     member_ids=["a","b","c"], leader_id="a",
                     cohesion=0.7, founded_year=1)
        f1 = Faction(id="f1", name="B", ideology="e",
                     member_ids=["d","e","f"], leader_id="d",
                     cohesion=0.7, founded_year=1)
        t = Treaty(id="t0", treaty_type="alliance",
                   faction_a="f0", faction_b="f1",
                   start_year=1, duration=5)
        state = DiplomacyState(factions=[f0, f1], treaties=[t],
                               next_faction_id=2)
        colonists = [make_colonist(cid) for cid in
                     ["a","b","c","d","e","f"]]
        edges = make_social_edges([c.id for c in colonists])
        result = tick_diplomacy(state, colonists, edges, [], {}, 7,
                                random.Random(42))
        assert not t.active
        assert len(result.treaties_expired) >= 1

    def test_colonist_in_at_most_one_faction(self):
        colonists, edges = self._make_population(15, trust=0.6)
        state = DiplomacyState()
        for year in range(1, 20):
            tick_diplomacy(state, colonists, edges, [], {}, year,
                           random.Random(42 + year))
        seen = set()
        for f in state.active_factions():
            for mid in f.member_ids:
                assert mid not in seen, f"{mid} in multiple factions"
                seen.add(mid)

    def test_leader_is_member(self):
        colonists, edges = self._make_population(12, trust=0.6)
        state = DiplomacyState()
        for year in range(1, 15):
            tick_diplomacy(state, colonists, edges, [], {}, year,
                           random.Random(42 + year))
        for f in state.active_factions():
            if f.leader_id is not None:
                assert f.leader_id in f.member_ids

    def test_cohesion_bounded(self):
        colonists, edges = self._make_population(12, trust=0.6)
        state = DiplomacyState()
        for year in range(1, 50):
            tick_diplomacy(state, colonists, edges, [], {}, year,
                           random.Random(42 + year))
        for f in state.factions:
            assert 0.0 <= f.cohesion <= 1.0

    def test_deterministic(self):
        colonists, edges = self._make_population(12, trust=0.6)
        results = []
        for _ in range(2):
            state = DiplomacyState()
            for year in range(1, 10):
                tick_diplomacy(state, colonists, edges, [], {}, year,
                               random.Random(42 + year))
            results.append(state.to_dict())
        assert results[0] == results[1]


# ===== 100-year smoke test =====

class TestSmokeDiplomacy:
    def test_100_year_run(self):
        rng = random.Random(42)
        colonists = [make_colonist(f"c{i}",
                     rng.choice(list(STAT_NAMES)), 0.8, 0.3)
                     for i in range(15)]
        ids = [c.id for c in colonists]
        edges = make_social_edges(ids, base_trust=0.5, rng=random.Random(1))
        state = DiplomacyState()
        for year in range(1, 101):
            actions = {c.id: rng.choice(["farm","cooperate","sabotage",
                                         "rest","terraform"])
                       for c in colonists if c.is_active()}
            tick_diplomacy(state, colonists, edges,
                           [{"severity": rng.random(), "name": "test"}],
                           actions, year, random.Random(42 + year))
            seen = set()
            for f in state.active_factions():
                assert 0.0 <= f.cohesion <= 1.0
                for mid in f.member_ids:
                    assert mid not in seen
                    seen.add(mid)
                if f.leader_id is not None:
                    assert f.leader_id in f.member_ids
        assert len(state.factions) >= 1
