"""Tests for the diplomacy organ (engine v11.0)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.diplomacy import (
    FACTION_MIN_SIZE, FACTION_TRUST_THRESHOLD, IDEOLOGY_HYSTERESIS_MARGIN,
    IDEOLOGY_NAMES, MAX_BLOC_PRESSURE, MAX_VOTE_BIAS,
    SCHISM_COHESION_THRESHOLD, SCHISM_MIN_SIZE,
    Alliance, DiplomacyState, DiplomacyTickResult, Faction,
    classify_ideology, compute_bloc_pressure, compute_faction_vote_bias,
    tick_diplomacy, _ideology_compatibility, _ideology_scores, _tension_key,
    _avg_pair_trust,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeRelationship:
    """Minimal relationship stub for testing."""
    def __init__(self, trust: float = 0.5, affection: float = 0.5):
        self.trust = trust
        self.affection = affection


def make_social_get(default_trust: float = 0.5):
    """Return a social_get callable with uniform trust."""
    def _get(a: str, b: str) -> FakeRelationship:
        return FakeRelationship(trust=default_trust)
    return _get


def make_colonist_dict(cid: str, *,
                        empathy: float = 0.5, resolve: float = 0.5,
                        faith: float = 0.5, paranoia: float = 0.5,
                        hoarding: float = 0.5, improvisation: float = 0.5,
                        coding: float = 0.0, mediation: float = 0.0,
                        prayer: float = 0.0, sabotage: float = 0.0,
                        terraforming: float = 0.0, hydroponics: float = 0.0,
                        ) -> dict:
    return {
        "id": cid,
        "stats": {
            "empathy": empathy, "resolve": resolve, "faith": faith,
            "paranoia": paranoia, "hoarding": hoarding,
            "improvisation": improvisation,
        },
        "skills": {
            "coding": coding, "mediation": mediation, "prayer": prayer,
            "sabotage": sabotage, "terraforming": terraforming,
            "hydroponics": hydroponics,
        },
    }


# ---------------------------------------------------------------------------
# Ideology classification
# ---------------------------------------------------------------------------

class TestIdeologyClassification:
    def test_cooperative_dominant(self):
        ideo = classify_ideology(
            {"empathy": 0.9, "paranoia": 0.1, "faith": 0.3,
             "hoarding": 0.1, "resolve": 0.3, "improvisation": 0.3},
            {"mediation": 0.8, "coding": 0.0, "prayer": 0.0,
             "sabotage": 0.0, "terraforming": 0.0, "hydroponics": 0.0})
        assert ideo == "cooperative"

    def test_survivalist_dominant(self):
        ideo = classify_ideology(
            {"paranoia": 0.9, "hoarding": 0.8, "resolve": 0.7,
             "empathy": 0.1, "faith": 0.1, "improvisation": 0.2},
            {"sabotage": 0.6, "coding": 0.0, "mediation": 0.0,
             "prayer": 0.0, "terraforming": 0.0, "hydroponics": 0.0})
        assert ideo == "survivalist"

    def test_spiritual_dominant(self):
        ideo = classify_ideology(
            {"faith": 0.9, "empathy": 0.6, "paranoia": 0.1,
             "hoarding": 0.2, "resolve": 0.3, "improvisation": 0.2},
            {"prayer": 0.8, "coding": 0.0, "mediation": 0.0,
             "sabotage": 0.0, "terraforming": 0.0, "hydroponics": 0.0})
        assert ideo == "spiritual"

    def test_technocratic_dominant(self):
        ideo = classify_ideology(
            {"improvisation": 0.8, "faith": 0.1, "paranoia": 0.2,
             "empathy": 0.3, "hoarding": 0.2, "resolve": 0.4},
            {"coding": 0.9, "terraforming": 0.7, "mediation": 0.0,
             "prayer": 0.0, "sabotage": 0.0, "hydroponics": 0.0})
        assert ideo == "technocratic"

    def test_isolationist_dominant(self):
        ideo = classify_ideology(
            {"hoarding": 0.8, "paranoia": 0.7, "empathy": 0.1,
             "faith": 0.6, "resolve": 0.3, "improvisation": 0.2},
            {"coding": 0.0, "mediation": 0.0, "prayer": 0.0,
             "sabotage": 0.0, "terraforming": 0.0, "hydroponics": 0.0})
        assert ideo == "isolationist"

    def test_hysteresis_keeps_prior(self):
        """Small score difference should NOT change ideology."""
        ideo = classify_ideology(
            {"empathy": 0.5, "paranoia": 0.3, "faith": 0.2,
             "hoarding": 0.2, "resolve": 0.4, "improvisation": 0.55},
            {"mediation": 0.15, "coding": 0.45, "prayer": 0.0,
             "sabotage": 0.0, "terraforming": 0.25, "hydroponics": 0.0},
            prior="technocratic")
        assert ideo == "technocratic"

    def test_hysteresis_switches_on_large_diff(self):
        """Large score difference should change ideology."""
        ideo = classify_ideology(
            {"empathy": 0.95, "paranoia": 0.0, "faith": 0.1,
             "hoarding": 0.0, "resolve": 0.2, "improvisation": 0.1},
            {"mediation": 0.9, "coding": 0.0, "prayer": 0.0,
             "sabotage": 0.0, "terraforming": 0.0, "hydroponics": 0.0},
            prior="survivalist")
        assert ideo == "cooperative"

    def test_all_ideologies_reachable(self):
        """Every ideology can be the result of classification."""
        reached = set()
        profiles = [
            ({"empathy": 0.9, "paranoia": 0.0, "faith": 0.0,
              "hoarding": 0.0, "resolve": 0.0, "improvisation": 0.0},
             {"mediation": 0.9}),
            ({"paranoia": 0.9, "hoarding": 0.9, "resolve": 0.8,
              "empathy": 0.0, "faith": 0.0, "improvisation": 0.0},
             {"sabotage": 0.8}),
            ({"faith": 0.95, "empathy": 0.5, "paranoia": 0.0,
              "hoarding": 0.0, "resolve": 0.0, "improvisation": 0.0},
             {"prayer": 0.9}),
            ({"improvisation": 0.9, "faith": 0.0, "paranoia": 0.0,
              "empathy": 0.0, "hoarding": 0.0, "resolve": 0.0},
             {"coding": 0.9, "terraforming": 0.8}),
            ({"hoarding": 0.9, "paranoia": 0.7, "empathy": 0.0,
              "faith": 0.6, "resolve": 0.0, "improvisation": 0.0},
             {"coding": 0.0}),
        ]
        for stats, skills in profiles:
            full_skills = {k: skills.get(k, 0.0) for k in
                           ["coding", "mediation", "prayer", "sabotage",
                            "terraforming", "hydroponics"]}
            reached.add(classify_ideology(stats, full_skills))
        assert reached == set(IDEOLOGY_NAMES)


# ---------------------------------------------------------------------------
# Ideology scores
# ---------------------------------------------------------------------------

class TestIdeologyScores:
    def test_scores_bounded(self):
        """All ideology scores should be in [0, 1]."""
        rng = random.Random(99)
        for _ in range(200):
            stats = {s: rng.random() for s in
                     ["empathy", "resolve", "faith", "paranoia",
                      "hoarding", "improvisation"]}
            skills = {s: rng.random() for s in
                      ["coding", "mediation", "prayer", "sabotage",
                       "terraforming", "hydroponics"]}
            scores = _ideology_scores(stats, skills)
            for ideo, score in scores.items():
                assert 0.0 <= score <= 1.0, f"{ideo}={score}"


# ---------------------------------------------------------------------------
# Faction data structures
# ---------------------------------------------------------------------------

class TestFaction:
    def test_to_from_dict_roundtrip(self):
        f = Faction(id="f-0", name="Red Compact", ideology="cooperative",
                    members=["a", "b", "c"], leader_id="a",
                    founding_year=10, cohesion=0.6, influence=0.3)
        d = f.to_dict()
        f2 = Faction.from_dict(d)
        assert f2.id == f.id
        assert f2.members == f.members
        assert f2.cohesion == f.cohesion

    def test_empty_faction(self):
        f = Faction(id="f-x", name="X", ideology="spiritual",
                    members=[], leader_id=None, founding_year=1)
        d = f.to_dict()
        assert d["members"] == []
        assert d["leader_id"] is None


class TestAlliance:
    def test_pair_key_normalized(self):
        a = Alliance(faction_a="f-2", faction_b="f-1",
                     strength=0.5, formed_year=5)
        assert a.pair_key() == ("f-1", "f-2")

    def test_to_from_dict(self):
        a = Alliance(faction_a="f-0", faction_b="f-1",
                     strength=0.7, formed_year=10)
        d = a.to_dict()
        a2 = Alliance.from_dict(d)
        assert a2.faction_a == a.faction_a
        assert abs(a2.strength - a.strength) < 1e-6


class TestDiplomacyState:
    def test_to_from_dict_roundtrip(self):
        state = DiplomacyState()
        f = Faction(id="f-0", name="Test", ideology="cooperative",
                    members=["a", "b", "c"], leader_id="a",
                    founding_year=5)
        state.factions["f-0"] = f
        state.alliances.append(
            Alliance(faction_a="f-0", faction_b="f-1",
                     strength=0.5, formed_year=6))
        state.tensions["f-0:f-1"] = 0.3
        state.next_faction_id = 2
        d = state.to_dict()
        s2 = DiplomacyState.from_dict(d)
        assert len(s2.factions) == 1
        assert s2.factions["f-0"].name == "Test"
        assert len(s2.alliances) == 1
        assert s2.next_faction_id == 2


# ---------------------------------------------------------------------------
# Tension key
# ---------------------------------------------------------------------------

class TestTensionKey:
    def test_normalized(self):
        assert _tension_key("b", "a") == "a:b"
        assert _tension_key("a", "b") == "a:b"

    def test_same(self):
        assert _tension_key("x", "x") == "x:x"


# ---------------------------------------------------------------------------
# Ideology compatibility
# ---------------------------------------------------------------------------

class TestIdeologyCompatibility:
    def test_same_ideology(self):
        assert _ideology_compatibility("cooperative", "cooperative") == 0.8

    def test_opposed_ideologies(self):
        compat = _ideology_compatibility("cooperative", "isolationist")
        assert compat < 0.2

    def test_symmetric(self):
        assert (_ideology_compatibility("spiritual", "cooperative")
                == _ideology_compatibility("cooperative", "spiritual"))


# ---------------------------------------------------------------------------
# Average pair trust
# ---------------------------------------------------------------------------

class TestAvgPairTrust:
    def test_single_member(self):
        assert _avg_pair_trust(["a"], make_social_get(0.8)) == 0.0

    def test_two_members(self):
        trust = _avg_pair_trust(["a", "b"], make_social_get(0.7))
        assert abs(trust - 0.7) < 1e-6

    def test_three_members(self):
        trust = _avg_pair_trust(["a", "b", "c"], make_social_get(0.6))
        assert abs(trust - 0.6) < 1e-6


# ---------------------------------------------------------------------------
# Faction formation via tick_diplomacy
# ---------------------------------------------------------------------------

class TestFactionFormation:
    def test_no_factions_with_few_colonists(self):
        """Fewer than FACTION_MIN_SIZE colonists → no factions."""
        state = DiplomacyState()
        colonists = [make_colonist_dict(f"c-{i}") for i in range(2)]
        result = tick_diplomacy(state, colonists, make_social_get(),
                                {}, year=5, rng=random.Random(42))
        assert result.faction_count == 0

    def test_factions_form_with_matching_ideology(self):
        """3+ cooperative colonists with high trust → faction forms."""
        state = DiplomacyState()
        # Pre-seed ideology cache with 2+ years of stability
        for i in range(4):
            state.ideology_cache[f"c-{i}"] = "cooperative"
            state.ideology_age[f"c-{i}"] = 3
        colonists = [
            make_colonist_dict(f"c-{i}", empathy=0.9, paranoia=0.1,
                               mediation=0.8)
            for i in range(4)
        ]
        result = tick_diplomacy(state, colonists, make_social_get(0.5),
                                {}, year=10, rng=random.Random(42))
        assert result.faction_count >= 1
        assert len(result.factions_formed) >= 1

    def test_no_faction_with_low_trust(self):
        """Matching ideology but low trust → no faction."""
        state = DiplomacyState()
        for i in range(4):
            state.ideology_cache[f"c-{i}"] = "cooperative"
            state.ideology_age[f"c-{i}"] = 3
        colonists = [
            make_colonist_dict(f"c-{i}", empathy=0.9, paranoia=0.1,
                               mediation=0.8)
            for i in range(4)
        ]
        result = tick_diplomacy(state, colonists, make_social_get(0.1),
                                {}, year=10, rng=random.Random(42))
        assert result.faction_count == 0

    def test_no_faction_with_unstable_ideology(self):
        """Ideology not stable for 2 years → no faction."""
        state = DiplomacyState()
        for i in range(4):
            state.ideology_cache[f"c-{i}"] = "cooperative"
            state.ideology_age[f"c-{i}"] = 0  # fresh
        colonists = [
            make_colonist_dict(f"c-{i}", empathy=0.9, paranoia=0.1,
                               mediation=0.8)
            for i in range(4)
        ]
        result = tick_diplomacy(state, colonists, make_social_get(0.5),
                                {}, year=10, rng=random.Random(42))
        assert result.faction_count == 0

    def test_already_affiliated_skipped(self):
        """Colonists already in a faction don't form a new one."""
        state = DiplomacyState()
        existing = Faction(id="f-0", name="Old", ideology="cooperative",
                           members=["c-0", "c-1", "c-2"],
                           leader_id="c-0", founding_year=5)
        state.factions["f-0"] = existing
        for i in range(5):
            state.ideology_cache[f"c-{i}"] = "cooperative"
            state.ideology_age[f"c-{i}"] = 5
        colonists = [make_colonist_dict(f"c-{i}", empathy=0.9) for i in range(5)]
        result = tick_diplomacy(state, colonists, make_social_get(0.5),
                                {}, year=15, rng=random.Random(42))
        # Only 2 unaffiliated, can't form a new faction
        assert len(result.factions_formed) == 0


# ---------------------------------------------------------------------------
# Faction pruning
# ---------------------------------------------------------------------------

class TestFactionPruning:
    def test_dead_members_removed(self):
        """Dead colonists pruned from factions."""
        state = DiplomacyState()
        f = Faction(id="f-0", name="Test", ideology="cooperative",
                    members=["c-0", "c-1", "c-2", "c-dead"],
                    leader_id="c-0", founding_year=5)
        state.factions["f-0"] = f
        active = [make_colonist_dict(f"c-{i}") for i in range(3)]
        tick_diplomacy(state, active, make_social_get(), {}, year=10,
                       rng=random.Random(42))
        assert "c-dead" not in state.factions["f-0"].members

    def test_faction_dissolves_when_too_small(self):
        """Faction with <2 active members dissolves."""
        state = DiplomacyState()
        f = Faction(id="f-0", name="Tiny", ideology="cooperative",
                    members=["c-0", "c-dead-1", "c-dead-2"],
                    leader_id="c-0", founding_year=5)
        state.factions["f-0"] = f
        active = [make_colonist_dict("c-0"),
                  make_colonist_dict("c-1"),
                  make_colonist_dict("c-2")]
        result = tick_diplomacy(state, active, make_social_get(), {},
                                year=10, rng=random.Random(42))
        assert "f-0" not in state.factions
        assert len(result.factions_dissolved) >= 1


# ---------------------------------------------------------------------------
# Schism
# ---------------------------------------------------------------------------

class TestSchism:
    def test_schism_on_low_cohesion(self):
        """Large faction with low cohesion splits."""
        state = DiplomacyState()
        members = [f"c-{i}" for i in range(6)]
        f = Faction(id="f-0", name="Fractured", ideology="cooperative",
                    members=members, leader_id="c-0",
                    founding_year=5, cohesion=0.1)
        state.factions["f-0"] = f
        for cid in members:
            state.ideology_cache[cid] = "cooperative"
            state.ideology_age[cid] = 5
        colonists = [make_colonist_dict(cid) for cid in members]
        result = tick_diplomacy(state, colonists, make_social_get(0.1),
                                {}, year=15, rng=random.Random(42))
        assert len(result.schisms) >= 1
        assert len(state.factions) >= 2

    def test_no_schism_on_small_faction(self):
        """Faction below SCHISM_MIN_SIZE doesn't split."""
        state = DiplomacyState()
        members = [f"c-{i}" for i in range(3)]
        f = Faction(id="f-0", name="Small", ideology="cooperative",
                    members=members, leader_id="c-0",
                    founding_year=5, cohesion=0.1)
        state.factions["f-0"] = f
        colonists = [make_colonist_dict(cid) for cid in members]
        result = tick_diplomacy(state, colonists, make_social_get(0.1),
                                {}, year=15, rng=random.Random(42))
        assert len(result.schisms) == 0


# ---------------------------------------------------------------------------
# Alliances
# ---------------------------------------------------------------------------

class TestAlliances:
    def test_alliance_forms_between_compatible_factions(self):
        """Compatible factions with low tension may form alliance."""
        state = DiplomacyState()
        f1 = Faction(id="f-0", name="A", ideology="cooperative",
                     members=["c-0", "c-1", "c-2"], leader_id="c-0",
                     founding_year=5)
        f2 = Faction(id="f-1", name="B", ideology="spiritual",
                     members=["c-3", "c-4", "c-5"], leader_id="c-3",
                     founding_year=6)
        state.factions["f-0"] = f1
        state.factions["f-1"] = f2
        for cid in f1.members + f2.members:
            state.ideology_cache[cid] = "cooperative" if cid in f1.members else "spiritual"
            state.ideology_age[cid] = 5
        colonists = [make_colonist_dict(cid) for cid in f1.members + f2.members]
        # Run many times to get probabilistic alliance
        formed = False
        for seed in range(100):
            s = DiplomacyState.from_dict(state.to_dict())
            s.factions["f-0"] = Faction.from_dict(f1.to_dict())
            s.factions["f-1"] = Faction.from_dict(f2.to_dict())
            r = tick_diplomacy(s, colonists, make_social_get(0.5),
                               {}, year=20, rng=random.Random(seed))
            if len(r.alliances_formed) > 0:
                formed = True
                break
        assert formed, "Alliance should form between compatible factions"


# ---------------------------------------------------------------------------
# Bloc pressure
# ---------------------------------------------------------------------------

class TestBlocPressure:
    def test_no_pressure_without_faction(self):
        state = DiplomacyState()
        pressure = compute_bloc_pressure(state, "c-0",
                                          ["cooperate", "hoard", "pray"])
        assert pressure == {}

    def test_pressure_capped(self):
        state = DiplomacyState()
        f = Faction(id="f-0", name="Test", ideology="cooperative",
                    members=["c-0", "c-1", "c-2"], leader_id="c-0",
                    founding_year=5, cohesion=1.0, influence=1.0)
        state.factions["f-0"] = f
        pressure = compute_bloc_pressure(
            state, "c-0",
            ["cooperate", "hoard", "sabotage", "mediate"])
        for v in pressure.values():
            assert abs(v) <= MAX_BLOC_PRESSURE + 1e-9

    def test_cooperative_encourages_cooperate(self):
        state = DiplomacyState()
        f = Faction(id="f-0", name="Test", ideology="cooperative",
                    members=["c-0"], leader_id="c-0",
                    founding_year=5, cohesion=0.8, influence=0.5)
        state.factions["f-0"] = f
        pressure = compute_bloc_pressure(state, "c-0",
                                          ["cooperate", "sabotage"])
        assert pressure.get("cooperate", 0) > 0
        assert pressure.get("sabotage", 0) < 0

    def test_low_cohesion_no_pressure(self):
        """Faction with cohesion < 0.3 exerts no pressure."""
        state = DiplomacyState()
        f = Faction(id="f-0", name="Weak", ideology="survivalist",
                    members=["c-0"], leader_id="c-0",
                    founding_year=5, cohesion=0.2, influence=0.5)
        state.factions["f-0"] = f
        pressure = compute_bloc_pressure(state, "c-0", ["hoard"])
        assert pressure == {}


# ---------------------------------------------------------------------------
# Vote bias
# ---------------------------------------------------------------------------

class TestFactionVoteBias:
    def test_same_faction_positive_bias(self):
        state = DiplomacyState()
        f = Faction(id="f-0", name="Test", ideology="cooperative",
                    members=["c-0", "c-1"], leader_id="c-0",
                    founding_year=5, cohesion=0.8)
        state.factions["f-0"] = f
        bias = compute_faction_vote_bias(state, "c-0", "c-1")
        assert bias > 0

    def test_rival_faction_negative_bias(self):
        state = DiplomacyState()
        f1 = Faction(id="f-0", name="A", ideology="cooperative",
                     members=["c-0"], leader_id="c-0", founding_year=5)
        f2 = Faction(id="f-1", name="B", ideology="survivalist",
                     members=["c-1"], leader_id="c-1", founding_year=5)
        state.factions["f-0"] = f1
        state.factions["f-1"] = f2
        state.tensions["f-0:f-1"] = 0.8
        bias = compute_faction_vote_bias(state, "c-0", "c-1")
        assert bias < 0

    def test_no_faction_no_bias(self):
        state = DiplomacyState()
        bias = compute_faction_vote_bias(state, "c-0", "c-1")
        assert bias == 0.0

    def test_allied_factions_positive_bias(self):
        state = DiplomacyState()
        f1 = Faction(id="f-0", name="A", ideology="cooperative",
                     members=["c-0"], leader_id="c-0", founding_year=5)
        f2 = Faction(id="f-1", name="B", ideology="spiritual",
                     members=["c-1"], leader_id="c-1", founding_year=5)
        state.factions["f-0"] = f1
        state.factions["f-1"] = f2
        state.alliances.append(Alliance(
            faction_a="f-0", faction_b="f-1", strength=0.6, formed_year=8))
        bias = compute_faction_vote_bias(state, "c-0", "c-1")
        assert bias > 0

    def test_bias_bounded(self):
        """Vote bias never exceeds MAX_VOTE_BIAS."""
        state = DiplomacyState()
        f = Faction(id="f-0", name="Max", ideology="cooperative",
                    members=["c-0", "c-1"], leader_id="c-0",
                    founding_year=5, cohesion=1.0)
        state.factions["f-0"] = f
        bias = compute_faction_vote_bias(state, "c-0", "c-1")
        assert abs(bias) <= MAX_VOTE_BIAS + 1e-9


# ---------------------------------------------------------------------------
# Property-based: determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_result(self):
        """Two runs with same seed produce identical results."""
        state1 = DiplomacyState()
        state2 = DiplomacyState()
        for i in range(5):
            state1.ideology_cache[f"c-{i}"] = "cooperative"
            state1.ideology_age[f"c-{i}"] = 5
            state2.ideology_cache[f"c-{i}"] = "cooperative"
            state2.ideology_age[f"c-{i}"] = 5
        colonists = [
            make_colonist_dict(f"c-{i}", empathy=0.8, mediation=0.6)
            for i in range(5)
        ]
        r1 = tick_diplomacy(state1, colonists, make_social_get(0.5),
                             {}, year=10, rng=random.Random(42))
        r2 = tick_diplomacy(state2, colonists, make_social_get(0.5),
                             {}, year=10, rng=random.Random(42))
        assert r1.to_dict() == r2.to_dict()


# ---------------------------------------------------------------------------
# Property-based: physical bounds
# ---------------------------------------------------------------------------

class TestBounds:
    def test_cohesion_bounded(self):
        """Faction cohesion stays in [0, 1]."""
        state = DiplomacyState()
        members = [f"c-{i}" for i in range(6)]
        f = Faction(id="f-0", name="Test", ideology="cooperative",
                    members=members, leader_id="c-0",
                    founding_year=1, cohesion=0.5)
        state.factions["f-0"] = f
        for cid in members:
            state.ideology_cache[cid] = "cooperative"
            state.ideology_age[cid] = 5
        colonists = [make_colonist_dict(cid) for cid in members]
        rng = random.Random(77)
        for yr in range(1, 50):
            tick_diplomacy(state, colonists, make_social_get(rng.random()),
                           {}, year=yr, rng=rng)
            for faction in state.factions.values():
                assert 0.0 <= faction.cohesion <= 1.0
                assert 0.0 <= faction.influence <= 1.0

    def test_tensions_bounded(self):
        """All tensions stay in [0, 1]."""
        state = DiplomacyState()
        rng = random.Random(88)
        for i in range(10):
            state.ideology_cache[f"c-{i}"] = rng.choice(list(IDEOLOGY_NAMES))
            state.ideology_age[f"c-{i}"] = 5
        colonists = [make_colonist_dict(f"c-{i}") for i in range(10)]
        for yr in range(1, 100):
            tick_diplomacy(state, colonists, make_social_get(rng.random()),
                           {}, year=yr, rng=rng)
            for v in state.tensions.values():
                assert 0.0 <= v <= 1.0, f"tension={v} at year {yr}"


# ---------------------------------------------------------------------------
# Integration: 10-year smoke test
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_10_year_run(self):
        """Run diplomacy for 10 years without crash."""
        state = DiplomacyState()
        rng = random.Random(42)
        colonists = [make_colonist_dict(f"c-{i}",
                                         empathy=rng.random(),
                                         paranoia=rng.random(),
                                         faith=rng.random(),
                                         hoarding=rng.random(),
                                         resolve=rng.random(),
                                         improvisation=rng.random(),
                                         coding=rng.random(),
                                         mediation=rng.random(),
                                         prayer=rng.random())
                     for i in range(10)]
        actions = {f"c-{i}": rng.choice(["cooperate", "hoard", "pray",
                                          "code", "terraform"])
                   for i in range(10)}
        for yr in range(1, 11):
            result = tick_diplomacy(state, colonists, make_social_get(0.5),
                                    actions, year=yr, rng=rng)
            assert isinstance(result, DiplomacyTickResult)
            d = result.to_dict()
            assert "faction_count" in d
            assert "alliance_count" in d

    def test_empty_colonists(self):
        """Zero colonists doesn't crash."""
        state = DiplomacyState()
        result = tick_diplomacy(state, [], make_social_get(), {},
                                year=5, rng=random.Random(1))
        assert result.faction_count == 0

    def test_single_colonist(self):
        """Single colonist doesn't crash."""
        state = DiplomacyState()
        result = tick_diplomacy(state, [make_colonist_dict("c-0")],
                                make_social_get(), {}, year=5,
                                rng=random.Random(1))
        assert result.faction_count == 0
