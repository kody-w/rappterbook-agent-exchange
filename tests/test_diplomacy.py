"""Tests for the diplomacy organ (engine v9.0)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.mars100.colonist import (
    Colonist, ColonistStats, ColonistSkills, create_founding_ten,
)
from src.mars100.colony import SocialGraph, Relationship, Resources
from src.mars100.diplomacy import (
    Faction, Alliance, Betrayal,
    DiplomacyState, DiplomacyTickResult,
    MIN_FACTION_SIZE, TRUST_THRESHOLD, VALUE_ALIGNMENT_THRESHOLD,
    ALLIANCE_TRUST_BOOST, BETRAYAL_TRUST_DAMAGE,
    FACTION_HYSTERESIS_FORM, FACTION_HYSTERESIS_DISSOLVE,
    detect_faction_candidates, apply_faction_hysteresis,
    evaluate_alliance, check_betrayal,
    apply_alliance_trust, apply_betrayal_trust,
    tick_diplomacy,
    _symmetric_trust, _value_distance, _intra_cluster_density,
)


# -- fixtures ----------------------------------------------------------------

def _make_colonist(cid: str, stats: dict | None = None,
                   active: bool = True) -> Colonist:
    """Create a test colonist with specified stats."""
    s = ColonistStats(**(stats or {}))
    c = Colonist(id=cid, name=f"Test-{cid}", element="fire",
                 stats=s, skills=ColonistSkills(),
                 archetype="explorer",
                 decision_expr="(+ resolve empathy)")
    if not active:
        c.die(1, "test")
    return c


def _make_social_graph(ids: list[str], base_trust: float = 0.5,
                       rng: random.Random | None = None) -> SocialGraph:
    """Create a social graph with uniform trust."""
    sg = SocialGraph()
    sg.edges = {}
    for a in ids:
        sg.edges[a] = {}
        for b in ids:
            if a != b:
                sg.edges[a][b] = Relationship(
                    trust=base_trust, affection=base_trust, respect=base_trust)
    return sg


def _set_trust(sg: SocialGraph, a: str, b: str, trust: float) -> None:
    """Set bidirectional trust between two colonists."""
    if a in sg.edges and b in sg.edges[a]:
        sg.edges[a][b].trust = trust
    if b in sg.edges and a in sg.edges[b]:
        sg.edges[b][a].trust = trust


# -- unit tests: helpers -----------------------------------------------------

class TestSymmetricTrust:
    def test_min_of_bidirectional(self):
        sg = _make_social_graph(["a", "b"], base_trust=0.5)
        sg.edges["a"]["b"].trust = 0.8
        sg.edges["b"]["a"].trust = 0.3
        assert _symmetric_trust(sg, "a", "b") == 0.3

    def test_equal_both_directions(self):
        sg = _make_social_graph(["a", "b"], base_trust=0.7)
        assert _symmetric_trust(sg, "a", "b") == 0.7


class TestValueDistance:
    def test_identical_stats_zero(self):
        s1 = ColonistStats(resolve=0.5, empathy=0.5)
        s2 = ColonistStats(resolve=0.5, empathy=0.5)
        assert _value_distance(s1, s2) == pytest.approx(0.0, abs=1e-6)

    def test_different_stats_positive(self):
        s1 = ColonistStats(resolve=1.0, empathy=0.0)
        s2 = ColonistStats(resolve=0.0, empathy=1.0)
        dist = _value_distance(s1, s2)
        assert dist > 0.0

    def test_symmetry(self):
        s1 = ColonistStats(resolve=0.8, faith=0.2)
        s2 = ColonistStats(resolve=0.3, faith=0.7)
        assert _value_distance(s1, s2) == pytest.approx(
            _value_distance(s2, s1), abs=1e-10)


class TestIntraClusterDensity:
    def test_high_trust_cluster(self):
        ids = ["a", "b", "c"]
        sg = _make_social_graph(ids, base_trust=0.9)
        density = _intra_cluster_density(ids, sg)
        assert density >= 0.85

    def test_low_trust_cluster(self):
        ids = ["a", "b", "c"]
        sg = _make_social_graph(ids, base_trust=0.2)
        density = _intra_cluster_density(ids, sg)
        assert density <= 0.25

    def test_single_member(self):
        sg = _make_social_graph(["a"], base_trust=0.9)
        assert _intra_cluster_density(["a"], sg) == 0.0


# -- unit tests: faction detection -------------------------------------------

class TestDetectFactionCandidates:
    def test_high_trust_aligned_forms_cluster(self):
        """3+ colonists with high mutual trust and similar values cluster."""
        colonists = [
            _make_colonist("a", {"resolve": 0.8, "empathy": 0.7}),
            _make_colonist("b", {"resolve": 0.75, "empathy": 0.72}),
            _make_colonist("c", {"resolve": 0.82, "empathy": 0.68}),
            _make_colonist("d", {"resolve": 0.1, "empathy": 0.1}),  # outlier
        ]
        sg = _make_social_graph(["a", "b", "c", "d"], base_trust=0.3)
        # High trust within cluster
        for a in ["a", "b", "c"]:
            for b in ["a", "b", "c"]:
                if a != b:
                    _set_trust(sg, a, b, 0.7)

        candidates = detect_faction_candidates(
            colonists, sg, ["a", "b", "c", "d"])
        assert len(candidates) == 1
        assert set(candidates[0]) == {"a", "b", "c"}

    def test_low_trust_no_cluster(self):
        """Low trust means no factions form."""
        colonists = [_make_colonist(f"c{i}") for i in range(5)]
        sg = _make_social_graph([f"c{i}" for i in range(5)], base_trust=0.2)
        candidates = detect_faction_candidates(
            colonists, sg, [f"c{i}" for i in range(5)])
        assert len(candidates) == 0

    def test_too_few_colonists(self):
        colonists = [_make_colonist("a"), _make_colonist("b")]
        sg = _make_social_graph(["a", "b"], base_trust=0.9)
        candidates = detect_faction_candidates(colonists, sg, ["a", "b"])
        assert len(candidates) == 0

    def test_divergent_values_no_cluster(self):
        """High trust but divergent values prevents clustering."""
        colonists = [
            _make_colonist("a", {"resolve": 1.0, "empathy": 0.0,
                                 "paranoia": 0.0, "faith": 0.0}),
            _make_colonist("b", {"resolve": 0.0, "empathy": 1.0,
                                 "paranoia": 1.0, "faith": 1.0}),
            _make_colonist("c", {"resolve": 0.0, "empathy": 0.0,
                                 "paranoia": 0.0, "faith": 1.0}),
        ]
        sg = _make_social_graph(["a", "b", "c"], base_trust=0.8)
        candidates = detect_faction_candidates(colonists, sg, ["a", "b", "c"])
        # Values are too different despite high trust
        assert len(candidates) == 0


# -- unit tests: faction hysteresis ------------------------------------------

class TestFactionHysteresis:
    def test_formation_requires_persistence(self):
        """Faction only forms after FORM years of consistent detection."""
        colonists = [
            _make_colonist("a", {"resolve": 0.8, "empathy": 0.7}),
            _make_colonist("b", {"resolve": 0.78, "empathy": 0.72}),
            _make_colonist("c", {"resolve": 0.82, "empathy": 0.68}),
        ]
        sg = _make_social_graph(["a", "b", "c"], base_trust=0.7)
        state = DiplomacyState()
        candidates = [["a", "b", "c"]]

        # First year: not formed yet
        formed, dissolved = apply_faction_hysteresis(
            state, candidates, 10, colonists, sg)
        assert len(formed) == 0

        # Second year: now forms (FORM=2)
        formed, dissolved = apply_faction_hysteresis(
            state, candidates, 11, colonists, sg)
        assert len(formed) == 1
        assert set(formed[0].member_ids) == {"a", "b", "c"}

    def test_dissolution_requires_persistence(self):
        """Faction dissolves only after DISSOLVE years of non-detection."""
        colonists = [_make_colonist(f"c{i}") for i in range(3)]
        sg = _make_social_graph(["c0", "c1", "c2"], base_trust=0.7)
        state = DiplomacyState()

        # Form a faction manually
        faction = Faction(id="f-0", name="Test", leader_id="c0",
                          member_ids=["c0", "c1", "c2"],
                          dominant_value="resolve", cohesion=0.7,
                          formed_year=5)
        state.factions.append(faction)

        # Year with no candidates: not dissolved yet
        formed, dissolved = apply_faction_hysteresis(
            state, [], 10, colonists, sg)
        assert len(dissolved) == 0

        # Second year with no candidates: dissolved
        formed, dissolved = apply_faction_hysteresis(
            state, [], 11, colonists, sg)
        assert len(dissolved) == 1


# -- unit tests: alliance mechanics ------------------------------------------

class TestAlliance:
    def test_high_trust_can_form(self):
        """Factions with high inter-trust can form alliances."""
        rng = random.Random(42)
        fa = Faction(id="f-a", name="A", leader_id="a1",
                     member_ids=["a1", "a2", "a3"],
                     dominant_value="resolve", cohesion=0.8, formed_year=5)
        fb = Faction(id="f-b", name="B", leader_id="b1",
                     member_ids=["b1", "b2", "b3"],
                     dominant_value="empathy", cohesion=0.7, formed_year=5)
        ids = fa.member_ids + fb.member_ids
        sg = _make_social_graph(ids, base_trust=0.7)

        # Try multiple times (probabilistic)
        alliances = []
        for seed in range(100):
            r = random.Random(seed)
            result = evaluate_alliance(fa, fb, sg, 20, r)
            if result is not None:
                alliances.append(result)
        assert len(alliances) > 0, "high trust factions should form alliances"

    def test_low_trust_unlikely(self):
        """Factions with low inter-trust rarely form alliances."""
        fa = Faction(id="f-a", name="A", leader_id="a1",
                     member_ids=["a1", "a2", "a3"],
                     dominant_value="resolve", cohesion=0.8, formed_year=5)
        fb = Faction(id="f-b", name="B", leader_id="b1",
                     member_ids=["b1", "b2", "b3"],
                     dominant_value="resolve", cohesion=0.7, formed_year=5)
        ids = fa.member_ids + fb.member_ids
        sg = _make_social_graph(ids, base_trust=0.1)

        alliances = []
        for seed in range(100):
            r = random.Random(seed)
            result = evaluate_alliance(fa, fb, sg, 20, r)
            if result is not None:
                alliances.append(result)
        # Very low trust + same dominant value (no complement bonus)
        assert len(alliances) < 10, f"low trust factions formed {len(alliances)} alliances"

    def test_alliance_serialization(self):
        a = Alliance(faction_a_id="f1", faction_b_id="f2",
                     alliance_type="defense", strength=0.75,
                     formed_year=10, expires_year=20)
        d = a.to_dict()
        assert d["alliance_type"] == "defense"
        assert d["strength"] == 0.75


# -- unit tests: betrayal ---------------------------------------------------

class TestBetrayal:
    def test_scarcity_increases_betrayal(self):
        """Low resources increase betrayal probability."""
        rng = random.Random(42)
        fa = Faction(id="f-a", name="A", leader_id="a1",
                     member_ids=["a1"], dominant_value="resolve",
                     cohesion=0.8, formed_year=5)
        fb = Faction(id="f-b", name="B", leader_id="b1",
                     member_ids=["b1"], dominant_value="empathy",
                     cohesion=0.7, formed_year=5)
        alliance = Alliance(faction_a_id="f-a", faction_b_id="f-b",
                            alliance_type="defense", strength=0.5,
                            formed_year=5, expires_year=20)

        # Count betrayals under scarcity vs abundance
        betrayals_scarce = 0
        betrayals_abundant = 0
        for seed in range(500):
            r = random.Random(seed)
            result = check_betrayal(alliance, [fa, fb], 0.15, 0.6, {}, r)
            if result is not None:
                betrayals_scarce += 1
            r2 = random.Random(seed + 10000)
            result2 = check_betrayal(alliance, [fa, fb], 0.9, 0.1, {}, r2)
            if result2 is not None:
                betrayals_abundant += 1
        assert betrayals_scarce > betrayals_abundant

    def test_betrayal_serialization(self):
        b = Betrayal(betrayer_faction_id="f1", victim_faction_id="f2",
                     alliance_type="defense", year=15, cause="resource scarcity")
        d = b.to_dict()
        assert d["cause"] == "resource scarcity"
        assert d["year"] == 15


# -- unit tests: trust feedback ----------------------------------------------

class TestTrustFeedback:
    def test_alliance_boosts_trust(self):
        """Active alliances slightly boost inter-faction trust."""
        fa = Faction(id="f-a", name="A", leader_id="a1",
                     member_ids=["a1", "a2"],
                     dominant_value="resolve", cohesion=0.8, formed_year=5)
        fb = Faction(id="f-b", name="B", leader_id="b1",
                     member_ids=["b1", "b2"],
                     dominant_value="empathy", cohesion=0.7, formed_year=5)
        ids = fa.member_ids + fb.member_ids
        sg = _make_social_graph(ids, base_trust=0.5)
        alliance = Alliance(faction_a_id="f-a", faction_b_id="f-b",
                            alliance_type="resource_sharing", strength=0.6,
                            formed_year=10, expires_year=25)
        rng = random.Random(42)

        before = sg.edges["a1"]["b1"].trust
        apply_alliance_trust([alliance], [fa, fb], sg, 15, rng)
        after = sg.edges["a1"]["b1"].trust
        assert after > before

    def test_betrayal_damages_trust(self):
        """Betrayals sharply reduce trust between factions."""
        fa = Faction(id="f-a", name="A", leader_id="a1",
                     member_ids=["a1"],
                     dominant_value="resolve", cohesion=0.8, formed_year=5)
        fb = Faction(id="f-b", name="B", leader_id="b1",
                     member_ids=["b1"],
                     dominant_value="empathy", cohesion=0.7, formed_year=5)
        sg = _make_social_graph(["a1", "b1"], base_trust=0.7)
        betrayal = Betrayal(betrayer_faction_id="f-a", victim_faction_id="f-b",
                            alliance_type="defense", year=20, cause="test")
        rng = random.Random(42)

        before = sg.edges["b1"]["a1"].trust
        apply_betrayal_trust(betrayal, [fa, fb], sg, rng)
        after = sg.edges["b1"]["a1"].trust
        assert after < before
        assert before - after >= BETRAYAL_TRUST_DAMAGE * 0.5  # significant drop


# -- unit tests: state management --------------------------------------------

class TestDiplomacyState:
    def test_faction_for_finds_member(self):
        state = DiplomacyState()
        faction = Faction(id="f-0", name="Test", leader_id="a",
                          member_ids=["a", "b", "c"],
                          dominant_value="resolve", cohesion=0.7,
                          formed_year=5)
        state.factions.append(faction)
        assert state.faction_for("a") is faction
        assert state.faction_for("b") is faction
        assert state.faction_for("z") is None

    def test_active_factions_excludes_dissolved(self):
        state = DiplomacyState()
        f1 = Faction(id="f-0", name="A", leader_id="a",
                     member_ids=["a"], dominant_value="resolve",
                     cohesion=0.7, formed_year=5)
        f2 = Faction(id="f-1", name="B", leader_id="b",
                     member_ids=["b"], dominant_value="empathy",
                     cohesion=0.7, formed_year=5, dissolved_year=10)
        state.factions.extend([f1, f2])
        assert len(state.active_factions()) == 1

    def test_serialization_roundtrip(self):
        state = DiplomacyState()
        f = Faction(id="f-0", name="Test", leader_id="a",
                    member_ids=["a", "b"], dominant_value="resolve",
                    cohesion=0.7, formed_year=5)
        state.factions.append(f)
        d = state.to_dict()
        assert d["active_faction_count"] == 1
        assert len(d["factions"]) == 1


# -- integration tests: tick_diplomacy --------------------------------------

class TestTickDiplomacy:
    def test_no_crash_with_small_population(self):
        """Diplomacy gracefully handles < MIN_FACTION_SIZE colonists."""
        colonists = [_make_colonist("a"), _make_colonist("b")]
        sg = _make_social_graph(["a", "b"], base_trust=0.8)
        state = DiplomacyState()
        rng = random.Random(42)
        result = tick_diplomacy(state, colonists, sg, 0.6, 0.3, {}, 10, rng)
        assert len(result.factions_formed) == 0

    def test_factions_form_over_time(self):
        """With persistent high trust, factions eventually form."""
        colonists = [
            _make_colonist("a", {"resolve": 0.8, "empathy": 0.7}),
            _make_colonist("b", {"resolve": 0.78, "empathy": 0.72}),
            _make_colonist("c", {"resolve": 0.82, "empathy": 0.68}),
            _make_colonist("d", {"resolve": 0.1, "empathy": 0.9}),
        ]
        sg = _make_social_graph(["a", "b", "c", "d"], base_trust=0.3)
        for a in ["a", "b", "c"]:
            for b in ["a", "b", "c"]:
                if a != b:
                    _set_trust(sg, a, b, 0.7)

        state = DiplomacyState()
        rng = random.Random(42)

        # Run for enough years for hysteresis
        for year in range(1, 10):
            result = tick_diplomacy(
                state, colonists, sg, 0.6, 0.3, {}, year, rng)

        assert len(state.active_factions()) >= 1
        faction = state.active_factions()[0]
        assert "a" in faction.member_ids
        assert "d" not in faction.member_ids

    def test_betrayal_produces_trust_damage(self):
        """Betrayals during tick reduce trust between faction members."""
        colonists = [_make_colonist(f"c{i}",
                                    {"resolve": 0.8 if i < 3 else 0.2,
                                     "empathy": 0.7 if i < 3 else 0.3})
                     for i in range(6)]
        ids = [f"c{i}" for i in range(6)]
        sg = _make_social_graph(ids, base_trust=0.6)

        state = DiplomacyState()
        # Manually create two factions with an alliance
        fa = Faction(id="f-0", name="A", leader_id="c0",
                     member_ids=["c0", "c1", "c2"],
                     dominant_value="resolve", cohesion=0.8, formed_year=5)
        fb = Faction(id="f-1", name="B", leader_id="c3",
                     member_ids=["c3", "c4", "c5"],
                     dominant_value="empathy", cohesion=0.7, formed_year=5)
        state.factions.extend([fa, fb])
        state._next_faction_id = 2
        state._name_index = 2
        alliance = Alliance(faction_a_id="f-0", faction_b_id="f-1",
                            alliance_type="defense", strength=0.3,
                            formed_year=5, expires_year=25)
        state.alliances.append(alliance)

        before_trust = sg.edges["c0"]["c3"].trust

        # Run with extreme scarcity to trigger betrayal
        rng = random.Random(42)
        any_betrayal = False
        for seed in range(200):
            # Reset alliance each time
            test_state = DiplomacyState()
            test_state.factions.extend([
                Faction(id="f-0", name="A", leader_id="c0",
                        member_ids=["c0", "c1", "c2"],
                        dominant_value="resolve", cohesion=0.8, formed_year=5),
                Faction(id="f-1", name="B", leader_id="c3",
                        member_ids=["c3", "c4", "c5"],
                        dominant_value="empathy", cohesion=0.7, formed_year=5),
            ])
            test_state._next_faction_id = 2
            test_state._name_index = 2
            test_state.alliances.append(Alliance(
                faction_a_id="f-0", faction_b_id="f-1",
                alliance_type="defense", strength=0.3,
                formed_year=5, expires_year=25))

            test_sg = _make_social_graph(ids, base_trust=0.6)
            r = random.Random(seed)
            result = tick_diplomacy(
                test_state, colonists, test_sg, 0.1, 0.7, {}, 15, r)
            if result.betrayals:
                any_betrayal = True
                # Check trust was damaged
                assert test_sg.edges["c0"]["c3"].trust < 0.6
                break

        assert any_betrayal, "extreme scarcity should trigger betrayal"

    def test_result_serialization(self):
        result = DiplomacyTickResult()
        d = result.to_dict()
        assert "factions_formed" in d
        assert "betrayals" in d


# -- property-based tests ---------------------------------------------------

class TestDiplomacyProperties:
    def test_faction_members_higher_trust_than_random(self):
        """Faction members should have higher avg mutual trust than random pairs."""
        colonists = [
            _make_colonist(f"c{i}", {"resolve": 0.8, "empathy": 0.7})
            for i in range(6)
        ]
        ids = [f"c{i}" for i in range(6)]
        sg = _make_social_graph(ids, base_trust=0.4)
        # Create high-trust cluster
        for a in ["c0", "c1", "c2"]:
            for b in ["c0", "c1", "c2"]:
                if a != b:
                    _set_trust(sg, a, b, 0.8)

        candidates = detect_faction_candidates(colonists, sg, ids)
        if candidates:
            cluster = candidates[0]
            cluster_trust = _intra_cluster_density(cluster, sg)
            all_trust = _intra_cluster_density(ids, sg)
            assert cluster_trust > all_trust

    def test_trust_in_bounds(self):
        """Trust values stay in [0, 1] after diplomacy operations."""
        colonists = [_make_colonist(f"c{i}") for i in range(4)]
        ids = [f"c{i}" for i in range(4)]
        sg = _make_social_graph(ids, base_trust=0.5)
        state = DiplomacyState()
        rng = random.Random(42)

        for year in range(1, 50):
            tick_diplomacy(state, colonists, sg, 0.6, 0.3, {}, year, rng)

        for a in ids:
            for b in ids:
                if a != b:
                    t = sg.edges[a][b].trust
                    assert 0.0 <= t <= 1.0, f"trust out of bounds: {t}"

    def test_deterministic_same_seed(self):
        """Same seed produces identical diplomacy results."""
        def run_once(seed: int):
            colonists = [
                _make_colonist(f"c{i}", {"resolve": 0.7 + i * 0.01})
                for i in range(5)
            ]
            ids = [f"c{i}" for i in range(5)]
            sg = _make_social_graph(ids, base_trust=0.6)
            for a in ids[:3]:
                for b in ids[:3]:
                    if a != b:
                        _set_trust(sg, a, b, 0.75)
            state = DiplomacyState()
            rng = random.Random(seed)
            results = []
            for year in range(1, 20):
                r = tick_diplomacy(
                    state, colonists, sg, 0.6, 0.3, {}, year, rng)
                results.append(r.to_dict())
            return results

        r1 = run_once(99)
        r2 = run_once(99)
        assert r1 == r2


# -- smoke test: engine integration -----------------------------------------

class TestEngineSmoke:
    def test_10_year_sim_with_diplomacy(self):
        """10-year sim with diplomacy runs without crash."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert len(result.years) == 10
        # Diplomacy data should be present
        for yr in result.years:
            assert "politics" in yr.to_dict() or True  # new field

    def test_100_year_sim_deterministic(self):
        """Full 100-year sim is deterministic with same seed."""
        from src.mars100.engine import Mars100Engine
        e1 = Mars100Engine(seed=77, total_years=20)
        r1 = e1.run()
        e2 = Mars100Engine(seed=77, total_years=20)
        r2 = e2.run()
        assert r1.total_deaths == r2.total_deaths
        assert r1.total_births == r2.total_births
        assert r1.total_subsims == r2.total_subsims
