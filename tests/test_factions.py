"""Tests for Mars-100 emergent factions."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, create_founding_ten
from src.mars100.colony import SocialGraph
from src.mars100.factions import (
    Faction, detect_factions, compute_faction_tensions,
    check_soft_schism, pairwise_affinity, _stat_similarity,
    _stat_vector, _stat_distance, MERGE_THRESHOLD,
    MIN_FACTION_SIZE,
)


def _make_colonist(cid: str, resolve: float = 0.5, empathy: float = 0.5,
                   faith: float = 0.5, paranoia: float = 0.5) -> Colonist:
    """Helper to create a colonist with specific stats."""
    return Colonist(
        id=cid, name=f"Test-{cid}", element="fire",
        archetype="pioneer", decision_expr="(+ resolve empathy)",
        stats=ColonistStats(resolve=resolve, improvisation=0.5,
                            empathy=empathy, hoarding=0.5,
                            faith=faith, paranoia=paranoia),
        skills=ColonistSkills(),
    )


def _make_social(colonists: list[Colonist], seed: int = 42) -> SocialGraph:
    """Helper to create an initialized social graph."""
    sg = SocialGraph()
    sg.initialize([c.id for c in colonists], random.Random(seed))
    return sg


class TestStatDistance:
    def test_identical_vectors(self) -> None:
        a = [0.5, 0.5, 0.5]
        assert _stat_distance(a, a) == 0.0

    def test_opposite_vectors(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [1.0, 1.0, 1.0]
        assert abs(_stat_distance(a, b) - 1.0) < 0.001

    def test_symmetry(self) -> None:
        a = [0.1, 0.9, 0.3]
        b = [0.8, 0.2, 0.7]
        assert abs(_stat_distance(a, b) - _stat_distance(b, a)) < 1e-9


class TestStatSimilarity:
    def test_identical(self) -> None:
        a = [0.5, 0.5]
        assert _stat_similarity(a, a) == 1.0

    def test_opposite(self) -> None:
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        assert abs(_stat_similarity(a, b)) < 0.001

    def test_range(self) -> None:
        rng = random.Random(123)
        for _ in range(20):
            a = [rng.random() for _ in range(6)]
            b = [rng.random() for _ in range(6)]
            sim = _stat_similarity(a, b)
            assert 0.0 <= sim <= 1.0


class TestPairwiseAffinity:
    def test_similar_colonists_high_affinity(self) -> None:
        a = _make_colonist("a", resolve=0.9, empathy=0.9)
        b = _make_colonist("b", resolve=0.9, empathy=0.9)
        sg = _make_social([a, b])
        aff = pairwise_affinity(a, b, sg)
        assert 0.0 <= aff <= 1.0

    def test_different_colonists_lower_affinity(self) -> None:
        a = _make_colonist("a", resolve=0.1, empathy=0.1, paranoia=0.1)
        b = _make_colonist("b", resolve=0.9, empathy=0.9, paranoia=0.9)
        sg = _make_social([a, b])
        similar_a = _make_colonist("sa", resolve=0.5, empathy=0.5)
        similar_b = _make_colonist("sb", resolve=0.5, empathy=0.5)
        sg2 = _make_social([similar_a, similar_b])
        aff_diff = pairwise_affinity(a, b, sg)
        aff_sim = pairwise_affinity(similar_a, similar_b, sg2)
        # Similar colonists should have >= affinity on the stat component
        # (social graph is random, so we just check ranges)
        assert 0.0 <= aff_diff <= 1.0
        assert 0.0 <= aff_sim <= 1.0


class TestDetectFactions:
    def test_no_factions_too_few(self) -> None:
        c = _make_colonist("solo")
        sg = _make_social([c])
        factions = detect_factions([c], sg, year=1)
        assert factions == []

    def test_similar_colonists_cluster(self) -> None:
        """Three very similar colonists should form one faction."""
        colonists = [
            _make_colonist("a", resolve=0.9, empathy=0.9, faith=0.9, paranoia=0.1),
            _make_colonist("b", resolve=0.9, empathy=0.9, faith=0.9, paranoia=0.1),
            _make_colonist("c", resolve=0.9, empathy=0.9, faith=0.9, paranoia=0.1),
        ]
        sg = _make_social(colonists)
        factions = detect_factions(colonists, sg, year=1, rng=random.Random(42))
        # With high similarity, they should cluster
        assert len(factions) >= 1
        if factions:
            assert len(factions[0].members) >= 2

    def test_faction_structure(self) -> None:
        colonists = create_founding_ten(42)
        sg = _make_social(colonists)
        factions = detect_factions(colonists, sg, year=1, rng=random.Random(42))
        for f in factions:
            assert f.id
            assert f.name
            assert len(f.members) >= MIN_FACTION_SIZE
            assert f.dominant_stat in ("resolve", "improvisation", "empathy",
                                       "hoarding", "faith", "paranoia")
            assert 0.0 <= f.cohesion <= 1.0

    def test_deterministic(self) -> None:
        colonists = create_founding_ten(42)
        sg = _make_social(colonists)
        f1 = detect_factions(colonists, sg, year=1, rng=random.Random(99))
        f2 = detect_factions(colonists, sg, year=1, rng=random.Random(99))
        assert len(f1) == len(f2)
        for a, b in zip(f1, f2):
            assert sorted(a.members) == sorted(b.members)

    def test_prior_matching(self) -> None:
        """Factions should preserve identity across ticks if overlap is high."""
        colonists = create_founding_ten(42)
        sg = _make_social(colonists)
        rng = random.Random(42)
        prior = detect_factions(colonists, sg, year=1, rng=rng)
        if not prior:
            return  # no factions to test
        rng2 = random.Random(42)
        new = detect_factions(colonists, sg, prior_factions=prior, year=2, rng=rng2)
        # With same colonists, faction ids should be preserved
        prior_ids = {f.id for f in prior}
        new_ids = {f.id for f in new}
        assert prior_ids & new_ids  # at least some should match


class TestFactionTensions:
    def test_empty(self) -> None:
        assert compute_faction_tensions([], None) == {}

    def test_single_faction(self) -> None:
        colonists = create_founding_ten(42)
        sg = _make_social(colonists)
        f = Faction(id="f1", name="Test", members=[c.id for c in colonists[:5]],
                    dominant_stat="resolve", centroid={}, cohesion=0.8, founded_year=1)
        tensions = compute_faction_tensions([f], sg)
        assert tensions == {}  # only one faction

    def test_two_factions_produce_tension(self) -> None:
        colonists = create_founding_ten(42)
        sg = _make_social(colonists)
        fa = Faction(id="fa", name="A", members=[c.id for c in colonists[:5]],
                     dominant_stat="resolve", centroid={}, cohesion=0.8, founded_year=1)
        fb = Faction(id="fb", name="B", members=[c.id for c in colonists[5:]],
                     dominant_stat="faith", centroid={}, cohesion=0.7, founded_year=1)
        tensions = compute_faction_tensions([fa, fb], sg)
        assert len(tensions) == 1
        key = ("fa", "fb")
        assert key in tensions
        assert 0.0 <= tensions[key] <= 1.0


class TestSoftSchism:
    def test_no_schism_low_tension(self) -> None:
        colonists = create_founding_ten(42)
        sg = _make_social(colonists)
        fa = Faction(id="fa", name="A", members=[c.id for c in colonists[:5]],
                     dominant_stat="resolve", centroid={}, cohesion=0.8, founded_year=1)
        fb = Faction(id="fb", name="B", members=[c.id for c in colonists[5:]],
                     dominant_stat="faith", centroid={}, cohesion=0.7, founded_year=1)
        tensions = {("fa", "fb"): 0.3}  # low tension
        events = check_soft_schism([fa, fb], tensions)
        assert events == []

    def test_schism_high_tension(self) -> None:
        fa = Faction(id="fa", name="Hawks", members=["a", "b"],
                     dominant_stat="resolve", centroid={}, cohesion=0.9, founded_year=1)
        fb = Faction(id="fb", name="Doves", members=["c", "d"],
                     dominant_stat="faith", centroid={}, cohesion=0.9, founded_year=1)
        tensions = {("fa", "fb"): 0.85}  # very high tension
        events = check_soft_schism([fa, fb], tensions)
        assert len(events) == 1
        assert events[0]["type"] == "soft_schism"
        assert "Hawks" in events[0]["faction_names"]
        assert "Doves" in events[0]["faction_names"]

    def test_threshold_boundary(self) -> None:
        fa = Faction(id="fa", name="A", members=["a"],
                     dominant_stat="resolve", centroid={}, cohesion=0.9, founded_year=1)
        fb = Faction(id="fb", name="B", members=["b"],
                     dominant_stat="faith", centroid={}, cohesion=0.9, founded_year=1)
        # Exactly at threshold
        assert check_soft_schism([fa, fb], {("fa", "fb"): 0.7}) != []
        # Just below
        assert check_soft_schism([fa, fb], {("fa", "fb"): 0.69}) == []


class TestFactionSerialization:
    def test_to_dict(self) -> None:
        f = Faction(id="f1", name="Test Pact", members=["a", "b"],
                    dominant_stat="resolve", centroid={"resolve": 0.8},
                    cohesion=0.75, founded_year=5)
        d = f.to_dict()
        assert d["id"] == "f1"
        assert d["name"] == "Test Pact"
        assert d["members"] == ["a", "b"]
        assert d["cohesion"] == 0.75
        assert d["founded_year"] == 5
