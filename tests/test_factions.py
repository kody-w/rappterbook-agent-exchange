"""Tests for Mars-100 cultural factions."""
from __future__ import annotations

import random
import pytest
from src.mars100.factions import (
    Faction, detect_factions, faction_vote_modifier,
    summarize_factions, _compute_centroid, _compute_cohesion,
)
from src.mars100.colonist import STAT_NAMES


def _make_snap(cid: str, **kwargs: float) -> dict:
    """Make a colonist snapshot dict with given stats."""
    stats = {s: kwargs.get(s, 0.5) for s in STAT_NAMES}
    return {"id": cid, "alive": True, "exiled": False, "stats": stats}


class TestDetectFactions:
    def test_identical_colonists_one_faction(self) -> None:
        snaps = [_make_snap(f"c{i}") for i in range(5)]
        factions = detect_factions(snaps, year=10)
        assert len(factions) == 1
        assert len(factions[0].member_ids) == 5

    def test_two_clusters(self) -> None:
        g1 = [_make_snap(f"a{i}", resolve=0.0, empathy=0.0, faith=0.0,
                         hoarding=0.0, improvisation=0.0, paranoia=0.0)
              for i in range(3)]
        g2 = [_make_snap(f"b{i}", resolve=1.0, empathy=1.0, faith=1.0,
                         hoarding=1.0, improvisation=1.0, paranoia=1.0)
              for i in range(3)]
        factions = detect_factions(g1 + g2, year=10)
        assert len(factions) == 2

    def test_too_few_colonists(self) -> None:
        factions = detect_factions([_make_snap("solo")], year=10)
        assert factions == []

    def test_dead_excluded(self) -> None:
        snaps = [_make_snap("a"), _make_snap("b"),
                 {**_make_snap("c"), "alive": False}]
        factions = detect_factions(snaps, year=10)
        total = sum(len(f.member_ids) for f in factions)
        assert total <= 2

    def test_exiled_excluded(self) -> None:
        snaps = [_make_snap("a"), _make_snap("b"),
                 {**_make_snap("c"), "exiled": True}]
        factions = detect_factions(snaps, year=10)
        total = sum(len(f.member_ids) for f in factions)
        assert total <= 2

    def test_faction_has_id_and_year(self) -> None:
        snaps = [_make_snap(f"c{i}") for i in range(3)]
        factions = detect_factions(snaps, year=42)
        assert factions[0].faction_id.startswith("faction-y42-")
        assert factions[0].year_formed == 42

    def test_min_size_respected(self) -> None:
        g1 = [_make_snap("a", resolve=0.0, empathy=0.0, faith=0.0,
                         hoarding=0.0, improvisation=0.0, paranoia=0.0)]
        g2 = [_make_snap(f"b{i}", resolve=1.0, empathy=1.0, faith=1.0,
                         hoarding=1.0, improvisation=1.0, paranoia=1.0)
              for i in range(3)]
        factions = detect_factions(g1 + g2, year=10, min_size=2)
        # g1 has only 1 member — shouldn't be a faction
        for f in factions:
            assert len(f.member_ids) >= 2

    def test_custom_threshold(self) -> None:
        # With very large threshold, everything is one cluster
        snaps = [_make_snap(f"c{i}", resolve=i * 0.1) for i in range(5)]
        factions = detect_factions(snaps, year=10, threshold=10.0)
        assert len(factions) == 1


class TestFactionVoteModifier:
    def test_same_faction_bonus(self) -> None:
        f = Faction(faction_id="f1", member_ids=["a", "b"],
                    centroid={}, dominant_stat="resolve",
                    cohesion=0.8, year_formed=10)
        mod = faction_vote_modifier("a", "b", [f])
        assert mod > 0

    def test_different_faction_penalty(self) -> None:
        f1 = Faction(faction_id="f1", member_ids=["a"],
                     centroid={}, dominant_stat="resolve",
                     cohesion=0.8, year_formed=10)
        f2 = Faction(faction_id="f2", member_ids=["b"],
                     centroid={}, dominant_stat="empathy",
                     cohesion=0.7, year_formed=10)
        mod = faction_vote_modifier("a", "b", [f1, f2])
        assert mod < 0

    def test_no_faction_neutral(self) -> None:
        mod = faction_vote_modifier("a", "b", [])
        assert mod == 0.0

    def test_one_not_in_faction_neutral(self) -> None:
        f = Faction(faction_id="f1", member_ids=["a"],
                    centroid={}, dominant_stat="resolve",
                    cohesion=0.8, year_formed=10)
        mod = faction_vote_modifier("a", "c", [f])
        assert mod == 0.0


class TestCentroid:
    def test_single_member(self) -> None:
        members = [_make_snap("a", resolve=0.8)]
        centroid = _compute_centroid(members)
        assert centroid["resolve"] == pytest.approx(0.8)

    def test_mean_of_two(self) -> None:
        members = [_make_snap("a", resolve=0.2), _make_snap("b", resolve=0.8)]
        centroid = _compute_centroid(members)
        assert centroid["resolve"] == pytest.approx(0.5)

    def test_empty_defaults(self) -> None:
        centroid = _compute_centroid([])
        assert all(v == 0.5 for v in centroid.values())


class TestCohesion:
    def test_single_member_perfect(self) -> None:
        members = [_make_snap("a")]
        centroid = _compute_centroid(members)
        assert _compute_cohesion(members, centroid) == 1.0

    def test_identical_perfect(self) -> None:
        members = [_make_snap(f"c{i}") for i in range(5)]
        centroid = _compute_centroid(members)
        assert _compute_cohesion(members, centroid) == 1.0

    def test_different_lower(self) -> None:
        members = [_make_snap("a", resolve=0.0), _make_snap("b", resolve=1.0)]
        centroid = _compute_centroid(members)
        cohesion = _compute_cohesion(members, centroid)
        assert 0.0 < cohesion < 1.0


class TestSummarizeFactions:
    def test_empty(self) -> None:
        summary = summarize_factions([])
        assert summary["count"] == 0

    def test_single_faction(self) -> None:
        f = Faction(faction_id="f1", member_ids=["a", "b"],
                    centroid={"resolve": 0.7}, dominant_stat="resolve",
                    cohesion=0.9, year_formed=10)
        summary = summarize_factions([f])
        assert summary["count"] == 1
        assert summary["total_members"] == 2
        assert summary["avg_cohesion"] == 0.9

    def test_to_dict_serializable(self) -> None:
        f = Faction(faction_id="f1", member_ids=["a", "b"],
                    centroid={s: 0.5 for s in STAT_NAMES},
                    dominant_stat="resolve", cohesion=0.8, year_formed=10)
        d = f.to_dict()
        import json
        json.dumps(d)  # should not raise


class TestPhysicalBounds:
    def test_cohesion_bounded(self) -> None:
        rng = random.Random(42)
        for _ in range(20):
            snaps = [_make_snap(f"c{i}", **{s: rng.random() for s in STAT_NAMES})
                     for i in range(rng.randint(2, 8))]
            factions = detect_factions(snaps, year=10)
            for f in factions:
                assert 0.0 <= f.cohesion <= 1.0
