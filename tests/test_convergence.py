"""Tests for Mars-100 value convergence analysis."""
from __future__ import annotations

import pytest
from src.mars100.convergence import (
    compute_stat_variances,
    pairwise_distances,
    detect_clusters,
    convergence_trend,
    analyze_year,
    CONVERGENCE_WINDOW,
)
from src.mars100.colonist import STAT_NAMES


def _make_snap(cid: str, **kwargs: float) -> dict:
    """Make a colonist snapshot dict with given stats."""
    stats = {s: kwargs.get(s, 0.5) for s in STAT_NAMES}
    return {"id": cid, "alive": True, "exiled": False, "stats": stats}


class TestStatVariances:
    def test_identical_colonists_zero_variance(self) -> None:
        snaps = [_make_snap("a"), _make_snap("b"), _make_snap("c")]
        variances = compute_stat_variances(snaps)
        for v in variances.values():
            assert v == 0.0

    def test_different_colonists_positive_variance(self) -> None:
        snaps = [_make_snap("a", resolve=0.0), _make_snap("b", resolve=1.0)]
        variances = compute_stat_variances(snaps)
        assert variances["resolve"] > 0

    def test_single_colonist_zero_variance(self) -> None:
        snaps = [_make_snap("a")]
        variances = compute_stat_variances(snaps)
        assert all(v == 0.0 for v in variances.values())

    def test_dead_colonists_excluded(self) -> None:
        snaps = [
            _make_snap("a", resolve=0.0),
            _make_snap("b", resolve=1.0),
            {**_make_snap("c", resolve=0.5), "alive": False},
        ]
        variances = compute_stat_variances(snaps)
        assert variances["resolve"] == pytest.approx(0.25)

    def test_exiled_colonists_excluded(self) -> None:
        snaps = [
            _make_snap("a", resolve=0.3),
            _make_snap("b", resolve=0.3),
            {**_make_snap("c", resolve=0.9), "exiled": True},
        ]
        variances = compute_stat_variances(snaps)
        assert variances["resolve"] == 0.0

    def test_all_stats_present(self) -> None:
        variances = compute_stat_variances([_make_snap("a"), _make_snap("b")])
        for stat in STAT_NAMES:
            assert stat in variances


class TestPairwiseDistances:
    def test_identical_zero_distances(self) -> None:
        snaps = [_make_snap("a"), _make_snap("b")]
        dists = pairwise_distances(snaps)
        assert len(dists) == 1
        assert dists[0] == 0.0

    def test_pair_count(self) -> None:
        snaps = [_make_snap(f"c{i}") for i in range(5)]
        dists = pairwise_distances(snaps)
        assert len(dists) == 10  # 5 choose 2

    def test_nonzero_for_different(self) -> None:
        snaps = [_make_snap("a", resolve=0.0), _make_snap("b", resolve=1.0)]
        dists = pairwise_distances(snaps)
        assert dists[0] > 0

    def test_single_colonist_no_distances(self) -> None:
        dists = pairwise_distances([_make_snap("a")])
        assert dists == []


class TestDetectClusters:
    def test_all_identical_one_cluster(self) -> None:
        snaps = [_make_snap(f"c{i}") for i in range(5)]
        clusters = detect_clusters(snaps)
        assert len(clusters) == 1
        assert len(clusters[0]) == 5

    def test_two_distant_groups(self) -> None:
        g1 = [_make_snap(f"a{i}", resolve=0.0, empathy=0.0, faith=0.0,
                         hoarding=0.0, improvisation=0.0, paranoia=0.0)
              for i in range(3)]
        g2 = [_make_snap(f"b{i}", resolve=1.0, empathy=1.0, faith=1.0,
                         hoarding=1.0, improvisation=1.0, paranoia=1.0)
              for i in range(3)]
        clusters = detect_clusters(g1 + g2)
        assert len(clusters) == 2

    def test_dead_excluded(self) -> None:
        snaps = [_make_snap("a"), _make_snap("b"),
                 {**_make_snap("c"), "alive": False}]
        clusters = detect_clusters(snaps)
        total = sum(len(c) for c in clusters)
        assert total == 2

    def test_single_returns_singleton(self) -> None:
        clusters = detect_clusters([_make_snap("solo")])
        assert len(clusters) == 1


class TestConvergenceTrend:
    def test_insufficient_data(self) -> None:
        assert convergence_trend([]) == "insufficient_data"
        assert convergence_trend([{"resolve": 0.1}]) == "insufficient_data"

    def test_converging(self) -> None:
        history = [{s: 0.1 - i * 0.02 for s in STAT_NAMES}
                   for i in range(CONVERGENCE_WINDOW)]
        assert convergence_trend(history) == "converging"

    def test_diverging(self) -> None:
        history = [{s: 0.01 + i * 0.02 for s in STAT_NAMES}
                   for i in range(CONVERGENCE_WINDOW)]
        assert convergence_trend(history) == "diverging"

    def test_stable(self) -> None:
        history = [{s: 0.05 for s in STAT_NAMES} for _ in range(CONVERGENCE_WINDOW)]
        assert convergence_trend(history) == "stable"


class TestAnalyzeYear:
    def test_returns_all_fields(self) -> None:
        snaps = [_make_snap(f"c{i}") for i in range(5)]
        result = analyze_year(snaps, [])
        for key in ("stat_variances", "mean_distance", "cluster_count",
                     "clusters", "trend"):
            assert key in result

    def test_variance_history_extended(self) -> None:
        snaps = [_make_snap("a"), _make_snap("b")]
        history: list[dict[str, float]] = []
        result = analyze_year(snaps, history)
        assert "stat_variances" in result

    def test_cluster_count_matches(self) -> None:
        snaps = [_make_snap(f"c{i}") for i in range(4)]
        result = analyze_year(snaps, [])
        assert result["cluster_count"] == len(result["clusters"])


class TestPhysicalBounds:
    """Property-based: all outputs should be in expected ranges."""

    def test_variances_nonnegative(self) -> None:
        import random
        rng = random.Random(42)
        for _ in range(20):
            snaps = [_make_snap(f"c{i}", **{s: rng.random() for s in STAT_NAMES})
                     for i in range(rng.randint(2, 10))]
            variances = compute_stat_variances(snaps)
            for v in variances.values():
                assert v >= 0.0

    def test_distances_nonnegative(self) -> None:
        import random
        rng = random.Random(42)
        snaps = [_make_snap(f"c{i}", **{s: rng.random() for s in STAT_NAMES})
                 for i in range(5)]
        for d in pairwise_distances(snaps):
            assert d >= 0.0
