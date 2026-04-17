"""Tests for Mars-100 value convergence analysis."""
from __future__ import annotations

import pytest
from src.mars100.convergence import (
    compute_stat_variances, compute_pairwise_distance,
    detect_clusters, analyze_year, convergence_trend,
    ConvergenceSnapshot,
)
from src.mars100.colonist import STAT_NAMES


def _make_snapshot(stats: dict[str, float], alive: bool = True,
                   exiled: bool = False) -> dict:
    return {"stats": stats, "alive": alive, "exiled": exiled}


def _uniform_stats(value: float) -> dict[str, float]:
    return {s: value for s in STAT_NAMES}


class TestStatVariances:
    def test_identical_colonists_zero_variance(self) -> None:
        snaps = [_make_snapshot(_uniform_stats(0.5)) for _ in range(5)]
        variances = compute_stat_variances(snaps)
        for v in variances.values():
            assert v == pytest.approx(0.0)

    def test_diverse_colonists_nonzero_variance(self) -> None:
        snaps = [
            _make_snapshot({s: 0.1 for s in STAT_NAMES}),
            _make_snapshot({s: 0.9 for s in STAT_NAMES}),
        ]
        variances = compute_stat_variances(snaps)
        for v in variances.values():
            assert v > 0.1

    def test_dead_excluded(self) -> None:
        snaps = [
            _make_snapshot(_uniform_stats(0.5)),
            _make_snapshot(_uniform_stats(0.9), alive=False),
        ]
        variances = compute_stat_variances(snaps)
        for v in variances.values():
            assert v == pytest.approx(0.0)

    def test_exiled_excluded(self) -> None:
        snaps = [
            _make_snapshot(_uniform_stats(0.5)),
            _make_snapshot(_uniform_stats(0.9), exiled=True),
        ]
        variances = compute_stat_variances(snaps)
        for v in variances.values():
            assert v == pytest.approx(0.0)

    def test_single_colonist(self) -> None:
        snaps = [_make_snapshot(_uniform_stats(0.5))]
        variances = compute_stat_variances(snaps)
        for v in variances.values():
            assert v == pytest.approx(0.0)


class TestPairwiseDistance:
    def test_identical_zero_distance(self) -> None:
        snaps = [_make_snapshot(_uniform_stats(0.5)) for _ in range(5)]
        dist = compute_pairwise_distance(snaps)
        assert dist == pytest.approx(0.0)

    def test_different_positive_distance(self) -> None:
        snaps = [
            _make_snapshot({s: 0.0 for s in STAT_NAMES}),
            _make_snapshot({s: 1.0 for s in STAT_NAMES}),
        ]
        dist = compute_pairwise_distance(snaps)
        assert dist > 0

    def test_single_colonist_zero(self) -> None:
        snaps = [_make_snapshot(_uniform_stats(0.5))]
        assert compute_pairwise_distance(snaps) == pytest.approx(0.0)

    def test_dead_excluded(self) -> None:
        snaps = [
            _make_snapshot(_uniform_stats(0.5)),
            _make_snapshot(_uniform_stats(0.9), alive=False),
        ]
        assert compute_pairwise_distance(snaps) == pytest.approx(0.0)


class TestDetectClusters:
    def test_two_identical_one_cluster(self) -> None:
        snaps = [_make_snapshot(_uniform_stats(0.5)) for _ in range(3)]
        assert detect_clusters(snaps) == 1

    def test_two_distant_two_clusters(self) -> None:
        snaps = [
            _make_snapshot({s: 0.0 for s in STAT_NAMES}),
            _make_snapshot({s: 1.0 for s in STAT_NAMES}),
        ]
        assert detect_clusters(snaps) == 2

    def test_mixed_groups(self) -> None:
        group_a = [_make_snapshot({s: 0.1 for s in STAT_NAMES}) for _ in range(3)]
        group_b = [_make_snapshot({s: 0.9 for s in STAT_NAMES}) for _ in range(3)]
        clusters = detect_clusters(group_a + group_b)
        assert clusters == 2

    def test_single_colonist(self) -> None:
        assert detect_clusters([_make_snapshot(_uniform_stats(0.5))]) == 1


class TestAnalyzeYear:
    def test_returns_snapshot(self) -> None:
        snaps = [_make_snapshot(_uniform_stats(0.5)) for _ in range(5)]
        result = analyze_year(10, snaps)
        assert isinstance(result, ConvergenceSnapshot)
        assert result.year == 10

    def test_to_dict(self) -> None:
        snaps = [_make_snapshot(_uniform_stats(0.5)) for _ in range(3)]
        result = analyze_year(5, snaps)
        d = result.to_dict()
        assert "year" in d
        assert "stat_variances" in d
        assert "mean_pairwise_distance" in d
        assert "cluster_count" in d


class TestConvergenceTrend:
    def test_insufficient_data(self) -> None:
        snaps = [ConvergenceSnapshot(y, {}, 0.5, 2) for y in range(5)]
        assert convergence_trend(snaps) == "insufficient_data"

    def test_converging(self) -> None:
        # Early years: high distance, recent: low distance
        early = [ConvergenceSnapshot(y, {}, 0.8, 3) for y in range(10)]
        recent = [ConvergenceSnapshot(y + 10, {}, 0.3, 1) for y in range(10)]
        assert convergence_trend(early + recent) == "converging"

    def test_diverging(self) -> None:
        early = [ConvergenceSnapshot(y, {}, 0.3, 1) for y in range(10)]
        recent = [ConvergenceSnapshot(y + 10, {}, 0.8, 3) for y in range(10)]
        assert convergence_trend(early + recent) == "diverging"

    def test_stable(self) -> None:
        snaps = [ConvergenceSnapshot(y, {}, 0.5, 2) for y in range(20)]
        assert convergence_trend(snaps) == "stable"

    def test_edge_zero_early_dist(self) -> None:
        snaps = [ConvergenceSnapshot(y, {}, 0.0, 1) for y in range(20)]
        assert convergence_trend(snaps) == "stable"
