"""Tests for Mars-100 value convergence tracking."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars100.convergence import (
    compute_stat_std, compute_convergence_score,
    convergence_trend, per_stat_convergence, convergence_summary,
)

STAT_NAMES = ("resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia")


def _make_colonist(cid: str, stats: dict, alive: bool = True) -> dict:
    return {"id": cid, "name": cid, "alive": alive, "exiled": False,
            "stats": stats, "skills": {}}


class TestComputeStatStd:
    def test_identical_stats_zero_std(self):
        colonists = [_make_colonist(f"c{i}", {"resolve": 0.5}) for i in range(5)]
        assert compute_stat_std(colonists, "resolve") == 0.0

    def test_varied_stats_nonzero_std(self):
        colonists = [
            _make_colonist("c0", {"resolve": 0.0}),
            _make_colonist("c1", {"resolve": 1.0}),
        ]
        std = compute_stat_std(colonists, "resolve")
        assert std == pytest.approx(0.5, abs=0.01)

    def test_dead_colonists_excluded(self):
        colonists = [
            _make_colonist("c0", {"resolve": 0.5}),
            _make_colonist("c1", {"resolve": 0.5}),
            _make_colonist("dead", {"resolve": 0.0}, alive=False),
        ]
        assert compute_stat_std(colonists, "resolve") == 0.0

    def test_single_colonist_zero_std(self):
        colonists = [_make_colonist("c0", {"resolve": 0.7})]
        assert compute_stat_std(colonists, "resolve") == 0.0

    def test_empty_list(self):
        assert compute_stat_std([], "resolve") == 0.0

    def test_missing_stat_excluded(self):
        colonists = [
            _make_colonist("c0", {"resolve": 0.5}),
            _make_colonist("c1", {}),
        ]
        assert compute_stat_std(colonists, "resolve") == 0.0

    def test_exiled_excluded(self):
        colonists = [
            _make_colonist("c0", {"resolve": 0.5}),
            _make_colonist("c1", {"resolve": 0.5}),
        ]
        colonists[1]["exiled"] = True
        assert compute_stat_std(colonists, "resolve") == 0.0


class TestComputeConvergenceScore:
    def test_identical_colonists_zero(self):
        colonists = [
            _make_colonist(f"c{i}", {s: 0.5 for s in STAT_NAMES})
            for i in range(5)
        ]
        assert compute_convergence_score(colonists, STAT_NAMES) == 0.0

    def test_diverse_colonists_positive(self):
        c0 = _make_colonist("c0", {s: 0.0 for s in STAT_NAMES})
        c1 = _make_colonist("c1", {s: 1.0 for s in STAT_NAMES})
        score = compute_convergence_score([c0, c1], STAT_NAMES)
        assert score > 0.0

    def test_empty_inputs(self):
        assert compute_convergence_score([], STAT_NAMES) == 0.0
        c = _make_colonist("c0", {s: 0.5 for s in STAT_NAMES})
        assert compute_convergence_score([c], ()) == 0.0


class TestConvergenceTrend:
    def test_converging(self):
        scores = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
        assert convergence_trend(scores) == "converging"

    def test_diverging(self):
        scores = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
        assert convergence_trend(scores) == "diverging"

    def test_stable(self):
        scores = [0.3, 0.3, 0.3, 0.3, 0.3]
        assert convergence_trend(scores) == "stable"

    def test_single_score(self):
        assert convergence_trend([0.5]) == "stable"

    def test_empty(self):
        assert convergence_trend([]) == "stable"


class TestPerStatConvergence:
    def test_returns_all_stats(self):
        colonists = [
            _make_colonist(f"c{i}", {s: 0.5 for s in STAT_NAMES})
            for i in range(3)
        ]
        result = per_stat_convergence(colonists, STAT_NAMES)
        assert set(result.keys()) == set(STAT_NAMES)
        for v in result.values():
            assert v == 0.0


class TestConvergenceSummary:
    def test_basic(self):
        scores = [{"year": i, "score": 0.5 - i * 0.01} for i in range(10)]
        s = convergence_summary(scores)
        assert "trend" in s
        assert s["initial"] == 0.5
        assert s["final"] < 0.5

    def test_empty(self):
        s = convergence_summary([])
        assert s["trend"] == "stable"
        assert s["initial"] == 0.0

    def test_peak_trough(self):
        scores = [{"year": 0, "score": 0.1}, {"year": 1, "score": 0.9},
                  {"year": 2, "score": 0.5}]
        s = convergence_summary(scores)
        assert s["peak"] == 0.9
        assert s["trough"] == 0.1
