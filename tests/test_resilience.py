"""
tests/test_resilience.py — Tests for src/resilience.py stress-test harness.

Proves resilience.py works, the import chain is intact, and
stress_test produces valid reports.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.resilience import ResilienceReport, stress_test


class TestResilienceReport:
    """ResilienceReport data structure."""

    def test_empty_report(self) -> None:
        r = ResilienceReport()
        assert r.scenarios_run == 0
        assert r.failures == []
        assert r.survivors == []

    def test_to_dict_keys(self) -> None:
        r = ResilienceReport(scenarios_run=1, summary="ok")
        d = r.to_dict()
        assert "scenarios_run" in d
        assert "failures" in d
        assert "survivors" in d
        assert "summary" in d

    def test_to_dict_roundtrip(self) -> None:
        r = ResilienceReport(scenarios_run=3, survivors=[100, 200, 150])
        d = r.to_dict()
        assert d["scenarios_run"] == 3
        assert d["survivors"] == [100, 200, 150]


class TestStressTest:
    """stress_test() integration."""

    def test_runs_without_crash(self) -> None:
        report = stress_test(sols=50, env_seed=42, n_scenarios=2)
        assert report.scenarios_run == 2

    def test_survivors_nonneg(self) -> None:
        report = stress_test(sols=100, env_seed=42, n_scenarios=3)
        for s in report.survivors:
            assert s >= 0

    def test_summary_present(self) -> None:
        report = stress_test(sols=50, env_seed=42, n_scenarios=2)
        assert len(report.summary) > 0
        assert "scenarios" in report.summary

    def test_failures_are_dicts(self) -> None:
        report = stress_test(sols=50, env_seed=42, n_scenarios=3)
        for f in report.failures:
            assert "seed" in f
            assert "colony" in f

    def test_deterministic(self) -> None:
        r1 = stress_test(sols=50, env_seed=42, n_scenarios=2)
        r2 = stress_test(sols=50, env_seed=42, n_scenarios=2)
        assert r1.survivors == r2.survivors
        assert r1.failures == r2.failures

    def test_default_params(self) -> None:
        """Default 5 scenarios, 365 sols."""
        report = stress_test()
        assert report.scenarios_run == 5
        assert len(report.survivors) == 5

    def test_report_to_dict_serializable(self) -> None:
        """to_dict() produces JSON-safe output."""
        import json
        report = stress_test(sols=30, n_scenarios=2)
        d = report.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
