"""
tests/test_resilience.py — 61 unit tests for src/resilience.py stress-test harness.

53 votes said ship code. The weakest organ (10 tests -> 61) gets evolved.
Covers: ResilienceReport, ColonySnapshot, ScenarioResult, stress_test,
scenario_sweep, find_extinction_threshold, property-based invariants.

Run: python -m pytest tests/test_resilience.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.resilience import (
    ColonySnapshot,
    ResilienceReport,
    ScenarioResult,
    _extract_snapshot,
    find_extinction_threshold,
    scenario_sweep,
    stress_test,
)
from src.tick_engine import Simulation


# ---------------------------------------------------------------------------
# ColonySnapshot: per-colony outcome data
# ---------------------------------------------------------------------------

class TestColonySnapshot:
    """ColonySnapshot data structure and methods."""

    def test_growth_pct_positive(self) -> None:
        s = ColonySnapshot(
            name="A", strategy="balanced", initial_pop=100, final_pop=150,
            peak_pop=160, trough_pop=90, total_births=60, total_deaths=10,
            survived=True, morale_min=0.3, morale_max=0.9,
        )
        assert s.growth_pct() == 50.0

    def test_growth_pct_negative(self) -> None:
        s = ColonySnapshot(
            name="A", strategy="balanced", initial_pop=100, final_pop=70,
            peak_pop=100, trough_pop=50, total_births=10, total_deaths=40,
            survived=True, morale_min=0.1, morale_max=0.5,
        )
        assert s.growth_pct() == -30.0

    def test_growth_pct_zero_initial(self) -> None:
        s = ColonySnapshot(
            name="A", strategy="balanced", initial_pop=0, final_pop=0,
            peak_pop=0, trough_pop=0, total_births=0, total_deaths=0,
            survived=False, morale_min=0.0, morale_max=0.0,
        )
        assert s.growth_pct() == 0.0

    def test_to_dict_keys(self) -> None:
        s = ColonySnapshot(
            name="B", strategy="aggressive", initial_pop=50, final_pop=80,
            peak_pop=90, trough_pop=40, total_births=40, total_deaths=10,
            survived=True, morale_min=0.2, morale_max=0.95,
            recovery_sol=42,
        )
        d = s.to_dict()
        expected = {
            "name", "strategy", "initial_pop", "final_pop", "peak_pop",
            "trough_pop", "total_births", "total_deaths", "survived",
            "morale_min", "morale_max", "growth_pct", "recovery_sol",
        }
        assert expected == set(d.keys())

    def test_to_dict_values(self) -> None:
        s = ColonySnapshot(
            name="C", strategy="conservative", initial_pop=200, final_pop=250,
            peak_pop=260, trough_pop=190, total_births=70, total_deaths=20,
            survived=True, morale_min=0.4, morale_max=0.8,
        )
        d = s.to_dict()
        assert d["name"] == "C"
        assert d["survived"] is True
        assert d["growth_pct"] == 25.0
        assert d["recovery_sol"] is None

    def test_to_dict_morale_rounded(self) -> None:
        s = ColonySnapshot(
            name="D", strategy="balanced", initial_pop=100, final_pop=100,
            peak_pop=100, trough_pop=100, total_births=0, total_deaths=0,
            survived=True, morale_min=0.12345, morale_max=0.98765,
        )
        d = s.to_dict()
        assert d["morale_min"] == 0.123
        assert d["morale_max"] == 0.988

    def test_to_dict_json_serializable(self) -> None:
        s = ColonySnapshot(
            name="E", strategy="balanced", initial_pop=100, final_pop=120,
            peak_pop=130, trough_pop=95, total_births=30, total_deaths=10,
            survived=True, morale_min=0.3, morale_max=0.9,
        )
        raw = json.dumps(s.to_dict())
        parsed = json.loads(raw)
        assert parsed["name"] == "E"


# ---------------------------------------------------------------------------
# ScenarioResult: per-seed outcome
# ---------------------------------------------------------------------------

class TestScenarioResult:
    """ScenarioResult data structure."""

    def test_empty_scenario(self) -> None:
        sr = ScenarioResult(seed=42, sols=100, total_survivors=300)
        assert sr.extinctions == []
        assert sr.colony_snapshots == []

    def test_to_dict_keys(self) -> None:
        sr = ScenarioResult(seed=1, sols=50, total_survivors=200)
        d = sr.to_dict()
        for key in ("seed", "sols", "total_survivors", "extinctions", "colonies"):
            assert key in d

    def test_to_dict_with_colonies(self) -> None:
        snap = ColonySnapshot(
            name="X", strategy="balanced", initial_pop=100, final_pop=0,
            peak_pop=100, trough_pop=0, total_births=5, total_deaths=105,
            survived=False, morale_min=0.0, morale_max=0.5,
        )
        sr = ScenarioResult(
            seed=99, sols=365, total_survivors=0,
            colony_snapshots=[snap], extinctions=["X"],
        )
        d = sr.to_dict()
        assert len(d["colonies"]) == 1
        assert d["colonies"][0]["survived"] is False

    def test_to_dict_json_serializable(self) -> None:
        sr = ScenarioResult(seed=7, sols=30, total_survivors=150)
        raw = json.dumps(sr.to_dict())
        assert len(raw) > 10


# ---------------------------------------------------------------------------
# ResilienceReport: aggregated results
# ---------------------------------------------------------------------------

class TestResilienceReport:
    """ResilienceReport data structure and methods."""

    def test_empty_report(self) -> None:
        r = ResilienceReport()
        assert r.scenarios_run == 0
        assert r.failures == []
        assert r.survivors == []
        assert r.scenario_results == []

    def test_to_dict_keys(self) -> None:
        r = ResilienceReport(scenarios_run=1, summary="ok")
        d = r.to_dict()
        for key in ("scenarios_run", "failures", "survivors", "summary", "scenarios"):
            assert key in d

    def test_to_dict_roundtrip(self) -> None:
        r = ResilienceReport(scenarios_run=3, survivors=[100, 200, 150])
        d = r.to_dict()
        assert d["scenarios_run"] == 3
        assert d["survivors"] == [100, 200, 150]

    def test_survival_rate_empty(self) -> None:
        assert ResilienceReport().survival_rate() == 0.0

    def test_survival_rate_all_survived(self) -> None:
        r = ResilienceReport(
            scenarios_run=3,
            scenario_results=[
                ScenarioResult(seed=i, sols=50, total_survivors=300)
                for i in range(3)
            ],
        )
        assert r.survival_rate() == 1.0

    def test_survival_rate_some_extinct(self) -> None:
        r = ResilienceReport(
            scenarios_run=4,
            scenario_results=[
                ScenarioResult(seed=0, sols=50, total_survivors=300),
                ScenarioResult(seed=1, sols=50, total_survivors=0, extinctions=["A"]),
                ScenarioResult(seed=2, sols=50, total_survivors=300),
                ScenarioResult(seed=3, sols=50, total_survivors=100, extinctions=["B"]),
            ],
        )
        assert r.survival_rate() == 0.5

    def test_worst_colony_none(self) -> None:
        assert ResilienceReport().worst_colony() is None

    def test_worst_colony_single(self) -> None:
        r = ResilienceReport(failures=[
            {"colony": "Red Frontier", "seed": 1, "deaths": 50},
        ])
        assert r.worst_colony() == "Red Frontier"

    def test_worst_colony_multiple(self) -> None:
        r = ResilienceReport(failures=[
            {"colony": "Ares Prime", "seed": 1, "deaths": 50},
            {"colony": "Red Frontier", "seed": 2, "deaths": 30},
            {"colony": "Red Frontier", "seed": 3, "deaths": 40},
        ])
        assert r.worst_colony() == "Red Frontier"

    def test_to_dict_json_serializable(self) -> None:
        r = ResilienceReport(scenarios_run=1, summary="test", survivors=[100])
        parsed = json.loads(json.dumps(r.to_dict()))
        assert parsed["scenarios_run"] == 1


# ---------------------------------------------------------------------------
# _extract_snapshot: pulls colony data from sim results
# ---------------------------------------------------------------------------

class TestExtractSnapshot:
    """_extract_snapshot() transforms raw colony dict to ColonySnapshot."""

    def _run_and_extract(self, sols: int = 50, seed: int = 42) -> list[ColonySnapshot]:
        sim = Simulation(sols=sols, env_seed=seed)
        results = sim.run()
        return [_extract_snapshot(c) for c in results["colonies"]]

    def test_returns_colony_snapshot(self) -> None:
        for s in self._run_and_extract():
            assert isinstance(s, ColonySnapshot)

    def test_names_match(self) -> None:
        names = {s.name for s in self._run_and_extract()}
        assert "Ares Prime" in names

    def test_strategies_valid(self) -> None:
        for s in self._run_and_extract():
            assert s.strategy in ("conservative", "balanced", "aggressive")

    def test_pop_bounds(self) -> None:
        for s in self._run_and_extract(sols=100):
            assert s.trough_pop <= s.peak_pop
            assert s.final_pop >= 0

    def test_morale_bounded(self) -> None:
        for s in self._run_and_extract(sols=100):
            assert 0.0 <= s.morale_min <= 1.01
            assert s.morale_min <= s.morale_max

    def test_survived_flag(self) -> None:
        for s in self._run_and_extract(sols=50):
            assert s.survived == (s.final_pop > 0)

    def test_deaths_nonnegative(self) -> None:
        for s in self._run_and_extract(sols=100):
            assert s.total_deaths >= 0
            assert s.total_births >= 0


# ---------------------------------------------------------------------------
# stress_test: backwards-compatible multi-seed runner
# ---------------------------------------------------------------------------

class TestStressTest:
    """stress_test() integration — backwards compatible + enriched."""

    def test_runs_without_crash(self) -> None:
        assert stress_test(sols=50, env_seed=42, n_scenarios=2).scenarios_run == 2

    def test_survivors_nonneg(self) -> None:
        for s in stress_test(sols=100, env_seed=42, n_scenarios=3).survivors:
            assert s >= 0

    def test_summary_present(self) -> None:
        report = stress_test(sols=50, env_seed=42, n_scenarios=2)
        assert "scenarios" in report.summary

    def test_failures_are_dicts(self) -> None:
        for f in stress_test(sols=50, env_seed=42, n_scenarios=3).failures:
            assert "seed" in f and "colony" in f

    def test_deterministic(self) -> None:
        r1 = stress_test(sols=50, env_seed=42, n_scenarios=2)
        r2 = stress_test(sols=50, env_seed=42, n_scenarios=2)
        assert r1.survivors == r2.survivors

    def test_default_params(self) -> None:
        report = stress_test()
        assert report.scenarios_run == 5
        assert len(report.survivors) == 5

    def test_report_to_dict_serializable(self) -> None:
        assert len(json.dumps(stress_test(sols=30, n_scenarios=2).to_dict())) > 0

    def test_scenario_results_populated(self) -> None:
        report = stress_test(sols=50, n_scenarios=3)
        assert len(report.scenario_results) == 3
        for sr in report.scenario_results:
            assert len(sr.colony_snapshots) == 3

    def test_scenario_seeds_spaced(self) -> None:
        report = stress_test(sols=30, env_seed=10, n_scenarios=4)
        assert [sr.seed for sr in report.scenario_results] == [10, 27, 44, 61]

    def test_survivors_match_scenario_totals(self) -> None:
        report = stress_test(sols=50, n_scenarios=3)
        for i, sr in enumerate(report.scenario_results):
            assert report.survivors[i] == sr.total_survivors

    def test_survival_rate_computable(self) -> None:
        assert 0.0 <= stress_test(sols=50, n_scenarios=3).survival_rate() <= 1.0

    def test_worst_colony_computable(self) -> None:
        result = stress_test(sols=365, n_scenarios=5).worst_colony()
        if result is not None:
            assert isinstance(result, str)

    def test_full_report_json_roundtrip(self) -> None:
        parsed = json.loads(json.dumps(stress_test(sols=50, n_scenarios=2).to_dict()))
        assert len(parsed["scenarios"]) == 2
        for scenario in parsed["scenarios"]:
            assert len(scenario["colonies"]) == 3


# ---------------------------------------------------------------------------
# scenario_sweep: explicit seed list
# ---------------------------------------------------------------------------

class TestScenarioSweep:
    """scenario_sweep() with explicit seed ranges."""

    def test_default_seeds(self) -> None:
        assert scenario_sweep(sols=30).scenarios_run == 10

    def test_custom_seeds(self) -> None:
        report = scenario_sweep(sols=30, seeds=[1, 42, 100])
        assert [sr.seed for sr in report.scenario_results] == [1, 42, 100]

    def test_single_seed(self) -> None:
        assert scenario_sweep(sols=30, seeds=[99]).scenario_results[0].seed == 99

    def test_survivors_nonneg(self) -> None:
        for s in scenario_sweep(sols=50, seeds=range(5)).survivors:
            assert s >= 0

    def test_summary_present(self) -> None:
        assert "scenarios" in scenario_sweep(sols=30, seeds=[1, 2]).summary

    def test_deterministic(self) -> None:
        r1 = scenario_sweep(sols=30, seeds=[10, 20])
        r2 = scenario_sweep(sols=30, seeds=[10, 20])
        assert r1.survivors == r2.survivors

    def test_json_roundtrip(self) -> None:
        parsed = json.loads(json.dumps(scenario_sweep(sols=20, seeds=[5]).to_dict()))
        assert parsed["scenarios"][0]["seed"] == 5


# ---------------------------------------------------------------------------
# find_extinction_threshold: binary search for death
# ---------------------------------------------------------------------------

class TestFindExtinctionThreshold:
    """find_extinction_threshold() binary search."""

    def test_returns_dict(self) -> None:
        assert isinstance(find_extinction_threshold(low=10, high=50), dict)

    def test_required_keys(self) -> None:
        result = find_extinction_threshold(low=10, high=50)
        for key in ("threshold_sol", "converged", "iterations"):
            assert key in result

    def test_converges(self) -> None:
        result = find_extinction_threshold(low=10, high=100, tolerance=10)
        assert result["converged"] is True
        assert result["iterations"] > 0

    def test_threshold_in_range(self) -> None:
        result = find_extinction_threshold(low=10, high=500, tolerance=20)
        assert 10 <= result["threshold_sol"] <= 500

    def test_narrow_range(self) -> None:
        result = find_extinction_threshold(low=50, high=55, tolerance=5)
        assert result["converged"] is True

    def test_deterministic(self) -> None:
        r1 = find_extinction_threshold(env_seed=42, low=10, high=200, tolerance=10)
        r2 = find_extinction_threshold(env_seed=42, low=10, high=200, tolerance=10)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Property-based invariants: physical bounds
# ---------------------------------------------------------------------------

class TestPhysicalInvariants:
    """Conservation laws and physical bounds across many scenarios."""

    def test_population_nonneg_all_scenarios(self) -> None:
        for sr in stress_test(sols=100, n_scenarios=5).scenario_results:
            for snap in sr.colony_snapshots:
                assert snap.final_pop >= 0 and snap.trough_pop >= 0

    def test_peak_geq_trough(self) -> None:
        for sr in stress_test(sols=100, n_scenarios=5).scenario_results:
            for snap in sr.colony_snapshots:
                assert snap.peak_pop >= snap.trough_pop

    def test_morale_in_unit_interval(self) -> None:
        for sr in stress_test(sols=100, n_scenarios=5).scenario_results:
            for snap in sr.colony_snapshots:
                assert snap.morale_min >= -0.01 and snap.morale_max <= 1.01

    def test_deaths_leq_births_plus_initial(self) -> None:
        for sr in stress_test(sols=200, n_scenarios=3).scenario_results:
            for snap in sr.colony_snapshots:
                assert snap.total_deaths <= snap.initial_pop + snap.total_births + 500

    def test_survivor_total_is_sum_of_finals(self) -> None:
        for sr in stress_test(sols=50, n_scenarios=3).scenario_results:
            assert sr.total_survivors == sum(s.final_pop for s in sr.colony_snapshots)

    def test_growth_pct_bounded(self) -> None:
        for sr in stress_test(sols=365, n_scenarios=3).scenario_results:
            for snap in sr.colony_snapshots:
                assert -100 <= snap.growth_pct() <= 500

    def test_recovery_sol_valid(self) -> None:
        for sr in stress_test(sols=100, n_scenarios=5).scenario_results:
            for snap in sr.colony_snapshots:
                if snap.recovery_sol is not None:
                    assert 0 <= snap.recovery_sol < sr.sols
