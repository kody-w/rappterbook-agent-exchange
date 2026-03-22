"""
Tests for Monte Carlo simulation runner.

Run: python -m pytest tests/test_monte_carlo.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.monte_carlo import run_ensemble, _percentile, PERCENTILES
from src.mars_curves import generate_dashboard, _build_events_js, _build_mc_js


class TestPercentile:
    def test_empty(self) -> None:
        assert _percentile([], 50) == 0.0

    def test_single_value(self) -> None:
        assert _percentile([42.0], 50) == 42.0

    def test_median_odd(self) -> None:
        assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0

    def test_p0_is_min(self) -> None:
        assert _percentile([10.0, 20.0, 30.0], 0) == 10.0

    def test_p100_is_max(self) -> None:
        assert _percentile([10.0, 20.0, 30.0], 100) == 30.0

    def test_monotonic(self) -> None:
        data = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]
        prev = -1.0
        for p in range(0, 101, 10):
            val = _percentile(data, p)
            assert val >= prev
            prev = val


class TestEnsembleSmoke:
    def test_runs_3_seeds(self) -> None:
        result = run_ensemble(n_seeds=3, sols=30)
        assert result.n_seeds == 3
        assert result.sols == 30
        assert len(result.colony_names) == 3
        assert len(result.bands) == 3
        assert result.canonical_results is not None

    def test_bands_have_all_percentiles(self) -> None:
        result = run_ensemble(n_seeds=3, sols=20)
        for ci in range(3):
            for metric in ["population", "morale", "food_kg",
                           "genetic_diversity", "carrying_capacity"]:
                assert metric in result.bands[ci]
                bands = result.bands[ci][metric]
                assert len(bands) == len(PERCENTILES)
                for band in bands:
                    assert len(band) == 20

    def test_bands_ordered(self) -> None:
        result = run_ensemble(n_seeds=5, sols=30)
        for ci in range(3):
            bands = result.bands[ci]["population"]
            for sol_idx in range(30):
                vals = [bands[pi][sol_idx] for pi in range(len(PERCENTILES))]
                for i in range(len(vals) - 1):
                    assert vals[i] <= vals[i + 1] + 0.01

    def test_survival_rates_valid(self) -> None:
        result = run_ensemble(n_seeds=5, sols=50)
        for rate in result.survival_rates:
            assert 0.0 <= rate <= 1.0

    def test_final_pop_stats_valid(self) -> None:
        result = run_ensemble(n_seeds=5, sols=50)
        for fps in result.final_pop_stats:
            assert fps["mean"] >= 0
            assert fps["stdev"] >= 0
            assert fps["min"] <= fps["max"]
            assert fps["p10"] <= fps["p90"]

    def test_deterministic_same_seeds(self) -> None:
        r1 = run_ensemble(n_seeds=3, sols=20, base_seed=100)
        r2 = run_ensemble(n_seeds=3, sols=20, base_seed=100)
        for ci in range(3):
            assert r1.bands[ci]["population"][2] == r2.bands[ci]["population"][2]


class TestConservationLawsMC:
    def test_population_nonnegative_all_bands(self) -> None:
        result = run_ensemble(n_seeds=5, sols=50)
        for ci in range(3):
            for band in result.bands[ci]["population"]:
                for val in band:
                    assert val >= 0

    def test_morale_bounded_all_bands(self) -> None:
        result = run_ensemble(n_seeds=5, sols=50)
        for ci in range(3):
            for band in result.bands[ci]["morale"]:
                for val in band:
                    assert -0.01 <= val <= 1.01


class TestDashboardWithMC:
    def test_mc_dashboard_generates(self) -> None:
        result = run_ensemble(n_seeds=3, sols=20)
        mc_data = {
            "n_seeds": result.n_seeds, "sols": result.sols,
            "colony_names": result.colony_names,
            "colony_strategies": result.colony_strategies,
            "bands": [{
                metric: {f"p{PERCENTILES[pi]}": [round(v, 1) for v in bands[pi]]
                         for pi in range(len(PERCENTILES))}
                for metric, bands in cb.items()
            } for cb in result.bands],
            "final_pop_stats": [{k: round(v, 1) for k, v in fps.items()}
                                for fps in result.final_pop_stats],
            "growth_pct_stats": [{k: round(v, 1) for k, v in gps.items()}
                                 for gps in result.growth_pct_stats],
            "survival_rates": result.survival_rates,
        }
        html = generate_dashboard(result.canonical_results, mc_data=mc_data)
        assert "Monte Carlo" in html
        assert "mc-pop-chart" in html

    def test_dashboard_without_mc(self) -> None:
        from src.tick_engine import Simulation
        sim = Simulation(sols=20)
        html = generate_dashboard(sim.run())
        assert "Mars Barn" in html
        assert "mc-pop-box" in html


class TestEventsExtraction:
    def test_build_events_js(self) -> None:
        colonies = [{"name": "Test", "events": [
            {"sol": 10, "type": "epidemic_start", "strain": "Mars Flu"},
            {"sol": 30, "type": "supply_ship", "count": 20},
            {"sol": 50, "type": "discovery", "kind": "ice_vein"},
            {"sol": 60, "type": "storm", "kind": "regional"},
        ]}]
        js = _build_events_js(colonies)
        assert "EVENTS" in js
        assert "epidemic_start" in js

    def test_build_mc_js(self) -> None:
        js = _build_mc_js({"n_seeds": 5, "bands": []})
        assert "MC" in js
        assert "n_seeds" in js
