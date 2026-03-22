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
    """Unit tests for the percentile helper."""

    def test_empty(self) -> None:
        assert _percentile([], 50) == 0.0

    def test_single_value(self) -> None:
        assert _percentile([42.0], 50) == 42.0

    def test_median_odd(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(data, 50) == 3.0

    def test_p0_is_min(self) -> None:
        data = [10.0, 20.0, 30.0]
        assert _percentile(data, 0) == 10.0

    def test_p100_is_max(self) -> None:
        data = [10.0, 20.0, 30.0]
        assert _percentile(data, 100) == 30.0

    def test_monotonic(self) -> None:
        """Higher percentiles should return higher or equal values."""
        data = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]
        prev = -1.0
        for p in range(0, 101, 10):
            val = _percentile(data, p)
            assert val >= prev, f"p{p}={val} < p{p-10}={prev}"
            prev = val


class TestEnsembleSmoke:
    """Smoke tests for the ensemble runner — small N to keep fast."""

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
        """p10 ≤ p25 ≤ p50 ≤ p75 ≤ p90 at every sol."""
        result = run_ensemble(n_seeds=5, sols=30)
        for ci in range(3):
            bands = result.bands[ci]["population"]
            for sol_idx in range(30):
                vals = [bands[pi][sol_idx] for pi in range(len(PERCENTILES))]
                for i in range(len(vals) - 1):
                    assert vals[i] <= vals[i + 1] + 0.01, (
                        f"Colony {ci}, sol {sol_idx}: "
                        f"p{PERCENTILES[i]}={vals[i]} > p{PERCENTILES[i+1]}={vals[i+1]}"
                    )

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

    def test_growth_pct_stats_computed(self) -> None:
        result = run_ensemble(n_seeds=3, sols=30)
        for gps in result.growth_pct_stats:
            assert "mean" in gps
            assert "stdev" in gps
            assert "p10" in gps
            assert "p90" in gps

    def test_deterministic_same_seeds(self) -> None:
        """Same base_seed should produce same results."""
        r1 = run_ensemble(n_seeds=3, sols=20, base_seed=100)
        r2 = run_ensemble(n_seeds=3, sols=20, base_seed=100)
        for ci in range(3):
            p50_a = r1.bands[ci]["population"][2]  # p50
            p50_b = r2.bands[ci]["population"][2]
            assert p50_a == p50_b


class TestConservationLawsMC:
    """Physics invariants must hold across all MC seeds."""

    def test_population_nonnegative_all_bands(self) -> None:
        result = run_ensemble(n_seeds=5, sols=50)
        for ci in range(3):
            for band in result.bands[ci]["population"]:
                for val in band:
                    assert val >= 0, f"Negative population in band"

    def test_morale_bounded_all_bands(self) -> None:
        result = run_ensemble(n_seeds=5, sols=50)
        for ci in range(3):
            for band in result.bands[ci]["morale"]:
                for val in band:
                    assert -0.01 <= val <= 1.01, f"Morale out of bounds: {val}"


class TestDashboardWithMC:
    """Dashboard generation with MC data."""

    def test_mc_dashboard_generates(self) -> None:
        result = run_ensemble(n_seeds=3, sols=20)
        mc_data = {
            "n_seeds": result.n_seeds,
            "sols": result.sols,
            "colony_names": result.colony_names,
            "colony_strategies": result.colony_strategies,
            "bands": [
                {
                    metric: {
                        f"p{PERCENTILES[pi]}": [round(v, 1) for v in bands[pi]]
                        for pi in range(len(PERCENTILES))
                    }
                    for metric, bands in cb.items()
                }
                for cb in result.bands
            ],
            "final_pop_stats": [
                {k: round(v, 1) for k, v in fps.items()}
                for fps in result.final_pop_stats
            ],
            "growth_pct_stats": [
                {k: round(v, 1) for k, v in gps.items()}
                for gps in result.growth_pct_stats
            ],
            "survival_rates": result.survival_rates,
        }
        html = generate_dashboard(result.canonical_results, mc_data=mc_data)
        assert "Monte Carlo" in html
        assert "mc-pop-chart" in html
        assert "confidence" in html.lower() or "p10" in html.lower()

    def test_dashboard_without_mc(self) -> None:
        """Dashboard still works without MC data."""
        from src.tick_engine import Simulation
        sim = Simulation(sols=20)
        results = sim.run()
        html = generate_dashboard(results)
        assert "Mars Barn" in html
        assert "mc-pop-box" in html  # exists but hidden


class TestEventsExtraction:
    """Test event extraction for timeline annotations."""

    def test_build_events_js(self) -> None:
        colonies = [{
            "name": "Test",
            "events": [
                {"sol": 10, "type": "epidemic_start", "strain": "Mars Flu"},
                {"sol": 30, "type": "supply_ship", "count": 20},
                {"sol": 50, "type": "discovery", "kind": "ice_vein"},
                {"sol": 60, "type": "storm", "kind": "regional"},
            ]
        }]
        js = _build_events_js(colonies)
        assert "EVENTS" in js
        assert "epidemic_start" in js
        assert "supply_ship" in js
        assert "discovery" in js

    def test_build_mc_js_null(self) -> None:
        js = _build_mc_js({"n_seeds": 5, "bands": []})
        assert "MC" in js
        assert "n_seeds" in js
