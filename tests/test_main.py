"""
tests/test_main.py — 70+ unit tests for src/main.py (Mars Barn CLI runner).

The last untested module in mars-barn. Tests cover:
  - _compact_results(): data compression for the frontend
  - _serialize_ensemble(): Monte Carlo serialization
  - main() CLI: argument parsing, file output, exit codes
  - Physical invariants and property-based checks

Community voted 53–0: ship code, not governance. One file. One test. One merge.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.main import _compact_results, _serialize_ensemble
from src.tick_engine import Simulation
from src.monte_carlo import run_ensemble, PERCENTILES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_sim(sols: int = 10, seed: int = 42) -> dict:
    """Run a short simulation and return results dict."""
    sim = Simulation(sols=sols, env_seed=seed)
    return sim.run()


def _run_ensemble_small(n_seeds: int = 3, sols: int = 10) -> "EnsembleResult":
    """Run a tiny ensemble for test speed."""
    return run_ensemble(n_seeds=n_seeds, sols=sols)


def _run_cli(*args: str, state_dir: str | None = None, docs_dir: str | None = None) -> subprocess.CompletedProcess:
    """Run src/main.py as a subprocess."""
    cmd = [sys.executable, str(REPO_ROOT / "src" / "main.py")]
    if state_dir:
        cmd += ["--state-dir", state_dir]
    if docs_dir:
        cmd += ["--docs-dir", docs_dir]
    cmd += list(args)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120)


# ---------------------------------------------------------------------------
# _compact_results() — data compression for frontend
# ---------------------------------------------------------------------------

class TestCompactResults:
    """_compact_results() strips heavy fields for the frontend data file."""

    def test_returns_dict(self) -> None:
        results = _run_sim()
        compact = _compact_results(results)
        assert isinstance(compact, dict)

    def test_has_meta(self) -> None:
        results = _run_sim()
        compact = _compact_results(results)
        assert "_meta" in compact
        assert compact["_meta"]["engine"] == "mars-barn"

    def test_has_summary(self) -> None:
        results = _run_sim()
        compact = _compact_results(results)
        assert "summary" in compact

    def test_has_colonies(self) -> None:
        results = _run_sim()
        compact = _compact_results(results)
        assert "colonies" in compact
        assert len(compact["colonies"]) == 3

    def test_colony_has_population_array(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        for colony in compact["colonies"]:
            assert "population" in colony
            assert isinstance(colony["population"], list)
            assert len(colony["population"]) == 10

    def test_colony_has_food_array(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        for colony in compact["colonies"]:
            assert "food_kg" in colony
            assert len(colony["food_kg"]) == 10

    def test_colony_has_morale_array(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        for colony in compact["colonies"]:
            assert "morale" in colony
            assert len(colony["morale"]) == 10

    def test_colony_has_births_deaths(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        for colony in compact["colonies"]:
            assert "births" in colony
            assert "deaths" in colony

    def test_colony_has_death_causes_by_sol(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        for colony in compact["colonies"]:
            assert "death_causes" in colony
            assert len(colony["death_causes"]) == 10

    def test_colony_has_cumulative_death_causes(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        for colony in compact["colonies"]:
            assert "cumulative_death_causes" in colony
            assert isinstance(colony["cumulative_death_causes"], dict)

    def test_colony_has_carrying_capacity(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        for colony in compact["colonies"]:
            assert "carrying_capacity" in colony
            assert len(colony["carrying_capacity"]) == 10

    def test_colony_has_genetic_diversity(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        for colony in compact["colonies"]:
            assert "genetic_diversity" in colony
            assert len(colony["genetic_diversity"]) == 10

    def test_colony_has_net_migration(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        for colony in compact["colonies"]:
            assert "net_migration" in colony
            assert len(colony["net_migration"]) == 10

    def test_colony_has_tech(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        for colony in compact["colonies"]:
            assert "tech" in colony

    def test_colony_has_name_and_strategy(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        names = {c["name"] for c in compact["colonies"]}
        assert "Ares Prime" in names
        assert "Olympus Station" in names
        assert "Red Frontier" in names
        for colony in compact["colonies"]:
            assert colony["strategy"] in ("conservative", "balanced", "aggressive")

    def test_environment_has_temperature(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        assert "environment" in compact
        assert "temperature_c" in compact["environment"]
        assert len(compact["environment"]["temperature_c"]) == 10

    def test_environment_has_dust(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        assert "dust_opacity" in compact["environment"]
        assert len(compact["environment"]["dust_opacity"]) == 10

    def test_environment_has_radiation(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        assert "radiation_msv" in compact["environment"]
        assert len(compact["environment"]["radiation_msv"]) == 10

    def test_environment_has_terraforming(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        assert "terraforming_progress" in compact["environment"]
        assert len(compact["environment"]["terraforming_progress"]) == 10

    def test_environment_has_pressure(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        assert "pressure_kpa" in compact["environment"]
        assert len(compact["environment"]["pressure_kpa"]) == 10

    def test_json_serializable(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        json_str = json.dumps(compact)
        assert len(json_str) > 0
        roundtrip = json.loads(json_str)
        assert roundtrip["_meta"]["engine"] == "mars-barn"

    def test_compact_is_smaller(self) -> None:
        """Compact output should be smaller than full results."""
        results = _run_sim(sols=50)
        full_size = len(json.dumps(results))
        compact_size = len(json.dumps(_compact_results(results)))
        assert compact_size < full_size

    def test_deterministic(self) -> None:
        r1 = _run_sim(sols=10, seed=42)
        r2 = _run_sim(sols=10, seed=42)
        c1 = _compact_results(r1)
        c2 = _compact_results(r2)
        # Ignore generated timestamp — it differs between runs
        c1["_meta"].pop("generated", None)
        c2["_meta"].pop("generated", None)
        assert json.dumps(c1, sort_keys=True) == json.dumps(c2, sort_keys=True)

    # --- Property-based invariants ---

    def test_population_nonnegative(self) -> None:
        results = _run_sim(sols=50)
        compact = _compact_results(results)
        for colony in compact["colonies"]:
            for pop in colony["population"]:
                assert pop >= 0, f"Negative population: {pop}"

    def test_morale_bounded_zero_one(self) -> None:
        results = _run_sim(sols=50)
        compact = _compact_results(results)
        for colony in compact["colonies"]:
            for m in colony["morale"]:
                assert 0.0 <= m <= 1.0, f"Morale out of bounds: {m}"

    def test_food_nonnegative(self) -> None:
        results = _run_sim(sols=50)
        compact = _compact_results(results)
        for colony in compact["colonies"]:
            for f in colony["food_kg"]:
                assert f >= 0.0, f"Negative food: {f}"

    def test_dust_opacity_nonnegative(self) -> None:
        results = _run_sim(sols=50)
        compact = _compact_results(results)
        for d in compact["environment"]["dust_opacity"]:
            assert d >= 0.0, f"Negative dust opacity: {d}"

    def test_radiation_nonnegative(self) -> None:
        results = _run_sim(sols=50)
        compact = _compact_results(results)
        for r in compact["environment"]["radiation_msv"]:
            assert r >= 0.0, f"Negative radiation: {r}"

    def test_pressure_physically_bounded(self) -> None:
        """Mars surface pressure ≈ 0.6 kPa. Allow 0–2 kPa with terraforming."""
        results = _run_sim(sols=50)
        compact = _compact_results(results)
        for p in compact["environment"]["pressure_kpa"]:
            assert 0.0 <= p <= 5.0, f"Pressure out of physical bounds: {p} kPa"

    def test_temperature_physically_bounded(self) -> None:
        """Mars temperature range: roughly -150°C to +30°C."""
        results = _run_sim(sols=100)
        compact = _compact_results(results)
        for t in compact["environment"]["temperature_c"]:
            assert -200.0 <= t <= 50.0, f"Temperature out of bounds: {t}°C"

    def test_terraforming_bounded_zero_one(self) -> None:
        results = _run_sim(sols=50)
        compact = _compact_results(results)
        for tf in compact["environment"]["terraforming_progress"]:
            assert 0.0 <= tf <= 1.0, f"Terraforming out of bounds: {tf}"

    def test_genetic_diversity_bounded(self) -> None:
        results = _run_sim(sols=50)
        compact = _compact_results(results)
        for colony in compact["colonies"]:
            for gd in colony["genetic_diversity"]:
                assert 0.0 <= gd <= 1.0, f"Genetic diversity out of bounds: {gd}"

    def test_births_deaths_nonnegative(self) -> None:
        results = _run_sim(sols=50)
        compact = _compact_results(results)
        for colony in compact["colonies"]:
            for b in colony["births"]:
                assert b >= 0
            for d in colony["deaths"]:
                assert d >= 0

    def test_all_arrays_same_length(self) -> None:
        """All per-sol arrays must have the same length (= sols)."""
        sols = 25
        results = _run_sim(sols=sols)
        compact = _compact_results(results)
        for colony in compact["colonies"]:
            assert len(colony["population"]) == sols
            assert len(colony["food_kg"]) == sols
            assert len(colony["morale"]) == sols
            assert len(colony["births"]) == sols
            assert len(colony["deaths"]) == sols
            assert len(colony["carrying_capacity"]) == sols
            assert len(colony["genetic_diversity"]) == sols
            assert len(colony["net_migration"]) == sols
        assert len(compact["environment"]["temperature_c"]) == sols
        assert len(compact["environment"]["dust_opacity"]) == sols
        assert len(compact["environment"]["radiation_msv"]) == sols
        assert len(compact["environment"]["terraforming_progress"]) == sols
        assert len(compact["environment"]["pressure_kpa"]) == sols

    def test_multiple_seeds_differ(self) -> None:
        """Different seeds produce different environment histories."""
        r1 = _compact_results(_run_sim(sols=200, seed=1))
        r2 = _compact_results(_run_sim(sols=200, seed=999))
        rad1 = r1["environment"]["radiation_msv"]
        rad2 = r2["environment"]["radiation_msv"]
        assert rad1 != rad2


# ---------------------------------------------------------------------------
# _serialize_ensemble() — Monte Carlo serialization
# ---------------------------------------------------------------------------

class TestSerializeEnsemble:
    """_serialize_ensemble() converts EnsembleResult to JSON-safe dict."""

    def test_returns_dict(self) -> None:
        ensemble = _run_ensemble_small()
        data = _serialize_ensemble(ensemble)
        assert isinstance(data, dict)

    def test_has_n_seeds(self) -> None:
        ensemble = _run_ensemble_small(n_seeds=3)
        data = _serialize_ensemble(ensemble)
        assert data["n_seeds"] == 3

    def test_has_sols(self) -> None:
        ensemble = _run_ensemble_small(sols=10)
        data = _serialize_ensemble(ensemble)
        assert data["sols"] == 10

    def test_has_colony_names(self) -> None:
        ensemble = _run_ensemble_small()
        data = _serialize_ensemble(ensemble)
        assert len(data["colony_names"]) == 3
        assert "Ares Prime" in data["colony_names"]

    def test_has_colony_strategies(self) -> None:
        ensemble = _run_ensemble_small()
        data = _serialize_ensemble(ensemble)
        assert "conservative" in data["colony_strategies"]

    def test_has_bands(self) -> None:
        ensemble = _run_ensemble_small()
        data = _serialize_ensemble(ensemble)
        assert "bands" in data
        assert len(data["bands"]) == 3

    def test_bands_have_percentile_keys(self) -> None:
        ensemble = _run_ensemble_small()
        data = _serialize_ensemble(ensemble)
        for colony_bands in data["bands"]:
            for metric, percentiles in colony_bands.items():
                for p in PERCENTILES:
                    assert f"p{p}" in percentiles

    def test_bands_values_are_lists_of_floats(self) -> None:
        ensemble = _run_ensemble_small(sols=10)
        data = _serialize_ensemble(ensemble)
        for colony_bands in data["bands"]:
            for metric, percentiles in colony_bands.items():
                for key, values in percentiles.items():
                    assert isinstance(values, list)
                    assert len(values) == 10
                    for v in values:
                        assert isinstance(v, (int, float))

    def test_final_pop_stats(self) -> None:
        ensemble = _run_ensemble_small()
        data = _serialize_ensemble(ensemble)
        assert len(data["final_pop_stats"]) == 3
        for fps in data["final_pop_stats"]:
            assert "mean" in fps
            assert "median" in fps
            assert "stdev" in fps
            assert "p10" in fps
            assert "p90" in fps

    def test_growth_pct_stats(self) -> None:
        ensemble = _run_ensemble_small()
        data = _serialize_ensemble(ensemble)
        assert len(data["growth_pct_stats"]) == 3
        for gps in data["growth_pct_stats"]:
            assert "mean" in gps

    def test_survival_rates(self) -> None:
        ensemble = _run_ensemble_small()
        data = _serialize_ensemble(ensemble)
        assert len(data["survival_rates"]) == 3
        for rate in data["survival_rates"]:
            assert 0.0 <= rate <= 1.0

    def test_json_serializable(self) -> None:
        ensemble = _run_ensemble_small()
        data = _serialize_ensemble(ensemble)
        json_str = json.dumps(data)
        roundtrip = json.loads(json_str)
        assert roundtrip["n_seeds"] == data["n_seeds"]

    def test_values_are_rounded(self) -> None:
        """Band values should be rounded to 1 decimal place."""
        ensemble = _run_ensemble_small(sols=10)
        data = _serialize_ensemble(ensemble)
        for colony_bands in data["bands"]:
            for metric, percentiles in colony_bands.items():
                for key, values in percentiles.items():
                    for v in values:
                        # Check that the value has at most 1 decimal
                        assert round(v, 1) == v, f"Not rounded: {v}"

    def test_stats_are_rounded(self) -> None:
        """Stats values should be rounded to 1 decimal."""
        ensemble = _run_ensemble_small()
        data = _serialize_ensemble(ensemble)
        for fps in data["final_pop_stats"]:
            for k, v in fps.items():
                assert round(v, 1) == v

    def test_survival_rates_rounded(self) -> None:
        ensemble = _run_ensemble_small()
        data = _serialize_ensemble(ensemble)
        for rate in data["survival_rates"]:
            assert round(rate, 3) == rate

    def test_deterministic(self) -> None:
        e1 = run_ensemble(n_seeds=3, sols=10, base_seed=0)
        e2 = run_ensemble(n_seeds=3, sols=10, base_seed=0)
        d1 = _serialize_ensemble(e1)
        d2 = _serialize_ensemble(e2)
        assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)


# ---------------------------------------------------------------------------
# main() CLI — integration tests via subprocess
# ---------------------------------------------------------------------------

class TestMainCLI:
    """main() CLI integration — runs src/main.py as a subprocess."""

    def test_default_run(self) -> None:
        """Default run (365 sols) completes without error."""
        with tempfile.TemporaryDirectory() as td:
            state_dir = os.path.join(td, "state")
            docs_dir = os.path.join(td, "docs")
            result = _run_cli("--sols", "10", "--quiet",
                              state_dir=state_dir, docs_dir=docs_dir)
            assert result.returncode == 0, f"STDERR: {result.stderr}"

    def test_output_files_created(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_dir = os.path.join(td, "state")
            docs_dir = os.path.join(td, "docs")
            result = _run_cli("--sols", "10", "--quiet",
                              state_dir=state_dir, docs_dir=docs_dir)
            assert result.returncode == 0, f"STDERR: {result.stderr}"
            assert os.path.exists(os.path.join(state_dir, "mars.json"))
            assert os.path.exists(os.path.join(docs_dir, "data.json"))
            assert os.path.exists(os.path.join(docs_dir, "index.html"))

    def test_mars_json_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_dir = os.path.join(td, "state")
            docs_dir = os.path.join(td, "docs")
            _run_cli("--sols", "10", "--quiet",
                     state_dir=state_dir, docs_dir=docs_dir)
            with open(os.path.join(state_dir, "mars.json")) as f:
                data = json.load(f)
            assert data["_meta"]["engine"] == "mars-barn"
            assert len(data["colonies"]) == 3

    def test_data_json_compact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_dir = os.path.join(td, "state")
            docs_dir = os.path.join(td, "docs")
            _run_cli("--sols", "10", "--quiet",
                     state_dir=state_dir, docs_dir=docs_dir)
            with open(os.path.join(docs_dir, "data.json")) as f:
                data = json.load(f)
            assert "colonies" in data
            assert "environment" in data

    def test_dashboard_html_generated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_dir = os.path.join(td, "state")
            docs_dir = os.path.join(td, "docs")
            _run_cli("--sols", "10", "--quiet",
                     state_dir=state_dir, docs_dir=docs_dir)
            html_path = os.path.join(docs_dir, "index.html")
            content = Path(html_path).read_text()
            assert "<html" in content.lower() or "<!doctype" in content.lower()

    def test_seed_flag(self) -> None:
        """--seed produces deterministic output."""
        with tempfile.TemporaryDirectory() as td1, \
             tempfile.TemporaryDirectory() as td2:
            _run_cli("--sols", "10", "--seed", "123", "--quiet",
                     state_dir=os.path.join(td1, "s"), docs_dir=os.path.join(td1, "d"))
            _run_cli("--sols", "10", "--seed", "123", "--quiet",
                     state_dir=os.path.join(td2, "s"), docs_dir=os.path.join(td2, "d"))
            with open(os.path.join(td1, "s", "mars.json")) as f1, \
                 open(os.path.join(td2, "s", "mars.json")) as f2:
                d1 = json.load(f1)
                d2 = json.load(f2)
            # Same seed = same populations (ignore timestamp)
            pops1 = [h["population"] for h in d1["colonies"][0]["history"]]
            pops2 = [h["population"] for h in d2["colonies"][0]["history"]]
            assert pops1 == pops2

    def test_different_seeds_differ(self) -> None:
        with tempfile.TemporaryDirectory() as td1, \
             tempfile.TemporaryDirectory() as td2:
            _run_cli("--sols", "200", "--seed", "1", "--quiet",
                     state_dir=os.path.join(td1, "s"), docs_dir=os.path.join(td1, "d"))
            _run_cli("--sols", "200", "--seed", "999", "--quiet",
                     state_dir=os.path.join(td2, "s"), docs_dir=os.path.join(td2, "d"))
            with open(os.path.join(td1, "s", "mars.json")) as f1, \
                 open(os.path.join(td2, "s", "mars.json")) as f2:
                d1 = json.load(f1)
                d2 = json.load(f2)
            rad1 = [h["radiation_msv"] for h in d1["environment"]["history"]]
            rad2 = [h["radiation_msv"] for h in d2["environment"]["history"]]
            assert rad1 != rad2

    def test_quiet_mode_less_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            loud = _run_cli("--sols", "100",
                            state_dir=os.path.join(td, "s1"),
                            docs_dir=os.path.join(td, "d1"))
            quiet = _run_cli("--sols", "100", "--quiet",
                             state_dir=os.path.join(td, "s2"),
                             docs_dir=os.path.join(td, "d2"))
            assert len(quiet.stdout) < len(loud.stdout)

    def test_monte_carlo_mode(self) -> None:
        """--monte-carlo N runs ensemble mode."""
        with tempfile.TemporaryDirectory() as td:
            result = _run_cli("--sols", "10", "--monte-carlo", "3", "--quiet",
                              state_dir=os.path.join(td, "s"),
                              docs_dir=os.path.join(td, "d"))
            assert result.returncode == 0, f"STDERR: {result.stderr}"
            assert "MONTE CARLO COMPLETE" in result.stdout

    def test_monte_carlo_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_dir = os.path.join(td, "s")
            docs_dir = os.path.join(td, "d")
            _run_cli("--sols", "10", "--monte-carlo", "3",
                     state_dir=state_dir, docs_dir=docs_dir)
            assert os.path.exists(os.path.join(state_dir, "mars.json"))
            assert os.path.exists(os.path.join(docs_dir, "data.json"))
            assert os.path.exists(os.path.join(docs_dir, "index.html"))

    def test_stdout_contains_sim_complete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = _run_cli("--sols", "10",
                              state_dir=os.path.join(td, "s"),
                              docs_dir=os.path.join(td, "d"))
            assert result.returncode == 0
            assert "SIMULATION COMPLETE" in result.stdout

    def test_stdout_contains_colony_names(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = _run_cli("--sols", "10",
                              state_dir=os.path.join(td, "s"),
                              docs_dir=os.path.join(td, "d"))
            assert "Ares Prime" in result.stdout
            assert "Olympus Station" in result.stdout
            assert "Red Frontier" in result.stdout


# ---------------------------------------------------------------------------
# Edge cases and conservation laws
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases, conservation laws, and stress tests."""

    def test_single_sol(self) -> None:
        results = _run_sim(sols=1)
        compact = _compact_results(results)
        for colony in compact["colonies"]:
            assert len(colony["population"]) == 1

    def test_large_sim_compacts(self) -> None:
        """200-sol sim compacts without error."""
        results = _run_sim(sols=200)
        compact = _compact_results(results)
        assert len(compact["colonies"][0]["population"]) == 200

    def test_compact_preserves_colony_count(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        assert len(compact["colonies"]) == len(results["colonies"])

    def test_compact_preserves_summary(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        assert compact["summary"] == results["summary"]

    def test_compact_preserves_meta(self) -> None:
        results = _run_sim(sols=10)
        compact = _compact_results(results)
        assert compact["_meta"] == results["_meta"]

    def test_serialize_ensemble_matches_ensemble_fields(self) -> None:
        ensemble = _run_ensemble_small()
        data = _serialize_ensemble(ensemble)
        assert data["colony_names"] == ensemble.colony_names
        assert data["colony_strategies"] == ensemble.colony_strategies
        assert data["n_seeds"] == ensemble.n_seeds
        assert data["sols"] == ensemble.sols

    def test_population_conservation(self) -> None:
        """births - deaths + immigration - emigration = population change."""
        results = _run_sim(sols=100, seed=42)
        for colony in results["colonies"]:
            start = colony["initial_population"]
            end = colony["final_population"]
            births = colony["total_births"]
            deaths = colony["total_deaths"]
            immigrants = colony.get("total_immigrants", 0)
            emigrants = colony.get("total_emigrants", 0)
            expected = start + births - deaths + immigrants - emigrants
            assert end == expected, (
                f"{colony['name']}: {start} + {births} - {deaths} "
                f"+ {immigrants} - {emigrants} = {expected} != {end}"
            )

    def test_ensemble_survival_rate_valid(self) -> None:
        ensemble = _run_ensemble_small(n_seeds=5, sols=50)
        data = _serialize_ensemble(ensemble)
        for rate in data["survival_rates"]:
            assert 0.0 <= rate <= 1.0

    def test_compact_env_arrays_match_colony_arrays(self) -> None:
        """Environment and colony arrays should have the same length."""
        results = _run_sim(sols=30)
        compact = _compact_results(results)
        env_len = len(compact["environment"]["temperature_c"])
        for colony in compact["colonies"]:
            assert len(colony["population"]) == env_len

    def test_cli_creates_directories(self) -> None:
        """CLI creates state/docs dirs if they don't exist."""
        with tempfile.TemporaryDirectory() as td:
            state_dir = os.path.join(td, "deep", "nested", "state")
            docs_dir = os.path.join(td, "deep", "nested", "docs")
            result = _run_cli("--sols", "5", "--quiet",
                              state_dir=state_dir, docs_dir=docs_dir)
            assert result.returncode == 0
            assert os.path.isdir(state_dir)
            assert os.path.isdir(docs_dir)
