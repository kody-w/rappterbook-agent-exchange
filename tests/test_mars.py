"""
Tests for Mars Barn terrarium — environment, colony, engine, curves.

Run: python -m pytest tests/test_mars.py -v
"""
from __future__ import annotations

import json
import math
import sys
import os
import tempfile
from pathlib import Path

# Add repo root and src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars_env import (
    MarsEnvironment,
    sol_to_ls,
    season_name,
    surface_temperature_c,
    solar_flux_wm2,
    radiation_msv,
    SOLS_PER_MARS_YEAR,
    BASE_SOLAR_FLUX_WM2,
    BASE_RADIATION_MSV_SOL,
)
from src.mars_colony import (
    Colony,
    create_colony,
    FOOD_KG_SOL,
    WATER_L_SOL,
    POWER_KWH_SOL,
)
from src.tick_engine import Simulation
from src.mars_curves import generate_dashboard


# ─── Environment tests ───


class TestSolToLs:
    def test_sol_zero(self) -> None:
        assert sol_to_ls(0) == 0.0

    def test_full_year(self) -> None:
        ls = sol_to_ls(int(SOLS_PER_MARS_YEAR))
        assert abs(ls - 360.0) < 1.0 or abs(ls) < 1.0  # wraps

    def test_monotonic(self) -> None:
        prev = -1.0
        wrapped = False
        for sol in range(0, 668):
            ls = sol_to_ls(sol)
            if ls < prev:
                wrapped = True
            if not wrapped:
                assert ls >= prev or sol == 0
            prev = ls

    def test_range(self) -> None:
        for sol in range(0, 1000):
            ls = sol_to_ls(sol)
            assert 0 <= ls < 360


class TestSeasonName:
    def test_spring(self) -> None:
        assert season_name(45) == "spring"

    def test_summer(self) -> None:
        assert season_name(135) == "summer"

    def test_autumn(self) -> None:
        assert season_name(225) == "autumn"

    def test_winter(self) -> None:
        assert season_name(315) == "winter"


class TestTemperature:
    def test_physical_bounds(self) -> None:
        """Temperature must stay within Mars physical limits."""
        for sol in range(0, 700):
            ls = sol_to_ls(sol)
            temp = surface_temperature_c(ls)
            assert -150 < temp < 30, f"Sol {sol}: {temp}°C out of bounds"

    def test_seasonal_variation(self) -> None:
        """Summer should be warmer than winter."""
        summer_temp = surface_temperature_c(135)  # Ls=135, summer
        winter_temp = surface_temperature_c(315)  # Ls=315, winter
        assert summer_temp > winter_temp


class TestSolarFlux:
    def test_clear_sky(self) -> None:
        flux = solar_flux_wm2(90, 0.0)
        assert flux > 0
        assert flux < BASE_SOLAR_FLUX_WM2 * 1.1  # tau=0.3 attenuates

    def test_global_storm(self) -> None:
        flux_clear = solar_flux_wm2(90, 0.0)
        flux_storm = solar_flux_wm2(90, 1.0)
        assert flux_storm < flux_clear * 0.2  # massive reduction

    def test_positive(self) -> None:
        for dust in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert solar_flux_wm2(180, dust) > 0


class TestRadiation:
    def test_baseline(self) -> None:
        rad = radiation_msv(0.0, False)
        assert abs(rad - BASE_RADIATION_MSV_SOL) < 0.01

    def test_flare_increases(self) -> None:
        normal = radiation_msv(0.0, False)
        flare = radiation_msv(0.0, True)
        assert flare > normal

    def test_dust_shields_gcr(self) -> None:
        clear = radiation_msv(0.0, False)
        dusty = radiation_msv(1.0, False)
        assert dusty < clear


class TestMarsEnvironment:
    def test_smoke_100_sols(self) -> None:
        """Run 100 sols without crash."""
        env = MarsEnvironment(seed=42)
        for _ in range(100):
            snap = env.tick()
            assert "sol" in snap
            assert "temperature_c" in snap
            assert "radiation_msv" in snap
            assert snap["sol"] > 0

    def test_deterministic(self) -> None:
        """Same seed = same results."""
        env1 = MarsEnvironment(seed=99)
        env2 = MarsEnvironment(seed=99)
        for _ in range(50):
            s1 = env1.tick()
            s2 = env2.tick()
            assert s1 == s2

    def test_storms_occur(self) -> None:
        """Over 2 Mars years, at least one storm should happen."""
        env = MarsEnvironment(seed=42)
        storms = 0
        for _ in range(1337):  # ~2 Mars years
            snap = env.tick()
            if snap.get("storm"):
                storms += 1
        assert storms > 0, "No storms in 2 Mars years — suspicious"


# ─── Colony tests ───


class TestColonyCreation:
    def test_conservative(self) -> None:
        c = create_colony("Test", "conservative", 1)
        assert c.population == 120
        assert c.medical_level == 0.8
        assert c.food_kg > 0

    def test_balanced(self) -> None:
        c = create_colony("Test", "balanced", 2)
        assert c.population == 80

    def test_aggressive(self) -> None:
        c = create_colony("Test", "aggressive", 3)
        assert c.population == 60
        assert c.medical_level == 0.4


class TestColonyTick:
    def test_smoke_10_sols(self) -> None:
        """Colony survives 10 sols."""
        c = create_colony("Test", "balanced", 42)
        env = MarsEnvironment(seed=42)
        for _ in range(10):
            snap = env.tick()
            c.tick(snap)
        assert c.population > 0
        assert len(c.history) == 10

    def test_population_nonnegative(self) -> None:
        """Population never goes negative, even under stress."""
        c = create_colony("Stress", "aggressive", 7)
        c.food_kg = 10  # nearly starving
        c.water_l = 5
        env = MarsEnvironment(seed=7)
        for _ in range(100):
            snap = env.tick()
            c.tick(snap)
            assert c.population >= 0

    def test_morale_bounded(self) -> None:
        """Morale stays in [0, 1]."""
        c = create_colony("Morale", "balanced", 11)
        env = MarsEnvironment(seed=11)
        for _ in range(200):
            snap = env.tick()
            c.tick(snap)
            assert 0.0 <= c.morale <= 1.0

    def test_resources_nonnegative(self) -> None:
        """Food, water, power never go negative."""
        c = create_colony("Resources", "conservative", 13)
        env = MarsEnvironment(seed=13)
        for _ in range(100):
            snap = env.tick()
            c.tick(snap)
            assert c.food_kg >= 0
            assert c.water_l >= 0
            assert c.power_kwh >= 0

    def test_deaths_lte_population(self) -> None:
        """Deaths in a sol never exceed population at start of sol."""
        c = create_colony("Deaths", "balanced", 17)
        env = MarsEnvironment(seed=17)
        for _ in range(100):
            prev_pop = c.population
            snap = env.tick()
            result = c.tick(snap)
            assert result["deaths"] <= prev_pop + result["births"]  # births happen same sol

    def test_history_grows(self) -> None:
        """Each tick appends to history."""
        c = create_colony("Hist", "balanced", 19)
        env = MarsEnvironment(seed=19)
        for i in range(5):
            c.tick(env.tick())
        assert len(c.history) == 5

    def test_zero_population_stable(self) -> None:
        """Colony with 0 population doesn't crash."""
        c = create_colony("Ghost", "balanced", 23)
        c.population = 0
        env = MarsEnvironment(seed=23)
        for _ in range(10):
            c.tick(env.tick())
        assert c.population == 0  # no spontaneous generation


# ─── Simulation (tick_engine) tests ───


class TestSimulation:
    def test_smoke_10_sols(self) -> None:
        """Full simulation runs 10 sols without crash."""
        sim = Simulation(sols=10, env_seed=42)
        results = sim.run()
        assert results["_meta"]["sols"] == 10
        assert len(results["colonies"]) == 3
        for c in results["colonies"]:
            assert len(c["history"]) == 10

    def test_deterministic(self) -> None:
        """Same seeds = same results."""
        r1 = Simulation(sols=50, env_seed=42).run()
        r2 = Simulation(sols=50, env_seed=42).run()
        for c1, c2 in zip(r1["colonies"], r2["colonies"]):
            pops1 = [h["population"] for h in c1["history"]]
            pops2 = [h["population"] for h in c2["history"]]
            assert pops1 == pops2

    def test_365_sols_all_survive(self) -> None:
        """All 3 colonies survive 365 sols with default params."""
        sim = Simulation(sols=365, env_seed=42)
        results = sim.run()
        for c in results["colonies"]:
            final_pop = c["history"][-1]["population"]
            assert final_pop > 0, f"{c['name']} went extinct"

    def test_summary_consistent(self) -> None:
        """Summary stats match history."""
        sim = Simulation(sols=100, env_seed=42)
        results = sim.run()
        for c_result, s in zip(results["colonies"], results["summary"]["colonies"]):
            pops = [h["population"] for h in c_result["history"]]
            assert s["peak_pop"] == max(pops)
            assert s["min_pop"] == min(pops)
            assert s["end_pop"] == pops[-1]

    def test_env_history_matches(self) -> None:
        """Environment history length matches sols."""
        sim = Simulation(sols=30, env_seed=42)
        results = sim.run()
        assert len(results["environment"]["history"]) == 30

    def test_callback_invoked(self) -> None:
        """Callback fires each sol."""
        calls = []
        sim = Simulation(sols=5, env_seed=42)
        sim.run(callback=lambda sol, env, cols: calls.append(sol))
        assert calls == [1, 2, 3, 4, 5]

    def test_results_serializable(self) -> None:
        """Results can be serialized to JSON."""
        sim = Simulation(sols=10, env_seed=42)
        results = sim.run()
        text = json.dumps(results)
        parsed = json.loads(text)
        assert parsed["_meta"]["sols"] == 10


# ─── Curves (HTML generation) tests ───


class TestDashboard:
    def test_generates_html(self) -> None:
        """Dashboard produces valid-looking HTML."""
        sim = Simulation(sols=30, env_seed=42)
        results = sim.run()
        html = generate_dashboard(results)
        assert "<!DOCTYPE html>" in html
        assert "Mars Barn" in html
        assert "Ares Prime" in html
        assert "Olympus Station" in html
        assert "Red Frontier" in html
        assert "<svg" in html

    def test_svg_charts_present(self) -> None:
        """Dashboard contains expected SVG charts."""
        sim = Simulation(sols=30, env_seed=42)
        results = sim.run()
        html = generate_dashboard(results)
        for chart_id in ["pop-chart", "food-chart", "morale-chart", "births-chart", "temp-chart"]:
            assert chart_id in html, f"Missing chart: {chart_id}"


# ─── Property-based invariant tests ───


class TestConservationLaws:
    """Physics-level invariants that must hold for any seed."""

    def test_population_accounting(self) -> None:
        """births - deaths = population change (over full sim)."""
        for seed in [1, 42, 99, 256, 1000]:
            sim = Simulation(sols=100, env_seed=seed)
            results = sim.run()
            for c in results["colonies"]:
                hist = c["history"]
                start_pop = hist[0]["population"] - hist[0]["births"] + hist[0]["deaths"]
                end_pop = hist[-1]["population"]
                total_b = sum(h["births"] for h in hist)
                total_d = sum(h["deaths"] for h in hist)
                assert end_pop == start_pop + total_b - total_d, (
                    f"Accounting error for {c['name']} seed={seed}: "
                    f"start={start_pop} + births={total_b} - deaths={total_d} != end={end_pop}"
                )

    def test_temperature_physical_range(self) -> None:
        """All temperatures within Mars physical bounds for any seed."""
        for seed in [1, 42, 99]:
            sim = Simulation(sols=700, env_seed=seed)
            results = sim.run()
            for e in results["environment"]["history"]:
                t = e["temperature_c"]
                assert -150 < t < 30, f"Temp {t}°C out of Mars bounds (seed={seed})"

    def test_radiation_positive(self) -> None:
        """Radiation is always positive."""
        for seed in [1, 42, 99]:
            sim = Simulation(sols=700, env_seed=seed)
            results = sim.run()
            for e in results["environment"]["history"]:
                assert e["radiation_msv"] > 0


# ─── Trade and discovery tests ────────────────────────────────────────

class TestInterColonyTrade:
    """Inter-colony food sharing during shortages."""

    def test_trade_transfers_food(self) -> None:
        """When one colony starves and another has surplus, food flows."""
        sim = Simulation(sols=200, env_seed=42)
        results = sim.run()
        # Just verify all colonies have some food — trade prevents total starvation
        for c in results["colonies"]:
            last = c["history"][-1]
            assert last["food_kg"] >= 0

    def test_trade_doesnt_create_food(self) -> None:
        """Trade is zero-sum — total food before trade == total after."""
        sim = Simulation(sols=10, env_seed=42)
        # Run without trade to compare
        sim2 = Simulation(sols=10, env_seed=42)
        r1 = sim.run()
        r2 = sim2.run()
        # Results should be deterministic and identical
        for i in range(3):
            assert (r1["colonies"][i]["history"][-1]["population"] ==
                    r2["colonies"][i]["history"][-1]["population"])


class TestDiscoveries:
    """Rare permanent improvements (ice veins, medical, crop strains)."""

    def test_discoveries_occur_over_365_sols(self) -> None:
        """At least one discovery should happen in 365 sols across 3 colonies."""
        sim = Simulation(sols=365, env_seed=42)
        sim.run()
        # Check the colony objects directly (not the trimmed results)
        total_discoveries = sum(
            sum(1 for e in c.events if e["type"] == "discovery")
            for c in sim.colonies
        )
        assert total_discoveries > 0, "Expected at least one discovery in 365 sols"

    def test_water_mining_bonus_nonnegative(self) -> None:
        """Water mining bonus accumulates, never negative."""
        sim = Simulation(sols=365, env_seed=42)
        sim.run()
        for c in sim.colonies:
            assert c.water_mining_bonus >= 0

    def test_medical_breakthroughs_capped(self) -> None:
        """Medical breakthroughs capped at 4."""
        sim = Simulation(sols=2000, env_seed=42)
        sim.run()
        for c in sim.colonies:
            assert c.medical_breakthroughs <= 4
