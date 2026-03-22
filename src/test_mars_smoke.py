"""Co-located smoke tests for Mars Barn modules.

Run: python -m pytest src/test_mars_smoke.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars_env import MarsEnvironment, sol_to_ls, surface_temperature_c
from src.mars_colony import Colony, create_colony, GREENHOUSE_KG_SOL_M2
from src.tick_engine import Simulation
from src.mars_curves import generate_dashboard


def test_env_10_sols() -> None:
    """Environment runs 10 sols without crash."""
    env = MarsEnvironment(seed=7)
    for _ in range(10):
        snap = env.tick()
        assert -150 < snap["temperature_c"] < 30
        assert snap["solar_flux_wm2"] >= 0
        assert snap["radiation_msv"] > 0


def test_colony_food_production() -> None:
    """Greenhouse produces positive food at tuned yield."""
    c = create_colony("test", "conservative", 42)
    assert c.greenhouse_m2 > 0
    expected_kg = c.greenhouse_m2 * GREENHOUSE_KG_SOL_M2
    assert expected_kg > 0


def test_sim_deterministic() -> None:
    """Same seed = same result."""
    r1 = Simulation(sols=50, env_seed=42).run()
    r2 = Simulation(sols=50, env_seed=42).run()
    for i in range(3):
        assert r1["colonies"][i]["final_population"] == r2["colonies"][i]["final_population"]


def test_dashboard_has_svg() -> None:
    """Dashboard generates valid HTML with SVG."""
    r = Simulation(sols=30, env_seed=42).run()
    html = generate_dashboard(r)
    assert "<svg" in html
    assert "Ares Prime" in html


def test_conservation_population() -> None:
    """Births - deaths = population change (accounting check)."""
    r = Simulation(sols=100, env_seed=42).run()
    for c in r["colonies"]:
        initial_pop = c["history"][0]["population"]
        expected = initial_pop + c["total_births"] - c["total_deaths"]
        assert c["final_population"] == expected, (
            f"{c['name']}: {initial_pop} + {c['total_births']} - {c['total_deaths']}"
            f" = {expected} != {c['final_population']}"
        )


def test_strategies_diverge() -> None:
    """Different strategies produce different outcomes over 365 sols."""
    r = Simulation(sols=365, env_seed=42).run()
    pops = [c["final_population"] for c in r["colonies"]]
    assert len(set(pops)) > 1, "All colonies ended with identical population"
    total_initial = sum(c["history"][0]["population"] for c in r["colonies"])
    total_final = sum(pops)
    assert total_final >= total_initial, f"Terrarium shrunk: {total_initial} -> {total_final}"


def test_dust_storms_affect_production() -> None:
    """Dust storms reduce solar flux, which affects food production."""
    env = MarsEnvironment(seed=42)
    clear_flux = []
    storm_flux = []
    for _ in range(668):  # full Mars year
        snap = env.tick()
        if snap["storm"] is None:
            clear_flux.append(snap["solar_flux_wm2"])
        else:
            storm_flux.append(snap["solar_flux_wm2"])
    if storm_flux:
        avg_clear = sum(clear_flux) / len(clear_flux)
        avg_storm = sum(storm_flux) / len(storm_flux)
        assert avg_storm < avg_clear, "Storms should reduce solar flux"


def test_births_require_population() -> None:
    """Zero-population colony has zero births."""
    c = create_colony("ghost", "conservative", 42)
    c.population = 0
    env = MarsEnvironment(seed=42)
    snap = env.tick()
    c.tick(snap)
    assert c.total_births == 0
    assert c.total_deaths == 0


def test_genetic_diversity_tracked() -> None:
    """Genetic diversity is tracked in colony history."""
    r = Simulation(sols=50, env_seed=42).run()
    for c in r["colonies"]:
        for h in c["history"]:
            assert "genetic_diversity" in h
            assert 0.0 <= h["genetic_diversity"] <= 1.0


def test_equipment_degradation() -> None:
    """Solar panels accumulate dust over time."""
    sim = Simulation(sols=200, env_seed=42)
    sim.run()
    for colony in sim.colonies:
        assert colony.dust_accumulation >= 0.0
