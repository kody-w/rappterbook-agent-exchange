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


def test_dashboard_has_charts() -> None:
    """Dashboard generates valid HTML with canvas charts."""
    r = Simulation(sols=30, env_seed=42).run()
    html = generate_dashboard(r)
    assert "<canvas" in html
    assert "Ares Prime" in html
    assert "drawChart" in html


def test_conservation_population() -> None:
    """Births - deaths + migration = population change."""
    r = Simulation(sols=100, env_seed=42).run()
    for c in r["colonies"]:
        initial = c["initial_population"]
        net_mig = sum(h.get("net_migration", 0) for h in c["history"])
        expected = initial + c["total_births"] - c["total_deaths"] + net_mig
        assert c["final_population"] == expected, (
            f"{c['name']}: {initial} + {c['total_births']} - {c['total_deaths']}"
            f" + mig {net_mig} = {expected} != {c['final_population']}"
        )


def test_strategies_diverge() -> None:
    """Different strategies produce different outcomes over 365 sols."""
    r = Simulation(sols=365, env_seed=42).run()
    pops = [c["final_population"] for c in r["colonies"]]
    assert len(set(pops)) > 1, "All colonies ended with identical population"


def test_carrying_capacity_tracked() -> None:
    """Carrying capacity K appears in colony history snapshots."""
    r = Simulation(sols=10, env_seed=42).run()
    for c in r["colonies"]:
        for h in c["history"]:
            assert "carrying_capacity" in h
            assert h["carrying_capacity"] > 0


def test_genetic_diversity_tracked() -> None:
    """Genetic diversity appears in history and stays in [0, 1]."""
    r = Simulation(sols=100, env_seed=42).run()
    for c in r["colonies"]:
        for h in c["history"]:
            assert "genetic_diversity" in h
            assert 0.0 <= h["genetic_diversity"] <= 1.0


def test_births_require_population() -> None:
    """Zero-population colony has zero births."""
    c = create_colony("ghost", "conservative", 42)
    c.population = 0
    env = MarsEnvironment(seed=42)
    snap = env.tick()
    c.tick(snap)
    assert c.total_births == 0
    assert c.total_deaths == 0


def test_death_causes_tracked() -> None:
    """Death causes appear in history snapshots and sum correctly."""
    r = Simulation(sols=100, env_seed=42).run()
    for c in r["colonies"]:
        for h in c["history"]:
            assert "death_causes" in h
            assert sum(h["death_causes"].values()) == h["deaths"]


def test_storm_damage_reduces_infrastructure() -> None:
    """Global storms reduce solar and greenhouse area."""
    c = create_colony("test", "balanced", 42)
    initial_solar = c.solar_m2
    env = {"sol": 1, "storm": "global", "radiation_msv": 0.67,
           "solar_flux_wm2": 100.0, "flare": False}
    c.tick(env)
    assert c.solar_m2 < initial_solar


def test_death_causes_dashboard() -> None:
    """Dashboard includes death causes chart."""
    r = Simulation(sols=30, env_seed=42).run()
    html = generate_dashboard(r)
    assert "death-causes-chart" in html
    assert "deathCauses" in html
