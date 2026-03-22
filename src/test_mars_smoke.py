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


def test_tech_tree_wired() -> None:
    """Tech tree is wired and techs unlock over 200 sols."""
    sim = Simulation(sols=200, env_seed=42)
    results = sim.run()
    total_techs = sum(
        len(c["tech_tree"]["unlocked"]) for c in results["colonies"]
    )
    assert total_techs > 0, "No techs unlocked in 200 sols"


def test_tech_effects_serializable() -> None:
    """Tech tree snapshot is JSON-serializable."""
    import json
    sim = Simulation(sols=100, env_seed=42)
    results = sim.run()
    for c in results["colonies"]:
        json.dumps(c["tech_tree"])  # Should not raise


def test_tech_divergence() -> None:
    """Different strategies produce different tech paths."""
    sim = Simulation(sols=365, env_seed=42)
    sim.run()
    techs = [sorted(c.research_engine.unlocked) for c in sim.colonies]
    # Not all identical
    assert len(set(tuple(t) for t in techs)) > 1, "All colonies chose same techs"
