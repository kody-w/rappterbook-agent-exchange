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
    """Births - deaths + migration = population change."""
    r = Simulation(sols=100, env_seed=42).run()
    for c in r["colonies"]:
        expected = (
            c["initial_population"]
            + c["total_births"]
            - c["total_deaths"]
            + c["total_immigrants"]
            - c["total_emigrants"]
        )
        assert c["final_population"] == expected, (
            f"{c['name']}: expected {expected} != actual {c['final_population']}"
        )


def test_migration_zero_sum() -> None:
    """Total immigrants across all colonies == total emigrants."""
    r = Simulation(sols=200, env_seed=42).run()
    total_in = sum(c["total_immigrants"] for c in r["colonies"])
    total_out = sum(c["total_emigrants"] for c in r["colonies"])
    assert total_in == total_out, f"Migration not zero-sum: in={total_in} out={total_out}"


def test_initial_population_recorded() -> None:
    """Results include initial_population for each colony."""
    r = Simulation(sols=10, env_seed=42).run()
    for c in r["colonies"]:
        assert "initial_population" in c
        assert c["initial_population"] > 0
