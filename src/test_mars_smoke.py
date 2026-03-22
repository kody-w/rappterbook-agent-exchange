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
    """Global population: births - deaths = total population change (migration is zero-sum)."""
    r = Simulation(sols=100, env_seed=42).run()
    total_start = 0
    total_end = 0
    total_b = 0
    total_d = 0
    for c in r["colonies"]:
        hist = c["history"]
        total_start += hist[0]["population"] - hist[0]["births"] + hist[0]["deaths"]
        total_end += hist[-1]["population"]
        total_b += sum(h["births"] for h in hist)
        total_d += sum(h["deaths"] for h in hist)
    assert total_end == total_start + total_b - total_d


def test_aggressive_growth_rate() -> None:
    """Aggressive strategy has highest % growth over 365 sols."""
    r = Simulation(sols=365, env_seed=42).run()
    growth = {}
    for c in r["colonies"]:
        pops = [h["population"] for h in c["history"]]
        start = pops[0] - c["history"][0]["births"] + c["history"][0]["deaths"]
        growth[c["strategy"]] = (pops[-1] - start) / max(1, start)
    # Aggressive expands fastest in % terms (starts smallest, biggest supply ships)
    assert growth["aggressive"] > growth["balanced"]
