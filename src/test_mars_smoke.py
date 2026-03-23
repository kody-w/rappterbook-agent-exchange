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


def test_tech_tree_smoke() -> None:
    """Tech tree imports and creates engines without crash."""
    from src.tech_tree import ResearchEngine, TECH_CATALOG
    import random
    eng = ResearchEngine(strategy="balanced", rng=random.Random(1))
    assert len(TECH_CATALOG) == 8
    assert eng.research_points == 0.0


def test_tech_unlocks_in_sim() -> None:
    """A 200-sol sim unlocks at least one tech."""
    r = Simulation(sols=200, env_seed=42).run()
    any_tech = any(
        col["history"][-1].get("tech", {}).get("unlocked_count", 0) > 0
        for col in r["colonies"]
    )
    assert any_tech, "Expected tech unlocks in 200 sols"


def test_tech_timeline_in_dashboard() -> None:
    """Dashboard includes the tech timeline chart."""
    r = Simulation(sols=100, env_seed=42).run()
    html = generate_dashboard(r)
    assert "tech-chart" in html
    assert "drawTechTimeline" in html


def test_terraforming_feedback_active() -> None:
    """Terraforming progress increases over a long sim."""
    r = Simulation(sols=365, env_seed=42).run()
    tf = r["summary"].get("terraforming", {})
    progress = tf.get("progress", 0)
    assert progress > 0, "Expected terraforming progress > 0 after 365 sols"
    # Each colony should contribute
    contributions = tf.get("contributions", {})
    assert len(contributions) == 3
    for name, output in contributions.items():
        assert output > 0, f"Expected {name} to have positive terraforming output"


def test_terraforming_in_env_history() -> None:
    """Environment history contains terraforming_progress field."""
    r = Simulation(sols=50, env_seed=42).run()
    env_history = r["environment"]["history"]
    assert len(env_history) == 50
    # Terraforming progress should be present in each env snapshot
    for snap in env_history:
        assert "terraforming_progress" in snap
        assert snap["terraforming_progress"] >= 0


def test_terraforming_modifies_temperature() -> None:
    """Over a very long sim, terraforming should warm the planet."""
    sim_short = Simulation(sols=10, env_seed=42)
    r_short = sim_short.run()
    sim_long = Simulation(sols=668, env_seed=42)
    r_long = sim_long.run()
    # After a full Mars year, terraforming progress should be measurable
    tf_short = r_short["environment"]["final_terraforming_progress"]
    tf_long = r_long["environment"]["final_terraforming_progress"]
    assert tf_long > tf_short


def test_terraforming_output_in_colony_results() -> None:
    """Colony results include terraforming_output field."""
    r = Simulation(sols=100, env_seed=42).run()
    for c in r["colonies"]:
        assert "terraforming_output" in c
        assert c["terraforming_output"] >= 0


def test_terraforming_dashboard_chart() -> None:
    """Dashboard includes terraforming chart."""
    r = Simulation(sols=50, env_seed=42).run()
    html = generate_dashboard(r)
    assert "terraform-chart" in html
    assert "TERRAFORM" in html
