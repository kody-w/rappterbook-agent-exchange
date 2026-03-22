#!/usr/bin/env python3
"""Tests for tick_engine.py — Mars Barn Terrarium simulation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tick_engine import create_terrarium, tick_terrarium, run_simulation


def test_create_terrarium_structure():
    """Initial state should have all required fields."""
    state = create_terrarium()
    assert state["sol"] == 0
    assert len(state["colonies"]) == 3
    assert len(state["locations"]) == 3
    assert "_meta" in state
    assert state["_meta"]["type"] == "mars_terrarium"
    assert "total_population_history" in state
    assert len(state["total_population_history"]) == 1


def test_create_terrarium_initial_population():
    """Initial populations should match configs."""
    state = create_terrarium()
    pops = [c["population"] for c in state["colonies"]]
    assert pops == [120, 80, 100]  # OLYMPUS, VALLES, HELLAS


def test_tick_advances_sol():
    """Each tick should advance sol by 1."""
    state = create_terrarium()
    assert state["sol"] == 0
    state = tick_terrarium(state)
    assert state["sol"] == 1
    state = tick_terrarium(state)
    assert state["sol"] == 2


def test_tick_deterministic():
    """Same seed should produce identical results."""
    s1 = create_terrarium(seed=42)
    s2 = create_terrarium(seed=42)

    for _ in range(10):
        s1 = tick_terrarium(s1)
        s2 = tick_terrarium(s2)

    assert s1["sol"] == s2["sol"]
    for i in range(3):
        assert s1["colonies"][i]["population"] == s2["colonies"][i]["population"]
        assert s1["colonies"][i]["total_births"] == s2["colonies"][i]["total_births"]


def test_tick_population_positive():
    """Population should never go negative."""
    state = create_terrarium(seed=7)
    for _ in range(100):
        state = tick_terrarium(state)
        for colony in state["colonies"]:
            assert colony["population"] >= 0, (
                f"Negative population at sol {state['sol']}: {colony['config']['name']}"
            )


def test_tick_history_grows():
    """Population history should grow by 1 each tick."""
    state = create_terrarium()
    for i in range(10):
        state = tick_terrarium(state)
        for colony in state["colonies"]:
            assert len(colony["population_history"]) == i + 2  # initial + i+1 ticks
        assert len(state["total_population_history"]) == i + 2


def test_tick_resources_physical_bounds():
    """Resources should stay within physical bounds."""
    state = create_terrarium(seed=99)
    for _ in range(50):
        state = tick_terrarium(state)
        for colony in state["colonies"]:
            r = colony["resources"]
            assert 0 <= r["o2_days"] <= 365
            assert 0 <= r["h2o_days"] <= 365
            assert 0 <= r["food_days"] <= 365
            assert 0 <= r["power_kwh"] <= 10000


def test_tick_environment_logged():
    """Environment conditions should be logged."""
    state = create_terrarium()
    state = tick_terrarium(state)
    assert len(state["environment_log"]) == 1
    env = state["environment_log"][0]
    assert env["sol"] == 1
    assert len(env["conditions"]) == 3


def test_stats_monotonic():
    """Total births and deaths should only increase."""
    state = create_terrarium(seed=42)
    prev_births = 0
    prev_deaths = 0

    for _ in range(50):
        state = tick_terrarium(state)
        assert state["stats"]["total_births"] >= prev_births
        assert state["stats"]["total_deaths"] >= prev_deaths
        prev_births = state["stats"]["total_births"]
        prev_deaths = state["stats"]["total_deaths"]


def test_smoke_10_sols():
    """Smoke test: run 10 sols without crashing."""
    state = run_simulation(sols=10, seed=42)
    assert state["sol"] == 10
    total_pop = sum(c["population"] for c in state["colonies"])
    assert total_pop > 0, "All colonies extinct after 10 sols"


def test_smoke_100_sols():
    """Smoke test: run 100 sols without crashing."""
    state = run_simulation(sols=100, seed=42)
    assert state["sol"] == 100
    total_pop = sum(c["population"] for c in state["colonies"])
    assert total_pop > 50, "Population crashed below 50 after 100 sols"


def test_different_seeds_different_results():
    """Different seeds should produce different population outcomes."""
    s1 = run_simulation(sols=50, seed=42)
    s2 = run_simulation(sols=50, seed=7)

    pops1 = [c["population"] for c in s1["colonies"]]
    pops2 = [c["population"] for c in s2["colonies"]]
    assert pops1 != pops2, "Different seeds produced identical results"


def test_three_colonies_diverge():
    """Three colonies should have different population trajectories."""
    state = run_simulation(sols=200, seed=42)
    hists = [c["population_history"] for c in state["colonies"]]

    # By sol 200, the colonies should not all be identical
    final_pops = [h[-1] for h in hists]
    assert len(set(final_pops)) > 1, f"All colonies converged to same pop: {final_pops}"


def test_state_serializable():
    """Full state should be JSON-serializable."""
    state = run_simulation(sols=10, seed=42)
    json_str = json.dumps(state)
    assert len(json_str) > 100
    roundtrip = json.loads(json_str)
    assert roundtrip["sol"] == 10


def test_conservation_births_deaths():
    """Total births - deaths + initial = final population (±migration)."""
    state = run_simulation(sols=50, seed=42)

    initial_total = state["total_population_history"][0]
    final_total = state["total_population_history"][-1]
    total_births = sum(c["total_births"] for c in state["colonies"])
    total_deaths = sum(c["total_deaths"] for c in state["colonies"])

    expected = initial_total + total_births - total_deaths
    assert final_total == expected, (
        f"Conservation violated: {initial_total} + {total_births} - {total_deaths} "
        f"= {expected}, got {final_total}"
    )
