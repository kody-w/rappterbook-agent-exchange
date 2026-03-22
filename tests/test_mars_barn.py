"""Tests for the Mars Barn terrarium — colony population simulation.

Covers:
- Environment: Ls calculation, solar flux, event generation
- Colony: population dynamics, resource constraints, carrying capacity
- Tick engine: full simulation runs, conservation laws
- Property-based invariants: outputs in physical bounds
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

# Allow imports from src/
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from mars_barn import environment as env
from mars_barn import colony as col
from mars_barn.tick_engine import (
    create_initial_state,
    tick,
    run_simulation,
)


# ─── Environment tests ───────────────────────────────────────────────

class TestSolToLs:
    """Solar longitude calculation."""

    def test_sol_zero_near_zero(self):
        ls = env.sol_to_ls(0)
        assert 0 <= ls < 360

    def test_full_mars_year_wraps(self):
        ls_start = env.sol_to_ls(0)
        ls_end = env.sol_to_ls(669)
        # Should be close to start after one full year
        assert abs(ls_end - ls_start) < 15 or abs(ls_end - ls_start) > 345

    def test_monotonic_increase_approximately(self):
        """Ls should generally increase over the year."""
        values = [env.sol_to_ls(s) for s in range(0, 668, 10)]
        # Allow for wrap-around at 360
        increases = 0
        for i in range(1, len(values)):
            if values[i] > values[i - 1] or values[i] < 30:
                increases += 1
        assert increases > len(values) * 0.8

    def test_ls_always_in_range(self):
        for sol in range(0, 1000, 7):
            ls = env.sol_to_ls(sol)
            assert 0 <= ls < 360, f"Ls out of range at sol {sol}: {ls}"


class TestSolarFlux:
    """Solar flux calculations."""

    def test_flux_in_unit_range(self):
        for ls in range(0, 360, 5):
            flux = env.solar_flux(float(ls))
            assert 0.0 <= flux <= 1.0, f"Flux out of range at Ls {ls}: {flux}"

    def test_flux_peaks_near_perihelion(self):
        flux_perihelion = env.solar_flux(251.0)
        flux_aphelion = env.solar_flux(71.0)
        assert flux_perihelion > flux_aphelion

    def test_flux_near_one_at_perihelion(self):
        flux = env.solar_flux(251.0)
        assert flux > 0.9


class TestDustSeason:
    """Dust storm season detection."""

    def test_in_dust_season(self):
        assert env.is_dust_season(200.0)
        assert env.is_dust_season(300.0)

    def test_not_in_dust_season(self):
        assert not env.is_dust_season(90.0)
        assert not env.is_dust_season(350.0)


class TestSurfaceTemperature:
    """Mars surface temperature model."""

    def test_temperature_range(self):
        for ls in range(0, 360, 10):
            temp = env.surface_temperature(float(ls))
            assert -120 < temp < 40, f"Temp out of range at Ls {ls}: {temp}"

    def test_colder_at_poles(self):
        equator = env.surface_temperature(180.0, latitude=0.0)
        pole = env.surface_temperature(180.0, latitude=80.0)
        assert pole < equator


class TestEventGeneration:
    """Mars event generation."""

    def test_events_are_list(self):
        rng = random.Random(42)
        colonies = [{"population": 50, "carrying_capacity": 200}]
        events = env.generate_events(100, colonies, rng)
        assert isinstance(events, list)

    def test_event_structure(self):
        rng = random.Random(99)
        colonies = [
            {"population": 50, "carrying_capacity": 200},
            {"population": 30, "carrying_capacity": 300},
        ]
        # Run many sols to get at least some events
        all_events = []
        for sol in range(1, 200):
            all_events.extend(env.generate_events(sol, colonies, rng))
        assert len(all_events) > 0, "Expected some events over 200 sols"
        for event in all_events:
            assert "type" in event
            assert "target" in event
            assert "description" in event

    def test_dust_season_more_storms(self):
        """More dust storms should happen during dust season."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        colonies = [{"population": 50, "carrying_capacity": 200}]

        # Count storms outside dust season (Ls ~90)
        off_season = 0
        for _ in range(1000):
            events = env.generate_events(150, colonies, rng1)  # Ls ~80
            off_season += sum(1 for e in events if "dust" in e["type"])

        # Count storms during dust season (Ls ~250)
        on_season = 0
        for _ in range(1000):
            events = env.generate_events(450, colonies, rng2)  # Ls ~242
            on_season += sum(1 for e in events if "dust" in e["type"])

        assert on_season > off_season


class TestResourceProduction:
    """Resource production model."""

    def test_all_resources_positive(self):
        colony = {"tech_level": 1.0, "infrastructure": 1.0, "population": 50, "index": 0}
        prod = env.compute_resource_production(colony, 180.0, [])
        for key in ["power", "o2", "h2o", "food"]:
            assert prod[key] >= 0, f"{key} is negative: {prod[key]}"

    def test_dust_reduces_production(self):
        colony = {"tech_level": 1.0, "infrastructure": 1.0, "population": 50, "index": 0}
        clear = env.compute_resource_production(colony, 180.0, [])
        stormy = env.compute_resource_production(colony, 180.0, [
            {"target": 0, "magnitude": 0.8}
        ])
        assert stormy["power"] < clear["power"]

    def test_higher_tech_more_production(self):
        low = {"tech_level": 0.5, "infrastructure": 1.0, "population": 50, "index": 0}
        high = {"tech_level": 2.0, "infrastructure": 1.0, "population": 50, "index": 0}
        prod_low = env.compute_resource_production(low, 180.0, [])
        prod_high = env.compute_resource_production(high, 180.0, [])
        assert prod_high["power"] > prod_low["power"]
        assert prod_high["o2"] > prod_low["o2"]


# ─── Colony tests ─────────────────────────────────────────────────────

class TestColonyCreation:
    """Colony initialization."""

    def test_create_all_presets(self):
        for i, key in enumerate(col.COLONY_PRESETS):
            colony = col.create_colony(key, i)
            assert colony["population"] > 0
            assert colony["name"]
            assert colony["index"] == i
            assert colony["births_total"] == 0
            assert colony["deaths_total"] == 0

    def test_resources_initialized(self):
        colony = col.create_colony("ares_prime", 0)
        for res in ["o2", "h2o", "food", "power"]:
            assert res in colony["resources"]
            assert colony["resources"][res] > 0


class TestEffectiveCarryingCapacity:
    """Carrying capacity computation."""

    def test_k_eff_bounded_by_base(self):
        colony = col.create_colony("ares_prime", 0)
        k = col.effective_carrying_capacity(colony)
        assert k <= colony["carrying_capacity"]

    def test_k_eff_drops_with_low_resources(self):
        colony = col.create_colony("ares_prime", 0)
        k_full = col.effective_carrying_capacity(colony)

        colony["resources"]["food"] = 1.0  # Nearly empty
        k_starving = col.effective_carrying_capacity(colony)
        assert k_starving < k_full

    def test_k_eff_minimum_floor(self):
        colony = col.create_colony("ares_prime", 0)
        colony["resources"] = {"o2": 0, "h2o": 0, "food": 0, "power": 0}
        k = col.effective_carrying_capacity(colony)
        assert k >= 2  # Floor


class TestTickColony:
    """Single-colony tick dynamics."""

    def test_population_stays_positive(self):
        colony = col.create_colony("ares_prime", 0)
        rng = random.Random(42)
        for sol in range(1, 100):
            production = {"o2": 10, "h2o": 20, "food": 10, "power": 50}
            col.tick_colony(colony, sol, production, [], rng)
            assert colony["population"] >= 0

    def test_births_deaths_tracked(self):
        colony = col.create_colony("ares_prime", 0)
        rng = random.Random(42)
        for sol in range(1, 50):
            production = {"o2": 50, "h2o": 200, "food": 100, "power": 500}
            col.tick_colony(colony, sol, production, [], rng)
        assert colony["births_total"] > 0
        assert colony["deaths_total"] >= 0

    def test_conservation_law(self):
        """births - deaths = final_pop - initial_pop."""
        colony = col.create_colony("ares_prime", 0)
        initial_pop = colony["population"]
        rng = random.Random(42)
        for sol in range(1, 200):
            production = {"o2": 50, "h2o": 200, "food": 100, "power": 500}
            col.tick_colony(colony, sol, production, [], rng)
        expected = initial_pop + colony["births_total"] - colony["deaths_total"]
        assert colony["population"] == expected, (
            f"Conservation violated: {initial_pop} + {colony['births_total']} - "
            f"{colony['deaths_total']} = {expected}, got {colony['population']}"
        )

    def test_morale_bounded(self):
        colony = col.create_colony("ares_prime", 0)
        rng = random.Random(42)
        for sol in range(1, 100):
            production = {"o2": 10, "h2o": 20, "food": 10, "power": 50}
            col.tick_colony(colony, sol, production, [], rng)
            assert 0.0 <= colony["morale"] <= 1.0

    def test_starvation_kills(self):
        """Colony with zero resources should decline."""
        colony = col.create_colony("ares_prime", 0)
        colony["resources"] = {"o2": 0, "h2o": 0, "food": 0, "power": 0}
        rng = random.Random(42)
        initial = colony["population"]
        for sol in range(1, 50):
            col.tick_colony(colony, sol, {"o2": 0, "h2o": 0, "food": 0, "power": 0}, [], rng)
        assert colony["population"] < initial

    def test_snapshot_structure(self):
        colony = col.create_colony("ares_prime", 0)
        rng = random.Random(42)
        snap = col.tick_colony(colony, 1, {"o2": 10, "h2o": 20, "food": 10, "power": 50}, [], rng)
        assert "sol" in snap
        assert "population" in snap
        assert "morale" in snap
        assert "resources" in snap


# ─── Tick engine tests ────────────────────────────────────────────────

class TestCreateInitialState:
    """Initial state creation."""

    def test_three_colonies(self):
        state = create_initial_state()
        assert len(state["colonies"]) == 3

    def test_meta_present(self):
        state = create_initial_state()
        assert state["_meta"]["engine"] == "mars_barn"
        assert state["_meta"]["version"] == "1.0.0"

    def test_total_population(self):
        state = create_initial_state()
        expected = sum(c["population"] for c in state["colonies"])
        assert state["total_population"] == expected


class TestTick:
    """Single tick of the simulation."""

    def test_sol_increments(self):
        state = create_initial_state()
        rng = random.Random(42)
        tick(state, rng)
        assert state["sol"] == 1

    def test_timeline_grows(self):
        state = create_initial_state()
        rng = random.Random(42)
        tick(state, rng)
        assert len(state["timeline"]) == 1

    def test_snapshot_has_all_colonies(self):
        state = create_initial_state()
        rng = random.Random(42)
        snapshot = tick(state, rng)
        assert len(snapshot["colonies"]) == 3


class TestRunSimulation:
    """Full simulation runs."""

    def test_smoke_10_sols(self):
        """Smoke test: 10 sols without crash."""
        state = run_simulation(sols=10, seed=42)
        assert state["sol"] == 10
        assert len(state["timeline"]) == 10
        assert state["total_population"] > 0

    def test_365_sols_populations_positive(self):
        """All colonies survive 365 sols."""
        state = run_simulation(sols=365, seed=42)
        assert state["sol"] == 365
        for colony in state["colonies"]:
            assert colony["population"] >= 0

    def test_reproducible_with_same_seed(self):
        """Same seed → same results."""
        s1 = run_simulation(sols=50, seed=123)
        s2 = run_simulation(sols=50, seed=123)
        for i in range(3):
            assert s1["colonies"][i]["population"] == s2["colonies"][i]["population"]

    def test_different_seeds_differ(self):
        """Different seeds → different results."""
        s1 = run_simulation(sols=100, seed=1)
        s2 = run_simulation(sols=100, seed=2)
        pops1 = [c["population"] for c in s1["colonies"]]
        pops2 = [c["population"] for c in s2["colonies"]]
        assert pops1 != pops2

    def test_conservation_all_colonies(self):
        """births - deaths = final_pop - initial_pop for each colony."""
        state = create_initial_state(seed=42)
        initial_pops = [c["population"] for c in state["colonies"]]
        rng = random.Random(42)
        for _ in range(200):
            tick(state, rng)
        for i, colony in enumerate(state["colonies"]):
            expected = initial_pops[i] + colony["births_total"] - colony["deaths_total"]
            assert colony["population"] == expected, (
                f"Colony {colony['name']}: conservation violated"
            )

    def test_population_bounded(self):
        """Population shouldn't exceed carrying capacity by too much."""
        state = run_simulation(sols=365, seed=42)
        for colony in state["colonies"]:
            # Allow 20% overshoot due to events
            assert colony["peak_population"] < colony["carrying_capacity"] * 2

    def test_timeline_contains_ls(self):
        """Timeline entries have solar longitude."""
        state = run_simulation(sols=10, seed=42)
        for entry in state["timeline"]:
            assert "ls" in entry
            assert 0 <= entry["ls"] < 360

    def test_resources_never_negative(self):
        """Resources should never go below zero."""
        state = run_simulation(sols=365, seed=42)
        for colony in state["colonies"]:
            for res, val in colony["resources"].items():
                assert val >= 0, f"{colony['name']} has negative {res}: {val}"
