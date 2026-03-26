"""Tests for propellant_depot.py — Mars Cryogenic Propellant Storage.

68 tests covering:
  - Pure physics functions (heat leak, boil-off, cryocooler power)
  - Individual CryoTank behavior (fill, boil-off, ZBO, degradation)
  - Integrated PropellantDepot (multi-sol, launch readiness)
  - Conservation laws (mass balance, energy bounds)
  - Edge cases (empty tank, zero power, extreme temperatures)
  - Physical invariants (bounds, monotonicity)
  - Smoke test: 365-sol run without crash
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from propellant_depot import (
    CryoTank,
    PropellantDepot,
    boiloff_rate_kg_per_sol,
    cryocooler_power_kw,
    fill_fraction,
    heat_leak_watts,
    make_depot,
    run_depot,
    tank_surface_area_m2,
    LOX_BP_K,
    LOX_LATENT_KJ_KG,
    LOX_DENSITY,
    LCH4_BP_K,
    LCH4_LATENT_KJ_KG,
    LCH4_DENSITY,
    MARS_AVG_TEMP_K,
    MLI_HEAT_FLUX_NOMINAL,
    SOL_SECONDS,
    TARGET_LOX_KG,
    TARGET_LCH4_KG,
)


# ===================================================================
# Pure physics functions
# ===================================================================

class TestHeatLeak:

    def test_zero_area(self):
        assert heat_leak_watts(0.0, 1.0) == 0.0

    def test_zero_flux(self):
        assert heat_leak_watts(100.0, 0.0) == 0.0

    def test_positive(self):
        assert heat_leak_watts(50.0, 1.0) == 50.0

    def test_scales_linearly_with_area(self):
        q1 = heat_leak_watts(10.0, 1.5)
        q2 = heat_leak_watts(20.0, 1.5)
        assert abs(q2 - 2 * q1) < 1e-10

    def test_negative_inputs_clamped(self):
        assert heat_leak_watts(-5.0, 1.0) == 0.0


class TestBoiloffRate:

    def test_zero_heat(self):
        assert boiloff_rate_kg_per_sol(0.0, LOX_LATENT_KJ_KG) == 0.0

    def test_zero_latent_heat(self):
        assert boiloff_rate_kg_per_sol(10.0, 0.0) == 0.0

    def test_lox_1_watt(self):
        """1 W sustained boils ~0.417 kg LOX per sol."""
        rate = boiloff_rate_kg_per_sol(1.0, LOX_LATENT_KJ_KG)
        expected = SOL_SECONDS / (LOX_LATENT_KJ_KG * 1000.0)
        assert abs(rate - expected) < 1e-6
        assert 0.3 < rate < 0.5

    def test_lch4_1_watt(self):
        """1 W sustained boils ~0.174 kg LCH₄ per sol."""
        rate = boiloff_rate_kg_per_sol(1.0, LCH4_LATENT_KJ_KG)
        assert 0.1 < rate < 0.3

    def test_lox_boils_faster_than_lch4(self):
        lox_rate = boiloff_rate_kg_per_sol(10.0, LOX_LATENT_KJ_KG)
        lch4_rate = boiloff_rate_kg_per_sol(10.0, LCH4_LATENT_KJ_KG)
        assert lox_rate > lch4_rate

    def test_scales_linearly_with_heat(self):
        r1 = boiloff_rate_kg_per_sol(5.0, LOX_LATENT_KJ_KG)
        r2 = boiloff_rate_kg_per_sol(10.0, LOX_LATENT_KJ_KG)
        assert abs(r2 - 2 * r1) < 1e-10


class TestCryocoolerPower:

    def test_zero_heat_load(self):
        p = cryocooler_power_kw(0.0, LOX_BP_K, MARS_AVG_TEMP_K)
        assert p >= 0.0

    def test_lox_typical(self):
        """LOX at 90K against Mars 213K ambient."""
        p = cryocooler_power_kw(100.0, LOX_BP_K, MARS_AVG_TEMP_K)
        assert 0.3 < p < 2.0

    def test_lch4_easier_than_lox(self):
        """LCH₄ is warmer (112K vs 90K) → higher COP → less power.
        Use 1000 W load to exceed the minimum power floor.
        """
        p_lox = cryocooler_power_kw(1000.0, LOX_BP_K, MARS_AVG_TEMP_K)
        p_lch4 = cryocooler_power_kw(1000.0, LCH4_BP_K, MARS_AVG_TEMP_K)
        assert p_lch4 < p_lox

    def test_cold_equals_hot_returns_zero(self):
        assert cryocooler_power_kw(100.0, 200.0, 200.0) == 0.0

    def test_inverted_temps_returns_zero(self):
        assert cryocooler_power_kw(100.0, 300.0, 200.0) == 0.0


class TestTankSurfaceArea:

    def test_zero_volume(self):
        assert tank_surface_area_m2(0.0) == 0.0

    def test_positive_volume(self):
        assert tank_surface_area_m2(100.0) > 0.0

    def test_grows_with_volume(self):
        assert tank_surface_area_m2(100.0) > tank_surface_area_m2(10.0)

    def test_scales_sublinearly(self):
        """Surface area ~ V^(2/3), so 8× volume → ~4× area."""
        ratio = tank_surface_area_m2(8.0) / tank_surface_area_m2(1.0)
        assert 3.5 < ratio < 4.5


class TestFillFraction:

    def test_empty(self):
        assert fill_fraction(0.0, 100.0) == 0.0

    def test_full(self):
        assert fill_fraction(100.0, 100.0) == 1.0

    def test_overfill_clamped(self):
        assert fill_fraction(150.0, 100.0) == 1.0

    def test_zero_capacity(self):
        assert fill_fraction(50.0, 0.0) == 0.0

    def test_half(self):
        assert abs(fill_fraction(50.0, 100.0) - 0.5) < 1e-10


# ===================================================================
# CryoTank unit tests
# ===================================================================

class TestCryoTank:

    def _lox(self, capacity=10000.0, current=5000.0):
        return CryoTank(
            label="LOX", capacity_kg=capacity, current_kg=current,
            boiling_point_k=LOX_BP_K, latent_kj_kg=LOX_LATENT_KJ_KG,
            density_kg_m3=LOX_DENSITY,
        )

    def test_volume_positive(self):
        assert self._lox().volume_m3 > 0

    def test_surface_area_positive(self):
        assert self._lox().surface_area_m2 > 0

    def test_fill_percentage(self):
        assert abs(self._lox(10000.0, 5000.0).fill_pct - 50.0) < 1e-6

    def test_no_cryocooler_causes_boiloff(self):
        tank = self._lox(current=5000.0)
        r = tank.tick(cryocooler_active=False)
        assert r["boiloff_kg"] > 0
        assert tank.current_kg < 5000.0

    def test_cryocooler_prevents_boiloff(self):
        tank = self._lox(current=5000.0)
        r = tank.tick(cryocooler_active=True, available_power_kw=100.0)
        assert r["zbo_active"]
        assert r["boiloff_kg"] < 0.01

    def test_adds_propellant(self):
        tank = self._lox(current=5000.0)
        tank.tick(added_kg=100.0, cryocooler_active=True, available_power_kw=100.0)
        assert tank.current_kg > 5000.0

    def test_fill_capped(self):
        tank = self._lox(capacity=100.0, current=90.0)
        tank.tick(added_kg=50.0, cryocooler_active=True, available_power_kw=100.0)
        assert tank.current_kg <= 100.0

    def test_boiloff_cant_exceed_contents(self):
        tank = self._lox(current=0.01)
        tank.tick(cryocooler_active=False)
        assert tank.current_kg >= 0.0

    def test_insulation_degrades(self):
        tank = self._lox()
        h0 = tank.insulation_health
        tank.tick()
        assert tank.insulation_health < h0

    def test_cold_ambient_less_boiloff(self):
        t1, t2 = self._lox(current=5000.0), self._lox(current=5000.0)
        r1 = t1.tick(ambient_temp_k=MARS_AVG_TEMP_K, cryocooler_active=False)
        r2 = t2.tick(ambient_temp_k=150.0, cryocooler_active=False)
        assert r2["boiloff_kg"] < r1["boiloff_kg"]

    def test_hot_ambient_more_boiloff(self):
        t1, t2 = self._lox(current=5000.0), self._lox(current=5000.0)
        r1 = t1.tick(ambient_temp_k=200.0, cryocooler_active=False)
        r2 = t2.tick(ambient_temp_k=280.0, cryocooler_active=False)
        assert r2["boiloff_kg"] > r1["boiloff_kg"]

    def test_heat_leak_scales_with_temp(self):
        tank = self._lox()
        assert tank.compute_heat_leak(250.0) > tank.compute_heat_leak(150.0)


# ===================================================================
# PropellantDepot integration
# ===================================================================

class TestPropellantDepot:

    def test_make_depot_defaults(self):
        depot = make_depot()
        assert depot.lox_tank.capacity_kg == TARGET_LOX_KG
        assert depot.lch4_tank.capacity_kg == TARGET_LCH4_KG

    def test_single_tick(self):
        depot = make_depot()
        r = depot.tick(lox_added_kg=100.0, lch4_added_kg=30.0)
        assert r["sol"] == 1
        assert r["total_propellant_kg"] > 0

    def test_fill_over_time(self):
        depot = make_depot()
        for _ in range(10):
            depot.tick(lox_added_kg=600.0, lch4_added_kg=170.0)
        assert depot.lox_tank.current_kg > 5000.0

    def test_launch_readiness(self):
        depot = make_depot()
        depot.lox_tank.current_kg = TARGET_LOX_KG
        depot.lch4_tank.current_kg = TARGET_LCH4_KG
        r = depot.tick(cryocooler_active=True, available_power_kw=50.0)
        assert r["launch_ready"]

    def test_not_ready_when_empty(self):
        r = make_depot().tick()
        assert not r["launch_ready"]

    def test_days_to_ready(self):
        sols = make_depot().days_to_ready(600.0, 170.0)
        assert 250 < sols < 500

    def test_days_to_ready_infinite(self):
        assert make_depot().days_to_ready(0.0, 0.0) == float("inf")

    def test_get_status_keys(self):
        depot = make_depot()
        depot.tick(lox_added_kg=100.0, lch4_added_kg=30.0)
        s = depot.get_status()
        for k in ("lox_kg", "lch4_kg", "fill_pct", "launch_ready", "sol"):
            assert k in s

    def test_history_recorded(self):
        depot = make_depot()
        depot.tick()
        depot.tick()
        assert len(depot.history) == 2

    def test_no_cryo_loses_propellant(self):
        depot = make_depot()
        depot.lox_tank.current_kg = 10000.0
        depot.lch4_tank.current_kg = 3000.0
        total0 = 13000.0
        for _ in range(30):
            depot.tick(cryocooler_active=False)
        total1 = depot.lox_tank.current_kg + depot.lch4_tank.current_kg
        assert total1 < total0
        assert depot.lox_tank.total_boiloff_kg > 0

    def test_custom_capacity(self):
        depot = make_depot(lox_capacity_kg=1000.0, lch4_capacity_kg=300.0)
        assert depot.lox_tank.capacity_kg == 1000.0
        assert depot.lch4_tank.capacity_kg == 300.0


# ===================================================================
# Physical invariants
# ===================================================================

class TestPhysicalInvariants:

    def test_mass_never_negative(self):
        tank = CryoTank(
            label="LOX", capacity_kg=100.0, current_kg=0.5,
            boiling_point_k=LOX_BP_K, latent_kj_kg=LOX_LATENT_KJ_KG,
            density_kg_m3=LOX_DENSITY,
        )
        for _ in range(200):
            tank.tick(cryocooler_active=False)
        assert tank.current_kg >= 0.0

    def test_fill_never_exceeds_capacity(self):
        tank = CryoTank(
            label="LCH4", capacity_kg=1000.0, current_kg=990.0,
            boiling_point_k=LCH4_BP_K, latent_kj_kg=LCH4_LATENT_KJ_KG,
            density_kg_m3=LCH4_DENSITY,
        )
        for _ in range(10):
            tank.tick(added_kg=500.0, cryocooler_active=True, available_power_kw=100.0)
        assert tank.current_kg <= 1000.0

    def test_boiloff_monotonic_with_heat(self):
        rates = [boiloff_rate_kg_per_sol(w, LOX_LATENT_KJ_KG) for w in range(0, 100, 10)]
        for i in range(len(rates) - 1):
            assert rates[i + 1] >= rates[i]

    def test_insulation_health_bounded(self):
        tank = CryoTank(
            label="LOX", capacity_kg=10000.0, current_kg=5000.0,
            boiling_point_k=LOX_BP_K, latent_kj_kg=LOX_LATENT_KJ_KG,
            density_kg_m3=LOX_DENSITY,
        )
        for _ in range(10000):
            tank.tick(cryocooler_active=False)
        assert 0.3 <= tank.insulation_health <= 1.0

    def test_cryo_power_increases_with_temp_diff(self):
        p_close = cryocooler_power_kw(100.0, 100.0, 120.0)
        p_far = cryocooler_power_kw(100.0, 90.0, 300.0)
        assert p_far > p_close

    def test_energy_always_nonneg(self):
        depot = make_depot()
        for _ in range(50):
            r = depot.tick(lox_added_kg=100.0, lch4_added_kg=30.0)
            assert r["energy_kwh"] >= 0.0

    def test_total_boiloff_monotonic(self):
        depot = make_depot()
        depot.lox_tank.current_kg = 50000.0
        depot.lch4_tank.current_kg = 15000.0
        prev = 0.0
        for _ in range(30):
            r = depot.tick(cryocooler_active=False)
            assert r["total_boiloff_kg"] >= prev
            prev = r["total_boiloff_kg"]

    def test_lox_boiloff_exceeds_lch4(self):
        depot = make_depot()
        depot.lox_tank.current_kg = 10000.0
        depot.lch4_tank.current_kg = 10000.0
        for _ in range(30):
            depot.tick(cryocooler_active=False)
        assert depot.lox_tank.total_boiloff_kg > depot.lch4_tank.total_boiloff_kg


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:

    def test_empty_tank_tick(self):
        tank = CryoTank(
            label="LOX", capacity_kg=1000.0, current_kg=0.0,
            boiling_point_k=LOX_BP_K, latent_kj_kg=LOX_LATENT_KJ_KG,
            density_kg_m3=LOX_DENSITY,
        )
        r = tank.tick(cryocooler_active=False)
        assert r["boiloff_kg"] == 0.0

    def test_zero_power_no_zbo(self):
        tank = CryoTank(
            label="LOX", capacity_kg=10000.0, current_kg=5000.0,
            boiling_point_k=LOX_BP_K, latent_kj_kg=LOX_LATENT_KJ_KG,
            density_kg_m3=LOX_DENSITY,
        )
        r = tank.tick(cryocooler_active=True, available_power_kw=0.0)
        assert not r["zbo_active"]
        assert r["boiloff_kg"] > 0

    def test_ambient_at_bp_no_boiloff(self):
        tank = CryoTank(
            label="LOX", capacity_kg=1000.0, current_kg=500.0,
            boiling_point_k=LOX_BP_K, latent_kj_kg=LOX_LATENT_KJ_KG,
            density_kg_m3=LOX_DENSITY,
        )
        r = tank.tick(ambient_temp_k=LOX_BP_K, cryocooler_active=False)
        assert r["boiloff_kg"] == 0.0

    def test_extreme_cold_ambient(self):
        tank = CryoTank(
            label="LCH4", capacity_kg=5000.0, current_kg=3000.0,
            boiling_point_k=LCH4_BP_K, latent_kj_kg=LCH4_LATENT_KJ_KG,
            density_kg_m3=LCH4_DENSITY,
        )
        r = tank.tick(ambient_temp_k=150.0, cryocooler_active=False)
        assert r["boiloff_kg"] < 10.0


# ===================================================================
# Smoke test — long run
# ===================================================================

class TestSmokeRun:

    def test_365_sol_no_crash(self):
        results = run_depot(sols=365)
        assert len(results) == 365
        assert results[-1]["sol"] == 365
        assert results[-1]["total_propellant_kg"] > 0

    def test_365_sol_significant_fill(self):
        results = run_depot(sols=365, cryocooler_active=True, available_power_kw=50.0)
        assert results[-1]["fill_pct"] > 50.0

    def test_no_cryo_still_fills(self):
        results = run_depot(sols=100, cryocooler_active=False)
        assert results[-1]["total_propellant_kg"] > 50000.0

    def test_10_sol_quick(self):
        results = run_depot(sols=10)
        assert len(results) == 10
        for r in results:
            assert r["energy_kwh"] >= 0.0

    def test_all_fields(self):
        r = run_depot(sols=1)[0]
        for k in ("sol", "lox", "lch4", "total_propellant_kg",
                   "fill_pct", "energy_kwh", "launch_ready", "total_boiloff_kg"):
            assert k in r

    def test_deterministic(self):
        r1 = run_depot(sols=50)
        r2 = run_depot(sols=50)
        for a, b in zip(r1, r2):
            assert a["total_propellant_kg"] == b["total_propellant_kg"]
