"""Tests for propellant_depot.py -- Mars cryogenic propellant storage."""
from __future__ import annotations

import math
import sys
import os

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from propellant_depot import (
    # Constants
    LCH4_BOILING_K, LCH4_DENSITY_KG_M3, LCH4_LATENT_KJ_KG,
    LOX_BOILING_K, LOX_DENSITY_KG_M3, LOX_LATENT_KJ_KG,
    MARS_AMBIENT_K, SECONDS_PER_SOL, MLI_U_W_M2_K,
    CRYO_COP_CH4, CRYO_COP_LOX, MIXTURE_RATIO_OF,
    STARSHIP_PROPELLANT_KG, STARSHIP_LOX_FRACTION, STARSHIP_CH4_FRACTION,
    MIN_ULLAGE_FRACTION, MAX_GAS_FRACTION,
    # Geometry
    sphere_volume_m3, sphere_surface_m2, radius_for_volume,
    tank_surface_for_capacity,
    # Physics
    heat_leak_w, boiloff_kg_per_sol, reliquefaction_kwh,
    # Operations
    compute_mixture_ratio, mission_readiness_fraction,
    receive_propellant, dispense_propellant, dispense_matched_pair,
    tick_tank, tick_depot,
    # Factory
    create_depot, depot_power_estimate_kwh,
    # State
    TankState, DepotState, DepotTickResult,
)


# ─── Physical constants ────────────────────────────────────────────────────────

class TestConstants:
    """Physical constants must be in physically reasonable ranges."""

    def test_ch4_boiling_above_absolute_zero(self):
        assert LCH4_BOILING_K > 0

    def test_lox_boiling_above_absolute_zero(self):
        assert LOX_BOILING_K > 0

    def test_lox_colder_than_ch4(self):
        assert LOX_BOILING_K < LCH4_BOILING_K

    def test_ch4_density_positive(self):
        assert LCH4_DENSITY_KG_M3 > 0

    def test_lox_density_positive(self):
        assert LOX_DENSITY_KG_M3 > 0

    def test_lox_denser_than_ch4(self):
        assert LOX_DENSITY_KG_M3 > LCH4_DENSITY_KG_M3

    def test_ch4_latent_heat_positive(self):
        assert LCH4_LATENT_KJ_KG > 0

    def test_lox_latent_heat_positive(self):
        assert LOX_LATENT_KJ_KG > 0

    def test_mars_ambient_warmer_than_cryogens(self):
        assert MARS_AMBIENT_K > LCH4_BOILING_K
        assert MARS_AMBIENT_K > LOX_BOILING_K

    def test_sol_duration(self):
        assert 88000 < SECONDS_PER_SOL < 89000

    def test_insulation_positive(self):
        assert MLI_U_W_M2_K > 0

    def test_cryo_cop_between_0_and_1(self):
        assert 0 < CRYO_COP_CH4 < 1
        assert 0 < CRYO_COP_LOX < 1

    def test_mixture_ratio_positive(self):
        assert MIXTURE_RATIO_OF > 0

    def test_starship_fractions_sum_to_one(self):
        assert abs(STARSHIP_LOX_FRACTION + STARSHIP_CH4_FRACTION - 1.0) < 1e-6

    def test_ullage_fraction_bounds(self):
        assert 0 < MIN_ULLAGE_FRACTION < 0.5

    def test_max_gas_fraction_greater_than_ullage(self):
        assert MAX_GAS_FRACTION > MIN_ULLAGE_FRACTION


# ─── Geometry ───────────────────────────────────────────────────────────────────

class TestGeometry:
    """Sphere geometry helpers."""

    def test_unit_sphere_volume(self):
        assert abs(sphere_volume_m3(1.0) - (4 / 3) * math.pi) < 1e-6

    def test_unit_sphere_surface(self):
        assert abs(sphere_surface_m2(1.0) - 4 * math.pi) < 1e-6

    def test_volume_scales_cubically(self):
        r = 2.0
        assert abs(sphere_volume_m3(r) / sphere_volume_m3(1.0) - r ** 3) < 1e-6

    def test_surface_scales_quadratically(self):
        r = 3.0
        assert abs(sphere_surface_m2(r) / sphere_surface_m2(1.0) - r ** 2) < 1e-6

    def test_radius_for_volume_roundtrip(self):
        v = 100.0
        r = radius_for_volume(v)
        assert abs(sphere_volume_m3(r) - v) < 1e-4

    def test_radius_for_zero_volume(self):
        assert radius_for_volume(0.0) == 0.0

    def test_radius_for_negative_volume(self):
        assert radius_for_volume(-10.0) == 0.0

    def test_tank_surface_positive(self):
        s = tank_surface_for_capacity(10_000.0, LCH4_DENSITY_KG_M3)
        assert s > 0

    def test_tank_surface_zero_capacity(self):
        assert tank_surface_for_capacity(0.0, LCH4_DENSITY_KG_M3) == 0.0

    def test_tank_surface_larger_for_bigger_tank(self):
        s1 = tank_surface_for_capacity(10_000, LCH4_DENSITY_KG_M3)
        s2 = tank_surface_for_capacity(50_000, LCH4_DENSITY_KG_M3)
        assert s2 > s1

    def test_lox_tank_smaller_surface_than_ch4_same_mass(self):
        """LOX is denser, so same mass needs less volume → less surface."""
        s_ch4 = tank_surface_for_capacity(10_000, LCH4_DENSITY_KG_M3)
        s_lox = tank_surface_for_capacity(10_000, LOX_DENSITY_KG_M3)
        assert s_lox < s_ch4


# ─── Boiloff physics ────────────────────────────────────────────────────────────

class TestBoiloff:
    """Heat leak and boiloff calculations."""

    def test_heat_leak_positive(self):
        hl = heat_leak_w(100.0, MLI_U_W_M2_K, 100.0)
        assert hl > 0

    def test_heat_leak_zero_dt(self):
        assert heat_leak_w(100.0, MLI_U_W_M2_K, 0.0) == 0.0

    def test_heat_leak_zero_area(self):
        assert heat_leak_w(0.0, MLI_U_W_M2_K, 100.0) == 0.0

    def test_heat_leak_negative_dt_clamped(self):
        """Negative dT means cryogen is warmer than environment — no leak."""
        assert heat_leak_w(100.0, MLI_U_W_M2_K, -10.0) == 0.0

    def test_heat_leak_proportional_to_area(self):
        h1 = heat_leak_w(100.0, MLI_U_W_M2_K, 100.0)
        h2 = heat_leak_w(200.0, MLI_U_W_M2_K, 100.0)
        assert abs(h2 / h1 - 2.0) < 1e-6

    def test_boiloff_positive(self):
        bo = boiloff_kg_per_sol(5000.0, LCH4_LATENT_KJ_KG)
        assert bo > 0

    def test_boiloff_zero_heat(self):
        assert boiloff_kg_per_sol(0.0, LCH4_LATENT_KJ_KG) == 0.0

    def test_boiloff_negative_heat_clamped(self):
        assert boiloff_kg_per_sol(-100.0, LCH4_LATENT_KJ_KG) == 0.0

    def test_lox_boiloff_higher_than_ch4_same_heat(self):
        """LOX has lower latent heat, so same heat boils more."""
        bo_ch4 = boiloff_kg_per_sol(5000.0, LCH4_LATENT_KJ_KG)
        bo_lox = boiloff_kg_per_sol(5000.0, LOX_LATENT_KJ_KG)
        assert bo_lox > bo_ch4

    def test_boiloff_in_reasonable_range(self):
        """For a 60t CH4 tank, boiloff should be < 1% per sol."""
        surface = tank_surface_for_capacity(60_000, LCH4_DENSITY_KG_M3)
        dt = MARS_AMBIENT_K - LCH4_BOILING_K
        hl = heat_leak_w(surface, MLI_U_W_M2_K, dt)
        bo = boiloff_kg_per_sol(hl, LCH4_LATENT_KJ_KG)
        assert bo < 0.01 * 60_000  # < 1% per sol


# ─── Reliquefaction ─────────────────────────────────────────────────────────────

class TestReliquefaction:
    """Cryocooler energy calculations."""

    def test_reliquefaction_positive(self):
        e = reliquefaction_kwh(10.0, LCH4_LATENT_KJ_KG, CRYO_COP_CH4)
        assert e > 0

    def test_reliquefaction_zero_mass(self):
        assert reliquefaction_kwh(0.0, LCH4_LATENT_KJ_KG, CRYO_COP_CH4) == 0

    def test_reliquefaction_zero_cop(self):
        assert reliquefaction_kwh(10.0, LCH4_LATENT_KJ_KG, 0.0) == 0

    def test_ch4_more_expensive_per_kg_to_reliquefy(self):
        """CH4 has higher latent heat, so despite better COP,
        it costs more energy per kg to reliquefy than LOX."""
        e_ch4 = reliquefaction_kwh(1.0, LCH4_LATENT_KJ_KG, CRYO_COP_CH4)
        e_lox = reliquefaction_kwh(1.0, LOX_LATENT_KJ_KG, CRYO_COP_LOX)
        assert e_ch4 > e_lox


# ─── Mixture ratio & mission readiness ──────────────────────────────────────────

class TestMixtureAndMission:
    """Mixture ratio and mission readiness tracking."""

    def test_ideal_mixture_ratio(self):
        r = compute_mixture_ratio(3600.0, 1000.0)
        assert abs(r - 3.6) < 1e-6

    def test_mixture_ratio_no_ch4(self):
        r = compute_mixture_ratio(100.0, 0.0)
        assert r == float("inf")

    def test_mixture_ratio_both_zero(self):
        assert compute_mixture_ratio(0.0, 0.0) == 0.0

    def test_mission_readiness_zero_propellant(self):
        assert mission_readiness_fraction(0.0, 0.0) == 0.0

    def test_mission_readiness_full(self):
        lox = STARSHIP_PROPELLANT_KG * STARSHIP_LOX_FRACTION
        ch4 = STARSHIP_PROPELLANT_KG * STARSHIP_CH4_FRACTION
        assert abs(mission_readiness_fraction(lox, ch4) - 1.0) < 1e-6

    def test_mission_readiness_half_ch4(self):
        lox = STARSHIP_PROPELLANT_KG * STARSHIP_LOX_FRACTION
        ch4 = STARSHIP_PROPELLANT_KG * STARSHIP_CH4_FRACTION * 0.5
        r = mission_readiness_fraction(lox, ch4)
        assert abs(r - 0.5) < 1e-6

    def test_mission_readiness_capped_at_one(self):
        r = mission_readiness_fraction(1e9, 1e9)
        assert r <= 1.0

    def test_mission_readiness_limited_by_bottleneck(self):
        """Readiness is limited by the lesser of the two propellants."""
        r = mission_readiness_fraction(
            0.0,  # no LOX
            STARSHIP_PROPELLANT_KG * STARSHIP_CH4_FRACTION  # full CH4
        )
        assert r == 0.0


# ─── Tank operations ────────────────────────────────────────────────────────────

class TestTankOps:
    """Receive and dispense propellant."""

    @pytest.fixture
    def ch4_tank(self):
        return TankState(
            label="ch4", capacity_kg=10_000, liquid_kg=5_000,
            surface_m2=50.0, boiling_k=LCH4_BOILING_K,
            latent_kj_kg=LCH4_LATENT_KJ_KG,
            density_kg_m3=LCH4_DENSITY_KG_M3, cryo_cop=CRYO_COP_CH4)

    def test_receive_normal(self, ch4_tank):
        stored = receive_propellant(ch4_tank, 1000.0)
        assert stored == 1000.0
        assert ch4_tank.liquid_kg == 6000.0

    def test_receive_overflow(self, ch4_tank):
        stored = receive_propellant(ch4_tank, 20_000.0)
        assert stored == 5000.0
        assert ch4_tank.liquid_kg == 10_000.0

    def test_receive_zero(self, ch4_tank):
        stored = receive_propellant(ch4_tank, 0.0)
        assert stored == 0.0

    def test_receive_negative(self, ch4_tank):
        stored = receive_propellant(ch4_tank, -100.0)
        assert stored == 0.0

    def test_dispense_normal(self, ch4_tank):
        dispensed = dispense_propellant(ch4_tank, 1000.0)
        assert dispensed == 1000.0
        assert ch4_tank.liquid_kg == 4000.0

    def test_dispense_more_than_available(self, ch4_tank):
        dispensed = dispense_propellant(ch4_tank, 8000.0)
        assert dispensed == 5000.0
        assert ch4_tank.liquid_kg == 0.0

    def test_dispense_zero(self, ch4_tank):
        assert dispense_propellant(ch4_tank, 0.0) == 0.0

    def test_dispense_negative(self, ch4_tank):
        assert dispense_propellant(ch4_tank, -100.0) == 0.0

    def test_fill_fraction(self, ch4_tank):
        assert abs(ch4_tank.fill_fraction() - 0.5) < 1e-6

    def test_fill_fraction_empty(self):
        t = TankState(label="ch4", capacity_kg=1000, liquid_kg=0)
        assert t.fill_fraction() == 0.0

    def test_fill_fraction_full(self):
        t = TankState(label="ch4", capacity_kg=1000, liquid_kg=1000)
        assert abs(t.fill_fraction() - 1.0) < 1e-6

    def test_fill_fraction_zero_capacity(self):
        t = TankState(label="ch4", capacity_kg=0, liquid_kg=0)
        assert t.fill_fraction() == 0.0


# ─── Matched dispensing ─────────────────────────────────────────────────────────

class TestMatchedDispensing:
    """Dispensing LOX + CH4 at correct mixture ratio."""

    @pytest.fixture
    def depot(self):
        return create_depot(
            ch4_capacity_kg=60_000, lox_capacity_kg=200_000,
            ch4_initial_kg=30_000, lox_initial_kg=100_000)

    def test_dispense_pair_ratio(self, depot):
        lox_out, ch4_out = dispense_matched_pair(depot, 4600.0)
        expected_ch4 = 4600.0 / (1 + MIXTURE_RATIO_OF)
        expected_lox = 4600.0 - expected_ch4
        assert abs(ch4_out - expected_ch4) < 0.1
        assert abs(lox_out - expected_lox) < 0.1

    def test_dispense_pair_zero(self, depot):
        lox, ch4 = dispense_matched_pair(depot, 0.0)
        assert lox == 0.0
        assert ch4 == 0.0

    def test_dispense_pair_conserves_mass(self, depot):
        before_ch4 = depot.ch4_tank.liquid_kg
        before_lox = depot.lox_tank.liquid_kg
        lox_out, ch4_out = dispense_matched_pair(depot, 1000.0)
        after_ch4 = depot.ch4_tank.liquid_kg
        after_lox = depot.lox_tank.liquid_kg
        assert abs((before_ch4 - after_ch4) - ch4_out) < 1e-6
        assert abs((before_lox - after_lox) - lox_out) < 1e-6


# ─── Tick tank ──────────────────────────────────────────────────────────────────

class TestTickTank:
    """Single-tank tick behavior."""

    @pytest.fixture
    def full_ch4_tank(self):
        return TankState(
            label="ch4", capacity_kg=60_000, liquid_kg=60_000,
            surface_m2=tank_surface_for_capacity(60_000, LCH4_DENSITY_KG_M3),
            boiling_k=LCH4_BOILING_K, latent_kj_kg=LCH4_LATENT_KJ_KG,
            density_kg_m3=LCH4_DENSITY_KG_M3, cryo_cop=CRYO_COP_CH4)

    def test_boiloff_occurs(self, full_ch4_tank):
        res = tick_tank(full_ch4_tank, MARS_AMBIENT_K, 0.0)
        assert res["boiloff_kg"] > 0

    def test_zero_power_no_reliquefaction(self, full_ch4_tank):
        res = tick_tank(full_ch4_tank, MARS_AMBIENT_K, 0.0)
        assert res["reliquefied_kg"] == 0.0

    def test_reliquefaction_with_power(self, full_ch4_tank):
        # First tick to generate boiloff gas
        tick_tank(full_ch4_tank, MARS_AMBIENT_K, 0.0)
        assert full_ch4_tank.boiloff_gas_kg > 0
        # Second tick with power recovers some
        res = tick_tank(full_ch4_tank, MARS_AMBIENT_K, 100.0)
        assert res["reliquefied_kg"] > 0

    def test_no_boiloff_if_empty(self):
        t = TankState(
            label="ch4", capacity_kg=60_000, liquid_kg=0,
            surface_m2=100.0, boiling_k=LCH4_BOILING_K,
            latent_kj_kg=LCH4_LATENT_KJ_KG,
            density_kg_m3=LCH4_DENSITY_KG_M3, cryo_cop=CRYO_COP_CH4)
        res = tick_tank(t, MARS_AMBIENT_K, 0.0)
        assert res["boiloff_kg"] == 0.0

    def test_boiloff_limited_by_liquid_mass(self):
        """Can't boil more than what's in the tank."""
        t = TankState(
            label="ch4", capacity_kg=60_000, liquid_kg=0.001,
            surface_m2=100.0, boiling_k=LCH4_BOILING_K,
            latent_kj_kg=LCH4_LATENT_KJ_KG,
            density_kg_m3=LCH4_DENSITY_KG_M3, cryo_cop=CRYO_COP_CH4)
        res = tick_tank(t, MARS_AMBIENT_K, 0.0)
        assert res["boiloff_kg"] <= 0.001 + 1e-9


# ─── Full depot tick ────────────────────────────────────────────────────────────

class TestTickDepot:
    """Full-depot one-sol advancement."""

    @pytest.fixture
    def depot(self):
        return create_depot(
            ch4_capacity_kg=60_000, lox_capacity_kg=200_000,
            ch4_initial_kg=30_000, lox_initial_kg=100_000)

    def test_tick_advances_sol(self, depot):
        depot, _ = tick_depot(depot)
        assert depot.sols_running == 1

    def test_tick_produces_boiloff(self, depot):
        _, r = tick_depot(depot)
        assert r.ch4_boiloff_kg > 0
        assert r.lox_boiloff_kg > 0

    def test_tick_receives_propellant(self, depot):
        _, r = tick_depot(depot, ch4_input_kg=100, lox_input_kg=360)
        assert r.ch4_received_kg == 100
        assert r.lox_received_kg == 360

    def test_tick_cumulative_tracking(self, depot):
        depot, _ = tick_depot(depot, ch4_input_kg=50, lox_input_kg=180)
        depot, _ = tick_depot(depot, ch4_input_kg=50, lox_input_kg=180)
        assert depot.total_ch4_received_kg == 100
        assert depot.total_lox_received_kg == 360
        assert depot.sols_running == 2

    def test_tick_energy_non_negative(self, depot):
        _, r = tick_depot(depot, power_budget_kwh=20.0)
        assert r.total_energy_kwh >= 0

    def test_tick_fill_fractions(self, depot):
        _, r = tick_depot(depot)
        assert 0 <= r.ch4_fill_fraction <= 1
        assert 0 <= r.lox_fill_fraction <= 1

    def test_tick_mission_readiness(self, depot):
        _, r = tick_depot(depot)
        assert 0 <= r.mission_readiness <= 1

    def test_alert_nominal(self, depot):
        _, r = tick_depot(depot)
        assert r.alert == "nominal"

    def test_alert_warning_near_full(self):
        d = create_depot(
            ch4_capacity_kg=10_000, lox_capacity_kg=10_000,
            ch4_initial_kg=9_900, lox_initial_kg=5_000)
        # Receive more to push past 95%
        receive_propellant(d.ch4_tank, 200)
        _, r = tick_depot(d)
        assert r.alert == "warning"

    def test_alert_low_on_empty(self):
        d = create_depot(
            ch4_capacity_kg=60_000, lox_capacity_kg=200_000,
            ch4_initial_kg=50, lox_initial_kg=50)
        _, r = tick_depot(d)
        assert r.alert == "low"


# ─── Multi-sol simulation ──────────────────────────────────────────────────────

class TestMultiSol:
    """Run the depot for many sols and check properties."""

    def test_ten_sol_no_crash(self):
        depot = create_depot(ch4_initial_kg=30_000, lox_initial_kg=100_000)
        for _ in range(10):
            depot, _ = tick_depot(depot, power_budget_kwh=20.0)
        assert depot.sols_running == 10

    def test_hundred_sol_no_crash(self):
        depot = create_depot(ch4_initial_kg=30_000, lox_initial_kg=100_000)
        for _ in range(100):
            depot, _ = tick_depot(depot, ch4_input_kg=50, lox_input_kg=180,
                                   power_budget_kwh=20.0)
        assert depot.sols_running == 100

    def test_propellant_accumulates_with_input(self):
        """Net propellant should increase if input > boiloff."""
        depot = create_depot(
            ch4_capacity_kg=15_000, lox_capacity_kg=50_000,
            ch4_initial_kg=1000, lox_initial_kg=3600)
        for _ in range(50):
            depot, _ = tick_depot(depot, ch4_input_kg=200, lox_input_kg=720,
                                   power_budget_kwh=50.0)
        assert depot.ch4_tank.liquid_kg > 1000
        assert depot.lox_tank.liquid_kg > 3600

    def test_propellant_depletes_without_input(self):
        """With no input and limited power, propellant should decrease."""
        depot = create_depot(ch4_initial_kg=30_000, lox_initial_kg=100_000)
        initial_ch4 = depot.ch4_tank.liquid_kg
        for _ in range(50):
            depot, _ = tick_depot(depot, power_budget_kwh=0.0)
        assert depot.ch4_tank.liquid_kg < initial_ch4

    def test_mission_readiness_grows(self):
        """Mission readiness should increase as propellant accumulates."""
        depot = create_depot(ch4_initial_kg=0, lox_initial_kg=0)
        _, r0 = tick_depot(depot, ch4_input_kg=500, lox_input_kg=1800)
        for _ in range(99):
            depot, r = tick_depot(depot, ch4_input_kg=500, lox_input_kg=1800,
                                   power_budget_kwh=50.0)
        assert r.mission_readiness > r0.mission_readiness


# ─── Factory ────────────────────────────────────────────────────────────────────

class TestFactory:
    """Factory function and power estimator."""

    def test_create_depot_default(self):
        d = create_depot()
        assert d.ch4_tank.capacity_kg == 60_000
        assert d.lox_tank.capacity_kg == 200_000
        assert d.ch4_tank.liquid_kg == 0
        assert d.lox_tank.liquid_kg == 0
        assert d.sols_running == 0

    def test_create_depot_with_initial(self):
        d = create_depot(ch4_initial_kg=5000, lox_initial_kg=18000)
        assert d.ch4_tank.liquid_kg == 5000
        assert d.lox_tank.liquid_kg == 18000

    def test_create_depot_initial_clamped_to_capacity(self):
        d = create_depot(ch4_capacity_kg=1000, ch4_initial_kg=5000)
        assert d.ch4_tank.liquid_kg == 1000

    def test_create_depot_surface_computed(self):
        d = create_depot()
        assert d.ch4_tank.surface_m2 > 0
        assert d.lox_tank.surface_m2 > 0

    def test_create_depot_physics_set(self):
        d = create_depot()
        assert d.ch4_tank.boiling_k == LCH4_BOILING_K
        assert d.lox_tank.boiling_k == LOX_BOILING_K

    def test_power_estimate_positive(self):
        d = create_depot(ch4_initial_kg=30_000, lox_initial_kg=100_000)
        p = depot_power_estimate_kwh(d)
        assert p > 0

    def test_power_estimate_reasonable(self):
        """Should be in reasonable kWh range for typical depot."""
        d = create_depot(ch4_initial_kg=30_000, lox_initial_kg=100_000)
        p = depot_power_estimate_kwh(d)
        assert 10 < p < 5000


# ─── Physical invariants ────────────────────────────────────────────────────────

class TestInvariants:
    """Property-based invariants that must hold across all runs."""

    def test_mass_conservation_single_tick(self):
        """Total mass (liquid + gas + vented) is conserved per tick."""
        depot = create_depot(ch4_initial_kg=30_000, lox_initial_kg=100_000)
        ch4_before = depot.ch4_tank.liquid_kg + depot.ch4_tank.boiloff_gas_kg
        lox_before = depot.lox_tank.liquid_kg + depot.lox_tank.boiloff_gas_kg

        _, r = tick_depot(depot, power_budget_kwh=10.0)

        ch4_after = depot.ch4_tank.liquid_kg + depot.ch4_tank.boiloff_gas_kg
        lox_after = depot.lox_tank.liquid_kg + depot.lox_tank.boiloff_gas_kg

        # Mass in = mass out (accounting for vented gas)
        assert abs(ch4_before - ch4_after - r.ch4_vented_kg) < 1e-6
        assert abs(lox_before - lox_after - r.lox_vented_kg) < 1e-6

    def test_mass_conservation_with_input(self):
        depot = create_depot(ch4_initial_kg=10_000, lox_initial_kg=36_000)
        ch4_before = depot.ch4_tank.liquid_kg + depot.ch4_tank.boiloff_gas_kg
        lox_before = depot.lox_tank.liquid_kg + depot.lox_tank.boiloff_gas_kg

        _, r = tick_depot(depot, ch4_input_kg=100, lox_input_kg=360,
                           power_budget_kwh=10.0)

        ch4_after = depot.ch4_tank.liquid_kg + depot.ch4_tank.boiloff_gas_kg
        lox_after = depot.lox_tank.liquid_kg + depot.lox_tank.boiloff_gas_kg

        assert abs((ch4_before + r.ch4_received_kg) - ch4_after
                    - r.ch4_vented_kg) < 1e-6
        assert abs((lox_before + r.lox_received_kg) - lox_after
                    - r.lox_vented_kg) < 1e-6

    def test_liquid_never_negative(self):
        depot = create_depot(ch4_initial_kg=1, lox_initial_kg=1)
        for _ in range(200):
            depot, _ = tick_depot(depot, power_budget_kwh=0.0)
            assert depot.ch4_tank.liquid_kg >= 0
            assert depot.lox_tank.liquid_kg >= 0

    def test_gas_never_negative(self):
        depot = create_depot(ch4_initial_kg=1000, lox_initial_kg=3600)
        for _ in range(50):
            depot, _ = tick_depot(depot, power_budget_kwh=100.0)
            assert depot.ch4_tank.boiloff_gas_kg >= 0
            assert depot.lox_tank.boiloff_gas_kg >= 0

    def test_fill_fraction_bounded(self):
        depot = create_depot(ch4_initial_kg=30_000, lox_initial_kg=100_000)
        for _ in range(50):
            depot, r = tick_depot(depot, ch4_input_kg=500, lox_input_kg=1800,
                                   power_budget_kwh=20.0)
            assert 0 <= r.ch4_fill_fraction <= 1
            assert 0 <= r.lox_fill_fraction <= 1

    def test_mission_readiness_bounded(self):
        depot = create_depot(ch4_initial_kg=50_000, lox_initial_kg=190_000)
        for _ in range(50):
            depot, r = tick_depot(depot, ch4_input_kg=500, lox_input_kg=1800,
                                   power_budget_kwh=30.0)
            assert 0 <= r.mission_readiness <= 1

    def test_energy_non_negative_always(self):
        depot = create_depot(ch4_initial_kg=30_000, lox_initial_kg=100_000)
        for _ in range(50):
            depot, r = tick_depot(depot, power_budget_kwh=20.0)
            assert r.total_energy_kwh >= 0
            assert r.ch4_reliq_kwh >= 0
            assert r.lox_reliq_kwh >= 0

    def test_cumulative_boiloff_monotonic(self):
        depot = create_depot(ch4_initial_kg=30_000, lox_initial_kg=100_000)
        prev_ch4 = 0.0
        prev_lox = 0.0
        for _ in range(30):
            depot, _ = tick_depot(depot, power_budget_kwh=5.0)
            assert depot.total_ch4_boiloff_kg >= prev_ch4
            assert depot.total_lox_boiloff_kg >= prev_lox
            prev_ch4 = depot.total_ch4_boiloff_kg
            prev_lox = depot.total_lox_boiloff_kg
