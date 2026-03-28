"""Tests for the Mars Parabolic Solar Thermal Concentrator module.

134 tests covering constants, orbital mechanics, optics, thermal physics,
dust degradation, salt storage, tick simulation, and physical invariants.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import solar_concentrator as sc


# ===================================================================
# TestConstants  (10 tests)
# ===================================================================
class TestConstants:
    """Sanity-check physical constants and default values."""

    def test_stefan_boltzmann_order_of_magnitude(self):
        assert 5.0e-8 < sc.STEFAN_BOLTZMANN < 6.0e-8

    def test_solar_constant_positive(self):
        assert sc.SOLAR_CONSTANT_1AU > 1300.0

    def test_mars_semi_major_au(self):
        assert 1.4 < sc.MARS_SEMI_MAJOR_AU < 1.7

    def test_mars_eccentricity_range(self):
        assert 0.0 < sc.MARS_ECCENTRICITY < 0.15

    def test_mars_year_sols(self):
        assert 660 < sc.MARS_YEAR_SOLS < 680

    def test_mars_sol_hours(self):
        assert 24.0 < sc.MARS_SOL_HOURS < 25.0

    def test_mars_ambient_temp(self):
        assert 180 < sc.MARS_AMBIENT_TEMP_K < 240

    def test_default_mirror_area(self):
        assert sc.DEFAULT_MIRROR_AREA_M2 == 100.0

    def test_salt_temp_range_positive(self):
        assert sc.SALT_MAX_TEMP_K > sc.SALT_MIN_TEMP_K

    def test_min_reflectivity_positive(self):
        assert 0.0 < sc.MIN_REFLECTIVITY < sc.DEFAULT_MIRROR_REFLECTIVITY


# ===================================================================
# TestMarsDistance  (5 tests)
# ===================================================================
class TestMarsDistance:
    """Orbital distance calculations."""

    def test_positive_distance(self):
        assert sc.mars_distance_au(0) > 0.0

    def test_perihelion_closer(self):
        # At sol=0, cos(0)=1 so r = a*(1-e) = minimum
        r0 = sc.mars_distance_au(0)
        r_half = sc.mars_distance_au(sc.MARS_YEAR_SOLS / 2)
        assert r0 < r_half

    def test_aphelion_farther(self):
        r_half = sc.mars_distance_au(sc.MARS_YEAR_SOLS / 2)
        assert r_half > sc.MARS_SEMI_MAJOR_AU

    def test_distance_periodic(self):
        r0 = sc.mars_distance_au(0)
        r_period = sc.mars_distance_au(sc.MARS_YEAR_SOLS)
        assert abs(r0 - r_period) < 1e-10

    def test_eccentricity_range(self):
        r_min = sc.mars_distance_au(0)
        r_max = sc.mars_distance_au(sc.MARS_YEAR_SOLS / 2)
        ratio = r_max / r_min
        assert 1.0 < ratio < 1.3


# ===================================================================
# TestSolarFlux  (8 tests)
# ===================================================================
class TestSolarFlux:
    """Instantaneous solar flux."""

    def test_positive_at_zenith(self):
        assert sc.solar_flux_w_m2(100, 0.5, 0.0) > 0.0

    def test_zero_at_horizon(self):
        assert sc.solar_flux_w_m2(100, 0.5, 90.0) == 0.0

    def test_zero_below_horizon(self):
        assert sc.solar_flux_w_m2(100, 0.5, 120.0) == 0.0

    def test_decreases_with_zenith(self):
        f0 = sc.solar_flux_w_m2(100, 0.5, 0.0)
        f60 = sc.solar_flux_w_m2(100, 0.5, 60.0)
        assert f0 > f60

    def test_decreases_with_dust(self):
        f_clear = sc.solar_flux_w_m2(100, 0.1, 30.0)
        f_dusty = sc.solar_flux_w_m2(100, 2.0, 30.0)
        assert f_clear > f_dusty

    def test_inverse_square(self):
        # Closer sol should yield higher flux
        f_peri = sc.solar_flux_w_m2(0, 0.5, 0.0)
        f_aph = sc.solar_flux_w_m2(sc.MARS_YEAR_SOLS / 2, 0.5, 0.0)
        assert f_peri > f_aph

    def test_no_dust_higher(self):
        f_no_dust = sc.solar_flux_w_m2(100, 0.0, 0.0)
        f_some = sc.solar_flux_w_m2(100, 0.5, 0.0)
        assert f_no_dust > f_some

    def test_flux_order_of_magnitude(self):
        f = sc.solar_flux_w_m2(100, 0.5, 0.0)
        assert 100 < f < 1000


# ===================================================================
# TestDailyAverageFlux  (4 tests)
# ===================================================================
class TestDailyAverageFlux:
    """Daily average flux."""

    def test_positive(self):
        assert sc.daily_average_flux_w_m2(100, 0.5) > 0.0

    def test_less_than_peak(self):
        peak = sc.solar_flux_w_m2(100, 0.5, 0.0)
        avg = sc.daily_average_flux_w_m2(100, 0.5)
        assert avg < peak

    def test_decreases_with_dust(self):
        avg_clear = sc.daily_average_flux_w_m2(100, 0.3)
        avg_dusty = sc.daily_average_flux_w_m2(100, 2.0)
        assert avg_clear > avg_dusty

    def test_varies_with_sol(self):
        avg_peri = sc.daily_average_flux_w_m2(0, 0.5)
        avg_aph = sc.daily_average_flux_w_m2(sc.MARS_YEAR_SOLS / 2, 0.5)
        assert avg_peri > avg_aph


# ===================================================================
# TestConcentration  (5 tests)
# ===================================================================
class TestConcentration:
    """Geometric concentration ratio."""

    def test_basic_ratio(self):
        assert sc.concentration_ratio(100.0, 0.05) == pytest.approx(2000.0)

    def test_capped_at_max(self):
        ratio = sc.concentration_ratio(100_000.0, 0.01)
        assert ratio == sc.MAX_PRACTICAL_CONCENTRATION

    def test_proportional_to_mirror(self):
        r1 = sc.concentration_ratio(50.0, 0.05)
        r2 = sc.concentration_ratio(100.0, 0.05)
        assert r2 == pytest.approx(2 * r1)

    def test_inverse_to_receiver(self):
        r1 = sc.concentration_ratio(100.0, 0.05)
        r2 = sc.concentration_ratio(100.0, 0.10)
        assert r1 == pytest.approx(2 * r2)

    def test_zero_receiver_gives_max(self):
        assert sc.concentration_ratio(100.0, 0.0) == sc.MAX_PRACTICAL_CONCENTRATION


# ===================================================================
# TestEquilibriumTemperature  (6 tests)
# ===================================================================
class TestEquilibriumTemperature:
    """Stagnation temperature."""

    def test_ambient_with_no_flux(self):
        t = sc.equilibrium_temperature_k(0.0, 2000, 0.95, 0.90)
        assert t == sc.MARS_AMBIENT_TEMP_K

    def test_increases_with_concentration(self):
        t1 = sc.equilibrium_temperature_k(400, 500, 0.95, 0.90)
        t2 = sc.equilibrium_temperature_k(400, 2000, 0.95, 0.90)
        assert t2 > t1

    def test_increases_with_flux(self):
        t1 = sc.equilibrium_temperature_k(200, 2000, 0.95, 0.90)
        t2 = sc.equilibrium_temperature_k(600, 2000, 0.95, 0.90)
        assert t2 > t1

    def test_reaches_industrial_temps(self):
        t = sc.equilibrium_temperature_k(500, 2000, 0.95, 0.90)
        assert t > 1000.0

    def test_fourth_root_scaling(self):
        t1 = sc.equilibrium_temperature_k(100, 1000, 0.95, 0.90)
        t16 = sc.equilibrium_temperature_k(1600, 1000, 0.95, 0.90)
        # 16x flux → 2x temperature (fourth root)
        assert t16 == pytest.approx(t1 * 2.0, rel=0.01)

    def test_ambient_with_zero_emissivity(self):
        t = sc.equilibrium_temperature_k(400, 2000, 0.95, 0.0)
        assert t == sc.MARS_AMBIENT_TEMP_K


# ===================================================================
# TestAbsorbedPower  (6 tests)
# ===================================================================
class TestAbsorbedPower:
    """Absorbed thermal power."""

    def test_positive_with_valid_inputs(self):
        p = sc.absorbed_power_kw(400, 100, 0.92, 0.95)
        assert p > 0.0

    def test_zero_with_zero_flux(self):
        assert sc.absorbed_power_kw(0.0, 100, 0.92, 0.95) == 0.0

    def test_zero_with_zero_area(self):
        assert sc.absorbed_power_kw(400, 0.0, 0.92, 0.95) == 0.0

    def test_proportional_to_flux(self):
        p1 = sc.absorbed_power_kw(200, 100, 0.92, 0.95)
        p2 = sc.absorbed_power_kw(400, 100, 0.92, 0.95)
        assert p2 == pytest.approx(2 * p1)

    def test_proportional_to_area(self):
        p1 = sc.absorbed_power_kw(400, 50, 0.92, 0.95)
        p2 = sc.absorbed_power_kw(400, 100, 0.92, 0.95)
        assert p2 == pytest.approx(2 * p1)

    def test_zero_reflectivity(self):
        assert sc.absorbed_power_kw(400, 100, 0.0, 0.95) == 0.0


# ===================================================================
# TestThermalLosses  (8 tests)
# ===================================================================
class TestThermalLosses:
    """Radiation, conduction, and net thermal power."""

    def test_radiation_zero_at_ambient(self):
        loss = sc.radiation_loss_kw(sc.MARS_AMBIENT_TEMP_K, 0.05, 0.90)
        assert abs(loss) < 1e-12

    def test_radiation_positive_above_ambient(self):
        loss = sc.radiation_loss_kw(800, 0.05, 0.90)
        assert loss > 0.0

    def test_radiation_t4_scaling(self):
        l1 = sc.radiation_loss_kw(500, 0.05, 0.90)
        l2 = sc.radiation_loss_kw(1000, 0.05, 0.90)
        # T^4 scaling: 2x temp → ~16x loss (minus ambient correction)
        assert l2 > 10 * l1

    def test_conduction_zero_at_ambient(self):
        loss = sc.conduction_loss_kw(sc.MARS_AMBIENT_TEMP_K)
        assert abs(loss) < 1e-12

    def test_conduction_linear(self):
        l1 = sc.conduction_loss_kw(410)  # 200 K above ambient
        l2 = sc.conduction_loss_kw(610)  # 400 K above ambient
        assert l2 == pytest.approx(2 * l1)

    def test_net_power_positive(self):
        # With large absorbed power and moderate temperature
        net = sc.net_thermal_power_kw(50.0, 800, 0.05, 0.90)
        assert net > 0.0

    def test_net_power_clamped_at_zero(self):
        net = sc.net_thermal_power_kw(0.001, 5000, 1.0, 1.0)
        assert net == 0.0

    def test_net_less_than_absorbed(self):
        absorbed = 50.0
        net = sc.net_thermal_power_kw(absorbed, 900, 0.05, 0.90)
        assert net <= absorbed


# ===================================================================
# TestDustDegradation  (8 tests)
# ===================================================================
class TestDustDegradation:
    """Mirror dust and cleaning."""

    def test_one_sol_loss(self):
        r0 = 0.92
        r1 = sc.reflectivity_after_dust(r0, 1)
        expected = r0 * (1 - sc.DUST_REFLECTIVITY_LOSS_PER_SOL)
        assert r1 == pytest.approx(expected)

    def test_monotonic_decrease(self):
        r0 = 0.90
        vals = [sc.reflectivity_after_dust(r0, s) for s in range(0, 50)]
        for a, b in zip(vals, vals[1:]):
            assert a >= b

    def test_never_below_minimum(self):
        r = sc.reflectivity_after_dust(0.92, 10_000)
        assert r >= sc.MIN_REFLECTIVITY

    def test_many_sols_approaches_min(self):
        r = sc.reflectivity_after_dust(0.92, 5000)
        assert r == pytest.approx(sc.MIN_REFLECTIVITY)

    def test_clean_restores(self):
        cleaned = sc.clean_mirror(0.50, 0.92, 0)
        assert cleaned > 0.50

    def test_clean_below_baseline(self):
        cleaned = sc.clean_mirror(0.50, 0.92, 0)
        expected = 0.92 - sc.PERMANENT_WEATHERING_PER_CLEAN
        assert cleaned == pytest.approx(expected)

    def test_weathering_cumulative(self):
        c1 = sc.clean_mirror(0.50, 0.92, 0)
        c10 = sc.clean_mirror(0.50, 0.92, 9)
        assert c1 > c10

    def test_weathering_never_below_min(self):
        cleaned = sc.clean_mirror(0.50, 0.92, 100_000)
        assert cleaned >= sc.MIN_REFLECTIVITY


# ===================================================================
# TestSaltStorage  (13 tests)
# ===================================================================
class TestSaltStorage:
    """Molten salt thermal storage."""

    def test_capacity_positive(self):
        assert sc.salt_energy_capacity_kwh(5000) > 0.0

    def test_capacity_proportional_to_mass(self):
        c1 = sc.salt_energy_capacity_kwh(5000)
        c2 = sc.salt_energy_capacity_kwh(10000)
        assert c2 == pytest.approx(2 * c1)

    def test_capacity_proportional_to_temp_range(self):
        c1 = sc.salt_energy_capacity_kwh(5000, 200)
        c2 = sc.salt_energy_capacity_kwh(5000, 400)
        assert c2 == pytest.approx(2 * c1)

    def test_capacity_default_temp_range(self):
        c_default = sc.salt_energy_capacity_kwh(5000)
        c_explicit = sc.salt_energy_capacity_kwh(5000, sc.SALT_MAX_TEMP_K - sc.SALT_MIN_TEMP_K)
        assert c_default == pytest.approx(c_explicit)

    def test_charge_basic(self):
        charged = sc.salt_charge_kwh(10.0, 5.0, 0.0, 1000.0)
        assert charged == pytest.approx(50.0)

    def test_charge_capped_at_capacity(self):
        charged = sc.salt_charge_kwh(100.0, 100.0, 900.0, 1000.0)
        assert charged == pytest.approx(100.0)

    def test_charge_zero_power(self):
        assert sc.salt_charge_kwh(0.0, 10.0, 0.0, 1000.0) == 0.0

    def test_discharge_basic(self):
        discharged = sc.salt_discharge_kwh(5.0, 10.0, 100.0)
        assert discharged == pytest.approx(50.0)

    def test_discharge_capped_at_stored(self):
        discharged = sc.salt_discharge_kwh(100.0, 100.0, 50.0)
        assert discharged == pytest.approx(50.0)

    def test_discharge_zero_stored(self):
        assert sc.salt_discharge_kwh(10.0, 10.0, 0.0) == 0.0

    def test_thermal_loss_rate(self):
        loss = sc.salt_thermal_loss_kwh(100.0)
        expected = 100.0 * sc.SALT_TANK_LOSS_FRAC_PER_SOL
        assert loss == pytest.approx(expected)

    def test_thermal_loss_zero_when_empty(self):
        assert sc.salt_thermal_loss_kwh(0.0) == 0.0

    def test_charge_never_negative(self):
        charged = sc.salt_charge_kwh(-5.0, 10.0, 500.0, 1000.0)
        assert charged >= 0.0


# ===================================================================
# TestCreate  (6 tests)
# ===================================================================
class TestCreate:
    """create_concentrator factory."""

    def test_returns_tuple(self):
        result = sc.create_concentrator()
        assert isinstance(result, tuple) and len(result) == 2

    def test_config_concentration(self):
        cfg, _ = sc.create_concentrator()
        assert cfg.concentration == pytest.approx(2000.0)

    def test_config_salt_capacity(self):
        cfg, _ = sc.create_concentrator()
        assert cfg.salt_capacity_kwh > 0.0

    def test_config_mirror_diameter(self):
        cfg, _ = sc.create_concentrator()
        expected = 2.0 * math.sqrt(100.0 / math.pi)
        assert cfg.mirror_diameter_m == pytest.approx(expected, rel=1e-4)

    def test_state_initial_reflectivity(self):
        cfg, st = sc.create_concentrator()
        assert st.reflectivity == cfg.baseline_reflectivity

    def test_clamps_negative_area(self):
        cfg, _ = sc.create_concentrator(mirror_area_m2=-5.0)
        assert cfg.mirror_area_m2 >= 0.0


# ===================================================================
# TestTick  (13 tests)
# ===================================================================
class TestTick:
    """Single-sol tick."""

    def test_sol_increments(self):
        cfg, st = sc.create_concentrator()
        sc.tick(cfg, st)
        assert st.sols_operated == 1

    def test_absorbs_energy(self):
        cfg, st = sc.create_concentrator()
        result = sc.tick(cfg, st)
        assert result["absorbed_kwh"] > 0.0

    def test_receiver_heated(self):
        cfg, st = sc.create_concentrator()
        sc.tick(cfg, st)
        assert st.receiver_temp_k > sc.MARS_AMBIENT_TEMP_K

    def test_reflectivity_degrades(self):
        cfg, st = sc.create_concentrator()
        r0 = st.reflectivity
        sc.tick(cfg, st)
        assert st.reflectivity < r0

    def test_dust_storm_reduces_flux(self):
        cfg1, st1 = sc.create_concentrator()
        cfg2, st2 = sc.create_concentrator()
        r_clear = sc.tick(cfg1, st1, is_dust_storm=False)
        r_storm = sc.tick(cfg2, st2, is_dust_storm=True)
        assert r_clear["flux_w_m2"] > r_storm["flux_w_m2"]

    def test_dust_storm_solar_hours(self):
        cfg, st = sc.create_concentrator()
        result = sc.tick(cfg, st, is_dust_storm=True)
        assert result["solar_hours"] == sc.SOLAR_HOURS_STORM

    def test_clear_day_solar_hours(self):
        cfg, st = sc.create_concentrator()
        result = sc.tick(cfg, st, is_dust_storm=False)
        assert result["solar_hours"] == sc.SOLAR_HOURS_CLEAR

    def test_storage_charges_with_surplus(self):
        cfg, st = sc.create_concentrator()
        result = sc.tick(cfg, st, thermal_demand_kw=0.0)
        assert result["charged_kwh"] > 0.0

    def test_storage_discharges_with_demand(self):
        cfg, st = sc.create_concentrator()
        # Pre-charge salt
        st.salt_stored_kwh = 200.0
        # Large demand + dust storm → mostly from salt
        result = sc.tick(cfg, st, thermal_demand_kw=500.0, is_dust_storm=True)
        assert result["discharged_kwh"] > 0.0

    def test_sol_of_year_wraps(self):
        cfg, st = sc.create_concentrator(initial_sol=sc.MARS_YEAR_SOLS - 1)
        sc.tick(cfg, st)
        assert st.sol_of_year < sc.MARS_YEAR_SOLS

    def test_explicit_sol_of_year(self):
        cfg, st = sc.create_concentrator()
        sc.tick(cfg, st, sol_of_year=300.0)
        assert st.sol_of_year == pytest.approx(300.0)

    def test_net_kwh_nonnegative(self):
        cfg, st = sc.create_concentrator()
        result = sc.tick(cfg, st)
        assert result["net_kwh"] >= 0.0

    def test_salt_stored_nonnegative_after_tick(self):
        cfg, st = sc.create_concentrator()
        st.salt_stored_kwh = 1.0
        sc.tick(cfg, st, thermal_demand_kw=9999.0)
        assert st.salt_stored_kwh >= 0.0


# ===================================================================
# TestClean  (4 tests)
# ===================================================================
class TestClean:
    """Mirror cleaning."""

    def test_restores_reflectivity(self):
        cfg, st = sc.create_concentrator()
        for _ in range(30):
            sc.tick(cfg, st)
        old = st.reflectivity
        sc.clean(cfg, st)
        assert st.reflectivity > old

    def test_resets_sols_since_clean(self):
        cfg, st = sc.create_concentrator()
        for _ in range(10):
            sc.tick(cfg, st)
        sc.clean(cfg, st)
        assert st.sols_since_clean == 0

    def test_increments_cleaning_count(self):
        cfg, st = sc.create_concentrator()
        sc.clean(cfg, st)
        assert st.cleanings_total == 1

    def test_returns_dict_with_keys(self):
        cfg, st = sc.create_concentrator()
        result = sc.clean(cfg, st)
        assert "old_reflectivity" in result
        assert "new_reflectivity" in result
        assert "cleanings_total" in result


# ===================================================================
# TestStatus  (3 tests)
# ===================================================================
class TestStatus:
    """Status snapshot."""

    def test_keys_present(self):
        cfg, st = sc.create_concentrator()
        s = sc.status(cfg, st)
        for key in (
            "sols_operated", "reflectivity", "receiver_temp_k",
            "salt_stored_kwh", "salt_capacity_kwh", "efficiency_pct",
            "concentration_ratio", "mirror_area_m2",
        ):
            assert key in s

    def test_efficiency_bounded(self):
        cfg, st = sc.create_concentrator()
        for _ in range(30):
            sc.tick(cfg, st, thermal_demand_kw=10.0)
        s = sc.status(cfg, st)
        assert 0.0 <= s["efficiency_pct"] <= 100.0

    def test_storage_pct_bounded(self):
        cfg, st = sc.create_concentrator()
        s = sc.status(cfg, st)
        assert 0.0 <= s["salt_storage_pct"] <= 100.0


# ===================================================================
# TestEnergyConservation  (6 tests)
# ===================================================================
class TestEnergyConservation:
    """Energy balance invariants."""

    def test_absorbed_ge_delivered_lifetime(self):
        cfg, st = sc.create_concentrator()
        for _ in range(100):
            sc.tick(cfg, st, thermal_demand_kw=20.0)
        assert st.total_absorbed_kwh >= st.total_delivered_kwh

    def test_salt_bounded_by_capacity(self):
        cfg, st = sc.create_concentrator()
        for _ in range(200):
            sc.tick(cfg, st, thermal_demand_kw=0.0)
        assert st.salt_stored_kwh <= cfg.salt_capacity_kwh + 1e-6

    def test_net_kwh_le_absorbed(self):
        cfg, st = sc.create_concentrator()
        result = sc.tick(cfg, st)
        assert result["net_kwh"] <= result["absorbed_kwh"] + 1e-9

    def test_delivered_le_demand(self):
        demand_kw = 10.0
        cfg, st = sc.create_concentrator()
        result = sc.tick(cfg, st, thermal_demand_kw=demand_kw)
        demand_kwh = demand_kw * sc.MARS_SOL_HOURS
        assert result["delivered_kwh"] <= demand_kwh + 1e-9

    def test_monotonic_total_absorbed(self):
        cfg, st = sc.create_concentrator()
        prev = 0.0
        for _ in range(50):
            sc.tick(cfg, st)
            assert st.total_absorbed_kwh >= prev
            prev = st.total_absorbed_kwh

    def test_monotonic_total_delivered(self):
        cfg, st = sc.create_concentrator()
        prev = 0.0
        for _ in range(50):
            sc.tick(cfg, st, thermal_demand_kw=5.0)
            assert st.total_delivered_kwh >= prev
            prev = st.total_delivered_kwh


# ===================================================================
# TestRunSimulation  (9 tests)
# ===================================================================
class TestRunSimulation:
    """Multi-sol simulation."""

    def test_correct_sol_count(self):
        result = sc.run_simulation(sols=50)
        assert len(result["history"]) == 50

    def test_returns_final_status(self):
        result = sc.run_simulation(sols=10)
        assert "efficiency_pct" in result["final_status"]

    def test_returns_config(self):
        result = sc.run_simulation(sols=10)
        assert "mirror_area_m2" in result["config"]

    def test_delivers_energy(self):
        result = sc.run_simulation(sols=30, thermal_demand_kw=10.0)
        total = sum(h["delivered_kwh"] for h in result["history"])
        assert total > 0.0

    def test_cleanings_happen(self):
        result = sc.run_simulation(sols=60, clean_interval_sols=30)
        assert result["final_status"]["cleanings_total"] >= 2

    def test_dust_storms_tracked(self):
        storms = {5, 6, 7}
        result = sc.run_simulation(sols=20, dust_storm_sols=storms)
        storm_ticks = [h for h in result["history"] if h["is_dust_storm"]]
        assert len(storm_ticks) == 3

    def test_default_creates_concentrator(self):
        result = sc.run_simulation(sols=5)
        assert result["sols"] == 5

    def test_custom_config(self):
        cfg, st = sc.create_concentrator(mirror_area_m2=200.0)
        result = sc.run_simulation(sols=5, config=cfg, state=st)
        assert result["config"]["mirror_area_m2"] == 200.0

    def test_zero_clean_interval(self):
        # clean_interval_sols=0 means no cleaning
        result = sc.run_simulation(sols=30, clean_interval_sols=0)
        assert result["final_status"]["cleanings_total"] == 0


# ===================================================================
# TestSmokeTests  (10 tests)
# ===================================================================
class TestSmokeTests:
    """Smoke tests — run without crashing."""

    def test_10_ticks(self):
        cfg, st = sc.create_concentrator()
        for _ in range(10):
            sc.tick(cfg, st)
        assert st.sols_operated == 10

    def test_100_ticks(self):
        cfg, st = sc.create_concentrator()
        for _ in range(100):
            sc.tick(cfg, st, thermal_demand_kw=20.0)
        assert st.sols_operated == 100

    def test_full_mars_year(self):
        result = sc.run_simulation(sols=668, thermal_demand_kw=30.0,
                                   clean_interval_sols=30)
        assert result["final_status"]["sols_operated"] == 668

    def test_zero_demand(self):
        cfg, st = sc.create_concentrator()
        result = sc.tick(cfg, st, thermal_demand_kw=0.0)
        assert result["delivered_kwh"] == 0.0

    def test_huge_demand(self):
        cfg, st = sc.create_concentrator()
        result = sc.tick(cfg, st, thermal_demand_kw=10_000.0)
        assert result["delivered_kwh"] >= 0.0

    def test_tiny_mirror(self):
        cfg, st = sc.create_concentrator(mirror_area_m2=0.01)
        result = sc.tick(cfg, st)
        assert result["absorbed_kwh"] >= 0.0

    def test_no_salt(self):
        cfg, st = sc.create_concentrator(salt_mass_kg=0.0)
        result = sc.tick(cfg, st, thermal_demand_kw=10.0)
        assert result["charged_kwh"] == 0.0

    def test_max_dust(self):
        cfg, st = sc.create_concentrator()
        result = sc.tick(cfg, st, dust_optical_depth=10.0)
        assert result["flux_w_m2"] >= 0.0

    def test_perihelion_start(self):
        cfg, st = sc.create_concentrator(initial_sol=0.0)
        result = sc.tick(cfg, st)
        assert result["absorbed_kwh"] > 0.0

    def test_aphelion_start(self):
        cfg, st = sc.create_concentrator(initial_sol=sc.MARS_YEAR_SOLS / 2)
        result = sc.tick(cfg, st)
        assert result["absorbed_kwh"] > 0.0


# ===================================================================
# TestPhysicalInvariants  (10 tests)
# ===================================================================
class TestPhysicalInvariants:
    """Physical invariants that must hold across any simulation run."""

    def test_reflectivity_bounded_above(self):
        cfg, st = sc.create_concentrator()
        for _ in range(100):
            sc.tick(cfg, st)
            assert st.reflectivity <= cfg.baseline_reflectivity

    def test_reflectivity_bounded_below(self):
        cfg, st = sc.create_concentrator()
        for _ in range(5000):
            sc.tick(cfg, st)
        assert st.reflectivity >= sc.MIN_REFLECTIVITY

    def test_receiver_temp_positive(self):
        cfg, st = sc.create_concentrator()
        for _ in range(30):
            sc.tick(cfg, st)
            assert st.receiver_temp_k > 0.0

    def test_energies_nonnegative(self):
        cfg, st = sc.create_concentrator()
        for _ in range(50):
            r = sc.tick(cfg, st, thermal_demand_kw=30.0)
            assert r["absorbed_kwh"] >= 0.0
            assert r["delivered_kwh"] >= 0.0
            assert r["lost_kwh"] >= 0.0

    def test_delivered_le_demand_always(self):
        demand = 20.0
        cfg, st = sc.create_concentrator()
        demand_kwh = demand * sc.MARS_SOL_HOURS
        for _ in range(60):
            r = sc.tick(cfg, st, thermal_demand_kw=demand)
            assert r["delivered_kwh"] <= demand_kwh + 1e-9

    def test_salt_never_negative(self):
        cfg, st = sc.create_concentrator()
        for _ in range(100):
            sc.tick(cfg, st, thermal_demand_kw=999.0)
            assert st.salt_stored_kwh >= -1e-12

    def test_salt_never_exceeds_capacity(self):
        cfg, st = sc.create_concentrator()
        for _ in range(200):
            sc.tick(cfg, st, thermal_demand_kw=0.0)
            assert st.salt_stored_kwh <= cfg.salt_capacity_kwh + 1e-6

    def test_absorbed_ge_net(self):
        cfg, st = sc.create_concentrator()
        for _ in range(30):
            r = sc.tick(cfg, st)
            assert r["absorbed_kwh"] >= r["net_kwh"] - 1e-9

    def test_cleaning_restores_above_dirty(self):
        cfg, st = sc.create_concentrator()
        for _ in range(50):
            sc.tick(cfg, st)
        dirty = st.reflectivity
        sc.clean(cfg, st)
        assert st.reflectivity > dirty

    def test_dust_storm_lower_absorbed(self):
        cfg1, st1 = sc.create_concentrator()
        cfg2, st2 = sc.create_concentrator()
        r_clear = sc.tick(cfg1, st1, is_dust_storm=False)
        r_storm = sc.tick(cfg2, st2, is_dust_storm=True)
        assert r_clear["absorbed_kwh"] > r_storm["absorbed_kwh"]
