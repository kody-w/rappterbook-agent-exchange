"""Tests for condensation_trap.py -- 141 tests across 19 sections."""
from __future__ import annotations
import math, sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from condensation_trap import (
    WATER_SUBLIMATION_ENTHALPY_J_KG, WATER_FUSION_ENTHALPY_J_KG,
    WATER_SPECIFIC_GAS_CONSTANT_J_KGK, ICE_SPECIFIC_HEAT_J_KGK,
    WATER_SPECIFIC_HEAT_J_KGK, T_REF_K, P_REF_PA,
    MARS_SURFACE_PRESSURE_PA, MARS_SURFACE_TEMP_K,
    MARS_H2O_MIXING_RATIO_DEFAULT, MARS_H2O_MIXING_RATIO_MIN,
    MARS_H2O_MIXING_RATIO_MAX, TEC_EFFICIENCY_FRACTION, TEC_IDLE_POWER_W,
    TEC_MAX_DELTA_T_K, DEFAULT_PLATE_AREA_M2, DEFAULT_FAN_POWER_W,
    DEFAULT_WIND_SPEED_M_S, MASS_TRANSFER_COEFF_BASE,
    DUST_ACCUMULATION_PER_SOL, DUST_CLEANING_EFFICIENCY,
    HOURS_PER_SOL, SECONDS_PER_SOL, COLLECTION_DUTY_CYCLE,
    EFFECTIVE_COLLECTION_SECONDS,
    saturation_pressure_ice, frost_point, water_partial_pressure,
    carnot_cop, tec_cop, condensation_rate_kg_s, cooling_power_w,
    tec_input_power_w, melt_energy_j, daily_yield_grams,
    CondensationTrap, TickResult, tick, run_simulation,
)


class TestPhysicalConstants:
    def test_sublimation_enthalpy_range(self):
        assert 2.5e6 < WATER_SUBLIMATION_ENTHALPY_J_KG < 3.0e6
    def test_fusion_enthalpy_range(self):
        assert 3.0e5 < WATER_FUSION_ENTHALPY_J_KG < 3.5e5
    def test_water_gas_constant(self):
        assert 460.0 < WATER_SPECIFIC_GAS_CONSTANT_J_KGK < 463.0
    def test_triple_point_temperature(self):
        assert T_REF_K == 273.16
    def test_triple_point_pressure(self):
        assert abs(P_REF_PA - 611.657) < 0.01
    def test_mars_surface_pressure(self):
        assert 500.0 < MARS_SURFACE_PRESSURE_PA < 700.0
    def test_mars_surface_temperature(self):
        assert 180.0 < MARS_SURFACE_TEMP_K < 240.0
    def test_mars_h2o_mixing_ratios(self):
        assert MARS_H2O_MIXING_RATIO_MIN < MARS_H2O_MIXING_RATIO_DEFAULT < MARS_H2O_MIXING_RATIO_MAX
    def test_sol_duration(self):
        assert 24.6 < HOURS_PER_SOL < 24.7
        assert abs(SECONDS_PER_SOL - HOURS_PER_SOL * 3600) < 1.0


class TestSaturationPressure:
    def test_at_triple_point(self):
        assert abs(saturation_pressure_ice(T_REF_K) - P_REF_PA) < 0.01
    def test_monotonically_increasing(self):
        temps = [150.0, 180.0, 200.0, 220.0, 250.0, 270.0]
        ps = [saturation_pressure_ice(t) for t in temps]
        for i in range(len(ps) - 1):
            assert ps[i] < ps[i + 1]
    def test_very_cold(self):
        assert 0.0 < saturation_pressure_ice(150.0) < 1.0
    def test_at_200k(self):
        assert 0.001 < saturation_pressure_ice(200.0) < 1.0
    def test_zero_temperature(self):
        assert saturation_pressure_ice(0.0) == 0.0
    def test_negative_temperature(self):
        assert saturation_pressure_ice(-10.0) == 0.0
    def test_above_triple_point(self):
        assert saturation_pressure_ice(300.0) == P_REF_PA
    def test_always_positive(self):
        for t in range(50, 274):
            assert saturation_pressure_ice(float(t)) >= 0.0


class TestFrostPoint:
    def test_roundtrip_at_200k(self):
        assert abs(frost_point(saturation_pressure_ice(200.0)) - 200.0) < 0.5
    def test_roundtrip_at_250k(self):
        assert abs(frost_point(saturation_pressure_ice(250.0)) - 250.0) < 0.5
    def test_at_triple_point_pressure(self):
        assert abs(frost_point(P_REF_PA) - T_REF_K) < 0.1
    def test_mars_conditions(self):
        assert 185.0 < frost_point(610.0 * 210e-6) < 210.0
    def test_zero_pressure(self):
        assert frost_point(0.0) == 0.0
    def test_negative_pressure(self):
        assert frost_point(-1.0) == 0.0
    def test_above_triple_pressure(self):
        assert frost_point(1000.0) == T_REF_K
    @pytest.mark.parametrize("tk", [160.0, 180.0, 195.0, 210.0, 240.0, 265.0])
    def test_roundtrip_parametric(self, tk):
        assert abs(frost_point(saturation_pressure_ice(tk)) - tk) < 1.0


class TestWaterPartialPressure:
    def test_mars_default(self):
        assert abs(water_partial_pressure(610.0, 210e-6) - 0.1281) < 0.001
    def test_zero_mixing_ratio(self):
        assert water_partial_pressure(610.0, 0.0) == 0.0
    def test_zero_pressure(self):
        assert water_partial_pressure(0.0, 210e-6) == 0.0
    def test_negative_clamp(self):
        assert water_partial_pressure(-100.0, 210e-6) == 0.0
    def test_high_humidity(self):
        assert abs(water_partial_pressure(610.0, 300e-6) - 0.183) < 0.001


class TestCOP:
    def test_carnot_cop_basic(self):
        assert abs(carnot_cop(180.0, 210.0) - 6.0) < 0.01
    def test_carnot_cop_no_delta(self):
        assert math.isinf(carnot_cop(200.0, 200.0))
    def test_carnot_cop_cold_hotter(self):
        assert math.isinf(carnot_cop(220.0, 200.0))
    def test_tec_cop_is_fraction_of_carnot(self):
        assert abs(tec_cop(180.0, 210.0) - TEC_EFFICIENCY_FRACTION * carnot_cop(180.0, 210.0)) < 0.01
    def test_tec_cop_minimum(self):
        assert tec_cop(100.0, 300.0) >= 0.01
    def test_tec_cop_small_delta(self):
        assert tec_cop(200.0, 205.0) > 5.0


class TestCondensationRate:
    def test_positive_when_below_frost_point(self):
        assert condensation_rate_kg_s(2.0, 3.0, 0.128, saturation_pressure_ice(180.0), 210.0) > 0.0
    def test_zero_when_above_frost_point(self):
        assert condensation_rate_kg_s(2.0, 3.0, 0.128, 1.0, 210.0) == 0.0
    def test_zero_area(self):
        assert condensation_rate_kg_s(0.0, 3.0, 0.128, 0.001, 210.0) == 0.0
    def test_dust_reduces_rate(self):
        ps = saturation_pressure_ice(180.0)
        c = condensation_rate_kg_s(2.0, 3.0, 0.128, ps, 210.0, 0.0)
        d = condensation_rate_kg_s(2.0, 3.0, 0.128, ps, 210.0, 0.5)
        assert d < c and d == pytest.approx(c * 0.5, rel=0.01)
    def test_full_dust_kills_rate(self):
        assert condensation_rate_kg_s(2.0, 3.0, 0.128, saturation_pressure_ice(180.0), 210.0, 1.0) == 0.0
    def test_higher_wind_increases_rate(self):
        ps = saturation_pressure_ice(180.0)
        assert condensation_rate_kg_s(2.0, 10.0, 0.128, ps, 210.0) > condensation_rate_kg_s(2.0, 1.0, 0.128, ps, 210.0)
    def test_larger_area_increases_rate(self):
        ps = saturation_pressure_ice(180.0)
        assert condensation_rate_kg_s(4.0, 3.0, 0.128, ps, 210.0) == pytest.approx(
            condensation_rate_kg_s(1.0, 3.0, 0.128, ps, 210.0) * 4.0, rel=0.01)
    def test_rate_is_non_negative(self):
        assert condensation_rate_kg_s(2.0, 3.0, 0.0, 1.0, 210.0) >= 0.0


class TestCoolingPower:
    def test_zero_rate(self):
        assert cooling_power_w(0.0) == 0.0
    def test_positive_rate(self):
        assert abs(cooling_power_w(1e-6) - 1e-6 * WATER_SUBLIMATION_ENTHALPY_J_KG) < 0.01
    def test_non_negative(self):
        assert cooling_power_w(-1.0) == 0.0


class TestTECInputPower:
    def test_basic(self):
        assert abs(tec_input_power_w(100.0, 2.0) - 50.0) < 0.01
    def test_zero_load(self):
        assert tec_input_power_w(0.0, 2.0) == 0.0
    def test_zero_cop(self):
        assert math.isinf(tec_input_power_w(100.0, 0.0))
    def test_high_cop(self):
        assert abs(tec_input_power_w(100.0, 10.0) - 10.0) < 0.01


class TestMeltEnergy:
    def test_zero_mass(self):
        assert melt_energy_j(0.0, 180.0) == 0.0
    def test_positive(self):
        e = melt_energy_j(1.0, 180.0)
        exp = ICE_SPECIFIC_HEAT_J_KGK * (T_REF_K - 180.0) + WATER_FUSION_ENTHALPY_J_KG + WATER_SPECIFIC_HEAT_J_KGK * (275.0 - T_REF_K)
        assert abs(e - exp) < 1.0
    def test_warmer_ice_less_energy(self):
        assert melt_energy_j(1.0, 250.0) < melt_energy_j(1.0, 150.0)
    def test_at_triple_point(self):
        exp = WATER_FUSION_ENTHALPY_J_KG + WATER_SPECIFIC_HEAT_J_KGK * (275.0 - T_REF_K)
        assert abs(melt_energy_j(1.0, T_REF_K) - exp) < 1.0
    def test_scales_linearly_with_mass(self):
        assert abs(melt_energy_j(2.0, 180.0) - 2.0 * melt_energy_j(1.0, 180.0)) < 1.0


class TestDailyYield:
    def test_default_conditions(self):
        assert 0.0 < daily_yield_grams(2.0, 3.0, 610.0, 210.0, 180.0, 210e-6) < 1000.0
    def test_zero_with_warm_plate(self):
        assert daily_yield_grams(2.0, 3.0, 610.0, 210.0, 210.0, 210e-6) >= 0.0
    def test_more_humidity_more_yield(self):
        assert daily_yield_grams(2.0, 3.0, 610.0, 210.0, 180.0, 300e-6) > daily_yield_grams(2.0, 3.0, 610.0, 210.0, 180.0, 100e-6)
    def test_dust_reduces_yield(self):
        assert daily_yield_grams(2.0, 3.0, 610.0, 210.0, 180.0, 210e-6, 0.5) < daily_yield_grams(2.0, 3.0, 610.0, 210.0, 180.0, 210e-6, 0.0)
    def test_non_negative(self):
        assert daily_yield_grams(2.0, 3.0, 610.0, 210.0, 250.0, 50e-6) >= 0.0


class TestCondensationTrapState:
    def test_default_construction(self):
        t = CondensationTrap()
        assert t.plate_area_m2 == DEFAULT_PLATE_AREA_M2 and t.dust_fraction == 0.0 and t.sol == 0
    def test_plate_temp_auto_set(self):
        assert abs(CondensationTrap().plate_temp_k - (MARS_SURFACE_TEMP_K - 30.0)) < 0.1
    def test_dust_clamped(self):
        assert CondensationTrap(dust_fraction=1.5).dust_fraction == 1.0
        assert CondensationTrap(dust_fraction=-0.5).dust_fraction == 0.0
    def test_ice_clamped(self):
        assert CondensationTrap(ice_collected_g=-10.0).ice_collected_g == 0.0
    def test_area_clamped(self):
        assert CondensationTrap(plate_area_m2=-1.0).plate_area_m2 >= 0.01
    def test_custom_values(self):
        t = CondensationTrap(plate_area_m2=5.0, mixing_ratio=300e-6)
        assert t.plate_area_m2 == 5.0 and t.mixing_ratio == 300e-6


class TestSerialization:
    def test_roundtrip_default(self):
        t = CondensationTrap()
        t2 = CondensationTrap.from_dict(t.to_dict())
        assert t2.plate_area_m2 == t.plate_area_m2 and t2.sol == t.sol
    def test_roundtrip_after_ticks(self):
        t = CondensationTrap(); tick(t); tick(t)
        t2 = CondensationTrap.from_dict(t.to_dict())
        assert t2.sol == 2 and t2.total_energy_wh == pytest.approx(t.total_energy_wh, rel=1e-6)
    def test_extra_keys_ignored(self):
        d = CondensationTrap().to_dict(); d["unknown"] = "x"
        CondensationTrap.from_dict(d)
    def test_dict_has_all_fields(self):
        d = CondensationTrap().to_dict()
        for f in CondensationTrap.__dataclass_fields__:
            assert f in d


class TestTick:
    def test_basic_tick(self):
        t = CondensationTrap(); r = tick(t)
        assert t.sol == 1 and r.yield_g >= 0.0 and r.total_power_w >= TEC_IDLE_POWER_W
    def test_tick_advances_sol(self):
        t = CondensationTrap()
        for _ in range(5): tick(t)
        assert t.sol == 5
    def test_ice_accumulates(self):
        t = CondensationTrap(); tick(t); i1 = t.ice_collected_g; tick(t)
        assert t.ice_collected_g >= i1
    def test_dust_accumulates(self):
        t = CondensationTrap(); tick(t)
        assert abs(t.dust_fraction - DUST_ACCUMULATION_PER_SOL) < 0.001
    def test_melt_transfers_ice_to_water(self):
        t = CondensationTrap(); tick(t); ib = t.ice_collected_g
        r = tick(t, melt_ice=True)
        if ib > 0: assert r.melted_this_sol and t.ice_collected_g == 0.0 and t.water_extracted_g > 0.0
    def test_clean_dust(self):
        t = CondensationTrap(dust_fraction=0.5); tick(t, clean_dust=True)
        assert t.dust_fraction < 0.5
    def test_energy_accumulates(self):
        t = CondensationTrap(); tick(t); e1 = t.total_energy_wh; tick(t)
        assert t.total_energy_wh > e1
    def test_peak_yield_tracked(self):
        t = CondensationTrap(); tick(t); assert t.peak_yield_g >= 0.0
    def test_frost_point_in_result(self):
        t = CondensationTrap(); r = tick(t)
        assert 0.0 < r.frost_point_k < MARS_SURFACE_TEMP_K


class TestConservation:
    def test_mass_conservation(self):
        t = CondensationTrap(); total = sum(tick(t).yield_g for _ in range(20))
        assert abs(t.ice_collected_g - total) < 0.01
    def test_mass_conservation_with_melt(self):
        t = CondensationTrap(); total = 0.0
        for s in range(20): total += tick(t, melt_ice=(s > 0 and s % 5 == 0)).yield_g
        assert abs(t.ice_collected_g + t.water_extracted_g - total) < 0.01
    def test_total_water_tracks_extracted(self):
        t = CondensationTrap()
        for s in range(10): tick(t, melt_ice=(s == 5))
        assert abs(t.total_water_g - t.water_extracted_g) < 0.01
    def test_energy_non_negative(self):
        t = CondensationTrap()
        for _ in range(10): assert tick(t).energy_wh >= 0.0
    def test_cop_le_carnot(self):
        t = CondensationTrap(); r = tick(t)
        c = carnot_cop(t.plate_temp_k, t.ambient_temp_k)
        if not math.isinf(c): assert r.cop <= c + 0.01
    def test_yield_le_max_theoretical(self):
        assert tick(CondensationTrap()).yield_g < 10000.0


class TestMultiSol:
    def test_100_sol_run(self):
        t = CondensationTrap(); assert len(run_simulation(t, 100)) == 100 and t.sol == 100
    def test_water_production(self):
        t = CondensationTrap(); run_simulation(t, 100, 10); assert t.total_water_g > 0.0
    def test_dust_cycles(self):
        t = CondensationTrap(); run_simulation(t, 60); assert t.dust_fraction < 0.18
    def test_yield_decreases_with_dust(self):
        t = CondensationTrap(); rs = run_simulation(t, 30)
        if rs[0].yield_g > 0: assert rs[-1].yield_g <= rs[0].yield_g + 0.001
    def test_energy_monotonic(self):
        t = CondensationTrap(); rs = run_simulation(t, 50)
        running = 0.0
        for r in rs: prev = running; running += r.energy_wh; assert running >= prev


class TestPropertyInvariants:
    @pytest.mark.parametrize("area", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_yield_scales_with_area(self, area):
        y = daily_yield_grams(area, 3.0, 610.0, 210.0, 180.0, 210e-6)
        ref = daily_yield_grams(1.0, 3.0, 610.0, 210.0, 180.0, 210e-6)
        if ref > 0: assert abs(y / ref - area) < 0.01 * area
    @pytest.mark.parametrize("mix", [50e-6, 100e-6, 150e-6, 210e-6, 300e-6])
    def test_yield_non_negative_all_humidities(self, mix):
        assert daily_yield_grams(2.0, 3.0, 610.0, 210.0, 180.0, mix) >= 0.0
    @pytest.mark.parametrize("pt", [130.0, 150.0, 170.0, 190.0, 200.0, 210.0])
    def test_colder_plate_more_yield(self, pt):
        assert daily_yield_grams(2.0, 3.0, 610.0, 210.0, 130.0, 210e-6) >= daily_yield_grams(2.0, 3.0, 610.0, 210.0, pt, 210e-6) - 0.001
    @pytest.mark.parametrize("dust", [0.0, 0.2, 0.5, 0.8, 1.0])
    def test_dust_reduces_or_kills_yield(self, dust):
        assert daily_yield_grams(2.0, 3.0, 610.0, 210.0, 180.0, 210e-6, dust) <= daily_yield_grams(2.0, 3.0, 610.0, 210.0, 180.0, 210e-6, 0.0) + 0.001
    @pytest.mark.parametrize("tk", [150.0, 180.0, 200.0, 220.0, 250.0])
    def test_saturation_pressure_positive(self, tk):
        assert saturation_pressure_ice(tk) > 0.0
    @pytest.mark.parametrize("sols", [1, 10, 50, 100])
    def test_simulation_length(self, sols):
        t = CondensationTrap(); assert len(run_simulation(t, sols)) == sols and t.sol == sols


class TestEdgeCases:
    def test_zero_wind(self):
        assert daily_yield_grams(2.0, 0.0, 610.0, 210.0, 180.0, 210e-6) >= 0.0
    def test_extreme_cold_plate(self):
        r = tick(CondensationTrap(plate_temp_k=80.0))
        assert r.yield_g >= 0.0 and r.total_power_w > 0.0
    def test_very_low_pressure(self):
        assert tick(CondensationTrap(atm_pressure_pa=100.0)).yield_g >= 0.0
    def test_very_high_humidity(self):
        assert tick(CondensationTrap(mixing_ratio=300e-6)).yield_g >= 0.0
    def test_very_low_humidity(self):
        assert tick(CondensationTrap(mixing_ratio=50e-6)).yield_g >= 0.0
    def test_already_at_full_dust(self):
        assert tick(CondensationTrap(dust_fraction=1.0)).yield_g == 0.0
    def test_melt_with_no_ice(self):
        t = CondensationTrap(); t.plate_temp_k = t.ambient_temp_k
        r = tick(t, melt_ice=True)
        assert not r.melted_this_sol or t.ice_collected_g == 0.0
    def test_simultaneous_clean_and_melt(self):
        t = CondensationTrap(dust_fraction=0.5, ice_collected_g=10.0)
        r = tick(t, clean_dust=True, melt_ice=True)
        assert t.dust_fraction < 0.5 and r.melted_this_sol


class TestDustEffects:
    def test_dust_accumulates_each_sol(self):
        t = CondensationTrap()
        for _ in range(10): tick(t)
        assert abs(t.dust_fraction - 10 * DUST_ACCUMULATION_PER_SOL) < 0.001
    def test_dust_never_exceeds_one(self):
        t = CondensationTrap()
        for _ in range(500): tick(t)
        assert t.dust_fraction <= 1.0
    def test_cleaning_removes_90_percent(self):
        t = CondensationTrap(dust_fraction=0.5); tick(t, clean_dust=True)
        assert abs(t.dust_fraction - (0.05 + DUST_ACCUMULATION_PER_SOL)) < 0.001
    def test_cleaning_from_zero(self):
        t = CondensationTrap(dust_fraction=0.0); tick(t, clean_dust=True)
        assert abs(t.dust_fraction - DUST_ACCUMULATION_PER_SOL) < 0.001


class TestSmoke:
    def test_10_sol_smoke(self):
        t = CondensationTrap(); rs = run_simulation(t, 10)
        assert len(rs) == 10 and t.sol == 10
    def test_365_sol_full_year(self):
        t = CondensationTrap(); run_simulation(t, 365, 30)
        assert t.total_water_g > 0.0 and t.total_sols_operated == 365
    def test_serialization_after_simulation(self):
        t = CondensationTrap(); run_simulation(t, 50)
        t2 = CondensationTrap.from_dict(t.to_dict())
        assert t2.sol == t.sol and t2.total_water_g == pytest.approx(t.total_water_g, rel=1e-6)
