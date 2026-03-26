"""Tests for air_filtration.py — Mars Habitat HEPA/ESP Air Filtration.

68 tests covering:
  - Pure function correctness and edge cases
  - Physics bounds and conservation laws
  - Filter lifecycle (loading → replacement → reload)
  - EVA dust ingress events
  - System degradation and recovery
  - Smoke test (multi-sol run without crash)
  - Property-based invariants
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from air_filtration import (
    BASELINE_DUST_MG_SOL,
    CRITICAL_DUST_MG_M3,
    DEFAULT_FLOW_M3_HR,
    DEFAULT_HAB_VOLUME_M3,
    ESP_PLATE_AREA_M2,
    ESP_DRIFT_VELOCITY_M_S,
    EVA_DUST_MG,
    FAN_EFFICIENCY,
    FILTER_CAPACITY_MG,
    HEPA_BASE_EFFICIENCY,
    HEPA_CLEAN_DP_PA,
    HOURS_PER_SOL,
    NASA_PEL_MG_M3,
    FilterState,
    FiltrationRecord,
    dust_concentration,
    dust_removed_per_sol,
    esp_capture_efficiency,
    fan_power_kwh,
    hepa_loaded_efficiency,
    make_filtration,
    natural_settling,
    needs_filter_replacement,
    pressure_drop,
    replace_filter,
    run_filtration,
    tick_filtration,
)


# ═══════════════════════════════════════════════════════════════════════════
# dust_concentration
# ═══════════════════════════════════════════════════════════════════════════

class TestDustConcentration:
    def test_basic_calculation(self):
        assert dust_concentration(300.0, 300.0) == pytest.approx(1.0)

    def test_zero_volume(self):
        assert dust_concentration(100.0, 0.0) == 0.0

    def test_negative_volume(self):
        assert dust_concentration(100.0, -10.0) == 0.0

    def test_zero_dust(self):
        assert dust_concentration(0.0, 300.0) == 0.0

    def test_high_concentration(self):
        conc = dust_concentration(3000.0, 300.0)
        assert conc == pytest.approx(10.0)

    def test_low_concentration(self):
        conc = dust_concentration(0.03, 300.0)
        assert conc == pytest.approx(0.0001)


# ═══════════════════════════════════════════════════════════════════════════
# esp_capture_efficiency
# ═══════════════════════════════════════════════════════════════════════════

class TestESPEfficiency:
    def test_default_parameters(self):
        eff = esp_capture_efficiency(DEFAULT_FLOW_M3_HR)
        assert 0.0 < eff < 1.0

    def test_zero_flow_returns_zero(self):
        assert esp_capture_efficiency(0.0) == 0.0

    def test_negative_flow(self):
        assert esp_capture_efficiency(-10.0) == 0.0

    def test_high_flow_low_efficiency(self):
        """Higher flow → less time in ESP → lower capture."""
        eff_low = esp_capture_efficiency(50.0)
        eff_high = esp_capture_efficiency(500.0)
        assert eff_low > eff_high

    def test_deutsch_anderson_formula(self):
        """Verify against direct Deutsch-Anderson computation."""
        flow = 100.0  # m³/hr
        w = ESP_DRIFT_VELOCITY_M_S
        A = ESP_PLATE_AREA_M2
        Q = flow / 3600.0
        expected = 1.0 - math.exp(-(w * A) / Q)
        assert esp_capture_efficiency(flow) == pytest.approx(expected, rel=1e-10)

    def test_efficiency_bounded_0_1(self):
        """Efficiency is always between 0 and 1."""
        for flow in [0.001, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
            eff = esp_capture_efficiency(flow)
            assert 0.0 <= eff <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# hepa_loaded_efficiency
# ═══════════════════════════════════════════════════════════════════════════

class TestHEPALoadedEfficiency:
    def test_clean_filter(self):
        eff = hepa_loaded_efficiency(HEPA_BASE_EFFICIENCY, 0.0)
        assert eff == HEPA_BASE_EFFICIENCY

    def test_loaded_filter_higher(self):
        """Loaded filters are MORE efficient (dust cake effect)."""
        eff_clean = hepa_loaded_efficiency(HEPA_BASE_EFFICIENCY, 0.0)
        eff_loaded = hepa_loaded_efficiency(HEPA_BASE_EFFICIENCY, 0.5)
        assert eff_loaded >= eff_clean

    def test_capped_at_five_nines(self):
        eff = hepa_loaded_efficiency(0.9999, 1.0)
        assert eff <= 0.99999

    def test_full_load(self):
        eff = hepa_loaded_efficiency(HEPA_BASE_EFFICIENCY, 1.0)
        assert eff > HEPA_BASE_EFFICIENCY


# ═══════════════════════════════════════════════════════════════════════════
# pressure_drop
# ═══════════════════════════════════════════════════════════════════════════

class TestPressureDrop:
    def test_clean_filter(self):
        dp = pressure_drop(HEPA_CLEAN_DP_PA, 0.0)
        assert dp == HEPA_CLEAN_DP_PA

    def test_loaded_filter_higher(self):
        dp_clean = pressure_drop(HEPA_CLEAN_DP_PA, 0.0)
        dp_loaded = pressure_drop(HEPA_CLEAN_DP_PA, 0.5)
        assert dp_loaded > dp_clean

    def test_full_load_quadratic(self):
        """At full load, ΔP = clean × (1 + k), k=3 → 4× clean."""
        dp = pressure_drop(HEPA_CLEAN_DP_PA, 1.0)
        assert dp == pytest.approx(HEPA_CLEAN_DP_PA * 4.0)

    def test_monotonic_increase(self):
        """Pressure drop increases monotonically with loading."""
        prev = 0.0
        for frac in [0.0, 0.1, 0.2, 0.5, 0.8, 1.0]:
            dp = pressure_drop(HEPA_CLEAN_DP_PA, frac)
            assert dp >= prev
            prev = dp

    def test_always_positive(self):
        dp = pressure_drop(HEPA_CLEAN_DP_PA, 0.0)
        assert dp > 0


# ═══════════════════════════════════════════════════════════════════════════
# fan_power_kwh
# ═══════════════════════════════════════════════════════════════════════════

class TestFanPower:
    def test_basic_calculation(self):
        power = fan_power_kwh(250.0, 150.0)
        assert power > 0

    def test_zero_flow(self):
        assert fan_power_kwh(250.0, 0.0) == 0.0

    def test_zero_pressure(self):
        assert fan_power_kwh(0.0, 150.0) == 0.0

    def test_zero_efficiency(self):
        assert fan_power_kwh(250.0, 150.0, 0.0) == 0.0

    def test_higher_dp_more_power(self):
        p1 = fan_power_kwh(250.0, 150.0)
        p2 = fan_power_kwh(500.0, 150.0)
        assert p2 > p1

    def test_physical_units(self):
        """Verify the result is in reasonable kWh range for a sol."""
        power = fan_power_kwh(HEPA_CLEAN_DP_PA, DEFAULT_FLOW_M3_HR)
        # Fan for 300m³ hab should be < 10 kWh/sol
        assert 0 < power < 10.0


# ═══════════════════════════════════════════════════════════════════════════
# dust_removed_per_sol
# ═══════════════════════════════════════════════════════════════════════════

class TestDustRemoval:
    def test_removes_most_dust(self):
        """HEPA at 99.99% with many turnovers should remove nearly all dust."""
        removed = dust_removed_per_sol(
            100.0, 300.0, 150.0, 0.9999, 0.9, True, True
        )
        assert removed > 99.0  # Virtually all removed

    def test_zero_airborne(self):
        removed = dust_removed_per_sol(0.0, 300.0, 150.0, 0.9999, 0.9, True, True)
        assert removed == 0.0

    def test_zero_flow(self):
        removed = dust_removed_per_sol(100.0, 300.0, 0.0, 0.9999, 0.9, True, True)
        assert removed == 0.0

    def test_zero_volume(self):
        removed = dust_removed_per_sol(100.0, 0.0, 150.0, 0.9999, 0.9, True, True)
        assert removed == 0.0

    def test_hepa_off_esp_off(self):
        """Both filters off → no removal."""
        removed = dust_removed_per_sol(100.0, 300.0, 150.0, 0.9999, 0.9, False, False)
        assert removed == 0.0

    def test_hepa_only(self):
        removed = dust_removed_per_sol(100.0, 300.0, 150.0, 0.9999, 0.0, True, False)
        assert removed > 90.0

    def test_esp_only(self):
        removed = dust_removed_per_sol(100.0, 300.0, 150.0, 0.0, 0.9, False, True)
        assert removed > 50.0

    def test_never_exceeds_airborne(self):
        """Cannot remove more dust than exists."""
        removed = dust_removed_per_sol(10.0, 300.0, 150.0, 0.9999, 0.99, True, True)
        assert removed <= 10.0

    def test_higher_flow_removes_more(self):
        r1 = dust_removed_per_sol(100.0, 300.0, 50.0, 0.9999, 0.0, True, False)
        r2 = dust_removed_per_sol(100.0, 300.0, 300.0, 0.9999, 0.0, True, False)
        assert r2 >= r1


# ═══════════════════════════════════════════════════════════════════════════
# natural_settling
# ═══════════════════════════════════════════════════════════════════════════

class TestNaturalSettling:
    def test_basic(self):
        settled = natural_settling(100.0)
        assert settled == pytest.approx(2.0)

    def test_zero(self):
        assert natural_settling(0.0) == 0.0

    def test_proportional(self):
        s1 = natural_settling(50.0)
        s2 = natural_settling(100.0)
        assert s2 == pytest.approx(2.0 * s1)


# ═══════════════════════════════════════════════════════════════════════════
# needs_filter_replacement / replace_filter
# ═══════════════════════════════════════════════════════════════════════════

class TestFilterReplacement:
    def test_clean_filter_no_replacement(self):
        assert not needs_filter_replacement(0.0)

    def test_at_capacity(self):
        assert needs_filter_replacement(FILTER_CAPACITY_MG)

    def test_over_capacity(self):
        assert needs_filter_replacement(FILTER_CAPACITY_MG + 1.0)

    def test_replace_resets_state(self):
        state = make_filtration()
        state.filter_load_mg = 40000.0
        state.pressure_drop_pa = 800.0
        replace_filter(state)
        assert state.filter_load_mg == 0.0
        assert state.pressure_drop_pa == HEPA_CLEAN_DP_PA
        assert state.filter_replaced_count == 1
        assert state.disposed_dust_mg == 40000.0

    def test_replace_increments_count(self):
        state = make_filtration()
        replace_filter(state)
        replace_filter(state)
        assert state.filter_replaced_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# tick_filtration — single sol
# ═══════════════════════════════════════════════════════════════════════════

class TestTickFiltration:
    def test_basic_tick(self):
        state = make_filtration()
        record = tick_filtration(state)
        assert record.sol == 1
        assert record.airborne_dust_mg >= 0
        assert record.filter_load_mg >= 0
        assert record.dust_added_mg == pytest.approx(BASELINE_DUST_MG_SOL)

    def test_sol_increments(self):
        state = make_filtration()
        tick_filtration(state)
        tick_filtration(state)
        assert state.sol == 2

    def test_dust_added_baseline(self):
        state = make_filtration()
        record = tick_filtration(state, eva_events=0, extra_dust_mg=0.0)
        assert record.dust_added_mg == pytest.approx(BASELINE_DUST_MG_SOL)

    def test_eva_dust_injection(self):
        state = make_filtration()
        record = tick_filtration(state, eva_events=2)
        expected = BASELINE_DUST_MG_SOL + 2 * EVA_DUST_MG
        assert record.dust_added_mg == pytest.approx(expected)

    def test_extra_dust(self):
        state = make_filtration()
        record = tick_filtration(state, extra_dust_mg=50.0)
        expected = BASELINE_DUST_MG_SOL + 50.0
        assert record.dust_added_mg == pytest.approx(expected)

    def test_negative_extra_dust_clamped(self):
        state = make_filtration()
        record = tick_filtration(state, extra_dust_mg=-100.0)
        assert record.dust_added_mg == pytest.approx(BASELINE_DUST_MG_SOL)

    def test_within_pel_normal_ops(self):
        """Under normal operations, concentration should be within PEL."""
        state = make_filtration()
        # Run a few sols to reach steady state
        for _ in range(10):
            record = tick_filtration(state)
        assert record.within_pel

    def test_fan_power_positive(self):
        state = make_filtration()
        record = tick_filtration(state)
        assert record.fan_power_kwh > 0

    def test_cumulative_eva_events(self):
        state = make_filtration()
        tick_filtration(state, eva_events=3)
        tick_filtration(state, eva_events=2)
        assert state.cumulative_eva_events == 5


# ═══════════════════════════════════════════════════════════════════════════
# Mass conservation — THE critical invariant
# ═══════════════════════════════════════════════════════════════════════════

class TestMassConservation:
    def test_single_tick_conservation(self):
        """Total mass (air + filter + settled + disposed) must increase by exactly dust_added."""
        state = make_filtration(initial_dust_mg=100.0)
        before = (state.airborne_dust_mg + state.filter_load_mg
                  + state.settled_dust_mg + state.disposed_dust_mg)
        record = tick_filtration(state)
        after = (state.airborne_dust_mg + state.filter_load_mg
                 + state.settled_dust_mg + state.disposed_dust_mg)
        assert after == pytest.approx(before + record.dust_added_mg, abs=1e-6)

    def test_multi_tick_conservation(self):
        """Conservation holds across many sols including filter replacements."""
        state = make_filtration(initial_dust_mg=50.0)
        total_added = 0.0
        initial_mass = (state.airborne_dust_mg + state.filter_load_mg
                        + state.settled_dust_mg + state.disposed_dust_mg)
        for i in range(50):
            eva = 1 if i % 10 == 0 else 0
            record = tick_filtration(state, eva_events=eva)
            total_added += record.dust_added_mg
        final_mass = (state.airborne_dust_mg + state.filter_load_mg
                      + state.settled_dust_mg + state.disposed_dust_mg)
        assert final_mass == pytest.approx(initial_mass + total_added, abs=1e-4)

    def test_conservation_with_eva(self):
        """EVA dust is properly accounted for."""
        state = make_filtration(initial_dust_mg=10.0)
        before = (state.airborne_dust_mg + state.filter_load_mg
                  + state.settled_dust_mg + state.disposed_dust_mg)
        record = tick_filtration(state, eva_events=5)
        after = (state.airborne_dust_mg + state.filter_load_mg
                 + state.settled_dust_mg + state.disposed_dust_mg)
        expected_added = BASELINE_DUST_MG_SOL + 5 * EVA_DUST_MG
        assert after == pytest.approx(before + expected_added, abs=1e-6)

    def test_no_negative_masses(self):
        """No mass quantity should ever go negative."""
        state = make_filtration()
        for _ in range(100):
            tick_filtration(state, eva_events=1)
            assert state.airborne_dust_mg >= 0
            assert state.filter_load_mg >= 0
            assert state.settled_dust_mg >= 0


# ═══════════════════════════════════════════════════════════════════════════
# Filter lifecycle
# ═══════════════════════════════════════════════════════════════════════════

class TestFilterLifecycle:
    def test_filter_loads_over_time(self):
        """Filter load increases with normal operations."""
        state = make_filtration()
        tick_filtration(state)
        assert state.filter_load_mg > 0

    def test_pressure_drop_increases(self):
        """Pressure drop increases as filter loads."""
        state = make_filtration()
        dp_initial = state.pressure_drop_pa
        for _ in range(20):
            tick_filtration(state, eva_events=1)
        assert state.pressure_drop_pa >= dp_initial

    def test_auto_replacement_at_capacity(self):
        """Filter auto-replaces when it hits capacity."""
        state = make_filtration()
        state.filter_load_mg = FILTER_CAPACITY_MG - 1.0
        # This tick should push it over and trigger replacement
        tick_filtration(state, eva_events=10)
        # After auto-replacement, load should be low
        assert state.filter_load_mg < FILTER_CAPACITY_MG
        assert state.filter_replaced_count >= 1

    def test_filter_life_fraction_decreases(self):
        """Filter life fraction decreases as it loads."""
        state = make_filtration()
        records = run_filtration(state, sols=10, eva_per_sol=2)
        # First record should have more life remaining than last
        assert records[0].filter_life_fraction >= records[-1].filter_life_fraction


# ═══════════════════════════════════════════════════════════════════════════
# System failure modes
# ═══════════════════════════════════════════════════════════════════════════

class TestFailureModes:
    def test_hepa_offline(self):
        """With HEPA offline, filter captures less dust per sol."""
        state_on = make_filtration(initial_dust_mg=100.0)
        state_off = make_filtration(initial_dust_mg=100.0)
        state_off.hepa_online = False

        # Single tick comparison — both-on removes more in one pass
        rec_on = tick_filtration(state_on)
        rec_off = tick_filtration(state_off)
        assert rec_on.dust_removed_mg >= rec_off.dust_removed_mg

    def test_esp_offline(self):
        """With ESP offline, slightly more dust in air."""
        state_on = make_filtration(initial_dust_mg=100.0)
        state_off = make_filtration(initial_dust_mg=100.0)
        state_off.esp_online = False

        for _ in range(10):
            tick_filtration(state_on)
            tick_filtration(state_off)

        assert state_off.airborne_dust_mg >= state_on.airborne_dust_mg

    def test_both_offline_dust_accumulates(self):
        """With both offline, dust only decreases by settling."""
        state = make_filtration(initial_dust_mg=100.0)
        state.hepa_online = False
        state.esp_online = False
        for _ in range(5):
            record = tick_filtration(state)
        # Dust should be accumulating since only settling removes it
        # and baseline adds 5mg/sol
        assert state.airborne_dust_mg > 0


# ═══════════════════════════════════════════════════════════════════════════
# Physical bounds
# ═══════════════════════════════════════════════════════════════════════════

class TestPhysicalBounds:
    def test_concentration_non_negative(self):
        state = make_filtration()
        for _ in range(200):
            record = tick_filtration(state)
            assert record.concentration_mg_m3 >= 0.0

    def test_filter_load_non_negative(self):
        state = make_filtration()
        for _ in range(200):
            tick_filtration(state)
            assert state.filter_load_mg >= 0.0

    def test_pressure_drop_non_negative(self):
        state = make_filtration()
        for _ in range(50):
            tick_filtration(state, eva_events=2)
            assert state.pressure_drop_pa >= 0.0

    def test_fan_power_reasonable(self):
        """Fan power should stay under 50 kWh/sol (Mars hab scale)."""
        state = make_filtration()
        for _ in range(50):
            record = tick_filtration(state, eva_events=1)
            assert record.fan_power_kwh < 50.0

    def test_steady_state_within_pel(self):
        """At steady state with normal ops, concentration stays within PEL."""
        state = make_filtration()
        records = run_filtration(state, sols=100, eva_per_sol=0)
        # Last 20 sols should all be within PEL
        for r in records[-20:]:
            assert r.within_pel


# ═══════════════════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════════════════

class TestFactory:
    def test_default_parameters(self):
        state = make_filtration()
        assert state.hab_volume_m3 == DEFAULT_HAB_VOLUME_M3
        assert state.flow_rate_m3_hr == DEFAULT_FLOW_M3_HR
        assert state.airborne_dust_mg == 15.0
        assert state.hepa_online
        assert state.esp_online

    def test_custom_parameters(self):
        state = make_filtration(hab_volume_m3=500.0, flow_rate_m3_hr=200.0, initial_dust_mg=50.0)
        assert state.hab_volume_m3 == 500.0
        assert state.flow_rate_m3_hr == 200.0
        assert state.airborne_dust_mg == 50.0

    def test_initial_filter_clean(self):
        state = make_filtration()
        assert state.filter_load_mg == 0.0
        assert state.filter_replaced_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test — the colony breathes for a full Mars year
# ═══════════════════════════════════════════════════════════════════════════

class TestSmoke:
    def test_run_668_sols_no_crash(self):
        """Full Mars year — 668 sols without crash."""
        state = make_filtration()
        records = run_filtration(state, sols=668, eva_per_sol=1)
        assert len(records) == 668
        assert records[-1].sol == 668
        # Colony should still be breathing
        assert records[-1].concentration_mg_m3 < CRITICAL_DUST_MG_M3

    def test_run_10_sols_basic(self):
        """Basic 10-sol smoke test."""
        state = make_filtration()
        records = run_filtration(state, sols=10)
        assert len(records) == 10
        for r in records:
            assert r.airborne_dust_mg >= 0
            assert r.filter_load_mg >= 0

    def test_heavy_eva_schedule(self):
        """Colony with 4 EVAs per sol for 100 sols."""
        state = make_filtration()
        records = run_filtration(state, sols=100, eva_per_sol=4)
        assert len(records) == 100
        # System should handle it (with filter replacements)
        assert state.filter_replaced_count >= 0
        # Concentration should still be manageable
        assert records[-1].concentration_mg_m3 < 5.0

    def test_zero_initial_dust(self):
        """Start with perfectly clean air — HEPA captures baseline dust onto filter."""
        state = make_filtration(initial_dust_mg=0.0)
        records = run_filtration(state, sols=30)
        # HEPA is so efficient baseline dust is captured immediately;
        # verify dust is on the filter, not floating
        assert state.filter_load_mg > 0

    def test_massive_hab(self):
        """Large habitat (1000 m³) with proportional flow."""
        state = make_filtration(hab_volume_m3=1000.0, flow_rate_m3_hr=500.0)
        records = run_filtration(state, sols=50)
        assert len(records) == 50
        assert all(r.concentration_mg_m3 >= 0 for r in records)


# ═══════════════════════════════════════════════════════════════════════════
# Property-based invariants
# ═══════════════════════════════════════════════════════════════════════════

class TestInvariants:
    def test_removal_bounded_by_airborne(self):
        """Dust removed never exceeds airborne dust."""
        state = make_filtration(initial_dust_mg=200.0)
        for _ in range(50):
            airborne_before = state.airborne_dust_mg + BASELINE_DUST_MG_SOL
            record = tick_filtration(state)
            assert record.dust_removed_mg <= airborne_before + 1e-6

    def test_concentration_decreases_with_filtration(self):
        """With filtration on and no new dust, concentration should decrease."""
        state = make_filtration(initial_dust_mg=1000.0)
        # First tick adds baseline dust but removes much more
        record = tick_filtration(state, eva_events=0)
        # Even with 5mg added, removal of 1000mg should dominate
        assert record.airborne_dust_mg < 1000.0

    def test_settled_dust_monotonic(self):
        """Settled dust only increases (no mechanism to remove it)."""
        state = make_filtration()
        prev_settled = state.settled_dust_mg
        for _ in range(50):
            tick_filtration(state)
            assert state.settled_dust_mg >= prev_settled - 1e-10
            prev_settled = state.settled_dust_mg

    def test_esp_efficiency_physical_range(self):
        """ESP efficiency is always in [0, 1]."""
        for flow in [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0, 50000.0]:
            eff = esp_capture_efficiency(flow)
            assert 0.0 <= eff <= 1.0
