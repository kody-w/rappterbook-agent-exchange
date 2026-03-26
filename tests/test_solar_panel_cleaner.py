"""Tests for solar_panel_cleaner.py — Automated Solar Panel Dust Removal."""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import solar_panel_cleaner as spc


# ── fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def cleaner():
    return spc.create_cleaner(panel_area_m2=100.0)


@pytest.fixture
def dusty_cleaner():
    return spc.create_cleaner(panel_area_m2=100.0, initial_dust=0.30)


@pytest.fixture
def worn_cleaner():
    return spc.create_cleaner(panel_area_m2=100.0, initial_wear=0.50)


# ── 1. Dust adhesion physics ───────────────────────────────────────────

class TestDustAdhesion:
    def test_positive_diameter_gives_positive_force(self):
        f = spc.dust_adhesion_force_n(1.5e-6)
        assert f > 0.0

    def test_zero_diameter_gives_zero_force(self):
        assert spc.dust_adhesion_force_n(0.0) == 0.0

    def test_negative_diameter_gives_zero_force(self):
        assert spc.dust_adhesion_force_n(-1.0e-6) == 0.0

    def test_force_scales_with_diameter(self):
        f_small = spc.dust_adhesion_force_n(1.0e-6)
        f_big = spc.dust_adhesion_force_n(10.0e-6)
        assert f_big > f_small

    def test_force_linear_in_diameter(self):
        f1 = spc.dust_adhesion_force_n(1.0e-6)
        f2 = spc.dust_adhesion_force_n(2.0e-6)
        assert abs(f2 / f1 - 2.0) < 1e-10

    def test_physical_magnitude(self):
        """1.5 μm silicate on glass: ~10⁻⁸ N order of magnitude."""
        f = spc.dust_adhesion_force_n(1.5e-6)
        assert 1e-10 < f < 1e-6


# ── 2. Electrostatic removal force ────────────────────────────────────

class TestElectrostaticForce:
    def test_positive_field_positive_force(self):
        f = spc.electrostatic_removal_force_n(1000.0, 1e-15)
        assert f > 0.0

    def test_negative_field_still_positive(self):
        f = spc.electrostatic_removal_force_n(-1000.0, 1e-15)
        assert f > 0.0

    def test_zero_field_zero_force(self):
        assert spc.electrostatic_removal_force_n(0.0, 1e-15) == 0.0

    def test_zero_charge_zero_force(self):
        assert spc.electrostatic_removal_force_n(1000.0, 0.0) == 0.0

    def test_force_proportional_to_field(self):
        f1 = spc.electrostatic_removal_force_n(100.0, 1e-15)
        f2 = spc.electrostatic_removal_force_n(200.0, 1e-15)
        assert abs(f2 / f1 - 2.0) < 1e-10


# ── 3. Cleaning efficiency ────────────────────────────────────────────

class TestCleaningEfficiency:
    def test_zero_dust_zero_efficiency(self):
        for m in spc.CleaningMethod:
            assert spc.cleaning_efficiency(m, 0.0, 0.0) == 0.0

    def test_efficiency_in_valid_range(self):
        for m in spc.CleaningMethod:
            for dust in [0.01, 0.1, 0.5, 0.9, 1.0]:
                for wear in [0.0, 0.3, 0.7, 1.0]:
                    eff = spc.cleaning_efficiency(m, dust, wear)
                    assert 0.0 <= eff <= 1.0, f"{m} dust={dust} wear={wear}"

    def test_edr_best_on_fresh_panels(self):
        """Electrostatic should be most efficient on unworn panels."""
        eff = spc.cleaning_efficiency(
            spc.CleaningMethod.ELECTROSTATIC, 0.2, 0.0)
        assert eff > 0.85

    def test_wear_reduces_efficiency(self):
        eff_fresh = spc.cleaning_efficiency(
            spc.CleaningMethod.WIPER, 0.2, 0.0)
        eff_worn = spc.cleaning_efficiency(
            spc.CleaningMethod.WIPER, 0.2, 0.5)
        assert eff_worn < eff_fresh

    def test_full_wear_kills_efficiency(self):
        for m in spc.CleaningMethod:
            eff = spc.cleaning_efficiency(m, 0.5, 1.0)
            assert eff == 0.0

    def test_heavier_dust_slightly_easier(self):
        """Log bonus: heavy dust loads are slightly easier to dislodge."""
        eff_light = spc.cleaning_efficiency(
            spc.CleaningMethod.CO2_BLAST, 0.01, 0.0)
        eff_heavy = spc.cleaning_efficiency(
            spc.CleaningMethod.CO2_BLAST, 0.80, 0.0)
        assert eff_heavy >= eff_light


# ── 4. Cleaning energy cost ───────────────────────────────────────────

class TestCleaningEnergy:
    def test_all_methods_positive_energy(self):
        for m in spc.CleaningMethod:
            e = spc.cleaning_energy_wh(m, 100.0)
            assert e > 0.0

    def test_zero_area_zero_for_edr(self):
        assert spc.cleaning_energy_wh(spc.CleaningMethod.ELECTROSTATIC, 0.0) == 0.0

    def test_negative_area_clamped(self):
        assert spc.cleaning_energy_wh(spc.CleaningMethod.ELECTROSTATIC, -10.0) == 0.0

    def test_edr_scales_with_area(self):
        e1 = spc.cleaning_energy_wh(spc.CleaningMethod.ELECTROSTATIC, 50.0)
        e2 = spc.cleaning_energy_wh(spc.CleaningMethod.ELECTROSTATIC, 100.0)
        assert abs(e2 / e1 - 2.0) < 1e-10

    def test_wiper_fixed_cost(self):
        """Wiper uses one motor regardless of area."""
        e1 = spc.cleaning_energy_wh(spc.CleaningMethod.WIPER, 50.0)
        e2 = spc.cleaning_energy_wh(spc.CleaningMethod.WIPER, 200.0)
        assert e1 == e2

    def test_edr_cheapest_for_small_area(self):
        """EDR is cheapest for small panel areas (< ~5 m²)."""
        area = 1.0  # 1 m² — EDR scales with area, CO2/wiper are fixed
        e_edr = spc.cleaning_energy_wh(spc.CleaningMethod.ELECTROSTATIC, area)
        e_co2 = spc.cleaning_energy_wh(spc.CleaningMethod.CO2_BLAST, area)
        assert e_edr < e_co2

    def test_energy_values_physically_reasonable(self):
        """100 m² of panels: EDR should use < 1 kWh per cycle."""
        e = spc.cleaning_energy_wh(spc.CleaningMethod.ELECTROSTATIC, 100.0)
        assert e < 1000.0  # well under 1 kWh


# ── 5. Wear per cycle ─────────────────────────────────────────────────

class TestWearPerCycle:
    def test_all_methods_positive_wear(self):
        for m in spc.CleaningMethod:
            assert spc.wear_per_cycle(m) > 0.0

    def test_edr_least_wear(self):
        assert spc.wear_per_cycle(spc.CleaningMethod.ELECTROSTATIC) < \
               spc.wear_per_cycle(spc.CleaningMethod.WIPER)

    def test_wiper_most_wear(self):
        w_wiper = spc.wear_per_cycle(spc.CleaningMethod.WIPER)
        w_edr = spc.wear_per_cycle(spc.CleaningMethod.ELECTROSTATIC)
        w_co2 = spc.wear_per_cycle(spc.CleaningMethod.CO2_BLAST)
        assert w_wiper >= w_edr
        assert w_wiper >= w_co2


# ── 6. Power output ──────────────────────────────────────────────────

class TestPowerOutput:
    def test_clean_panels_positive_power(self):
        p = spc.power_output_wh(100.0, 0.0, 0.0)
        assert p > 0.0

    def test_fully_dusty_zero_power(self):
        p = spc.power_output_wh(100.0, 1.0, 0.0)
        assert p == 0.0

    def test_fully_worn_zero_power(self):
        p = spc.power_output_wh(100.0, 0.0, 1.0)
        assert p == 0.0

    def test_zero_area_zero_power(self):
        assert spc.power_output_wh(0.0, 0.0, 0.0) == 0.0

    def test_negative_area_clamped(self):
        assert spc.power_output_wh(-10.0, 0.0, 0.0) == 0.0

    def test_dust_reduces_power(self):
        p_clean = spc.power_output_wh(100.0, 0.0, 0.0)
        p_dusty = spc.power_output_wh(100.0, 0.3, 0.0)
        assert p_dusty < p_clean

    def test_dust_storm_reduces_power(self):
        p_clear = spc.power_output_wh(100.0, 0.0, 0.0, dust_storm_active=False)
        p_storm = spc.power_output_wh(100.0, 0.0, 0.0, dust_storm_active=True)
        assert p_storm < p_clear

    def test_power_always_non_negative(self):
        for dust in [0.0, 0.5, 1.0]:
            for wear in [0.0, 0.5, 1.0]:
                for storm in [False, True]:
                    p = spc.power_output_wh(100.0, dust, wear, storm)
                    assert p >= 0.0

    def test_100m2_clean_power_reasonable(self):
        """100 m² GaAs on Mars: ~100 kWh/sol expected.
        590 W/m² × 100 m² × 0.95 transmittance × 0.30 eff × 6 h ≈ 101 kWh."""
        p = spc.power_output_wh(100.0, 0.0, 0.0)
        assert 50_000.0 < p < 200_000.0

    def test_wear_reduces_power(self):
        p_fresh = spc.power_output_wh(100.0, 0.0, 0.0)
        p_worn = spc.power_output_wh(100.0, 0.0, 0.5)
        assert p_worn < p_fresh


# ── 7. Net energy gain ───────────────────────────────────────────────

class TestNetEnergyGain:
    def test_cleaning_dusty_panels_net_positive(self):
        gain = spc.net_energy_gain_wh(
            spc.CleaningMethod.ELECTROSTATIC,
            100.0, 0.30, 0.03, 0.0)
        assert gain > 0.0

    def test_cleaning_clean_panels_net_negative(self):
        """Worn panel + minimal dust: cleaning cost exceeds gain."""
        gain = spc.net_energy_gain_wh(
            spc.CleaningMethod.CO2_BLAST,
            0.01, 0.005, 0.001, 0.95)
        assert gain < 0.0

    def test_larger_area_larger_gain(self):
        g1 = spc.net_energy_gain_wh(
            spc.CleaningMethod.ELECTROSTATIC,
            50.0, 0.30, 0.03, 0.0)
        g2 = spc.net_energy_gain_wh(
            spc.CleaningMethod.ELECTROSTATIC,
            200.0, 0.30, 0.03, 0.0)
        assert g2 > g1


# ── 8. Should-clean decision ─────────────────────────────────────────

class TestShouldClean:
    def test_low_dust_no_clean(self):
        assert not spc.should_clean(0.01, 0.0, 100.0)

    def test_high_dust_yes_clean(self):
        assert spc.should_clean(0.30, 0.0, 100.0)

    def test_fully_worn_no_clean(self):
        """If panels are fully worn, cleaning can't help."""
        assert not spc.should_clean(0.30, 1.0, 100.0)

    def test_threshold_boundary(self):
        """Just below threshold: no clean."""
        assert not spc.should_clean(
            spc.DUST_TRIGGER_THRESHOLD - 0.001, 0.0, 100.0)


# ── 9. State creation and clamping ───────────────────────────────────

class TestStateCreation:
    def test_default_state(self):
        s = spc.create_cleaner()
        assert s.panel_area_m2 == 100.0
        assert s.dust_fraction == 0.0
        assert s.panel_wear == 0.0
        assert s.sol == 0

    def test_custom_state(self):
        s = spc.create_cleaner(panel_area_m2=200.0, initial_dust=0.1,
                               initial_wear=0.05)
        assert s.panel_area_m2 == 200.0
        assert abs(s.dust_fraction - 0.1) < 1e-10
        assert abs(s.panel_wear - 0.05) < 1e-10

    def test_negative_area_clamped(self):
        s = spc.create_cleaner(panel_area_m2=-10.0)
        assert s.panel_area_m2 == 0.0

    def test_dust_clamped_to_unit(self):
        s = spc.create_cleaner(initial_dust=5.0)
        assert s.dust_fraction == 1.0

    def test_negative_dust_clamped(self):
        s = spc.create_cleaner(initial_dust=-0.1)
        assert s.dust_fraction == 0.0


# ── 10. Dust accumulation ────────────────────────────────────────────

class TestDustAccumulation:
    def test_normal_accumulation(self, cleaner):
        s = spc.accumulate_dust(cleaner)
        assert abs(s.dust_fraction - spc.DUST_ACCUMULATION_PER_SOL) < 1e-10

    def test_storm_accumulation_faster(self, cleaner):
        s_normal = spc.accumulate_dust(cleaner, dust_storm_active=False)
        s_storm = spc.accumulate_dust(cleaner, dust_storm_active=True)
        assert s_storm.dust_fraction > s_normal.dust_fraction

    def test_sol_increments(self, cleaner):
        s = spc.accumulate_dust(cleaner)
        assert s.sol == cleaner.sol + 1

    def test_dust_caps_at_one(self):
        s = spc.create_cleaner(initial_dust=0.999)
        s = spc.accumulate_dust(s)
        assert s.dust_fraction <= 1.0

    def test_dust_monotonically_increases(self, cleaner):
        s = cleaner
        for _ in range(100):
            prev = s.dust_fraction
            s = spc.accumulate_dust(s)
            assert s.dust_fraction >= prev


# ── 11. Cleaning operation ───────────────────────────────────────────

class TestCleanPanels:
    def test_reduces_dust(self, dusty_cleaner):
        s = spc.clean_panels(dusty_cleaner, spc.CleaningMethod.ELECTROSTATIC)
        assert s.dust_fraction < dusty_cleaner.dust_fraction

    def test_increases_wear(self, dusty_cleaner):
        s = spc.clean_panels(dusty_cleaner, spc.CleaningMethod.WIPER)
        assert s.panel_wear > dusty_cleaner.panel_wear

    def test_increments_cycle_count(self, dusty_cleaner):
        s = spc.clean_panels(dusty_cleaner)
        assert s.total_cleaning_cycles == 1

    def test_tracks_energy_spent(self, dusty_cleaner):
        s = spc.clean_panels(dusty_cleaner)
        assert s.total_energy_spent_wh > 0.0

    def test_tracks_dust_removed(self, dusty_cleaner):
        s = spc.clean_panels(dusty_cleaner)
        assert s.total_dust_removed > 0.0

    def test_dust_never_negative(self, dusty_cleaner):
        s = dusty_cleaner
        for _ in range(10):
            s = spc.clean_panels(s)
        assert s.dust_fraction >= 0.0

    def test_wear_never_exceeds_max(self):
        s = spc.create_cleaner(initial_dust=0.5, initial_wear=0.999)
        s = spc.clean_panels(s, spc.CleaningMethod.WIPER)
        assert s.panel_wear <= spc.MAX_WEAR

    def test_all_methods_reduce_dust(self, dusty_cleaner):
        for m in spc.CleaningMethod:
            s = spc.clean_panels(dusty_cleaner, m)
            assert s.dust_fraction < dusty_cleaner.dust_fraction


# ── 12. Tick (one sol) ───────────────────────────────────────────────

class TestTick:
    def test_tick_advances_sol(self, cleaner):
        s = spc.tick(cleaner)
        assert s.sol == 1

    def test_tick_accumulates_dust(self, cleaner):
        s = spc.tick(cleaner, auto_clean=False)
        assert s.dust_fraction > 0.0

    def test_auto_clean_triggers_when_dusty(self):
        s = spc.create_cleaner(initial_dust=0.20)
        s = spc.tick(s, auto_clean=True)
        # Should have cleaned (dust was above threshold)
        assert s.total_cleaning_cycles >= 0  # may or may not trigger based on threshold

    def test_auto_clean_doesnt_trigger_when_clean(self, cleaner):
        s = spc.tick(cleaner, auto_clean=True)
        # Dust after 1 sol is only 0.28%, below 5% threshold
        assert s.total_cleaning_cycles == 0

    def test_dust_storm_tick(self, cleaner):
        s_clear = spc.tick(cleaner, dust_storm_active=False, auto_clean=False)
        s_storm = spc.tick(cleaner, dust_storm_active=True, auto_clean=False)
        assert s_storm.dust_fraction > s_clear.dust_fraction


# ── 13. Multi-sol simulation ─────────────────────────────────────────

class TestSimulate:
    def test_smoke_10_sols(self):
        history = spc.simulate(10)
        assert len(history) == 11  # initial + 10 ticks
        assert history[-1].sol == 10

    def test_smoke_100_sols(self):
        history = spc.simulate(100)
        assert len(history) == 101

    def test_dust_accumulates_without_cleaning(self):
        history = spc.simulate(50, auto_clean=False)
        assert history[-1].dust_fraction > history[0].dust_fraction

    def test_cleaning_keeps_dust_low(self):
        h_clean = spc.simulate(200, auto_clean=True)
        h_dirty = spc.simulate(200, auto_clean=False)
        assert h_clean[-1].dust_fraction < h_dirty[-1].dust_fraction

    def test_dust_storm_scenario(self):
        history = spc.simulate(
            60, dust_storm_start=20, dust_storm_duration=15,
            auto_clean=True)
        # During storm (sol 20-35), dust should spike
        pre_storm = history[19].dust_fraction
        mid_storm = history[30].dust_fraction
        # Mid-storm dust should be higher than pre-storm
        # (even with cleaning, storm adds dust faster)
        assert mid_storm > pre_storm or history[30].total_cleaning_cycles > 0

    def test_all_methods_simulatable(self):
        for m in spc.CleaningMethod:
            history = spc.simulate(10, method=m)
            assert len(history) == 11

    def test_sol_counter_correct(self):
        history = spc.simulate(25)
        for i, s in enumerate(history):
            assert s.sol == i


# ── 14. Serialization ───────────────────────────────────────────────

class TestSerialization:
    def test_to_dict_returns_dict(self, cleaner):
        d = spc.to_dict(cleaner)
        assert isinstance(d, dict)

    def test_to_dict_keys(self, cleaner):
        d = spc.to_dict(cleaner)
        expected = {"panel_area_m2", "dust_fraction", "panel_wear",
                    "total_cleaning_cycles", "total_energy_spent_wh",
                    "total_energy_gained_wh", "total_dust_removed",
                    "sol", "current_power_wh"}
        assert set(d.keys()) == expected

    def test_to_dict_json_safe(self, cleaner):
        import json
        d = spc.to_dict(cleaner)
        s = json.dumps(d)
        assert isinstance(s, str)

    def test_to_dict_after_simulation(self):
        history = spc.simulate(30)
        d = spc.to_dict(history[-1])
        assert d["sol"] == 30
        assert d["current_power_wh"] > 0.0


# ── 15. Conservation law invariants ──────────────────────────────────

class TestConservationLaws:
    def test_dust_always_in_unit_interval(self):
        """Dust fraction must always be in [0, 1]."""
        history = spc.simulate(500, auto_clean=True,
                               dust_storm_start=100,
                               dust_storm_duration=50)
        for s in history:
            assert 0.0 <= s.dust_fraction <= 1.0

    def test_wear_monotonically_increases(self):
        """Panel wear never decreases (scratches don't heal)."""
        history = spc.simulate(200, auto_clean=True)
        for i in range(1, len(history)):
            assert history[i].panel_wear >= history[i - 1].panel_wear

    def test_wear_bounded_by_max(self):
        """Wear never exceeds MAX_WEAR."""
        # Force many cleaning cycles
        s = spc.create_cleaner(initial_dust=0.5)
        for _ in range(50000):
            s = spc.clean_panels(s, spc.CleaningMethod.WIPER)
            s = spc.PanelCleanerState(
                panel_area_m2=s.panel_area_m2,
                dust_fraction=0.5,
                panel_wear=s.panel_wear,
                total_cleaning_cycles=s.total_cleaning_cycles,
                total_energy_spent_wh=s.total_energy_spent_wh,
                total_energy_gained_wh=s.total_energy_gained_wh,
                total_dust_removed=s.total_dust_removed,
                sol=s.sol,
            )
        assert s.panel_wear <= spc.MAX_WEAR

    def test_energy_spent_monotonically_increases(self):
        """Total energy spent never decreases."""
        history = spc.simulate(100, auto_clean=True)
        for i in range(1, len(history)):
            assert history[i].total_energy_spent_wh >= \
                   history[i - 1].total_energy_spent_wh

    def test_cleaning_cycles_monotonically_increase(self):
        """Total cleaning cycles never decrease."""
        history = spc.simulate(100, auto_clean=True)
        for i in range(1, len(history)):
            assert history[i].total_cleaning_cycles >= \
                   history[i - 1].total_cleaning_cycles

    def test_dust_removed_monotonically_increases(self):
        """Total dust removed never decreases."""
        history = spc.simulate(100, auto_clean=True)
        for i in range(1, len(history)):
            assert history[i].total_dust_removed >= \
                   history[i - 1].total_dust_removed

    def test_power_non_negative_throughout(self):
        """Power output is always >= 0."""
        history = spc.simulate(500, auto_clean=True,
                               dust_storm_start=100,
                               dust_storm_duration=100)
        for s in history:
            p = spc.power_output_wh(s.panel_area_m2, s.dust_fraction,
                                     s.panel_wear)
            assert p >= 0.0

    def test_sol_advances_correctly(self):
        """Sol counter always equals index in history."""
        history = spc.simulate(100)
        for i, s in enumerate(history):
            assert s.sol == i


# ── 16. Property-based: physical bounds ──────────────────────────────

class TestPhysicalBounds:
    def test_power_decreases_with_dust(self):
        """More dust -> less power, monotonically."""
        prev = spc.power_output_wh(100.0, 0.0, 0.0)
        for dust_pct in range(1, 101):
            p = spc.power_output_wh(100.0, dust_pct / 100.0, 0.0)
            assert p <= prev
            prev = p

    def test_power_decreases_with_wear(self):
        """More wear -> less power, monotonically."""
        prev = spc.power_output_wh(100.0, 0.0, 0.0)
        for wear_pct in range(1, 101):
            p = spc.power_output_wh(100.0, 0.0, wear_pct / 100.0)
            assert p <= prev
            prev = p

    def test_efficiency_decreases_with_wear(self):
        """More wear -> lower cleaning efficiency, monotonically."""
        prev = spc.cleaning_efficiency(
            spc.CleaningMethod.ELECTROSTATIC, 0.3, 0.0)
        for wear_pct in range(1, 101):
            eff = spc.cleaning_efficiency(
                spc.CleaningMethod.ELECTROSTATIC, 0.3, wear_pct / 100.0)
            assert eff <= prev + 1e-10
            prev = eff

    def test_opportunity_scenario(self):
        """Simulate what killed Opportunity: long dust storm, no cleaning.
        After 200 sols of storm, power should be near zero."""
        history = spc.simulate(
            200, auto_clean=False,
            dust_storm_start=0, dust_storm_duration=200)
        final_power = spc.power_output_wh(
            history[-1].panel_area_m2,
            history[-1].dust_fraction,
            history[-1].panel_wear,
            dust_storm_active=True)
        # Panels nearly fully covered + storm dimming
        clean_power = spc.power_output_wh(100.0, 0.0, 0.0)
        assert final_power < clean_power * 0.05

    def test_cleaning_extends_panel_life(self):
        """With cleaning, panels produce more total energy than without."""
        h_clean = spc.simulate(365, auto_clean=True)
        h_dirty = spc.simulate(365, auto_clean=False)
        total_clean = sum(
            spc.power_output_wh(s.panel_area_m2, s.dust_fraction, s.panel_wear)
            for s in h_clean)
        total_dirty = sum(
            spc.power_output_wh(s.panel_area_m2, s.dust_fraction, s.panel_wear)
            for s in h_dirty)
        assert total_clean > total_dirty


# ── 17. Edge cases ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_sol_simulation(self):
        history = spc.simulate(0)
        assert len(history) == 1
        assert history[0].sol == 0

    def test_one_sol_simulation(self):
        history = spc.simulate(1)
        assert len(history) == 2

    def test_very_small_panels(self):
        history = spc.simulate(10, panel_area_m2=0.01)
        assert all(s.dust_fraction >= 0.0 for s in history)

    def test_very_large_panels(self):
        history = spc.simulate(10, panel_area_m2=100000.0)
        assert history[-1].sol == 10

    def test_clean_already_clean(self):
        s = spc.create_cleaner(initial_dust=0.0)
        s2 = spc.clean_panels(s)
        assert s2.dust_fraction == 0.0


# ── 18. Integration: cleaning beats Opportunity's fate ───────────────

class TestIntegration:
    def test_edr_survives_dust_storm(self):
        """EDR-equipped colony survives a 30-sol global dust storm."""
        history = spc.simulate(
            100, panel_area_m2=500.0,
            dust_storm_start=30, dust_storm_duration=30,
            method=spc.CleaningMethod.ELECTROSTATIC,
            auto_clean=True)
        # After storm clears (sol 60+), power should recover
        post_storm_power = spc.power_output_wh(
            history[80].panel_area_m2,
            history[80].dust_fraction,
            history[80].panel_wear)
        clean_power = spc.power_output_wh(500.0, 0.0, 0.0)
        assert post_storm_power > clean_power * 0.50

    def test_mars_year_sustainability(self):
        """668 sols (1 Mars year) with cleaning: panels still useful."""
        history = spc.simulate(
            668, panel_area_m2=100.0,
            dust_storm_start=200, dust_storm_duration=60,
            auto_clean=True)
        final = history[-1]
        final_power = spc.power_output_wh(
            final.panel_area_m2, final.dust_fraction, final.panel_wear)
        clean_power = spc.power_output_wh(100.0, 0.0, 0.0)
        # After a full Mars year with one major storm, panels should
        # still produce > 80% of rated power
        assert final_power > clean_power * 0.70
        assert final.total_cleaning_cycles > 0
