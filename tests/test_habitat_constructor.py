"""test_habitat_constructor.py -- 125 unit tests for Mars Habitat Module Assembly.

Tests cover:
  - Pure physics functions (hoop stress, safety factor, shielding, leak rate)
  - Construction phase progression
  - Material consumption and conservation laws
  - Environmental effects (dust, temperature, power)
  - Edge cases and boundary conditions
  - Multi-sol simulation smoke tests
  - Property-based invariants (physical bounds)
"""
from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import habitat_constructor as hc


# ===========================================================================
# 1. Hoop stress and structural safety
# ===========================================================================

class TestHoopStress:
    """Thin-wall pressure vessel mechanics."""

    def test_basic_hoop_stress(self):
        stress = hc.hoop_stress_mpa(70.0, 4.0, 0.006)
        assert abs(stress - 46.667) < 0.1

    def test_zero_pressure(self):
        assert hc.hoop_stress_mpa(0.0, 4.0, 0.006) == 0.0

    def test_zero_thickness(self):
        assert hc.hoop_stress_mpa(70.0, 4.0, 0.0) == 0.0

    def test_zero_radius(self):
        assert hc.hoop_stress_mpa(70.0, 0.0, 0.006) == 0.0

    def test_negative_thickness(self):
        assert hc.hoop_stress_mpa(70.0, 4.0, -0.001) == 0.0

    def test_stress_proportional_to_pressure(self):
        s1 = hc.hoop_stress_mpa(35.0, 4.0, 0.006)
        s2 = hc.hoop_stress_mpa(70.0, 4.0, 0.006)
        assert abs(s2 / s1 - 2.0) < 0.01

    def test_stress_proportional_to_radius(self):
        s1 = hc.hoop_stress_mpa(70.0, 2.0, 0.006)
        s2 = hc.hoop_stress_mpa(70.0, 4.0, 0.006)
        assert abs(s2 / s1 - 2.0) < 0.01

    def test_stress_inversely_proportional_to_thickness(self):
        s1 = hc.hoop_stress_mpa(70.0, 4.0, 0.006)
        s2 = hc.hoop_stress_mpa(70.0, 4.0, 0.012)
        assert abs(s1 / s2 - 2.0) < 0.01


class TestSafetyFactor:
    """Structural safety margins."""

    def test_basic_safety_factor(self):
        sf = hc.safety_factor(250.0, 46.667)
        assert abs(sf - 5.357) < 0.01

    def test_zero_stress_infinite(self):
        sf = hc.safety_factor(250.0, 0.0)
        assert sf == float("inf")

    def test_negative_stress_infinite(self):
        sf = hc.safety_factor(250.0, -10.0)
        assert sf == float("inf")

    def test_is_structurally_safe_nominal(self):
        assert hc.is_structurally_safe(
            hc.TARGET_PRESSURE_KPA, hc.HABITAT_RADIUS_M,
            hc.FRAME_WALL_THICKNESS_M)

    def test_unsafe_at_extreme_pressure(self):
        assert not hc.is_structurally_safe(
            1013.25, hc.HABITAT_RADIUS_M, hc.FRAME_WALL_THICKNESS_M)

    def test_safe_with_thick_walls(self):
        assert hc.is_structurally_safe(
            hc.TARGET_PRESSURE_KPA, hc.HABITAT_RADIUS_M, 0.1)

    def test_unsafe_with_thin_walls(self):
        assert not hc.is_structurally_safe(
            hc.TARGET_PRESSURE_KPA, hc.HABITAT_RADIUS_M, 0.0001)


# ===========================================================================
# 2. Radiation shielding
# ===========================================================================

class TestShielding:
    """Regolith radiation shielding effectiveness."""

    def test_gcr_zero_thickness(self):
        assert hc.shielding_gcr_reduction(0.0) == 0.0

    def test_gcr_negative_thickness(self):
        assert hc.shielding_gcr_reduction(-1.0) == 0.0

    def test_gcr_2m_meaningful(self):
        reduction = hc.shielding_gcr_reduction(2.0)
        assert 0.60 < reduction < 0.99

    def test_gcr_monotonically_increasing(self):
        prev = 0.0
        for thickness in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            val = hc.shielding_gcr_reduction(thickness)
            assert val > prev
            prev = val

    def test_gcr_never_exceeds_one(self):
        assert hc.shielding_gcr_reduction(100.0) <= 1.0

    def test_spe_zero_thickness(self):
        assert hc.shielding_spe_reduction(0.0) == 0.0

    def test_spe_negative_thickness(self):
        assert hc.shielding_spe_reduction(-1.0) == 0.0

    def test_spe_2m_high_effectiveness(self):
        reduction = hc.shielding_spe_reduction(2.0)
        assert reduction > 0.90

    def test_spe_more_effective_than_gcr(self):
        for thickness in [0.5, 1.0, 2.0]:
            spe = hc.shielding_spe_reduction(thickness)
            gcr = hc.shielding_gcr_reduction(thickness)
            assert spe >= gcr

    def test_spe_monotonically_increasing(self):
        prev = 0.0
        for thickness in [0.1, 0.5, 1.0, 2.0, 5.0]:
            val = hc.shielding_spe_reduction(thickness)
            assert val > prev
            prev = val

    def test_spe_never_exceeds_one(self):
        assert hc.shielding_spe_reduction(100.0) <= 1.0


# ===========================================================================
# 3. Leak rate model
# ===========================================================================

class TestLeakRate:
    """Pressure leak rate from seal quality and conditions."""

    def test_zero_seal_quality_total_leak(self):
        rate = hc.leak_rate_per_sol(0.0, 70.0, 210.0)
        assert rate == 1.0

    def test_no_pressure_diff_no_leak(self):
        rate = hc.leak_rate_per_sol(1.0, hc.MARS_AMBIENT_PRESSURE_KPA, 210.0)
        assert rate == 0.0

    def test_below_ambient_no_leak(self):
        rate = hc.leak_rate_per_sol(1.0, 0.3, 210.0)
        assert rate == 0.0

    def test_nominal_leak_bounded(self):
        rate = hc.leak_rate_per_sol(1.0, 70.0, 210.0)
        assert 0.0 < rate < 0.1

    def test_higher_pressure_more_leak(self):
        r1 = hc.leak_rate_per_sol(1.0, 50.0, 210.0)
        r2 = hc.leak_rate_per_sol(1.0, 100.0, 210.0)
        assert r2 > r1

    def test_better_seal_less_leak(self):
        r1 = hc.leak_rate_per_sol(0.5, 70.0, 210.0)
        r2 = hc.leak_rate_per_sol(1.0, 70.0, 210.0)
        assert r2 < r1

    def test_cold_increases_leak(self):
        r_warm = hc.leak_rate_per_sol(1.0, 70.0, 293.0)
        r_cold = hc.leak_rate_per_sol(1.0, 70.0, 150.0)
        assert r_cold > r_warm

    def test_leak_rate_never_negative(self):
        for sq in [0.01, 0.1, 0.5, 1.0]:
            for p in [0.0, 10.0, 70.0, 101.0]:
                for t in [100.0, 210.0, 293.0, 400.0]:
                    assert hc.leak_rate_per_sol(sq, p, t) >= 0.0

    def test_perfect_seal_achievable_leak_rate(self):
        """Bug regression: leak at seal_quality=1.0 must be <= ACCEPTABLE."""
        rate = hc.leak_rate_per_sol(1.0, hc.TARGET_PRESSURE_KPA,
                                     hc.MARS_AMBIENT_TEMP_K)
        assert rate <= hc.ACCEPTABLE_LEAK_RATE, (
            f"Bug: perfect seal leak {rate} > acceptable {hc.ACCEPTABLE_LEAK_RATE}")


# ===========================================================================
# 4. Construction helpers
# ===========================================================================

class TestConstructionHelpers:
    """Energy, materials, and dust penalty functions."""

    def test_energy_clamped_to_available(self):
        e = hc.construction_energy_kwh("foundation", 10.0)
        assert e == 10.0

    def test_energy_nominal(self):
        e = hc.construction_energy_kwh("foundation", 500.0)
        assert e == hc.ENERGY_FOUNDATION_KWH

    def test_energy_zero_available(self):
        e = hc.construction_energy_kwh("foundation", 0.0)
        assert e == 0.0

    def test_energy_negative_available(self):
        e = hc.construction_energy_kwh("foundation", -10.0)
        assert e == 0.0

    def test_energy_unknown_phase(self):
        e = hc.construction_energy_kwh("unknown", 100.0)
        assert e == 0.0

    def test_regolith_per_sol(self):
        vol = hc.regolith_per_sol_m3(16.0)
        assert abs(vol - 24.0) < 0.01

    def test_regolith_per_sol_zero(self):
        assert hc.regolith_per_sol_m3(0.0) == 0.0

    def test_regolith_per_sol_negative(self):
        assert hc.regolith_per_sol_m3(-5.0) == 0.0

    def test_iron_consumed_framing(self):
        iron = hc.iron_consumed_kg("framing")
        assert iron > 0.0
        assert iron == hc.FRAME_IRON_KG / hc.PHASE_FRAMING_SOLS

    def test_iron_consumed_other_phases(self):
        for phase in ["foundation", "inflation", "shielding",
                      "pressure_test", "outfitting"]:
            assert hc.iron_consumed_kg(phase) == 0.0

    def test_dust_seal_penalty_clear(self):
        assert hc.dust_seal_penalty(0.0) == 1.0

    def test_dust_seal_penalty_heavy(self):
        assert hc.dust_seal_penalty(0.5) == 0.5

    def test_dust_seal_penalty_storm(self):
        assert hc.dust_seal_penalty(1.0) == 0.5

    def test_dust_seal_penalty_moderate(self):
        p = hc.dust_seal_penalty(0.25)
        assert 0.5 < p < 1.0

    def test_dust_seal_penalty_negative(self):
        assert hc.dust_seal_penalty(-0.1) == 1.0


# ===========================================================================
# 5. Phase constants consistency
# ===========================================================================

class TestPhaseConstants:
    """Verify internal consistency of phase definitions."""

    def test_all_phases_have_durations(self):
        for phase in hc.PHASES[:-1]:
            assert phase in hc.PHASE_DURATIONS

    def test_all_phases_have_energy(self):
        for phase in hc.PHASES[:-1]:
            assert phase in hc.PHASE_ENERGY

    def test_total_min_sols_consistent(self):
        total = sum(hc.PHASE_DURATIONS.values())
        assert total == hc.TOTAL_MIN_SOLS

    def test_phases_ordered(self):
        expected = ["foundation", "framing", "inflation", "shielding",
                    "pressure_test", "outfitting", "complete"]
        assert hc.PHASES == expected

    def test_habitat_volume_positive(self):
        assert hc.HABITAT_VOLUME_M3 > 0

    def test_frame_iron_positive(self):
        assert hc.FRAME_IRON_KG > 0

    def test_regolith_total_positive(self):
        assert hc.REGOLITH_TOTAL_M3 > 0
        assert hc.REGOLITH_TOTAL_KG > 0

    def test_structural_safety_at_target_pressure(self):
        assert hc.is_structurally_safe(
            hc.TARGET_PRESSURE_KPA, hc.HABITAT_RADIUS_M,
            hc.FRAME_WALL_THICKNESS_M)


# ===========================================================================
# 6. HabitatConstructor dataclass
# ===========================================================================

class TestHabitatConstructor:
    """State serialization and defaults."""

    def test_default_state(self):
        s = hc.HabitatConstructor()
        assert s.phase == "foundation"
        assert s.phase_sol == 0
        assert s.iron_used_kg == 0.0
        assert s.integrity == 1.0
        assert s.modules_completed == 0

    def test_to_dict(self):
        s = hc.HabitatConstructor()
        d = s.to_dict()
        assert isinstance(d, dict)
        assert d["phase"] == "foundation"
        assert d["integrity"] == 1.0

    def test_from_dict_roundtrip(self):
        s = hc.HabitatConstructor(phase="shielding", phase_sol=3,
                                   iron_used_kg=100.5, integrity=0.98)
        d = s.to_dict()
        s2 = hc.HabitatConstructor.from_dict(d)
        assert s2.phase == "shielding"
        assert s2.phase_sol == 3
        assert abs(s2.iron_used_kg - 100.5) < 0.01

    def test_from_dict_empty(self):
        s = hc.HabitatConstructor.from_dict({})
        assert s.phase == "foundation"
        assert s.integrity == 1.0

    def test_from_dict_partial(self):
        s = hc.HabitatConstructor.from_dict({"phase": "outfitting"})
        assert s.phase == "outfitting"
        assert s.integrity == 1.0


# ===========================================================================
# 7. Tick engine -- phase progression
# ===========================================================================

class TestTickPhaseProgression:
    """Verify construction advances through phases."""

    def test_foundation_phase(self):
        s = hc.HabitatConstructor()
        result = hc.tick(s)
        assert s.phase_sol == 1
        assert result["phase"] == "foundation"

    def test_foundation_advances_after_5_sols(self):
        s = hc.HabitatConstructor()
        for _ in range(5):
            hc.tick(s)
        assert s.phase == "framing"
        assert s.phase_sol == 0

    def test_framing_uses_iron(self):
        s = hc.HabitatConstructor(phase="framing", phase_sol=0)
        hc.tick(s, iron_available_kg=5000.0)
        assert s.iron_used_kg > 0.0

    def test_framing_advances_after_3_sols(self):
        s = hc.HabitatConstructor(phase="framing", phase_sol=0)
        for _ in range(3):
            hc.tick(s, iron_available_kg=5000.0)
        assert s.phase == "inflation"

    def test_inflation_sets_pressure(self):
        s = hc.HabitatConstructor(phase="inflation", phase_sol=0)
        hc.tick(s)
        assert s.bladder_installed
        assert abs(s.pressure_kpa - hc.TARGET_PRESSURE_KPA) < 0.01

    def test_inflation_advances_after_1_sol(self):
        s = hc.HabitatConstructor(phase="inflation", phase_sol=0)
        hc.tick(s)
        assert s.phase == "shielding"

    def test_shielding_places_regolith(self):
        s = hc.HabitatConstructor(phase="shielding", phase_sol=0)
        hc.tick(s, regolith_available_m3=10000.0)
        assert s.regolith_placed_m3 > 0.0
        assert s.shielding_m > 0.0

    def test_pressure_test_builds_seal(self):
        s = hc.HabitatConstructor(
            phase="pressure_test", phase_sol=0,
            pressure_kpa=70.0, bladder_installed=True)
        hc.tick(s)
        assert s.seal_quality > 0.0
        assert s.sealant_used_kg > 0.0

    def test_outfitting_continues_seal_curing(self):
        s = hc.HabitatConstructor(
            phase="outfitting", phase_sol=0,
            seal_quality=0.5, pressure_kpa=70.0)
        hc.tick(s)
        assert s.seal_quality > 0.5

    def test_complete_phase_increments_modules(self):
        s = hc.HabitatConstructor()
        for _ in range(200):
            hc.tick(s, regolith_available_m3=100000.0)
            if s.phase == "complete":
                break
        assert s.modules_completed == 1


class TestTickEnvironmentalEffects:
    """Dust, temperature, power constraints."""

    def test_dust_storm_halts_foundation(self):
        s = hc.HabitatConstructor(phase="foundation", phase_sol=0)
        result = hc.tick(s, optical_depth=0.8)
        assert result["action"] == "dust_halt"

    def test_dust_storm_halts_shielding(self):
        s = hc.HabitatConstructor(phase="shielding", phase_sol=0)
        result = hc.tick(s, optical_depth=0.9)
        assert result["action"] == "dust_halt"

    def test_dust_storm_does_not_halt_framing(self):
        s = hc.HabitatConstructor(phase="framing", phase_sol=0)
        result = hc.tick(s, optical_depth=0.9, iron_available_kg=5000.0)
        assert result["action"] != "dust_halt"

    def test_low_power_reduces_energy(self):
        s1 = hc.HabitatConstructor(phase="foundation", phase_sol=0)
        s2 = hc.HabitatConstructor(phase="foundation", phase_sol=0)
        r1 = hc.tick(s1, available_power_kwh=200.0)
        r2 = hc.tick(s2, available_power_kwh=10.0)
        assert r1["energy_kwh"] >= r2["energy_kwh"]

    def test_no_iron_limits_framing(self):
        s = hc.HabitatConstructor(phase="framing", phase_sol=0)
        result = hc.tick(s, iron_available_kg=0.0)
        assert result["iron_kg"] == 0.0

    def test_no_regolith_limits_shielding(self):
        s = hc.HabitatConstructor(phase="shielding", phase_sol=0)
        result = hc.tick(s, regolith_available_m3=0.0)
        assert result["regolith_m3"] == 0.0

    def test_dust_reduces_seal_quality_gain(self):
        s1 = hc.HabitatConstructor(
            phase="pressure_test", phase_sol=0,
            pressure_kpa=70.0, seal_quality=0.0)
        s2 = hc.HabitatConstructor(
            phase="pressure_test", phase_sol=0,
            pressure_kpa=70.0, seal_quality=0.0)
        hc.tick(s1, optical_depth=0.0)
        hc.tick(s2, optical_depth=1.0)
        assert s1.seal_quality > s2.seal_quality


class TestTickConservationLaws:
    """Physical invariants that must hold across all ticks."""

    def test_integrity_never_exceeds_one(self):
        s = hc.HabitatConstructor()
        for _ in range(100):
            hc.tick(s, regolith_available_m3=100000.0)
        assert s.integrity <= 1.0

    def test_integrity_never_negative(self):
        s = hc.HabitatConstructor()
        for _ in range(10000):
            hc.tick(s, regolith_available_m3=100000.0)
            if s.integrity <= 0:
                break
        assert s.integrity >= 0.0

    def test_seal_quality_bounded_0_1(self):
        s = hc.HabitatConstructor()
        for _ in range(200):
            hc.tick(s, regolith_available_m3=100000.0)
        assert 0.0 <= s.seal_quality <= 1.0

    def test_pressure_bounded(self):
        s = hc.HabitatConstructor()
        for _ in range(200):
            hc.tick(s, regolith_available_m3=100000.0)
        assert 0.0 <= s.pressure_kpa <= hc.MAX_PRESSURE_KPA

    def test_shielding_bounded(self):
        s = hc.HabitatConstructor()
        for _ in range(200):
            hc.tick(s, regolith_available_m3=100000.0)
        assert 0.0 <= s.shielding_m <= hc.SHIELDING_TARGET_M

    def test_iron_consumption_positive(self):
        s = hc.HabitatConstructor()
        for _ in range(200):
            hc.tick(s, regolith_available_m3=100000.0)
        assert s.iron_used_kg >= 0.0

    def test_regolith_placement_positive(self):
        s = hc.HabitatConstructor()
        for _ in range(200):
            hc.tick(s, regolith_available_m3=100000.0)
        assert s.regolith_placed_m3 >= 0.0

    def test_total_energy_monotonically_increases(self):
        s = hc.HabitatConstructor()
        prev_energy = 0.0
        for _ in range(50):
            hc.tick(s, regolith_available_m3=100000.0)
            assert s.total_energy_kwh >= prev_energy
            prev_energy = s.total_energy_kwh

    def test_total_sols_monotonically_increases(self):
        s = hc.HabitatConstructor()
        for i in range(50):
            hc.tick(s, regolith_available_m3=100000.0)
            assert s.total_sols == i + 1

    def test_eva_hours_non_negative(self):
        s = hc.HabitatConstructor()
        for _ in range(100):
            result = hc.tick(s, regolith_available_m3=100000.0)
            assert result["eva_hours"] >= 0.0

    def test_leak_rate_non_negative(self):
        s = hc.HabitatConstructor()
        for _ in range(200):
            hc.tick(s, regolith_available_m3=100000.0)
        assert s.leak_rate >= 0.0

    def test_phase_never_goes_backward(self):
        s = hc.HabitatConstructor()
        highest_idx = 0
        for _ in range(200):
            hc.tick(s, regolith_available_m3=100000.0)
            idx = hc.PHASES.index(s.phase)
            assert idx >= highest_idx
            highest_idx = idx


# ===========================================================================
# 8. Complete module -- post-construction
# ===========================================================================

class TestCompleteModule:
    """Behavior after construction finishes."""

    def test_complete_module_maintenance(self):
        s = hc.HabitatConstructor(
            phase="complete", integrity=1.0, seal_quality=0.9,
            pressure_kpa=70.0, modules_completed=1)
        result = hc.tick(s)
        assert result["action"] == "maintenance"
        assert result["energy_kwh"] == 0.0

    def test_complete_module_integrity_degrades(self):
        s = hc.HabitatConstructor(
            phase="complete", integrity=1.0, seal_quality=0.9,
            pressure_kpa=70.0, modules_completed=1)
        initial = s.integrity
        hc.tick(s)
        assert s.integrity < initial

    def test_complete_module_tracks_leak(self):
        s = hc.HabitatConstructor(
            phase="complete", integrity=1.0, seal_quality=0.9,
            pressure_kpa=70.0, modules_completed=1)
        hc.tick(s)
        assert s.leak_rate >= 0.0


# ===========================================================================
# 9. Simulation runner
# ===========================================================================

class TestSimulation:
    """Full simulation runs."""

    def test_smoke_10_sols(self):
        history = hc.run_simulation(sols=10)
        assert len(history) >= 1
        assert history[0]["sol"] == 0

    def test_simulation_completes(self):
        history = hc.run_simulation(sols=200,
                                     regolith_available_m3=100000.0)
        final = history[-1]["state"]
        assert final["phase"] == "complete"

    def test_simulation_history_has_state(self):
        history = hc.run_simulation(sols=5)
        for entry in history:
            assert "state" in entry
            assert "phase" in entry["state"]

    def test_simulation_sols_sequential(self):
        history = hc.run_simulation(sols=10)
        for i, entry in enumerate(history):
            assert entry["sol"] == i

    def test_simulation_with_dust_storm(self):
        history = hc.run_simulation(sols=50, optical_depth=0.8)
        halts = [h for h in history if h.get("action") == "dust_halt"]
        assert len(halts) > 0

    def test_simulation_low_power(self):
        history = hc.run_simulation(sols=50, available_power_kwh=5.0)
        assert len(history) >= 1

    def test_simulation_zero_resources(self):
        history = hc.run_simulation(
            sols=50, iron_available_kg=0.0, regolith_available_m3=0.0)
        assert len(history) >= 1

    def test_simulation_cold_temperature(self):
        history = hc.run_simulation(sols=50, temperature_k=150.0)
        assert len(history) >= 1


# ===========================================================================
# 10. Property-based invariants (physical bounds)
# ===========================================================================

class TestPhysicalBounds:
    """Outputs must stay within physically meaningful ranges."""

    def test_habitat_volume_realistic(self):
        assert 100 < hc.HABITAT_VOLUME_M3 < 1000

    def test_frame_mass_realistic(self):
        assert 50 < hc.FRAME_IRON_KG < 5000

    def test_regolith_total_realistic(self):
        assert 100 < hc.REGOLITH_TOTAL_M3 < 5000

    def test_construction_time_realistic(self):
        assert 20 < hc.TOTAL_MIN_SOLS < 100

    def test_energy_per_sol_realistic(self):
        for phase in hc.PHASE_ENERGY:
            assert 1.0 <= hc.PHASE_ENERGY[phase] <= 200.0

    def test_target_pressure_earth_like(self):
        assert 50.0 <= hc.TARGET_PRESSURE_KPA <= 101.325

    def test_bladder_stronger_than_target(self):
        assert hc.BLADDER_BURST_PRESSURE_KPA > hc.TARGET_PRESSURE_KPA * 2

    def test_max_eva_hours_humane(self):
        assert hc.MAX_EVA_HOURS_PER_SOL <= 12.0

    def test_acceptable_leak_rate_tight(self):
        assert hc.ACCEPTABLE_LEAK_RATE < 0.01

    def test_shielding_effectiveness_at_target(self):
        gcr = hc.shielding_gcr_reduction(hc.SHIELDING_TARGET_M)
        spe = hc.shielding_spe_reduction(hc.SHIELDING_TARGET_M)
        assert gcr > 0.5
        assert spe > 0.9


# ===========================================================================
# 11. Edge cases
# ===========================================================================

class TestEdgeCases:
    """Unusual inputs and boundary conditions."""

    def test_tick_already_complete(self):
        s = hc.HabitatConstructor(
            phase="complete", modules_completed=1,
            seal_quality=0.9, pressure_kpa=70.0)
        result = hc.tick(s)
        assert result["phase"] == "complete"
        assert s.modules_completed == 1

    def test_shielding_exact_target(self):
        s = hc.HabitatConstructor(
            phase="shielding", phase_sol=9,
            regolith_placed_m3=hc.REGOLITH_TOTAL_M3)
        s.shielding_m = hc.SHIELDING_TARGET_M
        hc.tick(s, regolith_available_m3=100000.0)
        assert s.phase in ("shielding", "pressure_test")

    def test_pressure_test_poor_seal_stays(self):
        s = hc.HabitatConstructor(
            phase="pressure_test", phase_sol=0,
            pressure_kpa=70.0, seal_quality=0.01)
        hc.tick(s)
        assert s.phase == "pressure_test" or s.seal_quality > 0.01

    def test_very_high_optical_depth(self):
        s = hc.HabitatConstructor(phase="foundation")
        result = hc.tick(s, optical_depth=5.0)
        assert result["action"] == "dust_halt"

    def test_extreme_cold(self):
        s = hc.HabitatConstructor(
            phase="pressure_test", phase_sol=0,
            pressure_kpa=70.0, seal_quality=0.5)
        hc.tick(s, temperature_k=100.0)
        assert s.leak_rate >= 0.0

    def test_extreme_heat(self):
        s = hc.HabitatConstructor(
            phase="pressure_test", phase_sol=0,
            pressure_kpa=70.0, seal_quality=0.5)
        hc.tick(s, temperature_k=400.0)
        assert s.leak_rate >= 0.0

    def test_zero_sols_simulation(self):
        history = hc.run_simulation(sols=0)
        assert len(history) == 0

    def test_one_sol_simulation(self):
        history = hc.run_simulation(sols=1)
        assert len(history) == 1

    def test_massive_resources(self):
        history = hc.run_simulation(
            sols=200, available_power_kwh=10000.0,
            iron_available_kg=1e6, regolith_available_m3=1e6)
        final = history[-1]["state"]
        assert final["phase"] == "complete"

    def test_dust_halt_no_phase_sol_increment(self):
        """Dust halt should NOT advance phase_sol counter."""
        s = hc.HabitatConstructor(phase="foundation", phase_sol=2)
        hc.tick(s, optical_depth=0.9)
        assert s.phase_sol == 2


# ===========================================================================
# 12. Integration: full construction lifecycle
# ===========================================================================

class TestFullLifecycle:
    """End-to-end construction of a complete habitat module."""

    def test_full_build_tracks_all_resources(self):
        s = hc.HabitatConstructor()
        for _ in range(200):
            hc.tick(s, regolith_available_m3=100000.0)
            if s.phase == "complete":
                break

        assert s.phase == "complete"
        assert s.modules_completed == 1
        assert s.iron_used_kg > 0.0
        assert s.regolith_placed_m3 > 0.0
        assert s.total_energy_kwh > 0.0
        assert s.total_eva_hours > 0.0
        assert s.total_sols >= hc.TOTAL_MIN_SOLS
        assert s.bladder_installed
        assert s.sealant_used_kg > 0.0
        assert s.shielding_m > 0.0
        assert s.seal_quality > 0.0
        assert s.pressure_kpa > 0.0

    def test_full_build_all_phases_visited(self):
        s = hc.HabitatConstructor()
        visited = set()
        for _ in range(200):
            visited.add(s.phase)
            hc.tick(s, regolith_available_m3=100000.0)
            if s.phase == "complete":
                visited.add("complete")
                break
        assert visited == set(hc.PHASES)

    def test_integrity_stays_high_during_build(self):
        s = hc.HabitatConstructor()
        for _ in range(200):
            hc.tick(s, regolith_available_m3=100000.0)
            if s.phase == "complete":
                break
        assert s.integrity > 0.95
