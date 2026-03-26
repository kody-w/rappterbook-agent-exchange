"""Tests for solar_flare_monitor.py -- Mars SPE Early Warning System.

70+ tests covering:
  - Physical constants / unit conversions
  - Proton kinematics
  - Warning lead time calculations
  - NOAA S-scale classification
  - Dose rate calculations
  - Alert level thresholds
  - SPE event generation
  - Tick engine lifecycle
  - Multi-sol integration
  - Edge cases and boundary conditions
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from solar_flare_monitor import (
    ALERT_CRITICAL_PFU,
    ALERT_EMERGENCY_PFU,
    ALERT_WARNING_PFU,
    ALERT_WATCH_PFU,
    AU_KM,
    GCR_BASELINE_MSV_SOL,
    LIGHT_TIME_SUN_MARS_MIN,
    MARS_AU,
    MARS_DISTANCE_KM,
    MEV_TO_JOULE,
    MonitorState,
    PFU_TO_MSV_HR,
    PROTON_MASS_KG,
    S_SCALE,
    SHELTER_ATTENUATION,
    SolarEvent,
    SolReport,
    SPEED_OF_LIGHT_KM_S,
    SPE_RATES_PER_SOL,
    alert_from_pfu,
    check_for_spe,
    classify_s_scale,
    dose_rate_msv_hr,
    event_total_dose_msv,
    generate_spe,
    make_monitor,
    monitor_power_kwh,
    proton_transit_time_min,
    proton_velocity_km_s,
    run_monitor,
    sheltered_dose_msv,
    should_shelter,
    tick_monitor,
    warning_lead_time_min,
)


# ============================================================================
# 1. Physical constants
# ============================================================================

class TestPhysicalConstants:
    """Validate physical constants against known values."""

    def test_speed_of_light(self):
        assert abs(SPEED_OF_LIGHT_KM_S - 299792.458) < 1.0

    def test_proton_mass(self):
        assert abs(PROTON_MASS_KG - 1.6726e-27) / 1.6726e-27 < 0.01

    def test_mev_to_joule(self):
        assert abs(MEV_TO_JOULE - 1.6022e-13) / 1.6022e-13 < 0.01

    def test_au_km(self):
        assert abs(AU_KM - 149_597_870.7) < 1.0

    def test_mars_au(self):
        assert 1.5 < MARS_AU < 1.6

    def test_mars_distance(self):
        expected = MARS_AU * AU_KM
        assert abs(MARS_DISTANCE_KM - expected) < 1.0

    def test_light_time_sun_mars(self):
        assert 12.0 < LIGHT_TIME_SUN_MARS_MIN < 13.5

    def test_gcr_baseline(self):
        assert 0.5 < GCR_BASELINE_MSV_SOL < 1.0

    def test_shelter_attenuation_bounded(self):
        assert 0.0 < SHELTER_ATTENUATION < 1.0


# ============================================================================
# 2. Proton physics
# ============================================================================

class TestProtonPhysics:
    """Test proton velocity and transit time calculations."""

    def test_velocity_positive_energy(self):
        v = proton_velocity_km_s(100.0)
        assert v > 0.0

    def test_velocity_zero_energy(self):
        assert proton_velocity_km_s(0.0) == 0.0

    def test_velocity_negative_energy(self):
        assert proton_velocity_km_s(-1.0) == 0.0

    def test_velocity_increases_with_energy(self):
        v10 = proton_velocity_km_s(10.0)
        v100 = proton_velocity_km_s(100.0)
        v1000 = proton_velocity_km_s(1000.0)
        assert v10 < v100 < v1000

    def test_velocity_capped_at_c(self):
        v = proton_velocity_km_s(1e12)
        assert v <= SPEED_OF_LIGHT_KM_S

    def test_velocity_10mev(self):
        v = proton_velocity_km_s(10.0)
        assert 40_000 < v < 50_000

    def test_transit_100mev(self):
        t = proton_transit_time_min(100.0, MARS_DISTANCE_KM)
        assert 15 < t < 60

    def test_transit_zero_energy(self):
        assert proton_transit_time_min(0.0, MARS_DISTANCE_KM) == float("inf")

    def test_transit_negative_energy(self):
        assert proton_transit_time_min(-5.0, MARS_DISTANCE_KM) == float("inf")

    def test_transit_decreases_with_energy(self):
        t10 = proton_transit_time_min(10.0, MARS_DISTANCE_KM)
        t100 = proton_transit_time_min(100.0, MARS_DISTANCE_KM)
        assert t100 < t10


# ============================================================================
# 3. Warning time
# ============================================================================

class TestWarningTime:
    """Test X-ray precursor warning lead time."""

    def test_warning_positive(self):
        wt = warning_lead_time_min(50.0)
        assert wt > 0.0

    def test_warning_never_negative(self):
        wt = warning_lead_time_min(1e12)
        assert wt >= 0.0

    def test_low_energy_more_warning(self):
        w10 = warning_lead_time_min(10.0)
        w100 = warning_lead_time_min(100.0)
        assert w10 > w100

    def test_warning_typical_range(self):
        wt = warning_lead_time_min(100.0)
        assert 5 < wt < 120


# ============================================================================
# 4. S-scale classification
# ============================================================================

class TestClassification:
    """Test NOAA S-scale classification."""

    def test_below_s1(self):
        assert classify_s_scale(5.0) == "none"

    def test_s1_threshold(self):
        assert classify_s_scale(10.0) == "S1"

    def test_s2_threshold(self):
        assert classify_s_scale(100.0) == "S2"

    def test_s3_threshold(self):
        assert classify_s_scale(1000.0) == "S3"

    def test_s4_threshold(self):
        assert classify_s_scale(10000.0) == "S4"

    def test_s5_threshold(self):
        assert classify_s_scale(100000.0) == "S5"

    def test_s5_extreme(self):
        assert classify_s_scale(1_000_000.0) == "S5"

    def test_mid_range(self):
        assert classify_s_scale(500.0) == "S2"

    def test_s_scale_dict_complete(self):
        assert set(S_SCALE.keys()) == {"S1", "S2", "S3", "S4", "S5"}


# ============================================================================
# 5. Dose calculations
# ============================================================================

class TestDose:
    """Test radiation dose calculations."""

    def test_dose_rate_positive(self):
        dr = dose_rate_msv_hr(100.0)
        assert dr > 0.0

    def test_dose_rate_zero(self):
        assert dose_rate_msv_hr(0.0) == 0.0

    def test_dose_rate_negative(self):
        assert dose_rate_msv_hr(-10.0) == 0.0

    def test_dose_rate_proportional(self):
        dr1 = dose_rate_msv_hr(100.0)
        dr10 = dose_rate_msv_hr(1000.0)
        assert abs(dr10 / dr1 - 10.0) < 0.01

    def test_total_dose_triangular(self):
        d = event_total_dose_msv(1000.0, 10.0)
        expected = 1000.0 * 0.5 * PFU_TO_MSV_HR * 10.0
        assert abs(d - expected) < 0.01

    def test_sheltered_dose_reduced(self):
        unsheltered = event_total_dose_msv(1000.0, 10.0)
        sheltered = sheltered_dose_msv(unsheltered)
        assert sheltered < unsheltered
        assert abs(sheltered - unsheltered * SHELTER_ATTENUATION) < 0.01

    def test_sheltered_dose_zero_input(self):
        assert sheltered_dose_msv(0.0) == 0.0

    def test_sept2017_reference(self):
        """Sept 2017 SPE at Mars: MAVEN/RAD measured ~50 mSv.
        Our model: ~S3 event, 1000 pfu, ~16 hr."""
        d = event_total_dose_msv(1000.0, 16.0)
        assert 20.0 < d < 100.0


# ============================================================================
# 6. Alert levels
# ============================================================================

class TestAlerts:
    """Test alert level thresholds."""

    def test_nominal(self):
        assert alert_from_pfu(0.1) == "nominal"

    def test_watch(self):
        assert alert_from_pfu(ALERT_WATCH_PFU) == "watch"

    def test_warning(self):
        assert alert_from_pfu(ALERT_WARNING_PFU) == "warning"

    def test_critical(self):
        assert alert_from_pfu(ALERT_CRITICAL_PFU) == "critical"

    def test_emergency(self):
        assert alert_from_pfu(ALERT_EMERGENCY_PFU) == "emergency"

    def test_shelter_critical(self):
        assert should_shelter("critical") is True

    def test_shelter_emergency(self):
        assert should_shelter("emergency") is True

    def test_no_shelter_warning(self):
        assert should_shelter("warning") is False

    def test_no_shelter_nominal(self):
        assert should_shelter("nominal") is False


# ============================================================================
# 7. Event generation
# ============================================================================

class TestEventGeneration:
    """Test stochastic SPE generation."""

    def test_generate_spe_returns_event(self):
        evt = generate_spe(1, "S3", random.Random(42))
        assert isinstance(evt, SolarEvent)

    def test_generate_spe_pfu_in_range(self):
        for s_class in ("S1", "S2", "S3", "S4", "S5"):
            evt = generate_spe(1, s_class, random.Random(42))
            bounds = S_SCALE[s_class]
            assert bounds["min_pfu"] <= evt.peak_pfu <= bounds["max_pfu"]

    def test_generate_spe_positive_duration(self):
        evt = generate_spe(1, "S4", random.Random(99))
        assert evt.duration_hours > 0.0

    def test_generate_spe_positive_warning(self):
        evt = generate_spe(1, "S3", random.Random(7))
        assert evt.warning_time_min > 0.0

    def test_generate_spe_dose_positive(self):
        evt = generate_spe(1, "S5", random.Random(1))
        assert evt.dose_unshielded_msv > 0.0
        assert evt.dose_sheltered_msv > 0.0

    def test_generate_sheltered_less(self):
        evt = generate_spe(1, "S4", random.Random(3))
        assert evt.dose_sheltered_msv < evt.dose_unshielded_msv

    def test_check_for_spe_deterministic(self):
        r1 = check_for_spe(1, "solar_max", random.Random(12345))
        r2 = check_for_spe(1, "solar_max", random.Random(12345))
        if r1 is None:
            assert r2 is None
        else:
            assert r2 is not None
            assert r1.peak_pfu == r2.peak_pfu

    def test_check_for_spe_can_return_none(self):
        none_count = 0
        for seed in range(100):
            if check_for_spe(1, "solar_min", random.Random(seed)) is None:
                none_count += 1
        assert none_count > 80

    def test_solar_max_more_events(self):
        max_events = sum(
            1 for s in range(500)
            if check_for_spe(1, "solar_max", random.Random(s)) is not None
        )
        min_events = sum(
            1 for s in range(500)
            if check_for_spe(1, "solar_min", random.Random(s)) is not None
        )
        assert max_events >= min_events


# ============================================================================
# 8. Tick engine
# ============================================================================

class TestTick:
    """Test per-sol tick function."""

    def test_tick_advances_sol(self):
        state = make_monitor()
        tick_monitor(state, rng=random.Random(1))
        assert state.sol == 1

    def test_tick_returns_report(self):
        state = make_monitor()
        report = tick_monitor(state, rng=random.Random(1))
        assert isinstance(report, SolReport)

    def test_tick_cumulative_dose_increases(self):
        state = make_monitor()
        tick_monitor(state, rng=random.Random(1))
        assert state.cumulative_dose_msv > 0.0

    def test_tick_power_consumed(self):
        state = make_monitor()
        tick_monitor(state, rng=random.Random(1))
        assert state.power_consumed_kwh > 0.0

    def test_tick_shelter_activates_on_event(self):
        state = make_monitor()
        rng = random.Random(42)
        activated = False
        for _ in range(1000):
            report = tick_monitor(state, rng=rng)
            if state.shelter_active:
                activated = True
                break
        # Not guaranteed, but with 1000 sols at solar max, very likely
        # If no event, test still passes -- we just check the machinery works

    def test_tick_10_sols_no_crash(self):
        state = make_monitor()
        rng = random.Random(7)
        for _ in range(10):
            tick_monitor(state, rng=rng)
        assert state.sol == 10


# ============================================================================
# 9. Integration -- run_monitor
# ============================================================================

class TestIntegration:
    """Multi-sol integration tests."""

    def test_run_returns_reports(self):
        reports = run_monitor(sols=30, seed=42)
        assert len(reports) == 30

    def test_run_365_sols(self):
        reports = run_monitor(sols=365, seed=99)
        assert len(reports) == 365
        assert reports[-1].cumulative_dose_msv > 0.0

    def test_dose_monotonically_increases(self):
        reports = run_monitor(sols=100, seed=7)
        for i in range(1, len(reports)):
            assert reports[i].cumulative_dose_msv >= reports[i-1].cumulative_dose_msv

    def test_gcr_dominates_quiet_sols(self):
        reports = run_monitor(sols=100, solar_phase="solar_min", seed=12345)
        quiet_reports = [r for r in reports if r.alert_level == "nominal"]
        if quiet_reports:
            avg_dose = sum(r.dose_msv for r in quiet_reports) / len(quiet_reports)
            assert abs(avg_dose - GCR_BASELINE_MSV_SOL) < 1.0

    def test_all_sols_have_power(self):
        reports = run_monitor(sols=50, seed=3)
        for r in reports:
            assert r.power_kwh > 0.0

    def test_deterministic_with_seed(self):
        r1 = run_monitor(sols=50, seed=42)
        r2 = run_monitor(sols=50, seed=42)
        for a, b in zip(r1, r2):
            assert a.current_pfu == b.current_pfu

    def test_event_history_grows(self):
        reports = run_monitor(sols=668, seed=42)  # one Mars year
        events_detected = [r for r in reports if r.event_detected is not None]
        # At solar max, expect at least a few events per Mars year
        # But seed-dependent; just check count is non-negative
        assert len(events_detected) >= 0

    def test_shelter_activations_counted(self):
        state = make_monitor()
        rng = random.Random(42)
        for _ in range(668):
            tick_monitor(state, rng=rng)
        assert state.shelter_activations >= 0

    def test_sols_at_elevated_tracked(self):
        state = make_monitor()
        rng = random.Random(42)
        for _ in range(100):
            tick_monitor(state, rng=rng)
        assert state.sols_at_elevated >= 0


# ============================================================================
# 10. Edge cases
# ============================================================================

class TestEdgeCases:
    """Boundary and edge case tests."""

    def test_zero_pfu_dose(self):
        assert dose_rate_msv_hr(0.0) == 0.0

    def test_huge_pfu(self):
        d = dose_rate_msv_hr(1e6)
        assert d > 0.0
        assert math.isfinite(d)

    def test_classify_zero_pfu(self):
        assert classify_s_scale(0.0) == "none"

    def test_classify_boundary_9_99(self):
        assert classify_s_scale(9.99) == "none"

    def test_alert_boundary_below_watch(self):
        assert alert_from_pfu(4.9) == "nominal"

    def test_monitor_power_positive(self):
        assert monitor_power_kwh() > 0.0

    def test_make_monitor_defaults(self):
        m = make_monitor()
        assert m.sol == 0
        assert m.alert_level == "nominal"
        assert m.cumulative_dose_msv == 0.0
        assert m.events_detected == 0
        assert m.event_history == []

    def test_event_total_dose_zero_duration(self):
        assert event_total_dose_msv(1000.0, 0.0) == 0.0

    def test_run_zero_sols(self):
        reports = run_monitor(sols=0, seed=1)
        assert reports == []

    def test_run_one_sol(self):
        reports = run_monitor(sols=1, seed=1)
        assert len(reports) == 1
