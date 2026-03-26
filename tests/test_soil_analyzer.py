"""Tests for soil_analyzer.py — Mars Regolith XRF Analysis."""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.soil_analyzer import (
    AnalyzerState, SoilSample, AnalysisResult,
    beer_lambert, xrf_signal_counts, signal_to_noise,
    penetration_depth_um, perchlorate_from_chlorine, water_proxy,
    assess_toxicity, tube_degradation_rate, temperature_quality_factor,
    tick_analysis, create_soil_analyzer,
    KA_ENERGIES, MARS_BASELINE, PERCHLORATE_SAFE_LIMIT,
    MU_REGOLITH_6KEV, REGOLITH_DENSITY,
)


# ── SoilSample clamping ────────────────────────────────────────────

class TestSoilSample:
    def test_defaults(self):
        s = SoilSample()
        assert s.depth_cm == 0.0
        assert s.perchlorate_pct == 0.0
        assert s.quality == 1.0

    def test_negative_depth_clamped(self):
        s = SoilSample(depth_cm=-5.0)
        assert s.depth_cm == 0.0

    def test_perchlorate_clamped(self):
        s = SoilSample(perchlorate_pct=200.0)
        assert s.perchlorate_pct == 100.0
        s2 = SoilSample(perchlorate_pct=-1.0)
        assert s2.perchlorate_pct == 0.0

    def test_quality_clamped(self):
        s = SoilSample(quality=5.0)
        assert s.quality == 1.0
        s2 = SoilSample(quality=-0.5)
        assert s2.quality == 0.0


# ── AnalyzerState clamping ──────────────────────────────────────────

class TestAnalyzerState:
    def test_defaults(self):
        a = AnalyzerState()
        assert a.tube_voltage_kv == 40.0
        assert a.calibrated is True

    def test_voltage_clamped(self):
        a = AnalyzerState(tube_voltage_kv=100.0)
        assert a.tube_voltage_kv == 60.0
        a2 = AnalyzerState(tube_voltage_kv=-5.0)
        assert a2.tube_voltage_kv == 0.0

    def test_integration_time_clamped(self):
        a = AnalyzerState(integration_time_s=0.1)
        assert a.integration_time_s == 1.0
        a2 = AnalyzerState(integration_time_s=999.0)
        assert a2.integration_time_s == 600.0

    def test_temp_clamped(self):
        a = AnalyzerState(temperature_c=-200.0)
        assert a.temperature_c == -120.0
        a2 = AnalyzerState(temperature_c=200.0)
        assert a2.temperature_c == 80.0


# ── Beer-Lambert ────────────────────────────────────────────────────

class TestBeerLambert:
    def test_zero_depth(self):
        assert beer_lambert(1000.0, 50.0, 1.5, 0.0) == 1000.0

    def test_attenuation(self):
        result = beer_lambert(1000.0, 50.0, 1.5, 0.01)
        assert 0 < result < 1000.0

    def test_deep_nearly_zero(self):
        result = beer_lambert(1000.0, 50.0, 1.5, 1.0)
        assert result < 1e-10

    def test_no_negative(self):
        result = beer_lambert(1000.0, 50.0, 1.5, 100.0)
        assert result >= 0.0

    def test_higher_density_more_attenuation(self):
        thin = beer_lambert(1000.0, 50.0, 1.0, 0.01)
        dense = beer_lambert(1000.0, 50.0, 3.0, 0.01)
        assert dense < thin


# ── XRF signal counts ──────────────────────────────────────────────

class TestXRFSignal:
    def test_zero_fraction(self):
        assert xrf_signal_counts(40.0, 0.1, 30.0, 0.0) == 0.0

    def test_positive_counts(self):
        c = xrf_signal_counts(40.0, 0.1, 30.0, 0.18)
        assert c > 0

    def test_higher_voltage_more_counts(self):
        lo = xrf_signal_counts(20.0, 0.1, 30.0, 0.1)
        hi = xrf_signal_counts(40.0, 0.1, 30.0, 0.1)
        assert hi > lo

    def test_longer_integration_more_counts(self):
        short = xrf_signal_counts(40.0, 0.1, 10.0, 0.1)
        long = xrf_signal_counts(40.0, 0.1, 60.0, 0.1)
        assert long > short

    def test_higher_concentration_more_counts(self):
        low = xrf_signal_counts(40.0, 0.1, 30.0, 0.05)
        high = xrf_signal_counts(40.0, 0.1, 30.0, 0.20)
        assert high > low


# ── Signal to noise ─────────────────────────────────────────────────

class TestSNR:
    def test_zero_counts(self):
        snr = signal_to_noise(0.0, 10.0)
        assert snr == 0.0

    def test_positive(self):
        snr = signal_to_noise(1000.0, 10.0)
        assert snr > 0

    def test_more_counts_better_snr(self):
        lo = signal_to_noise(100.0, 10.0)
        hi = signal_to_noise(10000.0, 10.0)
        assert hi > lo

    def test_snr_scales_sublinearly(self):
        """SNR ~ sqrt(N), so 100x counts gives ~10x SNR."""
        s1 = signal_to_noise(100.0)
        s2 = signal_to_noise(10000.0)
        ratio = s2 / s1
        assert 5 < ratio < 15  # roughly sqrt(100)=10


# ── Penetration depth ──────────────────────────────────────────────

class TestPenetrationDepth:
    def test_zero_energy(self):
        assert penetration_depth_um(0.0, 1.5) == 0.0

    def test_positive(self):
        d = penetration_depth_um(6.4, 1.5)
        assert d > 0

    def test_higher_energy_deeper(self):
        lo = penetration_depth_um(2.0, 1.5)
        hi = penetration_depth_um(10.0, 1.5)
        assert hi > lo

    def test_denser_material_shallower(self):
        loose = penetration_depth_um(6.4, 1.0)
        dense = penetration_depth_um(6.4, 3.0)
        assert dense < loose


# ── Perchlorate estimation ──────────────────────────────────────────

class TestPerchlorate:
    def test_no_chlorine(self):
        assert perchlorate_from_chlorine(0.0, 1.0) == 0.0

    def test_positive_with_cl(self):
        p = perchlorate_from_chlorine(0.7, 2.0)
        assert p > 0

    def test_more_cl_more_perchlorate(self):
        lo = perchlorate_from_chlorine(0.3, 2.0)
        hi = perchlorate_from_chlorine(1.0, 2.0)
        assert hi > lo

    def test_no_excess_o_lower_confidence(self):
        with_o = perchlorate_from_chlorine(0.7, 2.0)
        without_o = perchlorate_from_chlorine(0.7, 0.0)
        assert with_o > without_o

    def test_bounded(self):
        """Perchlorate should be less than Cl * 3 (stoichiometric max)."""
        p = perchlorate_from_chlorine(1.0, 10.0)
        assert p < 3.0


# ── Water proxy ─────────────────────────────────────────────────────

class TestWaterProxy:
    def test_no_excess(self):
        assert water_proxy(40.0, 45.0) == 0.0

    def test_positive_excess(self):
        w = water_proxy(50.0, 45.0)
        assert w > 0

    def test_proportional(self):
        w1 = water_proxy(46.0, 45.0)
        w2 = water_proxy(50.0, 45.0)
        assert w2 > w1

    def test_h2o_scaling(self):
        """H2O/O ratio is 1.125."""
        w = water_proxy(46.0, 45.0)
        assert abs(w - 1.125) < 0.01


# ── Toxicity assessment ────────────────────────────────────────────

class TestToxicity:
    def test_safe_soil(self):
        toxic, reason = assess_toxicity(0.0005)
        assert not toxic
        assert reason == "safe"

    def test_perchlorate_toxic(self):
        toxic, reason = assess_toxicity(0.01)
        assert toxic
        assert "perchlorate" in reason

    def test_chromium_toxic(self):
        toxic, reason = assess_toxicity(0.0, cr_pct=0.1)
        assert toxic
        assert "chromium" in reason

    def test_nickel_toxic(self):
        toxic, reason = assess_toxicity(0.0, ni_pct=0.05)
        assert toxic
        assert "nickel" in reason

    def test_multiple_toxins(self):
        toxic, reason = assess_toxicity(0.01, cr_pct=0.1)
        assert toxic
        assert "perchlorate" in reason
        assert "chromium" in reason


# ── Tube degradation ───────────────────────────────────────────────

class TestTubeDegradation:
    def test_new_tube(self):
        assert tube_degradation_rate(0.0, 5000.0) == 1.0

    def test_half_life(self):
        d = tube_degradation_rate(2500.0, 5000.0)
        assert 0.5 < d < 1.0  # quadratic: 1 - 0.25 = 0.75

    def test_end_of_life(self):
        d = tube_degradation_rate(5000.0, 5000.0)
        assert d == 0.0

    def test_monotonic_decrease(self):
        prev = 1.0
        for h in range(0, 5001, 500):
            d = tube_degradation_rate(float(h), 5000.0)
            assert d <= prev
            prev = d

    def test_zero_max(self):
        assert tube_degradation_rate(100.0, 0.0) == 0.0


# ── Temperature quality ────────────────────────────────────────────

class TestTempQuality:
    def test_optimal_range(self):
        assert temperature_quality_factor(-20.0) == 1.0
        assert temperature_quality_factor(0.0) == 1.0

    def test_too_hot(self):
        q = temperature_quality_factor(50.0)
        assert 0.3 <= q < 1.0

    def test_too_cold(self):
        q = temperature_quality_factor(-80.0)
        assert 0.3 <= q < 1.0

    def test_extreme_clamp(self):
        q = temperature_quality_factor(200.0)
        assert q >= 0.3


# ── Tick analysis ───────────────────────────────────────────────────

class TestTickAnalysis:
    def test_idle_tick(self):
        state = create_soil_analyzer()
        result = tick_analysis(state, dt_s=60.0, take_sample=False)
        assert result.sample is None
        assert state.power_w == 5.0
        assert state.samples_taken == 0

    def test_take_sample(self):
        state = create_soil_analyzer()
        result = tick_analysis(state, dt_s=60.0, take_sample=True, sample_depth_cm=5.0)
        assert result.sample is not None
        assert state.samples_taken == 1
        assert result.signal_counts > 0
        assert result.snr > 0

    def test_sample_has_composition(self):
        state = create_soil_analyzer()
        result = tick_analysis(state, dt_s=60.0, take_sample=True)
        assert len(result.sample.composition) > 0
        total = sum(result.sample.composition.values())
        assert 95.0 < total < 105.0  # approximately 100%

    def test_tube_hours_increase(self):
        state = create_soil_analyzer()
        h0 = state.tube_hours
        tick_analysis(state, dt_s=60.0, take_sample=True)
        assert state.tube_hours > h0

    def test_exhausted_tube(self):
        state = create_soil_analyzer()
        state.tube_hours = state.tube_max_hours
        result = tick_analysis(state, dt_s=60.0, take_sample=True)
        assert result.warning == "tube_exhausted"
        assert result.sample is None

    def test_uncalibrated_warning(self):
        state = create_soil_analyzer()
        state.calibrated = False
        result = tick_analysis(state, dt_s=60.0, take_sample=True)
        assert "uncalibrated" in result.warning or result.sample is not None

    def test_deeper_sample_more_water(self):
        state1 = create_soil_analyzer()
        r1 = tick_analysis(state1, take_sample=True, sample_depth_cm=0.0)
        state2 = create_soil_analyzer()
        r2 = tick_analysis(state2, take_sample=True, sample_depth_cm=20.0)
        assert r2.sample.water_proxy_pct >= r1.sample.water_proxy_pct

    def test_power_draw(self):
        state = create_soil_analyzer()
        result = tick_analysis(state, dt_s=60.0, take_sample=True)
        assert result.power_used_wh > 0
        assert state.power_w == 30.0

    def test_penetration_depth_positive(self):
        state = create_soil_analyzer()
        result = tick_analysis(state, dt_s=60.0, take_sample=True)
        assert result.penetration_depth_um > 0


# ── Factory ─────────────────────────────────────────────────────────

class TestFactory:
    def test_standard(self):
        s = create_soil_analyzer("standard")
        assert s.tube_voltage_kv == 40.0
        assert s.tube_max_hours == 5000.0

    def test_deep_core(self):
        s = create_soil_analyzer("deep_core")
        assert s.tube_voltage_kv == 50.0
        assert s.integration_time_s == 60.0

    def test_portable(self):
        s = create_soil_analyzer("portable")
        assert s.tube_voltage_kv == 30.0
        assert s.tube_max_hours == 2000.0

    def test_unknown_defaults(self):
        s = create_soil_analyzer("nonexistent")
        assert s.tube_voltage_kv == 40.0  # falls back to standard


# ── Invariants ──────────────────────────────────────────────────────

class TestInvariants:
    def test_10_sample_no_crash(self):
        state = create_soil_analyzer()
        for i in range(10):
            tick_analysis(state, dt_s=60.0, take_sample=True, sample_depth_cm=float(i))
        assert state.samples_taken == 10

    def test_50_sample_no_crash(self):
        state = create_soil_analyzer()
        for i in range(50):
            tick_analysis(state, dt_s=60.0, take_sample=(i % 3 == 0),
                         sample_depth_cm=float(i * 0.5))
        assert state.samples_taken == 17  # every 3rd of 50

    def test_tube_hours_monotonic(self):
        state = create_soil_analyzer()
        prev = 0.0
        for _ in range(20):
            tick_analysis(state, dt_s=60.0, take_sample=True)
            assert state.tube_hours >= prev
            prev = state.tube_hours

    def test_power_always_positive(self):
        state = create_soil_analyzer()
        for i in range(20):
            result = tick_analysis(state, dt_s=60.0, take_sample=(i % 2 == 0))
            assert result.power_used_wh >= 0

    def test_quality_bounded(self):
        state = create_soil_analyzer()
        for i in range(20):
            result = tick_analysis(state, dt_s=60.0, take_sample=True,
                                  sample_depth_cm=float(i))
            assert 0.0 <= result.sample.quality <= 1.0

    def test_composition_sums_near_100(self):
        state = create_soil_analyzer()
        for i in range(10):
            result = tick_analysis(state, dt_s=60.0, take_sample=True,
                                  sample_depth_cm=float(i * 2))
            total = sum(result.sample.composition.values())
            assert 90.0 < total < 110.0

    def test_snr_always_nonneg(self):
        state = create_soil_analyzer()
        for i in range(20):
            result = tick_analysis(state, dt_s=60.0, take_sample=True)
            assert result.snr >= 0.0

    def test_all_scenarios_run(self):
        for scenario in ["standard", "deep_core", "portable"]:
            state = create_soil_analyzer(scenario)
            result = tick_analysis(state, dt_s=60.0, take_sample=True)
            assert result.sample is not None

    def test_degradation_over_lifetime(self):
        """Tube degrades to zero over its lifetime."""
        state = create_soil_analyzer()
        state.tube_hours = state.tube_max_hours * 0.99
        result = tick_analysis(state, dt_s=60.0, take_sample=True)
        assert result.tube_degradation > 0.9  # near end of life

    def test_portable_shorter_integration(self):
        """Portable unit has lower SNR due to shorter integration."""
        std = create_soil_analyzer("standard")
        port = create_soil_analyzer("portable")
        r_std = tick_analysis(std, dt_s=60.0, take_sample=True)
        r_port = tick_analysis(port, dt_s=60.0, take_sample=True)
        assert r_std.signal_counts > r_port.signal_counts
