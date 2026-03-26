"""Tests for radiation_monitor.py -- Mars Radiation Monitoring."""
from __future__ import annotations
import math, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.radiation_monitor import (
    RadiationMonitorState, CrewDosimetry, SPEEvent, TickResult,
    solar_cycle_phase, gcr_modulation_factor, gcr_dose_rate,
    spe_probability, generate_spe, spe_dose_rate_at_time,
    shielding_attenuation, thickness_to_gcm2, effective_dose,
    detector_degradation, check_dose_limits, integrate_spe_sol_dose,
    tick_radiation, create_radiation_monitor,
    GCR_BASELINE_MSV_DAY, GCR_SOLAR_MIN_FACTOR, GCR_SOLAR_MAX_FACTOR,
    SOLAR_CYCLE_SOLS, SPE_DURATION_HOURS_MIN, SPE_DURATION_HOURS_MAX,
    ATTEN_LENGTH_REGOLITH, ATTEN_LENGTH_POLYETHYLENE,
    ATTEN_LENGTH_WATER, ATTEN_LENGTH_ALUMINUM,
    QUALITY_FACTOR_GCR, QUALITY_FACTOR_SPE,
    CAREER_LIMIT_MSV, ANNUAL_LIMIT_MSV, THIRTY_DAY_LIMIT_MSV,
    ALERT_FRACTION, HOURS_PER_SOL,
)

class TestCrewDosimetry:
    def test_defaults(self):
        c = CrewDosimetry(); assert c.name == "crew-1" and c.cumulative_msv == 0.0
    def test_negative_clamped(self):
        assert CrewDosimetry(cumulative_msv=-10.0).cumulative_msv == 0.0
    def test_eva_high(self):
        assert CrewDosimetry(eva_fraction=1.5).eva_fraction == 1.0
    def test_eva_low(self):
        assert CrewDosimetry(eva_fraction=-0.5).eva_fraction == 0.0
    def test_thirty_day(self):
        assert CrewDosimetry(thirty_day_msv=-1.0).thirty_day_msv == 0.0
    def test_annual(self):
        assert CrewDosimetry(annual_msv=-1.0).annual_msv == 0.0

class TestSPEEvent:
    def test_defaults(self):
        assert SPEEvent().active is True
    def test_min_duration(self):
        assert SPEEvent(duration_hours=0.5).duration_hours == SPE_DURATION_HOURS_MIN
    def test_neg_rate(self):
        assert SPEEvent(peak_dose_rate_msv_hr=-5.0).peak_dose_rate_msv_hr == 0.0
    def test_neg_elapsed(self):
        assert SPEEvent(elapsed_hours=-10.0).elapsed_hours == 0.0
    def test_neg_delivered(self):
        assert SPEEvent(total_delivered_msv=-1.0).total_delivered_msv == 0.0

class TestRadiationMonitorState:
    def test_defaults(self):
        s = RadiationMonitorState(); assert s.detector_health == 1.0
    def test_health_high(self):
        assert RadiationMonitorState(detector_health=1.5).detector_health == 1.0
    def test_health_low(self):
        assert RadiationMonitorState(detector_health=-0.1).detector_health == 0.0
    def test_shielding(self):
        assert RadiationMonitorState(habitat_shielding_gcm2=-5.0).habitat_shielding_gcm2 == 0.0

class TestSolarCyclePhase:
    def test_zero(self):
        assert solar_cycle_phase(0) == 0.0
    def test_wrap(self):
        p = solar_cycle_phase(round(SOLAR_CYCLE_SOLS)); assert p < 0.01 or p > 0.99
    def test_half(self):
        assert 0.45 < solar_cycle_phase(int(SOLAR_CYCLE_SOLS / 2)) < 0.55
    def test_range(self):
        for s in [0, 100, 1000, 5000, 50000]: assert 0.0 <= solar_cycle_phase(s) < 1.0

class TestGCRModulation:
    def test_solar_min(self):
        assert abs(gcr_modulation_factor(0.0) - GCR_SOLAR_MIN_FACTOR) < 1e-6
    def test_solar_max(self):
        assert abs(gcr_modulation_factor(0.5) - GCR_SOLAR_MAX_FACTOR) < 1e-6
    def test_positive(self):
        for p in [i/20 for i in range(21)]: assert gcr_modulation_factor(p) > 0
    def test_bounded(self):
        for p in [i/100 for i in range(101)]:
            assert GCR_SOLAR_MAX_FACTOR - 1e-9 <= gcr_modulation_factor(p) <= GCR_SOLAR_MIN_FACTOR + 1e-9
    def test_symmetry(self):
        assert abs(gcr_modulation_factor(0.25) - gcr_modulation_factor(0.75)) < 1e-6

class TestGCRDoseRate:
    def test_positive(self):
        for s in range(0, 10000, 500): assert gcr_dose_rate(s) > 0
    def test_plausible(self):
        for s in range(0, 10000, 100): assert 0.3 < gcr_dose_rate(s) < 1.2
    def test_varies(self):
        assert len(set(gcr_dose_rate(s) for s in range(0, 8000, 1000))) > 1

class TestSPEProbability:
    def test_positive(self):
        for p in [i/10 for i in range(11)]: assert spe_probability(p) > 0
    def test_higher_at_max(self):
        assert spe_probability(0.5) > spe_probability(0.0)
    def test_reasonable(self):
        for p in [i/10 for i in range(11)]: assert spe_probability(p) < 0.05

class TestGenerateSPE:
    def test_active(self):
        e = generate_spe(100, random.Random(42)); assert e.active and e.start_sol == 100
    def test_duration(self):
        rng = random.Random(42)
        for _ in range(50):
            e = generate_spe(0, rng); assert SPE_DURATION_HOURS_MIN <= e.duration_hours <= SPE_DURATION_HOURS_MAX
    def test_peak_rate(self):
        rng = random.Random(42)
        for _ in range(50): assert generate_spe(0, rng).peak_dose_rate_msv_hr > 0
    def test_deterministic(self):
        assert generate_spe(10, random.Random(123)).duration_hours == generate_spe(10, random.Random(123)).duration_hours

class TestSPEDoseRate:
    def test_inactive(self):
        assert spe_dose_rate_at_time(SPEEvent(active=False), 1.0) == 0.0
    def test_neg_time(self):
        assert spe_dose_rate_at_time(SPEEvent(), -1.0) == 0.0
    def test_past(self):
        assert spe_dose_rate_at_time(SPEEvent(duration_hours=10.0), 11.0) == 0.0
    def test_peak(self):
        e = SPEEvent(duration_hours=10.0, peak_dose_rate_msv_hr=50.0)
        assert spe_dose_rate_at_time(e, 5.0) > spe_dose_rate_at_time(e, 1.0)
    def test_symmetric(self):
        e = SPEEvent(duration_hours=10.0, peak_dose_rate_msv_hr=50.0)
        assert abs(spe_dose_rate_at_time(e, 3.0) - spe_dose_rate_at_time(e, 7.0)) < 1e-6
    def test_non_negative(self):
        e = SPEEvent(duration_hours=20.0, peak_dose_rate_msv_hr=100.0)
        for t in [i*0.5 for i in range(41)]: assert spe_dose_rate_at_time(e, t) >= 0

class TestShielding:
    def test_zero(self):
        assert shielding_attenuation(0.0) == 1.0
    def test_monotonic(self):
        assert shielding_attenuation(50.0) < shielding_attenuation(10.0)
    def test_range(self):
        for x in [0,5,10,25,50,100,200]: assert 0 <= shielding_attenuation(float(x)) <= 1
    def test_materials(self):
        assert shielding_attenuation(25.0, "regolith") < shielding_attenuation(25.0, "polyethylene")
    def test_exp_law(self):
        assert abs(shielding_attenuation(ATTEN_LENGTH_REGOLITH, "regolith") - math.exp(-1)) < 1e-6
    def test_negative(self):
        assert shielding_attenuation(-10.0) == 1.0
    def test_unknown(self):
        assert abs(shielding_attenuation(25.0, "x") - shielding_attenuation(25.0, "regolith")) < 1e-10
    def test_50cm(self):
        assert shielding_attenuation(thickness_to_gcm2(50.0, "regolith"), "regolith") < 0.10

class TestThickness:
    def test_zero(self):
        assert thickness_to_gcm2(0.0) == 0.0
    def test_regolith(self):
        assert abs(thickness_to_gcm2(10.0, "regolith") - 15.0) < 1e-6
    def test_water(self):
        assert abs(thickness_to_gcm2(10.0, "water") - 10.0) < 1e-6
    def test_aluminum(self):
        assert abs(thickness_to_gcm2(10.0, "aluminum") - 27.0) < 1e-6
    def test_neg(self):
        assert thickness_to_gcm2(-5.0) == 0.0

class TestEffectiveDose:
    def test_unity(self):
        assert effective_dose(10.0, 1.0) == 10.0
    def test_gcr(self):
        assert effective_dose(10.0, QUALITY_FACTOR_GCR) == 30.0
    def test_zero(self):
        assert effective_dose(0.0, 5.0) == 0.0
    def test_neg_qf(self):
        assert effective_dose(10.0, -1.0) == 0.0

class TestDetector:
    def test_no_dose(self):
        assert detector_degradation(1.0, 0.0) == 1.0
    def test_reduces(self):
        assert detector_degradation(1.0, 1000.0) < 1.0
    def test_floor(self):
        assert detector_degradation(0.01, 1e6) >= 0.0
    def test_ceiling(self):
        assert detector_degradation(1.0, -100.0) <= 1.0
    def test_cumulative(self):
        h = 1.0
        for _ in range(100): h = detector_degradation(h, 10.0)
        assert 0 < h < 1

class TestDoseLimits:
    def test_safe(self):
        assert len(check_dose_limits(CrewDosimetry(cumulative_msv=100.0))) == 0
    def test_career_warn(self):
        assert any("career" in a.lower() for a in check_dose_limits(CrewDosimetry(cumulative_msv=CAREER_LIMIT_MSV * ALERT_FRACTION + 1)))
    def test_career_crit(self):
        assert any("CRITICAL" in a for a in check_dose_limits(CrewDosimetry(cumulative_msv=CAREER_LIMIT_MSV + 1)))
    def test_30day(self):
        assert any("30-day" in a for a in check_dose_limits(CrewDosimetry(thirty_day_msv=THIRTY_DAY_LIMIT_MSV * ALERT_FRACTION + 1)))
    def test_annual(self):
        assert any("annual" in a.lower() for a in check_dose_limits(CrewDosimetry(annual_msv=ANNUAL_LIMIT_MSV + 1)))
    def test_multi(self):
        assert len(check_dose_limits(CrewDosimetry(cumulative_msv=CAREER_LIMIT_MSV+1, thirty_day_msv=THIRTY_DAY_LIMIT_MSV+1, annual_msv=ANNUAL_LIMIT_MSV+1))) >= 3

class TestIntegrateSPE:
    def test_inactive(self):
        assert integrate_spe_sol_dose(SPEEvent(active=False)) == 0.0
    def test_positive(self):
        assert integrate_spe_sol_dose(SPEEvent(duration_hours=24.0, peak_dose_rate_msv_hr=10.0)) > 0
    def test_bounded(self):
        assert integrate_spe_sol_dose(SPEEvent(duration_hours=24.0, peak_dose_rate_msv_hr=10.0)) <= 10 * HOURS_PER_SOL
    def test_longer_more(self):
        assert integrate_spe_sol_dose(SPEEvent(duration_hours=40.0, peak_dose_rate_msv_hr=10.0)) > integrate_spe_sol_dose(SPEEvent(duration_hours=4.0, peak_dose_rate_msv_hr=10.0))

class TestTick:
    def test_sol(self):
        s = create_radiation_monitor(); tick_radiation(s, random.Random(42)); assert s.sol == 1
    def test_gcr(self):
        assert tick_radiation(create_radiation_monitor(), random.Random(42)).gcr_dose_msv > 0
    def test_hab_lt_unshielded(self):
        r = tick_radiation(create_radiation_monitor(), random.Random(42)); assert r.habitat_dose_msv <= r.total_unshielded_msv
    def test_shelter_lt_hab(self):
        r = tick_radiation(create_radiation_monitor(), random.Random(42)); assert r.shelter_dose_msv <= r.habitat_dose_msv
    def test_crew_cumulative(self):
        s = create_radiation_monitor(); tick_radiation(s, random.Random(42)); assert all(c.cumulative_msv > 0 for c in s.crew)
    def test_station(self):
        s = create_radiation_monitor(); tick_radiation(s, random.Random(42)); assert s.station_cumulative_msv > 0
    def test_detector(self):
        s = create_radiation_monitor(); h0 = s.detector_health; tick_radiation(s, random.Random(42)); assert s.detector_health <= h0
    def test_deterministic(self):
        s1, s2 = create_radiation_monitor(), create_radiation_monitor()
        assert tick_radiation(s1, random.Random(42)).gcr_dose_msv == tick_radiation(s2, random.Random(42)).gcr_dose_msv
    def test_four_crew(self):
        s = create_radiation_monitor(); tick_radiation(s, random.Random(42)); assert all(len(c.daily_history) == 1 for c in s.crew)

class TestSmoke:
    def test_10sol(self):
        s = create_radiation_monitor(); rng = random.Random(12345)
        for _ in range(10): assert tick_radiation(s, rng).gcr_dose_msv > 0
        assert s.sol == 10
    def test_100sol(self):
        s = create_radiation_monitor(); rng = random.Random(54321)
        for _ in range(100): tick_radiation(s, rng)
        assert all(c.cumulative_msv > 0 for c in s.crew)
    def test_1000sol_trim(self):
        s = create_radiation_monitor(); rng = random.Random(99999)
        for _ in range(1000): tick_radiation(s, rng)
        assert all(len(c.daily_history) <= 668 for c in s.crew)
    def test_dose_positive(self):
        s = create_radiation_monitor(); rng = random.Random(77777)
        for _ in range(50):
            r = tick_radiation(s, rng); assert r.gcr_dose_msv >= 0 and r.spe_dose_msv >= 0
    def test_shielding_reduces(self):
        s = create_radiation_monitor(); rng = random.Random(88888)
        for _ in range(50):
            r = tick_radiation(s, rng); assert r.habitat_dose_msv <= r.total_unshielded_msv + 1e-12
    def test_detector_mono(self):
        s = create_radiation_monitor(); rng = random.Random(11111); prev = s.detector_health
        for _ in range(50): tick_radiation(s, rng); assert s.detector_health <= prev + 1e-12; prev = s.detector_health

class TestFactory:
    def test_standard(self):
        assert create_radiation_monitor("standard").shielding_material == "regolith"
    def test_minimal(self):
        assert create_radiation_monitor("minimal").shielding_material == "aluminum"
    def test_bunker(self):
        assert create_radiation_monitor("bunker").shelter_shielding_gcm2 == 100.0
    def test_custom(self):
        s = create_radiation_monitor(crew_names=["a", "b"]); assert len(s.crew) == 2
    def test_unknown(self):
        assert create_radiation_monitor("x").habitat_shielding_gcm2 == create_radiation_monitor("standard").habitat_shielding_gcm2

class TestEVA:
    def test_increases(self):
        s1, s2 = create_radiation_monitor(crew_names=["h"]), create_radiation_monitor(crew_names=["e"])
        s2.crew[0].eva_fraction = 0.5
        tick_radiation(s1, random.Random(42)); tick_radiation(s2, random.Random(42))
        assert s2.crew[0].cumulative_msv > s1.crew[0].cumulative_msv
    def test_full(self):
        s = create_radiation_monitor(crew_names=["w"]); s.crew[0].eva_fraction = 1.0
        tick_radiation(s, random.Random(42)); assert s.crew[0].cumulative_msv > 0

class TestSPELifecycle:
    def test_delivers(self):
        s = create_radiation_monitor(); s.active_spe = SPEEvent(duration_hours=24.0, peak_dose_rate_msv_hr=20.0)
        assert tick_radiation(s, random.Random(42)).spe_dose_msv > 0
    def test_ends(self):
        s = create_radiation_monitor(); s.active_spe = SPEEvent(duration_hours=10.0, peak_dose_rate_msv_hr=20.0)
        assert tick_radiation(s, random.Random(42)).spe_ended is True
    def test_shelter(self):
        s = create_radiation_monitor(); s.active_spe = SPEEvent(duration_hours=48.0, peak_dose_rate_msv_hr=50.0); s.spe_alert = True
        assert tick_radiation(s, random.Random(42)).shelter_ordered is True
    def test_count(self):
        s = create_radiation_monitor(); rng = random.Random(42)
        for _ in range(500): tick_radiation(s, rng)
        assert s.total_spe_events >= 0

class TestPhysics:
    def test_gcr_range(self):
        for sol in range(0, 10000, 100): assert 0.3 <= gcr_dose_rate(sol) <= 1.2
    def test_exp_all(self):
        for m, l in [("regolith", ATTEN_LENGTH_REGOLITH), ("polyethylene", ATTEN_LENGTH_POLYETHYLENE),
                      ("water", ATTEN_LENGTH_WATER), ("aluminum", ATTEN_LENGTH_ALUMINUM)]:
            for x in [0, 10, 25, 50, 100]: assert abs(shielding_attenuation(float(x), m) - math.exp(-x/l)) < 1e-10
    def test_qf(self):
        for q in [QUALITY_FACTOR_GCR, QUALITY_FACTOR_SPE]: assert effective_dose(10.0, q) >= 10.0
    def test_sol_mono(self):
        s = create_radiation_monitor(); rng = random.Random(42)
        for e in range(1, 20): tick_radiation(s, rng); assert s.sol == e
    def test_cum_mono(self):
        s = create_radiation_monitor(); rng = random.Random(42); prev = [0.0]*4
        for _ in range(50):
            tick_radiation(s, rng)
            for i, c in enumerate(s.crew): assert c.cumulative_msv >= prev[i]; prev[i] = c.cumulative_msv
