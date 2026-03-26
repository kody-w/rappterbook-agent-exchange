"""test_fire_suppression.py -- Tests for Mars Colony Fire Suppression.

88 tests across 16 categories: fire physics, detection, suppression,
crew safety, gas dynamics, conservation laws, simulation smoke tests.
"""
from __future__ import annotations

import random
import sys
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.fire_suppression import (
    FireSuppressionState, tick, activate_suppression, recharge_extinguishers,
    compute_risk_score, check_ignition, run_simulation, clamp,
    NOMINAL_O2_KPA, MIN_O2_FOR_COMBUSTION_KPA, CRITICAL_O2_KPA,
    ALPHA_SLOW, ALPHA_MEDIUM, ALPHA_FAST,
    O2_CONSUMED_PER_KW_MIN, CO2_PRODUCED_PER_KW_MIN, CO_PRODUCED_PER_KW_MIN,
    SMOKE_DETECTOR_THRESHOLD, CO_LETHAL_KPA, CO_DANGEROUS_KPA,
    EXTINGUISHER_CHARGES, CHARGE_DURATION_MIN,
    DETECTION_POWER_KW, SUPPRESSION_POWER_KW, VENTILATION_POWER_KW,
)


def make_fire(**kw):
    d = dict(fire_active=True, fire_duration_min=0, fire_growth_alpha=ALPHA_MEDIUM,
             fires_total=1, rng_seed=42)
    d.update(kw)
    return FireSuppressionState(**d)


def run_n(state, n):
    for _ in range(n):
        state = tick(state)
    return state


class TestInitialState:
    def test_default_o2(self):
        assert FireSuppressionState().o2_kpa == NOMINAL_O2_KPA

    def test_no_fire(self):
        assert not FireSuppressionState().fire_active

    def test_no_smoke(self):
        assert FireSuppressionState().smoke_opacity == 0.0

    def test_no_co(self):
        assert FireSuppressionState().co_kpa == 0.0

    def test_full_charges(self):
        assert FireSuppressionState().extinguisher_charges == EXTINGUISHER_CHARGES

    def test_no_alarm(self):
        assert not FireSuppressionState().alarm_active

    def test_zero_crew_exposure(self):
        s = FireSuppressionState()
        assert s.crew_co_exposure == 0.0 and s.crew_smoke_exposure == 0.0

    def test_idle_power(self):
        s = tick(FireSuppressionState(rng_seed=99999))
        assert s.power_draw_kw >= DETECTION_POWER_KW


class TestRiskScore:
    def test_nominal_moderate(self):
        r = compute_risk_score(FireSuppressionState())
        assert 0.0 < r < 1.0

    def test_high_o2_increases(self):
        rn = compute_risk_score(FireSuppressionState(o2_kpa=NOMINAL_O2_KPA))
        rh = compute_risk_score(FireSuppressionState(o2_kpa=CRITICAL_O2_KPA))
        assert rh > rn

    def test_low_o2_low(self):
        r = compute_risk_score(FireSuppressionState(o2_kpa=MIN_O2_FOR_COMBUSTION_KPA - 1))
        assert r < 0.5

    def test_humidity_reduces(self):
        rd = compute_risk_score(FireSuppressionState(humidity_fraction=0.1))
        rw = compute_risk_score(FireSuppressionState(humidity_fraction=0.9))
        assert rw < rd

    def test_dust_increases(self):
        rc = compute_risk_score(FireSuppressionState(dust_loading=0.0))
        rd = compute_risk_score(FireSuppressionState(dust_loading=1.0))
        assert rd > rc

    def test_clamped(self):
        for o2 in [0, 10, 21, 30, 50]:
            for h in [0, 0.5, 1]:
                for d in [0, 0.5, 1]:
                    r = compute_risk_score(FireSuppressionState(o2_kpa=o2,
                        humidity_fraction=h, dust_loading=d))
                    assert 0.0 <= r <= 1.0


class TestIgnition:
    def test_no_ignition_low_o2(self):
        s = FireSuppressionState(o2_kpa=MIN_O2_FOR_COMBUSTION_KPA - 1)
        for _ in range(1000):
            assert not check_ignition(s, random.Random(42))

    def test_no_ignition_if_burning(self):
        assert not check_ignition(make_fire(), random.Random(42))

    def test_possible_high_o2(self):
        s = FireSuppressionState(o2_kpa=40, humidity_fraction=0.1, dust_loading=0.8)
        hits = sum(1 for i in range(10000) if check_ignition(s, random.Random(i)))
        assert hits > 0

    def test_humidity_reduces(self):
        dry = FireSuppressionState(o2_kpa=30, humidity_fraction=0, dust_loading=0.5)
        wet = FireSuppressionState(o2_kpa=30, humidity_fraction=1, dust_loading=0.5)
        dc = sum(1 for i in range(10000) if check_ignition(dry, random.Random(i)))
        wc = sum(1 for i in range(10000) if check_ignition(wet, random.Random(i)))
        assert wc <= dc


class TestFireGrowth:
    def test_intensity_increases(self):
        s = make_fire()
        s = tick(s); i1 = s.fire_intensity_kw
        s = tick(s); i2 = s.fire_intensity_kw
        assert i2 > i1

    def test_t_squared(self):
        s = tick(make_fire(fire_growth_alpha=ALPHA_MEDIUM))
        expected = ALPHA_MEDIUM * (60.0 ** 2) / 1000.0
        assert s.fire_intensity_kw <= expected * 1.1

    def test_self_extinguish_low_o2(self):
        s = tick(make_fire(o2_kpa=MIN_O2_FOR_COMBUSTION_KPA - 0.5))
        assert not s.fire_active
        assert any("self-extinguished" in a for a in s.alerts)

    def test_consumes_o2(self):
        s = make_fire()
        o2 = s.o2_kpa
        assert run_n(s, 5).o2_kpa < o2

    def test_produces_co2(self):
        s = make_fire()
        co2 = s.co2_kpa
        assert run_n(s, 5).co2_kpa > co2

    def test_produces_co(self):
        s = make_fire()
        co = s.co_kpa
        assert run_n(s, 5).co_kpa > co

    def test_slow_grows_slower(self):
        ss = run_n(make_fire(fire_growth_alpha=ALPHA_SLOW), 3)
        sf = run_n(make_fire(fire_growth_alpha=ALPHA_FAST), 3)
        assert ss.fire_intensity_kw < sf.fire_intensity_kw


class TestGasDynamics:
    def test_o2_decreases(self):
        assert run_n(make_fire(), 10).o2_kpa < NOMINAL_O2_KPA

    def test_o2_never_negative(self):
        for s in run_simulation(200, force_ignition_at=5):
            assert s.o2_kpa >= 0.0

    def test_co2_never_negative(self):
        for s in run_simulation(200, force_ignition_at=5):
            assert s.co2_kpa >= 0.0

    def test_co_never_negative(self):
        for s in run_simulation(200, force_ignition_at=5):
            assert s.co_kpa >= 0.0

    def test_co_scrubbed(self):
        s = run_n(make_fire(), 3)
        s.fire_active = False; s.fire_intensity_kw = 0
        co = s.co_kpa
        assert run_n(s, 20).co_kpa < co


class TestSmoke:
    def test_increases_during_fire(self):
        assert run_n(make_fire(), 5).smoke_opacity > 0

    def test_clamped_to_1(self):
        assert run_n(make_fire(), 100).smoke_opacity <= 1.0

    def test_dissipates_after(self):
        s = run_n(make_fire(), 5)
        s.fire_active = False; s.fire_intensity_kw = 0
        peak = s.smoke_opacity
        assert run_n(s, 50).smoke_opacity < peak

    def test_never_negative(self):
        for s in run_simulation(200, force_ignition_at=5):
            assert s.smoke_opacity >= 0


class TestDetection:
    def test_alarm_triggers(self):
        s = make_fire()
        for _ in range(50):
            s = tick(s)
            if s.alarm_active:
                break
        assert s.alarm_active

    def test_no_alarm_without_smoke(self):
        assert not run_n(FireSuppressionState(rng_seed=99999), 10).alarm_active

    def test_alarm_clears(self):
        s = run_n(make_fire(), 5)
        s.fire_active = False; s.fire_intensity_kw = 0
        s.smoke_opacity = 0; s.alarm_active = True
        assert not tick(s).alarm_active


class TestSuppression:
    def test_uses_charge(self):
        s = activate_suppression(FireSuppressionState())
        assert s.extinguisher_charges == EXTINGUISHER_CHARGES - 1
        assert s.suppression_active

    def test_no_charges(self):
        s = activate_suppression(FireSuppressionState(extinguisher_charges=0))
        assert not s.suppression_active

    def test_no_double(self):
        s = activate_suppression(FireSuppressionState())
        s = activate_suppression(s)
        assert any("already active" in a.lower() for a in s.alerts)

    def test_adds_co2(self):
        s = make_fire()
        co2 = s.co2_kpa
        s = activate_suppression(s)
        assert tick(s).co2_kpa > co2

    def test_smothers_fire(self):
        s = activate_suppression(make_fire())
        for _ in range(20):
            s = tick(s)
            if not s.fire_active:
                break
        assert not s.fire_active

    def test_duration_limited(self):
        s = activate_suppression(FireSuppressionState())
        assert not run_n(s, CHARGE_DURATION_MIN + 2).suppression_active

    def test_auto_after_detection(self):
        s = make_fire()
        for _ in range(50):
            s = tick(s)
        assert s.extinguisher_charges < EXTINGUISHER_CHARGES


class TestCrewImpact:
    def test_co_exposure(self):
        assert run_n(make_fire(), 10).crew_co_exposure > 0

    def test_smoke_exposure(self):
        assert run_n(make_fire(), 10).crew_smoke_exposure > 0

    def test_no_exposure_clean(self):
        s = run_n(FireSuppressionState(rng_seed=99999), 50)
        assert s.crew_co_exposure == 0 and s.crew_smoke_exposure == 0

    def test_lethal_co_alert(self):
        s = FireSuppressionState(co_kpa=CO_LETHAL_KPA + 0.01, rng_seed=99999)
        s.fire_active = True  # prevent scrubbing during tick
        s = tick(s)
        assert any("lethal" in a.lower() or "CRITICAL" in a for a in s.alerts)

    def test_dangerous_co_alert(self):
        s = FireSuppressionState(co_kpa=CO_DANGEROUS_KPA + 0.01, rng_seed=99999)
        s.fire_active = True  # prevent scrubbing during tick
        s = tick(s)
        assert any("danger" in a.lower() for a in s.alerts)


class TestExtinguisherMgmt:
    def test_recharge(self):
        s = recharge_extinguishers(FireSuppressionState(extinguisher_charges=1))
        assert s.extinguisher_charges == EXTINGUISHER_CHARGES

    def test_no_recharge_fire(self):
        s = recharge_extinguishers(make_fire())
        assert any("Cannot recharge" in a for a in s.alerts)

    def test_no_recharge_suppression(self):
        s = FireSuppressionState()
        s.suppression_active = True
        s = recharge_extinguishers(s)
        assert any("Cannot recharge" in a for a in s.alerts)

    def test_deplete_all(self):
        s = FireSuppressionState()
        for _ in range(EXTINGUISHER_CHARGES):
            s = activate_suppression(s)
            s.suppression_active = False
        assert s.extinguisher_charges == 0


class TestPostFire:
    def test_co_decreases(self):
        s = run_n(make_fire(), 5)
        s.fire_active = False; s.fire_intensity_kw = 0
        co = s.co_kpa
        assert run_n(s, 100).co_kpa < co

    def test_smoke_clears(self):
        s = run_n(make_fire(), 5)
        s.fire_active = False; s.fire_intensity_kw = 0
        assert run_n(s, 200).smoke_opacity < SMOKE_DETECTOR_THRESHOLD

    def test_ventilation_power(self):
        s = FireSuppressionState(rng_seed=99999)
        s.co_kpa = 0.05
        assert tick(s).power_draw_kw > DETECTION_POWER_KW


class TestFullLifecycle:
    def test_full_cycle(self):
        h = run_simulation(150, force_ignition_at=5)
        assert any(s.fire_active for s in h)
        assert any(not s.fire_active and s.fires_total > 0 for s in h[10:])
        assert h[-1].smoke_opacity < max(s.smoke_opacity for s in h)

    def test_multiple_fires(self):
        s = make_fire()
        s = run_n(s, 5)
        s.fire_active = False; s.fire_intensity_kw = 0
        s.suppression_active = False; s.alarm_active = False; s.smoke_opacity = 0
        s.fire_active = True; s.fires_total = 2; s.fire_duration_min = 0
        s = run_n(s, 5)
        assert s.fires_total == 2


class TestPhysicalBounds:
    def test_o2(self):
        for s in run_simulation(200, force_ignition_at=5):
            assert 0 <= s.o2_kpa <= 50

    def test_co2(self):
        for s in run_simulation(200, force_ignition_at=5):
            assert 0 <= s.co2_kpa <= 100

    def test_co(self):
        for s in run_simulation(200, force_ignition_at=5):
            assert 0 <= s.co_kpa <= 10

    def test_smoke(self):
        for s in run_simulation(200, force_ignition_at=5):
            assert 0 <= s.smoke_opacity <= 1

    def test_risk(self):
        for s in run_simulation(200, force_ignition_at=5):
            assert 0 <= s.risk_score <= 1

    def test_intensity(self):
        for s in run_simulation(200, force_ignition_at=5):
            assert s.fire_intensity_kw >= 0

    def test_power(self):
        for s in run_simulation(200, force_ignition_at=5):
            assert s.power_draw_kw > 0

    def test_tick_monotonic(self):
        h = run_simulation(50)
        for i in range(1, len(h)):
            assert h[i].tick == h[i-1].tick + 1


class TestConservation:
    def test_o2_to_products(self):
        s = make_fire()
        o2_0, co2_0, co_0 = s.o2_kpa, s.co2_kpa, s.co_kpa
        s = run_n(s, 5)
        if o2_0 - s.o2_kpa > 0:
            assert s.co2_kpa > co2_0 and s.co_kpa > co_0

    def test_no_free_energy(self):
        s = run_n(make_fire(o2_kpa=MIN_O2_FOR_COMBUSTION_KPA + 0.1), 20)
        assert not s.fire_active or s.o2_kpa >= MIN_O2_FOR_COMBUSTION_KPA


class TestEdgeCases:
    def test_fire_zero_o2(self):
        assert not tick(make_fire(o2_kpa=0)).fire_active

    def test_suppress_no_fire(self):
        s = tick(activate_suppression(FireSuppressionState(rng_seed=99999)))
        assert s.suppression_active or s.suppression_remaining_min == 0

    def test_high_o2_risk(self):
        assert compute_risk_score(FireSuppressionState(o2_kpa=45)) > 0.3

    def test_recharge_then_use(self):
        s = recharge_extinguishers(FireSuppressionState(extinguisher_charges=0))
        assert activate_suppression(s).suppression_active


class TestSimulation:
    def test_10(self):
        assert len(run_simulation(10)) == 10

    def test_100(self):
        assert len(run_simulation(100)) == 100

    def test_with_fire(self):
        h = run_simulation(100, force_ignition_at=10)
        assert any(s.fire_active for s in h)

    def test_500_no_crash(self):
        assert len(run_simulation(500, force_ignition_at=50)) == 500

    def test_deterministic(self):
        h1 = run_simulation(50, force_ignition_at=10)
        h2 = run_simulation(50, force_ignition_at=10)
        for a, b in zip(h1, h2):
            assert a.o2_kpa == b.o2_kpa and a.smoke_opacity == b.smoke_opacity


class TestClamp:
    def test_within(self):
        assert clamp(5, 0, 10) == 5

    def test_below(self):
        assert clamp(-1, 0, 10) == 0

    def test_above(self):
        assert clamp(15, 0, 10) == 10

