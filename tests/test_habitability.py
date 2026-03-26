"""
tests/test_habitability.py -- Unit tests for Mars Habitability Index.

Community voted 53-0: ship code, not governance.
One file. One test. One merge.

Run: python -m pytest tests/test_habitability.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.habitability import (
    temperature_score, pressure_score, radiation_score,
    dust_score, water_score, habitability_index,
    evaluate_sol, mars_ambient_mhi, earth_like_mhi,
    HabitabilityReport, W_TEMPERATURE, W_PRESSURE,
    W_RADIATION, W_DUST, W_WATER,
)


class TestTemperatureScore:
    def test_earth_comfort_high(self) -> None:
        assert temperature_score(20.0) > 0.9

    def test_mars_extreme_low(self) -> None:
        assert temperature_score(-120.0) < 0.01

    def test_mars_mean_moderate(self) -> None:
        s = temperature_score(-60.0)
        assert 0.0 < s < 0.3

    def test_monotonic_increasing(self) -> None:
        prev = 0.0
        for t in range(-120, 45, 5):
            s = temperature_score(float(t))
            assert s >= prev - 1e-9
            prev = s

    def test_heat_stroke_penalty(self) -> None:
        assert temperature_score(60.0) < temperature_score(40.0)

    def test_bounded_0_1(self) -> None:
        for t in range(-200, 200, 7):
            s = temperature_score(float(t))
            assert 0.0 <= s <= 1.0


class TestPressureScore:
    def test_earth_sea_level_high(self) -> None:
        assert pressure_score(101.3) > 0.95

    def test_mars_ambient_near_zero(self) -> None:
        assert pressure_score(0.636) < 0.05

    def test_zero_pressure_zero(self) -> None:
        assert pressure_score(0.0) == 0.0

    def test_negative_pressure_zero(self) -> None:
        assert pressure_score(-10.0) == 0.0

    def test_monotonic(self) -> None:
        prev = 0.0
        for p in range(0, 120):
            s = pressure_score(float(p))
            assert s >= prev - 1e-9
            prev = s

    def test_bounded(self) -> None:
        for p in [0, 0.1, 1, 10, 50, 100, 500]:
            assert 0.0 <= pressure_score(float(p)) <= 1.0


class TestRadiationScore:
    def test_earth_background_high(self) -> None:
        assert radiation_score(0.01) > 0.9

    def test_zero_radiation_perfect(self) -> None:
        assert radiation_score(0.0) == 1.0

    def test_lethal_near_zero(self) -> None:
        assert radiation_score(5.0) < 0.05

    def test_mars_gcr_moderate(self) -> None:
        s = radiation_score(0.67)
        assert 0.2 < s < 0.8

    def test_monotonic_decreasing(self) -> None:
        prev = 1.0
        for r_10x in range(0, 100):
            r = r_10x * 0.1
            s = radiation_score(r)
            assert s <= prev + 1e-9
            prev = s

    def test_bounded(self) -> None:
        for r in [0, 0.01, 0.1, 0.5, 1, 5, 10, 100]:
            assert 0.0 <= radiation_score(r) <= 1.0


class TestDustScore:
    def test_clear_sky_perfect(self) -> None:
        assert dust_score(0.3) == 1.0
        assert dust_score(0.1) == 1.0

    def test_storm_zero(self) -> None:
        assert dust_score(6.0) == 0.0
        assert dust_score(10.0) == 0.0

    def test_monotonic_decreasing(self) -> None:
        prev = 1.0
        for d_10x in range(0, 70):
            d = d_10x * 0.1
            s = dust_score(d)
            assert s <= prev + 1e-9
            prev = s

    def test_bounded(self) -> None:
        for d in [0, 0.3, 1, 3, 5, 6, 10]:
            assert 0.0 <= dust_score(d) <= 1.0


class TestWaterScore:
    def test_zero_water_zero(self) -> None:
        assert water_score(0.0) == 0.0

    def test_comfortable_supply_perfect(self) -> None:
        assert water_score(50.0) == 1.0
        assert water_score(150.0) == 1.0

    def test_survival_minimum_low(self) -> None:
        s = water_score(2.0)
        assert 0.05 < s < 0.2

    def test_monotonic(self) -> None:
        prev = 0.0
        for w_10x in range(0, 600):
            w = w_10x * 0.1
            s = water_score(w)
            assert s >= prev - 1e-9
            prev = s

    def test_bounded(self) -> None:
        for w in [0, 0.5, 1, 2, 5, 10, 50, 100, 500]:
            assert 0.0 <= water_score(w) <= 1.0

    def test_sub_survival_very_low(self) -> None:
        assert water_score(1.0) < 0.1


class TestHabitabilityIndex:
    def test_earth_like_near_one(self) -> None:
        assert earth_like_mhi() > 0.95

    def test_mars_ambient_zero(self) -> None:
        assert mars_ambient_mhi() == 0.0

    def test_any_axis_zero_kills(self) -> None:
        base = dict(temp_c=20.0, pressure_kpa=101.3, radiation_msv_sol=0.01,
                    dust_opacity=0.1, water_liters_per_person_sol=100.0)
        assert habitability_index(**dict(base, water_liters_per_person_sol=0.0)) == 0.0
        assert habitability_index(**dict(base, pressure_kpa=0.0)) == 0.0

    def test_bounded_0_1(self) -> None:
        cases = [(-120, 0.636, 0.67, 0.5, 0), (20, 101.3, 0.01, 0.1, 150),
                 (-60, 10, 0.5, 1.0, 5), (0, 50, 0.1, 0.3, 20)]
        for t, p, r, d, w in cases:
            mhi = habitability_index(t, p, r, d, w)
            assert 0.0 <= mhi <= 1.0

    def test_weights_sum_to_one(self) -> None:
        total = W_TEMPERATURE + W_PRESSURE + W_RADIATION + W_DUST + W_WATER
        assert abs(total - 1.0) < 1e-9

    def test_improving_axes_improves_mhi(self) -> None:
        bad = habitability_index(-60, 5, 2.0, 3.0, 3)
        good = habitability_index(15, 80, 0.05, 0.2, 40)
        assert good > bad

    def test_partial_terraforming(self) -> None:
        partial = habitability_index(-20, 20, 0.3, 0.5, 15)
        assert 0.05 < partial < 0.95

    def test_deterministic(self) -> None:
        a = habitability_index(-40, 30, 0.5, 1.0, 10)
        b = habitability_index(-40, 30, 0.5, 1.0, 10)
        assert a == b


class TestHabitabilityReport:
    def test_evaluate_sol_returns_report(self) -> None:
        report = evaluate_sol(100, -40.0, 10.0, 0.5, 0.8, 10.0)
        assert isinstance(report, HabitabilityReport)
        assert report.sol == 100

    def test_report_mhi_matches_direct(self) -> None:
        report = evaluate_sol(50, -30.0, 20.0, 0.4, 0.5, 15.0)
        direct = habitability_index(-30.0, 20.0, 0.4, 0.5, 15.0)
        assert report.mhi == direct

    def test_report_json_serializable(self) -> None:
        report = evaluate_sol(200, 10.0, 50.0, 0.1, 0.3, 30.0)
        d = report.to_dict()
        text = json.dumps(d)
        rt = json.loads(text)
        assert rt["sol"] == 200
        assert "breakdown" in rt

    def test_breakdown_all_bounded(self) -> None:
        report = evaluate_sol(1, -60.0, 0.636, 0.67, 0.5, 5.0)
        d = report.to_dict()
        for key, val in d["breakdown"].items():
            assert 0.0 <= val <= 1.0

    def test_breakdown_keys(self) -> None:
        report = evaluate_sol(1, 0, 50, 0.1, 0.3, 20)
        d = report.to_dict()
        expected = {"temperature", "pressure", "radiation", "dust", "water"}
        assert set(d["breakdown"].keys()) == expected


class TestPhysicalBounds:
    def test_exhaustive_sweep(self) -> None:
        temps = [-150, -120, -60, -10, 0, 20, 45, 75]
        pressures = [0, 0.636, 6.3, 30, 101.3, 200]
        radiations = [0, 0.01, 0.67, 1, 5, 50]
        dusts = [0, 0.3, 1, 3, 6, 10]
        waters = [0, 1, 2, 10, 50, 150]
        for t in temps:
            for p in pressures:
                for r in radiations:
                    for d in dusts:
                        for w in waters:
                            mhi = habitability_index(t, p, r, d, w)
                            assert 0.0 <= mhi <= 1.0

    def test_monotonicity_per_axis(self) -> None:
        base = dict(temp_c=-30.0, pressure_kpa=20.0, radiation_msv_sol=0.5,
                    dust_opacity=1.0, water_liters_per_person_sol=10.0)
        base_mhi = habitability_index(**base)
        assert habitability_index(**dict(base, temp_c=10.0)) >= base_mhi - 1e-9
        assert habitability_index(**dict(base, pressure_kpa=80.0)) >= base_mhi - 1e-9
        assert habitability_index(**dict(base, radiation_msv_sol=0.05)) >= base_mhi - 1e-9
        assert habitability_index(**dict(base, dust_opacity=0.2)) >= base_mhi - 1e-9
        assert habitability_index(**dict(base, water_liters_per_person_sol=40.0)) >= base_mhi - 1e-9
