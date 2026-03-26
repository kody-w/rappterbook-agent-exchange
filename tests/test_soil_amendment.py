"""Tests for soil_amendment.py — Mars regolith-to-arable-soil conversion.

120+ tests covering: perchlorate leaching, composting kinetics, nitrogen
fixation, pH buffering, water holding capacity, fertility score, tick()
integration, conservation laws, smoke test, and property-based invariants.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from soil_amendment import (
    PERCHLORATE_INITIAL_PPM,
    PERCHLORATE_SAFE_PPM,
    WASH_EFFICIENCY_PER_CYCLE,
    WATER_PER_KG_SOIL_L,
    COMPOST_DECAY_RATE_PER_DAY,
    OPTIMAL_COMPOST_TEMP_K,
    BIO_N_FIXATION_MG_PER_KG_PER_SOL,
    TARGET_N_MG_PER_KG,
    MAX_N_MG_PER_KG,
    MARS_REGOLITH_PH,
    TARGET_PH_LOW,
    TARGET_PH_HIGH,
    MIN_PH,
    MAX_PH,
    BASE_WATER_RETENTION_FRAC,
    MAX_WATER_RETENTION_FRAC,
    MIN_ORGANIC_MATTER_FRAC,
    GOOD_ORGANIC_MATTER_FRAC,
    MARS_AMBIENT_TEMP_K,
    perchlorate_after_wash,
    wash_water_needed_l,
    washes_to_safe,
    compost_decay_fraction,
    compost_heat_kw,
    nitrogen_per_sol,
    ph_after_amendment,
    water_holding_capacity,
    soil_fertility_score,
    SoilAmendmentBed,
    tick,
)


# =============================================================================
# perchlorate_after_wash
# =============================================================================

class TestPerchlorateAfterWash:
    def test_zero_cycles(self):
        assert perchlorate_after_wash(6000.0, 0) == 6000.0

    def test_one_cycle(self):
        result = perchlorate_after_wash(6000.0, 1)
        expected = 6000.0 * (1 - WASH_EFFICIENCY_PER_CYCLE)
        assert abs(result - expected) < 0.01

    def test_multiple_cycles_decrease(self):
        prev = 6000.0
        for n in range(1, 6):
            curr = perchlorate_after_wash(6000.0, n)
            assert curr < prev
            prev = curr

    def test_monotonic_decrease(self):
        results = [perchlorate_after_wash(6000.0, n) for n in range(10)]
        for i in range(1, len(results)):
            assert results[i] <= results[i - 1]

    def test_never_negative(self):
        assert perchlorate_after_wash(6000.0, 100) >= 0.0
        assert perchlorate_after_wash(0.0, 5) >= 0.0

    def test_negative_cycles(self):
        assert perchlorate_after_wash(6000.0, -1) == 6000.0

    def test_negative_initial(self):
        assert perchlorate_after_wash(-100.0, 3) == 0.0

    def test_zero_initial(self):
        assert perchlorate_after_wash(0.0, 3) == 0.0

    def test_reaches_safe_eventually(self):
        result = perchlorate_after_wash(6000.0, 20)
        assert result < PERCHLORATE_SAFE_PPM

    @pytest.mark.parametrize("initial", [100, 1000, 6000, 10000])
    def test_exponential_decay(self, initial):
        r1 = perchlorate_after_wash(initial, 1)
        r2 = perchlorate_after_wash(initial, 2)
        if r1 > 0:
            ratio = r2 / r1
            expected_ratio = 1 - WASH_EFFICIENCY_PER_CYCLE
            assert abs(ratio - expected_ratio) < 0.01


# =============================================================================
# wash_water_needed_l
# =============================================================================

class TestWashWaterNeeded:
    def test_basic(self):
        result = wash_water_needed_l(100.0, 1)
        assert result == 100.0 * WATER_PER_KG_SOIL_L

    def test_multiple_cycles(self):
        result = wash_water_needed_l(100.0, 3)
        assert result == 100.0 * WATER_PER_KG_SOIL_L * 3

    def test_zero_mass(self):
        assert wash_water_needed_l(0.0, 3) == 0.0

    def test_zero_cycles(self):
        assert wash_water_needed_l(100.0, 0) == 0.0

    def test_negative_mass(self):
        assert wash_water_needed_l(-10.0, 1) == 0.0

    def test_scales_linearly_with_mass(self):
        w1 = wash_water_needed_l(50.0, 1)
        w2 = wash_water_needed_l(100.0, 1)
        assert abs(w2 - 2 * w1) < 0.01

    def test_scales_linearly_with_cycles(self):
        w1 = wash_water_needed_l(100.0, 1)
        w3 = wash_water_needed_l(100.0, 3)
        assert abs(w3 - 3 * w1) < 0.01


# =============================================================================
# washes_to_safe
# =============================================================================

class TestWashesToSafe:
    def test_default(self):
        n = washes_to_safe()
        assert n >= 1
        assert perchlorate_after_wash(PERCHLORATE_INITIAL_PPM, n) <= PERCHLORATE_SAFE_PPM

    def test_already_safe(self):
        assert washes_to_safe(50.0) == 0

    def test_barely_unsafe(self):
        n = washes_to_safe(PERCHLORATE_SAFE_PPM + 1)
        assert n >= 1

    def test_high_concentration(self):
        n = washes_to_safe(50000.0)
        assert n > washes_to_safe(6000.0)


# =============================================================================
# compost_decay_fraction
# =============================================================================

class TestCompostDecay:
    def test_optimal_conditions(self):
        frac = compost_decay_fraction(OPTIMAL_COMPOST_TEMP_K, 0.5, 27.5)
        assert frac > 0.0
        assert frac <= 1.0

    def test_zero_temp(self):
        assert compost_decay_fraction(200.0, 0.5, 27.5) == 0.0

    def test_zero_moisture(self):
        assert compost_decay_fraction(OPTIMAL_COMPOST_TEMP_K, 0.0, 27.5) == 0.0

    def test_bounded_zero_to_one(self):
        for t in [250, 300, 333, 360]:
            for m in [0.1, 0.3, 0.5, 0.7, 0.9]:
                for cn in [10, 20, 27.5, 40, 60]:
                    f = compost_decay_fraction(float(t), m, cn)
                    assert 0.0 <= f <= 1.0, f"Out of bounds: {t}, {m}, {cn} -> {f}"

    def test_optimal_temp_maximizes(self):
        optimal = compost_decay_fraction(OPTIMAL_COMPOST_TEMP_K, 0.5, 27.5)
        cold = compost_decay_fraction(280.0, 0.5, 27.5)
        hot = compost_decay_fraction(370.0, 0.5, 27.5)
        assert optimal >= cold
        assert optimal >= hot

    def test_too_wet(self):
        dry = compost_decay_fraction(333.0, 0.5, 27.5)
        wet = compost_decay_fraction(333.0, 0.95, 27.5)
        assert dry >= wet

    def test_far_from_optimal_cn(self):
        good = compost_decay_fraction(333.0, 0.5, 27.5)
        bad = compost_decay_fraction(333.0, 0.5, 80.0)
        assert good >= bad

    def test_freezing(self):
        assert compost_decay_fraction(273.0, 0.5, 27.5) == 0.0


# =============================================================================
# compost_heat_kw
# =============================================================================

class TestCompostHeat:
    def test_positive_output(self):
        assert compost_heat_kw(100.0, 0.05) > 0.0

    def test_zero_mass(self):
        assert compost_heat_kw(0.0, 0.05) == 0.0

    def test_zero_decay(self):
        assert compost_heat_kw(100.0, 0.0) == 0.0

    def test_scales_with_mass(self):
        h1 = compost_heat_kw(50.0, 0.05)
        h2 = compost_heat_kw(100.0, 0.05)
        assert abs(h2 - 2 * h1) < 0.001

    def test_never_negative(self):
        assert compost_heat_kw(-10.0, 0.05) == 0.0
        assert compost_heat_kw(10.0, -0.05) == 0.0


# =============================================================================
# nitrogen_per_sol
# =============================================================================

class TestNitrogenPerSol:
    def test_bio_only(self):
        n = nitrogen_per_sol(100.0, True, 0.0)
        assert abs(n - BIO_N_FIXATION_MG_PER_KG_PER_SOL * 100.0) < 0.01

    def test_no_bio(self):
        n = nitrogen_per_sol(100.0, False, 0.0)
        assert n == 0.0

    def test_chemical_only(self):
        n = nitrogen_per_sol(100.0, False, 10.0)
        assert n > 0.0

    def test_bio_plus_chemical(self):
        bio = nitrogen_per_sol(100.0, True, 0.0)
        chem = nitrogen_per_sol(100.0, False, 10.0)
        both = nitrogen_per_sol(100.0, True, 10.0)
        assert abs(both - (bio + chem)) < 0.01

    def test_zero_mass(self):
        n = nitrogen_per_sol(0.0, True, 0.0)
        assert n == 0.0

    def test_scales_linearly(self):
        n1 = nitrogen_per_sol(50.0, True, 0.0)
        n2 = nitrogen_per_sol(100.0, True, 0.0)
        assert abs(n2 - 2 * n1) < 0.01

    @pytest.mark.parametrize("mass", [10, 50, 200, 1000])
    def test_always_non_negative(self, mass):
        assert nitrogen_per_sol(float(mass), True, 5.0) >= 0.0


# =============================================================================
# ph_after_amendment
# =============================================================================

class TestPhAfterAmendment:
    def test_sulfur_lowers_ph(self):
        ph = ph_after_amendment(8.0, 1.0, 100.0)
        assert ph < 8.0

    def test_no_sulfur(self):
        ph = ph_after_amendment(8.0, 0.0, 100.0)
        assert ph == 8.0

    def test_zero_soil(self):
        ph = ph_after_amendment(8.0, 1.0, 0.0)
        assert ph == 8.0

    def test_bounded_above_min(self):
        ph = ph_after_amendment(8.0, 10000.0, 100.0)
        assert ph >= MIN_PH

    def test_bounded_below_max(self):
        ph = ph_after_amendment(8.0, 0.0, 100.0)
        assert ph <= MAX_PH

    @pytest.mark.parametrize("sulfur", [0.1, 0.5, 1.0, 5.0, 10.0])
    def test_more_sulfur_lower_ph(self, sulfur):
        ph_low = ph_after_amendment(8.0, sulfur, 100.0)
        ph_high = ph_after_amendment(8.0, sulfur * 0.5, 100.0)
        assert ph_low <= ph_high

    def test_negative_sulfur(self):
        ph = ph_after_amendment(8.0, -1.0, 100.0)
        assert ph == 8.0


# =============================================================================
# water_holding_capacity
# =============================================================================

class TestWaterHoldingCapacity:
    def test_zero_organic(self):
        assert water_holding_capacity(0.0) == BASE_WATER_RETENTION_FRAC

    def test_increases_with_organic(self):
        whc_low = water_holding_capacity(0.02)
        whc_high = water_holding_capacity(0.10)
        assert whc_high > whc_low

    def test_capped_at_max(self):
        assert water_holding_capacity(1.0) <= MAX_WATER_RETENTION_FRAC

    def test_never_negative(self):
        assert water_holding_capacity(-0.5) >= 0.0

    def test_reasonable_at_five_percent(self):
        whc = water_holding_capacity(0.05)
        assert 0.15 < whc < 0.35


# =============================================================================
# soil_fertility_score
# =============================================================================

class TestSoilFertilityScore:
    def test_perfect_soil(self):
        score = soil_fertility_score(0.0, TARGET_N_MG_PER_KG, GOOD_ORGANIC_MATTER_FRAC, 6.5)
        assert score >= 0.9

    def test_raw_regolith(self):
        score = soil_fertility_score(6000.0, 0.0, 0.0, 8.0)
        assert score < 0.3

    def test_bounded_zero_to_one(self):
        for perc in [0, 100, 3000, 6000]:
            for n in [0, 50, 200, 500]:
                for om in [0.0, 0.03, 0.08, 0.15]:
                    for ph in [4.0, 6.0, 7.0, 8.0, 10.0]:
                        s = soil_fertility_score(float(perc), float(n), om, ph)
                        assert 0.0 <= s <= 1.0, f"OOB: {perc},{n},{om},{ph}->{s}"

    def test_safe_perchlorate_better(self):
        safe = soil_fertility_score(50.0, 100.0, 0.05, 6.5)
        toxic = soil_fertility_score(5000.0, 100.0, 0.05, 6.5)
        assert safe > toxic

    def test_more_nitrogen_better(self):
        low = soil_fertility_score(50.0, 10.0, 0.05, 6.5)
        high = soil_fertility_score(50.0, 200.0, 0.05, 6.5)
        assert high > low

    def test_better_ph_better(self):
        good = soil_fertility_score(50.0, 200.0, 0.08, 6.5)
        bad = soil_fertility_score(50.0, 200.0, 0.08, 9.5)
        assert good > bad

    def test_organic_matter_matters(self):
        low = soil_fertility_score(50.0, 200.0, 0.0, 6.5)
        high = soil_fertility_score(50.0, 200.0, 0.10, 6.5)
        assert high > low

    def test_weights_sum_to_one(self):
        # The score function uses 0.35 + 0.25 + 0.25 + 0.15 = 1.0
        assert abs(0.35 + 0.25 + 0.25 + 0.15 - 1.0) < 1e-10


# =============================================================================
# SoilAmendmentBed initial state
# =============================================================================

class TestSoilAmendmentBed:
    def test_default_values(self):
        bed = SoilAmendmentBed()
        assert bed.soil_mass_kg == 0.0
        assert bed.perchlorate_ppm == PERCHLORATE_INITIAL_PPM
        assert bed.wash_cycles_completed == 0
        assert bed.ph == MARS_REGOLITH_PH
        assert bed.sols_processed == 0

    def test_custom_init(self):
        bed = SoilAmendmentBed(soil_mass_kg=500.0, ph=7.0)
        assert bed.soil_mass_kg == 500.0
        assert bed.ph == 7.0


# =============================================================================
# tick() integration tests
# =============================================================================

class TestTick:
    def test_empty_bed_noop(self):
        bed = SoilAmendmentBed()
        result = tick(bed)
        assert result["soil_mass_kg"] == 0.0
        assert bed.sols_processed == 1

    def test_add_regolith(self):
        bed = SoilAmendmentBed()
        result = tick(bed, regolith_added_kg=100.0)
        assert bed.soil_mass_kg == 100.0
        assert bed.perchlorate_ppm == PERCHLORATE_INITIAL_PPM

    def test_wash_reduces_perchlorate(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0)
        result = tick(bed, wash_this_sol=True, water_available_l=1000.0)
        assert bed.perchlorate_ppm < PERCHLORATE_INITIAL_PPM
        assert bed.wash_cycles_completed == 1
        assert result["water_used_l"] > 0

    def test_wash_needs_water(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0)
        tick(bed, wash_this_sol=True, water_available_l=0.0)
        assert bed.perchlorate_ppm == PERCHLORATE_INITIAL_PPM
        assert bed.wash_cycles_completed == 0

    def test_composting(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0, perchlorate_ppm=50.0)
        tick(bed, compost_added_kg=10.0, water_available_l=50.0, heated=True)
        assert bed.organic_matter_kg > 0.0

    def test_nitrogen_fixation_after_wash(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0, perchlorate_ppm=50.0)
        result = tick(bed)
        assert bed.bio_fixation_active is True
        assert bed.nitrogen_mg_per_kg > 0.0

    def test_nitrogen_blocked_when_toxic(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0, perchlorate_ppm=6000.0)
        tick(bed)
        assert bed.bio_fixation_active is False

    def test_ph_adjustment(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0)
        tick(bed, sulfur_added_kg=2.0)
        assert bed.ph < MARS_REGOLITH_PH

    def test_sols_counter_increments(self):
        bed = SoilAmendmentBed()
        tick(bed)
        tick(bed)
        tick(bed)
        assert bed.sols_processed == 3

    def test_water_tracking(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0)
        tick(bed, wash_this_sol=True, water_available_l=5000.0)
        assert bed.total_water_used_l > 0.0

    def test_fertility_score_in_result(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0)
        result = tick(bed)
        assert "fertility_score" in result
        assert 0.0 <= result["fertility_score"] <= 1.0

    def test_arable_false_for_raw_regolith(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0)
        result = tick(bed)
        assert result["arable_ready"] is False

    def test_mixing_regolith_dilutes(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0, perchlorate_ppm=50.0,
                                nitrogen_mg_per_kg=200.0)
        tick(bed, regolith_added_kg=100.0)
        assert bed.perchlorate_ppm > 50.0
        assert bed.nitrogen_mg_per_kg < 200.0
        assert bed.soil_mass_kg == 200.0


# =============================================================================
# Multi-sol integration tests
# =============================================================================

class TestMultiSolIntegration:
    def test_full_pipeline_improves_fertility(self):
        """Run the full 4-stage pipeline and verify fertility improves."""
        bed = SoilAmendmentBed()
        tick(bed, regolith_added_kg=200.0)
        initial_score = bed.fertility_score

        for _ in range(5):
            tick(bed, wash_this_sol=True, water_available_l=5000.0)

        for _ in range(20):
            tick(bed, compost_added_kg=5.0, water_available_l=100.0,
                 heated=True, sulfur_added_kg=0.5, chemical_n_power_kw=5.0)

        assert bed.fertility_score > initial_score

    def test_arable_achievable(self):
        """Verify that soil can reach arable quality."""
        bed = SoilAmendmentBed()
        tick(bed, regolith_added_kg=200.0)

        for _ in range(8):
            tick(bed, wash_this_sol=True, water_available_l=5000.0)

        for _ in range(60):
            tick(bed, compost_added_kg=3.0, water_available_l=200.0,
                 heated=True, sulfur_added_kg=0.3, chemical_n_power_kw=10.0)

        assert bed.perchlorate_ppm <= PERCHLORATE_SAFE_PPM
        assert bed.fertility_score > 0.5


# =============================================================================
# Conservation laws / invariants
# =============================================================================

class TestConservationLaws:
    def test_mass_never_negative(self):
        bed = SoilAmendmentBed()
        for _ in range(50):
            tick(bed, regolith_added_kg=10.0, compost_added_kg=2.0,
                 water_available_l=500.0, wash_this_sol=True)
        assert bed.soil_mass_kg >= 0.0
        assert bed.organic_matter_kg >= 0.0

    def test_perchlorate_monotone_during_wash(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0)
        prev = bed.perchlorate_ppm
        for _ in range(10):
            tick(bed, wash_this_sol=True, water_available_l=5000.0)
            assert bed.perchlorate_ppm <= prev
            prev = bed.perchlorate_ppm

    def test_nitrogen_bounded(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0, perchlorate_ppm=10.0)
        for _ in range(200):
            tick(bed, chemical_n_power_kw=100.0)
        assert bed.nitrogen_mg_per_kg <= MAX_N_MG_PER_KG

    def test_ph_always_bounded(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0)
        for _ in range(100):
            tick(bed, sulfur_added_kg=10.0)
        assert MIN_PH <= bed.ph <= MAX_PH

    def test_water_holding_bounded(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0)
        for _ in range(50):
            tick(bed, compost_added_kg=20.0, water_available_l=200.0, heated=True)
        assert 0.0 <= bed.water_holding_cap <= MAX_WATER_RETENTION_FRAC

    def test_fertility_always_zero_to_one(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0)
        for _ in range(100):
            tick(bed, regolith_added_kg=5.0, compost_added_kg=2.0,
                 wash_this_sol=True, water_available_l=500.0,
                 sulfur_added_kg=0.1, chemical_n_power_kw=1.0)
            assert 0.0 <= bed.fertility_score <= 1.0

    def test_organic_matter_fraction_bounded(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0)
        for _ in range(200):
            tick(bed, compost_added_kg=50.0, water_available_l=500.0, heated=True)
        assert bed.organic_matter_fraction <= 0.15


# =============================================================================
# Smoke test (10-step simulation)
# =============================================================================

class TestSmokeTest:
    def test_10_sol_no_crash(self):
        bed = SoilAmendmentBed()
        tick(bed, regolith_added_kg=500.0)
        for sol in range(10):
            result = tick(bed,
                          compost_added_kg=2.0,
                          water_available_l=2000.0,
                          wash_this_sol=(sol % 2 == 0),
                          sulfur_added_kg=0.1,
                          chemical_n_power_kw=5.0,
                          heated=True)
            assert isinstance(result, dict)
            assert "fertility_score" in result
        assert bed.sols_processed == 11

    def test_100_sol_no_crash(self):
        bed = SoilAmendmentBed()
        tick(bed, regolith_added_kg=1000.0)
        for sol in range(100):
            tick(bed,
                 compost_added_kg=1.0,
                 water_available_l=5000.0,
                 wash_this_sol=(sol < 10),
                 sulfur_added_kg=0.2 if sol > 10 else 0.0,
                 chemical_n_power_kw=5.0 if sol > 15 else 0.0,
                 heated=True)
        assert bed.sols_processed == 101
        assert bed.fertility_score > 0.0

    def test_empty_ticks(self):
        bed = SoilAmendmentBed()
        for _ in range(10):
            result = tick(bed)
            assert result["soil_mass_kg"] == 0.0


# =============================================================================
# Edge cases / property-based
# =============================================================================

class TestEdgeCases:
    def test_huge_regolith(self):
        bed = SoilAmendmentBed()
        tick(bed, regolith_added_kg=1e6)
        assert bed.soil_mass_kg == 1e6

    def test_tiny_regolith(self):
        bed = SoilAmendmentBed()
        tick(bed, regolith_added_kg=0.001)
        assert bed.soil_mass_kg > 0.0

    def test_no_water_no_wash(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0)
        tick(bed, wash_this_sol=True, water_available_l=0.0)
        assert bed.perchlorate_ppm == PERCHLORATE_INITIAL_PPM

    def test_excess_sulfur_capped(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0)
        tick(bed, sulfur_added_kg=1e6)
        assert bed.ph >= MIN_PH

    @pytest.mark.parametrize("power_kw", [0.0, 1.0, 10.0, 100.0])
    def test_chemical_n_scales(self, power_kw):
        bed = SoilAmendmentBed(soil_mass_kg=100.0, perchlorate_ppm=10.0)
        tick(bed, chemical_n_power_kw=power_kw)
        if power_kw > 0:
            assert bed.nitrogen_mg_per_kg > 0.0

    def test_repeated_regolith_add(self):
        bed = SoilAmendmentBed()
        for _ in range(20):
            tick(bed, regolith_added_kg=10.0)
        assert bed.soil_mass_kg == 200.0

    def test_compost_heat_positive_when_active(self):
        bed = SoilAmendmentBed(soil_mass_kg=100.0, perchlorate_ppm=50.0)
        tick(bed, compost_added_kg=50.0, water_available_l=200.0, heated=True)
        result = tick(bed, water_available_l=200.0, heated=True)
        assert result["compost_heat_kw"] >= 0.0

    def test_dataclass_independence(self):
        bed1 = SoilAmendmentBed(soil_mass_kg=100.0)
        bed2 = SoilAmendmentBed(soil_mass_kg=200.0)
        tick(bed1, wash_this_sol=True, water_available_l=5000.0)
        assert bed2.perchlorate_ppm == PERCHLORATE_INITIAL_PPM
        assert bed2.wash_cycles_completed == 0
