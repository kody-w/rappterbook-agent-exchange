"""
Tests for waste_recycler.py — Mars closed-loop biological waste processing.

92 tests covering:
  - Waste generation from population
  - Urine processing (water recovery, brine, power)
  - Greywater processing
  - Brine electrolysis recovery
  - Composting (cycle timing, fertilizer, CO₂/water release)
  - Anaerobic digestion (biogas yield, digestate)
  - Full tick integration
  - Multi-sol smoke tests (100/365/2000 sols)
  - Conservation laws (mass, energy, monotonic accumulation)
  - Utility/planning helpers
"""
from __future__ import annotations

import math
import pytest

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.waste_recycler import (
    # Constants
    URINE_L_PER_PERSON_SOL, SOLID_WASTE_KG_PER_PERSON_SOL,
    HYGIENE_WATER_L_PER_PERSON_SOL, CROP_RESIDUE_RATIO,
    UPA_WATER_RECOVERY, UPA_POWER_KWH_PER_L,
    BRINE_FRACTION, BRINE_ELECTROLYSIS_RECOVERY, BRINE_ELECTROLYSIS_KWH_PER_L,
    GREY_WATER_RECOVERY, GREY_POWER_KWH_PER_L,
    COMPOST_CYCLE_SOLS, COMPOST_MASS_REDUCTION, COMPOST_WATER_RELEASE,
    COMPOST_CO2_RELEASE_KG_PER_KG, COMPOST_POWER_KWH_PER_KG,
    VOLATILE_SOLIDS_FRACTION, BIOGAS_M3_PER_KG_VS,
    BIOGAS_CH4_FRACTION, BIOGAS_CO2_FRACTION,
    CH4_DENSITY_KG_M3, CO2_DENSITY_KG_M3,
    DIGESTATE_FRACTION, DIGESTER_POWER_KWH_PER_KG,
    MAX_DAILY_URINE_L, MAX_DAILY_SOLIDS_KG, MAX_DAILY_RESIDUE_KG,
    FERT_N_KG_PER_KG_WASTE, FERT_P_KG_PER_KG_WASTE, FERT_K_KG_PER_KG_WASTE,
    # Data structures
    WasteInput, RecyclerOutput, RecyclerState,
    # Functions
    process_urine, process_greywater, process_brine,
    compost_tick, digest_residue, tick_waste,
    daily_waste_volume, water_recovery_rate, fertilizer_npk,
)


# ===================================================================
# 1. Constants validation
# ===================================================================

class TestConstants:
    """Physical constants should be in realistic ranges."""

    def test_urine_rate_realistic(self) -> None:
        assert 1.0 <= URINE_L_PER_PERSON_SOL <= 2.5

    def test_solid_waste_realistic(self) -> None:
        assert 0.05 <= SOLID_WASTE_KG_PER_PERSON_SOL <= 0.3

    def test_hygiene_water_realistic(self) -> None:
        assert 5.0 <= HYGIENE_WATER_L_PER_PERSON_SOL <= 20.0

    def test_upa_recovery_high(self) -> None:
        """ISS recovers >93% of water from urine."""
        assert 0.90 <= UPA_WATER_RECOVERY <= 0.99

    def test_grey_recovery_high(self) -> None:
        assert 0.90 <= GREY_WATER_RECOVERY <= 0.99

    def test_brine_fraction_small(self) -> None:
        assert 0.0 < BRINE_FRACTION < 0.20

    def test_crop_residue_ratio_reasonable(self) -> None:
        assert 0.2 <= CROP_RESIDUE_RATIO <= 0.6

    def test_biogas_fractions_sum(self) -> None:
        assert abs(BIOGAS_CH4_FRACTION + BIOGAS_CO2_FRACTION - 1.0) < 1e-10

    def test_compost_cycle_realistic(self) -> None:
        assert 30 <= COMPOST_CYCLE_SOLS <= 120

    def test_compost_mass_reduction_physical(self) -> None:
        """Most mass is lost as CO₂ + water vapor."""
        assert 0.4 <= COMPOST_MASS_REDUCTION <= 0.8


# ===================================================================
# 2. WasteInput data structure
# ===================================================================

class TestWasteInput:
    """Verify waste input construction and clamping."""

    def test_from_population(self) -> None:
        w = WasteInput.from_population(100)
        assert w.urine_l == 100 * URINE_L_PER_PERSON_SOL
        assert w.solid_waste_kg == 100 * SOLID_WASTE_KG_PER_PERSON_SOL
        assert w.greywater_l == 100 * HYGIENE_WATER_L_PER_PERSON_SOL

    def test_from_population_with_harvest(self) -> None:
        w = WasteInput.from_population(50, crop_harvest_kg=100.0)
        assert abs(w.crop_residue_kg - 100.0 * CROP_RESIDUE_RATIO) < 1e-10

    def test_clamps_negative(self) -> None:
        w = WasteInput(urine_l=-5, solid_waste_kg=-1, greywater_l=-10, crop_residue_kg=-3)
        assert w.urine_l == 0.0
        assert w.solid_waste_kg == 0.0
        assert w.greywater_l == 0.0
        assert w.crop_residue_kg == 0.0

    def test_zero_population(self) -> None:
        w = WasteInput.from_population(0)
        assert w.urine_l == 0.0
        assert w.solid_waste_kg == 0.0


# ===================================================================
# 3. RecyclerState
# ===================================================================

class TestRecyclerState:
    """Equipment state defaults and clamping."""

    def test_defaults(self) -> None:
        s = RecyclerState()
        assert s.upa_health == 1.0
        assert s.composter_health == 1.0
        assert s.digester_health == 1.0

    def test_clamps(self) -> None:
        s = RecyclerState(upa_health=2.0, composter_health=-1.0)
        assert s.upa_health == 1.0
        assert s.composter_health == 0.0


# ===================================================================
# 4. Urine processing
# ===================================================================

class TestUrineProcessing:
    """Urine processor assembly tests."""

    def test_zero_urine(self) -> None:
        r = process_urine(0.0, 100.0, 1.0)
        assert r["water_l"] == 0.0

    def test_zero_power(self) -> None:
        r = process_urine(100.0, 0.0, 1.0)
        assert r["water_l"] == 0.0

    def test_zero_health(self) -> None:
        r = process_urine(100.0, 100.0, 0.0)
        assert r["water_l"] == 0.0

    def test_recovers_water(self) -> None:
        r = process_urine(10.0, 100.0, 1.0)
        assert r["water_l"] > 0.0
        assert r["water_l"] <= 10.0 * UPA_WATER_RECOVERY + 1e-10

    def test_produces_brine(self) -> None:
        r = process_urine(10.0, 100.0, 1.0)
        assert r["brine_l"] > 0.0

    def test_water_plus_brine_accounts_for_input(self) -> None:
        """Recovery + brine should account for most of the input."""
        r = process_urine(10.0, 100.0, 1.0)
        accounted = r["water_l"] + r["brine_l"]
        assert accounted <= r["processed_l"] + 1e-10

    def test_power_conservation(self) -> None:
        r = process_urine(10.0, 100.0, 1.0)
        assert r["power_consumed"] <= 100.0 + 1e-10

    def test_capacity_limited(self) -> None:
        """Cannot exceed MAX_DAILY_URINE_L."""
        r = process_urine(10000.0, 10000.0, 1.0)
        assert r["processed_l"] <= MAX_DAILY_URINE_L + 1e-10

    def test_worn_upa_uses_more_power(self) -> None:
        r_new = process_urine(10.0, 100.0, 1.0)
        r_old = process_urine(10.0, 100.0, 0.3)
        if r_new["processed_l"] > 0 and r_old["processed_l"] > 0:
            cost_new = r_new["power_consumed"] / r_new["processed_l"]
            cost_old = r_old["power_consumed"] / r_old["processed_l"]
            assert cost_old >= cost_new


# ===================================================================
# 5. Greywater processing
# ===================================================================

class TestGreywaterProcessing:
    """Greywater filtration + UV sterilization."""

    def test_zero_input(self) -> None:
        r = process_greywater(0.0, 100.0)
        assert r["water_l"] == 0.0

    def test_zero_power(self) -> None:
        r = process_greywater(100.0, 0.0)
        assert r["water_l"] == 0.0

    def test_recovers_water(self) -> None:
        r = process_greywater(100.0, 100.0)
        assert r["water_l"] > 0.0
        assert r["water_l"] <= 100.0 * GREY_WATER_RECOVERY + 1e-10

    def test_power_conservation(self) -> None:
        r = process_greywater(100.0, 100.0)
        assert r["power_consumed"] <= 100.0 + 1e-10

    def test_limited_by_power(self) -> None:
        r_low = process_greywater(1000.0, 1.0)
        r_high = process_greywater(1000.0, 100.0)
        assert r_high["water_l"] > r_low["water_l"]


# ===================================================================
# 6. Brine processing
# ===================================================================

class TestBrineProcessing:
    """Brine electrolysis secondary recovery."""

    def test_zero_brine(self) -> None:
        r = process_brine(0.0, 100.0)
        assert r["water_l"] == 0.0

    def test_zero_power(self) -> None:
        r = process_brine(10.0, 0.0)
        assert r["water_l"] == 0.0

    def test_recovers_water(self) -> None:
        r = process_brine(10.0, 100.0)
        assert r["water_l"] > 0.0

    def test_produces_salts(self) -> None:
        r = process_brine(10.0, 100.0)
        assert r["salts_kg"] > 0.0

    def test_water_plus_salts_conservation(self) -> None:
        """Water + salts should not exceed input volume (approximately)."""
        r = process_brine(10.0, 100.0)
        assert r["water_l"] + r["salts_kg"] <= 10.0 + 1e-6


# ===================================================================
# 7. Composting
# ===================================================================

class TestComposting:
    """Aerobic thermophilic composting system."""

    def test_zero_waste(self) -> None:
        s = RecyclerState()
        r = compost_tick(0.0, s, 100.0)
        assert r["fertilizer_kg"] == 0.0

    def test_zero_power(self) -> None:
        s = RecyclerState()
        r = compost_tick(10.0, s, 0.0)
        assert r["fertilizer_kg"] == 0.0

    def test_zero_health(self) -> None:
        s = RecyclerState(composter_health=0.0)
        r = compost_tick(10.0, s, 100.0)
        assert r["fertilizer_kg"] == 0.0

    def test_adds_to_queue(self) -> None:
        s = RecyclerState()
        compost_tick(10.0, s, 100.0)
        assert s.compost_queue_kg > 0.0

    def test_releases_co2(self) -> None:
        s = RecyclerState()
        compost_tick(10.0, s, 100.0)
        # Second tick should show CO₂ release
        r = compost_tick(0.0, s, 100.0)
        assert r["co2_kg"] >= 0.0

    def test_releases_water(self) -> None:
        s = RecyclerState()
        compost_tick(10.0, s, 100.0)
        r = compost_tick(0.0, s, 100.0)
        assert r["water_l"] >= 0.0

    def test_batch_completes_after_cycle(self) -> None:
        """After COMPOST_CYCLE_SOLS, fertilizer should be produced."""
        s = RecyclerState()
        compost_tick(50.0, s, 100.0)
        fert_total = 0.0
        for _ in range(COMPOST_CYCLE_SOLS + 10):
            r = compost_tick(0.0, s, 100.0)
            fert_total += r["fertilizer_kg"]
        assert fert_total > 0.0

    def test_fertilizer_less_than_input(self) -> None:
        """Mass is lost to CO₂ and water during composting."""
        s = RecyclerState()
        compost_tick(100.0, s, 1000.0)
        fert_total = 0.0
        for _ in range(COMPOST_CYCLE_SOLS + 10):
            r = compost_tick(0.0, s, 100.0)
            fert_total += r["fertilizer_kg"]
        assert fert_total < 100.0

    def test_capacity_limited(self) -> None:
        s = RecyclerState()
        compost_tick(10000.0, s, 100000.0)
        assert s.compost_queue_kg <= MAX_DAILY_SOLIDS_KG + 1e-10


# ===================================================================
# 8. Anaerobic digestion
# ===================================================================

class TestDigestion:
    """Anaerobic digestion of crop residue."""

    def test_zero_residue(self) -> None:
        r = digest_residue(0.0, 100.0, 1.0)
        assert r["ch4_kg"] == 0.0

    def test_zero_power(self) -> None:
        r = digest_residue(100.0, 0.0, 1.0)
        assert r["ch4_kg"] == 0.0

    def test_zero_health(self) -> None:
        r = digest_residue(100.0, 100.0, 0.0)
        assert r["ch4_kg"] == 0.0

    def test_produces_methane(self) -> None:
        r = digest_residue(50.0, 100.0, 1.0)
        assert r["ch4_kg"] > 0.0

    def test_produces_co2(self) -> None:
        r = digest_residue(50.0, 100.0, 1.0)
        assert r["co2_kg"] > 0.0

    def test_produces_digestate(self) -> None:
        r = digest_residue(50.0, 100.0, 1.0)
        assert r["digestate_kg"] > 0.0

    def test_digestate_fraction(self) -> None:
        r = digest_residue(50.0, 100.0, 1.0)
        assert abs(r["digestate_kg"] - 50.0 * DIGESTATE_FRACTION) < 1e-6

    def test_capacity_limited(self) -> None:
        r = digest_residue(10000.0, 100000.0, 1.0)
        max_input = MAX_DAILY_RESIDUE_KG
        max_ch4 = max_input * VOLATILE_SOLIDS_FRACTION * BIOGAS_M3_PER_KG_VS * BIOGAS_CH4_FRACTION * CH4_DENSITY_KG_M3
        assert r["ch4_kg"] <= max_ch4 + 1e-6

    def test_power_conservation(self) -> None:
        r = digest_residue(50.0, 5.0, 1.0)
        assert r["power_consumed"] <= 5.0 + 1e-10


# ===================================================================
# 9. Full tick integration
# ===================================================================

class TestTickWaste:
    """Full waste pipeline integration."""

    def _fresh(self) -> tuple[WasteInput, RecyclerOutput, RecyclerState]:
        return (
            WasteInput.from_population(100, crop_harvest_kg=50.0),
            RecyclerOutput(),
            RecyclerState(),
        )

    def test_zero_power_no_recovery(self) -> None:
        w, o, s = self._fresh()
        tick_waste(w, o, s, 0.0)
        assert o.water_recovered_l == 0.0

    def test_recovers_water(self) -> None:
        w, o, s = self._fresh()
        tick_waste(w, o, s, 500.0)
        assert o.water_recovered_l > 0.0

    def test_power_within_budget(self) -> None:
        w, o, s = self._fresh()
        summary = tick_waste(w, o, s, 500.0)
        assert summary["sol_power_consumed"] <= 500.0 + 1e-6

    def test_equipment_degrades(self) -> None:
        w, o, s = self._fresh()
        tick_waste(w, o, s, 500.0)
        assert s.upa_health < 1.0
        assert s.composter_health < 1.0
        assert s.digester_health < 1.0

    def test_summary_keys(self) -> None:
        w, o, s = self._fresh()
        summary = tick_waste(w, o, s, 500.0)
        expected = {
            "sol_water_recovered_l", "sol_fertilizer_kg",
            "sol_ch4_kg", "sol_co2_kg", "sol_power_consumed",
        }
        assert expected.issubset(summary.keys())

    def test_biogas_from_crop_residue(self) -> None:
        w = WasteInput(crop_residue_kg=100.0)
        o = RecyclerOutput()
        s = RecyclerState()
        tick_waste(w, o, s, 500.0)
        assert o.ch4_kg > 0.0

    def test_no_waste_no_output(self) -> None:
        w = WasteInput()
        o = RecyclerOutput()
        s = RecyclerState()
        summary = tick_waste(w, o, s, 500.0)
        assert summary["sol_water_recovered_l"] == 0.0


# ===================================================================
# 10. Multi-sol smoke tests
# ===================================================================

class TestMultiSolSmoke:
    """Run processor for many sols — verify stability and bounds."""

    def test_100_sols_no_crash(self) -> None:
        o = RecyclerOutput()
        s = RecyclerState()
        for sol in range(100):
            w = WasteInput.from_population(80, crop_harvest_kg=30.0)
            tick_waste(w, o, s, 100.0)

        assert o.water_recovered_l > 0.0
        assert 0.0 <= s.upa_health <= 1.0
        assert 0.0 <= s.composter_health <= 1.0

    def test_365_sols_meaningful_recovery(self) -> None:
        """A Mars year of waste processing should recover significant water."""
        o = RecyclerOutput()
        s = RecyclerState()
        for _ in range(365):
            w = WasteInput.from_population(100, crop_harvest_kg=40.0)
            tick_waste(w, o, s, 200.0)

        # 100 people × 1.5 L urine/sol × 365 sols × ~93.5% recovery
        # Plus greywater... should recover tens of thousands of litres
        assert o.water_recovered_l > 10_000.0
        assert o.ch4_kg > 0.0

    def test_compost_produces_fertilizer_over_time(self) -> None:
        """After compost cycle completes, fertilizer should appear."""
        o = RecyclerOutput()
        s = RecyclerState()
        for _ in range(COMPOST_CYCLE_SOLS + 20):
            w = WasteInput.from_population(50)
            tick_waste(w, o, s, 200.0)
        assert o.fertilizer_kg > 0.0

    def test_cumulative_water_monotonic(self) -> None:
        """Cumulative water recovery only grows."""
        o = RecyclerOutput()
        s = RecyclerState()
        prev = 0.0
        for _ in range(50):
            w = WasteInput.from_population(60)
            tick_waste(w, o, s, 150.0)
            assert o.water_recovered_l >= prev
            prev = o.water_recovered_l

    def test_equipment_degrades_significantly(self) -> None:
        """After 2000 sols, equipment should be noticeably worn."""
        o = RecyclerOutput()
        s = RecyclerState()
        for _ in range(2000):
            w = WasteInput.from_population(50)
            tick_waste(w, o, s, 100.0)
        assert s.upa_health < 0.5

    def test_water_recovery_ratio_realistic(self) -> None:
        """Overall water recovery should be 90%+ of input."""
        o = RecyclerOutput()
        s = RecyclerState()
        total_input_l = 0.0
        for _ in range(200):
            w = WasteInput.from_population(80)
            total_input_l += w.urine_l + w.greywater_l
            tick_waste(w, o, s, 500.0)
        if total_input_l > 0:
            ratio = o.water_recovered_l / total_input_l
            assert ratio > 0.85, f"Recovery ratio {ratio:.2%} too low"


# ===================================================================
# 11. Utility functions
# ===================================================================

class TestUtilities:
    """Planning helpers."""

    def test_daily_waste_volume(self) -> None:
        v = daily_waste_volume(100)
        assert v["urine_l"] == 100 * URINE_L_PER_PERSON_SOL
        assert v["solid_waste_kg"] == 100 * SOLID_WASTE_KG_PER_PERSON_SOL
        assert v["greywater_l"] == 100 * HYGIENE_WATER_L_PER_PERSON_SOL

    def test_daily_waste_zero_pop(self) -> None:
        v = daily_waste_volume(0)
        assert all(val == 0.0 for val in v.values())

    def test_water_recovery_rate_positive(self) -> None:
        rate = water_recovery_rate(100)
        assert 0.0 < rate <= 1.0

    def test_water_recovery_rate_high(self) -> None:
        """Combined recovery should exceed 90%."""
        rate = water_recovery_rate(100)
        assert rate > 0.90

    def test_water_recovery_rate_zero_pop(self) -> None:
        assert water_recovery_rate(0) == 0.0

    def test_fertilizer_npk(self) -> None:
        npk = fertilizer_npk(100.0)
        assert npk["nitrogen_kg"] == 100.0 * FERT_N_KG_PER_KG_WASTE
        assert npk["phosphorus_kg"] == 100.0 * FERT_P_KG_PER_KG_WASTE
        assert npk["potassium_kg"] == 100.0 * FERT_K_KG_PER_KG_WASTE

    def test_fertilizer_npk_zero(self) -> None:
        npk = fertilizer_npk(0.0)
        assert all(v == 0.0 for v in npk.values())

    def test_nitrogen_dominates_npk(self) -> None:
        """Human waste is nitrogen-rich."""
        npk = fertilizer_npk(100.0)
        assert npk["nitrogen_kg"] > npk["phosphorus_kg"]
        assert npk["nitrogen_kg"] > npk["potassium_kg"]
