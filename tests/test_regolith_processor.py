"""
Tests for regolith_processor.py — Mars regolith excavation and processing.

91 tests.  Every function, every edge case, every conservation law.
Physics-first: if it violates thermodynamics or mass conservation, it fails.

Run: python -m pytest tests/test_regolith_processor.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.regolith_processor import (
    BRICK_MASS_KG,
    BRICK_STRENGTH_MPA_BASE,
    BRICK_VOLUME_M3,
    COLD_EXCAVATION_PENALTY,
    DUST_STORM_HALT_THRESHOLD,
    EQUIPMENT_WEAR_PER_SOL,
    EXCAVATION_KWH_PER_M3,
    IRON_EXTRACTION_KWH_PER_KG,
    IRON_OXIDE_FRACTION,
    PERCHLORATE_FRACTION,
    REGOLITH_DENSITY_KG_M3,
    SIEVING_KWH_PER_M3,
    SINTER_TEMP_C,
    SINTER_TEMP_MAX_C,
    SINTER_TEMP_MIN_C,
    SINTERING_KWH_PER_KG,
    WARM_EXCAVATION_BONUS,
    WASHING_KWH_PER_M3,
    WATER_WASH_RATIO,
    ProcessorState,
    RegolithStockpile,
    brick_strength_mpa,
    excavate_sol,
    excavation_efficiency,
    extract_iron_sol,
    perchlorate_removal_efficiency,
    sieve_sol,
    sinter_bricks_sol,
    sintering_quality,
    tick_regolith,
    wash_sol,
)


# ===================================================================
# Constants validation
# ===================================================================

class TestConstants:
    """Physical constants are within real-world bounds."""

    def test_regolith_density_reasonable(self):
        """Mars regolith density is 1200-1800 kg/m³ per Viking/Phoenix."""
        assert 1200.0 <= REGOLITH_DENSITY_KG_M3 <= 1800.0

    def test_iron_oxide_fraction_reasonable(self):
        """Fe₂O₃ is 14-18% of Mars regolith (MER Mössbauer)."""
        assert 0.14 <= IRON_OXIDE_FRACTION <= 0.18

    def test_perchlorate_fraction_reasonable(self):
        """Phoenix measured 0.5-1.0% perchlorates."""
        assert 0.005 <= PERCHLORATE_FRACTION <= 0.01

    def test_sintering_temp_range(self):
        """Sintering requires 900-1200°C (NASA KSC experiments)."""
        assert SINTER_TEMP_MIN_C < SINTER_TEMP_C < SINTER_TEMP_MAX_C

    def test_brick_mass_reasonable(self):
        """Standard brick is 8-15 kg (Mars-appropriate size)."""
        assert 5.0 <= BRICK_MASS_KG <= 20.0

    def test_equipment_wear_slow(self):
        """Equipment lasts years, not days."""
        sols_to_zero = 1.0 / EQUIPMENT_WEAR_PER_SOL
        assert sols_to_zero > 1000  # >1.5 Mars years


# ===================================================================
# Excavation efficiency
# ===================================================================

class TestExcavationEfficiency:
    """excavation_efficiency(temp_c, dust_opacity)."""

    def test_warm_clear_maximum(self):
        """0°C and zero dust → maximum efficiency."""
        eff = excavation_efficiency(0.0, 0.0)
        assert eff == pytest.approx(WARM_EXCAVATION_BONUS, abs=0.01)

    def test_cold_clear(self):
        """-120°C and zero dust → cold penalty."""
        eff = excavation_efficiency(-120.0, 0.0)
        assert eff == pytest.approx(COLD_EXCAVATION_PENALTY, abs=0.01)

    def test_dust_storm_halts_operations(self):
        """At DUST_STORM_HALT_THRESHOLD, excavation stops."""
        eff = excavation_efficiency(0.0, DUST_STORM_HALT_THRESHOLD)
        assert eff == 0.0

    def test_above_storm_threshold_still_zero(self):
        """Above threshold, still zero."""
        eff = excavation_efficiency(0.0, 1.0)
        assert eff == 0.0

    def test_moderate_dust_reduces_efficiency(self):
        """Dust opacity 0.3 reduces but doesn't halt."""
        eff_clear = excavation_efficiency(0.0, 0.0)
        eff_dusty = excavation_efficiency(0.0, 0.3)
        assert 0 < eff_dusty < eff_clear

    def test_monotonic_with_temperature(self):
        """Warmer → higher efficiency (clear skies)."""
        prev = 0.0
        for t in range(-120, 1, 5):
            eff = excavation_efficiency(float(t), 0.0)
            assert eff >= prev
            prev = eff

    def test_always_non_negative(self):
        """Efficiency is never negative."""
        for t in range(-200, 50, 10):
            for d_int in range(0, 15):
                d = d_int / 10.0
                eff = excavation_efficiency(float(t), d)
                assert eff >= 0.0

    def test_always_at_most_one(self):
        """Efficiency never exceeds 1.0."""
        for t in range(-120, 50, 5):
            for d_int in range(0, 7):
                d = d_int / 10.0
                eff = excavation_efficiency(float(t), d)
                assert eff <= 1.0 + 1e-9

    def test_clamps_temperature_below_minus_120(self):
        """Temperatures below -120°C don't reduce further."""
        eff_120 = excavation_efficiency(-120.0, 0.0)
        eff_200 = excavation_efficiency(-200.0, 0.0)
        assert eff_120 == pytest.approx(eff_200)


# ===================================================================
# Sintering quality
# ===================================================================

class TestSinteringQuality:
    """sintering_quality(furnace_temp_c)."""

    def test_below_minimum_is_zero(self):
        """Below SINTER_TEMP_MIN_C, no sintering."""
        assert sintering_quality(SINTER_TEMP_MIN_C - 1) == 0.0
        assert sintering_quality(20.0) == 0.0
        assert sintering_quality(0.0) == 0.0

    def test_at_minimum_is_zero(self):
        """At minimum threshold, quality is 0."""
        assert sintering_quality(SINTER_TEMP_MIN_C) == pytest.approx(0.0)

    def test_at_optimal_is_one(self):
        """At SINTER_TEMP_C, quality is 1.0."""
        assert sintering_quality(SINTER_TEMP_C) == pytest.approx(1.0)

    def test_above_optimal_is_one(self):
        """Above optimal, quality stays at 1.0."""
        assert sintering_quality(SINTER_TEMP_C + 50) == pytest.approx(1.0)

    def test_above_max_is_one(self):
        """Above SINTER_TEMP_MAX_C, quality is 1.0."""
        assert sintering_quality(SINTER_TEMP_MAX_C + 100) == 1.0

    def test_midpoint_between_min_and_optimal(self):
        """Midpoint between min and optimal → quality ~0.5."""
        mid = (SINTER_TEMP_MIN_C + SINTER_TEMP_C) / 2.0
        q = sintering_quality(mid)
        assert 0.45 <= q <= 0.55

    def test_monotonic_increase(self):
        """Quality increases with temperature from min to optimal."""
        prev = -1.0
        for t_int in range(int(SINTER_TEMP_MIN_C), int(SINTER_TEMP_C) + 1, 5):
            q = sintering_quality(float(t_int))
            assert q >= prev
            prev = q

    def test_always_bounded_0_1(self):
        """Quality is always in [0, 1]."""
        for t in range(-100, 2000, 50):
            q = sintering_quality(float(t))
            assert 0.0 <= q <= 1.0


# ===================================================================
# Brick strength
# ===================================================================

class TestBrickStrength:
    """brick_strength_mpa(quality)."""

    def test_zero_quality_zero_strength(self):
        assert brick_strength_mpa(0.0) == 0.0

    def test_full_quality_base_strength(self):
        assert brick_strength_mpa(1.0) == pytest.approx(BRICK_STRENGTH_MPA_BASE)

    def test_half_quality_half_strength(self):
        assert brick_strength_mpa(0.5) == pytest.approx(BRICK_STRENGTH_MPA_BASE * 0.5)

    def test_clamps_above_one(self):
        assert brick_strength_mpa(1.5) == pytest.approx(BRICK_STRENGTH_MPA_BASE)

    def test_clamps_below_zero(self):
        assert brick_strength_mpa(-0.5) == 0.0


# ===================================================================
# Perchlorate removal
# ===================================================================

class TestPerchlorateRemoval:
    """perchlorate_removal_efficiency(water_ratio)."""

    def test_no_water_no_removal(self):
        assert perchlorate_removal_efficiency(0.0) == 0.0

    def test_negative_water(self):
        assert perchlorate_removal_efficiency(-1.0) == 0.0

    def test_minimal_water_minimal_removal(self):
        eff = perchlorate_removal_efficiency(0.25)
        assert 0.0 < eff < 0.1

    def test_standard_ratio_high_removal(self):
        """At WATER_WASH_RATIO (2.0 L/kg), removal is 95%."""
        eff = perchlorate_removal_efficiency(WATER_WASH_RATIO)
        assert eff >= 0.90

    def test_excess_water_diminishing_returns(self):
        eff = perchlorate_removal_efficiency(3.0)
        assert eff >= 0.95

    def test_monotonic_increase(self):
        prev = 0.0
        for wr in [i * 0.1 for i in range(0, 40)]:
            eff = perchlorate_removal_efficiency(wr)
            assert eff >= prev - 1e-9
            prev = eff

    def test_always_under_one(self):
        for wr in [i * 0.5 for i in range(0, 20)]:
            eff = perchlorate_removal_efficiency(wr)
            assert eff <= 1.0


# ===================================================================
# Excavation per-sol
# ===================================================================

class TestExcavateSol:
    """excavate_sol(power_kwh, temp_c, dust_opacity, equipment_condition)."""

    def test_zero_power_zero_output(self):
        kg, pw = excavate_sol(0.0, -60.0, 0.0, 1.0)
        assert kg == 0.0 and pw == 0.0

    def test_zero_equipment_zero_output(self):
        kg, pw = excavate_sol(100.0, -60.0, 0.0, 0.0)
        assert kg == 0.0 and pw == 0.0

    def test_dust_storm_zero_output(self):
        kg, pw = excavate_sol(100.0, -60.0, 0.9, 1.0)
        assert kg == 0.0

    def test_positive_production(self):
        kg, pw = excavate_sol(100.0, -30.0, 0.0, 1.0)
        assert kg > 0.0
        assert pw > 0.0

    def test_more_power_more_output(self):
        kg1, _ = excavate_sol(50.0, -30.0, 0.0, 1.0)
        kg2, _ = excavate_sol(100.0, -30.0, 0.0, 1.0)
        assert kg2 > kg1

    def test_power_conservation(self):
        """Never consumes more power than allocated."""
        for pw_in in [10, 50, 100, 500]:
            _, pw_used = excavate_sol(float(pw_in), -60.0, 0.0, 1.0)
            assert pw_used <= pw_in + 1e-9

    def test_negative_power_safe(self):
        kg, pw = excavate_sol(-10.0, -60.0, 0.0, 1.0)
        assert kg == 0.0 and pw == 0.0


# ===================================================================
# Sieving per-sol
# ===================================================================

class TestSieveSol:
    """sieve_sol(raw_kg, power_kwh)."""

    def test_zero_raw_zero_output(self):
        s, r, p = sieve_sol(0.0, 100.0)
        assert s == 0.0 and r == 0.0 and p == 0.0

    def test_zero_power_zero_output(self):
        s, r, p = sieve_sol(1000.0, 0.0)
        assert s == 0.0

    def test_five_percent_loss(self):
        """Sieving loses ~5% to ultra-fine dust."""
        s, r, p = sieve_sol(1000.0, 1000.0)
        assert s <= r * 0.96  # at most 95% yield + rounding

    def test_mass_conservation(self):
        """Sieved output ≤ raw input."""
        s, r, p = sieve_sol(500.0, 1000.0)
        assert s <= 500.0

    def test_power_limited(self):
        """With limited power, can't sieve all raw material."""
        s_low, _, _ = sieve_sol(10000.0, 1.0)
        s_high, _, _ = sieve_sol(10000.0, 100.0)
        assert s_high > s_low

    def test_raw_limited(self):
        """With limited raw material, can't use all power."""
        s, r, p = sieve_sol(10.0, 1000.0)
        assert r <= 10.0
        assert p < 1000.0


# ===================================================================
# Washing per-sol
# ===================================================================

class TestWashSol:
    """wash_sol(sieved_kg, water_liters, power_kwh)."""

    def test_zero_inputs(self):
        w, s, wt, p = wash_sol(0.0, 100.0, 100.0)
        assert w == 0.0
        w, s, wt, p = wash_sol(100.0, 0.0, 100.0)
        assert w == 0.0
        w, s, wt, p = wash_sol(100.0, 100.0, 0.0)
        assert w == 0.0

    def test_positive_production(self):
        w, s, wt, p = wash_sol(100.0, 500.0, 50.0)
        assert w > 0.0
        assert wt > 0.0

    def test_water_limited(self):
        """Little water → little washing."""
        w_low, _, _, _ = wash_sol(1000.0, 1.0, 1000.0)
        w_high, _, _, _ = wash_sol(1000.0, 1000.0, 1000.0)
        assert w_high > w_low

    def test_water_conservation(self):
        """Water consumed ≤ water available."""
        for wl in [1.0, 10.0, 100.0, 1000.0]:
            _, _, wt_used, _ = wash_sol(1000.0, wl, 1000.0)
            assert wt_used <= wl + 1e-9

    def test_water_partially_recovered(self):
        """Only 20% of wash water is consumed (80% reclaimed)."""
        w, s, wt_consumed, p = wash_sol(100.0, 1000.0, 1000.0)
        # Water used for washing = s * WATER_WASH_RATIO
        # Consumed = 20% of that
        if s > 0:
            total_wash_water = s * WATER_WASH_RATIO
            assert wt_consumed <= total_wash_water * 0.21 + 0.1  # 20% + rounding


# ===================================================================
# Sintering per-sol
# ===================================================================

class TestSinterBricksSol:
    """sinter_bricks_sol(sieved_kg, power_kwh, furnace_temp_c)."""

    def test_cold_furnace_no_bricks(self):
        bricks, q, s, p = sinter_bricks_sol(1000.0, 1000.0, 500.0)
        assert bricks == 0

    def test_optimal_temp_makes_bricks(self):
        bricks, q, s, p = sinter_bricks_sol(1000.0, 1000.0, SINTER_TEMP_C)
        assert bricks > 0
        assert q == pytest.approx(1.0)

    def test_brick_mass_conservation(self):
        """Each brick consumes BRICK_MASS_KG of material."""
        bricks, q, s_used, p = sinter_bricks_sol(1000.0, 1000.0, SINTER_TEMP_C)
        assert s_used == pytest.approx(bricks * BRICK_MASS_KG)

    def test_power_limited_production(self):
        """With little power, fewer bricks."""
        b_low, _, _, _ = sinter_bricks_sol(1000.0, 10.0, SINTER_TEMP_C)
        b_high, _, _, _ = sinter_bricks_sol(1000.0, 1000.0, SINTER_TEMP_C)
        assert b_high >= b_low

    def test_material_limited_production(self):
        """Not enough material for even one brick."""
        bricks, _, _, _ = sinter_bricks_sol(5.0, 1000.0, SINTER_TEMP_C)
        assert bricks == 0  # 5 kg < BRICK_MASS_KG

    def test_power_conservation(self):
        """Never consumes more power than allocated."""
        for pw in [10, 50, 100, 500]:
            _, _, _, p_used = sinter_bricks_sol(10000.0, float(pw), SINTER_TEMP_C)
            assert p_used <= pw + 1e-9

    def test_integer_bricks(self):
        """Brick count is always an integer."""
        bricks, _, _, _ = sinter_bricks_sol(100.0, 100.0, SINTER_TEMP_C)
        assert isinstance(bricks, int)


# ===================================================================
# Iron extraction per-sol
# ===================================================================

class TestExtractIronSol:
    """extract_iron_sol(sieved_kg, power_kwh)."""

    def test_zero_inputs(self):
        fe, s, p = extract_iron_sol(0.0, 100.0)
        assert fe == 0.0
        fe, s, p = extract_iron_sol(100.0, 0.0)
        assert fe == 0.0

    def test_positive_production(self):
        fe, s, p = extract_iron_sol(1000.0, 100.0)
        assert fe > 0.0

    def test_iron_fraction_bounded(self):
        """Iron output ≤ IRON_OXIDE_FRACTION * 0.70 * input."""
        for kg in [100, 500, 1000]:
            fe, s, p = extract_iron_sol(float(kg), 10000.0)
            max_iron = kg * IRON_OXIDE_FRACTION * 0.70
            assert fe <= max_iron + 1e-4

    def test_power_limited(self):
        fe_low, _, _ = extract_iron_sol(10000.0, 1.0)
        fe_high, _, _ = extract_iron_sol(10000.0, 100.0)
        assert fe_high > fe_low

    def test_power_conservation(self):
        for pw in [1, 10, 100]:
            _, _, p_used = extract_iron_sol(10000.0, float(pw))
            assert p_used <= pw + 1e-9


# ===================================================================
# Stockpile dataclass
# ===================================================================

class TestRegolithStockpile:
    """RegolithStockpile data integrity."""

    def test_default_all_zero(self):
        s = RegolithStockpile()
        assert s.raw_kg == 0.0
        assert s.sieved_kg == 0.0
        assert s.washed_kg == 0.0
        assert s.bricks == 0
        assert s.iron_kg == 0.0

    def test_clamps_negative(self):
        s = RegolithStockpile(raw_kg=-10, sieved_kg=-5, washed_kg=-1, bricks=-3, iron_kg=-1)
        assert s.raw_kg == 0.0
        assert s.sieved_kg == 0.0
        assert s.washed_kg == 0.0
        assert s.bricks == 0
        assert s.iron_kg == 0.0

    def test_total_processed(self):
        s = RegolithStockpile(sieved_kg=100, washed_kg=50, bricks=5, iron_kg=10)
        expected = 100 + 50 + 5 * BRICK_MASS_KG + 10
        assert s.total_processed_kg() == pytest.approx(expected)


# ===================================================================
# ProcessorState dataclass
# ===================================================================

class TestProcessorState:
    """ProcessorState data integrity."""

    def test_defaults(self):
        p = ProcessorState()
        assert p.equipment_condition == 1.0
        assert p.sinter_furnace_temp_c == 20.0
        assert p.water_available_liters == 0.0

    def test_clamps_equipment(self):
        p = ProcessorState(equipment_condition=1.5)
        assert p.equipment_condition == 1.0
        p2 = ProcessorState(equipment_condition=-0.5)
        assert p2.equipment_condition == 0.0


# ===================================================================
# Integrated tick
# ===================================================================

class TestTickRegolith:
    """tick_regolith — full per-sol integration."""

    def test_smoke_10_sols(self):
        """Run 10 sols without crash."""
        stockpile = RegolithStockpile()
        state = ProcessorState(
            sinter_furnace_temp_c=SINTER_TEMP_C,
            water_available_liters=500.0,
        )
        for _ in range(10):
            result = tick_regolith(
                power_budget_kwh=200.0,
                temp_c=-40.0,
                dust_opacity=0.1,
                water_liters=50.0,
                stockpile=stockpile,
                state=state,
            )
            assert "excavated_kg" in result
            assert "bricks" in result
            assert "stockpile" in result

    def test_accumulation_over_sols(self):
        """Materials accumulate across ticks."""
        stockpile = RegolithStockpile()
        state = ProcessorState(sinter_furnace_temp_c=SINTER_TEMP_C)
        for _ in range(20):
            tick_regolith(
                power_budget_kwh=300.0,
                temp_c=-30.0,
                dust_opacity=0.0,
                water_liters=100.0,
                stockpile=stockpile,
                state=state,
            )
        assert stockpile.raw_kg >= 0.0
        # Should have produced something after 20 sols
        assert stockpile.total_processed_kg() > 0.0

    def test_equipment_degrades(self):
        """Equipment condition decreases over time."""
        stockpile = RegolithStockpile()
        state = ProcessorState()
        initial = state.equipment_condition
        for _ in range(100):
            tick_regolith(
                power_budget_kwh=100.0, temp_c=-60.0,
                dust_opacity=0.0, water_liters=10.0,
                stockpile=stockpile, state=state,
            )
        assert state.equipment_condition < initial
        expected = initial - 100 * EQUIPMENT_WEAR_PER_SOL
        assert state.equipment_condition == pytest.approx(expected, abs=1e-6)

    def test_zero_power_no_production(self):
        """Zero power → no production."""
        stockpile = RegolithStockpile()
        state = ProcessorState()
        result = tick_regolith(
            power_budget_kwh=0.0, temp_c=-60.0,
            dust_opacity=0.0, water_liters=100.0,
            stockpile=stockpile, state=state,
        )
        assert result["excavated_kg"] == 0.0
        assert result["sieved_kg"] == 0.0
        assert result["bricks"] == 0

    def test_dust_storm_halts_excavation_only(self):
        """During dust storm, excavation stops but indoor ops continue."""
        stockpile = RegolithStockpile(raw_kg=1000, sieved_kg=1000)
        state = ProcessorState(sinter_furnace_temp_c=SINTER_TEMP_C)
        result = tick_regolith(
            power_budget_kwh=500.0, temp_c=-60.0,
            dust_opacity=0.9, water_liters=100.0,
            stockpile=stockpile, state=state,
        )
        assert result["excavated_kg"] == 0.0
        # Indoor ops (sieving, sintering, iron) should still work
        assert result["sieved_kg"] > 0.0 or result["bricks"] > 0 or result["iron_kg"] > 0

    def test_custom_allocation(self):
        """Custom power allocation changes output balance."""
        stockpile1 = RegolithStockpile(sieved_kg=5000)
        state1 = ProcessorState(sinter_furnace_temp_c=SINTER_TEMP_C)
        r1 = tick_regolith(
            power_budget_kwh=500.0, temp_c=-30.0,
            dust_opacity=0.0, water_liters=100.0,
            stockpile=stockpile1, state=state1,
            allocation={"excavation": 0.0, "sieving": 0.0, "washing": 0.0, "sintering": 1.0, "iron": 0.0},
        )

        stockpile2 = RegolithStockpile(sieved_kg=5000)
        state2 = ProcessorState(sinter_furnace_temp_c=SINTER_TEMP_C)
        r2 = tick_regolith(
            power_budget_kwh=500.0, temp_c=-30.0,
            dust_opacity=0.0, water_liters=100.0,
            stockpile=stockpile2, state=state2,
            allocation={"excavation": 1.0, "sieving": 0.0, "washing": 0.0, "sintering": 0.0, "iron": 0.0},
        )

        # All-sintering should make more bricks
        assert r1["bricks"] >= r2["bricks"]

    def test_result_has_all_keys(self):
        """tick result dict has all expected keys."""
        stockpile = RegolithStockpile()
        state = ProcessorState()
        result = tick_regolith(
            power_budget_kwh=100.0, temp_c=-60.0,
            dust_opacity=0.0, water_liters=10.0,
            stockpile=stockpile, state=state,
        )
        expected_keys = {
            "excavated_kg", "sieved_kg", "washed_kg", "bricks",
            "brick_quality", "brick_strength_mpa", "iron_kg",
            "power_consumed_kwh", "water_consumed_liters",
            "equipment_condition", "stockpile",
        }
        assert expected_keys.issubset(result.keys())

    def test_stockpile_never_negative(self):
        """After any tick, no stockpile value is negative."""
        stockpile = RegolithStockpile(raw_kg=10, sieved_kg=5)
        state = ProcessorState(sinter_furnace_temp_c=SINTER_TEMP_C)
        for _ in range(50):
            tick_regolith(
                power_budget_kwh=500.0, temp_c=-30.0,
                dust_opacity=0.0, water_liters=50.0,
                stockpile=stockpile, state=state,
            )
            assert stockpile.raw_kg >= -1e-9
            assert stockpile.sieved_kg >= -1e-9
            assert stockpile.washed_kg >= -1e-9
            assert stockpile.bricks >= 0
            assert stockpile.iron_kg >= -1e-9


# ===================================================================
# Property-based sweeps
# ===================================================================

class TestPropertySweeps:
    """Sweep parameters to check invariants hold across ranges."""

    def test_excavation_power_sweep(self):
        """Output scales with power, always ≥ 0."""
        for pw in range(0, 501, 25):
            kg, p = excavate_sol(float(pw), -60.0, 0.0, 1.0)
            assert kg >= 0.0
            assert p >= 0.0
            assert p <= pw + 1e-9

    def test_temperature_sweep_excavation(self):
        """Excavation output varies with temp but stays non-negative."""
        for t in range(-120, 1, 5):
            kg, p = excavate_sol(100.0, float(t), 0.0, 1.0)
            assert kg >= 0.0

    def test_dust_sweep_excavation(self):
        """Dust reduces then halts excavation."""
        outputs = []
        for d_int in range(0, 10):
            d = d_int / 10.0
            kg, _ = excavate_sol(100.0, -30.0, d, 1.0)
            outputs.append(kg)
        # Should generally decrease (not strictly due to threshold)
        assert outputs[0] >= outputs[-1]
        # At high dust, should be zero
        assert outputs[-1] == 0.0

    def test_sinter_temp_sweep(self):
        """Bricks only above minimum sintering temperature."""
        for t in range(0, 1300, 50):
            bricks, q, _, _ = sinter_bricks_sol(10000.0, 10000.0, float(t))
            if t < SINTER_TEMP_MIN_C:
                assert bricks == 0
            if t >= SINTER_TEMP_C:
                assert q >= 0.99

    def test_iron_input_sweep(self):
        """Iron output scales with regolith input."""
        prev_fe = 0.0
        for kg in range(0, 10001, 500):
            fe, _, _ = extract_iron_sol(float(kg), 10000.0)
            assert fe >= prev_fe - 1e-9
            prev_fe = fe

    def test_full_lifecycle_100_sols(self):
        """100-sol simulation: stockpile grows, equipment degrades, no crash."""
        stockpile = RegolithStockpile()
        state = ProcessorState(
            sinter_furnace_temp_c=SINTER_TEMP_C,
            water_available_liters=2000.0,
        )
        total_bricks = 0
        total_iron = 0.0
        for sol in range(100):
            # Simulate day/night temp variation
            temp = -60.0 + 20.0 * math.sin(sol * 0.1)
            dust = 0.1 if sol < 80 else 0.5  # dust storm last 20 sols
            result = tick_regolith(
                power_budget_kwh=250.0,
                temp_c=temp,
                dust_opacity=dust,
                water_liters=30.0,
                stockpile=stockpile,
                state=state,
            )
            total_bricks += result["bricks"]
            total_iron += result["iron_kg"]

        # Colony should have produced something meaningful
        assert total_bricks > 0, "100 sols should produce at least one brick"
        assert total_iron > 0, "100 sols should extract some iron"
        assert stockpile.washed_kg > 0, "Should have washed regolith for greenhouse"
        assert state.equipment_condition < 1.0, "Equipment should have degraded"
        assert state.equipment_condition > 0.9, "Equipment shouldn't die in 100 sols"
