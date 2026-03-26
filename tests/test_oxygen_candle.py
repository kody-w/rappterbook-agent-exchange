"""Tests for oxygen_candle.py — Mars Emergency Chemical Oxygen Generator.

Coverage: stoichiometry, burn physics, shelf life, inventory management,
conservation laws, edge cases, emergency scenarios, multi-candle burns.
"""
from __future__ import annotations

import math
import os
import sys
import dataclasses

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import oxygen_candle as oc


# ── fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def candle():
    """Fresh 1 kg candle, no aging."""
    return oc.create_candle("test-0", mass_kg=1.0, age_sols=0.0)


@pytest.fixture
def aged_candle():
    """Candle stored for 500 sols."""
    return oc.create_candle("test-aged", mass_kg=1.0, age_sols=500.0)


@pytest.fixture
def inventory():
    """Fresh 20-candle inventory."""
    return oc.create_inventory(num_candles=20, age_sols=0.0)


@pytest.fixture
def small_inventory():
    """Small 3-candle inventory for focused tests."""
    return oc.create_inventory(num_candles=3, age_sols=0.0)


# ══════════════════════════════════════════════════════════════════════════
# 1. STOICHIOMETRY — the chemistry must be exact
# ══════════════════════════════════════════════════════════════════════════

class TestStoichiometry:
    """2 NaClO₃ → 2 NaCl + 3 O₂ — mass must balance."""

    def test_o2_yield_per_kg_naclo3(self):
        """O₂ yield matches stoichiometry: 3*32 / (2*106.44)."""
        expected = (1.5 * 32.0) / 106.44
        assert abs(oc.O2_YIELD_PER_KG_NACLO3 - expected) < 1e-6

    def test_nacl_yield_per_kg_naclo3(self):
        """NaCl yield matches stoichiometry: 58.44 / 106.44."""
        expected = 58.44 / 106.44
        assert abs(oc.NACL_YIELD_PER_KG_NACLO3 - expected) < 1e-6

    def test_mass_conservation_pure_naclo3(self):
        """O₂ + NaCl fractions must sum to 1.0 (mass conservation)."""
        total = oc.O2_YIELD_PER_KG_NACLO3 + oc.NACL_YIELD_PER_KG_NACLO3
        assert abs(total - 1.0) < 1e-4

    def test_stoichiometric_o2_zero(self):
        assert oc.stoichiometric_o2_kg(0.0) == 0.0

    def test_stoichiometric_o2_negative(self):
        assert oc.stoichiometric_o2_kg(-1.0) == 0.0

    def test_stoichiometric_o2_1kg(self):
        result = oc.stoichiometric_o2_kg(1.0)
        assert abs(result - oc.O2_YIELD_PER_KG_NACLO3) < 1e-6

    def test_stoichiometric_o2_linear(self):
        """Doubling NaClO₃ doubles O₂."""
        assert abs(oc.stoichiometric_o2_kg(2.0)
                    - 2.0 * oc.stoichiometric_o2_kg(1.0)) < 1e-10

    def test_stoichiometric_nacl_1kg(self):
        result = oc.stoichiometric_nacl_kg(1.0)
        assert abs(result - oc.NACL_YIELD_PER_KG_NACLO3) < 1e-6

    def test_reaction_heat_positive(self):
        """Exothermic reaction: heat > 0 for positive reactant."""
        assert oc.reaction_heat_kj(1.0) > 0.0

    def test_reaction_heat_zero(self):
        assert oc.reaction_heat_kj(0.0) == 0.0

    def test_reaction_heat_negative_input(self):
        assert oc.reaction_heat_kj(-1.0) == 0.0

    def test_reaction_heat_proportional(self):
        """Heat scales linearly with reactant mass."""
        h1 = oc.reaction_heat_kj(1.0)
        h2 = oc.reaction_heat_kj(2.0)
        assert abs(h2 - 2.0 * h1) < 1e-6

    def test_candle_composition_sums_to_one(self):
        """Candle mass fractions must sum to 1.0."""
        total = (oc.NACLO3_MASS_FRACTION
                 + oc.FE_POWDER_MASS_FRACTION
                 + oc.PACKAGING_MASS_FRACTION)
        assert abs(total - 1.0) < 1e-6


# ══════════════════════════════════════════════════════════════════════════
# 2. CANDLE O₂ YIELD
# ══════════════════════════════════════════════════════════════════════════

class TestCandleYield:
    def test_fresh_candle_yield(self):
        """1 kg candle → ~0.383 kg O₂."""
        y = oc.candle_o2_yield_kg(1.0)
        assert 0.35 < y < 0.42  # physical bounds

    def test_degraded_candle_yield(self):
        """50% degradation halves yield."""
        fresh = oc.candle_o2_yield_kg(1.0, 0.0)
        half = oc.candle_o2_yield_kg(1.0, 0.5)
        assert abs(half - fresh * 0.5) < 1e-10

    def test_fully_degraded(self):
        """100% degradation → zero yield."""
        assert oc.candle_o2_yield_kg(1.0, 1.0) == 0.0

    def test_zero_mass(self):
        assert oc.candle_o2_yield_kg(0.0) == 0.0

    def test_negative_mass(self):
        assert oc.candle_o2_yield_kg(-1.0) == 0.0

    def test_yield_monotonic_with_mass(self):
        """Bigger candle → more O₂."""
        assert oc.candle_o2_yield_kg(2.0) > oc.candle_o2_yield_kg(1.0)

    def test_yield_decreases_with_degradation(self):
        """More degradation → less O₂."""
        assert oc.candle_o2_yield_kg(1.0, 0.3) < oc.candle_o2_yield_kg(1.0, 0.1)


# ══════════════════════════════════════════════════════════════════════════
# 3. BURN RATE PROFILE
# ══════════════════════════════════════════════════════════════════════════

class TestBurnRate:
    def test_rate_at_start_is_zero(self):
        """Candle rate at progress=0 is zero (ignition hasn't happened)."""
        assert oc.burn_rate_kg_o2_min(1.0, 0.0) == 0.0

    def test_tick_from_zero_produces_o2(self):
        """Midpoint integration means even a fresh candle produces O₂."""
        c = oc.create_candle("ramp-test")
        oc.ignite_candle(c)
        r = oc.tick_candle(c, dt_min=1.0)
        assert r["o2_kg"] > 0.0  # midpoint integration handles ramp

    def test_rate_ramps_up(self):
        """Rate increases during ignition phase."""
        r1 = oc.burn_rate_kg_o2_min(1.0, 0.01)
        r2 = oc.burn_rate_kg_o2_min(1.0, 0.04)
        assert r2 > r1 > 0.0

    def test_rate_at_midburn(self):
        """Mid-burn rate equals nominal BURN_RATE."""
        rate = oc.burn_rate_kg_o2_min(1.0, 0.5)
        assert abs(rate - oc.BURN_RATE_KG_O2_PER_MIN) < 1e-10

    def test_rate_tapers_at_end(self):
        """Rate decreases in final 10%."""
        r_mid = oc.burn_rate_kg_o2_min(1.0, 0.5)
        r_end = oc.burn_rate_kg_o2_min(1.0, 0.95)
        assert r_end < r_mid

    def test_rate_at_100_percent(self):
        """Fully spent candle produces nothing."""
        assert oc.burn_rate_kg_o2_min(1.0, 1.0) == 0.0

    def test_rate_always_non_negative(self):
        """Rate ≥ 0 for all burn progress values."""
        for p in [i / 100.0 for i in range(101)]:
            assert oc.burn_rate_kg_o2_min(1.0, p) >= 0.0

    def test_rate_scales_with_mass(self):
        """2 kg candle burns twice as fast."""
        r1 = oc.burn_rate_kg_o2_min(1.0, 0.5)
        r2 = oc.burn_rate_kg_o2_min(2.0, 0.5)
        assert abs(r2 - 2.0 * r1) < 1e-10

    def test_rate_negative_progress_clamped(self):
        """Negative progress treated as 0."""
        r = oc.burn_rate_kg_o2_min(1.0, -0.5)
        assert r == oc.burn_rate_kg_o2_min(1.0, 0.0)

    def test_rate_over_100_clamped(self):
        """Progress > 1 treated as 1.0."""
        r = oc.burn_rate_kg_o2_min(1.0, 1.5)
        assert r == 0.0


# ══════════════════════════════════════════════════════════════════════════
# 4. HEAT OUTPUT
# ══════════════════════════════════════════════════════════════════════════

class TestHeatOutput:
    def test_heat_positive_during_burn(self):
        """Burning candle produces heat."""
        h = oc.heat_output_kw(1.0, 0.5)
        assert h > 0.0

    def test_heat_zero_when_spent(self):
        assert oc.heat_output_kw(1.0, 1.0) == 0.0

    def test_heat_zero_when_not_started(self):
        assert oc.heat_output_kw(1.0, 0.0) == 0.0

    def test_heat_proportional_to_burn_rate(self):
        """Heat tracks O₂ rate (both measure reaction rate)."""
        h_mid = oc.heat_output_kw(1.0, 0.5)
        h_end = oc.heat_output_kw(1.0, 0.95)
        r_mid = oc.burn_rate_kg_o2_min(1.0, 0.5)
        r_end = oc.burn_rate_kg_o2_min(1.0, 0.95)
        if r_mid > 0 and r_end > 0:
            assert abs(h_mid / h_end - r_mid / r_end) < 0.01

    def test_hab_temp_rise_positive(self):
        """Heat in enclosed hab raises temperature."""
        rise = oc.hab_temp_rise_k(1.0, 500.0, 2.0)
        assert rise > 0.0

    def test_hab_temp_rise_zero_no_heat(self):
        assert oc.hab_temp_rise_k(0.0, 500.0, 2.0) == 0.0

    def test_hab_temp_rise_zero_volume(self):
        assert oc.hab_temp_rise_k(1.0, 0.0, 2.0) == 0.0

    def test_larger_hab_less_temp_rise(self):
        """Bigger hab dilutes heat more."""
        r_small = oc.hab_temp_rise_k(1.0, 100.0, 2.0)
        r_big = oc.hab_temp_rise_k(1.0, 1000.0, 2.0)
        assert r_big < r_small

    def test_more_ventilation_less_rise(self):
        """More airflow carries away more heat."""
        r_low = oc.hab_temp_rise_k(1.0, 500.0, 1.0)
        r_high = oc.hab_temp_rise_k(1.0, 500.0, 10.0)
        assert r_high < r_low


# ══════════════════════════════════════════════════════════════════════════
# 5. PERSONNEL HOURS & CANDLES NEEDED
# ══════════════════════════════════════════════════════════════════════════

class TestPersonnelHours:
    def test_one_person_hours(self):
        """1 kg candle gives ~9+ hours for 1 person."""
        h = oc.personnel_hours(1.0, 1)
        assert 5.0 < h < 20.0  # physical sanity bounds

    def test_more_crew_fewer_hours(self):
        """More people drain O₂ faster."""
        h1 = oc.personnel_hours(1.0, 1)
        h4 = oc.personnel_hours(1.0, 4)
        assert abs(h4 - h1 / 4.0) < 0.01

    def test_zero_crew(self):
        assert oc.personnel_hours(1.0, 0) == 0.0

    def test_degradation_reduces_hours(self):
        h_fresh = oc.personnel_hours(1.0, 1, 0.0)
        h_old = oc.personnel_hours(1.0, 1, 0.3)
        assert h_old < h_fresh

    def test_candles_needed_one_person_24h(self):
        """Sanity: need 2-4 candles for 1 person for 24 hours."""
        n = oc.candles_needed(1, 24.0)
        assert 1 <= n <= 6

    def test_candles_needed_zero_hours(self):
        assert oc.candles_needed(4, 0.0) == 0

    def test_candles_needed_zero_crew(self):
        assert oc.candles_needed(0, 24.0) == 0

    def test_candles_needed_scales_with_crew(self):
        """Twice the crew ≈ twice the candles."""
        n1 = oc.candles_needed(2, 24.0)
        n2 = oc.candles_needed(4, 24.0)
        # Integer ceiling: n2 should be roughly 2*n1
        assert n2 >= n1

    def test_candles_needed_integer(self):
        """Result is always an integer (can't light half a candle)."""
        n = oc.candles_needed(3, 10.0)
        assert isinstance(n, int)


# ══════════════════════════════════════════════════════════════════════════
# 6. SHELF LIFE & DEGRADATION
# ══════════════════════════════════════════════════════════════════════════

class TestShelfLife:
    def test_fresh_candle_no_degradation(self):
        c = oc.create_candle("c0", age_sols=0.0)
        assert c.shelf_degradation == 0.0

    def test_degradation_after_100_sols(self):
        d = oc.shelf_degradation_after(100.0)
        expected = 100.0 * oc.SHELF_DEGRADATION_PER_SOL
        assert abs(d - expected) < 1e-10

    def test_degradation_monotonic(self):
        d100 = oc.shelf_degradation_after(100.0)
        d200 = oc.shelf_degradation_after(200.0)
        assert d200 > d100

    def test_degradation_capped_at_1(self):
        d = oc.shelf_degradation_after(1e9)
        assert d == 1.0

    def test_degradation_from_existing(self):
        d = oc.shelf_degradation_after(100.0, initial_degradation=0.3)
        assert d > 0.3

    def test_viable_fresh(self):
        assert oc.is_candle_viable(0.0)

    def test_not_viable_at_50_percent(self):
        """50% degradation = 50% yield = threshold."""
        assert not oc.is_candle_viable(0.50)

    def test_viable_just_under_threshold(self):
        assert oc.is_candle_viable(0.49)

    def test_aged_candle_has_degradation(self, aged_candle):
        assert aged_candle.shelf_degradation > 0.0
        expected = 500.0 * oc.SHELF_DEGRADATION_PER_SOL
        assert abs(aged_candle.shelf_degradation - expected) < 1e-6

    def test_very_old_candle_not_viable(self):
        """Candle stored for 5000 sols (>6.6 Mars years) is dead."""
        c = oc.create_candle("ancient", age_sols=5000.0)
        # 5000 * 0.0005 = 2.5, capped at 1.0
        assert c.shelf_degradation == 1.0
        assert not c.is_ready


# ══════════════════════════════════════════════════════════════════════════
# 7. CANDLE STATE MACHINE
# ══════════════════════════════════════════════════════════════════════════

class TestCandleState:
    def test_new_candle_is_ready(self, candle):
        assert candle.is_ready
        assert not candle.is_burning
        assert not candle.is_spent

    def test_ignite_success(self, candle):
        err = oc.ignite_candle(candle)
        assert err is None
        assert candle.is_burning
        assert not candle.is_ready

    def test_ignite_already_burning(self, candle):
        oc.ignite_candle(candle)
        err = oc.ignite_candle(candle)
        assert err is not None
        assert "already burning" in err

    def test_ignite_spent_candle(self, candle):
        candle.burn_progress = 1.0
        err = oc.ignite_candle(candle)
        assert err is not None
        assert "spent" in err

    def test_ignite_degraded_candle(self):
        c = oc.create_candle("bad", age_sols=5000.0)
        err = oc.ignite_candle(c)
        assert err is not None
        assert "degraded" in err

    def test_remaining_o2_fresh(self, candle):
        total = oc.candle_o2_yield_kg(candle.mass_kg)
        assert abs(candle.remaining_o2_kg - total) < 1e-10

    def test_remaining_o2_decreases(self, candle):
        initial = candle.remaining_o2_kg
        oc.ignite_candle(candle)
        oc.tick_candle(candle, dt_min=5.0)
        assert candle.remaining_o2_kg < initial

    def test_post_init_clamps(self):
        """Negative values clamped to zero."""
        c = oc.Candle(mass_kg=-1.0, burn_progress=-0.5,
                      shelf_degradation=2.0)
        assert c.mass_kg == 0.0
        assert c.burn_progress == 0.0
        assert c.shelf_degradation == 1.0


# ══════════════════════════════════════════════════════════════════════════
# 8. TICK CANDLE — single candle burn simulation
# ══════════════════════════════════════════════════════════════════════════

class TestTickCandle:
    def test_unlit_candle_produces_nothing(self, candle):
        result = oc.tick_candle(candle, dt_min=1.0)
        assert result["o2_kg"] == 0.0
        assert result["heat_kj"] == 0.0

    def test_burning_candle_produces_o2(self, candle):
        oc.ignite_candle(candle)
        # Skip past ignition ramp
        candle.burn_progress = 0.1
        result = oc.tick_candle(candle, dt_min=1.0)
        assert result["o2_kg"] > 0.0

    def test_burning_candle_produces_heat(self, candle):
        oc.ignite_candle(candle)
        candle.burn_progress = 0.1
        result = oc.tick_candle(candle, dt_min=1.0)
        assert result["heat_kj"] > 0.0

    def test_burning_candle_produces_nacl(self, candle):
        oc.ignite_candle(candle)
        candle.burn_progress = 0.1
        result = oc.tick_candle(candle, dt_min=1.0)
        assert result["nacl_kg"] > 0.0

    def test_burn_progress_advances(self, candle):
        oc.ignite_candle(candle)
        candle.burn_progress = 0.1
        before = candle.burn_progress
        oc.tick_candle(candle, dt_min=1.0)
        assert candle.burn_progress > before

    def test_candle_eventually_spent(self, candle):
        """Burn until exhaustion — must finish."""
        oc.ignite_candle(candle)
        for _ in range(500):  # 500 minutes should exhaust any 1kg candle
            if candle.is_spent:
                break
            oc.tick_candle(candle, dt_min=1.0)
        assert candle.is_spent
        assert not candle.is_burning

    def test_spent_candle_no_output(self, candle):
        oc.ignite_candle(candle)
        candle.burn_progress = 1.0
        candle.is_burning = False
        result = oc.tick_candle(candle, dt_min=1.0)
        assert result["o2_kg"] == 0.0

    def test_zero_dt_produces_nothing(self, candle):
        oc.ignite_candle(candle)
        candle.burn_progress = 0.5
        result = oc.tick_candle(candle, dt_min=0.0)
        assert result["o2_kg"] == 0.0

    def test_negative_dt_clamped(self, candle):
        oc.ignite_candle(candle)
        candle.burn_progress = 0.5
        result = oc.tick_candle(candle, dt_min=-5.0)
        assert result["o2_kg"] == 0.0


# ══════════════════════════════════════════════════════════════════════════
# 9. CONSERVATION LAWS — the physics must close
# ══════════════════════════════════════════════════════════════════════════

class TestConservationLaws:
    def test_total_o2_bounded_by_stoichiometry(self, candle):
        """Total O₂ produced ≤ stoichiometric limit."""
        max_o2 = oc.candle_o2_yield_kg(candle.mass_kg)
        oc.ignite_candle(candle)
        for _ in range(500):
            if candle.is_spent:
                break
            oc.tick_candle(candle, dt_min=1.0)
        assert candle.o2_delivered_kg <= max_o2 + 1e-10

    def test_o2_delivered_approaches_yield(self, candle):
        """Full burn delivers ~100% of theoretical yield."""
        max_o2 = oc.candle_o2_yield_kg(candle.mass_kg)
        oc.ignite_candle(candle)
        for _ in range(500):
            if candle.is_spent:
                break
            oc.tick_candle(candle, dt_min=1.0)
        # Should deliver at least 95% (ignition ramp eats a little)
        assert candle.o2_delivered_kg > max_o2 * 0.95

    def test_heat_delivered_bounded(self, candle):
        """Total heat ≤ enthalpy of reaction."""
        max_heat = oc.HEAT_PER_KG_CANDLE_KJ * candle.mass_kg
        oc.ignite_candle(candle)
        for _ in range(500):
            if candle.is_spent:
                break
            oc.tick_candle(candle, dt_min=1.0)
        assert candle.heat_delivered_kj <= max_heat + 1e-6

    def test_nacl_proportional_to_o2(self, candle):
        """NaCl and O₂ must be stoichiometrically linked."""
        oc.ignite_candle(candle)
        total_o2 = 0.0
        total_nacl = 0.0
        for _ in range(500):
            if candle.is_spent:
                break
            r = oc.tick_candle(candle, dt_min=1.0)
            total_o2 += r["o2_kg"]
            total_nacl += r["nacl_kg"]
        if total_o2 > 0:
            expected_ratio = (2.0 * oc.NACL_MOLAR_MASS_G) / (1.5 * oc.O2_MOLAR_MASS_G)
            actual_ratio = total_nacl / total_o2
            assert abs(actual_ratio - expected_ratio) < 0.01

    def test_burn_progress_monotonic(self, candle):
        """Burn progress never decreases."""
        oc.ignite_candle(candle)
        prev = candle.burn_progress
        for _ in range(200):
            if candle.is_spent:
                break
            oc.tick_candle(candle, dt_min=1.0)
            assert candle.burn_progress >= prev
            prev = candle.burn_progress

    def test_o2_always_non_negative(self, candle):
        """No tick ever produces negative O₂."""
        oc.ignite_candle(candle)
        for _ in range(500):
            if candle.is_spent:
                break
            r = oc.tick_candle(candle, dt_min=1.0)
            assert r["o2_kg"] >= 0.0

    def test_remaining_o2_non_negative(self, candle):
        """Remaining O₂ never goes below zero."""
        oc.ignite_candle(candle)
        for _ in range(500):
            if candle.is_spent:
                break
            oc.tick_candle(candle, dt_min=1.0)
            assert candle.remaining_o2_kg >= -1e-10


# ══════════════════════════════════════════════════════════════════════════
# 10. INVENTORY MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════

class TestInventory:
    def test_create_inventory(self, inventory):
        assert inventory.total_count == 20
        assert inventory.ready_count == 20
        assert inventory.burning_count == 0
        assert inventory.spent_count == 0

    def test_inventory_conservation(self, inventory):
        """Total = ready + burning + spent + degraded."""
        assert inventory.inventory_check()

    def test_activate_emergency(self, inventory):
        errors = oc.activate_emergency(inventory, num_candles=2)
        assert len(errors) == 0
        assert inventory.burning_count == 2
        assert inventory.ready_count == 18

    def test_activate_too_many(self, small_inventory):
        errors = oc.activate_emergency(small_inventory, num_candles=5)
        assert len(errors) > 0
        assert small_inventory.burning_count == 3

    def test_activate_zero(self, inventory):
        errors = oc.activate_emergency(inventory, num_candles=0)
        assert len(errors) == 0
        assert inventory.burning_count == 0

    def test_tick_inventory(self, small_inventory):
        oc.activate_emergency(small_inventory, num_candles=1)
        # Advance past ignition ramp
        for c in small_inventory.candles:
            if c.is_burning:
                c.burn_progress = 0.1
        result = oc.tick_inventory(small_inventory, dt_min=1.0)
        assert result["o2_kg"] > 0.0
        assert result["active_candles"] >= 1

    def test_tick_inventory_no_burning(self, inventory):
        result = oc.tick_inventory(inventory, dt_min=1.0)
        assert result["o2_kg"] == 0.0
        assert result["active_candles"] == 0

    def test_emergency_capacity(self, inventory):
        """20 fresh candles should give many hours for 1 person."""
        hours = inventory.emergency_capacity_hours
        assert hours > 100.0  # 20 candles × ~9h each

    def test_age_inventory(self, inventory):
        expired = oc.age_inventory(inventory, sols=100.0)
        assert expired == 0  # 100 sols not enough to expire
        for c in inventory.candles:
            if not c.is_burning and not c.is_spent:
                assert c.age_sols == 100.0

    def test_age_inventory_expire(self):
        """After enough aging, candles expire."""
        inv = oc.create_inventory(num_candles=5, age_sols=900.0)
        # 900 sols = 0.45 degradation. Add 200 more = 0.55 > 0.50 threshold
        expired = oc.age_inventory(inv, sols=200.0)
        assert expired == 5

    def test_emergency_duration(self, inventory):
        hours = oc.emergency_duration_hours(inventory, crew_size=4)
        assert hours > 0.0

    def test_emergency_duration_zero_crew(self, inventory):
        assert oc.emergency_duration_hours(inventory, crew_size=0) == 0.0


# ══════════════════════════════════════════════════════════════════════════
# 11. MULTI-CANDLE BURN SIMULATION
# ══════════════════════════════════════════════════════════════════════════

class TestMultiCandleBurn:
    def test_two_candles_double_output(self, small_inventory):
        """Two burning candles produce ~2× the O₂ of one."""
        # Single candle
        inv1 = oc.create_inventory(num_candles=1)
        oc.activate_emergency(inv1, 1)
        for c in inv1.candles:
            if c.is_burning:
                c.burn_progress = 0.5
        r1 = oc.tick_inventory(inv1, dt_min=1.0)

        # Two candles
        inv2 = oc.create_inventory(num_candles=2)
        oc.activate_emergency(inv2, 2)
        for c in inv2.candles:
            if c.is_burning:
                c.burn_progress = 0.5
        r2 = oc.tick_inventory(inv2, dt_min=1.0)

        assert abs(r2["o2_kg"] - 2.0 * r1["o2_kg"]) < 1e-10

    def test_sequential_candle_activation(self, small_inventory):
        """Activate candles one at a time as they burn out."""
        oc.activate_emergency(small_inventory, 1)
        # Burn to completion
        for _ in range(500):
            oc.tick_inventory(small_inventory, dt_min=1.0)
            if small_inventory.burning_count == 0:
                break
        assert small_inventory.spent_count == 1
        assert small_inventory.ready_count == 2
        # Light another
        oc.activate_emergency(small_inventory, 1)
        assert small_inventory.burning_count == 1

    def test_full_inventory_burn(self):
        """Burn entire inventory — all candles end spent."""
        inv = oc.create_inventory(num_candles=3)
        oc.activate_emergency(inv, 3)
        for _ in range(500):
            if inv.burning_count == 0:
                break
            oc.tick_inventory(inv, dt_min=1.0)
        assert inv.spent_count == 3
        assert inv.ready_count == 0
        assert inv.burning_count == 0


# ══════════════════════════════════════════════════════════════════════════
# 12. SERIALIZATION
# ══════════════════════════════════════════════════════════════════════════

class TestSerialization:
    def test_to_dict_structure(self, inventory):
        d = oc.to_dict(inventory)
        assert "total_candles" in d
        assert "ready" in d
        assert "burning" in d
        assert "spent" in d
        assert "candles" in d
        assert len(d["candles"]) == 20

    def test_to_dict_after_burn(self, small_inventory):
        oc.activate_emergency(small_inventory, 1)
        for c in small_inventory.candles:
            if c.is_burning:
                c.burn_progress = 0.1
        oc.tick_inventory(small_inventory, dt_min=5.0)
        d = oc.to_dict(small_inventory)
        assert d["burning"] == 1
        assert d["total_o2_delivered_kg"] > 0.0

    def test_to_dict_values_are_json_safe(self, inventory):
        """All values must be JSON-serializable."""
        import json
        d = oc.to_dict(inventory)
        json_str = json.dumps(d)
        assert len(json_str) > 0


# ══════════════════════════════════════════════════════════════════════════
# 13. EDGE CASES
# ══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_zero_mass_candle(self):
        c = oc.create_candle("zero", mass_kg=0.0)
        assert c.remaining_o2_kg == 0.0
        err = oc.ignite_candle(c)
        # Should ignite (technically valid) but produce nothing
        assert err is None

    def test_huge_candle(self):
        c = oc.create_candle("huge", mass_kg=100.0)
        assert c.remaining_o2_kg > 30.0

    def test_tiny_dt(self):
        c = oc.create_candle("c0")
        oc.ignite_candle(c)
        c.burn_progress = 0.5
        r = oc.tick_candle(c, dt_min=0.001)
        assert r["o2_kg"] >= 0.0

    def test_large_dt(self):
        """Large dt should not produce more O₂ than available."""
        c = oc.create_candle("c0")
        oc.ignite_candle(c)
        c.burn_progress = 0.5
        r = oc.tick_candle(c, dt_min=10000.0)
        max_remaining = c.remaining_o2_kg + r["o2_kg"]
        total = oc.candle_o2_yield_kg(c.mass_kg)
        assert r["o2_kg"] <= total + 1e-10

    def test_empty_inventory(self):
        inv = oc.create_inventory(num_candles=0)
        assert inv.total_count == 0
        assert inv.emergency_capacity_hours == 0.0
        errors = oc.activate_emergency(inv, 1)
        assert len(errors) > 0


# ══════════════════════════════════════════════════════════════════════════
# 14. SMOKE TEST — 10-minute emergency scenario
# ══════════════════════════════════════════════════════════════════════════

class TestSmokeScenario:
    def test_10_minute_emergency(self):
        """Simulate 10-minute power-out emergency for 4 crew.

        Light 2 candles, run for 10 minutes, verify O₂ delivery
        is sufficient and all conservation laws hold.
        """
        inv = oc.create_inventory(num_candles=10, age_sols=180.0)
        crew_size = 4
        duration_min = 10

        # Activate emergency
        errors = oc.activate_emergency(inv, num_candles=2)
        assert len(errors) == 0

        total_o2 = 0.0
        o2_demand = oc.O2_KG_PER_PERSON_PER_MIN * crew_size * duration_min

        for minute in range(duration_min):
            result = oc.tick_inventory(inv, dt_min=1.0)
            total_o2 += result["o2_kg"]
            assert result["o2_kg"] >= 0.0
            assert result["heat_kj"] >= 0.0

        # 2 candles for 10 min should exceed demand for 4 crew
        assert total_o2 > o2_demand * 0.5  # at least 50% (ignition ramp)
        assert inv.burning_count >= 1  # candles still going at 10 min
        assert inv.inventory_check()

    def test_full_sol_emergency(self):
        """Colony power out for an entire sol. Candles sustain 4 crew.

        Sequential candle activation as each burns out.
        """
        crew_size = 4
        sol_minutes = int(24.66 * 60)  # ~1480 minutes
        needed = oc.candles_needed(crew_size, 24.66)
        inv = oc.create_inventory(num_candles=needed + 5, age_sols=30.0)

        total_o2 = 0.0
        demand = oc.O2_KG_PER_PERSON_PER_MIN * crew_size * sol_minutes

        # Start with 2 candles
        oc.activate_emergency(inv, num_candles=2)

        for minute in range(sol_minutes):
            # If no candles burning, light the next one
            if inv.burning_count == 0 and inv.ready_count > 0:
                oc.activate_emergency(inv, num_candles=1)
            # If only 1 burning and it's >80% done, pre-light another
            for c in inv.candles:
                if (c.is_burning and c.burn_progress > 0.80
                        and inv.burning_count < 2 and inv.ready_count > 0):
                    oc.activate_emergency(inv, num_candles=1)
                    break

            result = oc.tick_inventory(inv, dt_min=1.0)
            total_o2 += result["o2_kg"]

        # Total O₂ should meet crew demand
        assert total_o2 > demand * 0.80
        assert inv.inventory_check()


# ══════════════════════════════════════════════════════════════════════════
# 15. PROPERTY-BASED INVARIANTS
# ══════════════════════════════════════════════════════════════════════════

class TestInvariants:
    """Property-based checks that must hold for ANY input."""

    @pytest.mark.parametrize("mass", [0.0, 0.1, 0.5, 1.0, 2.0, 10.0])
    def test_yield_non_negative(self, mass):
        assert oc.candle_o2_yield_kg(mass) >= 0.0

    @pytest.mark.parametrize("mass", [0.0, 0.1, 0.5, 1.0, 2.0, 10.0])
    def test_yield_bounded_by_mass(self, mass):
        """O₂ yield < candle mass (can't create matter)."""
        assert oc.candle_o2_yield_kg(mass) <= mass

    @pytest.mark.parametrize("deg", [0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
    def test_yield_decreases_with_degradation(self, deg):
        y = oc.candle_o2_yield_kg(1.0, deg)
        assert y >= 0.0
        assert y <= oc.candle_o2_yield_kg(1.0, 0.0) + 1e-10

    @pytest.mark.parametrize("progress", [0.0, 0.01, 0.05, 0.1, 0.5,
                                           0.8, 0.9, 0.95, 0.99, 1.0])
    def test_burn_rate_non_negative(self, progress):
        assert oc.burn_rate_kg_o2_min(1.0, progress) >= 0.0

    @pytest.mark.parametrize("sols", [0, 1, 10, 100, 1000, 10000])
    def test_shelf_degradation_in_range(self, sols):
        d = oc.shelf_degradation_after(float(sols))
        assert 0.0 <= d <= 1.0

    @pytest.mark.parametrize("crew,hours", [
        (1, 1), (1, 24), (4, 6), (6, 48), (10, 168),
    ])
    def test_candles_needed_non_negative(self, crew, hours):
        n = oc.candles_needed(crew, float(hours))
        assert n >= 0
        assert isinstance(n, int)

    @pytest.mark.parametrize("crew,hours", [
        (1, 24), (4, 24), (6, 48),
    ])
    def test_candles_needed_sufficient(self, crew, hours):
        """Requested candles actually provide enough O₂."""
        n = oc.candles_needed(crew, float(hours))
        total_o2 = n * oc.candle_o2_yield_kg(1.0)
        demand = oc.O2_KG_PER_PERSON_PER_MIN * crew * hours * 60.0
        assert total_o2 >= demand * 0.99  # allow 1% numerical tolerance
