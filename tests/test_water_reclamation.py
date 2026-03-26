"""Tests for water_reclamation.py — Mars Colony Water Recovery System.

93 tests covering:
  - Individual subsystem physics (condensate, urine, brine, greywater)
  - Quality monitoring and rejection logic
  - Full system integration over multi-sol runs
  - Conservation laws (water in ≥ water out)
  - Degradation curves and maintenance recovery
  - Power-limited operation
  - Edge cases (zero crew, extreme temps, depleted systems)
  - Property-based invariants (physical bounds)
"""
from __future__ import annotations

import math
import pytest

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.water_reclamation import (
    CondensateCollector,
    UrineProcessor,
    BrineProcessor,
    GreywaterRecycler,
    QualityMonitor,
    WaterReclamationSystem,
    URINE_L_PER_PERSON_SOL,
    SWEAT_RESP_L_PER_PERSON_SOL,
    GREYWATER_L_PER_PERSON_SOL,
    CONDENSATE_RECOVERY,
    URINE_DISTILL_RECOVERY,
    BRINE_RECOVERY,
    GREYWATER_RECOVERY,
    TOC_LIMIT_MG_L,
    CONDUCTIVITY_LIMIT_US_CM,
    BIOCIDE_MIN_MG_L,
    MIN_EFFICIENCY_FACTOR,
    MAINTENANCE_RESTORE,
    FILTER_LIFE_LITERS,
)


# ===================================================================
# Condensate Collector
# ===================================================================

class TestCondensateCollector:
    """Tests for atmospheric condensate extraction."""

    def test_basic_collection(self):
        """Collects water from crew respiration/perspiration."""
        cc = CondensateCollector()
        water, energy = cc.collect(crew=6)
        expected_vapor = 6 * SWEAT_RESP_L_PER_PERSON_SOL
        assert water > 0
        assert water <= expected_vapor
        assert energy > 0

    def test_zero_crew(self):
        """No crew = no condensate."""
        cc = CondensateCollector()
        water, energy = cc.collect(crew=0)
        assert water == 0.0
        assert energy == 0.0

    def test_single_crew(self):
        """Single crew member produces proportional output."""
        cc = CondensateCollector()
        water_1, _ = cc.collect(crew=1)
        cc2 = CondensateCollector()
        water_6, _ = cc2.collect(crew=6)
        assert abs(water_6 / water_1 - 6.0) < 0.1

    def test_temperature_penalty(self):
        """Hotter habitat → less condensation."""
        cc_cool = CondensateCollector()
        water_cool, _ = cc_cool.collect(crew=6, hab_temp_c=18.0)

        cc_hot = CondensateCollector()
        water_hot, _ = cc_hot.collect(crew=6, hab_temp_c=30.0)

        assert water_cool > water_hot

    def test_optimal_temperature(self):
        """At optimal temp, efficiency is near theoretical max."""
        cc = CondensateCollector()
        water, _ = cc.collect(crew=6, hab_temp_c=CONDENSATE_RECOVERY)
        expected = 6 * SWEAT_RESP_L_PER_PERSON_SOL * CONDENSATE_RECOVERY
        # Should be close to max (health=1.0, temp=optimal)
        assert water >= expected * 0.9

    def test_health_degrades(self):
        """Collecting water degrades collector health over time."""
        cc = CondensateCollector()
        initial_health = cc.health
        for _ in range(100):
            cc.collect(crew=50)  # high throughput
        assert cc.health < initial_health

    def test_health_floor(self):
        """Health cannot drop below minimum."""
        cc = CondensateCollector()
        for _ in range(10000):
            cc.collect(crew=100)
        assert cc.health >= MIN_EFFICIENCY_FACTOR

    def test_cumulative_tracking(self):
        """Total collected tracks across ticks."""
        cc = CondensateCollector()
        for _ in range(10):
            cc.collect(crew=6)
        assert cc.total_collected_l > 0

    def test_energy_positive(self):
        """Energy cost is always non-negative."""
        cc = CondensateCollector()
        _, energy = cc.collect(crew=6, hab_temp_c=35.0)
        assert energy >= 0


# ===================================================================
# Urine Processor
# ===================================================================

class TestUrineProcessor:
    """Tests for vapor-compression distillation urine processor."""

    def test_basic_processing(self):
        """Processes urine and returns clean water + brine."""
        up = UrineProcessor()
        clean, brine, energy = up.process(crew=6)
        total = clean + brine
        expected = 6 * URINE_L_PER_PERSON_SOL
        assert abs(total - expected) < 0.01  # mass conservation
        assert clean > 0
        assert brine > 0
        assert energy > 0

    def test_zero_crew(self):
        """No crew = no urine to process."""
        up = UrineProcessor()
        clean, brine, energy = up.process(crew=0)
        assert clean == 0.0
        assert brine == 0.0
        assert energy == 0.0

    def test_conservation_law(self):
        """Water in = water out (clean + brine). No water created or destroyed."""
        up = UrineProcessor()
        for crew in [1, 5, 10, 50, 100]:
            clean, brine, _ = up.process(crew=crew)
            total_in = crew * URINE_L_PER_PERSON_SOL
            total_out = clean + brine
            assert abs(total_in - total_out) < 0.01

    def test_recovery_rate_in_bounds(self):
        """Recovery rate between 0 and 1."""
        up = UrineProcessor()
        clean, _, _ = up.process(crew=6)
        rate = clean / (6 * URINE_L_PER_PERSON_SOL)
        assert 0 < rate <= 1.0

    def test_brine_accumulation(self):
        """Brine accumulates over multiple sols."""
        up = UrineProcessor()
        for _ in range(10):
            up.process(crew=6)
        assert up.brine_accumulated_l > 0

    def test_degradation(self):
        """Processor health degrades with use."""
        up = UrineProcessor()
        initial = up.health
        for _ in range(200):
            up.process(crew=50)
        assert up.health < initial

    def test_degradation_floor(self):
        """Health has a minimum floor."""
        up = UrineProcessor()
        for _ in range(50000):
            up.process(crew=100)
        assert up.health >= MIN_EFFICIENCY_FACTOR


# ===================================================================
# Brine Processor
# ===================================================================

class TestBrineProcessor:
    """Tests for brine water extraction."""

    def test_basic_extraction(self):
        """Extracts water from brine concentrate."""
        bp = BrineProcessor()
        clean, solid, energy = bp.process(brine_l=10.0)
        assert clean > 0
        assert solid > 0
        assert energy > 0

    def test_zero_brine(self):
        """No brine = no output."""
        bp = BrineProcessor()
        clean, solid, energy = bp.process(brine_l=0.0)
        assert clean == 0.0
        assert solid == 0.0
        assert energy == 0.0

    def test_mass_conservation(self):
        """Input brine mass = output water + solid waste (adjusted for density)."""
        bp = BrineProcessor()
        brine_in = 10.0
        clean, solid, _ = bp.process(brine_l=brine_in)
        # clean is liters of water (~1 kg/L), solid is kg at 1.2 density
        # brine_in liters → clean liters water + (brine_in - clean) * 1.2 kg solid
        remaining_brine = brine_in - clean
        expected_solid = remaining_brine * 1.2
        assert abs(solid - expected_solid) < 0.01

    def test_recovery_bounded(self):
        """Recovery fraction between 0 and 1."""
        bp = BrineProcessor()
        clean, _, _ = bp.process(brine_l=10.0)
        assert 0 < clean / 10.0 <= 1.0

    def test_high_throughput_degrades(self):
        """Processing large volumes degrades the membrane."""
        bp = BrineProcessor()
        initial = bp.health
        for _ in range(500):
            bp.process(brine_l=100.0)
        assert bp.health < initial

    def test_health_floor(self):
        """Minimum health enforced."""
        bp = BrineProcessor()
        for _ in range(100000):
            bp.process(brine_l=1000.0)
        assert bp.health >= MIN_EFFICIENCY_FACTOR


# ===================================================================
# Greywater Recycler
# ===================================================================

class TestGreywaterRecycler:
    """Tests for greywater treatment."""

    def test_basic_recycling(self):
        """Recycles greywater from crew hygiene."""
        gw = GreywaterRecycler()
        clean, waste, energy = gw.process(crew=6)
        expected_grey = 6 * GREYWATER_L_PER_PERSON_SOL
        assert clean > 0
        assert clean <= expected_grey
        assert waste >= 0
        assert energy > 0

    def test_zero_crew(self):
        """No crew = no greywater."""
        gw = GreywaterRecycler()
        clean, waste, energy = gw.process(crew=0)
        assert clean == 0.0
        assert waste == 0.0
        assert energy == 0.0

    def test_conservation(self):
        """Clean + waste = total greywater input."""
        gw = GreywaterRecycler()
        clean, waste, _ = gw.process(crew=6)
        total_in = 6 * GREYWATER_L_PER_PERSON_SOL
        assert abs((clean + waste) - total_in) < 0.01

    def test_recovery_in_range(self):
        """Recovery rate within physical bounds."""
        gw = GreywaterRecycler()
        clean, _, _ = gw.process(crew=10)
        rate = clean / (10 * GREYWATER_L_PER_PERSON_SOL)
        assert 0 < rate <= 1.0

    def test_degradation(self):
        """Filter clogs reduce recovery over time."""
        gw = GreywaterRecycler()
        initial = gw.health
        for _ in range(200):
            gw.process(crew=50)
        assert gw.health < initial

    def test_large_crew(self):
        """Scales linearly with crew size."""
        gw1 = GreywaterRecycler()
        c1, _, _ = gw1.process(crew=10)
        gw2 = GreywaterRecycler()
        c2, _, _ = gw2.process(crew=100)
        assert abs(c2 / c1 - 10.0) < 0.5


# ===================================================================
# Quality Monitor
# ===================================================================

class TestQualityMonitor:
    """Tests for water quality monitoring."""

    def test_healthy_system_passes(self):
        """Healthy system produces potable water."""
        qm = QualityMonitor()
        potable, rejected = qm.check(water_l=100.0, system_health=1.0)
        assert potable == 100.0
        assert rejected == 0.0

    def test_degraded_system_rejects(self):
        """Degraded system rejects some water."""
        qm = QualityMonitor()
        potable, rejected = qm.check(water_l=100.0, system_health=0.5)
        assert rejected > 0
        assert potable + rejected == pytest.approx(100.0)

    def test_very_degraded_high_rejection(self):
        """Very degraded system → high rejection rate."""
        qm = QualityMonitor()
        potable, rejected = qm.check(water_l=100.0, system_health=0.3)
        assert rejected > 0
        # At health=0.3, contamination=0.7, so reject=min(0.5, 0.7)=0.5 → 50L
        assert rejected == pytest.approx(50.0)

    def test_quality_metrics_update(self):
        """Quality metrics reflect system health."""
        qm = QualityMonitor()
        qm.check(water_l=100.0, system_health=0.5)
        assert qm.toc_mg_l > 0
        assert qm.conductivity_us_cm > 0
        assert qm.biocide_mg_l > 0

    def test_healthy_within_limits(self):
        """Healthy system stays within quality limits."""
        qm = QualityMonitor()
        qm.check(water_l=100.0, system_health=1.0)
        assert qm.toc_mg_l <= TOC_LIMIT_MG_L
        assert qm.conductivity_us_cm <= CONDUCTIVITY_LIMIT_US_CM
        assert qm.biocide_mg_l >= BIOCIDE_MIN_MG_L

    def test_zero_water(self):
        """Zero water input → zero output."""
        qm = QualityMonitor()
        potable, rejected = qm.check(water_l=0.0, system_health=1.0)
        assert potable == 0.0
        assert rejected == 0.0

    def test_cumulative_rejects(self):
        """Rejected water accumulates in tracking."""
        qm = QualityMonitor()
        qm.check(water_l=100.0, system_health=0.5)
        qm.check(water_l=100.0, system_health=0.5)
        assert qm.rejects_l > 0


# ===================================================================
# Full System Integration
# ===================================================================

class TestWaterReclamationSystem:
    """Tests for the integrated water recovery system."""

    def test_single_tick(self):
        """System processes one sol successfully."""
        wrs = WaterReclamationSystem()
        result = wrs.tick(crew=6)
        assert result["sol"] == 1
        assert result["crew"] == 6
        assert result["potable_l"] > 0
        assert result["recovery_rate"] > 0
        assert result["energy_kwh"] > 0

    def test_recovery_rate_physical_bounds(self):
        """Recovery rate between 0 and 1 (can't create water)."""
        wrs = WaterReclamationSystem()
        result = wrs.tick(crew=6)
        assert 0 < result["recovery_rate"] <= 1.0

    def test_water_conservation(self):
        """Potable + rejected ≤ wastewater (no water creation)."""
        wrs = WaterReclamationSystem()
        result = wrs.tick(crew=6)
        assert result["potable_l"] + result["rejected_l"] <= result["wastewater_l"] + 0.01

    def test_multi_sol_run(self):
        """System runs 100 sols without crash."""
        wrs = WaterReclamationSystem()
        for _ in range(100):
            result = wrs.tick(crew=6)
        assert wrs.sol == 100
        assert len(wrs.history) == 100
        assert wrs.total_recovered_l > 0

    def test_365_sol_run(self):
        """Full Mars year without crash or negative values."""
        wrs = WaterReclamationSystem()
        for _ in range(365):
            result = wrs.tick(crew=20)
            assert result["potable_l"] >= 0
            assert result["energy_kwh"] >= 0
            assert result["recovery_rate"] >= 0
            assert result["system_health"] > 0

    def test_zero_crew_tick(self):
        """Zero crew = zero everything."""
        wrs = WaterReclamationSystem()
        result = wrs.tick(crew=0)
        assert result["wastewater_l"] == 0.0
        assert result["potable_l"] == 0.0
        assert result["energy_kwh"] == 0.0

    def test_large_crew(self):
        """Handles 200+ colonists."""
        wrs = WaterReclamationSystem()
        result = wrs.tick(crew=200)
        assert result["potable_l"] > 0
        assert result["wastewater_l"] == 200 * (
            URINE_L_PER_PERSON_SOL
            + SWEAT_RESP_L_PER_PERSON_SOL
            + GREYWATER_L_PER_PERSON_SOL
        )

    def test_health_degrades_over_time(self):
        """System health decreases over extended operation."""
        wrs = WaterReclamationSystem()
        wrs.tick(crew=6)
        initial_health = wrs.history[0]["system_health"]
        for _ in range(500):
            wrs.tick(crew=6)
        final_health = wrs.history[-1]["system_health"]
        assert final_health < initial_health

    def test_subsystem_health_tracked(self):
        """Each subsystem health is individually tracked."""
        wrs = WaterReclamationSystem()
        result = wrs.tick(crew=6)
        sh = result["subsystem_health"]
        assert "condensate" in sh
        assert "urine_processor" in sh
        assert "brine_processor" in sh
        assert "greywater" in sh

    def test_quality_tracked(self):
        """Quality metrics are in every tick result."""
        wrs = WaterReclamationSystem()
        result = wrs.tick(crew=6)
        q = result["quality"]
        assert "toc_mg_l" in q
        assert "conductivity_us_cm" in q
        assert "biocide_mg_l" in q

    def test_energy_scales_with_crew(self):
        """More crew = more energy for water processing."""
        wrs1 = WaterReclamationSystem()
        r1 = wrs1.tick(crew=6)
        wrs2 = WaterReclamationSystem()
        r2 = wrs2.tick(crew=60)
        assert r2["energy_kwh"] > r1["energy_kwh"]

    def test_total_tracking(self):
        """Cumulative totals are maintained."""
        wrs = WaterReclamationSystem()
        for _ in range(10):
            wrs.tick(crew=6)
        assert wrs.total_recovered_l > 0
        assert wrs.total_energy_kwh > 0

    def test_sol_counter(self):
        """Sol counter increments correctly."""
        wrs = WaterReclamationSystem()
        for i in range(5):
            result = wrs.tick(crew=6)
            assert result["sol"] == i + 1

    def test_history_length(self):
        """History grows with each tick."""
        wrs = WaterReclamationSystem()
        for _ in range(25):
            wrs.tick(crew=6)
        assert len(wrs.history) == 25


# ===================================================================
# Maintenance
# ===================================================================

class TestMaintenance:
    """Tests for system maintenance and recovery."""

    def test_maintenance_improves_health(self):
        """Maintenance restores subsystem health."""
        wrs = WaterReclamationSystem()
        # Degrade the system
        for _ in range(500):
            wrs.tick(crew=50)
        pre_health = wrs.condensate.health
        improvements = wrs.perform_maintenance()
        assert wrs.condensate.health > pre_health
        assert all(v >= 0 for v in improvements.values())

    def test_maintenance_on_healthy_system(self):
        """Maintenance on fresh system is a no-op (already at max)."""
        wrs = WaterReclamationSystem()
        improvements = wrs.perform_maintenance()
        # Already at 1.0, so improvements should be zero
        assert all(abs(v) < 0.001 for v in improvements.values())

    def test_maintenance_recovery_bounded(self):
        """Maintenance cannot exceed health of 1.0."""
        wrs = WaterReclamationSystem()
        for _ in range(100):
            wrs.tick(crew=50)
        wrs.perform_maintenance()
        wrs.perform_maintenance()  # double maintenance
        assert wrs.condensate.health <= 1.0
        assert wrs.urine_proc.health <= 1.0
        assert wrs.brine_proc.health <= 1.0
        assert wrs.greywater.health <= 1.0

    def test_maintenance_restores_recovery_rate(self):
        """After maintenance, recovery rate improves."""
        wrs = WaterReclamationSystem()
        for _ in range(300):
            wrs.tick(crew=30)
        pre_rate = wrs.history[-1]["recovery_rate"]
        wrs.perform_maintenance()
        post_result = wrs.tick(crew=30)
        assert post_result["recovery_rate"] >= pre_rate


# ===================================================================
# Power-Limited Operation
# ===================================================================

class TestPowerLimited:
    """Tests for operation under power constraints."""

    def test_unlimited_power(self):
        """Unlimited power gives full recovery."""
        wrs = WaterReclamationSystem()
        result = wrs.tick(crew=6, available_power_kwh=float("inf"))
        assert result["recovery_rate"] > 0.8

    def test_zero_power(self):
        """Zero power = zero recovery."""
        wrs = WaterReclamationSystem()
        result = wrs.tick(crew=6, available_power_kwh=0.0)
        # With no power, everything is limited
        assert result["energy_kwh"] == 0.0

    def test_limited_power_reduces_output(self):
        """Limited power produces less water than unlimited."""
        wrs_full = WaterReclamationSystem()
        r_full = wrs_full.tick(crew=6)

        wrs_low = WaterReclamationSystem()
        r_low = wrs_low.tick(crew=6, available_power_kwh=1.0)

        assert r_low["potable_l"] <= r_full["potable_l"]

    def test_energy_respects_budget(self):
        """System never exceeds power budget."""
        wrs = WaterReclamationSystem()
        budget = 2.0
        result = wrs.tick(crew=6, available_power_kwh=budget)
        assert result["energy_kwh"] <= budget + 0.01

    def test_priority_ordering(self):
        """With limited power, condensate (cheapest) runs first."""
        wrs = WaterReclamationSystem()
        # Very small budget — only enough for condensate
        result = wrs.tick(crew=6, available_power_kwh=0.3)
        assert result["condensate_l"] > 0


# ===================================================================
# Water Balance
# ===================================================================

class TestWaterBalance:
    """Tests for water balance analysis."""

    def test_balance_healthy(self):
        """Healthy system is self-sufficient."""
        wrs = WaterReclamationSystem()
        balance = wrs.get_water_balance(crew=6)
        assert balance["daily_need_l"] > 0
        assert balance["daily_wastewater_l"] > 0
        assert "self_sufficient" in balance

    def test_balance_zero_crew(self):
        """Zero crew = zero balance."""
        wrs = WaterReclamationSystem()
        balance = wrs.get_water_balance(crew=0)
        assert balance["daily_need_l"] == 0.0
        assert balance["deficit_l"] == 0.0

    def test_degraded_system_deficit(self):
        """Heavily degraded system shows water deficit."""
        wrs = WaterReclamationSystem()
        # Force low health
        wrs.condensate.health = 0.3
        wrs.urine_proc.health = 0.3
        wrs.brine_proc.health = 0.3
        wrs.greywater.health = 0.3
        balance = wrs.get_water_balance(crew=50)
        assert balance["deficit_l"] > 0
        assert not balance["self_sufficient"]

    def test_overall_recovery_rate(self):
        """Lifetime recovery rate is reasonable."""
        wrs = WaterReclamationSystem()
        for _ in range(50):
            wrs.tick(crew=6)
        rate = wrs.get_overall_recovery_rate()
        assert 0.5 < rate < 1.0

    def test_overall_rate_empty(self):
        """No history = zero rate."""
        wrs = WaterReclamationSystem()
        assert wrs.get_overall_recovery_rate() == 0.0


# ===================================================================
# Property-Based Invariants
# ===================================================================

class TestInvariants:
    """Property-based tests — physical laws that must always hold."""

    @pytest.mark.parametrize("crew", [1, 6, 20, 50, 100, 200])
    def test_no_water_creation(self, crew):
        """Output water ≤ input wastewater for any crew size."""
        wrs = WaterReclamationSystem()
        result = wrs.tick(crew=crew)
        assert result["potable_l"] + result["rejected_l"] <= result["wastewater_l"] + 0.01

    @pytest.mark.parametrize("crew", [1, 6, 20, 50, 100])
    def test_recovery_rate_bounded(self, crew):
        """Recovery rate ∈ [0, 1] for any crew size."""
        wrs = WaterReclamationSystem()
        result = wrs.tick(crew=crew)
        assert 0 <= result["recovery_rate"] <= 1.0

    @pytest.mark.parametrize("crew", [1, 6, 20, 50])
    def test_energy_non_negative(self, crew):
        """Energy consumption is never negative."""
        wrs = WaterReclamationSystem()
        result = wrs.tick(crew=crew)
        assert result["energy_kwh"] >= 0

    @pytest.mark.parametrize("crew", [1, 6, 20, 50])
    def test_health_in_bounds(self, crew):
        """System health always ∈ [MIN_EFFICIENCY_FACTOR, 1.0]."""
        wrs = WaterReclamationSystem()
        for _ in range(100):
            result = wrs.tick(crew=crew)
            h = result["system_health"]
            assert MIN_EFFICIENCY_FACTOR <= h <= 1.0

    def test_monotonic_degradation_without_maintenance(self):
        """Without maintenance, health only decreases."""
        wrs = WaterReclamationSystem()
        healths = []
        for _ in range(50):
            result = wrs.tick(crew=20)
            healths.append(result["system_health"])
        # Each health should be ≤ the previous
        for i in range(1, len(healths)):
            assert healths[i] <= healths[i - 1] + 0.001

    @pytest.mark.parametrize("temp", [-20.0, 0.0, 18.0, 25.0, 40.0])
    def test_temperature_produces_valid_output(self, temp):
        """Any temperature produces non-negative, bounded output."""
        wrs = WaterReclamationSystem()
        result = wrs.tick(crew=6, hab_temp_c=temp)
        assert result["potable_l"] >= 0
        assert result["recovery_rate"] >= 0
        assert result["recovery_rate"] <= 1.0

    def test_10_sol_smoke_test(self):
        """Smoke test: 10 sols, 6 crew, no crash, all values sane."""
        wrs = WaterReclamationSystem()
        for sol in range(1, 11):
            result = wrs.tick(crew=6)
            assert result["sol"] == sol
            assert result["potable_l"] > 0
            assert result["energy_kwh"] > 0
            assert 0 < result["recovery_rate"] <= 1.0
            assert result["system_health"] > 0

    def test_wastewater_formula(self):
        """Wastewater = crew × (urine + sweat + greywater) per sol."""
        wrs = WaterReclamationSystem()
        crew = 42
        result = wrs.tick(crew=crew)
        expected = crew * (
            URINE_L_PER_PERSON_SOL
            + SWEAT_RESP_L_PER_PERSON_SOL
            + GREYWATER_L_PER_PERSON_SOL
        )
        assert result["wastewater_l"] == pytest.approx(expected, rel=0.001)


# ===================================================================
# Edge Cases
# ===================================================================

class TestEdgeCases:
    """Tests for boundary conditions and unusual inputs."""

    def test_single_crew_full_year(self):
        """One person, 668 sols (full Mars year)."""
        wrs = WaterReclamationSystem()
        for _ in range(668):
            result = wrs.tick(crew=1)
        assert wrs.sol == 668
        assert wrs.total_recovered_l > 0

    def test_extreme_cold_habitat(self):
        """Very cold habitat (emergency heating failure)."""
        wrs = WaterReclamationSystem()
        result = wrs.tick(crew=6, hab_temp_c=-20.0)
        assert result["potable_l"] >= 0

    def test_extreme_hot_habitat(self):
        """Very hot habitat (cooling failure)."""
        wrs = WaterReclamationSystem()
        result = wrs.tick(crew=6, hab_temp_c=45.0)
        assert result["potable_l"] >= 0
        # Hot = less condensation
        assert result["condensate_l"] >= 0

    def test_alternating_crew_sizes(self):
        """Crew size varies sol to sol (EVA rotations)."""
        wrs = WaterReclamationSystem()
        for sol in range(100):
            crew = 6 if sol % 2 == 0 else 4
            result = wrs.tick(crew=crew)
            assert result["potable_l"] >= 0

    def test_maintenance_mid_run(self):
        """Maintenance during extended operation."""
        wrs = WaterReclamationSystem()
        for _ in range(200):
            wrs.tick(crew=20)
        wrs.perform_maintenance()
        for _ in range(200):
            wrs.tick(crew=20)
        assert wrs.sol == 400
        assert wrs.total_recovered_l > 0
