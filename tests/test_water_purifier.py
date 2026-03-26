"""Tests for water_purifier.py — Mars Colony Water Purification."""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.water_purifier import (
    WaterBatch, PurifierState, PurifyResult,
    ix_removal_efficiency, ix_bed_volumes_used, ix_regen_capacity_after,
    ro_flux, ro_rejection, ro_energy_per_liter, ro_fouling_rate,
    uv_dose, uv_log_reduction, uv_lamp_degradation,
    tick_purify, regenerate_ix_resin, create_water_purifier,
    PERCHLORATE_SMAC_MG_L, IX_BED_VOLUMES_TO_BREAKTHROUGH,
    IX_REGEN_RECOVERY, RO_PERMEABILITY, RO_REJECTION,
    UV_D10_MJ_CM2, UV_LAMP_POWER_W,
    _clamp, _clamp_int,
)


# ── WaterBatch clamping ────────────────────────────────────────────

class TestWaterBatch:
    def test_defaults(self):
        b = WaterBatch()
        assert b.volume_liters == 0.0
        assert b.potable is False
        assert b.energy_wh == 0.0

    def test_negative_volume_clamped(self):
        b = WaterBatch(volume_liters=-10.0)
        assert b.volume_liters == 0.0

    def test_negative_perchlorate_clamped(self):
        b = WaterBatch(influent_perchlorate_mg_l=-5.0)
        assert b.influent_perchlorate_mg_l == 0.0

    def test_effluent_cannot_exceed_influent(self):
        b = WaterBatch(influent_perchlorate_mg_l=100.0,
                       effluent_perchlorate_mg_l=200.0)
        assert b.effluent_perchlorate_mg_l <= b.influent_perchlorate_mg_l

    def test_negative_energy_clamped(self):
        b = WaterBatch(energy_wh=-50.0)
        assert b.energy_wh == 0.0

    def test_uv_dose_clamped(self):
        b = WaterBatch(uv_dose_mj_cm2=-10.0)
        assert b.uv_dose_mj_cm2 == 0.0

    def test_log_reduction_clamped(self):
        b = WaterBatch(log_reduction=-3.0)
        assert b.log_reduction == 0.0


# ── PurifierState clamping ──────────────────────────────────────────

class TestPurifierState:
    def test_defaults(self):
        s = PurifierState()
        assert s.ix_resin_volume_l == 50.0
        assert s.ro_membrane_area_m2 == 10.0
        assert s.uv_lamp_count == 2

    def test_resin_volume_clamped(self):
        s = PurifierState(ix_resin_volume_l=0.01)
        assert s.ix_resin_volume_l == 1.0
        s2 = PurifierState(ix_resin_volume_l=9999.0)
        assert s2.ix_resin_volume_l == 500.0

    def test_ro_pressure_clamped(self):
        s = PurifierState(ro_pressure_bar=-5.0)
        assert s.ro_pressure_bar == 1.0
        s2 = PurifierState(ro_pressure_bar=200.0)
        assert s2.ro_pressure_bar == 80.0

    def test_capacity_fraction_clamped(self):
        s = PurifierState(ix_capacity_fraction=2.0)
        assert s.ix_capacity_fraction == 1.0
        s2 = PurifierState(ix_capacity_fraction=-0.5)
        assert s2.ix_capacity_fraction == 0.0

    def test_lamp_count_clamped(self):
        s = PurifierState(uv_lamp_count=0)
        assert s.uv_lamp_count == 1
        s2 = PurifierState(uv_lamp_count=100)
        assert s2.uv_lamp_count == 20

    def test_temperature_clamped(self):
        s = PurifierState(water_temperature_c=-50.0)
        assert s.water_temperature_c == 1.0
        s2 = PurifierState(water_temperature_c=100.0)
        assert s2.water_temperature_c == 45.0

    def test_fouling_clamped(self):
        s = PurifierState(ro_fouling_factor=5.0)
        assert s.ro_fouling_factor == 1.0
        s2 = PurifierState(ro_fouling_factor=-1.0)
        assert s2.ro_fouling_factor == 0.0


# ── Ion-exchange physics ────────────────────────────────────────────

class TestIonExchange:
    def test_fresh_resin_near_complete_removal(self):
        """Fresh resin at low BV should remove >99% perchlorate."""
        eff = ix_removal_efficiency(10.0, 1.0)
        assert eff > 0.99

    def test_efficiency_drops_near_breakthrough(self):
        """Efficiency must drop as BV approaches breakthrough."""
        eff_early = ix_removal_efficiency(100.0, 1.0)
        eff_late = ix_removal_efficiency(450.0, 1.0)
        assert eff_early > eff_late

    def test_past_breakthrough_very_low(self):
        """Past breakthrough, removal drops to near zero."""
        eff = ix_removal_efficiency(600.0, 1.0)
        assert eff < 0.1

    def test_zero_capacity_no_removal(self):
        eff = ix_removal_efficiency(10.0, 0.0)
        assert eff == 0.0

    def test_efficiency_always_bounded(self):
        """Efficiency must be in [0, 1] for any inputs."""
        for bv in [0, 100, 400, 500, 600, 1000, 5000]:
            for cap in [0.0, 0.3, 0.5, 0.8, 1.0]:
                eff = ix_removal_efficiency(float(bv), cap)
                assert 0.0 <= eff <= 1.0, f"bv={bv}, cap={cap}, eff={eff}"

    def test_bed_volumes_math(self):
        bv = ix_bed_volumes_used(100.0, 50.0)
        assert abs(bv - 2.0) < 1e-9

    def test_bed_volumes_zero_resin(self):
        bv = ix_bed_volumes_used(100.0, 0.0)
        assert bv == 0.0

    def test_regen_capacity_diminishes(self):
        """Each regen cycle reduces max capacity."""
        c0 = ix_regen_capacity_after(0)
        c1 = ix_regen_capacity_after(1)
        c5 = ix_regen_capacity_after(5)
        assert c0 == 1.0
        assert c1 == IX_REGEN_RECOVERY
        assert c5 < c1
        assert c5 > 0.0

    def test_regen_negative_cycles(self):
        c = ix_regen_capacity_after(-3)
        assert c == 1.0  # clamped to 0 cycles


# ── Reverse-osmosis physics ────────────────────────────────────────

class TestReverseOsmosis:
    def test_flux_positive_at_normal_conditions(self):
        f = ro_flux(15.0, 3.0, 1.0, 25.0)
        assert f > 0.0

    def test_flux_zero_when_pressure_below_osmotic(self):
        """If pressure < osmotic, no flux."""
        f = ro_flux(1.0, 50.0, 1.0, 25.0)
        assert f == 0.0

    def test_flux_increases_with_pressure(self):
        f_lo = ro_flux(10.0, 3.0, 1.0, 25.0)
        f_hi = ro_flux(20.0, 3.0, 1.0, 25.0)
        assert f_hi > f_lo

    def test_flux_decreases_with_fouling(self):
        f_clean = ro_flux(15.0, 3.0, 1.0, 25.0)
        f_fouled = ro_flux(15.0, 3.0, 0.3, 25.0)
        assert f_clean > f_fouled

    def test_flux_temperature_effect(self):
        """Higher temp increases flux."""
        f_cold = ro_flux(15.0, 3.0, 1.0, 5.0)
        f_warm = ro_flux(15.0, 3.0, 1.0, 35.0)
        assert f_warm > f_cold

    def test_rejection_clean_membrane(self):
        r = ro_rejection(1.0)
        assert abs(r - RO_REJECTION) < 1e-9

    def test_rejection_degrades_with_fouling(self):
        r_clean = ro_rejection(1.0)
        r_fouled = ro_rejection(0.5)
        assert r_clean > r_fouled

    def test_energy_per_liter_positive(self):
        e = ro_energy_per_liter(15.0)
        assert e > 0.0

    def test_energy_proportional_to_pressure(self):
        e_lo = ro_energy_per_liter(10.0)
        e_hi = ro_energy_per_liter(20.0)
        assert e_hi > e_lo

    def test_fouling_rate_clean_at_start(self):
        f = ro_fouling_rate(0.0, 8760.0)
        assert f == 1.0

    def test_fouling_rate_degrades_monotonically(self):
        prev = 1.0
        for h in range(0, 10000, 500):
            f = ro_fouling_rate(float(h), 8760.0)
            assert f <= prev + 1e-9
            prev = f

    def test_fouling_rate_dead_membrane(self):
        f = ro_fouling_rate(0.0, 0.0)
        assert f == 0.0

    def test_fouling_bounded(self):
        for h in [0, 1000, 5000, 8760, 15000, 50000]:
            f = ro_fouling_rate(float(h), 8760.0)
            assert 0.0 <= f <= 1.0


# ── UV-C sterilization physics ─────────────────────────────────────

class TestUVSterilization:
    def test_dose_positive_at_normal_flow(self):
        d = uv_dose(40.0, 2, 100.0, 500.0, 1.0)
        assert d > 0.0

    def test_dose_zero_no_flow(self):
        d = uv_dose(40.0, 2, 0.0, 500.0, 1.0)
        assert d == 0.0

    def test_dose_increases_with_lamps(self):
        d1 = uv_dose(40.0, 1, 100.0, 500.0, 1.0)
        d2 = uv_dose(40.0, 2, 100.0, 500.0, 1.0)
        assert d2 > d1

    def test_dose_decreases_with_higher_flow(self):
        """Faster flow = less exposure time = lower dose."""
        d_slow = uv_dose(40.0, 2, 50.0, 500.0, 1.0)
        d_fast = uv_dose(40.0, 2, 200.0, 500.0, 1.0)
        assert d_slow > d_fast

    def test_dose_decreases_with_degradation(self):
        d_new = uv_dose(40.0, 2, 100.0, 500.0, 1.0)
        d_old = uv_dose(40.0, 2, 100.0, 500.0, 0.5)
        assert d_new > d_old

    def test_log_reduction_4_log_at_target(self):
        """At the target dose (40 mJ/cm²), should get ~5.7 log kill."""
        lr = uv_log_reduction(40.0)
        assert lr > 4.0

    def test_log_reduction_zero_dose(self):
        lr = uv_log_reduction(0.0)
        assert lr == 0.0

    def test_log_reduction_capped_at_6(self):
        lr = uv_log_reduction(10000.0)
        assert lr == 6.0

    def test_log_reduction_monotonic(self):
        prev = 0.0
        for dose in range(0, 100, 5):
            lr = uv_log_reduction(float(dose))
            assert lr >= prev
            prev = lr

    def test_lamp_degradation_new(self):
        d = uv_lamp_degradation(0.0, 9000.0)
        assert d == 1.0

    def test_lamp_degradation_end_of_life(self):
        d = uv_lamp_degradation(9000.0, 9000.0)
        assert abs(d - 0.7) < 0.01

    def test_lamp_degradation_past_life(self):
        """Past rated life, output drops rapidly."""
        d_at = uv_lamp_degradation(9000.0, 9000.0)
        d_past = uv_lamp_degradation(12000.0, 9000.0)
        assert d_past < d_at

    def test_lamp_degradation_bounded(self):
        for h in [0, 1000, 5000, 9000, 15000, 50000]:
            d = uv_lamp_degradation(float(h), 9000.0)
            assert 0.0 <= d <= 1.0

    def test_lamp_degradation_zero_life(self):
        d = uv_lamp_degradation(100.0, 0.0)
        assert d == 0.0


# ── Tick integration ────────────────────────────────────────────────

class TestTickPurify:
    def test_idle_tick_no_batch(self):
        """Idle tick (no water) should produce empty result."""
        state = create_water_purifier()
        result = tick_purify(state, dt_hours=1.0, raw_volume_liters=0.0)
        assert result.batch.volume_liters == 0.0
        assert state.total_batches == 0

    def test_single_batch_produces_potable_water(self):
        """Fresh system should produce potable water from typical Mars input."""
        state = create_water_purifier()
        result = tick_purify(state, dt_hours=1.0,
                             raw_volume_liters=100.0,
                             raw_perchlorate_mg_l=500.0,
                             raw_tds_mg_l=3000.0)
        assert result.batch.potable is True
        assert result.batch.effluent_perchlorate_mg_l <= PERCHLORATE_SMAC_MG_L
        assert state.total_batches == 1
        assert state.total_liters_processed == 100.0

    def test_effluent_always_less_than_influent(self):
        """Conservation: effluent concentration ≤ influent."""
        state = create_water_purifier()
        for _ in range(20):
            result = tick_purify(state, dt_hours=1.0,
                                 raw_volume_liters=50.0,
                                 raw_perchlorate_mg_l=800.0,
                                 raw_tds_mg_l=5000.0)
            assert result.batch.effluent_perchlorate_mg_l <= 800.0
            assert result.batch.effluent_tds_mg_l <= 5000.0

    def test_energy_always_positive_for_batch(self):
        state = create_water_purifier()
        result = tick_purify(state, dt_hours=1.0,
                             raw_volume_liters=100.0)
        assert result.energy_wh > 0.0

    def test_energy_zero_for_idle(self):
        state = create_water_purifier()
        result = tick_purify(state, dt_hours=0.0, raw_volume_liters=0.0)
        assert result.energy_wh == 0.0

    def test_batches_accumulate(self):
        state = create_water_purifier()
        for i in range(10):
            tick_purify(state, dt_hours=1.0, raw_volume_liters=50.0)
        assert state.total_batches == 10
        assert state.total_liters_processed == 500.0
        assert len(state.batches) == 10

    def test_liters_monotonically_increase(self):
        state = create_water_purifier()
        prev = 0.0
        for _ in range(15):
            tick_purify(state, dt_hours=1.0, raw_volume_liters=30.0)
            assert state.total_liters_processed >= prev
            prev = state.total_liters_processed

    def test_equipment_ages_over_time(self):
        """RO hours and UV hours increase with ticks."""
        state = create_water_purifier()
        for _ in range(100):
            tick_purify(state, dt_hours=10.0, raw_volume_liters=50.0)
        assert state.ro_hours == 1000.0
        assert state.uv_lamp_hours == 1000.0

    def test_negative_inputs_clamped(self):
        state = create_water_purifier()
        result = tick_purify(state, dt_hours=-5.0,
                             raw_volume_liters=-100.0,
                             raw_perchlorate_mg_l=-500.0)
        assert state.total_batches == 0
        assert result.batch.volume_liters == 0.0

    def test_ix_breakthrough_warning(self):
        """Processing enough water triggers breakthrough warning."""
        state = create_water_purifier()
        # Process enough water to near breakthrough
        # 50L resin × 500 BV × 0.7 = 17500L
        for _ in range(200):
            result = tick_purify(state, dt_hours=0.1,
                                 raw_volume_liters=100.0)
        assert result.ix_near_breakthrough is True

    def test_ro_membrane_fouling_warning(self):
        """Running membrane past its life triggers fouling warning."""
        state = create_water_purifier()
        state.ro_hours = state.ro_membrane_life_hours * 0.9
        tick_purify(state, dt_hours=2000.0)
        result = tick_purify(state, dt_hours=1.0, raw_volume_liters=50.0)
        assert result.ro_membrane_fouled is True

    def test_uv_lamp_degradation_warning(self):
        """Old UV lamps trigger weak-lamp warning."""
        state = create_water_purifier()
        state.uv_lamp_hours = state.uv_lamp_life_hours * 0.9
        tick_purify(state, dt_hours=5000.0)
        result = tick_purify(state, dt_hours=1.0, raw_volume_liters=50.0)
        assert result.uv_lamp_weak is True


# ── Regeneration ────────────────────────────────────────────────────

class TestRegeneration:
    def test_regen_resets_bed_volumes(self):
        state = create_water_purifier()
        state.ix_bed_volumes_processed = 400.0
        regenerate_ix_resin(state)
        assert state.ix_bed_volumes_processed == 0.0
        assert state.ix_regen_cycles == 1

    def test_regen_reduces_capacity(self):
        state = create_water_purifier()
        regenerate_ix_resin(state)
        assert state.ix_capacity_fraction == IX_REGEN_RECOVERY

    def test_multiple_regens(self):
        state = create_water_purifier()
        for _ in range(5):
            regenerate_ix_resin(state)
        assert state.ix_regen_cycles == 5
        expected = IX_REGEN_RECOVERY ** 5
        assert abs(state.ix_capacity_fraction - expected) < 1e-9

    def test_regen_restores_removal_efficiency(self):
        """After regen, IX efficiency should be high again."""
        state = create_water_purifier()
        # Exhaust the resin well past breakthrough (600 BV)
        for _ in range(300):
            tick_purify(state, dt_hours=0.1, raw_volume_liters=100.0)
        # Efficiency should be low (past breakthrough)
        eff_before = ix_removal_efficiency(
            state.ix_bed_volumes_processed, state.ix_capacity_fraction)
        regenerate_ix_resin(state)
        eff_after = ix_removal_efficiency(
            state.ix_bed_volumes_processed, state.ix_capacity_fraction)
        assert eff_after > eff_before


# ── Factory ─────────────────────────────────────────────────────────

class TestFactory:
    def test_all_scenarios_create(self):
        for scenario in ["colony", "outpost", "emergency"]:
            state = create_water_purifier(scenario)
            assert state.ix_resin_volume_l > 0
            assert state.ro_membrane_area_m2 > 0
            assert state.uv_lamp_count >= 1

    def test_colony_larger_than_outpost(self):
        colony = create_water_purifier("colony")
        outpost = create_water_purifier("outpost")
        assert colony.ix_resin_volume_l > outpost.ix_resin_volume_l
        assert colony.ro_membrane_area_m2 > outpost.ro_membrane_area_m2

    def test_emergency_smallest(self):
        emergency = create_water_purifier("emergency")
        outpost = create_water_purifier("outpost")
        assert emergency.ix_resin_volume_l < outpost.ix_resin_volume_l
        assert emergency.uv_lamp_count <= outpost.uv_lamp_count

    def test_unknown_scenario_defaults_to_colony(self):
        state = create_water_purifier("nonexistent")
        colony = create_water_purifier("colony")
        assert state.ix_resin_volume_l == colony.ix_resin_volume_l

    def test_all_scenarios_produce_potable(self):
        """All scenarios should produce potable water when fresh."""
        for scenario in ["colony", "outpost", "emergency"]:
            state = create_water_purifier(scenario)
            result = tick_purify(state, dt_hours=1.0,
                                 raw_volume_liters=10.0,
                                 raw_perchlorate_mg_l=500.0,
                                 raw_tds_mg_l=3000.0)
            assert result.batch.potable is True, f"{scenario} failed potability"


# ── Smoke test: 100-batch simulation ────────────────────────────────

class TestSmoke:
    def test_100_batch_no_crash(self):
        """Run 100 batches without crashing."""
        state = create_water_purifier()
        for i in range(100):
            tick_purify(state, dt_hours=1.0,
                        raw_volume_liters=50.0,
                        raw_perchlorate_mg_l=500.0 + i * 5,
                        raw_tds_mg_l=3000.0 + i * 10)
        assert state.total_batches == 100
        assert state.total_liters_processed == 5000.0

    def test_mixed_idle_and_batch(self):
        """Alternating idle and batch ticks work correctly."""
        state = create_water_purifier()
        for i in range(50):
            if i % 2 == 0:
                tick_purify(state, dt_hours=1.0, raw_volume_liters=0.0)
            else:
                tick_purify(state, dt_hours=1.0, raw_volume_liters=100.0)
        assert state.total_batches == 25

    def test_full_lifecycle(self):
        """Process water → exhaust resin → regen → continue."""
        state = create_water_purifier("outpost")
        # Phase 1: process until near breakthrough
        for _ in range(100):
            tick_purify(state, dt_hours=0.5, raw_volume_liters=50.0)
        bv_before = state.ix_bed_volumes_processed
        # Regenerate
        regenerate_ix_resin(state)
        assert state.ix_bed_volumes_processed == 0.0
        assert state.ix_capacity_fraction < 1.0
        # Phase 2: continue processing
        for _ in range(50):
            result = tick_purify(state, dt_hours=0.5, raw_volume_liters=50.0)
        assert state.total_batches == 150

    def test_energy_conservation(self):
        """Total energy should equal sum of batch energies."""
        state = create_water_purifier()
        batch_sum = 0.0
        for _ in range(20):
            result = tick_purify(state, dt_hours=1.0, raw_volume_liters=50.0)
            batch_sum += result.energy_wh
        assert abs(state.total_energy_wh - batch_sum) < 0.01


# ── Helpers ─────────────────────────────────────────────────────────

class TestHelpers:
    def test_clamp_within_range(self):
        assert _clamp(5.0, 0.0, 10.0) == 5.0

    def test_clamp_below(self):
        assert _clamp(-5.0, 0.0, 10.0) == 0.0

    def test_clamp_above(self):
        assert _clamp(15.0, 0.0, 10.0) == 10.0

    def test_clamp_int(self):
        assert _clamp_int(5, 0, 10) == 5
        assert _clamp_int(-5, 0, 10) == 0
        assert _clamp_int(15, 0, 10) == 10
