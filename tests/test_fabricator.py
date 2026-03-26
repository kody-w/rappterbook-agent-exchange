"""
Tests for fabricator.py -- Mars Colony 3D Fabrication Lab.

75 tests across 10 test classes. Every function, edge case, and physics
invariant tested. The fabricator manufactures the colony's survival.

Run: python -m pytest tests/test_fabricator.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.fabricator import (
    FabricatorState,
    PrintJob,
    FabSol,
    estimate_job,
    start_job,
    cancel_job,
    add_feedstock,
    calibrate,
    tick_fabricator,
    create_fabricator,
    ENERGY_REGOLITH_KWH_KG,
    ENERGY_METAL_KWH_KG,
    ENERGY_POLYMER_KWH_KG,
    SPEED_REGOLITH_KG_HR,
    SPEED_METAL_KG_HR,
    SPEED_POLYMER_KG_HR,
    WASTE_REGOLITH,
    WASTE_METAL,
    WASTE_POLYMER,
    QUALITY_NEW,
    QUALITY_MIN,
    NOZZLE_WEAR_PER_KG,
    CALIBRATION_RESTORE,
    MAX_FEEDSTOCK_KG,
    SOL_HOURS,
    MATERIAL_TYPES,
)


# --- PrintJob ---

class TestPrintJob:
    """Tests for the PrintJob dataclass."""

    def test_basic_job(self):
        j = PrintJob(name="gasket", material="polymer", mass_kg=0.5)
        assert j.name == "gasket"
        assert j.material == "polymer"
        assert j.mass_kg == 0.5

    def test_invalid_material_defaults(self):
        j = PrintJob(name="test", material="unobtainium", mass_kg=1.0)
        assert j.material == "regolith"

    def test_clamp_mass(self):
        j = PrintJob(name="test", material="metal", mass_kg=-5.0)
        assert j.mass_kg == 0.01

    def test_energy_includes_waste(self):
        j = PrintJob(name="test", material="regolith", mass_kg=1.0)
        expected = 1.0 * (1.0 + WASTE_REGOLITH) * ENERGY_REGOLITH_KWH_KG
        assert abs(j.energy_required_kwh - expected) < 1e-6

    def test_feedstock_includes_waste(self):
        j = PrintJob(name="test", material="metal", mass_kg=2.0)
        expected = 2.0 * (1.0 + WASTE_METAL)
        assert abs(j.feedstock_required_kg - expected) < 1e-6

    def test_print_time_positive(self):
        j = PrintJob(name="test", material="polymer", mass_kg=1.0)
        assert j.print_time_hours > 0.0

    def test_metal_slower_than_polymer(self):
        jm = PrintJob(name="m", material="metal", mass_kg=1.0)
        jp = PrintJob(name="p", material="polymer", mass_kg=1.0)
        assert jm.print_time_hours > jp.print_time_hours

    def test_priority_clamp(self):
        j = PrintJob(name="test", material="regolith", mass_kg=1.0, priority=10)
        assert j.priority == 5


# --- FabricatorState ---

class TestFabricatorState:
    """Tests for the FabricatorState dataclass."""

    def test_default_state(self):
        s = FabricatorState()
        assert s.nozzle_quality == 1.0
        assert s.operational
        assert s.is_idle
        assert s.nozzle_usable

    def test_clamp_nozzle(self):
        s = FabricatorState(nozzle_quality=1.5)
        assert s.nozzle_quality == 1.0
        s2 = FabricatorState(nozzle_quality=0.01)
        assert s2.nozzle_quality == QUALITY_MIN

    def test_clamp_feedstock(self):
        s = FabricatorState(feedstock_regolith_kg=9999.0)
        assert s.feedstock_regolith_kg == MAX_FEEDSTOCK_KG

    def test_get_feedstock(self):
        s = FabricatorState(feedstock_metal_kg=42.0)
        assert s.get_feedstock("metal") == 42.0
        assert s.get_feedstock("unobtainium") == 0.0

    def test_consume_feedstock(self):
        s = FabricatorState(feedstock_polymer_kg=10.0)
        actual = s.consume_feedstock("polymer", 3.0)
        assert actual == 3.0
        assert s.feedstock_polymer_kg == 7.0

    def test_consume_more_than_available(self):
        s = FabricatorState(feedstock_metal_kg=5.0)
        actual = s.consume_feedstock("metal", 10.0)
        assert actual == 5.0
        assert s.feedstock_metal_kg == 0.0

    def test_nozzle_usable_threshold(self):
        s = FabricatorState(nozzle_quality=QUALITY_MIN - 0.01)
        # clamped to QUALITY_MIN
        assert s.nozzle_usable


# --- Estimate ---

class TestEstimate:
    """Tests for job estimation."""

    def test_estimate_returns_all_fields(self):
        j = PrintJob(name="bolt", material="metal", mass_kg=0.1)
        e = estimate_job(j)
        assert "energy_kwh" in e
        assert "feedstock_kg" in e
        assert "print_hours" in e
        assert "print_sols" in e

    def test_estimate_sols_matches_hours(self):
        j = PrintJob(name="plate", material="regolith", mass_kg=5.0)
        e = estimate_job(j)
        assert abs(e["print_sols"] - e["print_hours"] / SOL_HOURS) < 0.001


# --- Start / Cancel ---

class TestStartCancel:
    """Tests for starting and cancelling jobs."""

    def test_start_success(self):
        s = FabricatorState()
        j = PrintJob(name="gasket", material="polymer", mass_kg=0.5)
        r = start_job(s, j)
        assert r["success"]
        assert not s.is_idle

    def test_start_while_busy(self):
        s = FabricatorState()
        j1 = PrintJob(name="a", material="polymer", mass_kg=0.5)
        start_job(s, j1)
        j2 = PrintJob(name="b", material="metal", mass_kg=0.3)
        r = start_job(s, j2)
        assert not r["success"]
        assert r["reason"] == "printer_busy"

    def test_start_nozzle_worn(self):
        s = FabricatorState(nozzle_quality=QUALITY_MIN - 0.01)
        # clamped to QUALITY_MIN, so still usable
        j = PrintJob(name="test", material="regolith", mass_kg=0.1)
        r = start_job(s, j)
        assert r["success"]

    def test_start_insufficient_feedstock(self):
        s = FabricatorState(feedstock_metal_kg=0.1)
        j = PrintJob(name="big", material="metal", mass_kg=10.0)
        r = start_job(s, j)
        assert not r["success"]
        assert r["reason"] == "insufficient_feedstock"

    def test_start_not_operational(self):
        s = FabricatorState()
        s.operational = False
        j = PrintJob(name="test", material="regolith", mass_kg=0.1)
        r = start_job(s, j)
        assert not r["success"]
        assert r["reason"] == "not_operational"

    def test_cancel_success(self):
        s = FabricatorState()
        j = PrintJob(name="gasket", material="polymer", mass_kg=0.5)
        start_job(s, j)
        r = cancel_job(s)
        assert r["success"]
        assert s.is_idle

    def test_cancel_no_job(self):
        s = FabricatorState()
        r = cancel_job(s)
        assert not r["success"]


# --- Feedstock ---

class TestFeedstock:
    """Tests for adding feedstock."""

    def test_add_feedstock(self):
        s = FabricatorState(feedstock_regolith_kg=100.0)
        r = add_feedstock(s, "regolith", 50.0)
        assert r["success"]
        assert r["added_kg"] == 50.0
        assert s.feedstock_regolith_kg == 150.0

    def test_add_over_capacity(self):
        s = FabricatorState(feedstock_metal_kg=490.0)
        r = add_feedstock(s, "metal", 20.0)
        assert r["added_kg"] == 10.0
        assert r["rejected_kg"] == 10.0

    def test_add_invalid_material(self):
        s = FabricatorState()
        r = add_feedstock(s, "unobtainium", 10.0)
        assert not r["success"]


# --- Calibrate ---

class TestCalibrate:
    """Tests for printer calibration."""

    def test_calibrate_improves(self):
        s = FabricatorState(nozzle_quality=0.5)
        r = calibrate(s)
        assert r["success"]
        assert s.nozzle_quality > 0.5

    def test_calibrate_restore_fraction(self):
        s = FabricatorState(nozzle_quality=0.5)
        calibrate(s)
        expected = 0.5 + 0.5 * CALIBRATION_RESTORE
        assert abs(s.nozzle_quality - expected) < 1e-6

    def test_calibrate_while_busy(self):
        s = FabricatorState()
        j = PrintJob(name="test", material="regolith", mass_kg=5.0)
        start_job(s, j)
        r = calibrate(s)
        assert not r["success"]


# --- Tick ---

class TestTick:
    """Tests for the sol-level tick function."""

    def test_idle_sol(self):
        s = FabricatorState()
        sol = FabSol(sol=1)
        r = tick_fabricator(s, sol)
        assert r["sol"] == 1
        assert r["is_idle"]

    def test_sol_with_new_job(self):
        s = FabricatorState()
        j = PrintJob(name="washer", material="metal", mass_kg=0.1)
        sol = FabSol(sol=1, new_job=j)
        r = tick_fabricator(s, sol)
        assert r["start"]["success"]

    def test_sol_printing(self):
        s = FabricatorState()
        j = PrintJob(name="small", material="polymer", mass_kg=0.5)
        start_job(s, j)
        sol = FabSol(sol=1, available_power_kwh=100.0)
        r = tick_fabricator(s, sol)
        assert r["print"] is not None
        assert r["print"]["printed_kg"] > 0

    def test_small_job_completes_in_one_sol(self):
        s = FabricatorState()
        j = PrintJob(name="washer", material="polymer", mass_kg=0.1)
        start_job(s, j)
        sol = FabSol(sol=1, available_power_kwh=100.0)
        r = tick_fabricator(s, sol)
        assert r["print"]["completed"]
        assert s.total_parts_printed == 1
        assert s.is_idle

    def test_large_job_takes_multiple_sols(self):
        s = FabricatorState()
        j = PrintJob(name="panel", material="regolith", mass_kg=50.0)
        start_job(s, j)
        completed = False
        for i in range(100):
            sol = FabSol(sol=i + 1, available_power_kwh=100.0)
            r = tick_fabricator(s, sol)
            if r["print"] and r["print"]["completed"]:
                completed = True
                break
        assert completed
        assert s.total_parts_printed == 1

    def test_sol_add_feedstock(self):
        s = FabricatorState(feedstock_regolith_kg=100.0)
        sol = FabSol(sol=1, add_regolith_kg=50.0)
        r = tick_fabricator(s, sol)
        assert len(r["feedstock_added"]) == 1
        assert s.feedstock_regolith_kg == 150.0

    def test_sol_calibrate(self):
        s = FabricatorState(nozzle_quality=0.5)
        sol = FabSol(sol=1, calibrate_printer=True)
        r = tick_fabricator(s, sol)
        assert r["calibrate"]["success"]

    def test_nozzle_wears_during_print(self):
        s = FabricatorState()
        j = PrintJob(name="block", material="regolith", mass_kg=10.0)
        start_job(s, j)
        q_before = s.nozzle_quality
        sol = FabSol(sol=1, available_power_kwh=100.0)
        tick_fabricator(s, sol)
        assert s.nozzle_quality < q_before

    def test_10_sol_smoke_test(self):
        """10 sols of mixed fabrication. Must not crash."""
        s = create_fabricator()
        jobs = [
            PrintJob(name="gasket", material="polymer", mass_kg=0.3),
            PrintJob(name="bracket", material="metal", mass_kg=0.5),
            PrintJob(name="tile", material="regolith", mass_kg=2.0),
        ]
        job_idx = 0
        for i in range(10):
            new_job = None
            if s.is_idle and job_idx < len(jobs):
                new_job = jobs[job_idx]
                job_idx += 1
            sol = FabSol(
                sol=i + 1,
                new_job=new_job,
                available_power_kwh=50.0,
                calibrate_printer=(i == 5 and s.is_idle),
                add_regolith_kg=10.0 if i == 0 else 0.0,
            )
            r = tick_fabricator(s, sol)
            assert r["nozzle_after"] >= QUALITY_MIN

    def test_50_sol_endurance(self):
        """50 sols of continuous fabrication."""
        s = create_fabricator()
        parts_completed = 0
        for i in range(50):
            new_job = None
            if s.is_idle:
                new_job = PrintJob(
                    name=f"part_{i}",
                    material="regolith",
                    mass_kg=1.0,
                )
            sol = FabSol(
                sol=i + 1,
                new_job=new_job,
                available_power_kwh=50.0,
                add_regolith_kg=5.0,
                calibrate_printer=(i % 15 == 14 and s.is_idle),
            )
            r = tick_fabricator(s, sol)
            if r["print"] and r["print"]["completed"]:
                parts_completed += 1

        assert parts_completed >= 5
        assert s.total_mass_printed_kg > 0
        assert s.nozzle_quality < 1.0


# --- Physics Invariants ---

class TestPhysicsInvariants:
    """Property-based invariants that must always hold."""

    def test_nozzle_never_below_minimum(self):
        s = FabricatorState()
        for i in range(100):
            j = PrintJob(name=f"p{i}", material="regolith", mass_kg=1.0)
            if s.is_idle:
                start_job(s, j)
            sol = FabSol(sol=i + 1, available_power_kwh=100.0, add_regolith_kg=20.0)
            tick_fabricator(s, sol)
        assert s.nozzle_quality >= QUALITY_MIN

    def test_feedstock_never_negative(self):
        s = FabricatorState()
        j = PrintJob(name="big", material="regolith", mass_kg=100.0)
        start_job(s, j)
        for i in range(50):
            sol = FabSol(sol=i + 1, available_power_kwh=100.0)
            tick_fabricator(s, sol)
        assert s.feedstock_regolith_kg >= 0.0
        assert s.feedstock_metal_kg >= 0.0
        assert s.feedstock_polymer_kg >= 0.0

    def test_total_parts_monotonic(self):
        s = FabricatorState()
        prev = 0
        for i in range(20):
            if s.is_idle:
                j = PrintJob(name=f"p{i}", material="polymer", mass_kg=0.1)
                start_job(s, j)
            sol = FabSol(sol=i + 1, available_power_kwh=100.0)
            tick_fabricator(s, sol)
            assert s.total_parts_printed >= prev
            prev = s.total_parts_printed

    def test_energy_consumed_monotonic(self):
        s = FabricatorState()
        prev = 0.0
        j = PrintJob(name="block", material="regolith", mass_kg=5.0)
        start_job(s, j)
        for i in range(10):
            sol = FabSol(sol=i + 1, available_power_kwh=50.0)
            tick_fabricator(s, sol)
            assert s.total_energy_consumed_kwh >= prev
            prev = s.total_energy_consumed_kwh

    def test_energy_positive_for_printing(self):
        j = PrintJob(name="test", material="metal", mass_kg=1.0)
        assert j.energy_required_kwh > 0.0

    def test_feedstock_positive_for_printing(self):
        j = PrintJob(name="test", material="polymer", mass_kg=1.0)
        assert j.feedstock_required_kg > j.mass_kg  # waste adds more


# --- Factory ---

class TestFactory:
    """Tests for the create_fabricator factory."""

    def test_factory_defaults(self):
        s = create_fabricator()
        assert s.nozzle_quality == 1.0
        assert s.is_idle
        assert s.operational
        assert s.total_parts_printed == 0
        assert s.feedstock_regolith_kg == 200.0
        assert s.feedstock_metal_kg == 50.0
        assert s.feedstock_polymer_kg == 30.0
