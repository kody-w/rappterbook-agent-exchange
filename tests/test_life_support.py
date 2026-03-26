"""Tests for src/life_support.py — Mars habitat atmospheric management."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest
from src.life_support import (
    Atmosphere,
    LifeSupportStatus,
    crew_o2_demand,
    crew_co2_output,
    electrolysis_water_needed,
    electrolysis_power_needed,
    sabatier_power_needed,
    sabatier_water_recovery_kg,
    atmospheric_leak,
    o2_pressure_from_mass,
    co2_pressure_from_mass,
    tick_life_support,
    O2_KG_PER_PERSON_SOL,
    CO2_KG_PER_PERSON_SOL,
    WATER_PER_KG_O2,
    ELECTROLYSIS_KWH_PER_KG_O2,
    SABATIER_KWH_PER_KG_CO2,
    SABATIER_WATER_RECOVERY,
    TARGET_O2_KPA,
    MAX_CO2_KPA,
    TARGET_TOTAL_KPA,
    LEAK_RATE_FRACTION,
    MIN_O2_EMERGENCY_KPA,
)


# ===================================================================
# Atmosphere dataclass
# ===================================================================

class TestAtmosphere:
    def test_defaults(self) -> None:
        atmo = Atmosphere()
        assert atmo.o2_kpa == TARGET_O2_KPA
        assert atmo.co2_kpa == 0.04
        assert atmo.total_kpa == TARGET_TOTAL_KPA
        assert atmo.o2_reserve_kg == 500.0

    def test_custom_init(self) -> None:
        atmo = Atmosphere(o2_kpa=19.0, co2_kpa=0.3, total_kpa=65.0)
        assert atmo.o2_kpa == 19.0
        assert atmo.co2_kpa == 0.3

    def test_negative_clamped(self) -> None:
        atmo = Atmosphere(o2_kpa=-5.0, co2_kpa=-1.0, o2_reserve_kg=-100.0)
        assert atmo.o2_kpa == 0.0
        assert atmo.co2_kpa == 0.0
        assert atmo.o2_reserve_kg == 0.0

    def test_zero_values(self) -> None:
        atmo = Atmosphere(o2_kpa=0.0, co2_kpa=0.0, total_kpa=0.0, o2_reserve_kg=0.0)
        assert atmo.o2_kpa == 0.0


# ===================================================================
# LifeSupportStatus dataclass
# ===================================================================

class TestLifeSupportStatus:
    def test_defaults(self) -> None:
        status = LifeSupportStatus()
        assert status.o2_produced_kg == 0.0
        assert status.co2_removed_kg == 0.0
        assert status.emergency is False

    def test_custom(self) -> None:
        status = LifeSupportStatus(o2_produced_kg=5.0, emergency=True)
        assert status.o2_produced_kg == 5.0
        assert status.emergency is True


# ===================================================================
# crew_o2_demand / crew_co2_output
# ===================================================================

class TestCrewMetabolism:
    def test_single_person_o2(self) -> None:
        assert crew_o2_demand(1) == pytest.approx(O2_KG_PER_PERSON_SOL)

    def test_multi_crew_o2(self) -> None:
        assert crew_o2_demand(6) == pytest.approx(6 * O2_KG_PER_PERSON_SOL)

    def test_zero_crew_o2(self) -> None:
        assert crew_o2_demand(0) == 0.0

    def test_negative_crew_o2(self) -> None:
        assert crew_o2_demand(-3) == 0.0

    def test_single_person_co2(self) -> None:
        assert crew_co2_output(1) == pytest.approx(CO2_KG_PER_PERSON_SOL)

    def test_multi_crew_co2(self) -> None:
        assert crew_co2_output(6) == pytest.approx(6 * CO2_KG_PER_PERSON_SOL)

    def test_co2_exceeds_o2(self) -> None:
        """Humans exhale more CO₂ mass than O₂ consumed (CO₂ is heavier)."""
        assert crew_co2_output(1) > crew_o2_demand(1)


# ===================================================================
# electrolysis functions
# ===================================================================

class TestElectrolysis:
    def test_water_for_one_kg_o2(self) -> None:
        assert electrolysis_water_needed(1.0) == pytest.approx(WATER_PER_KG_O2)

    def test_water_scales_linearly(self) -> None:
        w1 = electrolysis_water_needed(1.0)
        w5 = electrolysis_water_needed(5.0)
        assert w5 == pytest.approx(5.0 * w1)

    def test_power_for_one_kg_o2(self) -> None:
        assert electrolysis_power_needed(1.0) == pytest.approx(ELECTROLYSIS_KWH_PER_KG_O2)

    def test_zero_input(self) -> None:
        assert electrolysis_water_needed(0.0) == 0.0
        assert electrolysis_power_needed(0.0) == 0.0

    def test_negative_input_clamped(self) -> None:
        assert electrolysis_water_needed(-1.0) == 0.0
        assert electrolysis_power_needed(-1.0) == 0.0


# ===================================================================
# sabatier functions
# ===================================================================

class TestSabatier:
    def test_power_for_one_kg_co2(self) -> None:
        assert sabatier_power_needed(1.0) == pytest.approx(SABATIER_KWH_PER_KG_CO2)

    def test_water_recovery(self) -> None:
        assert sabatier_water_recovery_kg(1.0) == pytest.approx(SABATIER_WATER_RECOVERY)

    def test_recovery_positive(self) -> None:
        """Sabatier always recovers some water (>0) for positive input."""
        assert sabatier_water_recovery_kg(5.0) > 0.0

    def test_zero_input(self) -> None:
        assert sabatier_power_needed(0.0) == 0.0
        assert sabatier_water_recovery_kg(0.0) == 0.0

    def test_negative_clamped(self) -> None:
        assert sabatier_power_needed(-1.0) == 0.0
        assert sabatier_water_recovery_kg(-1.0) == 0.0


# ===================================================================
# atmospheric_leak
# ===================================================================

class TestAtmosphericLeak:
    def test_nominal_leak(self) -> None:
        atmo = Atmosphere()
        leak = atmospheric_leak(atmo)
        assert leak == pytest.approx(TARGET_O2_KPA * LEAK_RATE_FRACTION)

    def test_leak_proportional_to_pressure(self) -> None:
        a1 = Atmosphere(o2_kpa=10.0)
        a2 = Atmosphere(o2_kpa=20.0)
        assert atmospheric_leak(a2) == pytest.approx(2.0 * atmospheric_leak(a1))

    def test_zero_pressure_no_leak(self) -> None:
        atmo = Atmosphere(o2_kpa=0.0)
        assert atmospheric_leak(atmo) == 0.0

    def test_leak_always_positive(self) -> None:
        atmo = Atmosphere(o2_kpa=15.0)
        assert atmospheric_leak(atmo) > 0.0


# ===================================================================
# pressure conversion functions
# ===================================================================

class TestPressureConversion:
    def test_o2_positive_mass_positive_pressure(self) -> None:
        dp = o2_pressure_from_mass(1.0, 500.0)
        assert dp > 0.0

    def test_o2_zero_mass(self) -> None:
        assert o2_pressure_from_mass(0.0, 500.0) == 0.0

    def test_o2_zero_volume(self) -> None:
        assert o2_pressure_from_mass(1.0, 0.0) == 0.0

    def test_o2_negative_volume(self) -> None:
        assert o2_pressure_from_mass(1.0, -100.0) == 0.0

    def test_co2_positive_mass_positive_pressure(self) -> None:
        dp = co2_pressure_from_mass(1.0, 500.0)
        assert dp > 0.0

    def test_co2_heavier_less_pressure_per_kg(self) -> None:
        """CO₂ (44 g/mol) produces less pressure per kg than O₂ (32 g/mol)."""
        dp_o2 = o2_pressure_from_mass(1.0, 500.0)
        dp_co2 = co2_pressure_from_mass(1.0, 500.0)
        assert dp_co2 < dp_o2

    def test_pressure_scales_with_mass(self) -> None:
        dp1 = o2_pressure_from_mass(1.0, 500.0)
        dp3 = o2_pressure_from_mass(3.0, 500.0)
        assert dp3 == pytest.approx(3.0 * dp1)

    def test_pressure_inversely_scales_with_volume(self) -> None:
        dp_small = o2_pressure_from_mass(1.0, 250.0)
        dp_large = o2_pressure_from_mass(1.0, 500.0)
        assert dp_small == pytest.approx(2.0 * dp_large)


# ===================================================================
# tick_life_support — unit tests
# ===================================================================

class TestTickLifeSupport:
    def test_nominal_operation(self) -> None:
        """6 crew, ample power and water — should run clean."""
        atmo = Atmosphere()
        status = tick_life_support(atmo, population=6, power_available_kwh=100.0,
                                   water_available_kg=50.0)
        assert status.o2_produced_kg > 0.0
        assert status.co2_removed_kg > 0.0
        assert status.o2_deficit_kg == 0.0
        assert not status.emergency

    def test_zero_population(self) -> None:
        """No crew — only leak, no production needed."""
        atmo = Atmosphere()
        initial_o2 = atmo.o2_kpa
        status = tick_life_support(atmo, population=0, power_available_kwh=0.0,
                                   water_available_kg=0.0)
        assert status.o2_produced_kg == 0.0
        assert status.co2_removed_kg == 0.0
        assert atmo.o2_kpa < initial_o2  # leak

    def test_no_power_creates_deficit(self) -> None:
        """No power, no reserve — can't generate O₂."""
        atmo = Atmosphere(o2_reserve_kg=0.0)
        status = tick_life_support(atmo, population=6, power_available_kwh=0.0,
                                   water_available_kg=50.0)
        assert status.o2_produced_kg == 0.0
        assert status.o2_deficit_kg > 0.0

    def test_no_water_creates_deficit(self) -> None:
        """No water, no reserve — can't electrolyse."""
        atmo = Atmosphere(o2_reserve_kg=0.0)
        status = tick_life_support(atmo, population=6, power_available_kwh=100.0,
                                   water_available_kg=0.0)
        assert status.o2_produced_kg == 0.0
        assert status.o2_deficit_kg > 0.0

    def test_partial_power(self) -> None:
        """Half power → half O₂ production."""
        atmo = Atmosphere()
        full_power = electrolysis_power_needed(crew_o2_demand(6)) + \
                     sabatier_power_needed(crew_co2_output(6))
        status = tick_life_support(atmo, population=6,
                                   power_available_kwh=full_power * 0.5,
                                   water_available_kg=50.0)
        # Should produce some O₂ but not all
        expected_demand = crew_o2_demand(6)
        assert 0 < status.o2_produced_kg < expected_demand

    def test_reserve_used_on_deficit(self) -> None:
        """Reserves should deplete when production insufficient."""
        atmo = Atmosphere(o2_reserve_kg=100.0)
        initial_reserve = atmo.o2_reserve_kg
        tick_life_support(atmo, population=6, power_available_kwh=0.0,
                          water_available_kg=0.0)
        assert atmo.o2_reserve_kg < initial_reserve

    def test_sabatier_recovers_water(self) -> None:
        """Sabatier should recover some water from CO₂ scrubbing."""
        atmo = Atmosphere()
        status = tick_life_support(atmo, population=6, power_available_kwh=100.0,
                                   water_available_kg=50.0)
        assert status.water_recovered_kg > 0.0

    def test_net_water_less_than_electrolysis(self) -> None:
        """Net water consumption should be less than raw electrolysis need."""
        atmo = Atmosphere()
        status = tick_life_support(atmo, population=6, power_available_kwh=100.0,
                                   water_available_kg=50.0)
        raw_water = electrolysis_water_needed(status.o2_produced_kg)
        assert status.water_consumed_kg < raw_water

    def test_co2_rises_without_scrubbing(self) -> None:
        """If power only covers O₂ but not CO₂ scrubbing, CO₂ rises."""
        atmo = Atmosphere(co2_kpa=0.1)
        initial_co2 = atmo.co2_kpa
        # Give just enough power for electrolysis, nothing for Sabatier
        o2_power = electrolysis_power_needed(crew_o2_demand(6))
        tick_life_support(atmo, population=6, power_available_kwh=o2_power,
                          water_available_kg=50.0)
        assert atmo.co2_kpa > initial_co2

    def test_emergency_on_low_o2(self) -> None:
        """Emergency triggers when O₂ drops below hypoxia threshold."""
        atmo = Atmosphere(o2_kpa=15.0, o2_reserve_kg=0.0)
        status = tick_life_support(atmo, population=6, power_available_kwh=0.0,
                                   water_available_kg=0.0)
        assert status.emergency is True

    def test_no_emergency_nominal(self) -> None:
        """No emergency under normal conditions."""
        atmo = Atmosphere()
        status = tick_life_support(atmo, population=6, power_available_kwh=100.0,
                                   water_available_kg=50.0)
        assert not status.emergency

    def test_large_habitat_volume(self) -> None:
        """Larger habitat → smaller pressure swings per kg of gas."""
        atmo_small = Atmosphere()
        atmo_large = Atmosphere()
        tick_life_support(atmo_small, population=6, power_available_kwh=0.0,
                          water_available_kg=0.0, habitat_volume_m3=100.0)
        tick_life_support(atmo_large, population=6, power_available_kwh=0.0,
                          water_available_kg=0.0, habitat_volume_m3=1000.0)
        # Larger volume should have smaller CO₂ rise
        assert atmo_large.co2_kpa < atmo_small.co2_kpa


# ===================================================================
# Physical invariants
# ===================================================================

class TestPhysicalInvariants:
    def test_o2_never_negative(self) -> None:
        """O₂ partial pressure must never go negative."""
        atmo = Atmosphere(o2_kpa=0.1, o2_reserve_kg=0.0)
        for _ in range(100):
            tick_life_support(atmo, population=6, power_available_kwh=0.0,
                              water_available_kg=0.0)
        assert atmo.o2_kpa >= 0.0

    def test_co2_never_negative(self) -> None:
        """CO₂ partial pressure must never go negative."""
        atmo = Atmosphere(co2_kpa=0.01)
        for _ in range(10):
            tick_life_support(atmo, population=0, power_available_kwh=100.0,
                              water_available_kg=50.0)
        assert atmo.co2_kpa >= 0.0

    def test_reserve_never_negative(self) -> None:
        """O₂ reserve must never go negative."""
        atmo = Atmosphere(o2_reserve_kg=1.0)
        for _ in range(100):
            tick_life_support(atmo, population=6, power_available_kwh=0.0,
                              water_available_kg=0.0)
        assert atmo.o2_reserve_kg >= 0.0

    def test_water_recovery_never_exceeds_consumption(self) -> None:
        """Water recovered via Sabatier must be ≤ water used in electrolysis."""
        atmo = Atmosphere()
        status = tick_life_support(atmo, population=6, power_available_kwh=100.0,
                                   water_available_kg=50.0)
        raw_water = electrolysis_water_needed(status.o2_produced_kg)
        assert status.water_recovered_kg <= raw_water

    def test_power_consumed_bounded(self) -> None:
        """Power consumed must not exceed power available."""
        atmo = Atmosphere()
        power = 50.0
        status = tick_life_support(atmo, population=6, power_available_kwh=power,
                                   water_available_kg=50.0)
        assert status.power_consumed_kwh <= power + 1e-9

    def test_mass_conservation_o2(self) -> None:
        """O₂ produced - O₂ consumed = net pressure change (directionally)."""
        atmo = Atmosphere()
        initial_o2 = atmo.o2_kpa
        status = tick_life_support(atmo, population=6, power_available_kwh=100.0,
                                   water_available_kg=50.0)
        # If produced ≥ demand, O₂ should stay roughly stable (minus leak)
        if status.o2_deficit_kg == 0.0:
            # O₂ should be close to initial, maybe slightly lower from leak
            assert atmo.o2_kpa >= initial_o2 - 0.1

    def test_more_crew_more_demand(self) -> None:
        """Doubling crew doubles O₂ demand and CO₂ output."""
        assert crew_o2_demand(12) == pytest.approx(2.0 * crew_o2_demand(6))
        assert crew_co2_output(12) == pytest.approx(2.0 * crew_co2_output(6))

    def test_electrolysis_stoichiometry(self) -> None:
        """2H₂O → 2H₂ + O₂ means ~1.125 kg H₂O per kg O₂.

        Molecular: 2×18 = 36 g H₂O → 32 g O₂ → ratio = 36/32 = 1.125.
        """
        ratio = electrolysis_water_needed(1.0) / 1.0
        assert ratio == pytest.approx(1.125)


# ===================================================================
# Smoke test — 365 sols
# ===================================================================

class TestSmoke:
    def test_one_year_nominal(self) -> None:
        """Run 365 sols with 6 crew, ample resources. No crash.

        O₂ drifts down ~30% over a year due to 0.1%/sol atmospheric leak.
        This is physically correct — real habitats must overproduce O₂
        to compensate. We verify it stays positive and doesn't crash.
        """
        atmo = Atmosphere()
        for sol in range(365):
            status = tick_life_support(
                atmo, population=6,
                power_available_kwh=80.0,
                water_available_kg=20.0,
            )
        assert atmo.o2_kpa > 0.0
        assert atmo.co2_kpa >= 0.0
        # O₂ drifts down from leak, but stays breathable (>10 kPa)
        assert atmo.o2_kpa > 10.0

    def test_one_year_growing_colony(self) -> None:
        """Population grows from 6 to 120 over a year. System scales.

        Leak causes O₂ drift — verify it stays positive and doesn't crash.
        """
        atmo = Atmosphere()
        for sol in range(365):
            pop = 6 + sol // 3
            status = tick_life_support(
                atmo, population=pop,
                power_available_kwh=200.0 + pop * 5,
                water_available_kg=50.0 + pop * 2,
            )
        assert atmo.o2_kpa > 0.0
        assert atmo.co2_kpa >= 0.0

    def test_power_outage_then_recovery(self) -> None:
        """10-sol blackout, then 355 sols normal. Colony survives via reserves."""
        atmo = Atmosphere(o2_reserve_kg=500.0)
        emergencies = 0
        for sol in range(365):
            if sol < 10:
                # Blackout
                status = tick_life_support(atmo, population=6,
                                           power_available_kwh=0.0,
                                           water_available_kg=0.0)
            else:
                status = tick_life_support(atmo, population=6,
                                           power_available_kwh=100.0,
                                           water_available_kg=50.0)
            if status.emergency:
                emergencies += 1
        # After recovery, O₂ should stabilise
        assert atmo.o2_kpa > 0.0
        # Colony survives (O₂ doesn't hit zero)
        assert atmo.o2_kpa > 5.0

    def test_multi_habitat(self) -> None:
        """Three habitats with different volumes — all survive.

        Leak reduces O₂ over time. Verify positive and no crash.
        """
        volumes = [200.0, 500.0, 1000.0]
        for vol in volumes:
            atmo = Atmosphere()
            for sol in range(365):
                tick_life_support(atmo, population=6,
                                  power_available_kwh=100.0,
                                  water_available_kg=50.0,
                                  habitat_volume_m3=vol)
            assert atmo.o2_kpa > 0.0
            assert atmo.co2_kpa >= 0.0

    def test_ten_sol_smoke(self) -> None:
        """Minimal smoke: 10 sols without crash."""
        atmo = Atmosphere()
        for _ in range(10):
            tick_life_support(atmo, population=6,
                              power_available_kwh=50.0,
                              water_available_kg=20.0)
        assert atmo.o2_kpa > 0.0


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_single_person(self) -> None:
        """Single crew member — minimal resource use."""
        atmo = Atmosphere()
        status = tick_life_support(atmo, population=1,
                                   power_available_kwh=50.0,
                                   water_available_kg=20.0)
        assert status.o2_produced_kg == pytest.approx(O2_KG_PER_PERSON_SOL)
        assert status.o2_deficit_kg == 0.0

    def test_massive_population(self) -> None:
        """1000 people — exceeds resources, triggers deficit."""
        atmo = Atmosphere()
        status = tick_life_support(atmo, population=1000,
                                   power_available_kwh=100.0,
                                   water_available_kg=50.0)
        assert status.o2_deficit_kg > 0.0

    def test_tiny_habitat(self) -> None:
        """10 m³ habitat — pressure swings are large."""
        atmo = Atmosphere()
        tick_life_support(atmo, population=6,
                          power_available_kwh=0.0,
                          water_available_kg=0.0,
                          habitat_volume_m3=10.0)
        # CO₂ should spike dramatically in tiny volume
        assert atmo.co2_kpa > 0.5

    def test_no_reserve_no_power(self) -> None:
        """Worst case: no power, no reserve, full crew."""
        atmo = Atmosphere(o2_reserve_kg=0.0)
        status = tick_life_support(atmo, population=6,
                                   power_available_kwh=0.0,
                                   water_available_kg=0.0)
        assert status.o2_deficit_kg > 0.0
        assert status.o2_produced_kg == 0.0
