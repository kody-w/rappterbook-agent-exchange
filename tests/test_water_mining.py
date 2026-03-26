"""
Tests for water_mining.py — Mars ice extraction model.

Run: python -m pytest tests/test_water_mining.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.water_mining import (
    IceDeposit,
    mine_water_sol,
    temperature_efficiency,
    create_colony_deposit,
    ICE_SUBLIMATION_KJ_KG,
    KJ_PER_KWH,
    COLD_EFFICIENCY_FLOOR,
    WARM_EFFICIENCY_CEIL,
)


# --- Temperature efficiency tests ---


class TestTemperatureEfficiency:
    def test_cold_floor(self) -> None:
        """At -120C, efficiency hits the floor."""
        assert temperature_efficiency(-120.0) == COLD_EFFICIENCY_FLOOR

    def test_warm_ceiling(self) -> None:
        """At 0C, efficiency hits the ceiling."""
        assert temperature_efficiency(0.0) == WARM_EFFICIENCY_CEIL

    def test_monotonic(self) -> None:
        """Warmer is always more efficient."""
        prev = 0.0
        for t in range(-120, 1, 5):
            eff = temperature_efficiency(float(t))
            assert eff >= prev, f"Efficiency dropped at {t}C"
            prev = eff

    def test_bounded(self) -> None:
        """Efficiency always in [floor, ceiling] even for extreme inputs."""
        for t in range(-200, 50, 3):
            eff = temperature_efficiency(float(t))
            assert COLD_EFFICIENCY_FLOOR <= eff <= WARM_EFFICIENCY_CEIL

    def test_mars_mean_temp(self) -> None:
        """At Mars mean (-60C), efficiency is mid-range."""
        eff = temperature_efficiency(-60.0)
        assert COLD_EFFICIENCY_FLOOR < eff < WARM_EFFICIENCY_CEIL


# --- IceDeposit tests ---


class TestIceDeposit:
    def test_concentration_clamped_high(self) -> None:
        d = IceDeposit(concentration=1.5, depth_m=1.0, reserve_kg=1000)
        assert d.concentration == 1.0

    def test_concentration_clamped_low(self) -> None:
        d = IceDeposit(concentration=-0.5, depth_m=1.0, reserve_kg=1000)
        assert d.concentration == 0.0

    def test_reserve_nonnegative(self) -> None:
        d = IceDeposit(concentration=0.1, depth_m=1.0, reserve_kg=-50)
        assert d.reserve_kg == 0.0

    def test_depth_minimum(self) -> None:
        d = IceDeposit(concentration=0.1, depth_m=-1.0, reserve_kg=1000)
        assert d.depth_m >= 0.1


# --- mine_water_sol tests ---


class TestMineWaterSol:
    def test_zero_power_zero_water(self) -> None:
        """No power = no water."""
        d = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=10000)
        water, power = mine_water_sol(0.0, -60.0, d)
        assert water == 0.0
        assert power == 0.0

    def test_positive_extraction(self) -> None:
        """With power and ice, we get water."""
        d = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=100000)
        water, power = mine_water_sol(50.0, -60.0, d)
        assert water > 0, "Should extract some water"
        assert power > 0, "Should consume some power"

    def test_conservation_reserve(self) -> None:
        """Never extract more water than the reserve."""
        d = IceDeposit(concentration=0.5, depth_m=1.0, reserve_kg=1.0)
        water, _ = mine_water_sol(1000.0, 0.0, d)
        assert water <= 1.0, f"Extracted {water} from 1.0 kg reserve"

    def test_conservation_power(self) -> None:
        """Never consume more power than allocated."""
        d = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=100000)
        _, power = mine_water_sol(10.0, -60.0, d)
        assert power <= 10.0

    def test_reserve_depletes(self) -> None:
        """After extraction, reserve decreases."""
        d = IceDeposit(concentration=0.3, depth_m=1.0, reserve_kg=5000)
        initial = d.reserve_kg
        mine_water_sol(100.0, -40.0, d)
        assert d.reserve_kg < initial

    def test_empty_deposit_zero(self) -> None:
        """Empty deposit yields nothing."""
        d = IceDeposit(concentration=0.3, depth_m=1.0, reserve_kg=0.0)
        water, power = mine_water_sol(100.0, -40.0, d)
        assert water == 0.0
        assert power == 0.0

    def test_warmer_more_water(self) -> None:
        """Warmer temperatures extract more water (same power)."""
        d1 = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=100000)
        d2 = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=100000)
        cold_water, _ = mine_water_sol(50.0, -100.0, d1)
        warm_water, _ = mine_water_sol(50.0, -20.0, d2)
        assert warm_water > cold_water

    def test_broken_drill_zero(self) -> None:
        """Drill condition 0 = no extraction."""
        d = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=10000)
        water, power = mine_water_sol(50.0, -60.0, d, drill_condition=0.0)
        assert water == 0.0

    def test_degraded_drill_less(self) -> None:
        """Degraded drill extracts less than fresh drill."""
        d1 = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=100000)
        d2 = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=100000)
        fresh, _ = mine_water_sol(50.0, -60.0, d1, drill_condition=1.0)
        worn, _ = mine_water_sol(50.0, -60.0, d2, drill_condition=0.3)
        assert worn < fresh

    def test_higher_concentration_more_water(self) -> None:
        """Richer deposits yield more water per sol."""
        d_lean = IceDeposit(concentration=0.05, depth_m=1.0, reserve_kg=100000)
        d_rich = IceDeposit(concentration=0.50, depth_m=1.0, reserve_kg=100000)
        lean_water, _ = mine_water_sol(50.0, -60.0, d_lean)
        rich_water, _ = mine_water_sol(50.0, -60.0, d_rich)
        assert rich_water > lean_water

    def test_ten_sol_smoke(self) -> None:
        """Run 10 sols of mining without crash."""
        d = IceDeposit(concentration=0.15, depth_m=2.0, reserve_kg=50000)
        total_water = 0.0
        for _ in range(10):
            water, _ = mine_water_sol(30.0, -60.0, d)
            total_water += water
            assert d.reserve_kg >= 0
        assert total_water > 0
        assert d.reserve_kg < 50000


# --- create_colony_deposit tests ---


class TestCreateColonyDeposit:
    def test_conservative(self) -> None:
        d = create_colony_deposit("conservative")
        assert d.concentration > 0.2, "Conservative should have rich deposit"
        assert d.reserve_kg > 400000

    def test_balanced(self) -> None:
        d = create_colony_deposit("balanced")
        assert 0.05 < d.concentration < 0.5

    def test_aggressive(self) -> None:
        d = create_colony_deposit("aggressive")
        assert d.concentration < 0.15, "Aggressive should have lean deposit"

    def test_unknown_defaults_balanced(self) -> None:
        d = create_colony_deposit("unknown")
        balanced = create_colony_deposit("balanced")
        assert d.concentration == balanced.concentration


# --- Physical bounds property test ---


class TestPhysicalBounds:
    def test_energy_conservation(self) -> None:
        """For any inputs, water yield obeys energy conservation.

        Upper bound: all power goes to sublimation of pure ice.
        max_water = power * KJ_PER_KWH / ICE_SUBLIMATION_KJ_KG
        """
        for power in [1, 10, 50, 100, 500]:
            for conc in [0.01, 0.1, 0.3, 0.6, 1.0]:
                for temp in [-120, -80, -40, 0]:
                    d = IceDeposit(
                        concentration=conc, depth_m=1.0, reserve_kg=1e9
                    )
                    water, consumed = mine_water_sol(
                        float(power), float(temp), d
                    )
                    # Physical upper bound: perfect efficiency, all power to sublimation
                    theoretical_max = power * KJ_PER_KWH / ICE_SUBLIMATION_KJ_KG
                    assert water <= theoretical_max * 1.01, (
                        f"Violated energy bound: {water:.2f} > {theoretical_max:.2f}"
                    )
                    assert water >= 0
                    assert consumed >= 0
                    assert consumed <= power
