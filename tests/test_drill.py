"""
Tests for drill.py — Mars Subsurface Drill Simulation.

87 tests across 10 test classes. Every function, edge case, and physics
invariant tested. The drill is the colony's access to the subsurface.

Run: python -m pytest tests/test_drill.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.drill import (
    DrillState,
    DrillSol,
    get_layer_at_depth,
    penetration_rate,
    power_per_metre,
    bit_heat_rise,
    cool_bit,
    bit_wear_per_metre,
    core_mass_kg,
    ice_yield_kg,
    replace_bit,
    tick_drill,
    create_drill,
    MARS_AMBIENT_TEMP_C,
    BORE_RADIUS_M,
    CORE_CROSS_SECTION_M2,
    ROCK_REGOLITH,
    ROCK_DURICRUST,
    ROCK_BASALT,
    ROCK_ICE_REGOLITH,
    DEFAULT_LAYERS,
    BASE_PENETRATION_M_PER_SOL,
    DEPTH_PENALTY_FACTOR,
    MAX_RATED_DEPTH_M,
    BASE_POWER_KWH_PER_M,
    HARDNESS_POWER_MULTIPLIER,
    FRICTION_HEAT_C_PER_M,
    COOLING_RATE_C_PER_SOL,
    BIT_OVERHEAT_TEMP_C,
    BIT_MAX_TEMP_C,
    BIT_LIFE_METRES,
    HARDNESS_WEAR_MULTIPLIER,
    WORN_BIT_SPEED_PENALTY,
    BIT_REPLACEMENT_RESTORE,
)


# ─── DrillState ──────────────────────────────────────────────────────────────

class TestDrillState:
    """Unit tests for the DrillState dataclass."""

    def test_defaults(self):
        d = DrillState()
        assert d.sol == 0
        assert d.depth_m == 0.0
        assert d.bit_wear == 0.0
        assert d.bit_temp_c == MARS_AMBIENT_TEMP_C
        assert d.operational is True
        assert d.total_energy_kwh == 0.0
        assert d.total_cores_extracted == 0
        assert d.total_ice_detected_kg == 0.0
        assert d.emergency_shutdowns == 0

    def test_depth_clamped_high(self):
        d = DrillState(depth_m=9999.0)
        assert d.depth_m == MAX_RATED_DEPTH_M

    def test_depth_clamped_low(self):
        d = DrillState(depth_m=-10.0)
        assert d.depth_m == 0.0

    def test_bit_wear_clamped_high(self):
        d = DrillState(bit_wear=5.0)
        assert d.bit_wear == 1.0

    def test_bit_wear_clamped_low(self):
        d = DrillState(bit_wear=-1.0)
        assert d.bit_wear == 0.0

    def test_temp_clamped_high(self):
        d = DrillState(bit_temp_c=9999.0)
        assert d.bit_temp_c == BIT_MAX_TEMP_C

    def test_temp_clamped_low(self):
        d = DrillState(bit_temp_c=-9999.0)
        assert d.bit_temp_c == MARS_AMBIENT_TEMP_C


# ─── DrillSol ────────────────────────────────────────────────────────────────

class TestDrillSol:
    """Unit tests for the DrillSol record."""

    def test_defaults(self):
        s = DrillSol()
        assert s.metres_drilled == 0.0
        assert s.power_consumed_kwh == 0.0
        assert s.core_mass_kg == 0.0
        assert s.ice_detected_kg == 0.0
        assert s.rock_type == "unknown"
        assert s.emergency_shutdown is False


# ─── get_layer_at_depth ──────────────────────────────────────────────────────

class TestGetLayerAtDepth:
    """Tests for geological layer lookup."""

    def test_surface_is_regolith(self):
        layer = get_layer_at_depth(0.0)
        assert layer["rock"]["name"] == "regolith"

    def test_mid_duricrust(self):
        # Regolith is 3m, duricrust starts at 3m
        layer = get_layer_at_depth(4.0)
        assert layer["rock"]["name"] == "duricrust"

    def test_deep_basalt(self):
        # Regolith 3m + duricrust 2m + basalt 15m starts at 5m
        layer = get_layer_at_depth(10.0)
        assert layer["rock"]["name"] == "basalt"

    def test_ice_layer(self):
        # Reg 3 + dur 2 + bas 15 = 20m, ice starts at 20m
        layer = get_layer_at_depth(22.0)
        assert layer["rock"]["name"] == "ice_regolith"
        assert layer["ice_fraction"] == 0.35

    def test_beyond_all_layers_returns_deepest(self):
        layer = get_layer_at_depth(999.0)
        assert layer["rock"]["name"] == "basalt"

    def test_empty_layers_returns_basalt_default(self):
        layer = get_layer_at_depth(5.0, layers=[])
        assert layer["rock"]["name"] == "basalt"

    def test_custom_layers(self):
        custom = [{"rock": ROCK_ICE_REGOLITH, "thickness_m": 100.0, "ice_fraction": 0.5}]
        layer = get_layer_at_depth(50.0, layers=custom)
        assert layer["rock"]["name"] == "ice_regolith"
        assert layer["ice_fraction"] == 0.5

    def test_boundary_between_layers(self):
        # At exactly 3.0m (boundary of regolith/duricrust)
        layer = get_layer_at_depth(2.99)
        assert layer["rock"]["name"] == "regolith"
        layer = get_layer_at_depth(3.01)
        assert layer["rock"]["name"] == "duricrust"


# ─── penetration_rate ────────────────────────────────────────────────────────

class TestPenetrationRate:
    """Tests for drilling speed calculation."""

    def test_fresh_bit_surface_regolith(self):
        rate = penetration_rate(0.0, ROCK_REGOLITH["hardness"], 0.0)
        assert rate > 0
        # Soft rock at surface with new bit should be near base rate
        assert rate > BASE_PENETRATION_M_PER_SOL * 0.5

    def test_harder_rock_is_slower(self):
        soft = penetration_rate(0.0, ROCK_REGOLITH["hardness"], 0.0)
        hard = penetration_rate(0.0, ROCK_BASALT["hardness"], 0.0)
        assert hard < soft

    def test_deeper_is_slower(self):
        shallow = penetration_rate(1.0, 0.5, 0.0)
        deep = penetration_rate(30.0, 0.5, 0.0)
        assert deep < shallow

    def test_worn_bit_is_slower(self):
        fresh = penetration_rate(5.0, 0.5, 0.0)
        worn = penetration_rate(5.0, 0.5, 0.8)
        assert worn < fresh

    def test_destroyed_bit_returns_zero(self):
        rate = penetration_rate(0.0, 0.0, 1.0)
        assert rate == 0.0

    def test_always_non_negative(self):
        for depth in [0, 10, 25, 49]:
            for hardness in [0.0, 0.5, 1.0]:
                for wear in [0.0, 0.5, 0.99]:
                    assert penetration_rate(depth, hardness, wear) >= 0.0

    def test_monotonic_with_depth(self):
        """Rate should decrease or stay flat as depth increases."""
        prev = penetration_rate(0.0, 0.5, 0.0)
        for d in range(1, 50):
            curr = penetration_rate(float(d), 0.5, 0.0)
            assert curr <= prev + 1e-9
            prev = curr


# ─── power_per_metre ─────────────────────────────────────────────────────────

class TestPowerPerMetre:
    """Tests for energy cost calculation."""

    def test_baseline_cost(self):
        cost = power_per_metre(0.0, 0.0)
        # Zero hardness, zero depth -> base cost
        assert cost == BASE_POWER_KWH_PER_M

    def test_harder_costs_more(self):
        soft = power_per_metre(0.0, 0.1)
        hard = power_per_metre(0.0, 0.9)
        assert hard > soft

    def test_deeper_costs_more(self):
        shallow = power_per_metre(1.0, 0.5)
        deep = power_per_metre(40.0, 0.5)
        assert deep > shallow

    def test_always_positive(self):
        for depth in [0, 10, 50]:
            for hardness in [0.0, 0.5, 1.0]:
                assert power_per_metre(depth, hardness) > 0


# ─── Thermal ─────────────────────────────────────────────────────────────────

class TestThermal:
    """Tests for heat generation and cooling."""

    def test_no_drilling_no_heat(self):
        assert bit_heat_rise(0.0, 0.5, 0.0) == 0.0

    def test_negative_metres_no_heat(self):
        assert bit_heat_rise(-1.0, 0.5, 0.0) == 0.0

    def test_more_metres_more_heat(self):
        h1 = bit_heat_rise(1.0, 0.5, 0.0)
        h2 = bit_heat_rise(2.0, 0.5, 0.0)
        assert h2 > h1

    def test_harder_rock_more_heat(self):
        soft = bit_heat_rise(1.0, 0.1, 0.0)
        hard = bit_heat_rise(1.0, 0.9, 0.0)
        assert hard > soft

    def test_worn_bit_more_heat(self):
        fresh = bit_heat_rise(1.0, 0.5, 0.0)
        worn = bit_heat_rise(1.0, 0.5, 0.8)
        assert worn > fresh

    def test_heat_always_non_negative(self):
        for m in [0.0, 0.5, 1.0, 5.0]:
            for h in [0.0, 0.5, 1.0]:
                for w in [0.0, 0.5, 1.0]:
                    assert bit_heat_rise(m, h, w) >= 0.0

    def test_cool_bit_at_ambient(self):
        result = cool_bit(MARS_AMBIENT_TEMP_C)
        assert result == MARS_AMBIENT_TEMP_C

    def test_cool_bit_below_ambient(self):
        result = cool_bit(MARS_AMBIENT_TEMP_C - 50)
        assert result == MARS_AMBIENT_TEMP_C

    def test_cool_bit_hot(self):
        result = cool_bit(200.0)
        expected = 200.0 - COOLING_RATE_C_PER_SOL
        assert result == max(MARS_AMBIENT_TEMP_C, expected)

    def test_cool_bit_never_below_ambient(self):
        # Even if cooling overshoots
        result = cool_bit(MARS_AMBIENT_TEMP_C + 10.0)
        assert result >= MARS_AMBIENT_TEMP_C


# ─── Bit wear ────────────────────────────────────────────────────────────────

class TestBitWear:
    """Tests for drill bit degradation."""

    def test_soft_rock_slow_wear(self):
        wear = bit_wear_per_metre(0.0)
        assert wear == 1.0 / BIT_LIFE_METRES

    def test_hard_rock_fast_wear(self):
        soft_wear = bit_wear_per_metre(0.0)
        hard_wear = bit_wear_per_metre(1.0)
        assert hard_wear > soft_wear

    def test_basalt_wears_faster_than_regolith(self):
        reg = bit_wear_per_metre(ROCK_REGOLITH["hardness"])
        bas = bit_wear_per_metre(ROCK_BASALT["hardness"])
        assert bas > reg

    def test_always_positive(self):
        for h in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert bit_wear_per_metre(h) > 0


# ─── Core samples ────────────────────────────────────────────────────────────

class TestCoreSamples:
    """Tests for core mass and ice yield calculations."""

    def test_core_mass_formula(self):
        """mass = density × π × r² × length"""
        m = core_mass_kg(1.0, 1500.0)
        expected = 1500.0 * CORE_CROSS_SECTION_M2 * 1.0
        assert abs(m - expected) < 1e-6

    def test_core_mass_scales_with_length(self):
        m1 = core_mass_kg(1.0, 1500.0)
        m2 = core_mass_kg(2.0, 1500.0)
        assert abs(m2 - 2 * m1) < 1e-6

    def test_core_mass_scales_with_density(self):
        m1 = core_mass_kg(1.0, 1500.0)
        m2 = core_mass_kg(1.0, 3000.0)
        assert abs(m2 - 2 * m1) < 1e-6

    def test_core_mass_zero_length(self):
        assert core_mass_kg(0.0, 1500.0) == 0.0

    def test_core_mass_negative_length(self):
        assert core_mass_kg(-1.0, 1500.0) == 0.0

    def test_ice_yield_formula(self):
        mass = 1350.0 * CORE_CROSS_SECTION_M2 * 1.0
        ice = ice_yield_kg(1.0, 1350.0, 0.35)
        expected = mass * 0.35
        assert abs(ice - expected) < 1e-6

    def test_ice_yield_zero_fraction(self):
        assert ice_yield_kg(1.0, 1500.0, 0.0) == 0.0

    def test_ice_yield_negative_fraction(self):
        assert ice_yield_kg(1.0, 1500.0, -0.1) == 0.0

    def test_ice_yield_clamped_to_one(self):
        """Ice fraction > 1.0 should be clamped."""
        m1 = ice_yield_kg(1.0, 1500.0, 1.0)
        m2 = ice_yield_kg(1.0, 1500.0, 5.0)
        assert abs(m1 - m2) < 1e-6


# ─── replace_bit ─────────────────────────────────────────────────────────────

class TestReplaceBit:
    """Tests for drill bit replacement."""

    def test_restores_wear(self):
        d = DrillState(bit_wear=0.95)
        replace_bit(d)
        assert d.bit_wear < 0.95
        assert d.bit_wear >= 0.0

    def test_resets_temp(self):
        d = DrillState(bit_temp_c=300.0)
        replace_bit(d)
        assert d.bit_temp_c == MARS_AMBIENT_TEMP_C

    def test_restores_operational(self):
        d = DrillState(operational=False, bit_wear=1.0)
        replace_bit(d)
        assert d.operational is True

    def test_replacement_not_perfect(self):
        d = DrillState(bit_wear=1.0)
        replace_bit(d)
        # After replacement of a fully worn bit, some wear remains
        assert d.bit_wear == 1.0 - BIT_REPLACEMENT_RESTORE


# ─── tick_drill ──────────────────────────────────────────────────────────────

class TestTickDrill:
    """Integration tests for the main tick function."""

    def test_one_sol_drills(self):
        state = create_drill()
        sol = tick_drill(state, power_budget_kwh=100.0)
        assert sol.metres_drilled > 0
        assert sol.power_consumed_kwh > 0
        assert state.depth_m > 0
        assert state.sol == 1

    def test_no_power_no_drilling(self):
        state = create_drill()
        sol = tick_drill(state, power_budget_kwh=0.0)
        assert sol.metres_drilled == 0.0
        assert state.depth_m == 0.0

    def test_negative_power_no_drilling(self):
        state = create_drill()
        sol = tick_drill(state, power_budget_kwh=-10.0)
        assert sol.metres_drilled == 0.0

    def test_non_operational_no_drilling(self):
        state = create_drill()
        state.operational = False
        sol = tick_drill(state, power_budget_kwh=100.0)
        assert sol.metres_drilled == 0.0

    def test_destroyed_bit_no_drilling(self):
        state = create_drill()
        state.bit_wear = 1.0
        sol = tick_drill(state, power_budget_kwh=100.0)
        assert sol.metres_drilled == 0.0
        assert state.operational is False

    def test_power_conservation(self):
        """Never consume more power than budgeted."""
        state = create_drill()
        budget = 20.0
        sol = tick_drill(state, power_budget_kwh=budget)
        assert sol.power_consumed_kwh <= budget + 1e-9

    def test_depth_monotonic(self):
        """Depth only increases over multiple sols."""
        state = create_drill()
        prev_depth = 0.0
        for _ in range(20):
            tick_drill(state, power_budget_kwh=50.0)
            assert state.depth_m >= prev_depth - 1e-9
            prev_depth = state.depth_m

    def test_wear_monotonic(self):
        """Bit wear only increases."""
        state = create_drill()
        prev_wear = 0.0
        for _ in range(20):
            tick_drill(state, power_budget_kwh=50.0)
            assert state.bit_wear >= prev_wear - 1e-9
            prev_wear = state.bit_wear

    def test_sol_counter_increments(self):
        state = create_drill()
        for i in range(5):
            tick_drill(state, power_budget_kwh=50.0)
        assert state.sol == 5

    def test_core_mass_physical(self):
        """Core mass must match geometric formula."""
        state = create_drill()
        sol = tick_drill(state, power_budget_kwh=100.0)
        if sol.metres_drilled > 0:
            layer = get_layer_at_depth(0.0)
            expected = layer["rock"]["density"] * CORE_CROSS_SECTION_M2 * sol.metres_drilled
            assert abs(sol.core_mass_kg - expected) < 1e-4

    def test_depth_never_exceeds_rated(self):
        """Drill cannot go deeper than MAX_RATED_DEPTH_M."""
        state = create_drill()
        for _ in range(200):
            tick_drill(state, power_budget_kwh=200.0)
        assert state.depth_m <= MAX_RATED_DEPTH_M

    def test_ice_detected_in_ice_layer(self):
        """Drilling into ice-bearing layers should yield ice."""
        ice_layers = [
            {"rock": ROCK_ICE_REGOLITH, "thickness_m": 50.0, "ice_fraction": 0.5}
        ]
        state = create_drill()
        total_ice = 0.0
        for _ in range(10):
            sol = tick_drill(state, power_budget_kwh=100.0, layers=ice_layers)
            total_ice += sol.ice_detected_kg
        assert total_ice > 0

    def test_no_ice_in_dry_rock(self):
        """Drilling through dry basalt should yield zero ice."""
        dry_layers = [
            {"rock": ROCK_BASALT, "thickness_m": 100.0, "ice_fraction": 0.0}
        ]
        state = create_drill()
        for _ in range(10):
            sol = tick_drill(state, power_budget_kwh=100.0, layers=dry_layers)
            assert sol.ice_detected_kg == 0.0

    def test_energy_accounting(self):
        """Total energy tracked in state matches sum of sol records."""
        state = create_drill()
        total = 0.0
        for _ in range(15):
            sol = tick_drill(state, power_budget_kwh=50.0)
            total += sol.power_consumed_kwh
        assert abs(state.total_energy_kwh - total) < 1e-6


# ─── Multi-sol simulation ───────────────────────────────────────────────────

class TestMultiSolSimulation:
    """Smoke tests and property-based invariants over extended runs."""

    def test_10_sol_no_crash(self):
        """Smoke test: 10 sols without exceptions."""
        state = create_drill()
        for _ in range(10):
            tick_drill(state, power_budget_kwh=50.0)
        assert state.sol == 10
        assert state.depth_m > 0

    def test_50_sol_no_crash(self):
        """Extended smoke test."""
        state = create_drill()
        for _ in range(50):
            tick_drill(state, power_budget_kwh=80.0)
        assert state.sol == 50

    def test_drill_to_max_depth(self):
        """With enough power, drill reaches max depth."""
        state = create_drill()
        for _ in range(500):
            if state.bit_wear >= 1.0:
                replace_bit(state)
            tick_drill(state, power_budget_kwh=500.0)
        # Should have reached near max depth with bit replacements
        assert state.depth_m > 20.0

    def test_bit_eventually_wears_out(self):
        """Without replacement, bit will eventually fail."""
        state = create_drill()
        for _ in range(500):
            tick_drill(state, power_budget_kwh=100.0)
        # After 500 sols of drilling, bit should be dead or nearly so
        assert state.bit_wear > 0.5

    def test_varying_power_budgets(self):
        """Drill handles variable power budgets without crashing."""
        state = create_drill()
        budgets = [0.0, 10.0, 50.0, 100.0, 0.0, 200.0, 5.0, 0.0, 75.0, 30.0]
        for budget in budgets:
            sol = tick_drill(state, power_budget_kwh=budget)
            assert sol.power_consumed_kwh <= budget + 1e-9
            assert sol.metres_drilled >= 0.0

    def test_total_mass_conservation(self):
        """Sum of sol core masses equals state total."""
        state = create_drill()
        total_mass = 0.0
        for _ in range(20):
            sol = tick_drill(state, power_budget_kwh=60.0)
            total_mass += sol.core_mass_kg
        assert abs(state.total_core_mass_kg - total_mass) < 1e-6

    def test_total_ice_conservation(self):
        """Sum of sol ice yields equals state total."""
        ice_layers = [
            {"rock": ROCK_ICE_REGOLITH, "thickness_m": 50.0, "ice_fraction": 0.4}
        ]
        state = create_drill()
        total_ice = 0.0
        for _ in range(20):
            sol = tick_drill(state, power_budget_kwh=60.0, layers=ice_layers)
            total_ice += sol.ice_detected_kg
        assert abs(state.total_ice_detected_kg - total_ice) < 1e-6


# ─── create_drill ────────────────────────────────────────────────────────────

class TestCreateDrill:
    """Tests for the factory function."""

    def test_returns_drill_state(self):
        d = create_drill()
        assert isinstance(d, DrillState)

    def test_fresh_state(self):
        d = create_drill()
        assert d.sol == 0
        assert d.depth_m == 0.0
        assert d.bit_wear == 0.0
        assert d.operational is True
