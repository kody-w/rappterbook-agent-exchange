"""Tests for the ecology organ (engine v9.0)."""
from __future__ import annotations

import pytest
from src.mars100.ecology import (
    EcologyState,
    EcologyMilestones,
    EcologyTickResult,
    MARS_PRESSURE_ATM,
    MARS_TEMP_C,
    MARS_O2_FRACTION,
    MARS_CO2_FRACTION,
    LICHEN_PRESSURE_THRESHOLD,
    LICHEN_TEMP_THRESHOLD,
    WATER_CYCLE_TEMP_THRESHOLD,
    MASK_BREATHING_PARTIAL_PRESSURE,
    OUTDOOR_FARMING_SOIL_THRESHOLD,
    OUTDOOR_FARMING_TEMP_THRESHOLD,
    compute_terraforming_contribution,
    compute_event_ecology_effects,
    check_milestones,
    compute_temperature,
    compute_dust_storm_modifier,
    compute_ecology_modifiers,
    tick_ecology,
    _update_gas_fractions,
    _clamp,
)


# ── helpers ──────────────────────────────────────────────────────────

def _fresh() -> EcologyState:
    """Fresh Mars ecology at t=0."""
    return EcologyState()


def _assert_fractions_sum_to_one(eco: EcologyState, tol: float = 1e-9) -> None:
    """Assert gas fractions sum to 1.0 within tolerance."""
    total = eco.o2_fraction + eco.co2_fraction + eco.other_fraction
    assert abs(total - 1.0) < tol, f"fractions sum to {total}, not 1.0"


def _assert_bounds(eco: EcologyState) -> None:
    """Assert all ecology values are in physical bounds."""
    assert eco.atmosphere_pressure >= MARS_PRESSURE_ATM
    assert eco.atmosphere_pressure <= 2.0
    assert 0.0 <= eco.o2_fraction <= 1.0
    assert 0.0 <= eco.co2_fraction <= 1.0
    assert eco.other_fraction >= 0.0
    assert MARS_TEMP_C - 1 <= eco.temperature_avg <= 30.0
    assert 0.0 <= eco.soil_fertility <= 1.0
    assert 0.0 <= eco.water_cycle_strength <= 1.0
    assert 0.0 <= eco.biodiversity_index <= 1.0
    assert 0.0 <= eco.outdoor_coverage <= 1.0
    _assert_fractions_sum_to_one(eco)


# ── EcologyState basics ─────────────────────────────────────────────

class TestEcologyState:

    def test_initial_values(self):
        eco = _fresh()
        assert eco.atmosphere_pressure == MARS_PRESSURE_ATM
        assert eco.temperature_avg == MARS_TEMP_C
        assert eco.o2_fraction == MARS_O2_FRACTION
        assert eco.co2_fraction == MARS_CO2_FRACTION
        assert eco.soil_fertility == 0.0
        assert eco.water_cycle_strength == 0.0
        assert eco.biodiversity_index == 0.0
        assert eco.outdoor_coverage == 0.0
        _assert_fractions_sum_to_one(eco)

    def test_gas_fractions_sum_to_one(self):
        eco = _fresh()
        _assert_fractions_sum_to_one(eco)

    def test_o2_partial_pressure(self):
        eco = _fresh()
        expected = MARS_PRESSURE_ATM * MARS_O2_FRACTION
        assert abs(eco.o2_partial_pressure - expected) < 1e-12

    def test_other_fraction(self):
        eco = _fresh()
        assert eco.other_fraction > 0.04  # ~4.55% N₂+Ar

    def test_roundtrip_serialization(self):
        eco = _fresh()
        eco.soil_fertility = 0.3
        eco.milestones.lichen_viable = True
        d = eco.to_dict()
        eco2 = EcologyState.from_dict(d)
        assert abs(eco2.soil_fertility - 0.3) < 1e-6
        assert eco2.milestones.lichen_viable is True
        _assert_fractions_sum_to_one(eco2)


# ── Gas fraction invariant ───────────────────────────────────────────

class TestGasFractions:

    def test_update_preserves_sum(self):
        eco = _fresh()
        _update_gas_fractions(eco, 0.01)
        _assert_fractions_sum_to_one(eco)

    def test_o2_increase_decreases_co2(self):
        eco = _fresh()
        old_co2 = eco.co2_fraction
        _update_gas_fractions(eco, 0.01)
        assert eco.co2_fraction < old_co2

    def test_other_stays_constant(self):
        eco = _fresh()
        other_before = eco.other_fraction
        _update_gas_fractions(eco, 0.05)
        assert abs(eco.other_fraction - other_before) < 1e-9

    def test_extreme_o2_clamped(self):
        eco = _fresh()
        _update_gas_fractions(eco, 10.0)  # way too much
        _assert_fractions_sum_to_one(eco)
        assert eco.o2_fraction < 1.0
        assert eco.co2_fraction > 0.0

    def test_negative_o2_clamped(self):
        eco = _fresh()
        _update_gas_fractions(eco, -10.0)
        _assert_fractions_sum_to_one(eco)
        assert eco.o2_fraction >= 0.0


# ── Terraforming contribution ────────────────────────────────────────

class TestTerraformingContribution:

    def test_zero_actions_zero_contribution(self):
        c = compute_terraforming_contribution({}, [])
        assert c["pressure"] == 0.0
        assert c["o2"] == 0.0
        assert c["soil"] == 0.0
        assert c["biodiversity"] == 0.0

    def test_terraform_increases_pressure(self):
        c = compute_terraforming_contribution({"terraform": 5}, [])
        assert c["pressure"] > 0.0
        assert c["o2"] > 0.0

    def test_farm_increases_soil_and_biodiversity(self):
        c = compute_terraforming_contribution({"farm": 3}, [])
        assert c["soil"] > 0.0
        assert c["biodiversity"] > 0.0
        assert c["pressure"] == 0.0  # farming doesn't affect pressure

    def test_research_boosts_both(self):
        c = compute_terraforming_contribution({"research": 2}, [])
        assert c["pressure"] > 0.0
        assert c["soil"] > 0.0

    def test_monotonic_in_colonist_count(self):
        c1 = compute_terraforming_contribution({"terraform": 1}, [])
        c5 = compute_terraforming_contribution({"terraform": 5}, [])
        assert c5["pressure"] > c1["pressure"]
        assert c5["o2"] > c1["o2"]

    def test_infra_bonus_atmospheric_processor(self):
        c_base = compute_terraforming_contribution({"terraform": 3}, [])
        c_infra = compute_terraforming_contribution(
            {"terraform": 3}, ["atmospheric_processor"])
        assert c_infra["pressure"] > c_base["pressure"]
        assert c_infra["o2"] > c_base["o2"]

    def test_infra_bonus_greenhouse(self):
        c_base = compute_terraforming_contribution({"farm": 3}, [])
        c_infra = compute_terraforming_contribution(
            {"farm": 3}, ["advanced_greenhouse"])
        assert c_infra["biodiversity"] > c_base["biodiversity"]


# ── Event effects ────────────────────────────────────────────────────

class TestEventEcologyEffects:

    def test_empty_events(self):
        e = compute_event_ecology_effects([])
        assert e["soil"] == 0.0
        assert e["outdoor_coverage"] == 0.0
        assert e["biodiversity"] == 0.0

    def test_dust_storm_reduces_coverage(self):
        e = compute_event_ecology_effects([{"name": "dust_storm", "severity": 0.5}])
        assert e["outdoor_coverage"] < 0.0

    def test_resource_strike_improves_soil(self):
        e = compute_event_ecology_effects([{"name": "resource_strike", "severity": 0.1}])
        assert e["soil"] > 0.0

    def test_breakthrough_improves_biodiversity(self):
        e = compute_event_ecology_effects([{"name": "breakthrough", "severity": 0.1}])
        assert e["biodiversity"] > 0.0


# ── Temperature model ────────────────────────────────────────────────

class TestTemperature:

    def test_baseline_pressure_baseline_temp(self):
        t = compute_temperature(MARS_PRESSURE_ATM)
        assert abs(t - MARS_TEMP_C) < 0.1

    def test_higher_pressure_warmer(self):
        t_lo = compute_temperature(MARS_PRESSURE_ATM)
        t_hi = compute_temperature(0.1)
        assert t_hi > t_lo

    def test_temperature_capped(self):
        t = compute_temperature(10.0)
        assert t <= 30.0

    def test_temperature_floored(self):
        t = compute_temperature(0.0)
        assert t >= MARS_TEMP_C


# ── Dust storm modifier ─────────────────────────────────────────────

class TestDustStormModifier:

    def test_cold_baseline(self):
        m = compute_dust_storm_modifier(-65.0)
        assert m == 1.0

    def test_warming_increases(self):
        m = compute_dust_storm_modifier(-35.0)
        assert m > 1.0

    def test_hot_decreases(self):
        m = compute_dust_storm_modifier(10.0)
        assert m < 1.3

    def test_never_negative(self):
        for temp in range(-80, 40, 5):
            m = compute_dust_storm_modifier(float(temp))
            assert m >= 0.5


# ── Milestones ───────────────────────────────────────────────────────

class TestMilestones:

    def test_no_milestones_at_start(self):
        eco = _fresh()
        fired = check_milestones(eco, 1)
        assert fired == []

    def test_lichen_milestone(self):
        eco = _fresh()
        eco.atmosphere_pressure = LICHEN_PRESSURE_THRESHOLD + 0.001
        eco.temperature_avg = LICHEN_TEMP_THRESHOLD + 1.0
        fired = check_milestones(eco, 20)
        assert "lichen_viable" in fired
        assert eco.milestones.lichen_viable is True

    def test_lichen_needs_both_conditions(self):
        eco = _fresh()
        eco.atmosphere_pressure = LICHEN_PRESSURE_THRESHOLD + 0.001
        eco.temperature_avg = LICHEN_TEMP_THRESHOLD - 5.0  # too cold
        fired = check_milestones(eco, 20)
        assert "lichen_viable" not in fired

    def test_milestone_fires_once(self):
        eco = _fresh()
        eco.atmosphere_pressure = LICHEN_PRESSURE_THRESHOLD + 0.001
        eco.temperature_avg = LICHEN_TEMP_THRESHOLD + 1.0
        fired1 = check_milestones(eco, 20)
        fired2 = check_milestones(eco, 21)
        assert "lichen_viable" in fired1
        assert "lichen_viable" not in fired2

    def test_water_cycle_milestone(self):
        eco = _fresh()
        eco.temperature_avg = WATER_CYCLE_TEMP_THRESHOLD + 1.0
        fired = check_milestones(eco, 30)
        assert "water_cycle_active" in fired

    def test_mask_breathing_uses_partial_pressure(self):
        eco = _fresh()
        # Low fraction but high pressure → still triggers
        eco.atmosphere_pressure = 0.5
        eco.o2_fraction = 0.02  # partial = 0.01 > threshold
        fired = check_milestones(eco, 50)
        assert "mask_breathing" in fired

    def test_mask_breathing_high_fraction_low_pressure_no_trigger(self):
        eco = _fresh()
        eco.atmosphere_pressure = 0.01
        eco.o2_fraction = 0.2  # partial = 0.002 < threshold
        fired = check_milestones(eco, 50)
        assert "mask_breathing" not in fired

    def test_outdoor_farming_needs_both(self):
        eco = _fresh()
        eco.soil_fertility = OUTDOOR_FARMING_SOIL_THRESHOLD + 0.1
        eco.temperature_avg = OUTDOOR_FARMING_TEMP_THRESHOLD - 20.0  # too cold
        fired = check_milestones(eco, 80)
        assert "outdoor_farming" not in fired

    def test_milestone_events_logged(self):
        eco = _fresh()
        eco.atmosphere_pressure = LICHEN_PRESSURE_THRESHOLD + 0.001
        eco.temperature_avg = LICHEN_TEMP_THRESHOLD + 1.0
        check_milestones(eco, 25)
        assert len(eco.milestone_events) == 1
        assert eco.milestone_events[0]["year"] == 25


# ── Ecology modifiers ────────────────────────────────────────────────

class TestEcologyModifiers:

    def test_zero_at_start(self):
        eco = _fresh()
        mods = compute_ecology_modifiers(eco)
        assert len(mods) == 0  # all below threshold

    def test_soil_gives_food_bonus(self):
        eco = _fresh()
        eco.soil_fertility = 0.5
        mods = compute_ecology_modifiers(eco)
        assert "food_ecology_bonus" in mods
        assert mods["food_ecology_bonus"] > 0.0

    def test_water_cycle_gives_water_bonus(self):
        eco = _fresh()
        eco.water_cycle_strength = 0.5
        mods = compute_ecology_modifiers(eco)
        assert "water_ecology_bonus" in mods
        assert mods["water_ecology_bonus"] > 0.0

    def test_biodiversity_gives_air_bonus(self):
        eco = _fresh()
        eco.biodiversity_index = 0.5
        mods = compute_ecology_modifiers(eco)
        assert "air_ecology_bonus" in mods
        assert mods["air_ecology_bonus"] > 0.0

    def test_modifiers_capped(self):
        eco = _fresh()
        eco.soil_fertility = 1.0
        eco.water_cycle_strength = 1.0
        eco.biodiversity_index = 1.0
        eco.outdoor_coverage = 1.0
        eco.milestones.outdoor_farming = True
        mods = compute_ecology_modifiers(eco)
        for v in mods.values():
            assert v <= 0.20 + 0.001


# ── tick_ecology integration ─────────────────────────────────────────

class TestTickEcology:

    def test_one_tick_basic(self):
        eco = _fresh()
        result = tick_ecology(eco, {"terraform": 3, "farm": 2}, [], [], 1)
        _assert_bounds(eco)
        assert result.pressure_delta > 0
        assert result.soil_delta > 0

    def test_no_actions_minimal_change(self):
        eco = _fresh()
        result = tick_ecology(eco, {}, [], [], 1)
        _assert_bounds(eco)
        assert result.pressure_delta == 0.0
        assert result.soil_delta == 0.0

    def test_ten_year_progression(self):
        eco = _fresh()
        for year in range(1, 11):
            tick_ecology(eco, {"terraform": 5, "farm": 3, "research": 2}, [], [], year)
            _assert_bounds(eco)
        assert eco.atmosphere_pressure > MARS_PRESSURE_ATM
        assert eco.soil_fertility > 0.0
        assert eco.biodiversity_index > 0.0

    def test_hundred_year_progression(self):
        """100 years of active terraforming should show progress but not Earth-like."""
        eco = _fresh()
        for year in range(1, 101):
            tick_ecology(
                eco,
                {"terraform": 5, "farm": 3, "research": 2},
                [], [], year,
            )
            _assert_bounds(eco)
        # Pressure should increase but stay far from Earth
        assert eco.atmosphere_pressure > MARS_PRESSURE_ATM
        assert eco.atmosphere_pressure < 0.1  # nowhere near 1 atm
        # Soil should be substantially improved
        assert eco.soil_fertility > 0.3
        # Temperature should warm but not reach Earth-like
        assert eco.temperature_avg > MARS_TEMP_C
        assert eco.temperature_avg < 0.0  # still subzero

    def test_events_affect_ecology(self):
        eco = _fresh()
        eco.milestones.lichen_viable = True
        eco.outdoor_coverage = 0.5
        events = [{"name": "dust_storm", "severity": 0.8}]
        result = tick_ecology(eco, {}, events, [], 10)
        assert result.outdoor_coverage_delta < 0

    def test_milestones_trigger_during_tick(self):
        eco = _fresh()
        # Lichen viable needs: pressure >= 0.01 AND temp >= -40
        # At pressure 0.6, temp ≈ -38.2 (above -40), and pressure >> 0.01
        eco.atmosphere_pressure = 0.6
        eco.temperature_avg = compute_temperature(0.6)
        assert eco.temperature_avg >= LICHEN_TEMP_THRESHOLD
        assert eco.atmosphere_pressure >= LICHEN_PRESSURE_THRESHOLD
        result = tick_ecology(eco, {"terraform": 1}, [], [], 25)
        assert "lichen_viable" in result.new_milestones

    def test_water_cycle_only_active_after_milestone(self):
        eco = _fresh()
        result = tick_ecology(eco, {"terraform": 3}, [], [], 1)
        assert result.water_cycle_delta == 0.0
        # Activate milestone
        eco.milestones.water_cycle_active = True
        result2 = tick_ecology(eco, {"terraform": 3}, [], [], 2)
        assert result2.water_cycle_delta > 0.0

    def test_modifiers_returned(self):
        eco = _fresh()
        eco.soil_fertility = 0.5
        result = tick_ecology(eco, {}, [], [], 1)
        assert isinstance(result.modifiers, dict)

    def test_serialization_roundtrip(self):
        eco = _fresh()
        tick_ecology(eco, {"terraform": 3, "farm": 2}, [], [], 5)
        d = eco.to_dict()
        eco2 = EcologyState.from_dict(d)
        _assert_bounds(eco2)
        assert abs(eco.atmosphere_pressure - eco2.atmosphere_pressure) < 1e-6

    def test_infra_techs_amplify(self):
        eco1 = _fresh()
        eco2 = _fresh()
        tick_ecology(eco1, {"terraform": 5}, [], [], 1)
        tick_ecology(eco2, {"terraform": 5}, [],
                     ["atmospheric_processor"], 1)
        assert eco2.atmosphere_pressure > eco1.atmosphere_pressure


# ── Property-based invariants ────────────────────────────────────────

class TestPropertyInvariants:

    @pytest.mark.parametrize("seed", range(20))
    def test_bounds_never_violated(self, seed):
        """Fuzz with different action combos: bounds always hold."""
        import random
        rng = random.Random(seed)
        eco = _fresh()
        action_pool = ["terraform", "farm", "research", "code", "rest",
                        "mediate", "pray", "sabotage"]
        event_pool = ["dust_storm", "resource_strike", "breakthrough",
                       "ice_volcano", "cave_discovery"]
        for year in range(1, 51):
            actions: dict[str, int] = {}
            for _ in range(rng.randint(0, 10)):
                act = rng.choice(action_pool)
                actions[act] = actions.get(act, 0) + 1
            events = []
            if rng.random() < 0.3:
                events.append({"name": rng.choice(event_pool),
                                "severity": rng.random()})
            tick_ecology(eco, actions, events, [], year)
            _assert_bounds(eco)

    @pytest.mark.parametrize("n_terraform", [0, 1, 5, 10])
    def test_pressure_monotonically_nondecreasing(self, n_terraform):
        """Terraforming should never decrease pressure."""
        eco = _fresh()
        prev = eco.atmosphere_pressure
        for year in range(1, 21):
            tick_ecology(eco, {"terraform": n_terraform}, [], [], year)
            assert eco.atmosphere_pressure >= prev - 1e-12
            prev = eco.atmosphere_pressure

    def test_gas_fraction_invariant_across_100_years(self):
        eco = _fresh()
        for year in range(1, 101):
            tick_ecology(eco, {"terraform": 5, "farm": 3}, [], [], year)
            _assert_fractions_sum_to_one(eco)

    def test_no_ecology_change_without_relevant_actions(self):
        """Actions like 'pray', 'mediate', 'sabotage' shouldn't touch ecology."""
        eco1 = _fresh()
        eco2 = _fresh()
        tick_ecology(eco1, {}, [], [], 1)
        tick_ecology(eco2, {"pray": 5, "mediate": 3, "sabotage": 2}, [], [], 1)
        assert abs(eco1.atmosphere_pressure - eco2.atmosphere_pressure) < 1e-12
        assert abs(eco1.soil_fertility - eco2.soil_fertility) < 1e-12

    def test_saturation_stays_realistic(self):
        """Even with max effort for 100 years, Mars doesn't become Earth."""
        eco = _fresh()
        for year in range(1, 101):
            tick_ecology(eco, {"terraform": 20, "farm": 20, "research": 20},
                         [], ["atmospheric_processor", "advanced_greenhouse"],
                         year)
        assert eco.atmosphere_pressure < 0.5   # not Earth-like
        assert eco.temperature_avg < 20.0      # not Earth-like
        assert eco.o2_fraction < 0.21          # not breathable
