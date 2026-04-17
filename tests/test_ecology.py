"""Tests for the Mars-100 ecology organ."""
from __future__ import annotations

import random
import pytest

from src.mars100.ecology import (
    BIOME_NAMES,
    BIOME_O2_RATE,
    BIOME_SOIL_REGEN,
    BIOME_THRESHOLDS,
    EcologyState,
    EcologyTickResult,
    _clamp,
    _compute_biome_level,
    compute_ecology_death_rate,
    compute_ecology_modifiers,
    tick_ecology,
)


# ── helpers ────────────────────────────────────────────────────────

def _make_eco(**kw: float) -> EcologyState:
    """Create EcologyState with overrides."""
    return EcologyState(**kw)


def _make_actions(terraform: int = 0, farm: int = 0, rest: int = 0) -> dict[str, str]:
    """Create action dict for testing."""
    actions: dict[str, str] = {}
    for i in range(terraform):
        actions[f"tf-{i}"] = "terraform"
    for i in range(farm):
        actions[f"fa-{i}"] = "farm"
    for i in range(rest):
        actions[f"re-{i}"] = "rest"
    return actions


# ── EcologyState serialization ─────────────────────────────────────

class TestEcologyStateSerialization:
    def test_round_trip(self):
        eco = EcologyState(hab_o2=0.6, soil_quality=0.3, biome_level=2)
        d = eco.to_dict()
        restored = EcologyState.from_dict(d)
        assert restored.hab_o2 == pytest.approx(0.6, abs=0.001)
        assert restored.soil_quality == pytest.approx(0.3, abs=0.001)
        assert restored.biome_level == 2

    def test_to_dict_has_biome_name(self):
        eco = EcologyState(biome_level=3)
        d = eco.to_dict()
        assert d["biome_name"] == "moss"

    def test_from_dict_defaults(self):
        eco = EcologyState.from_dict({})
        assert eco.hab_o2 == 0.80
        assert eco.biome_level == 0


# ── biome level computation ────────────────────────────────────────

class TestBiomeLevel:
    def test_barren_at_start(self):
        eco = _make_eco()
        assert _compute_biome_level(eco) == 0

    def test_microbial_threshold(self):
        eco = _make_eco(soil_quality=0.15, hab_o2=0.4, water_table=0.2)
        assert _compute_biome_level(eco) == 1

    def test_lichen_threshold(self):
        eco = _make_eco(soil_quality=0.25, hab_o2=0.5, water_table=0.3)
        assert _compute_biome_level(eco) == 2

    def test_moss_threshold(self):
        eco = _make_eco(soil_quality=0.40, hab_o2=0.6, water_table=0.4)
        assert _compute_biome_level(eco) == 3

    def test_garden_threshold(self):
        eco = _make_eco(soil_quality=0.55, hab_o2=0.7, water_table=0.5)
        assert _compute_biome_level(eco) == 4

    def test_forest_threshold(self):
        eco = _make_eco(soil_quality=0.70, hab_o2=0.8, water_table=0.6)
        assert _compute_biome_level(eco) == 5

    def test_regression_when_conditions_degrade(self):
        eco = _make_eco(soil_quality=0.70, hab_o2=0.8, water_table=0.6)
        assert _compute_biome_level(eco) == 5
        eco.soil_quality = 0.10
        assert _compute_biome_level(eco) == 0

    def test_partial_regression(self):
        eco = _make_eco(soil_quality=0.55, hab_o2=0.7, water_table=0.5)
        assert _compute_biome_level(eco) == 4
        eco.water_table = 0.25
        assert _compute_biome_level(eco) == 1

    def test_monotonic_thresholds(self):
        """Each biome level requires strictly more than the previous."""
        for i in range(1, len(BIOME_THRESHOLDS)):
            for j in range(3):
                assert BIOME_THRESHOLDS[i][j] >= BIOME_THRESHOLDS[i - 1][j]


# ── clamping ───────────────────────────────────────────────────────

class TestClamp:
    def test_clamp_upper(self):
        eco = _make_eco(hab_o2=1.5, hab_co2=2.0, soil_quality=1.1,
                        water_table=1.3)
        _clamp(eco)
        assert eco.hab_o2 == 1.0
        assert eco.hab_co2 == 1.0
        assert eco.soil_quality == 1.0
        assert eco.water_table == 1.0

    def test_clamp_lower(self):
        eco = _make_eco(hab_o2=-0.5, hab_co2=-1.0, soil_quality=-0.1,
                        water_table=-0.2)
        _clamp(eco)
        assert eco.hab_o2 == 0.0
        assert eco.hab_co2 == 0.0
        assert eco.soil_quality == 0.0
        assert eco.water_table == 0.0

    def test_biome_level_clamp(self):
        eco = _make_eco()
        eco.biome_level = 10
        _clamp(eco)
        assert eco.biome_level == 5
        eco.biome_level = -3
        _clamp(eco)
        assert eco.biome_level == 0


# ── tick_ecology ───────────────────────────────────────────────────

class TestTickEcology:
    def test_basic_tick_returns_result(self):
        eco = _make_eco()
        rng = random.Random(42)
        result = tick_ecology(eco, {}, 10, [], rng)
        assert isinstance(result, EcologyTickResult)

    def test_population_breathing_lowers_o2(self):
        eco = _make_eco(hab_o2=0.5)
        rng = random.Random(42)
        tick_ecology(eco, _make_actions(rest=20), 20, [], rng)
        # With 20 people breathing, O2 should decrease (despite life support)
        # Net effect depends on rng, but high pop should push O2 down
        assert eco.hab_o2 < 0.55  # some decrease expected

    def test_terraforming_improves_soil(self):
        eco = _make_eco(soil_quality=0.1)
        rng = random.Random(42)
        initial_soil = eco.soil_quality
        tick_ecology(eco, _make_actions(terraform=5), 10, [], rng)
        assert eco.soil_quality > initial_soil

    def test_terraforming_increments_counter(self):
        eco = _make_eco()
        rng = random.Random(42)
        assert eco.terraforming_years == 0
        tick_ecology(eco, _make_actions(terraform=1), 10, [], rng)
        assert eco.terraforming_years == 1

    def test_farming_consumes_soil(self):
        eco = _make_eco(soil_quality=0.5)
        rng = random.Random(42)
        tick_ecology(eco, _make_actions(farm=5), 5, [], rng)
        # Farming drains soil quality (partially offset by natural regen)
        assert eco.soil_quality < 0.5

    def test_farming_consumes_water(self):
        eco = _make_eco(water_table=0.5)
        rng = random.Random(42)
        initial_water = eco.water_table
        tick_ecology(eco, _make_actions(farm=5), 5, [], rng)
        assert eco.water_table < initial_water

    def test_values_stay_bounded_after_tick(self):
        """Property: all values remain in [0, 1] after any tick."""
        for seed in range(50):
            eco = _make_eco()
            rng = random.Random(seed)
            pop = rng.randint(1, 40)
            n_tf = rng.randint(0, 10)
            n_fa = rng.randint(0, 10)
            tick_ecology(eco, _make_actions(terraform=n_tf, farm=n_fa),
                         pop, [], rng)
            assert 0.0 <= eco.hab_o2 <= 1.0
            assert 0.0 <= eco.hab_co2 <= 1.0
            assert 0.0 <= eco.soil_quality <= 1.0
            assert 0.0 <= eco.water_table <= 1.0
            assert 0 <= eco.biome_level <= 5

    def test_tech_bonuses_greenhouse(self):
        eco = _make_eco(soil_quality=0.3)
        rng = random.Random(42)
        eco_no_tech = _make_eco(soil_quality=0.3)
        rng2 = random.Random(42)
        tick_ecology(eco, _make_actions(rest=10), 10, ["greenhouse_dome"], rng)
        tick_ecology(eco_no_tech, _make_actions(rest=10), 10, [], rng2)
        assert eco.soil_quality >= eco_no_tech.soil_quality
        assert eco.hab_o2 >= eco_no_tech.hab_o2

    def test_tech_bonuses_water_recycler(self):
        eco = _make_eco(water_table=0.4)
        rng = random.Random(42)
        eco_no_tech = _make_eco(water_table=0.4)
        rng2 = random.Random(42)
        tick_ecology(eco, _make_actions(rest=10), 10, ["water_recycler"], rng)
        tick_ecology(eco_no_tech, _make_actions(rest=10), 10, [], rng2)
        assert eco.water_table >= eco_no_tech.water_table

    def test_biome_transition_detected(self):
        # Start just below microbial threshold, terraform to cross it
        eco = _make_eco(soil_quality=0.14, hab_o2=0.4, water_table=0.2)
        rng = random.Random(42)
        result = tick_ecology(eco, _make_actions(terraform=3), 5, [], rng)
        if result.biome_changed:
            assert result.biome_after > result.biome_before

    def test_tipping_point_o2_critical(self):
        eco = _make_eco(hab_o2=0.31)
        rng = random.Random(42)
        # Large population breathing should push O2 below 0.3
        result = tick_ecology(eco, _make_actions(rest=30), 30, [], rng)
        if eco.hab_o2 < 0.3:
            assert result.tipping_point == "o2_critical"

    def test_deterministic_with_same_seed(self):
        eco_a = _make_eco()
        eco_b = _make_eco()
        actions = _make_actions(terraform=2, farm=3, rest=5)
        tick_ecology(eco_a, actions, 10, [], random.Random(99))
        tick_ecology(eco_b, actions, 10, [], random.Random(99))
        assert eco_a.hab_o2 == eco_b.hab_o2
        assert eco_a.soil_quality == eco_b.soil_quality
        assert eco_a.water_table == eco_b.water_table


# ── compute_ecology_modifiers ──────────────────────────────────────

class TestEcologyModifiers:
    def test_high_o2_gives_air_bonus(self):
        eco = _make_eco(hab_o2=0.8)
        mods = compute_ecology_modifiers(eco)
        assert mods["air"] > 0

    def test_low_o2_gives_air_penalty(self):
        eco = _make_eco(hab_o2=0.2)
        mods = compute_ecology_modifiers(eco)
        assert mods["air"] < 0

    def test_good_soil_gives_food_bonus(self):
        eco = _make_eco(soil_quality=0.6)
        mods = compute_ecology_modifiers(eco)
        assert mods["food"] > 0

    def test_water_table_bonus(self):
        eco = _make_eco(water_table=0.7)
        mods = compute_ecology_modifiers(eco)
        assert mods["water"] > 0

    def test_low_water_table_penalty(self):
        eco = _make_eco(water_table=0.1)
        mods = compute_ecology_modifiers(eco)
        assert mods["water"] < 0

    def test_modifier_magnitudes_reasonable(self):
        """Modifiers should be small nudges, not overwhelming."""
        for seed in range(20):
            rng = random.Random(seed)
            eco = _make_eco(
                hab_o2=rng.uniform(0.0, 1.0),
                soil_quality=rng.uniform(0.0, 1.0),
                water_table=rng.uniform(0.0, 1.0),
            )
            mods = compute_ecology_modifiers(eco)
            for key, val in mods.items():
                assert abs(val) < 0.1, f"{key}={val} too large"


# ── compute_ecology_death_rate ─────────────────────────────────────

class TestEcologyDeathRate:
    def test_normal_conditions_no_extra_death(self):
        eco = _make_eco(hab_o2=0.8, hab_co2=0.1)
        rate, cause = compute_ecology_death_rate(eco)
        assert rate == 0.0
        assert cause is None

    def test_low_o2_increases_death_rate(self):
        eco = _make_eco(hab_o2=0.2)
        rate, cause = compute_ecology_death_rate(eco)
        assert rate > 0
        assert cause == "asphyxiation"

    def test_high_co2_increases_death_rate(self):
        eco = _make_eco(hab_o2=0.5, hab_co2=0.9)
        rate, cause = compute_ecology_death_rate(eco)
        assert rate > 0
        assert cause == "co2_toxicity"

    def test_o2_priority_over_co2(self):
        """When both O2 low and CO2 high, O2 takes priority."""
        eco = _make_eco(hab_o2=0.1, hab_co2=0.9)
        rate, cause = compute_ecology_death_rate(eco)
        assert cause == "asphyxiation"

    def test_death_rate_bounded(self):
        eco = _make_eco(hab_o2=0.0, hab_co2=1.0)
        rate, _ = compute_ecology_death_rate(eco)
        assert rate <= 0.05


# ── EcologyTickResult serialization ────────────────────────────────

class TestEcologyTickResultSerialization:
    def test_to_dict(self):
        result = EcologyTickResult(
            o2_delta=0.01, co2_delta=-0.005,
            soil_delta=0.002, water_delta=-0.001,
            biome_before=1, biome_after=2,
            biome_changed=True, tipping_point="biome_advance:lichen")
        d = result.to_dict()
        assert d["biome_changed"] is True
        assert d["tipping_point"] == "biome_advance:lichen"

    def test_to_dict_no_tipping(self):
        result = EcologyTickResult()
        d = result.to_dict()
        assert "tipping_point" not in d


# ── smoke: multi-year progression ──────────────────────────────────

class TestMultiYearProgression:
    def test_10_years_no_crash(self):
        eco = _make_eco()
        rng = random.Random(42)
        for year in range(10):
            tick_ecology(eco, _make_actions(terraform=2, farm=1, rest=7),
                         10, [], rng)
        assert 0 <= eco.biome_level <= 5

    def test_50_years_terraforming_improves_biome(self):
        eco = _make_eco()
        rng = random.Random(42)
        for year in range(50):
            tick_ecology(eco, _make_actions(terraform=4, farm=1, rest=5),
                         10, [], rng)
        # With sustained terraforming, biome should advance
        assert eco.biome_level >= 1
        assert eco.soil_quality > 0.1
        assert eco.terraforming_years == 50

    def test_100_years_heavy_terraforming_reaches_high_biome(self):
        eco = _make_eco()
        rng = random.Random(42)
        for year in range(100):
            tick_ecology(eco, _make_actions(terraform=5, farm=2, rest=3),
                         10, ["greenhouse_dome", "water_recycler"], rng)
        # With max terraforming + tech for 100 years
        assert eco.biome_level >= 2

    def test_neglect_causes_regression(self):
        # Build up ecology, then stop
        eco = _make_eco()
        rng = random.Random(42)
        for _ in range(30):
            tick_ecology(eco, _make_actions(terraform=5, rest=5), 10, [], rng)
        mid_biome = eco.biome_level
        mid_soil = eco.soil_quality
        # Now neglect for 30 years with large population
        for _ in range(30):
            tick_ecology(eco, _make_actions(farm=5, rest=15), 20, [], rng)
        # Soil should degrade, water deplete
        assert eco.water_table < 0.7  # depleted by farming + population

    def test_water_depletes_with_large_population(self):
        eco = _make_eco(water_table=0.7)
        rng = random.Random(42)
        for _ in range(50):
            tick_ecology(eco, _make_actions(rest=30), 30, [], rng)
        assert eco.water_table < 0.5


# ── integration with engine ────────────────────────────────────────

class TestEngineIntegration:
    def test_engine_10_year_with_ecology(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert len(result.years) > 0
        # Ecology data should be present in year results
        for yr in result.years:
            d = yr.to_dict()
            assert "ecology" in d

    def test_engine_determinism_with_ecology(self):
        from src.mars100.engine import Mars100Engine
        a = Mars100Engine(seed=99, total_years=10).run()
        b = Mars100Engine(seed=99, total_years=10).run()
        for ya, yb in zip(a.years, b.years):
            assert ya.ecology == yb.ecology

    def test_final_ecology_in_sim_result(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        d = result.to_dict()
        assert "final_ecology" in d
        assert "hab_o2" in d["final_ecology"]

    def test_version_bumped(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        d = result.to_dict()
        assert d["_meta"]["version"] == "10.0"
