"""Tests for the Mars-100 ecology organ (engine v9.0).

Property-based tests: physical bounds, conservation laws, milestone
idempotence, modifier caps, determinism, no runaway feedback.
"""
from __future__ import annotations

import random
import pytest

from src.mars100.ecology import (
    EcologyState, EcologyTickResult,
    tick_ecology, compute_ecology_modifiers, compute_ecology_psych_effects,
    _check_milestones,
    BIOME_MAX, O2_MIN, O2_MAX, TEMP_MIN, TEMP_MAX,
    MAX_FOOD_BONUS, MAX_WATER_PENALTY, MAX_EVENT_DAMAGE_REDUCTION,
    MAX_DEATH_RATE_REDUCTION,
    MILESTONE_ALGAE_BLOOM, MILESTONE_STABLE_BIOME,
    MILESTONE_WATER_INDEPENDENCE, MILESTONE_BREATHABLE_HAB,
    ICE_RECYCLING_FLOOR,
)


# -- helpers -----------------------------------------------------------------

def _make_state(**overrides) -> EcologyState:
    """Create an EcologyState with optional overrides."""
    return EcologyState(**overrides)


def _tick(state: EcologyState, actions: dict[str, int] | None = None,
          population: int = 10, power: float = 0.7,
          infra: list[str] | None = None, ship: bool = False,
          seed: int = 42) -> EcologyTickResult:
    """Convenience tick wrapper."""
    return tick_ecology(
        state=state,
        action_counts=actions or {},
        population=population,
        resource_power=power,
        infra_completed=infra or [],
        earth_ship_this_year=ship,
        rng=random.Random(seed),
    )


# -- physical bounds ---------------------------------------------------------

class TestBounds:
    """All state variables remain within physical bounds after any tick."""

    @pytest.mark.parametrize("seed", range(20))
    def test_bounds_random_actions(self, seed):
        rng = random.Random(seed)
        state = _make_state()
        for _ in range(50):
            actions = {
                "terraform": rng.randint(0, 5),
                "farm": rng.randint(0, 5),
                "research": rng.randint(0, 3),
            }
            _tick(state, actions, population=rng.randint(1, 30),
                  power=rng.random(), seed=seed)
        assert 0.0 <= state.soil_toxicity <= 1.0
        assert 0.0 <= state.biome_health <= BIOME_MAX
        assert 0.0 <= state.water_ice_reserve <= 1.0
        assert 0.0 <= state.water_recycling <= 1.0
        assert O2_MIN <= state.hab_o2_quality <= O2_MAX
        assert TEMP_MIN <= state.temp_stability <= TEMP_MAX

    def test_extreme_actions_bounded(self):
        """Even with extreme action counts, state stays bounded."""
        state = _make_state()
        _tick(state, {"terraform": 100, "farm": 100, "research": 100},
              population=100, power=1.0)
        assert 0.0 <= state.soil_toxicity <= 1.0
        assert 0.0 <= state.biome_health <= BIOME_MAX
        assert 0.0 <= state.water_ice_reserve <= 1.0

    def test_zero_colonists_bounded(self):
        """With zero colonists, ecology still behaves."""
        state = _make_state()
        result = _tick(state, {}, population=0, power=0.0)
        assert 0.0 <= state.soil_toxicity <= 1.0
        assert isinstance(result.terraforming_score, float)


# -- soil remediation --------------------------------------------------------

class TestSoilRemediation:
    def test_terraforming_reduces_toxicity(self):
        state = _make_state(soil_toxicity=0.8)
        _tick(state, {"terraform": 5})
        assert state.soil_toxicity < 0.8

    def test_farming_reduces_toxicity(self):
        state = _make_state(soil_toxicity=0.8)
        _tick(state, {"farm": 5})
        assert state.soil_toxicity < 0.8

    def test_natural_decay(self):
        """Even with no actions, soil slowly improves."""
        state = _make_state(soil_toxicity=0.5)
        _tick(state, {})
        assert state.soil_toxicity < 0.5

    def test_toxicity_floor_is_zero(self):
        state = _make_state(soil_toxicity=0.01)
        for _ in range(20):
            _tick(state, {"terraform": 10})
        assert state.soil_toxicity >= 0.0

    def test_scrubber_tech_accelerates(self):
        a = _make_state(soil_toxicity=0.8)
        b = _make_state(soil_toxicity=0.8)
        _tick(a, {"terraform": 3}, seed=99)
        _tick(b, {"terraform": 3}, infra=["perchlorate_scrubber"], seed=99)
        assert b.soil_toxicity < a.soil_toxicity


# -- biome health ------------------------------------------------------------

class TestBiomeHealth:
    def test_farming_grows_biome(self):
        state = _make_state(biome_health=0.1)
        _tick(state, {"farm": 3})
        assert state.biome_health > 0.1

    def test_neglect_decays_biome(self):
        state = _make_state(biome_health=0.5)
        _tick(state, {"farm": 0})
        assert state.biome_health < 0.5

    def test_greenhouse_tech_boost(self):
        a = _make_state(biome_health=0.2)
        b = _make_state(biome_health=0.2)
        _tick(a, {"farm": 3}, seed=77)
        _tick(b, {"farm": 3}, infra=["greenhouse_dome"], seed=77)
        assert b.biome_health > a.biome_health

    def test_low_toxicity_helps_biome(self):
        """Remediated soil grows biomes faster."""
        a = _make_state(biome_health=0.2, soil_toxicity=0.8)
        b = _make_state(biome_health=0.2, soil_toxicity=0.1)
        _tick(a, {"farm": 3}, seed=55)
        _tick(b, {"farm": 3}, seed=55)
        assert b.biome_health > a.biome_health


# -- water cycle -------------------------------------------------------------

class TestWaterCycle:
    def test_terraforming_depletes_ice(self):
        state = _make_state(water_ice_reserve=1.0)
        _tick(state, {"terraform": 5})
        assert state.water_ice_reserve < 1.0

    def test_recycler_reduces_depletion(self):
        a = _make_state(water_ice_reserve=0.8)
        b = _make_state(water_ice_reserve=0.8)
        _tick(a, {"terraform": 3}, seed=33)
        _tick(b, {"terraform": 3}, infra=["water_recycler"], seed=33)
        assert b.water_ice_reserve > a.water_ice_reserve

    def test_earth_ship_replenishes(self):
        a = _make_state(water_ice_reserve=0.5)
        b = _make_state(water_ice_reserve=0.5)
        _tick(a, {"terraform": 2}, seed=44)
        _tick(b, {"terraform": 2}, ship=True, seed=44)
        assert b.water_ice_reserve > a.water_ice_reserve

    def test_recycling_floor_prevents_total_depletion(self):
        """High recycling + low ice should hit the floor."""
        state = _make_state(water_ice_reserve=0.12, water_recycling=0.7)
        for _ in range(30):
            _tick(state, {"terraform": 5}, infra=["water_recycler"])
        assert state.water_ice_reserve >= 0.0

    def test_research_improves_recycling(self):
        state = _make_state(water_recycling=0.2)
        _tick(state, {"research": 5})
        assert state.water_recycling > 0.2


# -- hab O2 quality ----------------------------------------------------------

class TestHabO2:
    def test_farming_improves_o2(self):
        state = _make_state(hab_o2_quality=0.5, biome_health=0.3)
        _tick(state, {"farm": 5}, population=2)
        assert state.hab_o2_quality > 0.5

    def test_population_drains_o2(self):
        state = _make_state(hab_o2_quality=0.6)
        _tick(state, {}, population=30)
        assert state.hab_o2_quality < 0.6

    def test_o2_has_minimum(self):
        state = _make_state(hab_o2_quality=0.15)
        for _ in range(20):
            _tick(state, {}, population=50)
        assert state.hab_o2_quality >= O2_MIN


# -- temperature stability ---------------------------------------------------

class TestTempStability:
    def test_power_helps_temp(self):
        state = _make_state(temp_stability=0.5)
        _tick(state, {}, population=2, power=0.9)
        assert state.temp_stability > 0.5

    def test_population_strains_temp(self):
        state = _make_state(temp_stability=0.7)
        _tick(state, {}, population=30, power=0.3)
        assert state.temp_stability < 0.7

    def test_thermal_regulator_helps(self):
        a = _make_state(temp_stability=0.5)
        b = _make_state(temp_stability=0.5)
        _tick(a, {}, population=10, power=0.5, seed=11)
        _tick(b, {}, population=10, power=0.5,
              infra=["thermal_regulator"], seed=11)
        assert b.temp_stability > a.temp_stability


# -- milestones (edge-triggered, fire once) ----------------------------------

class TestMilestones:
    def test_algae_bloom_fires_once(self):
        state = _make_state(soil_toxicity=0.35)
        r1 = EcologyTickResult()
        _check_milestones(state, r1)
        assert "first_algae_bloom" in r1.new_milestones
        assert "first_algae_bloom" in state.milestones

        r2 = EcologyTickResult()
        _check_milestones(state, r2)
        assert r2.new_milestones == []

    def test_stable_biome_milestone(self):
        state = _make_state(biome_health=0.65)
        r = EcologyTickResult()
        _check_milestones(state, r)
        assert "stable_biome" in r.new_milestones

    def test_water_independence_milestone(self):
        state = _make_state(water_recycling=0.85)
        r = EcologyTickResult()
        _check_milestones(state, r)
        assert "water_independence" in r.new_milestones

    def test_breathable_hab_milestone(self):
        state = _make_state(hab_o2_quality=0.90)
        r = EcologyTickResult()
        _check_milestones(state, r)
        assert "breathable_hab" in r.new_milestones

    def test_no_milestone_below_threshold(self):
        state = _make_state(soil_toxicity=0.9, biome_health=0.1,
                            water_recycling=0.1, hab_o2_quality=0.3)
        r = EcologyTickResult()
        _check_milestones(state, r)
        assert r.new_milestones == []

    def test_milestones_accumulate(self):
        """Multiple milestones can fire in the same tick."""
        state = _make_state(
            soil_toxicity=0.2, biome_health=0.7,
            water_recycling=0.9, hab_o2_quality=0.9)
        r = EcologyTickResult()
        _check_milestones(state, r)
        assert len(r.new_milestones) == 4


# -- modifiers ---------------------------------------------------------------

class TestModifiers:
    def test_food_bonus_capped(self):
        state = _make_state(biome_health=1.0)
        mods = compute_ecology_modifiers(state)
        assert mods["food_production_mult"] <= 1.0 + MAX_FOOD_BONUS

    def test_water_penalty_capped(self):
        state = _make_state(water_ice_reserve=0.0, water_recycling=0.0)
        mods = compute_ecology_modifiers(state)
        assert mods["water_production_mult"] >= 1.0 - MAX_WATER_PENALTY

    def test_event_damage_reduction_capped(self):
        state = _make_state(temp_stability=1.0, hab_o2_quality=1.0)
        mods = compute_ecology_modifiers(state)
        assert mods["event_damage_mult"] >= 1.0 - MAX_EVENT_DAMAGE_REDUCTION

    def test_death_rate_reduction_capped(self):
        state = _make_state(
            soil_toxicity=0.0, biome_health=1.0,
            water_recycling=1.0, hab_o2_quality=1.0,
            temp_stability=1.0)
        mods = compute_ecology_modifiers(state)
        assert mods["death_rate_ecology_mult"] >= 1.0 - MAX_DEATH_RATE_REDUCTION

    def test_neutral_state_modifiers_near_one(self):
        """Default state shouldn't wildly distort resource production."""
        state = _make_state()
        mods = compute_ecology_modifiers(state)
        for key, val in mods.items():
            assert 0.5 < val < 1.5, f"{key}={val}"

    def test_modifiers_all_positive(self):
        """All multipliers must be positive (no negative production)."""
        state = _make_state(water_ice_reserve=0.0, water_recycling=0.0)
        mods = compute_ecology_modifiers(state)
        for key, val in mods.items():
            assert val > 0, f"{key}={val}"


# -- psych effects -----------------------------------------------------------

class TestPsychEffects:
    def test_milestone_boosts_purpose(self):
        state = _make_state()
        effects = compute_ecology_psych_effects(state, ["first_algae_bloom"])
        assert effects["purpose_boost"] > 0

    def test_no_milestones_no_purpose_boost(self):
        state = _make_state()
        effects = compute_ecology_psych_effects(state, [])
        assert effects["purpose_boost"] == 0.0

    def test_low_ice_increases_stress(self):
        state = _make_state(water_ice_reserve=0.1)
        effects = compute_ecology_psych_effects(state, [])
        assert effects["stress_from_ice"] > 0

    def test_normal_ice_no_stress(self):
        state = _make_state(water_ice_reserve=0.5)
        effects = compute_ecology_psych_effects(state, [])
        assert effects["stress_from_ice"] == 0.0

    def test_biome_reduces_loneliness(self):
        state = _make_state(biome_health=0.6)
        effects = compute_ecology_psych_effects(state, [])
        assert effects["loneliness_reduction"] > 0

    def test_low_biome_no_loneliness_effect(self):
        state = _make_state(biome_health=0.2)
        effects = compute_ecology_psych_effects(state, [])
        assert effects["loneliness_reduction"] == 0.0


# -- serialization -----------------------------------------------------------

class TestSerialization:
    def test_round_trip(self):
        state = _make_state(soil_toxicity=0.33, biome_health=0.66,
                            milestones=["first_algae_bloom"])
        d = state.to_dict()
        restored = EcologyState.from_dict(d)
        assert abs(restored.soil_toxicity - 0.33) < 0.001
        assert abs(restored.biome_health - 0.66) < 0.001
        assert restored.milestones == ["first_algae_bloom"]

    def test_tick_result_serializable(self):
        state = _make_state()
        result = _tick(state, {"farm": 3})
        d = result.to_dict()
        assert isinstance(d["soil_delta"], float)
        assert isinstance(d["new_milestones"], list)
        assert isinstance(d["terraforming_score"], float)


# -- determinism -------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_result(self):
        for seed in [42, 99, 7]:
            a = _make_state()
            b = _make_state()
            ra = _tick(a, {"terraform": 3, "farm": 2}, seed=seed)
            rb = _tick(b, {"terraform": 3, "farm": 2}, seed=seed)
            assert a.to_dict() == b.to_dict()
            assert ra.to_dict() == rb.to_dict()

    def test_different_seed_different_result(self):
        a = _make_state()
        b = _make_state()
        _tick(a, {"terraform": 3}, seed=1)
        _tick(b, {"terraform": 3}, seed=2)
        assert a.to_dict() != b.to_dict()


# -- smoke test (multi-year run) ---------------------------------------------

class TestSmoke:
    def test_50_year_ecology_run(self):
        """Run ecology for 50 years with mixed actions — nothing crashes."""
        state = _make_state()
        rng = random.Random(42)
        for year in range(50):
            actions = {
                "terraform": rng.randint(0, 4),
                "farm": rng.randint(0, 4),
                "research": rng.randint(0, 2),
            }
            result = tick_ecology(
                state, actions, population=rng.randint(5, 20),
                resource_power=rng.random(),
                infra_completed=["greenhouse_dome", "water_recycler"],
                earth_ship_this_year=(year % 10 == 0),
                rng=random.Random(42 + year),
            )
            assert isinstance(result, EcologyTickResult)
            assert isinstance(result.terraforming_score, float)
            assert 0.0 <= result.terraforming_score <= 1.0

    def test_100_year_no_runaway(self):
        """100-year run: no resource modifier exceeds safe bounds."""
        state = _make_state()
        rng = random.Random(99)
        for year in range(100):
            actions = {"terraform": 3, "farm": 3, "research": 2}
            tick_ecology(state, actions, population=15,
                         resource_power=0.7,
                         infra_completed=["greenhouse_dome", "water_recycler",
                                          "perchlorate_scrubber"],
                         earth_ship_this_year=False,
                         rng=random.Random(99 + year))
            mods = compute_ecology_modifiers(state)
            for key, val in mods.items():
                assert 0.3 < val < 2.0, f"Year {year}: {key}={val}"
