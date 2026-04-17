"""Tests for Mars-100 health organ (engine v9.0)."""
from __future__ import annotations

import random
import pytest

from src.mars100.health import (
    HealthState, ColonistHealthContext, HealthTickResult,
    compute_aging_delta, compute_radiation_delta, check_new_conditions,
    compute_condition_penalty, health_death_modifier, health_death_cause,
    tick_health, _clamp,
    MARS_YEAR_IN_EARTH_YEARS, AGING_ONSET_EARTH_YEARS,
    AGING_RATE_YOUNG, AGING_RATE_OLD,
    RADIATION_BACKGROUND, SOLAR_FLARE_RADIATION, SHELTER_RADIATION_MULT,
    BONE_LOSS_ONSET_YEAR, BONE_LOSS_PROBABILITY,
    RADIATION_DAMAGE_THRESHOLD, RADIATION_DAMAGE_PROBABILITY,
    FITNESS_CRITICAL, OLD_AGE_MIN_EARTH_YEARS, RADIATION_LETHAL,
)


# -- helpers -----------------------------------------------------------------

def _make_ctx(cid: str = "c-0", birth_year: int = 0,
              solar_flare: bool = False, medicine: float = 0.5,
              med_bay: bool = False, shelter: bool = False,
              years_in_colony: int = 10) -> ColonistHealthContext:
    return ColonistHealthContext(
        colonist_id=cid, birth_year=birth_year,
        has_solar_flare=solar_flare, medicine_level=medicine,
        has_med_bay=med_bay, has_shelter=shelter,
        mars_years_in_colony=years_in_colony,
    )


# -- HealthState tests -------------------------------------------------------

class TestHealthState:
    def test_default_values(self):
        h = HealthState()
        assert h.fitness == 0.85
        assert h.radiation == 0.0
        assert h.chronic_conditions == []
        assert h.initial_age == 30.0

    def test_biological_age(self):
        h = HealthState(initial_age=30.0)
        age = h.biological_age(birth_year=0, current_year=10)
        expected = 30.0 + 10 * MARS_YEAR_IN_EARTH_YEARS
        assert abs(age - expected) < 0.01

    def test_biological_age_at_birth(self):
        h = HealthState(initial_age=0.0)
        assert h.biological_age(birth_year=5, current_year=5) == 0.0

    def test_to_dict_roundtrip(self):
        h = HealthState(fitness=0.7, radiation=0.3,
                        chronic_conditions=["bone_loss"], initial_age=28.0)
        d = h.to_dict()
        h2 = HealthState.from_dict(d)
        assert abs(h2.fitness - 0.7) < 0.001
        assert abs(h2.radiation - 0.3) < 0.001
        assert h2.chronic_conditions == ["bone_loss"]
        assert abs(h2.initial_age - 28.0) < 0.1


# -- aging tests -------------------------------------------------------------

class TestAging:
    def test_young_colonist_ages_slowly(self):
        h = HealthState(initial_age=25.0)
        delta = compute_aging_delta(h, birth_year=0, current_year=5)
        assert delta == AGING_RATE_YOUNG

    def test_middle_age_colonist_ages_faster(self):
        h = HealthState(initial_age=25.0)
        # At year 15: age = 25 + 15*1.88 = 53.2 → past onset (45)
        delta = compute_aging_delta(h, birth_year=0, current_year=15)
        assert delta == AGING_RATE_OLD

    def test_elderly_colonist_accelerates(self):
        h = HealthState(initial_age=30.0)
        # At year 25: age = 30 + 25*1.88 = 77 → past acceleration (70)
        delta = compute_aging_delta(h, birth_year=0, current_year=25)
        assert delta > AGING_RATE_OLD

    def test_aging_never_negative(self):
        """Aging delta is always positive (fitness always decreases from age)."""
        h = HealthState(initial_age=20.0)
        for year in range(0, 100):
            delta = compute_aging_delta(h, birth_year=0, current_year=year)
            assert delta >= 0


# -- radiation tests ---------------------------------------------------------

class TestRadiation:
    def test_background_radiation(self):
        ctx = _make_ctx(solar_flare=False, shelter=False)
        delta = compute_radiation_delta(ctx)
        assert abs(delta - RADIATION_BACKGROUND) < 0.001

    def test_solar_flare_spike(self):
        ctx = _make_ctx(solar_flare=True, shelter=False)
        delta = compute_radiation_delta(ctx)
        assert abs(delta - (RADIATION_BACKGROUND + SOLAR_FLARE_RADIATION)) < 0.001

    def test_shelter_reduces_radiation(self):
        ctx = _make_ctx(solar_flare=True, shelter=True)
        delta = compute_radiation_delta(ctx)
        expected = (RADIATION_BACKGROUND + SOLAR_FLARE_RADIATION) * SHELTER_RADIATION_MULT
        assert abs(delta - expected) < 0.001

    def test_no_shelter_no_reduction(self):
        ctx_no = _make_ctx(solar_flare=False, shelter=False)
        ctx_yes = _make_ctx(solar_flare=False, shelter=True)
        assert compute_radiation_delta(ctx_no) > compute_radiation_delta(ctx_yes)


# -- chronic conditions tests ------------------------------------------------

class TestChronicConditions:
    def test_bone_loss_not_before_onset(self):
        h = HealthState()
        rng = random.Random(42)
        ctx = _make_ctx(years_in_colony=BONE_LOSS_ONSET_YEAR - 1)
        conditions = check_new_conditions(h, ctx, rng)
        assert "bone_loss" not in conditions

    def test_bone_loss_possible_after_onset(self):
        """Over many trials, bone_loss should eventually appear."""
        h = HealthState()
        ctx = _make_ctx(years_in_colony=BONE_LOSS_ONSET_YEAR + 5)
        found = False
        for seed in range(200):
            rng = random.Random(seed)
            h_copy = HealthState()
            if "bone_loss" in check_new_conditions(h_copy, ctx, rng):
                found = True
                break
        assert found, "bone_loss never triggered in 200 trials"

    def test_no_duplicate_conditions(self):
        h = HealthState(chronic_conditions=["bone_loss"])
        ctx = _make_ctx(years_in_colony=50)
        rng = random.Random(42)
        conditions = check_new_conditions(h, ctx, rng)
        assert "bone_loss" not in conditions

    def test_radiation_damage_requires_threshold(self):
        h = HealthState(radiation=RADIATION_DAMAGE_THRESHOLD - 0.01)
        ctx = _make_ctx()
        rng = random.Random(42)
        conditions = check_new_conditions(h, ctx, rng)
        assert "radiation_damage" not in conditions

    def test_radiation_damage_possible_above_threshold(self):
        h = HealthState(radiation=RADIATION_DAMAGE_THRESHOLD + 0.1)
        ctx = _make_ctx()
        found = False
        for seed in range(200):
            rng = random.Random(seed)
            h_copy = HealthState(radiation=RADIATION_DAMAGE_THRESHOLD + 0.1)
            if "radiation_damage" in check_new_conditions(h_copy, ctx, rng):
                found = True
                break
        assert found, "radiation_damage never triggered"


class TestConditionPenalty:
    def test_no_conditions_no_penalty(self):
        h = HealthState()
        assert compute_condition_penalty(h, medicine_level=0.5, has_med_bay=False) == 0.0

    def test_bone_loss_penalty(self):
        h = HealthState(chronic_conditions=["bone_loss"])
        penalty = compute_condition_penalty(h, medicine_level=0.2, has_med_bay=False)
        assert penalty > 0

    def test_med_bay_reduces_penalty(self):
        h = HealthState(chronic_conditions=["bone_loss", "radiation_damage"])
        p_no_bay = compute_condition_penalty(h, medicine_level=0.5, has_med_bay=False)
        p_bay = compute_condition_penalty(h, medicine_level=0.5, has_med_bay=True)
        assert p_bay < p_no_bay

    def test_high_medicine_reduces_penalty(self):
        h = HealthState(chronic_conditions=["bone_loss"])
        p_low = compute_condition_penalty(h, medicine_level=0.2, has_med_bay=False)
        p_high = compute_condition_penalty(h, medicine_level=0.8, has_med_bay=False)
        assert p_high < p_low


# -- death modifier tests ---------------------------------------------------

class TestDeathModifier:
    def test_healthy_colonist_modifier_is_one(self):
        h = HealthState(fitness=0.85, radiation=0.0)
        assert health_death_modifier(h) == 1.0

    def test_low_fitness_increases_modifier(self):
        h = HealthState(fitness=0.1, radiation=0.0)
        assert health_death_modifier(h) > 1.0

    def test_high_radiation_increases_modifier(self):
        h = HealthState(fitness=0.85, radiation=0.8)
        assert health_death_modifier(h) > 1.0

    def test_chronic_conditions_increase_modifier(self):
        h = HealthState(fitness=0.85, radiation=0.0,
                        chronic_conditions=["bone_loss", "radiation_damage"])
        assert health_death_modifier(h) > 1.0

    def test_modifier_always_at_least_one(self):
        h = HealthState(fitness=1.0, radiation=0.0)
        assert health_death_modifier(h) >= 1.0


# -- death cause tests -------------------------------------------------------

class TestDeathCause:
    def test_healthy_colonist_no_cause(self):
        h = HealthState(fitness=0.85, radiation=0.0)
        assert health_death_cause(h, birth_year=0, current_year=10) is None

    def test_old_age_cause(self):
        h = HealthState(fitness=0.1, radiation=0.0, initial_age=30.0)
        # At year 30: age = 30 + 30*1.88 = 86.4 → past old age threshold
        cause = health_death_cause(h, birth_year=0, current_year=30)
        assert cause == "old_age"

    def test_radiation_sickness_cause(self):
        h = HealthState(fitness=0.5, radiation=0.9, initial_age=30.0)
        cause = health_death_cause(h, birth_year=0, current_year=10)
        assert cause == "radiation_sickness"

    def test_chronic_illness_cause(self):
        h = HealthState(fitness=0.05, radiation=0.3,
                        chronic_conditions=["radiation_damage"],
                        initial_age=30.0)
        cause = health_death_cause(h, birth_year=0, current_year=5)
        assert cause == "chronic_illness"

    def test_young_low_fitness_not_old_age(self):
        """Young colonist with low fitness shouldn't die of old_age."""
        h = HealthState(fitness=0.1, radiation=0.0, initial_age=25.0)
        # Year 5: age = 25 + 5*1.88 = 34.4 → too young for old_age
        cause = health_death_cause(h, birth_year=0, current_year=5)
        assert cause != "old_age"  # should be None or chronic_illness


# -- tick_health tests -------------------------------------------------------

class TestTickHealth:
    def test_creates_health_for_new_colonist(self):
        health_map: dict[str, HealthState] = {}
        ctx = [_make_ctx(cid="new-1", birth_year=0, years_in_colony=1)]
        rng = random.Random(42)
        tick_health(health_map, ctx, year=1, rng=rng)
        assert "new-1" in health_map

    def test_fitness_decreases_over_time(self):
        health_map: dict[str, HealthState] = {
            "c-0": HealthState(fitness=0.85, initial_age=30.0),
        }
        rng = random.Random(42)
        ctx = [_make_ctx(cid="c-0", birth_year=0, years_in_colony=20)]
        result = tick_health(health_map, ctx, year=20, rng=rng)
        assert health_map["c-0"].fitness < 0.85

    def test_radiation_increases_over_time(self):
        health_map: dict[str, HealthState] = {
            "c-0": HealthState(radiation=0.0, initial_age=30.0),
        }
        rng = random.Random(42)
        ctx = [_make_ctx(cid="c-0", birth_year=0)]
        tick_health(health_map, ctx, year=10, rng=rng)
        assert health_map["c-0"].radiation > 0.0

    def test_solar_flare_increases_radiation_more(self):
        h1 = {"c-0": HealthState(radiation=0.1, initial_age=30.0)}
        h2 = {"c-1": HealthState(radiation=0.1, initial_age=30.0)}
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        tick_health(h1, [_make_ctx(cid="c-0", solar_flare=False)], year=10, rng=rng1)
        tick_health(h2, [_make_ctx(cid="c-1", solar_flare=True)], year=10, rng=rng2)
        assert h2["c-1"].radiation > h1["c-0"].radiation

    def test_medicine_enables_recovery(self):
        health_map: dict[str, HealthState] = {
            "c-0": HealthState(fitness=0.5, initial_age=25.0),
        }
        rng = random.Random(42)
        ctx = [_make_ctx(cid="c-0", birth_year=0, medicine=0.8,
                         med_bay=True, years_in_colony=1)]
        tick_health(health_map, ctx, year=1, rng=rng)
        # Fitness should not have dropped as much due to recovery
        # (aging still applies, but recovery partially offsets)
        assert health_map["c-0"].fitness > 0.45

    def test_result_tracks_changes(self):
        health_map: dict[str, HealthState] = {
            "c-0": HealthState(fitness=0.85, radiation=0.1, initial_age=30.0),
        }
        rng = random.Random(42)
        ctx = [_make_ctx(cid="c-0", birth_year=0)]
        result = tick_health(health_map, ctx, year=10, rng=rng)
        assert "c-0" in result.fitness_changes
        assert "c-0" in result.radiation_changes

    def test_values_clamped_zero_one(self):
        """Property: all health values stay in [0, 1]."""
        rng = random.Random(99)
        health_map: dict[str, HealthState] = {}
        for i in range(10):
            health_map[f"c-{i}"] = HealthState(
                fitness=rng.uniform(0.0, 1.0),
                radiation=rng.uniform(0.0, 1.0),
                initial_age=rng.uniform(20, 60),
            )
        for year in range(1, 101):
            ctxs = [
                _make_ctx(
                    cid=f"c-{i}", birth_year=0,
                    solar_flare=(rng.random() < 0.15),
                    medicine=rng.uniform(0.2, 0.8),
                    years_in_colony=year,
                )
                for i in range(10)
            ]
            tick_health(health_map, ctxs, year=year, rng=rng)
            for h in health_map.values():
                assert 0.0 <= h.fitness <= 1.0, f"fitness out of range: {h.fitness}"
                assert 0.0 <= h.radiation <= 1.0, f"radiation out of range: {h.radiation}"


# -- integration: multi-year trajectory --------------------------------------

class TestMultiYearTrajectory:
    def test_colonist_ages_over_50_years(self):
        """A founder should see significant fitness decline over 50 Mars years."""
        health_map: dict[str, HealthState] = {
            "kira-sol": HealthState(fitness=0.85, initial_age=30.0),
        }
        rng = random.Random(42)
        for year in range(1, 51):
            ctx = [_make_ctx(
                cid="kira-sol", birth_year=0,
                solar_flare=(year % 7 == 0),
                medicine=0.5,
                years_in_colony=year,
            )]
            tick_health(health_map, ctx, year=year, rng=rng)
        h = health_map["kira-sol"]
        # After 50 Mars years (94 Earth years starting from 30 → 124):
        # significant fitness decline expected
        assert h.fitness < 0.5, f"Fitness too high after 50 years: {h.fitness}"
        assert h.radiation > 0.2, f"Radiation too low after 50 years: {h.radiation}"

    def test_radiation_accumulates_lethally(self):
        """Many solar flares should push radiation toward lethal levels."""
        health_map: dict[str, HealthState] = {
            "c-0": HealthState(fitness=0.85, initial_age=25.0),
        }
        rng = random.Random(42)
        for year in range(1, 51):
            ctx = [_make_ctx(
                cid="c-0", birth_year=0,
                solar_flare=True,  # every year!
                medicine=0.3,
                years_in_colony=year,
            )]
            tick_health(health_map, ctx, year=year, rng=rng)
        h = health_map["c-0"]
        assert h.radiation > 0.8, f"Radiation should be near-lethal: {h.radiation}"

    def test_med_bay_improves_long_term_outcomes(self):
        """Colonist with med_bay should have higher fitness after 30 years."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        h_no_bay: dict[str, HealthState] = {"c-0": HealthState(fitness=0.85, initial_age=30.0)}
        h_bay: dict[str, HealthState] = {"c-0": HealthState(fitness=0.85, initial_age=30.0)}
        for year in range(1, 31):
            ctx_no = [_make_ctx(cid="c-0", birth_year=0, medicine=0.5,
                                med_bay=False, years_in_colony=year)]
            ctx_yes = [_make_ctx(cid="c-0", birth_year=0, medicine=0.5,
                                 med_bay=True, years_in_colony=year)]
            tick_health(h_no_bay, ctx_no, year=year, rng=rng1)
            tick_health(h_bay, ctx_yes, year=year, rng=rng2)
        assert h_bay["c-0"].fitness >= h_no_bay["c-0"].fitness


# -- edge cases --------------------------------------------------------------

class TestEdgeCases:
    def test_zero_fitness_clamps(self):
        h = HealthState(fitness=0.01, initial_age=80.0)
        ctx = [_make_ctx(cid="c-0", birth_year=0, years_in_colony=30)]
        rng = random.Random(42)
        health_map = {"c-0": h}
        tick_health(health_map, ctx, year=30, rng=rng)
        assert health_map["c-0"].fitness >= 0.0

    def test_max_radiation_clamps(self):
        h = HealthState(radiation=0.99, initial_age=30.0)
        ctx = [_make_ctx(cid="c-0", solar_flare=True)]
        rng = random.Random(42)
        health_map = {"c-0": h}
        tick_health(health_map, ctx, year=10, rng=rng)
        assert health_map["c-0"].radiation <= 1.0

    def test_empty_context_list(self):
        health_map: dict[str, HealthState] = {}
        result = tick_health(health_map, [], year=1, rng=random.Random(42))
        assert result.new_conditions == []
        assert result.fitness_changes == {}

    def test_child_starts_young(self):
        """Children born in colony should have initial_age=0."""
        health_map: dict[str, HealthState] = {}
        ctx = [_make_ctx(cid="child-1", birth_year=20, years_in_colony=0)]
        rng = random.Random(42)
        tick_health(health_map, ctx, year=20, rng=rng)
        h = health_map["child-1"]
        # Initial age assigned by rng, but we should have a health entry
        assert h.fitness > 0.5
