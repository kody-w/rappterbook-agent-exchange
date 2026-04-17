"""Tests for the ecology organ (engine v10.0)."""
from __future__ import annotations

import random
from typing import Any
import pytest

from src.mars100.ecology import (
    EcologyState, EcologyYearContext, EcologyTickResult,
    BIOME_NAMES, BIOME_UP_THRESHOLDS, BIOME_DOWN_THRESHOLDS,
    MAX_FOOD_BONUS, MAX_AIR_BONUS, MAX_WATER_BONUS,
    MAX_NATURE_STRESS_REDUCTION,
    PRESSURE_GATE_KPA, TEMPERATURE_GATE_C, PERCHLORATE_GATE,
    compute_atmosphere_score, compute_soil_score, compute_flora_score,
    compute_biosphere_index, outdoor_habitable, has_greenhouse_tech,
    compute_ecology_bonuses, compute_resource_modifiers,
    compute_nature_stress_reduction,
    update_biome_level, tick_ecology,
)


class TestEcologyStateDefaults:
    def test_defaults(self) -> None:
        s = EcologyState()
        assert s.pressure_kpa == pytest.approx(0.6)
        assert s.o2_kpa == pytest.approx(0.001)
        assert s.temperature_c == pytest.approx(-60.0)
        assert s.perchlorate == pytest.approx(0.8)
        assert s.soil_organic == pytest.approx(0.0)
        assert s.greenhouse_crops == pytest.approx(0.0)
        assert s.outdoor_plants == pytest.approx(0.0)
        assert s.biome_level == 0
        assert s.biosphere_index == pytest.approx(0.0)

    def test_roundtrip(self) -> None:
        s = EcologyState(pressure_kpa=5.0, o2_kpa=1.2, temperature_c=-30.0,
                         perchlorate=0.2, soil_organic=0.5, soil_moisture=0.3,
                         greenhouse_crops=0.4, outdoor_plants=0.1,
                         biome_level=2, biosphere_index=0.25,
                         biome_unlocks=[1, 2])
        d = s.to_dict()
        s2 = EcologyState.from_dict(d)
        for attr in ("pressure_kpa", "o2_kpa", "temperature_c", "perchlorate",
                     "soil_organic", "soil_moisture", "greenhouse_crops",
                     "outdoor_plants", "biosphere_index"):
            assert getattr(s2, attr) == pytest.approx(getattr(s, attr), abs=1e-3)
        assert s2.biome_level == s.biome_level
        assert s2.biome_unlocks == s.biome_unlocks

    def test_clamp(self) -> None:
        s = EcologyState(pressure_kpa=3.0, o2_kpa=50.0, temperature_c=100.0,
                         perchlorate=-1.0, soil_organic=2.0,
                         biosphere_index=5.0)
        s.clamp()
        assert s.o2_kpa <= s.pressure_kpa
        assert s.temperature_c == 30.0
        assert s.perchlorate == 0.0
        assert s.soil_organic == 1.0
        assert s.biosphere_index == 1.0

    def test_health_alias(self) -> None:
        s = EcologyState(biosphere_index=0.42)
        assert s.health() == pytest.approx(0.42)


class TestScores:
    def test_atmosphere_score_default_low(self) -> None:
        s = EcologyState()
        score = compute_atmosphere_score(s)
        assert 0.0 <= score <= 0.05

    def test_atmosphere_score_earthlike_high(self) -> None:
        s = EcologyState(pressure_kpa=50.0, o2_kpa=16.0, temperature_c=20.0)
        score = compute_atmosphere_score(s)
        assert score == pytest.approx(1.0)

    def test_soil_score_default_low(self) -> None:
        s = EcologyState()
        score = compute_soil_score(s)
        assert 0.0 <= score <= 0.15

    def test_soil_score_good(self) -> None:
        s = EcologyState(perchlorate=0.0, soil_organic=1.0, soil_moisture=1.0)
        score = compute_soil_score(s)
        assert score == pytest.approx(1.0)

    def test_flora_score_default_zero(self) -> None:
        s = EcologyState()
        assert compute_flora_score(s) == pytest.approx(0.0)

    def test_flora_score_greenhouse(self) -> None:
        s = EcologyState(greenhouse_crops=1.0)
        assert compute_flora_score(s) == pytest.approx(0.6)

    def test_biosphere_index_composition(self) -> None:
        s = EcologyState(pressure_kpa=50.0, o2_kpa=16.0, temperature_c=20.0,
                         perchlorate=0.0, soil_organic=1.0, soil_moisture=1.0,
                         greenhouse_crops=1.0, outdoor_plants=1.0)
        idx = compute_biosphere_index(s)
        assert idx == pytest.approx(1.0)


class TestOutdoorHabitable:
    def test_mars_default_not_habitable(self) -> None:
        assert not outdoor_habitable(EcologyState())

    def test_habitable_when_all_gates_met(self) -> None:
        s = EcologyState(pressure_kpa=6.0, temperature_c=-35.0, perchlorate=0.1)
        assert outdoor_habitable(s)

    def test_pressure_too_low(self) -> None:
        s = EcologyState(pressure_kpa=4.9, temperature_c=-35.0, perchlorate=0.1)
        assert not outdoor_habitable(s)

    def test_temp_too_low(self) -> None:
        s = EcologyState(pressure_kpa=6.0, temperature_c=-41.0, perchlorate=0.1)
        assert not outdoor_habitable(s)

    def test_perchlorate_too_high(self) -> None:
        s = EcologyState(pressure_kpa=6.0, temperature_c=-35.0, perchlorate=0.31)
        assert not outdoor_habitable(s)


class TestHasGreenhouseTech:
    def test_no_infra(self) -> None:
        assert not has_greenhouse_tech([])

    def test_greenhouse_present(self) -> None:
        assert has_greenhouse_tech(["advanced_greenhouse"])

    def test_case_insensitive(self) -> None:
        assert has_greenhouse_tech(["Greenhouse_mk2"])


class TestBonuses:
    def test_zero_on_default(self) -> None:
        b = compute_ecology_bonuses(EcologyState())
        assert b["food"] == pytest.approx(0.0, abs=1e-5)
        assert b["water"] == pytest.approx(0.05 * MAX_WATER_BONUS, abs=1e-5)

    def test_food_from_greenhouse(self) -> None:
        s = EcologyState(greenhouse_crops=1.0)
        b = compute_ecology_bonuses(s)
        assert b["food"] > 0.01

    def test_bonuses_bounded(self) -> None:
        s = EcologyState(greenhouse_crops=1.0, outdoor_plants=1.0,
                         soil_moisture=1.0, o2_kpa=20.0)
        b = compute_ecology_bonuses(s)
        assert b["food"] <= MAX_FOOD_BONUS * 2
        assert b["water"] <= MAX_WATER_BONUS


class TestResourceModifiers:
    def test_default_near_one(self) -> None:
        m = compute_resource_modifiers(EcologyState())
        assert m["food_spoilage_mult"] == pytest.approx(1.0)
        assert m["air_maintenance_mult"] == pytest.approx(1.0)

    def test_improves_with_biosphere(self) -> None:
        s = EcologyState(biosphere_index=0.5)
        m = compute_resource_modifiers(s)
        assert m["food_spoilage_mult"] < 1.0
        assert m["food_spoilage_mult"] >= 0.5


class TestNatureStress:
    def test_zero_on_default(self) -> None:
        assert compute_nature_stress_reduction(EcologyState()) == pytest.approx(0.0)

    def test_positive_with_biosphere(self) -> None:
        s = EcologyState(biosphere_index=0.5)
        val = compute_nature_stress_reduction(s)
        assert val == pytest.approx(0.5 * MAX_NATURE_STRESS_REDUCTION)

    def test_max_cap(self) -> None:
        s = EcologyState(biosphere_index=1.0)
        val = compute_nature_stress_reduction(s)
        assert val == pytest.approx(MAX_NATURE_STRESS_REDUCTION)


class TestBiomeLevels:
    def test_stays_barren_at_zero(self) -> None:
        s = EcologyState(biosphere_index=0.0, biome_level=0)
        assert update_biome_level(s) is None
        assert s.biome_level == 0

    def test_promote_to_lichen(self) -> None:
        s = EcologyState(biosphere_index=0.08, biome_level=0)
        t = update_biome_level(s)
        assert t is not None
        assert t["to_name"] == "lichen"
        assert t["direction"] == "up"
        assert s.biome_level == 1

    def test_hysteresis_prevents_thrash(self) -> None:
        s = EcologyState(biosphere_index=0.06, biome_level=1, biome_unlocks=[1])
        t = update_biome_level(s)
        assert t is None
        assert s.biome_level == 1

    def test_demote_below_down_threshold(self) -> None:
        s = EcologyState(biosphere_index=0.04, biome_level=1, biome_unlocks=[1])
        t = update_biome_level(s)
        assert t is not None
        assert t["direction"] == "down"
        assert s.biome_level == 0

    def test_first_time_tracked(self) -> None:
        s = EcologyState(biosphere_index=0.08, biome_level=0)
        t = update_biome_level(s)
        assert t is not None
        assert t["first_time"] is True
        assert 1 in s.biome_unlocks

    def test_revisit_not_first_time(self) -> None:
        s = EcologyState(biosphere_index=0.08, biome_level=0, biome_unlocks=[1])
        t = update_biome_level(s)
        assert t is not None
        assert t["first_time"] is False


class TestTickEcology:
    def _make_ctx(self, **kw: Any) -> EcologyYearContext:
        defaults: dict[str, Any] = dict(
            year=1, terraform_count=0, farm_count=0,
            research_count=0, population=10,
            infrastructure_completed=[])
        defaults.update(kw)
        return EcologyYearContext(**defaults)

    def test_no_action_slight_decay(self) -> None:
        eco = EcologyState(greenhouse_crops=0.5)
        rng = random.Random(42)
        tick_ecology(eco, self._make_ctx(), rng)
        assert eco.greenhouse_crops < 0.5

    def test_terraform_raises_pressure(self) -> None:
        eco = EcologyState()
        rng = random.Random(42)
        tick_ecology(eco, self._make_ctx(terraform_count=3), rng)
        assert eco.pressure_kpa > 0.6

    def test_farm_reduces_perchlorate(self) -> None:
        eco = EcologyState(perchlorate=0.5)
        rng = random.Random(42)
        tick_ecology(eco, self._make_ctx(farm_count=3), rng)
        assert eco.perchlorate < 0.5

    def test_bonuses_from_lagged_state(self) -> None:
        eco = EcologyState(greenhouse_crops=0.5, soil_moisture=0.3)
        rng = random.Random(42)
        result = tick_ecology(eco, self._make_ctx(), rng)
        assert result.resource_bonuses["food"] > 0

    def test_greenhouse_tech_warms(self) -> None:
        eco = EcologyState()
        rng = random.Random(42)
        tick_ecology(eco, self._make_ctx(
            infrastructure_completed=["greenhouse_mk1"]), rng)
        assert eco.temperature_c > -60.0

    def test_outdoor_plants_die_without_habitat(self) -> None:
        eco = EcologyState(outdoor_plants=0.5)
        rng = random.Random(42)
        tick_ecology(eco, self._make_ctx(), rng)
        assert eco.outdoor_plants < 0.5

    def test_outdoor_plants_grow_when_habitable(self) -> None:
        eco = EcologyState(pressure_kpa=6.0, temperature_c=-30.0,
                           perchlorate=0.1, greenhouse_crops=0.3,
                           soil_organic=0.5, soil_moisture=0.5,
                           outdoor_plants=0.1)
        rng = random.Random(42)
        tick_ecology(eco, self._make_ctx(farm_count=2), rng)
        assert eco.outdoor_plants > 0.1

    def test_biosphere_index_updated(self) -> None:
        eco = EcologyState()
        rng = random.Random(42)
        tick_ecology(eco, self._make_ctx(terraform_count=3, farm_count=3), rng)
        assert eco.biosphere_index > 0.0

    def test_tick_result_has_fields(self) -> None:
        eco = EcologyState()
        rng = random.Random(42)
        result = tick_ecology(eco, self._make_ctx(terraform_count=1), rng)
        assert "food" in result.resource_bonuses
        assert "air" in result.resource_bonuses
        assert "water" in result.resource_bonuses
        assert isinstance(result.nature_stress_reduction, float)

    def test_to_dict(self) -> None:
        result = EcologyTickResult(
            resource_bonuses={"food": 0.01},
            nature_stress_reduction=0.02,
            biome_transition={"to_name": "lichen"},
            tipping_event="ecological_bloom")
        d = result.to_dict()
        assert d["tipping_event"] == "ecological_bloom"
        assert "biome_transition" in d


class TestDeterminism:
    def test_same_seed_same_result(self) -> None:
        for seed in range(5):
            eco1 = EcologyState()
            eco2 = EcologyState()
            ctx = EcologyYearContext(year=1, terraform_count=2, farm_count=2,
                                     research_count=1, population=10,
                                     infrastructure_completed=[])
            r1 = tick_ecology(eco1, ctx, random.Random(seed))
            r2 = tick_ecology(eco2, ctx, random.Random(seed))
            assert eco1.to_dict() == eco2.to_dict()
            assert r1.resource_bonuses == r2.resource_bonuses


class TestBoundsInvariants:
    """Property-based: 10 seeds x 50 years, all values in physical bounds."""

    @pytest.mark.parametrize("seed", range(10))
    def test_50_year_bounds(self, seed: int) -> None:
        eco = EcologyState()
        rng = random.Random(seed)
        for year in range(50):
            terraform = rng.randint(0, 4)
            farm = rng.randint(0, 4)
            infra = ["greenhouse_mk1"] if year > 10 else []
            ctx = EcologyYearContext(
                year=year, terraform_count=terraform,
                farm_count=farm, research_count=1,
                population=10,
                infrastructure_completed=infra)
            tick_ecology(eco, ctx, random.Random(seed * 1000 + year))
            assert 0.0 <= eco.pressure_kpa <= 200.0
            assert 0.0 <= eco.o2_kpa <= eco.pressure_kpa + 0.001
            assert -80.0 <= eco.temperature_c <= 30.0
            assert 0.0 <= eco.perchlorate <= 1.0
            assert 0.0 <= eco.soil_organic <= 1.0
            assert 0.0 <= eco.soil_moisture <= 1.0
            assert 0.0 <= eco.greenhouse_crops <= 1.0
            assert 0.0 <= eco.outdoor_plants <= 1.0
            assert 0.0 <= eco.biosphere_index <= 1.0
            assert 0 <= eco.biome_level <= len(BIOME_NAMES) - 1


class TestReachability:
    """Ensure biosphere can advance beyond barren with sustained effort."""

    def test_100_year_advancement(self) -> None:
        eco = EcologyState()
        rng = random.Random(42)
        for year in range(100):
            infra = ["greenhouse_mk1"] if year > 5 else []
            ctx = EcologyYearContext(
                year=year, terraform_count=3,
                farm_count=3, research_count=2,
                population=10,
                infrastructure_completed=infra)
            tick_ecology(eco, ctx, random.Random(42 * 1000 + year))
        assert eco.biosphere_index > 0.15
        assert eco.biome_level >= 1
        assert eco.perchlorate < 0.1
        assert eco.greenhouse_crops > 0.1


class TestSmokeTest:
    """Full smoke test: 100 years, no crash."""

    def test_100_years_no_crash(self) -> None:
        eco = EcologyState()
        rng = random.Random(777)
        for year in range(100):
            ctx = EcologyYearContext(
                year=year,
                terraform_count=rng.randint(0, 5),
                farm_count=rng.randint(0, 5),
                research_count=rng.randint(0, 3),
                population=10,
                infrastructure_completed=(
                    ["greenhouse_mk1"] if year > 10 else []))
            tick_ecology(eco, ctx, random.Random(777 * 1000 + year))
        assert eco.biosphere_index >= 0.0
