"""Tests for the Mars-100 ecology organ (engine v9.0).

Covers: Biosphere state, biome progression, tipping points, modifier
computation, tick mechanics, integration with the engine, and physical
bounds.
"""
from __future__ import annotations

import random
import pytest

from src.mars100.ecology import (
    Biosphere, EcologyEvent, EcologyTickResult,
    tick_ecology, compute_ecology_modifiers,
    _compute_biome_level, _check_tipping_points,
    _generate_ecology_events, _apply_biome_feedback,
    BIOME_THRESHOLDS, BIOME_NAMES, TIPPING_POINTS, ECOLOGY_EVENTS,
    TERRAFORM_EFFORT_BASE, RESEARCH_EFFORT_BASE,
    ATMOSPHERE_DECAY, TEMPERATURE_DECAY, SOIL_DECAY, WATER_DECAY,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_bio(**overrides) -> Biosphere:
    return Biosphere(**overrides)


def make_rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


# ── Biosphere tests ─────────────────────────────────────────────────────────

class TestBiosphere:
    def test_defaults(self):
        b = Biosphere()
        assert b.atmosphere_pressure == pytest.approx(0.006)
        assert b.o2_fraction == pytest.approx(0.001)
        assert b.temperature == pytest.approx(0.0)
        assert b.soil_fertility == pytest.approx(0.01)
        assert b.water_coverage == pytest.approx(0.0)
        assert b.biome_level == 0
        assert b.tipping_points_hit == []

    def test_to_dict_roundtrip(self):
        b = make_bio(atmosphere_pressure=0.2, soil_fertility=0.3)
        b.tipping_points_hit = ["first_microbes"]
        d = b.to_dict()
        b2 = Biosphere.from_dict(d)
        assert b2.atmosphere_pressure == pytest.approx(0.2, abs=1e-5)
        assert b2.soil_fertility == pytest.approx(0.3, abs=1e-5)
        assert b2.tipping_points_hit == ["first_microbes"]

    def test_from_dict_empty(self):
        b = Biosphere.from_dict({})
        assert b.atmosphere_pressure == pytest.approx(0.006)

    def test_from_dict_none(self):
        b = Biosphere.from_dict(None)
        assert b.atmosphere_pressure == pytest.approx(0.006)

    def test_clamp(self):
        b = make_bio(atmosphere_pressure=1.5, o2_fraction=-0.1,
                     temperature=2.0, soil_fertility=-1.0,
                     water_coverage=3.0, biome_level=10)
        b.clamp()
        assert b.atmosphere_pressure == 1.0
        assert b.o2_fraction == 0.0
        assert b.temperature == 1.0
        assert b.soil_fertility == 0.0
        assert b.water_coverage == 1.0
        assert b.biome_level == 5

    def test_breathable_quality(self):
        b = make_bio(atmosphere_pressure=0.5, o2_fraction=0.2)
        assert b.breathable_quality() == pytest.approx(0.1, abs=1e-6)

    def test_habitability_score(self):
        b = Biosphere()
        score = b.habitability_score()
        assert 0.0 <= score <= 1.0

    def test_biome_name(self):
        for i, name in enumerate(BIOME_NAMES):
            b = make_bio(biome_level=i)
            assert b.biome_name() == name


# ── Biome level computation ─────────────────────────────────────────────────

class TestBiomeLevel:
    def test_barren_at_start(self):
        b = Biosphere()
        assert _compute_biome_level(b) == 0

    def test_level_1_microbes(self):
        b = make_bio(soil_fertility=0.10, temperature=0.08)
        assert _compute_biome_level(b) >= 1

    def test_level_2_lichen(self):
        b = make_bio(o2_fraction=0.04, soil_fertility=0.18, temperature=0.12)
        assert _compute_biome_level(b) >= 2

    def test_monotonic_advancement(self):
        """Higher values should never yield a lower biome level."""
        prev_level = 0
        for mult in range(1, 20):
            b = make_bio(
                atmosphere_pressure=0.03 * mult,
                o2_fraction=0.01 * mult,
                temperature=0.02 * mult,
                soil_fertility=0.03 * mult,
                water_coverage=0.01 * mult,
            )
            b.clamp()
            level = _compute_biome_level(b)
            assert level >= prev_level
            prev_level = level


# ── Tipping points ──────────────────────────────────────────────────────────

class TestTippingPoints:
    def test_none_at_start(self):
        b = Biosphere()
        new = _check_tipping_points(b)
        assert new == []

    def test_first_microbes(self):
        b = make_bio(soil_fertility=0.10, temperature=0.08)
        new = _check_tipping_points(b)
        assert "first_microbes" in new

    def test_not_repeated(self):
        b = make_bio(soil_fertility=0.10, temperature=0.08)
        b.tipping_points_hit = ["first_microbes"]
        new = _check_tipping_points(b)
        assert "first_microbes" not in new

    def test_multiple_at_once(self):
        b = make_bio(soil_fertility=0.3, temperature=0.3,
                     o2_fraction=0.1, water_coverage=0.1)
        new = _check_tipping_points(b)
        assert len(new) >= 2

    def test_all_tipping_points_reachable(self):
        """All tipping points can be hit with sufficiently high values."""
        b = make_bio(atmosphere_pressure=0.9, o2_fraction=0.9,
                     temperature=0.9, soil_fertility=0.9,
                     water_coverage=0.9)
        _check_tipping_points(b)
        assert set(TIPPING_POINTS.keys()) == set(b.tipping_points_hit)


# ── Ecology events ──────────────────────────────────────────────────────────

class TestEcologyEvents:
    def test_events_respect_min_biome(self):
        b = Biosphere()  # biome_level=0
        rng = make_rng()
        events = _generate_ecology_events(b, year=1, rng=rng)
        for e in events:
            template = next(t for t in ECOLOGY_EVENTS if t["name"] == e.name)
            assert template["min_biome"] <= b.biome_level

    def test_events_with_high_biome(self):
        b = make_bio(biome_level=3)
        rng = make_rng(seed=1)
        # Run many times to get at least one event
        all_events = []
        for y in range(100):
            all_events.extend(_generate_ecology_events(b, year=y, rng=rng))
        assert len(all_events) > 0, "Should get at least one event in 100 years"

    def test_event_effects_are_numeric(self):
        b = make_bio(biome_level=3)
        rng = make_rng()
        for _ in range(50):
            for e in _generate_ecology_events(b, year=1, rng=rng):
                for v in e.effects.values():
                    assert isinstance(v, (int, float))


# ── Biome feedback ──────────────────────────────────────────────────────────

class TestBiomeFeedback:
    def test_no_feedback_at_barren(self):
        b = make_bio(biome_level=0)
        fb = _apply_biome_feedback(b)
        assert fb == {}

    def test_feedback_at_level_1(self):
        b = make_bio(biome_level=1)
        fb = _apply_biome_feedback(b)
        assert fb.get("o2_fraction", 0) > 0
        assert fb.get("soil_fertility", 0) > 0

    def test_feedback_increases_with_biome(self):
        fb1 = _apply_biome_feedback(make_bio(biome_level=1))
        fb3 = _apply_biome_feedback(make_bio(biome_level=3))
        assert fb3.get("o2_fraction", 0) > fb1.get("o2_fraction", 0)


# ── Modifier computation ────────────────────────────────────────────────────

class TestModifiers:
    def test_starting_modifiers(self):
        b = Biosphere()
        mods = compute_ecology_modifiers(b)
        # Starting soil ~0.01 → food_production_mult ≈ 1.0025
        assert mods["food_production_mult"] == pytest.approx(1.0, abs=0.01)
        assert mods["air_maintenance_mult"] <= 1.0
        assert mods["medicine_production_mult"] >= 1.0
        assert mods["morale_ecology_bonus"] == pytest.approx(0.0)

    def test_improved_soil_boosts_food(self):
        b = make_bio(soil_fertility=0.5)
        mods = compute_ecology_modifiers(b)
        assert mods["food_production_mult"] > 1.1

    def test_breathable_air_reduces_maintenance(self):
        b = make_bio(atmosphere_pressure=0.3, o2_fraction=0.15)
        mods = compute_ecology_modifiers(b)
        assert mods["air_maintenance_mult"] < 1.0

    def test_biome_level_provides_morale(self):
        b = make_bio(biome_level=3)
        mods = compute_ecology_modifiers(b)
        assert mods["morale_ecology_bonus"] > 0

    def test_all_modifiers_positive(self):
        b = make_bio(atmosphere_pressure=0.5, o2_fraction=0.3,
                     soil_fertility=0.5, temperature=0.3,
                     water_coverage=0.2, biome_level=3)
        mods = compute_ecology_modifiers(b)
        for k, v in mods.items():
            assert v >= 0.0, f"{k} must be non-negative, got {v}"

    def test_air_maintenance_floor(self):
        b = make_bio(atmosphere_pressure=0.9, o2_fraction=0.9)
        mods = compute_ecology_modifiers(b)
        assert mods["air_maintenance_mult"] >= 0.5


# ── tick_ecology tests ───────────────────────────────────────────────────────

class TestTickEcology:
    def test_no_effort_minimal_change(self):
        b = Biosphere()
        rng = make_rng()
        result = tick_ecology(b, terraform_count=0, research_count=0,
                              population=10, infra_completed=[], year=1, rng=rng)
        assert isinstance(result, EcologyTickResult)
        assert result.year == 1
        assert result.terraform_effort < 0.01

    def test_terraform_increases_values(self):
        b = Biosphere()
        initial_atm = b.atmosphere_pressure
        rng = make_rng()
        tick_ecology(b, terraform_count=5, research_count=2,
                     population=10, infra_completed=[], year=1, rng=rng)
        assert b.atmosphere_pressure > initial_atm

    def test_greenhouse_amplifies(self):
        b1, b2 = Biosphere(), Biosphere()
        r1, r2 = make_rng(99), make_rng(99)
        tick_ecology(b1, terraform_count=3, research_count=1,
                     population=10, infra_completed=[], year=1, rng=r1)
        tick_ecology(b2, terraform_count=3, research_count=1,
                     population=10, infra_completed=["greenhouse"], year=1, rng=r2)
        assert b2.atmosphere_pressure > b1.atmosphere_pressure

    def test_all_values_clamped(self):
        b = Biosphere()
        rng = make_rng()
        for year in range(1, 201):
            tick_ecology(b, terraform_count=10, research_count=5,
                         population=20, infra_completed=["greenhouse", "soil_processor"],
                         year=year, rng=rng)
        assert 0.0 <= b.atmosphere_pressure <= 1.0
        assert 0.0 <= b.o2_fraction <= 1.0
        assert 0.0 <= b.temperature <= 1.0
        assert 0.0 <= b.soil_fertility <= 1.0
        assert 0.0 <= b.water_coverage <= 1.0

    def test_result_has_snapshot(self):
        b = Biosphere()
        rng = make_rng()
        result = tick_ecology(b, terraform_count=3, research_count=1,
                              population=10, infra_completed=[], year=5, rng=rng)
        snap = result.biosphere_snapshot
        assert "atmosphere_pressure" in snap
        assert "biome_name" in snap

    def test_tipping_points_fire(self):
        b = make_bio(soil_fertility=0.09, temperature=0.07)
        rng = make_rng()
        # Push just past threshold
        tick_ecology(b, terraform_count=5, research_count=2,
                     population=10, infra_completed=["greenhouse", "soil_processor"],
                     year=1, rng=rng)
        # Might or might not cross threshold in 1 year, run a few more
        for y in range(2, 20):
            tick_ecology(b, terraform_count=5, research_count=2,
                         population=10, infra_completed=["greenhouse", "soil_processor"],
                         year=y, rng=rng)
        assert len(b.tipping_points_hit) >= 1

    def test_deterministic(self):
        results = []
        for _ in range(3):
            b = Biosphere()
            rng = make_rng(seed=99)
            tick_ecology(b, terraform_count=3, research_count=1,
                         population=10, infra_completed=[], year=5, rng=rng)
            results.append(b.to_dict())
        assert results[0] == results[1] == results[2]


# ── 100-year properties ─────────────────────────────────────────────────────

class TestLongRun:
    def test_100_years_heavy_terraform(self):
        b = Biosphere()
        rng = make_rng(seed=42)
        for year in range(1, 101):
            tick_ecology(b, terraform_count=5, research_count=2,
                         population=10, infra_completed=["greenhouse", "soil_processor"],
                         year=year, rng=rng)
        assert len(b.tipping_points_hit) >= 1
        assert b.atmosphere_pressure > 0.006 * 5  # significantly more than start

    def test_100_years_no_effort(self):
        b = Biosphere()
        rng = make_rng(seed=42)
        for year in range(1, 101):
            tick_ecology(b, terraform_count=0, research_count=0,
                         population=5, infra_completed=[], year=year, rng=rng)
        # With industrial CO2 from 5 people but no terraforming: tiny growth
        assert b.biome_level <= 1
        assert len(b.tipping_points_hit) <= 1

    def test_biome_level_advances(self):
        b = Biosphere()
        rng = make_rng(seed=42)
        max_level = 0
        for year in range(1, 201):
            tick_ecology(b, terraform_count=8, research_count=3,
                         population=15,
                         infra_completed=["greenhouse", "soil_processor", "water_extractor"],
                         year=year, rng=rng)
            max_level = max(max_level, b.biome_level)
        assert max_level >= 2, "200 years of heavy terraforming should reach biome 2+"


# ── Engine integration ───────────────────────────────────────────────────────

class TestEngineIntegration:
    def test_ecology_in_year_result(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        for yr in result.years:
            d = yr.to_dict()
            assert "ecology" in d
            assert "biosphere" in d["ecology"]
            assert "biome_level" in d["ecology"]

    def test_ecology_in_final_result(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        d = result.to_dict()
        assert d["_meta"]["version"] == "9.0"
        assert "final_ecology" in d
        assert "atmosphere_pressure" in d["final_ecology"]
        assert "max_biome_level" in d["summary"]

    def test_engine_deterministic(self):
        from src.mars100.engine import Mars100Engine
        a = Mars100Engine(seed=99, total_years=10).run()
        b = Mars100Engine(seed=99, total_years=10).run()
        for ya, yb in zip(a.years, b.years):
            assert ya.ecology == yb.ecology

    def test_100_year_smoke(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        assert len(result.years) > 0
        d = result.to_dict()
        eco = d["final_ecology"]
        assert 0.0 <= eco["atmosphere_pressure"] <= 1.0
        assert 0.0 <= eco["habitability_score"] <= 1.0
