#!/usr/bin/env python3
"""Tests for colony.py — colony population model."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.colony import (
    Resources,
    ColonyConfig,
    Colony,
    OLYMPUS_CONFIG,
    VALLES_CONFIG,
    HELLAS_CONFIG,
    carrying_capacity,
    produce_resources,
    compute_morale,
    population_delta,
    compute_migration,
)


def test_resources_min_supply():
    """min_supply_days returns the lowest resource."""
    r = Resources(o2_days=30, h2o_days=50, food_days=20, power_kwh=100)
    assert r.min_supply_days() == 20


def test_resources_serialization():
    """Resources should round-trip through to_dict."""
    r = Resources(o2_days=45.678, h2o_days=60.123, food_days=30.999, power_kwh=500.5)
    d = r.to_dict()
    assert d["o2_days"] == 45.7
    assert d["food_days"] == 31.0


def test_carrying_capacity_positive():
    """K must always be at least 10."""
    for config in [OLYMPUS_CONFIG, VALLES_CONFIG, HELLAS_CONFIG]:
        r = Resources(o2_days=1, h2o_days=1, food_days=1, power_kwh=0)
        k = carrying_capacity(config, r)
        assert k >= 10, f"K={k} for {config.name}"


def test_carrying_capacity_resource_sensitive():
    """K should decrease when resources are low."""
    config = OLYMPUS_CONFIG
    r_good = Resources(o2_days=90, h2o_days=90, food_days=90, power_kwh=500)
    r_bad = Resources(o2_days=10, h2o_days=10, food_days=10, power_kwh=50)

    k_good = carrying_capacity(config, r_good)
    k_bad = carrying_capacity(config, r_bad)
    assert k_good > k_bad, f"Good resources K={k_good} should exceed bad K={k_bad}"


def test_carrying_capacity_space_limited():
    """Larger habitats should support more people."""
    small = ColonyConfig("small", "greenhouse", habitat_volume_m3=1000, greenhouse_area_m2=100)
    big = ColonyConfig("big", "greenhouse", habitat_volume_m3=10000, greenhouse_area_m2=1000)
    r = Resources()

    k_small = carrying_capacity(small, r)
    k_big = carrying_capacity(big, r)
    assert k_big > k_small


def test_produce_resources_power():
    """Solar panels should produce power proportional to solar factor."""
    config = OLYMPUS_CONFIG
    r = Resources(power_kwh=500)
    pop = 50

    r_bright = produce_resources(config, Resources(power_kwh=500), pop, 1.0, -60, random.Random(1))
    r_dark = produce_resources(config, Resources(power_kwh=500), pop, 0.1, -60, random.Random(1))

    assert r_bright.power_kwh > r_dark.power_kwh


def test_produce_resources_bounds():
    """Resources should stay within bounds after production."""
    config = HELLAS_CONFIG
    r = Resources(o2_days=50, h2o_days=50, food_days=50, power_kwh=500)
    rng = random.Random(42)

    for _ in range(100):
        r = produce_resources(config, r, 100, 0.5, -50, rng)
        assert 0 <= r.o2_days <= 365
        assert 0 <= r.h2o_days <= 365
        assert 0 <= r.food_days <= 365
        assert 0 <= r.power_kwh <= 10000


def test_morale_baseline():
    """Morale with good conditions should be reasonable."""
    r = Resources(o2_days=90, h2o_days=90, food_days=90, power_kwh=500)
    m = compute_morale(r, 50, HELLAS_CONFIG, 0.5, False)
    assert 0.5 < m < 1.0, f"Baseline morale: {m}"


def test_morale_bounds():
    """Morale must stay in [0, 1]."""
    for config in [OLYMPUS_CONFIG, VALLES_CONFIG, HELLAS_CONFIG]:
        for supply in [1, 10, 30, 90]:
            for pop in [10, 50, 200]:
                for rad in [0.1, 1.0, 5.0]:
                    for storm in [True, False]:
                        r = Resources(
                            o2_days=supply, h2o_days=supply,
                            food_days=supply, power_kwh=100,
                        )
                        m = compute_morale(r, pop, config, rad, storm)
                        assert 0.0 <= m <= 1.0, f"Morale {m} out of bounds"


def test_morale_dust_storm_penalty():
    """Dust storms should reduce morale."""
    r = Resources(o2_days=90, h2o_days=90, food_days=90, power_kwh=500)
    m_calm = compute_morale(r, 50, HELLAS_CONFIG, 0.5, False)
    m_storm = compute_morale(r, 50, HELLAS_CONFIG, 0.5, True)
    assert m_storm < m_calm


def test_population_delta_conservation():
    """Births and deaths should be non-negative integers."""
    rng = random.Random(42)
    r = Resources(o2_days=60, h2o_days=60, food_days=60, power_kwh=500)

    for _ in range(200):
        births, deaths, events = population_delta(100, 200, 0.7, r, 0.5, rng)
        assert births >= 0
        assert deaths >= 0
        assert isinstance(births, int)
        assert isinstance(deaths, int)


def test_population_delta_zero_pop():
    """Zero population should produce zero births/deaths."""
    rng = random.Random(42)
    r = Resources()
    births, deaths, events = population_delta(0, 100, 0.7, r, 0.5, rng)
    assert births == 0
    assert deaths == 0


def test_population_delta_extinction_event():
    """Very low resources should increase deaths."""
    rng = random.Random(42)
    r_good = Resources(o2_days=90, h2o_days=90, food_days=90, power_kwh=500)
    r_bad = Resources(o2_days=3, h2o_days=3, food_days=3, power_kwh=10)

    deaths_good = sum(
        population_delta(200, 300, 0.7, r_good, 0.5, random.Random(i))[1]
        for i in range(100)
    )
    deaths_bad = sum(
        population_delta(200, 300, 0.7, r_bad, 0.5, random.Random(i))[1]
        for i in range(100)
    )
    assert deaths_bad > deaths_good, "Bad resources should cause more deaths"


def test_migration_requires_differential():
    """Migration should only happen with significant resource differential."""
    colonies = [
        {"resources": Resources(o2_days=90, h2o_days=90, food_days=90, power_kwh=500),
         "population": 100},
        {"resources": Resources(o2_days=90, h2o_days=90, food_days=90, power_kwh=500),
         "population": 100},
    ]
    moves = compute_migration(colonies, random.Random(42))
    assert len(moves) == 0, "Equal resources should not trigger migration"


def test_migration_from_poor_to_rich():
    """People should migrate from resource-poor to resource-rich."""
    colonies = [
        {"resources": Resources(o2_days=10, h2o_days=10, food_days=10, power_kwh=50),
         "population": 100},
        {"resources": Resources(o2_days=90, h2o_days=90, food_days=90, power_kwh=500),
         "population": 100},
    ]
    moves = compute_migration(colonies, random.Random(42))
    if moves:
        src, dst, count = moves[0]
        assert src == 0 and dst == 1, "Should migrate from poor (0) to rich (1)"
        assert count > 0


def test_colony_serialization_roundtrip():
    """Colony should survive to_dict → from_dict."""
    c = Colony(config=OLYMPUS_CONFIG)
    c.population = 150
    c.total_births = 20
    c.morale = 0.85
    c.population_history = [100, 110, 120, 130, 140, 150]

    d = c.to_dict()
    c2 = Colony.from_dict(d)

    assert c2.population == 150
    assert c2.total_births == 20
    assert c2.config.name == "Olympus Greenhouse"
    assert len(c2.population_history) == 6


def test_three_configs_distinct():
    """All three colony configs should have distinct names and strategies."""
    names = [c.name for c in [OLYMPUS_CONFIG, VALLES_CONFIG, HELLAS_CONFIG]]
    strategies = [c.strategy for c in [OLYMPUS_CONFIG, VALLES_CONFIG, HELLAS_CONFIG]]
    assert len(set(names)) == 3
    assert len(set(strategies)) == 3


def test_population_delta_logistic_damping():
    """Birth rate should decrease as population approaches K.

    Uses larger sample (2000 trials) because individual birth probability is low.
    """
    r = Resources(o2_days=90, h2o_days=90, food_days=90, power_kwh=500)

    births_low_pop = sum(
        population_delta(50, 200, 0.9, r, 0.1, random.Random(i))[0]
        for i in range(2000)
    )
    births_near_k = sum(
        population_delta(195, 200, 0.9, r, 0.1, random.Random(i + 10000))[0]
        for i in range(2000)
    )
    # Near K, total births should be lower (logistic damping)
    assert births_near_k < births_low_pop, (
        f"Births near K ({births_near_k}) should be less than far from K ({births_low_pop})"
    )
