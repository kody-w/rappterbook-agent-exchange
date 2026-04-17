"""Tests for the ecology organ (engine v10.0)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.ecology import (
    Biosphere, EcologyTickResult, tick_ecology,
    _clamp, DUST_EVENTS, WATER_EVENTS, DAMAGE_EVENTS,
)


def make_bio(**kw) -> Biosphere:
    defaults = dict(soil_fertility=0.05, perchlorate_level=0.80,
                    flora_coverage=0.0, water_table=0.10,
                    atmosphere_density=0.01, self_sustaining_year=None)
    defaults.update(kw)
    return Biosphere(**defaults)


def run_tick(bio, seed=42, **kw) -> EcologyTickResult:
    defaults = dict(year=1, terraformers=2, avg_terraform_skill=0.5,
                    farmers=1, population=10, event_type="calm",
                    event_severity=0.3, infra_completed=[], rng=random.Random(seed))
    defaults.update(kw)
    return tick_ecology(bio, **defaults)


class TestClamp:
    def test_within(self):  assert _clamp(0.5) == 0.5
    def test_below(self):   assert _clamp(-0.1) == 0.0
    def test_above(self):   assert _clamp(1.5) == 1.0
    def test_custom(self):  assert _clamp(5, 0, 10) == 5


class TestBiosphere:
    def test_default_health(self):
        assert 0.0 <= Biosphere().health() <= 1.0

    def test_health_improves(self):
        assert Biosphere(soil_fertility=0.9).health() > Biosphere(soil_fertility=0.1).health()

    def test_bonuses_zero_no_flora(self):
        b = Biosphere(flora_coverage=0.0).resource_bonuses()
        assert b["air"] == 0.0 and b["food"] == 0.0

    def test_bonuses_positive_with_flora(self):
        b = Biosphere(flora_coverage=0.5, soil_fertility=0.5, perchlorate_level=0.2).resource_bonuses()
        assert b["air"] > 0 and b["food"] > 0 and b["water"] > 0

    def test_water_bonus_scales(self):
        lo = Biosphere(water_table=0.1).resource_bonuses()["water"]
        hi = Biosphere(water_table=0.9).resource_bonuses()["water"]
        assert hi > lo

    def test_not_sustaining_initially(self):
        assert not Biosphere().is_self_sustaining()

    def test_sustaining_good_bio(self):
        assert Biosphere(soil_fertility=0.8, water_table=0.6,
                         perchlorate_level=0.1, flora_coverage=0.3).is_self_sustaining()

    def test_not_sustaining_low_flora(self):
        assert not Biosphere(soil_fertility=0.8, water_table=0.6,
                             perchlorate_level=0.1, flora_coverage=0.01).is_self_sustaining()

    def test_clamp(self):
        bio = Biosphere(soil_fertility=1.5, perchlorate_level=-0.1,
                        flora_coverage=2.0, water_table=-1.0, atmosphere_density=3.0)
        bio.clamp()
        assert (bio.soil_fertility, bio.perchlorate_level, bio.flora_coverage,
                bio.water_table, bio.atmosphere_density) == (1.0, 0.0, 1.0, 0.0, 1.0)

    def test_to_dict_keys(self):
        expected = {"soil_fertility", "perchlorate_level", "flora_coverage",
                    "water_table", "atmosphere_density", "health",
                    "self_sustaining", "self_sustaining_year", "resource_bonuses"}
        assert set(Biosphere().to_dict().keys()) == expected

    def test_roundtrip(self):
        bio = Biosphere(soil_fertility=0.5, perchlorate_level=0.3,
                        flora_coverage=0.2, water_table=0.4,
                        atmosphere_density=0.05, self_sustaining_year=42)
        r = Biosphere.from_dict(bio.to_dict())
        assert abs(r.soil_fertility - 0.5) < 1e-4
        assert r.self_sustaining_year == 42

    def test_from_dict_empty(self):
        assert Biosphere.from_dict({}).soil_fertility == 0.05


class TestTickResult:
    def test_keys(self):
        d = EcologyTickResult().to_dict()
        assert "soil_delta" in d and "events" in d and "resource_bonuses" in d


class TestTickEcology:
    def test_perc_decreases(self):
        bio = make_bio()
        p0 = bio.perchlorate_level
        run_tick(bio, terraformers=3, avg_terraform_skill=0.7)
        assert bio.perchlorate_level < p0

    def test_perc_stable_no_terraformers(self):
        changes = [make_bio() for _ in range(50)]
        for i, b in enumerate(changes):
            run_tick(b, seed=i, terraformers=0, avg_terraform_skill=0.0)
        avg = sum(b.perchlorate_level - 0.80 for b in changes) / 50
        assert abs(avg) < 0.01

    def test_soil_improves_terraform(self):
        bio = make_bio(perchlorate_level=0.3)
        s0 = bio.soil_fertility
        run_tick(bio, terraformers=3, avg_terraform_skill=0.6)
        assert bio.soil_fertility > s0

    def test_soil_improves_farming(self):
        bio = make_bio(perchlorate_level=0.3)
        s0 = bio.soil_fertility
        run_tick(bio, terraformers=0, avg_terraform_skill=0.0, farmers=5)
        assert bio.soil_fertility > s0

    def test_water_decreases_large_pop(self):
        bio = make_bio(water_table=0.5)
        run_tick(bio, population=30, infra_completed=[])
        assert bio.water_table < 0.5

    def test_water_recycler_helps(self):
        a, b = make_bio(water_table=0.5), make_bio(water_table=0.5)
        run_tick(a, population=15, infra_completed=[])
        run_tick(b, population=15, infra_completed=["water_recycler"])
        assert b.water_table > a.water_table

    def test_flora_blocked_high_perc(self):
        bio = make_bio(perchlorate_level=0.9, soil_fertility=0.3, water_table=0.3)
        run_tick(bio)
        assert bio.flora_coverage == 0.0

    def test_flora_grows_good(self):
        bio = make_bio(soil_fertility=0.5, perchlorate_level=0.2,
                       water_table=0.4, flora_coverage=0.01)
        run_tick(bio, population=5)
        assert bio.flora_coverage > 0.01

    def test_greenhouse_boosts_flora(self):
        a = make_bio(soil_fertility=0.5, perchlorate_level=0.2,
                     water_table=0.4, flora_coverage=0.01)
        b = make_bio(soil_fertility=0.5, perchlorate_level=0.2,
                     water_table=0.4, flora_coverage=0.01)
        run_tick(a, population=5, infra_completed=[])
        run_tick(b, population=5, infra_completed=["greenhouse_dome"])
        assert b.flora_coverage > a.flora_coverage

    def test_dust_damages_flora(self):
        bio = make_bio(flora_coverage=0.5, soil_fertility=0.5,
                       perchlorate_level=0.2, water_table=0.4)
        r = run_tick(bio, event_type="dust_storm", event_severity=0.8, population=5)
        assert r.flora_delta < 0 or bio.flora_coverage < 0.5

    def test_water_event_boosts(self):
        a, b = make_bio(water_table=0.3), make_bio(water_table=0.3)
        run_tick(a, event_type="ice_strike", event_severity=0.7)
        run_tick(b, event_type="calm", event_severity=0.0)
        assert a.water_table > b.water_table

    def test_atmosphere_grows(self):
        bio = make_bio(flora_coverage=0.5, atmosphere_density=0.01)
        a0 = bio.atmosphere_density
        run_tick(bio, population=5)
        assert bio.atmosphere_density >= a0

    def test_sustaining_transition(self):
        bio = make_bio(soil_fertility=0.8, water_table=0.6,
                       perchlorate_level=0.1, flora_coverage=0.3)
        run_tick(bio, year=50, population=5)
        if bio.is_self_sustaining():
            assert bio.self_sustaining_year is not None

    def test_sustaining_reversible(self):
        bio = make_bio(soil_fertility=0.8, water_table=0.6,
                       perchlorate_level=0.1, flora_coverage=0.3,
                       self_sustaining_year=30)
        bio.flora_coverage = 0.01
        bio.water_table = 0.02
        r = run_tick(bio, year=50, population=20)
        if not bio.is_self_sustaining():
            assert bio.self_sustaining_year is None
            assert r.lost_sustaining

    def test_deltas_consistent(self):
        bio = make_bio(soil_fertility=0.3, perchlorate_level=0.4,
                       water_table=0.3, flora_coverage=0.1)
        s0 = bio.soil_fertility
        r = run_tick(bio, population=5)
        assert abs(r.soil_delta - (bio.soil_fertility - s0)) < 1e-9

    def test_events_populated(self):
        bio = make_bio()
        r = run_tick(bio, infra_completed=["water_recycler"])
        assert any("water_recycler" in e for e in r.events)


class TestBoundsInvariant:
    @pytest.mark.parametrize("seed", range(20))
    def test_bounded(self, seed):
        rng = random.Random(seed)
        bio = Biosphere(soil_fertility=rng.random(), perchlorate_level=rng.random(),
                        flora_coverage=rng.random(), water_table=rng.random(),
                        atmosphere_density=rng.random())
        tick_ecology(bio, year=rng.randint(1, 100), terraformers=rng.randint(0, 5),
                     avg_terraform_skill=rng.random(), farmers=rng.randint(0, 3),
                     population=rng.randint(1, 40),
                     event_type=rng.choice(["calm", "dust_storm", "ice_strike",
                                            "equipment_failure", "mega_storm"]),
                     event_severity=rng.random(),
                     infra_completed=rng.sample(["greenhouse_dome", "water_recycler",
                                                  "air_recycler"], k=rng.randint(0, 3)),
                     rng=rng)
        for a in ("soil_fertility", "perchlorate_level", "flora_coverage",
                  "water_table", "atmosphere_density"):
            assert 0.0 <= getattr(bio, a) <= 1.0, f"{a}={getattr(bio, a)}"


class TestMultiYear:
    def test_100yr_no_crash(self):
        bio = Biosphere()
        rng = random.Random(42)
        for yr in range(1, 101):
            tick_ecology(bio, year=yr, terraformers=rng.randint(1, 3),
                         avg_terraform_skill=0.5 + yr * 0.003,
                         farmers=rng.randint(1, 2), population=10 + yr // 10,
                         event_type=rng.choice(["calm", "calm", "dust_storm",
                                                "ice_strike", "equipment_failure"]),
                         event_severity=rng.random() * 0.5,
                         infra_completed=(["greenhouse_dome"] if yr > 20 else [])
                                         + (["water_recycler"] if yr > 30 else []),
                         rng=rng)
        assert 0.0 <= bio.flora_coverage <= 1.0
        assert bio.perchlorate_level < 0.80

    def test_heavy_terraform_grows_flora(self):
        bio = Biosphere()
        rng = random.Random(99)
        for yr in range(1, 61):
            tick_ecology(bio, year=yr, terraformers=4, avg_terraform_skill=0.7,
                         farmers=2, population=10, event_type="calm",
                         event_severity=0.0,
                         infra_completed=["greenhouse_dome", "water_recycler"],
                         rng=rng)
        assert bio.flora_coverage > 0.0, f"flora={bio.flora_coverage}, soil={bio.soil_fertility}, water={bio.water_table}, perc={bio.perchlorate_level}"
        assert bio.perchlorate_level < 0.5

    def test_no_terraform_barren(self):
        bio = Biosphere()
        rng = random.Random(7)
        for yr in range(1, 51):
            tick_ecology(bio, year=yr, terraformers=0, avg_terraform_skill=0.0,
                         farmers=0, population=10, event_type="calm",
                         event_severity=0.0, infra_completed=[], rng=rng)
        assert bio.flora_coverage == 0.0


class TestDeterminism:
    def test_same_seed(self):
        results = []
        for _ in range(2):
            bio = Biosphere()
            rng = random.Random(42)
            for yr in range(1, 21):
                tick_ecology(bio, year=yr, terraformers=2, avg_terraform_skill=0.5,
                             farmers=1, population=10, event_type="calm",
                             event_severity=0.3, infra_completed=[], rng=rng)
            results.append(bio.to_dict())
        assert results[0] == results[1]


class TestEngineIntegration:
    def test_engine_with_ecology(self):
        """Full engine run with ecology populates ecology data."""
        from src.mars100.engine import Mars100Engine
        e = Mars100Engine(seed=42, total_years=10)
        r = e.run()
        for yr in r.years:
            eco = yr.ecology
            assert isinstance(eco, dict), f"year {yr.year}: ecology not dict"
            assert "soil_fertility" in eco or "biosphere" in eco or "health" in eco
