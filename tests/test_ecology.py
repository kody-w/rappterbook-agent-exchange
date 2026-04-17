"""Tests for the ecology organ (Mars-100 engine v10.0)."""
from __future__ import annotations

import random
from src.mars100.ecology import (
    Atmosphere, SoilState, WaterCycle, Flora, Biosphere,
    EcologyResult, biome_from_progress,
    tick_ecology, compute_ecology_modifiers, compute_ecology_upkeep,
    FLORA_TIPPING_THRESHOLD,
)


class TestAtmosphere:
    def test_defaults(self) -> None:
        a = Atmosphere()
        assert a.o2 == 0.002
        assert a.co2 == 0.95
        assert a.pressure == 0.01

    def test_clamp_negative(self) -> None:
        a = Atmosphere(o2=-0.1, co2=1.5, pressure=2.0)
        a.clamp()
        assert a.o2 >= 0.0
        assert a.co2 <= 1.0
        assert a.pressure <= 1.0

    def test_o2_co2_constraint(self) -> None:
        a = Atmosphere(o2=0.6, co2=0.7, pressure=0.5)
        a.clamp()
        assert a.o2 + a.co2 <= 1.0 + 1e-9
        assert a.o2 > 0
        assert a.co2 > 0

    def test_roundtrip(self) -> None:
        a = Atmosphere(o2=0.05, co2=0.80, pressure=0.03)
        a2 = Atmosphere.from_dict(a.to_dict())
        assert abs(a2.o2 - a.o2) < 1e-5
        assert abs(a2.co2 - a.co2) < 1e-5


class TestSoilState:
    def test_defaults(self) -> None:
        s = SoilState()
        assert s.perchlorate == 0.80
        assert s.nutrient_level == 0.02

    def test_clamp(self) -> None:
        s = SoilState(nutrient_level=-0.5, perchlorate=2.0)
        s.clamp()
        assert s.nutrient_level == 0.0
        assert s.perchlorate == 1.0

    def test_roundtrip(self) -> None:
        s = SoilState(nutrient_level=0.5, perchlorate=0.3)
        s2 = SoilState.from_dict(s.to_dict())
        assert abs(s2.nutrient_level - s.nutrient_level) < 1e-5


class TestWaterCycle:
    def test_defaults(self) -> None:
        w = WaterCycle()
        assert w.ice_reserves == 0.60
        assert w.recycling_efficiency == 0.30
        assert w.aquifer_discovered is False

    def test_roundtrip(self) -> None:
        w = WaterCycle(ice_reserves=0.4, recycling_efficiency=0.7,
                       aquifer_discovered=True)
        w2 = WaterCycle.from_dict(w.to_dict())
        assert w2.aquifer_discovered is True
        assert abs(w2.recycling_efficiency - 0.7) < 1e-5


class TestFlora:
    def test_defaults(self) -> None:
        f = Flora()
        assert f.coverage == 0.0
        assert f.crop_health == 0.30

    def test_clamp(self) -> None:
        f = Flora(coverage=1.5, biodiversity=-0.1, crop_health=2.0)
        f.clamp()
        assert f.coverage == 1.0
        assert f.biodiversity == 0.0
        assert f.crop_health == 1.0


class TestBiosphere:
    def test_default_biome_is_barren(self) -> None:
        b = Biosphere()
        assert b.biome == "barren"

    def test_terraforming_progress_derived(self) -> None:
        b = Biosphere()
        p1 = b.terraforming_progress
        b.atmosphere.o2 = 0.21
        b.atmosphere.pressure = 0.5
        p2 = b.terraforming_progress
        assert p2 > p1

    def test_biome_thresholds(self) -> None:
        assert biome_from_progress(0.0) == "barren"
        assert biome_from_progress(0.14) == "barren"
        assert biome_from_progress(0.15) == "pioneer"
        assert biome_from_progress(0.30) == "greenhouse"
        assert biome_from_progress(0.50) == "garden"
        assert biome_from_progress(0.70) == "forest"
        assert biome_from_progress(0.85) == "earthlike"
        assert biome_from_progress(1.0) == "earthlike"

    def test_roundtrip(self) -> None:
        b = Biosphere()
        b.atmosphere.o2 = 0.10
        b.flora.coverage = 0.3
        b2 = Biosphere.from_dict(b.to_dict())
        assert abs(b2.atmosphere.o2 - 0.10) < 1e-5
        assert abs(b2.flora.coverage - 0.3) < 1e-5

    def test_clamp_cascades(self) -> None:
        b = Biosphere()
        b.atmosphere.o2 = 2.0
        b.soil.perchlorate = -1.0
        b.clamp()
        assert b.atmosphere.o2 <= 1.0
        assert b.soil.perchlorate >= 0.0


class TestTickEcology:
    def _rng(self, seed: int = 42) -> random.Random:
        return random.Random(seed)

    def test_no_actions_decays(self) -> None:
        bio = Biosphere()
        bio.flora.coverage = 0.20
        cov0 = bio.flora.coverage
        tick_ecology(bio, year=5, action_counts={},
                     event_names=[], event_severities=[],
                     infra_completed=[], rng=self._rng())
        assert bio.flora.coverage < cov0

    def test_terraform_increases_o2(self) -> None:
        bio = Biosphere()
        o2_0 = bio.atmosphere.o2
        tick_ecology(bio, year=5, action_counts={"terraform": 3},
                     event_names=[], event_severities=[],
                     infra_completed=[], rng=self._rng())
        assert bio.atmosphere.o2 > o2_0

    def test_terraform_reduces_perchlorate(self) -> None:
        bio = Biosphere()
        p0 = bio.soil.perchlorate
        tick_ecology(bio, year=5, action_counts={"terraform": 2},
                     event_names=[], event_severities=[],
                     infra_completed=[], rng=self._rng())
        assert bio.soil.perchlorate < p0

    def test_farming_boosts_flora(self) -> None:
        bio = Biosphere()
        bio.soil.nutrient_level = 0.8
        bio.soil.perchlorate = 0.2
        bio.flora.coverage = 0.1
        cov0 = bio.flora.coverage
        # Need high soil quality (nutrient * (1 - perchlorate)) for gains to exceed decay
        tick_ecology(bio, year=5, action_counts={"farm": 5},
                     event_names=[], event_severities=[],
                     infra_completed=[], rng=self._rng())
        assert bio.flora.coverage > cov0

    def test_dust_storm_damages_flora(self) -> None:
        bio = Biosphere()
        bio.flora.coverage = 0.40
        bio.flora.crop_health = 0.60
        cov0 = bio.flora.coverage
        result = tick_ecology(bio, year=5, action_counts={},
                              event_names=["dust_storm"],
                              event_severities=[0.8],
                              infra_completed=[], rng=self._rng())
        assert bio.flora.coverage < cov0
        assert result.flora_damage > 0

    def test_ice_volcano_boosts_water(self) -> None:
        bio = Biosphere()
        ice0 = bio.water_cycle.ice_reserves
        tick_ecology(bio, year=5, action_counts={},
                     event_names=["ice_volcano"],
                     event_severities=[0.5],
                     infra_completed=[], rng=self._rng())
        assert bio.water_cycle.ice_reserves > ice0

    def test_tipping_point(self) -> None:
        bio = Biosphere()
        bio.flora.coverage = 0.55
        o2_0 = bio.atmosphere.o2
        result = tick_ecology(bio, year=50, action_counts={},
                              event_names=[], event_severities=[],
                              infra_completed=[], rng=self._rng())
        assert result.tipping_point_triggered is True
        bio2 = Biosphere()
        bio2.flora.coverage = 0.40
        bio2.atmosphere.o2 = o2_0
        tick_ecology(bio2, year=50, action_counts={},
                     event_names=[], event_severities=[],
                     infra_completed=[], rng=random.Random(42))
        assert bio.atmosphere.o2 > bio2.atmosphere.o2

    def test_aquifer_discovery(self) -> None:
        discovered = False
        for seed in range(200):
            bio = Biosphere()
            tick_ecology(bio, year=20, action_counts={"explore": 5},
                         event_names=[], event_severities=[],
                         infra_completed=[], rng=random.Random(seed))
            if bio.water_cycle.aquifer_discovered:
                discovered = True
                break
        assert discovered, "Aquifer should be discoverable"

    def test_aquifer_not_before_year_15(self) -> None:
        for seed in range(50):
            bio = Biosphere()
            tick_ecology(bio, year=5, action_counts={"explore": 5},
                         event_names=[], event_severities=[],
                         infra_completed=[], rng=random.Random(seed))
            assert not bio.water_cycle.aquifer_discovered

    def test_infra_tech_bonuses(self) -> None:
        bio = Biosphere()
        bio.flora.crop_health = 0.50
        tick_ecology(bio, year=10, action_counts={},
                     event_names=[], event_severities=[],
                     infra_completed=["greenhouse_dome"],
                     rng=self._rng())
        bio2 = Biosphere()
        bio2.flora.crop_health = 0.50
        tick_ecology(bio2, year=10, action_counts={},
                     event_names=[], event_severities=[],
                     infra_completed=[],
                     rng=random.Random(42))
        assert bio.flora.crop_health > bio2.flora.crop_health

    def test_result_structure(self) -> None:
        bio = Biosphere()
        result = tick_ecology(bio, year=1, action_counts={},
                              event_names=[], event_severities=[],
                              infra_completed=[], rng=self._rng())
        d = result.to_dict()
        assert "biome_before" in d
        assert "biome_after" in d
        assert "terraforming_before" in d
        assert "terraforming_after" in d


class TestEcologyBounds:
    def test_100_year_all_values_physical(self) -> None:
        rng = random.Random(42)
        bio = Biosphere()
        for year in range(1, 101):
            n_tf = rng.randint(0, 3)
            n_farm = rng.randint(0, 3)
            n_explore = rng.randint(0, 2)
            events = rng.sample(
                ["dust_storm", "ice_volcano", "resource_strike",
                 "solar_flare", "calm"],
                k=min(2, 5))
            sevs = [rng.uniform(0.1, 0.9) for _ in events]
            tick_ecology(bio, year=year,
                         action_counts={"terraform": n_tf, "farm": n_farm,
                                        "explore": n_explore},
                         event_names=events, event_severities=sevs,
                         infra_completed=["greenhouse_dome", "water_recycler"],
                         rng=rng)
            assert 0.0 <= bio.atmosphere.o2 <= 1.0
            assert 0.0 <= bio.atmosphere.co2 <= 1.0
            assert bio.atmosphere.o2 + bio.atmosphere.co2 <= 1.0 + 1e-9
            assert 0.0 <= bio.atmosphere.pressure <= 1.0
            assert 0.0 <= bio.soil.nutrient_level <= 1.0
            assert 0.0 <= bio.soil.perchlorate <= 1.0
            assert 0.0 <= bio.water_cycle.ice_reserves <= 1.0
            assert 0.0 <= bio.water_cycle.recycling_efficiency <= 1.0
            assert 0.0 <= bio.flora.coverage <= 1.0
            assert 0.0 <= bio.flora.biodiversity <= 1.0
            assert 0.0 <= bio.flora.crop_health <= 1.0
            assert 0.0 <= bio.terraforming_progress <= 1.0

    def test_multiple_seeds(self) -> None:
        for seed in range(10):
            rng = random.Random(seed)
            bio = Biosphere()
            for year in range(1, 51):
                tick_ecology(bio, year=year,
                             action_counts={"terraform": 2, "farm": 2},
                             event_names=["dust_storm"],
                             event_severities=[0.9],
                             infra_completed=[], rng=rng)
                assert 0.0 <= bio.atmosphere.o2 <= 1.0
                assert bio.atmosphere.o2 + bio.atmosphere.co2 <= 1.0 + 1e-9
                assert 0.0 <= bio.flora.coverage <= 1.0

    def test_deterministic(self) -> None:
        def run(seed: int) -> dict:
            rng = random.Random(seed)
            bio = Biosphere()
            for year in range(1, 21):
                tick_ecology(bio, year=year,
                             action_counts={"terraform": 1, "farm": 1},
                             event_names=[], event_severities=[],
                             infra_completed=[], rng=rng)
            return bio.to_dict()
        assert run(99) == run(99)

    def test_different_seeds_diverge(self) -> None:
        def run(seed: int) -> float:
            rng = random.Random(seed)
            bio = Biosphere()
            for year in range(1, 21):
                tick_ecology(bio, year=year,
                             action_counts={"terraform": 1, "farm": 1},
                             event_names=[], event_severities=[],
                             infra_completed=[], rng=rng)
            return bio.terraforming_progress
        assert run(42) != run(43)


class TestEcologyModifiers:
    def test_default_biosphere_near_neutral(self) -> None:
        bio = Biosphere()
        mods = compute_ecology_modifiers(bio)
        assert 0.8 <= mods["food_spoilage_mult"] <= 1.0
        assert mods["death_rate_ecology_mult"] > 1.0

    def test_improved_biosphere(self) -> None:
        bio = Biosphere()
        bio.flora.crop_health = 0.80
        bio.water_cycle.recycling_efficiency = 0.70
        bio.atmosphere.o2 = 0.15
        bio.soil.perchlorate = 0.20
        mods = compute_ecology_modifiers(bio)
        assert mods["food_spoilage_mult"] < 0.9
        assert mods["water_maintenance_mult"] < 0.9
        assert mods["death_rate_ecology_mult"] == 1.0

    def test_high_perchlorate_penalty(self) -> None:
        bio = Biosphere()
        bio.soil.perchlorate = 0.90
        mods = compute_ecology_modifiers(bio)
        assert mods["death_rate_ecology_mult"] > 1.1

    def test_modifiers_bounded(self) -> None:
        bio = Biosphere()
        bio.flora.crop_health = 1.0
        bio.water_cycle.recycling_efficiency = 1.0
        bio.atmosphere.o2 = 0.21
        mods = compute_ecology_modifiers(bio)
        assert mods["food_spoilage_mult"] >= 0.3
        assert mods["water_maintenance_mult"] >= 0.4
        assert mods["air_spoilage_mult"] >= 0.5


class TestEcologyUpkeep:
    def test_baseline_costs(self) -> None:
        bio = Biosphere()
        costs = compute_ecology_upkeep(bio)
        assert costs["power"] > 0
        assert costs["water"] > 0

    def test_higher_flora_higher_costs(self) -> None:
        bio_low = Biosphere()
        bio_low.flora.coverage = 0.1
        bio_high = Biosphere()
        bio_high.flora.coverage = 0.8
        assert compute_ecology_upkeep(bio_high)["power"] > compute_ecology_upkeep(bio_low)["power"]


class TestEcologySmoke:
    def test_100_year_smoke(self) -> None:
        rng = random.Random(42)
        bio = Biosphere()
        biome_history: list[str] = []
        for year in range(1, 101):
            n_tf = max(0, 2 + rng.randint(-1, 1))
            n_farm = max(0, 3 + rng.randint(-1, 1))
            n_explore = 1 if rng.random() < 0.3 else 0
            events: list[str] = []
            sevs: list[float] = []
            if rng.random() < 0.3:
                events.append("dust_storm")
                sevs.append(rng.uniform(0.3, 0.8))
            if rng.random() < 0.1:
                events.append("ice_volcano")
                sevs.append(rng.uniform(0.2, 0.6))
            infra: list[str] = []
            if year > 10:
                infra.append("greenhouse_dome")
            if year > 15:
                infra.append("water_recycler")
            result = tick_ecology(bio, year=year,
                                  action_counts={"terraform": n_tf, "farm": n_farm,
                                                  "explore": n_explore},
                                  event_names=events, event_severities=sevs,
                                  infra_completed=infra, rng=rng)
            biome_history.append(result.biome_after)
            mods = compute_ecology_modifiers(bio)
            assert all(v > 0 for v in mods.values())
            upkeep = compute_ecology_upkeep(bio)
            assert all(v >= 0 for v in upkeep.values())
        assert biome_history[-1] != "barren"

    def test_no_activity_stays_barren(self) -> None:
        rng = random.Random(42)
        bio = Biosphere()
        for year in range(1, 101):
            tick_ecology(bio, year=year, action_counts={},
                         event_names=[], event_severities=[],
                         infra_completed=[], rng=rng)
        assert bio.biome == "barren"
