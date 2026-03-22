"""Tests for Mars colony model."""
from __future__ import annotations

import random
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from mars.colony import (
    Colony,
    ColonyConfig,
    COLONY_CONFIGS,
    WATER_PER_PERSON_KG,
    FOOD_PER_PERSON_KG,
    POWER_PER_PERSON_KWH,
    HABITAT_VOLUME_PER_PERSON_M3,
)
from mars.environment import MarsEnvironment


def _make_env(sol: int = 100, latitude: float = 0.0,
              altitude_km: float = 0.0, seed: int = 42) -> MarsEnvironment:
    """Helper to create a test environment."""
    return MarsEnvironment(
        sol=sol, latitude=latitude, altitude_km=altitude_km,
        has_shielding=True, rng=random.Random(seed),
    )


class TestColonyInit:
    """Test colony initialization."""

    def test_all_configs_create(self) -> None:
        for cfg in COLONY_CONFIGS:
            colony = Colony(cfg)
            assert colony.population == cfg.initial_crew
            assert colony.population > 0

    def test_starts_with_reserves(self) -> None:
        colony = Colony(COLONY_CONFIGS[0])
        assert colony.water_kg > 0
        assert colony.food_kg > 0
        assert colony.stored_power_kwh > 0

    def test_infrastructure_proportional(self) -> None:
        colony = Colony(COLONY_CONFIGS[0])
        assert colony.habitat_volume_m3 >= colony.population * 20
        assert colony.greenhouse_area_m2 > 0
        assert colony.solar_panel_area_m2 > 0


class TestCarryingCapacity:
    """Test carrying capacity calculation."""

    def test_positive(self) -> None:
        for cfg in COLONY_CONFIGS:
            colony = Colony(cfg)
            cap = colony.carrying_capacity()
            assert cap >= 1

    def test_increases_with_infrastructure(self) -> None:
        colony = Colony(COLONY_CONFIGS[0])
        cap1 = colony.carrying_capacity()
        colony.habitat_volume_m3 *= 3
        colony.greenhouse_area_m2 *= 3
        colony.solar_panel_area_m2 *= 3
        colony.ice_miners *= 3
        colony.stored_power_kwh *= 3
        cap2 = colony.carrying_capacity()
        assert cap2 > cap1


class TestColonyTick:
    """Test single-sol colony tick."""

    def test_tick_returns_report(self) -> None:
        colony = Colony(COLONY_CONFIGS[0])
        env = _make_env(sol=1, latitude=colony.config.latitude,
                        altitude_km=colony.config.altitude_km)
        rng = random.Random(42)
        report = colony.tick(env, rng)
        assert "population" in report
        assert "births" in report
        assert "deaths" in report
        assert report["sol"] == 1

    def test_population_non_negative(self) -> None:
        """Population can never go below 0."""
        colony = Colony(COLONY_CONFIGS[0])
        for sol in range(1, 100):
            env = _make_env(sol=sol, latitude=colony.config.latitude,
                            altitude_km=colony.config.altitude_km, seed=sol)
            rng = random.Random(sol)
            colony.tick(env, rng)
            assert colony.population >= 0

    def test_history_grows(self) -> None:
        colony = Colony(COLONY_CONFIGS[0])
        for sol in range(1, 11):
            env = _make_env(sol=sol, latitude=colony.config.latitude,
                            altitude_km=colony.config.altitude_km, seed=sol)
            colony.tick(env, random.Random(sol))
        assert len(colony.history) == 10

    def test_resources_bounded(self) -> None:
        """Resources should never go negative."""
        colony = Colony(COLONY_CONFIGS[0])
        for sol in range(1, 100):
            env = _make_env(sol=sol, latitude=colony.config.latitude,
                            altitude_km=colony.config.altitude_km, seed=sol)
            colony.tick(env, random.Random(sol))
            assert colony.water_kg >= 0
            assert colony.food_kg >= 0
            assert colony.stored_power_kwh >= 0

    def test_morale_bounded(self) -> None:
        """Morale stays in [0.1, 1.0]."""
        colony = Colony(COLONY_CONFIGS[0])
        for sol in range(1, 200):
            env = _make_env(sol=sol, latitude=colony.config.latitude,
                            altitude_km=colony.config.altitude_km, seed=sol)
            colony.tick(env, random.Random(sol))
            assert 0.1 <= colony.morale <= 1.0


class TestColonySerialization:
    """Test colony serialization round-trip."""

    def test_to_dict(self) -> None:
        colony = Colony(COLONY_CONFIGS[0])
        d = colony.to_dict()
        assert d["name"] == "Ares Prime"
        assert "infrastructure" in d
        assert "resources" in d
        assert "demographics" in d
        assert "history" in d

    def test_keys_are_json_safe(self) -> None:
        """All values should be JSON-serializable."""
        import json
        colony = Colony(COLONY_CONFIGS[0])
        env = _make_env(sol=1, latitude=colony.config.latitude,
                        altitude_km=colony.config.altitude_km)
        colony.tick(env, random.Random(42))
        d = colony.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 100


class TestDustStormImpact:
    """Test that dust storms meaningfully impact colonies."""

    def test_storm_reduces_power(self) -> None:
        """During a dust storm, power generation should drop."""
        colony1 = Colony(COLONY_CONFIGS[0])
        colony2 = Colony(COLONY_CONFIGS[0])

        # Normal conditions
        env_clear = _make_env(sol=100, latitude=colony1.config.latitude,
                              altitude_km=colony1.config.altitude_km, seed=42)
        # Force clear weather
        env_clear.tau = 0.3
        env_clear.dust_storm = False
        env_clear.irradiance = 200.0
        report1 = colony1.tick(env_clear, random.Random(42))

        # Dust storm
        env_storm = _make_env(sol=100, latitude=colony2.config.latitude,
                              altitude_km=colony2.config.altitude_km, seed=42)
        env_storm.tau = 6.0
        env_storm.dust_storm = True
        env_storm.irradiance = 30.0
        report2 = colony2.tick(env_storm, random.Random(42))

        assert report2["power_kwh"] < report1["power_kwh"]


class TestThreeColonies365Sols:
    """Integration test: run all 3 colonies for 365 sols."""

    def test_smoke_365(self) -> None:
        """All 3 colonies survive 365 sols without crash."""
        colonies = [Colony(cfg) for cfg in COLONY_CONFIGS]
        for sol in range(1, 366):
            for colony in colonies:
                env = MarsEnvironment(
                    sol=sol, latitude=colony.config.latitude,
                    altitude_km=colony.config.altitude_km,
                    has_shielding=colony.has_shielding,
                    rng=random.Random(42 * 1000000 + sol),
                )
                colony.tick(env, random.Random(42 * 1000000 + sol + 999))

        # At least some colonies should survive
        total = sum(c.population for c in colonies)
        assert total > 0, "All colonies died — model needs tuning"

        # History should have 365 entries each
        for colony in colonies:
            assert len(colony.history) == 365

    def test_population_growth_trend(self) -> None:
        """Over 365 sols, total population should generally increase
        (infrastructure expands, immigration happens)."""
        colonies = [Colony(cfg) for cfg in COLONY_CONFIGS]
        initial_pop = sum(c.population for c in colonies)

        for sol in range(1, 366):
            for colony in colonies:
                env = MarsEnvironment(
                    sol=sol, latitude=colony.config.latitude,
                    altitude_km=colony.config.altitude_km,
                    has_shielding=colony.has_shielding,
                    rng=random.Random(42 * 1000000 + sol),
                )
                colony.tick(env, random.Random(42 * 1000000 + sol + 999))

        final_pop = sum(c.population for c in colonies)
        # Immigration alone adds ~6 per 260 sols, so we expect growth
        assert final_pop >= initial_pop, (
            f"Population declined: {initial_pop} → {final_pop}"
        )
