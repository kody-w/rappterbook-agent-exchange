"""Tests for the Mars-100 ecology engine."""
from __future__ import annotations

import random
import pytest

from src.mars100.ecology import (
    EcologyState,
    compute_ecology_deltas,
    tick_ecology,
    compute_ecology_modifiers,
    ecology_stress,
    ATMOSPHERE_BREATHABLE,
    BIODIVERSITY_FLOURISH,
    POLLUTION_TOXIC,
    RADIATION_LETHAL,
    TERRAFORM_HABITABLE,
    TERRAFORM_EDEN,
)


class TestEcologyState:
    def test_defaults(self):
        s = EcologyState()
        assert s.terraforming == 0.0
        assert s.pollution == 0.0
        assert s.atmosphere == 0.1
        assert s.biodiversity == 0.0
        assert s.radiation == 0.5
        assert not s.tp_atmosphere_breathable
        assert not s.tp_terraform_eden

    def test_to_dict_roundtrip(self):
        s = EcologyState(terraforming=0.12345)
        d = s.to_dict()
        assert d["terraforming"] == 0.1235
        assert isinstance(d["tipping_points"], dict)
        assert len(d["tipping_points"]) == 6


class TestComputeEcologyDeltas:
    def test_empty_produces_no_deltas(self):
        d = compute_ecology_deltas([], [], [], {})
        assert d == {}

    def test_terraform_action(self):
        actions = [{"action": "terraform"}]
        d = compute_ecology_deltas(actions, [], [], {})
        assert d["terraforming"] == pytest.approx(0.02)
        assert d["atmosphere"] == pytest.approx(0.008)

    def test_hydroponics_biodiversity(self):
        actions = [{"action": "hydroponics"}, {"action": "hydroponics"}]
        d = compute_ecology_deltas(actions, [], [], {})
        assert d["biodiversity"] == pytest.approx(0.03)

    def test_sabotage_pollution(self):
        actions = [{"action": "sabotage"}]
        d = compute_ecology_deltas(actions, [], [], {})
        assert d["pollution"] == pytest.approx(0.05)

    def test_tech_industrial_pollution(self):
        d = compute_ecology_deltas([], [], ["solar_array", "water_recycler"], {})
        assert "pollution" in d
        assert d["pollution"] == pytest.approx(2 * 0.003)

    def test_atmospheric_processor_bonus(self):
        d = compute_ecology_deltas([], [], ["atmospheric_processor"], {})
        assert d["atmosphere"] == pytest.approx(0.015)
        assert d["radiation"] == pytest.approx(-0.01)
        assert d["pollution"] == pytest.approx(0.003)

    def test_dust_storm_effects(self):
        d = compute_ecology_deltas([], [], [], {"dust_severity": 0.5})
        assert d["radiation"] == pytest.approx(0.01)
        assert d["atmosphere"] == pytest.approx(-0.005)

    def test_multiple_actions_combine(self):
        actions = [{"action": "terraform"}, {"action": "sabotage"},
                   {"action": "hydroponics"}]
        d = compute_ecology_deltas(actions, [], ["solar_array"], {})
        assert "terraforming" in d
        assert "pollution" in d
        assert "biodiversity" in d


class TestTickEcology:
    def test_natural_pollution_decay(self):
        s = EcologyState(pollution=0.1)
        tick_ecology(s, {})
        assert s.pollution < 0.1

    def test_natural_radiation_decay(self):
        s = EcologyState(radiation=0.5)
        tick_ecology(s, {})
        assert s.radiation < 0.5

    def test_values_clamped_to_bounds(self):
        s = EcologyState(pollution=0.002)
        tick_ecology(s, {"pollution": -0.5})
        assert s.pollution >= 0.0

        s2 = EcologyState(terraforming=0.99)
        tick_ecology(s2, {"terraforming": 0.5})
        assert s2.terraforming <= 1.0

    def test_tipping_point_fires_once(self):
        s = EcologyState(atmosphere=ATMOSPHERE_BREATHABLE - 0.01)
        events = tick_ecology(s, {"atmosphere": 0.02})
        tp_names = [e["name"] for e in events]
        assert "atmosphere_breathable" in tp_names

        events2 = tick_ecology(s, {})
        tp_names2 = [e["name"] for e in events2]
        assert "atmosphere_breathable" not in tp_names2
        assert s.tp_atmosphere_breathable

    def test_pollution_toxic_tipping(self):
        s = EcologyState(pollution=POLLUTION_TOXIC - 0.01)
        events = tick_ecology(s, {"pollution": 0.02})
        assert any(e["name"] == "pollution_toxic" for e in events)

    def test_terraform_habitable_tipping(self):
        s = EcologyState(terraforming=TERRAFORM_HABITABLE - 0.01)
        events = tick_ecology(s, {"terraforming": 0.02})
        assert any(e["name"] == "terraform_habitable" for e in events)

    def test_biodiversity_self_reinforcement(self):
        s = EcologyState(biodiversity=0.15)
        tick_ecology(s, {})
        assert s.biodiversity == pytest.approx(0.153, abs=0.001)

    def test_100_years_no_crash(self):
        s = EcologyState()
        rng = random.Random(42)
        for _ in range(100):
            deltas = {"terraforming": rng.uniform(0, 0.02),
                      "pollution": rng.uniform(-0.01, 0.02)}
            tick_ecology(s, deltas, rng)
        for attr in ("terraforming", "pollution", "atmosphere",
                     "biodiversity", "radiation"):
            v = getattr(s, attr)
            assert 0.0 <= v <= 1.0, f"{attr}={v} out of bounds"


class TestComputeEcologyModifiers:
    def test_baseline_modifiers(self):
        s = EcologyState()
        m = compute_ecology_modifiers(s)
        assert m["death_rate_mult"] == pytest.approx(1.0, abs=0.01)
        assert "food_bonus" not in m
        assert "water_bonus" not in m

    def test_terraform_water_bonus(self):
        s = EcologyState(terraforming=0.5)
        m = compute_ecology_modifiers(s)
        assert m["water_bonus"] > 0

    def test_terraform_air_bonus(self):
        s = EcologyState(terraforming=0.7)
        m = compute_ecology_modifiers(s)
        assert m["air_bonus"] > 0
        assert m["water_bonus"] > 0

    def test_biodiversity_food_bonus(self):
        s = EcologyState(biodiversity=0.4)
        m = compute_ecology_modifiers(s)
        assert m["food_bonus"] > 0

    def test_pollution_penalties(self):
        s = EcologyState(pollution=0.8)
        m = compute_ecology_modifiers(s)
        assert m["food_penalty"] < 0
        assert m["water_penalty"] < 0

    def test_pollution_death_rate(self):
        s = EcologyState(pollution=0.8)
        m = compute_ecology_modifiers(s)
        assert m["death_rate_mult"] > 1.0

    def test_radiation_death_rate(self):
        s = EcologyState(radiation=0.9)
        m = compute_ecology_modifiers(s)
        assert m["death_rate_mult"] > 1.0

    def test_event_severity_bounded(self):
        for _ in range(50):
            s = EcologyState(
                atmosphere=random.random(),
                pollution=random.random(),
            )
            m = compute_ecology_modifiers(s)
            assert 0.5 <= m["event_severity_mult"] <= 1.5


class TestEcologyStress:
    def test_default_state_stress(self):
        s = EcologyState()
        stress = ecology_stress(s)
        assert 0.0 <= stress <= 1.0

    def test_high_pollution_high_stress(self):
        s = EcologyState(pollution=0.9, radiation=0.8)
        stress = ecology_stress(s)
        assert stress > 0.5

    def test_terraform_reduces_stress(self):
        base = EcologyState(pollution=0.5, radiation=0.5)
        improved = EcologyState(pollution=0.5, radiation=0.5,
                                terraforming=0.8, biodiversity=0.5)
        assert ecology_stress(improved) < ecology_stress(base)


class TestEcologyEngineIntegration:
    def test_engine_has_ecology_state(self):
        from src.mars100.engine import Mars100Engine
        eng = Mars100Engine(seed=42)
        assert hasattr(eng, "ecology")
        assert isinstance(eng.ecology, EcologyState)

    def test_10_year_ecology_evolves(self):
        from src.mars100.engine import Mars100Engine
        eng = Mars100Engine(seed=42)
        result = eng.run()
        d = result.to_dict()
        assert "ecology" in d
        eco = d["ecology"]
        assert eco["terraforming"] >= 0.0
        assert eco["radiation"] <= 0.7  # may spike from dust storms

    def test_year_result_has_ecology(self):
        from src.mars100.engine import Mars100Engine
        eng = Mars100Engine(seed=42)
        result = eng.run()
        for yr in result.years:
            yd = yr.to_dict()
            assert "ecology" in yd
            assert "ecology_events" in yd
            assert isinstance(yd["ecology_events"], list)

    def test_50_year_no_crash(self):
        from src.mars100.engine import Mars100Engine
        eng = Mars100Engine(seed=123)
        result = eng.run()
        assert len(result.years) >= 10
        eco = result.ecology
        assert isinstance(eco, dict)

    def test_version_is_5(self):
        from src.mars100.engine import Mars100Engine
        eng = Mars100Engine(seed=42)
        result = eng.run()
        d = result.to_dict()
        assert d["_meta"]["version"] == "5.0"
