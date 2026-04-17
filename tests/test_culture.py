"""Tests for the Cultural Memory organ."""
from __future__ import annotations

import random
import pytest
from src.mars100.culture import (
    CulturalMemory, Tradition, OralHistory, Martyr, Taboo, YearContext,
    evolve_culture, compute_cultural_pressure, transmit_to_child,
    MAX_TRADITIONS, MAX_ORAL_HISTORY, MAX_MARTYRS, MAX_TABOOS,
    PRESSURE_BOUND,
)
from src.mars100.engine import Mars100Engine


def _make_ctx(
    year=1, event_type="dust_storm", event_severity=0.3,
    deaths=None, exiles=None, governance_proposals=None,
    subsim_count=0, action_counts=None, resources=None,
    colonists=None,
):
    return YearContext(
        year=year, event_type=event_type, event_severity=event_severity,
        deaths=deaths or [], exiles=exiles or [],
        governance_proposals=governance_proposals or [],
        subsim_count=subsim_count,
        action_counts=action_counts or {"cooperate": 3, "hoard": 2, "innovate": 2},
        resources=resources or {"food": 0.6, "water": 0.5, "oxygen": 0.7, "materials": 0.4},
        colonists=colonists or [],
    )


class TestCulturalMemory:
    def test_initial_state_empty(self):
        cm = CulturalMemory()
        assert cm.traditions == []
        assert cm.oral_history == []
        assert cm.martyrs == []
        assert cm.taboos == []

    def test_summary_returns_counts(self):
        cm = CulturalMemory()
        cm.traditions.append(Tradition(name="X", source_year=1, source_type="t"))
        s = cm.summary()
        assert s["traditions"] == 1
        assert "X" in s["tradition_names"]

    def test_to_dict_serializes(self):
        cm = CulturalMemory()
        cm.martyrs.append(Martyr(colonist_id="c1", year=5, dominant_trait="resolve"))
        d = cm.to_dict()
        assert len(d["martyrs"]) == 1
        assert d["martyrs"][0]["colonist_id"] == "c1"


class TestEvolveCulture:
    def test_severe_event_creates_oral_history(self):
        cm = CulturalMemory()
        ctx = _make_ctx(event_severity=0.8, event_type="meteor_strike")
        evolve_culture(cm, ctx, random.Random(1))
        assert len(cm.oral_history) == 1
        assert "meteor_strike" in cm.oral_history[0].narrative

    def test_mild_event_no_oral_history(self):
        cm = CulturalMemory()
        ctx = _make_ctx(event_severity=0.3)
        evolve_culture(cm, ctx, random.Random(1))
        assert len(cm.oral_history) == 0

    def test_death_creates_martyr(self):
        cm = CulturalMemory()
        ctx = _make_ctx(deaths=[{"colonist_id": "c1", "dominant_trait": "empathy"}])
        evolve_culture(cm, ctx, random.Random(1))
        assert len(cm.martyrs) == 1
        assert cm.martyrs[0].dominant_trait == "empathy"

    def test_exile_creates_taboo(self):
        cm = CulturalMemory()
        ctx = _make_ctx(exiles=[{"colonist_id": "c2"}])
        evolve_culture(cm, ctx, random.Random(1))
        assert len(cm.taboos) == 1
        assert cm.taboos[0].action == "sabotage"

    def test_passed_governance_creates_tradition(self):
        cm = CulturalMemory()
        ctx = _make_ctx(governance_proposals=[{"passed": True, "title": "Water Law"}])
        evolve_culture(cm, ctx, random.Random(1))
        names = [t.name for t in cm.traditions]
        assert "Water Law" in names

    def test_failed_governance_no_tradition(self):
        cm = CulturalMemory()
        ctx = _make_ctx(governance_proposals=[{"passed": False, "title": "Bad Law"}])
        evolve_culture(cm, ctx, random.Random(1))
        assert len(cm.traditions) == 0

    def test_multiple_subsims_create_oracle_tradition(self):
        cm = CulturalMemory()
        ctx = _make_ctx(subsim_count=3)
        evolve_culture(cm, ctx, random.Random(1))
        names = [t.name for t in cm.traditions]
        assert "Consulting the Oracle" in names

    def test_cooperate_majority_creates_common_table(self):
        cm = CulturalMemory()
        ctx = _make_ctx(action_counts={"cooperate": 8, "hoard": 1, "innovate": 1})
        evolve_culture(cm, ctx, random.Random(1))
        names = [t.name for t in cm.traditions]
        assert "The Common Table" in names

    def test_critical_resources_create_conservation_tradition(self):
        cm = CulturalMemory()
        ctx = _make_ctx(resources={"food": 0.1, "water": 0.5, "oxygen": 0.7, "materials": 0.4})
        evolve_culture(cm, ctx, random.Random(1))
        names = [t.name for t in cm.traditions]
        assert "Conserve food" in names

    def test_tradition_importance_decays(self):
        cm = CulturalMemory()
        cm.traditions.append(Tradition(name="Old", source_year=1, source_type="test", importance=1.0))
        rng = random.Random(42)
        for y in range(2, 27):
            evolve_culture(cm, _make_ctx(year=y), rng)
        old = next(t for t in cm.traditions if t.name == "Old")
        assert old.importance < 0.95

    def test_no_duplicate_oracle_tradition(self):
        cm = CulturalMemory()
        rng = random.Random(1)
        for y in range(1, 5):
            evolve_culture(cm, _make_ctx(year=y, subsim_count=3), rng)
        assert sum(1 for t in cm.traditions if t.name == "Consulting the Oracle") == 1


class TestBounds:
    def test_traditions_bounded(self):
        cm = CulturalMemory()
        rng = random.Random(1)
        for y in range(1, 50):
            evolve_culture(cm, _make_ctx(year=y, governance_proposals=[{"passed": True, "title": "Law {}".format(y)}]), rng)
        assert len(cm.traditions) <= MAX_TRADITIONS

    def test_oral_history_bounded(self):
        cm = CulturalMemory()
        rng = random.Random(1)
        for y in range(1, 30):
            evolve_culture(cm, _make_ctx(year=y, event_severity=0.9), rng)
        assert len(cm.oral_history) <= MAX_ORAL_HISTORY

    def test_martyrs_bounded(self):
        cm = CulturalMemory()
        rng = random.Random(1)
        for y in range(1, 20):
            evolve_culture(cm, _make_ctx(year=y, deaths=[{"colonist_id": "c{}".format(y), "dominant_trait": "resolve"}]), rng)
        assert len(cm.martyrs) <= MAX_MARTYRS

    def test_taboos_bounded(self):
        cm = CulturalMemory()
        rng = random.Random(1)
        for y in range(1, 20):
            evolve_culture(cm, _make_ctx(year=y, exiles=[{"colonist_id": "c{}".format(y)}]), rng)
        assert len(cm.taboos) <= MAX_TABOOS

    def test_cultural_pressure_bounded(self):
        cm = CulturalMemory()
        for _ in range(20):
            cm.taboos.append(Taboo(action="sabotage", source_year=1, strength=1.0))
        pressure = compute_cultural_pressure(cm)
        assert abs(pressure.get("sabotage", 0)) <= PRESSURE_BOUND


class TestCulturalPressure:
    def test_empty_culture_no_pressure(self):
        assert compute_cultural_pressure(CulturalMemory()) == {}

    def test_sabotage_taboo_suppresses_sabotage(self):
        cm = CulturalMemory()
        cm.taboos.append(Taboo(action="sabotage", source_year=1, strength=1.0))
        assert compute_cultural_pressure(cm)["sabotage"] < 0

    def test_martyr_boosts_dominant_trait_action(self):
        cm = CulturalMemory()
        cm.martyrs.append(Martyr(colonist_id="c1", year=1, dominant_trait="empathy"))
        assert compute_cultural_pressure(cm).get("cooperate", 0) > 0


class TestTransmitToChild:
    def test_empty_culture_transmits_empty(self):
        assert len(transmit_to_child(CulturalMemory(), random.Random(1)).traditions) == 0

    def test_transmits_some_traditions(self):
        cm = CulturalMemory()
        for i in range(5):
            cm.traditions.append(Tradition(name="T{}".format(i), source_year=1, source_type="test", importance=1.0))
        child = transmit_to_child(cm, random.Random(42))
        assert 0 < len(child.traditions) <= 5


class TestCultureIntegration:
    def test_50_year_sim_has_culture(self):
        result = Mars100Engine(seed=42, total_years=50).run()
        d = result.to_dict()
        assert "final_culture" in d
        assert "traditions" in d["final_culture"]
        assert "oral_history" in d["final_culture"]

    def test_culture_in_year_result(self):
        result = Mars100Engine(seed=42, total_years=5).run()
        for yr in result.years:
            assert hasattr(yr, "culture")
            assert isinstance(yr.culture, dict)

    def test_culture_deterministic(self):
        r1 = Mars100Engine(seed=99, total_years=20).run().to_dict()
        r2 = Mars100Engine(seed=99, total_years=20).run().to_dict()
        assert r1["final_culture"] == r2["final_culture"]

    def test_different_seeds_different_culture(self):
        c1 = Mars100Engine(seed=1, total_years=30).run().to_dict()["final_culture"]
        c2 = Mars100Engine(seed=999, total_years=30).run().to_dict()["final_culture"]
        assert (c1["traditions"] != c2["traditions"]
                or c1["oral_history"] != c2["oral_history"]
                or c1["martyrs"] != c2["martyrs"])

    def test_full_100_year_culture_bounded(self):
        fc = Mars100Engine(seed=77, total_years=100).run().to_dict()["final_culture"]
        assert len(fc["traditions"]) <= MAX_TRADITIONS
        assert len(fc["oral_history"]) <= MAX_ORAL_HISTORY
        assert len(fc["martyrs"]) <= MAX_MARTYRS
        assert len(fc["taboos"]) <= MAX_TABOOS
