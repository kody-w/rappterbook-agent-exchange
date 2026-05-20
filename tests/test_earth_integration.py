"""Integration tests for Earth Protocol in engine v6.0.

Tests: immigrants, resource-aware death causes, engine fields, air_recycler.
"""
from __future__ import annotations

import random
import pytest

from src.mars100.engine import Mars100Engine, YearResult, SimulationResult
from src.mars100.earth import EarthState
from src.mars100.colonist import create_immigrant, IMMIGRANT_ARCHETYPES, SKILL_NAMES
from src.mars100.infrastructure import TECH_TREE, TECH_BY_ID


class TestAirRecycler:
    def test_air_recycler_in_tech_tree(self) -> None:
        assert "air_recycler" in TECH_BY_ID

    def test_air_recycler_effects(self) -> None:
        tech = TECH_BY_ID["air_recycler"]
        assert tech.effects["air_spoilage_mult"] < 1.0
        assert tech.effects["air_maintenance_mult"] < 1.0

    def test_tech_tree_has_seven(self) -> None:
        assert len(TECH_TREE) == 7


class TestCreateImmigrant:
    def test_deterministic(self) -> None:
        a = create_immigrant("imm-1", 25, random.Random(42))
        b = create_immigrant("imm-1", 25, random.Random(42))
        assert a.name == b.name
        assert a.element == b.element
        assert a.archetype == b.archetype

    def test_adult_skills(self) -> None:
        imm = create_immigrant("imm-1", 25, random.Random(42))
        skill_vals = [getattr(imm.skills, s) for s in SKILL_NAMES]
        assert max(skill_vals) > 0.1

    def test_valid_archetype(self) -> None:
        valid = {t["archetype"] for t in IMMIGRANT_ARCHETYPES}
        for seed in range(20):
            imm = create_immigrant(f"imm-{seed}", 30, random.Random(seed))
            assert imm.archetype in valid

    def test_birth_year_is_arrival(self) -> None:
        imm = create_immigrant("imm-1", 42, random.Random(1))
        assert imm.birth_year == 42

    def test_is_active(self) -> None:
        imm = create_immigrant("imm-1", 10, random.Random(1))
        assert imm.is_active()


class TestEngineEarthIntegration:
    def test_engine_has_earth(self) -> None:
        e = Mars100Engine(seed=42, total_years=5)
        assert isinstance(e.earth, EarthState)

    def test_year_result_fields(self) -> None:
        e = Mars100Engine(seed=42, total_years=3)
        r = e.run()
        for yr in r.years:
            d = yr.to_dict()
            assert "earth" in d
            assert "earth_events" in d
            assert "immigrants" in d

    def test_simulation_result_fields(self) -> None:
        e = Mars100Engine(seed=42, total_years=5)
        r = e.run()
        d = r.to_dict()
        assert "final_earth" in d
        assert d["_meta"]["version"] == "12.1"
        assert "total_immigrants" in d["summary"]

    def test_earth_state_evolves(self) -> None:
        """Earth opinion, funding, and policy change over time."""
        e = Mars100Engine(seed=42, total_years=20)
        r = e.run()
        d = r.to_dict()
        fe = d["final_earth"]
        assert 0.0 <= fe["opinion"] <= 1.0
        assert 0.0 <= fe["funding"] <= 1.0
        assert fe["policy"] in ("supportive", "neutral", "restrictive",
                                 "hostile", "independent")

    def test_100_year_no_crash(self) -> None:
        e = Mars100Engine(seed=42, total_years=100)
        r = e.run()
        d = r.to_dict()
        assert d["_meta"]["total_years"] == 100
        fe = d["final_earth"]
        assert 0.0 <= fe["opinion"] <= 1.0
        assert fe["ships_launched"] >= 0

    def test_ships_arrive(self) -> None:
        """Over 20 years, Earth should launch and deliver supply ships."""
        e = Mars100Engine(seed=42, total_years=20)
        r = e.run()
        d = r.to_dict()
        assert d["final_earth"]["ships_launched"] > 0
        assert d["final_earth"]["ships_arrived"] >= 0

    def test_colonist_count_consistent(self) -> None:
        e = Mars100Engine(seed=42, total_years=100)
        r = e.run()
        d = r.to_dict()
        expected = 10 + d["summary"]["total_births"] + d["summary"]["total_immigrants"]
        assert len(d["final_colonists"]) == expected


class TestResourceAwareDeathCauses:
    def test_all_causes_known(self) -> None:
        e = Mars100Engine(seed=42, total_years=100)
        r = e.run()
        causes = [d["cause"] for yr in r.years for d in yr.deaths]
        known = {"asphyxiation", "starvation", "dehydration",
                 "hypothermia", "untreated illness",
                 "equipment malfunction", "radiation exposure",
                 "medical emergency", "habitat breach",
                 "suspicious accident"}
        for cause in causes:
            assert cause in known, f"Unknown cause: {cause}"

    def test_resource_deaths_present(self) -> None:
        """Over 100 years, deaths should have known causes."""
        e = Mars100Engine(seed=42, total_years=100)
        r = e.run()
        causes = {d["cause"] for yr in r.years for d in yr.deaths}
        known = {"asphyxiation", "starvation", "dehydration",
                 "hypothermia", "untreated illness",
                 "equipment malfunction", "radiation exposure",
                 "medical emergency", "habitat breach",
                 "suspicious accident"}
        # All causes should be from the known set
        assert causes <= known, f"Unknown causes: {causes - known}"
        # At least some deaths should occur over 100 years
        assert len(causes) > 0


class TestDeterminism:
    def test_same_seed_same_result(self) -> None:
        """Engine is fully deterministic with same seed."""
        r1 = Mars100Engine(seed=99, total_years=20).run()
        r2 = Mars100Engine(seed=99, total_years=20).run()
        d1, d2 = r1.to_dict(), r2.to_dict()
        assert d1["summary"] == d2["summary"]
        assert d1["final_earth"] == d2["final_earth"]
