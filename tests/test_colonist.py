"""Tests for the colonist model."""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.colonist import (
    Colonist, create_colony, STATS, SKILLS, ELEMENTS, MEMORY_CAP,
)


class TestColonistCreation:
    def test_create_colony_returns_10(self):
        colony = create_colony(seed=42)
        assert len(colony) == 10

    def test_unique_ids(self):
        colony = create_colony(seed=42)
        ids = [c.id for c in colony]
        assert len(set(ids)) == 10

    def test_all_elements_represented(self):
        colony = create_colony(seed=42)
        elements = {c.element for c in colony}
        assert elements == set(ELEMENTS)

    def test_stats_bounded(self):
        colony = create_colony(seed=42)
        for c in colony:
            for stat_name in STATS:
                val = c.stat(stat_name)
                assert 0.0 <= val <= 1.0, f"{c.id}.{stat_name} = {val}"

    def test_skills_bounded(self):
        colony = create_colony(seed=42)
        for c in colony:
            for skill_name in SKILLS:
                val = c.skill(skill_name)
                assert 0.0 <= val <= 1.0, f"{c.id}.{skill_name} = {val}"

    def test_relationships_initialized(self):
        colony = create_colony(seed=42)
        for c in colony:
            assert len(c.relationships) == 9  # 10 - self

    def test_relationships_bounded(self):
        colony = create_colony(seed=42)
        for c in colony:
            for other_id, trust in c.relationships.items():
                assert -1.0 <= trust <= 1.0

    def test_all_alive(self):
        colony = create_colony(seed=42)
        assert all(c.alive for c in colony)

    def test_deterministic(self):
        """Same seed produces same colony."""
        c1 = create_colony(seed=99)
        c2 = create_colony(seed=99)
        for a, b in zip(c1, c2):
            assert a.id == b.id
            assert a.stats == b.stats

    def test_different_seeds(self):
        """Different seeds produce different colonies."""
        c1 = create_colony(seed=1)
        c2 = create_colony(seed=2)
        # Stats should differ
        assert c1[0].stats != c2[0].stats


class TestColonistBehavior:
    def test_stat_clamping(self):
        c = create_colony(seed=42)[0]
        c.adjust_stat("resolve", 10.0)
        assert c.stat("resolve") == 1.0
        c.adjust_stat("resolve", -20.0)
        assert c.stat("resolve") == 0.0

    def test_skill_clamping(self):
        c = create_colony(seed=42)[0]
        c.adjust_skill("terraforming", 10.0)
        assert c.skill("terraforming") == 1.0

    def test_trust_clamping(self):
        c = create_colony(seed=42)[0]
        other_id = list(c.relationships.keys())[0]
        c.adjust_trust(other_id, 10.0)
        assert c.trust(other_id) == 1.0
        c.adjust_trust(other_id, -20.0)
        assert c.trust(other_id) == -1.0

    def test_memory_add(self):
        c = create_colony(seed=42)[0]
        initial = len(c.memory)
        c.add_memory(1, "test event", 0.5)
        assert len(c.memory) == initial + 1

    def test_memory_cap(self):
        c = create_colony(seed=42)[0]
        for i in range(MEMORY_CAP + 20):
            c.add_memory(i, f"event {i}", significance=float(i) / 100)
        assert len(c.memory) <= MEMORY_CAP

    def test_memory_evicts_least_significant(self):
        c = Colonist(
            id="test", name="Test", element="fire",
            stats={s: 0.5 for s in STATS},
            skills={s: 0.5 for s in SKILLS},
            relationships={}, memory=[],
        )
        for i in range(MEMORY_CAP + 5):
            c.add_memory(i, f"event {i}", significance=float(i) / 100)
        significances = [m["significance"] for m in c.memory]
        # Least significant should have been evicted
        assert min(significances) >= 0.04  # lowest remaining > lowest added

    def test_death(self):
        c = create_colony(seed=42)[0]
        c.die(50, "starvation")
        assert not c.alive
        assert c.year_died == 50
        assert c.cause_of_death == "starvation"

    def test_effectiveness_bounded(self):
        c = create_colony(seed=42)[0]
        eff = c.effectiveness()
        assert 0.0 <= eff <= 1.0

    def test_cooperation_bounded(self):
        c = create_colony(seed=42)[0]
        coop = c.cooperation_tendency()
        assert 0.0 <= coop <= 1.0

    def test_discovery_potential_increases_with_year(self):
        c = create_colony(seed=42)[0]
        early = c.discovery_potential(5)
        late = c.discovery_potential(80)
        assert late > early


class TestSerialization:
    def test_to_dict_round_trip(self):
        colony = create_colony(seed=42)
        for original in colony:
            data = original.to_dict()
            restored = Colonist.from_dict(data)
            assert restored.id == original.id
            assert restored.name == original.name
            assert restored.element == original.element
            assert restored.alive == original.alive

    def test_to_dict_stats_rounded(self):
        c = create_colony(seed=42)[0]
        data = c.to_dict()
        for val in data["stats"].values():
            # Should be rounded to 4 decimal places
            assert val == round(val, 4)

    def test_from_dict_clamps_values(self):
        data = {
            "id": "test", "name": "Test", "element": "fire",
            "stats": {"resolve": 5.0, "empathy": -2.0},
            "skills": {"coding": 10.0},
            "relationships": {"other": 5.0},
        }
        c = Colonist.from_dict(data)
        assert c.stat("resolve") == 1.0
        assert c.stat("empathy") == 0.0
        assert c.skill("coding") == 1.0
        assert c.trust("other") == 1.0

    def test_to_view_compact(self):
        c = create_colony(seed=42)[0]
        view = c.to_view()
        assert "id" in view
        assert "element" in view
        assert "resolve" in view
        assert "memory" not in view  # view is compact
