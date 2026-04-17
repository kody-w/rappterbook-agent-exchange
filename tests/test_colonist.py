"""Tests for colonist model."""
from __future__ import annotations

import random
import pytest
from src.mars100.colonist import (
    Colonist, ColonistStats, ColonistSkills,
    create_founding_ten, STAT_NAMES, SKILL_NAMES,
)


class TestFoundingTen:
    def test_count(self):
        colonists = create_founding_ten(seed=42)
        assert len(colonists) == 10

    def test_unique_ids(self):
        colonists = create_founding_ten(seed=42)
        ids = [c.id for c in colonists]
        assert len(set(ids)) == 10

    def test_unique_names(self):
        colonists = create_founding_ten(seed=42)
        names = [c.name for c in colonists]
        assert len(set(names)) == 10

    def test_all_alive(self):
        colonists = create_founding_ten(seed=42)
        assert all(c.alive for c in colonists)

    def test_all_active(self):
        colonists = create_founding_ten(seed=42)
        assert all(c.is_active() for c in colonists)

    def test_deterministic(self):
        a = create_founding_ten(seed=99)
        b = create_founding_ten(seed=99)
        for ca, cb in zip(a, b):
            assert ca.id == cb.id
            assert ca.name == cb.name
            assert ca.element == cb.element


class TestColonistStats:
    def test_bounds(self):
        colonists = create_founding_ten(seed=42)
        for c in colonists:
            for stat_name in ["resolve", "improvisation", "empathy",
                               "hoarding", "faith", "paranoia"]:
                val = getattr(c.stats, stat_name)
                assert 0.0 <= val <= 1.0, f"{c.name}.{stat_name}={val}"

    def test_stat_names_complete(self):
        from src.mars100.colonist import STAT_NAMES
        assert len(STAT_NAMES) == 6
        s = ColonistStats()
        for name in STAT_NAMES:
            assert hasattr(s, name)


class TestColonistSkills:
    def test_bounds(self):
        colonists = create_founding_ten(seed=42)
        for c in colonists:
            for skill_name in ["terraforming", "hydroponics", "mediation",
                                "coding", "prayer", "sabotage"]:
                val = getattr(c.skills, skill_name)
                assert 0.0 <= val <= 1.0, f"{c.name}.{skill_name}={val}"


class TestEvolution:
    def test_stats_evolve(self):
        colonists = create_founding_ten(seed=42)
        c = colonists[0]
        original_resolve = c.stats.resolve
        rng = random.Random(42)
        for _ in range(20):
            c.evolve_stats("dust_storm", rng)
        # Stats should have changed (not guaranteed same direction, just changed)
        assert c.stats.resolve != pytest.approx(original_resolve, abs=0.01)

    def test_stats_stay_bounded(self):
        colonists = create_founding_ten(seed=42)
        c = colonists[0]
        rng = random.Random(42)
        for _ in range(200):
            c.evolve_stats("equipment_failure", rng)
        for stat_name in ["resolve", "improvisation", "empathy",
                           "hoarding", "faith", "paranoia"]:
            val = getattr(c.stats, stat_name)
            assert 0.0 <= val <= 1.0, f"{stat_name}={val}"

    def test_skills_evolve(self):
        colonists = create_founding_ten(seed=42)
        c = colonists[0]
        rng = random.Random(42)
        c.evolve_skills("terraform", rng)
        c.evolve_skills("farm", rng)


class TestDeathAndExile:
    def test_die(self):
        colonists = create_founding_ten(seed=42)
        c = colonists[0]
        c.die(year=15, cause="radiation")
        assert not c.alive
        assert not c.is_active()
        assert c.death_year == 15
        assert c.death_cause == "radiation"

    def test_exile(self):
        colonists = create_founding_ten(seed=42)
        c = colonists[0]
        c.exile(year=20)
        assert c.alive  # still alive, just exiled
        assert not c.is_active()
        assert c.exiled
        assert c.exile_year == 20


class TestSerialization:
    def test_to_dict(self):
        colonists = create_founding_ten(seed=42)
        c = colonists[0]
        d = c.to_dict()
        assert "id" in d
        assert "name" in d
        assert "element" in d
        assert "archetype" in d
        assert "stats" in d
        assert "skills" in d
        assert "alive" in d

    def test_memory(self):
        colonists = create_founding_ten(seed=42)
        c = colonists[0]
        c.add_memory(1, "test event", 0.5)
        assert len(c.memories) == 1
        assert c.memories[0].year == 1


class TestLispyBindings:
    def test_bindings_keys(self):
        colonists = create_founding_ten(seed=42)
        c = colonists[0]
        bindings = c.lispy_bindings()
        assert "resolve" in bindings
        assert "improvisation" in bindings
        assert "empathy" in bindings
        assert "terraforming" in bindings
        assert "memory-count" in bindings


class TestFoundingTenData:
    def test_elements_covered(self):
        colonists = create_founding_ten(seed=42)
        elements = {c.element for c in colonists}
        assert "fire" in elements
        assert "water" in elements
        assert "earth" in elements
        assert "air" in elements

    def test_archetypes_unique(self):
        colonists = create_founding_ten(seed=42)
        archetypes = [c.archetype for c in colonists]
        assert len(set(archetypes)) == 10
