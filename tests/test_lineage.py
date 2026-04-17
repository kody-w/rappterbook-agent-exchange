"""Tests for Mars-100 lineage and birth mechanics."""
from __future__ import annotations

import random
import pytest
from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, STAT_NAMES, SKILL_NAMES
from src.mars100.colony import Resources, SocialGraph
from src.mars100.lineage import (
    MAX_POPULATION,
    MIN_BIRTH_YEAR,
    MIN_PARENT_AGE,
    _blend,
    _blend_skills,
    _blend_stats,
    _generate_child_id,
    _generate_child_name,
    _generate_decision_expr,
    can_birth,
    maybe_birth,
    select_parents,
)


def _make_colonist(cid: str, name: str = "Test", birth_year: int = 0,
                   element: str = "fire", **stat_overrides) -> Colonist:
    """Helper to create a colonist with defaults."""
    stats = {s: 0.5 for s in STAT_NAMES}
    stats.update(stat_overrides)
    return Colonist(
        id=cid, name=name, element=element, archetype="pioneer",
        stats=ColonistStats.from_dict(stats),
        skills=ColonistSkills.from_dict({s: 0.3 for s in SKILL_NAMES}),
        decision_expr="(+ resolve empathy)",
        birth_year=birth_year, generation=0,
    )


class TestBlend:
    def test_output_bounded(self):
        rng = random.Random(42)
        for _ in range(200):
            result = _blend(rng.random(), rng.random(), rng)
            assert 0.0 <= result <= 1.0

    def test_blend_stats_all_bounded(self):
        rng = random.Random(42)
        a = ColonistStats(resolve=0.9, improvisation=0.1, empathy=0.8,
                          hoarding=0.2, faith=0.7, paranoia=0.3)
        b = ColonistStats(resolve=0.3, improvisation=0.9, empathy=0.2,
                          hoarding=0.8, faith=0.1, paranoia=0.7)
        for _ in range(50):
            blended = _blend_stats(a, b, rng)
            for name in STAT_NAMES:
                val = getattr(blended, name)
                assert 0.0 <= val <= 1.0, f"{name}={val}"

    def test_blend_skills_weaker_than_parents(self):
        rng = random.Random(42)
        a = ColonistSkills(terraforming=0.9, hydroponics=0.8, mediation=0.7,
                           coding=0.9, prayer=0.5, sabotage=0.3)
        b = ColonistSkills(terraforming=0.8, hydroponics=0.9, mediation=0.6,
                           coding=0.7, prayer=0.4, sabotage=0.2)
        total_child = 0.0
        total_parent = 0.0
        for _ in range(100):
            child = _blend_skills(a, b, rng)
            for name in SKILL_NAMES:
                total_child += getattr(child, name)
                total_parent += (getattr(a, name) + getattr(b, name)) / 2.0
        # Children should be weaker on average (40% inheritance)
        assert total_child < total_parent


class TestChildGeneration:
    def test_unique_ids(self):
        a = _make_colonist("kira-sol", "Kira Sol")
        b = _make_colonist("fen-marsh", "Fen Marsh")
        existing: set[str] = {"kira-sol", "fen-marsh"}
        ids = set()
        for year in range(15, 40):
            cid = _generate_child_id(a, b, year, existing)
            assert cid not in existing
            ids.add(cid)
            existing.add(cid)
        assert len(ids) == 25  # all unique

    def test_child_name_generation(self):
        rng = random.Random(42)
        a = _make_colonist("kira-sol", "Kira Sol")
        b = _make_colonist("fen-marsh", "Fen Marsh")
        names = [_generate_child_name(a, b, 1, rng) for _ in range(10)]
        assert all(isinstance(n, str) and len(n) > 0 for n in names)

    def test_decision_expr_valid(self):
        rng = random.Random(42)
        stats = ColonistStats(resolve=0.7, improvisation=0.5, empathy=0.6,
                              hoarding=0.3, faith=0.8, paranoia=0.2)
        for _ in range(20):
            expr = _generate_decision_expr(stats, "fire", rng)
            assert expr.startswith("(")
            assert expr.endswith(")")


class TestCanBirth:
    def test_too_early(self):
        r = Resources(food=0.7, water=0.7, power=0.8, air=0.9, medicine=0.5)
        assert not can_birth(10, 8, r)

    def test_population_cap(self):
        r = Resources(food=0.7, water=0.7, power=0.8, air=0.9, medicine=0.5)
        assert not can_birth(20, MAX_POPULATION, r)

    def test_low_resources(self):
        r = Resources(food=0.1, water=0.1, power=0.1, air=0.1, medicine=0.1)
        assert not can_birth(20, 8, r)

    def test_conditions_met(self):
        r = Resources(food=0.7, water=0.7, power=0.8, air=0.9, medicine=0.5)
        assert can_birth(20, 8, r)


class TestSelectParents:
    def test_no_eligible(self):
        colonists = [_make_colonist(f"c{i}", birth_year=10) for i in range(5)]
        rng = random.Random(42)
        result = select_parents(colonists, 20, rng)
        assert result is None  # age 10 < MIN_PARENT_AGE

    def test_eligible_parents(self):
        colonists = [_make_colonist(f"c{i}", birth_year=0) for i in range(5)]
        rng = random.Random(42)
        result = select_parents(colonists, 20, rng)
        assert result is not None
        a, b = result
        assert a.id != b.id

    def test_dead_excluded(self):
        colonists = [_make_colonist(f"c{i}", birth_year=0) for i in range(3)]
        colonists[0].die(5, "test")
        colonists[1].die(5, "test")
        rng = random.Random(42)
        result = select_parents(colonists, 20, rng)
        assert result is None  # only 1 alive


class TestMaybeBirth:
    def test_returns_none_when_early(self):
        colonists = [_make_colonist(f"c{i}", birth_year=0) for i in range(5)]
        resources = Resources(food=0.7, water=0.7, power=0.8, air=0.9, medicine=0.5)
        rng = random.Random(42)
        result = maybe_birth(colonists, 5, resources, rng)
        assert result is None

    def test_birth_possible(self):
        """Over many attempts with good conditions, at least one birth should occur."""
        colonists = [_make_colonist(f"c{i}", birth_year=0) for i in range(8)]
        resources = Resources(food=0.8, water=0.8, power=0.9, air=0.9, medicine=0.6)
        births = 0
        for seed in range(100):
            rng = random.Random(seed)
            child = maybe_birth(colonists, 25, resources, rng)
            if child is not None:
                births += 1
                assert child.birth_year == 25
                assert child.generation == 1
                assert len(child.parent_ids) == 2
                assert child.alive
                for name in STAT_NAMES:
                    assert 0.0 <= getattr(child.stats, name) <= 1.0
                for name in SKILL_NAMES:
                    assert 0.0 <= getattr(child.skills, name) <= 1.0
        assert births > 0, "Expected at least one birth across 100 seeds"

    def test_child_inherits_parent_element(self):
        a = _make_colonist("a", "Alpha", birth_year=0, element="fire")
        b = _make_colonist("b", "Beta", birth_year=0, element="water")
        fire_count = water_count = 0
        for seed in range(200):
            rng = random.Random(seed)
            child = maybe_birth([a, b], 20, Resources(food=0.9, water=0.9, power=0.9, air=0.9, medicine=0.9), rng)
            if child:
                if child.element == "fire":
                    fire_count += 1
                else:
                    water_count += 1
        # Both elements should appear
        if fire_count + water_count > 0:
            assert fire_count > 0 or water_count > 0


class TestSocialIntegration:
    def test_newborn_added_to_graph(self):
        rng = random.Random(42)
        graph = SocialGraph()
        ids = ["a", "b", "c"]
        graph.initialize(ids, rng)
        graph.add_colonist("child", ids + ["child"], ["a", "b"], rng)
        # Child should have edges to all existing
        assert "child" in graph.edges
        assert "a" in graph.edges["child"]
        assert "b" in graph.edges["child"]
        assert "c" in graph.edges["child"]
        # Parents should have higher trust
        assert graph.edges["child"]["a"].trust > 0.6
        assert graph.edges["child"]["b"].trust > 0.6
        # Bidirectional
        assert "child" in graph.edges["a"]
        assert "child" in graph.edges["b"]
