"""Tests for the Mars-100 birth system and value convergence tracking."""
from __future__ import annotations

import random

import pytest

from src.mars100.colonist import (
    Colonist, ColonistStats, ColonistSkills, STAT_NAMES, SKILL_NAMES,
    create_founding_ten, create_mars_born, blend_stats, blend_skills,
    reset_birth_counter, MARS_BORN_NAMES,
)
from src.mars100.colony import SocialGraph, Relationship
from src.mars100.engine import (
    Mars100Engine, MIN_BIRTH_POP, MIN_BIRTH_YEAR, BASE_BIRTH_RATE,
)


# ---- Fixtures ----

@pytest.fixture
def rng() -> random.Random:
    return random.Random(99)


@pytest.fixture
def parents(rng: random.Random) -> tuple[Colonist, Colonist]:
    ten = create_founding_ten(42)
    return ten[0], ten[1]


# ---- blend_stats ----

class TestBlendStats:
    def test_stats_within_bounds(self, parents: tuple, rng: random.Random) -> None:
        a, b = parents
        child = blend_stats(a.stats, b.stats, rng)
        for name in STAT_NAMES:
            val = getattr(child, name)
            assert 0.0 <= val <= 1.0, f"{name} = {val} out of bounds"

    def test_deterministic(self, parents: tuple) -> None:
        a, b = parents
        r1 = random.Random(7)
        r2 = random.Random(7)
        c1 = blend_stats(a.stats, b.stats, r1)
        c2 = blend_stats(a.stats, b.stats, r2)
        for name in STAT_NAMES:
            assert getattr(c1, name) == getattr(c2, name)

    def test_child_differs_from_parents(self, parents: tuple, rng: random.Random) -> None:
        a, b = parents
        child = blend_stats(a.stats, b.stats, rng)
        diffs_a = sum(1 for n in STAT_NAMES if getattr(child, n) != getattr(a.stats, n))
        diffs_b = sum(1 for n in STAT_NAMES if getattr(child, n) != getattr(b.stats, n))
        assert diffs_a > 0
        assert diffs_b > 0


# ---- blend_skills ----

class TestBlendSkills:
    def test_skills_within_bounds(self, parents: tuple, rng: random.Random) -> None:
        a, b = parents
        child = blend_skills(a.skills, b.skills, rng)
        for name in SKILL_NAMES:
            val = getattr(child, name)
            assert 0.0 <= val <= 1.0, f"{name} = {val} out of bounds"

    def test_skills_diminished(self, parents: tuple, rng: random.Random) -> None:
        """Children inherit at most 40% of parents' peak skill."""
        a, b = parents
        child = blend_skills(a.skills, b.skills, rng)
        for name in SKILL_NAMES:
            parent_max = max(getattr(a.skills, name), getattr(b.skills, name))
            child_val = getattr(child, name)
            # Allow some gaussian noise overshoot, but child should be << parent_max
            assert child_val < parent_max * 0.6 + 0.15, f"{name}: child={child_val} parent_max={parent_max}"


# ---- create_mars_born ----

class TestCreateMarsBorn:
    def test_returns_colonist(self, parents: tuple, rng: random.Random) -> None:
        reset_birth_counter()
        a, b = parents
        child = create_mars_born(a, b, 25, rng)
        assert isinstance(child, Colonist)

    def test_id_format(self, parents: tuple, rng: random.Random) -> None:
        reset_birth_counter()
        a, b = parents
        child = create_mars_born(a, b, 25, rng)
        assert child.id.startswith("mars-born-")

    def test_name_contains_year(self, parents: tuple, rng: random.Random) -> None:
        reset_birth_counter()
        a, b = parents
        child = create_mars_born(a, b, 25, rng)
        assert "-25" in child.name

    def test_element_from_parents(self, parents: tuple, rng: random.Random) -> None:
        reset_birth_counter()
        a, b = parents
        child = create_mars_born(a, b, 25, rng)
        assert child.element in [a.element, b.element]

    def test_incremental_ids(self, parents: tuple, rng: random.Random) -> None:
        reset_birth_counter()
        a, b = parents
        c1 = create_mars_born(a, b, 10, rng)
        c2 = create_mars_born(a, b, 15, rng)
        assert c1.id == "mars-born-1"
        assert c2.id == "mars-born-2"

    def test_names_cycle_through_list(self, parents: tuple, rng: random.Random) -> None:
        reset_birth_counter()
        a, b = parents
        names = []
        for i in range(len(MARS_BORN_NAMES) + 1):
            c = create_mars_born(a, b, 10 + i, rng)
            names.append(c.name)
        # First name wraps around after exhausting the list
        assert names[0].startswith(MARS_BORN_NAMES[0])
        assert names[len(MARS_BORN_NAMES)].startswith(MARS_BORN_NAMES[0])


# ---- SocialGraph.add_colonist ----

class TestSocialGraphAdd:
    def test_new_colonist_has_edges(self, rng: random.Random) -> None:
        sg = SocialGraph()
        sg.initialize(["a", "b", "c"], rng)
        sg.add_colonist("d", ["a", "b", "c", "d"], rng)
        for other in ["a", "b", "c"]:
            assert sg.get("d", other).trust > 0
            assert sg.get(other, "d").trust > 0

    def test_existing_edges_untouched(self, rng: random.Random) -> None:
        sg = SocialGraph()
        sg.initialize(["a", "b"], rng)
        old_trust = sg.get("a", "b").trust
        sg.add_colonist("c", ["a", "b", "c"], rng)
        assert sg.get("a", "b").trust == old_trust


# ---- Engine births ----

class TestEngineBirths:
    def test_no_births_before_min_year(self) -> None:
        engine = Mars100Engine(seed=42, total_years=MIN_BIRTH_YEAR - 1)
        result = engine.run()
        assert result.total_births == 0

    def test_births_occur_in_100_years(self) -> None:
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        assert result.total_births > 0

    def test_birth_records_have_parents(self) -> None:
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        for yr in result.years:
            for birth in yr.births:
                assert "parents" in birth
                assert len(birth["parents"]) == 2
                assert birth["parents"][0] != birth["parents"][1]

    def test_births_in_summary(self) -> None:
        engine = Mars100Engine(seed=42, total_years=100)
        d = engine.run().to_dict()
        assert "total_births" in d["summary"]
        assert d["summary"]["total_births"] >= 0

    def test_deterministic_births(self) -> None:
        r1 = Mars100Engine(seed=42, total_years=50).run()
        r2 = Mars100Engine(seed=42, total_years=50).run()
        assert r1.total_births == r2.total_births

    def test_born_colonists_appear_in_snapshots(self) -> None:
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        if result.total_births > 0:
            all_ids = {c["id"] for c in result.final_colonists}
            born_ids = set()
            for yr in result.years:
                for b in yr.births:
                    born_ids.add(b["id"])
            assert born_ids.issubset(all_ids)

    def test_small_pop_no_births(self) -> None:
        """If population drops below MIN_BIRTH_POP, no births occur."""
        engine = Mars100Engine(seed=42, total_years=5)
        # Kill colonists to get below threshold
        for c in engine.colonists[:7]:
            c.die(1, "test")
        assert len(engine._active_colonists()) == 3
        for _ in range(5):
            yr = engine.tick()
            assert yr.births == []


# ---- Value convergence ----

class TestValueConvergence:
    def test_convergence_in_year_results(self) -> None:
        engine = Mars100Engine(seed=42, total_years=20)
        result = engine.run()
        has_convergence = any(yr.value_convergence for yr in result.years)
        assert has_convergence

    def test_convergence_has_all_stats(self) -> None:
        engine = Mars100Engine(seed=42, total_years=20)
        result = engine.run()
        for yr in result.years:
            if yr.value_convergence:
                for stat in STAT_NAMES:
                    assert stat in yr.value_convergence
                    assert 0.0 <= yr.value_convergence[stat] <= 1.0

    def test_convergence_in_to_dict(self) -> None:
        engine = Mars100Engine(seed=42, total_years=20)
        d = engine.run().to_dict()
        found = False
        for yr in d["years"]:
            if "value_convergence" in yr:
                found = True
                break
        assert found

    def test_convergence_empty_with_one_colonist(self) -> None:
        engine = Mars100Engine(seed=42, total_years=3)
        for c in engine.colonists[1:]:
            c.die(0, "test")
        yr = engine.tick()
        assert yr.value_convergence == {}
