"""Tests for Mars-100 birth system."""
from __future__ import annotations
import random, sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars100.births import (
    can_birth, blend_stats, blend_skills, create_mars_born,
    maybe_birth, reset_birth_counter, MARS_NAMES, ARCHETYPES_BORN,
)
from src.mars100.colonist import (
    Colonist, ColonistStats, ColonistSkills, create_founding_ten,
    STAT_NAMES, SKILL_NAMES, ELEMENTS,
)


@pytest.fixture(autouse=True)
def reset_counter():
    reset_birth_counter()
    yield
    reset_birth_counter()


def _make_colonist(cid: str, element: str = "fire") -> Colonist:
    return Colonist(
        id=cid, name=cid.title(), element=element, archetype="test",
        stats=ColonistStats(resolve=0.6, improvisation=0.5, empathy=0.7,
                            hoarding=0.3, faith=0.4, paranoia=0.2),
        skills=ColonistSkills(terraforming=0.5, hydroponics=0.6,
                              mediation=0.3, coding=0.4, prayer=0.2, sabotage=0.1),
        decision_expr="(+ resolve empathy)",
    )


class TestCanBirth:
    def test_too_early(self):
        rng = random.Random(42)
        assert not can_birth(year=5, active_count=8, resources_avg=0.7, rng=rng)

    def test_too_few_colonists(self):
        rng = random.Random(42)
        assert not can_birth(year=20, active_count=3, resources_avg=0.7, rng=rng)

    def test_low_resources(self):
        rng = random.Random(42)
        assert not can_birth(year=20, active_count=8, resources_avg=0.2, rng=rng)

    def test_conditions_met_probabilistic(self):
        births = sum(
            can_birth(year=40, active_count=8, resources_avg=0.7, rng=random.Random(i))
            for i in range(100)
        )
        assert 5 < births < 50  # should see some births


class TestBlendStats:
    def test_blend_is_between_parents(self):
        rng = random.Random(42)
        a = ColonistStats(resolve=0.2, improvisation=0.8, empathy=0.5,
                          hoarding=0.3, faith=0.9, paranoia=0.1)
        b = ColonistStats(resolve=0.8, improvisation=0.2, empathy=0.5,
                          hoarding=0.7, faith=0.1, paranoia=0.9)
        child = blend_stats(a, b, rng)
        for name in STAT_NAMES:
            val = getattr(child, name)
            assert 0.0 <= val <= 1.0, f"{name} out of range: {val}"

    def test_blend_is_deterministic(self):
        a = ColonistStats(resolve=0.5, improvisation=0.5, empathy=0.5,
                          hoarding=0.5, faith=0.5, paranoia=0.5)
        b = ColonistStats(resolve=0.5, improvisation=0.5, empathy=0.5,
                          hoarding=0.5, faith=0.5, paranoia=0.5)
        r1 = blend_stats(a, b, random.Random(42))
        r2 = blend_stats(a, b, random.Random(42))
        for name in STAT_NAMES:
            assert getattr(r1, name) == getattr(r2, name)


class TestBlendSkills:
    def test_offspring_lower_than_parents(self):
        rng = random.Random(42)
        a = ColonistSkills(terraforming=0.8, hydroponics=0.9, mediation=0.7,
                           coding=0.8, prayer=0.6, sabotage=0.5)
        b = ColonistSkills(terraforming=0.7, hydroponics=0.8, mediation=0.6,
                           coding=0.7, prayer=0.5, sabotage=0.4)
        child = blend_skills(a, b, rng)
        for name in SKILL_NAMES:
            parent_avg = (getattr(a, name) + getattr(b, name)) / 2.0
            assert getattr(child, name) < parent_avg  # 30% of average

    def test_skills_in_range(self):
        rng = random.Random(42)
        a = ColonistSkills()
        b = ColonistSkills()
        child = blend_skills(a, b, rng)
        for name in SKILL_NAMES:
            assert 0.0 <= getattr(child, name) <= 1.0


class TestCreateMarsBorn:
    def test_creates_valid_colonist(self):
        rng = random.Random(42)
        parent_a = _make_colonist("a", "fire")
        parent_b = _make_colonist("b", "water")
        child = create_mars_born(year=20, parent_a=parent_a, parent_b=parent_b, rng=rng)
        assert child.id.startswith("mars-born-")
        assert child.name in MARS_NAMES
        assert child.element in ELEMENTS
        assert child.archetype in ARCHETYPES_BORN
        assert child.alive
        assert not child.exiled

    def test_unique_ids(self):
        rng = random.Random(42)
        pa = _make_colonist("a")
        pb = _make_colonist("b")
        c1 = create_mars_born(20, pa, pb, rng)
        c2 = create_mars_born(25, pa, pb, rng)
        assert c1.id != c2.id


class TestMaybeBirth:
    def test_no_birth_too_early(self):
        colonists = create_founding_ten(42)
        rng = random.Random(42)
        result = maybe_birth(year=5, colonists=colonists, resources_avg=0.7, rng=rng)
        assert result is None

    def test_birth_happens_eventually(self):
        colonists = create_founding_ten(42)
        births = 0
        for i in range(200):
            reset_birth_counter()
            rng = random.Random(i)
            result = maybe_birth(year=40, colonists=colonists, resources_avg=0.7, rng=rng)
            if result is not None:
                births += 1
                assert isinstance(result, Colonist)
        assert births > 0  # at least some births

    def test_no_birth_with_one_colonist(self):
        colonist = _make_colonist("solo")
        rng = random.Random(42)
        result = maybe_birth(year=40, colonists=[colonist], resources_avg=0.7, rng=rng)
        assert result is None
