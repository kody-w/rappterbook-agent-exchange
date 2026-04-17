"""Tests for Mars-100 birth system."""
from __future__ import annotations

import random
import pytest
from src.mars100.births import (
    attempt_birth, find_eligible_pair,
    _inherit_stat, _inherit_stats, _inherit_skills,
    BIRTH_MIN_YEAR, BIRTH_COOLDOWN, MAX_BIRTHS,
)
from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, STAT_NAMES, SKILL_NAMES
from src.mars100.colony import SocialGraph, Relationship


def _make_colonist(cid: str, seed: int = 42) -> Colonist:
    rng = random.Random(seed)
    return Colonist(
        id=cid, name=f"Test-{cid}", element="fire", archetype="engineer",
        stats=ColonistStats(**{s: rng.random() for s in STAT_NAMES}),
        skills=ColonistSkills(**{s: rng.random() for s in SKILL_NAMES}),
        decision_expr="(+ resolve empathy)",
    )


def _make_colony(n: int = 5, seed: int = 42) -> tuple[list[Colonist], SocialGraph]:
    rng = random.Random(seed)
    colonists = [_make_colonist(f"c{i}", seed=seed + i) for i in range(n)]
    social = SocialGraph()
    ids = [c.id for c in colonists]
    social.initialize(ids, rng)
    # Set high trust to ensure pairing is possible
    for a in ids:
        for b in ids:
            if a != b:
                social.edges[a][b].trust = 0.8
    return colonists, social


class TestInheritStat:
    def test_blends_parents(self) -> None:
        rng = random.Random(42)
        val = _inherit_stat(0.2, 0.8, rng)
        assert 0.0 <= val <= 1.0

    def test_clamped(self) -> None:
        rng = random.Random(42)
        for _ in range(100):
            val = _inherit_stat(1.0, 1.0, rng)
            assert 0.0 <= val <= 1.0

    def test_deterministic(self) -> None:
        v1 = _inherit_stat(0.5, 0.5, random.Random(99))
        v2 = _inherit_stat(0.5, 0.5, random.Random(99))
        assert v1 == v2


class TestInheritStats:
    def test_all_stats_present(self) -> None:
        a = ColonistStats(**{s: 0.3 for s in STAT_NAMES})
        b = ColonistStats(**{s: 0.7 for s in STAT_NAMES})
        child = _inherit_stats(a, b, random.Random(42))
        for s in STAT_NAMES:
            assert hasattr(child, s)
            assert 0.0 <= getattr(child, s) <= 1.0

    def test_blend_near_midpoint(self) -> None:
        a = ColonistStats(**{s: 0.0 for s in STAT_NAMES})
        b = ColonistStats(**{s: 1.0 for s in STAT_NAMES})
        rng = random.Random(42)
        results = []
        for _ in range(50):
            child = _inherit_stats(a, b, rng)
            results.append(child.resolve)
        avg = sum(results) / len(results)
        assert 0.3 < avg < 0.7  # should be near 0.5


class TestInheritSkills:
    def test_all_skills_present(self) -> None:
        a = ColonistSkills(**{s: 0.5 for s in SKILL_NAMES})
        b = ColonistSkills(**{s: 0.5 for s in SKILL_NAMES})
        child = _inherit_skills(a, b, random.Random(42))
        for s in SKILL_NAMES:
            assert hasattr(child, s)
            assert 0.0 <= getattr(child, s) <= 1.0


class TestFindEligiblePair:
    def test_finds_pair_with_high_trust(self) -> None:
        colonists, social = _make_colony(5)
        pair = find_eligible_pair(colonists, social, year=20, rng=random.Random(42))
        assert pair is not None
        assert len(pair) == 2

    def test_no_pair_if_all_dead(self) -> None:
        colonists, social = _make_colony(5)
        for c in colonists:
            c.alive = False
        pair = find_eligible_pair(colonists, social, year=20, rng=random.Random(42))
        assert pair is None

    def test_cooldown_respected(self) -> None:
        colonists, social = _make_colony(2)
        for c in colonists:
            c.last_birth_year = 18
        pair = find_eligible_pair(colonists, social, year=20, rng=random.Random(42))
        assert pair is None  # within cooldown

    def test_cooldown_expired(self) -> None:
        colonists, social = _make_colony(4)
        for c in colonists:
            c.last_birth_year = 10
        pair = find_eligible_pair(colonists, social, year=20, rng=random.Random(42))
        assert pair is not None


class TestAttemptBirth:
    def test_no_birth_before_min_year(self) -> None:
        colonists, social = _make_colony(5)
        result = attempt_birth(colonists, social, year=5,
                               total_births=0, rng=random.Random(42))
        assert result is None

    def test_no_birth_at_max(self) -> None:
        colonists, social = _make_colony(5)
        result = attempt_birth(colonists, social, year=20,
                               total_births=MAX_BIRTHS, rng=random.Random(42))
        assert result is None

    def test_no_birth_too_few_colonists(self) -> None:
        colonists, social = _make_colony(2)
        result = attempt_birth(colonists, social, year=20,
                               total_births=0, rng=random.Random(42))
        assert result is None

    def test_birth_produces_child(self) -> None:
        """Run many attempts to get at least one birth."""
        for seed in range(100):
            colonists, social = _make_colony(6, seed=seed)
            result = attempt_birth(colonists, social, year=25,
                                   total_births=0, rng=random.Random(seed))
            if result is not None:
                assert "child_id" in result
                assert "parent_a" in result
                assert "parent_b" in result
                assert result["year"] == 25
                # Child should be appended to colonists list
                assert len(colonists) == 7
                child = colonists[-1]
                assert child.id == result["child_id"]
                assert child.is_active()
                return
        pytest.fail("No birth occurred in 100 attempts")

    def test_child_has_memories(self) -> None:
        for seed in range(100):
            colonists, social = _make_colony(6, seed=seed)
            result = attempt_birth(colonists, social, year=30,
                                   total_births=0, rng=random.Random(seed))
            if result is not None:
                child = colonists[-1]
                assert len(child.memories) >= 1
                assert "Born" in child.memories[0].event
                return
        pytest.fail("No birth for memory test")

    def test_birth_updates_social_graph(self) -> None:
        for seed in range(100):
            colonists, social = _make_colony(6, seed=seed)
            original_edges = len(social.edges)
            result = attempt_birth(colonists, social, year=25,
                                   total_births=0, rng=random.Random(seed))
            if result is not None:
                child = colonists[-1]
                assert child.id in social.edges
                return
        pytest.fail("No birth for social graph test")


class TestPhysicalBounds:
    def test_inherited_stats_bounded(self) -> None:
        rng = random.Random(42)
        for _ in range(50):
            a = ColonistStats(**{s: rng.random() for s in STAT_NAMES})
            b = ColonistStats(**{s: rng.random() for s in STAT_NAMES})
            child = _inherit_stats(a, b, rng)
            for s in STAT_NAMES:
                assert 0.0 <= getattr(child, s) <= 1.0

    def test_inherited_skills_bounded(self) -> None:
        rng = random.Random(42)
        for _ in range(50):
            a = ColonistSkills(**{s: rng.random() for s in SKILL_NAMES})
            b = ColonistSkills(**{s: rng.random() for s in SKILL_NAMES})
            child = _inherit_skills(a, b, rng)
            for s in SKILL_NAMES:
                assert 0.0 <= getattr(child, s) <= 1.0
