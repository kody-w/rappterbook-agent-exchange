"""Tests for Mars-100 birth system."""
from __future__ import annotations

import random
import pytest
from src.mars100.births import (
    attempt_birth, can_birth, find_eligible_pair, is_juvenile,
    BirthEvent, MAX_POPULATION, MIN_FOOD_FOR_BIRTH, MIN_MEDICINE_FOR_BIRTH,
    MIN_TRUST_FOR_PAIR, BIRTH_FOOD_COST, BIRTH_MEDICINE_COST,
    PARENT_COOLDOWN_YEARS, JUVENILE_YEARS,
)
from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, create_founding_ten
from src.mars100.colony import Resources, SocialGraph, Relationship


def _make_colonist(cid: str, name: str = "Test") -> Colonist:
    return Colonist(id=cid, name=name, element="fire", archetype="settler",
                    stats=ColonistStats(), skills=ColonistSkills(),
                    decision_expr="(+ resolve 0.1)")


def _make_pair_with_trust(trust: float) -> tuple[list[Colonist], SocialGraph]:
    a = _make_colonist("a", "Alice")
    b = _make_colonist("b", "Bob")
    social = SocialGraph()
    social.edges = {
        "a": {"b": Relationship(trust=trust, affection=0.5, respect=0.5)},
        "b": {"a": Relationship(trust=trust, affection=0.5, respect=0.5)},
    }
    return [a, b], social


class TestCanBirth:
    def test_allows_when_conditions_met(self) -> None:
        r = Resources(food=0.7, medicine=0.5)
        assert can_birth(r, 5) is True

    def test_blocks_at_max_population(self) -> None:
        r = Resources(food=0.8, medicine=0.5)
        assert can_birth(r, MAX_POPULATION) is False

    def test_blocks_low_food(self) -> None:
        r = Resources(food=0.2, medicine=0.5)
        assert can_birth(r, 5) is False

    def test_blocks_low_medicine(self) -> None:
        r = Resources(food=0.7, medicine=0.1)
        assert can_birth(r, 5) is False


class TestFindEligiblePair:
    def test_finds_trusted_pair(self) -> None:
        colonists, social = _make_pair_with_trust(0.8)
        pair = find_eligible_pair(colonists, social, 10, {})
        assert pair is not None
        ids = {pair[0].id, pair[1].id}
        assert ids == {"a", "b"}

    def test_rejects_low_trust(self) -> None:
        colonists, social = _make_pair_with_trust(0.3)
        pair = find_eligible_pair(colonists, social, 10, {})
        assert pair is None

    def test_respects_cooldown(self) -> None:
        colonists, social = _make_pair_with_trust(0.9)
        cooldowns = {"a": 20}  # cooldown until year 20
        pair = find_eligible_pair(colonists, social, 10, cooldowns)
        assert pair is None

    def test_allows_after_cooldown_expires(self) -> None:
        colonists, social = _make_pair_with_trust(0.9)
        cooldowns = {"a": 5}
        pair = find_eligible_pair(colonists, social, 10, cooldowns)
        assert pair is not None

    def test_single_colonist_returns_none(self) -> None:
        a = _make_colonist("a")
        social = SocialGraph()
        pair = find_eligible_pair([a], social, 10, {})
        assert pair is None


class TestAttemptBirth:
    def test_successful_birth(self) -> None:
        colonists, social = _make_pair_with_trust(0.9)
        resources = Resources(food=0.8, water=0.7, medicine=0.6)
        rng = random.Random(42)
        # Attempt many times to overcome probability
        births = []
        for seed in range(100):
            rng_try = random.Random(seed)
            cs = [_make_colonist("a", "Alice"), _make_colonist("b", "Bob")]
            social_try = SocialGraph()
            social_try.edges = {
                "a": {"b": Relationship(trust=0.9, affection=0.8, respect=0.7)},
                "b": {"a": Relationship(trust=0.9, affection=0.8, respect=0.7)},
            }
            res = Resources(food=0.8, water=0.7, medicine=0.6)
            birth = attempt_birth(cs, res, social_try, 10, rng_try, {})
            if birth is not None:
                births.append(birth)
        assert len(births) > 0, "At least one birth should occur across 100 seeds"

    def test_birth_costs_resources(self) -> None:
        # Find a seed that produces a birth
        for seed in range(200):
            rng = random.Random(seed)
            cs = [_make_colonist("a", "Alice"), _make_colonist("b", "Bob")]
            social = SocialGraph()
            social.edges = {
                "a": {"b": Relationship(trust=0.9, affection=0.8, respect=0.7)},
                "b": {"a": Relationship(trust=0.9, affection=0.8, respect=0.7)},
            }
            res = Resources(food=0.8, water=0.7, medicine=0.6)
            birth = attempt_birth(cs, res, social, 10, rng, {})
            if birth is not None:
                assert res.food < 0.8 - BIRTH_FOOD_COST + 0.01
                assert res.medicine < 0.6 - BIRTH_MEDICINE_COST + 0.01
                return
        pytest.skip("No birth produced in 200 seeds")

    def test_birth_adds_colonist(self) -> None:
        for seed in range(200):
            rng = random.Random(seed)
            cs = [_make_colonist("a", "Alice"), _make_colonist("b", "Bob")]
            social = SocialGraph()
            social.edges = {
                "a": {"b": Relationship(trust=0.9, affection=0.8, respect=0.7)},
                "b": {"a": Relationship(trust=0.9, affection=0.8, respect=0.7)},
            }
            res = Resources(food=0.8, water=0.7, medicine=0.6)
            birth = attempt_birth(cs, res, social, 10, rng, {})
            if birth is not None:
                assert len(cs) == 3
                child = cs[-1]
                assert child.id == birth.child_id
                assert child.alive is True
                return
        pytest.skip("No birth produced in 200 seeds")

    def test_birth_sets_cooldown(self) -> None:
        for seed in range(200):
            rng = random.Random(seed)
            cs = [_make_colonist("a", "Alice"), _make_colonist("b", "Bob")]
            social = SocialGraph()
            social.edges = {
                "a": {"b": Relationship(trust=0.9, affection=0.8, respect=0.7)},
                "b": {"a": Relationship(trust=0.9, affection=0.8, respect=0.7)},
            }
            res = Resources(food=0.8, water=0.7, medicine=0.6)
            cooldowns: dict[str, int] = {}
            birth = attempt_birth(cs, res, social, 10, rng, cooldowns)
            if birth is not None:
                assert cooldowns["a"] == 10 + PARENT_COOLDOWN_YEARS
                assert cooldowns["b"] == 10 + PARENT_COOLDOWN_YEARS
                return
        pytest.skip("No birth produced in 200 seeds")

    def test_child_has_social_edges(self) -> None:
        for seed in range(200):
            rng = random.Random(seed)
            cs = [_make_colonist("a", "Alice"), _make_colonist("b", "Bob")]
            social = SocialGraph()
            social.edges = {
                "a": {"b": Relationship(trust=0.9, affection=0.8, respect=0.7)},
                "b": {"a": Relationship(trust=0.9, affection=0.8, respect=0.7)},
            }
            res = Resources(food=0.8, water=0.7, medicine=0.6)
            birth = attempt_birth(cs, res, social, 10, rng, {})
            if birth is not None:
                child = cs[-1]
                assert child.id in social.edges
                assert "a" in social.edges[child.id]
                assert "b" in social.edges[child.id]
                # Parent trust should be high
                assert social.edges[child.id]["a"].trust > 0.5
                return
        pytest.skip("No birth produced in 200 seeds")


class TestBirthEventSerialization:
    def test_to_dict(self) -> None:
        be = BirthEvent(year=10, child_id="c1", child_name="Test Jr",
                        parent_a_id="a", parent_b_id="b")
        d = be.to_dict()
        assert d["year"] == 10
        assert d["child_id"] == "c1"
        assert d["child_name"] == "Test Jr"


class TestIsJuvenile:
    def test_founding_colonist_not_juvenile(self) -> None:
        c = _make_colonist("a")
        assert is_juvenile(c, 50) is False

    def test_child_is_juvenile(self) -> None:
        c = _make_colonist("child")
        c.add_memory(10, "Born to Alice and Bob", 0.8)
        assert is_juvenile(c, 12) is True

    def test_child_grows_up(self) -> None:
        c = _make_colonist("child")
        c.add_memory(10, "Born to Alice and Bob", 0.8)
        assert is_juvenile(c, 10 + JUVENILE_YEARS) is False

    def test_child_just_before_maturity(self) -> None:
        c = _make_colonist("child")
        c.add_memory(10, "Born to Alice and Bob", 0.8)
        assert is_juvenile(c, 10 + JUVENILE_YEARS - 1) is True


class TestChildStatInheritance:
    def test_child_stats_between_parents(self) -> None:
        """Child stats should roughly average parents, with noise."""
        rng = random.Random(42)
        for _ in range(20):
            cs = [_make_colonist("a", "Alice"), _make_colonist("b", "Bob")]
            cs[0].stats.resolve = 0.9
            cs[1].stats.resolve = 0.1
            social = SocialGraph()
            social.edges = {
                "a": {"b": Relationship(trust=0.9, affection=0.8, respect=0.7)},
                "b": {"a": Relationship(trust=0.9, affection=0.8, respect=0.7)},
            }
            res = Resources(food=0.8, water=0.7, medicine=0.6)
            birth = attempt_birth(cs, res, social, 10, rng, {})
            if birth is not None:
                child = cs[-1]
                # Should be roughly 0.5 ± noise, not 0.9 or 0.1
                assert 0.1 < child.stats.resolve < 0.9
                return
        pytest.skip("No birth in 20 attempts")


class TestPopulationCap:
    def test_no_birth_at_max(self) -> None:
        colonists = [_make_colonist(f"c{i}") for i in range(MAX_POPULATION)]
        social = SocialGraph()
        social.initialize([c.id for c in colonists], random.Random(1))
        res = Resources(food=0.9, medicine=0.9)
        birth = attempt_birth(colonists, res, social, 10, random.Random(42), {})
        assert birth is None
