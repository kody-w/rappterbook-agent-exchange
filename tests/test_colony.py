"""Tests for colony resource model and social graph."""
from __future__ import annotations

import random
import pytest
from src.mars100.colony import (
    Resources, Relationship, SocialGraph, tick_resources, RESOURCE_NAMES,
)


class TestResources:
    def test_defaults(self):
        r = Resources()
        assert r.food == 0.7
        assert r.water == 0.7
        assert r.power == 0.8
        assert r.air == 0.9
        assert r.medicine == 0.5

    def test_to_dict(self):
        r = Resources()
        d = r.to_dict()
        assert set(d.keys()) == set(RESOURCE_NAMES)

    def test_from_dict(self):
        d = {"food": 0.5, "water": 0.6, "power": 0.7, "air": 0.8, "medicine": 0.9}
        r = Resources.from_dict(d)
        assert r.food == 0.5
        assert r.medicine == 0.9

    def test_clamp(self):
        r = Resources(food=1.5, water=-0.3, power=0.5, air=0.5, medicine=0.5)
        r.clamp()
        assert r.food == 1.0
        assert r.water == 0.0

    def test_critical(self):
        r = Resources(food=0.1, water=0.8, power=0.05, air=0.9, medicine=0.5)
        crit = r.critical()
        assert "food" in crit
        assert "power" in crit
        assert "water" not in crit

    def test_average(self):
        r = Resources(food=0.5, water=0.5, power=0.5, air=0.5, medicine=0.5)
        assert r.average() == pytest.approx(0.5)

    def test_total(self):
        r = Resources(food=0.2, water=0.2, power=0.2, air=0.2, medicine=0.2)
        assert r.total() == pytest.approx(1.0)


class TestTickResources:
    def test_basic_tick(self):
        r = Resources()
        delta = tick_resources(r, active_count=10, skill_bonuses={}, event_effects={})
        assert isinstance(delta, dict)
        for name in RESOURCE_NAMES:
            assert name in delta

    def test_bounded_after_tick(self):
        r = Resources()
        for _ in range(50):
            tick_resources(r, active_count=10, skill_bonuses={}, event_effects={})
        for name in RESOURCE_NAMES:
            val = getattr(r, name)
            assert 0.0 <= val <= 1.0, f"{name}={val}"

    def test_skill_bonus_effect(self):
        r1 = Resources(food=0.5, water=0.5, power=0.5, air=0.5, medicine=0.5)
        r2 = Resources(food=0.5, water=0.5, power=0.5, air=0.5, medicine=0.5)
        tick_resources(r1, active_count=5, skill_bonuses={"food": 0.5}, event_effects={})
        tick_resources(r2, active_count=5, skill_bonuses={}, event_effects={})
        assert r1.food > r2.food  # bonus should help

    def test_event_effect(self):
        r = Resources(food=0.5, water=0.5, power=0.5, air=0.5, medicine=0.5)
        tick_resources(r, active_count=5, skill_bonuses={}, event_effects={"food": -0.3})
        assert r.food < 0.5


class TestRelationship:
    def test_defaults(self):
        r = Relationship()
        assert r.trust == 0.5
        assert r.affection == 0.5
        assert r.respect == 0.5

    def test_score(self):
        r = Relationship(trust=1.0, affection=1.0, respect=1.0)
        assert r.score() == pytest.approx(1.0)

    def test_serialization(self):
        r = Relationship(trust=0.3, affection=0.7, respect=0.5)
        d = r.to_dict()
        r2 = Relationship.from_dict(d)
        assert r2.trust == pytest.approx(0.3)


class TestSocialGraph:
    def test_initialize(self):
        sg = SocialGraph()
        sg.initialize(["a", "b", "c"], random.Random(42))
        assert "a" in sg.edges
        assert "b" in sg.edges["a"]
        assert "a" not in sg.edges["a"]  # no self-edges

    def test_get_default(self):
        sg = SocialGraph()
        r = sg.get("nonexistent", "also_nonexistent")
        assert r.trust == 0.5

    def test_cohesion(self):
        sg = SocialGraph()
        sg.initialize(["a", "b", "c"], random.Random(42))
        c = sg.colony_cohesion(["a", "b", "c"])
        assert 0.0 <= c <= 1.0

    def test_cooperation_increases_trust(self):
        sg = SocialGraph()
        sg.initialize(["a", "b"], random.Random(42))
        before = sg.get("a", "b").trust
        for _ in range(20):
            sg.update_from_cooperation("a", "b", random.Random(42))
        after = sg.get("a", "b").trust
        assert after >= before

    def test_conflict_decreases_trust(self):
        sg = SocialGraph()
        sg.initialize(["a", "b"], random.Random(42))
        before = sg.get("a", "b").trust
        for _ in range(20):
            sg.update_from_conflict("a", "b", random.Random(42))
        after = sg.get("a", "b").trust
        assert after <= before

    def test_most_trusted_by(self):
        sg = SocialGraph()
        sg.initialize(["a", "b", "c"], random.Random(42))
        sg.edges["a"]["b"].trust = 0.9
        sg.edges["a"]["c"].trust = 0.1
        most = sg.most_trusted_by("a", ["a", "b", "c"])
        assert most == "b"

    def test_to_dict(self):
        sg = SocialGraph()
        sg.initialize(["a", "b"], random.Random(42))
        d = sg.to_dict()
        assert "a" in d
        assert "b" in d["a"]


class TestEventEffect:
    def test_update_from_event(self):
        sg = SocialGraph()
        sg.initialize(["a", "b", "c"], random.Random(42))
        sg.update_from_event(["a", "b"], valence=0.5, rng=random.Random(42))
        # Should not crash — values may go up or down slightly
