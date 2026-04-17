"""Tests for Mars-100 simulation kernel."""
from __future__ import annotations

import json
import random
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100 import (
    Resources, Intent, YearResult,
    pick_event, decide_action, resolve_intents, tick_year,
    EVENTS, ACTIONS,
)
from src.colonist import Colonist, create_colony
from src.governance import GovernanceState


class TestResources:
    def test_initial_values(self):
        r = Resources()
        assert 0.0 <= r.food <= 2.0
        assert 0.0 <= r.water <= 2.0
        assert 0.0 <= r.power <= 2.0

    def test_apply_event(self):
        r = Resources(food=1.0)
        r.apply_event({"food": -0.1})
        assert r.food == pytest.approx(0.9, abs=0.001)

    def test_apply_event_clamped(self):
        r = Resources(food=0.01)
        r.apply_event({"food": -1.0})
        assert r.food == 0.0

    def test_consumption_tick(self):
        r = Resources(food=1.0, water=1.0, oxygen=1.0)
        initial_food = r.food
        r.consumption_tick(10)
        assert r.food < initial_food

    def test_crisis_level_healthy(self):
        r = Resources(food=1.0, water=1.0, power=1.0, oxygen=1.0, hab_integrity=1.0)
        assert r.crisis_level() == 0.0

    def test_crisis_level_dire(self):
        r = Resources(food=0.0, water=0.0, power=0.0, oxygen=0.0, hab_integrity=0.0)
        assert r.crisis_level() == 1.0

    def test_serialization_round_trip(self):
        r = Resources(food=0.7, water=1.3, power=0.5)
        d = r.to_dict()
        r2 = Resources.from_dict(d)
        assert r2.food == pytest.approx(r.food, abs=0.001)
        assert r2.water == pytest.approx(r.water, abs=0.001)


class TestEvents:
    def test_event_weights_positive(self):
        for e in EVENTS:
            assert e["weight"] > 0

    def test_pick_event_returns_valid(self):
        rng = random.Random(42)
        for _ in range(50):
            event = pick_event(rng)
            assert "id" in event
            assert "description" in event
            assert event["id"] in [e["id"] for e in EVENTS]

    def test_event_distribution(self):
        """All events should be reachable."""
        rng = random.Random(42)
        seen = set()
        for _ in range(10000):
            event = pick_event(rng)
            seen.add(event["id"])
        # Most events should appear in 10000 trials
        assert len(seen) >= len(EVENTS) - 1  # rare events might not appear


class TestIntentResolution:
    def test_decide_action_returns_valid(self):
        colonists = create_colony(seed=42)
        r = Resources()
        gov = GovernanceState()
        rng = random.Random(42)
        for c in colonists:
            intent = decide_action(c, r, 1, {"id": "calm_year"}, gov, rng)
            assert intent.action in ACTIONS
            assert intent.colonist_id == c.id

    def test_crisis_drives_survival_actions(self):
        colonists = create_colony(seed=42)
        r = Resources(food=0.0, water=0.0, power=0.0, hab_integrity=0.0)
        gov = GovernanceState()
        rng = random.Random(42)
        survival_actions = {"gather_food", "gather_water", "generate_power", "repair_hab"}
        for c in colonists:
            intent = decide_action(c, r, 1, {"id": "dust_storm"}, gov, rng)
            assert intent.action in survival_actions

    def test_resolve_intents_modifies_resources(self):
        colonists = create_colony(seed=42)
        r = Resources(food=0.5)
        gov = GovernanceState()
        rng = random.Random(42)
        intents = [Intent(c.id, "gather_food", effectiveness=0.8) for c in colonists[:5]]
        initial_food = r.food
        resolve_intents(intents, colonists, r, 1, gov, rng)
        assert r.food > initial_food  # food should increase


class TestYearTick:
    def test_tick_returns_year_result(self):
        colonists = create_colony(seed=42)
        r = Resources()
        gov = GovernanceState()
        rng = random.Random(42)
        result = tick_year(1, colonists, r, gov, rng)
        assert isinstance(result, YearResult)
        assert result.year == 1
        assert result.alive_count > 0

    def test_10_year_smoke(self):
        """Simulation should run 10 years without crash."""
        colonists = create_colony(seed=42)
        r = Resources()
        gov = GovernanceState()
        rng = random.Random(42)
        for year in range(1, 11):
            result = tick_year(year, colonists, r, gov, rng)
            assert result.alive_count >= 0
            assert 0.0 <= r.food <= 2.0
            assert 0.0 <= r.water <= 2.0

    def test_resources_conservation(self):
        """Resources should stay within physical bounds across many years."""
        colonists = create_colony(seed=42)
        r = Resources()
        gov = GovernanceState()
        rng = random.Random(42)
        for year in range(1, 51):
            tick_year(year, colonists, r, gov, rng)
            assert 0.0 <= r.food <= 2.0
            assert 0.0 <= r.water <= 2.0
            assert 0.0 <= r.power <= 2.0
            assert 0.0 <= r.oxygen <= 2.0
            assert 0.0 <= r.hab_integrity <= 2.0
            assert 0.0 <= r.morale <= 1.0

    def test_extinct_colony_returns_collapsed(self):
        colonists = create_colony(seed=42)
        for c in colonists:
            c.die(1, "test")
        r = Resources()
        gov = GovernanceState()
        rng = random.Random(42)
        result = tick_year(2, colonists, r, gov, rng)
        assert result.alive_count == 0
        assert result.governance_label == "collapsed"

    def test_governance_label_assigned(self):
        colonists = create_colony(seed=42)
        r = Resources()
        gov = GovernanceState()
        rng = random.Random(42)
        result = tick_year(1, colonists, r, gov, rng)
        assert result.governance_label is not None
        assert isinstance(result.governance_label, str)

    def test_diaries_generated(self):
        colonists = create_colony(seed=42)
        r = Resources()
        gov = GovernanceState()
        rng = random.Random(42)
        result = tick_year(1, colonists, r, gov, rng)
        assert len(result.colonist_diaries) >= 1
        assert len(result.colonist_diaries) <= 3
        for diary in result.colonist_diaries:
            assert "colonist_id" in diary
            assert "entry" in diary

    def test_year_result_serializable(self):
        colonists = create_colony(seed=42)
        r = Resources()
        gov = GovernanceState()
        rng = random.Random(42)
        result = tick_year(1, colonists, r, gov, rng)
        d = result.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_colonist_memory_grows(self):
        colonists = create_colony(seed=42)
        r = Resources()
        gov = GovernanceState()
        rng = random.Random(42)
        initial_memories = {c.id: len(c.memory) for c in colonists}
        for year in range(1, 11):
            tick_year(year, colonists, r, gov, rng)
        for c in colonists:
            if c.alive:
                assert len(c.memory) > initial_memories[c.id]


class TestFullSimulation:
    def test_100_year_deterministic(self):
        """Full 100-year sim with same seed produces same result."""
        def run_sim(seed: int) -> dict:
            colonists = create_colony(seed=seed)
            r = Resources()
            gov = GovernanceState()
            rng = random.Random(seed)
            final_year = 0
            for year in range(1, 101):
                result = tick_year(year, colonists, r, gov, rng)
                final_year = year
                if result.alive_count == 0:
                    break
            return {
                "final_year": final_year,
                "alive": sum(1 for c in colonists if c.alive),
                "food": round(r.food, 4),
            }

        r1 = run_sim(42)
        r2 = run_sim(42)
        assert r1 == r2

    def test_50_year_smoke_no_crash(self):
        """50-year simulation completes without exception."""
        colonists = create_colony(seed=123)
        r = Resources()
        gov = GovernanceState()
        rng = random.Random(123)
        for year in range(1, 51):
            result = tick_year(year, colonists, r, gov, rng)
            if result.alive_count == 0:
                break

    def test_different_seeds_different_outcomes(self):
        """Different seeds produce different simulation histories."""
        def final_state(seed: int) -> tuple:
            colonists = create_colony(seed=seed)
            r = Resources()
            gov = GovernanceState()
            rng = random.Random(seed)
            for year in range(1, 31):
                result = tick_year(year, colonists, r, gov, rng)
                if result.alive_count == 0:
                    break
            alive = sum(1 for c in colonists if c.alive)
            return (alive, round(r.food, 2))

        results = [final_state(s) for s in range(10)]
        # Should have at least 2 different outcomes across 10 seeds
        assert len(set(results)) >= 2
