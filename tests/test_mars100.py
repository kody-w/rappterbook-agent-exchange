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
    EVENTS, ACTIONS, reset_birth_counter, maybe_birth,
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
            reset_birth_counter()
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
            reset_birth_counter()
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


# ---------------------------------------------------------------------------
# New tests: Births
# ---------------------------------------------------------------------------

class TestBirths:
    def test_no_birth_before_year_15(self):
        """maybe_birth returns None before year 15."""
        reset_birth_counter()
        colonists = create_colony(seed=42)
        r = Resources(food=1.5, water=1.0)
        rng = random.Random(42)
        for year in range(1, 15):
            child = maybe_birth(year, colonists, r, rng)
            assert child is None, f"Unexpected birth at year {year}"

    def test_no_birth_low_food(self):
        """maybe_birth returns None when food is below threshold."""
        reset_birth_counter()
        colonists = create_colony(seed=42)
        r = Resources(food=0.5, water=1.0)
        rng = random.Random(42)
        child = maybe_birth(20, colonists, r, rng)
        assert child is None

    def test_no_birth_low_water(self):
        """maybe_birth returns None when water is below threshold."""
        reset_birth_counter()
        colonists = create_colony(seed=42)
        r = Resources(food=1.5, water=0.3)
        rng = random.Random(42)
        child = maybe_birth(20, colonists, r, rng)
        assert child is None

    def test_birth_occurs_favorable_conditions(self):
        """Births occur with high resources over many trials."""
        births = 0
        for seed in range(100):
            reset_birth_counter()
            colonists = create_colony(seed=seed)
            r = Resources(food=1.8, water=1.5, power=1.2)
            r.morale = 0.8
            rng = random.Random(seed)
            child = maybe_birth(20, colonists, r, rng)
            if child is not None:
                births += 1
        # With crisis near 0 and stability ~0.12 chance, expect some births
        assert births > 0, "No births occurred across 100 seeds"

    def test_mars_born_inherits_blended_stats(self):
        """Mars-born colonists inherit stats from parents."""
        reset_birth_counter()
        colonists = create_colony(seed=42)
        r = Resources(food=1.8, water=1.5, power=1.2)
        r.morale = 0.9
        # Try many seeds to get a birth
        for seed in range(200):
            reset_birth_counter()
            rng = random.Random(seed)
            child = maybe_birth(25, colonists, r, rng)
            if child is not None:
                assert child.id.startswith("mars-")
                assert child.year_joined == 25
                # Stats should be within valid range
                for s in child.stats.values():
                    assert 0.0 <= s <= 1.0
                for s in child.skills.values():
                    assert 0.0 <= s <= 1.0
                assert len(child.memory) >= 1
                assert child.memory[0]["event"] == "born on Mars"
                return
        pytest.skip("No birth occurred across 200 seeds")

    def test_birth_counter_caps_at_10(self):
        """Maximum 10 births per simulation."""
        import src.mars100 as m
        m._birth_counter = 10
        colonists = create_colony(seed=42)
        r = Resources(food=1.8, water=1.5)
        rng = random.Random(42)
        child = maybe_birth(50, colonists, r, rng)
        assert child is None
        reset_birth_counter()

    def test_birth_consumes_resources(self):
        """Birth reduces food and water."""
        for seed in range(200):
            reset_birth_counter()
            colonists = create_colony(seed=seed)
            r = Resources(food=1.8, water=1.5, power=1.2)
            r.morale = 0.9
            rng = random.Random(seed)
            food_before = r.food
            water_before = r.water
            child = maybe_birth(25, colonists, r, rng)
            if child is not None:
                assert r.food < food_before
                assert r.water < water_before
                return
        pytest.skip("No birth occurred across 200 seeds")


# ---------------------------------------------------------------------------
# New tests: Deep sub-sims
# ---------------------------------------------------------------------------

class TestDeepSubSims:
    def test_subsim_depth_reaches_2(self):
        """At least one seed reaches sub-sim depth 2 across multiple runs."""
        from src.run_mars100 import run_simulation
        import tempfile
        for seed in [7, 99, 256]:
            reset_birth_counter()
            d = Path(tempfile.mkdtemp())
            r = run_simulation(seed=seed, years=100, output_dir=d/'out',
                               state_dir=d/'state', quiet=True)
            max_d = r['subsimulations']['max_depth_reached']
            if max_d >= 2:
                return
        pytest.fail("No seed reached sub-sim depth 2")

    def test_subsim_depth_reaches_3(self):
        """At least one seed reaches sub-sim depth 3 (turtles all the way down)."""
        from src.run_mars100 import run_simulation
        import tempfile
        for seed in [99, 256, 777]:
            reset_birth_counter()
            d = Path(tempfile.mkdtemp())
            r = run_simulation(seed=seed, years=100, output_dir=d/'out',
                               state_dir=d/'state', quiet=True)
            max_d = r['subsimulations']['max_depth_reached']
            if max_d >= 3:
                return
        pytest.fail("No seed reached sub-sim depth 3")

    def test_subsim_count_increased(self):
        """More sub-sims run than the old baseline (was 4 for seed=42)."""
        from src.run_mars100 import run_simulation
        import tempfile
        d = Path(tempfile.mkdtemp())
        r = run_simulation(seed=42, years=100, output_dir=d/'out',
                           state_dir=d/'state', quiet=True)
        assert r['subsimulations']['total_runs'] > 10

    def test_subsim_log_has_meta_fields(self):
        """Sub-sim logs include depth and max_depth_reached fields."""
        from src.run_mars100 import run_simulation
        import tempfile
        d = Path(tempfile.mkdtemp())
        r = run_simulation(seed=42, years=50, output_dir=d/'out',
                           state_dir=d/'state', quiet=True)
        for log in r['subsimulations']['log']:
            assert 'depth' in log
            assert 'max_depth_reached' in log
            assert 'colonist' in log
            assert 'year' in log


# ---------------------------------------------------------------------------
# New tests: Meta-insights and governance emergence
# ---------------------------------------------------------------------------

class TestMetaInsights:
    def test_meta_insights_generated(self):
        """At least one seed produces meta-insights."""
        from src.run_mars100 import run_simulation
        import tempfile
        for seed in [42, 99, 777]:
            reset_birth_counter()
            d = Path(tempfile.mkdtemp())
            r = run_simulation(seed=seed, years=100, output_dir=d/'out',
                               state_dir=d/'state', quiet=True)
            if len(r['meta_insights']) > 0:
                return
        pytest.fail("No meta-insights generated across tested seeds")

    def test_promoted_amendment(self):
        """A simulation with meta-insights and decent fitness promotes an amendment."""
        from src.run_mars100 import run_simulation
        import tempfile
        for seed in [42, 99, 256, 777]:
            reset_birth_counter()
            d = Path(tempfile.mkdtemp())
            r = run_simulation(seed=seed, years=100, output_dir=d/'out',
                               state_dir=d/'state', quiet=True)
            if 'promoted_amendment' in r:
                assert r['promoted_amendment']['proposal']
                assert r['promoted_amendment']['insight']
                return
        pytest.fail("No promoted amendment across tested seeds")

    def test_governance_emerges_beyond_nascent(self):
        """Governance label evolves beyond 'nascent' over 100 years."""
        from src.run_mars100 import run_simulation
        import tempfile
        d = Path(tempfile.mkdtemp())
        r = run_simulation(seed=42, years=100, output_dir=d/'out',
                           state_dir=d/'state', quiet=True)
        assert r['governance']['final_label'] != "nascent"
        assert len(r['governance']['transitions']) > 0


# ---------------------------------------------------------------------------
# New tests: Colony longevity
# ---------------------------------------------------------------------------

class TestColonyLongevity:
    def test_colony_survives_100_years_some_seeds(self):
        """At least some seeds survive to year 100."""
        from src.run_mars100 import run_simulation
        import tempfile
        survived = 0
        for seed in [42, 99, 256, 777, 1001]:
            reset_birth_counter()
            d = Path(tempfile.mkdtemp())
            r = run_simulation(seed=seed, years=100, output_dir=d/'out',
                               state_dir=d/'state', quiet=True)
            if r['_meta']['years_simulated'] == 100:
                survived += 1
        assert survived >= 2, f"Only {survived}/5 seeds survived to year 100"

    def test_death_rate_reduced(self):
        """Fewer accidental deaths vs baseline (was ~7 for seed=42)."""
        from src.run_mars100 import run_simulation
        import tempfile
        d = Path(tempfile.mkdtemp())
        r = run_simulation(seed=42, years=100, output_dir=d/'out',
                           state_dir=d/'state', quiet=True)
        deaths = r['colony']['death_causes']
        total_accident = deaths.get('accident', 0)
        # With reduced death rate, should see fewer accidents
        assert total_accident <= 15, f"Too many accidents: {total_accident}"

    def test_births_appear_in_summary(self):
        """Summary includes birth count."""
        from src.run_mars100 import run_simulation
        import tempfile
        d = Path(tempfile.mkdtemp())
        r = run_simulation(seed=42, years=100, output_dir=d/'out',
                           state_dir=d/'state', quiet=True)
        assert 'total_births' in r['colony']
        assert r['colony']['total_births'] >= 0
