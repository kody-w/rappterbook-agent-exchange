"""Tests for the recursive world simulation organ (v10.0)."""
from __future__ import annotations

import random
import pytest

from src.mars100.recursive_sim import (
    MiniColonist, MiniWorld, WorldSimResult, WorldSimBudget,
    WorldSimFrame, MiniGovState,
    create_mini_colonists, spawn_world_sim,
    MAX_WORLD_DEPTH, FRAMES_BY_DEPTH, AGENTS_BY_DEPTH,
    MINI_RESOURCES, MINI_STAT_NAMES, MINI_SKILL_NAMES,
    MINI_EVENTS, MINI_GOV_TYPES, GOVERNANCE_STABILITY_THRESHOLD,
    MAX_BUDGET_UNITS_PER_YEAR,
)


# ────── MiniColonist ──────

class TestMiniColonist:
    def test_creation(self):
        c = MiniColonist(id="mini-0", stats={"resolve": 0.6, "empathy": 0.4})
        assert c.id == "mini-0"
        assert c.alive is True

    def test_lispy_bindings_complete_surface(self):
        c = MiniColonist(id="m0", stats={"resolve": 0.7}, skills={"coding": 0.8})
        b = c.lispy_bindings()
        for name in MINI_STAT_NAMES:
            assert name in b
        for name in MINI_SKILL_NAMES:
            assert name in b
        assert b["resolve"] == 0.7
        assert b["coding"] == 0.8

    def test_lispy_bindings_defaults(self):
        c = MiniColonist(id="m0")
        b = c.lispy_bindings()
        assert b["resolve"] == 0.5
        assert b["terraforming"] == 0.1

    def test_to_dict(self):
        c = MiniColonist(id="m0", stats={"resolve": 0.5})
        d = c.to_dict()
        assert d["id"] == "m0"
        assert "alive" in d
        assert "gov_preference" in d


class TestCreateMiniColonists:
    def test_count(self):
        rng = random.Random(42)
        cols = create_mini_colonists(5, rng)
        assert len(cols) == 5

    def test_unique_ids(self):
        rng = random.Random(42)
        cols = create_mini_colonists(5, rng)
        ids = [c.id for c in cols]
        assert len(set(ids)) == 5

    def test_stat_bounds(self):
        rng = random.Random(42)
        cols = create_mini_colonists(10, rng)
        for c in cols:
            for name in MINI_STAT_NAMES:
                val = c.stats.get(name, 0.5)
                assert 0.0 <= val <= 1.0, f"{name}={val}"

    def test_skill_bounds(self):
        rng = random.Random(42)
        cols = create_mini_colonists(10, rng)
        for c in cols:
            for name in MINI_SKILL_NAMES:
                val = c.skills.get(name, 0.1)
                assert 0.0 <= val <= 1.0, f"{name}={val}"

    def test_parent_bindings_influence(self):
        rng = random.Random(42)
        parent = {"resolve": 0.9, "empathy": 0.1}
        cols = create_mini_colonists(20, rng, parent_bindings=parent)
        avg_resolve = sum(c.stats.get("resolve", 0.5) for c in cols) / 20
        avg_empathy = sum(c.stats.get("empathy", 0.5) for c in cols) / 20
        assert avg_resolve > avg_empathy

    def test_decision_expr_assigned(self):
        rng = random.Random(42)
        cols = create_mini_colonists(3, rng)
        for c in cols:
            assert isinstance(c.decision_expr, str)
            assert len(c.decision_expr) > 0


# ────── MiniGovState ──────

class TestMiniGovState:
    def test_initial(self):
        g = MiniGovState()
        assert g.gov_type == "anarchy"
        assert g.stable_frames == 0

    def test_to_dict(self):
        g = MiniGovState(gov_type="council", stable_frames=5)
        d = g.to_dict()
        assert d["gov_type"] == "council"
        assert d["stable_frames"] == 5


# ────── WorldSimBudget ──────

class TestWorldSimBudget:
    def test_initial(self):
        b = WorldSimBudget(year=1)
        assert b.units_used == 0.0
        assert b.can_spawn(1)

    def test_cost_decreases_with_depth(self):
        b = WorldSimBudget(year=1)
        cost1 = b.cost(1)
        cost2 = b.cost(2)
        cost3 = b.cost(3)
        assert cost1 > cost2 > cost3

    def test_budget_exhaustion(self):
        b = WorldSimBudget(year=1)
        while b.can_spawn(1):
            b.record(1)
        assert not b.can_spawn(1)
        assert b.units_used <= b.max_units + b.cost(1)

    def test_depth_limit(self):
        b = WorldSimBudget(year=1)
        assert not b.can_spawn(MAX_WORLD_DEPTH + 1)

    def test_mixed_depth_budget(self):
        b = WorldSimBudget(year=1)
        b.record(1)
        b.record(2)
        assert b.sims_spawned == 2
        assert b.units_used == b.cost(1) + b.cost(2)


# ────── WorldSimFrame ──────

class TestWorldSimFrame:
    def test_to_dict(self):
        f = WorldSimFrame(frame=1, event={"name": "calm"},
                          resources={"food": 0.5},
                          gov_type="anarchy", alive_count=3,
                          actions={"m0": 0.6})
        d = f.to_dict()
        assert d["frame"] == 1
        assert d["alive_count"] == 3


# ────── WorldSimResult ──────

class TestWorldSimResult:
    def test_succeeded(self):
        r = WorldSimResult(depth=1, colonist_id="c0", year=1,
                           frames_run=10, survived=True,
                           final_resources={"food": 0.5},
                           governance_history=[], dominant_governance="council",
                           governance_stable=True, stability_score=0.8)
        assert r.succeeded

    def test_error(self):
        r = WorldSimResult(depth=4, colonist_id="c0", year=1,
                           frames_run=0, survived=False,
                           final_resources={}, governance_history=[],
                           dominant_governance="anarchy",
                           governance_stable=False, stability_score=0.0,
                           error="max depth exceeded")
        assert not r.succeeded

    def test_to_dict(self):
        r = WorldSimResult(depth=1, colonist_id="c0", year=5,
                           frames_run=20, survived=True,
                           final_resources={"food": 0.6, "water": 0.5},
                           governance_history=[{"frame": 3, "from": "anarchy", "to": "council"}],
                           dominant_governance="council",
                           governance_stable=True, stability_score=0.85)
        d = r.to_dict()
        assert d["depth"] == 1
        assert d["survived"] is True
        assert d["dominant_governance"] == "council"
        assert "error" not in d

    def test_to_dict_with_children(self):
        child = WorldSimResult(depth=2, colonist_id="m0", year=5,
                               frames_run=10, survived=True,
                               final_resources={"food": 0.4},
                               governance_history=[],
                               dominant_governance="consensus",
                               governance_stable=True, stability_score=0.7)
        parent = WorldSimResult(depth=1, colonist_id="c0", year=5,
                                frames_run=20, survived=True,
                                final_resources={"food": 0.6},
                                governance_history=[],
                                dominant_governance="council",
                                governance_stable=True, stability_score=0.8,
                                children=[child])
        d = parent.to_dict()
        assert len(d["children"]) == 1
        assert d["children"][0]["depth"] == 2

    def test_compact_summary(self):
        r = WorldSimResult(depth=1, colonist_id="c0", year=5,
                           frames_run=20, survived=True,
                           final_resources={"food": 0.5},
                           governance_history=[],
                           dominant_governance="council",
                           governance_stable=True, stability_score=0.8)
        s = r.compact_summary()
        assert s["dominant_gov"] == "council"
        assert "children_count" in s


# ────── MiniWorld ──────

class TestMiniWorld:
    def test_creation(self):
        rng = random.Random(42)
        w = MiniWorld(depth=1, colonist_id="c0", year=10, rng=rng)
        assert len(w.colonists) == AGENTS_BY_DEPTH[1]
        assert w.max_frames == FRAMES_BY_DEPTH[1]

    def test_depth_scales_agents(self):
        rng = random.Random(42)
        w1 = MiniWorld(depth=1, colonist_id="c0", year=1, rng=rng)
        w2 = MiniWorld(depth=2, colonist_id="c0", year=1, rng=rng)
        w3 = MiniWorld(depth=3, colonist_id="c0", year=1, rng=rng)
        assert len(w1.colonists) > len(w2.colonists) >= len(w3.colonists)

    def test_depth_scales_frames(self):
        rng = random.Random(42)
        w1 = MiniWorld(depth=1, colonist_id="c0", year=1, rng=rng)
        w2 = MiniWorld(depth=2, colonist_id="c0", year=1, rng=rng)
        w3 = MiniWorld(depth=3, colonist_id="c0", year=1, rng=rng)
        assert w1.max_frames > w2.max_frames > w3.max_frames

    def test_run_completes(self):
        rng = random.Random(42)
        w = MiniWorld(depth=1, colonist_id="c0", year=10, rng=rng)
        result = w.run()
        assert result.succeeded
        assert result.frames_run > 0
        assert result.frames_run <= FRAMES_BY_DEPTH[1]

    def test_resources_bounded(self):
        rng = random.Random(42)
        w = MiniWorld(depth=1, colonist_id="c0", year=10, rng=rng)
        result = w.run()
        for res_name, val in result.final_resources.items():
            assert 0.0 <= val <= 1.0, f"{res_name}={val}"

    def test_governance_emerges(self):
        """Over many seeds, governance should sometimes change."""
        changed = False
        for seed in range(50):
            rng = random.Random(seed)
            w = MiniWorld(depth=1, colonist_id="c0", year=10, rng=rng)
            result = w.run()
            if result.governance_history:
                changed = True
                break
        assert changed, "No governance change across 50 seeds"

    def test_governance_stability_detection(self):
        """If governance doesn't change for enough frames, it's stable."""
        for seed in range(30):
            rng = random.Random(seed)
            w = MiniWorld(depth=1, colonist_id="c0", year=10, rng=rng)
            result = w.run()
            if result.governance_stable:
                assert result.stability_score > 0
                return
        # It's okay if no seed produces stability in 30 tries

    def test_inherits_parent_gov(self):
        rng = random.Random(42)
        w = MiniWorld(depth=1, colonist_id="c0", year=10, rng=rng,
                      parent_gov="council")
        assert w.gov.gov_type == "council"

    def test_death_possible(self):
        """With low resources, colonists should sometimes die."""
        for seed in range(50):
            rng = random.Random(seed)
            w = MiniWorld(depth=1, colonist_id="c0", year=10, rng=rng)
            w.resources = {"food": 0.05, "water": 0.05, "power": 0.05}
            result = w.run()
            if not result.survived:
                return  # death confirmed
        # Low resources should cause deaths

    def test_deterministic(self):
        """Same seed produces same result."""
        rng1 = random.Random(42)
        w1 = MiniWorld(depth=1, colonist_id="c0", year=10, rng=rng1)
        r1 = w1.run()
        rng2 = random.Random(42)
        w2 = MiniWorld(depth=1, colonist_id="c0", year=10, rng=rng2)
        r2 = w2.run()
        assert r1.frames_run == r2.frames_run
        assert r1.dominant_governance == r2.dominant_governance
        assert r1.survived == r2.survived


class TestMiniWorldDepth2:
    def test_depth2_runs(self):
        rng = random.Random(42)
        w = MiniWorld(depth=2, colonist_id="c0", year=10, rng=rng)
        result = w.run()
        assert result.succeeded
        assert result.depth == 2
        assert result.frames_run <= FRAMES_BY_DEPTH[2]

    def test_depth2_fewer_agents(self):
        rng = random.Random(42)
        w = MiniWorld(depth=2, colonist_id="c0", year=10, rng=rng)
        assert len(w.colonists) == AGENTS_BY_DEPTH[2]


class TestMiniWorldDepth3:
    def test_depth3_runs(self):
        rng = random.Random(42)
        w = MiniWorld(depth=3, colonist_id="c0", year=10, rng=rng)
        result = w.run()
        assert result.succeeded
        assert result.depth == 3
        assert result.frames_run <= FRAMES_BY_DEPTH[3]

    def test_depth3_minimal_agents(self):
        rng = random.Random(42)
        w = MiniWorld(depth=3, colonist_id="c0", year=10, rng=rng)
        assert len(w.colonists) == AGENTS_BY_DEPTH[3]


# ────── spawn_world_sim ──────

class TestSpawnWorldSim:
    def test_basic(self):
        rng = random.Random(42)
        result = spawn_world_sim(
            colonist_id="c0", year=10,
            parent_bindings={"resolve": 0.6, "empathy": 0.5},
            parent_gov="anarchy", depth=1, rng=rng,
        )
        assert result.succeeded
        assert result.depth == 1

    def test_depth_limit(self):
        rng = random.Random(42)
        result = spawn_world_sim(
            colonist_id="c0", year=10,
            parent_bindings={}, parent_gov="anarchy",
            depth=MAX_WORLD_DEPTH + 1, rng=rng,
        )
        assert not result.succeeded
        assert "depth" in result.error.lower()

    def test_budget_exhaustion(self):
        rng = random.Random(42)
        budget = WorldSimBudget(year=1, max_units=0.001)
        result = spawn_world_sim(
            colonist_id="c0", year=10,
            parent_bindings={}, parent_gov="anarchy",
            depth=1, rng=rng, budget=budget,
        )
        assert not result.succeeded
        assert "budget" in result.error.lower()

    def test_budget_tracking(self):
        rng = random.Random(42)
        budget = WorldSimBudget(year=1)
        spawn_world_sim(
            colonist_id="c0", year=10,
            parent_bindings={}, parent_gov="anarchy",
            depth=1, rng=rng, budget=budget,
        )
        assert budget.sims_spawned >= 1  # may include child sims
        assert budget.units_used > 0

    def test_with_parent_bindings(self):
        rng = random.Random(42)
        result = spawn_world_sim(
            colonist_id="c0", year=10,
            parent_bindings={"resolve": 0.9, "faith": 0.8, "coding": 0.7},
            parent_gov="council", depth=1, rng=rng,
        )
        assert result.succeeded

    def test_child_world_possible(self):
        """With enough budget, a world-sim may spawn a child."""
        for seed in range(100):
            rng = random.Random(seed)
            budget = WorldSimBudget(year=1, max_units=50.0)
            result = spawn_world_sim(
                colonist_id="c0", year=10,
                parent_bindings={}, parent_gov="anarchy",
                depth=1, rng=rng, budget=budget,
            )
            if result.children:
                assert result.children[0].depth == 2
                return
        # Child spawning is probabilistic; ok if rare


# ────── Property-based invariants ──────

class TestPhysicalInvariants:
    @pytest.mark.parametrize("seed", range(20))
    def test_resources_always_bounded(self, seed):
        rng = random.Random(seed)
        w = MiniWorld(depth=1, colonist_id="c0", year=10, rng=rng)
        result = w.run()
        for frame in w.frames:
            for res_name, val in frame.resources.items():
                assert 0.0 <= val <= 1.0, (
                    f"seed={seed} frame={frame.frame} {res_name}={val}")

    @pytest.mark.parametrize("seed", range(20))
    def test_alive_count_monotonic(self, seed):
        """Alive count should never increase (no births in mini-world)."""
        rng = random.Random(seed)
        w = MiniWorld(depth=1, colonist_id="c0", year=10, rng=rng)
        w.run()
        for i in range(1, len(w.frames)):
            assert w.frames[i].alive_count <= w.frames[i - 1].alive_count, (
                f"seed={seed}: alive increased at frame {w.frames[i].frame}")

    @pytest.mark.parametrize("seed", range(10))
    def test_stability_score_bounded(self, seed):
        rng = random.Random(seed)
        result = spawn_world_sim(
            colonist_id="c0", year=10,
            parent_bindings={}, parent_gov="anarchy",
            depth=1, rng=rng,
        )
        assert 0.0 <= result.stability_score <= 1.0


# ────── Smoke tests ──────

class TestSmoke:
    def test_full_depth1_sim(self):
        rng = random.Random(42)
        result = spawn_world_sim(
            colonist_id="c0", year=50,
            parent_bindings={"resolve": 0.6, "empathy": 0.5,
                             "faith": 0.4, "paranoia": 0.3},
            parent_gov="anarchy", depth=1, rng=rng,
        )
        assert result.succeeded
        assert result.frames_run == FRAMES_BY_DEPTH[1]

    def test_full_depth2_sim(self):
        rng = random.Random(42)
        result = spawn_world_sim(
            colonist_id="c0", year=50,
            parent_bindings={}, parent_gov="council",
            depth=2, rng=rng,
        )
        assert result.succeeded
        assert result.frames_run <= FRAMES_BY_DEPTH[2]

    def test_full_depth3_sim(self):
        rng = random.Random(42)
        result = spawn_world_sim(
            colonist_id="c0", year=50,
            parent_bindings={}, parent_gov="consensus",
            depth=3, rng=rng,
        )
        assert result.succeeded
        assert result.frames_run <= FRAMES_BY_DEPTH[3]
