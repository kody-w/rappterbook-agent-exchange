"""Tests for sub-simulation spawner."""
from __future__ import annotations

import pytest
from src.mars100.subsim import (
    SubSimBudget, SubSimResult, spawn_subsim,
    MAX_SUBSIM_DEPTH, MAX_SUBSIMS_PER_COLONIST_PER_YEAR, MAX_SUBSIMS_PER_YEAR,
)


class TestSubSimBudget:
    def test_initial_state(self):
        b = SubSimBudget(year=1)
        assert b.colony_total == 0
        assert b.can_spawn("test")

    def test_per_colonist_limit(self):
        b = SubSimBudget(year=1)
        for _ in range(MAX_SUBSIMS_PER_COLONIST_PER_YEAR):
            assert b.can_spawn("c1")
            b.record("c1")
        assert not b.can_spawn("c1")
        assert b.can_spawn("c2")

    def test_colony_limit(self):
        b = SubSimBudget(year=1)
        for i in range(MAX_SUBSIMS_PER_YEAR):
            cid = f"c{i}"
            assert b.can_spawn(cid)
            b.record(cid)
        assert not b.can_spawn("extra")


class TestSpawnSubsim:
    def test_basic(self):
        result = spawn_subsim(
            expression="(+ 1 2)",
            colonist_id="test",
            year=1,
            bindings={},
            depth=1,
        )
        assert result.succeeded
        assert result.result == 3
        assert result.depth == 1

    def test_with_bindings(self):
        result = spawn_subsim(
            expression="(+ x 10)",
            colonist_id="test",
            year=1,
            bindings={"x": 5},
            depth=1,
        )
        assert result.succeeded
        assert result.result == 15

    def test_depth_limit(self):
        result = spawn_subsim(
            expression="(+ 1 2)",
            colonist_id="test",
            year=1,
            bindings={},
            depth=MAX_SUBSIM_DEPTH + 1,
        )
        assert not result.succeeded
        assert "depth" in result.error.lower()

    def test_budget_exhausted(self):
        budget = SubSimBudget(year=1)
        for _ in range(MAX_SUBSIMS_PER_COLONIST_PER_YEAR):
            budget.record("test")
        result = spawn_subsim(
            expression="(+ 1 2)",
            colonist_id="test",
            year=1,
            bindings={},
            depth=1,
            budget=budget,
        )
        assert not result.succeeded
        assert "budget" in result.error.lower()

    def test_logging(self):
        log: list[SubSimResult] = []
        spawn_subsim(
            expression="(+ 1 2)",
            colonist_id="test",
            year=1,
            bindings={},
            depth=1,
            log=log,
        )
        assert len(log) == 1
        assert log[0].colonist_id == "test"

    def test_error_handling(self):
        result = spawn_subsim(
            expression="(/ 1 0)",
            colonist_id="test",
            year=1,
            bindings={},
            depth=1,
        )
        assert not result.succeeded
        assert result.error is not None

    def test_depth_reduces_budget(self):
        """Deeper sub-sims get fewer steps."""
        result = spawn_subsim(
            expression="(+ 1 2)",
            colonist_id="test",
            year=1,
            bindings={},
            depth=3,
        )
        assert result.succeeded

    def test_sim_depth_binding(self):
        result = spawn_subsim(
            expression="sim-depth",
            colonist_id="test",
            year=5,
            bindings={},
            depth=2,
        )
        assert result.succeeded
        assert result.result == 2

    def test_sim_year_binding(self):
        result = spawn_subsim(
            expression="sim-year",
            colonist_id="test",
            year=42,
            bindings={},
            depth=1,
        )
        assert result.succeeded
        assert result.result == 42


class TestSubSimResult:
    def test_to_dict(self):
        r = SubSimResult(depth=1, colonist_id="c1", year=5,
                         expression="(+ 1 2)", result=3)
        d = r.to_dict()
        assert d["depth"] == 1
        assert d["result"] == 3
        assert d["colonist_id"] == "c1"

    def test_to_dict_with_children(self):
        child = SubSimResult(depth=2, colonist_id="c1", year=5,
                             expression="(+ 3 4)", result=7)
        parent = SubSimResult(depth=1, colonist_id="c1", year=5,
                              expression="(+ 1 2)", result=3,
                              children=[child])
        d = parent.to_dict()
        assert len(d["children"]) == 1
        assert d["children"][0]["depth"] == 2

    def test_to_dict_with_error(self):
        r = SubSimResult(depth=1, colonist_id="c1", year=5,
                         expression="bad", error="oops")
        d = r.to_dict()
        assert "error" in d
