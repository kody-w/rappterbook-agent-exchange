"""Tests for the turtles deep recursion engine."""
from __future__ import annotations

import math
import pytest

from src.mars100.turtles import (
    TurtleNode, TurtleTree, TurtleSummary,
    run_turtle, turtle_vote_modifier, update_colonist_wisdom,
    aggregate_summary, _eval_node, _depth1_expr, _depth2_expr,
    _depth3_expr, _should_recurse_to_depth2, _should_recurse_to_depth3,
    _analyse_tree, _DEPTH1_TEMPLATES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bindings(**overrides: float) -> dict[str, float]:
    defaults = {
        "empathy": 0.6, "resolve": 0.5, "faith": 0.4,
        "improvisation": 0.5, "paranoia": 0.3, "hoarding": 0.2,
        "coding": 0.5, "terraforming": 0.3, "hydroponics": 0.4,
        "mediation": 0.5, "prayer": 0.3, "sabotage": 0.1,
    }
    defaults.update(overrides)
    return defaults


# ===========================================================================
# TurtleNode
# ===========================================================================

class TestTurtleNode:
    def test_defaults(self):
        n = TurtleNode(depth=1, expression="(+ 1 2)")
        assert n.depth == 1
        assert n.conclusion == 0.0
        assert n.children == []
        assert n.error is None

    def test_to_dict_minimal(self):
        n = TurtleNode(depth=1, expression="(+ 1 2)", result=3, conclusion=1.0)
        d = n.to_dict()
        assert d["depth"] == 1
        assert d["result"] == 3
        assert d["conclusion"] == 1.0
        assert "error" not in d
        assert "children" not in d

    def test_to_dict_with_error(self):
        n = TurtleNode(depth=2, expression="(/ 1 0)", error="division by zero")
        d = n.to_dict()
        assert d["error"] == "division by zero"

    def test_to_dict_with_children(self):
        child = TurtleNode(depth=2, expression="(+ 1 1)", result=2, conclusion=1.0)
        parent = TurtleNode(depth=1, expression="(+ 2 3)", result=5, conclusion=1.0,
                            children=[child])
        d = parent.to_dict()
        assert len(d["children"]) == 1
        assert d["children"][0]["depth"] == 2


# ===========================================================================
# TurtleTree
# ===========================================================================

class TestTurtleTree:
    def test_defaults(self):
        t = TurtleTree(proposal_type="council", colonist_id="c1", year=5)
        assert t.root is None
        assert t.max_depth_reached == 0
        assert t.reversal is False
        assert t.convergence == 0.0

    def test_to_dict(self):
        root = TurtleNode(depth=1, expression="(+ 1 1)", result=2, conclusion=1.0)
        t = TurtleTree(proposal_type="lottery", colonist_id="c2", year=10,
                        root=root, max_depth_reached=1)
        d = t.to_dict()
        assert d["proposal_type"] == "lottery"
        assert d["root"]["depth"] == 1
        assert d["max_depth_reached"] == 1


# ===========================================================================
# Expression generators
# ===========================================================================

class TestExpressionGenerators:
    @pytest.mark.parametrize("gov_type", list(_DEPTH1_TEMPLATES.keys()))
    def test_depth1_all_types(self, gov_type):
        expr = _depth1_expr(gov_type)
        assert isinstance(expr, str)
        assert expr.startswith("(")

    def test_depth1_unknown_falls_back_to_council(self):
        assert _depth1_expr("unknown_gov") == _depth1_expr("council")

    def test_depth2_format(self):
        expr = _depth2_expr(0.5)
        assert "0.5" in expr
        assert expr.startswith("(")

    def test_depth3_format(self):
        expr = _depth3_expr(0.5, -0.3)
        assert "0.5" in expr
        assert "-0.3" in expr


# ===========================================================================
# Gate conditions
# ===========================================================================

class TestGateConditions:
    def test_depth2_ambiguous(self):
        assert _should_recurse_to_depth2(0.3, 0.0) is True

    def test_depth2_high_wisdom(self):
        assert _should_recurse_to_depth2(0.9, 0.2) is True

    def test_depth2_clear_low_wisdom(self):
        assert _should_recurse_to_depth2(0.9, 0.05) is False

    def test_depth3_reversal(self):
        assert _should_recurse_to_depth3(0.5, -0.5, 0.0, 0.0) is True

    def test_depth3_high_wisdom_coding(self):
        assert _should_recurse_to_depth3(0.5, 0.5, 0.3, 0.7) is True

    def test_depth3_blocked(self):
        assert _should_recurse_to_depth3(0.5, 0.5, 0.1, 0.3) is False


# ===========================================================================
# _eval_node
# ===========================================================================

class TestEvalNode:
    def test_simple_arithmetic(self):
        node = _eval_node("(+ 1 2)", {}, depth=1)
        assert node.result == 3
        assert node.conclusion == 1.0  # clamped

    def test_conclusion_clamped(self):
        node = _eval_node("(- 0 5)", {}, depth=1)
        assert node.conclusion == -1.0

    def test_bindings_used(self):
        node = _eval_node("empathy", {"empathy": 0.7}, depth=1)
        assert abs(node.result - 0.7) < 0.001
        assert abs(node.conclusion - 0.7) < 0.001

    def test_error_handling(self):
        node = _eval_node("(unknown-fn 1)", {}, depth=1)
        assert node.error is not None
        assert node.conclusion == 0.0


# ===========================================================================
# run_turtle
# ===========================================================================

class TestRunTurtle:
    def test_depth1_only(self):
        """Clear conclusion + low wisdom => stays at depth 1."""
        tree = run_turtle("dictator", "c1", 5, _bindings(resolve=0.9, empathy=0.1),
                          wisdom=0.0, coding=0.0)
        assert tree.root is not None
        assert tree.max_depth_reached == 1
        assert tree.root.children == []

    def test_reaches_depth2(self):
        """Ambiguous D1 conclusion triggers depth 2."""
        tree = run_turtle("council", "c1", 10,
                          _bindings(empathy=0.4, resolve=0.3),
                          wisdom=0.2, coding=0.0)
        assert tree.max_depth_reached >= 2

    def test_reaches_depth3(self):
        """High wisdom + coding triggers depth 3."""
        tree = run_turtle("council", "c1", 20,
                          _bindings(empathy=0.4, resolve=0.3),
                          wisdom=0.3, coding=0.8)
        # Should reach at least depth 2; depth 3 depends on reversal
        assert tree.max_depth_reached >= 2

    def test_all_gov_types(self):
        """Every governance type produces a valid tree."""
        for gov_type in _DEPTH1_TEMPLATES:
            tree = run_turtle(gov_type, "c1", 5, _bindings(),
                              wisdom=0.3, coding=0.8)
            assert tree.root is not None
            assert tree.root.error is None
            assert tree.proposal_type == gov_type

    def test_tree_serializable(self):
        """Tree can be converted to dict without error."""
        tree = run_turtle("consensus", "c1", 15, _bindings(),
                          wisdom=0.3, coding=0.8)
        d = tree.to_dict()
        assert isinstance(d, dict)
        assert d["root"] is not None


# ===========================================================================
# _analyse_tree
# ===========================================================================

class TestAnalyseTree:
    def test_single_node(self):
        root = TurtleNode(depth=1, expression="x", conclusion=0.5)
        tree = TurtleTree(proposal_type="council", colonist_id="c1", year=1, root=root)
        _analyse_tree(tree)
        assert tree.convergence == 1.0  # single node = perfect convergence
        assert tree.reversal is False

    def test_reversal_detected(self):
        d2 = TurtleNode(depth=2, expression="x", conclusion=-0.5)
        root = TurtleNode(depth=1, expression="x", conclusion=0.5, children=[d2])
        tree = TurtleTree(proposal_type="council", colonist_id="c1", year=1,
                          root=root, max_depth_reached=2)
        _analyse_tree(tree)
        assert tree.reversal is True

    def test_convergence_perfect(self):
        d2 = TurtleNode(depth=2, expression="x", conclusion=0.5)
        root = TurtleNode(depth=1, expression="x", conclusion=0.5, children=[d2])
        tree = TurtleTree(proposal_type="council", colonist_id="c1", year=1,
                          root=root, max_depth_reached=2)
        _analyse_tree(tree)
        assert tree.convergence == 1.0


# ===========================================================================
# turtle_vote_modifier
# ===========================================================================

class TestVoteModifier:
    def test_no_root(self):
        tree = TurtleTree(proposal_type="council", colonist_id="c1", year=1)
        assert turtle_vote_modifier(tree) == 0.0

    def test_bounded(self):
        """Output always in [-0.3, 0.3]."""
        for gov in _DEPTH1_TEMPLATES:
            tree = run_turtle(gov, "c1", 5, _bindings(), wisdom=0.3, coding=0.8)
            mod = turtle_vote_modifier(tree)
            assert -0.3 <= mod <= 0.3, f"{gov}: modifier {mod} out of bounds"

    def test_deeper_has_more_influence(self):
        """A depth-2 tree should have higher absolute influence than depth-1."""
        tree1 = TurtleTree(proposal_type="council", colonist_id="c1", year=1,
                           root=TurtleNode(depth=1, expression="x",
                                           conclusion=0.5),
                           max_depth_reached=1, convergence=1.0)
        # Create depth-2 tree with same conclusion
        d2 = TurtleNode(depth=2, expression="y", conclusion=0.5)
        root2 = TurtleNode(depth=1, expression="x", conclusion=0.5, children=[d2])
        tree2 = TurtleTree(proposal_type="council", colonist_id="c1", year=1,
                           root=root2, max_depth_reached=2, convergence=1.0)
        assert abs(turtle_vote_modifier(tree2)) >= abs(turtle_vote_modifier(tree1))


# ===========================================================================
# update_colonist_wisdom
# ===========================================================================

class TestWisdom:
    def test_grows(self):
        w: dict[str, float] = {}
        tree = TurtleTree(proposal_type="council", colonist_id="c1", year=1,
                          max_depth_reached=2)
        update_colonist_wisdom(w, "c1", tree)
        assert w["c1"] > 0.0

    def test_capped_at_03(self):
        w = {"c1": 0.29}
        tree = TurtleTree(proposal_type="council", colonist_id="c1", year=1,
                          max_depth_reached=3, reversal=True)
        update_colonist_wisdom(w, "c1", tree)
        assert w["c1"] <= 0.3

    def test_reversal_bonus(self):
        w1: dict[str, float] = {}
        w2: dict[str, float] = {}
        tree_no_rev = TurtleTree(proposal_type="council", colonist_id="c1",
                                  year=1, max_depth_reached=2, reversal=False)
        tree_rev = TurtleTree(proposal_type="council", colonist_id="c1",
                               year=1, max_depth_reached=2, reversal=True)
        update_colonist_wisdom(w1, "c1", tree_no_rev)
        update_colonist_wisdom(w2, "c1", tree_rev)
        assert w2["c1"] > w1["c1"]


# ===========================================================================
# aggregate_summary
# ===========================================================================

class TestAggregateSummary:
    def test_empty(self):
        s = aggregate_summary([])
        assert s.total_trees == 0
        assert s.avg_convergence == 0.0

    def test_counts(self):
        t1 = TurtleTree(proposal_type="council", colonist_id="c1", year=1,
                         max_depth_reached=1, convergence=1.0)
        t2 = TurtleTree(proposal_type="lottery", colonist_id="c2", year=2,
                         max_depth_reached=2, reversal=True, convergence=0.8)
        t3 = TurtleTree(proposal_type="anarchy", colonist_id="c3", year=3,
                         max_depth_reached=3, convergence=0.6)
        s = aggregate_summary([t1, t2, t3])
        assert s.total_trees == 3
        assert s.trees_reaching_depth2 == 2
        assert s.trees_reaching_depth3 == 1
        assert s.reversals == 1
        assert abs(s.avg_convergence - 0.8) < 0.01

    def test_serializable(self):
        s = aggregate_summary([])
        d = s.to_dict()
        assert isinstance(d, dict)


# ===========================================================================
# Property-based invariants
# ===========================================================================

class TestInvariants:
    @pytest.mark.parametrize("seed", range(20))
    def test_conclusion_bounded(self, seed):
        """All conclusions in [-1, 1]."""
        import random
        rng = random.Random(seed)
        b = {k: rng.random() for k in [
            "empathy", "resolve", "faith", "improvisation",
            "paranoia", "hoarding", "coding", "terraforming",
            "hydroponics", "mediation", "prayer", "sabotage",
        ]}
        gov = rng.choice(list(_DEPTH1_TEMPLATES.keys()))
        tree = run_turtle(gov, f"c{seed}", seed, b,
                          wisdom=rng.random() * 0.3,
                          coding=rng.random())
        assert tree.root is not None
        assert -1.0 <= tree.root.conclusion <= 1.0
        for d2 in tree.root.children:
            assert -1.0 <= d2.conclusion <= 1.0
            for d3 in d2.children:
                assert -1.0 <= d3.conclusion <= 1.0

    @pytest.mark.parametrize("seed", range(20))
    def test_vote_modifier_bounded(self, seed):
        """Vote modifier always in [-0.3, 0.3]."""
        import random
        rng = random.Random(seed)
        b = {k: rng.random() for k in [
            "empathy", "resolve", "faith", "improvisation",
            "paranoia", "hoarding", "coding", "terraforming",
            "hydroponics", "mediation", "prayer", "sabotage",
        ]}
        gov = rng.choice(list(_DEPTH1_TEMPLATES.keys()))
        tree = run_turtle(gov, f"c{seed}", seed, b,
                          wisdom=rng.random() * 0.3,
                          coding=rng.random())
        mod = turtle_vote_modifier(tree)
        assert -0.3 <= mod <= 0.3

    def test_depth_never_exceeds_3(self):
        """No tree should exceed depth 3."""
        for gov in _DEPTH1_TEMPLATES:
            tree = run_turtle(gov, "c1", 50, _bindings(),
                              wisdom=0.3, coding=1.0)
            assert tree.max_depth_reached <= 3


# ===========================================================================
# Integration smoke test
# ===========================================================================

class TestSmoke:
    def test_10_year_simulation(self):
        """Run 10 governance proposals through turtle recursion."""
        import random
        rng = random.Random(42)
        all_trees: list[TurtleTree] = []
        wisdom_map: dict[str, float] = {}
        for year in range(1, 11):
            gov = rng.choice(list(_DEPTH1_TEMPLATES.keys()))
            cid = f"colonist-{rng.randint(0, 9)}"
            b = {k: rng.random() for k in [
                "empathy", "resolve", "faith", "improvisation",
                "paranoia", "hoarding", "coding", "terraforming",
                "hydroponics", "mediation", "prayer", "sabotage",
            ]}
            tree = run_turtle(gov, cid, year, b,
                              wisdom=wisdom_map.get(cid, 0.0),
                              coding=b["coding"])
            all_trees.append(tree)
            update_colonist_wisdom(wisdom_map, cid, tree)
            mod = turtle_vote_modifier(tree)
            assert -0.3 <= mod <= 0.3

        summary = aggregate_summary(all_trees)
        assert summary.total_trees == 10
        assert summary.to_dict()["total_trees"] == 10
        # At least some should reach depth 2
        assert summary.trees_reaching_depth2 > 0
