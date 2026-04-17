"""
Turtles All the Way Down — deep recursive sub-simulation engine.

Colonists spawn nested LisPy simulations (up to depth 3) to model
governance proposals before committing.  Each depth answers a different
question:

  Depth 1: "What happens if we adopt this governance model?"
  Depth 2: "What happens to the meta-governance that evaluates D1?"
  Depth 3: "What philosophical principle underlies D1 and D2?"

Results bubble back as evidence that modifies governance votes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.mars100.lispy_vm import run as lispy_run, LispyError


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TurtleNode:
    """Single recursion level within a turtle tree."""
    depth: int
    expression: str
    result: Any = None
    error: str | None = None
    conclusion: float = 0.0          # [-1, 1] normalised verdict
    children: list[TurtleNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "depth": self.depth,
            "expression": self.expression,
            "result": _safe(self.result),
            "conclusion": self.conclusion,
        }
        if self.error:
            d["error"] = self.error
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class TurtleTree:
    """Full recursion tree for one governance proposal."""
    proposal_type: str
    colonist_id: str
    year: int
    root: TurtleNode | None = None
    max_depth_reached: int = 0
    reversal: bool = False           # did deeper sims reverse the D1 verdict?
    convergence: float = 0.0         # how much deeper sims agree [-1, 1]

    def to_dict(self) -> dict:
        return {
            "proposal_type": self.proposal_type,
            "colonist_id": self.colonist_id,
            "year": self.year,
            "max_depth_reached": self.max_depth_reached,
            "reversal": self.reversal,
            "convergence": self.convergence,
            "root": self.root.to_dict() if self.root else None,
        }


@dataclass
class TurtleSummary:
    """Aggregate stats across all turtle trees in a simulation."""
    total_trees: int = 0
    trees_reaching_depth2: int = 0
    trees_reaching_depth3: int = 0
    reversals: int = 0
    avg_convergence: float = 0.0
    deepest_insight: str = ""

    def to_dict(self) -> dict:
        return {
            "total_trees": self.total_trees,
            "trees_reaching_depth2": self.trees_reaching_depth2,
            "trees_reaching_depth3": self.trees_reaching_depth3,
            "reversals": self.reversals,
            "avg_convergence": round(self.avg_convergence, 4),
            "deepest_insight": self.deepest_insight,
        }


# ---------------------------------------------------------------------------
# Expression generators — governance-specific LisPy at each depth
# ---------------------------------------------------------------------------

_DEPTH1_TEMPLATES: dict[str, str] = {
    "council": (
        "(let ((voice (+ empathy (* resolve 0.5))))"
        " (if (> voice 1.0) 0.8 (if (> voice 0.5) 0.3 -0.4)))"
    ),
    "dictator": (
        "(let ((power (- resolve empathy)))"
        " (if (> power 0.3) 0.7 (if (> power 0.0) 0.1 -0.6)))"
    ),
    "lottery": (
        "(let ((luck (+ faith (* improvisation 0.7))))"
        " (if (> luck 1.0) 0.6 (if (> luck 0.4) 0.2 -0.3)))"
    ),
    "consensus": (
        "(let ((harmony (+ empathy faith)))"
        " (if (> harmony 1.2) 0.9 (if (> harmony 0.6) 0.4 -0.2)))"
    ),
    "ai_governor": (
        "(let ((trust-ai (+ coding (* improvisation 0.5))))"
        " (if (> trust-ai 1.0) 0.7 (if (> trust-ai 0.4) 0.2 -0.5)))"
    ),
    "anarchy": (
        "(let ((rebel (+ paranoia (* improvisation 0.6))))"
        " (if (> rebel 1.0) 0.5 (if (> rebel 0.5) 0.1 -0.7)))"
    ),
}

_DEPTH2_TEMPLATE = (
    "(let ((d1-result {d1_result}))"
    " (let ((meta (if (> d1-result 0) (- 1.0 (* d1-result 0.3))"
    "                                  (+ -1.0 (* d1-result -0.3)))))"
    "  meta))"
)

_DEPTH3_TEMPLATE = (
    "(let ((d1 {d1_result}) (d2 {d2_result}))"
    " (let ((phi (* (+ d1 d2) 0.5)))"
    "  (if (> phi 0.3) 1.0 (if (< phi -0.3) -1.0 (* phi 2.0)))))"
)


def _depth1_expr(gov_type: str) -> str:
    """Return a governance-specific depth-1 LisPy expression."""
    return _DEPTH1_TEMPLATES.get(gov_type, _DEPTH1_TEMPLATES["council"])


def _depth2_expr(d1_result: float) -> str:
    """Return a meta-governance depth-2 LisPy expression."""
    return _DEPTH2_TEMPLATE.format(d1_result=round(d1_result, 4))


def _depth3_expr(d1_result: float, d2_result: float) -> str:
    """Return a philosophical depth-3 LisPy expression."""
    return _DEPTH3_TEMPLATE.format(
        d1_result=round(d1_result, 4),
        d2_result=round(d2_result, 4),
    )


# ---------------------------------------------------------------------------
# Gate conditions — should we go deeper?
# ---------------------------------------------------------------------------

def _should_recurse_to_depth2(d1_conclusion: float, wisdom: float) -> bool:
    """Depth 2 triggers on ambiguity or high wisdom."""
    ambiguous = abs(d1_conclusion) < 0.5
    return ambiguous or wisdom > 0.15


def _should_recurse_to_depth3(
    d1_conclusion: float,
    d2_conclusion: float,
    wisdom: float,
    coding: float,
) -> bool:
    """Depth 3 triggers on reversal or very high wisdom + coding."""
    reversal = (d1_conclusion > 0) != (d2_conclusion > 0)
    return reversal or (wisdom > 0.25 and coding > 0.6)


# ---------------------------------------------------------------------------
# Core recursion
# ---------------------------------------------------------------------------

def _eval_node(expr: str, bindings: dict[str, Any], depth: int) -> TurtleNode:
    """Evaluate a single LisPy expression and return a TurtleNode."""
    node = TurtleNode(depth=depth, expression=expr)
    max_steps = 10000 // depth
    try:
        value = lispy_run(expr, extra_bindings=bindings,
                          max_steps=max_steps, max_depth=200 // depth)
        node.result = value
        if isinstance(value, (int, float)):
            node.conclusion = max(-1.0, min(1.0, float(value)))
        else:
            node.conclusion = 0.0
    except LispyError as exc:
        node.error = str(exc)
        node.conclusion = 0.0
    except Exception as exc:
        node.error = str(exc)
        node.conclusion = 0.0
    return node


def run_turtle(
    gov_type: str,
    colonist_id: str,
    year: int,
    bindings: dict[str, Any],
    wisdom: float = 0.0,
    coding: float = 0.0,
) -> TurtleTree:
    """Run a full turtle recursion tree for one governance proposal.

    Returns a TurtleTree with up to 3 levels of depth.
    """
    tree = TurtleTree(
        proposal_type=gov_type,
        colonist_id=colonist_id,
        year=year,
    )

    # --- Depth 1 ---
    d1_expr = _depth1_expr(gov_type)
    d1_node = _eval_node(d1_expr, bindings, depth=1)
    tree.root = d1_node
    tree.max_depth_reached = 1

    if d1_node.error:
        return tree

    # --- Depth 2 gate ---
    if not _should_recurse_to_depth2(d1_node.conclusion, wisdom):
        return tree

    d2_expr = _depth2_expr(d1_node.conclusion)
    d2_bindings = dict(bindings)
    d2_bindings["parent-result"] = d1_node.conclusion
    d2_node = _eval_node(d2_expr, d2_bindings, depth=2)
    d1_node.children.append(d2_node)
    tree.max_depth_reached = 2

    if d2_node.error:
        _analyse_tree(tree)
        return tree

    # --- Depth 3 gate ---
    if not _should_recurse_to_depth3(
        d1_node.conclusion, d2_node.conclusion, wisdom, coding
    ):
        _analyse_tree(tree)
        return tree

    d3_expr = _depth3_expr(d1_node.conclusion, d2_node.conclusion)
    d3_bindings = dict(bindings)
    d3_bindings["parent-result"] = d2_node.conclusion
    d3_bindings["grandparent-result"] = d1_node.conclusion
    d3_node = _eval_node(d3_expr, d3_bindings, depth=3)
    d2_node.children.append(d3_node)
    tree.max_depth_reached = 3

    _analyse_tree(tree)
    return tree


# ---------------------------------------------------------------------------
# Post-hoc analysis
# ---------------------------------------------------------------------------

def _analyse_tree(tree: TurtleTree) -> None:
    """Compute reversal and convergence on a finished tree."""
    if tree.root is None:
        return
    conclusions = [tree.root.conclusion]
    for d2 in tree.root.children:
        conclusions.append(d2.conclusion)
        for d3 in d2.children:
            conclusions.append(d3.conclusion)

    if len(conclusions) >= 2:
        signs = [1 if c >= 0 else -1 for c in conclusions]
        tree.reversal = signs[0] != signs[-1]
        mean = sum(conclusions) / len(conclusions)
        variance = sum((c - mean) ** 2 for c in conclusions) / len(conclusions)
        tree.convergence = 1.0 - min(1.0, math.sqrt(variance))
    else:
        tree.convergence = 1.0


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

def turtle_vote_modifier(tree: TurtleTree) -> float:
    """Convert a turtle tree into a vote bias in [-0.3, 0.3].

    Deeper and more convergent trees have stronger influence.
    """
    if tree.root is None:
        return 0.0
    base = tree.root.conclusion * 0.15
    depth_mult = 1.0 + 0.25 * (tree.max_depth_reached - 1)  # 1.0 / 1.25 / 1.5
    convergence_mult = 0.5 + 0.5 * tree.convergence
    raw = base * depth_mult * convergence_mult
    return max(-0.3, min(0.3, raw))


def update_colonist_wisdom(
    wisdom_map: dict[str, float],
    colonist_id: str,
    tree: TurtleTree,
) -> None:
    """Update a colonist's turtle wisdom based on recursion experience.

    Wisdom grows slowly, capped at 0.3.
    """
    current = wisdom_map.get(colonist_id, 0.0)
    gain = 0.01 * tree.max_depth_reached
    if tree.reversal:
        gain += 0.005
    wisdom_map[colonist_id] = min(0.3, current + gain)


def aggregate_summary(trees: list[TurtleTree]) -> TurtleSummary:
    """Compute aggregate stats from all turtle trees."""
    summary = TurtleSummary()
    summary.total_trees = len(trees)
    convergence_vals: list[float] = []
    for t in trees:
        if t.max_depth_reached >= 2:
            summary.trees_reaching_depth2 += 1
        if t.max_depth_reached >= 3:
            summary.trees_reaching_depth3 += 1
        if t.reversal:
            summary.reversals += 1
        convergence_vals.append(t.convergence)
    if convergence_vals:
        summary.avg_convergence = sum(convergence_vals) / len(convergence_vals)
    # Find deepest insight expression
    deepest_depth = 0
    for t in trees:
        if t.root:
            for d2 in t.root.children:
                for d3 in d2.children:
                    if d3.depth > deepest_depth and not d3.error:
                        deepest_depth = d3.depth
                        summary.deepest_insight = d3.expression[:200]
    return summary


def _safe(value: Any) -> Any:
    """Make a value JSON-serializable."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_safe(v) for v in value]
    return str(value)
