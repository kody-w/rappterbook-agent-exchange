"""
Recursive sub-simulation engine for Mars-100.

Implements "Turtles All the Way Down" (Amendment XIII): sub-simulations
that spawn sub-sub-simulations to model consequences at increasing depth.
Max depth 3.  Each level narrows the question.  Results are normalised
scalars in [-1, 1] so recurse-or-stop thresholds are well-defined.

Scenario types
--------------
governance_test    – will a proposed governance change survive crises?
resource_forecast  – will we survive the next decade given current trends?
conflict_probe     – can a dispute between two colonists be resolved?
existential_probe  – (rare) meta-insight about simulation nature

Budget model: one recursive *tree* consumes a single "scenario slot" from
the per-colonist budget.  Internal depth increments are free.  This
prevents recursive chains from starving the yearly budget.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.mars100.lispy_vm import run as lispy_run, LispyError, LispyBudgetExceeded

MAX_DEPTH = 3
RECURSE_LO = -0.4
RECURSE_HI = 0.4

SCENARIO_TYPES = (
    "governance_test",
    "resource_forecast",
    "conflict_probe",
    "existential_probe",
)


@dataclass
class RecursiveResult:
    """Result of one node in the recursion tree."""
    scenario: str
    depth: int
    colonist_id: str
    year: int
    expression: str
    raw_result: Any = None
    normalised: float = 0.0
    error: str | None = None
    children: list["RecursiveResult"] = field(default_factory=list)
    insight: str | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "scenario": self.scenario,
            "depth": self.depth,
            "colonist_id": self.colonist_id,
            "year": self.year,
            "expression": self.expression,
            "raw_result": _safe(self.raw_result),
            "normalised": round(self.normalised, 4),
        }
        if self.error:
            d["error"] = self.error
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        if self.insight:
            d["insight"] = self.insight
        return d

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def flatten(self) -> list["RecursiveResult"]:
        """Return this node + all descendants as a flat list."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.flatten())
        return nodes


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_safe(v) for v in value]
    return str(value)


def _normalise(raw: Any) -> float:
    """Clamp any numeric result to [-1, 1]."""
    if isinstance(raw, bool):
        return 1.0 if raw else -1.0
    if isinstance(raw, (int, float)) and math.isfinite(raw):
        return max(-1.0, min(1.0, float(raw)))
    return 0.0


def _should_recurse(normalised: float) -> bool:
    """Recurse when the result is ambiguous (near zero)."""
    return RECURSE_LO <= normalised <= RECURSE_HI


# ── Expression generators per scenario × depth ────────────────────────

def _gov_test_expr(depth: int, bindings: dict, parent_result: float | None) -> str:
    """Generate LisPy expression for governance test at given depth."""
    if depth == 1:
        return (
            "(let ((stability (+ (* empathy 0.3) (* resolve 0.3) (* faith 0.2) (* improvisation 0.2))))"
            "  (let ((crisis-risk (* paranoia 0.5)))"
            "    (- stability crisis-risk)))"
        )
    if depth == 2:
        pr = parent_result if parent_result is not None else 0.0
        return (
            f"(let ((base {pr:.4f}))"
            "  (let ((crisis-pressure (* paranoia resolve)))"
            "    (let ((council-strength (* empathy mediation)))"
            "      (- (+ base council-strength) crisis-pressure))))"
        )
    # depth 3 — meta-governance
    pr = parent_result if parent_result is not None else 0.0
    return (
        f"(let ((d2-result {pr:.4f}))"
        "  (let ((fracture-risk (- paranoia empathy)))"
        "    (let ((reform-potential (* faith improvisation)))"
        "      (if (> fracture-risk 0.3)"
        "        (- d2-result fracture-risk)"
        "        (+ d2-result reform-potential)))))"
    )


def _resource_expr(depth: int, bindings: dict, parent_result: float | None) -> str:
    """Generate LisPy expression for resource forecast at given depth."""
    if depth == 1:
        return (
            "(let ((supply (+ (* food 0.3) (* water 0.3) (* power 0.2) (* air 0.2))))"
            "  (let ((demand (* 0.06 population)))"
            "    (- supply demand)))"
        )
    if depth == 2:
        pr = parent_result if parent_result is not None else 0.0
        return (
            f"(let ((trend {pr:.4f}))"
            "  (let ((emergency-food (* hydroponics 0.3)))"
            "    (let ((drain (* population 0.02)))"
            "      (+ trend (- emergency-food drain)))))"
        )
    pr = parent_result if parent_result is not None else 0.0
    return (
        f"(let ((d2-trend {pr:.4f}))"
        "  (let ((recovery (* terraforming resolve 0.5)))"
        "    (let ((collapse-risk (* paranoia (- 1.0 food))))"
        "      (if (> collapse-risk 0.5)"
        "        (- d2-trend collapse-risk)"
        "        (+ d2-trend recovery)))))"
    )


def _conflict_expr(depth: int, bindings: dict, parent_result: float | None) -> str:
    """Generate LisPy expression for conflict probe at given depth."""
    if depth == 1:
        return (
            "(let ((mediation-power (* mediation empathy)))"
            "  (let ((aggression (* paranoia sabotage)))"
            "    (- mediation-power aggression)))"
        )
    if depth == 2:
        pr = parent_result if parent_result is not None else 0.0
        return (
            f"(let ((initial {pr:.4f}))"
            "  (let ((retaliation (* paranoia hoarding)))"
            "    (let ((deescalation (* empathy faith)))"
            "      (+ initial (- deescalation retaliation)))))"
        )
    pr = parent_result if parent_result is not None else 0.0
    return (
        f"(let ((d2-state {pr:.4f}))"
        "  (let ((peace-signal (* faith empathy resolve)))"
        "    (let ((grudge (* paranoia (- 1.0 empathy))))"
        "      (if (> peace-signal grudge)"
        "        (+ d2-state (* peace-signal 0.3))"
        "        (- d2-state (* grudge 0.3))))))"
    )


def _existential_expr(depth: int, bindings: dict, parent_result: float | None) -> str:
    """Generate LisPy expression for existential probing at given depth."""
    if depth == 1:
        return (
            "(let ((pattern-score (+ (* faith 0.4) (* paranoia 0.3) (* improvisation 0.3))))"
            "  (let ((noise (- (* resolve 0.2) (* hoarding 0.1))))"
            "    (- pattern-score noise)))"
        )
    if depth == 2:
        pr = parent_result if parent_result is not None else 0.0
        return (
            f"(let ((awareness {pr:.4f}))"
            "  (let ((question-depth (* faith improvisation)))"
            "    (let ((denial (* resolve (- 1.0 faith))))"
            "      (if (> question-depth denial)"
            "        (+ awareness (* question-depth 0.2))"
            "        (- awareness 0.1)))))"
        )
    # depth 3 — the meta-insight
    pr = parent_result if parent_result is not None else 0.0
    return (
        f"(let ((d2-awareness {pr:.4f}))"
        "  (let ((sim-depth-factor (* sim-depth 0.01)))"
        "    (let ((self-model (* faith improvisation empathy)))"
        "      (+ d2-awareness self-model sim-depth-factor))))"
    )


_EXPR_GENERATORS = {
    "governance_test": _gov_test_expr,
    "resource_forecast": _resource_expr,
    "conflict_probe": _conflict_expr,
    "existential_probe": _existential_expr,
}


# ── Insight extraction ────────────────────────────────────────────────

_INSIGHT_TEMPLATES = {
    "governance_test": {
        2: "Council governance shows {quality} under stress (stability={val:.2f}).",
        3: "Meta-governance insight: when councils fracture, {outcome} (score={val:.2f}).",
    },
    "resource_forecast": {
        2: "Resource outlook is {quality} under crisis conditions (trend={val:.2f}).",
        3: "Recovery modeling suggests {outcome} (score={val:.2f}).",
    },
    "conflict_probe": {
        2: "Conflict resolution attempt shows {quality} prospects (score={val:.2f}).",
        3: "Long-term peace assessment: {outcome} (score={val:.2f}).",
    },
    "existential_probe": {
        2: "The patterns in colony data are {quality} regular (score={val:.2f}).",
        3: "Recursive self-model converges: the simulation sees itself seeing itself (score={val:.2f}).",
    },
}


def _generate_insight(scenario: str, depth: int, normalised: float) -> str | None:
    """Generate a human-readable insight string from a result, or None."""
    if depth < 2:
        return None
    templates = _INSIGHT_TEMPLATES.get(scenario, {})
    tmpl = templates.get(depth)
    if tmpl is None:
        return None
    quality = "strong" if normalised > 0.3 else "weak" if normalised < -0.3 else "ambiguous"
    outcome = ("reform emerges" if normalised > 0.2
               else "collapse is likely" if normalised < -0.2
               else "the system oscillates")
    return tmpl.format(quality=quality, val=normalised, outcome=outcome)


# ── Core recursive runner ─────────────────────────────────────────────

def run_scenario(
    scenario: str,
    colonist_id: str,
    year: int,
    bindings: dict[str, Any],
    depth: int = 1,
    parent_result: float | None = None,
    max_steps_base: int = 10000,
) -> RecursiveResult:
    """Run a recursive scenario at the given depth.

    Each depth level gets half the step budget of the previous.
    Recursion continues automatically when the result is ambiguous
    (normalised value between RECURSE_LO and RECURSE_HI) and depth < MAX_DEPTH.
    """
    generator = _EXPR_GENERATORS.get(scenario)
    if generator is None:
        return RecursiveResult(
            scenario=scenario, depth=depth, colonist_id=colonist_id,
            year=year, expression="", error=f"unknown scenario: {scenario}",
        )

    if depth > MAX_DEPTH:
        return RecursiveResult(
            scenario=scenario, depth=depth, colonist_id=colonist_id,
            year=year, expression="", error="max depth exceeded",
        )

    expression = generator(depth, bindings, parent_result)
    max_steps = max(100, max_steps_base // (2 ** (depth - 1)))
    max_vm_depth = max(20, 200 // depth)

    node = RecursiveResult(
        scenario=scenario, depth=depth, colonist_id=colonist_id,
        year=year, expression=expression,
    )

    vm_bindings = dict(bindings)
    vm_bindings["sim-depth"] = depth
    vm_bindings["sim-year"] = year

    try:
        raw = lispy_run(expression, extra_bindings=vm_bindings,
                        max_steps=max_steps, max_depth=max_vm_depth)
        node.raw_result = raw
        node.normalised = _normalise(raw)
    except (LispyError, LispyBudgetExceeded) as exc:
        node.error = str(exc)
        return node
    except Exception as exc:  # pragma: no cover
        node.error = f"unexpected: {exc}"
        return node

    node.insight = _generate_insight(scenario, depth, node.normalised)

    # Recurse if ambiguous and depth allows
    if _should_recurse(node.normalised) and depth < MAX_DEPTH:
        child = run_scenario(
            scenario=scenario,
            colonist_id=colonist_id,
            year=year,
            bindings=bindings,
            depth=depth + 1,
            parent_result=node.normalised,
            max_steps_base=max_steps_base,
        )
        node.children.append(child)

    return node


def choose_scenario(
    colonist_stats: dict[str, float],
    colonist_skills: dict[str, float],
    resource_avg: float,
    has_conflict: bool,
    rng_value: float,
) -> str | None:
    """Choose which scenario a colonist should run, or None.

    Returns one of SCENARIO_TYPES or None if no scenario is triggered.
    Deterministic given the same inputs (rng_value is a pre-rolled float).
    """
    faith = colonist_stats.get("faith", 0.5)
    paranoia = colonist_stats.get("paranoia", 0.5)
    improvisation = colonist_stats.get("improvisation", 0.5)
    empathy = colonist_stats.get("empathy", 0.5)
    coding = colonist_skills.get("coding", 0.0)

    # Existential probe: rare, needs high faith + paranoia
    if faith > 0.6 and paranoia > 0.4 and rng_value < 0.08:
        return "existential_probe"

    # Conflict probe: triggered by active conflicts
    if has_conflict and empathy > 0.3 and rng_value < 0.25:
        return "conflict_probe"

    # Resource forecast: when resources are strained
    if resource_avg < 0.5 and improvisation > 0.3 and rng_value < 0.30:
        return "resource_forecast"

    # Governance test: general purpose, most common
    if coding > 0.2 and improvisation > 0.3 and rng_value < 0.35:
        return "governance_test"

    return None


def max_depth_reached(result: RecursiveResult) -> int:
    """Return the maximum depth achieved in a recursion tree."""
    deepest = result.depth
    for child in result.children:
        deepest = max(deepest, max_depth_reached(child))
    return deepest


def collect_insights(result: RecursiveResult) -> list[dict]:
    """Collect all insights from a recursion tree as flat dicts."""
    insights: list[dict] = []
    for node in result.flatten():
        if node.insight:
            insights.append({
                "scenario": node.scenario,
                "depth": node.depth,
                "colonist_id": node.colonist_id,
                "year": node.year,
                "insight": node.insight,
                "normalised": node.normalised,
            })
    return insights
