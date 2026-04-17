"""
The Dreaming Engine — colonists that rewrite themselves.

Colonists are LisPy data structures AND LisPy programs.  The homoiconic
property means they can rewrite their own ``decision_expr`` through a
sub-simulation-driven genetic programming loop.

During a *dream*, a colonist:
1. Generates mutations of their decision expression at the AST level.
2. Screens each mutation via lightweight LisPy VM evaluation.
3. Optionally validates the best candidate through a proper sub-sim.
4. Adopts the mutation only if it strictly improves fitness.

All mutations produce valid, evaluable LisPy that returns a finite numeric
scalar.  Expression bloat is prevented by an AST node cap.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.lispy_vm import (
    LispyError,
    parse_all,
    _count_nodes,
    evaluate,
    make_env,
    VMState,
    MAX_AST_NODES,
)
from src.mars100.subsim import SubSimBudget, SubSimResult, spawn_subsim

MAX_EXPR_NODES = 60
ADOPTION_THRESHOLD = 0.05
DREAM_CHANCE_BASE = 0.20
CANDIDATES_PER_DREAM = 3
SAMPLE_ROUNDS = 5


@dataclass
class DreamResult:
    """Outcome of one colonist's dream session."""
    colonist_id: str
    year: int
    old_expr: str
    new_expr: str | None
    adopted: bool
    fitness_delta: float
    candidates_tried: int
    subsim_used: bool = False

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "colonist_id": self.colonist_id,
            "year": self.year,
            "old_expr": self.old_expr,
            "adopted": self.adopted,
            "fitness_delta": round(self.fitness_delta, 4),
            "candidates_tried": self.candidates_tried,
            "subsim_used": self.subsim_used,
        }
        if self.new_expr is not None:
            d["new_expr"] = self.new_expr
        return d


# ---------------------------------------------------------------------------
# AST manipulation
# ---------------------------------------------------------------------------

def _serialize(ast: Any) -> str:
    """Convert a parsed AST node back to a LisPy source string."""
    if isinstance(ast, list):
        return "(" + " ".join(_serialize(a) for a in ast) + ")"
    if isinstance(ast, bool):
        return "#t" if ast else "#f"
    if ast is None:
        return "nil"
    if isinstance(ast, float):
        return f"{ast:.4f}"
    return str(ast)


def _collect_numbers(ast: Any) -> list[tuple[list, int]]:
    """Find (parent_list, index) pairs for every numeric literal in the AST."""
    results: list[tuple[list, int]] = []
    if isinstance(ast, list):
        for i, child in enumerate(ast):
            if isinstance(child, (int, float)) and not isinstance(child, bool):
                results.append((ast, i))
            elif isinstance(child, list):
                results.extend(_collect_numbers(child))
    return results


def _collect_operators(ast: Any) -> list[tuple[list, int]]:
    """Find (parent_list, index) pairs for swappable operators in the AST."""
    swappable = {"+", "-", "*", "/", ">", "<", ">=", "<=", "min", "max"}
    results: list[tuple[list, int]] = []
    if isinstance(ast, list):
        for i, child in enumerate(ast):
            if isinstance(child, str) and child in swappable:
                results.append((ast, i))
            elif isinstance(child, list):
                results.extend(_collect_operators(child))
    return results


SWAP_GROUPS = [
    {"+", "-"},
    {"*", "/"},
    {">", "<"},
    {">=", "<="},
    {"min", "max"},
]


def _swap_operator(op: str, rng: random.Random) -> str:
    """Swap an operator with another from the same group."""
    for group in SWAP_GROUPS:
        if op in group:
            others = [o for o in group if o != op]
            if others:
                return rng.choice(others)
    return op


def mutate_expr(expr: str, rng: random.Random) -> str | None:
    """Generate a single AST-level mutation of a LisPy expression.

    Returns None if the expression cannot be mutated or the result
    is too large.
    """
    try:
        parsed = parse_all(expr)
    except LispyError:
        return None

    if not parsed:
        return None
    ast = copy.deepcopy(parsed[0])

    mutations_available: list[str] = []

    numbers = _collect_numbers(ast)
    if numbers:
        mutations_available.append("point")

    operators = _collect_operators(ast)
    if operators:
        mutations_available.append("swap_op")

    if isinstance(ast, list) and len(ast) >= 3:
        for child in ast:
            if isinstance(child, list) and len(child) >= 3:
                if child[0] == "if":
                    mutations_available.append("swap_branch")
                    break

    if not mutations_available:
        return None

    mutation_type = rng.choice(mutations_available)

    if mutation_type == "point" and numbers:
        parent, idx = rng.choice(numbers)
        old_val = parent[idx]
        delta = rng.gauss(0, 0.15)
        parent[idx] = round(float(old_val) + delta, 4)
    elif mutation_type == "swap_op" and operators:
        parent, idx = rng.choice(operators)
        parent[idx] = _swap_operator(parent[idx], rng)
    elif mutation_type == "swap_branch":
        # Find an if-expression and swap then/else
        _swap_if_branches(ast, rng)

    result = _serialize(ast)

    try:
        reparsed = parse_all(result)
        if not reparsed:
            return None
        if _count_nodes(reparsed[0]) > MAX_EXPR_NODES:
            return None
    except LispyError:
        return None

    return result


def _swap_if_branches(ast: Any, rng: random.Random) -> bool:
    """Find and swap the branches of an if-expression in the AST."""
    if isinstance(ast, list):
        if ast[0] == "if" and len(ast) == 4:
            ast[2], ast[3] = ast[3], ast[2]
            return True
        for child in ast:
            if isinstance(child, list) and _swap_if_branches(child, rng):
                return True
    return False


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def _eval_expr_numeric(expr: str, bindings: dict[str, Any]) -> float | None:
    """Evaluate an expression and return its value only if finite numeric."""
    try:
        env = make_env(bindings)
        parsed = parse_all(expr)
        if not parsed:
            return None
        state = VMState(max_steps=1000, max_depth=50)
        value = evaluate(parsed[0], env, state)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            fval = float(value)
            if not (fval != fval) and abs(fval) < 1e6:  # finite check
                return fval
    except Exception:
        pass
    return None


def _sample_bindings(base: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Create a perturbed copy of bindings for robustness testing."""
    perturbed = dict(base)
    for key, val in base.items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            perturbed[key] = max(0.0, min(1.0, float(val) + rng.gauss(0, 0.1)))
    return perturbed


def evaluate_fitness(expr: str, bindings: dict[str, Any],
                     rng: random.Random) -> float:
    """Evaluate an expression's fitness across multiple binding samples.

    Fitness measures:
    - Consistency: does it return numeric values across varied inputs?
    - Range: does it produce varied outputs (not constant)?
    - Reasonableness: are outputs in a useful range for action weighting?

    Returns a score in [0, 1].  0 = useless, 1 = excellent.
    """
    values: list[float] = []
    for _ in range(SAMPLE_ROUNDS):
        sampled = _sample_bindings(bindings, rng)
        result = _eval_expr_numeric(expr, sampled)
        if result is not None:
            values.append(result)

    if len(values) < SAMPLE_ROUNDS * 0.6:
        return 0.0  # too many failures

    consistency = len(values) / SAMPLE_ROUNDS

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    diversity = min(1.0, (variance ** 0.5) * 5.0)

    in_range = sum(1 for v in values if -2.0 <= v <= 2.0) / len(values)

    return consistency * 0.3 + diversity * 0.3 + in_range * 0.4


# ---------------------------------------------------------------------------
# Dream loop
# ---------------------------------------------------------------------------

def should_dream(colonist_stats: dict[str, float], rng: random.Random) -> bool:
    """Determine if a colonist enters a dream state this year."""
    coding = colonist_stats.get("coding", 0.0)
    faith = colonist_stats.get("faith", 0.0)
    improvisation = colonist_stats.get("improvisation", 0.0)
    chance = DREAM_CHANCE_BASE + coding * 0.15 + faith * 0.10 + improvisation * 0.05
    return rng.random() < chance


def dream(
    colonist_id: str,
    year: int,
    current_expr: str,
    bindings: dict[str, Any],
    budget: SubSimBudget | None,
    rng: random.Random,
) -> DreamResult:
    """Run one colonist's dream session.

    Generates ``CANDIDATES_PER_DREAM`` mutations, evaluates fitness via
    plain VM eval, and adopts the best if it clears the threshold.
    """
    current_fitness = evaluate_fitness(current_expr, bindings, rng)

    best_expr: str | None = None
    best_fitness = current_fitness
    tried = 0

    for _ in range(CANDIDATES_PER_DREAM):
        candidate = mutate_expr(current_expr, rng)
        if candidate is None or candidate == current_expr:
            continue
        tried += 1
        fitness = evaluate_fitness(candidate, bindings, rng)
        if fitness > best_fitness + ADOPTION_THRESHOLD:
            best_fitness = fitness
            best_expr = candidate

    adopted = best_expr is not None
    subsim_used = False

    # Optional: validate via subsim if budget allows
    if adopted and budget is not None and budget.can_spawn(colonist_id):
        assert best_expr is not None
        subsim_used = True
        sr = spawn_subsim(
            expression=best_expr,
            colonist_id=colonist_id,
            year=year,
            bindings=bindings,
            depth=1,
            budget=budget,
        )
        if not sr.succeeded or not isinstance(sr.result, (int, float)):
            adopted = False
            best_expr = None
            best_fitness = current_fitness

    return DreamResult(
        colonist_id=colonist_id,
        year=year,
        old_expr=current_expr,
        new_expr=best_expr if adopted else None,
        adopted=adopted,
        fitness_delta=round(best_fitness - current_fitness, 4),
        candidates_tried=tried,
        subsim_used=subsim_used,
    )
