"""
Sub-simulation spawner for Mars-100.

Colonists spawn sandboxed LisPy sub-simulations to model governance
proposals, economic scenarios, or survival strategies before committing.
Max depth: 3 levels. Each level inherits the parent constitution but
may propose amendments within its scope.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from src.mars100.lispy_vm import (
    LispyError, LispyBudgetExceeded,
    run as lispy_run, make_env, VMState, evaluate, parse_all,
)

MAX_SUBSIM_DEPTH = 3
SUBSIM_BUDGET_PER_YEAR = 5  # max sub-sims per year
SUBSIM_MAX_STEPS = 2000     # tighter budget than top-level


@dataclass
class SubSimBudget:
    """Track sub-simulation budget per frame."""
    remaining: int = SUBSIM_BUDGET_PER_YEAR
    total_spawned: int = 0

    def can_spawn(self) -> bool:
        """Check if budget allows another sub-sim."""
        return self.remaining > 0

    def spend(self) -> None:
        """Consume one sub-sim from the budget."""
        self.remaining -= 1
        self.total_spawned += 1

    def reset(self) -> None:
        """Reset per-frame budget."""
        self.remaining = SUBSIM_BUDGET_PER_YEAR


@dataclass
class SubSimResult:
    """Result from a sub-simulation."""
    depth: int
    colonist_id: str
    expression: str
    result: Any
    error: str = ""
    steps_used: int = 0
    nested_sims: int = 0

    def to_dict(self) -> dict:
        """Serialize."""
        return {
            "depth": self.depth,
            "colonist_id": self.colonist_id,
            "expression": self.expression[:200],
            "result": _safe_serialize(self.result),
            "error": self.error,
            "steps_used": self.steps_used,
            "nested_sims": self.nested_sims,
        }


def _safe_serialize(value: Any) -> Any:
    """Safely serialize a LisPy value to JSON-compatible form."""
    if value is None:
        return None
    if isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, list):
        return [_safe_serialize(v) for v in value[:20]]
    return str(value)[:100]


def spawn_subsim(expression: str, colonist_id: str,
                 context_bindings: dict,
                 depth: int = 1,
                 rng: random.Random = None) -> SubSimResult:
    """Spawn a sandboxed LisPy sub-simulation.

    Each sub-sim gets a fresh environment with context bindings
    injected (resource levels, social data, etc.) but NO access
    to the parent's mutable state.
    """
    if depth > MAX_SUBSIM_DEPTH:
        return SubSimResult(
            depth=depth, colonist_id=colonist_id,
            expression=expression,
            result=None,
            error=f"max depth {MAX_SUBSIM_DEPTH} exceeded",
        )

    # Build context: inject read-only colony state
    bindings = dict(context_bindings)
    nested_count = 0

    # Allow nested sub-sim at depth+1 via a callable
    if depth < MAX_SUBSIM_DEPTH:
        def sub_sim_fn(expr_str):
            nonlocal nested_count
            nested_count += 1
            if nested_count > 3:
                return "sub-sim-budget-exceeded"
            inner = spawn_subsim(
                str(expr_str), colonist_id,
                context_bindings, depth + 1, rng,
            )
            return inner.result
        bindings["sub-sim"] = sub_sim_fn

    try:
        result = lispy_run(
            expression,
            extra_bindings=bindings,
            max_steps=SUBSIM_MAX_STEPS,
            max_depth=100,
        )
        return SubSimResult(
            depth=depth, colonist_id=colonist_id,
            expression=expression,
            result=result,
            nested_sims=nested_count,
        )
    except LispyBudgetExceeded as exc:
        return SubSimResult(
            depth=depth, colonist_id=colonist_id,
            expression=expression,
            result=None, error=f"budget: {exc}",
            nested_sims=nested_count,
        )
    except LispyError as exc:
        return SubSimResult(
            depth=depth, colonist_id=colonist_id,
            expression=expression,
            result=None, error=str(exc),
            nested_sims=nested_count,
        )
