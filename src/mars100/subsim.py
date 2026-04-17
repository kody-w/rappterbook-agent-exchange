"""
Sub-simulation spawner for Mars-100.

Colonists can spawn sandboxed LisPy sub-simulations to model consequences
before committing to decisions. Max depth 3. All sub-sims are logged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.mars100.lispy_vm import (
    LispyBudgetExceeded,
    LispyError,
    run as lispy_run,
)

MAX_SUBSIM_DEPTH = 3
MAX_SUBSIMS_PER_COLONIST_PER_YEAR = 2
MAX_SUBSIMS_PER_YEAR = 8


@dataclass
class SubSimResult:
    """Result of a sub-simulation."""
    depth: int
    colonist_id: str
    year: int
    expression: str
    result: Any = None
    error: str | None = None
    steps_used: int = 0
    children: list[SubSimResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "depth": self.depth, "colonist_id": self.colonist_id,
            "year": self.year, "expression": self.expression,
            "result": _safe_serialize(self.result), "steps_used": self.steps_used,
        }
        if self.error:
            d["error"] = self.error
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    @property
    def succeeded(self) -> bool:
        return self.error is None


def _safe_serialize(value: Any) -> Any:
    """Make a value JSON-serializable."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_safe_serialize(v) for v in value]
    return str(value)


@dataclass
class SubSimBudget:
    """Track sub-simulation budgets per year."""
    year: int
    colony_total: int = 0
    per_colonist: dict[str, int] = field(default_factory=dict)

    def can_spawn(self, colonist_id: str) -> bool:
        """Check if a colonist can spawn another sub-sim this year."""
        if self.colony_total >= MAX_SUBSIMS_PER_YEAR:
            return False
        count = self.per_colonist.get(colonist_id, 0)
        return count < MAX_SUBSIMS_PER_COLONIST_PER_YEAR

    def record(self, colonist_id: str) -> None:
        """Record a sub-sim spawn."""
        self.colony_total += 1
        self.per_colonist[colonist_id] = self.per_colonist.get(colonist_id, 0) + 1


def spawn_subsim(
    expression: str,
    colonist_id: str,
    year: int,
    bindings: dict[str, Any],
    depth: int = 1,
    budget: SubSimBudget | None = None,
    log: list[SubSimResult] | None = None,
) -> SubSimResult:
    """Spawn a sandboxed sub-simulation."""
    result = SubSimResult(depth=depth, colonist_id=colonist_id,
                          year=year, expression=expression)

    if depth > MAX_SUBSIM_DEPTH:
        result.error = "max depth exceeded"
        if log is not None:
            log.append(result)
        return result

    if budget is not None and not budget.can_spawn(colonist_id):
        result.error = "sub-sim budget exhausted"
        if log is not None:
            log.append(result)
        return result

    if budget is not None:
        budget.record(colonist_id)

    subsim_bindings = dict(bindings)
    subsim_bindings["sim-depth"] = depth
    subsim_bindings["sim-year"] = year

    max_steps = 10000 // depth

    try:
        value = lispy_run(expression, extra_bindings=subsim_bindings,
                          max_steps=max_steps, max_depth=200 // depth)
        result.result = value
    except LispyBudgetExceeded as exc:
        result.error = str(exc)
    except LispyError as exc:
        result.error = str(exc)
    except Exception as exc:
        result.error = str(exc)

    if log is not None:
        log.append(result)
    return result
