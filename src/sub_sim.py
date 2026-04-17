"""sub_sim.py -- Recursive sub-simulation spawner for Mars-100.

Colonists may spawn sandboxed LisPy simulations to model governance
proposals, economic scenarios, or survival strategies before committing.
Max depth 3, per the Turtles All the Way Down doctrine (Amendment XIII).

Python stdlib only.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone

from src.lispy import (
    Env, Budget, Procedure, Symbol,
    default_env, evaluate, parse, run, LispyError, BudgetExhausted,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_sub_sim(
    lispy_code: str,
    parent_env: Env,
    budget: Budget,
    context: dict | None = None,
    rng_seed: int | None = None,
) -> dict:
    """Run a sub-simulation in a sandboxed LisPy environment.

    Args:
        lispy_code: LisPy source to evaluate
        parent_env: Parent environment (read from, not written to)
        budget: SHARED budget — depth/steps count against parent
        context: Optional context dict injected as 'sim-context' binding
        rng_seed: Optional seed for deterministic sub-sim randomness

    Returns: dict with depth, result, steps_used, success, error (if any)
    """
    steps_before = budget.steps_used
    depth_before = budget.current_depth

    try:
        budget.enter_sub_sim()
    except BudgetExhausted as exc:
        return {
            'depth': budget.current_depth,
            'result': None,
            'steps_used': 0,
            'success': False,
            'error': str(exc),
            'timestamp': now_iso(),
        }

    try:
        # Create sandboxed environment (no parent writes)
        sandbox = default_env()
        # Copy parent bindings (shallow — mutations don't leak)
        for key, value in parent_env.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                sandbox[key] = value
            elif isinstance(value, (list, dict)):
                sandbox[key] = copy.deepcopy(value)

        # Inject context
        if context:
            sandbox[Symbol('sim-context')] = _dict_to_assoc(context)
            sandbox[Symbol('sim-depth')] = budget.current_depth
            if rng_seed is not None:
                sandbox[Symbol('sim-seed')] = rng_seed

        ast = parse(lispy_code)
        result = evaluate(ast, sandbox, budget)

        return {
            'depth': budget.current_depth,
            'result': _sanitize_result(result),
            'steps_used': budget.steps_used - steps_before,
            'success': True,
            'error': None,
            'timestamp': now_iso(),
        }

    except (LispyError, BudgetExhausted) as exc:
        return {
            'depth': budget.current_depth,
            'result': None,
            'steps_used': budget.steps_used - steps_before,
            'success': False,
            'error': str(exc),
            'timestamp': now_iso(),
        }
    finally:
        budget.exit_sub_sim()


def _dict_to_assoc(d: dict) -> list:
    """Convert a Python dict to a LisPy assoc list."""
    return [[Symbol(str(k)), v] for k, v in d.items()]


def _sanitize_result(result) -> object:
    """Ensure result is JSON-serializable."""
    if result is None:
        return None
    if isinstance(result, (int, float, str, bool)):
        return result
    if isinstance(result, Symbol):
        return str(result)
    if isinstance(result, list):
        return [_sanitize_result(item) for item in result]
    if isinstance(result, dict):
        return {str(k): _sanitize_result(v) for k, v in result.items()}
    if isinstance(result, Procedure):
        return f"<lambda ({' '.join(result.params)})>"
    return str(result)


# --- Sub-sim templates for colonist governance modeling ---

GOVERNANCE_SIM_TEMPLATE = """
(begin
  (define sim-population {population})
  (define sim-food {food})
  (define sim-morale {morale})
  (define sim-year {year})
  (define sim-years-forward {years_forward})
  (define food-per-year (* sim-population 100))
  (define simulate-year
    (lambda (yr food-left mor)
      (if (> yr (+ sim-year sim-years-forward))
          (list 'outcome "stable" 'final-food food-left 'final-morale mor)
          (let ((consumed food-per-year))
            (let ((produced (* sim-population 80)))
              (let ((new-food (- (+ food-left produced) consumed)))
                (let ((new-morale (if (< new-food 0) (- mor 20) (+ mor 2))))
                  (if (< new-morale 0)
                      (list 'outcome "collapse" 'year yr 'cause "morale")
                      (if (< new-food (* -2 food-per-year))
                          (list 'outcome "collapse" 'year yr 'cause "famine")
                          (simulate-year (+ yr 1) new-food (min 100 new-morale)))))))))))
  (simulate-year sim-year sim-food sim-morale))
"""

PHILOSOPHY_SIM_TEMPLATE = """
(let ((depth sim-depth)
      (context sim-context))
  (if (>= depth 3)
      (list 'insight "The governance that works is the one that knows it is temporary."
            'meta "At depth 3, all models converge on impermanence."
            'depth depth)
      (list 'insight "Stability requires flexibility."
            'meta "Models at this depth suggest adaptive governance outperforms rigid structures."
            'depth depth)))
"""


def build_governance_sim(
    population: int,
    food: float,
    morale: float,
    year: int,
    years_forward: int = 10,
) -> str:
    """Build a LisPy governance sub-simulation for economic modeling."""
    return GOVERNANCE_SIM_TEMPLATE.format(
        population=population,
        food=int(food),
        morale=int(morale),
        year=year,
        years_forward=years_forward,
    )


def build_philosophy_sim() -> str:
    """Build a depth-aware philosophical sub-simulation."""
    return PHILOSOPHY_SIM_TEMPLATE
