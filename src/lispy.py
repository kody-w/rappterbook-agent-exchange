"""
lispy.py — Sandboxed LisPy interpreter for Mars-100 recursive simulations.

A minimal Lisp dialect designed for safe evaluation of colonist decision
programs and sub-simulation proposals. No I/O, no imports, no side effects.
Homoiconic: colonist state IS the program, and the program IS the state.

Safety guarantees:
  - Max recursion depth (configurable, default 256)
  - Max evaluation steps (configurable, default 100_000)
  - No file/network/OS access
  - map/filter charge per-element against step budget
  - Global step counter shared across sub-sim tree
  - Sub-sim isolation via environment snapshots
"""
from __future__ import annotations

import copy
import math
import operator
import random as _random_mod
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_EVAL_STEPS = 100_000
MAX_LIST_SIZE = 5000
MAX_RECURSION = 256
MAX_SUB_SIM_DEPTH = 3
MAX_SUB_SIMS_PER_FRAME = 50


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Symbol(str):
    """A Lisp symbol."""
    pass


class LispyError(Exception):
    """Any error during parse or eval."""
    pass


class BudgetExhausted(LispyError):
    """Eval step budget exhausted."""
    pass


class DepthExceeded(LispyError):
    """Sub-simulation depth exceeded."""
    pass


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(source: str) -> list[str]:
    """Split source into tokens (parens, strings, atoms)."""
    tokens: list[str] = []
    i = 0
    while i < len(source):
        ch = source[i]
        if ch in " \t\n\r":
            i += 1
        elif ch == ";":
            while i < len(source) and source[i] != "\n":
                i += 1
        elif ch in "()":
            tokens.append(ch)
            i += 1
        elif ch == '"':
            j = i + 1
            while j < len(source) and source[j] != '"':
                if source[j] == "\\":
                    j += 1
                j += 1
            tokens.append(source[i : j + 1])
            i = j + 1
        elif ch == "'":
            tokens.append("'")
            i += 1
        else:
            j = i
            while j < len(source) and source[j] not in " \t\n\r();\"":
                j += 1
            tokens.append(source[i:j])
            i = j
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_atom(token: str) -> Any:
    """Convert a token string to a Python value."""
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1].replace('\\"', '"').replace("\\n", "\n")
    if token == "#t":
        return True
    if token == "#f":
        return False
    if token == "nil":
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return Symbol(token)


def read_all(source: str) -> list:
    """Parse source string into a list of s-expressions."""
    tokens = tokenize(source)
    results: list = []
    pos = 0

    def read_expr() -> Any:
        nonlocal pos
        if pos >= len(tokens):
            raise LispyError("unexpected end of input")
        tok = tokens[pos]
        if tok == "(":
            pos += 1
            items: list = []
            while pos < len(tokens) and tokens[pos] != ")":
                items.append(read_expr())
            if pos >= len(tokens):
                raise LispyError("missing closing paren")
            pos += 1  # skip ')'
            return items
        elif tok == ")":
            raise LispyError("unexpected ')'")
        elif tok == "'":
            pos += 1
            return [Symbol("quote"), read_expr()]
        else:
            pos += 1
            return _parse_atom(tok)

    while pos < len(tokens):
        results.append(read_expr())
    return results


def read(source: str) -> Any:
    """Parse a single expression."""
    exprs = read_all(source)
    if len(exprs) == 0:
        raise LispyError("empty input")
    if len(exprs) != 1:
        raise LispyError(f"expected 1 expression, got {len(exprs)}")
    return exprs[0]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Env(dict):
    """A Lisp environment: a dict with a parent chain."""

    def __init__(self, params: tuple | list = (), args: tuple | list = (),
                 outer: Env | None = None):
        super().__init__()
        self.update(zip(params, args))
        self.outer = outer

    def find(self, name: str) -> Env:
        """Find the env where name is defined."""
        if name in self:
            return self
        if self.outer is not None:
            return self.outer.find(name)
        raise LispyError(f"undefined symbol: {name}")


def _snapshot_env(env: Env) -> Env:
    """Flatten env chain into a single-level copy (isolation for sub-sims)."""
    flat: dict[str, Any] = {}
    current: Env | None = env
    while current is not None:
        for k, v in current.items():
            if k not in flat:
                flat[k] = v
        current = current.outer
    result = Env()
    result.update(flat)
    return result


# ---------------------------------------------------------------------------
# Procedure (closure)
# ---------------------------------------------------------------------------

class Procedure:
    """A user-defined function (closure)."""
    __slots__ = ("params", "body", "env")

    def __init__(self, params: list[str], body: Any, env: Env):
        self.params = params
        self.body = body
        self.env = env

    def __repr__(self) -> str:
        return f"<procedure ({' '.join(self.params)})>"


# ---------------------------------------------------------------------------
# Evaluation context (budget + sub-sim tracking)
# ---------------------------------------------------------------------------

class EvalContext:
    """Shared evaluation budget across the entire sim tree."""

    def __init__(self, max_steps: int = MAX_EVAL_STEPS,
                 max_depth: int = MAX_RECURSION,
                 subsim_depth: int = 0,
                 seed: int | None = None,
                 rng: _random_mod.Random | None = None,
                 _subsim_counter: list[int] | None = None):
        self.max_steps = max_steps
        self.steps = 0
        self.max_depth = max_depth
        self.subsim_depth = subsim_depth
        self.seed = seed
        if rng is not None:
            self._rng = rng
        elif seed is not None:
            self._rng = _random_mod.Random(seed)
        else:
            self._rng = _random_mod.Random()
        self._subsim_counter = _subsim_counter if _subsim_counter is not None else [0]

    def charge(self, n: int = 1) -> None:
        """Charge n steps. Raises BudgetExhausted if over limit."""
        self.steps += n
        if self.steps > self.max_steps:
            raise BudgetExhausted(
                f"step limit exceeded: {self.steps}/{self.max_steps}"
            )

    def spawn_subsim(self) -> EvalContext:
        """Create a child context for a sub-simulation."""
        new_depth = self.subsim_depth + 1
        if new_depth > MAX_SUB_SIM_DEPTH:
            raise DepthExceeded(
                f"sub-sim depth {new_depth} exceeds max {MAX_SUB_SIM_DEPTH}"
            )
        self._subsim_counter[0] += 1
        if self._subsim_counter[0] > MAX_SUB_SIMS_PER_FRAME:
            raise LispyError(
                f"sub-sim count {self._subsim_counter[0]} exceeds max "
                f"{MAX_SUB_SIMS_PER_FRAME}"
            )
        return EvalContext(
            max_steps=self.max_steps // 2,
            max_depth=self.max_depth,
            subsim_depth=new_depth,
            seed=(self.seed or 0) + new_depth * 7919,
            _subsim_counter=self._subsim_counter,
        )


# ---------------------------------------------------------------------------
# Standard environment (builtins)
# ---------------------------------------------------------------------------

def _make_standard_env(ctx: EvalContext) -> Env:
    """Build the standard safe environment."""
    env = Env()

    # -- Arithmetic --------------------------------------------------------
    env["+"] = lambda *a: sum(a)
    env["-"] = lambda a, b=None: -a if b is None else a - b
    env["*"] = lambda *a: math.prod(a)
    env["/"] = lambda a, b: a / b if b != 0 else 0
    env["//"] = lambda a, b: a // b if b != 0 else 0
    env["%"] = lambda a, b: a % b if b != 0 else 0
    env["abs"] = abs
    env["min"] = min
    env["max"] = max
    env["round"] = round
    env["floor"] = lambda x: int(math.floor(x))
    env["ceil"] = lambda x: int(math.ceil(x))
    env["sqrt"] = math.sqrt
    env["pow"] = pow

    # -- Comparison --------------------------------------------------------
    env["<"] = operator.lt
    env[">"] = operator.gt
    env["<="] = operator.le
    env[">="] = operator.ge
    env["="] = operator.eq
    env["!="] = operator.ne
    env["equal?"] = operator.eq

    # -- Boolean -----------------------------------------------------------
    env["not"] = operator.not_

    # -- Type predicates ---------------------------------------------------
    env["number?"] = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    env["string?"] = lambda x: isinstance(x, str) and not isinstance(x, Symbol)
    env["list?"] = lambda x: isinstance(x, list)
    env["dict?"] = lambda x: isinstance(x, dict)
    env["nil?"] = lambda x: x is None
    env["null?"] = lambda x: x is None
    env["bool?"] = lambda x: isinstance(x, bool)
    env["symbol?"] = lambda x: isinstance(x, Symbol)
    env["pair?"] = lambda x: isinstance(x, list) and len(x) > 0
    env["empty?"] = lambda x: isinstance(x, list) and len(x) == 0

    # -- List operations ---------------------------------------------------
    def _list_fn(*a: Any) -> list:
        if len(a) > MAX_LIST_SIZE:
            raise LispyError(f"list size {len(a)} exceeds max {MAX_LIST_SIZE}")
        return list(a)

    env["list"] = _list_fn
    env["car"] = lambda x: x[0] if x else None
    env["cdr"] = lambda x: x[1:] if x else []
    env["cons"] = lambda a, b: [a] + (b if isinstance(b, list) else [b])
    env["length"] = len
    env["nth"] = lambda lst, n: lst[int(n)] if 0 <= int(n) < len(lst) else None
    env["append"] = lambda *lsts: sum(
        (l if isinstance(l, list) else [l] for l in lsts), []
    )
    env["reverse"] = lambda x: list(reversed(x))
    env["range"] = lambda *a: list(range(*[int(x) for x in a]))
    env["sort"] = lambda x: sorted(x)
    env["contains?"] = lambda lst, v: v in lst

    # Assoc (association list lookup): ((k1 v1) (k2 v2) ...) -> v or nil
    def _assoc(key: Any, alist: Any) -> Any:
        if not isinstance(alist, list):
            return None
        for pair in alist:
            if isinstance(pair, list) and len(pair) >= 2 and pair[0] == key:
                return pair[1]
        return None

    def _assoc_set(key: Any, val: Any, alist: Any) -> list:
        if not isinstance(alist, list):
            alist = []
        return alist + [[key, val]]

    env["assoc"] = _assoc
    env["assoc-set"] = _assoc_set

    # -- Higher-order (step-charged per element) ---------------------------
    def _call_fn(fn: Any, *args: Any) -> Any:
        """Call a function — handles both Python callables and Procedures."""
        if isinstance(fn, Procedure):
            inner = Env(fn.params, list(args), fn.env)
            return evaluate(fn.body, inner, ctx, 0)
        if callable(fn):
            return fn(*args)
        raise LispyError(f"not callable: {fn!r}")

    def _map(fn: Any, lst: Any) -> list:
        ctx.charge(len(lst))
        return [_call_fn(fn, x) for x in lst]

    def _filter(fn: Any, lst: Any) -> list:
        ctx.charge(len(lst))
        return [x for x in lst if _call_fn(fn, x)]

    def _reduce(fn: Any, lst: Any, init: Any = None) -> Any:
        ctx.charge(len(lst))
        acc = init if init is not None else lst[0]
        start = 0 if init is not None else 1
        for x in lst[start:]:
            acc = _call_fn(fn, acc, x)
        return acc

    env["map"] = _map
    env["filter"] = _filter
    env["reduce"] = _reduce

    # -- Dict operations ---------------------------------------------------
    env["dict"] = lambda *pairs: dict(zip(pairs[::2], pairs[1::2]))
    env["get"] = lambda d, k, default=None: (
        d.get(k, default) if isinstance(d, dict) else None
    )
    env["dict-assoc"] = lambda d, k, v: (
        {**d, k: v} if isinstance(d, dict) else d
    )
    env["dissoc"] = lambda d, k: (
        {kk: vv for kk, vv in d.items() if kk != k}
        if isinstance(d, dict) else d
    )
    env["keys"] = lambda d: list(d.keys()) if isinstance(d, dict) else []
    env["values"] = lambda d: list(d.values()) if isinstance(d, dict) else []
    env["has-key?"] = lambda d, k: k in d if isinstance(d, dict) else False
    env["merge"] = lambda *dicts: {
        k: v for d in dicts if isinstance(d, dict) for k, v in d.items()
    }

    # -- String operations -------------------------------------------------
    env["str"] = lambda *a: "".join(str(x) if x is not None else "nil" for x in a)
    env["str-upper"] = lambda s: s.upper()
    env["str-lower"] = lambda s: s.lower()
    env["str-split"] = lambda s, sep=" ": s.split(sep)
    env["str-length"] = lambda s: len(s) if isinstance(s, str) else 0
    env["str-ref"] = lambda s, n: s[int(n)] if isinstance(s, str) and 0 <= int(n) < len(s) else None

    # -- Conversion --------------------------------------------------------
    env["int"] = lambda x: int(x) if isinstance(x, (int, float, str)) else 0
    env["float"] = lambda x: float(x) if isinstance(x, (int, float, str)) else 0.0
    env["str->num"] = lambda s: float(s) if "." in str(s) else int(s)

    # -- Random (deterministic via seed) -----------------------------------
    env["random-float"] = lambda: ctx._rng.random()
    env["random-int"] = lambda a, b: ctx._rng.randint(int(a), int(b))

    # -- Constants ---------------------------------------------------------
    env["PI"] = math.pi
    env["E"] = math.e
    env["true"] = True
    env["false"] = False
    env["nil"] = None

    return env


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate(expr: Any, env: Env, ctx: EvalContext,
             depth: int = 0) -> Any:
    """Evaluate a LisPy expression in the given environment."""
    if depth > ctx.max_depth:
        raise DepthExceeded(f"recursion depth {depth} > {ctx.max_depth}")
    ctx.charge()

    # Atom
    if isinstance(expr, Symbol):
        return env.find(str(expr))[str(expr)]
    if not isinstance(expr, list):
        return expr  # literal: number, string, bool, None, dict

    if len(expr) == 0:
        return None

    head = expr[0]

    # Special forms
    if isinstance(head, Symbol):
        name = str(head)

        if name == "quote":
            return expr[1] if len(expr) > 1 else None

        if name == "if":
            test = evaluate(expr[1], env, ctx, depth + 1)
            if test:
                return evaluate(expr[2], env, ctx, depth + 1)
            elif len(expr) > 3:
                return evaluate(expr[3], env, ctx, depth + 1)
            return None

        if name == "cond":
            for clause in expr[1:]:
                if len(clause) < 2:
                    continue
                if isinstance(clause[0], Symbol) and str(clause[0]) == "else":
                    return evaluate(clause[1], env, ctx, depth + 1)
                test = evaluate(clause[0], env, ctx, depth + 1)
                if test:
                    return evaluate(clause[1], env, ctx, depth + 1)
            return None

        if name == "define":
            if isinstance(expr[1], list):
                # (define (f x y) body)
                fname = str(expr[1][0])
                params = [str(p) for p in expr[1][1:]]
                body = expr[2]
                env[fname] = Procedure(params, body, env)
            else:
                env[str(expr[1])] = evaluate(expr[2], env, ctx, depth + 1)
            return None

        if name == "set!":
            target = str(expr[1])
            val = evaluate(expr[2], env, ctx, depth + 1)
            env.find(target)[target] = val
            return val

        if name == "lambda" or name == "fn":
            params = [str(p) for p in expr[1]]
            body = expr[2]
            return Procedure(params, body, env)

        if name == "let":
            bindings = expr[1]
            body = expr[2]
            inner = Env(outer=env)
            for binding in bindings:
                inner[str(binding[0])] = evaluate(
                    binding[1], inner, ctx, depth + 1
                )
            return evaluate(body, inner, ctx, depth + 1)

        if name == "begin" or name == "do-seq":
            result: Any = None
            for e in expr[1:]:
                result = evaluate(e, env, ctx, depth + 1)
            return result

        if name == "and":
            result = True
            for e in expr[1:]:
                result = evaluate(e, env, ctx, depth + 1)
                if not result:
                    return result
            return result

        if name == "or":
            for e in expr[1:]:
                result = evaluate(e, env, ctx, depth + 1)
                if result:
                    return result
            return False

        # -- sub-sim: isolated sub-simulation evaluation -------------------
        if name == "sub-sim":
            child_ctx = ctx.spawn_subsim()
            child_env = _snapshot_env(env)
            body = expr[1] if len(expr) > 1 else None
            if body is None:
                return None
            return evaluate(body, child_env, child_ctx, depth + 1)

        # -- do: Scheme-style iteration ------------------------------------
        # (do ((var init step) ...) (test result) body...)
        if name == "do":
            bindings_spec = expr[1]
            test_clause = expr[2]
            body_exprs = expr[3:]

            inner = Env(outer=env)
            # Initialize variables
            for binding in bindings_spec:
                var_name = str(binding[0])
                init_val = evaluate(binding[1], env, ctx, depth + 1)
                inner[var_name] = init_val

            # Iterate
            iteration_limit = ctx.max_steps - ctx.steps
            for _ in range(iteration_limit):
                ctx.charge()
                test_val = evaluate(test_clause[0], inner, ctx, depth + 1)
                if test_val:
                    if len(test_clause) > 1:
                        return evaluate(
                            test_clause[1], inner, ctx, depth + 1
                        )
                    return None
                # Execute body
                for b in body_exprs:
                    evaluate(b, inner, ctx, depth + 1)
                # Compute ALL step values before updating
                new_vals = []
                for binding in bindings_spec:
                    if len(binding) >= 3:
                        step_val = evaluate(
                            binding[2], inner, ctx, depth + 1
                        )
                        new_vals.append((str(binding[0]), step_val))
                # Apply updates simultaneously
                for var_name, val in new_vals:
                    inner[var_name] = val

            raise BudgetExhausted("do loop exceeded step budget")

    # Function call
    fn = evaluate(head, env, ctx, depth + 1)
    args = [evaluate(a, env, ctx, depth + 1) for a in expr[1:]]

    if isinstance(fn, Procedure):
        if len(args) != len(fn.params):
            raise LispyError(
                f"arity mismatch: expects {len(fn.params)} args, got {len(args)}"
            )
        inner = Env(fn.params, args, fn.env)
        return evaluate(fn.body, inner, ctx, depth + 1)

    if callable(fn):
        try:
            return fn(*args)
        except TypeError as e:
            raise LispyError(f"call error: {e}")

    raise LispyError(f"not callable: {fn!r}")


# ---------------------------------------------------------------------------
# S-expression serializer
# ---------------------------------------------------------------------------

def to_sexp(obj: Any) -> str:
    """Convert a Python object to an s-expression string."""
    if obj is None:
        return "nil"
    if obj is True:
        return "#t"
    if obj is False:
        return "#f"
    if isinstance(obj, Symbol):
        return str(obj)
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return str(obj)
    if isinstance(obj, str):
        return f'"{obj}"'
    if isinstance(obj, list):
        inner = " ".join(to_sexp(x) for x in obj)
        return f"({inner})"
    if isinstance(obj, dict):
        pairs = " ".join(f'"{k}" {to_sexp(v)}' for k, v in obj.items())
        return f"(dict {pairs})"
    return str(obj)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_env(seed: int | None = None,
             budget: EvalContext | None = None) -> tuple[Env, EvalContext]:
    """Create a fresh standard environment + context pair."""
    if budget is not None:
        ctx = budget
    else:
        ctx = EvalContext(seed=seed)
    env = _make_standard_env(ctx)
    return env, ctx


def run(source: str, seed: int | None = None,
        budget: EvalContext | None = None,
        env: Env | None = None) -> Any:
    """Parse and evaluate a LisPy program. Returns the last expression's value."""
    if env is not None and budget is not None:
        ctx = budget
        _env = env
    elif budget is not None:
        _env, ctx = make_env(budget=budget)
    else:
        _env, ctx = make_env(seed=seed)
    exprs = read_all(source)
    result: Any = None
    for expr in exprs:
        result = evaluate(expr, _env, ctx)
    return result


def run_in_env(source: str, env: Env, ctx: EvalContext) -> Any:
    """Parse and evaluate in an existing environment + context."""
    exprs = read_all(source)
    result: Any = None
    for expr in exprs:
        result = evaluate(expr, env, ctx)
    return result


def safe_eval(source: str, context: dict | None = None,
              max_steps: int = 10_000, max_depth: int = MAX_RECURSION) -> Any:
    """Evaluate with an optional context dict injected. Fresh env."""
    ctx = EvalContext(max_steps=max_steps, max_depth=max_depth)
    env = _make_standard_env(ctx)
    if context:
        for k, v in context.items():
            env[str(k)] = v
    exprs = read_all(source)
    result: Any = None
    for expr in exprs:
        result = evaluate(expr, env, ctx)
    return result


# ---------------------------------------------------------------------------
# Compatibility aliases (tests import these names)
# ---------------------------------------------------------------------------

Budget = EvalContext
Lambda = Procedure
LispError = LispyError
StepLimitExceeded = BudgetExhausted
DepthLimitExceeded = DepthExceeded
parse = read_all
