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


class Env(dict):
    """Lexically scoped environment."""

    def __init__(self, params=(), args=(), outer=None):
        super().__init__(zip(params, args))
        self.outer = outer

    def find(self, var: str) -> Env:
        if var in self:
            return self
        if self.outer is not None:
            return self.outer.find(var)
        raise LispyError(f"undefined symbol: {var}")


class Procedure:
    """A user-defined lambda/function."""

    def __init__(self, params, body, env):
        self.params = params
        self.body = body
        self.env = env


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(source: str) -> list[str]:
    """Tokenize LisPy source into a flat list of tokens."""
    tokens: list[str] = []
    i = 0
    while i < len(source):
        c = source[i]
        if c == ';':
            while i < len(source) and source[i] != '\n':
                i += 1
            continue
        if c.isspace():
            i += 1
            continue
        if c in '()':
            tokens.append(c)
            i += 1
            continue
        if c == "'":
            tokens.append("'")
            i += 1
            continue
        if c == '"':
            j = i + 1
            while j < len(source) and source[j] != '"':
                if source[j] == '\\':
                    j += 1
                j += 1
            tokens.append(source[i:j + 1])
            i = j + 1
            continue
        j = i
        while j < len(source) and source[j] not in ' \t\n\r();"':
            j += 1
        tokens.append(source[i:j])
        i = j
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _atom(token: str) -> Any:
    """Parse a single token into an atom."""
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1].replace('\\"', '"')
    if token == "#t" or token == "true":
        return True
    if token == "#f" or token == "false":
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


def _parse_tokens(tokens: list[str], pos: int) -> tuple[Any, int]:
    """Recursive descent parse from token list."""
    if pos >= len(tokens):
        raise LispyError("unexpected end of input")
    token = tokens[pos]
    if token == "'":
        expr, pos = _parse_tokens(tokens, pos + 1)
        return [Symbol("quote"), expr], pos
    if token == "(":
        lst: list[Any] = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ")":
            expr, pos = _parse_tokens(tokens, pos)
            lst.append(expr)
        if pos >= len(tokens):
            raise LispyError("missing closing parenthesis")
        return lst, pos + 1
    if token == ")":
        raise LispyError("unexpected )")
    return _atom(token), pos + 1


def read_all(source: str) -> list:
    """Parse source into a list of expressions."""
    tokens = tokenize(source)
    results: list[Any] = []
    pos = 0
    while pos < len(tokens):
        expr, pos = _parse_tokens(tokens, pos)
        results.append(expr)
    return results


def read(source: str) -> Any:
    """Parse source, return the first expression."""
    exprs = read_all(source)
    return exprs[0] if exprs else None


# ---------------------------------------------------------------------------
# Evaluation context (shared budget)
# ---------------------------------------------------------------------------

class EvalContext:
    """Shared evaluation budget across the entire sim tree."""

    def __init__(self, max_steps: int = MAX_EVAL_STEPS,
                 max_depth: int = MAX_RECURSION,
                 subsim_depth: int = 0,
                 seed: int | None = None,
                 _subsim_counter: list[int] | None = None):
        self.max_steps = max_steps
        self.steps = 0
        self.max_depth = max_depth
        self.subsim_depth = subsim_depth
        self.seed = seed
        self._rng = _random_mod.Random(seed) if seed is not None else _random_mod.Random()
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
    env["min"] = lambda *a: min(a)
    env["max"] = lambda *a: max(a)
    env["sqrt"] = math.sqrt
    env["pow"] = pow
    env["floor"] = math.floor
    env["ceil"] = math.ceil
    env["round"] = round
    env["pi"] = math.pi
    env["e"] = math.e

    # -- Comparison --------------------------------------------------------
    env["="] = lambda a, b: a == b
    env["!="] = lambda a, b: a != b
    env["<"] = lambda a, b: a < b
    env[">"] = lambda a, b: a > b
    env["<="] = lambda a, b: a <= b
    env[">="] = lambda a, b: a >= b

    # -- Boolean -----------------------------------------------------------
    env["not"] = operator.not_
    env["and"] = lambda *a: all(a)
    env["or"] = lambda *a: any(a)

    # -- List ops ----------------------------------------------------------
    env["list"] = lambda *a: list(a)
    env["cons"] = lambda a, b: [a] + (b if isinstance(b, list) else [b])
    env["car"] = lambda a: a[0]
    env["cdr"] = lambda a: a[1:]
    env["length"] = len
    env["nth"] = lambda lst, i: lst[int(i)]
    env["append"] = lambda *lists: sum((l if isinstance(l, list) else [l] for l in lists), [])
    env["reverse"] = lambda a: list(reversed(a))
    env["sort"] = lambda a: sorted(a)
    env["range"] = lambda *a: list(range(*[int(x) for x in a]))
    env["empty?"] = lambda a: len(a) == 0 if isinstance(a, (list, dict, str)) else a is None

    # -- String ops --------------------------------------------------------
    env["str"] = lambda *a: "".join(str(x) for x in a)
    env["string-length"] = lambda s: len(s)
    env["substring"] = lambda s, a, b=None: s[int(a):int(b)] if b else s[int(a):]
    env["string-append"] = lambda *a: "".join(str(x) for x in a)
    env["string-upcase"] = lambda s: s.upper()
    env["string-downcase"] = lambda s: s.lower()

    # -- Dict ops ----------------------------------------------------------
    env["dict"] = lambda *kv: dict(zip(kv[::2], kv[1::2]))
    env["get"] = lambda d, k, default=None: d.get(k, default) if isinstance(d, dict) else None
    env["assoc"] = lambda d, k, v: {**d, k: v}
    env["dissoc"] = lambda d, k: {kk: vv for kk, vv in d.items() if kk != k}
    env["keys"] = lambda d: list(d.keys())
    env["values"] = lambda d: list(d.values())
    env["has-key?"] = lambda d, k: k in d

    # -- Type predicates ---------------------------------------------------
    env["number?"] = lambda a: isinstance(a, (int, float))
    env["string?"] = lambda a: isinstance(a, str) and not isinstance(a, Symbol)
    env["list?"] = lambda a: isinstance(a, list)
    env["dict?"] = lambda a: isinstance(a, dict)
    env["nil?"] = lambda a: a is None
    env["symbol?"] = lambda a: isinstance(a, Symbol)
    env["procedure?"] = lambda a: callable(a) or isinstance(a, Procedure)
    env["boolean?"] = lambda a: isinstance(a, bool)

    # -- Type conversion ---------------------------------------------------
    env["int"] = lambda a: int(a)
    env["float"] = lambda a: float(a)
    env["string"] = str

    # -- Random (deterministic via ctx._rng) -------------------------------
    rng = ctx._rng
    env["random-float"] = lambda: rng.random()
    env["random-int"] = lambda a, b: rng.randint(int(a), int(b))
    env["random-choice"] = lambda lst: rng.choice(lst)
    env["random-gauss"] = lambda mu, sigma: rng.gauss(mu, sigma)

    # -- I/O stubs (safe — collect, don't print) ---------------------------
    _output: list[str] = []
    env["display"] = lambda *a: _output.append(" ".join(str(x) for x in a))
    env["print"] = lambda *a: _output.append(" ".join(str(x) for x in a))
    env["_output"] = _output

    # -- Constants ---------------------------------------------------------
    env["true"] = True
    env["false"] = False
    env["nil"] = None

    return env


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate(expr: Any, env: Env, ctx: EvalContext, depth: int = 0) -> Any:
    """Evaluate a LisPy expression in the given environment."""
    if depth > ctx.max_depth:
        raise DepthExceeded(f"recursion depth {depth} exceeds max {ctx.max_depth}")
    ctx.charge()

    # Atom
    if isinstance(expr, Symbol):
        return env.find(str(expr))[str(expr)]
    if not isinstance(expr, list):
        return expr

    if len(expr) == 0:
        return None

    head = expr[0]

    # Special forms
    if isinstance(head, Symbol):
        name = str(head)

        if name == "quote":
            return expr[1]

        if name == "if":
            test = evaluate(expr[1], env, ctx, depth + 1)
            if test:
                return evaluate(expr[2], env, ctx, depth + 1)
            elif len(expr) > 3:
                return evaluate(expr[3], env, ctx, depth + 1)
            return None

        if name == "cond":
            for clause in expr[1:]:
                if isinstance(clause[0], Symbol) and str(clause[0]) == "else":
                    return evaluate(clause[1], env, ctx, depth + 1)
                if evaluate(clause[0], env, ctx, depth + 1):
                    return evaluate(clause[1], env, ctx, depth + 1)
            return None

        if name == "define":
            if isinstance(expr[1], list):
                fname = str(expr[1][0])
                params = [str(p) for p in expr[1][1:]]
                body = expr[2]
                env[fname] = Procedure(params, body, env)
                return None
            else:
                var = str(expr[1])
                val = evaluate(expr[2], env, ctx, depth + 1)
                env[var] = val
                return val

        if name == "set!":
            var = str(expr[1])
            val = evaluate(expr[2], env, ctx, depth + 1)
            target = env.find(var)
            target[var] = val
            return val

        if name == "lambda":
            params = [str(p) for p in expr[1]]
            body = expr[2]
            return Procedure(params, body, env)

        if name == "let":
            bindings = expr[1]
            body = expr[2:]
            inner = Env(outer=env)
            for b in bindings:
                inner[str(b[0])] = evaluate(b[1], env, ctx, depth + 1)
            result = None
            for b_expr in body:
                result = evaluate(b_expr, inner, ctx, depth + 1)
            return result

        if name == "begin":
            result = None
            for sub in expr[1:]:
                result = evaluate(sub, env, ctx, depth + 1)
            return result

        if name == "do":
            result = None
            for sub in expr[1:]:
                result = evaluate(sub, env, ctx, depth + 1)
            return result

        if name == "and":
            result: Any = True
            for sub in expr[1:]:
                result = evaluate(sub, env, ctx, depth + 1)
                if not result:
                    return result
            return result

        if name == "or":
            for sub in expr[1:]:
                result = evaluate(sub, env, ctx, depth + 1)
                if result:
                    return result
            return False

        # map / filter / reduce — charge per element
        if name == "map":
            fn = evaluate(expr[1], env, ctx, depth + 1)
            lst = evaluate(expr[2], env, ctx, depth + 1)
            out = []
            for item in lst:
                ctx.charge()
                out.append(_apply(fn, [item], env, ctx, depth + 1))
            return out

        if name == "filter":
            fn = evaluate(expr[1], env, ctx, depth + 1)
            lst = evaluate(expr[2], env, ctx, depth + 1)
            out = []
            for item in lst:
                ctx.charge()
                if _apply(fn, [item], env, ctx, depth + 1):
                    out.append(item)
            return out

        if name == "reduce":
            fn = evaluate(expr[1], env, ctx, depth + 1)
            lst = evaluate(expr[2], env, ctx, depth + 1)
            acc = evaluate(expr[3], env, ctx, depth + 1)
            for item in lst:
                ctx.charge()
                acc = _apply(fn, [acc, item], env, ctx, depth + 1)
            return acc

        # sub-sim — spawn an isolated child simulation
        if name == "sub-sim":
            child_ctx = ctx.spawn_subsim()
            child_env = _deep_copy_env(env)
            result = None
            for sub in expr[1:]:
                result = evaluate(sub, child_env, child_ctx, 0)
            # charge sub-sim steps back to parent
            ctx.steps += child_ctx.steps
            return result

    # Function call
    fn = evaluate(head, env, ctx, depth + 1)
    args = [evaluate(a, env, ctx, depth + 1) for a in expr[1:]]
    return _apply(fn, args, env, ctx, depth + 1)


def _apply(fn: Any, args: list, env: Env, ctx: EvalContext, depth: int) -> Any:
    """Apply a function (builtin or Procedure) to arguments."""
    if isinstance(fn, Procedure):
        if len(args) != len(fn.params):
            raise LispyError(
                f"expected {len(fn.params)} args, got {len(args)}"
            )
        inner = Env(fn.params, args, fn.env)
        return evaluate(fn.body, inner, ctx, depth)
    if callable(fn):
        try:
            return fn(*args)
        except TypeError as e:
            raise LispyError(str(e))
    raise LispyError(f"not callable: {fn}")


def _deep_copy_env(env: Env) -> Env:
    """Deep copy an environment chain for sub-sim isolation."""
    new_env = Env()
    for key, val in env.items():
        if isinstance(val, (int, float, str, bool, type(None))):
            new_env[key] = val
        elif isinstance(val, list):
            new_env[key] = copy.deepcopy(val)
        elif isinstance(val, dict):
            new_env[key] = copy.deepcopy(val)
        elif callable(val) or isinstance(val, Procedure):
            new_env[key] = val  # share callables (they're stateless)
        else:
            new_env[key] = val
    if env.outer is not None:
        new_env.outer = _deep_copy_env(env.outer)
    return new_env


# ---------------------------------------------------------------------------
# S-expression serialization
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
        return f"{obj:.6g}" if obj != int(obj) else str(int(obj))
    if isinstance(obj, str):
        return f'"{obj}"'
    if isinstance(obj, list):
        return "(" + " ".join(to_sexp(x) for x in obj) + ")"
    if isinstance(obj, dict):
        pairs = " ".join(f'"{k}" {to_sexp(v)}' for k, v in obj.items())
        return f"(dict {pairs})"
    return str(obj)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_env(seed: int | None = None) -> tuple[Env, EvalContext]:
    """Create a fresh standard environment + context pair."""
    ctx = EvalContext(seed=seed)
    env = _make_standard_env(ctx)
    return env, ctx


def run(source: str, seed: int | None = None) -> Any:
    """Parse and evaluate a LisPy program. Returns the last expression's value."""
    env, ctx = make_env(seed=seed)
    exprs = read_all(source)
    result: Any = None
    for expr in exprs:
        result = evaluate(expr, env, ctx)
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
