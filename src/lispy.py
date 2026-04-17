"""
LisPy — A safe-eval Lisp interpreter for sub-simulations.

Used by Mars-100 colonists to model governance proposals, economic
scenarios, and survival strategies before committing. Sub-simulations
are sandboxed: no I/O, no imports, pure computation.

Supports: atoms, numbers, strings, booleans, lists, lambda, define,
if, cond, let, begin, quote, arithmetic, comparison, list ops,
and sub-sim spawning with depth tracking.

The homoiconic property (code = data) means colonists can rewrite
themselves — their state IS a LisPy expression, and their decisions
ARE LisPy programs that transform that state.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

class Symbol:
    """An interned symbol (variable name or keyword)."""
    __slots__ = ("name",)
    _cache: dict[str, Symbol] = {}

    def __new__(cls, name: str) -> Symbol:
        if name in cls._cache:
            return cls._cache[name]
        obj = super().__new__(cls)
        obj.name = name
        cls._cache[name] = obj
        return obj

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Symbol) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


class _Nil:
    """Singleton nil value."""
    _instance: _Nil | None = None

    def __new__(cls) -> _Nil:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "nil"

    def __bool__(self) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Nil)

    def __hash__(self) -> int:
        return hash(None)


NIL = _Nil()


class Lambda:
    """A user-defined function (closure)."""
    __slots__ = ("params", "body", "env", "name")

    def __init__(self, params: list[Symbol], body: Any, env: Env, name: str = "<lambda>"):
        self.params = params
        self.body = body
        self.env = env
        self.name = name

    def __repr__(self) -> str:
        param_str = " ".join(p.name for p in self.params)
        return f"(lambda ({param_str}) ...)"


# ---------------------------------------------------------------------------
# Environment (scope chain)
# ---------------------------------------------------------------------------

class Env:
    """A scope with parent chain lookup."""
    __slots__ = ("bindings", "parent")

    def __init__(self, bindings: dict[str, Any] | None = None, parent: Env | None = None):
        self.bindings: dict[str, Any] = bindings or {}
        self.parent = parent

    def lookup(self, name: str) -> Any:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        raise LispyError(f"Unbound symbol: {name}")

    def set(self, name: str, value: Any) -> None:
        self.bindings[name] = value


# ---------------------------------------------------------------------------
# Evaluation context (step + depth tracking)
# ---------------------------------------------------------------------------

MAX_DEPTH = 3         # Max sub-sim nesting depth
MAX_STEPS = 5000      # Max eval steps per top-level run
MAX_CALL_DEPTH = 200  # Max function call depth (recursion limit)
MAX_TOKENS = 4000     # Max tokens in input expression
MAX_AST_DEPTH = 40    # Max AST nesting depth
MAX_LIST_LEN = 500    # Max list length
MAX_SUBSIM_PER_TICK = 20  # Max sub-sims per tick


class EvalContext:
    """Tracks evaluation budget and sub-simulation depth."""

    def __init__(self, max_steps: int = MAX_STEPS, max_depth: int = MAX_DEPTH):
        self.steps = 0
        self.max_steps = max_steps
        self.call_depth = 0
        self.sim_depth = 0
        self.max_depth = max_depth
        self.sub_sims_run = 0

    def tick(self) -> None:
        """Count one evaluation step."""
        self.steps += 1
        if self.steps > self.max_steps:
            raise StepBudgetExceeded(
                f"Step budget exceeded ({self.max_steps} steps)")

    def enter_subsim(self) -> int:
        """Enter a sub-simulation. Returns previous depth for restore."""
        prev = self.sim_depth
        self.sim_depth += 1
        self.sub_sims_run += 1
        if self.sim_depth > self.max_depth:
            self.sim_depth = prev
            raise DepthExceeded(
                f"Sub-sim depth {self.sim_depth + 1} exceeds max {self.max_depth}")
        if self.sub_sims_run > MAX_SUBSIM_PER_TICK:
            self.sim_depth = prev
            raise StepBudgetExceeded(
                f"Too many sub-sims ({self.sub_sims_run} > {MAX_SUBSIM_PER_TICK})")
        return prev

    def exit_subsim(self, prev_depth: int) -> None:
        """Restore sub-simulation depth."""
        self.sim_depth = prev_depth


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LispyError(Exception):
    """Base error for LisPy evaluation."""
    pass


class StepBudgetExceeded(LispyError):
    """Evaluation took too many steps."""
    pass


class DepthExceeded(LispyError):
    """Sub-simulation depth exceeded."""
    pass


# ---------------------------------------------------------------------------
# Tokenizer + Parser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)                    |  # whitespace (skip)
    (?P<comment>;[^\n]*)           |  # comment (skip)
    (?P<string>"(?:[^"\\]|\\.)*")  |  # string literal
    (?P<open>\()                   |  # open paren
    (?P<close>\))                  |  # close paren
    (?P<quote>')                   |  # quote shorthand
    (?P<atom>[^\s()"';]+)             # atom (symbol or number)
""", re.VERBOSE)


def tokenize(source: str) -> list[str]:
    """Tokenize a LisPy source string."""
    if len(source) > MAX_TOKENS * 10:
        raise LispyError(f"Input too large ({len(source)} chars)")
    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(source):
        if m.group("ws") or m.group("comment"):
            continue
        tokens.append(m.group())
    return tokens


def _parse_tokens(tokens: list[str], pos: int, depth: int = 0) -> tuple[Any, int]:
    """Parse tokens into an AST. Returns (ast, next_pos)."""
    if depth > MAX_AST_DEPTH:
        raise LispyError(f"AST depth exceeds {MAX_AST_DEPTH}")
    if pos >= len(tokens):
        raise LispyError("Unexpected end of input")

    token = tokens[pos]

    if token == "(":
        lst: list[Any] = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ")":
            if len(lst) > MAX_LIST_LEN:
                raise LispyError(f"List too long (> {MAX_LIST_LEN})")
            val, pos = _parse_tokens(tokens, pos, depth + 1)
            lst.append(val)
        if pos >= len(tokens):
            raise LispyError("Missing closing parenthesis")
        pos += 1  # skip ')'
        return lst, pos

    if token == ")":
        raise LispyError("Unexpected closing parenthesis")

    if token == "'":
        val, pos = _parse_tokens(tokens, pos + 1, depth + 1)
        return [Symbol("quote"), val], pos

    # Atom: number, string, bool, nil, or symbol
    return _parse_atom(token), pos + 1


def _parse_atom(token: str) -> Any:
    """Parse an atom token into a value."""
    if token == "true":
        return True
    if token == "false":
        return False
    if token == "nil":
        return NIL
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1].replace('\\"', '"').replace("\\n", "\n")
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return Symbol(token)


def parse(source: str) -> Any:
    """Parse a LisPy source string into an AST."""
    tokens = tokenize(source)
    if not tokens:
        return NIL
    ast, pos = _parse_tokens(tokens, 0)
    return ast


def parse_all(source: str) -> list[Any]:
    """Parse multiple top-level expressions."""
    tokens = tokenize(source)
    results: list[Any] = []
    pos = 0
    while pos < len(tokens):
        ast, pos = _parse_tokens(tokens, pos)
        results.append(ast)
    return results


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate(expr: Any, env: Env, ctx: EvalContext) -> Any:
    """Evaluate a LisPy expression in the given environment."""
    ctx.tick()

    # Atoms
    if isinstance(expr, (int, float, bool, str)):
        return expr
    if isinstance(expr, _Nil):
        return NIL
    if isinstance(expr, Symbol):
        return env.lookup(expr.name)

    # Lists (function calls and special forms)
    if not isinstance(expr, list) or len(expr) == 0:
        return NIL

    head = expr[0]

    # Special forms
    if isinstance(head, Symbol):
        name = head.name

        if name == "quote":
            if len(expr) < 2:
                raise LispyError("quote requires an argument")
            return expr[1]

        if name == "if":
            if len(expr) < 3:
                raise LispyError("if requires at least 2 arguments")
            cond = evaluate(expr[1], env, ctx)
            if cond and cond is not NIL:
                return evaluate(expr[2], env, ctx)
            elif len(expr) > 3:
                return evaluate(expr[3], env, ctx)
            return NIL

        if name == "cond":
            for clause in expr[1:]:
                if not isinstance(clause, list) or len(clause) < 2:
                    raise LispyError("cond clause must be (test body)")
                test = evaluate(clause[0], env, ctx)
                if test and test is not NIL:
                    return evaluate(clause[1], env, ctx)
            return NIL

        if name == "define":
            if len(expr) != 3:
                raise LispyError("define requires exactly 2 arguments")
            if not isinstance(expr[1], Symbol):
                raise LispyError("define name must be a symbol")
            val = evaluate(expr[2], env, ctx)
            env.set(expr[1].name, val)
            return val

        if name == "lambda":
            if len(expr) != 3:
                raise LispyError("lambda requires params and body")
            params = expr[1]
            if not isinstance(params, list):
                raise LispyError("lambda params must be a list")
            for p in params:
                if not isinstance(p, Symbol):
                    raise LispyError(f"lambda param must be symbol, got {p}")
            return Lambda(params, expr[2], env)

        if name == "let":
            if len(expr) < 3:
                raise LispyError("let requires bindings and body")
            bindings = expr[1]
            if not isinstance(bindings, list):
                raise LispyError("let bindings must be a list")
            child = Env(parent=env)
            for binding in bindings:
                if not isinstance(binding, list) or len(binding) != 2:
                    raise LispyError("let binding must be (name value)")
                if not isinstance(binding[0], Symbol):
                    raise LispyError("let binding name must be a symbol")
                child.set(binding[0].name, evaluate(binding[1], env, ctx))
            result = NIL
            for body_expr in expr[2:]:
                result = evaluate(body_expr, child, ctx)
            return result

        if name == "begin":
            result = NIL
            for sub in expr[1:]:
                result = evaluate(sub, env, ctx)
            return result

        if name == "and":
            result: Any = True
            for sub in expr[1:]:
                result = evaluate(sub, env, ctx)
                if not result or result is NIL:
                    return False
            return result

        if name == "or":
            for sub in expr[1:]:
                result = evaluate(sub, env, ctx)
                if result and result is not NIL:
                    return result
            return False

        if name == "not":
            if len(expr) != 2:
                raise LispyError("not requires exactly 1 argument")
            val = evaluate(expr[1], env, ctx)
            return not val or val is NIL

        if name == "sub-sim":
            return _run_subsim(expr[1:], env, ctx)

    # Function call
    fn = evaluate(head, env, ctx)
    args = [evaluate(a, env, ctx) for a in expr[1:]]

    if isinstance(fn, Lambda):
        if len(args) != len(fn.params):
            raise LispyError(
                f"{fn.name} expects {len(fn.params)} args, got {len(args)}")
        ctx.call_depth += 1
        if ctx.call_depth > MAX_CALL_DEPTH:
            ctx.call_depth -= 1
            raise LispyError(f"Call depth exceeded ({MAX_CALL_DEPTH})")
        child = Env(
            {p.name: a for p, a in zip(fn.params, args)},
            fn.env,
        )
        result = evaluate(fn.body, child, ctx)
        ctx.call_depth -= 1
        return result

    if callable(fn):
        return fn(*args)

    raise LispyError(f"Cannot call {type(fn).__name__}: {fn}")


def _run_subsim(args: list[Any], env: Env, ctx: EvalContext) -> Any:
    """Execute a sub-simulation at the next depth level."""
    if not args:
        raise LispyError("sub-sim requires at least one expression")
    prev = ctx.enter_subsim()
    try:
        result = NIL
        for expr in args:
            result = evaluate(expr, env, ctx)
        return result
    finally:
        ctx.exit_subsim(prev)


# ---------------------------------------------------------------------------
# Built-in functions
# ---------------------------------------------------------------------------

def make_builtins() -> dict[str, Any]:
    """Create the built-in function environment."""
    def _add(*args: Any) -> Any:
        return sum(args)

    def _sub(*args: Any) -> Any:
        if len(args) == 1:
            return -args[0]
        result = args[0]
        for a in args[1:]:
            result -= a
        return result

    def _mul(*args: Any) -> Any:
        result = 1
        for a in args:
            result *= a
        return result

    def _div(a: Any, b: Any) -> Any:
        if b == 0:
            raise LispyError("Division by zero")
        return a / b

    def _mod(a: Any, b: Any) -> Any:
        if b == 0:
            raise LispyError("Modulo by zero")
        return a % b

    def _eq(*args: Any) -> bool:
        return all(a == args[0] for a in args[1:])

    def _lt(a: Any, b: Any) -> bool:
        return a < b

    def _gt(a: Any, b: Any) -> bool:
        return a > b

    def _le(a: Any, b: Any) -> bool:
        return a <= b

    def _ge(a: Any, b: Any) -> bool:
        return a >= b

    def _list(*args: Any) -> list:
        return list(args)

    def _car(lst: Any) -> Any:
        if not isinstance(lst, list) or len(lst) == 0:
            raise LispyError("car requires non-empty list")
        return lst[0]

    def _cdr(lst: Any) -> Any:
        if not isinstance(lst, list):
            raise LispyError("cdr requires a list")
        return lst[1:]

    def _cons(item: Any, lst: Any) -> list:
        if not isinstance(lst, list):
            return [item, lst]
        return [item] + lst

    def _length(lst: Any) -> int:
        if isinstance(lst, (list, str)):
            return len(lst)
        raise LispyError(f"length requires list or string, got {type(lst).__name__}")

    def _append(*args: Any) -> list:
        result: list[Any] = []
        for a in args:
            if isinstance(a, list):
                result.extend(a)
            else:
                result.append(a)
        return result

    def _map_fn(fn: Any, lst: Any) -> list:
        if not isinstance(lst, list):
            raise LispyError("map requires a list")
        return [fn(x) if callable(fn) else x for x in lst]

    def _filter_fn(fn: Any, lst: Any) -> list:
        if not isinstance(lst, list):
            raise LispyError("filter requires a list")
        return [x for x in lst if fn(x) if callable(fn)]

    def _reduce_fn(fn: Any, lst: Any, init: Any = NIL) -> Any:
        if not isinstance(lst, list):
            raise LispyError("reduce requires a list")
        if not lst:
            return init
        acc = init if init is not NIL else lst[0]
        start = 0 if init is not NIL else 1
        for x in lst[start:]:
            acc = fn(acc, x) if callable(fn) else acc
        return acc

    def _abs_fn(x: Any) -> Any:
        return abs(x)

    def _max_fn(*args: Any) -> Any:
        flat = []
        for a in args:
            if isinstance(a, list):
                flat.extend(a)
            else:
                flat.append(a)
        return max(flat)

    def _min_fn(*args: Any) -> Any:
        flat = []
        for a in args:
            if isinstance(a, list):
                flat.extend(a)
            else:
                flat.append(a)
        return min(flat)

    def _round_fn(x: Any, n: int = 0) -> Any:
        return round(x, n)

    def _floor(x: Any) -> int:
        return math.floor(x)

    def _ceil(x: Any) -> int:
        return math.ceil(x)

    def _sqrt(x: Any) -> float:
        return math.sqrt(x)

    def _is_list(x: Any) -> bool:
        return isinstance(x, list)

    def _is_number(x: Any) -> bool:
        return isinstance(x, (int, float))

    def _is_string(x: Any) -> bool:
        return isinstance(x, str)

    def _is_nil(x: Any) -> bool:
        return x is NIL or x is None

    def _str_fn(*args: Any) -> str:
        return "".join(str(a) for a in args)

    def _print_fn(*args: Any) -> Any:
        # No-op in safe mode (no I/O)
        return NIL

    return {
        "+": _add, "-": _sub, "*": _mul, "/": _div, "%": _mod,
        "=": _eq, "<": _lt, ">": _gt, "<=": _le, ">=": _ge,
        "list": _list, "car": _car, "cdr": _cdr, "cons": _cons,
        "length": _length, "append": _append,
        "map": _map_fn, "filter": _filter_fn, "reduce": _reduce_fn,
        "abs": _abs_fn, "max": _max_fn, "min": _min_fn,
        "round": _round_fn, "floor": _floor, "ceil": _ceil, "sqrt": _sqrt,
        "list?": _is_list, "number?": _is_number, "string?": _is_string,
        "nil?": _is_nil,
        "str": _str_fn, "print": _print_fn,
        "pi": math.pi, "e": math.e,
        "true": True, "false": False, "nil": NIL,
    }


def make_env(extra: dict[str, Any] | None = None) -> Env:
    """Create a standard environment with builtins."""
    builtins = make_builtins()
    if extra:
        builtins.update(extra)
    return Env(builtins)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(source: str, env: Env | None = None, ctx: EvalContext | None = None) -> tuple[Any, EvalContext]:
    """Parse and evaluate a LisPy source string.

    Returns (result, context) tuple.
    """
    if env is None:
        env = make_env()
    if ctx is None:
        ctx = EvalContext()
    exprs = parse_all(source)
    result = NIL
    for expr in exprs:
        result = evaluate(expr, env, ctx)
    return result, ctx


def run_in_context(source: str, env: Env, ctx: EvalContext) -> Any:
    """Parse and evaluate, sharing an existing context."""
    exprs = parse_all(source)
    result = NIL
    for expr in exprs:
        result = evaluate(expr, env, ctx)
    return result


def safe_eval(source: str, max_steps: int = MAX_STEPS, env: Env | None = None) -> dict:
    """Safe wrapper that catches errors. Returns result dict."""
    if env is None:
        env = make_env()
    ctx = EvalContext(max_steps=max_steps)
    try:
        result, ctx = run(source, env, ctx)
        return {
            "ok": True,
            "value": result,
            "steps": ctx.steps,
            "sub_sims": ctx.sub_sims_run,
        }
    except LispyError as e:
        return {
            "ok": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "steps": ctx.steps,
            "sub_sims": ctx.sub_sims_run,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"Internal error: {e}",
            "error_type": "InternalError",
            "steps": ctx.steps,
            "sub_sims": ctx.sub_sims_run,
        }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def to_sexpr(value: Any) -> str:
    """Convert a Python value back to an s-expression string."""
    if value is NIL or value is None:
        return "nil"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, str):
        escaped = value.replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    if isinstance(value, Symbol):
        return value.name
    if isinstance(value, list):
        inner = " ".join(to_sexpr(v) for v in value)
        return f"({inner})"
    if isinstance(value, Lambda):
        params = " ".join(p.name for p in value.params)
        return f"(lambda ({params}) ...)"
    return str(value)


def serialize(value: Any) -> Any:
    """Convert a LisPy value to a JSON-safe Python value."""
    if value is NIL or value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Symbol):
        return value.name
    if isinstance(value, list):
        return [serialize(v) for v in value]
    if isinstance(value, Lambda):
        return {"__type__": "lambda", "params": [p.name for p in value.params]}
    return str(value)
