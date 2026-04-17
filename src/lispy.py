"""
lispy.py — Safe-eval LisPy interpreter for sub-simulations.

Homoiconic s-expression language. No I/O, no imports, pure computation.
Sub-simulations up to depth 3 (Turtles All the Way Down, Amendment XIII).

Usage:
    from src.lispy import Lispy, LispyError, LispyTimeout
    vm = Lispy(seed=42)
    result = vm.eval_string("(+ 1 2)")  # => 3

Python stdlib only.
"""
from __future__ import annotations

import math
import random
import re
from typing import Any

MAX_SUB_SIM_DEPTH = 3
DEFAULT_STEP_LIMIT = 50_000


class LispyError(Exception):
    """Any LisPy evaluation error."""


class LispyTimeout(LispyError):
    """Step limit exceeded."""


class Symbol:
    """Interned symbol."""
    _cache: dict[str, Symbol] = {}

    def __new__(cls, name: str) -> Symbol:
        if name not in cls._cache:
            obj = super().__new__(cls)
            obj.name = name
            cls._cache[name] = obj
        return cls._cache[name]

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Symbol) and self.name == other.name


class Lambda:
    """User-defined function (closure)."""

    def __init__(self, params: list[Symbol], body: Any, env: Env):
        self.params = params
        self.body = body
        self.env = env

    def __repr__(self) -> str:
        return f"(lambda ({' '.join(str(p) for p in self.params)}) ...)"


class Env:
    """Environment with parent chain."""

    def __init__(self, bindings: dict[str, Any] | None = None,
                 parent: Env | None = None):
        self.data: dict[str, Any] = bindings or {}
        self.parent = parent

    def get(self, name: str) -> Any:
        if name in self.data:
            return self.data[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise LispyError(f"Unbound symbol: {name}")

    def set(self, name: str, value: Any) -> None:
        self.data[name] = value

    def find(self, name: str) -> Env:
        """Find the env where name is defined."""
        if name in self.data:
            return self
        if self.parent is not None:
            return self.parent.find(name)
        raise LispyError(f"Unbound symbol: {name}")


# --- Tokenizer ---

TOKEN_RE = re.compile(r"""(\s+|;[^\n]*|"(?:[^"\\]|\\.)*"|'|\(|\)|[^\s()"';]+)""")


def tokenize(source: str) -> list[str]:
    """Tokenize source into a list of tokens."""
    tokens = []
    for m in TOKEN_RE.finditer(source):
        tok = m.group(1)
        if tok.isspace() or tok.startswith(";"):
            continue
        tokens.append(tok)
    return tokens


def parse_all(tokens: list[str]) -> list[Any]:
    """Parse all expressions from token list."""
    exprs = []
    pos = 0
    while pos < len(tokens):
        expr, pos = _parse(tokens, pos)
        exprs.append(expr)
    return exprs


def _parse(tokens: list[str], pos: int) -> tuple[Any, int]:
    """Parse one expression starting at pos."""
    if pos >= len(tokens):
        raise LispyError("Unexpected end of input")

    tok = tokens[pos]

    if tok == "(":
        return _parse_list(tokens, pos + 1)
    elif tok == "'":
        # Quote sugar: 'x => (quote x)
        expr, npos = _parse(tokens, pos + 1)
        return [Symbol("quote"), expr], npos
    elif tok == ")":
        raise LispyError("Unexpected )")
    else:
        return _parse_atom(tok), pos + 1


def _parse_list(tokens: list[str], pos: int) -> tuple[list, int]:
    """Parse list contents after opening paren."""
    items: list[Any] = []
    while pos < len(tokens) and tokens[pos] != ")":
        expr, pos = _parse(tokens, pos)
        items.append(expr)
    if pos >= len(tokens):
        raise LispyError("Missing )")
    return items, pos + 1  # skip )


def _parse_atom(tok: str) -> Any:
    """Parse an atom: number, string, bool, nil, or symbol."""
    if tok.startswith('"') and tok.endswith('"'):
        return tok[1:-1].replace('\\"', '"').replace("\\n", "\n")
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    if tok == "#t" or tok == "true":
        return True
    if tok == "#f" or tok == "false":
        return False
    if tok == "nil":
        return None
    return Symbol(tok)


# --- Interpreter ---

class Lispy:
    """Safe-eval LisPy virtual machine."""

    def __init__(self, seed: int = 42, sim_depth: int = 0,
                 step_limit: int = DEFAULT_STEP_LIMIT):
        self.seed = seed
        self.sim_depth = sim_depth
        self.step_limit = step_limit
        self.steps = 0
        self.rng = random.Random(seed)
        self.sub_sim_log: list[dict] = []
        self.global_env = self._make_global_env()

    def eval_string(self, source: str) -> Any:
        """Parse and evaluate a LisPy source string."""
        tokens = tokenize(source)
        exprs = parse_all(tokens)
        result = None
        for expr in exprs:
            result = self._eval(expr, self.global_env)
        return result

    def _tick(self) -> None:
        """Count a step; raise on limit."""
        self.steps += 1
        if self.steps > self.step_limit:
            raise LispyTimeout(f"Step limit exceeded ({self.step_limit})")

    def _eval(self, expr: Any, env: Env) -> Any:
        """Evaluate one expression."""
        self._tick()

        # Atoms
        if isinstance(expr, (int, float, bool, str)):
            return expr
        if expr is None:
            return None
        if isinstance(expr, Symbol):
            return env.get(expr.name)
        if not isinstance(expr, list) or len(expr) == 0:
            return expr

        head = expr[0]

        # Special forms
        if isinstance(head, Symbol):
            name = head.name

            if name == "quote":
                return expr[1] if len(expr) > 1 else None

            if name == "if":
                cond = self._eval(expr[1], env)
                if cond and cond is not None:
                    return self._eval(expr[2], env)
                elif len(expr) > 3:
                    return self._eval(expr[3], env)
                return None

            if name == "define":
                val = self._eval(expr[2], env)
                if isinstance(expr[1], Symbol):
                    env.set(expr[1].name, val)
                return val

            if name == "set!":
                val = self._eval(expr[2], env)
                target = env.find(expr[1].name)
                target.set(expr[1].name, val)
                return val

            if name == "lambda":
                params = [p if isinstance(p, Symbol) else Symbol(str(p))
                          for p in expr[1]]
                return Lambda(params, expr[2], env)

            if name == "begin":
                result = None
                for e in expr[1:]:
                    result = self._eval(e, env)
                return result

            if name == "let":
                bindings = expr[1]
                child = Env(parent=env)
                for b in bindings:
                    child.set(b[0].name if isinstance(b[0], Symbol) else str(b[0]),
                              self._eval(b[1], env))
                result = None
                for e in expr[2:]:
                    result = self._eval(e, child)
                return result

            if name == "cond":
                for clause in expr[1:]:
                    if isinstance(clause[0], Symbol) and clause[0].name == "else":
                        return self._eval(clause[1], env)
                    if self._eval(clause[0], env):
                        return self._eval(clause[1], env)
                return None

            if name == "and":
                result: Any = True
                for e in expr[1:]:
                    result = self._eval(e, env)
                    if not result:
                        return result
                return result

            if name == "or":
                for e in expr[1:]:
                    result = self._eval(e, env)
                    if result:
                        return result
                return False

            if name == "not":
                return not self._eval(expr[1], env)

            if name == "sub-sim":
                return self._run_sub_sim(expr, env)

        # Function application
        fn = self._eval(head, env)
        args = [self._eval(a, env) for a in expr[1:]]

        if isinstance(fn, Lambda):
            if len(args) != len(fn.params):
                raise LispyError(
                    f"Arity mismatch: {fn} expects {len(fn.params)}, got {len(args)}")
            child = Env(
                {p.name: a for p, a in zip(fn.params, args)},
                fn.env,
            )
            return self._eval(fn.body, child)

        if callable(fn):
            return fn(*args)

        raise LispyError(f"Not callable: {fn}")

    def _run_sub_sim(self, expr: list, env: Env) -> Any:
        """Run a sandboxed sub-simulation."""
        if self.sim_depth >= MAX_SUB_SIM_DEPTH:
            raise LispyError(
                f"Sub-sim depth limit reached ({MAX_SUB_SIM_DEPTH})")

        label = "anonymous"
        body_expr = expr[1]
        if len(expr) >= 3:
            label = self._eval(expr[1], env)
            body_expr = expr[2]

        # Evaluate quoted body to get the expression
        source = self._eval(body_expr, env)

        child = Lispy(
            seed=self.seed + self.steps,
            sim_depth=self.sim_depth + 1,
            step_limit=min(self.step_limit // 2, 10_000),
        )

        result = child._eval(source, child.global_env)

        log_entry = {
            "label": str(label),
            "depth": self.sim_depth + 1,
            "source": str(source)[:300],
            "result": str(result)[:200],
            "steps": child.steps,
            "child_logs": child.sub_sim_log,
        }
        self.sub_sim_log.append(log_entry)

        return result

    def _make_global_env(self) -> Env:
        """Create global environment with builtins."""
        rng = self.rng
        env = Env()

        # Arithmetic
        env.set("+", lambda *args: sum(args))
        env.set("-", lambda a, b=None: -a if b is None else a - b)
        env.set("*", lambda *args: math.prod(args))
        env.set("/", lambda a, b: a / b if b != 0 else 0)
        env.set("//", lambda a, b: a // b if b != 0 else 0)
        env.set("%", lambda a, b: a % b if b != 0 else 0)
        env.set("abs", abs)
        env.set("min", min)
        env.set("max", max)
        env.set("round", lambda x, n=0: round(x, n))
        env.set("floor", math.floor)
        env.set("ceil", math.ceil)
        env.set("sqrt", math.sqrt)
        env.set("pow", pow)
        env.set("log", math.log)
        env.set("pi", math.pi)
        env.set("e", math.e)

        # Comparison
        env.set("=", lambda a, b: a == b)
        env.set("!=", lambda a, b: a != b)
        env.set("<", lambda a, b: a < b)
        env.set(">", lambda a, b: a > b)
        env.set("<=", lambda a, b: a <= b)
        env.set(">=", lambda a, b: a >= b)

        # List ops
        env.set("list", lambda *args: list(args))
        env.set("cons", lambda a, b: [a] + (b if isinstance(b, list) else [b]))
        env.set("car", lambda lst: lst[0] if lst else None)
        env.set("cdr", lambda lst: lst[1:] if lst else [])
        env.set("length", lambda x: len(x) if x else 0)
        env.set("append", lambda *lsts: sum((l if isinstance(l, list) else [l] for l in lsts), []))
        env.set("nth", lambda lst, n: lst[n] if 0 <= n < len(lst) else None)
        env.set("range", lambda a, b=None: list(range(a)) if b is None else list(range(a, b)))
        env.set("reverse", lambda lst: list(reversed(lst)))
        env.set("sort", lambda lst: sorted(lst))
        env.set("empty?", lambda x: len(x) == 0 if x else True)

        # Higher-order (need _call helper for Lambda support)
        def _call(f: Any, *args: Any) -> Any:
            """Call a function — works for both Python callables and Lambda."""
            if isinstance(f, Lambda):
                child = Env(
                    {p.name: a for p, a in zip(f.params, args)},
                    f.env,
                )
                return self._eval(f.body, child)
            return f(*args)

        env.set("map", lambda f, lst: [_call(f, x) for x in lst])
        env.set("filter", lambda f, lst: [x for x in lst if _call(f, x)])
        env.set("reduce", lambda f, lst, init: _reduce_with(_call, f, lst, init))

        # String ops
        env.set("str", lambda *args: "".join(str(a) for a in args))
        env.set("str-length", lambda s: len(s) if isinstance(s, str) else 0)
        env.set("str-upper", lambda s: s.upper())
        env.set("str-lower", lambda s: s.lower())
        env.set("substr", lambda s, start, end=None: s[start:end])

        # Dict ops
        env.set("dict", lambda *pairs: dict(zip(pairs[::2], pairs[1::2])))
        env.set("dict-get", lambda d, k, default=None: d.get(k, default) if isinstance(d, dict) else default)
        env.set("dict-set", lambda d, k, v: {**d, k: v})
        env.set("dict-keys", lambda d: list(d.keys()) if isinstance(d, dict) else [])

        # Type checks
        env.set("number?", lambda x: isinstance(x, (int, float)))
        env.set("string?", lambda x: isinstance(x, str))
        env.set("list?", lambda x: isinstance(x, list))
        env.set("nil?", lambda x: x is None)
        env.set("bool?", lambda x: isinstance(x, bool))

        # Random (seeded, deterministic)
        env.set("random", lambda: rng.random())
        env.set("random-int", lambda a, b: rng.randint(a, b))
        env.set("random-choice", lambda lst: rng.choice(lst))

        # I/O deliberately omitted — safe eval

        return env


def _reduce(f: Any, lst: list, init: Any) -> Any:
    """Reduce helper for Python callables."""
    acc = init
    for x in lst:
        acc = f(acc, x)
    return acc


def _reduce_with(caller: Any, f: Any, lst: list, init: Any) -> Any:
    """Reduce helper that uses _call for Lambda support."""
    acc = init
    for x in lst:
        acc = caller(f, acc, x)
    return acc
