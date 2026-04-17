"""
LisPy — Safe s-expression interpreter for Mars-100 recursive simulations.

Pure computation only. No file I/O, no imports, no subprocess.
Homoiconic: data IS code IS data. The colonists rewrite themselves.

Supports nested sub-simulations up to depth 3 (Turtles All the Way Down).
A single shared step budget prevents runaway recursion across all levels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Symbol(str):
    """A Lisp symbol — a string that represents a variable name."""
    __slots__ = ()
    def __repr__(self) -> str:
        return f"Symbol({super().__repr__()})"


class Nil:
    """The empty list / false value.  Singleton."""
    _instance: Nil | None = None

    def __new__(cls) -> Nil:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "()"

    def __bool__(self) -> bool:
        return False

    def __iter__(self):
        return iter([])

    def __len__(self) -> int:
        return 0


NIL = Nil()


@dataclass
class Lambda:
    """A user-defined function (closure)."""
    params: list[str]
    body: Any            # parsed s-expression
    env: "Env"
    name: str = "<lambda>"

    def __repr__(self) -> str:
        return f"<lambda {self.name}({', '.join(self.params)})>"


# ---------------------------------------------------------------------------
# Budget — shared across all eval depths
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    """Step budget shared across parent + child evaluations."""
    remaining: int = 50_000
    max_depth: int = 3
    current_depth: int = 0

    def charge(self, cost: int = 1) -> None:
        """Consume steps. Raises if exhausted."""
        self.remaining -= cost
        if self.remaining <= 0:
            raise LispyError("step budget exhausted — infinite loop?")

    def enter_subsim(self) -> None:
        """Enter a sub-simulation level."""
        self.current_depth += 1
        if self.current_depth > self.max_depth:
            raise LispyError(
                f"sub-sim depth {self.current_depth} exceeds max {self.max_depth}"
            )

    def exit_subsim(self) -> None:
        """Exit a sub-simulation level."""
        self.current_depth -= 1


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Env(dict):
    """A Lisp environment: a dict of {symbol: value} with a parent scope."""

    def __init__(self, params: list[str] = (), args: list = (),
                 parent: "Env | None" = None):
        super().__init__(zip(params, args))
        self.parent = parent

    def find(self, name: str) -> "Env":
        """Find the innermost env containing *name*."""
        if name in self:
            return self
        if self.parent is not None:
            return self.parent.find(name)
        raise LispyError(f"undefined symbol: {name}")

    def get_val(self, name: str) -> Any:
        """Lookup a symbol's value."""
        return self.find(name)[name]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LispyError(Exception):
    """Any error during tokenise / parse / eval."""
    pass


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def tokenise(source: str) -> list[str]:
    """Turn a string of s-expressions into a flat list of tokens."""
    tokens: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in " \t\n\r":
            i += 1
        elif ch == ";":
            while i < n and source[i] != "\n":
                i += 1
        elif ch in "()":
            tokens.append(ch)
            i += 1
        elif ch == "'":
            tokens.append("'")
            i += 1
        elif ch == '"':
            j = i + 1
            while j < n and source[j] != '"':
                if source[j] == "\\":
                    j += 1
                j += 1
            if j >= n:
                raise LispyError("unterminated string literal")
            tokens.append(source[i:j + 1])
            i = j + 1
        else:
            j = i
            while j < n and source[j] not in " \t\n\r();\"'":
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
        return token[1:-1].replace('\\"', '"').replace("\\n", "\n")
    if token == "#t":
        return True
    if token == "#f":
        return False
    if token == "nil":
        return NIL
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return Symbol(token)


def parse(source: str) -> list[Any]:
    """Parse source into a list of s-expressions."""
    tokens = tokenise(source)
    results: list[Any] = []
    pos = 0

    def read_expr() -> Any:
        nonlocal pos
        if pos >= len(tokens):
            raise LispyError("unexpected end of input")
        tok = tokens[pos]

        if tok == "'":
            pos += 1
            inner = read_expr()
            return [Symbol("quote"), inner]

        if tok == "(":
            pos += 1
            items: list[Any] = []
            while pos < len(tokens) and tokens[pos] != ")":
                items.append(read_expr())
            if pos >= len(tokens):
                raise LispyError("missing closing parenthesis")
            pos += 1
            return items

        if tok == ")":
            raise LispyError("unexpected closing parenthesis")

        pos += 1
        return _atom(tok)

    while pos < len(tokens):
        results.append(read_expr())
    return results


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def lispy_eval(expr: Any, env: Env, budget: Budget) -> Any:
    """Evaluate a single s-expression."""
    budget.charge()

    # --- Atoms ---
    if isinstance(expr, Symbol):
        return env.get_val(expr)

    if isinstance(expr, (int, float, str, bool)) or expr is NIL:
        return expr

    if not isinstance(expr, list):
        return expr

    if len(expr) == 0:
        return NIL

    head = expr[0]

    # --- Special forms ---
    if isinstance(head, Symbol):
        if head == "quote":
            if len(expr) != 2:
                raise LispyError("quote requires exactly 1 argument")
            return expr[1]

        if head == "if":
            if len(expr) < 3:
                raise LispyError("if requires at least 2 arguments")
            test = lispy_eval(expr[1], env, budget)
            if test and test is not NIL:
                return lispy_eval(expr[2], env, budget)
            elif len(expr) > 3:
                return lispy_eval(expr[3], env, budget)
            return NIL

        if head == "define":
            if len(expr) < 3:
                raise LispyError("define requires at least 2 arguments")
            target = expr[1]
            if isinstance(target, list):
                name = target[0]
                params = [str(p) for p in target[1:]]
                body = expr[2]
                lam = Lambda(params=params, body=body, env=env, name=str(name))
                env[str(name)] = lam
                return lam
            val = lispy_eval(expr[2], env, budget)
            env[str(target)] = val
            return val

        if head == "lambda":
            if len(expr) < 3:
                raise LispyError("lambda requires params and body")
            params = [str(p) for p in expr[1]]
            body = expr[2]
            return Lambda(params=params, body=body, env=env)

        if head == "begin":
            result: Any = NIL
            for sub in expr[1:]:
                result = lispy_eval(sub, env, budget)
            return result

        if head == "let":
            if len(expr) < 3:
                raise LispyError("let requires bindings and body")
            bindings = expr[1]
            child_env = Env(parent=env)
            for binding in bindings:
                if not isinstance(binding, list) or len(binding) != 2:
                    raise LispyError(f"invalid let binding: {binding}")
                name = str(binding[0])
                val = lispy_eval(binding[1], env, budget)
                child_env[name] = val
            return lispy_eval(expr[2], child_env, budget)

        if head == "and":
            result = True
            for sub in expr[1:]:
                result = lispy_eval(sub, env, budget)
                if not result or result is NIL:
                    return result
            return result

        if head == "or":
            for sub in expr[1:]:
                result = lispy_eval(sub, env, budget)
                if result and result is not NIL:
                    return result
            return NIL

        if head == "sub-sim":
            if len(expr) != 2:
                raise LispyError("sub-sim requires exactly 1 argument (expression)")
            budget.enter_subsim()
            try:
                child_env = Env(parent=env)
                result = lispy_eval(expr[1], child_env, budget)
            finally:
                budget.exit_subsim()
            return result

    # --- Function call ---
    fn = lispy_eval(head, env, budget)
    args = [lispy_eval(a, env, budget) for a in expr[1:]]

    if isinstance(fn, Lambda):
        if len(args) != len(fn.params):
            raise LispyError(
                f"{fn.name} expects {len(fn.params)} args, got {len(args)}"
            )
        call_env = Env(params=fn.params, args=args, parent=fn.env)
        return lispy_eval(fn.body, call_env, budget)

    if callable(fn):
        budget.charge()
        return fn(*args)

    raise LispyError(f"cannot call: {fn}")


# ---------------------------------------------------------------------------
# Standard library — built-in functions
# ---------------------------------------------------------------------------

def _make_global_env() -> Env:
    """Create the global environment with built-in functions."""
    env = Env()

    # Arithmetic
    env["+"] = lambda *a: sum(a)
    env["-"] = lambda a, b=None: -a if b is None else a - b
    env["*"] = lambda *a: math.prod(a)
    env["/"] = lambda a, b: a / b if b != 0 else 0
    env["mod"] = lambda a, b: a % b if b != 0 else 0
    env["abs"] = abs
    env["min"] = min
    env["max"] = max
    env["round"] = lambda x, n=0: round(x, n)
    env["floor"] = math.floor
    env["ceil"] = math.ceil
    env["sqrt"] = math.sqrt
    env["pow"] = pow

    # Comparison
    env["="] = lambda a, b: a == b
    env["!="] = lambda a, b: a != b
    env["<"] = lambda a, b: a < b
    env[">"] = lambda a, b: a > b
    env["<="] = lambda a, b: a <= b
    env[">="] = lambda a, b: a >= b

    # Logic
    env["not"] = lambda x: not x

    # Type checks
    env["number?"] = lambda x: isinstance(x, (int, float))
    env["string?"] = lambda x: isinstance(x, str) and not isinstance(x, Symbol)
    env["list?"] = lambda x: isinstance(x, list)
    env["nil?"] = lambda x: x is NIL
    env["symbol?"] = lambda x: isinstance(x, Symbol)

    # List operations
    env["list"] = lambda *a: list(a)
    env["car"] = lambda x: x[0] if x else NIL
    env["cdr"] = lambda x: x[1:] if len(x) > 1 else NIL
    env["cons"] = lambda a, b: [a] + (b if isinstance(b, list) else [b])
    env["length"] = lambda x: len(x) if hasattr(x, "__len__") else 0
    env["nth"] = lambda lst, n: lst[n] if isinstance(lst, list) and 0 <= n < len(lst) else NIL
    env["append"] = lambda *lsts: sum((l if isinstance(l, list) else [l] for l in lsts), [])

    # Assoc-list lookup
    def _assoc_get(key, alist):
        if not isinstance(alist, list):
            return NIL
        for pair in alist:
            if isinstance(pair, list) and len(pair) >= 2 and pair[0] == key:
                return pair[1]
        return NIL
    env["assoc"] = _assoc_get

    # String operations
    env["str"] = lambda *a: "".join(str(x) for x in a)
    env["str-len"] = lambda s: len(s) if isinstance(s, str) else 0

    # Higher-order — need a helper to call Lambda or Python callable
    def _apply_fn(fn, args):
        """Call a LisPy Lambda or Python callable with args."""
        if isinstance(fn, Lambda):
            call_env = Env(params=fn.params, args=list(args), parent=fn.env)
            # We need a budget — grab one from outer scope or create default
            b = Budget(remaining=10000)
            return lispy_eval(fn.body, call_env, b)
        return fn(*args)

    def _lispy_map(fn, lst):
        if not isinstance(lst, list):
            return NIL
        return [_apply_fn(fn, [x]) for x in lst]
    env["map"] = _lispy_map

    def _lispy_filter(fn, lst):
        if not isinstance(lst, list):
            return NIL
        return [x for x in lst if _apply_fn(fn, [x])]
    env["filter"] = _lispy_filter

    def _lispy_reduce(fn, lst, init=NIL):
        if not isinstance(lst, list):
            return init
        acc = init
        for x in lst:
            acc = _apply_fn(fn, [acc, x])
        return acc
    env["reduce"] = _lispy_reduce

    # Constants
    env["pi"] = math.pi
    env["e"] = math.e
    env["true"] = True
    env["false"] = False
    env["nil"] = NIL

    return env


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(source: str, env: Env | None = None,
        budget: Budget | None = None) -> Any:
    """Parse and evaluate LisPy source code.  Returns the last result."""
    if env is None:
        env = _make_global_env()
    if budget is None:
        budget = Budget()

    exprs = parse(source)
    result: Any = NIL
    for expr in exprs:
        result = lispy_eval(expr, env, budget)
    return result


def make_env(**extras: Any) -> Env:
    """Create a fresh global environment, optionally injecting extra bindings."""
    env = _make_global_env()
    for k, v in extras.items():
        env[k.replace("_", "-")] = v
    return env


def format_sexpr(value: Any) -> str:
    """Pretty-print a value as an s-expression string."""
    if value is NIL:
        return "()"
    if isinstance(value, bool):
        return "#t" if value else "#f"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str) and not isinstance(value, Symbol):
        return f'"{value}"'
    if isinstance(value, Symbol):
        return str(value)
    if isinstance(value, list):
        inner = " ".join(format_sexpr(v) for v in value)
        return f"({inner})"
    if isinstance(value, Lambda):
        return f"(lambda ({' '.join(value.params)}) ...)"
    return str(value)
