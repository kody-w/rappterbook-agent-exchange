"""
lispy.py — Safe, sandboxed LisPy interpreter for Mars-100 sub-simulations.

No I/O. No imports. No eval of Python. Pure computation only.
Supports recursive sub-simulations up to depth 3 (Turtles All the Way Down).

Usage:
    from src.lispy import Lispy
    vm = Lispy(seed=42)
    result = vm.eval_string("(+ 1 2)")  # -> 3
    result = vm.eval_string("(sub-sim 1 '(+ 10 20))")  # -> 30
"""
from __future__ import annotations

import math
import random
import re
from typing import Any


# --- Safety limits ---
MAX_STEPS = 50_000
MAX_DEPTH = 100
MAX_SUB_SIM_DEPTH = 3
MAX_LIST_SIZE = 10_000


class LispyError(Exception):
    """All LisPy errors."""
    pass


class LispyTimeout(LispyError):
    """Step limit exceeded."""
    pass


# --- Tokenizer ---

TOKEN_RE = re.compile(
    r"""(\s+|;[^\n]*|"(?:[^"\\]|\\.)*"|'|\(|\)|[^\s()"';]+)""",
    re.DOTALL,
)


def tokenize(source: str) -> list[str]:
    """Split source into tokens, discarding whitespace and comments."""
    tokens: list[str] = []
    for m in TOKEN_RE.finditer(source):
        tok = m.group(1)
        if tok[0] in " \t\n\r" or tok[0] == ";":
            continue
        tokens.append(tok)
    return tokens


# --- Parser ---

class Symbol:
    """Interned symbol."""
    __slots__ = ("name",)
    _intern: dict[str, "Symbol"] = {}

    def __new__(cls, name: str) -> "Symbol":
        if name in cls._intern:
            return cls._intern[name]
        obj = super().__new__(cls)
        obj.name = name
        cls._intern[name] = obj
        return obj

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Symbol) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


# Sentinel for quote
QUOTE = Symbol("quote")


def parse(tokens: list[str]) -> Any:
    """Parse tokens into a LisPy AST."""
    if not tokens:
        raise LispyError("Unexpected EOF")
    return _parse_expr(tokens, 0)[0]


def parse_all(tokens: list[str]) -> list[Any]:
    """Parse all expressions from tokens."""
    results: list[Any] = []
    pos = 0
    while pos < len(tokens):
        expr, pos = _parse_expr(tokens, pos)
        results.append(expr)
    return results


def _parse_expr(tokens: list[str], pos: int) -> tuple[Any, int]:
    """Parse one expression starting at pos. Returns (expr, new_pos)."""
    if pos >= len(tokens):
        raise LispyError("Unexpected EOF")

    tok = tokens[pos]

    if tok == "'":
        # Quote shorthand
        inner, pos2 = _parse_expr(tokens, pos + 1)
        return [QUOTE, inner], pos2

    if tok == "(":
        lst: list[Any] = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ")":
            expr, pos = _parse_expr(tokens, pos)
            lst.append(expr)
        if pos >= len(tokens):
            raise LispyError("Missing closing )")
        return lst, pos + 1  # skip )

    if tok == ")":
        raise LispyError("Unexpected )")

    # Atom
    return _parse_atom(tok), pos + 1


def _parse_atom(tok: str) -> Any:
    """Parse an atom: number, string, bool, nil, or symbol."""
    if tok == "#t":
        return True
    if tok == "#f":
        return False
    if tok == "nil":
        return None

    # String literal
    if tok.startswith('"') and tok.endswith('"'):
        return tok[1:-1].replace('\\"', '"').replace("\\n", "\n")

    # Number
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass

    return Symbol(tok)


# --- Environment ---

class Env:
    """Lexically scoped environment."""
    __slots__ = ("bindings", "parent")

    def __init__(self, bindings: dict[str, Any] | None = None, parent: "Env | None" = None):
        self.bindings: dict[str, Any] = bindings or {}
        self.parent = parent

    def lookup(self, name: str) -> Any:
        """Find a binding, walking up the scope chain."""
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        raise LispyError(f"Unbound symbol: {name}")

    def set(self, name: str, value: Any) -> None:
        """Set a binding in this scope."""
        self.bindings[name] = value


class Lambda:
    """A user-defined function (closure)."""
    __slots__ = ("params", "body", "closure")

    def __init__(self, params: list[str], body: list[Any], closure: Env):
        self.params = params
        self.body = body
        self.closure = closure


# --- Evaluator ---

class Lispy:
    """Safe LisPy evaluator with sub-simulation support."""

    def __init__(self, seed: int = 42, sim_depth: int = 0, step_limit: int = MAX_STEPS):
        self.rng = random.Random(seed)
        self.sim_depth = sim_depth
        self.step_limit = step_limit
        self.steps = 0
        self.sub_sim_log: list[dict] = []
        self.global_env = self._make_global_env()

    def _tick(self) -> None:
        """Count one evaluation step; raise if limit exceeded."""
        self.steps += 1
        if self.steps > self.step_limit:
            raise LispyTimeout(
                f"Step limit ({self.step_limit}) exceeded at depth {self.sim_depth}"
            )

    def eval_string(self, source: str) -> Any:
        """Parse and evaluate a LisPy source string."""
        tokens = tokenize(source)
        exprs = parse_all(tokens)
        result = None
        for expr in exprs:
            result = self.eval(expr, self.global_env)
        return result

    def eval(self, expr: Any, env: Env, depth: int = 0) -> Any:
        """Evaluate a LisPy expression."""
        if depth > MAX_DEPTH:
            raise LispyError(f"Recursion depth exceeded ({MAX_DEPTH})")
        self._tick()

        # Atoms
        if isinstance(expr, (int, float, bool, str)) or expr is None:
            return expr
        if isinstance(expr, Symbol):
            return env.lookup(expr.name)

        # List (function call or special form)
        if not isinstance(expr, list) or len(expr) == 0:
            raise LispyError(f"Cannot evaluate: {expr}")

        head = expr[0]

        # Special forms
        if isinstance(head, Symbol):
            name = head.name

            if name == "quote":
                if len(expr) != 2:
                    raise LispyError("quote takes exactly 1 argument")
                return expr[1]

            if name == "if":
                if len(expr) < 3:
                    raise LispyError("if requires condition and then-branch")
                cond = self.eval(expr[1], env, depth + 1)
                if cond and cond is not None:
                    return self.eval(expr[2], env, depth + 1)
                elif len(expr) > 3:
                    return self.eval(expr[3], env, depth + 1)
                return None

            if name == "cond":
                for clause in expr[1:]:
                    if not isinstance(clause, list) or len(clause) < 2:
                        raise LispyError("cond clause must be (test expr...)")
                    test = clause[0]
                    if (isinstance(test, Symbol) and test.name == "else") or \
                       self.eval(test, env, depth + 1):
                        result = None
                        for e in clause[1:]:
                            result = self.eval(e, env, depth + 1)
                        return result
                return None

            if name == "define":
                if len(expr) < 3:
                    raise LispyError("define requires name and value")
                if isinstance(expr[1], Symbol):
                    val = self.eval(expr[2], env, depth + 1)
                    env.set(expr[1].name, val)
                    return val
                elif isinstance(expr[1], list):
                    # (define (f x y) body...)
                    fname = expr[1][0]
                    if not isinstance(fname, Symbol):
                        raise LispyError("define function name must be symbol")
                    params = [p.name for p in expr[1][1:] if isinstance(p, Symbol)]
                    lam = Lambda(params, expr[2:], env)
                    env.set(fname.name, lam)
                    return lam
                raise LispyError("define: invalid syntax")

            if name == "lambda":
                if len(expr) < 3:
                    raise LispyError("lambda requires params and body")
                params = [p.name for p in expr[1] if isinstance(p, Symbol)]
                return Lambda(params, expr[2:], env)

            if name == "let":
                if len(expr) < 3:
                    raise LispyError("let requires bindings and body")
                local = Env(parent=env)
                for binding in expr[1]:
                    if not isinstance(binding, list) or len(binding) != 2:
                        raise LispyError("let binding must be (name value)")
                    bname = binding[0]
                    if not isinstance(bname, Symbol):
                        raise LispyError("let binding name must be symbol")
                    local.set(bname.name, self.eval(binding[1], env, depth + 1))
                result = None
                for body_expr in expr[2:]:
                    result = self.eval(body_expr, local, depth + 1)
                return result

            if name == "begin":
                result = None
                for e in expr[1:]:
                    result = self.eval(e, env, depth + 1)
                return result

            if name == "when":
                if len(expr) < 3:
                    raise LispyError("when requires condition and body")
                cond = self.eval(expr[1], env, depth + 1)
                if cond and cond is not None:
                    result = None
                    for e in expr[2:]:
                        result = self.eval(e, env, depth + 1)
                    return result
                return None

            if name == "set!":
                if len(expr) != 3:
                    raise LispyError("set! requires name and value")
                sym = expr[1]
                if not isinstance(sym, Symbol):
                    raise LispyError("set! name must be symbol")
                val = self.eval(expr[2], env, depth + 1)
                # Walk scope chain to find and update
                e = env
                while e is not None:
                    if sym.name in e.bindings:
                        e.bindings[sym.name] = val
                        return val
                    e = e.parent
                raise LispyError(f"set!: unbound symbol: {sym.name}")

            if name == "and":
                result: Any = True
                for e in expr[1:]:
                    result = self.eval(e, env, depth + 1)
                    if not result:
                        return result
                return result

            if name == "or":
                result = False
                for e in expr[1:]:
                    result = self.eval(e, env, depth + 1)
                    if result:
                        return result
                return result

            if name == "sub-sim":
                return self._run_sub_sim(expr, env, depth)

        # Function call
        func = self.eval(head, env, depth + 1)
        args = [self.eval(a, env, depth + 1) for a in expr[1:]]

        if isinstance(func, Lambda):
            if len(args) != len(func.params):
                raise LispyError(
                    f"Arity mismatch: {func.params} expects {len(func.params)}, got {len(args)}"
                )
            call_env = Env(
                dict(zip(func.params, args)),
                func.closure,
            )
            result = None
            for body_expr in func.body:
                result = self.eval(body_expr, call_env, depth + 1)
            return result

        if callable(func):
            return func(*args)

        raise LispyError(f"Not callable: {func}")

    def _run_sub_sim(self, expr: list, env: Env, depth: int) -> Any:
        """Run a sub-simulation at the next depth level."""
        if len(expr) < 2:
            raise LispyError("sub-sim requires at least an expression")

        if self.sim_depth >= MAX_SUB_SIM_DEPTH:
            raise LispyError(
                f"Sub-simulation depth limit ({MAX_SUB_SIM_DEPTH}) reached"
            )

        # (sub-sim expr) or (sub-sim label expr)
        if len(expr) == 2:
            label = f"sub-sim-depth-{self.sim_depth + 1}"
            sim_expr = self.eval(expr[1], env, depth + 1)
        else:
            label = self.eval(expr[1], env, depth + 1)
            sim_expr = self.eval(expr[2], env, depth + 1)

        child_seed = self.rng.randint(0, 2**31)
        child = Lispy(
            seed=child_seed,
            sim_depth=self.sim_depth + 1,
            step_limit=self.step_limit // 2,
        )

        # If sim_expr is a list (quoted s-expression), convert back to source
        if isinstance(sim_expr, list):
            source = _to_source(sim_expr)
        elif isinstance(sim_expr, str):
            source = sim_expr
        else:
            source = str(sim_expr)

        try:
            result = child.eval_string(source)
        except LispyError as exc:
            result = f"sub-sim-error: {exc}"

        log_entry = {
            "label": str(label),
            "depth": self.sim_depth + 1,
            "source": source[:500],
            "result": _safe_repr(result),
            "steps": child.steps,
            "child_logs": child.sub_sim_log,
        }
        self.sub_sim_log.append(log_entry)
        return result

    def _make_global_env(self) -> Env:
        """Create the global environment with safe builtins."""
        rng = self.rng
        builtins: dict[str, Any] = {}

        # Arithmetic
        builtins["+"] = lambda *args: sum(args)
        builtins["-"] = lambda a, b=None: -a if b is None else a - b
        builtins["*"] = lambda *args: _product(args)
        builtins["/"] = lambda a, b: a / b if b != 0 else float("inf")
        builtins["mod"] = lambda a, b: a % b if b != 0 else 0
        builtins["abs"] = lambda a: abs(a)
        builtins["min"] = lambda *args: min(args)
        builtins["max"] = lambda *args: max(args)
        builtins["floor"] = lambda a: int(math.floor(a))
        builtins["ceil"] = lambda a: int(math.ceil(a))
        builtins["round"] = lambda a: round(a)
        builtins["sqrt"] = lambda a: math.sqrt(max(0, a))
        builtins["pow"] = lambda a, b: a ** b
        builtins["log"] = lambda a: math.log(max(1e-10, a))

        # Comparison
        builtins["<"] = lambda a, b: a < b
        builtins[">"] = lambda a, b: a > b
        builtins["="] = lambda a, b: a == b
        builtins["!="] = lambda a, b: a != b
        builtins["<="] = lambda a, b: a <= b
        builtins[">="] = lambda a, b: a >= b

        # Logic
        builtins["not"] = lambda a: not a

        # Type checks
        builtins["number?"] = lambda a: isinstance(a, (int, float))
        builtins["string?"] = lambda a: isinstance(a, str)
        builtins["list?"] = lambda a: isinstance(a, list)
        builtins["nil?"] = lambda a: a is None
        builtins["bool?"] = lambda a: isinstance(a, bool)
        builtins["symbol?"] = lambda a: isinstance(a, Symbol)

        # List operations
        builtins["list"] = lambda *args: list(args)
        builtins["car"] = lambda lst: lst[0] if lst else None
        builtins["cdr"] = lambda lst: lst[1:] if lst else []
        builtins["cons"] = lambda a, lst: [a] + (lst if isinstance(lst, list) else [lst])
        builtins["length"] = lambda lst: len(lst) if isinstance(lst, (list, str)) else 0
        builtins["nth"] = lambda lst, n: lst[n] if isinstance(lst, list) and 0 <= n < len(lst) else None
        builtins["append"] = lambda *lsts: sum((l if isinstance(l, list) else [l] for l in lsts), [])
        builtins["reverse"] = lambda lst: list(reversed(lst)) if isinstance(lst, list) else lst
        builtins["range"] = lambda a, b=None: list(range(a)) if b is None else list(range(a, b))
        builtins["sort"] = lambda lst: sorted(lst) if isinstance(lst, list) else lst

        # Higher-order
        def lispy_map(fn, lst):
            if not isinstance(lst, list):
                return []
            results = []
            for item in lst:
                if isinstance(fn, Lambda):
                    r = self.eval([Symbol("__map_fn__"), item], Env({"__map_fn__": fn}, self.global_env))
                elif callable(fn):
                    r = fn(item)
                else:
                    raise LispyError(f"map: not callable: {fn}")
                results.append(r)
            return results
        builtins["map"] = lispy_map

        def lispy_filter(fn, lst):
            if not isinstance(lst, list):
                return []
            results = []
            for item in lst:
                if isinstance(fn, Lambda):
                    r = self.eval([Symbol("__filter_fn__"), item], Env({"__filter_fn__": fn}, self.global_env))
                elif callable(fn):
                    r = fn(item)
                else:
                    raise LispyError(f"filter: not callable: {fn}")
                if r:
                    results.append(item)
            return results
        builtins["filter"] = lispy_filter

        def lispy_reduce(fn, lst, init=None):
            if not isinstance(lst, list) or len(lst) == 0:
                return init
            acc = init if init is not None else lst[0]
            start = 0 if init is not None else 1
            for item in lst[start:]:
                if isinstance(fn, Lambda):
                    acc = self.eval(
                        [Symbol("__reduce_fn__"), acc, item],
                        Env({"__reduce_fn__": fn}, self.global_env),
                    )
                elif callable(fn):
                    acc = fn(acc, item)
                else:
                    raise LispyError(f"reduce: not callable: {fn}")
            return acc
        builtins["reduce"] = lispy_reduce

        # String operations
        builtins["str"] = lambda *args: "".join(str(a) for a in args)
        builtins["str-append"] = lambda *args: "".join(str(a) for a in args)
        builtins["str-length"] = lambda s: len(s) if isinstance(s, str) else 0
        builtins["str-upper"] = lambda s: s.upper() if isinstance(s, str) else s
        builtins["str-lower"] = lambda s: s.lower() if isinstance(s, str) else s
        builtins["str-contains"] = lambda s, sub: sub in s if isinstance(s, str) else False
        builtins["str-split"] = lambda s, sep=" ": s.split(sep) if isinstance(s, str) else []
        builtins["number->string"] = lambda n: str(n)
        builtins["string->number"] = lambda s: float(s) if isinstance(s, str) else 0

        # Hash maps (as lists of pairs)
        builtins["make-dict"] = lambda *pairs: {pairs[i]: pairs[i + 1] for i in range(0, len(pairs), 2)}
        builtins["dict-get"] = lambda d, k, default=None: d.get(k, default) if isinstance(d, dict) else default
        builtins["dict-set"] = lambda d, k, v: {**d, k: v} if isinstance(d, dict) else {k: v}
        builtins["dict-keys"] = lambda d: list(d.keys()) if isinstance(d, dict) else []
        builtins["dict-values"] = lambda d: list(d.values()) if isinstance(d, dict) else []

        # Randomness (seeded, deterministic)
        builtins["random"] = lambda: rng.random()
        builtins["random-int"] = lambda lo, hi: rng.randint(lo, hi)
        builtins["random-float"] = lambda lo, hi: rng.uniform(lo, hi)
        builtins["random-choice"] = lambda lst: rng.choice(lst) if isinstance(lst, list) and lst else None
        builtins["random-gauss"] = lambda mu, sigma: rng.gauss(mu, sigma)

        # Utility
        builtins["print"] = lambda *args: " ".join(str(a) for a in args)
        builtins["type-of"] = lambda a: type(a).__name__
        builtins["equal?"] = lambda a, b: a == b

        # Math constants
        builtins["pi"] = math.pi
        builtins["e"] = math.e

        return Env(builtins)


def _product(args: tuple) -> int | float:
    """Multiply all arguments."""
    result: int | float = 1
    for a in args:
        result *= a
    return result


def _to_source(expr: Any) -> str:
    """Convert a LisPy AST back to source string."""
    if isinstance(expr, list):
        return "(" + " ".join(_to_source(e) for e in expr) + ")"
    if isinstance(expr, Symbol):
        return expr.name
    if isinstance(expr, str):
        return f'"{expr}"'
    if isinstance(expr, bool):
        return "#t" if expr else "#f"
    if expr is None:
        return "nil"
    return str(expr)


def _safe_repr(value: Any) -> str:
    """Safe string representation of a value for logging."""
    s = _to_source(value) if isinstance(value, (list, Symbol)) else str(value)
    return s[:200]
