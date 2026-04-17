"""lispy.py -- Sandboxed LisPy interpreter for Mars-100 sub-simulations.

A minimal, safe-eval s-expression interpreter. No I/O, no imports,
no file access. Pure computation with step-metered execution.

Supports recursive sub-simulations via (sub-sim depth body) special
form. Max depth 3. Each sub-sim gets a copied environment with
decremented depth budget and fresh step counters.

Usage:
    from lispy import parse, evaluate, default_env, run

    env = default_env(depth_budget=3, max_steps=10000)
    ast = parse("(+ 1 2)")
    result = evaluate(ast, env)  # -> 3

    result = run("(let ((x 10)) (* x x))")  # -> 100
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LispyError(Exception):
    """Base error for all LisPy failures."""

class ParseError(LispyError):
    """Malformed s-expression."""

class EvalError(LispyError):
    """Runtime evaluation failure."""

class StepLimitError(LispyError):
    """Exceeded maximum evaluation steps."""

class DepthLimitError(LispyError):
    """Sub-simulation depth exceeded."""

class RecursionLimitError(LispyError):
    """Interpreter recursion too deep."""


# ---------------------------------------------------------------------------
# Tokenizer + Parser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""(\(|\)|'|"(?:[^"\\]|\\.)*"|;[^\n]*|[^\s()'";]+)""")


def tokenize(source: str) -> list[str]:
    """Split source into tokens, discarding comments."""
    tokens = _TOKEN_RE.findall(source)
    return [t for t in tokens if not t.startswith(";")]


def _atom(token: str):
    """Convert a token string to a Python value."""
    if token.startswith('"') and token.endswith('"'):
        # Keep quotes so evaluator can distinguish strings from symbols
        return token
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    if token == "#t":
        return True
    if token == "#f":
        return False
    if token == "nil":
        return None
    return token  # symbol


def parse(source: str):
    """Parse an s-expression string into nested Python lists."""
    tokens = tokenize(source)
    if not tokens:
        raise ParseError("empty expression")
    result, pos = _read(tokens, 0)
    return result


def parse_all(source: str) -> list:
    """Parse multiple top-level expressions."""
    tokens = tokenize(source)
    results = []
    pos = 0
    while pos < len(tokens):
        expr, pos = _read(tokens, pos)
        results.append(expr)
    return results


def _read(tokens: list[str], pos: int):
    """Recursive-descent reader."""
    if pos >= len(tokens):
        raise ParseError("unexpected end of input")
    token = tokens[pos]
    if token == "'":
        inner, pos = _read(tokens, pos + 1)
        return ["quote", inner], pos
    if token == "(":
        lst: list = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ")":
            elem, pos = _read(tokens, pos)
            lst.append(elem)
        if pos >= len(tokens):
            raise ParseError("unmatched '('")
        return lst, pos + 1
    if token == ")":
        raise ParseError("unexpected ')'")
    return _atom(token), pos + 1


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class Env:
    """Lexical environment with step metering and depth tracking."""
    bindings: dict = field(default_factory=dict)
    parent: Env | None = None
    steps: list[int] = field(default_factory=lambda: [0])
    max_steps: int = 10000
    depth_budget: int = 3
    max_recursion: int = 200
    _recursion: list[int] = field(default_factory=lambda: [0])

    def lookup(self, name: str):
        """Find a binding, walking up the chain."""
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        raise EvalError(f"unbound symbol: {name}")

    def define(self, name: str, value) -> None:
        """Bind a name in this environment."""
        self.bindings[name] = value

    def child(self, bindings: dict | None = None) -> Env:
        """Create a child environment sharing step/recursion counters."""
        return Env(
            bindings=bindings or {},
            parent=self,
            steps=self.steps,
            max_steps=self.max_steps,
            depth_budget=self.depth_budget,
            max_recursion=self.max_recursion,
            _recursion=self._recursion,
        )

    def charge(self, n: int = 1) -> None:
        """Charge n steps. Raise if budget exhausted."""
        self.steps[0] += n
        if self.steps[0] > self.max_steps:
            raise StepLimitError(
                f"exceeded {self.max_steps} steps (at {self.steps[0]})"
            )

    def enter_recursion(self) -> None:
        """Track interpreter recursion depth."""
        self._recursion[0] += 1
        if self._recursion[0] > self.max_recursion:
            raise RecursionLimitError(
                f"interpreter recursion depth exceeded {self.max_recursion}"
            )

    def exit_recursion(self) -> None:
        """Unwind one recursion level."""
        self._recursion[0] -= 1


# ---------------------------------------------------------------------------
# Lambda
# ---------------------------------------------------------------------------

@dataclass
class Lambda:
    """A user-defined function (closure)."""
    params: list[str]
    body: list
    closure: Env
    name: str | None = None

    def __repr__(self) -> str:
        name = self.name or "anonymous"
        return f"<lambda:{name}({', '.join(self.params)})>"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

SPECIAL_FORMS = frozenset({
    "quote", "if", "cond", "and", "or", "define", "set!",
    "let", "lambda", "begin", "do", "sub-sim",
})


def evaluate(expr, env: Env):
    """Evaluate a LisPy expression in the given environment."""
    env.charge()
    env.enter_recursion()
    try:
        return _eval_inner(expr, env)
    finally:
        env.exit_recursion()


def _eval_inner(expr, env: Env):
    """Core evaluator -- dispatches on expression type."""
    # Atoms
    if isinstance(expr, (int, float, bool)) or expr is None:
        return expr
    if isinstance(expr, str):
        if expr.startswith('"') and expr.endswith('"'):
            return expr[1:-1].replace('\\"', '"').replace("\\n", "\n")
        return env.lookup(expr)

    if not isinstance(expr, list) or len(expr) == 0:
        raise EvalError(f"cannot evaluate: {expr!r}")

    head = expr[0]

    # --- Special forms (short-circuiting, binding, control flow) ---

    if head == "quote":
        if len(expr) != 2:
            raise EvalError("quote requires exactly 1 argument")
        return expr[1]

    if head == "if":
        if len(expr) not in (3, 4):
            raise EvalError("if requires 2 or 3 arguments")
        cond_val = evaluate(expr[1], env)
        if cond_val and cond_val is not None:
            return evaluate(expr[2], env)
        if len(expr) == 4:
            return evaluate(expr[3], env)
        return None

    if head == "cond":
        for clause in expr[1:]:
            if not isinstance(clause, list) or len(clause) < 2:
                raise EvalError(f"bad cond clause: {clause!r}")
            test = clause[0]
            if test == "else" or evaluate(test, env):
                result = None
                for body_expr in clause[1:]:
                    result = evaluate(body_expr, env)
                return result
        return None

    if head == "and":
        result = True
        for arg in expr[1:]:
            result = evaluate(arg, env)
            if not result:
                return result
        return result

    if head == "or":
        result = False
        for arg in expr[1:]:
            result = evaluate(arg, env)
            if result:
                return result
        return result

    if head == "define":
        if len(expr) != 3:
            raise EvalError("define requires name and value")
        name = expr[1]
        if not isinstance(name, str):
            raise EvalError(f"define name must be a symbol, got {name!r}")
        val = evaluate(expr[2], env)
        env.define(name, val)
        return val

    if head == "set!":
        if len(expr) != 3:
            raise EvalError("set! requires name and value")
        name = expr[1]
        val = evaluate(expr[2], env)
        e = env
        while e is not None:
            if name in e.bindings:
                e.bindings[name] = val
                return val
            e = e.parent
        raise EvalError(f"set!: unbound symbol: {name}")

    if head == "let":
        if len(expr) < 3:
            raise EvalError("let requires bindings and body")
        bindings_list = expr[1]
        if not isinstance(bindings_list, list):
            raise EvalError("let bindings must be a list")
        child = env.child()
        for binding in bindings_list:
            if not isinstance(binding, list) or len(binding) != 2:
                raise EvalError(f"bad let binding: {binding!r}")
            bname, val_expr = binding
            child.define(bname, evaluate(val_expr, env))
        result = None
        for body_expr in expr[2:]:
            result = evaluate(body_expr, child)
        return result

    if head == "lambda":
        if len(expr) < 3:
            raise EvalError("lambda requires params and body")
        params = expr[1]
        if not isinstance(params, list):
            raise EvalError("lambda params must be a list")
        for p in params:
            if not isinstance(p, str):
                raise EvalError(f"lambda param must be symbol, got {p!r}")
        return Lambda(params=params, body=expr[2:], closure=env)

    if head == "begin":
        result = None
        for sub in expr[1:]:
            result = evaluate(sub, env)
        return result

    if head == "do":
        if len(expr) != 3:
            raise EvalError("do requires count and body")
        n = evaluate(expr[1], env)
        if not isinstance(n, int) or n < 0:
            raise EvalError(f"do count must be non-negative int, got {n!r}")
        env.charge(n)
        result = None
        for _ in range(n):
            result = evaluate(expr[2], env)
        return result

    if head == "sub-sim":
        if len(expr) != 3:
            raise EvalError("sub-sim requires depth-budget and body")
        requested_depth = evaluate(expr[1], env)
        if not isinstance(requested_depth, int) or requested_depth < 0:
            raise EvalError("sub-sim depth must be non-negative int")
        if requested_depth >= env.depth_budget:
            raise DepthLimitError(
                f"sub-sim depth {requested_depth} exceeds budget "
                f"{env.depth_budget}"
            )
        # Isolated child: shallow-copied bindings, fresh counters
        child_env = Env(
            bindings=dict(env.bindings),
            parent=None,
            steps=[0],
            max_steps=env.max_steps,
            depth_budget=requested_depth,
            max_recursion=env.max_recursion,
            _recursion=[0],
        )
        _install_builtins(child_env)
        return evaluate(expr[2], child_env)

    # --- Function application ---
    func = evaluate(head, env)
    args = [evaluate(a, env) for a in expr[1:]]
    env.charge(len(args))

    if callable(func):
        return func(*args)
    if isinstance(func, Lambda):
        if len(args) != len(func.params):
            raise EvalError(
                f"{func} expects {len(func.params)} args, got {len(args)}"
            )
        call_env = func.closure.child(dict(zip(func.params, args)))
        result = None
        for body_expr in func.body:
            result = evaluate(body_expr, call_env)
        return result

    raise EvalError(f"not callable: {func!r}")


# ---------------------------------------------------------------------------
# Builtins
# ---------------------------------------------------------------------------

def _install_builtins(env: Env) -> None:
    """Install built-in functions into an environment."""

    def _add(*args):
        return sum(args)

    def _sub(*args):
        if len(args) == 1:
            return -args[0]
        return args[0] - sum(args[1:])

    def _mul(*args):
        r = 1
        for a in args:
            r *= a
        return r

    def _div(a, b):
        if b == 0:
            raise EvalError("division by zero")
        return a / b

    def _mod(a, b):
        if b == 0:
            raise EvalError("modulo by zero")
        return a % b

    def _eq(*args):
        return all(a == args[0] for a in args[1:])

    builtins: dict = {
        "+": _add, "-": _sub, "*": _mul, "/": _div, "mod": _mod,
        "=": _eq,
        "<": lambda a, b: a < b,
        ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        ">=": lambda a, b: a >= b,
        "not": lambda a: not a,
        "abs": lambda x: abs(x),
        "min": lambda *a: min(a[0]) if len(a) == 1 and isinstance(a[0], list) else min(a),
        "max": lambda *a: max(a[0]) if len(a) == 1 and isinstance(a[0], list) else max(a),
        "round": lambda x: round(x),
        "str": lambda *a: " ".join(str(x) for x in a),
        "list?": lambda x: isinstance(x, list),
        "number?": lambda x: isinstance(x, (int, float)),
        "string?": lambda x: isinstance(x, str),
        "nil?": lambda x: x is None,
    }

    # List operations (charge steps proportional to list size)
    def _list(*args):
        env.charge(len(args))
        return list(args)

    def _car(lst):
        if not isinstance(lst, list) or len(lst) == 0:
            raise EvalError("car of empty or non-list")
        return lst[0]

    def _cdr(lst):
        if not isinstance(lst, list):
            raise EvalError("cdr of non-list")
        return lst[1:]

    def _cons(a, b):
        if not isinstance(b, list):
            raise EvalError("cons: second arg must be list")
        return [a] + b

    def _length(lst):
        if not isinstance(lst, list):
            raise EvalError("length of non-list")
        return len(lst)

    def _append(*lsts):
        result = []
        for lst in lsts:
            if not isinstance(lst, list):
                raise EvalError("append: all args must be lists")
            env.charge(len(lst))
            result.extend(lst)
        return result

    def _nth(lst, n):
        if not isinstance(lst, list):
            raise EvalError("nth: first arg must be list")
        if not isinstance(n, int) or n < 0 or n >= len(lst):
            raise EvalError(f"nth: index {n} out of range")
        return lst[n]

    def _map_fn(fn, lst):
        if not isinstance(lst, list):
            raise EvalError("map: second arg must be list")
        env.charge(len(lst))
        results = []
        for item in lst:
            if callable(fn):
                results.append(fn(item))
            elif isinstance(fn, Lambda):
                ce = fn.closure.child(dict(zip(fn.params, [item])))
                r = None
                for b in fn.body:
                    r = evaluate(b, ce)
                results.append(r)
            else:
                raise EvalError("map: first arg must be function")
        return results

    def _filter_fn(fn, lst):
        if not isinstance(lst, list):
            raise EvalError("filter: second arg must be list")
        env.charge(len(lst))
        results = []
        for item in lst:
            if callable(fn):
                keep = fn(item)
            elif isinstance(fn, Lambda):
                ce = fn.closure.child(dict(zip(fn.params, [item])))
                keep = None
                for b in fn.body:
                    keep = evaluate(b, ce)
            else:
                raise EvalError("filter: first arg must be function")
            if keep:
                results.append(item)
        return results

    builtins.update({
        "list": _list, "car": _car, "cdr": _cdr, "cons": _cons,
        "length": _length, "append": _append, "nth": _nth,
        "map": _map_fn, "filter": _filter_fn,
    })

    # Hash (dict) operations
    def _hash_get(h, k):
        if not isinstance(h, dict):
            raise EvalError("hash-get: first arg must be a dict")
        return h.get(k)

    def _hash_set(h, k, v):
        if not isinstance(h, dict):
            raise EvalError("hash-set: first arg must be a dict")
        new_h = dict(h)
        new_h[k] = v
        return new_h

    def _hash_keys(h):
        if not isinstance(h, dict):
            raise EvalError("hash-keys: arg must be a dict")
        return list(h.keys())

    def _make_hash(*pairs):
        if len(pairs) % 2 != 0:
            raise EvalError("make-hash requires even number of args")
        return dict(zip(pairs[0::2], pairs[1::2]))

    builtins.update({
        "hash-get": _hash_get, "hash-set": _hash_set,
        "hash-keys": _hash_keys, "make-hash": _make_hash,
    })

    env.bindings.update(builtins)


def default_env(
    depth_budget: int = 3,
    max_steps: int = 10000,
    max_recursion: int = 200,
    extra_bindings: dict | None = None,
) -> Env:
    """Create a fresh environment with builtins installed."""
    env = Env(
        depth_budget=depth_budget,
        max_steps=max_steps,
        max_recursion=max_recursion,
    )
    _install_builtins(env)
    if extra_bindings:
        env.bindings.update(extra_bindings)
    return env


def run(source: str, env: Env | None = None):
    """Parse and evaluate a LisPy source string. Returns last result."""
    if env is None:
        env = default_env()
    exprs = parse_all(source)
    result = None
    for expr in exprs:
        result = evaluate(expr, env)
    return result
