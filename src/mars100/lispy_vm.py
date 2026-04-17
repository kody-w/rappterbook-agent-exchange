"""
Minimal safe-eval LisPy interpreter for Mars-100 sub-simulations.

S-expression evaluator with arithmetic, comparisons, let/if/cond/lambda/define,
list ops. NO I/O, NO imports, NO file access — pure computation only.

Safety limits: max recursion depth, max evaluation steps, max AST nodes,
max list size. Deterministic (no built-in randomness).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


MAX_STEPS = 10_000
MAX_DEPTH = 200
MAX_AST_NODES = 500
MAX_LIST_SIZE = 100
MAX_STRING_LEN = 1000


class LispyError(Exception):
    """Base error for the LisPy VM."""


class LispySyntaxError(LispyError):
    """Parsing error."""


class LispyRuntimeError(LispyError):
    """Evaluation error."""


class LispyBudgetExceeded(LispyError):
    """Step or depth budget exhausted."""


TOKEN_RE = re.compile(r"""(\(|\)|'|"[^"]*"|;[^\n]*|[^\s()"';]+)""")


def tokenize(source: str) -> list[str]:
    """Tokenize a LisPy source string into a list of tokens."""
    tokens = TOKEN_RE.findall(source)
    return [t for t in tokens if not t.startswith(";")]


from typing import Union
Atom = Union[int, float, str, bool, None]
Expr = Union[Atom, list]


def _parse_atom(token: str) -> Atom:
    """Parse a single atom token."""
    if token == "#t":
        return True
    if token == "#f":
        return False
    if token == "nil":
        return None
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _count_nodes(expr: Expr) -> int:
    """Count AST nodes for safety limit."""
    if not isinstance(expr, list):
        return 1
    return 1 + sum(_count_nodes(e) for e in expr)


def _parse_expr(tokens: list[str], pos: int) -> tuple[Expr, int]:
    """Parse one expression starting at pos."""
    if pos >= len(tokens):
        raise LispySyntaxError("unexpected EOF")
    token = tokens[pos]
    if token == "(":
        return _parse_list(tokens, pos + 1)
    if token == ")":
        raise LispySyntaxError("unexpected ')'")
    if token == "'":
        inner, new_pos = _parse_expr(tokens, pos + 1)
        return ["quote", inner], new_pos
    return _parse_atom(token), pos + 1


def _parse_list(tokens: list[str], pos: int) -> tuple[list, int]:
    """Parse a list starting after the opening '('."""
    items: list[Expr] = []
    while pos < len(tokens) and tokens[pos] != ")":
        expr, pos = _parse_expr(tokens, pos)
        items.append(expr)
    if pos >= len(tokens):
        raise LispySyntaxError("missing closing ')'")
    return items, pos + 1


def parse_all(source: str) -> list[Expr]:
    """Parse a source string into a list of top-level expressions."""
    tokens = tokenize(source)
    exprs: list[Expr] = []
    pos = 0
    while pos < len(tokens):
        expr, pos = _parse_expr(tokens, pos)
        exprs.append(expr)
    return exprs


class Env:
    """Lexically scoped environment."""
    def __init__(self, bindings: dict[str, Any] | None = None,
                 parent: "Env | None" = None) -> None:
        self.bindings: dict[str, Any] = bindings or {}
        self.parent = parent

    def lookup(self, name: str) -> Any:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        raise LispyRuntimeError(f"unbound symbol: {name}")

    def define(self, name: str, value: Any) -> None:
        self.bindings[name] = value


@dataclass
class Closure:
    """A lambda closure."""
    params: list[str]
    body: Expr
    env: Env


@dataclass
class VMState:
    """Mutable evaluation state for budget tracking."""
    steps: int = 0
    max_steps: int = MAX_STEPS
    max_depth: int = MAX_DEPTH


def evaluate(expr: Expr, env: Env, state: VMState | None = None,
             depth: int = 0) -> Any:
    """Evaluate a LisPy expression in the given environment."""
    if state is None:
        state = VMState()
    state.steps += 1
    if state.steps > state.max_steps:
        raise LispyBudgetExceeded(f"step budget exceeded ({state.max_steps} steps)")
    if depth > state.max_depth:
        raise LispyBudgetExceeded(f"recursion depth exceeded ({state.max_depth})")

    if isinstance(expr, (int, float, bool)) or expr is None:
        return expr
    if isinstance(expr, str):
        if expr.startswith('"'):
            return expr
        return env.lookup(expr)
    if not isinstance(expr, list) or len(expr) == 0:
        return expr

    head = expr[0]

    if head == "quote":
        if len(expr) != 2:
            raise LispyRuntimeError("quote requires exactly 1 argument")
        return expr[1]
    if head == "if":
        if len(expr) not in (3, 4):
            raise LispyRuntimeError("if requires 2 or 3 arguments")
        cond = evaluate(expr[1], env, state, depth + 1)
        if cond and cond is not None:
            return evaluate(expr[2], env, state, depth + 1)
        if len(expr) == 4:
            return evaluate(expr[3], env, state, depth + 1)
        return None
    if head == "cond":
        for clause in expr[1:]:
            if not isinstance(clause, list) or len(clause) != 2:
                raise LispyRuntimeError("cond clause must be (test expr)")
            test = evaluate(clause[0], env, state, depth + 1)
            if test and test is not None:
                return evaluate(clause[1], env, state, depth + 1)
        return None
    if head == "define":
        if len(expr) != 3:
            raise LispyRuntimeError("define requires name and value")
        name = expr[1]
        if not isinstance(name, str):
            raise LispyRuntimeError("define name must be a symbol")
        val = evaluate(expr[2], env, state, depth + 1)
        env.define(name, val)
        return val
    if head == "let":
        if len(expr) != 3:
            raise LispyRuntimeError("let requires bindings and body")
        bindings = expr[1]
        if not isinstance(bindings, list):
            raise LispyRuntimeError("let bindings must be a list")
        child = Env(parent=env)
        for b in bindings:
            if not isinstance(b, list) or len(b) != 2:
                raise LispyRuntimeError("let binding must be (name value)")
            bname, val_expr = b
            if not isinstance(bname, str):
                raise LispyRuntimeError("let binding name must be a symbol")
            child.define(bname, evaluate(val_expr, env, state, depth + 1))
        return evaluate(expr[2], child, state, depth + 1)
    if head == "lambda":
        if len(expr) != 3:
            raise LispyRuntimeError("lambda requires params and body")
        params = expr[1]
        if not isinstance(params, list):
            raise LispyRuntimeError("lambda params must be a list")
        for p in params:
            if not isinstance(p, str):
                raise LispyRuntimeError("lambda param must be a symbol")
        return Closure(params=params, body=expr[2], env=env)
    if head == "begin":
        result: Any = None
        for sub in expr[1:]:
            result = evaluate(sub, env, state, depth + 1)
        return result
    if head == "and":
        result = True
        for sub in expr[1:]:
            result = evaluate(sub, env, state, depth + 1)
            if not result:
                return result
        return result
    if head == "or":
        for sub in expr[1:]:
            result = evaluate(sub, env, state, depth + 1)
            if result:
                return result
        return False
    if head == "not":
        if len(expr) != 2:
            raise LispyRuntimeError("not requires exactly 1 argument")
        return not evaluate(expr[1], env, state, depth + 1)

    func = evaluate(head, env, state, depth + 1)
    args = [evaluate(a, env, state, depth + 1) for a in expr[1:]]

    if isinstance(func, Closure):
        if len(args) != len(func.params):
            raise LispyRuntimeError(f"expected {len(func.params)} args, got {len(args)}")
        child = Env(bindings=dict(zip(func.params, args)), parent=func.env)
        return evaluate(func.body, child, state, depth + 1)
    if callable(func):
        return func(*args)
    raise LispyRuntimeError(f"cannot call {type(func).__name__}: {func}")


def _make_stdlib() -> dict[str, Any]:
    """Create the standard library of built-in functions."""
    def _add(*args): return sum(args)
    def _sub(a, b): return a - b
    def _mul(*args):
        r = 1
        for x in args: r *= x
        return r
    def _div(a, b):
        if b == 0: raise LispyRuntimeError("division by zero")
        return a / b
    def _mod(a, b):
        if b == 0: raise LispyRuntimeError("modulo by zero")
        return a % b
    def _gt(a, b): return a > b
    def _lt(a, b): return a < b
    def _gte(a, b): return a >= b
    def _lte(a, b): return a <= b
    def _eq(a, b): return a == b
    def _neq(a, b): return a != b
    def _abs_val(x): return abs(x)
    def _min_val(*args): return min(args)
    def _max_val(*args): return max(args)
    def _floor(x): return int(math.floor(x))
    def _ceil(x): return int(math.ceil(x))
    def _sqrt(x):
        if x < 0: raise LispyRuntimeError("sqrt of negative")
        return math.sqrt(x)
    def _list_fn(*args):
        result = list(args)
        if len(result) > MAX_LIST_SIZE:
            raise LispyBudgetExceeded(f"list size exceeds {MAX_LIST_SIZE}")
        return result
    def _car(lst):
        if not isinstance(lst, list) or len(lst) == 0:
            raise LispyRuntimeError("car of empty or non-list")
        return lst[0]
    def _cdr(lst):
        if not isinstance(lst, list):
            raise LispyRuntimeError("cdr of non-list")
        return lst[1:]
    def _cons(head, tail):
        if not isinstance(tail, list):
            raise LispyRuntimeError("cons: tail must be a list")
        result = [head] + tail
        if len(result) > MAX_LIST_SIZE:
            raise LispyBudgetExceeded(f"list size exceeds {MAX_LIST_SIZE}")
        return result
    def _length(lst):
        if not isinstance(lst, list):
            raise LispyRuntimeError("length of non-list")
        return len(lst)
    def _nth(lst, n):
        if not isinstance(lst, list):
            raise LispyRuntimeError("nth of non-list")
        idx = int(n)
        if idx < 0 or idx >= len(lst):
            raise LispyRuntimeError(f"index {idx} out of range")
        return lst[idx]
    def _is_list(x): return isinstance(x, list)
    def _is_number(x): return isinstance(x, (int, float))
    def _is_nil(x): return x is None
    def _string_append(*args):
        result = "".join(str(a) for a in args)
        if len(result) > MAX_STRING_LEN:
            raise LispyBudgetExceeded(f"string exceeds {MAX_STRING_LEN}")
        return result

    return {
        "+": _add, "-": _sub, "*": _mul, "/": _div, "%": _mod,
        ">": _gt, "<": _lt, ">=": _gte, "<=": _lte, "=": _eq, "!=": _neq,
        "abs": _abs_val, "min": _min_val, "max": _max_val,
        "floor": _floor, "ceil": _ceil, "sqrt": _sqrt,
        "list": _list_fn, "car": _car, "cdr": _cdr, "cons": _cons,
        "length": _length, "nth": _nth,
        "list?": _is_list, "number?": _is_number, "nil?": _is_nil,
        "string-append": _string_append,
        "pi": math.pi, "e": math.e,
        "#t": True, "#f": False, "nil": None,
    }


STDLIB = _make_stdlib()


def make_env(extra: dict[str, Any] | None = None) -> Env:
    """Create a fresh environment with stdlib + optional extra bindings."""
    bindings = dict(STDLIB)
    if extra:
        bindings.update(extra)
    return Env(bindings=bindings)


def run(source: str, extra_bindings: dict[str, Any] | None = None,
        max_steps: int = MAX_STEPS, max_depth: int = MAX_DEPTH) -> Any:
    """Parse and evaluate a LisPy program. Returns the last expression's value."""
    exprs = parse_all(source)
    total_nodes = sum(_count_nodes(e) for e in exprs)
    if total_nodes > MAX_AST_NODES:
        raise LispyBudgetExceeded(f"AST size {total_nodes} exceeds limit {MAX_AST_NODES}")
    env = make_env(extra_bindings)
    state = VMState(max_steps=max_steps, max_depth=max_depth)
    result: Any = None
    for expr in exprs:
        result = evaluate(expr, env, state)
    return result
