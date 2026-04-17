"""lispy.py -- Safe LisPy evaluator for recursive sub-simulations.

A minimal Lisp dialect designed for safe evaluation inside simulations.
No I/O, no imports, no file access — pure computation in a sandbox.

Features:
  - Arithmetic: +, -, *, /, %, abs, min, max, round
  - Comparison: =, <, >, <=, >=, !=
  - Logic: and, or, not
  - Lists: list, car, cdr, cons, length, nth, map, filter, append
  - Strings: str-append, str-length, substring
  - Control: if, cond, begin, let, define, lambda
  - Sub-sim: (sub-sim "label" expr) — spawns a child simulation
  - Quoting: (quote ...) or '(...)

Limits enforced:
  - max_steps: total evaluation steps before abort
  - max_depth: call stack depth
  - max_sim_depth: recursive sub-simulation nesting depth
  - max_subsims_per_frame: sub-sim count per evaluation
"""
from __future__ import annotations

from typing import Any, Callable


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LispyError(Exception):
    """Base error for LisPy evaluation."""
    pass


class LispySandboxError(LispyError):
    """Raised when sandbox limits are exceeded."""
    pass


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Str:
    """Wrapper for string values in LisPy."""
    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f'"{self.value}"'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Str):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)


class Lambda:
    """User-defined function."""
    __slots__ = ("params", "body", "env")

    def __init__(self, params: list[str], body: Any, env: "Env") -> None:
        self.params = params
        self.body = body
        self.env = env

    def __repr__(self) -> str:
        return f"<lambda ({' '.join(self.params)})>"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Env:
    """Variable environment with parent chain."""

    def __init__(self, bindings: dict[str, Any] | None = None,
                 parent: "Env | None" = None) -> None:
        self.bindings: dict[str, Any] = bindings or {}
        self.parent = parent

    def lookup(self, name: str) -> Any:
        """Look up a variable, walking the parent chain."""
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        raise LispyError(f"Undefined variable: {name}")

    def define(self, name: str, value: Any) -> None:
        """Define a variable in this environment."""
        self.bindings[name] = value

    def set(self, name: str, value: Any) -> None:
        """Set an existing variable (walks parent chain)."""
        if name in self.bindings:
            self.bindings[name] = value
            return
        if self.parent is not None:
            self.parent.set(name, value)
            return
        raise LispyError(f"Cannot set undefined variable: {name}")


# ---------------------------------------------------------------------------
# Tokenizer + Parser
# ---------------------------------------------------------------------------

def tokenize(source: str) -> list[str]:
    """Tokenize a LisPy expression into a list of tokens."""
    tokens: list[str] = []
    i = 0
    while i < len(source):
        ch = source[i]
        if ch in " \t\n\r":
            i += 1
        elif ch == ";":
            # Comment — skip to end of line
            while i < len(source) and source[i] != "\n":
                i += 1
        elif ch in "()":
            tokens.append(ch)
            i += 1
        elif ch == "'":
            tokens.append("'")
            i += 1
        elif ch == '"':
            # String literal
            j = i + 1
            while j < len(source) and source[j] != '"':
                if source[j] == "\\":
                    j += 1  # skip escaped char
                j += 1
            tokens.append(source[i:j + 1])
            i = j + 1
        else:
            # Symbol or number
            j = i
            while j < len(source) and source[j] not in " \t\n\r()\"';":
                j += 1
            tokens.append(source[i:j])
            i = j
    return tokens


def parse(source: str) -> Any:
    """Parse a LisPy source string into an AST."""
    tokens = tokenize(source)
    if not tokens:
        raise LispyError("Empty expression")
    ast, pos = _parse_expr(tokens, 0)
    return ast


def _parse_expr(tokens: list[str], pos: int) -> tuple[Any, int]:
    """Parse one expression starting at pos. Returns (ast, new_pos)."""
    if pos >= len(tokens):
        raise LispyError("Unexpected end of input")

    token = tokens[pos]

    if token == "'":
        # Quote shorthand
        inner, pos = _parse_expr(tokens, pos + 1)
        return ["quote", inner], pos

    if token == "(":
        pos += 1
        elements: list[Any] = []
        while pos < len(tokens) and tokens[pos] != ")":
            elem, pos = _parse_expr(tokens, pos)
            elements.append(elem)
        if pos >= len(tokens):
            raise LispyError("Unmatched opening parenthesis")
        return elements, pos + 1  # skip ')'

    if token == ")":
        raise LispyError("Unexpected closing parenthesis")

    # Atom
    return _parse_atom(token), pos + 1


def _parse_atom(token: str) -> Any:
    """Parse an atom: number, string, bool, or symbol."""
    if token.startswith('"') and token.endswith('"'):
        return Str(token[1:-1])  # string → Str wrapper (distinguishes from symbols)

    # Booleans
    if token == "#t" or token == "true":
        return True
    if token == "#f" or token == "false":
        return False
    if token == "nil":
        return None

    # Numbers
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass

    # Symbol
    return token


def to_sexp(ast: Any) -> str:
    """Convert an AST back to an s-expression string."""
    if ast is None:
        return "nil"
    if isinstance(ast, bool):
        return "#t" if ast else "#f"
    if isinstance(ast, (int, float)):
        return str(ast)
    if isinstance(ast, str):
        # Check if it looks like a symbol or string
        if any(c in ast for c in ' "()') or ast == "":
            return f'"{ast}"'
        return ast
    if isinstance(ast, Str):
        return f'"{ast.value}"'
    if isinstance(ast, list):
        inner = " ".join(to_sexp(x) for x in ast)
        return f"({inner})"
    return str(ast)


# ---------------------------------------------------------------------------
# Standard environment (built-in functions)
# ---------------------------------------------------------------------------

def make_standard_env() -> Env:
    """Create an environment with standard built-in functions."""
    env = Env()

    # Arithmetic
    env.define("+", lambda *args: sum(args))
    env.define("-", lambda a, b=None: -a if b is None else a - b)
    env.define("*", lambda *args: _product(args))
    env.define("/", lambda a, b: a / b if b != 0 else 0)
    env.define("%", lambda a, b: a % b if b != 0 else 0)
    env.define("abs", lambda x: abs(x))
    env.define("min", lambda *args: min(args))
    env.define("max", lambda *args: max(args))
    env.define("round", lambda x, n=0: round(x, n))

    # Comparison
    env.define("=", lambda a, b: a == b)
    env.define("<", lambda a, b: a < b)
    env.define(">", lambda a, b: a > b)
    env.define("<=", lambda a, b: a <= b)
    env.define(">=", lambda a, b: a >= b)
    env.define("!=", lambda a, b: a != b)

    # Logic
    env.define("not", lambda x: not x)

    # List operations
    env.define("list", lambda *args: list(args))
    env.define("car", lambda lst: lst[0] if lst else None)
    env.define("cdr", lambda lst: lst[1:] if lst else [])
    env.define("cons", lambda a, b: [a] + (b if isinstance(b, list) else [b]))
    env.define("length", lambda lst: len(lst) if isinstance(lst, (list, str)) else 0)
    env.define("nth", lambda lst, n: lst[n] if isinstance(lst, list) and 0 <= n < len(lst) else None)
    env.define("append", lambda *args: sum((a if isinstance(a, list) else [a] for a in args), []))
    env.define("null?", lambda x: x is None or (isinstance(x, list) and len(x) == 0))
    env.define("pair?", lambda x: isinstance(x, list) and len(x) > 0)

    # String operations
    env.define("str-append", lambda *args: "".join(str(a) for a in args))
    env.define("str-length", lambda s: len(s) if isinstance(s, str) else 0)

    # Type checks
    env.define("number?", lambda x: isinstance(x, (int, float)))
    env.define("string?", lambda x: isinstance(x, str))
    env.define("list?", lambda x: isinstance(x, list))
    env.define("boolean?", lambda x: isinstance(x, bool))

    # Constants
    env.define("pi", 3.141592653589793)
    env.define("e", 2.718281828459045)

    return env


def _product(args: tuple) -> float | int:
    """Multiply all arguments."""
    result: float | int = 1
    for a in args:
        result *= a
    return result


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Safe LisPy evaluator with resource limits.

    Attributes:
        max_steps: Maximum evaluation steps before LispySandboxError.
        max_depth: Maximum call stack depth.
        max_sim_depth: Maximum recursive sub-sim nesting.
        max_subsims_per_frame: Max sub-sim invocations per evaluation.
        sim_depth: Current sub-sim depth (0 = top level).
        steps: Current step count.
        subsim_callback: Optional callback for sub-sim handling.
    """

    def __init__(
        self,
        max_steps: int = 10000,
        max_depth: int = 64,
        max_sim_depth: int = 3,
        max_subsims_per_frame: int = 6,
        sim_depth: int = 0,
        subsim_callback: Callable | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.max_sim_depth = max_sim_depth
        self.max_subsims_per_frame = max_subsims_per_frame
        self.sim_depth = sim_depth
        self.subsim_callback = subsim_callback
        self.steps = 0
        self._subsim_count = 0

    def eval(self, expr: Any, env: Env, depth: int = 0) -> Any:
        """Evaluate a LisPy expression in the given environment."""
        self.steps += 1
        if self.steps > self.max_steps:
            raise LispySandboxError(
                f"Step limit exceeded ({self.max_steps})"
            )
        if depth > self.max_depth:
            raise LispySandboxError(
                f"Depth limit exceeded ({self.max_depth})"
            )

        # Atoms
        if expr is None:
            return None
        if isinstance(expr, (bool, int, float)):
            return expr
        if isinstance(expr, Str):
            return expr.value
        if isinstance(expr, str):
            # Could be a string literal or a symbol
            if expr.startswith('"') and expr.endswith('"'):
                return expr[1:-1]
            return env.lookup(expr)

        # List (compound expression)
        if not isinstance(expr, list):
            return expr
        if len(expr) == 0:
            return []

        head = expr[0]

        # Special forms
        if head == "quote":
            return expr[1] if len(expr) > 1 else None

        if head == "if":
            cond = self.eval(expr[1], env, depth + 1)
            if cond:
                return self.eval(expr[2], env, depth + 1)
            elif len(expr) > 3:
                return self.eval(expr[3], env, depth + 1)
            return None

        if head == "cond":
            for clause in expr[1:]:
                if not isinstance(clause, list) or len(clause) < 2:
                    continue
                test = clause[0]
                if test == "else" or self.eval(test, env, depth + 1):
                    return self.eval(clause[1], env, depth + 1)
            return None

        if head == "define":
            name = expr[1]
            value = self.eval(expr[2], env, depth + 1)
            env.define(name, value)
            return value

        if head == "set!":
            name = expr[1]
            value = self.eval(expr[2], env, depth + 1)
            env.set(name, value)
            return value

        if head == "let":
            bindings = expr[1]
            body = expr[2:]
            child_env = Env(parent=env)
            for binding in bindings:
                name = binding[0]
                val = self.eval(binding[1], env, depth + 1)
                child_env.define(name, val)
            result = None
            for b in body:
                result = self.eval(b, child_env, depth + 1)
            return result

        if head == "lambda":
            params = expr[1]
            body = expr[2] if len(expr) == 3 else ["begin"] + expr[2:]
            return Lambda(params, body, env)

        if head == "begin":
            result = None
            for e in expr[1:]:
                result = self.eval(e, env, depth + 1)
            return result

        if head == "and":
            result: Any = True
            for e in expr[1:]:
                result = self.eval(e, env, depth + 1)
                if not result:
                    return result
            return result

        if head == "or":
            for e in expr[1:]:
                result = self.eval(e, env, depth + 1)
                if result:
                    return result
            return False

        if head == "map":
            fn = self.eval(expr[1], env, depth + 1)
            lst = self.eval(expr[2], env, depth + 1)
            if not isinstance(lst, list):
                return []
            return [self._apply(fn, [x], env, depth + 1) for x in lst]

        if head == "filter":
            fn = self.eval(expr[1], env, depth + 1)
            lst = self.eval(expr[2], env, depth + 1)
            if not isinstance(lst, list):
                return []
            return [x for x in lst if self._apply(fn, [x], env, depth + 1)]

        # Sub-sim
        if head == "sub-sim":
            return self._handle_subsim(expr, env, depth)

        # Function application
        fn = self.eval(head, env, depth + 1)
        args = [self.eval(a, env, depth + 1) for a in expr[1:]]
        return self._apply(fn, args, env, depth + 1)

    def _apply(self, fn: Any, args: list, env: Env, depth: int) -> Any:
        """Apply a function to arguments."""
        if callable(fn):
            try:
                return fn(*args)
            except TypeError as e:
                raise LispyError(f"Function call error: {e}")
        if isinstance(fn, Lambda):
            if len(args) != len(fn.params):
                raise LispyError(
                    f"Arity mismatch: expected {len(fn.params)}, got {len(args)}"
                )
            child_env = Env(
                bindings=dict(zip(fn.params, args)),
                parent=fn.env,
            )
            return self.eval(fn.body, child_env, depth)
        raise LispyError(f"Not callable: {fn}")

    def _handle_subsim(self, expr: list, env: Env, depth: int) -> Any:
        """Handle (sub-sim "label" body) expressions."""
        if len(expr) < 3:
            raise LispyError("sub-sim requires label and body")

        label = self.eval(expr[1], env, depth + 1)
        body = expr[2]

        new_depth = self.sim_depth + 1
        if new_depth > self.max_sim_depth:
            raise LispySandboxError(
                f"Sub-sim depth limit exceeded ({self.max_sim_depth})"
            )
        self._subsim_count += 1
        if self._subsim_count > self.max_subsims_per_frame:
            raise LispySandboxError(
                f"Sub-sim count limit exceeded ({self.max_subsims_per_frame})"
            )

        # If we have a callback, delegate to it
        if self.subsim_callback is not None:
            return self.subsim_callback(label, body, env, new_depth)

        # Default: evaluate in a child evaluator
        child_eval = Evaluator(
            max_steps=self.max_steps // 2,
            max_depth=self.max_depth,
            max_sim_depth=self.max_sim_depth,
            max_subsims_per_frame=max(1, self.max_subsims_per_frame // 2),
            sim_depth=new_depth,
        )
        child_env = Env(parent=env)
        return child_eval.eval(body, child_env, 0)
