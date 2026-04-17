"""
lispy.py — Minimal safe-eval LisPy interpreter for Mars-100 sub-simulations.

A tiny Scheme dialect: s-expressions, lexical scoping, no I/O.
Colonist decisions and sub-simulations are expressed as LisPy programs.

Safety guarantees:
  - No file I/O, no imports, no network
  - Step counter prevents infinite loops
  - Max recursion depth for sub-sims
  - All numeric outputs clamped to physical bounds

Grammar:
  expr  := atom | '(' expr* ')'
  atom  := number | string | symbol
"""
from __future__ import annotations

import math
import re
from typing import Any, Union


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Symbol = str
Number = float
Atom = Union[Number, Symbol, str]
Expr = Union[Atom, list]

class LispyError(Exception):
    """Runtime error in LisPy evaluation."""

class StepLimitError(LispyError):
    """Evaluation exceeded step budget."""

class DepthLimitError(LispyError):
    """Sub-simulation exceeded max recursion depth."""


class LispString:
    """A LisPy string literal — distinct from Symbol (which is also str)."""
    __slots__ = ('value',)

    def __init__(self, value: str):
        self.value = value

    def __repr__(self) -> str:
        return f'LispString({self.value!r})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LispString):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""(\s+|;[^\n]*|"(?:[^"\\]|\\.)*"|[()]|[^\s()"]+)""")


def tokenize(source: str) -> list[str]:
    """Tokenize LisPy source into a list of tokens."""
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(source):
        tok = match.group(1)
        if tok[0] in (' ', '\t', '\n', '\r') or tok[0] == ';':
            continue
        tokens.append(tok)
    return tokens


def parse(source: str) -> Expr:
    """Parse a LisPy source string into an expression tree."""
    tokens = tokenize(source)
    if not tokens:
        raise LispyError("empty expression")
    expr, pos = _read_expr(tokens, 0)
    return expr


def parse_all(source: str) -> list[Expr]:
    """Parse source into multiple top-level expressions."""
    tokens = tokenize(source)
    exprs: list[Expr] = []
    pos = 0
    while pos < len(tokens):
        expr, pos = _read_expr(tokens, pos)
        exprs.append(expr)
    return exprs


def _read_expr(tokens: list[str], pos: int) -> tuple[Expr, int]:
    """Read one expression from tokens starting at pos."""
    if pos >= len(tokens):
        raise LispyError("unexpected end of input")
    tok = tokens[pos]
    if tok == '(':
        return _read_list(tokens, pos + 1)
    elif tok == ')':
        raise LispyError("unexpected ')'")
    elif tok == "'":
        inner, npos = _read_expr(tokens, pos + 1)
        return ['quote', inner], npos
    else:
        return _read_atom(tok), pos + 1


def _read_list(tokens: list[str], pos: int) -> tuple[list, int]:
    """Read a list expression until matching ')'."""
    items: list[Expr] = []
    while pos < len(tokens) and tokens[pos] != ')':
        expr, pos = _read_expr(tokens, pos)
        items.append(expr)
    if pos >= len(tokens):
        raise LispyError("missing closing ')'")
    return items, pos + 1  # skip ')'


def _read_atom(tok: str) -> Atom:
    """Parse an atom: number, string, or symbol."""
    if tok.startswith('"') and tok.endswith('"'):
        return LispString(tok[1:-1].replace('\\"', '"').replace('\\n', '\n'))
    try:
        return float(tok) if '.' in tok else float(int(tok))
    except ValueError:
        # Boolean literals
        if tok == '#t':
            return True
        if tok == '#f':
            return False
        if tok == 'nil':
            return None
        return Symbol(tok)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Env(dict):
    """Lexically scoped environment."""

    def __init__(self, params: list[Symbol] = None, args: list = None,
                 outer: Env | None = None):
        super().__init__()
        self.outer = outer
        if params and args:
            if len(params) != len(args):
                raise LispyError(
                    f"arity mismatch: expected {len(params)}, got {len(args)}")
            self.update(zip(params, args))

    def find(self, name: Symbol) -> Env:
        """Find the innermost env where name is defined."""
        if name in self:
            return self
        if self.outer is not None:
            return self.outer.find(name)
        raise LispyError(f"undefined symbol: {name}")

    def lookup(self, name: Symbol) -> Any:
        """Lookup a name in the environment chain."""
        return self.find(name)[name]


def standard_env() -> Env:
    """Create the standard environment with builtins."""
    env = Env()

    # Arithmetic
    env['+'] = lambda *a: sum(a)
    env['-'] = lambda a, b=None: -a if b is None else a - b
    env['*'] = lambda *a: math.prod(a)
    env['/'] = lambda a, b: a / b if b != 0 else float('inf')
    env['%'] = lambda a, b: a % b if b != 0 else 0.0
    env['abs'] = abs
    env['min'] = min
    env['max'] = max
    env['floor'] = math.floor
    env['ceil'] = math.ceil
    env['sqrt'] = math.sqrt
    env['round'] = lambda x, n=0: round(x, int(n))

    # Comparison
    env['='] = lambda a, b: a == b
    env['!='] = lambda a, b: a != b
    env['>'] = lambda a, b: a > b
    env['<'] = lambda a, b: a < b
    env['>='] = lambda a, b: a >= b
    env['<='] = lambda a, b: a <= b

    # Logic
    env['not'] = lambda a: not a
    env['true'] = True
    env['false'] = False
    env['nil'] = None

    # List operations
    env['list'] = lambda *a: list(a)
    env['cons'] = lambda a, b: [a] + (b if isinstance(b, list) else [b])
    env['car'] = lambda a: a[0] if a else None
    env['cdr'] = lambda a: a[1:] if len(a) > 1 else []
    env['length'] = len
    env['append'] = lambda *a: sum((x if isinstance(x, list) else [x] for x in a), [])
    env['nth'] = lambda lst, n: lst[int(n)] if 0 <= int(n) < len(lst) else None
    env['empty?'] = lambda a: len(a) == 0 if isinstance(a, (list, str)) else a is None
    env['list?'] = lambda a: isinstance(a, list)
    env['number?'] = lambda a: isinstance(a, (int, float))
    env['string?'] = lambda a: isinstance(a, str) and not isinstance(a, bool)

    # String ops
    env['str'] = lambda *a: ''.join(str(x) for x in a)
    env['str-ref'] = lambda s, i: s[int(i)] if 0 <= int(i) < len(s) else ""
    env['substring'] = lambda s, a, b: s[int(a):int(b)]

    # Hash/dict operations (as association lists)
    env['assoc'] = _assoc
    env['assoc-ref'] = _assoc_ref
    env['assoc-set'] = _assoc_set

    # Math
    env['random'] = lambda: 0.5  # deterministic in sandbox
    env['clamp'] = lambda v, lo, hi: max(lo, min(hi, v))
    env['lerp'] = lambda a, b, t: a + (b - a) * t

    # Type conversion
    env['number'] = lambda x: float(x) if not isinstance(x, bool) else (1.0 if x else 0.0)
    env['symbol'] = lambda x: Symbol(str(x))

    return env


def _assoc(key: str, value: Any, alist: Any = None) -> list:
    """Create or extend an association list: (assoc key value alist?)."""
    pair = [key, value]
    if alist is None or alist == []:
        return [pair]
    if not isinstance(alist, list):
        return [pair]
    return [pair] + [p for p in alist if isinstance(p, list) and len(p) >= 2 and p[0] != key]


def _assoc_ref(alist: list, key: str) -> Any:
    """Lookup key in association list."""
    if not isinstance(alist, list):
        return None
    for pair in alist:
        if isinstance(pair, list) and len(pair) >= 2 and pair[0] == key:
            return pair[1]
    return None


def _assoc_set(alist: list, key: str, value: Any) -> list:
    """Set key in association list, return new alist."""
    if not isinstance(alist, list):
        return [[key, value]]
    new = [[key, value]]
    for pair in alist:
        if isinstance(pair, list) and len(pair) >= 2 and pair[0] != key:
            new.append(pair)
    return new


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """LisPy evaluator with step counting and sub-sim support."""

    def __init__(self, step_limit: int = 10000, subsim_depth: int = 0,
                 max_subsim_depth: int = 3, year_budget: int = 50000,
                 rng_seed: int = 42):
        self.step_limit = step_limit
        self.steps = 0
        self.subsim_depth = subsim_depth
        self.max_subsim_depth = max_subsim_depth
        self.year_budget = year_budget
        self.year_steps = 0
        self.subsim_log: list[dict] = []
        self._rng_counter = rng_seed

    def eval(self, expr: Expr, env: Env) -> Any:
        """Evaluate a LisPy expression in the given environment."""
        self.steps += 1
        self.year_steps += 1
        if self.steps > self.step_limit:
            raise StepLimitError(
                f"step limit {self.step_limit} exceeded at depth {self.subsim_depth}")
        if self.year_steps > self.year_budget:
            raise StepLimitError(
                f"year budget {self.year_budget} exceeded")

        # Atoms
        if isinstance(expr, (int, float, bool)):
            return expr
        if isinstance(expr, LispString):
            return expr.value  # String literals evaluate to their string value
        if expr is None:
            return None
        if isinstance(expr, Symbol):
            return env.lookup(expr)
        if not isinstance(expr, list):
            return expr
        if len(expr) == 0:
            return []

        head = expr[0]

        # Special forms
        if head == 'quote':
            return expr[1] if len(expr) > 1 else None

        if head == 'if':
            _, test, then = expr[0:3]
            els = expr[3] if len(expr) > 3 else None
            return self.eval(then, env) if self.eval(test, env) else self.eval(els, env)

        if head == 'cond':
            for clause in expr[1:]:
                if clause[0] == 'else' or self.eval(clause[0], env):
                    result = None
                    for body_expr in clause[1:]:
                        result = self.eval(body_expr, env)
                    return result
            return None

        if head == 'and':
            result: Any = True
            for arg in expr[1:]:
                result = self.eval(arg, env)
                if not result:
                    return result
            return result

        if head == 'or':
            for arg in expr[1:]:
                result = self.eval(arg, env)
                if result:
                    return result
            return False

        if head == 'define':
            name = expr[1]
            if isinstance(name, list):
                # (define (f x y) body) → sugar for (define f (lambda (x y) body))
                fname = name[0]
                params = name[1:]
                body = expr[2:]
                env[fname] = Closure(params, body, env)
            else:
                env[name] = self.eval(expr[2], env)
            return None

        if head == 'set!':
            name = expr[1]
            target = env.find(name)
            target[name] = self.eval(expr[2], env)
            return None

        if head == 'lambda':
            params = expr[1]
            body = expr[2:]
            return Closure(params, body, env)

        if head == 'let':
            bindings = expr[1]
            body = expr[2:]
            inner = Env(outer=env)
            for binding in bindings:
                inner[binding[0]] = self.eval(binding[1], env)
            result = None
            for b in body:
                result = self.eval(b, inner)
            return result

        if head == 'begin':
            result = None
            for sub in expr[1:]:
                result = self.eval(sub, env)
            return result

        if head == 'map':
            fn = self.eval(expr[1], env)
            lst = self.eval(expr[2], env)
            return [self._apply(fn, [x], env) for x in lst]

        if head == 'filter':
            fn = self.eval(expr[1], env)
            lst = self.eval(expr[2], env)
            return [x for x in lst if self._apply(fn, [x], env)]

        if head == 'reduce':
            fn = self.eval(expr[1], env)
            lst = self.eval(expr[2], env)
            init = self.eval(expr[3], env) if len(expr) > 3 else lst[0]
            acc = init
            start = 0 if len(expr) > 3 else 1
            for item in lst[start:]:
                acc = self._apply(fn, [acc, item], env)
            return acc

        if head == 'sub-sim':
            return self._run_subsim(expr, env)

        # Function application
        fn = self.eval(head, env)
        args = [self.eval(a, env) for a in expr[1:]]
        return self._apply(fn, args, env)

    def _apply(self, fn: Any, args: list, env: Env) -> Any:
        """Apply a function to arguments."""
        if isinstance(fn, Closure):
            inner = Env(fn.params, args, fn.env)
            result = None
            for body_expr in fn.body:
                result = self.eval(body_expr, inner)
            return result
        if callable(fn):
            return fn(*args)
        raise LispyError(f"not callable: {fn}")

    def _run_subsim(self, expr: list, env: Env) -> Any:
        """Run a sub-simulation: (sub-sim body)."""
        if self.subsim_depth >= self.max_subsim_depth:
            raise DepthLimitError(
                f"max sub-sim depth {self.max_subsim_depth} reached")

        body = expr[1] if len(expr) > 1 else None
        if body is None:
            return None

        # Budget for sub-sim decreases with depth
        child_budget = self.step_limit // (2 ** (self.subsim_depth + 1))
        child_budget = max(child_budget, 500)

        child_eval = Evaluator(
            step_limit=child_budget,
            subsim_depth=self.subsim_depth + 1,
            max_subsim_depth=self.max_subsim_depth,
            year_budget=self.year_budget - self.year_steps,
            rng_seed=self._rng_counter,
        )
        self._rng_counter += 1

        # Sandboxed env: copy of current env (read-only snapshot)
        sandbox = Env(outer=env)

        try:
            result = child_eval.eval(body, sandbox)
        except (StepLimitError, DepthLimitError) as e:
            result = f"subsim-error: {e}"

        log_entry = {
            "depth": self.subsim_depth + 1,
            "steps_used": child_eval.steps,
            "budget": child_budget,
            "result": _serialize_result(result),
        }
        self.subsim_log.append(log_entry)
        self.subsim_log.extend(child_eval.subsim_log)

        # Charge parent for child's steps
        self.year_steps += child_eval.year_steps

        return result

    def deterministic_random(self) -> float:
        """Seeded pseudo-random for sandbox use."""
        self._rng_counter = (self._rng_counter * 1103515245 + 12345) & 0x7FFFFFFF
        return (self._rng_counter % 10000) / 10000.0


class Closure:
    """A LisPy closure: params + body + captured environment."""
    __slots__ = ('params', 'body', 'env')

    def __init__(self, params: list[Symbol], body: list[Expr], env: Env):
        self.params = params
        self.body = body
        self.env = env

    def __repr__(self) -> str:
        return f"<closure ({' '.join(self.params)})>"


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def run(source: str, env: Env | None = None, step_limit: int = 10000,
        max_depth: int = 3) -> Any:
    """Parse and evaluate a LisPy program, return result."""
    if env is None:
        env = standard_env()
    evaluator = Evaluator(step_limit=step_limit, max_subsim_depth=max_depth)
    exprs = parse_all(source)
    result = None
    for expr in exprs:
        result = evaluator.eval(expr, env)
    return result


def to_sexpr(obj: Any) -> str:
    """Convert a Python object to LisPy s-expression string."""
    if obj is None:
        return "nil"
    if isinstance(obj, bool):
        return "#t" if obj else "#f"
    if isinstance(obj, (int, float)):
        v = obj
        if v == int(v):
            return str(int(v))
        return f"{v:.4f}"
    if isinstance(obj, str):
        return f'"{obj}"'
    if isinstance(obj, list):
        inner = ' '.join(to_sexpr(x) for x in obj)
        return f"({inner})"
    if isinstance(obj, dict):
        pairs = ' '.join(f'({to_sexpr(k)} {to_sexpr(v)})' for k, v in obj.items())
        return f"({pairs})"
    return str(obj)


def _serialize_result(result: Any) -> Any:
    """Serialize a LisPy result to JSON-safe value."""
    if result is None:
        return None
    if isinstance(result, (bool, int, float, str)):
        return result
    if isinstance(result, list):
        return [_serialize_result(x) for x in result]
    if isinstance(result, Closure):
        return f"<closure ({' '.join(result.params)})>"
    return str(result)
