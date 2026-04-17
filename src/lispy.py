"""
lispy.py — Safe, minimal LisPy evaluator for Mars-100 sub-simulations.

A tiny Lisp interpreter: tokenize → parse → evaluate.
No I/O, no imports, no file access, no network. Pure computation.
Colonists express actions as s-expressions; Python validates and applies.

Usage:
    from src.lispy import evaluate, parse, LispError
    result = evaluate('(if (> food 10) "share" "hoard")')
"""
from __future__ import annotations

import math
import operator
import random as _random_mod
from typing import Any


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class LispError(Exception):
    """Base error for LisPy evaluation."""


class LispyError(LispError):
    """Alias for LispError (class-API consumers)."""


class LispyParseError(LispError):
    """Syntax / parse error."""


class LispyRuntimeError(LispError):
    """Runtime evaluation error."""


class LispyBudgetExhausted(LispError):
    """Evaluation step limit exceeded."""


class LispyDepthExceeded(LispError):
    """Sub-simulation depth limit exceeded."""


class LispyMemoryError(LispError):
    """AST depth or collection size limit exceeded."""


# Backward-compat aliases
DepthExceeded = LispyDepthExceeded
StepLimitExceeded = LispyBudgetExhausted


# ---------------------------------------------------------------------------
# Symbol type — distinguishes identifiers from string literals
# ---------------------------------------------------------------------------

class Symbol(str):
    """A Lisp symbol (identifier). Distinguished from string literals."""
    __slots__ = ()

    def __repr__(self) -> str:
        return f"Symbol({super().__repr__()})"


def _sym(name: str) -> Symbol:
    return Symbol(name)


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

MAX_LIST_SIZE = 500
MAX_AST_DEPTH = 30
MAX_SUB_SIM_FRAMES = 10  # Cap per sub-sim level to prevent combinatorial explosion
DEFAULT_BUDGET = 1000


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(source: str) -> list[str]:
    """Split source into tokens: parens, quote, strings, atoms."""
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
                raise LispError("Unclosed string literal")
            tokens.append(source[i : j + 1])
            i = j + 1
        else:
            j = i
            while j < n and source[j] not in ' \t\n\r()";\':':
                j += 1
            tokens.append(source[i:j])
            i = j
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _atom(tok: str) -> Any:
    """Convert a token to a Python value."""
    if tok.startswith('"') and tok.endswith('"'):
        return tok[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if tok in ("#t", "true"):
        return True
    if tok in ("#f", "false"):
        return False
    if tok in ("nil", "null"):
        return None
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    return _sym(tok)  # symbol


def _parse_one(tokens: list[str], pos: int, depth: int = 0) -> tuple[Any, int]:
    """Parse one expression starting at *pos*."""
    if depth > MAX_AST_DEPTH:
        raise LispError(f"AST depth exceeds limit ({MAX_AST_DEPTH})")
    if pos >= len(tokens):
        raise LispError("unexpected end of input")
    tok = tokens[pos]

    if tok == "'":
        inner, nxt = _parse_one(tokens, pos + 1, depth + 1)
        return [_sym("quote"), inner], nxt

    if tok == "(":
        lst: list[Any] = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ")":
            if len(lst) >= MAX_LIST_SIZE:
                raise LispError(f"List too large (>{MAX_LIST_SIZE} elements)")
            elem, pos = _parse_one(tokens, pos, depth + 1)
            lst.append(elem)
        if pos >= len(tokens):
            raise LispError("missing closing )")
        return lst, pos + 1

    if tok == ")":
        raise LispError("unexpected )")

    return _atom(tok), pos + 1


def parse(source: str) -> Any:
    """Parse a single LisPy expression from *source*.

    Returns the AST (atom or list). Raises on empty input.
    """
    tokens = tokenize(source)
    if not tokens:
        raise LispError("empty input")
    expr, pos = _parse_one(tokens, 0)
    return expr


def parse_all(source: str) -> list[Any]:
    """Parse all top-level expressions from *source*."""
    tokens = tokenize(source)
    if not tokens:
        return []
    results: list[Any] = []
    pos = 0
    while pos < len(tokens):
        expr, pos = _parse_one(tokens, pos)
        results.append(expr)
    return results


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Env(dict):
    """Dict-based environment with parent chain."""

    def __init__(self, bindings: dict | None = None, parent: "Env | None" = None):
        super().__init__(bindings or {})
        self.parent = parent

    def lookup(self, name: str) -> Any:
        if name in self:
            return self[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        raise LispError(f"unbound symbol: {name}")

    def set_existing(self, name: str, value: Any) -> None:
        """Set *name* in the nearest scope where it exists."""
        if name in self:
            self[name] = value
            return
        if self.parent is not None:
            self.parent.set_existing(name, value)
            return
        raise LispError(f"cannot set! undefined variable: {name}")


# ---------------------------------------------------------------------------
# Lambda (user-defined function)
# ---------------------------------------------------------------------------

class _Lambda:
    __slots__ = ("params", "body", "closure_env")

    def __init__(self, params: list[str], body: Any, closure_env: Env):
        self.params = params
        self.body = body
        self.closure_env = closure_env


# ---------------------------------------------------------------------------
# Standard environment builder
# ---------------------------------------------------------------------------

def _product(args: tuple) -> int | float:
    result: int | float = 1
    for a in args:
        result *= a
    return result


def _div(a: int | float, b: int | float) -> float:
    if b == 0:
        raise LispError("division by zero")
    return a / b


def standard_env(seed: int | None = None) -> Env:
    """Create a standard environment with safe builtins."""
    rng = _random_mod.Random(seed)
    env = Env()

    # Arithmetic
    env["+"] = lambda *args: sum(args)
    env["-"] = lambda a, b=None: -a if b is None else a - b
    env["*"] = lambda *args: _product(args)
    env["/"] = _div
    env["mod"] = lambda a, b: a % b if b != 0 else 0
    env["abs"] = abs
    env["min"] = lambda *args: min(args)
    env["max"] = lambda *args: max(args)
    env["round"] = round
    env["floor"] = math.floor
    env["ceil"] = math.ceil
    env["sqrt"] = lambda x: math.sqrt(max(0, x))
    env["clamp"] = lambda v, lo, hi: max(lo, min(hi, v))

    # Comparison
    env["<"] = operator.lt
    env[">"] = operator.gt
    env["="] = operator.eq
    env["!="] = lambda a, b: a != b
    env["<="] = operator.le
    env[">="] = operator.ge
    env["not"] = operator.not_

    # List ops
    env["list"] = lambda *args: list(args)
    env["car"] = lambda lst: lst[0] if lst else None
    env["cdr"] = lambda lst: lst[1:] if lst else []
    env["cons"] = lambda a, b: [a] + (b if isinstance(b, list) else [b])
    env["length"] = lambda x: len(x) if x is not None else 0
    env["nth"] = lambda lst, n: lst[int(n)] if lst and 0 <= int(n) < len(lst) else None
    env["append"] = lambda *lsts: sum((l if isinstance(l, list) else [l] for l in lsts), [])
    env["empty?"] = lambda x: (len(x) == 0) if isinstance(x, (list, dict)) else x is None
    env["sort"] = lambda lst: sorted(lst)
    env["reverse"] = lambda lst: list(reversed(lst))
    env["range"] = lambda n: list(range(int(n)))

    # Dict ops
    env["dict"] = lambda *args: _make_dict(args)
    env["get"] = lambda d, k, default=None: d.get(k, default) if isinstance(d, dict) else default
    env["put"] = lambda d, k, v: {**d, k: v} if isinstance(d, dict) else {k: v}
    env["keys"] = lambda d: list(d.keys()) if isinstance(d, dict) else []
    env["values"] = lambda d: list(d.values()) if isinstance(d, dict) else []
    env["has?"] = lambda d, k: k in d if isinstance(d, dict) else False
    env["merge"] = lambda a, b: {**a, **b} if isinstance(a, dict) and isinstance(b, dict) else a

    # Type checks
    env["number?"] = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    env["num?"] = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    env["string?"] = lambda x: isinstance(x, str) and not isinstance(x, Symbol)
    env["str?"] = lambda x: isinstance(x, str) and not isinstance(x, Symbol)
    env["list?"] = lambda x: isinstance(x, list)
    env["dict?"] = lambda x: isinstance(x, dict) and not isinstance(x, Env)
    env["nil?"] = lambda x: x is None
    env["bool?"] = lambda x: isinstance(x, bool)

    # Type conversion
    env["int"] = lambda x: int(x)
    env["float"] = lambda x: float(x)
    env["str"] = lambda *args: "".join(str(a) for a in args)

    # Random (deterministic)
    env["random"] = lambda: rng.random()
    env["random-int"] = lambda a, b: rng.randint(int(a), int(b))
    env["randint"] = lambda a, b: rng.randint(int(a), int(b))
    env["choice"] = lambda lst: rng.choice(lst) if lst else None

    # Constants
    env["true"] = True
    env["false"] = False
    env["nil"] = None
    env["null"] = None
    env["else"] = True

    return env


def _make_dict(args: tuple) -> dict:
    d: dict = {}
    it = iter(args)
    for k in it:
        v = next(it, None)
        d[k] = v
    return d


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def lisp_eval(
    expr: Any,
    env: Env,
    *,
    step_limit: int = 10000,
    _counter: list[int] | None = None,
    _ast_depth: int = 0,
    _sim_depth: int = 0,
    _max_sim_depth: int = 3,
) -> Any:
    """Evaluate a LisPy expression in *env*."""
    if _counter is None:
        _counter = [0]

    _counter[0] += 1
    if _counter[0] > step_limit:
        raise StepLimitExceeded(f"step budget exhausted ({step_limit})")
    if _ast_depth > MAX_AST_DEPTH:
        raise LispError(f"AST depth exceeds limit ({MAX_AST_DEPTH})")

    def _eval(e: Any, ev: Env, depth: int = _ast_depth + 1) -> Any:
        return lisp_eval(
            e, ev, step_limit=step_limit, _counter=_counter,
            _ast_depth=depth, _sim_depth=_sim_depth,
            _max_sim_depth=_max_sim_depth,
        )

    # --- Self-evaluating atoms ---
    if isinstance(expr, bool):
        return expr
    if isinstance(expr, (int, float)):
        return expr
    if expr is None:
        return None

    # Symbol → look up in env. Plain str → string literal (self-evaluating).
    if isinstance(expr, Symbol):
        return env.lookup(expr)
    if isinstance(expr, str):
        return expr  # string literal

    if not isinstance(expr, list):
        return expr

    if len(expr) == 0:
        return []

    head = expr[0]
    head_name = str(head) if isinstance(head, Symbol) else None

    # --- Special forms (dispatch on Symbol name) ---

    if head_name == "quote":
        return expr[1] if len(expr) > 1 else None

    if head_name == "if":
        cond = _eval(expr[1], env)
        if cond and cond is not None and cond != 0:
            return _eval(expr[2], env) if len(expr) > 2 else None
        return _eval(expr[3], env) if len(expr) > 3 else None

    if head_name == "cond":
        for clause in expr[1:]:
            if not isinstance(clause, list) or len(clause) < 2:
                raise LispError("cond clause must be (test body)")
            test = _eval(clause[0], env)
            if test and test is not None and test != 0:
                return _eval(clause[1], env)
        return None

    if head_name == "define":
        name = str(expr[1])
        val = _eval(expr[2], env)
        env[name] = val
        return val

    if head_name == "set!":
        name = str(expr[1])
        val = _eval(expr[2], env)
        env.set_existing(name, val)
        return val

    if head_name == "let":
        bindings = expr[1]
        child = Env(parent=env)
        if isinstance(bindings, list):
            for binding in bindings:
                if isinstance(binding, list) and len(binding) == 2:
                    child[str(binding[0])] = _eval(binding[1], env)
        result = None
        for body in expr[2:]:
            result = _eval(body, child)
        return result

    if head_name == "begin":
        result = None
        for sub in expr[1:]:
            result = _eval(sub, env)
        return result

    if head_name == "lambda":
        params = expr[1]
        body = expr[2] if len(expr) == 3 else [_sym("begin")] + expr[2:]
        return _Lambda([str(p) for p in params], body, env)

    if head_name == "and":
        result: Any = True
        for sub in expr[1:]:
            result = _eval(sub, env)
            if not result:
                return result
        return result

    if head_name == "or":
        for sub in expr[1:]:
            result = _eval(sub, env)
            if result:
                return result
        return False

    if head_name == "sub-sim":
        return _eval_sub_sim(expr, env, step_limit, _counter, _sim_depth, _max_sim_depth)

    # --- map / filter / reduce (special because they take function args) ---
    if head_name == "map":
        fn = _eval(expr[1], env)
        lst = _eval(expr[2], env)
        return [_apply_fn(fn, [x], env, _eval) for x in lst]

    if head_name == "filter":
        fn = _eval(expr[1], env)
        lst = _eval(expr[2], env)
        return [x for x in lst if _apply_fn(fn, [x], env, _eval)]

    if head_name == "reduce":
        fn = _eval(expr[1], env)
        lst = _eval(expr[2], env)
        init = _eval(expr[3], env) if len(expr) > 3 else None
        acc = init
        for x in lst:
            acc = x if acc is None else _apply_fn(fn, [acc, x], env, _eval)
        return acc

    # --- Function application ---
    fn = _eval(head, env)
    args = [_eval(a, env) for a in expr[1:]]
    return _apply_fn(fn, args, env, _eval)


def _apply_fn(fn: Any, args: list, env: Env, _eval: Any) -> Any:
    """Apply a function (builtin or lambda) to args."""
    if isinstance(fn, _Lambda):
        if len(args) != len(fn.params):
            raise LispError(f"Expected {len(fn.params)} args, got {len(args)}")
        call_env = Env(dict(zip(fn.params, args)), parent=fn.closure_env)
        return _eval(fn.body, call_env)

    if callable(fn):
        try:
            return fn(*args)
        except LispError:
            raise
        except TypeError as exc:
            raise LispError(f"call error: {exc}") from exc
        except Exception as exc:
            raise LispError(f"builtin error: {exc}") from exc

    raise LispError(f"not callable: {fn}")


def _eval_sub_sim(
    expr: list,
    env: Env,
    step_limit: int,
    counter: list[int],
    sim_depth: int,
    max_sim_depth: int,
) -> Any:
    """Evaluate (sub-sim frames body).

    Runs body min(frames, MAX_SUB_SIM_FRAMES) times sharing the parent's
    step counter.  Errors are caught and returned as "sub-sim-error: …" strings.
    """
    new_depth = sim_depth + 1
    if new_depth > max_sim_depth:
        return f"sub-sim-error: depth {new_depth} exceeds limit {max_sim_depth}"

    try:
        frames_expr = lisp_eval(
            expr[1], env, step_limit=step_limit, _counter=counter,
            _sim_depth=sim_depth, _max_sim_depth=max_sim_depth,
        )
        frames = min(int(frames_expr), MAX_SUB_SIM_FRAMES)
        body = expr[2]
        prev_result: Any = None

        for frame in range(frames):
            sub_env = Env(parent=env)
            sub_env["sim-frame"] = frame
            sub_env["sim-depth"] = new_depth
            sub_env["prev-result"] = prev_result

            prev_result = lisp_eval(
                body, sub_env, step_limit=step_limit, _counter=counter,
                _sim_depth=new_depth, _max_sim_depth=max_sim_depth,
            )

        return prev_result

    except (StepLimitExceeded, LispyBudgetExhausted):
        return f"sub-sim-error: budget exhausted at depth {new_depth}"
    except (DepthExceeded, LispyDepthExceeded):
        return f"sub-sim-error: depth limit exceeded"
    except LispError as exc:
        return f"sub-sim-error: {exc}"


# ---------------------------------------------------------------------------
# Lispy VM class (for mars100.py class-based API)
# ---------------------------------------------------------------------------

class Lispy:
    """Safe LisPy virtual machine (class-based API)."""

    def __init__(
        self,
        budget: int = DEFAULT_BUDGET,
        max_depth: int = 3,
        seed: int | None = None,
    ) -> None:
        self.budget = budget
        self.max_depth = max_depth
        self.seed = seed
        self.env = standard_env(seed=seed)

    def eval_string(self, source: str, env: Env | None = None) -> Any:
        """Parse and evaluate a LisPy source string."""
        exprs = parse_all(source)
        if not exprs:
            return None
        target_env = env if env is not None else self.env
        counter: list[int] = [0]
        result = None
        for expr in exprs:
            result = lisp_eval(
                expr, target_env, step_limit=self.budget, _counter=counter,
                _sim_depth=0, _max_sim_depth=self.max_depth,
            )
        return result

    def make_env(self, bindings: dict[str, Any] | None = None) -> Env:
        """Create a child environment with additional bindings."""
        return Env(bindings, parent=self.env)


# ---------------------------------------------------------------------------
# Convenience (functional API)
# ---------------------------------------------------------------------------

def evaluate(
    source: str,
    env: Env | None = None,
    *,
    seed: int | None = None,
    step_limit: int = 10000,
) -> Any:
    """Parse and evaluate a LisPy source string."""
    if env is None:
        env = standard_env(seed=seed)
    exprs = parse_all(source)
    if not exprs:
        return None
    counter: list[int] = [0]
    result = None
    for expr in exprs:
        result = lisp_eval(expr, env, step_limit=step_limit, _counter=counter)
    return result


def to_sexpr(obj: Any) -> str:
    """Serialize a Python object to an s-expression string."""
    if obj is None:
        return "nil"
    if isinstance(obj, bool):
        return "#t" if obj else "#f"
    if isinstance(obj, (int, float)):
        return str(obj)
    if isinstance(obj, str):
        return f'"{obj}"'
    if isinstance(obj, list):
        inner = " ".join(to_sexpr(x) for x in obj)
        return f"({inner})"
    if isinstance(obj, dict):
        pairs = " ".join(f'"{k}" {to_sexpr(v)}' for k, v in obj.items())
        return f"(dict {pairs})"
    return str(obj)
