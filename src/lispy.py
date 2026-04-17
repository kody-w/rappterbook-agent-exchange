"""lispy.py -- Minimal safe-eval LisPy interpreter for Mars-100 sub-simulations.

Homoiconic: data IS code. Colonists are s-expressions that evaluate themselves.
No I/O, no imports, no system access. Pure computation with budget limits.

Usage:
    from src.lispy import parse, evaluate, default_env, LispyError, Budget

    env = default_env()
    budget = Budget(max_steps=10000, max_depth=3)
    ast = parse("(+ 1 2)")
    result = evaluate(ast, env, budget)  # -> 3
"""
from __future__ import annotations

import copy
import math
import operator
from dataclasses import dataclass, field


class LispyError(Exception):
    """Any error during LisPy evaluation."""


class BudgetExhausted(LispyError):
    """Step or depth budget exceeded."""


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(source: str) -> list[str]:
    """Split source into tokens: parens, strings, numbers, symbols."""
    tokens: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in ' \t\n\r':
            i += 1
        elif ch == ';':
            while i < n and source[i] != '\n':
                i += 1
        elif ch in '()':
            tokens.append(ch)
            i += 1
        elif ch == "'":
            tokens.append("'")
            i += 1
        elif ch == '"':
            j = i + 1
            while j < n and source[j] != '"':
                if source[j] == '\\':
                    j += 1
                j += 1
            tokens.append(source[i:j + 1])
            i = j + 1
        else:
            j = i
            while j < n and source[j] not in ' \t\n\r();"':
                j += 1
            tokens.append(source[i:j])
            i = j
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class Symbol(str):
    """A named symbol (distinct from a string)."""
    __slots__ = ()

    def __repr__(self) -> str:
        return f"Symbol({super().__repr__()})"


def _atom(token: str):
    """Convert a token to an atom: number, string, or symbol."""
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1].replace('\\"', '"').replace('\\n', '\n')
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    if token == '#t':
        return True
    if token == '#f':
        return False
    return Symbol(token)


def parse(source: str):
    """Parse a LisPy source string into an AST (nested lists + atoms)."""
    tokens = tokenize(source)
    if not tokens:
        raise LispyError("empty expression")
    ast, pos = _read(tokens, 0)
    return ast


def parse_all(source: str) -> list:
    """Parse multiple top-level expressions from source."""
    tokens = tokenize(source)
    results = []
    pos = 0
    while pos < len(tokens):
        ast, pos = _read(tokens, pos)
        results.append(ast)
    return results


def _read(tokens: list[str], pos: int):
    """Recursive descent reader."""
    if pos >= len(tokens):
        raise LispyError("unexpected EOF")
    token = tokens[pos]
    if token == "'":
        inner, pos = _read(tokens, pos + 1)
        return [Symbol('quote'), inner], pos
    if token == '(':
        lst = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ')':
            item, pos = _read(tokens, pos)
            lst.append(item)
        if pos >= len(tokens):
            raise LispyError("missing closing paren")
        return lst, pos + 1
    if token == ')':
        raise LispyError("unexpected )")
    return _atom(token), pos + 1


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Env(dict):
    """Environment with parent chain for lexical scoping."""

    def __init__(self, bindings: dict | None = None, parent: Env | None = None):
        super().__init__(bindings or {})
        self.parent = parent

    def lookup(self, name: str):
        if name in self:
            return self[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        raise LispyError(f"unbound symbol: {name}")

    def set_existing(self, name: str, value) -> None:
        """Set a variable in the nearest scope that defines it."""
        if name in self:
            self[name] = value
            return
        if self.parent is not None:
            self.parent.set_existing(name, value)
            return
        raise LispyError(f"cannot set! unbound symbol: {name}")


# ---------------------------------------------------------------------------
# Budget (execution limits)
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    """Shared execution budget across all nested evaluations."""
    max_steps: int = 10_000
    max_depth: int = 3
    steps_used: int = 0
    current_depth: int = 0
    sub_sims_spawned: int = 0
    max_sub_sims: int = 50
    _recursion_depth: int = field(default=0, repr=False)
    max_recursion: int = 200

    def tick(self) -> None:
        self.steps_used += 1
        if self.steps_used > self.max_steps:
            raise BudgetExhausted(
                f"step budget exhausted ({self.max_steps} steps)"
            )

    def enter_recursion(self) -> None:
        self._recursion_depth += 1
        if self._recursion_depth > self.max_recursion:
            raise BudgetExhausted(
                f"recursion limit ({self.max_recursion})"
            )

    def exit_recursion(self) -> None:
        self._recursion_depth -= 1

    def enter_sub_sim(self) -> None:
        self.sub_sims_spawned += 1
        if self.sub_sims_spawned > self.max_sub_sims:
            raise BudgetExhausted(
                f"sub-sim limit ({self.max_sub_sims})"
            )
        self.current_depth += 1
        if self.current_depth > self.max_depth:
            raise BudgetExhausted(
                f"sub-sim depth limit ({self.max_depth})"
            )

    def exit_sub_sim(self) -> None:
        self.current_depth -= 1


# ---------------------------------------------------------------------------
# Lambda / Procedure
# ---------------------------------------------------------------------------

class Procedure:
    """A user-defined lambda."""
    __slots__ = ('params', 'body', 'closure_env')

    def __init__(self, params: list[str], body, closure_env: Env):
        self.params = params
        self.body = body
        self.closure_env = closure_env

    def __repr__(self) -> str:
        return f"<lambda ({' '.join(self.params)})>"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate(expr, env: Env, budget: Budget):
    """Evaluate a LisPy expression in the given environment."""
    budget.tick()
    budget.enter_recursion()
    try:
        return _eval_inner(expr, env, budget)
    finally:
        budget.exit_recursion()


def _eval_inner(expr, env: Env, budget: Budget):
    # Atom
    if isinstance(expr, Symbol):
        return env.lookup(expr)
    if not isinstance(expr, list):
        return expr  # number, string, bool

    if len(expr) == 0:
        return []

    head = expr[0]

    # Special forms
    if isinstance(head, Symbol):
        if head == 'quote':
            if len(expr) != 2:
                raise LispyError("quote takes exactly 1 argument")
            return expr[1]

        if head == 'if':
            if len(expr) < 3:
                raise LispyError("if needs at least (if test then)")
            test = evaluate(expr[1], env, budget)
            if test and test is not False and test != 0:
                return evaluate(expr[2], env, budget)
            elif len(expr) > 3:
                return evaluate(expr[3], env, budget)
            return None

        if head == 'define':
            if len(expr) != 3:
                raise LispyError("define takes (define name value)")
            name = expr[1]
            if not isinstance(name, Symbol):
                raise LispyError(f"define name must be a symbol, got {type(name)}")
            val = evaluate(expr[2], env, budget)
            env[name] = val
            return val

        if head == 'set!':
            if len(expr) != 3:
                raise LispyError("set! takes (set! name value)")
            name = expr[1]
            if not isinstance(name, Symbol):
                raise LispyError("set! name must be a symbol")
            val = evaluate(expr[2], env, budget)
            env.set_existing(name, val)
            return val

        if head == 'lambda':
            if len(expr) < 3:
                raise LispyError("lambda needs (lambda (params) body)")
            params = expr[1]
            if not isinstance(params, list):
                raise LispyError("lambda params must be a list")
            param_names = []
            for p in params:
                if not isinstance(p, Symbol):
                    raise LispyError(f"lambda param must be symbol, got {p}")
                param_names.append(str(p))
            body = expr[2] if len(expr) == 3 else [Symbol('begin')] + expr[2:]
            return Procedure(param_names, body, env)

        if head == 'let':
            if len(expr) < 3:
                raise LispyError("let needs (let ((name val) ...) body)")
            bindings = expr[1]
            local = Env(parent=env)
            for b in bindings:
                if not isinstance(b, list) or len(b) != 2:
                    raise LispyError(f"bad let binding: {b}")
                local[str(b[0])] = evaluate(b[1], env, budget)
            result = None
            for body_expr in expr[2:]:
                result = evaluate(body_expr, local, budget)
            return result

        if head == 'begin':
            result = None
            for sub in expr[1:]:
                result = evaluate(sub, env, budget)
            return result

        if head == 'and':
            result = True
            for sub in expr[1:]:
                result = evaluate(sub, env, budget)
                if not result:
                    return result
            return result

        if head == 'or':
            result = False
            for sub in expr[1:]:
                result = evaluate(sub, env, budget)
                if result:
                    return result
            return result

        if head == 'cond':
            for clause in expr[1:]:
                if not isinstance(clause, list) or len(clause) < 2:
                    raise LispyError("bad cond clause")
                test_expr = clause[0]
                if isinstance(test_expr, Symbol) and test_expr == 'else':
                    return evaluate(clause[1], env, budget)
                if evaluate(test_expr, env, budget):
                    return evaluate(clause[1], env, budget)
            return None

        if head == 'sub-sim':
            # (sub-sim body) — run body in a sandboxed sub-environment
            if len(expr) != 2:
                raise LispyError("sub-sim takes (sub-sim body)")
            budget.enter_sub_sim()
            try:
                sandbox = Env(parent=env)
                return evaluate(expr[1], sandbox, budget)
            finally:
                budget.exit_sub_sim()

    # Function application
    func = evaluate(head, env, budget)
    args = [evaluate(a, env, budget) for a in expr[1:]]

    if isinstance(func, Procedure):
        if len(args) != len(func.params):
            raise LispyError(
                f"arity mismatch: {func} expects {len(func.params)} args, got {len(args)}"
            )
        local = Env(dict(zip(func.params, args)), parent=func.closure_env)
        return evaluate(func.body, local, budget)

    if callable(func):
        try:
            return func(*args)
        except LispyError:
            raise
        except Exception as exc:
            raise LispyError(f"builtin error: {exc}") from exc

    raise LispyError(f"not callable: {func}")


# ---------------------------------------------------------------------------
# Built-in functions
# ---------------------------------------------------------------------------

def _make_builtins() -> dict:
    """Create the default built-in function bindings."""
    def _list(*args):
        return list(args)

    def _car(lst):
        if not isinstance(lst, list) or len(lst) == 0:
            raise LispyError("car of empty/non-list")
        return lst[0]

    def _cdr(lst):
        if not isinstance(lst, list):
            raise LispyError("cdr of non-list")
        return lst[1:]

    def _cons(a, b):
        if not isinstance(b, list):
            return [a, b]
        return [a] + b

    def _length(lst):
        if isinstance(lst, (list, str)):
            return len(lst)
        raise LispyError(f"length of non-sequence: {type(lst)}")

    def _append(*lists):
        result = []
        for lst in lists:
            if isinstance(lst, list):
                result.extend(lst)
            else:
                result.append(lst)
        return result

    def _nth(lst, n):
        if not isinstance(lst, list):
            raise LispyError("nth of non-list")
        if not isinstance(n, int) or n < 0 or n >= len(lst):
            raise LispyError(f"nth index {n} out of range (len={len(lst)})")
        return lst[n]

    def _get(obj, key):
        """Get a value from a dict-like assoc list or actual dict."""
        if isinstance(obj, dict):
            return obj.get(key)
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, list) and len(item) >= 2 and item[0] == key:
                    return item[1]
            return None
        raise LispyError(f"get: not a dict or assoc list: {type(obj)}")

    def _put(obj, key, val):
        """Return new assoc list with key set to val."""
        if isinstance(obj, dict):
            result = dict(obj)
            result[key] = val
            return result
        if isinstance(obj, list):
            result = [item for item in obj
                      if not (isinstance(item, list) and len(item) >= 2 and item[0] == key)]
            result.append([key, val])
            return result
        raise LispyError("put: not a dict or assoc list")

    def _dict_fn(*pairs):
        """Create a dict from key-value pairs: (dict 'a 1 'b 2)."""
        if len(pairs) % 2 != 0:
            raise LispyError("dict needs even number of args")
        return {pairs[i]: pairs[i + 1] for i in range(0, len(pairs), 2)}

    def _keys(obj):
        if isinstance(obj, dict):
            return list(obj.keys())
        if isinstance(obj, list):
            return [item[0] for item in obj if isinstance(item, list) and len(item) >= 2]
        raise LispyError("keys: not a dict or assoc list")

    def _equal(a, b):
        return a == b

    def _not_equal(a, b):
        return a != b

    def _display(*args):
        """Display is a no-op in safe eval — returns the concatenated repr."""
        return ' '.join(str(a) for a in args)

    def _number_p(x):
        return isinstance(x, (int, float))

    def _list_p(x):
        return isinstance(x, list)

    def _string_p(x):
        return isinstance(x, str) and not isinstance(x, Symbol)

    def _symbol_p(x):
        return isinstance(x, Symbol)

    def _null_p(x):
        return x is None or (isinstance(x, list) and len(x) == 0)

    def _apply(func, args):
        if isinstance(func, Procedure):
            if len(args) != len(func.params):
                raise LispyError(f"apply arity mismatch")
            local = Env(dict(zip(func.params, args)), parent=func.closure_env)
            # Note: cannot call evaluate without budget here, so apply is limited
            raise LispyError("apply not supported for lambdas in safe mode")
        if callable(func):
            return func(*args)
        raise LispyError(f"apply: not callable: {func}")

    builtins = {
        # Arithmetic
        '+': lambda *args: sum(args),
        '-': lambda a, b=None: -a if b is None else a - b,
        '*': lambda *args: math.prod(args) if args else 1,
        '/': lambda a, b: a / b if b != 0 else float('inf'),
        '%': lambda a, b: a % b,
        'abs': abs,
        'min': min,
        'max': max,
        'round': round,
        'floor': math.floor,
        'ceil': math.ceil,
        'sqrt': math.sqrt,
        'expt': lambda a, b: a ** b,

        # Comparison
        '<': operator.lt,
        '>': operator.gt,
        '<=': operator.le,
        '>=': operator.ge,
        '=': _equal,
        '!=': _not_equal,

        # Logic
        'not': lambda x: not x,

        # List ops
        'list': _list,
        'car': _car,
        'cdr': _cdr,
        'cons': _cons,
        'length': _length,
        'append': _append,
        'nth': _nth,
        'reverse': lambda lst: list(reversed(lst)) if isinstance(lst, list) else lst,
        'sort': lambda lst: sorted(lst) if isinstance(lst, list) else lst,

        # Dict / assoc ops
        'get': _get,
        'put': _put,
        'dict': _dict_fn,
        'keys': _keys,

        # Type checks
        'number?': _number_p,
        'list?': _list_p,
        'string?': _string_p,
        'symbol?': _symbol_p,
        'null?': _null_p,

        # I/O (safe no-ops)
        'display': _display,

        # Constants
        'true': True,
        'false': False,
        'nil': None,
        'pi': math.pi,
    }
    return builtins


def default_env() -> Env:
    """Create a default environment with all built-in bindings."""
    return Env(_make_builtins())


# ---------------------------------------------------------------------------
# Convenience: eval a string
# ---------------------------------------------------------------------------

def run(source: str, env: Env | None = None, budget: Budget | None = None):
    """Parse and evaluate a LisPy source string. Returns final result."""
    if env is None:
        env = default_env()
    if budget is None:
        budget = Budget()
    exprs = parse_all(source)
    result = None
    for expr in exprs:
        result = evaluate(expr, env, budget)
    return result


def to_sexp(obj) -> str:
    """Convert a Python object back to an s-expression string."""
    if obj is None:
        return "nil"
    if obj is True:
        return "#t"
    if obj is False:
        return "#f"
    if isinstance(obj, Symbol):
        return str(obj)
    if isinstance(obj, str):
        return f'"{obj}"'
    if isinstance(obj, (int, float)):
        return str(obj)
    if isinstance(obj, dict):
        pairs = ' '.join(f'{to_sexp(k)} {to_sexp(v)}' for k, v in obj.items())
        return f"(dict {pairs})"
    if isinstance(obj, list):
        return '(' + ' '.join(to_sexp(x) for x in obj) + ')'
    return str(obj)
