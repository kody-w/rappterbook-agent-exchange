"""
lispy.py — Pure-computation LisPy interpreter for sandboxed simulations.

A minimal Lisp interpreter with NO I/O, NO imports, NO file access.
Designed for safe-eval sub-simulations (Amendment XIII: Turtles All the Way Down).

Features:
  - S-expression parsing (tokenize → read → eval)
  - Lexical scoping with closures
  - First-class functions (lambda)
  - List operations, dict operations
  - Sub-simulation spawning with depth limits
  - Step-counting execution limits
  - Tagged error results (never throws into host)

Usage:
    from src.lispy import Lispy
    vm = Lispy(max_steps=10000, max_depth=3)
    result = vm.run('(+ 1 2)')
    assert result == 3
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Symbol(str):
    """A Lisp symbol — a name that resolves in an environment."""
    __slots__ = ()
    def __repr__(self) -> str:
        return f"'{self}'"


class LispError:
    """Tagged error value — errors are data, not exceptions."""
    __slots__ = ("message",)
    def __init__(self, message: str) -> None:
        self.message = message
    def __repr__(self) -> str:
        return f"(error \"{self.message}\")"
    def __eq__(self, other: object) -> bool:
        return isinstance(other, LispError) and self.message == other.message


@dataclass
class Closure:
    """A lambda with captured environment."""
    params: list[str]
    body: object
    env: "Env"
    name: str = ""
    def __repr__(self) -> str:
        name = self.name or "lambda"
        return f"<{name}({', '.join(self.params)})>"


NIL: list = []


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Env:
    """Lexically-scoped environment chain."""
    __slots__ = ("data", "parent")

    def __init__(self, data: dict | None = None, parent: "Env | None" = None) -> None:
        self.data: dict = data or {}
        self.parent: Env | None = parent

    def lookup(self, name: str) -> object:
        if name in self.data:
            return self.data[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return LispError(f"unbound symbol: {name}")

    def set(self, name: str, value: object) -> None:
        self.data[name] = value

    def find_frame(self, name: str) -> "Env | None":
        """Find the frame that owns this binding (for set!)."""
        if name in self.data:
            return self
        if self.parent is not None:
            return self.parent.find_frame(name)
        return None


# ---------------------------------------------------------------------------
# Tokenizer + Parser
# ---------------------------------------------------------------------------

def tokenize(source: str) -> list[str]:
    """Split source into tokens: parens, quotes, strings, atoms."""
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
            tokens.append(source[i : j + 1])
            i = j + 1
        else:
            j = i
            while j < n and source[j] not in " \t\n\r();\"'":
                j += 1
            tokens.append(source[i:j])
            i = j
    return tokens


def parse(source: str) -> list:
    """Parse source string into a list of AST forms."""
    tokens = tokenize(source)
    forms: list = []
    pos = [0]

    def read_form() -> object:
        if pos[0] >= len(tokens):
            return LispError("unexpected end of input")
        tok = tokens[pos[0]]
        if tok == "(":
            return read_list()
        elif tok == "'":
            pos[0] += 1
            inner = read_form()
            return [Symbol("quote"), inner]
        elif tok == ")":
            return LispError("unexpected ')'")
        else:
            pos[0] += 1
            return read_atom(tok)

    def read_list() -> object:
        pos[0] += 1  # skip '('
        items: list = []
        while pos[0] < len(tokens) and tokens[pos[0]] != ")":
            form = read_form()
            if isinstance(form, LispError):
                return form
            items.append(form)
        if pos[0] >= len(tokens):
            return LispError("unclosed '('")
        pos[0] += 1  # skip ')'
        return items

    def read_atom(tok: str) -> object:
        if tok.startswith('"') and tok.endswith('"'):
            return tok[1:-1].replace('\\"', '"').replace("\\n", "\n")
        if tok == "true":
            return True
        if tok == "false":
            return False
        if tok == "nil":
            return NIL
        try:
            return int(tok)
        except ValueError:
            pass
        try:
            return float(tok)
        except ValueError:
            pass
        return Symbol(tok)

    while pos[0] < len(tokens):
        form = read_form()
        if isinstance(form, LispError):
            return [form]
        forms.append(form)
    return forms


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------

class Lispy:
    """Pure-computation LisPy interpreter with execution limits.

    Args:
        max_steps: maximum evaluation steps before halting.
        max_depth: maximum sub-simulation nesting depth.
        current_depth: current nesting level (0 = top).
        step_budget: shared mutable list [remaining_steps] across sub-sims.
    """

    def __init__(
        self,
        max_steps: int = 50000,
        max_depth: int = 3,
        current_depth: int = 0,
        step_budget: list[int] | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.current_depth = current_depth
        self.step_budget: list[int] = step_budget or [max_steps]
        self.global_env = self._build_global_env()

    def _build_global_env(self) -> Env:
        """Construct the global environment with builtins."""
        env = Env()

        # Arithmetic
        env.set("+", lambda *args: sum(args))
        env.set("-", lambda a, b=None: -a if b is None else a - b)
        env.set("*", lambda *args: math.prod(args))
        env.set("/", lambda a, b: a / b if b != 0 else LispError("division by zero"))
        env.set("mod", lambda a, b: a % b if b != 0 else LispError("modulo by zero"))
        env.set("abs", abs)
        env.set("min", min)
        env.set("max", max)
        env.set("floor", lambda x: int(math.floor(x)))
        env.set("ceil", lambda x: int(math.ceil(x)))
        env.set("round", lambda x, n=0: round(x, n))
        env.set("sqrt", lambda x: math.sqrt(x) if x >= 0 else LispError("sqrt of negative"))
        env.set("pow", lambda a, b: a ** b)

        # Comparison
        env.set("=", lambda a, b: a == b)
        env.set("!=", lambda a, b: a != b)
        env.set("<", lambda a, b: a < b)
        env.set(">", lambda a, b: a > b)
        env.set("<=", lambda a, b: a <= b)
        env.set(">=", lambda a, b: a >= b)

        # Logic
        env.set("not", lambda x: not _truthy(x))

        # Type checks
        def _error_check(x):
            return isinstance(x, LispError)
        _error_check._inspects_errors = True
        env.set("error?", _error_check)

        def _type_check(x):
            return _type_name(x)
        _type_check._inspects_errors = True
        env.set("type", _type_check)

        def _identity_fn(x):
            return x
        _identity_fn._inspects_errors = True
        env.set("identity", _identity_fn)

        env.set("number?", lambda x: isinstance(x, (int, float)))
        env.set("string?", lambda x: isinstance(x, str) and not isinstance(x, Symbol))
        env.set("list?", lambda x: isinstance(x, list))
        env.set("dict?", lambda x: isinstance(x, dict))
        env.set("nil?", lambda x: x == NIL or x is None)
        env.set("symbol?", lambda x: isinstance(x, Symbol))

        # String ops
        env.set("str", lambda *args: "".join(str(a) for a in args))
        env.set("str-len", lambda s: len(s) if isinstance(s, str) else LispError("not a string"))
        env.set("str-upper", lambda s: s.upper() if isinstance(s, str) else LispError("not a string"))
        env.set("str-lower", lambda s: s.lower() if isinstance(s, str) else LispError("not a string"))
        env.set("str-contains", lambda s, sub: sub in s if isinstance(s, str) else False)

        # List ops
        env.set("list", lambda *args: list(args))
        env.set("car", lambda xs: xs[0] if xs else LispError("car of empty list"))
        env.set("cdr", lambda xs: xs[1:] if xs else NIL)
        env.set("cons", lambda x, xs: [x] + (xs if isinstance(xs, list) else [xs]))
        env.set("length", lambda xs: len(xs) if isinstance(xs, (list, str, dict)) else 0)
        env.set("nth", lambda xs, n: xs[n] if 0 <= n < len(xs) else LispError(f"index {n} out of bounds"))
        env.set("append", lambda *lists: sum((l if isinstance(l, list) else [l] for l in lists), []))
        env.set("reverse", lambda xs: list(reversed(xs)) if isinstance(xs, list) else LispError("not a list"))
        env.set("range", lambda a, b=None: list(range(a)) if b is None else list(range(a, b)))
        env.set("sort", lambda xs: sorted(xs) if isinstance(xs, list) else LispError("not a list"))
        env.set("contains", lambda xs, x: x in xs if isinstance(xs, (list, dict)) else False)
        env.set("slice", lambda xs, a, b=None: xs[a:b] if isinstance(xs, list) else LispError("not a list"))

        # Dict ops
        env.set("dict", lambda *pairs: dict(zip(pairs[::2], pairs[1::2])))
        env.set("get", lambda d, k, default=None: d.get(k, default) if isinstance(d, dict) else LispError("not a dict"))
        env.set("assoc", lambda d, k, v: {**d, k: v} if isinstance(d, dict) else LispError("not a dict"))
        env.set("dissoc", lambda d, k: {kk: vv for kk, vv in d.items() if kk != k} if isinstance(d, dict) else LispError("not a dict"))
        env.set("keys", lambda d: list(d.keys()) if isinstance(d, dict) else LispError("not a dict"))
        env.set("values", lambda d: list(d.values()) if isinstance(d, dict) else LispError("not a dict"))
        env.set("merge", lambda *dicts: {k: v for d in dicts if isinstance(d, dict) for k, v in d.items()})

        # Utility
        env.set("print", lambda *args: " ".join(str(a) for a in args))
        env.set("error", lambda msg="error": LispError(str(msg)))

        # Constants
        env.set("pi", math.pi)
        env.set("e", math.e)
        env.set("true", True)
        env.set("false", False)
        env.set("nil", NIL)

        return env

    def run(self, source: str, env: Env | None = None) -> object:
        """Parse and evaluate source code. Returns final expression value."""
        forms = parse(source)
        if not forms:
            return NIL
        if len(forms) == 1 and isinstance(forms[0], LispError):
            return forms[0]
        result: object = NIL
        target_env = env or self.global_env
        for form in forms:
            result = self.eval_expr(form, target_env)
            if isinstance(result, LispError):
                return result
        return result

    def eval_expr(self, expr: object, env: Env) -> object:
        """Evaluate a single expression in the given environment."""
        # Step budget check
        if self.step_budget[0] <= 0:
            return LispError("step limit exceeded")
        self.step_budget[0] -= 1

        # Atoms
        if isinstance(expr, (int, float, bool)):
            return expr
        if isinstance(expr, str) and not isinstance(expr, Symbol):
            return expr
        if isinstance(expr, LispError):
            return expr
        if isinstance(expr, Symbol):
            return env.lookup(expr)

        # Lists (function calls and special forms)
        if not isinstance(expr, list):
            return LispError(f"cannot eval: {type(expr).__name__}")
        if len(expr) == 0:
            return NIL

        head = expr[0]

        # Special forms
        if isinstance(head, Symbol):
            if head == "quote":
                return expr[1] if len(expr) > 1 else NIL

            if head == "if":
                if len(expr) < 3:
                    return LispError("if needs at least 2 args")
                cond = self.eval_expr(expr[1], env)
                if isinstance(cond, LispError):
                    return cond
                if _truthy(cond):
                    return self.eval_expr(expr[2], env)
                elif len(expr) > 3:
                    return self.eval_expr(expr[3], env)
                return NIL

            if head == "cond":
                for clause in expr[1:]:
                    if not isinstance(clause, list) or len(clause) < 2:
                        return LispError("cond clause must be (test body)")
                    test = self.eval_expr(clause[0], env)
                    if isinstance(test, LispError):
                        return test
                    if _truthy(test):
                        return self.eval_expr(clause[1], env)
                return NIL

            if head == "define":
                if len(expr) != 3:
                    return LispError("define needs name and value")
                name = expr[1]
                if not isinstance(name, Symbol):
                    return LispError("define name must be a symbol")
                val = self.eval_expr(expr[2], env)
                if isinstance(val, LispError):
                    return val
                env.set(str(name), val)
                return val

            if head == "set!":
                if len(expr) != 3:
                    return LispError("set! needs name and value")
                name = expr[1]
                if not isinstance(name, Symbol):
                    return LispError("set! name must be a symbol")
                frame = env.find_frame(str(name))
                if frame is None:
                    return LispError(f"set!: unbound symbol: {name}")
                val = self.eval_expr(expr[2], env)
                if isinstance(val, LispError):
                    return val
                frame.data[str(name)] = val
                return val

            if head == "lambda":
                if len(expr) < 3:
                    return LispError("lambda needs params and body")
                params = expr[1]
                if not isinstance(params, list):
                    return LispError("lambda params must be a list")
                param_names = []
                for p in params:
                    if not isinstance(p, Symbol):
                        return LispError(f"lambda param must be a symbol, got {p}")
                    param_names.append(str(p))
                body = expr[2] if len(expr) == 3 else [Symbol("begin")] + expr[2:]
                return Closure(param_names, body, env)

            if head == "let":
                if len(expr) < 3:
                    return LispError("let needs bindings and body")
                bindings = expr[1]
                if not isinstance(bindings, list):
                    return LispError("let bindings must be a list")
                let_env = Env(parent=env)
                for binding in bindings:
                    if not isinstance(binding, list) or len(binding) != 2:
                        return LispError("let binding must be (name value)")
                    bname, bexpr = binding
                    if not isinstance(bname, Symbol):
                        return LispError("let binding name must be a symbol")
                    bval = self.eval_expr(bexpr, let_env)
                    if isinstance(bval, LispError):
                        return bval
                    let_env.set(str(bname), bval)
                result: object = NIL
                for body_expr in expr[2:]:
                    result = self.eval_expr(body_expr, let_env)
                    if isinstance(result, LispError):
                        return result
                return result

            if head == "begin":
                result: object = NIL
                for sub in expr[1:]:
                    result = self.eval_expr(sub, env)
                    if isinstance(result, LispError):
                        return result
                return result

            if head == "and":
                result: object = True
                for sub in expr[1:]:
                    result = self.eval_expr(sub, env)
                    if isinstance(result, LispError):
                        return result
                    if not _truthy(result):
                        return result
                return result

            if head == "or":
                result: object = False
                for sub in expr[1:]:
                    result = self.eval_expr(sub, env)
                    if isinstance(result, LispError):
                        return result
                    if _truthy(result):
                        return result
                return result

            if head == "do":
                # (do var-name start end body) — simple loop
                if len(expr) != 5:
                    return LispError("do needs (do var start end body)")
                var = expr[1]
                if not isinstance(var, Symbol):
                    return LispError("do variable must be a symbol")
                start = self.eval_expr(expr[2], env)
                end = self.eval_expr(expr[3], env)
                if isinstance(start, LispError):
                    return start
                if isinstance(end, LispError):
                    return end
                if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                    return LispError("do bounds must be numbers")
                result: object = NIL
                loop_env = Env(parent=env)
                for i in range(int(start), int(end)):
                    loop_env.set(str(var), i)
                    result = self.eval_expr(expr[4], loop_env)
                    if isinstance(result, LispError):
                        return result
                return result

            if head == "map":
                if len(expr) != 3:
                    return LispError("map needs function and list")
                fn = self.eval_expr(expr[1], env)
                lst = self.eval_expr(expr[2], env)
                if isinstance(fn, LispError):
                    return fn
                if isinstance(lst, LispError):
                    return lst
                if not isinstance(lst, list):
                    return LispError("map: second arg must be a list")
                results = []
                for item in lst:
                    val = self._apply(fn, [item], env)
                    if isinstance(val, LispError):
                        return val
                    results.append(val)
                return results

            if head == "filter":
                if len(expr) != 3:
                    return LispError("filter needs function and list")
                fn = self.eval_expr(expr[1], env)
                lst = self.eval_expr(expr[2], env)
                if isinstance(fn, LispError):
                    return fn
                if isinstance(lst, LispError):
                    return lst
                if not isinstance(lst, list):
                    return LispError("filter: second arg must be a list")
                results = []
                for item in lst:
                    val = self._apply(fn, [item], env)
                    if isinstance(val, LispError):
                        return val
                    if _truthy(val):
                        results.append(item)
                return results

            if head == "reduce":
                if len(expr) != 4:
                    return LispError("reduce needs function, initial, and list")
                fn = self.eval_expr(expr[1], env)
                acc = self.eval_expr(expr[2], env)
                lst = self.eval_expr(expr[3], env)
                if isinstance(fn, LispError):
                    return fn
                if isinstance(acc, LispError):
                    return acc
                if isinstance(lst, LispError):
                    return lst
                if not isinstance(lst, list):
                    return LispError("reduce: third arg must be a list")
                for item in lst:
                    acc = self._apply(fn, [acc, item], env)
                    if isinstance(acc, LispError):
                        return acc
                return acc

            if head == "sub-sim":
                return self._eval_sub_sim(expr, env)

        # Function application
        fn = self.eval_expr(head, env)
        if isinstance(fn, LispError):
            return fn

        args: list = []
        for arg_expr in expr[1:]:
            val = self.eval_expr(arg_expr, env)
            # Allow LispError values to pass through to functions that
            # explicitly check them (error?, type, etc.)
            if isinstance(val, LispError) and not _is_error_inspector(fn):
                return val
            args.append(val)

        return self._apply(fn, args, env)

    def _apply(self, fn: object, args: list, env: Env) -> object:
        """Apply a function to arguments."""
        if isinstance(fn, Closure):
            if len(args) != len(fn.params):
                return LispError(
                    f"arity mismatch: {fn} expects {len(fn.params)} args, got {len(args)}"
                )
            call_env = Env(
                data=dict(zip(fn.params, args)),
                parent=fn.env,
            )
            return self.eval_expr(fn.body, call_env)
        if callable(fn):
            try:
                return fn(*args)
            except TypeError as exc:
                return LispError(f"builtin error: {exc}")
            except Exception as exc:
                return LispError(f"runtime error: {exc}")
        return LispError(f"not callable: {fn}")

    def _eval_sub_sim(self, expr: list, env: Env) -> object:
        """(sub-sim body) — spawn a nested LisPy evaluation.

        Inherits the parent's step budget (shared counter).
        Returns the sub-sim's result without mutating parent state.
        """
        if len(expr) != 2:
            return LispError("sub-sim needs exactly 1 argument (body expression)")
        if self.current_depth >= self.max_depth:
            return LispError(
                f"sub-sim depth limit reached ({self.max_depth})"
            )
        child = Lispy(
            max_steps=self.max_steps,
            max_depth=self.max_depth,
            current_depth=self.current_depth + 1,
            step_budget=self.step_budget,  # shared budget
        )
        # Sub-sim gets a snapshot of parent env (read-only semantics via copy)
        child_env = _snapshot_env(env, child.global_env)
        return child.eval_expr(expr[1], child_env)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_error_inspector(fn: object) -> bool:
    """Check if a function is tagged to receive error values as arguments."""
    return getattr(fn, "_inspects_errors", False)


def _truthy(val: object) -> bool:
    """LisPy truthiness: false and nil/[] are falsy, everything else truthy."""
    if val is False or val is None:
        return False
    if isinstance(val, list) and len(val) == 0:
        return False
    if isinstance(val, LispError):
        return False
    return True


def _type_name(val: object) -> str:
    """Return the LisPy type name of a value."""
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int"
    if isinstance(val, float):
        return "float"
    if isinstance(val, Symbol):
        return "symbol"
    if isinstance(val, str):
        return "string"
    if isinstance(val, list):
        return "list"
    if isinstance(val, dict):
        return "dict"
    if isinstance(val, Closure):
        return "function"
    if isinstance(val, LispError):
        return "error"
    return "unknown"


def _snapshot_env(source: Env, base: Env) -> Env:
    """Create a shallow snapshot of source env layered on top of base.

    Sub-sims see parent bindings but cannot mutate them — writes go
    to the snapshot layer.
    """
    snapshot_data: dict = {}
    _collect_bindings(source, snapshot_data)
    return Env(data=snapshot_data, parent=base)


def _collect_bindings(env: Env, target: dict) -> None:
    """Recursively collect all bindings from env chain (child overrides parent)."""
    if env.parent is not None:
        _collect_bindings(env.parent, target)
    target.update(env.data)


def to_sexp(val: object) -> str:
    """Serialize a LisPy value back to s-expression string."""
    if val is None or (isinstance(val, list) and len(val) == 0):
        return "nil"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return f"{val:.6g}"
    if isinstance(val, Symbol):
        return str(val)
    if isinstance(val, str):
        return f'"{val}"'
    if isinstance(val, LispError):
        return f'(error "{val.message}")'
    if isinstance(val, Closure):
        return repr(val)
    if isinstance(val, dict):
        pairs = " ".join(f'"{k}" {to_sexp(v)}' for k, v in val.items())
        return f"(dict {pairs})"
    if isinstance(val, list):
        inner = " ".join(to_sexp(v) for v in val)
        return f"({inner})"
    return str(val)
