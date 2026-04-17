"""lispy_vm.py — Safe-eval LisPy interpreter for Mars-100 governance DSL.

A minimal Lisp dialect for modeling colonist decisions, governance proposals,
and sub-simulation scenarios. No I/O, no imports, no side effects — pure
computation only. Homoiconic: data structures ARE programs.

Constitutional constraints (Amendment XIII — Turtles All the Way Down):
  - Max recursion depth: 3 levels of sub-simulation
  - Sandboxed: no file access, no network, no imports
  - Inherit parent constitution, may propose amendments within scope
  - Results bubble back as evidence for colonist decisions
  - Ephemeral: dissolve after returning result

Usage:
    from src.lispy_vm import LispyVM
    vm = LispyVM(max_depth=3)
    result = vm.eval_str("(+ 1 2)")  # => 3
    result = vm.eval_str("(define x 10) (+ x 5)")  # => 15
"""
from __future__ import annotations

import math
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Symbol(str):
    """A Lisp symbol — distinct from string literals."""
    pass


@dataclass
class Lambda:
    """A user-defined function (closure)."""
    params: list[str]
    body: Any
    env: "Env"
    name: str = "<lambda>"

    def __repr__(self) -> str:
        return f"<lambda {self.name}({', '.join(self.params)})>"


class LispyError(Exception):
    """Runtime error in the LisPy VM."""
    pass


class LispyDepthError(LispyError):
    """Sub-simulation recursion depth exceeded."""
    pass


class LispySafetyError(LispyError):
    """Attempted unsafe operation."""
    pass


class LispyStepLimitError(LispyError):
    """Evaluation step limit exceeded."""
    pass


# ---------------------------------------------------------------------------
# Environment (scope chain)
# ---------------------------------------------------------------------------

class Env(dict):
    """An environment: a dict of {'var': val} pairs, with an outer Env."""

    def __init__(self, params: tuple = (), args: tuple = (), outer: "Env | None" = None):
        super().__init__(zip(params, args))
        self.outer = outer

    def find(self, var: str) -> "Env":
        """Find the innermost Env where var appears."""
        if var in self:
            return self
        if self.outer is not None:
            return self.outer.find(var)
        raise LispyError(f"Undefined symbol: {var}")

    def lookup(self, var: str) -> Any:
        """Look up a variable value."""
        return self.find(var)[var]


# ---------------------------------------------------------------------------
# Tokenizer + Parser
# ---------------------------------------------------------------------------

# Token regex: parens, quoted strings, comments, atoms
_TOKEN_RE = re.compile(
    r"""(\s+|;[^\n]*|"(?:[^"\\]|\\.)*"|'?\(|\)|'|[^\s()"';]+)""",
    re.VERBOSE,
)


def tokenize(source: str) -> list[str]:
    """Tokenize a LisPy source string."""
    tokens = []
    for match in _TOKEN_RE.finditer(source):
        tok = match.group(1)
        if tok.startswith(";") or tok.strip() == "":
            continue
        tokens.append(tok)
    return tokens


def parse(source: str) -> list[Any]:
    """Parse a LisPy source string into a list of expressions."""
    tokens = tokenize(source)
    exprs = []
    pos = 0
    while pos < len(tokens):
        expr, pos = _parse_expr(tokens, pos)
        exprs.append(expr)
    return exprs


def _parse_expr(tokens: list[str], pos: int) -> tuple[Any, int]:
    """Parse one expression starting at pos. Returns (expr, new_pos)."""
    if pos >= len(tokens):
        raise LispyError("Unexpected end of input")

    tok = tokens[pos]

    if tok == "'":
        # Quote shorthand: 'x => (quote x)
        inner, pos = _parse_expr(tokens, pos + 1)
        return [Symbol("quote"), inner], pos

    if tok == "(":
        lst: list[Any] = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ")":
            elem, pos = _parse_expr(tokens, pos)
            lst.append(elem)
        if pos >= len(tokens):
            raise LispyError("Missing closing parenthesis")
        return lst, pos + 1  # skip ')'

    if tok == "'(":
        # Quoted list shorthand
        lst = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ")":
            elem, pos = _parse_expr(tokens, pos)
            lst.append(elem)
        if pos >= len(tokens):
            raise LispyError("Missing closing parenthesis")
        return [Symbol("quote"), lst], pos + 1

    if tok == ")":
        raise LispyError("Unexpected closing parenthesis")

    # Atom
    return _parse_atom(tok), pos + 1


def _parse_atom(token: str) -> Any:
    """Parse an atom: number, string, bool, nil, or symbol."""
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1].replace('\\"', '"').replace("\\n", "\n")

    if token == "#t" or token == "true":
        return True
    if token == "#f" or token == "false":
        return False
    if token == "nil" or token == "null":
        return None

    # Try integer first, then float
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass

    return Symbol(token)


# ---------------------------------------------------------------------------
# Standard library (built-in functions)
# ---------------------------------------------------------------------------

def _stdlib() -> dict[str, Any]:
    """Build the standard library of built-in functions."""
    env: dict[str, Any] = {}

    # Arithmetic
    env["+"] = lambda *args: sum(args)
    env["-"] = lambda a, b=None: -a if b is None else a - b
    env["*"] = lambda *args: math.prod(args)
    env["/"] = lambda a, b: a / b if b != 0 else float("inf")
    env["//"] = lambda a, b: a // b if b != 0 else 0
    env["%"] = lambda a, b: a % b if b != 0 else 0
    env["mod"] = env["%"]
    env["abs"] = abs
    env["min"] = min
    env["max"] = max
    env["pow"] = pow
    env["sqrt"] = math.sqrt
    env["floor"] = math.floor
    env["ceil"] = math.ceil
    env["round"] = round

    # Comparison
    env["="] = operator.eq
    env["=="] = operator.eq
    env["!="] = operator.ne
    env["<"] = operator.lt
    env[">"] = operator.gt
    env["<="] = operator.le
    env[">="] = operator.ge

    # Boolean
    env["and"] = lambda *args: all(args)
    env["or"] = lambda *args: any(args)
    env["not"] = operator.not_

    # List operations
    env["list"] = lambda *args: list(args)
    env["cons"] = lambda a, b: [a] + (b if isinstance(b, list) else [b])
    env["car"] = lambda lst: lst[0] if lst else None
    env["cdr"] = lambda lst: lst[1:] if lst else []
    env["first"] = env["car"]
    env["rest"] = env["cdr"]
    env["len"] = len
    env["length"] = len
    env["nth"] = lambda lst, n: lst[n] if 0 <= n < len(lst) else None
    env["append"] = lambda *lsts: sum((l if isinstance(l, list) else [l] for l in lsts), [])
    env["reverse"] = lambda lst: list(reversed(lst))
    env["range"] = lambda *args: list(range(*args))
    env["empty?"] = lambda lst: len(lst) == 0 if isinstance(lst, list) else lst is None
    env["contains?"] = lambda lst, x: x in lst

    # Dict / assoc operations
    env["dict"] = lambda *pairs: dict(zip(pairs[::2], pairs[1::2]))
    env["get"] = lambda d, k, default=None: d.get(k, default) if isinstance(d, dict) else default
    env["put"] = lambda d, k, v: {**d, k: v} if isinstance(d, dict) else d
    env["keys"] = lambda d: list(d.keys()) if isinstance(d, dict) else []
    env["values"] = lambda d: list(d.values()) if isinstance(d, dict) else []
    env["has?"] = lambda d, k: k in d if isinstance(d, dict) else False

    # String operations
    env["str"] = lambda *args: "".join(str(a) for a in args)
    env["str-join"] = lambda sep, lst: sep.join(str(x) for x in lst)
    env["str-split"] = lambda s, sep=" ": s.split(sep) if isinstance(s, str) else []
    env["str-upper"] = lambda s: s.upper() if isinstance(s, str) else s
    env["str-lower"] = lambda s: s.lower() if isinstance(s, str) else s

    # Type checks
    env["number?"] = lambda x: isinstance(x, (int, float))
    env["string?"] = lambda x: isinstance(x, str) and not isinstance(x, Symbol)
    env["list?"] = lambda x: isinstance(x, list)
    env["dict?"] = lambda x: isinstance(x, dict)
    env["nil?"] = lambda x: x is None
    env["symbol?"] = lambda x: isinstance(x, Symbol)
    env["bool?"] = lambda x: isinstance(x, bool)

    # Math constants
    env["pi"] = math.pi
    env["e"] = math.e
    env["inf"] = float("inf")

    # Functional (will be enhanced by special form map/filter in eval)
    # These are stubs — actual higher-order use goes through eval
    env["identity"] = lambda x: x

    return env


# ---------------------------------------------------------------------------
# The VM
# ---------------------------------------------------------------------------

MAX_STEPS_DEFAULT = 50_000


class LispyVM:
    """Safe-eval LisPy virtual machine.

    Args:
        max_depth: Maximum sub-simulation nesting depth (default 3).
        max_steps: Maximum evaluation steps before abort (default 50000).
        rng_seed: Optional seed for deterministic random in governance DSL.
    """

    def __init__(
        self,
        max_depth: int = 3,
        max_steps: int = MAX_STEPS_DEFAULT,
        rng_seed: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.rng_seed = rng_seed
        self.steps = 0
        self.sub_sim_log: list[dict] = []
        self.global_env = self._make_global_env()

    def _make_global_env(self) -> Env:
        """Create the global environment with stdlib."""
        env = Env()
        env.update(_stdlib())
        return env

    def reset(self) -> None:
        """Reset the VM state."""
        self.steps = 0
        self.sub_sim_log = []
        self.global_env = self._make_global_env()

    def eval_str(self, source: str, env: Env | None = None) -> Any:
        """Parse and evaluate a LisPy source string. Returns last expression value."""
        exprs = parse(source)
        if not exprs:
            return None
        result = None
        target_env = env if env is not None else self.global_env
        for expr in exprs:
            result = self._eval(expr, target_env, depth=0)
        return result

    def eval_expr(self, expr: Any, env: Env | None = None, depth: int = 0) -> Any:
        """Evaluate a pre-parsed expression."""
        target_env = env if env is not None else self.global_env
        return self._eval(expr, target_env, depth=depth)

    def _step(self) -> None:
        """Increment step counter, raise if limit exceeded."""
        self.steps += 1
        if self.steps > self.max_steps:
            raise LispyStepLimitError(
                f"Evaluation exceeded {self.max_steps} steps"
            )

    def _eval(self, expr: Any, env: Env, depth: int = 0) -> Any:
        """Evaluate an expression in an environment."""
        self._step()

        # Atoms
        if isinstance(expr, Symbol):
            return env.lookup(expr)
        if not isinstance(expr, list):
            return expr  # number, string, bool, None
        if len(expr) == 0:
            return []

        head = expr[0]

        # Special forms
        if isinstance(head, Symbol):
            # quote
            if head == "quote":
                if len(expr) != 2:
                    raise LispyError("quote requires exactly 1 argument")
                return expr[1]

            # define
            if head == "define":
                if len(expr) == 3 and isinstance(expr[1], Symbol):
                    val = self._eval(expr[2], env, depth)
                    env[expr[1]] = val
                    return val
                # Function shorthand: (define (f x y) body)
                if len(expr) >= 3 and isinstance(expr[1], list):
                    name = expr[1][0]
                    params = [str(p) for p in expr[1][1:]]
                    body = expr[2] if len(expr) == 3 else [Symbol("begin")] + expr[2:]
                    lam = Lambda(params=params, body=body, env=env, name=str(name))
                    env[name] = lam
                    return lam
                raise LispyError(f"Bad define form: {expr}")

            # set!
            if head == "set!":
                if len(expr) != 3:
                    raise LispyError("set! requires symbol and value")
                var = expr[1]
                val = self._eval(expr[2], env, depth)
                env.find(var)[var] = val
                return val

            # if
            if head == "if":
                if len(expr) not in (3, 4):
                    raise LispyError("if requires 2 or 3 arguments")
                cond = self._eval(expr[1], env, depth)
                if cond:
                    return self._eval(expr[2], env, depth)
                elif len(expr) == 4:
                    return self._eval(expr[3], env, depth)
                return None

            # cond
            if head == "cond":
                for clause in expr[1:]:
                    if not isinstance(clause, list) or len(clause) < 2:
                        raise LispyError(f"Bad cond clause: {clause}")
                    test = clause[0]
                    if (isinstance(test, Symbol) and test == "else") or self._eval(test, env, depth):
                        result = None
                        for body_expr in clause[1:]:
                            result = self._eval(body_expr, env, depth)
                        return result
                return None

            # let / let*
            if head in ("let", "let*"):
                if len(expr) < 3:
                    raise LispyError(f"{head} requires bindings and body")
                bindings = expr[1]
                inner = Env(outer=env)
                for binding in bindings:
                    if not isinstance(binding, list) or len(binding) != 2:
                        raise LispyError(f"Bad {head} binding: {binding}")
                    var_name = str(binding[0])
                    # let* evaluates in the growing env; let evaluates in outer
                    eval_env = inner if head == "let*" else env
                    inner[var_name] = self._eval(binding[1], eval_env, depth)
                result = None
                for body_expr in expr[2:]:
                    result = self._eval(body_expr, inner, depth)
                return result

            # lambda
            if head == "lambda" or head == "fn":
                if len(expr) < 3:
                    raise LispyError("lambda requires params and body")
                params = [str(p) for p in expr[1]]
                body = expr[2] if len(expr) == 3 else [Symbol("begin")] + expr[2:]
                return Lambda(params=params, body=body, env=env)

            # begin (sequential evaluation)
            if head == "begin":
                result = None
                for body_expr in expr[1:]:
                    result = self._eval(body_expr, env, depth)
                return result

            # do (alias for begin)
            if head == "do":
                result = None
                for body_expr in expr[1:]:
                    result = self._eval(body_expr, env, depth)
                return result

            # map
            if head == "map":
                if len(expr) != 3:
                    raise LispyError("map requires function and list")
                fn = self._eval(expr[1], env, depth)
                lst = self._eval(expr[2], env, depth)
                if not isinstance(lst, list):
                    raise LispyError("map second argument must be a list")
                return [self._apply(fn, [x], depth) for x in lst]

            # filter
            if head == "filter":
                if len(expr) != 3:
                    raise LispyError("filter requires predicate and list")
                pred = self._eval(expr[1], env, depth)
                lst = self._eval(expr[2], env, depth)
                if not isinstance(lst, list):
                    raise LispyError("filter second argument must be a list")
                return [x for x in lst if self._apply(pred, [x], depth)]

            # reduce
            if head == "reduce":
                if len(expr) != 4:
                    raise LispyError("reduce requires function, initial, and list")
                fn = self._eval(expr[1], env, depth)
                acc = self._eval(expr[2], env, depth)
                lst = self._eval(expr[3], env, depth)
                if not isinstance(lst, list):
                    raise LispyError("reduce third argument must be a list")
                for item in lst:
                    acc = self._apply(fn, [acc, item], depth)
                return acc

            # sub-sim — the recursive simulation primitive
            if head == "sub-sim":
                return self._eval_sub_sim(expr, env, depth)

            # propose — governance proposal primitive
            if head == "propose":
                return self._eval_propose(expr, env, depth)

            # vote — governance voting primitive
            if head == "vote":
                return self._eval_vote(expr, env, depth)

        # Function application
        fn = self._eval(head, env, depth)
        args = [self._eval(arg, env, depth) for arg in expr[1:]]
        return self._apply(fn, args, depth)

    def _apply(self, fn: Any, args: list[Any], depth: int) -> Any:
        """Apply a function (Lambda or built-in) to arguments."""
        if isinstance(fn, Lambda):
            if len(args) != len(fn.params):
                raise LispyError(
                    f"{fn.name} expected {len(fn.params)} args, got {len(args)}"
                )
            inner = Env(tuple(fn.params), tuple(args), fn.env)
            return self._eval(fn.body, inner, depth)
        if callable(fn):
            try:
                return fn(*args)
            except TypeError as e:
                raise LispyError(f"Built-in call error: {e}") from e
        raise LispyError(f"Not a function: {fn}")

    def _eval_sub_sim(self, expr: list, env: Env, depth: int) -> Any:
        """Evaluate a sub-simulation expression.

        Syntax: (sub-sim <label> <body-expr>)
        The body is evaluated in a fresh sandboxed child VM at depth+1.
        """
        if len(expr) < 3:
            raise LispyError("sub-sim requires label and body")

        label = self._eval(expr[1], env, depth)
        body = expr[2]
        new_depth = depth + 1

        if new_depth > self.max_depth:
            raise LispyDepthError(
                f"Sub-simulation depth {new_depth} exceeds max {self.max_depth}"
            )

        # Create child VM with reduced step budget
        child_steps = max(1000, self.max_steps // 4)
        child_vm = LispyVM(
            max_depth=self.max_depth,
            max_steps=child_steps,
            rng_seed=(self.rng_seed + new_depth) if self.rng_seed is not None else None,
        )

        # Copy parent env bindings into child (read-only snapshot)
        for key, val in env.items():
            if not callable(val) and not isinstance(val, Lambda):
                child_vm.global_env[key] = val

        try:
            result = child_vm._eval(body, child_vm.global_env, new_depth)
            log_entry = {
                "depth": new_depth,
                "label": str(label),
                "status": "completed",
                "result": _serialize_value(result),
                "steps_used": child_vm.steps,
                "sub_sims": child_vm.sub_sim_log,
            }
        except LispyDepthError:
            # Depth errors propagate up — they mean the whole chain is too deep
            raise
        except LispyError as e:
            result = None
            log_entry = {
                "depth": new_depth,
                "label": str(label),
                "status": "error",
                "error": str(e),
                "steps_used": child_vm.steps,
                "sub_sims": child_vm.sub_sim_log,
            }

        self.sub_sim_log.append(log_entry)
        return result

    def _eval_propose(self, expr: list, env: Env, depth: int) -> dict:
        """Create a governance proposal.

        Syntax: (propose <type> <description> <value>)
        Returns a dict representing the proposal.
        """
        if len(expr) < 4:
            raise LispyError("propose requires type, description, and value")

        prop_type = self._eval(expr[1], env, depth)
        description = self._eval(expr[2], env, depth)
        value = self._eval(expr[3], env, depth)

        return {
            "type": "proposal",
            "proposal_type": str(prop_type),
            "description": str(description),
            "value": _serialize_value(value),
            "depth": depth,
        }

    def _eval_vote(self, expr: list, env: Env, depth: int) -> dict:
        """Cast a governance vote.

        Syntax: (vote <proposal-id> <position> <weight>)
        Position: "for", "against", "abstain"
        """
        if len(expr) < 4:
            raise LispyError("vote requires proposal-id, position, and weight")

        proposal_id = self._eval(expr[1], env, depth)
        position = self._eval(expr[2], env, depth)
        weight = self._eval(expr[3], env, depth)

        if position not in ("for", "against", "abstain"):
            raise LispyError(f"Invalid vote position: {position}")
        if not isinstance(weight, (int, float)):
            raise LispyError(f"Vote weight must be a number, got {type(weight)}")
        # Clamp weight to [0, 10] to prevent governance runaway
        weight = max(0.0, min(10.0, float(weight)))

        return {
            "type": "vote",
            "proposal_id": str(proposal_id),
            "position": str(position),
            "weight": weight,
            "depth": depth,
        }


def _serialize_value(val: Any) -> Any:
    """Convert a LisPy value to JSON-serializable form."""
    if val is None:
        return None
    if isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, Symbol):
        return str(val)
    if isinstance(val, list):
        return [_serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _serialize_value(v) for k, v in val.items()}
    if isinstance(val, Lambda):
        return f"<lambda {val.name}>"
    return str(val)


def format_sexpr(val: Any) -> str:
    """Format a Python value as an s-expression string."""
    if val is None:
        return "nil"
    if isinstance(val, bool):
        return "#t" if val else "#f"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        if isinstance(val, Symbol):
            return str(val)
        return f'"{val}"'
    if isinstance(val, list):
        inner = " ".join(format_sexpr(v) for v in val)
        return f"({inner})"
    if isinstance(val, dict):
        pairs = " ".join(f"{format_sexpr(k)} {format_sexpr(v)}" for k, v in val.items())
        return f"(dict {pairs})"
    return str(val)
