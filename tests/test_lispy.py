"""Tests for the LisPy safe-eval interpreter."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lispy import (
    DepthExceeded,
    Env,
    EvalContext,
    Lambda,
    LispyError,
    NIL,
    StepBudgetExceeded,
    Symbol,
    evaluate,
    make_builtins,
    make_env,
    parse,
    parse_all,
    run,
    run_in_context,
    safe_eval,
    serialize,
    to_sexpr,
    tokenize,
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenizer:
    def test_empty(self):
        assert tokenize("") == []

    def test_simple_expr(self):
        tokens = tokenize("(+ 1 2)")
        assert tokens == ["(", "+", "1", "2", ")"]

    def test_nested(self):
        tokens = tokenize("(if (> x 0) x (- x))")
        assert "if" in tokens and ">" in tokens

    def test_string_literal(self):
        tokens = tokenize('(str "hello world")')
        assert '"hello world"' in tokens

    def test_comments_stripped(self):
        tokens = tokenize("; this is a comment\n(+ 1 2)")
        assert ";" not in "".join(tokens)
        assert "+" in tokens

    def test_quote_shorthand(self):
        tokens = tokenize("'(1 2 3)")
        assert "'" in tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParser:
    def test_integer(self):
        assert parse("42") == 42

    def test_float(self):
        result = parse("3.14")
        assert abs(result - 3.14) < 0.01

    def test_string(self):
        assert parse('"hello"') == "hello"

    def test_symbol(self):
        result = parse("foo")
        assert isinstance(result, Symbol)
        assert result.name == "foo"

    def test_true(self):
        assert parse("true") is True

    def test_false(self):
        assert parse("false") is False

    def test_nil(self):
        assert parse("nil") is NIL

    def test_list(self):
        result = parse("(1 2 3)")
        assert result == [1, 2, 3]

    def test_nested_list(self):
        result = parse("(+ (- 3 1) 4)")
        assert len(result) == 3

    def test_quote_shorthand(self):
        result = parse("'(1 2 3)")
        assert result[0] == Symbol("quote")

    def test_parse_all_multiple(self):
        results = parse_all("1 2 3")
        assert results == [1, 2, 3]

    def test_empty_parens(self):
        result = parse("()")
        assert result == []


# ---------------------------------------------------------------------------
# Evaluator — arithmetic
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_add(self):
        result, _ = run("(+ 1 2 3)")
        assert result == 6

    def test_subtract(self):
        result, _ = run("(- 10 3)")
        assert result == 7

    def test_negate(self):
        result, _ = run("(- 5)")
        assert result == -5

    def test_multiply(self):
        result, _ = run("(* 3 4)")
        assert result == 12

    def test_divide(self):
        result, _ = run("(/ 10 3)")
        assert abs(result - 10 / 3) < 0.001

    def test_divide_by_zero(self):
        result = safe_eval("(/ 1 0)")
        assert not result["ok"]
        assert "zero" in result["error"].lower()

    def test_modulo(self):
        result, _ = run("(% 10 3)")
        assert result == 1

    def test_nested_arithmetic(self):
        result, _ = run("(+ (* 2 3) (- 10 4))")
        assert result == 12

    def test_abs(self):
        result, _ = run("(abs -5)")
        assert result == 5

    def test_max(self):
        result, _ = run("(max 1 5 3)")
        assert result == 5

    def test_min(self):
        result, _ = run("(min 1 5 3)")
        assert result == 1

    def test_sqrt(self):
        result, _ = run("(sqrt 16)")
        assert result == 4.0

    def test_floor(self):
        result, _ = run("(floor 3.7)")
        assert result == 3

    def test_ceil(self):
        result, _ = run("(ceil 3.2)")
        assert result == 4

    def test_round(self):
        result, _ = run("(round 3.456 2)")
        assert abs(result - 3.46) < 0.001


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


class TestComparisons:
    def test_eq(self):
        result, _ = run("(= 5 5)")
        assert result is True

    def test_eq_false(self):
        result, _ = run("(= 5 3)")
        assert result is False

    def test_lt(self):
        result, _ = run("(< 3 5)")
        assert result is True

    def test_gt(self):
        result, _ = run("(> 5 3)")
        assert result is True

    def test_le(self):
        result, _ = run("(<= 5 5)")
        assert result is True

    def test_ge(self):
        result, _ = run("(>= 5 3)")
        assert result is True


# ---------------------------------------------------------------------------
# Special forms
# ---------------------------------------------------------------------------


class TestSpecialForms:
    def test_if_true(self):
        result, _ = run("(if true 1 2)")
        assert result == 1

    def test_if_false(self):
        result, _ = run("(if false 1 2)")
        assert result == 2

    def test_if_no_else(self):
        result, _ = run("(if false 1)")
        assert result is NIL

    def test_cond(self):
        result, _ = run("(cond ((= 1 2) 10) ((= 1 1) 20) (true 30))")
        assert result == 20

    def test_define(self):
        result, _ = run("(begin (define x 42) x)")
        assert result == 42

    def test_lambda(self):
        result, _ = run("((lambda (x) (* x x)) 5)")
        assert result == 25

    def test_let(self):
        result, _ = run("(let ((x 3) (y 4)) (+ x y))")
        assert result == 7

    def test_begin(self):
        result, _ = run("(begin 1 2 3)")
        assert result == 3

    def test_and(self):
        result, _ = run("(and true true)")
        assert result is True

    def test_and_short_circuit(self):
        result, _ = run("(and false true)")
        assert result is False

    def test_or(self):
        result, _ = run("(or false 42)")
        assert result == 42

    def test_not_true(self):
        result, _ = run("(not true)")
        assert result is False

    def test_not_false(self):
        result, _ = run("(not false)")
        assert result is True

    def test_quote(self):
        result, _ = run("(quote (1 2 3))")
        assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# List operations
# ---------------------------------------------------------------------------


class TestListOps:
    def test_list(self):
        result, _ = run("(list 1 2 3)")
        assert result == [1, 2, 3]

    def test_car(self):
        result, _ = run("(car (list 1 2 3))")
        assert result == 1

    def test_cdr(self):
        result, _ = run("(cdr (list 1 2 3))")
        assert result == [2, 3]

    def test_cons(self):
        result, _ = run("(cons 0 (list 1 2))")
        assert result == [0, 1, 2]

    def test_length(self):
        result, _ = run("(length (list 1 2 3 4))")
        assert result == 4

    def test_append(self):
        result, _ = run("(append (list 1 2) (list 3 4))")
        assert result == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Type predicates
# ---------------------------------------------------------------------------


class TestTypePredicates:
    def test_is_list(self):
        result, _ = run("(list? (list 1))")
        assert result is True

    def test_is_number(self):
        result, _ = run("(number? 42)")
        assert result is True

    def test_is_string(self):
        result, _ = run('(string? "hello")')
        assert result is True

    def test_is_nil(self):
        result, _ = run("(nil? nil)")
        assert result is True


# ---------------------------------------------------------------------------
# Lambda and closures
# ---------------------------------------------------------------------------


class TestClosures:
    def test_closure_captures_env(self):
        result, _ = run("""
            (let ((x 10))
              (let ((f (lambda (y) (+ x y))))
                (f 5)))
        """)
        assert result == 15

    def test_higher_order(self):
        result, _ = run("""
            (let ((apply-twice (lambda (f x) (f (f x)))))
              (apply-twice (lambda (n) (+ n 1)) 0))
        """)
        assert result == 2

    def test_recursive_via_define(self):
        result, _ = run("""
            (begin
              (define fact (lambda (n) (if (<= n 1) 1 (* n (fact (- n 1))))))
              (fact 5))
        """)
        assert result == 120


# ---------------------------------------------------------------------------
# Sub-simulations
# ---------------------------------------------------------------------------


class TestSubSim:
    def test_depth_1(self):
        result = safe_eval("(sub-sim (+ 1 2))")
        assert result["ok"]
        assert result["value"] == 3
        assert result["sub_sims"] == 1

    def test_depth_2(self):
        result = safe_eval("(sub-sim (sub-sim (* 3 4)))")
        assert result["ok"]
        assert result["value"] == 12
        assert result["sub_sims"] == 2

    def test_depth_3(self):
        result = safe_eval("(sub-sim (sub-sim (sub-sim (+ 10 20))))")
        assert result["ok"]
        assert result["value"] == 30
        assert result["sub_sims"] == 3

    def test_depth_4_fails(self):
        result = safe_eval("(sub-sim (sub-sim (sub-sim (sub-sim 1))))")
        assert not result["ok"]
        assert "depth" in result["error"].lower() or "Depth" in result["error"]

    def test_subsim_shares_budget(self):
        """Sub-sims share the global step budget."""
        result = safe_eval("(sub-sim (+ 1 1))", max_steps=10)
        assert result["ok"]
        # Steps should be > 0 (shared context)
        assert result["steps"] > 0


# ---------------------------------------------------------------------------
# Safety limits
# ---------------------------------------------------------------------------


class TestSafetyLimits:
    def test_step_budget(self):
        # Infinite loop should hit step budget
        result = safe_eval("""
            (begin
              (define loop (lambda (n) (loop (+ n 1))))
              (loop 0))
        """, max_steps=100)
        assert not result["ok"]
        assert "step" in result["error"].lower() or "budget" in result["error"].lower()

    def test_call_depth(self):
        # Deep recursion
        result = safe_eval("""
            (begin
              (define deep (lambda (n) (deep (+ n 1))))
              (deep 0))
        """, max_steps=50000)
        assert not result["ok"]

    def test_unbound_symbol(self):
        result = safe_eval("undefined-var")
        assert not result["ok"]
        assert "unbound" in result["error"].lower()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_sexpr_int(self):
        assert to_sexpr(42) == "42"

    def test_to_sexpr_float(self):
        assert "3.14" in to_sexpr(3.14)

    def test_to_sexpr_string(self):
        assert to_sexpr("hello") == '"hello"'

    def test_to_sexpr_bool(self):
        assert to_sexpr(True) == "true"
        assert to_sexpr(False) == "false"

    def test_to_sexpr_nil(self):
        assert to_sexpr(NIL) == "nil"

    def test_to_sexpr_list(self):
        assert to_sexpr([1, 2, 3]) == "(1 2 3)"

    def test_to_sexpr_symbol(self):
        assert to_sexpr(Symbol("foo")) == "foo"

    def test_serialize_basic(self):
        assert serialize(42) == 42
        assert serialize("hello") == "hello"
        assert serialize(True) is True
        assert serialize(NIL) is None

    def test_serialize_list(self):
        assert serialize([1, Symbol("x"), NIL]) == [1, "x", None]

    def test_serialize_lambda(self):
        lam = Lambda([Symbol("x")], [Symbol("+"), Symbol("x"), 1], Env())
        result = serialize(lam)
        assert result["__type__"] == "lambda"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TestEnvironment:
    def test_lookup(self):
        env = Env({"x": 42})
        assert env.lookup("x") == 42

    def test_parent_lookup(self):
        parent = Env({"x": 42})
        child = Env({}, parent)
        assert child.lookup("x") == 42

    def test_unbound_raises(self):
        env = Env()
        try:
            env.lookup("missing")
            assert False, "Should have raised"
        except LispyError:
            pass

    def test_set(self):
        env = Env()
        env.set("x", 99)
        assert env.lookup("x") == 99


# ---------------------------------------------------------------------------
# EvalContext
# ---------------------------------------------------------------------------


class TestEvalContext:
    def test_tick_counts(self):
        ctx = EvalContext(max_steps=10)
        for _ in range(10):
            ctx.tick()
        assert ctx.steps == 10

    def test_tick_exceeds(self):
        ctx = EvalContext(max_steps=5)
        try:
            for _ in range(10):
                ctx.tick()
            assert False
        except StepBudgetExceeded:
            pass

    def test_enter_exit_subsim(self):
        ctx = EvalContext()
        prev = ctx.enter_subsim()
        assert ctx.sim_depth == 1
        ctx.exit_subsim(prev)
        assert ctx.sim_depth == 0

    def test_max_depth(self):
        ctx = EvalContext(max_depth=2)
        ctx.enter_subsim()
        ctx.enter_subsim()
        try:
            ctx.enter_subsim()
            assert False
        except DepthExceeded:
            pass


# ---------------------------------------------------------------------------
# Integration: safe_eval
# ---------------------------------------------------------------------------


class TestSafeEval:
    def test_ok_result(self):
        result = safe_eval("(+ 1 2)")
        assert result["ok"] is True
        assert result["value"] == 3

    def test_error_result(self):
        result = safe_eval("(/ 1 0)")
        assert result["ok"] is False
        assert "error" in result

    def test_steps_tracked(self):
        result = safe_eval("(+ 1 2)")
        assert result["steps"] > 0

    def test_subsims_tracked(self):
        result = safe_eval("(sub-sim (+ 1 1))")
        assert result["sub_sims"] == 1

    def test_custom_env(self):
        env = make_env({"my-var": 99})
        result = safe_eval("my-var", env=env)
        assert result["ok"]
        assert result["value"] == 99


# ---------------------------------------------------------------------------
# run_in_context
# ---------------------------------------------------------------------------


class TestRunInContext:
    def test_shares_context(self):
        env = make_env()
        ctx = EvalContext()
        run_in_context("(define x 10)", env, ctx)
        result = run_in_context("x", env, ctx)
        assert result == 10
        assert ctx.steps > 0


# ---------------------------------------------------------------------------
# Property-based: builtins are all callable
# ---------------------------------------------------------------------------


class TestBuiltins:
    def test_all_builtins_exist(self):
        b = make_builtins()
        for name in ["+", "-", "*", "/", "%", "=", "<", ">", "list",
                     "car", "cdr", "cons", "length", "abs", "max", "min"]:
            assert name in b

    def test_pi_and_e(self):
        env = make_env()
        result, _ = run("pi", env)
        assert abs(result - 3.14159) < 0.001
        result, _ = run("e", env)
        assert abs(result - 2.71828) < 0.001
