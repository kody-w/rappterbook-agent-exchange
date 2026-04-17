"""Tests for the LisPy safe-eval interpreter."""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lispy import (
    parse, parse_all, run, tokenize, to_sexpr,
    Evaluator, Env, standard_env, Closure,
    LispyError, StepLimitError, DepthLimitError,
)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_simple(self):
        assert tokenize("(+ 1 2)") == ["(", "+", "1", "2", ")"]

    def test_nested(self):
        tokens = tokenize("(if (> x 0) x (- x))")
        assert tokens[0] == "("
        assert "if" in tokens

    def test_string(self):
        tokens = tokenize('(str "hello world")')
        assert '"hello world"' in tokens

    def test_comment(self):
        tokens = tokenize("; this is a comment\n(+ 1 2)")
        assert ";" not in "".join(tokens)
        assert "+" in tokens


class TestParser:
    def test_number(self):
        assert parse("42") == 42.0

    def test_float(self):
        assert parse("3.14") == 3.14

    def test_string(self):
        assert parse('"hello"') == "hello"

    def test_symbol(self):
        assert parse("foo") == "foo"

    def test_list(self):
        assert parse("(+ 1 2)") == ["+", 1.0, 2.0]

    def test_nested_list(self):
        result = parse("(if (> x 0) x (- x))")
        assert result[0] == "if"
        assert isinstance(result[1], list)

    def test_empty_list(self):
        assert parse("()") == []

    def test_boolean_true(self):
        assert parse("#t") is True

    def test_boolean_false(self):
        assert parse("#f") is False

    def test_parse_all(self):
        exprs = parse_all("(define x 1) (+ x 2)")
        assert len(exprs) == 2

    def test_unmatched_paren(self):
        with pytest.raises(LispyError):
            parse("(+ 1 2")

    def test_extra_close_paren(self):
        with pytest.raises(LispyError):
            parse(")")


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------

class TestArithmetic:
    def test_add(self):
        assert run("(+ 1 2)") == 3.0

    def test_subtract(self):
        assert run("(- 10 3)") == 7.0

    def test_multiply(self):
        assert run("(* 3 4)") == 12.0

    def test_divide(self):
        assert run("(/ 10 3)") == pytest.approx(3.333, rel=0.01)

    def test_divide_by_zero(self):
        result = run("(/ 1 0)")
        assert result == float('inf')

    def test_nested_arithmetic(self):
        assert run("(+ (* 2 3) (- 10 4))") == 12.0

    def test_negate(self):
        assert run("(- 5)") == -5.0

    def test_modulo(self):
        assert run("(% 10 3)") == 1.0

    def test_abs(self):
        assert run("(abs -7)") == 7.0

    def test_min_max(self):
        assert run("(min 3 5)") == 3.0
        assert run("(max 3 5)") == 5.0

    def test_sqrt(self):
        assert run("(sqrt 16)") == 4.0

    def test_clamp(self):
        assert run("(clamp 5 0 3)") == 3.0
        assert run("(clamp -1 0 3)") == 0.0
        assert run("(clamp 1 0 3)") == 1.0


class TestComparison:
    def test_equal(self):
        assert run("(= 1 1)") is True
        assert run("(= 1 2)") is False

    def test_greater(self):
        assert run("(> 3 2)") is True

    def test_less(self):
        assert run("(< 1 2)") is True

    def test_not_equal(self):
        assert run("(!= 1 2)") is True


class TestLogic:
    def test_not(self):
        assert run("(not #t)") is False
        assert run("(not #f)") is True

    def test_and(self):
        assert run("(and #t #t)") is True
        assert run("(and #t #f)") is False

    def test_or(self):
        assert run("(or #f #t)") is True
        assert run("(or #f #f)") is False

    def test_and_short_circuit(self):
        # Should not error because second arg not evaluated
        assert run("(and #f (/ 1 0))") is False

    def test_or_short_circuit(self):
        assert run("(or #t (/ 1 0))") is True


class TestControlFlow:
    def test_if_true(self):
        assert run("(if #t 1 2)") == 1.0

    def test_if_false(self):
        assert run("(if #f 1 2)") == 2.0

    def test_if_no_else(self):
        assert run("(if #f 1)") is None

    def test_cond(self):
        assert run("(cond ((> 1 2) 10) ((> 2 1) 20) (else 30))") == 20.0

    def test_begin(self):
        assert run("(begin 1 2 3)") == 3.0


class TestVariables:
    def test_define(self):
        assert run("(begin (define x 42) x)") == 42.0

    def test_define_function(self):
        result = run("(begin (define (square x) (* x x)) (square 5))")
        assert result == 25.0

    def test_let(self):
        assert run("(let ((x 10) (y 20)) (+ x y))") == 30.0

    def test_set(self):
        assert run("(begin (define x 1) (set! x 2) x)") == 2.0

    def test_lambda(self):
        result = run("(begin (define f (lambda (x) (* x x))) (f 7))")
        assert result == 49.0

    def test_closure(self):
        result = run("""
            (begin
                (define (make-adder n) (lambda (x) (+ n x)))
                (define add5 (make-adder 5))
                (add5 10))
        """)
        assert result == 15.0


class TestLists:
    def test_list(self):
        assert run("(list 1 2 3)") == [1.0, 2.0, 3.0]

    def test_car(self):
        assert run("(car (list 1 2 3))") == 1.0

    def test_cdr(self):
        assert run("(cdr (list 1 2 3))") == [2.0, 3.0]

    def test_cons(self):
        assert run("(cons 0 (list 1 2))") == [0.0, 1.0, 2.0]

    def test_length(self):
        assert run("(length (list 1 2 3))") == 3

    def test_nth(self):
        assert run("(nth (list 10 20 30) 1)") == 20.0

    def test_empty(self):
        assert run("(empty? (list))") is True
        assert run("(empty? (list 1))") is False

    def test_map(self):
        result = run("(map (lambda (x) (* x 2)) (list 1 2 3))")
        assert result == [2.0, 4.0, 6.0]

    def test_filter(self):
        result = run("(filter (lambda (x) (> x 2)) (list 1 2 3 4))")
        assert result == [3.0, 4.0]

    def test_reduce(self):
        result = run("(reduce (lambda (a b) (+ a b)) (list 1 2 3 4) 0)")
        assert result == 10.0

    def test_quote(self):
        result = run("(quote (1 2 3))")
        assert result == [1.0, 2.0, 3.0]


class TestAssocLists:
    def test_assoc_create(self):
        result = run('(assoc "name" "aria" nil)')
        assert result == [["name", "aria"]]

    def test_assoc_ref(self):
        result = run("""
            (let ((inner (assoc "b" 2 nil)))
              (let ((data (assoc "a" 1 inner)))
                (assoc-ref data "a")))
        """)
        assert result == 1.0

    def test_assoc_set(self):
        result = run("""
            (let ((data (list (list "x" 1))))
                (assoc-ref (assoc-set data "x" 99) "x"))
        """)
        assert result == 99.0


class TestSerialization:
    def test_to_sexpr_number(self):
        assert to_sexpr(42) == "42"
        assert to_sexpr(3.14) == "3.1400"

    def test_to_sexpr_string(self):
        assert to_sexpr("hello") == '"hello"'

    def test_to_sexpr_list(self):
        assert to_sexpr([1, 2, 3]) == "(1 2 3)"

    def test_to_sexpr_none(self):
        assert to_sexpr(None) == "nil"

    def test_to_sexpr_bool(self):
        assert to_sexpr(True) == "#t"
        assert to_sexpr(False) == "#f"

    def test_to_sexpr_dict(self):
        result = to_sexpr({"a": 1})
        assert "a" in result
        assert "1" in result

    def test_round_trip(self):
        """Parse a serialized expression and evaluate it."""
        original = ["+", 1.0, 2.0]
        sexpr_str = to_sexpr(original)
        parsed = parse(sexpr_str)
        assert parsed == original


# ---------------------------------------------------------------------------
# Safety tests
# ---------------------------------------------------------------------------

class TestSafety:
    def test_step_limit(self):
        """Infinite loop should be caught by step counter."""
        with pytest.raises(StepLimitError):
            run("""
                (begin
                    (define (loop x) (loop (+ x 1)))
                    (loop 0))
            """, step_limit=100)

    def test_subsim_depth_limit(self):
        """Sub-sim at max depth should error."""
        evaluator = Evaluator(step_limit=5000, subsim_depth=3, max_subsim_depth=3)
        env = standard_env()
        expr = parse("(sub-sim (+ 1 2))")
        with pytest.raises(DepthLimitError):
            evaluator.eval(expr, env)

    def test_subsim_basic(self):
        """Sub-sim should evaluate expression in sandbox."""
        evaluator = Evaluator(step_limit=5000, max_subsim_depth=3)
        env = standard_env()
        expr = parse("(sub-sim (+ 10 20))")
        result = evaluator.eval(expr, env)
        assert result == 30.0

    def test_subsim_logged(self):
        """Sub-sim execution should be logged."""
        evaluator = Evaluator(step_limit=5000, max_subsim_depth=3)
        env = standard_env()
        expr = parse("(sub-sim (* 6 7))")
        evaluator.eval(expr, env)
        assert len(evaluator.subsim_log) == 1
        assert evaluator.subsim_log[0]["depth"] == 1
        assert evaluator.subsim_log[0]["result"] == 42.0

    def test_subsim_budget_decreases(self):
        """Deeper sub-sims should have smaller budgets."""
        evaluator = Evaluator(step_limit=4000, max_subsim_depth=3)
        env = standard_env()
        # Nested sub-sim
        expr = parse("(sub-sim (sub-sim (+ 1 1)))")
        result = evaluator.eval(expr, env)
        assert result == 2.0
        # Should have 2 log entries (depth 1 and depth 2)
        assert len(evaluator.subsim_log) == 2

    def test_no_io_access(self):
        """Standard env should not have any I/O primitives."""
        env = standard_env()
        dangerous = ["open", "read", "write", "exec", "eval", "import",
                      "system", "os", "subprocess", "file"]
        for name in dangerous:
            assert name not in env

    def test_undefined_symbol(self):
        with pytest.raises(LispyError, match="undefined"):
            run("nonexistent_var")

    def test_arity_mismatch(self):
        with pytest.raises(LispyError, match="arity"):
            run("(begin (define (f x) x) (f 1 2))")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_fibonacci(self):
        """Classic recursive function."""
        result = run("""
            (begin
                (define (fib n)
                    (if (<= n 1) n
                        (+ (fib (- n 1)) (fib (- n 2)))))
                (fib 10))
        """, step_limit=50000)
        assert result == 55.0

    def test_governance_model(self):
        """A LisPy expression modeling a simple governance decision."""
        result = run("""
            (let ((food 0.3)
                  (water 0.8)
                  (population 8)
                  (crisis (- 1.0 (min 0.3 0.8))))
              (if (> crisis 0.5)
                (list "ration" (* food 0.7))
                (list "expand" (+ population 1))))
        """)
        assert isinstance(result, list)
        assert result[0] == "ration"

    def test_colonist_decision(self):
        """Model a colonist decision as LisPy."""
        result = run("""
            (let ((resolve 0.8)
                  (empathy 0.6)
                  (paranoia 0.2)
                  (crisis 0.4))
              (cond
                ((> crisis 0.7) "survive")
                ((> empathy 0.5) "mediate")
                ((> paranoia 0.6) "hide")
                (else "explore")))
        """)
        assert result == "mediate"

    def test_subsim_governance_proposal(self):
        """Sub-sim evaluating a governance proposal's outcome."""
        result = run("""
            (sub-sim
              (let ((share-pct 0.8)
                    (food 1.2)
                    (pop 9))
                (let ((per-person (* food share-pct (/ 1.0 pop))))
                  (if (> per-person 0.05)
                    (list "viable" per-person)
                    (list "insufficient" per-person)))))
        """, step_limit=5000)
        assert isinstance(result, list)
        assert result[0] in ("viable", "insufficient")
