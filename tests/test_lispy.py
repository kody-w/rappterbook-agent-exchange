"""Tests for the LisPy sandboxed interpreter."""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lispy import (
    parse, parse_all, tokenize, evaluate, default_env, run,
    Env, Lambda,
    ParseError, EvalError, StepLimitError, DepthLimitError,
    RecursionLimitError,
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_simple_tokens(self):
        assert tokenize("(+ 1 2)") == ["(", "+", "1", "2", ")"]

    def test_string_token(self):
        tokens = tokenize('(str "hello world")')
        assert '"hello world"' in tokens

    def test_comments_stripped(self):
        tokens = tokenize("; this is a comment\n(+ 1 2)")
        assert ";" not in "".join(tokens)
        assert "+" in tokens

    def test_nested(self):
        tokens = tokenize("(if (> x 3) (+ x 1) 0)")
        assert tokens.count("(") == 3
        assert tokens.count(")") == 3

    def test_quote_shorthand(self):
        tokens = tokenize("'(1 2 3)")
        assert tokens[0] == "'"


class TestParser:
    def test_integer(self):
        assert parse("42") == 42

    def test_float(self):
        assert parse("3.14") == 3.14

    def test_string(self):
        # Strings keep quotes at parse level, eval strips them
        assert parse('"hello"') == '"hello"'

    def test_boolean_true(self):
        assert parse("#t") is True

    def test_boolean_false(self):
        assert parse("#f") is False

    def test_nil(self):
        assert parse("nil") is None

    def test_symbol(self):
        assert parse("foo") == "foo"

    def test_simple_list(self):
        assert parse("(+ 1 2)") == ["+", 1, 2]

    def test_nested_list(self):
        result = parse("(if (> x 3) (+ x 1))")
        assert result == ["if", [">", "x", 3], ["+", "x", 1]]

    def test_quote(self):
        result = parse("'(1 2 3)")
        assert result == ["quote", [1, 2, 3]]

    def test_empty_raises(self):
        with pytest.raises(ParseError):
            parse("")

    def test_unmatched_paren(self):
        with pytest.raises(ParseError):
            parse("(+ 1 2")

    def test_unexpected_close(self):
        with pytest.raises(ParseError):
            parse(")")

    def test_parse_all_multiple(self):
        results = parse_all("(+ 1 2) (* 3 4)")
        assert results == [["+", 1, 2], ["*", 3, 4]]


# ---------------------------------------------------------------------------
# Evaluation - arithmetic
# ---------------------------------------------------------------------------

class TestArithmetic:
    def test_add(self):
        assert run("(+ 1 2)") == 3

    def test_add_multiple(self):
        assert run("(+ 1 2 3 4)") == 10

    def test_subtract(self):
        assert run("(- 10 3)") == 7

    def test_negate(self):
        assert run("(- 5)") == -5

    def test_multiply(self):
        assert run("(* 3 4)") == 12

    def test_divide(self):
        assert run("(/ 10 3)") == pytest.approx(3.333, abs=0.01)

    def test_divide_by_zero(self):
        with pytest.raises(EvalError, match="division by zero"):
            run("(/ 1 0)")

    def test_modulo(self):
        assert run("(mod 10 3)") == 1

    def test_nested_arithmetic(self):
        assert run("(+ (* 2 3) (- 10 4))") == 12

    def test_abs(self):
        assert run("(abs -5)") == 5

    def test_round(self):
        assert run("(round 3.7)") == 4


# ---------------------------------------------------------------------------
# Evaluation - comparison & logic
# ---------------------------------------------------------------------------

class TestComparison:
    def test_equal(self):
        assert run("(= 3 3)") is True

    def test_not_equal(self):
        assert run("(= 3 4)") is False

    def test_less_than(self):
        assert run("(< 2 5)") is True

    def test_greater_than(self):
        assert run("(> 5 2)") is True

    def test_lte(self):
        assert run("(<= 3 3)") is True

    def test_gte(self):
        assert run("(>= 3 3)") is True

    def test_not(self):
        assert run("(not #f)") is True

    def test_and_short_circuit(self):
        """and should short-circuit on false."""
        # If and evaluates both, the second would cause an error
        assert run("(and #f (/ 1 0))") is False

    def test_and_all_true(self):
        assert run("(and 1 2 3)") == 3

    def test_or_short_circuit(self):
        """or should short-circuit on truth."""
        assert run("(or 42 (/ 1 0))") == 42

    def test_or_all_false(self):
        assert run("(or #f #f #f)") is False


# ---------------------------------------------------------------------------
# Evaluation - control flow
# ---------------------------------------------------------------------------

class TestControlFlow:
    def test_if_true(self):
        assert run("(if #t 1 2)") == 1

    def test_if_false(self):
        assert run("(if #f 1 2)") == 2

    def test_if_no_else(self):
        assert run("(if #f 1)") is None

    def test_cond_first_match(self):
        assert run("(cond ((= 1 1) 42) ((= 1 2) 99))") == 42

    def test_cond_else(self):
        assert run("(cond ((= 1 2) 42) (else 99))") == 99

    def test_cond_no_match(self):
        assert run("(cond ((= 1 2) 42))") is None

    def test_begin(self):
        assert run("(begin 1 2 3)") == 3


# ---------------------------------------------------------------------------
# Evaluation - bindings
# ---------------------------------------------------------------------------

class TestBindings:
    def test_define(self):
        assert run("(begin (define x 10) x)") == 10

    def test_let(self):
        assert run("(let ((x 5) (y 3)) (+ x y))") == 8

    def test_let_body_multiple(self):
        assert run("(let ((x 1)) x (+ x 1))") == 2

    def test_set_bang(self):
        assert run("(begin (define x 1) (set! x 42) x)") == 42

    def test_set_unbound_raises(self):
        with pytest.raises(EvalError, match="unbound"):
            run("(set! nonexistent 1)")

    def test_unbound_symbol(self):
        with pytest.raises(EvalError, match="unbound"):
            run("undefined_var")


# ---------------------------------------------------------------------------
# Evaluation - lambda
# ---------------------------------------------------------------------------

class TestLambda:
    def test_lambda_call(self):
        assert run("((lambda (x) (* x x)) 5)") == 25

    def test_lambda_closure(self):
        result = run("""
            (let ((make-adder (lambda (n) (lambda (x) (+ n x)))))
              (let ((add5 (make-adder 5)))
                (add5 10)))
        """)
        assert result == 15

    def test_lambda_wrong_arity(self):
        with pytest.raises(EvalError, match="expects"):
            run("((lambda (x y) (+ x y)) 1)")

    def test_define_lambda(self):
        result = run("""
            (begin
              (define square (lambda (x) (* x x)))
              (square 7))
        """)
        assert result == 49


# ---------------------------------------------------------------------------
# Evaluation - lists
# ---------------------------------------------------------------------------

class TestLists:
    def test_list_create(self):
        assert run("(list 1 2 3)") == [1, 2, 3]

    def test_car(self):
        assert run("(car (list 1 2 3))") == 1

    def test_cdr(self):
        assert run("(cdr (list 1 2 3))") == [2, 3]

    def test_cons(self):
        assert run("(cons 0 (list 1 2))") == [0, 1, 2]

    def test_length(self):
        assert run("(length (list 1 2 3))") == 3

    def test_nth(self):
        assert run("(nth (list 10 20 30) 1)") == 20

    def test_nth_out_of_range(self):
        with pytest.raises(EvalError, match="out of range"):
            run("(nth (list 1) 5)")

    def test_append(self):
        assert run("(append (list 1 2) (list 3 4))") == [1, 2, 3, 4]

    def test_map(self):
        result = run("(map (lambda (x) (* x 2)) (list 1 2 3))")
        assert result == [2, 4, 6]

    def test_filter(self):
        result = run("(filter (lambda (x) (> x 2)) (list 1 2 3 4))")
        assert result == [3, 4]

    def test_quote(self):
        result = run("'(1 2 3)")
        assert result == [1, 2, 3]

    def test_type_predicates(self):
        assert run("(list? (list 1))") is True
        assert run("(number? 42)") is True
        assert run('(string? "hi")') is True
        assert run("(nil? nil)") is True


# ---------------------------------------------------------------------------
# Evaluation - hash maps
# ---------------------------------------------------------------------------

class TestHashMaps:
    def test_make_hash(self):
        result = run('(make-hash "a" 1 "b" 2)')
        assert result == {"a": 1, "b": 2}

    def test_hash_get(self):
        result = run('(hash-get (make-hash "x" 42) "x")')
        assert result == 42

    def test_hash_get_missing(self):
        result = run('(hash-get (make-hash "x" 42) "y")')
        assert result is None

    def test_hash_set(self):
        result = run('(hash-set (make-hash "a" 1) "b" 2)')
        assert result == {"a": 1, "b": 2}

    def test_hash_keys(self):
        result = run('(hash-keys (make-hash "a" 1 "b" 2))')
        assert set(result) == {"a", "b"}


# ---------------------------------------------------------------------------
# Evaluation - do loop
# ---------------------------------------------------------------------------

class TestDoLoop:
    def test_do_basic(self):
        result = run("(begin (define x 0) (do 5 (set! x (+ x 1))) x)")
        assert result == 5

    def test_do_zero(self):
        result = run("(begin (define x 99) (do 0 (set! x 0)) x)")
        assert result == 99


# ---------------------------------------------------------------------------
# Safety - step limits
# ---------------------------------------------------------------------------

class TestStepLimit:
    def test_step_limit_exceeded(self):
        env = default_env(max_steps=50)
        with pytest.raises(StepLimitError):
            run("(do 100 (+ 1 1))", env)

    def test_step_limit_ok(self):
        env = default_env(max_steps=500)
        result = run("(do 10 (+ 1 1))", env)
        assert result == 2


# ---------------------------------------------------------------------------
# Safety - recursion limits
# ---------------------------------------------------------------------------

class TestRecursionLimit:
    def test_deep_recursion_caught(self):
        env = default_env(max_recursion=20)
        with pytest.raises(RecursionLimitError):
            run("""
                (begin
                  (define countdown (lambda (n)
                    (if (= n 0) 0 (countdown (- n 1)))))
                  (countdown 100))
            """, env)


# ---------------------------------------------------------------------------
# Sub-simulation
# ---------------------------------------------------------------------------

class TestSubSim:
    def test_sub_sim_basic(self):
        result = run("(sub-sim 2 (+ 10 20))")
        assert result == 30

    def test_sub_sim_depth_2(self):
        """Depth 3 -> sub-sim at depth 2 -> sub-sim at depth 1."""
        result = run("(sub-sim 2 (sub-sim 1 (+ 1 1)))")
        assert result == 2

    def test_sub_sim_depth_exceeded(self):
        with pytest.raises(DepthLimitError):
            run("(sub-sim 3 (+ 1 1))")  # budget is 3, requesting 3

    def test_sub_sim_inherits_bindings(self):
        """Sub-sim gets a copy of parent bindings."""
        result = run("""
            (begin
              (define x 42)
              (sub-sim 2 x))
        """)
        assert result == 42

    def test_sub_sim_isolation(self):
        """Sub-sim mutations don't affect parent."""
        result = run("""
            (begin
              (define x 1)
              (sub-sim 2 (begin (define x 999) x))
              x)
        """)
        assert result == 1

    def test_sub_sim_fresh_step_counter(self):
        """Sub-sim has its own step budget."""
        env = default_env(max_steps=500)
        # Use up most of parent's steps, then sub-sim should still work
        result = run("""
            (begin
              (do 30 (+ 1 1))
              (sub-sim 2 (+ 1 1)))
        """, env)
        assert result == 2

    def test_triple_nesting(self):
        """Depth 3 -> 2 -> 1 -> evaluates."""
        result = run("""
            (sub-sim 2
              (sub-sim 1
                (sub-sim 0
                  (* 6 7))))
        """)
        assert result == 42

    def test_sub_sim_at_depth_0_cannot_nest(self):
        """At depth 0, no further sub-sims allowed."""
        with pytest.raises(DepthLimitError):
            run("(sub-sim 0 (sub-sim 0 1))")


# ---------------------------------------------------------------------------
# Integration: colonist-like LisPy programs
# ---------------------------------------------------------------------------

class TestColonistPrograms:
    def test_governance_prediction(self):
        """Colonist runs a sub-sim to predict if rationing is needed."""
        result = run("""
            (sub-sim 2
              (let ((food 60) (water 80) (pop 8))
                (if (< food (* pop 10))
                  (list "recommend" "ration")
                  (list "recommend" "expand"))))
        """)
        # food=60 < pop*10=80, so ration
        assert result == ["recommend", "ration"]

    def test_work_action(self):
        """Colonist generates a work action."""
        result = run("""
            (let ((skill 45))
              (list "work" "farm" (/ skill 8.0)))
        """)
        assert result[0] == "work"
        assert result[1] == "farm"
        assert result[2] == pytest.approx(5.625)

    def test_vote_action(self):
        """Colonist votes on a proposal."""
        result = run('(list "vote" "prop-1" #t)')
        assert result == ["vote", "prop-1", True]

    def test_complex_decision(self):
        """Multi-step decision with sub-sim evidence."""
        result = run("""
            (let ((food 200) (pop 10) (my-paranoia 70))
              (if (> my-paranoia 60)
                (sub-sim 2
                  (let ((projected-food (- food (* pop 12))))
                    (if (< projected-food 0)
                      (list "action" "hoard" 20)
                      (list "action" "work" "farm"))))
                (list "action" "work" "terraform")))
        """)
        assert result[0] == "action"
        # With food=200, pop=10: projected = 200 - 120 = 80 > 0
        assert result[1] == "work"
