"""test_lispy.py -- Tests for the minimal safe-eval LisPy interpreter.

Covers: tokenizer, parser, evaluator, all builtins, safety limits,
sub-sim depth tracking, error handling.
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lispy import (
    tokenize, parse, parse_all, evaluate, run, to_sexp,
    Symbol, Env, Budget, Procedure,
    default_env, LispyError, BudgetExhausted,
)


# ── Tokenizer ──────────────────────────────────────────────────────────

class TestTokenizer:
    def test_simple(self):
        assert tokenize("(+ 1 2)") == ['(', '+', '1', '2', ')']

    def test_nested(self):
        tokens = tokenize("(define x (+ 1 2))")
        assert tokens == ['(', 'define', 'x', '(', '+', '1', '2', ')', ')']

    def test_string(self):
        tokens = tokenize('(display "hello world")')
        assert '"hello world"' in tokens

    def test_whitespace(self):
        tokens = tokenize("  ( + \n 1  \t 2 )  ")
        assert tokens == ['(', '+', '1', '2', ')']

    def test_comment(self):
        tokens = tokenize("; this is a comment\n(+ 1 2)")
        assert tokens == ['(', '+', '1', '2', ')']

    def test_quote_shorthand(self):
        tokens = tokenize("'hello")
        assert tokens == ["'", 'hello']

    def test_empty(self):
        assert tokenize("") == []
        assert tokenize("   ") == []
        assert tokenize("; just a comment") == []


# ── Parser ─────────────────────────────────────────────────────────────

class TestParser:
    def test_number(self):
        assert parse("42") == 42

    def test_float(self):
        assert parse("3.14") == 3.14

    def test_symbol(self):
        result = parse("hello")
        assert isinstance(result, Symbol)
        assert result == "hello"

    def test_string(self):
        result = parse('"hello world"')
        assert result == "hello world"
        assert not isinstance(result, Symbol)

    def test_list(self):
        result = parse("(+ 1 2)")
        assert result == [Symbol('+'), 1, 2]

    def test_nested_list(self):
        result = parse("(if (> x 0) 1 -1)")
        assert len(result) == 4
        assert result[0] == Symbol('if')

    def test_quote(self):
        result = parse("'hello")
        assert result == [Symbol('quote'), Symbol('hello')]

    def test_bool_true(self):
        assert parse("#t") is True

    def test_bool_false(self):
        assert parse("#f") is False

    def test_empty_raises(self):
        with pytest.raises(LispyError, match="empty expression"):
            parse("")

    def test_unmatched_paren(self):
        with pytest.raises(LispyError, match="missing closing paren"):
            parse("(+ 1 2")

    def test_unexpected_close(self):
        with pytest.raises(LispyError):
            parse(")")

    def test_parse_all(self):
        results = parse_all("(+ 1 2) (* 3 4)")
        assert len(results) == 2
        assert results[0] == [Symbol('+'), 1, 2]
        assert results[1] == [Symbol('*'), 3, 4]


# ── Evaluator: Arithmetic ──────────────────────────────────────────────

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
        assert run("(/ 10 3)") == pytest.approx(10 / 3)

    def test_modulo(self):
        assert run("(% 10 3)") == 1

    def test_abs(self):
        assert run("(abs -5)") == 5

    def test_min_max(self):
        assert run("(min 3 1 4 1 5)") == 1
        assert run("(max 3 1 4 1 5)") == 5

    def test_sqrt(self):
        assert run("(sqrt 16)") == 4.0

    def test_expt(self):
        assert run("(expt 2 10)") == 1024

    def test_floor_ceil(self):
        assert run("(floor 3.7)") == 3
        assert run("(ceil 3.2)") == 4


# ── Evaluator: Comparison & Logic ──────────────────────────────────────

class TestComparison:
    def test_less(self):
        assert run("(< 1 2)") is True
        assert run("(< 2 1)") is False

    def test_greater(self):
        assert run("(> 3 2)") is True

    def test_equal(self):
        assert run("(= 5 5)") is True
        assert run('(= "a" "a")') is True

    def test_not_equal(self):
        assert run("(!= 1 2)") is True

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
        # Should not error because second expr is never reached
        assert run("(and #f (/ 1 0))") is False

    def test_or_short_circuit(self):
        assert run("(or #t (/ 1 0))") is True


# ── Evaluator: Conditionals ───────────────────────────────────────────

class TestConditionals:
    def test_if_true(self):
        assert run("(if #t 1 2)") == 1

    def test_if_false(self):
        assert run("(if #f 1 2)") == 2

    def test_if_no_else(self):
        assert run("(if #f 1)") is None

    def test_if_truthy(self):
        assert run("(if 1 'yes 'no)") == Symbol('yes')

    def test_if_zero_is_falsy(self):
        assert run("(if 0 'yes 'no)") == Symbol('no')

    def test_cond(self):
        code = """
        (cond
          ((= 1 2) 'nope)
          ((= 1 1) 'yes)
          (else 'default))
        """
        assert run(code) == Symbol('yes')

    def test_cond_else(self):
        code = """
        (cond
          ((= 1 2) 'nope)
          (else 'default))
        """
        assert run(code) == Symbol('default')


# ── Evaluator: Variables & Lambdas ────────────────────────────────────

class TestVariables:
    def test_define(self):
        assert run("(begin (define x 42) x)") == 42

    def test_set(self):
        assert run("(begin (define x 1) (set! x 2) x)") == 2

    def test_set_unbound(self):
        with pytest.raises(LispyError, match="unbound"):
            run("(set! x 1)")

    def test_let(self):
        assert run("(let ((x 5) (y 3)) (+ x y))") == 8

    def test_let_body_multiple(self):
        assert run("(let ((x 1)) (define y 2) (+ x y))") == 3

    def test_lambda(self):
        assert run("((lambda (x) (* x x)) 5)") == 25

    def test_closure(self):
        code = """
        (begin
          (define make-adder (lambda (n) (lambda (x) (+ n x))))
          (define add5 (make-adder 5))
          (add5 3))
        """
        assert run(code) == 8

    def test_recursion(self):
        code = """
        (begin
          (define factorial
            (lambda (n)
              (if (<= n 1) 1 (* n (factorial (- n 1))))))
          (factorial 5))
        """
        assert run(code) == 120

    def test_begin(self):
        assert run("(begin 1 2 3)") == 3


# ── Evaluator: List Operations ─────────────────────────────────────────

class TestLists:
    def test_list(self):
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

    def test_append(self):
        assert run("(append (list 1 2) (list 3 4))") == [1, 2, 3, 4]

    def test_reverse(self):
        assert run("(reverse (list 1 2 3))") == [3, 2, 1]

    def test_sort(self):
        assert run("(sort (list 3 1 2))") == [1, 2, 3]

    def test_quote(self):
        result = run("'(1 2 3)")
        assert result == [1, 2, 3]

    def test_empty_list(self):
        assert run("(list)") == []

    def test_null_check(self):
        assert run("(null? (list))") is True
        assert run("(null? nil)") is True
        assert run("(null? (list 1))") is False

    def test_car_empty_error(self):
        with pytest.raises(LispyError, match="car"):
            run("(car (list))")

    def test_nth_out_of_range(self):
        with pytest.raises(LispyError, match="index"):
            run("(nth (list 1 2) 5)")


# ── Evaluator: Dict/Assoc Operations ──────────────────────────────────

class TestDict:
    def test_dict(self):
        result = run("(dict 'a 1 'b 2)")
        assert isinstance(result, dict)
        assert result[Symbol('a')] == 1

    def test_get_dict(self):
        code = """
        (let ((d (dict 'x 42 'y 99)))
          (get d 'x))
        """
        assert run(code) == 42

    def test_get_assoc(self):
        code = """
        (let ((a (list (list 'name "Ares") (list 'age 30))))
          (get a 'name))
        """
        assert run(code) == "Ares"

    def test_get_missing(self):
        assert run("(get (dict 'a 1) 'z)") is None

    def test_put(self):
        code = """
        (let ((d (dict 'a 1)))
          (get (put d 'b 2) 'b))
        """
        assert run(code) == 2

    def test_keys(self):
        result = run("(keys (dict 'a 1 'b 2))")
        assert set(result) == {Symbol('a'), Symbol('b')}


# ── Evaluator: Type Checks ────────────────────────────────────────────

class TestTypes:
    def test_number(self):
        assert run("(number? 42)") is True
        assert run('(number? "hi")') is False

    def test_string(self):
        assert run('(string? "hi")') is True
        assert run("(string? 42)") is False

    def test_list(self):
        assert run("(list? (list 1 2))") is True
        assert run("(list? 42)") is False

    def test_symbol(self):
        assert run("(symbol? 'hello)") is True
        assert run("(symbol? 42)") is False


# ── Safety: Budget ─────────────────────────────────────────────────────

class TestBudget:
    def test_step_limit(self):
        budget = Budget(max_steps=10)
        with pytest.raises(BudgetExhausted, match="step budget"):
            run("(begin 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)", budget=budget)

    def test_recursion_limit(self):
        budget = Budget(max_recursion=10)
        code = """
        (begin
          (define f (lambda (n) (if (= n 0) 1 (f (- n 1)))))
          (f 100))
        """
        with pytest.raises(BudgetExhausted, match="recursion"):
            run(code, budget=budget)

    def test_default_budget_sufficient(self):
        # Default budget should handle normal programs
        result = run("(+ 1 2)")
        assert result == 3

    def test_budget_steps_tracked(self):
        budget = Budget()
        run("(+ 1 2)", budget=budget)
        assert budget.steps_used > 0

    def test_budget_monotonic(self):
        """Step budget only increases, never decreases."""
        budget = Budget()
        run("(+ 1 2)", budget=budget)
        steps1 = budget.steps_used
        run("(+ 3 4)", budget=budget)
        assert budget.steps_used > steps1


# ── Sub-sim ────────────────────────────────────────────────────────────

class TestSubSim:
    def test_basic_sub_sim(self):
        result = run("(sub-sim (+ 1 2))")
        assert result == 3

    def test_sub_sim_depth(self):
        budget = Budget(max_depth=2)
        env = default_env()
        # Depth 1
        result = run("(sub-sim (sub-sim (+ 1 2)))", env=env, budget=budget)
        assert result == 3

    def test_sub_sim_depth_exceeded(self):
        budget = Budget(max_depth=1)
        with pytest.raises(BudgetExhausted, match="depth"):
            run("(sub-sim (sub-sim (+ 1 2)))", budget=budget)

    def test_sub_sim_isolation(self):
        """Variables defined in sub-sim don't leak to parent."""
        code = """
        (begin
          (define x 1)
          (sub-sim (define x 999))
          x)
        """
        assert run(code) == 1

    def test_sub_sim_reads_parent(self):
        """Sub-sim can read parent variables."""
        code = """
        (begin
          (define x 42)
          (sub-sim x))
        """
        assert run(code) == 42

    def test_sub_sim_budget_shared(self):
        """Sub-sims share the step budget with parent."""
        budget = Budget(max_steps=50)
        env = default_env()
        try:
            run("(sub-sim (begin 1 2 3 4 5))", env=env, budget=budget)
        except BudgetExhausted:
            pass
        # Steps from sub-sim counted in budget
        assert budget.steps_used > 0


# ── to_sexp ────────────────────────────────────────────────────────────

class TestToSexp:
    def test_number(self):
        assert to_sexp(42) == "42"

    def test_string(self):
        assert to_sexp("hello") == '"hello"'

    def test_bool(self):
        assert to_sexp(True) == "#t"
        assert to_sexp(False) == "#f"

    def test_none(self):
        assert to_sexp(None) == "nil"

    def test_list(self):
        assert to_sexp([1, 2, 3]) == "(1 2 3)"

    def test_nested(self):
        assert to_sexp([Symbol('+'), 1, [Symbol('*'), 2, 3]]) == "(+ 1 (* 2 3))"

    def test_symbol(self):
        assert to_sexp(Symbol('hello')) == "hello"

    def test_dict(self):
        result = to_sexp({Symbol('a'): 1})
        assert 'dict' in result


# ── Edge cases ─────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_unbound_symbol(self):
        with pytest.raises(LispyError, match="unbound"):
            run("nonexistent")

    def test_not_callable(self):
        with pytest.raises(LispyError, match="not callable"):
            run("(42 1 2)")

    def test_define_non_symbol(self):
        with pytest.raises(LispyError, match="symbol"):
            run("(define 42 1)")

    def test_lambda_no_body(self):
        with pytest.raises(LispyError):
            run("(lambda (x))")

    def test_lambda_arity_mismatch(self):
        with pytest.raises(LispyError, match="arity"):
            run("((lambda (x y) (+ x y)) 1)")

    def test_nested_environments(self):
        code = """
        (let ((x 1))
          (let ((y 2))
            (let ((z 3))
              (+ x (+ y z)))))
        """
        assert run(code) == 6

    def test_display(self):
        result = run('(display "hello" 42)')
        assert "hello" in result
        assert "42" in result

    def test_pi(self):
        import math
        assert run("pi") == pytest.approx(math.pi)
