"""Tests for the LisPy safe s-expression interpreter."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.lispy import (
    Budget, Env, Lambda, LispyError, NIL, Symbol,
    format_sexpr, lispy_eval, make_env, parse, run, tokenise,
)


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

class TestTokeniser:
    def test_empty(self):
        assert tokenise("") == []

    def test_parens(self):
        assert tokenise("()") == ["(", ")"]

    def test_nested(self):
        assert tokenise("(+ 1 2)") == ["(", "+", "1", "2", ")"]

    def test_string(self):
        tokens = tokenise('(str "hello world")')
        assert '"hello world"' in tokens

    def test_comment(self):
        tokens = tokenise("; comment\n(+ 1 2)")
        assert tokens == ["(", "+", "1", "2", ")"]

    def test_quote_sugar(self):
        tokens = tokenise("'(1 2 3)")
        assert tokens[0] == "'"

    def test_unterminated_string(self):
        with pytest.raises(LispyError, match="unterminated"):
            tokenise('"hello')


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TestParser:
    def test_integer(self):
        assert parse("42") == [42]

    def test_float(self):
        assert parse("3.14") == [3.14]

    def test_string(self):
        assert parse('"hello"') == ["hello"]

    def test_symbol(self):
        result = parse("foo")
        assert len(result) == 1
        assert isinstance(result[0], Symbol)
        assert result[0] == "foo"

    def test_boolean(self):
        assert parse("#t") == [True]
        assert parse("#f") == [False]

    def test_nil(self):
        assert parse("nil") == [NIL]

    def test_list(self):
        result = parse("(+ 1 2)")
        assert result == [[Symbol("+"), 1, 2]]

    def test_nested_list(self):
        result = parse("(if (> x 0) x (- 0 x))")
        assert len(result) == 1
        assert len(result[0]) == 4  # if, test, then, else

    def test_quote(self):
        result = parse("'(1 2 3)")
        assert result == [[Symbol("quote"), [1, 2, 3]]]

    def test_multiple_expressions(self):
        result = parse("1 2 3")
        assert result == [1, 2, 3]

    def test_missing_close_paren(self):
        with pytest.raises(LispyError, match="missing closing"):
            parse("(+ 1 2")

    def test_unexpected_close_paren(self):
        with pytest.raises(LispyError, match="unexpected closing"):
            parse(")")

    def test_empty_list(self):
        assert parse("()") == [[]]


# ---------------------------------------------------------------------------
# Evaluator — Basics
# ---------------------------------------------------------------------------

class TestEvalBasics:
    def test_integer(self):
        assert run("42") == 42

    def test_float(self):
        assert run("3.14") == 3.14

    def test_string(self):
        assert run('"hello"') == "hello"

    def test_boolean(self):
        assert run("#t") is True
        assert run("#f") is False

    def test_nil(self):
        assert run("nil") is NIL

    def test_empty_list(self):
        assert run("()") is NIL


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

class TestArithmetic:
    def test_addition(self):
        assert run("(+ 1 2)") == 3

    def test_addition_multiple(self):
        assert run("(+ 1 2 3 4)") == 10

    def test_subtraction(self):
        assert run("(- 10 3)") == 7

    def test_negation(self):
        assert run("(- 5)") == -5

    def test_multiplication(self):
        assert run("(* 3 4)") == 12

    def test_division(self):
        assert run("(/ 10 3)") == pytest.approx(3.333, rel=0.01)

    def test_division_by_zero(self):
        assert run("(/ 10 0)") == 0

    def test_modulo(self):
        assert run("(mod 10 3)") == 1

    def test_abs(self):
        assert run("(abs -5)") == 5

    def test_sqrt(self):
        assert run("(sqrt 9)") == 3.0

    def test_nested_arithmetic(self):
        assert run("(+ (* 2 3) (- 10 4))") == 12


# ---------------------------------------------------------------------------
# Comparison and Logic
# ---------------------------------------------------------------------------

class TestComparison:
    def test_equal(self):
        assert run("(= 1 1)") is True
        assert run("(= 1 2)") is False

    def test_less_than(self):
        assert run("(< 1 2)") is True
        assert run("(< 2 1)") is False

    def test_greater_than(self):
        assert run("(> 2 1)") is True

    def test_not(self):
        assert run("(not #t)") is False
        assert run("(not #f)") is True

    def test_and_short_circuit(self):
        # and should not evaluate the second arg if first is false
        assert run("(and #f (/ 1 0))") is False

    def test_and_truthy(self):
        assert run("(and 1 2 3)") == 3

    def test_or_short_circuit(self):
        assert run("(or 5 (/ 1 0))") == 5

    def test_or_falsy(self):
        assert run("(or #f nil)") is NIL


# ---------------------------------------------------------------------------
# Special Forms
# ---------------------------------------------------------------------------

class TestSpecialForms:
    def test_if_true(self):
        assert run("(if #t 1 2)") == 1

    def test_if_false(self):
        assert run("(if #f 1 2)") == 2

    def test_if_no_else(self):
        assert run("(if #f 1)") is NIL

    def test_define_variable(self):
        assert run("(begin (define x 42) x)") == 42

    def test_define_function(self):
        assert run("(begin (define (add a b) (+ a b)) (add 3 4))") == 7

    def test_lambda(self):
        assert run("((lambda (x) (* x x)) 5)") == 25

    def test_closure(self):
        result = run("""
        (begin
          (define (make-adder n) (lambda (x) (+ x n)))
          (define add5 (make-adder 5))
          (add5 10))
        """)
        assert result == 15

    def test_begin(self):
        assert run("(begin 1 2 3)") == 3

    def test_let(self):
        assert run("(let ((x 5) (y 3)) (+ x y))") == 8

    def test_let_isolation(self):
        """Let bindings should not leak to outer scope."""
        with pytest.raises(LispyError, match="undefined"):
            run("(begin (let ((x 5)) x) x)")

    def test_quote(self):
        result = run("'(1 2 3)")
        assert result == [1, 2, 3]

    def test_quote_symbol(self):
        result = run("'foo")
        assert isinstance(result, Symbol)
        assert result == "foo"


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------

class TestLists:
    def test_list(self):
        assert run("(list 1 2 3)") == [1, 2, 3]

    def test_car(self):
        assert run("(car (list 1 2 3))") == 1

    def test_cdr(self):
        assert run("(cdr (list 1 2 3))") == [2, 3]

    def test_cons(self):
        assert run("(cons 1 (list 2 3))") == [1, 2, 3]

    def test_length(self):
        assert run("(length (list 1 2 3))") == 3

    def test_nth(self):
        assert run("(nth (list 10 20 30) 1)") == 20

    def test_nth_out_of_bounds(self):
        assert run("(nth (list 1 2) 5)") is NIL

    def test_append(self):
        assert run("(append (list 1 2) (list 3 4))") == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Higher-order functions
# ---------------------------------------------------------------------------

class TestHigherOrder:
    def test_map(self):
        result = run("(map (lambda (x) (* x 2)) (list 1 2 3))")
        assert result == [2, 4, 6]

    def test_filter(self):
        result = run("(filter (lambda (x) (> x 2)) (list 1 2 3 4 5))")
        assert result == [3, 4, 5]

    def test_reduce(self):
        result = run("(reduce (lambda (acc x) (+ acc x)) (list 1 2 3 4) 0)")
        assert result == 10


# ---------------------------------------------------------------------------
# Sub-simulation
# ---------------------------------------------------------------------------

class TestSubSim:
    def test_basic_subsim(self):
        result = run("(sub-sim (+ 1 2))")
        assert result == 3

    def test_subsim_depth_1(self):
        """Sub-sim inside sub-sim (depth 2)."""
        result = run("(sub-sim (sub-sim (+ 10 20)))")
        assert result == 30

    def test_subsim_depth_3(self):
        """Maximum depth: 3 levels of nesting."""
        result = run("(sub-sim (sub-sim (sub-sim (* 5 5))))")
        assert result == 25

    def test_subsim_depth_exceeded(self):
        """Depth 4 should fail."""
        with pytest.raises(LispyError, match="depth"):
            run("(sub-sim (sub-sim (sub-sim (sub-sim 1))))")

    def test_subsim_isolation(self):
        """Child env defines should not leak to parent."""
        result = run("""
        (begin
          (define x 10)
          (sub-sim (begin (define y 20) (+ x y)))
          x)
        """)
        assert result == 10
        # y should not be visible
        with pytest.raises(LispyError, match="undefined"):
            run("""
            (begin
              (sub-sim (define y 42))
              y)
            """)

    def test_subsim_shared_budget(self):
        """Budget is shared across parent and child evals."""
        budget = Budget(remaining=100, max_depth=3)
        env = make_env()
        run("(sub-sim (+ 1 (sub-sim (+ 2 3))))", env=env, budget=budget)
        assert budget.remaining < 100  # steps were consumed

    def test_subsim_budget_exhaustion(self):
        """A tight budget should prevent deep sub-sim work."""
        budget = Budget(remaining=10, max_depth=3)
        env = make_env()
        with pytest.raises(LispyError, match="budget"):
            run("(sub-sim (sub-sim (sub-sim (begin (+ 1 2) (+ 3 4) (+ 5 6) (+ 7 8) (+ 9 10)))))",
                env=env, budget=budget)

    def test_subsim_depth_restores_on_error(self):
        """Current depth restores even if sub-sim errors."""
        budget = Budget(remaining=5000, max_depth=3)
        env = make_env()
        try:
            run("(sub-sim (/ 1 undefined-var))", env=env, budget=budget)
        except LispyError:
            pass
        assert budget.current_depth == 0


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

class TestBudget:
    def test_budget_exhaustion(self):
        """Infinite recursion should be caught by budget."""
        with pytest.raises(LispyError, match="budget"):
            run("""
            (begin
              (define (loop n) (loop (+ n 1)))
              (loop 0))
            """, budget=Budget(remaining=1000))

    def test_budget_charges_builtins(self):
        """Built-in function calls should also charge budget."""
        budget = Budget(remaining=50000)
        run("(+ 1 2)", budget=budget)
        assert budget.remaining < 50000


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------

class TestFormat:
    def test_format_int(self):
        assert format_sexpr(42) == "42"

    def test_format_string(self):
        assert format_sexpr("hello") == '"hello"'

    def test_format_bool(self):
        assert format_sexpr(True) == "#t"
        assert format_sexpr(False) == "#f"

    def test_format_nil(self):
        assert format_sexpr(NIL) == "()"

    def test_format_list(self):
        assert format_sexpr([1, 2, 3]) == "(1 2 3)"

    def test_format_nested_list(self):
        assert format_sexpr([[1, 2], [3, 4]]) == "((1 2) (3 4))"

    def test_format_symbol(self):
        assert format_sexpr(Symbol("foo")) == "foo"

    def test_format_lambda(self):
        lam = Lambda(params=["x"], body=[Symbol("+"), Symbol("x"), 1],
                     env=Env())
        assert "lambda" in format_sexpr(lam)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_deeply_nested_arithmetic(self):
        result = run("(+ (+ (+ (+ 1 1) 1) 1) 1)")
        assert result == 5

    def test_empty_begin(self):
        assert run("(begin)") is NIL

    def test_define_overwrites(self):
        assert run("(begin (define x 1) (define x 2) x)") == 2

    def test_function_wrong_args(self):
        with pytest.raises(LispyError, match="expects"):
            run("(begin (define (f x) x) (f 1 2))")

    def test_call_non_function(self):
        with pytest.raises(LispyError, match="cannot call"):
            run("(42 1 2)")

    def test_undefined_symbol(self):
        with pytest.raises(LispyError, match="undefined"):
            run("undefined-var")

    def test_recursive_function(self):
        result = run("""
        (begin
          (define (factorial n) (if (<= n 1) 1 (* n (factorial (- n 1)))))
          (factorial 5))
        """)
        assert result == 120

    def test_mutual_recursion_budget(self):
        """Mutual recursion should eventually exhaust budget."""
        with pytest.raises(LispyError, match="budget"):
            run("""
            (begin
              (define (even? n) (if (= n 0) #t (odd? (- n 1))))
              (define (odd? n) (if (= n 0) #f (even? (- n 1))))
              (even? 10000))
            """, budget=Budget(remaining=500))

    def test_string_concat(self):
        assert run('(str "hello" " " "world")') == "hello world"

    def test_type_checks(self):
        assert run("(number? 42)") is True
        assert run('(string? "hi")') is True
        assert run("(list? (list 1))") is True
        assert run("(nil? nil)") is True
        assert run("(symbol? 'foo)") is True

    def test_assoc(self):
        result = run('(assoc "b" (list (list "a" 1) (list "b" 2) (list "c" 3)))')
        assert result == 2

    def test_assoc_missing(self):
        result = run('(assoc "z" (list (list "a" 1)))')
        assert result is NIL


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self):
        """Identical programs produce identical results."""
        prog = "(begin (define x (+ 1 2)) (define y (* x 3)) (list x y))"
        assert run(prog) == run(prog)

    def test_make_env_extras(self):
        env = make_env(my_val=42, colony_size=10)
        assert run("my-val", env=env) == 42
        assert run("colony-size", env=env) == 10
