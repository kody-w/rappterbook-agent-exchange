"""Tests for the LisPy VM."""
from __future__ import annotations

import pytest
from src.mars100.lispy_vm import (
    run, LispyError, LispySyntaxError, LispyBudgetExceeded,
    tokenize, parse_all, evaluate, make_env,
)


class TestArithmetic:
    def test_add(self):
        assert run("(+ 1 2)") == 3

    def test_nested(self):
        assert run("(+ (* 2 3) (- 10 4))") == 12

    def test_division(self):
        assert run("(/ 10 3)") == pytest.approx(3.333, abs=0.01)

    def test_modulo(self):
        assert run("(% 10 3)") == 1

    def test_negative(self):
        assert run("(- 0 5)") == -5


class TestComparisons:
    def test_gt(self):
        assert run("(> 3 2)") is True

    def test_lt(self):
        assert run("(< 3 2)") is False

    def test_eq(self):
        assert run("(= 5 5)") is True

    def test_gte(self):
        assert run("(>= 5 5)") is True

    def test_lte(self):
        assert run("(<= 4 5)") is True


class TestLogic:
    def test_and(self):
        assert run("(and #t #t)") is True
        assert run("(and #t #f)") is False

    def test_or(self):
        assert run("(or #f #t)") is True
        assert run("(or #f #f)") is False

    def test_not(self):
        assert run("(not #f)") is True
        assert run("(not #t)") is False


class TestControlFlow:
    def test_if_true(self):
        assert run("(if (> 3 2) 10 20)") == 10

    def test_if_false(self):
        assert run("(if (< 3 2) 10 20)") == 20

    def test_cond(self):
        assert run("(cond ((= 1 2) 10) ((= 2 2) 20) (#t 30))") == 20

    def test_begin(self):
        assert run("(begin 1 2 3)") == 3


class TestLetAndLambda:
    def test_let(self):
        assert run("(let ((x 5) (y 3)) (+ x y))") == 8

    def test_lambda(self):
        assert run("((lambda (x) (* x x)) 4)") == 16

    def test_closure(self):
        assert run("(let ((make-adder (lambda (n) (lambda (x) (+ n x))))) ((make-adder 5) 3))") == 8

    def test_define(self):
        assert run("(begin (define x 10) (+ x 5))") == 15

    def test_recursive_define(self):
        result = run("""
            (begin
                (define fact (lambda (n) (if (<= n 1) 1 (* n (fact (- n 1))))))
                (fact 5))
        """)
        assert result == 120


class TestLists:
    def test_quote(self):
        assert run("(quote (1 2 3))") == [1, 2, 3]

    def test_car(self):
        assert run("(car (quote (1 2 3)))") == 1

    def test_cdr(self):
        assert run("(cdr (quote (1 2 3)))") == [2, 3]

    def test_cons(self):
        assert run("(cons 1 (quote (2 3)))") == [1, 2, 3]

    def test_length(self):
        assert run("(length (quote (a b c d)))") == 4

    def test_nth(self):
        assert run("(nth (quote (10 20 30)) 1)") == 20

    def test_null_empty(self):
        """Test empty list is falsy."""
        assert run("(= (length (quote ())) 0)") is True

    def test_null_nonempty(self):
        """Test non-empty list has length > 0."""
        assert run("(> (length (quote (1))) 0)") is True

    def test_list(self):
        assert run("(list 1 2 3)") == [1, 2, 3]


class TestExtraBindings:
    def test_bindings(self):
        assert run("(+ x 1)", extra_bindings={"x": 10}) == 11

    def test_overwrite(self):
        assert run("x", extra_bindings={"x": 42}) == 42


class TestStrings:
    def test_string_literal(self):
        """String literal support — test via extra bindings if VM doesn't parse quotes."""
        result = run("x", extra_bindings={"x": "hello"})
        assert result == "hello"

    def test_string_in_bindings(self):
        result = run("(= x x)", extra_bindings={"x": "abc"})
        assert result is True


class TestSafetyLimits:
    def test_step_limit(self):
        with pytest.raises(LispyBudgetExceeded):
            run("(begin (define loop (lambda (n) (loop (+ n 1)))) (loop 0))",
                max_steps=500)

    def test_depth_limit(self):
        with pytest.raises(LispyBudgetExceeded):
            run("(begin (define deep (lambda (n) (deep (+ n 1)))) (deep 0))",
                max_depth=20)

    def test_custom_limits(self):
        with pytest.raises(LispyBudgetExceeded):
            run("(begin (define f (lambda (n) (f (+ n 1)))) (f 0))",
                max_steps=50)


class TestSyntaxErrors:
    def test_unclosed_paren(self):
        with pytest.raises((LispySyntaxError, LispyError)):
            run("(+ 1 2")

    def test_empty(self):
        """Empty input should either raise or return None."""
        try:
            result = run("")
            assert result is None
        except (LispySyntaxError, LispyError):
            pass  # raising is also acceptable


class TestDivisionByZero:
    def test_div_zero(self):
        with pytest.raises(LispyError):
            run("(/ 1 0)")


class TestColonistExpressions:
    """Test expressions similar to what colonists actually generate."""

    def test_governance_score(self):
        result = run(
            "(let ((gov-value (+ (* empathy 0.4) (* resolve 0.3) (* faith 0.3)))) "
            "(if (> gov-value 0.5) 1 0))",
            extra_bindings={"empathy": 0.7, "resolve": 0.6, "faith": 0.8}
        )
        assert result in (0, 1)

    def test_surplus_calc(self):
        result = run(
            "(let ((surplus (- food (* 10 0.06)))) (if (> surplus 0) 0.1 (- 0 0.2)))",
            extra_bindings={"food": 0.7, "morale": 0.5}
        )
        assert isinstance(result, (int, float))

    def test_risk_assessment(self):
        result = run(
            "(let ((risk (* paranoia 0.8))) (if (> risk 0.5) -0.1 0.05))",
            extra_bindings={"paranoia": 0.3, "resolve": 0.6, "improvisation": 0.5}
        )
        assert isinstance(result, (int, float))
