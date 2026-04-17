"""Tests for the LisPy evaluator."""
from __future__ import annotations

import pytest
from src.lispy import (
    Evaluator, Env, Lambda, LispyError, LispySandboxError, Str,
    make_standard_env, parse, to_sexp, tokenize,
)


class TestTokenizer:
    def test_simple(self) -> None:
        assert tokenize("(+ 1 2)") == ["(", "+", "1", "2", ")"]

    def test_string(self) -> None:
        tokens = tokenize('(list "hello" "world")')
        assert '"hello"' in tokens
        assert '"world"' in tokens

    def test_nested(self) -> None:
        tokens = tokenize("(+ (* 2 3) 4)")
        assert tokens == ["(", "+", "(", "*", "2", "3", ")", "4", ")"]

    def test_comment(self) -> None:
        tokens = tokenize("; this is a comment\n(+ 1 2)")
        assert tokens == ["(", "+", "1", "2", ")"]

    def test_quote(self) -> None:
        tokens = tokenize("'(1 2 3)")
        assert tokens[0] == "'"


class TestParser:
    def test_number(self) -> None:
        assert parse("42") == 42

    def test_float(self) -> None:
        assert parse("3.14") == 3.14

    def test_string(self) -> None:
        result = parse('"hello"')
        assert isinstance(result, Str) and result.value == "hello"

    def test_symbol(self) -> None:
        assert parse("foo") == "foo"

    def test_list(self) -> None:
        assert parse("(+ 1 2)") == ["+", 1, 2]

    def test_nested(self) -> None:
        assert parse("(+ (* 2 3) 4)") == ["+", ["*", 2, 3], 4]

    def test_boolean_true(self) -> None:
        assert parse("#t") is True

    def test_boolean_false(self) -> None:
        assert parse("#f") is False

    def test_nil(self) -> None:
        assert parse("nil") is None

    def test_quote_shorthand(self) -> None:
        assert parse("'(1 2)") == ["quote", [1, 2]]

    def test_empty_raises(self) -> None:
        with pytest.raises(LispyError):
            parse("")


class TestToSexp:
    def test_number(self) -> None:
        assert to_sexp(42) == "42"

    def test_string(self) -> None:
        assert to_sexp("hello") == "hello"

    def test_list(self) -> None:
        assert to_sexp(["+", 1, 2]) == "(+ 1 2)"

    def test_none(self) -> None:
        assert to_sexp(None) == "nil"

    def test_bool(self) -> None:
        assert to_sexp(True) == "#t"
        assert to_sexp(False) == "#f"


class TestArithmetic:
    def test_add(self) -> None:
        ev = Evaluator()
        assert ev.eval(parse("(+ 1 2)"), make_standard_env()) == 3

    def test_subtract(self) -> None:
        ev = Evaluator()
        assert ev.eval(parse("(- 10 3)"), make_standard_env()) == 7

    def test_multiply(self) -> None:
        ev = Evaluator()
        assert ev.eval(parse("(* 4 5)"), make_standard_env()) == 20

    def test_divide(self) -> None:
        ev = Evaluator()
        assert ev.eval(parse("(/ 10 2)"), make_standard_env()) == 5.0

    def test_divide_by_zero(self) -> None:
        ev = Evaluator()
        assert ev.eval(parse("(/ 10 0)"), make_standard_env()) == 0

    def test_nested(self) -> None:
        ev = Evaluator()
        result = ev.eval(parse("(+ (* 2 3) (- 10 4))"), make_standard_env())
        assert result == 12

    def test_abs(self) -> None:
        ev = Evaluator()
        assert ev.eval(parse("(abs -5)"), make_standard_env()) == 5


class TestComparison:
    def test_equal(self) -> None:
        ev = Evaluator()
        assert ev.eval(parse("(= 1 1)"), make_standard_env()) is True

    def test_less_than(self) -> None:
        ev = Evaluator()
        assert ev.eval(parse("(< 1 2)"), make_standard_env()) is True

    def test_greater_than(self) -> None:
        ev = Evaluator()
        assert ev.eval(parse("(> 2 1)"), make_standard_env()) is True


class TestLogic:
    def test_and_true(self) -> None:
        ev = Evaluator()
        assert ev.eval(parse("(and #t #t)"), make_standard_env()) is True

    def test_and_false(self) -> None:
        ev = Evaluator()
        result = ev.eval(parse("(and #t #f)"), make_standard_env())
        assert not result

    def test_or(self) -> None:
        ev = Evaluator()
        assert ev.eval(parse("(or #f #t)"), make_standard_env()) is True

    def test_not(self) -> None:
        ev = Evaluator()
        assert ev.eval(parse("(not #f)"), make_standard_env()) is True


class TestLists:
    def test_list_create(self) -> None:
        ev = Evaluator()
        result = ev.eval(parse("(list 1 2 3)"), make_standard_env())
        assert result == [1, 2, 3]

    def test_car(self) -> None:
        ev = Evaluator()
        result = ev.eval(parse("(car (list 1 2 3))"), make_standard_env())
        assert result == 1

    def test_cdr(self) -> None:
        ev = Evaluator()
        result = ev.eval(parse("(cdr (list 1 2 3))"), make_standard_env())
        assert result == [2, 3]

    def test_length(self) -> None:
        ev = Evaluator()
        result = ev.eval(parse("(length (list 1 2 3))"), make_standard_env())
        assert result == 3

    def test_nth(self) -> None:
        ev = Evaluator()
        result = ev.eval(parse("(nth (list 10 20 30) 1)"), make_standard_env())
        assert result == 20


class TestControlFlow:
    def test_if_true(self) -> None:
        ev = Evaluator()
        result = ev.eval(parse("(if #t 1 2)"), make_standard_env())
        assert result == 1

    def test_if_false(self) -> None:
        ev = Evaluator()
        result = ev.eval(parse("(if #f 1 2)"), make_standard_env())
        assert result == 2

    def test_cond(self) -> None:
        ev = Evaluator()
        result = ev.eval(
            parse("(cond ((= 1 2) 10) ((= 1 1) 20) (else 30))"),
            make_standard_env()
        )
        assert result == 20

    def test_begin(self) -> None:
        ev = Evaluator()
        result = ev.eval(
            parse("(begin (define x 1) (define y 2) (+ x y))"),
            make_standard_env()
        )
        assert result == 3


class TestDefineAndLet:
    def test_define(self) -> None:
        ev = Evaluator()
        env = make_standard_env()
        ev.eval(parse("(define x 42)"), env)
        assert ev.eval(parse("x"), env) == 42

    def test_let(self) -> None:
        ev = Evaluator()
        result = ev.eval(
            parse("(let ((x 10) (y 20)) (+ x y))"),
            make_standard_env()
        )
        assert result == 30

    def test_let_scoping(self) -> None:
        """Let bindings don't leak to outer scope."""
        ev = Evaluator()
        env = make_standard_env()
        ev.eval(parse("(define x 1)"), env)
        result = ev.eval(parse("(let ((x 99)) x)"), env)
        assert result == 99
        assert ev.eval(parse("x"), env) == 1


class TestLambda:
    def test_basic_lambda(self) -> None:
        ev = Evaluator()
        result = ev.eval(
            parse("((lambda (x) (* x x)) 5)"),
            make_standard_env()
        )
        assert result == 25

    def test_named_lambda(self) -> None:
        ev = Evaluator()
        env = make_standard_env()
        ev.eval(parse("(define square (lambda (x) (* x x)))"), env)
        assert ev.eval(parse("(square 7)"), env) == 49

    def test_closure(self) -> None:
        ev = Evaluator()
        env = make_standard_env()
        ev.eval(parse("(define make-adder (lambda (n) (lambda (x) (+ n x))))"), env)
        ev.eval(parse("(define add5 (make-adder 5))"), env)
        assert ev.eval(parse("(add5 3)"), env) == 8


class TestMapFilter:
    def test_map(self) -> None:
        ev = Evaluator()
        env = make_standard_env()
        ev.eval(parse("(define double (lambda (x) (* x 2)))"), env)
        result = ev.eval(parse("(map double (list 1 2 3))"), env)
        assert result == [2, 4, 6]

    def test_filter(self) -> None:
        ev = Evaluator()
        env = make_standard_env()
        ev.eval(parse("(define pos? (lambda (x) (> x 0)))"), env)
        result = ev.eval(parse("(filter pos? (list -1 2 -3 4))"), env)
        assert result == [2, 4]


class TestSubSim:
    def test_basic_subsim(self) -> None:
        ev = Evaluator(max_sim_depth=3, max_subsims_per_frame=3)
        result = ev.eval(
            parse('(sub-sim "test" (+ 1 2))'),
            make_standard_env()
        )
        assert result == 3

    def test_depth_limit(self) -> None:
        ev = Evaluator(max_sim_depth=1, sim_depth=1, max_subsims_per_frame=3)
        with pytest.raises(LispySandboxError, match="depth"):
            ev.eval(parse('(sub-sim "test" 42)'), make_standard_env())

    def test_count_limit(self) -> None:
        ev = Evaluator(max_sim_depth=3, max_subsims_per_frame=1)
        ev.eval(parse('(sub-sim "first" 1)'), make_standard_env())
        with pytest.raises(LispySandboxError, match="count"):
            ev.eval(parse('(sub-sim "second" 2)'), make_standard_env())

    def test_callback(self) -> None:
        results = []
        def cb(label, expr, env, depth):
            results.append({"label": label, "depth": depth})
            return {"status": "complete", "result": 42}

        ev = Evaluator(
            max_sim_depth=3, max_subsims_per_frame=3,
            subsim_callback=cb,
        )
        result = ev.eval(parse('(sub-sim "test-cb" (+ 1 2))'), make_standard_env())
        assert result["status"] == "complete"
        assert len(results) == 1
        assert results[0]["label"] == "test-cb"


class TestSandboxLimits:
    def test_step_limit(self) -> None:
        ev = Evaluator(max_steps=5)
        with pytest.raises(LispySandboxError, match="Step"):
            # This creates infinite recursion
            env = make_standard_env()
            ev.eval(parse("(begin (define f (lambda () (f))) (f))"), env)

    def test_depth_limit(self) -> None:
        ev = Evaluator(max_depth=3)
        with pytest.raises(LispySandboxError, match="Depth"):
            env = make_standard_env()
            ev.eval(parse("(begin (define f (lambda () (f))) (f))"), env)


class TestQuote:
    def test_quote(self) -> None:
        ev = Evaluator()
        result = ev.eval(parse("(quote (1 2 3))"), make_standard_env())
        assert result == [1, 2, 3]

    def test_quote_shorthand(self) -> None:
        ev = Evaluator()
        result = ev.eval(parse("'(1 2 3)"), make_standard_env())
        assert result == [1, 2, 3]


class TestEnv:
    def test_define_and_lookup(self) -> None:
        env = Env()
        env.define("x", 42)
        assert env.lookup("x") == 42

    def test_parent_lookup(self) -> None:
        parent = Env()
        parent.define("x", 42)
        child = Env(parent=parent)
        assert child.lookup("x") == 42

    def test_undefined_raises(self) -> None:
        env = Env()
        with pytest.raises(LispyError, match="Undefined"):
            env.lookup("nonexistent")

    def test_set(self) -> None:
        env = Env()
        env.define("x", 1)
        env.set("x", 99)
        assert env.lookup("x") == 99
