"""Tests for the LisPy safe interpreter — including sub-sim support."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.lispy import (
    Budget, DepthLimitExceeded, Env, Lambda, LispError,
    StepLimitExceeded, Symbol, make_env, parse, run,
    safe_eval, to_sexp, tokenize,
)

class TestTokenizer:
    def test_simple_expr(self):
        assert tokenize("(+ 1 2)") == ["(", "+", "1", "2", ")"]
    def test_nested(self):
        assert tokenize("(define x (+ 1 2))") == ["(", "define", "x", "(", "+", "1", "2", ")", ")"]
    def test_string_literal(self):
        assert '"hello world"' in tokenize('(str "hello world")')
    def test_comment(self):
        assert tokenize("; comment\n(+ 1 2)") == ["(", "+", "1", "2", ")"]
    def test_quote_shorthand(self):
        assert tokenize("'(1 2 3)")[0] == "'"
    def test_empty(self):
        assert tokenize("") == []

class TestParser:
    def test_atom_int(self):
        assert parse("42") == [42]
    def test_atom_float(self):
        assert parse("3.14") == [3.14]
    def test_atom_string(self):
        assert parse('"hello"') == ["hello"]
    def test_atom_symbol(self):
        assert isinstance(parse("foo")[0], Symbol)
    def test_atom_bool(self):
        assert parse("#t") == [True]
        assert parse("#f") == [False]
    def test_atom_nil(self):
        assert parse("nil") == [None]
    def test_list(self):
        assert len(parse("(+ 1 2)")[0]) == 3
    def test_nested_list(self):
        assert len(parse("(if (> x 0) x (- 0 x))")[0]) == 4
    def test_quote(self):
        assert parse("'(1 2 3)")[0] == [Symbol("quote"), [1, 2, 3]]
    def test_multiple(self):
        assert len(parse("(+ 1 2) (- 3 4)")) == 2
    def test_unbalanced_parens(self):
        with pytest.raises(LispError): parse("(+ 1 2")
    def test_unexpected_close(self):
        with pytest.raises(LispError): parse(")")

class TestArithmetic:
    def test_add(self): assert run("(+ 1 2)") == 3
    def test_add_many(self): assert run("(+ 1 2 3 4 5)") == 15
    def test_subtract(self): assert run("(- 10 3)") == 7
    def test_negate(self): assert run("(- 5)") == -5
    def test_multiply(self): assert run("(* 3 4)") == 12
    def test_divide(self): assert abs(run("(/ 10 3)") - 3.333) < 0.01
    def test_divide_by_zero(self): assert run("(/ 10 0)") == 0
    def test_modulo(self): assert run("(% 10 3)") == 1
    def test_abs(self): assert run("(abs -5)") == 5
    def test_min_max(self):
        assert run("(min 3 1 2)") == 1
        assert run("(max 3 1 2)") == 3
    def test_sqrt(self): assert run("(sqrt 16)") == 4.0
    def test_floor_ceil(self):
        assert run("(floor 3.7)") == 3
        assert run("(ceil 3.2)") == 4
    def test_nested(self): assert run("(+ (* 2 3) (- 10 4))") == 12

class TestComparison:
    def test_equal(self):
        assert run("(= 1 1)") is True
        assert run("(= 1 2)") is False
    def test_not_equal(self): assert run("(!= 1 2)") is True
    def test_less_greater(self):
        assert run("(< 1 2)") is True
        assert run("(> 2 1)") is True
        assert run("(<= 2 2)") is True
        assert run("(>= 1 2)") is False
    def test_not(self):
        assert run("(not true)") is False
        assert run("(not false)") is True
    def test_and(self):
        assert run("(and true true)") is True
        assert run("(and true false)") is False
    def test_or(self):
        assert run("(or false true)") is True
        assert run("(or false false)") is False

class TestSpecialForms:
    def test_define_variable(self): assert run("(begin (define x 42) x)") == 42
    def test_define_function(self): assert run("(begin (define (square x) (* x x)) (square 5))") == 25
    def test_lambda(self): assert run("((lambda (x) (* x x)) 7)") == 49
    def test_if_true(self): assert run("(if true 1 2)") == 1
    def test_if_false(self): assert run("(if false 1 2)") == 2
    def test_if_no_else(self): assert run("(if false 1)") is None
    def test_cond(self): assert run("(cond ((= 1 2) 10) ((= 1 1) 20) (else 30))") == 20
    def test_cond_else(self): assert run("(cond ((= 1 2) 10) (else 30))") == 30
    def test_let(self): assert run("(let ((x 5) (y 3)) (+ x y))") == 8
    def test_begin(self): assert run("(begin 1 2 3)") == 3
    def test_set_bang(self): assert run("(begin (define x 1) (set! x 42) x)") == 42
    def test_quote(self): assert run("(quote (1 2 3))") == [1, 2, 3]
    def test_quote_shorthand(self): assert len(run("'(a b c)")) == 3
    def test_closure(self):
        assert run("(begin (define (make-adder n) (lambda (x) (+ n x))) (define add5 (make-adder 5)) (add5 10))") == 15
    def test_recursion(self):
        assert run("(begin (define (factorial n) (if (<= n 1) 1 (* n (factorial (- n 1))))) (factorial 10))") == 3628800

class TestListOps:
    def test_list(self): assert run("(list 1 2 3)") == [1, 2, 3]
    def test_cons(self): assert run("(cons 1 (list 2 3))") == [1, 2, 3]
    def test_car(self): assert run("(car (list 1 2 3))") == 1
    def test_cdr(self): assert run("(cdr (list 1 2 3))") == [2, 3]
    def test_length(self): assert run("(length (list 1 2 3))") == 3
    def test_nth(self): assert run("(nth (list 10 20 30) 1)") == 20
    def test_append(self): assert run("(append (list 1 2) (list 3 4))") == [1, 2, 3, 4]
    def test_reverse(self): assert run("(reverse (list 1 2 3))") == [3, 2, 1]
    def test_map(self): assert run("(map (lambda (x) (* x 2)) (list 1 2 3))") == [2, 4, 6]
    def test_filter(self): assert run("(filter (lambda (x) (> x 2)) (list 1 2 3 4 5))") == [3, 4, 5]
    def test_reduce(self): assert run("(reduce (lambda (acc x) (+ acc x)) (list 1 2 3 4) 0)") == 10

class TestDictOps:
    def test_dict(self): assert run('(dict "a" 1 "b" 2)') == {"a": 1, "b": 2}
    def test_get(self): assert run('(get (dict "x" 42) "x")') == 42
    def test_assoc(self): assert run('(assoc (dict "a" 1) "b" 2)') == {"a": 1, "b": 2}
    def test_keys(self): assert sorted(run('(keys (dict "a" 1 "b" 2))')) == ["a", "b"]

class TestRandom:
    def test_deterministic(self):
        a = run("(random-float)", seed=42)
        b = run("(random-float)", seed=42)
        assert a == b
    def test_random_int(self):
        val = run("(random-int 1 10)")
        assert 1 <= val <= 10

class TestSubSim:
    def test_basic_subsim(self): assert run("(sub-sim (+ 1 2))") == 3
    def test_subsim_inherits_env(self): assert run("(begin (define x 10) (sub-sim (+ x 5)))") == 15
    def test_subsim_cannot_mutate_parent(self):
        assert run("(begin (define x 10) (sub-sim (begin (set! x 999) x)) x)") == 10
    def test_subsim_depth_3_works(self): assert run("(sub-sim (sub-sim (sub-sim 42)))") == 42
    def test_subsim_depth_4_fails(self):
        with pytest.raises(DepthLimitExceeded):
            run("(sub-sim (sub-sim (sub-sim (sub-sim 42))))")
    def test_subsim_count_limit(self):
        calls = " ".join(f"(sub-sim {i})" for i in range(55))
        with pytest.raises(LispError): run(f"(begin {calls})")
    def test_subsim_budget_charged_to_parent(self):
        env, ctx = make_env(seed=42)
        from src.lispy import run_in_env
        run_in_env("(sub-sim (begin (define x 1) (+ x 1)))", env, ctx)
        assert ctx.steps > 0

class TestSafety:
    def test_step_limit(self):
        with pytest.raises((StepLimitExceeded, DepthLimitExceeded)):
            run("(begin (define (loop) (loop)) (loop))")
    def test_undefined_symbol(self):
        with pytest.raises(LispError, match="undefined"):
            run("nonexistent_var")
    def test_wrong_arity(self):
        with pytest.raises(LispError):
            run("((lambda (x y) (+ x y)) 1)")

class TestSafeEval:
    def test_context_injection(self): assert safe_eval("(+ x 10)", context={"x": 32}) == 42
    def test_context_string(self): assert safe_eval('(str "hello " name)', context={"name": "Mars"}) == "hello Mars"

class TestSexp:
    def test_int(self): assert to_sexp(42) == "42"
    def test_float(self): assert to_sexp(3.14) == "3.14"
    def test_string(self): assert to_sexp("hello") == '"hello"'
    def test_list(self): assert to_sexp([1, 2, 3]) == "(1 2 3)"
    def test_nested(self): assert to_sexp([Symbol("+"), 1, [Symbol("*"), 2, 3]]) == "(+ 1 (* 2 3))"
    def test_bool(self):
        assert to_sexp(True) == "#t"
        assert to_sexp(False) == "#f"
    def test_nil(self): assert to_sexp(None) == "nil"

class TestPredicates:
    def test_number(self):
        assert run("(number? 42)") is True
        assert run('(number? "hi")') is False
    def test_string(self):
        assert run('(string? "hi")') is True
        assert run("(string? 42)") is False
    def test_list(self):
        assert run("(list? (list 1))") is True
        assert run("(list? 42)") is False
    def test_nil(self):
        assert run("(nil? nil)") is True
        assert run("(nil? 42)") is False
    def test_dict(self):
        assert run('(dict? (dict "a" 1))') is True
        assert run("(dict? 42)") is False
