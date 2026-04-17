"""Tests for the LisPy safe interpreter — including sub-sim support."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.lispy import (
    Budget,
    DepthLimitExceeded,
    Env,
    Lambda,
    LispError,
    StepLimitExceeded,
    Symbol,
    make_env,
    parse,
    run,
    safe_eval,
    to_sexp,
    tokenize,
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_simple_expr(self):
        assert tokenize("(+ 1 2)") == ["(", "+", "1", "2", ")"]

    def test_nested(self):
        tokens = tokenize("(define x (+ 1 2))")
        assert tokens == ["(", "define", "x", "(", "+", "1", "2", ")", ")"]

    def test_string_literal(self):
        tokens = tokenize('(str "hello world")')
        assert '"hello world"' in tokens

    def test_comment(self):
        tokens = tokenize("; this is a comment\n(+ 1 2)")
        assert tokens == ["(", "+", "1", "2", ")"]

    def test_quote_shorthand(self):
        tokens = tokenize("'(1 2 3)")
        assert tokens[0] == "'"

    def test_empty(self):
        assert tokenize("") == []
        assert tokenize("   ") == []


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TestParser:
    def test_atom_int(self):
        result = parse("42")
        assert result == [42]

    def test_atom_float(self):
        result = parse("3.14")
        assert result == [3.14]

    def test_atom_string(self):
        result = parse('"hello"')
        assert result == ["hello"]

    def test_atom_symbol(self):
        result = parse("foo")
        assert len(result) == 1
        assert isinstance(result[0], Symbol)

    def test_atom_bool(self):
        assert parse("#t") == [True]
        assert parse("#f") == [False]

    def test_atom_nil(self):
        assert parse("nil") == [None]

    def test_list(self):
        result = parse("(+ 1 2)")
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_nested_list(self):
        result = parse("(if (> x 0) x (- 0 x))")
        assert len(result[0]) == 4

    def test_quote(self):
        result = parse("'(1 2 3)")
        assert result[0] == [Symbol("quote"), [1, 2, 3]]

    def test_multiple(self):
        results = parse("(+ 1 2) (- 3 4)")
        assert len(results) == 2

    def test_unbalanced_parens(self):
        with pytest.raises(LispError):
            parse("(+ 1 2")

    def test_unexpected_close(self):
        with pytest.raises(LispError):
            parse(")")


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

class TestArithmetic:
    def test_add(self):
        assert run("(+ 1 2)") == 3

    def test_add_many(self):
        assert run("(+ 1 2 3 4 5)") == 15

    def test_subtract(self):
        assert run("(- 10 3)") == 7

    def test_negate(self):
        assert run("(- 5)") == -5

    def test_multiply(self):
        assert run("(* 3 4)") == 12

    def test_divide(self):
        result = run("(/ 10 3)")
        assert abs(result - 3.333) < 0.01

    def test_divide_by_zero(self):
        result = run("(/ 10 0)")
        assert result == 0  # safe division returns 0

    def test_modulo(self):
        assert run("(% 10 3)") == 1

    def test_abs(self):
        assert run("(abs -5)") == 5

    def test_min_max(self):
        assert run("(min 3 1 2)") == 1
        assert run("(max 3 1 2)") == 3

    def test_sqrt(self):
        assert run("(sqrt 16)") == 4.0

    def test_floor_ceil(self):
        assert run("(floor 3.7)") == 3
        assert run("(ceil 3.2)") == 4

    def test_nested(self):
        assert run("(+ (* 2 3) (- 10 4))") == 12


# ---------------------------------------------------------------------------
# Comparison and boolean
# ---------------------------------------------------------------------------

class TestComparison:
    def test_equal(self):
        assert run("(= 1 1)") is True
        assert run("(= 1 2)") is False

    def test_not_equal(self):
        assert run("(!= 1 2)") is True

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


# ---------------------------------------------------------------------------
# Special forms
# ---------------------------------------------------------------------------

class TestSpecialForms:
    def test_define_variable(self):
        assert run("(begin (define x 42) x)") == 42

    def test_define_function(self):
        assert run("(begin (define (square x) (* x x)) (square 5))") == 25

    def test_lambda(self):
        assert run("((lambda (x) (* x x)) 7)") == 49

    def test_if_true(self):
        assert run("(if true 1 2)") == 1

    def test_if_false(self):
        assert run("(if false 1 2)") == 2

    def test_if_no_else(self):
        assert run("(if false 1)") is None

    def test_cond(self):
        result = run("(cond ((= 1 2) 10) ((= 1 1) 20) (else 30))")
        assert result == 20

    def test_cond_else(self):
        result = run("(cond ((= 1 2) 10) (else 30))")
        assert result == 30

    def test_let(self):
        assert run("(let ((x 5) (y 3)) (+ x y))") == 8

    def test_begin(self):
        assert run("(begin 1 2 3)") == 3

    def test_set_bang(self):
        assert run("(begin (define x 1) (set! x 42) x)") == 42

    def test_quote(self):
        result = run("(quote (1 2 3))")
        assert result == [1, 2, 3]

    def test_quote_shorthand(self):
        result = run("'(a b c)")
        assert len(result) == 3

    def test_closure(self):
        code = """
        (begin
          (define (make-adder n) (lambda (x) (+ n x)))
          (define add5 (make-adder 5))
          (add5 10))
        """
        assert run(code) == 15

    def test_recursion(self):
        code = """
        (begin
          (define (factorial n)
            (if (<= n 1) 1 (* n (factorial (- n 1)))))
          (factorial 10))
        """
        assert run(code) == 3628800


# ---------------------------------------------------------------------------
# List operations
# ---------------------------------------------------------------------------

class TestListOps:
    def test_list(self):
        assert run("(list 1 2 3)") == [1, 2, 3]

    def test_cons(self):
        assert run("(cons 1 (list 2 3))") == [1, 2, 3]

    def test_car(self):
        assert run("(car (list 1 2 3))") == 1

    def test_cdr(self):
        assert run("(cdr (list 1 2 3))") == [2, 3]

    def test_length(self):
        assert run("(length (list 1 2 3))") == 3

    def test_nth(self):
        assert run("(nth (list 10 20 30) 1)") == 20

    def test_append(self):
        assert run("(append (list 1 2) (list 3 4))") == [1, 2, 3, 4]

    def test_reverse(self):
        assert run("(reverse (list 1 2 3))") == [3, 2, 1]

    def test_map(self):
        code = "(map (lambda (x) (* x 2)) (list 1 2 3))"
        assert run(code) == [2, 4, 6]

    def test_filter(self):
        code = "(filter (lambda (x) (> x 2)) (list 1 2 3 4 5))"
        assert run(code) == [3, 4, 5]

    def test_reduce(self):
        code = "(reduce (lambda (acc x) (+ acc x)) (list 1 2 3 4) 0)"
        assert run(code) == 10


# ---------------------------------------------------------------------------
# Dict operations
# ---------------------------------------------------------------------------

class TestDictOps:
    def test_dict(self):
        result = run('(dict "a" 1 "b" 2)')
        assert result == {"a": 1, "b": 2}

    def test_get(self):
        assert run('(get (dict "x" 42) "x")') == 42

    def test_assoc(self):
        # dict-assoc adds key to a dict
        result = run('(dict-assoc (dict "a" 1) "b" 2)')
        assert result == {"a": 1, "b": 2}

    def test_keys(self):
        result = run('(keys (dict "a" 1 "b" 2))')
        assert sorted(result) == ["a", "b"]


# ---------------------------------------------------------------------------
# Random (deterministic)
# ---------------------------------------------------------------------------

class TestRandom:
    def test_deterministic(self):
        a = run("(random-float)", seed=42)
        b = run("(random-float)", seed=42)
        assert a == b

    def test_random_int(self):
        val = run("(random-int 1 10)")
        assert 1 <= val <= 10


# ---------------------------------------------------------------------------
# Sub-simulation (Turtles All the Way Down)
# ---------------------------------------------------------------------------

class TestSubSim:
    def test_basic_subsim(self):
        result = run("(sub-sim (+ 1 2))")
        assert result == 3

    def test_subsim_inherits_env(self):
        result = run("(begin (define x 10) (sub-sim (+ x 5)))")
        assert result == 15

    def test_subsim_cannot_mutate_parent(self):
        """Critical: sub-sim mutations must NOT leak to parent."""
        code = """
        (begin
          (define x 10)
          (sub-sim (begin (set! x 999) x))
          x)
        """
        result = run(code)
        assert result == 10  # parent x unchanged

    def test_subsim_depth_3_works(self):
        """Depth 3 is the constitutional limit — it should work."""
        code = "(sub-sim (sub-sim (sub-sim 42)))"
        assert run(code) == 42

    def test_subsim_depth_4_fails(self):
        """Depth 4 should raise DepthLimitExceeded."""
        code = "(sub-sim (sub-sim (sub-sim (sub-sim 42))))"
        with pytest.raises(DepthLimitExceeded):
            run(code)

    def test_subsim_count_limit(self):
        """Should limit total sub-sims."""
        calls = " ".join(f"(sub-sim {i})" for i in range(55))
        code = f"(begin {calls})"
        with pytest.raises(LispError):
            run(code)

    def test_subsim_budget_charged_to_parent(self):
        """Sub-sim steps should count against parent budget."""
        env, ctx = make_env(seed=0)
        from src.lispy import run_in_env
        run_in_env("(sub-sim (begin (define x 1) (+ x 1)))", env, ctx)
        assert ctx.steps > 0


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

class TestSafety:
    def test_step_limit(self):
        """Infinite recursion should be caught."""
        code = "(begin (define (loop) (loop)) (loop))"
        with pytest.raises((StepLimitExceeded, DepthLimitExceeded)):
            run(code)

    def test_undefined_symbol(self):
        with pytest.raises(LispError, match="undefined"):
            run("nonexistent_var")

    def test_wrong_arity(self):
        with pytest.raises(LispError):
            run("((lambda (x y) (+ x y)) 1)")


# ---------------------------------------------------------------------------
# safe_eval with context
# ---------------------------------------------------------------------------

class TestSafeEval:
    def test_context_injection(self):
        result = safe_eval("(+ x 10)", context={"x": 32})
        assert result == 42

    def test_context_string(self):
        result = safe_eval('(str "hello " name)', context={"name": "Mars"})
        assert result == "hello Mars"


# ---------------------------------------------------------------------------
# S-expression serialization
# ---------------------------------------------------------------------------

class TestSexp:
    def test_int(self):
        assert to_sexp(42) == "42"

    def test_float(self):
        assert to_sexp(3.14) == "3.14"

    def test_string(self):
        assert to_sexp("hello") == '"hello"'

    def test_list(self):
        assert to_sexp([1, 2, 3]) == "(1 2 3)"

    def test_nested(self):
        assert to_sexp([Symbol("+"), 1, [Symbol("*"), 2, 3]]) == "(+ 1 (* 2 3))"

    def test_bool(self):
        assert to_sexp(True) == "#t"
        assert to_sexp(False) == "#f"

    def test_nil(self):
        assert to_sexp(None) == "nil"


# ---------------------------------------------------------------------------
# Type predicates
# ---------------------------------------------------------------------------

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
