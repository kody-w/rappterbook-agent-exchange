"""Tests for the LisPy safe-eval interpreter (src/lispy.py).

Covers: tokenizer, parser, evaluator, special forms, builtins,
sub-simulation depth limits, step budget, closures, dict access,
and safety guarantees (no I/O escape).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.lispy import (
    Env,
    LispError,
    DepthExceeded,
    StepLimitExceeded,
    evaluate,
    lisp_eval,
    parse,
    parse_all,
    standard_env,
    to_sexpr,
    tokenize,
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenizer:
    def test_simple_expr(self) -> None:
        assert tokenize("(+ 1 2)") == ["(", "+", "1", "2", ")"]

    def test_nested(self) -> None:
        tokens = tokenize("(if (> x 5) 1 0)")
        assert tokens[0] == "("
        assert "if" in tokens
        assert tokens.count("(") == 2
        assert tokens.count(")") == 2

    def test_string_literal(self) -> None:
        tokens = tokenize('(str "hello world")')
        assert '"hello world"' in tokens

    def test_comment(self) -> None:
        tokens = tokenize("; this is a comment\n(+ 1 2)")
        assert tokens == ["(", "+", "1", "2", ")"]

    def test_quote(self) -> None:
        tokens = tokenize("'(1 2 3)")
        assert tokens[0] == "'"

    def test_empty_input(self) -> None:
        assert tokenize("") == []
        assert tokenize("   ") == []

    def test_booleans(self) -> None:
        tokens = tokenize("#t #f")
        assert tokens == ["#t", "#f"]

    def test_numbers(self) -> None:
        tokens = tokenize("42 3.14")
        assert tokens == ["42", "3.14"]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParser:
    def test_atom_int(self) -> None:
        assert parse("42") == 42

    def test_atom_float(self) -> None:
        assert parse("3.14") == 3.14

    def test_atom_symbol(self) -> None:
        assert parse("foo") == "foo"

    def test_atom_string(self) -> None:
        assert parse('"hello"') == "hello"

    def test_boolean(self) -> None:
        assert parse("#t") is True
        assert parse("#f") is False

    def test_list(self) -> None:
        result = parse("(+ 1 2)")
        assert result == ["+", 1, 2]

    def test_nested_list(self) -> None:
        result = parse("(if (> x 5) 1 0)")
        assert result == ["if", [">", "x", 5], 1, 0]

    def test_quote_shorthand(self) -> None:
        result = parse("'(1 2 3)")
        assert result == ["quote", [1, 2, 3]]

    def test_parse_all_multiple(self) -> None:
        results = parse_all("(+ 1 2) (* 3 4)")
        assert len(results) == 2
        assert results[0] == ["+", 1, 2]
        assert results[1] == ["*", 3, 4]

    def test_empty_raises(self) -> None:
        with pytest.raises(LispError, match="empty"):
            parse("")

    def test_unmatched_paren(self) -> None:
        with pytest.raises(LispError, match="missing"):
            parse("(+ 1 2")

    def test_unexpected_close(self) -> None:
        with pytest.raises(LispError, match="unexpected"):
            parse(")")


# ---------------------------------------------------------------------------
# Evaluator — arithmetic
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_addition(self) -> None:
        assert evaluate("(+ 1 2)") == 3

    def test_subtraction(self) -> None:
        assert evaluate("(- 10 3)") == 7

    def test_negation(self) -> None:
        assert evaluate("(- 5)") == -5

    def test_multiplication(self) -> None:
        assert evaluate("(* 3 4)") == 12

    def test_division(self) -> None:
        assert evaluate("(/ 10 3)") == pytest.approx(3.333, rel=0.01)

    def test_division_by_zero(self) -> None:
        with pytest.raises(LispError, match="division by zero"):
            evaluate("(/ 1 0)")

    def test_modulo(self) -> None:
        assert evaluate("(mod 10 3)") == 1

    def test_nested_arithmetic(self) -> None:
        assert evaluate("(+ (* 2 3) (- 10 4))") == 12

    def test_abs(self) -> None:
        assert evaluate("(abs -7)") == 7

    def test_min_max(self) -> None:
        assert evaluate("(min 3 7)") == 3
        assert evaluate("(max 3 7)") == 7

    def test_floor_ceil(self) -> None:
        assert evaluate("(floor 3.7)") == 3
        assert evaluate("(ceil 3.2)") == 4

    def test_sqrt(self) -> None:
        assert evaluate("(sqrt 16)") == 4.0


# ---------------------------------------------------------------------------
# Comparisons and logic
# ---------------------------------------------------------------------------


class TestComparisons:
    def test_less_than(self) -> None:
        assert evaluate("(< 1 2)") is True
        assert evaluate("(< 2 1)") is False

    def test_greater_than(self) -> None:
        assert evaluate("(> 5 3)") is True

    def test_equality(self) -> None:
        assert evaluate("(= 42 42)") is True
        assert evaluate("(= 1 2)") is False

    def test_not(self) -> None:
        assert evaluate("(not true)") is False
        assert evaluate("(not false)") is True

    def test_and(self) -> None:
        assert evaluate("(and true true)") is True
        assert evaluate("(and true false)") is False

    def test_or(self) -> None:
        assert evaluate("(or false true)") is True
        assert evaluate("(or false false)") is False


# ---------------------------------------------------------------------------
# Special forms
# ---------------------------------------------------------------------------


class TestSpecialForms:
    def test_if_true(self) -> None:
        assert evaluate("(if true 1 2)") == 1

    def test_if_false(self) -> None:
        assert evaluate("(if false 1 2)") == 2

    def test_if_no_else(self) -> None:
        assert evaluate("(if false 1)") is None

    def test_cond(self) -> None:
        result = evaluate("(cond (false 1) (true 2) (true 3))")
        assert result == 2

    def test_cond_no_match(self) -> None:
        assert evaluate("(cond (false 1))") is None

    def test_quote(self) -> None:
        assert evaluate("'(1 2 3)") == [1, 2, 3]

    def test_begin(self) -> None:
        assert evaluate("(begin 1 2 3)") == 3

    def test_let(self) -> None:
        assert evaluate("(let ((x 10) (y 20)) (+ x y))") == 30

    def test_define(self) -> None:
        env = standard_env()
        evaluate("(define x 42)", env=env)
        assert evaluate("x", env=env) == 42

    def test_lambda(self) -> None:
        result = evaluate("((lambda (x y) (+ x y)) 3 4)")
        assert result == 7

    def test_lambda_closure(self) -> None:
        env = standard_env()
        evaluate("(define add-n (lambda (n) (lambda (x) (+ x n))))", env=env)
        evaluate("(define add-5 (add-n 5))", env=env)
        assert evaluate("(add-5 10)", env=env) == 15


# ---------------------------------------------------------------------------
# List operations
# ---------------------------------------------------------------------------


class TestLists:
    def test_list(self) -> None:
        assert evaluate("(list 1 2 3)") == [1, 2, 3]

    def test_car(self) -> None:
        assert evaluate("(car (list 1 2 3))") == 1

    def test_cdr(self) -> None:
        assert evaluate("(cdr (list 1 2 3))") == [2, 3]

    def test_cons(self) -> None:
        assert evaluate("(cons 1 (list 2 3))") == [1, 2, 3]

    def test_length(self) -> None:
        assert evaluate("(length (list 1 2 3))") == 3

    def test_nth(self) -> None:
        assert evaluate("(nth (list 10 20 30) 1)") == 20

    def test_empty(self) -> None:
        assert evaluate("(empty? (list))") is True
        assert evaluate("(empty? (list 1))") is False

    def test_append(self) -> None:
        assert evaluate("(append (list 1 2) (list 3 4))") == [1, 2, 3, 4]

    def test_map(self) -> None:
        env = standard_env()
        evaluate("(define double (lambda (x) (* x 2)))", env=env)
        assert evaluate("(map double (list 1 2 3))", env=env) == [2, 4, 6]

    def test_filter(self) -> None:
        env = standard_env()
        evaluate("(define positive? (lambda (x) (> x 0)))", env=env)
        assert evaluate("(filter positive? (list -1 2 -3 4))", env=env) == [2, 4]

    def test_reduce(self) -> None:
        assert evaluate("(reduce + (list 1 2 3 4) 0)") == 10

    def test_sort(self) -> None:
        assert evaluate("(sort (list 3 1 2))") == [1, 2, 3]

    def test_reverse(self) -> None:
        assert evaluate("(reverse (list 1 2 3))") == [3, 2, 1]

    def test_range(self) -> None:
        assert evaluate("(range 5)") == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Dict operations
# ---------------------------------------------------------------------------


class TestDicts:
    def test_dict_create(self) -> None:
        result = evaluate('(dict "a" 1 "b" 2)')
        assert result == {"a": 1, "b": 2}

    def test_get(self) -> None:
        env = standard_env()
        evaluate('(define d (dict "food" 100 "water" 50))', env=env)
        assert evaluate('(get d "food")', env=env) == 100

    def test_get_default(self) -> None:
        env = standard_env()
        evaluate('(define d (dict "a" 1))', env=env)
        assert evaluate('(get d "missing" 0)', env=env) == 0

    def test_put(self) -> None:
        env = standard_env()
        evaluate('(define d (dict "a" 1))', env=env)
        result = evaluate('(put d "b" 2)', env=env)
        assert result == {"a": 1, "b": 2}

    def test_keys(self) -> None:
        result = evaluate('(keys (dict "a" 1 "b" 2))')
        assert sorted(result) == ["a", "b"]

    def test_values(self) -> None:
        result = evaluate('(values (dict "a" 1 "b" 2))')
        assert sorted(result) == [1, 2]

    def test_dict_predicate(self) -> None:
        assert evaluate('(dict? (dict "a" 1))') is True
        assert evaluate("(dict? 42)") is False


# ---------------------------------------------------------------------------
# Sub-simulation
# ---------------------------------------------------------------------------


class TestSubSim:
    def test_basic_sub_sim(self) -> None:
        result = evaluate("(sub-sim 100 (+ 1 2))")
        assert result == 3

    def test_sub_sim_with_parent_env(self) -> None:
        env = standard_env()
        evaluate("(define x 10)", env=env)
        result = evaluate("(sub-sim 100 (+ x 5))", env=env)
        assert result == 15

    def test_sub_sim_depth_2(self) -> None:
        result = evaluate("(sub-sim 200 (sub-sim 100 (+ 1 1)))")
        assert result == 2

    def test_sub_sim_depth_3(self) -> None:
        result = evaluate("(sub-sim 1000 (sub-sim 500 (sub-sim 200 42)))")
        assert result == 42

    def test_sub_sim_depth_exceeds(self) -> None:
        # Depth 4 should fail — returned as error string from inner sub-sim
        result = evaluate(
            "(sub-sim 2000 (sub-sim 1000 (sub-sim 500 (sub-sim 200 1))))"
        )
        assert isinstance(result, str) and "sub-sim-error" in result

    def test_sub_sim_budget_exceeded(self) -> None:
        # Tiny budget → should hit limit inside sub-sim
        result = evaluate(
            "(sub-sim 100 (begin 1 2 3 4 5 6 7 8 9 10))",
            step_limit=30,
        )
        # Sub-sim catches budget exhaustion and returns error string
        assert isinstance(result, str) and "sub-sim-error" in result


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


class TestSafety:
    def test_step_limit(self) -> None:
        with pytest.raises(StepLimitExceeded):
            # Infinite-ish recursion via lambda
            env = standard_env()
            evaluate("(define f (lambda (n) (f (+ n 1))))", env=env)
            evaluate("(f 0)", env=env, step_limit=100)

    def test_unbound_symbol(self) -> None:
        with pytest.raises(LispError, match="unbound"):
            evaluate("nonexistent")

    def test_call_non_function(self) -> None:
        with pytest.raises(LispError, match="not callable"):
            evaluate("(42 1 2)")

    def test_no_file_access(self) -> None:
        """Ensure no file/import builtins exist."""
        env = standard_env()
        for dangerous in ("open", "import", "exec", "eval", "compile",
                          "__import__", "system", "os", "subprocess"):
            assert dangerous not in env

    def test_deterministic_with_seed(self) -> None:
        r1 = evaluate("(random)", seed=42)
        r2 = evaluate("(random)", seed=42)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_sexpr_int(self) -> None:
        assert to_sexpr(42) == "42"

    def test_to_sexpr_string(self) -> None:
        assert to_sexpr("hello") == '"hello"'

    def test_to_sexpr_list(self) -> None:
        assert to_sexpr([1, 2, 3]) == "(1 2 3)"

    def test_to_sexpr_dict(self) -> None:
        result = to_sexpr({"a": 1})
        assert "dict" in result
        assert '"a"' in result

    def test_to_sexpr_nil(self) -> None:
        assert to_sexpr(None) == "nil"

    def test_to_sexpr_bool(self) -> None:
        assert to_sexpr(True) == "#t"
        assert to_sexpr(False) == "#f"


# ---------------------------------------------------------------------------
# Clamp
# ---------------------------------------------------------------------------


class TestClamp:
    def test_clamp_within(self) -> None:
        assert evaluate("(clamp 0.5 0 1)") == 0.5

    def test_clamp_below(self) -> None:
        assert evaluate("(clamp -1 0 1)") == 0

    def test_clamp_above(self) -> None:
        assert evaluate("(clamp 2 0 1)") == 1


# ---------------------------------------------------------------------------
# String ops
# ---------------------------------------------------------------------------


class TestStrings:
    def test_str_concat(self) -> None:
        assert evaluate('(str "hello" " " "world")') == "hello world"

    def test_str_predicate(self) -> None:
        assert evaluate('(str? "hello")') is True
        assert evaluate("(str? 42)") is False

    def test_num_predicate(self) -> None:
        assert evaluate("(num? 42)") is True
        assert evaluate('(num? "hello")') is False

    def test_list_predicate(self) -> None:
        assert evaluate("(list? (list 1 2))") is True
        assert evaluate("(list? 42)") is False
