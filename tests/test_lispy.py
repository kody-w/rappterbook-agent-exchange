"""Tests for the LisPy safe-eval interpreter."""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lispy import (
    Lispy, LispyError, LispyTimeout,
    Symbol, Env, Lambda, tokenize, parse_all,
)


# --- Tokenizer ---

class TestTokenizer:
    def test_basic(self):
        assert tokenize("(+ 1 2)") == ["(", "+", "1", "2", ")"]

    def test_nested(self):
        toks = tokenize("(if (> x 0) 'yes 'no)")
        assert toks[0] == "("
        assert "if" in toks

    def test_string(self):
        toks = tokenize('(str "hello world")')
        assert '"hello world"' in toks

    def test_comment(self):
        toks = tokenize("; comment\n(+ 1 2)")
        assert ";" not in "".join(toks)
        assert "+" in toks

    def test_quote_sugar(self):
        toks = tokenize("'foo")
        assert "'" in toks
        assert "foo" in toks


# --- Parser ---

class TestParser:
    def test_atom_int(self):
        exprs = parse_all(tokenize("42"))
        assert exprs == [42]

    def test_atom_float(self):
        exprs = parse_all(tokenize("3.14"))
        assert exprs == [3.14]

    def test_atom_string(self):
        exprs = parse_all(tokenize('"hello"'))
        assert exprs == ["hello"]

    def test_atom_bool(self):
        assert parse_all(tokenize("#t")) == [True]
        assert parse_all(tokenize("#f")) == [False]

    def test_atom_nil(self):
        assert parse_all(tokenize("nil")) == [None]

    def test_list(self):
        exprs = parse_all(tokenize("(+ 1 2)"))
        assert len(exprs) == 1
        assert len(exprs[0]) == 3

    def test_nested_list(self):
        exprs = parse_all(tokenize("(if (> x 1) 2 3)"))
        assert isinstance(exprs[0][1], list)

    def test_quote(self):
        exprs = parse_all(tokenize("'foo"))
        assert exprs[0][0] == Symbol("quote")


# --- Arithmetic ---

class TestArithmetic:
    def test_add(self):
        assert Lispy().eval_string("(+ 1 2)") == 3

    def test_subtract(self):
        assert Lispy().eval_string("(- 10 3)") == 7

    def test_multiply(self):
        assert Lispy().eval_string("(* 4 5)") == 20

    def test_divide(self):
        assert Lispy().eval_string("(/ 10 3)") == pytest.approx(3.333, abs=0.01)

    def test_divide_by_zero(self):
        assert Lispy().eval_string("(/ 1 0)") == 0

    def test_nested_arithmetic(self):
        assert Lispy().eval_string("(+ (* 2 3) (- 10 4))") == 12

    def test_abs(self):
        assert Lispy().eval_string("(abs -5)") == 5

    def test_min_max(self):
        assert Lispy().eval_string("(min 3 7)") == 3
        assert Lispy().eval_string("(max 3 7)") == 7

    def test_floor_ceil(self):
        assert Lispy().eval_string("(floor 3.7)") == 3
        assert Lispy().eval_string("(ceil 3.2)") == 4

    def test_sqrt(self):
        assert Lispy().eval_string("(sqrt 16)") == 4.0

    def test_pow(self):
        assert Lispy().eval_string("(pow 2 10)") == 1024

    def test_variadic_add(self):
        assert Lispy().eval_string("(+ 1 2 3 4 5)") == 15

    def test_variadic_multiply(self):
        assert Lispy().eval_string("(* 2 3 4)") == 24


# --- Comparison ---

class TestComparison:
    def test_equal(self):
        assert Lispy().eval_string("(= 1 1)") is True
        assert Lispy().eval_string("(= 1 2)") is False

    def test_not_equal(self):
        assert Lispy().eval_string("(!= 1 2)") is True

    def test_less_than(self):
        assert Lispy().eval_string("(< 1 2)") is True

    def test_greater_than(self):
        assert Lispy().eval_string("(> 5 3)") is True

    def test_leq_geq(self):
        assert Lispy().eval_string("(<= 3 3)") is True
        assert Lispy().eval_string("(>= 5 3)") is True


# --- Control flow ---

class TestControlFlow:
    def test_if_true(self):
        assert Lispy().eval_string("(if #t 1 2)") == 1

    def test_if_false(self):
        assert Lispy().eval_string("(if #f 1 2)") == 2

    def test_if_no_else(self):
        assert Lispy().eval_string("(if #f 1)") is None

    def test_cond(self):
        result = Lispy().eval_string("""
            (cond
                ((> 1 5) "a")
                ((< 1 5) "b")
                (else "c"))
        """)
        assert result == "b"

    def test_and(self):
        assert Lispy().eval_string("(and #t #t)") is True
        assert Lispy().eval_string("(and #t #f)") is False

    def test_or(self):
        assert Lispy().eval_string("(or #f #t)") is True
        assert Lispy().eval_string("(or #f #f)") is False

    def test_not(self):
        assert Lispy().eval_string("(not #t)") is False
        assert Lispy().eval_string("(not #f)") is True


# --- Variables and lambdas ---

class TestVariables:
    def test_define(self):
        assert Lispy().eval_string("(begin (define x 42) x)") == 42

    def test_set(self):
        assert Lispy().eval_string("(begin (define x 1) (set! x 2) x)") == 2

    def test_lambda(self):
        result = Lispy().eval_string("((lambda (x) (* x x)) 5)")
        assert result == 25

    def test_closure(self):
        result = Lispy().eval_string("""
            (begin
                (define make-adder (lambda (n) (lambda (x) (+ x n))))
                (define add5 (make-adder 5))
                (add5 10))
        """)
        assert result == 15

    def test_let(self):
        result = Lispy().eval_string("(let ((x 10) (y 20)) (+ x y))")
        assert result == 30

    def test_begin(self):
        result = Lispy().eval_string("(begin 1 2 3)")
        assert result == 3


# --- Lists ---

class TestLists:
    def test_list_create(self):
        result = Lispy().eval_string("(list 1 2 3)")
        assert result == [1, 2, 3]

    def test_car_cdr(self):
        assert Lispy().eval_string("(car (list 1 2 3))") == 1
        assert Lispy().eval_string("(cdr (list 1 2 3))") == [2, 3]

    def test_cons(self):
        assert Lispy().eval_string("(cons 0 (list 1 2))") == [0, 1, 2]

    def test_length(self):
        assert Lispy().eval_string("(length (list 1 2 3))") == 3

    def test_nth(self):
        assert Lispy().eval_string("(nth (list 10 20 30) 1)") == 20

    def test_range(self):
        assert Lispy().eval_string("(range 5)") == [0, 1, 2, 3, 4]
        assert Lispy().eval_string("(range 2 5)") == [2, 3, 4]

    def test_reverse(self):
        assert Lispy().eval_string("(reverse (list 1 2 3))") == [3, 2, 1]

    def test_empty(self):
        assert Lispy().eval_string("(empty? (list))") is True
        assert Lispy().eval_string("(empty? (list 1))") is False


# --- Higher-order ---

class TestHigherOrder:
    def test_map(self):
        result = Lispy().eval_string("(map (lambda (x) (* x 2)) (list 1 2 3))")
        assert result == [2, 4, 6]

    def test_filter(self):
        result = Lispy().eval_string("(filter (lambda (x) (> x 2)) (list 1 2 3 4))")
        assert result == [3, 4]

    def test_reduce(self):
        result = Lispy().eval_string("(reduce + (list 1 2 3 4) 0)")
        assert result == 10


# --- Strings ---

class TestStrings:
    def test_str_concat(self):
        assert Lispy().eval_string('(str "hello" " " "world")') == "hello world"

    def test_str_length(self):
        assert Lispy().eval_string('(str-length "hello")') == 5

    def test_str_upper(self):
        assert Lispy().eval_string('(str-upper "hello")') == "HELLO"


# --- Dicts ---

class TestDicts:
    def test_dict_create(self):
        result = Lispy().eval_string('(dict "a" 1 "b" 2)')
        assert result == {"a": 1, "b": 2}

    def test_dict_get(self):
        result = Lispy().eval_string('(dict-get (dict "x" 42) "x")')
        assert result == 42

    def test_dict_set(self):
        result = Lispy().eval_string('(dict-set (dict "a" 1) "b" 2)')
        assert result == {"a": 1, "b": 2}

    def test_dict_keys(self):
        result = Lispy().eval_string('(dict-keys (dict "a" 1 "b" 2))')
        assert set(result) == {"a", "b"}


# --- Random ---

class TestRandom:
    def test_deterministic(self):
        a = Lispy(seed=42).eval_string("(random)")
        b = Lispy(seed=42).eval_string("(random)")
        assert a == b

    def test_different_seeds(self):
        a = Lispy(seed=42).eval_string("(random)")
        b = Lispy(seed=99).eval_string("(random)")
        assert a != b

    def test_random_int(self):
        r = Lispy(seed=42).eval_string("(random-int 1 10)")
        assert 1 <= r <= 10

    def test_random_choice(self):
        r = Lispy(seed=42).eval_string("(random-choice (list 1 2 3))")
        assert r in [1, 2, 3]


# --- Safety ---

class TestSafety:
    def test_step_limit(self):
        with pytest.raises(LispyTimeout):
            Lispy(step_limit=10).eval_string("""
                (begin
                    (define f (lambda (n) (f (+ n 1))))
                    (f 0))
            """)

    def test_recursion_depth(self):
        """Deep recursion should hit step limit, not crash."""
        with pytest.raises((LispyTimeout, LispyError, RecursionError)):
            Lispy(step_limit=100).eval_string("""
                (begin
                    (define f (lambda (n) (if (= n 0) 0 (f (- n 1)))))
                    (f 10000))
            """)

    def test_unbound_symbol(self):
        with pytest.raises(LispyError, match="Unbound"):
            Lispy().eval_string("undefined_var")

    def test_arity_mismatch(self):
        with pytest.raises(LispyError, match="Arity"):
            Lispy().eval_string("((lambda (x y) (+ x y)) 1)")

    def test_no_io_builtins(self):
        """No open, print, exec, eval, import available."""
        for name in ["open", "exec", "eval", "__import__", "print"]:
            with pytest.raises(LispyError, match="Unbound"):
                Lispy().eval_string(f"{name}")


# --- Sub-simulations ---

class TestSubSim:
    def test_basic_sub_sim(self):
        result = Lispy(seed=42).eval_string("""
            (sub-sim "test" '(+ 1 2))
        """)
        assert result == 3

    def test_labeled_sub_sim(self):
        vm = Lispy(seed=42)
        vm.eval_string('(sub-sim "my-label" \'(* 3 4))')
        assert len(vm.sub_sim_log) == 1
        assert vm.sub_sim_log[0]["label"] == "my-label"

    def test_sub_sim_depth_limit(self):
        """Depth-3 sub-sims should raise."""
        with pytest.raises(LispyError, match="depth"):
            vm = Lispy(seed=42, sim_depth=3)
            vm.eval_string("(sub-sim \"deep\" '(+ 1 1))")

    def test_depth_2_sub_sim(self):
        """Nested sub-sims should work up to depth 2."""
        result = Lispy(seed=42).eval_string("""
            (sub-sim "outer"
                '(sub-sim "inner" '(+ 10 20)))
        """)
        assert result == 30

    def test_sub_sim_logging(self):
        vm = Lispy(seed=42)
        vm.eval_string('(sub-sim "s1" \'(+ 1 1))')
        assert len(vm.sub_sim_log) == 1
        log = vm.sub_sim_log[0]
        assert "depth" in log
        assert "steps" in log
        assert "result" in log

    def test_sub_sim_isolation(self):
        """Sub-sim cannot access parent variables."""
        vm = Lispy(seed=42)
        vm.global_env.set("secret", 42)
        with pytest.raises(LispyError):
            vm.eval_string("(sub-sim \"test\" '(+ secret 1))")

    def test_sub_sim_deterministic(self):
        a = Lispy(seed=42).eval_string('(sub-sim "t" \'(random))')
        b = Lispy(seed=42).eval_string('(sub-sim "t" \'(random))')
        assert a == b


# --- Type checks ---

class TestTypeChecks:
    def test_number(self):
        assert Lispy().eval_string("(number? 42)") is True
        assert Lispy().eval_string('(number? "x")') is False

    def test_string(self):
        assert Lispy().eval_string('(string? "hello")') is True
        assert Lispy().eval_string("(string? 42)") is False

    def test_list(self):
        assert Lispy().eval_string("(list? (list 1 2))") is True
        assert Lispy().eval_string("(list? 42)") is False

    def test_nil(self):
        assert Lispy().eval_string("(nil? nil)") is True
        assert Lispy().eval_string("(nil? 0)") is False

    def test_bool(self):
        assert Lispy().eval_string("(bool? #t)") is True
        assert Lispy().eval_string("(bool? 1)") is False


# --- Integration ---

class TestIntegration:
    def test_fibonacci(self):
        result = Lispy(step_limit=10000).eval_string("""
            (begin
                (define fib (lambda (n)
                    (if (<= n 1) n
                        (+ (fib (- n 1)) (fib (- n 2))))))
                (fib 10))
        """)
        assert result == 55

    def test_colonist_decision(self):
        """Simulate a colonist personality evaluation."""
        vm = Lispy(seed=42)
        vm.global_env.set("danger", 0.8)
        vm.global_env.set("food", 30)
        vm.global_env.set("morale", 0.3)
        result = vm.eval_string(
            "(if (> danger 0.7) 'fortify (if (< food 50) 'farm 'explore))"
        )
        assert str(result) == "fortify"

    def test_governance_vote(self):
        """Simulate a governance vote."""
        result = Lispy(seed=42).eval_string("""
            (begin
                (define citizens (range 1 11))
                (define votes (map (lambda (c) (if (> (random) 0.5) 1 0)) citizens))
                (define total (reduce + votes 0))
                (> total 5))
        """)
        assert isinstance(result, bool)

    def test_recursive_governance_sim(self):
        """Test the actual recursive governance simulation pattern."""
        result = Lispy(seed=42, step_limit=20000).eval_string("""
            (sub-sim "governance"
                '(begin
                    (define citizens (range 1 6))
                    (define votes (map (lambda (c) (if (> (random) 0.5) 1 0)) citizens))
                    (define approval (/ (reduce + votes 0) (length citizens)))
                    (define future
                        (sub-sim "consequence"
                            '(begin
                                (define stability 0.5)
                                (define years (range 1 6))
                                (define final
                                    (reduce
                                        (lambda (s y) (+ (* s 0.9) (* (random) 0.2)))
                                        years
                                        stability))
                                final)))
                    (list approval future)))
        """)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_math_constants(self):
        assert Lispy().eval_string("pi") == pytest.approx(3.14159, abs=0.001)
        assert Lispy().eval_string("e") == pytest.approx(2.71828, abs=0.001)
