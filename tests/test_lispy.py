"""Tests for the safe LisPy interpreter."""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lispy import Lispy, LispyError, LispyTimeout, tokenize, parse, Symbol


# --- Tokenizer ---

class TestTokenizer:
    def test_simple_expr(self):
        assert tokenize("(+ 1 2)") == ["(", "+", "1", "2", ")"]

    def test_nested(self):
        tokens = tokenize("(if (> x 3) 'yes 'no)")
        assert tokens[0] == "("
        assert "if" in tokens
        assert tokens[-1] == ")"

    def test_string_literal(self):
        tokens = tokenize('(str "hello world")')
        assert '"hello world"' in tokens

    def test_comments_stripped(self):
        tokens = tokenize("; comment\n(+ 1 2)")
        assert ";" not in "".join(tokens)
        assert tokens == ["(", "+", "1", "2", ")"]

    def test_quote_shorthand(self):
        tokens = tokenize("'(1 2 3)")
        assert tokens[0] == "'"

    def test_booleans(self):
        tokens = tokenize("#t #f nil")
        assert tokens == ["#t", "#f", "nil"]

    def test_empty_input(self):
        assert tokenize("") == []
        assert tokenize("   ") == []
        assert tokenize("; just a comment") == []


# --- Parser ---

class TestParser:
    def test_integer(self):
        assert parse(tokenize("42")) == 42

    def test_float(self):
        assert parse(tokenize("3.14")) == 3.14

    def test_string(self):
        assert parse(tokenize('"hello"')) == "hello"

    def test_symbol(self):
        result = parse(tokenize("foo"))
        assert isinstance(result, Symbol)
        assert result.name == "foo"

    def test_booleans(self):
        assert parse(tokenize("#t")) is True
        assert parse(tokenize("#f")) is False

    def test_nil(self):
        assert parse(tokenize("nil")) is None

    def test_list(self):
        result = parse(tokenize("(+ 1 2)"))
        assert isinstance(result, list)
        assert len(result) == 3

    def test_nested_list(self):
        result = parse(tokenize("(if (> 1 2) 3 4)"))
        assert isinstance(result, list)
        assert isinstance(result[1], list)

    def test_quote(self):
        result = parse(tokenize("'(1 2 3)"))
        assert isinstance(result, list)
        assert result[0] == Symbol("quote")

    def test_empty_list(self):
        result = parse(tokenize("()"))
        assert result == []

    def test_missing_close_paren(self):
        with pytest.raises(LispyError, match="Missing closing"):
            parse(tokenize("(+ 1 2"))

    def test_unexpected_close_paren(self):
        with pytest.raises(LispyError, match="Unexpected"):
            parse(tokenize(")"))

    def test_empty_raises(self):
        with pytest.raises(LispyError, match="Unexpected EOF"):
            parse([])


# --- Evaluator: Atoms ---

class TestEvalAtoms:
    def test_integer(self):
        vm = Lispy()
        assert vm.eval_string("42") == 42

    def test_float(self):
        vm = Lispy()
        assert vm.eval_string("3.14") == 3.14

    def test_string(self):
        vm = Lispy()
        assert vm.eval_string('"hello"') == "hello"

    def test_bool(self):
        vm = Lispy()
        assert vm.eval_string("#t") is True
        assert vm.eval_string("#f") is False

    def test_nil(self):
        vm = Lispy()
        assert vm.eval_string("nil") is None


# --- Evaluator: Arithmetic ---

class TestArithmetic:
    def test_add(self):
        assert Lispy().eval_string("(+ 1 2)") == 3

    def test_add_many(self):
        assert Lispy().eval_string("(+ 1 2 3 4 5)") == 15

    def test_subtract(self):
        assert Lispy().eval_string("(- 10 3)") == 7

    def test_negate(self):
        assert Lispy().eval_string("(- 5)") == -5

    def test_multiply(self):
        assert Lispy().eval_string("(* 3 4)") == 12

    def test_divide(self):
        assert Lispy().eval_string("(/ 10 3)") == pytest.approx(3.333, abs=0.01)

    def test_divide_by_zero(self):
        result = Lispy().eval_string("(/ 1 0)")
        assert result == float("inf")

    def test_mod(self):
        assert Lispy().eval_string("(mod 10 3)") == 1

    def test_abs(self):
        assert Lispy().eval_string("(abs -5)") == 5

    def test_min_max(self):
        assert Lispy().eval_string("(min 1 2 3)") == 1
        assert Lispy().eval_string("(max 1 2 3)") == 3

    def test_floor_ceil(self):
        assert Lispy().eval_string("(floor 3.7)") == 3
        assert Lispy().eval_string("(ceil 3.2)") == 4

    def test_sqrt(self):
        assert Lispy().eval_string("(sqrt 16)") == 4.0

    def test_pow(self):
        assert Lispy().eval_string("(pow 2 10)") == 1024

    def test_log(self):
        result = Lispy().eval_string("(log 1)")
        assert result == pytest.approx(0.0, abs=0.001)

    def test_nested_arithmetic(self):
        assert Lispy().eval_string("(+ (* 2 3) (- 10 4))") == 12


# --- Evaluator: Comparisons ---

class TestComparisons:
    def test_less_than(self):
        assert Lispy().eval_string("(< 1 2)") is True
        assert Lispy().eval_string("(< 2 1)") is False

    def test_greater_than(self):
        assert Lispy().eval_string("(> 2 1)") is True

    def test_equal(self):
        assert Lispy().eval_string("(= 1 1)") is True
        assert Lispy().eval_string("(= 1 2)") is False

    def test_not_equal(self):
        assert Lispy().eval_string("(!= 1 2)") is True

    def test_leq_geq(self):
        assert Lispy().eval_string("(<= 1 1)") is True
        assert Lispy().eval_string("(>= 2 1)") is True


# --- Evaluator: Logic ---

class TestLogic:
    def test_not(self):
        assert Lispy().eval_string("(not #t)") is False
        assert Lispy().eval_string("(not #f)") is True

    def test_and(self):
        assert Lispy().eval_string("(and #t #t)") is True
        assert Lispy().eval_string("(and #t #f)") is False

    def test_or(self):
        assert Lispy().eval_string("(or #f #t)") is True
        assert Lispy().eval_string("(or #f #f)") is False

    def test_and_short_circuit(self):
        vm = Lispy()
        assert vm.eval_string("(and #f (/ 1 0))") is False

    def test_or_short_circuit(self):
        vm = Lispy()
        result = vm.eval_string("(or #t (/ 1 0))")
        assert result is True


# --- Evaluator: Control Flow ---

class TestControlFlow:
    def test_if_true(self):
        assert Lispy().eval_string("(if #t 1 2)") == 1

    def test_if_false(self):
        assert Lispy().eval_string("(if #f 1 2)") == 2

    def test_if_no_else(self):
        assert Lispy().eval_string("(if #f 1)") is None

    def test_cond(self):
        result = Lispy().eval_string("(cond ((= 1 2) 'a) ((= 1 1) 'b) (else 'c))")
        assert isinstance(result, Symbol)
        assert result.name == "b"

    def test_cond_else(self):
        result = Lispy().eval_string("(cond ((= 1 2) 'a) (else 'fallback))")
        assert isinstance(result, Symbol)
        assert result.name == "fallback"

    def test_when_true(self):
        assert Lispy().eval_string("(when #t 42)") == 42

    def test_when_false(self):
        assert Lispy().eval_string("(when #f 42)") is None

    def test_begin(self):
        assert Lispy().eval_string("(begin 1 2 3)") == 3


# --- Evaluator: Bindings ---

class TestBindings:
    def test_define(self):
        vm = Lispy()
        vm.eval_string("(define x 42)")
        assert vm.eval_string("x") == 42

    def test_define_function(self):
        vm = Lispy()
        vm.eval_string("(define (square x) (* x x))")
        assert vm.eval_string("(square 5)") == 25

    def test_lambda(self):
        vm = Lispy()
        vm.eval_string("(define double (lambda (x) (* x 2)))")
        assert vm.eval_string("(double 7)") == 14

    def test_let(self):
        assert Lispy().eval_string("(let ((x 10) (y 20)) (+ x y))") == 30

    def test_let_scoping(self):
        vm = Lispy()
        vm.eval_string("(define x 1)")
        result = vm.eval_string("(let ((x 100)) x)")
        assert result == 100
        assert vm.eval_string("x") == 1

    def test_set_bang(self):
        vm = Lispy()
        vm.eval_string("(define x 1)")
        vm.eval_string("(set! x 42)")
        assert vm.eval_string("x") == 42

    def test_set_bang_unbound(self):
        with pytest.raises(LispyError, match="unbound"):
            Lispy().eval_string("(set! nonexistent 1)")

    def test_closure(self):
        vm = Lispy()
        vm.eval_string("(define (make-adder n) (lambda (x) (+ n x)))")
        vm.eval_string("(define add5 (make-adder 5))")
        assert vm.eval_string("(add5 10)") == 15

    def test_recursion(self):
        vm = Lispy()
        vm.eval_string("""
            (define (factorial n)
                (if (<= n 1) 1 (* n (factorial (- n 1)))))
        """)
        assert vm.eval_string("(factorial 5)") == 120


# --- Evaluator: Lists ---

class TestLists:
    def test_list_create(self):
        assert Lispy().eval_string("(list 1 2 3)") == [1, 2, 3]

    def test_car(self):
        assert Lispy().eval_string("(car (list 1 2 3))") == 1

    def test_cdr(self):
        assert Lispy().eval_string("(cdr (list 1 2 3))") == [2, 3]

    def test_cons(self):
        assert Lispy().eval_string("(cons 0 (list 1 2))") == [0, 1, 2]

    def test_length(self):
        assert Lispy().eval_string("(length (list 1 2 3))") == 3

    def test_nth(self):
        assert Lispy().eval_string("(nth (list 10 20 30) 1)") == 20

    def test_append(self):
        assert Lispy().eval_string("(append (list 1 2) (list 3 4))") == [1, 2, 3, 4]

    def test_reverse(self):
        assert Lispy().eval_string("(reverse (list 1 2 3))") == [3, 2, 1]

    def test_range(self):
        assert Lispy().eval_string("(range 5)") == [0, 1, 2, 3, 4]
        assert Lispy().eval_string("(range 2 5)") == [2, 3, 4]

    def test_sort(self):
        assert Lispy().eval_string("(sort (list 3 1 2))") == [1, 2, 3]

    def test_quote(self):
        result = Lispy().eval_string("'(1 2 3)")
        assert result == [1, 2, 3]

    def test_map(self):
        vm = Lispy()
        result = vm.eval_string("(map (lambda (x) (* x 2)) (list 1 2 3))")
        assert result == [2, 4, 6]

    def test_filter(self):
        vm = Lispy()
        result = vm.eval_string("(filter (lambda (x) (> x 2)) (list 1 2 3 4))")
        assert result == [3, 4]

    def test_reduce(self):
        vm = Lispy()
        result = vm.eval_string("(reduce + (list 1 2 3 4) 0)")
        assert result == 10


# --- Evaluator: Strings ---

class TestStrings:
    def test_str_append(self):
        assert Lispy().eval_string('(str-append "hello" " " "world")') == "hello world"

    def test_str_length(self):
        assert Lispy().eval_string('(str-length "hello")') == 5

    def test_str_upper(self):
        assert Lispy().eval_string('(str-upper "hello")') == "HELLO"

    def test_str_contains(self):
        assert Lispy().eval_string('(str-contains "hello world" "world")') is True

    def test_str_split(self):
        assert Lispy().eval_string('(str-split "a,b,c" ",")') == ["a", "b", "c"]


# --- Evaluator: Dicts ---

class TestDicts:
    def test_make_dict(self):
        result = Lispy().eval_string('(make-dict "a" 1 "b" 2)')
        assert result == {"a": 1, "b": 2}

    def test_dict_get(self):
        vm = Lispy()
        vm.eval_string('(define d (make-dict "x" 42))')
        assert vm.eval_string('(dict-get d "x")') == 42

    def test_dict_get_default(self):
        vm = Lispy()
        vm.eval_string('(define d (make-dict "x" 1))')
        assert vm.eval_string('(dict-get d "y" 99)') == 99

    def test_dict_set(self):
        vm = Lispy()
        vm.eval_string('(define d (make-dict "a" 1))')
        result = vm.eval_string('(dict-set d "b" 2)')
        assert result == {"a": 1, "b": 2}

    def test_dict_keys(self):
        result = Lispy().eval_string('(dict-keys (make-dict "a" 1 "b" 2))')
        assert sorted(result) == ["a", "b"]


# --- Evaluator: Random (deterministic) ---

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
        result = Lispy(seed=42).eval_string("(random-int 1 10)")
        assert 1 <= result <= 10

    def test_random_choice(self):
        result = Lispy(seed=42).eval_string("(random-choice (list 'a 'b 'c))")
        assert isinstance(result, Symbol)


# --- Safety ---

class TestSafety:
    def test_step_limit(self):
        with pytest.raises(LispyTimeout):
            vm = Lispy(step_limit=100)
            vm.eval_string("""
                (begin
                    (define (loop n) (if (> n 0) (loop (- n 1)) 0))
                    (loop 10000))
            """)

    def test_recursion_depth(self):
        with pytest.raises(LispyError, match="Recursion depth"):
            vm = Lispy(step_limit=100000)
            vm.eval_string("""
                (begin
                    (define (deep n) (deep (+ n 1)))
                    (deep 0))
            """)

    def test_unbound_symbol(self):
        with pytest.raises(LispyError, match="Unbound symbol"):
            Lispy().eval_string("nonexistent_var")

    def test_arity_mismatch(self):
        with pytest.raises(LispyError, match="Arity"):
            vm = Lispy()
            vm.eval_string("(define (f x) x)")
            vm.eval_string("(f 1 2)")

    def test_no_io_builtins(self):
        """Verify no dangerous builtins exist."""
        vm = Lispy()
        dangerous = ["open", "read-file", "write-file", "exec", "system",
                      "import", "require", "eval-python", "shell"]
        for name in dangerous:
            with pytest.raises(LispyError, match="Unbound"):
                vm.eval_string(f"({name})")


# --- Sub-simulations ---

class TestSubSim:
    def test_basic_sub_sim(self):
        vm = Lispy(seed=42, sim_depth=0)
        result = vm.eval_string("(sub-sim '(+ 10 20))")
        assert result == 30

    def test_labeled_sub_sim(self):
        vm = Lispy(seed=42, sim_depth=0)
        result = vm.eval_string('(sub-sim "test-label" \'(* 3 4))')
        assert result == 12
        assert len(vm.sub_sim_log) == 1
        assert vm.sub_sim_log[0]["label"] == "test-label"

    def test_sub_sim_depth_limit(self):
        vm = Lispy(seed=42, sim_depth=3)
        with pytest.raises(LispyError, match="depth limit"):
            vm.eval_string("(sub-sim '(+ 1 1))")

    def test_depth_2_sub_sim(self):
        vm = Lispy(seed=42, sim_depth=0, step_limit=20000)
        result = vm.eval_string("""
            (sub-sim "level-1"
                '(begin
                    (define x 10)
                    (define y (sub-sim "level-2" '(+ 5 5)))
                    (+ x y)))
        """)
        assert result == 20

    def test_sub_sim_logging(self):
        vm = Lispy(seed=42, sim_depth=0)
        vm.eval_string("(sub-sim '(+ 1 1))")
        assert len(vm.sub_sim_log) == 1
        log = vm.sub_sim_log[0]
        assert log["depth"] == 1
        assert "result" in log
        assert "steps" in log

    def test_sub_sim_isolation(self):
        """Sub-sim cannot affect parent environment."""
        vm = Lispy(seed=42, sim_depth=0)
        vm.eval_string("(define x 100)")
        vm.eval_string("(sub-sim '(define x 999))")
        assert vm.eval_string("x") == 100

    def test_sub_sim_deterministic(self):
        a = Lispy(seed=42, sim_depth=0)
        b = Lispy(seed=42, sim_depth=0)
        ra = a.eval_string("(sub-sim '(random-int 1 100))")
        rb = b.eval_string("(sub-sim '(random-int 1 100))")
        assert ra == rb


# --- Type checks ---

class TestTypeChecks:
    def test_number(self):
        assert Lispy().eval_string("(number? 42)") is True
        assert Lispy().eval_string('(number? "hi")') is False

    def test_string(self):
        assert Lispy().eval_string('(string? "hi")') is True

    def test_list(self):
        assert Lispy().eval_string("(list? (list 1 2))") is True

    def test_nil(self):
        assert Lispy().eval_string("(nil? nil)") is True

    def test_bool(self):
        assert Lispy().eval_string("(bool? #t)") is True


# --- Integration ---

class TestIntegration:
    def test_fibonacci(self):
        vm = Lispy()
        vm.eval_string("""
            (define (fib n)
                (if (<= n 1) n (+ (fib (- n 1)) (fib (- n 2)))))
        """)
        assert vm.eval_string("(fib 10)") == 55

    def test_colonist_decision(self):
        """Simulate a colonist making a decision."""
        vm = Lispy(seed=42)
        vm.eval_string("""
            (define danger 0.8)
            (define food 30)
            (define resolve 0.9)
        """)
        result = vm.eval_string("""
            (if (> danger 0.7)
                'fortify
                (if (< food 50) 'ration 'terraform))
        """)
        assert isinstance(result, Symbol)
        assert result.name == "fortify"

    def test_governance_vote(self):
        """Simulate a governance vote."""
        vm = Lispy(seed=42)
        result = vm.eval_string("""
            (begin
                (define citizens (range 1 11))
                (define votes (map (lambda (c) (if (> (random) 0.5) 1 0)) citizens))
                (define total (reduce + votes 0))
                (> total 5))
        """)
        assert isinstance(result, bool)

    def test_recursive_governance_sim(self):
        """Test the full recursive simulation pattern."""
        vm = Lispy(seed=42, sim_depth=0, step_limit=30000)
        result = vm.eval_string("""
            (begin
                (define proposal-test
                    (sub-sim "governance"
                        '(begin
                            (define votes (map (lambda (x) (random-int 0 1)) (range 10)))
                            (define approval (/ (reduce + votes 0) 10))
                            (list approval (> approval 0.5)))))
                proposal-test)
        """)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_math_constants(self):
        vm = Lispy()
        pi = vm.eval_string("pi")
        assert pi == pytest.approx(3.14159, abs=0.001)
