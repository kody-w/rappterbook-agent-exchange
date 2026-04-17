"""
test_lispy.py — Tests for the pure-computation LisPy interpreter.

Covers: parsing, arithmetic, comparisons, logic, lambdas, closures,
let bindings, list ops, dict ops, sub-sim spawning, depth limits,
step limits, error handling, homoiconicity.
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lispy import Lispy, LispError, Symbol, Closure, parse, tokenize, to_sexp, NIL


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_simple_expression(self):
        tokens = tokenize("(+ 1 2)")
        assert tokens == ["(", "+", "1", "2", ")"]

    def test_nested_expression(self):
        tokens = tokenize("(+ (* 2 3) 4)")
        assert tokens == ["(", "+", "(", "*", "2", "3", ")", "4", ")"]

    def test_string_literal(self):
        tokens = tokenize('(print "hello world")')
        assert '"hello world"' in tokens

    def test_comment_ignored(self):
        tokens = tokenize("; this is a comment\n(+ 1 2)")
        assert tokens == ["(", "+", "1", "2", ")"]

    def test_quote_shorthand(self):
        tokens = tokenize("'(1 2 3)")
        assert tokens[0] == "'"

    def test_empty_input(self):
        assert tokenize("") == []

    def test_whitespace_variants(self):
        tokens = tokenize("(+\t1\n2\r3)")
        assert tokens == ["(", "+", "1", "2", "3", ")"]


class TestParser:
    def test_integer(self):
        forms = parse("42")
        assert forms == [42]

    def test_float(self):
        forms = parse("3.14")
        assert forms == [3.14]

    def test_string(self):
        forms = parse('"hello"')
        assert forms == ["hello"]

    def test_symbol(self):
        forms = parse("foo")
        assert len(forms) == 1
        assert isinstance(forms[0], Symbol)
        assert forms[0] == "foo"

    def test_booleans(self):
        assert parse("true") == [True]
        assert parse("false") == [False]

    def test_nil(self):
        assert parse("nil") == [NIL]

    def test_list(self):
        forms = parse("(1 2 3)")
        assert forms == [[1, 2, 3]]

    def test_nested_list(self):
        forms = parse("(+ (- 5 3) 1)")
        assert len(forms) == 1
        assert len(forms[0]) == 3

    def test_quote(self):
        forms = parse("'(1 2 3)")
        assert forms == [[Symbol("quote"), [1, 2, 3]]]

    def test_multiple_forms(self):
        forms = parse("1 2 3")
        assert forms == [1, 2, 3]

    def test_unclosed_paren(self):
        forms = parse("(+ 1 2")
        assert isinstance(forms[0], LispError)

    def test_unexpected_close(self):
        forms = parse(")")
        assert isinstance(forms[0], LispError)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

class TestArithmetic:
    def setup_method(self):
        self.vm = Lispy()

    def test_addition(self):
        assert self.vm.run("(+ 1 2)") == 3

    def test_addition_multiple(self):
        assert self.vm.run("(+ 1 2 3 4)") == 10

    def test_subtraction(self):
        assert self.vm.run("(- 10 3)") == 7

    def test_negation(self):
        assert self.vm.run("(- 5)") == -5

    def test_multiplication(self):
        assert self.vm.run("(* 3 4)") == 12

    def test_division(self):
        assert self.vm.run("(/ 10 3)") == pytest.approx(3.333, abs=0.01)

    def test_division_by_zero(self):
        result = self.vm.run("(/ 1 0)")
        assert isinstance(result, LispError)

    def test_modulo(self):
        assert self.vm.run("(mod 10 3)") == 1

    def test_nested_arithmetic(self):
        assert self.vm.run("(+ (* 2 3) (- 10 4))") == 12

    def test_abs(self):
        assert self.vm.run("(abs -5)") == 5

    def test_min_max(self):
        assert self.vm.run("(min 3 1 4 1 5)") == 1
        assert self.vm.run("(max 3 1 4 1 5)") == 5

    def test_floor_ceil(self):
        assert self.vm.run("(floor 3.7)") == 3
        assert self.vm.run("(ceil 3.2)") == 4

    def test_sqrt(self):
        assert self.vm.run("(sqrt 16)") == 4.0

    def test_sqrt_negative(self):
        result = self.vm.run("(sqrt -1)")
        assert isinstance(result, LispError)

    def test_pow(self):
        assert self.vm.run("(pow 2 10)") == 1024


# ---------------------------------------------------------------------------
# Comparisons and Logic
# ---------------------------------------------------------------------------

class TestComparisons:
    def setup_method(self):
        self.vm = Lispy()

    def test_equal(self):
        assert self.vm.run("(= 1 1)") is True
        assert self.vm.run("(= 1 2)") is False

    def test_not_equal(self):
        assert self.vm.run("(!= 1 2)") is True

    def test_less_than(self):
        assert self.vm.run("(< 1 2)") is True
        assert self.vm.run("(< 2 1)") is False

    def test_greater_than(self):
        assert self.vm.run("(> 5 3)") is True

    def test_less_equal(self):
        assert self.vm.run("(<= 3 3)") is True
        assert self.vm.run("(<= 4 3)") is False

    def test_and(self):
        assert self.vm.run("(and true true)") is True
        assert self.vm.run("(and true false)") is False

    def test_or(self):
        assert self.vm.run("(or false true)") is True
        assert self.vm.run("(or false false)") is False

    def test_not(self):
        assert self.vm.run("(not true)") is False
        assert self.vm.run("(not false)") is True

    def test_if_true(self):
        assert self.vm.run("(if true 1 2)") == 1

    def test_if_false(self):
        assert self.vm.run("(if false 1 2)") == 2

    def test_if_no_else(self):
        result = self.vm.run("(if false 1)")
        assert result == NIL

    def test_cond(self):
        result = self.vm.run("""
            (cond
                ((= 1 2) "no")
                ((= 1 1) "yes")
                (true "fallback"))
        """)
        assert result == "yes"


# ---------------------------------------------------------------------------
# Variables and Functions
# ---------------------------------------------------------------------------

class TestVariablesAndFunctions:
    def setup_method(self):
        self.vm = Lispy()

    def test_define(self):
        assert self.vm.run("(begin (define x 42) x)") == 42

    def test_define_expression(self):
        assert self.vm.run("(begin (define x (+ 1 2)) x)") == 3

    def test_set_bang(self):
        assert self.vm.run("(begin (define x 1) (set! x 2) x)") == 2

    def test_set_bang_unbound(self):
        result = self.vm.run("(set! nonexistent 1)")
        assert isinstance(result, LispError)

    def test_lambda_basic(self):
        assert self.vm.run("((lambda (x) (+ x 1)) 5)") == 6

    def test_lambda_multiple_args(self):
        assert self.vm.run("((lambda (a b) (* a b)) 3 4)") == 12

    def test_closure(self):
        result = self.vm.run("""
            (begin
                (define make-adder (lambda (n) (lambda (x) (+ x n))))
                (define add5 (make-adder 5))
                (add5 10))
        """)
        assert result == 15

    def test_let_binding(self):
        assert self.vm.run("(let ((x 10) (y 20)) (+ x y))") == 30

    def test_let_sequential(self):
        assert self.vm.run("(let ((x 5) (y (* x 2))) (+ x y))") == 15

    def test_arity_mismatch(self):
        result = self.vm.run("((lambda (x) x) 1 2)")
        assert isinstance(result, LispError)

    def test_begin_returns_last(self):
        assert self.vm.run("(begin 1 2 3)") == 3

    def test_begin_empty(self):
        assert self.vm.run("(begin)") == NIL

    def test_do_loop(self):
        result = self.vm.run("""
            (begin
                (define sum 0)
                (do i 0 5 (set! sum (+ sum i)))
                sum)
        """)
        assert result == 10  # 0+1+2+3+4


# ---------------------------------------------------------------------------
# List Operations
# ---------------------------------------------------------------------------

class TestListOps:
    def setup_method(self):
        self.vm = Lispy()

    def test_list_create(self):
        assert self.vm.run("(list 1 2 3)") == [1, 2, 3]

    def test_car(self):
        assert self.vm.run("(car (list 1 2 3))") == 1

    def test_cdr(self):
        assert self.vm.run("(cdr (list 1 2 3))") == [2, 3]

    def test_cons(self):
        assert self.vm.run("(cons 0 (list 1 2))") == [0, 1, 2]

    def test_length(self):
        assert self.vm.run("(length (list 1 2 3))") == 3

    def test_nth(self):
        assert self.vm.run("(nth (list 10 20 30) 1)") == 20

    def test_nth_out_of_bounds(self):
        result = self.vm.run("(nth (list 1) 5)")
        assert isinstance(result, LispError)

    def test_append(self):
        assert self.vm.run("(append (list 1 2) (list 3 4))") == [1, 2, 3, 4]

    def test_reverse(self):
        assert self.vm.run("(reverse (list 1 2 3))") == [3, 2, 1]

    def test_range(self):
        assert self.vm.run("(range 5)") == [0, 1, 2, 3, 4]
        assert self.vm.run("(range 2 5)") == [2, 3, 4]

    def test_sort(self):
        assert self.vm.run("(sort (list 3 1 4 1 5))") == [1, 1, 3, 4, 5]

    def test_contains(self):
        assert self.vm.run("(contains (list 1 2 3) 2)") is True
        assert self.vm.run("(contains (list 1 2 3) 9)") is False

    def test_map(self):
        result = self.vm.run("(map (lambda (x) (* x 2)) (list 1 2 3))")
        assert result == [2, 4, 6]

    def test_filter(self):
        result = self.vm.run("(filter (lambda (x) (> x 2)) (list 1 2 3 4 5))")
        assert result == [3, 4, 5]

    def test_reduce(self):
        result = self.vm.run("(reduce (lambda (acc x) (+ acc x)) 0 (list 1 2 3 4))")
        assert result == 10

    def test_quote(self):
        result = self.vm.run("'(1 2 3)")
        assert result == [1, 2, 3]

    def test_quote_preserves_symbols(self):
        result = self.vm.run("'(a b c)")
        assert all(isinstance(x, Symbol) for x in result)


# ---------------------------------------------------------------------------
# Dict Operations
# ---------------------------------------------------------------------------

class TestDictOps:
    def setup_method(self):
        self.vm = Lispy()

    def test_dict_create(self):
        result = self.vm.run('(dict "a" 1 "b" 2)')
        assert result == {"a": 1, "b": 2}

    def test_dict_get(self):
        assert self.vm.run('(get (dict "x" 42) "x")') == 42

    def test_dict_get_default(self):
        assert self.vm.run('(get (dict "x" 42) "y" 0)') == 0

    def test_dict_assoc(self):
        result = self.vm.run('(assoc (dict "a" 1) "b" 2)')
        assert result == {"a": 1, "b": 2}

    def test_dict_dissoc(self):
        result = self.vm.run('(dissoc (dict "a" 1 "b" 2) "a")')
        assert result == {"b": 2}

    def test_dict_keys(self):
        result = self.vm.run('(sort (keys (dict "b" 2 "a" 1)))')
        assert result == ["a", "b"]

    def test_dict_values(self):
        result = self.vm.run('(sort (values (dict "a" 1 "b" 2)))')
        assert result == [1, 2]

    def test_dict_merge(self):
        result = self.vm.run('(merge (dict "a" 1) (dict "b" 2))')
        assert result == {"a": 1, "b": 2}

    def test_dict_merge_override(self):
        result = self.vm.run('(merge (dict "a" 1) (dict "a" 2))')
        assert result == {"a": 2}


# ---------------------------------------------------------------------------
# Sub-simulation
# ---------------------------------------------------------------------------

class TestSubSim:
    def setup_method(self):
        self.vm = Lispy(max_depth=3)

    def test_sub_sim_basic(self):
        result = self.vm.run("(sub-sim (+ 1 2))")
        assert result == 3

    def test_sub_sim_reads_parent_env(self):
        result = self.vm.run("(begin (define x 10) (sub-sim (+ x 5)))")
        assert result == 15

    def test_sub_sim_cannot_mutate_parent(self):
        result = self.vm.run("""
            (begin
                (define x 10)
                (sub-sim (begin (define x 99) x))
                x)
        """)
        assert result == 10  # parent's x unchanged

    def test_sub_sim_depth_2(self):
        result = self.vm.run("(sub-sim (sub-sim (+ 1 1)))")
        assert result == 2

    def test_sub_sim_depth_3(self):
        result = self.vm.run("(sub-sim (sub-sim (sub-sim (+ 1 1))))")
        assert result == 2

    def test_sub_sim_depth_limit(self):
        result = self.vm.run(
            "(sub-sim (sub-sim (sub-sim (sub-sim (+ 1 1)))))"
        )
        assert isinstance(result, LispError)
        assert "depth limit" in result.message

    def test_sub_sim_shares_step_budget(self):
        vm = Lispy(max_steps=100, max_depth=3)
        # Each sub-sim eats from the shared budget
        result = vm.run("(sub-sim (sub-sim (+ 1 1)))")
        assert result == 2
        assert vm.step_budget[0] < 100  # some steps consumed


# ---------------------------------------------------------------------------
# Execution Limits
# ---------------------------------------------------------------------------

class TestLimits:
    def test_step_limit(self):
        vm = Lispy(max_steps=10)
        result = vm.run("(do i 0 1000 (+ i 1))")
        assert isinstance(result, LispError)
        assert "step limit" in result.message

    def test_large_computation_succeeds(self):
        vm = Lispy(max_steps=100000)
        result = vm.run("(reduce (lambda (a x) (+ a x)) 0 (range 100))")
        assert result == 4950  # sum 0..99


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------

class TestErrors:
    def setup_method(self):
        self.vm = Lispy()

    def test_unbound_symbol(self):
        result = self.vm.run("nonexistent")
        assert isinstance(result, LispError)

    def test_error_function(self):
        result = self.vm.run('(error "boom")')
        assert isinstance(result, LispError)
        assert result.message == "boom"

    def test_error_check(self):
        assert self.vm.run('(error? (error "x"))') is True
        assert self.vm.run("(error? 42)") is False

    def test_not_callable(self):
        result = self.vm.run("(42 1 2)")
        assert isinstance(result, LispError)

    def test_error_propagates(self):
        result = self.vm.run("(+ 1 (/ 1 0))")
        assert isinstance(result, LispError)


# ---------------------------------------------------------------------------
# Type Checks
# ---------------------------------------------------------------------------

class TestTypeChecks:
    def setup_method(self):
        self.vm = Lispy()

    def test_number(self):
        assert self.vm.run("(number? 42)") is True
        assert self.vm.run('(number? "hi")') is False

    def test_string(self):
        assert self.vm.run('(string? "hi")') is True
        assert self.vm.run("(string? 42)") is False

    def test_list(self):
        assert self.vm.run("(list? (list 1 2))") is True
        assert self.vm.run("(list? 42)") is False

    def test_dict(self):
        assert self.vm.run('(dict? (dict "a" 1))') is True

    def test_type_names(self):
        assert self.vm.run("(type 42)") == "int"
        assert self.vm.run("(type 3.14)") == "float"
        assert self.vm.run('(type "hello")') == "string"
        assert self.vm.run("(type (list 1))") == "list"
        assert self.vm.run('(type (dict "a" 1))') == "dict"
        assert self.vm.run("(type true)") == "bool"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_sexp_int(self):
        assert to_sexp(42) == "42"

    def test_to_sexp_float(self):
        assert to_sexp(3.14) == "3.14"

    def test_to_sexp_string(self):
        assert to_sexp("hello") == '"hello"'

    def test_to_sexp_list(self):
        assert to_sexp([1, 2, 3]) == "(1 2 3)"

    def test_to_sexp_nil(self):
        assert to_sexp(NIL) == "nil"
        assert to_sexp(None) == "nil"

    def test_to_sexp_bool(self):
        assert to_sexp(True) == "true"
        assert to_sexp(False) == "false"

    def test_to_sexp_dict(self):
        result = to_sexp({"a": 1})
        assert "dict" in result
        assert '"a"' in result

    def test_to_sexp_error(self):
        assert to_sexp(LispError("oops")) == '(error "oops")'

    def test_roundtrip_simple(self):
        vm = Lispy()
        original = [1, "hello", True]
        sexp = to_sexp(original)
        # The sexp is a list literal like (1 "hello" true)
        # Evaluating it would try to call 1 as a function
        # So we use quote to preserve it as data
        result = vm.run(f"'{sexp}")
        assert result == original


# ---------------------------------------------------------------------------
# Integration: Colonist Brain Programs
# ---------------------------------------------------------------------------

class TestColonistBrains:
    """Test the kind of LisPy programs Mars-100 colonists actually run."""

    def setup_method(self):
        self.vm = Lispy(max_steps=10000, max_depth=3)
        # Set up a typical colonist environment
        self.vm.global_env.set("colony-food", 500.0)
        self.vm.global_env.set("colony-water", 1000.0)
        self.vm.global_env.set("colony-morale", 0.6)
        self.vm.global_env.set("colony-integrity", 0.9)
        self.vm.global_env.set("colony-terraform", 0.001)
        self.vm.global_env.set("event-name", "dust_storm")
        self.vm.global_env.set("event-severity", 0.5)
        self.vm.global_env.set("year", 15)
        self.vm.global_env.set("my-name", "Ares")
        self.vm.global_env.set("my-resolve", 0.7)
        self.vm.global_env.set("my-improvisation", 0.5)
        self.vm.global_env.set("my-empathy", 0.6)
        self.vm.global_env.set("my-hoarding", 0.3)
        self.vm.global_env.set("my-faith", 0.4)
        self.vm.global_env.set("my-paranoia", 0.2)
        self.vm.global_env.set("my-terraforming", 0.3)
        self.vm.global_env.set("my-hydroponics", 0.5)
        self.vm.global_env.set("my-mediation", 0.4)
        self.vm.global_env.set("my-coding", 0.7)
        self.vm.global_env.set("my-prayer", 0.2)
        self.vm.global_env.set("my-sabotage", 0.1)

    def test_repair_decision(self):
        result = self.vm.run("""
            (let ((cost (* event-severity 50))
                  (benefit (* my-coding 0.8))
                  (risk (if (> event-severity 0.7) 0.3 0.1)))
              (if (> benefit risk)
                (list "repair" "worth_it" (- benefit risk))
                (list "repair" "risky" risk)))
        """)
        assert isinstance(result, list)
        assert result[0] == "repair"
        assert result[1] == "worth_it"  # coding 0.7 * 0.8 = 0.56 > 0.1

    def test_farm_decision(self):
        result = self.vm.run("""
            (let ((yield (* my-hydroponics 30))
                  (food-need (* 10 1.8))
                  (surplus (- (+ colony-food yield) food-need)))
              (if (> surplus 0)
                (list "farm" "surplus" surplus)
                (list "farm" "deficit" surplus)))
        """)
        assert isinstance(result, list)
        assert result[0] == "farm"

    def test_terraform_with_sub_sim(self):
        """A colonist runs a sub-sim to project terraforming progress."""
        result = self.vm.run("""
            (let ((progress (* my-terraforming 0.001))
                  (cost (* 20 (- 1 my-resolve))))
              (sub-sim
                (let ((projected (+ colony-terraform progress)))
                  (if (> projected 0.01)
                    (list "terraform" "milestone" projected)
                    (list "terraform" "incremental" projected)))))
        """)
        assert isinstance(result, list)
        assert result[0] == "terraform"

    def test_nested_sub_sim_governance(self):
        """A colonist models a governance vote in a sub-sim within a sub-sim."""
        result = self.vm.run("""
            (sub-sim
              (let ((proposal-support (* colony-morale my-mediation)))
                (sub-sim
                  (if (> proposal-support 0.3)
                    (list "governance" "viable" proposal-support)
                    (list "governance" "doomed" proposal-support)))))
        """)
        assert isinstance(result, list)
        assert result[0] == "governance"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_program_same_result(self):
        """Same program should produce identical results."""
        program = """
            (begin
                (define fib (lambda (n)
                    (if (<= n 1) n
                        (+ (fib (- n 1)) (fib (- n 2))))))
                (fib 10))
        """
        vm1 = Lispy()
        vm2 = Lispy()
        assert vm1.run(program) == vm2.run(program) == 55
