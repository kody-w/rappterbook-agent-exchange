"""test_lispy_vm.py — Comprehensive tests for the LisPy safe-eval interpreter.

Covers: parsing, atoms, arithmetic, comparison, boolean, list ops, dict ops,
string ops, special forms (define, if, cond, let, lambda, begin), map/filter/reduce,
sub-sim recursion, governance primitives, safety limits, and invariants.
"""
from __future__ import annotations

import math
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lispy_vm import (
    LispyVM, LispyError, LispyDepthError, LispyStepLimitError,
    LispySafetyError, Symbol, Lambda, Env,
    parse, tokenize, format_sexpr, _serialize_value,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vm() -> LispyVM:
    """Fresh VM for each test."""
    return LispyVM(max_depth=3, max_steps=10_000)


@pytest.fixture
def vm_small() -> LispyVM:
    """VM with small step limit for safety tests."""
    return LispyVM(max_depth=3, max_steps=100)


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_simple_expression(self):
        assert tokenize("(+ 1 2)") == ["(", "+", "1", "2", ")"]

    def test_nested(self):
        tokens = tokenize("(+ (* 2 3) 4)")
        assert tokens == ["(", "+", "(", "*", "2", "3", ")", "4", ")"]

    def test_string_literal(self):
        tokens = tokenize('(str "hello world")')
        assert '"hello world"' in tokens

    def test_comment_ignored(self):
        tokens = tokenize("; this is a comment\n(+ 1 2)")
        assert ";" not in "".join(tokens)
        assert tokens == ["(", "+", "1", "2", ")"]

    def test_quote_shorthand(self):
        tokens = tokenize("'(1 2 3)")
        assert tokens[0] == "'("

    def test_empty_string(self):
        assert tokenize("") == []

    def test_whitespace_only(self):
        assert tokenize("   \n\t  ") == []


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParser:
    def test_integer(self):
        assert parse("42") == [42]

    def test_float(self):
        assert parse("3.14") == [3.14]

    def test_string(self):
        assert parse('"hello"') == ["hello"]

    def test_bool_true(self):
        assert parse("#t") == [True]

    def test_bool_false(self):
        assert parse("#f") == [False]

    def test_nil(self):
        assert parse("nil") == [None]

    def test_symbol(self):
        result = parse("foo")
        assert len(result) == 1
        assert isinstance(result[0], Symbol)
        assert result[0] == "foo"

    def test_list(self):
        result = parse("(1 2 3)")
        assert result == [[1, 2, 3]]

    def test_nested_list(self):
        result = parse("(+ (- 3 1) 2)")
        assert len(result) == 1
        assert result[0][0] == Symbol("+")

    def test_multiple_expressions(self):
        result = parse("1 2 3")
        assert result == [1, 2, 3]

    def test_quote(self):
        result = parse("'(1 2 3)")
        assert result == [[Symbol("quote"), [1, 2, 3]]]

    def test_unmatched_paren(self):
        with pytest.raises(LispyError, match="Missing closing"):
            parse("(+ 1 2")

    def test_extra_close_paren(self):
        with pytest.raises(LispyError, match="Unexpected closing"):
            parse(")")

    def test_empty_list(self):
        assert parse("()") == [[]]


# ---------------------------------------------------------------------------
# Arithmetic tests
# ---------------------------------------------------------------------------

class TestArithmetic:
    def test_add(self, vm):
        assert vm.eval_str("(+ 1 2)") == 3

    def test_add_multi(self, vm):
        assert vm.eval_str("(+ 1 2 3 4)") == 10

    def test_subtract(self, vm):
        assert vm.eval_str("(- 10 3)") == 7

    def test_negate(self, vm):
        assert vm.eval_str("(- 5)") == -5

    def test_multiply(self, vm):
        assert vm.eval_str("(* 3 4)") == 12

    def test_divide(self, vm):
        assert vm.eval_str("(/ 10 3)") == pytest.approx(3.333, abs=0.01)

    def test_divide_by_zero(self, vm):
        assert vm.eval_str("(/ 10 0)") == float("inf")

    def test_floor_div(self, vm):
        assert vm.eval_str("(// 7 2)") == 3

    def test_modulo(self, vm):
        assert vm.eval_str("(% 7 3)") == 1

    def test_abs(self, vm):
        assert vm.eval_str("(abs -5)") == 5

    def test_min(self, vm):
        assert vm.eval_str("(min 3 1 4 1 5)") == 1

    def test_max(self, vm):
        assert vm.eval_str("(max 3 1 4 1 5)") == 5

    def test_pow(self, vm):
        assert vm.eval_str("(pow 2 10)") == 1024

    def test_sqrt(self, vm):
        assert vm.eval_str("(sqrt 16)") == 4.0

    def test_floor(self, vm):
        assert vm.eval_str("(floor 3.7)") == 3

    def test_ceil(self, vm):
        assert vm.eval_str("(ceil 3.2)") == 4

    def test_round(self, vm):
        assert vm.eval_str("(round 3.5)") == 4

    def test_nested_arithmetic(self, vm):
        assert vm.eval_str("(+ (* 2 3) (- 10 4))") == 12

    def test_pi(self, vm):
        assert vm.eval_str("pi") == pytest.approx(math.pi)


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------

class TestComparison:
    def test_equal(self, vm):
        assert vm.eval_str("(= 3 3)") is True

    def test_not_equal(self, vm):
        assert vm.eval_str("(!= 3 4)") is True

    def test_less_than(self, vm):
        assert vm.eval_str("(< 3 4)") is True

    def test_greater_than(self, vm):
        assert vm.eval_str("(> 4 3)") is True

    def test_less_equal(self, vm):
        assert vm.eval_str("(<= 3 3)") is True

    def test_greater_equal(self, vm):
        assert vm.eval_str("(>= 4 3)") is True


# ---------------------------------------------------------------------------
# Boolean tests
# ---------------------------------------------------------------------------

class TestBoolean:
    def test_and_true(self, vm):
        assert vm.eval_str("(and #t #t)") is True

    def test_and_false(self, vm):
        assert vm.eval_str("(and #t #f)") is False

    def test_or_true(self, vm):
        assert vm.eval_str("(or #f #t)") is True

    def test_or_false(self, vm):
        assert vm.eval_str("(or #f #f)") is False

    def test_not(self, vm):
        assert vm.eval_str("(not #t)") is False


# ---------------------------------------------------------------------------
# List operations
# ---------------------------------------------------------------------------

class TestListOps:
    def test_list_create(self, vm):
        assert vm.eval_str("(list 1 2 3)") == [1, 2, 3]

    def test_car(self, vm):
        assert vm.eval_str("(car (list 1 2 3))") == 1

    def test_cdr(self, vm):
        assert vm.eval_str("(cdr (list 1 2 3))") == [2, 3]

    def test_cons(self, vm):
        assert vm.eval_str("(cons 0 (list 1 2))") == [0, 1, 2]

    def test_length(self, vm):
        assert vm.eval_str("(len (list 1 2 3 4))") == 4

    def test_nth(self, vm):
        assert vm.eval_str("(nth (list 10 20 30) 1)") == 20

    def test_append(self, vm):
        assert vm.eval_str("(append (list 1 2) (list 3 4))") == [1, 2, 3, 4]

    def test_reverse(self, vm):
        assert vm.eval_str("(reverse (list 1 2 3))") == [3, 2, 1]

    def test_range(self, vm):
        assert vm.eval_str("(range 5)") == [0, 1, 2, 3, 4]

    def test_empty_check(self, vm):
        assert vm.eval_str("(empty? (list))") is True
        assert vm.eval_str("(empty? (list 1))") is False

    def test_contains(self, vm):
        assert vm.eval_str("(contains? (list 1 2 3) 2)") is True
        assert vm.eval_str("(contains? (list 1 2 3) 9)") is False


# ---------------------------------------------------------------------------
# Dict operations
# ---------------------------------------------------------------------------

class TestDictOps:
    def test_dict_create(self, vm):
        result = vm.eval_str('(dict "a" 1 "b" 2)')
        assert result == {"a": 1, "b": 2}

    def test_get(self, vm):
        assert vm.eval_str('(get (dict "x" 42) "x")') == 42

    def test_get_default(self, vm):
        assert vm.eval_str('(get (dict "x" 42) "y" 0)') == 0

    def test_put(self, vm):
        result = vm.eval_str('(put (dict "a" 1) "b" 2)')
        assert result == {"a": 1, "b": 2}

    def test_keys(self, vm):
        result = vm.eval_str('(keys (dict "a" 1 "b" 2))')
        assert set(result) == {"a", "b"}

    def test_has(self, vm):
        assert vm.eval_str('(has? (dict "x" 1) "x")') is True
        assert vm.eval_str('(has? (dict "x" 1) "y")') is False


# ---------------------------------------------------------------------------
# String operations
# ---------------------------------------------------------------------------

class TestStringOps:
    def test_str_concat(self, vm):
        assert vm.eval_str('(str "hello" " " "world")') == "hello world"

    def test_str_join(self, vm):
        assert vm.eval_str('(str-join ", " (list "a" "b" "c"))') == "a, b, c"

    def test_str_upper(self, vm):
        assert vm.eval_str('(str-upper "hello")') == "HELLO"

    def test_str_lower(self, vm):
        assert vm.eval_str('(str-lower "HELLO")') == "hello"


# ---------------------------------------------------------------------------
# Type check tests
# ---------------------------------------------------------------------------

class TestTypeChecks:
    def test_number(self, vm):
        assert vm.eval_str("(number? 42)") is True
        assert vm.eval_str('(number? "hi")') is False

    def test_string(self, vm):
        assert vm.eval_str('(string? "hello")') is True

    def test_list(self, vm):
        assert vm.eval_str("(list? (list 1 2))") is True

    def test_dict(self, vm):
        assert vm.eval_str('(dict? (dict "a" 1))') is True

    def test_nil(self, vm):
        assert vm.eval_str("(nil? nil)") is True

    def test_bool(self, vm):
        assert vm.eval_str("(bool? #t)") is True


# ---------------------------------------------------------------------------
# Special forms
# ---------------------------------------------------------------------------

class TestSpecialForms:
    def test_define_var(self, vm):
        assert vm.eval_str("(define x 42) x") == 42

    def test_define_function(self, vm):
        assert vm.eval_str("(define (square n) (* n n)) (square 5)") == 25

    def test_set_bang(self, vm):
        assert vm.eval_str("(define x 1) (set! x 2) x") == 2

    def test_if_true(self, vm):
        assert vm.eval_str("(if #t 1 2)") == 1

    def test_if_false(self, vm):
        assert vm.eval_str("(if #f 1 2)") == 2

    def test_if_no_else(self, vm):
        assert vm.eval_str("(if #f 1)") is None

    def test_cond(self, vm):
        result = vm.eval_str("""
            (cond
                ((= 1 2) "nope")
                ((= 1 1) "yes")
                (else "default"))
        """)
        assert result == "yes"

    def test_cond_else(self, vm):
        result = vm.eval_str("""
            (cond
                ((= 1 2) "nope")
                (else "default"))
        """)
        assert result == "default"

    def test_let(self, vm):
        assert vm.eval_str("(let ((x 10) (y 20)) (+ x y))") == 30

    def test_let_star(self, vm):
        assert vm.eval_str("(let* ((x 10) (y (* x 2))) y)") == 20

    def test_lambda(self, vm):
        result = vm.eval_str("((lambda (x) (* x x)) 7)")
        assert result == 49

    def test_lambda_closure(self, vm):
        result = vm.eval_str("""
            (define (make-adder n) (lambda (x) (+ x n)))
            (define add5 (make-adder 5))
            (add5 10)
        """)
        assert result == 15

    def test_begin(self, vm):
        result = vm.eval_str("(begin 1 2 3)")
        assert result == 3

    def test_do(self, vm):
        result = vm.eval_str("(do 1 2 3)")
        assert result == 3

    def test_quote(self, vm):
        result = vm.eval_str("(quote (1 2 3))")
        assert result == [1, 2, 3]

    def test_recursive_function(self, vm):
        result = vm.eval_str("""
            (define (factorial n)
                (if (<= n 1) 1 (* n (factorial (- n 1)))))
            (factorial 5)
        """)
        assert result == 120


# ---------------------------------------------------------------------------
# Higher-order functions
# ---------------------------------------------------------------------------

class TestHigherOrder:
    def test_map(self, vm):
        result = vm.eval_str("""
            (define (double x) (* x 2))
            (map double (list 1 2 3))
        """)
        assert result == [2, 4, 6]

    def test_map_lambda(self, vm):
        result = vm.eval_str("(map (lambda (x) (+ x 1)) (list 10 20 30))")
        assert result == [11, 21, 31]

    def test_filter(self, vm):
        result = vm.eval_str("""
            (define (positive? x) (> x 0))
            (filter positive? (list -1 2 -3 4 -5))
        """)
        assert result == [2, 4]

    def test_reduce(self, vm):
        result = vm.eval_str("(reduce + 0 (list 1 2 3 4 5))")
        assert result == 15

    def test_reduce_custom(self, vm):
        result = vm.eval_str("""
            (reduce (lambda (acc x) (+ acc (* x x))) 0 (list 1 2 3))
        """)
        assert result == 14  # 1 + 4 + 9


# ---------------------------------------------------------------------------
# Sub-simulation tests
# ---------------------------------------------------------------------------

class TestSubSim:
    def test_basic_sub_sim(self, vm):
        result = vm.eval_str('(sub-sim "test" (+ 1 2))')
        assert result == 3
        assert len(vm.sub_sim_log) == 1
        assert vm.sub_sim_log[0]["depth"] == 1
        assert vm.sub_sim_log[0]["status"] == "completed"

    def test_sub_sim_depth_2(self, vm):
        result = vm.eval_str('''
            (sub-sim "outer"
                (sub-sim "inner" (+ 10 20)))
        ''')
        assert result == 30
        assert len(vm.sub_sim_log) == 1
        outer = vm.sub_sim_log[0]
        assert outer["depth"] == 1
        assert len(outer["sub_sims"]) == 1
        inner = outer["sub_sims"][0]
        assert inner["depth"] == 2

    def test_sub_sim_depth_3(self, vm):
        result = vm.eval_str('''
            (sub-sim "L1"
                (sub-sim "L2"
                    (sub-sim "L3" (* 7 6))))
        ''')
        assert result == 42

    def test_sub_sim_depth_exceeded(self, vm):
        with pytest.raises(LispyDepthError):
            vm.eval_str('''
                (sub-sim "L1"
                    (sub-sim "L2"
                        (sub-sim "L3"
                            (sub-sim "L4" 1))))
            ''')

    def test_sub_sim_inherits_bindings(self, vm):
        result = vm.eval_str("""
            (define colony-food 5000)
            (sub-sim "ration-check"
                (/ colony-food 10))
        """)
        assert result == 500.0

    def test_sub_sim_log_recorded(self, vm):
        vm.eval_str('(sub-sim "logged" (* 2 21))')
        assert len(vm.sub_sim_log) == 1
        log = vm.sub_sim_log[0]
        assert log["label"] == "logged"
        assert log["result"] == 42

    def test_sub_sim_error_logged(self, vm):
        result = vm.eval_str('(sub-sim "bad" (/ 1 undefined-var))')
        assert result is None
        assert vm.sub_sim_log[0]["status"] == "error"


# ---------------------------------------------------------------------------
# Governance primitives
# ---------------------------------------------------------------------------

class TestGovernance:
    def test_propose(self, vm):
        result = vm.eval_str(
            '(propose "leadership" "Elect new leader" "kael-terraform")'
        )
        assert result["type"] == "proposal"
        assert result["proposal_type"] == "leadership"

    def test_vote_for(self, vm):
        result = vm.eval_str('(vote "prop-1" "for" 3.0)')
        assert result["type"] == "vote"
        assert result["position"] == "for"
        assert result["weight"] == 3.0

    def test_vote_against(self, vm):
        result = vm.eval_str('(vote "prop-1" "against" 2.0)')
        assert result["position"] == "against"

    def test_vote_weight_clamped(self, vm):
        result = vm.eval_str('(vote "prop-1" "for" 999)')
        assert result["weight"] == 10.0  # clamped to MAX_GOVERNANCE_WEIGHT

    def test_vote_invalid_position(self, vm):
        with pytest.raises(LispyError, match="Invalid vote position"):
            vm.eval_str('(vote "prop-1" "maybe" 1.0)')


# ---------------------------------------------------------------------------
# Safety tests
# ---------------------------------------------------------------------------

class TestSafety:
    def test_step_limit(self, vm_small):
        with pytest.raises(LispyStepLimitError):
            vm_small.eval_str("""
                (define (loop n) (if (= n 0) 0 (loop (- n 1))))
                (loop 10000)
            """)

    def test_no_eval(self, vm):
        """eval is not in the stdlib — can't dynamically evaluate strings."""
        with pytest.raises(LispyError, match="Undefined symbol"):
            vm.eval_str('(eval "(+ 1 2)")')

    def test_no_import(self, vm):
        with pytest.raises(LispyError, match="Undefined symbol"):
            vm.eval_str('(import "os")')

    def test_no_file_ops(self, vm):
        with pytest.raises(LispyError, match="Undefined symbol"):
            vm.eval_str('(open "test.txt")')

    def test_reset_clears_state(self, vm):
        vm.eval_str("(define x 42)")
        vm.reset()
        with pytest.raises(LispyError, match="Undefined symbol"):
            vm.eval_str("x")


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_serialize_number(self):
        assert _serialize_value(42) == 42

    def test_serialize_string(self):
        assert _serialize_value("hello") == "hello"

    def test_serialize_list(self):
        assert _serialize_value([1, 2, 3]) == [1, 2, 3]

    def test_serialize_dict(self):
        assert _serialize_value({"a": 1}) == {"a": 1}

    def test_serialize_none(self):
        assert _serialize_value(None) is None

    def test_serialize_lambda(self):
        lam = Lambda(params=["x"], body=[Symbol("+"), Symbol("x"), 1], env=Env())
        assert "<lambda" in _serialize_value(lam)

    def test_format_sexpr_number(self):
        assert format_sexpr(42) == "42"

    def test_format_sexpr_string(self):
        assert format_sexpr("hello") == '"hello"'

    def test_format_sexpr_list(self):
        assert format_sexpr([1, 2, 3]) == "(1 2 3)"

    def test_format_sexpr_nil(self):
        assert format_sexpr(None) == "nil"

    def test_format_sexpr_bool(self):
        assert format_sexpr(True) == "#t"
        assert format_sexpr(False) == "#f"


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestEnvironment:
    def test_nested_scope(self):
        outer = Env()
        outer["x"] = 10
        inner = Env(outer=outer)
        inner["y"] = 20
        assert inner.lookup("x") == 10
        assert inner.lookup("y") == 20

    def test_undefined_raises(self):
        env = Env()
        with pytest.raises(LispyError, match="Undefined symbol"):
            env.lookup("nonexistent")


# ---------------------------------------------------------------------------
# Determinism invariant
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_program_same_result(self):
        """Two VMs with same setup produce identical results."""
        program = """
            (define (fib n)
                (if (<= n 1) n (+ (fib (- n 1)) (fib (- n 2)))))
            (fib 10)
        """
        vm1 = LispyVM(max_depth=3, max_steps=50_000)
        vm2 = LispyVM(max_depth=3, max_steps=50_000)
        assert vm1.eval_str(program) == vm2.eval_str(program) == 55

    def test_sub_sim_deterministic(self):
        """Sub-sims with same input produce same output."""
        program = '(sub-sim "test" (+ (* 3 7) 21))'
        vm1 = LispyVM(max_depth=3, rng_seed=42)
        vm2 = LispyVM(max_depth=3, rng_seed=42)
        r1 = vm1.eval_str(program)
        r2 = vm2.eval_str(program)
        assert r1 == r2 == 42


# ---------------------------------------------------------------------------
# Integration / complex programs
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_governance_scenario(self, vm):
        """Full governance scenario: propose, model in sub-sim, vote."""
        result = vm.eval_str("""
            (define colony-food 1000)
            (define colony-pop 8)

            ; Model in sub-sim first
            (define ration-result
                (sub-sim "ration-model"
                    (let* ((daily-need (* 8 1.8))
                           (days-left (/ 1000 daily-need)))
                        (if (< days-left 30)
                            "emergency-rations"
                            "normal-rations"))))

            ; Create proposal based on evidence
            (propose "resource_allocation" 
                     (str "Ration plan based on " ration-result)
                     ration-result)
        """)
        assert result["type"] == "proposal"
        assert "ration" in result["description"].lower()

    def test_colonist_data_as_sexpr(self, vm):
        """Colonists are data structures that can be queried."""
        result = vm.eval_str("""
            (define kael (dict
                "name" "Kael"
                "element" "earth"
                "resolve" 0.8
                "terraforming" 0.9))

            (if (> (get kael "resolve") 0.5)
                (str (get kael "name") " leads the repair effort")
                (str (get kael "name") " assists"))
        """)
        assert result == "Kael leads the repair effort"

    def test_empty_list_eval(self, vm):
        result = vm.eval_str("(list)")
        assert result == []
