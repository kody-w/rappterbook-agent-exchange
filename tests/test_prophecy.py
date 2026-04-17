"""Tests for the Prophecy Engine module."""
from __future__ import annotations

import random
import pytest

from src.mars100.prophecy import (
    Prophecy, ProphecyEngine, _safe_eval_predicate, _resolve_value,
)


class TestProphecyDataclass:
    def test_creation(self):
        p = Prophecy(author_id="c-1", expression="(> food 0.5)",
                     target_year=5, created_year=3, prophecy_type="intuitive")
        assert p.author_id == "c-1"
        assert p.resolved is False
        assert p.outcome is None

    def test_to_dict_roundtrip(self):
        p = Prophecy(author_id="c-2", expression="(< water 0.2)",
                     target_year=10, created_year=8, prophecy_type="empirical",
                     resource_tag="water", description="drought")
        d = p.to_dict()
        assert d["author_id"] == "c-2"
        assert d["type"] == "empirical"
        p2 = Prophecy.from_dict(d)
        assert p2.author_id == p.author_id
        assert p2.expression == p.expression
        assert p2.prophecy_type == p.prophecy_type

    def test_from_dict_defaults(self):
        d = {"author_id": "c-3", "expression": "(> food 0.5)",
             "target_year": 5, "created_year": 3}
        p = Prophecy.from_dict(d)
        assert p.prophecy_type == "intuitive"
        assert p.resolved is False


class TestProphecyCreation:
    def test_make_prophecy_returns_prophecy(self):
        engine = ProphecyEngine()
        rng = random.Random(42)
        stats = {"faith": 0.8, "paranoia": 0.6}
        skills = {"coding": 0.5}
        resources = {"food": 0.1, "water": 0.5, "air": 0.5, "power": 0.5, "medicine": 0.5}
        for _ in range(50):
            engine.begin_year()
            p = engine.make_prophecy("c-1", stats, skills, resources, 5, False, rng)
            if p is not None:
                assert isinstance(p, Prophecy)
                assert p.author_id == "c-1"
                assert p.target_year > 5
                return
        pytest.skip("Low probability path — no prophecy generated in 50 tries")

    def test_max_per_year_cap(self):
        engine = ProphecyEngine()
        rng = random.Random(1)
        engine.begin_year()
        stats = {"faith": 1.0, "paranoia": 1.0}
        skills = {}
        resources = {"food": 0.1, "water": 0.1, "air": 0.1, "power": 0.1, "medicine": 0.1}
        made = 0
        for _ in range(200):
            p = engine.make_prophecy("c-1", stats, skills, resources, 1, True, rng)
            if p:
                made += 1
        assert made <= 2

    def test_empirical_type(self):
        engine = ProphecyEngine()
        rng = random.Random(42)
        stats = {"faith": 1.0, "paranoia": 1.0}
        skills = {}
        resources = {"food": 0.1, "water": 0.1, "air": 0.1, "power": 0.1, "medicine": 0.1}
        for _ in range(100):
            engine.begin_year()
            p = engine.make_prophecy("c-1", stats, skills, resources, 5, True, rng)
            if p:
                assert p.prophecy_type == "empirical"
                return
        pytest.skip("No prophecy generated")


class TestProphecyResolution:
    def _make_resolved_engine(self):
        engine = ProphecyEngine()
        p = Prophecy(author_id="c-1", expression="(> food 0.5)",
                     target_year=5, created_year=3, prophecy_type="intuitive",
                     resource_tag="food", description="abundance")
        engine.prophecies.append(p)
        return engine

    def test_resolve_correct(self):
        engine = self._make_resolved_engine()
        resolutions = engine.resolve(5, {"food": 0.8, "water": 0.5}, {"c-1"})
        assert len(resolutions) == 1
        assert resolutions[0]["outcome"] is True

    def test_resolve_incorrect(self):
        engine = self._make_resolved_engine()
        resolutions = engine.resolve(5, {"food": 0.3, "water": 0.5}, {"c-1"})
        assert len(resolutions) == 1
        assert resolutions[0]["outcome"] is False

    def test_resolve_wrong_year_skips(self):
        engine = self._make_resolved_engine()
        resolutions = engine.resolve(4, {"food": 0.8, "water": 0.5}, {"c-1"})
        assert len(resolutions) == 0
        assert engine.prophecies[0].resolved is False

    def test_resolve_dead_author(self):
        engine = self._make_resolved_engine()
        resolutions = engine.resolve(5, {"food": 0.8, "water": 0.5}, {"c-2"})
        assert len(resolutions) == 1
        assert resolutions[0]["outcome"] is False

    def test_resolve_already_resolved_skips(self):
        engine = self._make_resolved_engine()
        engine.resolve(5, {"food": 0.8}, {"c-1"})
        resolutions = engine.resolve(5, {"food": 0.8}, {"c-1"})
        assert len(resolutions) == 0


class TestAccuracyAndInfluence:
    def _engine_with_history(self, outcomes):
        engine = ProphecyEngine()
        for i, outcome in enumerate(outcomes):
            p = Prophecy(author_id="c-1", expression="(> food 0.5)",
                         target_year=i+1, created_year=i, prophecy_type="intuitive",
                         resolved=True, outcome=outcome, resolution_year=i+1)
            engine.prophecies.append(p)
        return engine

    def test_accuracy_all_correct(self):
        engine = self._engine_with_history([True, True, True])
        assert engine.accuracy("c-1") == 1.0

    def test_accuracy_all_wrong(self):
        engine = self._engine_with_history([False, False, False])
        assert engine.accuracy("c-1") == 0.0

    def test_accuracy_mixed(self):
        engine = self._engine_with_history([True, False, True, False])
        assert engine.accuracy("c-1") == 0.5

    def test_accuracy_no_history(self):
        engine = ProphecyEngine()
        assert engine.accuracy("c-1") == 0.5

    def test_influence_modifier_high_accuracy(self):
        engine = self._engine_with_history([True] * 10)
        mod = engine.influence_modifier("c-1")
        assert mod > 0.0
        assert mod <= 0.3

    def test_influence_modifier_low_accuracy(self):
        engine = self._engine_with_history([False] * 10)
        mod = engine.influence_modifier("c-1")
        assert mod < 0.0
        assert mod >= -0.3

    def test_influence_modifier_no_history(self):
        engine = ProphecyEngine()
        assert engine.influence_modifier("c-1") == 0.0


class TestWarnings:
    def test_active_warnings(self):
        engine = ProphecyEngine()
        engine.prophecies.append(
            Prophecy(author_id="c-1", expression="(< food 0.2)",
                     target_year=5, created_year=3, prophecy_type="intuitive",
                     resource_tag="food"))
        engine.prophecies.append(
            Prophecy(author_id="c-2", expression="(> water 0.7)",
                     target_year=6, created_year=3, prophecy_type="intuitive",
                     resource_tag="water"))
        warnings = engine.active_warnings(4)
        assert len(warnings) == 1
        assert warnings[0].resource_tag == "food"

    def test_warning_resources(self):
        engine = ProphecyEngine()
        engine.prophecies.append(
            Prophecy(author_id="c-1", expression="(< food 0.2)",
                     target_year=5, created_year=3, prophecy_type="intuitive",
                     resource_tag="food"))
        engine.prophecies.append(
            Prophecy(author_id="c-2", expression="(< power 0.2)",
                     target_year=6, created_year=3, prophecy_type="intuitive",
                     resource_tag="power"))
        warned = engine.warning_resources(4)
        assert warned == {"food", "power"}

    def test_resolved_not_warned(self):
        engine = ProphecyEngine()
        p = Prophecy(author_id="c-1", expression="(< food 0.2)",
                     target_year=5, created_year=3, prophecy_type="intuitive",
                     resource_tag="food", resolved=True, outcome=True)
        engine.prophecies.append(p)
        assert engine.active_warnings(4) == []


class TestSummary:
    def test_empty_summary(self):
        engine = ProphecyEngine()
        s = engine.summary()
        assert s["total"] == 0
        assert s["accuracy"] == 0.0

    def test_mixed_summary(self):
        engine = ProphecyEngine()
        for i in range(4):
            engine.prophecies.append(
                Prophecy(author_id=f"c-{i}", expression="(> food 0.5)",
                         target_year=5, created_year=3,
                         prophecy_type="intuitive" if i < 2 else "empirical",
                         resolved=True, outcome=i % 2 == 0))
        s = engine.summary()
        assert s["total"] == 4
        assert s["resolved"] == 4
        assert s["correct"] == 2
        assert s["accuracy"] == 0.5


class TestSafeEval:
    def test_greater_than_true(self):
        assert _safe_eval_predicate("(> food 0.5)", {"food": 0.8}) is True

    def test_greater_than_false(self):
        assert _safe_eval_predicate("(> food 0.5)", {"food": 0.3}) is False

    def test_less_than(self):
        assert _safe_eval_predicate("(< water 0.2)", {"water": 0.1}) is True

    def test_equality(self):
        assert _safe_eval_predicate("(= x 1.0)", {"x": 1.0}) is True

    def test_unknown_variable(self):
        with pytest.raises(ValueError, match="Unknown variable"):
            _safe_eval_predicate("(> unknown 0.5)", {})

    def test_unknown_operator(self):
        with pytest.raises(ValueError, match="Unknown operator"):
            _safe_eval_predicate("(& x 0.5)", {"x": 0.5})

    def test_invalid_expression(self):
        with pytest.raises(ValueError, match="Invalid"):
            _safe_eval_predicate("no_parens", {})


class TestProphecyCycle:
    def test_full_cycle(self):
        """Integration test: make prophecy, advance time, resolve."""
        engine = ProphecyEngine()
        rng = random.Random(42)
        stats = {"faith": 1.0, "paranoia": 1.0}
        skills = {}
        resources = {"food": 0.1, "water": 0.1, "air": 0.1, "power": 0.1, "medicine": 0.1}
        made = None
        for year in range(1, 50):
            engine.begin_year()
            p = engine.make_prophecy("c-1", stats, skills, resources, year, False, rng)
            if p and made is None:
                made = p
            engine.resolve(year, resources, {"c-1"})
        assert made is not None, "Should have made at least one prophecy in 50 years"
        assert any(p.resolved for p in engine.prophecies)

    def test_prophecy_appears_in_summary(self):
        engine = ProphecyEngine()
        p = Prophecy(author_id="c-1", expression="(> food 0.5)",
                     target_year=5, created_year=3, prophecy_type="intuitive",
                     resolved=True, outcome=True, resolution_year=5)
        engine.prophecies.append(p)
        s = engine.summary()
        assert s["total"] == 1
        assert s["correct"] == 1
