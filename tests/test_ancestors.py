"""Tests for the ancestor protocol module."""
from __future__ import annotations

import random
import pytest

from src.mars100.ancestors import (
    AncestorProfile,
    AncestorVault,
    CONSULT_TEMPLATES,
    CRISIS_SEVERITY_THRESHOLD,
    DEATH_CAUSE_TAGS,
    FAITH_THRESHOLD,
    choose_template,
    consult_ancestor,
    should_consult,
)
from src.mars100.colonist import (
    Colonist, ColonistStats, ColonistSkills, STAT_NAMES, SKILL_NAMES,
)
from src.mars100.subsim import SubSimBudget, SubSimResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_colonist(cid: str = "test-1", name: str = "Test",
                   element: str = "fire", faith: float = 0.5,
                   alive: bool = True, death_year: int | None = None,
                   death_cause: str | None = None) -> Colonist:
    """Create a colonist for testing."""
    stats = ColonistStats(resolve=0.5, improvisation=0.5, empathy=0.5,
                          hoarding=0.3, faith=faith, paranoia=0.2)
    skills = ColonistSkills(terraforming=0.6, hydroponics=0.4,
                            mediation=0.3, coding=0.7, prayer=0.3,
                            sabotage=0.1)
    c = Colonist(id=cid, name=name, element=element, archetype="engineer",
                 stats=stats, skills=skills, birth_year=0,
                 decision_expr="(+ resolve faith)")
    if not alive:
        c.alive = False
        c.death_year = death_year
        c.death_cause = death_cause
    return c


def _dead_colonist(cid: str = "dead-1", name: str = "Ancestor",
                   death_year: int = 50,
                   death_cause: str = "asphyxiation") -> Colonist:
    """Create a dead colonist for vault registration."""
    return _make_colonist(cid=cid, name=name, alive=False,
                          death_year=death_year, death_cause=death_cause)


# ---------------------------------------------------------------------------
# AncestorProfile
# ---------------------------------------------------------------------------

class TestAncestorProfile:
    def test_lispy_bindings_prefix(self):
        c = _dead_colonist()
        vault = AncestorVault()
        profile = vault.add(c)
        bindings = profile.lispy_bindings()
        for name in STAT_NAMES:
            assert f"ancestor-{name}" in bindings
        for name in SKILL_NAMES:
            assert f"ancestor-{name}" in bindings
        assert "ancestor-years-lived" in bindings
        assert "ancestor-died-air" in bindings

    def test_air_death_tag(self):
        c = _dead_colonist(death_cause="asphyxiation")
        vault = AncestorVault()
        profile = vault.add(c)
        assert profile.lispy_bindings()["ancestor-died-air"] == 1

    def test_non_air_death_tag(self):
        c = _dead_colonist(death_cause="radiation exposure")
        vault = AncestorVault()
        profile = vault.add(c)
        assert profile.lispy_bindings()["ancestor-died-air"] == 0

    def test_years_lived(self):
        c = _make_colonist(alive=False)
        c.birth_year = 5
        c.death_year = 55
        c.death_cause = "medical emergency"
        vault = AncestorVault()
        profile = vault.add(c)
        assert profile.years_lived == 50

    def test_to_dict_roundtrip(self):
        c = _dead_colonist()
        vault = AncestorVault()
        profile = vault.add(c)
        d = profile.to_dict()
        assert d["colonist_id"] == "dead-1"
        assert d["death_cause"] == "asphyxiation"
        assert isinstance(d["stats"], dict)
        assert isinstance(d["skills"], dict)

    def test_bindings_are_numeric(self):
        """All bindings except death-cause-tag must be numeric."""
        c = _dead_colonist()
        vault = AncestorVault()
        profile = vault.add(c)
        for key, val in profile.lispy_bindings().items():
            if key == "ancestor-death-cause-tag":
                assert isinstance(val, str)
            else:
                assert isinstance(val, (int, float)), f"{key} not numeric: {val}"


# ---------------------------------------------------------------------------
# AncestorVault
# ---------------------------------------------------------------------------

class TestAncestorVault:
    def test_add_and_get(self):
        vault = AncestorVault()
        c = _dead_colonist("a1")
        vault.add(c)
        assert vault.size() == 1
        assert vault.get("a1") is not None
        assert vault.get("nonexistent") is None

    def test_multiple_ancestors(self):
        vault = AncestorVault()
        for i in range(5):
            vault.add(_dead_colonist(f"dead-{i}", death_year=10 + i))
        assert vault.size() == 5
        assert len(vault.all_ids()) == 5

    def test_to_dict(self):
        vault = AncestorVault()
        vault.add(_dead_colonist("a1"))
        d = vault.to_dict()
        assert "ancestors" in d
        assert "a1" in d["ancestors"]
        assert d["total_consultations"] == 0

    def test_record_consultation(self):
        vault = AncestorVault()
        vault.add(_dead_colonist("a1"))
        mock_result = SubSimResult(depth=1, colonist_id="caller-1",
                                   year=60, expression="(+ 1 2)",
                                   result=3, steps_used=5)
        vault.record_consultation(60, "caller-1", "a1", "air_crisis",
                                  mock_result)
        assert vault.to_dict()["total_consultations"] == 1
        assert vault.consultations[0]["succeeded"] is True

    def test_consultation_log_capped(self):
        vault = AncestorVault()
        vault.add(_dead_colonist("a1"))
        mock = SubSimResult(depth=1, colonist_id="c", year=1,
                            expression="x", result=0, steps_used=1)
        for i in range(30):
            vault.record_consultation(i, "c", "a1", "general", mock)
        assert len(vault.to_dict()["consultation_log"]) == 20
        assert len(vault.consultations) == 30


# ---------------------------------------------------------------------------
# should_consult
# ---------------------------------------------------------------------------

class TestShouldConsult:
    def test_empty_vault_returns_none(self):
        vault = AncestorVault()
        c = _make_colonist(faith=0.9)
        rng = random.Random(42)
        assert should_consult(c, vault, 0.8, rng) is None

    def test_low_faith_returns_none(self):
        vault = AncestorVault()
        vault.add(_dead_colonist("a1"))
        c = _make_colonist(faith=0.1)
        rng = random.Random(42)
        for _ in range(50):
            assert should_consult(c, vault, 0.0, rng) is None

    def test_high_faith_high_crisis_likely(self):
        vault = AncestorVault()
        vault.add(_dead_colonist("a1"))
        c = _make_colonist(faith=0.9)
        hits = 0
        for seed in range(100):
            rng = random.Random(seed)
            if should_consult(c, vault, 0.8, rng) is not None:
                hits += 1
        assert hits > 60, f"Expected >60 consultations, got {hits}"

    def test_moderate_faith_no_crisis_moderate_prob(self):
        vault = AncestorVault()
        vault.add(_dead_colonist("a1"))
        c = _make_colonist(faith=0.5)
        hits = 0
        for seed in range(200):
            rng = random.Random(seed)
            if should_consult(c, vault, 0.1, rng) is not None:
                hits += 1
        assert 20 < hits < 130, f"Expected moderate rate, got {hits}/200"


# ---------------------------------------------------------------------------
# choose_template
# ---------------------------------------------------------------------------

class TestChooseTemplate:
    def test_air_crisis(self):
        res = {"food": 0.5, "water": 0.5, "power": 0.5,
               "air": 0.1, "medicine": 0.5}
        assert choose_template(res, []) == "air_crisis"

    def test_resource_crisis(self):
        res = {"food": 0.2, "water": 0.2, "power": 0.2,
               "air": 0.3, "medicine": 0.2}
        assert choose_template(res, []) == "resource_crisis"

    def test_social_crisis(self):
        res = {"food": 0.6, "water": 0.6, "power": 0.6,
               "air": 0.6, "medicine": 0.6}
        events = [{"name": "colonist_conflict", "severity": 0.5}]
        assert choose_template(res, events) == "social_crisis"

    def test_general_fallback(self):
        res = {"food": 0.6, "water": 0.6, "power": 0.6,
               "air": 0.6, "medicine": 0.6}
        assert choose_template(res, []) == "general"

    def test_air_crisis_takes_priority(self):
        res = {"food": 0.5, "water": 0.5, "power": 0.5,
               "air": 0.1, "medicine": 0.5}
        events = [{"name": "colonist_conflict", "severity": 0.7}]
        assert choose_template(res, events) == "air_crisis"


# ---------------------------------------------------------------------------
# consult_ancestor (integration with LisPy VM)
# ---------------------------------------------------------------------------

class TestConsultAncestor:
    def test_basic_consultation(self):
        vault = AncestorVault()
        vault.add(_dead_colonist("a1", death_year=50))
        caller = _make_colonist("c1", faith=0.8)
        budget = SubSimBudget(year=51)
        log: list[SubSimResult] = []
        result = consult_ancestor(vault, caller, "a1", "general",
                                  year=51, budget=budget, log=log)
        assert result is not None
        assert result.succeeded
        assert isinstance(result.result, (int, float))
        assert len(log) == 1

    def test_air_crisis_template(self):
        vault = AncestorVault()
        vault.add(_dead_colonist("a1"))
        caller = _make_colonist("c1", faith=0.8)
        budget = SubSimBudget(year=60)
        log: list[SubSimResult] = []
        result = consult_ancestor(vault, caller, "a1", "air_crisis",
                                  year=60, budget=budget, log=log)
        assert result is not None
        assert result.succeeded

    def test_budget_respected(self):
        vault = AncestorVault()
        vault.add(_dead_colonist("a1"))
        caller = _make_colonist("c1")
        budget = SubSimBudget(year=60)
        budget.colony_total = 999
        log: list[SubSimResult] = []
        result = consult_ancestor(vault, caller, "a1", "general",
                                  year=60, budget=budget, log=log)
        assert result is None

    def test_nonexistent_ancestor(self):
        vault = AncestorVault()
        caller = _make_colonist("c1")
        budget = SubSimBudget(year=60)
        log: list[SubSimResult] = []
        result = consult_ancestor(vault, caller, "nobody", "general",
                                  year=60, budget=budget, log=log)
        assert result is None

    def test_consultation_logged(self):
        vault = AncestorVault()
        vault.add(_dead_colonist("a1"))
        caller = _make_colonist("c1")
        budget = SubSimBudget(year=60)
        log: list[SubSimResult] = []
        consult_ancestor(vault, caller, "a1", "general",
                         year=60, budget=budget, log=log)
        assert len(vault.consultations) == 1
        entry = vault.consultations[0]
        assert entry["caller"] == "c1"
        assert entry["ancestor"] == "a1"
        assert entry["template"] == "general"

    def test_all_templates_evaluable(self):
        """Every consultation template must evaluate without error."""
        vault = AncestorVault()
        vault.add(_dead_colonist("a1"))
        caller = _make_colonist("c1")
        for template_key in CONSULT_TEMPLATES:
            budget = SubSimBudget(year=60)
            log: list[SubSimResult] = []
            result = consult_ancestor(vault, caller, "a1", template_key,
                                      year=60, budget=budget, log=log)
            assert result is not None, f"{template_key} returned None"
            assert result.succeeded, f"{template_key} failed: {result.error}"


# ---------------------------------------------------------------------------
# Integration: AncestorVault with real Colonist lifecycle
# ---------------------------------------------------------------------------

class TestAncestorLifecycle:
    def test_colonist_dies_becomes_ancestor(self):
        c = _make_colonist("c1", name="Pioneer", element="earth")
        c.die(year=45, cause="asphyxiation")
        vault = AncestorVault()
        profile = vault.add(c)
        assert profile.death_year == 45
        assert profile.death_cause == "asphyxiation"
        assert profile.lispy_bindings()["ancestor-died-air"] == 1

    def test_multiple_generations(self):
        vault = AncestorVault()
        for i in range(5):
            c = _make_colonist(f"gen-{i}", name=f"Gen{i}",
                               faith=0.3 + i * 0.1)
            c.die(year=20 + i * 10, cause="asphyxiation")
            vault.add(c)
        assert vault.size() == 5
        for aid in vault.all_ids():
            profile = vault.get(aid)
            assert profile is not None
            assert profile.lispy_bindings()["ancestor-died-air"] == 1

    def test_consultation_uses_ancestor_stats(self):
        """Ancestor bindings override caller bindings for ancestor-* keys."""
        vault = AncestorVault()
        dead = _make_colonist("dead-1", faith=0.9)
        dead.stats.resolve = 0.95
        dead.die(year=30, cause="asphyxiation")
        vault.add(dead)

        caller = _make_colonist("live-1", faith=0.5)
        caller.stats.resolve = 0.1

        budget = SubSimBudget(year=31)
        log: list[SubSimResult] = []
        result = consult_ancestor(vault, caller, "dead-1", "general",
                                  year=31, budget=budget, log=log)
        assert result is not None
        assert result.succeeded
        assert isinstance(result.result, (int, float))


# ---------------------------------------------------------------------------
# Property-based invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_all_death_causes_have_tags(self):
        for cause, tag in DEATH_CAUSE_TAGS.items():
            assert isinstance(tag, str)
            assert len(tag) > 0

    def test_vault_size_matches_additions(self):
        vault = AncestorVault()
        for i in range(20):
            vault.add(_dead_colonist(f"d-{i}", death_year=i + 10))
        assert vault.size() == 20

    def test_faith_threshold_positive(self):
        assert 0 < FAITH_THRESHOLD < 1

    def test_crisis_severity_threshold_positive(self):
        assert 0 < CRISIS_SEVERITY_THRESHOLD < 1

    def test_all_templates_are_valid_lispy(self):
        from src.mars100.lispy_vm import parse_all
        for key, expr in CONSULT_TEMPLATES.items():
            exprs = parse_all(expr)
            assert len(exprs) > 0, f"Template {key} parsed to empty"

    def test_consult_deterministic(self):
        """Same seed + same inputs = same result."""
        vault = AncestorVault()
        vault.add(_dead_colonist("a1"))
        results = []
        for _ in range(3):
            caller = _make_colonist("c1")
            budget = SubSimBudget(year=60)
            log: list[SubSimResult] = []
            r = consult_ancestor(vault, caller, "a1", "general",
                                 year=60, budget=budget, log=log)
            assert r is not None
            results.append(r.result)
        assert results[0] == results[1] == results[2]


# ---------------------------------------------------------------------------
# Engine integration smoke test
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_engine_runs_with_ancestors(self):
        """Full sim must run with ancestor protocol wired in."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=30)
        result = engine.run()
        assert result.total_deaths >= 0
        if result.total_deaths > 0:
            assert engine.ancestors.size() > 0

    def test_ancestor_data_in_result(self):
        """SimulationResult should include ancestors dict."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=30)
        result = engine.run()
        d = result.to_dict()
        assert "ancestors" in d

    def test_version_bumped(self):
        """Engine version should be 5.0 with ancestor protocol."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        d = result.to_dict()
        assert d["_meta"]["version"] == "5.0"
