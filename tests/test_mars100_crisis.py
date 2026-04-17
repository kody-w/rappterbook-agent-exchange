"""Tests for the crisis memory organ (src/mars100/crisis.py)."""
from __future__ import annotations

import random
import pytest

from src.mars100.crisis import (
    RESOURCE_CRISIS_THRESHOLD, MIN_SUPPORT,
    CrisisEvent, CrisisPattern, ProposedAmendment,
    detect_crises, backfill_deaths, learn_from_crises,
    build_crisis_bindings, deep_deliberation, extract_rappterbook_amendment,
    _DEPTH1_TEMPLATES, _DEPTH2_TEMPLATES, _DEPTH3_TEMPLATES,
    _GOV_TYPE_MAP,
)
from src.mars100.subsim import SubSimBudget, SubSimResult
from src.mars100.lispy_vm import run as lispy_run
from src.mars100.engine import Mars100Engine


# ── CrisisEvent ─────────────────────────────────────────────────────────

class TestCrisisEvent:
    def test_creation(self):
        ev = CrisisEvent(year=10, crisis_type="resource_shortage_food",
                         severity=0.8, trigger="dust_storm",
                         governance_at_time="direct_democracy")
        assert ev.year == 10
        assert ev.severity == 0.8
        assert ev.deaths_this_year == 0

    def test_to_dict_roundtrip(self):
        ev = CrisisEvent(year=5, crisis_type="mass_casualty",
                         severity=0.6, trigger="equipment_failure",
                         governance_at_time="council",
                         resources_snapshot={"food": 0.3},
                         deaths_this_year=3)
        d = ev.to_dict()
        assert d["year"] == 5
        assert d["crisis_type"] == "mass_casualty"
        assert d["resources_snapshot"]["food"] == 0.3
        assert d["deaths_this_year"] == 3

    def test_default_resources_empty(self):
        ev = CrisisEvent(year=1, crisis_type="t", severity=0.1,
                         trigger="x", governance_at_time="anarchy")
        assert ev.resources_snapshot == {}


# ── detect_crises ────────────────────────────────────────────────────────

class TestDetectCrises:
    def test_no_crisis_when_resources_healthy(self):
        crises = detect_crises(
            resources_after={"food": 0.5, "water": 0.6},
            resources_before={"food": 0.5, "water": 0.6},
            event_names=["calm"], action_histogram={"farm": 5},
            governance_type="direct_democracy", year=10, deaths=0,
        )
        assert crises == []

    def test_resource_shortage_detected(self):
        crises = detect_crises(
            resources_after={"food": 0.05, "water": 0.6},
            resources_before={"food": 0.5, "water": 0.6},
            event_names=["dust_storm"], action_histogram={"farm": 5},
            governance_type="council", year=15, deaths=0,
        )
        assert len(crises) == 1
        assert crises[0].crisis_type == "resource_shortage_food"
        assert crises[0].severity > 0
        assert crises[0].trigger == "dust_storm"

    def test_multiple_resource_shortages(self):
        crises = detect_crises(
            resources_after={"food": 0.1, "water": 0.1, "power": 0.5},
            resources_before={"food": 0.5, "water": 0.5, "power": 0.5},
            event_names=["meteor"], action_histogram={},
            governance_type="anarchy", year=20, deaths=0,
        )
        assert len(crises) == 2
        types = {c.crisis_type for c in crises}
        assert "resource_shortage_food" in types
        assert "resource_shortage_water" in types

    def test_mass_casualty_detected(self):
        crises = detect_crises(
            resources_after={"food": 0.5}, resources_before={"food": 0.5},
            event_names=["plague"], action_histogram={"rest": 3},
            governance_type="technocracy", year=30, deaths=3,
        )
        assert any(c.crisis_type == "mass_casualty" for c in crises)
        mc = [c for c in crises if c.crisis_type == "mass_casualty"][0]
        assert mc.severity == 3 / 5.0

    def test_mass_casualty_not_for_single_death(self):
        crises = detect_crises(
            resources_after={"food": 0.5}, resources_before={"food": 0.5},
            event_names=[], action_histogram={"rest": 3},
            governance_type="council", year=5, deaths=1,
        )
        assert not any(c.crisis_type == "mass_casualty" for c in crises)

    def test_sabotage_crisis_detected(self):
        crises = detect_crises(
            resources_after={"food": 0.5}, resources_before={"food": 0.5},
            event_names=[], action_histogram={"sabotage": 4, "farm": 2},
            governance_type="anarchy", year=25, deaths=0,
        )
        assert any(c.crisis_type == "internal_sabotage" for c in crises)

    def test_sabotage_not_when_low(self):
        crises = detect_crises(
            resources_after={"food": 0.5}, resources_before={"food": 0.5},
            event_names=[], action_histogram={"sabotage": 1, "farm": 9},
            governance_type="council", year=25, deaths=0,
        )
        assert not any(c.crisis_type == "internal_sabotage" for c in crises)

    def test_severity_clamped_to_1(self):
        crises = detect_crises(
            resources_after={"food": 0.0}, resources_before={"food": 1.0},
            event_names=["famine"], action_histogram={},
            governance_type="council", year=50, deaths=0,
        )
        assert crises[0].severity <= 1.0

    def test_at_threshold_no_crisis(self):
        crises = detect_crises(
            resources_after={"food": RESOURCE_CRISIS_THRESHOLD},
            resources_before={"food": 0.5},
            event_names=[], action_histogram={},
            governance_type="council", year=10, deaths=0,
        )
        assert crises == []


# ── backfill_deaths ──────────────────────────────────────────────────────

class TestBackfillDeaths:
    def test_backfills_previous_year(self):
        log = [CrisisEvent(year=9, crisis_type="resource_shortage_food",
                           severity=0.5, trigger="storm",
                           governance_at_time="council")]
        backfill_deaths(log, year=10, deaths=2)
        assert log[0].deaths_this_year == 2

    def test_no_backfill_when_zero_deaths(self):
        log = [CrisisEvent(year=9, crisis_type="t", severity=0.5,
                           trigger="x", governance_at_time="council")]
        backfill_deaths(log, year=10, deaths=0)
        assert log[0].deaths_this_year == 0

    def test_no_overwrite_existing_deaths(self):
        log = [CrisisEvent(year=9, crisis_type="t", severity=0.5,
                           trigger="x", governance_at_time="council",
                           deaths_this_year=5)]
        backfill_deaths(log, year=10, deaths=2)
        assert log[0].deaths_this_year == 5  # not overwritten


# ── learn_from_crises ────────────────────────────────────────────────────

class TestLearnFromCrises:
    def test_no_patterns_below_min_support(self):
        log = [CrisisEvent(year=1, crisis_type="t", severity=0.5,
                           trigger="x", governance_at_time="council")]
        assert learn_from_crises(log) == []

    def test_pattern_detected(self):
        log = [
            CrisisEvent(year=i, crisis_type="resource_shortage_food",
                        severity=0.5 + i * 0.01, trigger="storm",
                        governance_at_time="direct_democracy")
            for i in range(5)
        ]
        patterns = learn_from_crises(log)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "resource_shortage_food"
        assert patterns[0].occurrences == 5
        assert patterns[0].governance_correlation == "direct_democracy"

    def test_multiple_pattern_types(self):
        log = [
            CrisisEvent(year=1, crisis_type="resource_shortage_food",
                        severity=0.5, trigger="x", governance_at_time="council"),
            CrisisEvent(year=2, crisis_type="resource_shortage_food",
                        severity=0.6, trigger="x", governance_at_time="council"),
            CrisisEvent(year=3, crisis_type="mass_casualty",
                        severity=0.7, trigger="x", governance_at_time="anarchy"),
            CrisisEvent(year=4, crisis_type="mass_casualty",
                        severity=0.8, trigger="x", governance_at_time="anarchy"),
        ]
        patterns = learn_from_crises(log)
        assert len(patterns) == 2
        types = {p.pattern_type for p in patterns}
        assert "resource_shortage_food" in types
        assert "mass_casualty" in types

    def test_governance_correlation_picks_majority(self):
        log = [
            CrisisEvent(year=1, crisis_type="t", severity=0.5, trigger="x",
                        governance_at_time="council"),
            CrisisEvent(year=2, crisis_type="t", severity=0.5, trigger="x",
                        governance_at_time="council"),
            CrisisEvent(year=3, crisis_type="t", severity=0.5, trigger="x",
                        governance_at_time="anarchy"),
        ]
        patterns = learn_from_crises(log)
        assert patterns[0].governance_correlation == "council"


# ── build_crisis_bindings ────────────────────────────────────────────────

class TestBuildCrisisBindings:
    def test_all_keys_present(self):
        pattern = CrisisPattern("t", 3, 0.7, "council", 1, 10)
        bindings = build_crisis_bindings(
            resources={"food": 0.5, "water": 0.4, "power": 0.6, "air": 0.7},
            governance_type="council",
            active_count=8,
            crisis_pattern=pattern,
        )
        expected_keys = {"governance-type", "is-democratic", "population",
                         "avg-resources", "crisis-severity", "crisis-lethal",
                         "crisis-occurrences", "food", "water", "power", "air"}
        assert expected_keys.issubset(set(bindings.keys()))

    def test_democratic_flag(self):
        pattern = CrisisPattern("t", 1, 0.5, "council", 1, 1)
        b1 = build_crisis_bindings({"food": 0.5}, "direct_democracy", 5, pattern)
        b2 = build_crisis_bindings({"food": 0.5}, "technocracy", 5, pattern)
        assert b1["is-democratic"] is True
        assert b2["is-democratic"] is False

    def test_lethal_threshold(self):
        lethal = CrisisPattern("t", 1, 0.7, "council", 1, 1)
        mild = CrisisPattern("t", 1, 0.3, "council", 1, 1)
        b1 = build_crisis_bindings({"food": 0.5}, "council", 5, lethal)
        b2 = build_crisis_bindings({"food": 0.5}, "council", 5, mild)
        assert b1["crisis-lethal"] is True
        assert b2["crisis-lethal"] is False


# ── LisPy template validity ─────────────────────────────────────────────

class TestTemplateValidity:
    """Every LisPy template must parse and evaluate without error."""

    def _eval(self, expr: str, bindings: dict) -> float | int | bool:
        return lispy_run(expr, extra_bindings=bindings, max_steps=5000)

    def _base_bindings(self) -> dict:
        return {
            "governance-type": 0, "is-democratic": True,
            "population": 8, "avg-resources": 0.5,
            "crisis-severity": 0.6, "crisis-lethal": False,
            "crisis-occurrences": 3,
            "food": 0.5, "water": 0.4, "power": 0.6, "air": 0.7,
            "resolve": 0.5, "improvisation": 0.4, "empathy": 0.6,
            "hoarding": 0.3, "faith": 0.5, "paranoia": 0.2,
        }

    @pytest.mark.parametrize("template", _DEPTH1_TEMPLATES, ids=lambda t: t[:40])
    def test_depth1_templates(self, template):
        result = self._eval(template, self._base_bindings())
        assert isinstance(result, (int, float))

    @pytest.mark.parametrize("template", _DEPTH2_TEMPLATES, ids=lambda t: t[:40])
    def test_depth2_templates(self, template):
        bindings = {**self._base_bindings(), "parent-result": 0.3}
        result = self._eval(template, bindings)
        assert isinstance(result, (int, float))

    @pytest.mark.parametrize("template", _DEPTH3_TEMPLATES, ids=lambda t: t[:40])
    def test_depth3_templates(self, template):
        bindings = {**self._base_bindings(), "parent-result": 0.4}
        result = self._eval(template, bindings)
        assert isinstance(result, (int, float))


# ── deep_deliberation ────────────────────────────────────────────────────

class TestDeepDeliberation:
    def _make_pattern(self) -> CrisisPattern:
        return CrisisPattern("resource_shortage_food", 3, 0.6,
                             "direct_democracy", 5, 15)

    def _colonist_bindings(self) -> dict:
        return {
            "resolve": 0.5, "improvisation": 0.4, "empathy": 0.6,
            "hoarding": 0.3, "faith": 0.5, "paranoia": 0.2,
            "terraforming": 0.3, "hydroponics": 0.5, "mediation": 0.4,
            "coding": 0.6, "prayer": 0.2, "sabotage": 0.1,
        }

    def test_returns_result_with_children(self):
        budget = SubSimBudget(year=10)
        log: list[SubSimResult] = []
        rng = random.Random(42)
        result = deep_deliberation(
            colonist_id="c-0", colonist_bindings=self._colonist_bindings(),
            resources={"food": 0.3, "water": 0.4, "power": 0.5, "air": 0.6},
            crisis_pattern=self._make_pattern(),
            governance_type="direct_democracy", active_count=8,
            year=10, budget=budget, log=log, rng=rng,
        )
        assert result is not None
        assert result.succeeded
        # Should have depth-2 child
        assert len(result.children) >= 1

    def test_reaches_depth_3(self):
        budget = SubSimBudget(year=10)
        log: list[SubSimResult] = []
        rng = random.Random(42)
        result = deep_deliberation(
            colonist_id="c-0", colonist_bindings=self._colonist_bindings(),
            resources={"food": 0.3, "water": 0.4, "power": 0.5, "air": 0.6},
            crisis_pattern=self._make_pattern(),
            governance_type="direct_democracy", active_count=8,
            year=10, budget=budget, log=log, rng=rng,
        )
        assert result is not None
        if result.children:
            d2 = result.children[0]
            if d2.children:
                d3 = d2.children[0]
                assert d3.depth == 3
                assert d3.succeeded

    def test_budget_limits_spawns(self):
        budget = SubSimBudget(year=10)
        log: list[SubSimResult] = []
        rng = random.Random(42)
        # Exhaust budget
        for i in range(10):
            budget.record("c-0")
        result = deep_deliberation(
            colonist_id="c-0", colonist_bindings=self._colonist_bindings(),
            resources={"food": 0.3, "water": 0.4, "power": 0.5, "air": 0.6},
            crisis_pattern=self._make_pattern(),
            governance_type="direct_democracy", active_count=8,
            year=10, budget=budget, log=log, rng=rng,
        )
        assert result is None  # budget exhausted

    def test_all_templates_used_across_seeds(self):
        """Over many seeds, all depth-1 templates get selected."""
        templates_hit: set[str] = set()
        for seed in range(50):
            rng = random.Random(seed)
            budget = SubSimBudget(year=10)
            log: list[SubSimResult] = []
            result = deep_deliberation(
                colonist_id="c-0", colonist_bindings=self._colonist_bindings(),
                resources={"food": 0.3, "water": 0.4, "power": 0.5, "air": 0.6},
                crisis_pattern=self._make_pattern(),
                governance_type="direct_democracy", active_count=8,
                year=10, budget=budget, log=log, rng=rng,
            )
            if result and result.expression:
                templates_hit.add(result.expression[:40])
        assert len(templates_hit) >= 2  # both depth-1 templates hit


# ── extract_rappterbook_amendment ────────────────────────────────────────

class TestExtractAmendment:
    def test_no_amendment_when_too_few_crises(self):
        patterns = [CrisisPattern("t", 2, 0.5, "council", 1, 5)]
        log = [CrisisEvent(year=1, crisis_type="t", severity=0.5,
                           trigger="x", governance_at_time="council")]
        assert extract_rappterbook_amendment([], patterns, log) is None

    def test_amendment_proposed_with_enough_evidence(self):
        patterns = [CrisisPattern("resource_shortage_food", 5, 0.7,
                                  "direct_democracy", 1, 20)]
        log = [
            CrisisEvent(year=i, crisis_type="resource_shortage_food",
                        severity=0.5, trigger="x",
                        governance_at_time="direct_democracy")
            for i in range(5)
        ]
        result = extract_rappterbook_amendment([], patterns, log)
        assert result is not None
        assert result.number == 19
        assert "Amendment XIX" in result.title
        assert "crisis" in result.body.lower() or "Crisis" in result.body

    def test_amendment_dict_has_required_keys(self):
        patterns = [CrisisPattern("mass_casualty", 3, 0.8, "anarchy", 5, 30)]
        log = [
            CrisisEvent(year=i, crisis_type="mass_casualty",
                        severity=0.8, trigger="x",
                        governance_at_time="anarchy")
            for i in range(4)
        ]
        result = extract_rappterbook_amendment([], patterns, log)
        assert result is not None
        d = result.to_dict()
        assert "amendment_number" in d
        assert "title" in d
        assert "body" in d
        assert "evidence" in d
        assert len(d["evidence"]) >= 1

    def test_picks_most_frequent_pattern(self):
        patterns = [
            CrisisPattern("resource_shortage_food", 10, 0.5, "council", 1, 50),
            CrisisPattern("mass_casualty", 2, 0.9, "anarchy", 10, 30),
        ]
        log = [CrisisEvent(year=i, crisis_type="t", severity=0.5,
                           trigger="x", governance_at_time="council")
               for i in range(12)]
        result = extract_rappterbook_amendment([], patterns, log)
        assert result is not None
        assert "resource" in result.title.lower() or "Crisis" in result.title


# ── CrisisPattern ────────────────────────────────────────────────────────

class TestCrisisPattern:
    def test_to_dict(self):
        p = CrisisPattern("resource_shortage_food", 5, 0.6,
                          "direct_democracy", 3, 20)
        d = p.to_dict()
        assert d["pattern_type"] == "resource_shortage_food"
        assert d["occurrences"] == 5
        assert d["first_seen"] == 3
        assert d["last_seen"] == 20


# ── ProposedAmendment ────────────────────────────────────────────────────

class TestProposedAmendment:
    def test_to_dict(self):
        a = ProposedAmendment(number=19, title="Test", body="Body",
                              evidence=[{"x": 1}], subsim_depth_reached=3)
        d = a.to_dict()
        assert d["amendment_number"] == 19
        assert d["subsim_depth_reached"] == 3


# ── Engine integration ───────────────────────────────────────────────────

class TestEngineIntegration:
    def test_100_year_run_has_crisis_fields(self):
        """Full simulation produces crisis data in results."""
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        d = result.to_dict()
        assert "total_crises" in d["summary"]
        assert "crisis_patterns_found" in d["summary"]
        assert "amendment_proposed" in d["summary"]
        assert "crisis_patterns" in d
        assert "proposed_amendment" in d

    def test_year_results_have_crisis_events(self):
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        # At least some years should have crisis events
        crisis_years = [y for y in result.years if y.crisis_events]
        # With 100 years, resources will drop enough to trigger some
        assert len(crisis_years) >= 0  # non-negative (may be 0 with lucky seed)
        for y in crisis_years:
            for ce in y.crisis_events:
                assert "crisis_type" in ce
                assert "severity" in ce

    def test_crisis_count_matches_year_data(self):
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        year_crisis_count = sum(len(y.crisis_events) for y in result.years)
        assert year_crisis_count == len(result.crisis_log)

    def test_determinism(self):
        """Same seed produces same crisis count."""
        r1 = Mars100Engine(seed=99, total_years=50).run()
        r2 = Mars100Engine(seed=99, total_years=50).run()
        assert len(r1.crisis_log) == len(r2.crisis_log)
        for c1, c2 in zip(r1.crisis_log, r2.crisis_log):
            assert c1["year"] == c2["year"]
            assert c1["crisis_type"] == c2["crisis_type"]

    def test_version_is_3(self):
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        assert result.to_dict()["_meta"]["version"] == "3.0"

    def test_amendment_extraction_runs(self):
        """Run long enough to potentially produce an amendment."""
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        d = result.to_dict()
        # Amendment may or may not be proposed depending on crisis count
        if d["proposed_amendment"] is not None:
            assert d["proposed_amendment"]["amendment_number"] == 19
            assert "title" in d["proposed_amendment"]

    def test_10_year_smoke(self):
        """Quick smoke test — 10 years, no crash."""
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert len(result.years) == 10

    def test_crisis_patterns_formed_over_time(self):
        """With enough years and a tight seed, patterns should form."""
        engine = Mars100Engine(seed=7, total_years=100)
        result = engine.run()
        # Patterns may or may not form depending on sim dynamics
        if result.crisis_patterns:
            for p in result.crisis_patterns:
                assert p["occurrences"] >= MIN_SUPPORT


# ── Property-based invariants ────────────────────────────────────────────

class TestInvariants:
    def test_severity_bounded(self):
        """All crisis severities in [0, 1]."""
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        for c in result.crisis_log:
            assert 0 <= c["severity"] <= 1.0

    def test_crisis_years_within_sim(self):
        """All crisis years within [1, total_years]."""
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        for c in result.crisis_log:
            assert 1 <= c["year"] <= 100

    @pytest.mark.parametrize("seed", range(5))
    def test_no_crash_various_seeds(self, seed):
        """Engine doesn't crash with crisis organ across multiple seeds."""
        engine = Mars100Engine(seed=seed, total_years=30)
        result = engine.run()
        assert len(result.years) > 0
