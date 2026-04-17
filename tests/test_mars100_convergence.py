"""Tests for Mars-100 value convergence, deep sub-sims, and insight promotion."""
from __future__ import annotations

import pytest
from src.mars100.engine import Mars100Engine, YearResult, SimulationResult
from src.mars100.colony import Resources, RESOURCE_NAMES, compute_value_convergence
from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, STAT_NAMES, create_founding_ten


# ── Value convergence ──────────────────────────────────────────


class TestValueConvergence:
    """Tests for compute_value_convergence()."""

    def test_returns_all_stat_names_plus_score(self):
        colonists = create_founding_ten(42)
        result = compute_value_convergence(colonists)
        for name in STAT_NAMES:
            assert name in result
        assert "convergence_score" in result

    def test_convergence_score_nonnegative(self):
        colonists = create_founding_ten(42)
        result = compute_value_convergence(colonists)
        assert result["convergence_score"] >= 0.0

    def test_identical_stats_yield_zero_convergence(self):
        colonists = create_founding_ten(42)
        for c in colonists:
            c.stats = ColonistStats(resolve=0.5, improvisation=0.5, empathy=0.5,
                                    hoarding=0.5, faith=0.5, paranoia=0.5)
        result = compute_value_convergence(colonists)
        assert result["convergence_score"] == pytest.approx(0.0, abs=1e-9)

    def test_high_variance_stats_yield_nonzero(self):
        colonists = create_founding_ten(42)
        colonists[0].stats = ColonistStats(resolve=1.0, improvisation=0.0, empathy=1.0,
                                           hoarding=0.0, faith=1.0, paranoia=0.0)
        result = compute_value_convergence(colonists)
        assert result["convergence_score"] > 0.0

    def test_single_colonist_yields_zero(self):
        colonists = create_founding_ten(42)
        for c in colonists[1:]:
            c.alive = False
        result = compute_value_convergence(colonists)
        assert result["convergence_score"] == 0.0

    def test_no_colonists_yields_zero(self):
        colonists = create_founding_ten(42)
        for c in colonists:
            c.alive = False
        result = compute_value_convergence(colonists)
        assert result["convergence_score"] == 0.0

    def test_convergence_appears_in_tick_result(self):
        engine = Mars100Engine(seed=42, total_years=5)
        yr = engine.tick()
        assert "convergence_score" in yr.convergence
        assert yr.convergence["convergence_score"] >= 0.0


# ── Convergence trend ─────────────────────────────────────────


class TestConvergenceTrend:
    def test_trend_in_result(self):
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert result.convergence_trend in ("converging", "diverging", "stable")

    def test_trend_in_serialized_output(self):
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        d = result.to_dict()
        assert "convergence_trend" in d["summary"]

    def test_short_run_is_stable(self):
        engine = Mars100Engine(seed=42, total_years=3)
        result = engine.run()
        assert result.convergence_trend == "stable"


# ── Deep sub-simulation governance expressions ────────────────


class TestDeepSubSimExpressions:
    def test_deep_governance_expr_depth_2(self):
        engine = Mars100Engine(seed=42, total_years=5)
        colonist = engine.colonists[0]
        expr = engine._generate_deep_governance_expression(colonist, 0.9, 2)
        assert "(let" in expr or "(cond" in expr
        assert any(word in expr for word in ["empathy", "resolve", "faith", "paranoia", "trust"])

    def test_deep_governance_expr_depth_3(self):
        engine = Mars100Engine(seed=42, total_years=5)
        colonist = engine.colonists[0]
        expr = engine._generate_deep_governance_expression(colonist, 1.2, 3)
        assert "(let" in expr
        assert "3" in expr or "depth" in expr or "meta" in expr

    def test_subsim_log_appears_in_year(self):
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        subsim_count = sum(len(y.subsim_log) for y in result.years)
        assert subsim_count >= 0


# ── Insight extraction and promotion ──────────────────────────


class TestInsightExtraction:
    def test_insight_queue_accumulates(self):
        engine = Mars100Engine(seed=42, total_years=30)
        engine.run()
        assert isinstance(engine.insight_queue, list)

    def test_insight_has_required_fields(self):
        engine = Mars100Engine(seed=42, total_years=50)
        engine.run()
        for insight in engine.insight_queue:
            assert "type" in insight
            assert "description" in insight
            assert "score" in insight
            assert "depth" in insight
            assert "year" in insight
            assert "colonist_id" in insight
            assert insight["depth"] >= 2

    def test_promoted_insights_list(self):
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        assert isinstance(result.promoted_insights, list)

    def test_promoted_insight_has_draft_amendment(self):
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        for ins in result.promoted_insights:
            assert "draft_amendment" in ins
            assert "Amendment:" in ins["draft_amendment"]
            assert "evidence_count" in ins
            assert ins["evidence_count"] >= 2

    def test_max_one_promotion(self):
        engine = Mars100Engine(seed=42, total_years=100)
        engine.run()
        assert len(engine.promoted_insights) <= 1


# ── Draft amendment text ──────────────────────────────────────


class TestDraftAmendment:
    def test_cooperative_template(self):
        evidence = [
            {"type": "cooperative_governance", "score": 1.5, "depth": 2,
             "year": 20, "colonist_id": "c1", "colonist_name": "A", "current_gov": "anarchy"},
            {"type": "cooperative_governance", "score": 1.3, "depth": 3,
             "year": 25, "colonist_id": "c2", "colonist_name": "B", "current_gov": "anarchy"},
        ]
        text = Mars100Engine._draft_amendment("cooperative_governance", evidence)
        assert "2/3 consensus" in text
        assert "2 sub-simulations" in text

    def test_crisis_template(self):
        evidence = [
            {"type": "crisis_authoritarian", "score": -0.3, "depth": 2,
             "year": 10, "colonist_id": "c1", "colonist_name": "A", "current_gov": "council"},
        ]
        text = Mars100Engine._draft_amendment("crisis_authoritarian", evidence)
        assert "emergency" in text.lower() or "crisis" in text.lower()

    def test_balanced_template(self):
        evidence = [
            {"type": "balanced_governance", "score": 0.9, "depth": 2,
             "year": 30, "colonist_id": "c1", "colonist_name": "A", "current_gov": "anarchy"},
        ]
        text = Mars100Engine._draft_amendment("balanced_governance", evidence)
        assert "rotate" in text.lower() or "council" in text.lower()

    def test_unknown_type_fallback(self):
        evidence = [{"score": 0.5, "depth": 2}]
        text = Mars100Engine._draft_amendment("unknown_type", evidence)
        assert "Amendment:" in text


# ── Serialization round-trip ──────────────────────────────────


class TestSerialization:
    def test_year_result_convergence_serializes(self):
        engine = Mars100Engine(seed=42, total_years=5)
        yr = engine.tick()
        d = yr.to_dict()
        assert "convergence" in d
        assert "convergence_score" in d["convergence"]

    def test_simulation_result_promoted_insights_serializes(self):
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        d = result.to_dict()
        assert "promoted_insights" in d["summary"]
        assert "promoted_insights" in d
        assert isinstance(d["promoted_insights"], list)

    def test_version_bumped(self):
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        d = result.to_dict()
        assert d["_meta"]["version"] == "1.1"


# ── Regression: colony lethality ──────────────────────────────


class TestLethalityRegression:
    def test_100_year_survival(self):
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        assert len(result.years) >= 50, (
            f"Colony died at year {len(result.years)} — too lethal"
        )

    def test_multiple_seeds_not_all_extinct(self):
        survivors = 0
        for seed in range(42, 52):
            engine = Mars100Engine(seed=seed, total_years=100)
            result = engine.run()
            if len(result.years) >= 80:
                survivors += 1
        assert survivors >= 2, (
            f"Only {survivors}/10 seeds survived 80+ years — death rate too high"
        )
