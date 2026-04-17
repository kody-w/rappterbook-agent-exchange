"""Tests for the Mars-100 governance report generator."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars100.report import (
    generate_governance_report,
    _interpret_convergence,
    _generate_conclusion,
)


def _make_analysis(overall_trend: str = "stable",
                   amendment: dict | None = None,
                   total_proposals: int = 50,
                   total_passed: int = 20,
                   total_subsims: int = 100,
                   meta_total: int = 30) -> dict:
    """Create a sample analysis dict for testing."""
    return {
        "convergence": {
            "overall_trend": overall_trend,
            "early_pairwise_avg": 0.3,
            "late_pairwise_avg": 0.2 if overall_trend == "converging" else 0.4,
            "stat_trends": {
                "resolve": "converging" if overall_trend == "converging" else "stable",
                "improvisation": "stable",
                "empathy": "converging" if overall_trend == "converging" else "diverging",
                "hoarding": "stable",
                "faith": "diverging" if overall_trend == "diverging" else "stable",
                "paranoia": "stable",
            },
        },
        "governance": {
            "total_proposals": total_proposals,
            "total_passed": total_passed,
            "type_breakdown": {
                "council": {"proposed": 20, "passed": 12, "rejected": 8},
                "consensus": {"proposed": 15, "passed": 5, "rejected": 10},
                "dictator": {"proposed": 10, "passed": 2, "rejected": 8},
            },
            "governance_timeline": [
                {"year": 10, "gov_type": "council", "resource_before": 3.5,
                 "resource_after": 3.8, "resource_delta": 0.3},
            ],
            "subsim_effectiveness": {
                "subsim_backed_pass_rate": 0.7,
                "non_subsim_pass_rate": 0.4,
                "subsim_backed_total": 30,
                "non_subsim_total": 20,
                "subsim_advantage": 0.3,
            },
        },
        "subsim_analysis": {
            "total_subsims": total_subsims,
            "depth_distribution": {1: 80, 2: 15, 3: 5},
            "depth3_findings": [
                {"year": 45, "colonist_id": "kira-sol",
                 "expression": "(+ resolve empathy)", "result": 1.2},
            ],
            "has_depth3": True,
            "depth3_count": 5,
        },
        "meta_insights": {
            "total_events": meta_total,
            "first_year": 31,
            "last_year": 95,
            "clusters": {
                "simulation_awareness": {"count": 12, "events": [
                    {"year": 31, "colonist_id": "a",
                     "insight": "Are we in a simulation?"},
                ]},
                "pattern_recognition": {"count": 8, "events": []},
                "data_sloshing": {"count": 5, "events": []},
                "existential": {"count": 5, "events": []},
            },
            "unclustered": [],
        },
        "proposed_amendment": amendment,
    }


class TestGovernanceReport:
    def test_generates_markdown(self):
        analysis = _make_analysis()
        report = generate_governance_report(analysis, {})
        assert report.startswith("# Emergent Governance Patterns")
        assert "## Executive Summary" in report
        assert "## Value Convergence" in report
        assert "## Governance Patterns" in report

    def test_includes_convergence_trend(self):
        analysis = _make_analysis(overall_trend="converging")
        report = generate_governance_report(analysis, {})
        assert "CONVERGING" in report

    def test_includes_subsim_effectiveness(self):
        analysis = _make_analysis()
        report = generate_governance_report(analysis, {})
        assert "Sub-simulation Effectiveness" in report
        assert "70%" in report  # subsim pass rate

    def test_includes_depth3_findings(self):
        analysis = _make_analysis()
        report = generate_governance_report(analysis, {})
        assert "Depth-3 Findings" in report
        assert "kira-sol" in report

    def test_includes_meta_awareness(self):
        analysis = _make_analysis()
        report = generate_governance_report(analysis, {})
        assert "Meta-Awareness Events" in report
        assert "31" in report  # first year

    def test_with_amendment(self):
        amendment = {
            "proposed": True,
            "text": "All proposals must include simulation evidence",
            "evidence": [{"type": "subsim_effectiveness",
                          "finding": "Sub-sims improve outcomes",
                          "confidence": 0.8}],
            "confidence": 0.75,
            "source": "mars-100",
            "provenance": {"simulation_years": 100},
        }
        analysis = _make_analysis(amendment=amendment)
        report = generate_governance_report(analysis, {})
        assert "Proposed Rappterbook Amendment" in report
        assert "simulation evidence" in report

    def test_without_amendment(self):
        analysis = _make_analysis(amendment=None)
        report = generate_governance_report(analysis, {})
        assert "No amendment proposed" in report

    def test_includes_conclusion(self):
        analysis = _make_analysis()
        report = generate_governance_report(analysis, {})
        assert "Recursive Lesson" in report
        assert "LisPy interpreter" in report


class TestInterpretConvergence:
    def test_converging(self):
        conv = {"overall_trend": "converging",
                "stat_trends": {"resolve": "converging", "empathy": "converging",
                                "faith": "stable", "paranoia": "stable",
                                "improvisation": "stable", "hoarding": "stable"}}
        result = _interpret_convergence(conv)
        assert "converge" in result.lower()
        assert "resolve" in result

    def test_diverging(self):
        conv = {"overall_trend": "diverging",
                "stat_trends": {"faith": "diverging", "resolve": "stable",
                                "empathy": "stable", "paranoia": "stable",
                                "improvisation": "stable", "hoarding": "stable"}}
        result = _interpret_convergence(conv)
        assert "diverge" in result.lower()

    def test_stable(self):
        conv = {"overall_trend": "stable",
                "stat_trends": {s: "stable" for s in
                                ["resolve", "empathy", "faith", "paranoia",
                                 "improvisation", "hoarding"]}}
        result = _interpret_convergence(conv)
        assert "stable" in result.lower()


class TestGenerateConclusion:
    def test_with_amendment_and_meta(self):
        analysis = _make_analysis(
            amendment={"proposed": True, "text": "test"},
            meta_total=20)
        conclusion = _generate_conclusion(analysis)
        assert "turtle" in conclusion.lower() or "looks up" in conclusion.lower()

    def test_without_amendment(self):
        analysis = _make_analysis(amendment=None)
        conclusion = _generate_conclusion(analysis)
        assert "recursive frontier" in conclusion.lower()
