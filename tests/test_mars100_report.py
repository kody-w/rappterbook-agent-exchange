"""Tests for Mars-100 governance report generator."""
from __future__ import annotations

import pytest
from src.mars100.engine import Mars100Engine
from src.mars100.analysis import full_analysis
from src.mars100.report import (
    generate_governance_report,
    _interpret_convergence,
    _generate_conclusion,
)


@pytest.fixture
def sim_and_analysis() -> tuple[dict, dict]:
    """Run a short sim and return (sim_dict, analysis)."""
    engine = Mars100Engine(seed=42, total_years=20)
    sim = engine.run().to_dict()
    return sim, full_analysis(sim)


class TestInterpretConvergence:
    def test_converging(self) -> None:
        data = {"verdict": "converging", "overall_convergence": 0.3,
                "convergence_scores": {"resolve": 0.4, "empathy": 0.1}}
        text = _interpret_convergence(data)
        assert "converged" in text.lower()
        assert "resolve" in text

    def test_diverging(self) -> None:
        data = {"verdict": "diverging", "overall_convergence": -0.2,
                "convergence_scores": {"resolve": -0.1, "empathy": -0.3}}
        text = _interpret_convergence(data)
        assert "diverged" in text.lower()

    def test_stable(self) -> None:
        data = {"verdict": "stable", "overall_convergence": 0.0,
                "convergence_scores": {"resolve": 0.0}}
        text = _interpret_convergence(data)
        assert "stable" in text.lower()

    def test_empty(self) -> None:
        text = _interpret_convergence({})
        assert "insufficient" in text.lower()


class TestGenerateConclusion:
    def test_high_fitness(self) -> None:
        analysis = {"fitness": {"composite": 0.7},
                     "amendment_proposal": {"proposed": False},
                     "meta_emergence": {"total_events": 0}}
        text = _generate_conclusion(analysis)
        assert "thrived" in text.lower()

    def test_low_fitness(self) -> None:
        analysis = {"fitness": {"composite": 0.2},
                     "amendment_proposal": {"proposed": False},
                     "meta_emergence": {"total_events": 0}}
        text = _generate_conclusion(analysis)
        assert "barely" in text.lower()

    def test_amendment_mentioned(self) -> None:
        analysis = {"fitness": {"composite": 0.5},
                     "amendment_proposal": {
                         "proposed": True,
                         "amendment": {"title": "Test Amendment"},
                     },
                     "meta_emergence": {"total_events": 0}}
        text = _generate_conclusion(analysis)
        assert "Test Amendment" in text

    def test_meta_mentioned(self) -> None:
        analysis = {"fitness": {"composite": 0.5},
                     "amendment_proposal": {"proposed": False},
                     "meta_emergence": {"total_events": 5, "first_year": 12}}
        text = _generate_conclusion(analysis)
        assert "meta-awareness" in text.lower() or "year 12" in text


class TestGovernanceReport:
    def test_returns_string(self, sim_and_analysis: tuple[dict, dict]) -> None:
        sim, analysis = sim_and_analysis
        report = generate_governance_report(analysis, sim.get("summary"))
        assert isinstance(report, str)

    def test_has_all_sections(self, sim_and_analysis: tuple[dict, dict]) -> None:
        sim, analysis = sim_and_analysis
        report = generate_governance_report(analysis, sim.get("summary"))
        for section in ("Executive Summary", "Value Convergence",
                        "Governance Stability", "Sub-Simulation Effectiveness",
                        "Meta-Awareness", "Proposed Constitutional Amendment",
                        "Conclusion"):
            assert section in report, f"Missing section: {section}"

    def test_contains_fitness_score(self, sim_and_analysis: tuple[dict, dict]) -> None:
        sim, analysis = sim_and_analysis
        report = generate_governance_report(analysis, sim.get("summary"))
        assert "fitness score" in report.lower()

    def test_without_summary(self, sim_and_analysis: tuple[dict, dict]) -> None:
        _, analysis = sim_and_analysis
        report = generate_governance_report(analysis)
        assert isinstance(report, str)
        assert "Executive Summary" in report

    def test_empty_analysis(self) -> None:
        report = generate_governance_report({})
        assert isinstance(report, str)
        assert "Mars-100" in report

    def test_deterministic(self) -> None:
        """Same inputs produce same report."""
        engine = Mars100Engine(seed=42, total_years=10)
        sim = engine.run().to_dict()
        a = full_analysis(sim)
        r1 = generate_governance_report(a, sim.get("summary"))
        r2 = generate_governance_report(a, sim.get("summary"))
        assert r1 == r2
