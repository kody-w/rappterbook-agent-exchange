"""Tests for the Constitutional Crossover Engine."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest
from src.mars100.crossover import (
    GovernancePattern,
    BehavioralConvergence,
    TransferabilityScore,
    AmendmentProposal,
    analyze_governance_transitions,
    detect_behavioral_convergence,
    score_transferability,
    analyze_value_trends,
    analyze_subsim_effectiveness,
    generate_rappterbook_amendment,
    crossover_analysis,
    RAPPTERBOOK_ANALOGUES,
    AMENDMENT_TEMPLATES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_year(year: int, gov_type: str | None = None, passed: bool = False,
               cohesion: float = 0.5, resources: dict | None = None,
               actions: dict | None = None, events: list | None = None,
               subsim_result: dict | None = None,
               convergence: dict | None = None) -> dict:
    """Build a minimal year dict for testing."""
    d: dict = {"year": year, "social_cohesion": cohesion}
    if resources:
        d["resources_after"] = resources
    else:
        d["resources_after"] = {"food": 0.6, "water": 0.6, "power": 0.7, "air": 0.8, "medicine": 0.5}
    if actions:
        d["actions"] = actions
    if events:
        d["events"] = events
    if convergence:
        d["convergence"] = convergence
    if gov_type is not None:
        gov: dict = {"gov_type": gov_type, "passed": passed}
        if subsim_result is not None:
            gov["subsim_result"] = subsim_result
        d["governance"] = gov
    return d


def _make_years(n: int = 100, gov_transitions: list[tuple[int, str]] | None = None,
                cohesion: float = 0.6) -> list[dict]:
    """Build a list of year dicts with optional governance transitions."""
    transitions = dict(gov_transitions) if gov_transitions else {}
    years = []
    for y in range(1, n + 1):
        gov_type = transitions.get(y)
        passed = gov_type is not None
        conv = {"convergence_score": 0.15 - y * 0.0005}
        for stat in ("resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia"):
            conv[stat] = 0.12 - y * 0.0003
        years.append(_make_year(y, gov_type=gov_type, passed=passed,
                                cohesion=cohesion, convergence=conv))
    return years


# ---------------------------------------------------------------------------
# GovernancePattern dataclass
# ---------------------------------------------------------------------------

class TestGovernancePattern:

    def test_to_dict(self) -> None:
        p = GovernancePattern(name="council", description="test", first_seen_year=1,
                              last_seen_year=50, duration_years=50, stability_score=0.5,
                              cohesion_during=0.7, resource_health=0.6,
                              evidence=["stable regime"])
        d = p.to_dict()
        assert d["name"] == "council"
        assert d["duration_years"] == 50
        assert d["evidence"] == ["stable regime"]

    def test_rounding(self) -> None:
        p = GovernancePattern(name="x", description="x", first_seen_year=1,
                              last_seen_year=1, duration_years=1,
                              stability_score=0.33333, cohesion_during=0.66666,
                              resource_health=0.12345)
        d = p.to_dict()
        assert d["stability_score"] == 0.333
        assert d["cohesion_during"] == 0.667


# ---------------------------------------------------------------------------
# BehavioralConvergence dataclass
# ---------------------------------------------------------------------------

class TestBehavioralConvergence:

    def test_to_dict(self) -> None:
        bc = BehavioralConvergence(year=10, action="farm", fraction=0.8,
                                   context="dust_storm", colonist_ids=["a", "b", "c"])
        d = bc.to_dict()
        assert d["year"] == 10
        assert d["colonist_count"] == 3
        assert d["fraction"] == 0.8


# ---------------------------------------------------------------------------
# analyze_governance_transitions
# ---------------------------------------------------------------------------

class TestGovernanceTransitions:

    def test_empty_years(self) -> None:
        assert analyze_governance_transitions([], 100) == []

    def test_no_transitions(self) -> None:
        years = _make_years(10)
        patterns = analyze_governance_transitions(years, 10)
        assert len(patterns) == 1
        assert patterns[0].name == "anarchy"
        assert patterns[0].duration_years == 10

    def test_single_transition(self) -> None:
        years = _make_years(20, gov_transitions=[(10, "council")])
        patterns = analyze_governance_transitions(years, 20)
        assert len(patterns) == 2
        assert patterns[0].name == "anarchy"
        assert patterns[0].duration_years == 9
        assert patterns[1].name == "council"
        assert patterns[1].duration_years == 11

    def test_multiple_transitions(self) -> None:
        years = _make_years(30, gov_transitions=[(5, "council"), (15, "consensus")])
        patterns = analyze_governance_transitions(years, 30)
        assert len(patterns) == 3
        names = [p.name for p in patterns]
        assert names == ["anarchy", "council", "consensus"]

    def test_stability_score_normalized(self) -> None:
        years = _make_years(100)
        patterns = analyze_governance_transitions(years, 100)
        assert len(patterns) == 1
        assert abs(patterns[0].stability_score - 1.0) < 0.01

    def test_cohesion_tracked(self) -> None:
        years = _make_years(10, cohesion=0.75)
        patterns = analyze_governance_transitions(years, 10)
        assert patterns[0].cohesion_during > 0.7

    def test_resource_health_tracked(self) -> None:
        years = []
        for y in range(1, 11):
            years.append(_make_year(y, resources={"food": 0.9, "water": 0.8}))
        patterns = analyze_governance_transitions(years, 10)
        assert patterns[0].resource_health > 0.8

    def test_evidence_populated(self) -> None:
        years = _make_years(50, cohesion=0.7)
        patterns = analyze_governance_transitions(years, 50)
        assert len(patterns[0].evidence) >= 1


# ---------------------------------------------------------------------------
# detect_behavioral_convergence
# ---------------------------------------------------------------------------

class TestBehavioralConvergenceDetection:

    def test_no_convergence(self) -> None:
        years = [_make_year(1, actions={"a": "farm", "b": "code", "c": "pray", "d": "mediate"})]
        result = detect_behavioral_convergence(years, threshold=0.6)
        assert result == []

    def test_clear_convergence(self) -> None:
        years = [_make_year(1, actions={"a": "farm", "b": "farm", "c": "farm", "d": "code"},
                            events=[{"name": "dust_storm"}])]
        result = detect_behavioral_convergence(years, threshold=0.6)
        assert len(result) == 1
        assert result[0].action == "farm"
        assert result[0].fraction == 0.75

    def test_threshold_respected(self) -> None:
        years = [_make_year(1, actions={"a": "farm", "b": "farm", "c": "code", "d": "pray"})]
        low = detect_behavioral_convergence(years, threshold=0.4)
        high = detect_behavioral_convergence(years, threshold=0.6)
        assert len(low) == 1
        assert len(high) == 0

    def test_multiple_years(self) -> None:
        years = [
            _make_year(1, actions={"a": "farm", "b": "farm", "c": "farm"}),
            _make_year(2, actions={"a": "code", "b": "code", "c": "code"}),
        ]
        result = detect_behavioral_convergence(years, threshold=0.6)
        assert len(result) == 2

    def test_too_few_colonists_skipped(self) -> None:
        years = [_make_year(1, actions={"a": "farm", "b": "farm"})]
        result = detect_behavioral_convergence(years, threshold=0.6)
        assert len(result) == 0  # < 3 colonists


# ---------------------------------------------------------------------------
# score_transferability
# ---------------------------------------------------------------------------

class TestTransferability:

    def test_empty_input(self) -> None:
        assert score_transferability([], [], 100) == []

    def test_stable_high_cohesion_scores_well(self) -> None:
        p = GovernancePattern(name="council", description="", first_seen_year=1,
                              last_seen_year=80, duration_years=80,
                              stability_score=0.8, cohesion_during=0.75,
                              resource_health=0.65)
        scores = score_transferability([p], [], 100)
        assert len(scores) == 1
        assert scores[0].score > 0.5
        assert scores[0].evidence_strength in ("strong", "moderate")

    def test_unstable_low_cohesion_scores_poorly(self) -> None:
        p = GovernancePattern(name="anarchy", description="", first_seen_year=1,
                              last_seen_year=5, duration_years=5,
                              stability_score=0.05, cohesion_during=0.3,
                              resource_health=0.3)
        scores = score_transferability([p], [], 100)
        assert scores[0].score < 0.3
        assert scores[0].evidence_strength == "weak"

    def test_convergences_boost_score(self) -> None:
        p = GovernancePattern(name="council", description="", first_seen_year=1,
                              last_seen_year=50, duration_years=50,
                              stability_score=0.5, cohesion_during=0.6,
                              resource_health=0.5)
        conv = [BehavioralConvergence(year=y, action="farm", fraction=0.7,
                                      context="calm", colonist_ids=["a", "b"])
                for y in range(10, 30)]
        score_without = score_transferability([p], [], 100)[0].score
        score_with = score_transferability([p], conv, 100)[0].score
        assert score_with > score_without

    def test_rappterbook_analogue_populated(self) -> None:
        for gov_type, analogue in RAPPTERBOOK_ANALOGUES.items():
            p = GovernancePattern(name=gov_type, description="", first_seen_year=1,
                                  last_seen_year=10, duration_years=10,
                                  stability_score=0.1, cohesion_during=0.5,
                                  resource_health=0.5)
            scores = score_transferability([p], [], 100)
            assert scores[0].rappterbook_analogue == analogue

    def test_score_bounded(self) -> None:
        p = GovernancePattern(name="council", description="", first_seen_year=1,
                              last_seen_year=100, duration_years=100,
                              stability_score=1.0, cohesion_during=1.0,
                              resource_health=1.0)
        scores = score_transferability([p], [], 100)
        assert 0.0 <= scores[0].score <= 1.0


# ---------------------------------------------------------------------------
# analyze_value_trends
# ---------------------------------------------------------------------------

class TestValueTrends:

    def test_empty_years(self) -> None:
        result = analyze_value_trends([])
        assert result["trend"] == "no_data"

    def test_insufficient_data(self) -> None:
        years = [_make_year(y, convergence={"convergence_score": 0.1}) for y in range(1, 6)]
        result = analyze_value_trends(years)
        assert result["trend"] == "insufficient_data"

    def test_stable_trend(self) -> None:
        years = []
        for y in range(1, 101):
            conv = {"convergence_score": 0.15}
            for stat in ("resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia"):
                conv[stat] = 0.12
            years.append(_make_year(y, convergence=conv))
        result = analyze_value_trends(years)
        assert result["trend"] == "stable"

    def test_converging_trend(self) -> None:
        years = []
        for y in range(1, 101):
            score = 0.2 - y * 0.0015  # clearly decreasing → converging
            conv = {"convergence_score": max(0, score)}
            for stat in ("resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia"):
                conv[stat] = max(0, score)
            years.append(_make_year(y, convergence=conv))
        result = analyze_value_trends(years)
        assert result["trend"] == "converging"

    def test_per_stat_trends(self) -> None:
        years = []
        for y in range(1, 101):
            conv = {"convergence_score": 0.15,
                    "resolve": 0.15, "improvisation": 0.15,
                    "empathy": 0.15, "hoarding": 0.15,
                    "faith": 0.15, "paranoia": 0.15}
            years.append(_make_year(y, convergence=conv))
        result = analyze_value_trends(years)
        assert "resolve" in result["stats"]


# ---------------------------------------------------------------------------
# analyze_subsim_effectiveness
# ---------------------------------------------------------------------------

class TestSubsimEffectiveness:

    def test_no_proposals(self) -> None:
        result = analyze_subsim_effectiveness([_make_year(1)])
        assert result["total_proposals"] == 0

    def test_with_subsim_proposals(self) -> None:
        years = [
            _make_year(1, gov_type="council", passed=True, subsim_result={"result": 0.8}),
            _make_year(2, gov_type="dictator", passed=False),
            _make_year(3, gov_type="lottery", passed=True, subsim_result={"result": 0.5}),
        ]
        result = analyze_subsim_effectiveness(years)
        assert result["total_proposals"] == 3
        assert result["with_subsim"] == 2
        assert result["without_subsim"] == 1
        assert result["with_subsim_passed"] == 2
        assert result["with_subsim_rate"] == 1.0

    def test_subsim_advantage_positive(self) -> None:
        years = [
            _make_year(y, gov_type="council", passed=True,
                       subsim_result={"result": 0.5})
            for y in range(1, 6)
        ] + [
            _make_year(y, gov_type="dictator", passed=False)
            for y in range(6, 11)
        ]
        result = analyze_subsim_effectiveness(years)
        assert result["subsim_advantage"] > 0


# ---------------------------------------------------------------------------
# generate_rappterbook_amendment
# ---------------------------------------------------------------------------

class TestAmendmentGeneration:

    def test_basic_amendment(self) -> None:
        patterns = [GovernancePattern(name="council", description="", first_seen_year=1,
                                      last_seen_year=80, duration_years=80,
                                      stability_score=0.8, cohesion_during=0.75,
                                      resource_health=0.65)]
        amendment = generate_rappterbook_amendment(
            patterns, [], [], {"subsim_advantage": 0.1, "verdict": "moderate"},
            {"trend": "stable"})
        assert amendment.title is not None
        assert amendment.number is not None
        assert amendment.confidence > 0
        assert amendment.lispy_expression != ""
        assert len(amendment.evidence) > 0

    def test_confidence_bounded(self) -> None:
        patterns = [GovernancePattern(name="council", description="", first_seen_year=1,
                                      last_seen_year=100, duration_years=100,
                                      stability_score=1.0, cohesion_during=0.9,
                                      resource_health=0.9)]
        convergences = [BehavioralConvergence(year=y, action="farm", fraction=0.8,
                                              context="calm", colonist_ids=["a"])
                        for y in range(1, 20)]
        strong_transfer = [TransferabilityScore(pattern_name="council", score=0.9,
                                                rationale="", rappterbook_analogue="",
                                                evidence_strength="strong")]
        amendment = generate_rappterbook_amendment(
            patterns, convergences, strong_transfer,
            {"subsim_advantage": 0.3, "verdict": "significant"},
            {"trend": "converging"})
        assert amendment.confidence <= 0.95

    def test_amendment_has_lispy(self) -> None:
        amendment = generate_rappterbook_amendment([], [], [], {}, {})
        assert "(define" in amendment.lispy_expression or "(if" in amendment.lispy_expression

    def test_amendment_serializable(self) -> None:
        amendment = generate_rappterbook_amendment([], [], [], {}, {})
        d = amendment.to_dict()
        assert isinstance(d["title"], str)
        assert isinstance(d["confidence"], float)
        assert isinstance(d["evidence"], list)


# ---------------------------------------------------------------------------
# crossover_analysis (integration)
# ---------------------------------------------------------------------------

class TestCrossoverAnalysis:

    def test_empty_sim(self) -> None:
        result = crossover_analysis({"years": []})
        assert result["_meta"]["engine"] == "mars-100-crossover"
        assert result["summary"]["patterns_found"] == 0

    def test_basic_sim(self) -> None:
        years = _make_years(20, gov_transitions=[(10, "council")])
        result = crossover_analysis({"years": years})
        assert result["summary"]["patterns_found"] == 2
        assert result["amendment_proposal"]["title"] is not None

    def test_full_sim(self) -> None:
        years = _make_years(100, gov_transitions=[(10, "council"), (50, "consensus")])
        # Add some actions for convergence detection
        for y in years:
            y["actions"] = {"a": "farm", "b": "farm", "c": "farm", "d": "code"}
        result = crossover_analysis({"years": years})
        assert result["summary"]["patterns_found"] == 3
        assert result["summary"]["convergences_found"] > 0
        assert result["amendment_proposal"]["confidence"] > 0

    def test_result_serializable(self) -> None:
        """All values must be JSON-serializable."""
        import json
        years = _make_years(20)
        result = crossover_analysis({"years": years})
        serialized = json.dumps(result)
        assert len(serialized) > 100

    def test_amendment_templates_valid(self) -> None:
        """All amendment templates must have required keys."""
        for tmpl in AMENDMENT_TEMPLATES:
            assert "trigger" in tmpl
            assert "title" in tmpl
            assert "number" in tmpl
            assert "text" in tmpl
            assert "lispy" in tmpl

    def test_deterministic(self) -> None:
        """Same input produces same output."""
        years = _make_years(50, gov_transitions=[(10, "council")])
        r1 = crossover_analysis({"years": years})
        r2 = crossover_analysis({"years": years})
        assert r1["summary"] == r2["summary"]
        assert r1["governance_patterns"] == r2["governance_patterns"]
