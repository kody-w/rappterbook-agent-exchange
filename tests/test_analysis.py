"""Tests for the Mars-100 analysis module."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars100.analysis import (
    _pairwise_distance,
    _std_dev,
    _traverse_subsim,
    analyze_governance_patterns,
    analyze_subsim_depths,
    compute_value_convergence,
    extract_meta_insights,
    propose_rappterbook_amendment,
    run_full_analysis,
)
from src.mars100.colonist import STAT_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_colonist_snapshot(cid: str, stats: dict, alive: bool = True,
                            exiled: bool = False) -> dict:
    """Create a minimal colonist snapshot for testing."""
    return {
        "id": cid, "name": f"Agent-{cid}", "element": "fire",
        "archetype": "test", "stats": stats,
        "skills": {}, "alive": alive, "exiled": exiled,
    }


def _make_year_result(year: int, colonists: list[dict],
                      subsims: list[dict] | None = None,
                      governance: dict | None = None,
                      meta: list | None = None,
                      resources_after: dict | None = None) -> dict:
    """Create a minimal year result dict for testing."""
    return {
        "year": year,
        "colonist_snapshots": colonists,
        "subsim_log": subsims or [],
        "governance": governance,
        "meta_awareness": meta or [],
        "resources_after": resources_after or {"food": 0.7, "water": 0.7,
                                               "power": 0.8, "air": 0.9,
                                               "medicine": 0.5},
        "events": [], "actions": {},
        "resources_before": {}, "resource_delta": {},
        "deaths": [], "exiles": [],
        "social_cohesion": 0.5, "governance_state": {},
    }


# ---------------------------------------------------------------------------
# Standard deviation
# ---------------------------------------------------------------------------

class TestStdDev:
    def test_uniform_values(self):
        assert _std_dev([5.0, 5.0, 5.0]) == 0.0

    def test_known_values(self):
        result = _std_dev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert 1.9 < result < 2.1

    def test_single_value(self):
        assert _std_dev([1.0]) == 0.0

    def test_empty(self):
        assert _std_dev([]) == 0.0

    def test_two_values(self):
        result = _std_dev([0.0, 1.0])
        assert abs(result - 0.5) < 0.001


# ---------------------------------------------------------------------------
# Pairwise distance
# ---------------------------------------------------------------------------

class TestPairwiseDistance:
    def test_identical_vectors(self):
        vecs = [{"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}]
        assert _pairwise_distance(vecs, ("a", "b")) == 0.0

    def test_known_distance(self):
        vecs = [{"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 0.0}]
        result = _pairwise_distance(vecs, ("a", "b"))
        assert abs(result - 1.0) < 0.001

    def test_single_vector(self):
        assert _pairwise_distance([{"a": 0.5}], ("a",)) == 0.0

    def test_empty(self):
        assert _pairwise_distance([], ("a",)) == 0.0

    def test_multiple_vectors(self):
        vecs = [{"a": 0.0}, {"a": 0.5}, {"a": 1.0}]
        result = _pairwise_distance(vecs, ("a",))
        # pairs: (0,0.5)=0.5, (0,1.0)=1.0, (0.5,1.0)=0.5 → avg = 2/3
        assert abs(result - 2.0 / 3) < 0.001

    def test_multidimensional(self):
        vecs = [{"a": 0.0, "b": 0.0}, {"a": 3.0, "b": 4.0}]
        result = _pairwise_distance(vecs, ("a", "b"))
        assert abs(result - 5.0) < 0.001


# ---------------------------------------------------------------------------
# Value convergence
# ---------------------------------------------------------------------------

class TestValueConvergence:
    def test_converging_colony(self):
        """Colonists with shrinking stat variance over time → converging."""
        years = []
        for y in range(1, 21):
            spread = 0.4 * (1 - y / 20)  # shrinking spread
            colonists = []
            for i in range(5):
                stats = {s: 0.5 + (i - 2) * spread / 4 for s in STAT_NAMES}
                colonists.append(_make_colonist_snapshot(f"c{i}", stats))
            years.append(_make_year_result(y, colonists))

        result = compute_value_convergence(years)
        assert result["overall_trend"] == "converging"
        assert result["late_pairwise_avg"] < result["early_pairwise_avg"]

    def test_diverging_colony(self):
        """Colonists with growing stat variance → diverging."""
        years = []
        for y in range(1, 21):
            spread = 0.1 + 0.3 * (y / 20)  # growing spread
            colonists = []
            for i in range(5):
                stats = {s: 0.5 + (i - 2) * spread / 4 for s in STAT_NAMES}
                colonists.append(_make_colonist_snapshot(f"c{i}", stats))
            years.append(_make_year_result(y, colonists))

        result = compute_value_convergence(years)
        assert result["overall_trend"] == "diverging"

    def test_stable_colony(self):
        """Colonists with constant stat variance → stable."""
        years = []
        for y in range(1, 21):
            colonists = []
            for i in range(5):
                stats = {s: 0.3 + i * 0.1 for s in STAT_NAMES}
                colonists.append(_make_colonist_snapshot(f"c{i}", stats))
            years.append(_make_year_result(y, colonists))

        result = compute_value_convergence(years)
        assert result["overall_trend"] == "stable"

    def test_tracks_population(self):
        """Population count tracks alive colonists per year."""
        years = []
        for y in range(1, 11):
            alive_count = 5 if y < 6 else 3
            colonists = []
            for i in range(5):
                alive = i < alive_count
                stats = {s: 0.5 for s in STAT_NAMES}
                colonists.append(_make_colonist_snapshot(f"c{i}", stats, alive=alive))
            years.append(_make_year_result(y, colonists))

        result = compute_value_convergence(years)
        assert result["population"][0] == 5
        assert result["population"][-1] == 3

    def test_empty_years(self):
        result = compute_value_convergence([])
        assert result["overall_trend"] == "stable"
        assert result["pairwise_distance"] == []

    def test_survivor_only_tracking(self):
        """Survivor pairwise distance only includes founding colonists."""
        years = []
        for y in range(1, 11):
            colonists = [
                _make_colonist_snapshot("founder-1", {s: 0.3 for s in STAT_NAMES}),
                _make_colonist_snapshot("founder-2", {s: 0.7 for s in STAT_NAMES}),
                _make_colonist_snapshot("newcomer", {s: 0.9 for s in STAT_NAMES}),
            ]
            years.append(_make_year_result(y, colonists))

        result = compute_value_convergence(years)
        # Survivor pairwise should only consider founder-1 and founder-2
        assert len(result["survivor_pairwise_distance"]) == 10
        assert all(d >= 0 for d in result["survivor_pairwise_distance"])


# ---------------------------------------------------------------------------
# Governance patterns
# ---------------------------------------------------------------------------

class TestGovernancePatterns:
    def test_basic_analysis(self):
        years = [
            _make_year_result(1, [], governance={
                "gov_type": "council", "passed": True,
                "votes_for": ["a", "b"], "votes_against": ["c"],
                "subsim_result": {"result": 0.8}, "proposer_id": "a",
            }),
            _make_year_result(2, [], governance={
                "gov_type": "dictator", "passed": False,
                "votes_for": ["a"], "votes_against": ["b", "c"],
                "subsim_result": None, "proposer_id": "b",
            }),
        ]
        result = analyze_governance_patterns(years)
        assert result["total_proposals"] == 2
        assert result["total_passed"] == 1
        assert "council" in result["type_breakdown"]
        assert result["type_breakdown"]["council"]["passed"] == 1

    def test_subsim_effectiveness(self):
        years = []
        for i in range(10):
            has_subsim = i < 5
            passed = i < 3 or (not has_subsim and i < 7)
            years.append(_make_year_result(i + 1, [], governance={
                "gov_type": "council", "passed": passed,
                "votes_for": ["a"], "votes_against": [],
                "subsim_result": {"result": 0.5} if has_subsim else None,
                "proposer_id": "a",
            }))
        result = analyze_governance_patterns(years)
        eff = result["subsim_effectiveness"]
        assert eff["subsim_backed_total"] == 5
        assert eff["non_subsim_total"] == 5

    def test_no_proposals(self):
        years = [_make_year_result(1, [])]
        result = analyze_governance_patterns(years)
        assert result["total_proposals"] == 0

    def test_governance_timeline_resource_delta(self):
        years = []
        for i in range(10):
            gov = None
            if i == 5:
                gov = {"gov_type": "council", "passed": True,
                       "votes_for": ["a"], "votes_against": [],
                       "subsim_result": None, "proposer_id": "a"}
            years.append(_make_year_result(i + 1, [],
                         governance=gov,
                         resources_after={"food": 0.5 + i * 0.03,
                                          "water": 0.6}))
        result = analyze_governance_patterns(years)
        timeline = result["governance_timeline"]
        assert len(timeline) == 1
        assert "resource_delta" in timeline[0]


# ---------------------------------------------------------------------------
# Sub-sim depth analysis
# ---------------------------------------------------------------------------

class TestSubsimDepths:
    def test_flat_subsims(self):
        years = [
            _make_year_result(1, [], subsims=[
                {"depth": 1, "colonist_id": "a", "expression": "(+ 1 2)", "result": 3},
                {"depth": 1, "colonist_id": "b", "expression": "(+ 3 4)", "result": 7},
            ]),
        ]
        result = analyze_subsim_depths(years)
        assert result["total_subsims"] == 2
        assert result["depth_distribution"][1] == 2
        assert not result["has_depth3"]

    def test_nested_subsims(self):
        years = [
            _make_year_result(1, [], subsims=[
                {"depth": 1, "colonist_id": "a", "expression": "(+ 1 2)", "result": 3,
                 "children": [
                     {"depth": 2, "colonist_id": "a", "expression": "(* 3 2)", "result": 6,
                      "children": [
                          {"depth": 3, "colonist_id": "a", "expression": "(- 6 1)",
                           "result": 5, "children": []},
                      ]},
                 ]},
            ]),
        ]
        result = analyze_subsim_depths(years)
        assert result["depth_distribution"][1] == 1
        assert result["depth_distribution"][2] == 1
        assert result["depth_distribution"][3] == 1
        assert result["has_depth3"]
        assert len(result["depth3_findings"]) == 1

    def test_empty(self):
        result = analyze_subsim_depths([_make_year_result(1, [])])
        assert result["total_subsims"] == 0

    def test_per_year_tracking(self):
        years = [
            _make_year_result(1, [], subsims=[
                {"depth": 1, "colonist_id": "a", "expression": "x", "result": 1},
            ]),
            _make_year_result(2, [], subsims=[]),
            _make_year_result(3, [], subsims=[
                {"depth": 1, "colonist_id": "a", "expression": "x", "result": 1},
                {"depth": 1, "colonist_id": "b", "expression": "y", "result": 2},
            ]),
        ]
        result = analyze_subsim_depths(years)
        assert result["per_year_subsims"] == [1, 0, 2]


# ---------------------------------------------------------------------------
# Meta insights
# ---------------------------------------------------------------------------

class TestMetaInsights:
    def test_clustering(self):
        years = [
            _make_year_result(31, [], meta=[
                {"colonist_id": "a", "insight": "Are we a simulation?"},
            ]),
            _make_year_result(32, [], meta=[
                {"colonist_id": "b", "insight": "I see patterns in the dust"},
            ]),
        ]
        result = extract_meta_insights(years)
        assert result["total_events"] == 2
        assert result["first_year"] == 31
        clusters = result["clusters"]
        assert clusters["simulation_awareness"]["count"] == 1
        assert clusters["pattern_recognition"]["count"] == 1

    def test_string_meta(self):
        """Handles string-format meta events."""
        years = [
            _make_year_result(50, [], meta="A sudden realization strikes"),
        ]
        result = extract_meta_insights(years)
        assert result["total_events"] == 1

    def test_empty_meta(self):
        result = extract_meta_insights([_make_year_result(1, [])])
        assert result["total_events"] == 0

    def test_data_sloshing_cluster(self):
        years = [
            _make_year_result(35, [], meta=[
                {"colonist_id": "c", "insight": "data sloshing through our frame input"},
            ]),
        ]
        result = extract_meta_insights(years)
        assert result["clusters"]["data_sloshing"]["count"] == 1


# ---------------------------------------------------------------------------
# Amendment proposal
# ---------------------------------------------------------------------------

class TestAmendmentProposal:
    def test_proposes_when_evidence_strong(self):
        convergence = {"overall_trend": "converging"}
        governance = {
            "subsim_effectiveness": {
                "subsim_backed_pass_rate": 0.8,
                "non_subsim_pass_rate": 0.4,
                "subsim_backed_total": 10,
                "non_subsim_total": 10,
                "subsim_advantage": 0.4,
            },
        }
        subsim = {"depth3_findings": [{"year": 50, "colonist_id": "a",
                                        "result": 42}],
                  "total_subsims": 100}
        meta = {"total_events": 20,
                "clusters": {"simulation_awareness": {"count": 8}}}
        result = propose_rappterbook_amendment(convergence, governance, subsim, meta)
        assert result is not None
        assert result["proposed"]
        assert len(result["evidence"]) >= 2
        assert result["confidence"] > 0.5
        assert "text" in result

    def test_no_amendment_when_evidence_weak(self):
        convergence = {"overall_trend": "stable"}
        governance = {
            "subsim_effectiveness": {
                "subsim_backed_pass_rate": 0.5,
                "non_subsim_pass_rate": 0.5,
                "subsim_backed_total": 1,
                "non_subsim_total": 1,
                "subsim_advantage": 0.0,
            },
        }
        subsim = {"depth3_findings": [], "total_subsims": 5}
        meta = {"total_events": 2,
                "clusters": {"simulation_awareness": {"count": 1}}}
        result = propose_rappterbook_amendment(convergence, governance, subsim, meta)
        assert result is None

    def test_amendment_includes_provenance(self):
        convergence = {"overall_trend": "converging"}
        governance = {
            "subsim_effectiveness": {
                "subsim_backed_pass_rate": 0.9,
                "non_subsim_pass_rate": 0.3,
                "subsim_backed_total": 5,
                "non_subsim_total": 5,
                "subsim_advantage": 0.6,
            },
        }
        subsim = {"depth3_findings": [], "total_subsims": 200}
        meta = {"total_events": 30,
                "clusters": {"simulation_awareness": {"count": 10}}}
        result = propose_rappterbook_amendment(convergence, governance, subsim, meta)
        assert result is not None
        assert "provenance" in result
        assert result["provenance"]["simulation_years"] == 100


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

class TestRunFullAnalysis:
    def test_returns_all_sections(self):
        years = [_make_year_result(y, [
            _make_colonist_snapshot(f"c{i}", {s: 0.5 for s in STAT_NAMES})
            for i in range(5)
        ]) for y in range(1, 11)]
        result = run_full_analysis(years)
        assert "convergence" in result
        assert "governance" in result
        assert "subsim_analysis" in result
        assert "meta_insights" in result
        assert "proposed_amendment" in result  # may be None

    def test_with_engine_output(self):
        """Run actual engine and analyze — integration test."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        sim = engine.run()
        year_dicts = [yr.to_dict() for yr in sim.years]
        result = run_full_analysis(year_dicts)
        assert result["convergence"]["population"][-1] > 0
        assert isinstance(result["subsim_analysis"]["total_subsims"], int)


# ---------------------------------------------------------------------------
# Traverse subsim helper
# ---------------------------------------------------------------------------

class TestTraverseSubsim:
    def test_depth_1(self):
        counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
        d3: list[dict] = []
        _traverse_subsim({"depth": 1, "colonist_id": "a",
                          "expression": "x", "result": 1}, counts, d3, 5)
        assert counts[1] == 1
        assert len(d3) == 0

    def test_nested(self):
        counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
        d3: list[dict] = []
        subsim = {
            "depth": 1, "colonist_id": "a", "expression": "x", "result": 1,
            "children": [{
                "depth": 2, "colonist_id": "a", "expression": "y", "result": 2,
                "children": [{
                    "depth": 3, "colonist_id": "a", "expression": "z",
                    "result": 3, "children": [],
                }],
            }],
        }
        _traverse_subsim(subsim, counts, d3, 10)
        assert counts == {1: 1, 2: 1, 3: 1}
        assert len(d3) == 1
        assert d3[0]["year"] == 10
