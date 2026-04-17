"""Tests for the constitutional amendment system (v10.0)."""
from __future__ import annotations

import pytest

from src.mars100.amendment import (
    GovernanceEvidence, AmendmentProposal,
    extract_governance_evidence, is_independent, count_independent,
    score_amendment, evaluate_amendments, format_amendment_text,
    MIN_INDEPENDENT_SIMS, MIN_DISTINCT_COLONISTS, MIN_DISTINCT_YEARS,
    MIN_STABILITY_SCORE, GOVERNANCE_LABELS,
)


# ────── GovernanceEvidence ──────

class TestGovernanceEvidence:
    def test_creation(self):
        ev = GovernanceEvidence(
            colonist_id="c0", year=10, depth=1,
            gov_type="council", stability_score=0.8,
            survived=True, frames_run=20,
        )
        assert ev.gov_type == "council"

    def test_to_dict(self):
        ev = GovernanceEvidence(
            colonist_id="c0", year=10, depth=1,
            gov_type="council", stability_score=0.8,
            survived=True, frames_run=20,
        )
        d = ev.to_dict()
        assert d["colonist_id"] == "c0"
        assert d["stability_score"] == 0.8


# ────── independence checks ──────

class TestIndependence:
    def test_same_colonist_same_year(self):
        a = GovernanceEvidence("c0", 10, 1, "council", 0.8, True, 20)
        b = GovernanceEvidence("c0", 10, 2, "council", 0.7, True, 10)
        assert not is_independent(a, b)

    def test_different_colonist(self):
        a = GovernanceEvidence("c0", 10, 1, "council", 0.8, True, 20)
        b = GovernanceEvidence("c1", 10, 1, "council", 0.7, True, 20)
        assert is_independent(a, b)

    def test_different_year(self):
        a = GovernanceEvidence("c0", 10, 1, "council", 0.8, True, 20)
        b = GovernanceEvidence("c0", 20, 1, "council", 0.7, True, 20)
        assert is_independent(a, b)

    def test_count_independent_simple(self):
        evidence = [
            GovernanceEvidence("c0", 10, 1, "council", 0.8, True, 20),
            GovernanceEvidence("c1", 10, 1, "council", 0.7, True, 20),
            GovernanceEvidence("c0", 20, 1, "council", 0.6, True, 20),
        ]
        assert count_independent(evidence) == 3

    def test_count_independent_with_duplicates(self):
        evidence = [
            GovernanceEvidence("c0", 10, 1, "council", 0.8, True, 20),
            GovernanceEvidence("c0", 10, 2, "council", 0.7, True, 10),  # same c0, y10
            GovernanceEvidence("c1", 10, 1, "council", 0.6, True, 20),
        ]
        assert count_independent(evidence) == 2

    def test_count_independent_empty(self):
        assert count_independent([]) == 0


# ────── extract_governance_evidence ──────

class TestExtractEvidence:
    def test_from_world_sim_dict(self):
        data = [{
            "colonist_id": "c0", "year": 10, "depth": 1,
            "dominant_governance": "council",
            "stability_score": 0.8, "survived": True,
            "frames_run": 20,
        }]
        evidence = extract_governance_evidence(data)
        assert len(evidence) == 1
        assert evidence[0].gov_type == "council"

    def test_skips_errors(self):
        data = [{
            "colonist_id": "c0", "year": 10, "depth": 1,
            "error": "budget exhausted",
        }]
        evidence = extract_governance_evidence(data)
        assert len(evidence) == 0

    def test_extracts_children(self):
        data = [{
            "colonist_id": "c0", "year": 10, "depth": 1,
            "dominant_governance": "council",
            "stability_score": 0.8, "survived": True,
            "frames_run": 20,
            "children": [{
                "colonist_id": "m0", "year": 10, "depth": 2,
                "dominant_governance": "consensus",
                "stability_score": 0.6, "survived": True,
                "frames_run": 10,
            }],
        }]
        evidence = extract_governance_evidence(data)
        assert len(evidence) == 2
        assert evidence[1].depth == 2

    def test_handles_compact_summary_keys(self):
        """Compact summaries use shorter key names."""
        data = [{
            "colonist_id": "c0", "year": 10, "depth": 1,
            "dominant_gov": "council",
            "stability": 0.8, "survived": True,
            "frames_run": 20,
        }]
        evidence = extract_governance_evidence(data)
        assert len(evidence) == 1
        assert evidence[0].gov_type == "council"


# ────── score_amendment ──────

class TestScoreAmendment:
    def test_empty_evidence(self):
        assert score_amendment("council", []) == 0.0

    def test_strong_evidence(self):
        evidence = [
            GovernanceEvidence(f"c{i}", yr, 1, "council", 0.9, True, 20)
            for i in range(5) for yr in [10, 20, 30]
        ]
        score = score_amendment("council", evidence)
        assert score > 0.5

    def test_depth_bonus(self):
        shallow = [
            GovernanceEvidence("c0", 10, 1, "council", 0.4, True, 20),
            GovernanceEvidence("c1", 20, 1, "council", 0.4, True, 20),
        ]
        deep = [
            GovernanceEvidence("c0", 10, 3, "council", 0.4, True, 5),
            GovernanceEvidence("c1", 20, 3, "council", 0.4, True, 5),
        ]
        score_shallow = score_amendment("council", shallow)
        score_deep = score_amendment("council", deep)
        assert score_deep > score_shallow

    def test_survival_matters(self):
        survived = [
            GovernanceEvidence("c0", 10, 1, "council", 0.8, True, 20),
            GovernanceEvidence("c1", 20, 1, "council", 0.8, True, 20),
        ]
        failed = [
            GovernanceEvidence("c0", 10, 1, "council", 0.8, False, 20),
            GovernanceEvidence("c1", 20, 1, "council", 0.8, False, 20),
        ]
        assert score_amendment("council", survived) > score_amendment("council", failed)

    def test_score_bounded(self):
        evidence = [
            GovernanceEvidence(f"c{i}", yr, 3, "council", 1.0, True, 20)
            for i in range(20) for yr in range(10, 100, 5)
        ]
        score = score_amendment("council", evidence)
        assert 0.0 <= score <= 1.0


# ────── evaluate_amendments ──────

class TestEvaluateAmendments:
    def test_insufficient_evidence(self):
        evidence = [
            GovernanceEvidence("c0", 10, 1, "council", 0.8, True, 20),
        ]
        proposals = evaluate_amendments(evidence)
        assert len(proposals) == 0

    def test_sufficient_evidence(self):
        evidence = [
            GovernanceEvidence("c0", 10, 1, "council", 0.8, True, 20),
            GovernanceEvidence("c1", 10, 1, "council", 0.7, True, 20),
            GovernanceEvidence("c0", 20, 1, "council", 0.9, True, 20),
            GovernanceEvidence("c2", 30, 1, "council", 0.6, True, 20),
        ]
        proposals = evaluate_amendments(evidence)
        assert len(proposals) == 1
        assert proposals[0].gov_type == "council"

    def test_filters_low_stability(self):
        evidence = [
            GovernanceEvidence("c0", 10, 1, "anarchy", 0.1, True, 20),
            GovernanceEvidence("c1", 20, 1, "anarchy", 0.1, True, 20),
            GovernanceEvidence("c2", 30, 1, "anarchy", 0.1, True, 20),
        ]
        proposals = evaluate_amendments(evidence)
        assert len(proposals) == 0

    def test_requires_distinct_colonists(self):
        evidence = [
            GovernanceEvidence("c0", 10, 1, "council", 0.8, True, 20),
            GovernanceEvidence("c0", 20, 1, "council", 0.8, True, 20),
            GovernanceEvidence("c0", 30, 1, "council", 0.8, True, 20),
        ]
        proposals = evaluate_amendments(evidence)
        assert len(proposals) == 0

    def test_requires_distinct_years(self):
        evidence = [
            GovernanceEvidence("c0", 10, 1, "council", 0.8, True, 20),
            GovernanceEvidence("c1", 10, 1, "council", 0.8, True, 20),
            GovernanceEvidence("c2", 10, 1, "council", 0.8, True, 20),
        ]
        proposals = evaluate_amendments(evidence)
        assert len(proposals) == 0

    def test_multiple_gov_types(self):
        evidence = [
            GovernanceEvidence("c0", 10, 1, "council", 0.8, True, 20),
            GovernanceEvidence("c1", 20, 1, "council", 0.7, True, 20),
            GovernanceEvidence("c2", 30, 1, "council", 0.9, True, 20),
            GovernanceEvidence("c0", 10, 1, "consensus", 0.6, True, 20),
            GovernanceEvidence("c1", 20, 1, "consensus", 0.7, True, 20),
            GovernanceEvidence("c2", 30, 1, "consensus", 0.8, True, 20),
        ]
        proposals = evaluate_amendments(evidence)
        assert len(proposals) == 2

    def test_sorted_by_score(self):
        evidence = [
            GovernanceEvidence("c0", 10, 1, "council", 0.9, True, 20),
            GovernanceEvidence("c1", 20, 1, "council", 0.9, True, 20),
            GovernanceEvidence("c2", 30, 1, "council", 0.9, True, 20),
            GovernanceEvidence("c0", 10, 1, "consensus", 0.5, False, 20),
            GovernanceEvidence("c1", 20, 1, "consensus", 0.5, False, 20),
            GovernanceEvidence("c2", 30, 1, "consensus", 0.5, False, 20),
        ]
        proposals = evaluate_amendments(evidence)
        if len(proposals) >= 2:
            assert proposals[0].score >= proposals[1].score


# ────── AmendmentProposal ──────

class TestAmendmentProposal:
    def test_to_dict(self):
        evidence = [
            GovernanceEvidence("c0", 10, 1, "council", 0.8, True, 20),
            GovernanceEvidence("c1", 20, 1, "council", 0.7, True, 20),
            GovernanceEvidence("c2", 30, 2, "council", 0.9, True, 10),
        ]
        p = AmendmentProposal(
            gov_type="council", evidence=evidence,
            score=0.75, text="test amendment",
        )
        d = p.to_dict()
        assert d["gov_type"] == "council"
        assert d["evidence_count"] == 3
        assert d["distinct_colonists"] == 3
        assert d["distinct_years"] == 3
        assert d["max_depth"] == 2


# ────── format_amendment_text ──────

class TestFormatAmendmentText:
    def test_contains_governance_label(self):
        evidence = [
            GovernanceEvidence("c0", 10, 1, "council", 0.8, True, 20),
            GovernanceEvidence("c1", 20, 1, "council", 0.7, True, 20),
        ]
        text = format_amendment_text("council", evidence, 0.7)
        assert "representative council governance" in text

    def test_contains_evidence_counts(self):
        evidence = [
            GovernanceEvidence("c0", 10, 1, "council", 0.8, True, 20),
            GovernanceEvidence("c1", 20, 2, "council", 0.7, True, 10),
            GovernanceEvidence("c2", 30, 1, "council", 0.9, True, 20),
        ]
        text = format_amendment_text("council", evidence, 0.85)
        assert "3 independent" in text
        assert "3 distinct years" in text
        assert "depth 2" in text

    def test_contains_survival_rate(self):
        evidence = [
            GovernanceEvidence("c0", 10, 1, "council", 0.8, True, 20),
            GovernanceEvidence("c1", 20, 1, "council", 0.7, False, 20),
        ]
        text = format_amendment_text("council", evidence, 0.5)
        assert "50%" in text
