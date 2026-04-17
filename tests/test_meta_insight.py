"""Tests for Mars-100 meta-insight extraction."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars100.meta_insight import (
    extract_meta_insight, should_promote_amendment,
    format_amendment_proposal, _compute_strength, _safe_float,
)


class TestExtractMetaInsight:
    def test_depth_1_returns_none(self):
        result = extract_meta_insight(
            {"result": 0.9, "error": None}, depth=1, year=50)
        assert result is None

    def test_depth_2_with_high_result(self):
        insight = extract_meta_insight(
            {"result": 0.9, "error": None}, depth=2, year=50)
        assert insight is not None
        assert insight["depth"] == 2
        assert insight["year"] == 50
        assert "proposed_amendment" in insight

    def test_depth_3_with_high_result(self):
        insight = extract_meta_insight(
            {"result": 1.5, "error": None}, depth=3, year=75)
        assert insight is not None
        assert insight["depth"] == 3
        assert insight["strength"] > 0

    def test_error_result_returns_none(self):
        result = extract_meta_insight(
            {"result": 0.9, "error": "some error"}, depth=2, year=50)
        assert result is None

    def test_none_result_returns_none(self):
        result = extract_meta_insight(
            {"result": None, "error": None}, depth=2, year=50)
        assert result is None

    def test_low_result_may_not_trigger(self):
        result = extract_meta_insight(
            {"result": 0.1, "error": None}, depth=2, year=50)
        # Low values might not match any template conditions
        # (some templates require > 0.6 or > 0.8)

    def test_list_result(self):
        insight = extract_meta_insight(
            {"result": [0.8, 0.9], "error": None}, depth=2, year=50)
        assert insight is not None
        assert insight["type"] == "consensus_emergence"

    def test_negative_result(self):
        insight = extract_meta_insight(
            {"result": -0.7, "error": None}, depth=2, year=50)
        assert insight is not None
        assert insight["type"] == "exile_justice"


class TestShouldPromoteAmendment:
    def test_no_insights(self):
        assert should_promote_amendment([]) is None

    def test_below_threshold(self):
        insights = [{"strength": 0.3, "type": "test", "proposed_amendment": "x"}]
        assert should_promote_amendment(insights) is None

    def test_above_threshold(self):
        insights = [{"strength": 0.8, "type": "test", "proposed_amendment": "x"}]
        result = should_promote_amendment(insights)
        assert result is not None
        assert result["strength"] == 0.8

    def test_picks_strongest(self):
        insights = [
            {"strength": 0.6, "type": "a", "proposed_amendment": "x"},
            {"strength": 0.9, "type": "b", "proposed_amendment": "y"},
            {"strength": 0.7, "type": "c", "proposed_amendment": "z"},
        ]
        result = should_promote_amendment(insights)
        assert result["type"] == "b"

    def test_custom_threshold(self):
        insights = [{"strength": 0.3, "type": "a", "proposed_amendment": "x"}]
        assert should_promote_amendment(insights, threshold=0.2) is not None
        assert should_promote_amendment(insights, threshold=0.5) is None


class TestFormatAmendmentProposal:
    def test_format_structure(self):
        insight = {
            "type": "governance_quality", "year": 50, "depth": 3,
            "proposed_amendment": "Test amendment text",
            "rationale": "Test rationale",
            "strength": 0.85,
        }
        text = format_amendment_proposal(insight)
        assert "## Proposed Amendment" in text
        assert "Test amendment text" in text
        assert "depth-3" in text
        assert "0.85" in text


class TestHelpers:
    def test_safe_float_number(self):
        assert _safe_float(0.5) == 0.5
        assert _safe_float(42) == 42.0

    def test_safe_float_list(self):
        assert _safe_float([0.7, 0.8]) == 0.7

    def test_safe_float_none(self):
        assert _safe_float(None) == 0.0

    def test_safe_float_string(self):
        assert _safe_float("hello") == 0.0

    def test_compute_strength(self):
        s1 = _compute_strength(0.9, depth=1)
        s2 = _compute_strength(0.9, depth=2)
        s3 = _compute_strength(0.9, depth=3)
        assert s2 > s1  # deeper = stronger
        assert s3 > s2

    def test_strength_clamped(self):
        s = _compute_strength(100.0, depth=3)
        assert s <= 1.0
        s2 = _compute_strength(0.0, depth=0)
        assert s2 >= 0.0
