"""Tests for the archaeology engine."""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.archaeology import (
    _normalize_event,
    _normalize_resources,
    load_year,
    TimeSeries,
    extract_time_series,
    detect_change_points,
    _deduplicate_points,
    detect_epochs,
    extract_artifacts,
    aggregate_subsims,
    propose_amendment,
    run_archaeology,
    SubSimSummary,
    CulturalArtifact,
    Epoch,
    ArchaeologyReport,
)


def _make_year(year=1, population=10, event=None, resources=None,
               sub_sims=None, governance_results=None, births=None,
               diary_entries=None, meta_awareness=""):
    return {
        "year": year, "population": population, "event": event,
        "event_effects": {}, "colonist_actions": {},
        "sub_sims": sub_sims or [],
        "governance_results": governance_results or [],
        "resources_snapshot": resources or {"food": 100, "water": 100, "morale": 50},
        "births": births or [], "diary_entries": diary_entries or [],
        "meta_awareness": meta_awareness,
    }


def _write_year_file(tmpdir, year_data):
    p = Path(tmpdir) / ("year-%d.json" % year_data["year"])
    p.write_text(json.dumps(year_data))
    return p


def _make_year_normalized(year, pop=10, morale=50, food=100,
                         sub_sims=None, governance_results=None,
                         diary_entries=None, meta_awareness=""):
    return {
        "year": year, "population": pop,
        "event": {"type": "none", "description": ""},
        "event_effects": {}, "actions": {},
        "sub_sims": sub_sims if sub_sims is not None else [],
        "governance_results": governance_results if governance_results is not None else [],
        "resources": {"food": float(food), "morale": float(morale)},
        "births": [], "diary_entries": diary_entries if diary_entries is not None else [],
        "meta_awareness": meta_awareness,
    }


class TestNormalizeEvent:
    def test_none(self):
        assert _normalize_event(None) == {"type": "none", "description": ""}

    def test_string(self):
        r = _normalize_event("dust_storm")
        assert r["type"] == "dust_storm"

    def test_dict(self):
        r = _normalize_event({"type": "quake", "description": "big one"})
        assert r["type"] == "quake"

    def test_dict_name_key(self):
        r = _normalize_event({"name": "flood"})
        assert r["type"] == "flood"

    def test_other(self):
        r = _normalize_event(42)
        assert r["type"] == "unknown"


class TestNormalizeResources:
    def test_normal(self):
        r = _normalize_resources({"food": 100, "water": "50.5"})
        assert r == {"food": 100.0, "water": 50.5}

    def test_bad_values(self):
        assert _normalize_resources({"food": "oops"}) == {}

    def test_not_dict(self):
        assert _normalize_resources("nope") == {}
        assert _normalize_resources(None) == {}


class TestLoadYear:
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            yd = _make_year(year=5, population=12, event="dust_storm",
                            resources={"food": 80, "morale": 60})
            p = _write_year_file(d, yd)
            loaded = load_year(p)
            assert loaded["year"] == 5
            assert loaded["population"] == 12
            assert loaded["event"]["type"] == "dust_storm"
            assert loaded["resources"]["food"] == 80.0


class TestTimeSeries:
    def test_basic(self):
        ts = TimeSeries(name="pop")
        ts.add(1, 10); ts.add(2, 20); ts.add(3, 30)
        assert ts.mean() == 20.0
        assert ts.min_val() == 10
        assert ts.max_val() == 30

    def test_empty(self):
        ts = TimeSeries(name="x")
        assert ts.mean() == 0.0
        assert ts.stdev() == 0.0

    def test_to_dict(self):
        ts = TimeSeries(name="y")
        ts.add(1, 5); ts.add(2, 15)
        d = ts.to_dict()
        assert d["name"] == "y"
        assert d["n"] == 2
        assert d["mean"] == 10.0


class TestExtractTimeSeries:
    def test_basic(self):
        years = [
            {"year": 1, "population": 10, "resources": {"food": 100, "morale": 50}},
            {"year": 2, "population": 12, "resources": {"food": 90, "morale": 55}},
        ]
        series = extract_time_series(years)
        assert series["population"].values == [10.0, 12.0]
        assert series["food"].values == [100.0, 90.0]


class TestChangePointDetection:
    def test_constant_signal(self):
        assert detect_change_points([50.0] * 40) == []

    def test_step_function(self):
        vals = [10.0] * 20 + [50.0] * 20
        pts = detect_change_points(vals, window=5, threshold=1.5)
        assert len(pts) > 0
        assert any(15 <= p <= 25 for p in pts)

    def test_too_short(self):
        assert detect_change_points([1, 2, 3]) == []

    def test_dedup(self):
        pts = _deduplicate_points([5, 6, 7, 20, 21], min_gap=5)
        assert pts == [5, 20]


class TestEpochDetection:
    def test_empty(self):
        assert detect_epochs([]) == []

    def test_single_epoch_from_flat_data(self):
        years = [_make_year_normalized(i) for i in range(1, 31)]
        epochs = detect_epochs(years)
        assert len(epochs) >= 1
        assert epochs[0].label == "The Founding"

    def test_step_creates_two_epochs(self):
        years = []
        for i in range(1, 21):
            years.append(_make_year_normalized(i, pop=10, morale=50))
        for i in range(21, 41):
            years.append(_make_year_normalized(i, pop=30, morale=80))
        epochs = detect_epochs(years, window=5, threshold=1.5)
        assert len(epochs) >= 2


class TestArtifactExtraction:
    def test_subsim_patterns(self):
        years = [_make_year_normalized(i, sub_sims=[
            {"colonist": "ada", "proposal": "water_ration", "depth": 1, "s_expr": "(sim water)"},
        ] * 4) for i in range(1, 11)]
        artifacts = extract_artifacts(years)
        assert any(a.artifact_type == "sub_sim_tradition" for a in artifacts)

    def test_governance_patterns(self):
        years = [_make_year_normalized(i, governance_results=["election held", "directive approved"])
                 for i in range(1, 6)]
        artifacts = extract_artifacts(years)
        assert any(a.artifact_type == "governance_ritual" for a in artifacts)


class TestSubSimAggregation:
    def test_basic(self):
        years = [
            _make_year_normalized(1, sub_sims=[
                {"colonist": "ada", "proposal": "water", "depth": 1, "s_expr": "()"},
                {"colonist": "bob", "proposal": "food", "depth": 2, "s_expr": "(nested)"},
            ]),
            _make_year_normalized(2, sub_sims=[
                {"colonist": "ada", "proposal": "water", "depth": 1, "s_expr": "()"},
            ]),
        ]
        summary = aggregate_subsims(years)
        assert summary.total == 3
        assert summary.by_colonist["ada"] == 2
        assert summary.max_depth == 2

    def test_empty(self):
        summary = aggregate_subsims([])
        assert summary.total == 0


class TestAmendmentProposal:
    def test_strong_evidence(self):
        epochs = [Epoch("Founding", 1, 50, "start", 15, 60, [], [], 10, 0),
                  Epoch("Expansion", 51, 100, "boom", 25, 70, [], [], 20, 0)]
        subsim = SubSimSummary(total=600, max_depth=3)
        years = [_make_year_normalized(i, meta_awareness="simulation detected")
                 for i in range(1, 20)]
        result = propose_amendment(epochs, [], subsim, years)
        assert "xix" in result["id"]
        assert result["evidence"]["total_subsims"] == 600
        assert "Self-Archaeology" in result["title"]

    def test_weak_evidence(self):
        result = propose_amendment([], [], SubSimSummary(total=5), [])
        assert "xix" in result["id"]


class TestFullReport:
    def test_with_year_files(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(1, 31):
                _write_year_file(d, _make_year(year=i, population=10 + i,
                                               resources={"food": 100 - i, "morale": 50 + i}))
            report = run_archaeology(d)
            assert report.years_analyzed == 30
            assert report.final_population == 40
            assert len(report.epochs) >= 1
            d_out = report.to_dict()
            assert "colony_name" in d_out

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            report = run_archaeology(d)
            assert report.years_analyzed == 0


class TestInvariants:
    def test_epochs_cover_all_years(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(1, 51):
                pop = 10 if i < 25 else 30
                _write_year_file(d, _make_year(year=i, population=pop,
                                               resources={"food": 100, "morale": 50}))
            report = run_archaeology(d)
            if report.epochs:
                assert report.epochs[0].start_year == 1
                assert report.epochs[-1].end_year == 50

    def test_subsim_total_is_sum(self):
        years = [_make_year_normalized(i, sub_sims=[
            {"colonist": "c%d" % (i % 3), "proposal": "p", "depth": 1, "s_expr": "()"}
        ]) for i in range(1, 20)]
        summary = aggregate_subsims(years)
        assert summary.total == sum(summary.by_colonist.values())

    def test_artifact_significance_bounded(self):
        years = [_make_year_normalized(
            i, sub_sims=[{"colonist": "a", "proposal": "test", "depth": 1, "s_expr": "()"}] * 5,
            governance_results=["election", "directive"],
            diary_entries=[{"entry": "hope and fear and storm"}],
            meta_awareness="simulation awareness level %d" % i,
        ) for i in range(1, 50)]
        artifacts = extract_artifacts(years)
        for a in artifacts:
            assert 0 <= a.significance <= 1.0

    def test_time_series_population_nonnegative(self):
        years = [_make_year_normalized(i, pop=max(0, 10 - i)) for i in range(1, 20)]
        series = extract_time_series(years)
        for v in series["population"].values:
            assert v >= 0

    def test_report_serializable(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(1, 11):
                _write_year_file(d, _make_year(year=i))
            report = run_archaeology(d)
            data = report.to_dict()
            json_str = json.dumps(data)
            assert json_str
            parsed = json.loads(json_str)
            assert parsed["colony_name"] == "Mars-100"
