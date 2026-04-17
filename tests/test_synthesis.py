"""Tests for Mars-100 post-simulation synthesis engine."""
from __future__ import annotations

import json
import math
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.synthesis import (
    detect_factions, Faction,
    analyze_governance_cycles, GovernanceCycle,
    analyze_stagnation, StagnationReport,
    find_resource_crises, ResourceCrisis,
    analyze_meta_awareness, MetaAwarenessArc,
    generate_amendment, AmendmentProposal,
    synthesize, SynthesisResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_colonist(cid: int, alive: bool = True, element: str = "fire",
                   relationships: dict | None = None) -> dict:
    """Create a minimal colonist dict for testing."""
    return {
        "id": cid, "name": f"Col-{cid}", "element": element,
        "alive": alive, "year_born": 0,
        "stats": {"resolve": 50, "improvisation": 50, "empathy": 50,
                  "hoarding": 50, "faith": 50, "paranoia": 50},
        "skills": {"terraforming": 30, "hydroponics": 30, "mediation": 30,
                   "coding": 30, "prayer": 30, "sabotage": 30},
        "relationships": relationships or {},
        "sub_sims_run": 0, "diary": [],
    }


def _make_two_factions(n_per_faction: int = 4) -> list[dict]:
    """Create colonists in two clearly separated factions."""
    colonists = []
    for i in range(n_per_faction):
        rels = {}
        for j in range(n_per_faction):
            if i != j:
                rels[str(j)] = 80.0
        for j in range(n_per_faction, 2 * n_per_faction):
            rels[str(j)] = -50.0
        colonists.append(_make_colonist(i, element="fire", relationships=rels))

    for i in range(n_per_faction, 2 * n_per_faction):
        rels = {}
        for j in range(n_per_faction, 2 * n_per_faction):
            if i != j:
                rels[str(j)] = 80.0
        for j in range(n_per_faction):
            rels[str(j)] = -50.0
        colonists.append(_make_colonist(i, element="water", relationships=rels))

    return colonists


def _make_deltas(n_years: int = 100) -> list[dict]:
    """Create minimal year deltas for testing."""
    deltas = []
    for y in range(1, n_years + 1):
        actions = []
        if y > 40:
            actions.append({
                "colonist": "Col-0",
                "action": "Col-0 considers proposing constitutional_amendment but it already exists"
            })
        else:
            actions.append({
                "colonist": "Col-0",
                "action": "Col-0 proposes leadership_election: new leader needed"
            })
        deltas.append({
            "year": y,
            "population": 10,
            "event": {"id": "dust_storm", "desc": "Storm", "severity": 0.5},
            "event_effects": [],
            "colonist_actions": actions,
            "sub_sims": [],
            "governance_results": [],
            "resource_effects": [],
            "births": [],
            "diary_entries": [],
            "meta_awareness": "",
            "resources_snapshot": {
                "food": 1000.0 if y < 70 else 150.0,
                "water": 1200.0, "power": 400.0,
                "oxygen": 800.0, "materials": 500.0, "morale": 60.0,
            },
        })
    return deltas


def _make_governance(amendments: list[dict] | None = None) -> dict:
    return {
        "system": "direct_democracy",
        "leader": 0,
        "amendments": amendments or [
            {"year": 11, "text": "right to spiritual practice", "proposer": 2},
            {"year": 24, "text": "knowledge shared openly", "proposer": 0},
            {"year": 32, "text": "surplus distributed equally", "proposer": 10},
            {"year": 42, "text": "emergency powers expire", "proposer": 5},
            {"year": 58, "text": "sub-sims are a right", "proposer": 6},
        ],
    }


def _make_summary(meta_events: list[str] | None = None) -> dict:
    return {
        "years_survived": 100,
        "final_population": 25,
        "peak_population": 35,
        "total_births": 29,
        "total_deaths": 14,
        "total_sub_simulations": 1304,
        "total_proposals": 116,
        "governance_system": "direct_democracy",
        "meta_awareness_events": meta_events or [
            "Titan (year 31): 'Are we in a sub-simulation?'",
            "Ares-18 (year 32): 'I wonder if the data sloshing through our colony matters.'",
            "Phobos (year 33): 'The sub-sim predicted our decision before we made it.'",
            "Lyra (year 34): 'Three levels of simulation deep and the pattern repeats at every scale.'",
        ],
        "population_curve": list(range(10, 36)) + list(range(35, 25, -1)),
        "morale_curve": [60.0 + i * 0.1 for i in range(36)],
    }


def _make_full_data() -> dict:
    colonists = _make_two_factions(4)
    return {
        "_meta": {"engine": "mars-100", "version": "1.0"},
        "colony": {
            "colonists": colonists,
            "governance": _make_governance(),
        },
        "deltas": _make_deltas(),
        "summary": _make_summary(),
    }


# ---------------------------------------------------------------------------
# Faction detection tests
# ---------------------------------------------------------------------------

class TestFactionDetection:
    def test_two_clear_factions(self):
        colonists = _make_two_factions(4)
        factions = detect_factions(colonists)
        assert len(factions) >= 2
        member_sets = [set(f.members) for f in factions]
        assert all(len(s) >= 2 for s in member_sets)

    def test_empty_colony(self):
        assert detect_factions([]) == []

    def test_all_dead(self):
        colonists = [_make_colonist(i, alive=False) for i in range(5)]
        assert detect_factions(colonists) == []

    def test_faction_labels_are_strings(self):
        colonists = _make_two_factions(3)
        factions = detect_factions(colonists, min_faction_size=2)
        for f in factions:
            assert isinstance(f.label, str)
            assert len(f.label) > 0

    def test_serialization(self):
        colonists = _make_two_factions(3)
        factions = detect_factions(colonists, min_faction_size=2)
        for f in factions:
            d = f.to_dict()
            assert "members" in d
            assert "avg_internal_trust" in d
            assert isinstance(d["avg_internal_trust"], float)

    def test_single_colonist_no_factions(self):
        colonists = [_make_colonist(0, relationships={})]
        assert detect_factions(colonists) == []

    def test_faction_members_are_alive(self):
        mixed = [_make_colonist(i, alive=(i < 6), element="fire",
                               relationships={str(j): 90.0 for j in range(8) if j != i})
                 for i in range(8)]
        factions = detect_factions(mixed)
        alive_ids = {c["id"] for c in mixed if c["alive"]}
        for f in factions:
            for m in f.members:
                assert m in alive_ids


# ---------------------------------------------------------------------------
# Governance analysis tests
# ---------------------------------------------------------------------------

class TestGovernanceCycles:
    def test_single_system(self):
        deltas = _make_deltas(50)
        gov = _make_governance([])
        cycles = analyze_governance_cycles(deltas, gov)
        assert len(cycles) >= 1
        assert cycles[-1].end_year is None

    def test_cycle_serialization(self):
        deltas = _make_deltas(20)
        gov = _make_governance()
        cycles = analyze_governance_cycles(deltas, gov)
        for c in cycles:
            d = c.to_dict()
            assert "system" in d
            assert "duration" in d
            assert isinstance(d["duration"], int)


class TestStagnation:
    def test_detects_redundancy(self):
        deltas = _make_deltas(100)
        gov = _make_governance()
        report = analyze_stagnation(deltas, gov)
        assert report.total_redundant_proposals > 0
        assert report.redundancy_rate > 0.0

    def test_onset_year_reasonable(self):
        deltas = _make_deltas(100)
        gov = _make_governance()
        report = analyze_stagnation(deltas, gov)
        if report.stagnation_onset_year is not None:
            assert 1 <= report.stagnation_onset_year <= 100

    def test_amendment_timeline(self):
        gov = _make_governance()
        report = analyze_stagnation(_make_deltas(), gov)
        assert len(report.amendment_timeline) == len(gov["amendments"])

    def test_serialization(self):
        report = analyze_stagnation(_make_deltas(), _make_governance())
        d = report.to_dict()
        assert "redundancy_rate" in d
        assert isinstance(d["redundancy_rate"], float)

    def test_no_stagnation_with_active_governance(self):
        deltas = []
        for y in range(1, 101):
            deltas.append({
                "year": y, "colonist_actions": [
                    {"colonist": "Col-0", "action": "Col-0 proposes new_rule: fresh idea"}
                ],
                "governance_results": [], "resources_snapshot": {},
            })
        report = analyze_stagnation(deltas, _make_governance())
        assert report.total_redundant_proposals == 0


# ---------------------------------------------------------------------------
# Resource crisis tests
# ---------------------------------------------------------------------------

class TestResourceCrises:
    def test_finds_food_crisis(self):
        deltas = _make_deltas(100)
        crises = find_resource_crises(deltas, threshold=200.0)
        food_crises = [c for c in crises if c.resource == "food"]
        assert len(food_crises) > 0

    def test_no_crisis_when_abundant(self):
        deltas = _make_deltas(10)
        crises = find_resource_crises(deltas, threshold=50.0)
        assert len(crises) == 0

    def test_crisis_serialization(self):
        crises = find_resource_crises(_make_deltas(), threshold=200.0)
        if crises:
            d = crises[0].to_dict()
            assert "year" in d
            assert "resource" in d
            assert isinstance(d["level"], float)


# ---------------------------------------------------------------------------
# Meta-awareness tests
# ---------------------------------------------------------------------------

class TestMetaAwareness:
    def test_parses_events(self):
        summary = _make_summary()
        arc = analyze_meta_awareness(summary)
        assert arc.total_events == 4
        assert arc.first_event_year == 31
        assert arc.unique_colonists >= 3

    def test_detects_themes(self):
        summary = _make_summary()
        arc = analyze_meta_awareness(summary)
        assert len(arc.theme_distribution) > 0

    def test_empty_events(self):
        arc = analyze_meta_awareness({"meta_awareness_events": []})
        assert arc.total_events == 0
        assert arc.first_event_year is None

    def test_serialization(self):
        arc = analyze_meta_awareness(_make_summary())
        d = arc.to_dict()
        assert "theme_distribution" in d
        assert isinstance(d["total_events"], int)


# ---------------------------------------------------------------------------
# Amendment proposal tests
# ---------------------------------------------------------------------------

class TestAmendmentProposal:
    def test_generates_proposal(self):
        stagnation = analyze_stagnation(_make_deltas(), _make_governance())
        factions = detect_factions(_make_two_factions())
        meta_arc = analyze_meta_awareness(_make_summary())
        amendment = generate_amendment(stagnation, factions, meta_arc, _make_governance())
        assert amendment.title
        assert amendment.number == "XVIII"
        assert len(amendment.text) > 50
        assert 0.0 <= amendment.confidence <= 1.0

    def test_evidence_not_empty(self):
        stagnation = analyze_stagnation(_make_deltas(), _make_governance())
        factions = detect_factions(_make_two_factions())
        meta_arc = analyze_meta_awareness(_make_summary())
        amendment = generate_amendment(stagnation, factions, meta_arc, _make_governance())
        assert len(amendment.evidence) > 0

    def test_serialization(self):
        stagnation = analyze_stagnation(_make_deltas(), _make_governance())
        amendment = generate_amendment(stagnation, [], analyze_meta_awareness({}), {})
        d = amendment.to_dict()
        assert "title" in d
        assert "confidence" in d
        assert isinstance(d["confidence"], float)


# ---------------------------------------------------------------------------
# Full synthesis tests
# ---------------------------------------------------------------------------

class TestFullSynthesis:
    def test_smoke(self):
        data = _make_full_data()
        result = synthesize(data)
        assert isinstance(result, SynthesisResult)
        assert len(result.key_findings) > 0

    def test_serialization_roundtrip(self):
        data = _make_full_data()
        result = synthesize(data)
        d = result.to_dict()
        json_str = json.dumps(d, indent=2)
        parsed = json.loads(json_str)
        assert "factions" in parsed
        assert "stagnation" in parsed
        assert "amendment_proposal" in parsed
        assert "key_findings" in parsed

    def test_empty_data(self):
        result = synthesize({})
        assert isinstance(result, SynthesisResult)
        assert result.stagnation.total_redundant_proposals == 0

    def test_findings_are_strings(self):
        data = _make_full_data()
        result = synthesize(data)
        for finding in result.key_findings:
            assert isinstance(finding, str)
            assert len(finding) > 10

    def test_on_real_data(self):
        """Run synthesis on actual published data if available."""
        data_path = Path(__file__).resolve().parent.parent / "docs" / "mars-100" / "data.json"
        if not data_path.exists():
            pytest.skip("published data.json not available")
        data = json.loads(data_path.read_text())
        result = synthesize(data)
        assert result.stagnation.total_redundant_proposals > 0
        assert result.amendment.confidence > 0.0
        assert len(result.key_findings) >= 3
        d = result.to_dict()
        json.dumps(d)  # must be serializable


# ---------------------------------------------------------------------------
# Property-based invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_confidence_bounded(self):
        for _ in range(5):
            stagnation = StagnationReport(
                stagnation_onset_year=30, total_redundant_proposals=100,
                total_successful_proposals=20, redundancy_rate=0.83,
                saturation_year=60, amendment_timeline=[])
            factions = detect_factions(_make_two_factions())
            amendment = generate_amendment(
                stagnation, factions, analyze_meta_awareness(_make_summary()),
                _make_governance())
            assert 0.0 <= amendment.confidence <= 1.0

    def test_faction_members_disjoint(self):
        colonists = _make_two_factions(5)
        factions = detect_factions(colonists)
        all_members: set[int] = set()
        for f in factions:
            overlap = all_members & set(f.members)
            assert len(overlap) == 0, f"Overlapping members: {overlap}"
            all_members.update(f.members)

    def test_crisis_years_within_range(self):
        deltas = _make_deltas(100)
        crises = find_resource_crises(deltas)
        for c in crises:
            assert 1 <= c.year <= 100

    def test_stagnation_rate_bounded(self):
        report = analyze_stagnation(_make_deltas(), _make_governance())
        assert 0.0 <= report.redundancy_rate <= 1.0
