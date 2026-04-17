"""Tests for Mars-100 narrator module."""
from __future__ import annotations

import random
import pytest
from src.mars100.narrator import narrate_year, generate_diary_entries, generate_final_report


SAMPLE_YEAR = {
    "year": 5,
    "events": [{"name": "dust_storm", "category": "environment",
                "description": "Massive dust storm", "severity": 0.7,
                "effects": {"food": -0.1}}],
    "actions": {"c1": "farm", "c2": "terraform", "c3": "mediate"},
    "subsim_log": [{"depth": 1, "colonist_id": "c1", "year": 5,
                    "expression": "(+ 1 2)", "result": 3}],
    "governance": {"gov_type": "council", "passed": True,
                   "votes_for": ["c1", "c2"], "votes_against": ["c3"]},
    "resources_before": {"food": 0.6, "water": 0.7, "power": 0.8, "air": 0.9, "medicine": 0.5},
    "resources_after": {"food": 0.5, "water": 0.65, "power": 0.75, "air": 0.85, "medicine": 0.45},
    "resource_delta": {"food": -0.1, "water": -0.05, "power": -0.05, "air": -0.05, "medicine": -0.05},
    "deaths": [{"id": "c4", "name": "Orinthe", "cause": "radiation", "year": 5}],
    "exiles": [],
    "meta_awareness": [{"colonist_id": "c1", "year": 5,
                        "insight": "What if we are variables?"}],
    "social_cohesion": 0.55,
    "governance_state": {"gov_type": "council"},
    "colonist_snapshots": [
        {"id": "c1", "name": "Kael", "element": "fire", "archetype": "commander",
         "alive": True, "exiled": False},
        {"id": "c2", "name": "Seren", "element": "water", "archetype": "healer",
         "alive": True, "exiled": False},
        {"id": "c3", "name": "Thorne", "element": "earth", "archetype": "engineer",
         "alive": True, "exiled": False},
    ],
}


class TestNarrateYear:
    def test_basic(self):
        text = narrate_year(SAMPLE_YEAR, random.Random(42))
        assert "Year 5" in text
        assert "dust_storm" in text

    def test_contains_resources(self):
        text = narrate_year(SAMPLE_YEAR, random.Random(42))
        assert "food" in text.lower()

    def test_contains_governance(self):
        text = narrate_year(SAMPLE_YEAR, random.Random(42))
        assert "council" in text.lower()

    def test_contains_death(self):
        text = narrate_year(SAMPLE_YEAR, random.Random(42))
        assert "Orinthe" in text

    def test_contains_meta(self):
        text = narrate_year(SAMPLE_YEAR, random.Random(42))
        assert "variables" in text.lower()


class TestDiaryEntries:
    def test_generates_entries(self):
        entries = generate_diary_entries(SAMPLE_YEAR, SAMPLE_YEAR["colonist_snapshots"],
                                        random.Random(42), count=2)
        assert len(entries) >= 1

    def test_entry_structure(self):
        entries = generate_diary_entries(SAMPLE_YEAR, SAMPLE_YEAR["colonist_snapshots"],
                                        random.Random(42))
        for e in entries:
            assert "colonist_id" in e
            assert "year" in e
            assert "text" in e

    def test_empty_colonists(self):
        entries = generate_diary_entries(SAMPLE_YEAR, [], random.Random(42))
        assert entries == []


SAMPLE_SIM_RESULT = {
    "_meta": {"engine": "mars-100"},
    "summary": {"total_deaths": 3, "total_exiles": 1,
                "total_subsims": 15, "governance_changes": 4,
                "meta_awareness_events": 7, "final_cohesion": 0.45},
    "final_governance": {"gov_type": "consensus", "history": [
        {"year": 5, "from": "anarchy", "to": "council"},
        {"year": 20, "from": "council", "to": "consensus"},
    ], "constitution": ["Rule 1", "Rule 2"]},
    "final_colonists": [
        {"id": "c1", "name": "Kael", "element": "fire", "archetype": "commander",
         "alive": True, "exiled": False},
        {"id": "c2", "name": "Seren", "element": "water", "archetype": "healer",
         "alive": False, "death_year": 50, "death_cause": "radiation", "exiled": False},
        {"id": "c3", "name": "Jax", "element": "fire", "archetype": "trickster",
         "alive": True, "exiled": True, "exile_year": 30},
    ],
    "final_resources": {"food": 0.4, "water": 0.5, "power": 0.6, "air": 0.7, "medicine": 0.3},
    "years": [],
}


class TestFinalReport:
    def test_basic(self):
        report = generate_final_report(SAMPLE_SIM_RESULT)
        assert "Mars-100" in report
        assert "Emergent Governance" in report

    def test_contains_summary(self):
        report = generate_final_report(SAMPLE_SIM_RESULT)
        assert "Deaths" in report and "3" in report

    def test_contains_governance_timeline(self):
        report = generate_final_report(SAMPLE_SIM_RESULT)
        assert "anarchy" in report.lower()
        assert "council" in report.lower()

    def test_contains_amendment(self):
        report = generate_final_report(SAMPLE_SIM_RESULT)
        assert "Amendment" in report

    def test_contains_roster(self):
        report = generate_final_report(SAMPLE_SIM_RESULT)
        assert "Kael" in report

    def test_contains_dead(self):
        report = generate_final_report(SAMPLE_SIM_RESULT)
        assert "Seren" in report
        assert "radiation" in report
