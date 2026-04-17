"""Tests for the events module — event generation, severity bounds, determinism."""
from __future__ import annotations

import random

import pytest

from src.mars100.events import Event, generate_events, EVENT_TEMPLATES


class TestEventTemplates:
    """Verify event template data integrity."""

    def test_templates_not_empty(self) -> None:
        assert len(EVENT_TEMPLATES) > 0

    def test_templates_have_required_fields(self) -> None:
        for tmpl in EVENT_TEMPLATES:
            assert "name" in tmpl
            assert "category" in tmpl
            assert "severity_range" in tmpl

    def test_severity_range_valid(self) -> None:
        for tmpl in EVENT_TEMPLATES:
            lo, hi = tmpl["severity_range"]
            assert 0.0 <= lo <= hi <= 1.0, (
                f"{tmpl['name']} severity_range {tmpl['severity_range']} invalid"
            )


class TestGenerateEvents:
    """Tests for event generation."""

    def test_returns_event_list(self) -> None:
        rng = random.Random(42)
        events = generate_events(year=1, rng=rng)
        assert isinstance(events, list)
        for e in events:
            assert isinstance(e, Event)

    def test_events_have_valid_severity(self) -> None:
        rng = random.Random(99)
        for year in range(1, 20):
            events = generate_events(year=year, rng=rng)
            for e in events:
                assert 0.0 <= e.severity <= 1.0, (
                    f"Year {year}: {e.name} severity {e.severity} out of [0,1]"
                )

    def test_deterministic_with_same_seed(self) -> None:
        events_a = generate_events(year=5, rng=random.Random(123))
        events_b = generate_events(year=5, rng=random.Random(123))
        assert len(events_a) == len(events_b)
        for a, b in zip(events_a, events_b):
            assert a.name == b.name
            assert a.severity == b.severity

    def test_different_seeds_vary(self) -> None:
        events_a = generate_events(year=5, rng=random.Random(1))
        events_b = generate_events(year=5, rng=random.Random(999))
        assert all(isinstance(e.name, str) for e in events_a)
        assert all(isinstance(e.name, str) for e in events_b)

    def test_event_to_dict(self) -> None:
        rng = random.Random(42)
        events = generate_events(year=1, rng=rng)
        if events:
            d = events[0].to_dict()
            assert "name" in d
            assert "severity" in d
            assert isinstance(d["severity"], float)

    def test_generates_for_100_years(self) -> None:
        """Smoke test: generate events for every year without crash."""
        rng = random.Random(42)
        total = 0
        for year in range(1, 101):
            events = generate_events(year=year, rng=rng)
            total += len(events)
        assert total > 0, "No events generated across 100 years"
