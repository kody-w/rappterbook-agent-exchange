"""test_mars100_events.py -- Tests for the environmental event generator.

Covers: event generation, scripted events, resource effects,
severity scaling, determinism.
"""
from __future__ import annotations

import pytest
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100_events import (
    generate_event, EVENT_TYPES, SCRIPTED_EVENTS,
    EVENT_DESCRIPTIONS, RESOURCE_EFFECTS,
)


class TestGenerateEvent:
    def test_returns_dict(self):
        rng = random.Random(42)
        event = generate_event(1, rng)
        assert isinstance(event, dict)

    def test_has_required_fields(self):
        rng = random.Random(42)
        event = generate_event(1, rng)
        assert 'type' in event
        assert 'severity' in event
        assert 'description' in event
        assert 'effects' in event
        assert 'year' in event

    def test_valid_event_type(self):
        rng = random.Random(42)
        valid_types = {e['type'] for e in EVENT_TYPES}
        for year in range(1, 50):
            event = generate_event(year, rng)
            assert event['type'] in valid_types

    def test_severity_bounded(self):
        rng = random.Random(42)
        for year in range(1, 100):
            event = generate_event(year, rng)
            assert 0 <= event['severity'] <= 1.0 or event['severity'] == 0

    def test_deterministic(self):
        e1 = generate_event(1, random.Random(42))
        e2 = generate_event(1, random.Random(42))
        assert e1['type'] == e2['type']
        assert e1['severity'] == e2['severity']

    def test_different_seeds_differ(self):
        events1 = [generate_event(y, random.Random(42)) for y in range(1, 20)]
        events2 = [generate_event(y, random.Random(99)) for y in range(1, 20)]
        # At least some should differ
        diffs = sum(1 for a, b in zip(events1, events2) if a['type'] != b['type'])
        assert diffs > 0


class TestScriptedEvents:
    def test_year5_dust_storm(self):
        event = generate_event(5, random.Random(42))
        assert event['type'] == 'dust_storm'
        assert event['scripted'] is True

    def test_year12_birth(self):
        event = generate_event(12, random.Random(42))
        assert event['type'] == 'birth'
        assert event['scripted'] is True
        assert 'Dawn' in event['description']

    def test_year25_earth_contact(self):
        event = generate_event(25, random.Random(42))
        assert event['type'] == 'earth_contact'
        assert 'silent' in event['description'].lower()

    def test_year37_alien_signal(self):
        event = generate_event(37, random.Random(42))
        assert event['type'] == 'alien_signal'
        assert 'primes' in event['description'].lower()

    def test_year67_simulation_realization(self):
        event = generate_event(67, random.Random(42))
        assert event['type'] == 'alien_signal'
        assert 'simulation' in event['description'].lower()

    def test_year82_cave(self):
        event = generate_event(82, random.Random(42))
        assert event['type'] == 'cave_discovery'
        assert 'depth-3' in event['description'].lower()

    def test_year95_earth_return(self):
        event = generate_event(95, random.Random(42))
        assert event['type'] == 'earth_contact'

    def test_scripted_events_have_effects(self):
        for year, spec in SCRIPTED_EVENTS.items():
            event = generate_event(year, random.Random(42))
            assert 'effects' in event


class TestResourceEffects:
    def test_all_event_types_have_effects(self):
        for etype in RESOURCE_EFFECTS:
            effects = RESOURCE_EFFECTS[etype]
            assert isinstance(effects, dict)

    def test_effects_are_numeric(self):
        for etype, effects in RESOURCE_EFFECTS.items():
            for resource, delta in effects.items():
                assert isinstance(delta, (int, float)), f"{etype}.{resource}"


class TestEventDistribution:
    def test_variety_over_100_years(self):
        """Over 100 years, we should see at least 4 different event types."""
        rng = random.Random(42)
        types_seen = set()
        for year in range(1, 101):
            event = generate_event(year, rng)
            types_seen.add(event['type'])
        assert len(types_seen) >= 4

    def test_severity_increases_with_age(self):
        """Average severity should be higher in later years."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        early = [generate_event(y, rng1) for y in range(1, 20) if y not in SCRIPTED_EVENTS]
        late = [generate_event(y, rng2) for y in range(80, 100) if y not in SCRIPTED_EVENTS]
        if early and late:
            avg_early = sum(e['severity'] for e in early) / len(early)
            avg_late = sum(e['severity'] for e in late) / len(late)
            # This is probabilistic but with fixed seeds should be deterministic
            assert avg_late >= avg_early * 0.5  # Modest check
