"""test_colonist.py -- Tests for the colonist model.

Covers: creation, stats bounds, LisPy conversion, relationship
symmetry, stat evolution, serialization.
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.colonist import (
    create_colonists, colonist_to_lispy, evolve_stats,
    evolve_relationships, serialize_colonist, clamp_stat,
    COLONIST_TEMPLATES, STAT_NAMES, SKILL_NAMES, ELEMENTS,
)
from src.lispy import Symbol


class TestCreateColonists:
    def test_creates_10(self):
        colonists = create_colonists()
        assert len(colonists) == 10

    def test_unique_ids(self):
        colonists = create_colonists()
        ids = [c['id'] for c in colonists]
        assert len(set(ids)) == 10

    def test_all_alive(self):
        colonists = create_colonists()
        assert all(c['alive'] for c in colonists)

    def test_valid_elements(self):
        colonists = create_colonists()
        for c in colonists:
            assert c['element'] in ELEMENTS

    def test_stats_in_range(self):
        colonists = create_colonists()
        for c in colonists:
            for stat in STAT_NAMES:
                assert 0 <= c['stats'][stat] <= 100, f"{c['id']}.{stat} = {c['stats'][stat]}"

    def test_skills_in_range(self):
        colonists = create_colonists()
        for c in colonists:
            for skill in SKILL_NAMES:
                assert 0 <= c['skills'][skill] <= 100, f"{c['id']}.{skill}"

    def test_relationships_initialized(self):
        colonists = create_colonists()
        for c in colonists:
            assert len(c['relationships']) == 9  # 10 colonists - self

    def test_relationship_range(self):
        colonists = create_colonists()
        for c in colonists:
            for other_id, affinity in c['relationships'].items():
                assert -100 <= affinity <= 100

    def test_no_self_relationship(self):
        colonists = create_colonists()
        for c in colonists:
            assert c['id'] not in c['relationships']

    def test_behavior_ast_parsed(self):
        colonists = create_colonists()
        for c in colonists:
            assert c['behavior_ast'] is not None
            assert isinstance(c['behavior_ast'], list)

    def test_empty_memory(self):
        colonists = create_colonists()
        for c in colonists:
            assert c['memory'] == []

    def test_deterministic(self):
        c1 = create_colonists(seed=42)
        c2 = create_colonists(seed=42)
        for a, b in zip(c1, c2):
            assert a['stats'] == b['stats']
            assert a['relationships'] == b['relationships']

    def test_different_seeds_differ(self):
        c1 = create_colonists(seed=42)
        c2 = create_colonists(seed=99)
        # At least some relationships should differ
        diffs = sum(1 for a, b in zip(c1, c2)
                    if a['relationships'] != b['relationships'])
        assert diffs > 0


class TestColonistToLispy:
    def test_returns_assoc_list(self):
        colonists = create_colonists()
        result = colonist_to_lispy(colonists[0])
        assert isinstance(result, list)
        assert all(isinstance(item, list) for item in result)

    def test_has_id(self):
        colonists = create_colonists()
        result = colonist_to_lispy(colonists[0])
        ids = [item[1] for item in result if item[0] == Symbol('id')]
        assert ids == ['ares']

    def test_has_stats(self):
        colonists = create_colonists()
        result = colonist_to_lispy(colonists[0])
        keys = [str(item[0]) for item in result]
        for stat in STAT_NAMES:
            assert stat in keys

    def test_has_skills(self):
        colonists = create_colonists()
        result = colonist_to_lispy(colonists[0])
        keys = [str(item[0]) for item in result]
        for skill in SKILL_NAMES:
            assert skill in keys


class TestEvolveStats:
    def test_stats_stay_bounded(self):
        import random
        rng = random.Random(42)
        colonists = create_colonists()
        for _ in range(100):
            for c in colonists:
                evolve_stats(c, 50, 0.5, rng)
                for stat in STAT_NAMES:
                    assert 0 <= c['stats'][stat] <= 100

    def test_dead_colonists_unchanged(self):
        import random
        rng = random.Random(42)
        colonists = create_colonists()
        colonists[0]['alive'] = False
        original = dict(colonists[0]['stats'])
        evolve_stats(colonists[0], 50, 0.5, rng)
        assert colonists[0]['stats'] == original

    def test_paranoia_increases_with_stress(self):
        import random
        rng = random.Random(42)
        colonists = create_colonists()
        col = colonists[0]
        initial_paranoia = col['stats']['paranoia']
        # High stress over many iterations should increase paranoia
        for _ in range(50):
            evolve_stats(col, 80, 1.0, rng)
        assert col['stats']['paranoia'] >= initial_paranoia


class TestEvolveRelationships:
    def test_relationships_stay_bounded(self):
        import random
        rng = random.Random(42)
        colonists = create_colonists()
        for _ in range(100):
            evolve_relationships(colonists, 50, rng)
            for c in colonists:
                for _, affinity in c['relationships'].items():
                    assert -100 <= affinity <= 100


class TestClampStat:
    def test_normal(self):
        assert clamp_stat(50) == 50

    def test_below_zero(self):
        assert clamp_stat(-10) == 0

    def test_above_100(self):
        assert clamp_stat(150) == 100

    def test_float(self):
        assert clamp_stat(50.7) == 51


class TestSerialize:
    def test_no_ast(self):
        colonists = create_colonists()
        serial = serialize_colonist(colonists[0])
        assert 'behavior_ast' not in serial

    def test_has_behavior_source(self):
        colonists = create_colonists()
        serial = serialize_colonist(colonists[0])
        assert 'behavior_source' in serial

    def test_json_safe(self):
        """Serialized colonist should be JSON-serializable."""
        import json
        colonists = create_colonists()
        for c in colonists:
            serial = serialize_colonist(c)
            json.dumps(serial)  # Should not raise
