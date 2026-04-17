"""Tests for the Mars-100 colonist model (src/mars100_colonist.py).

Covers: colonist creation, stat bounds, skill bounds, relationship init,
memory management, death/archival, stat evolution.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.mars100_colonist import (
    ELEMENTS,
    FOUNDERS,
    MAX_MEMORY,
    SKILL_NAMES,
    STAT_NAMES,
    add_memory,
    create_all_colonists,
    create_colonist,
    evolve_stats,
    init_relationships,
    kill_colonist,
    update_relationship,
)


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


class TestCreation:
    def test_create_all_returns_10(self) -> None:
        colonists = create_all_colonists(seed=42)
        assert len(colonists) == 10

    def test_unique_ids(self) -> None:
        colonists = create_all_colonists(seed=42)
        ids = [c["id"] for c in colonists]
        assert len(set(ids)) == 10

    def test_all_alive(self) -> None:
        colonists = create_all_colonists(seed=42)
        assert all(c["alive"] for c in colonists)

    def test_valid_elements(self) -> None:
        colonists = create_all_colonists(seed=42)
        for c in colonists:
            assert c["element"] in ELEMENTS

    def test_has_all_stats(self) -> None:
        colonists = create_all_colonists(seed=42)
        for c in colonists:
            for stat in STAT_NAMES:
                assert stat in c["stats"]

    def test_has_all_skills(self) -> None:
        colonists = create_all_colonists(seed=42)
        for c in colonists:
            for skill in SKILL_NAMES:
                assert skill in c["skills"]

    def test_deterministic(self) -> None:
        c1 = create_all_colonists(seed=42)
        c2 = create_all_colonists(seed=42)
        for a, b in zip(c1, c2):
            assert a["stats"] == b["stats"]
            assert a["skills"] == b["skills"]

    def test_different_seeds_differ(self) -> None:
        c1 = create_all_colonists(seed=42)
        c2 = create_all_colonists(seed=99)
        # At least some stats should differ
        diffs = sum(
            1 for a, b in zip(c1, c2)
            if a["stats"]["resolve"] != b["stats"]["resolve"]
        )
        assert diffs > 0


# ---------------------------------------------------------------------------
# Stat bounds (property-based via seeded sweep)
# ---------------------------------------------------------------------------


class TestStatBounds:
    @pytest.mark.parametrize("seed", range(10))
    def test_stats_in_zero_one(self, seed: int) -> None:
        colonists = create_all_colonists(seed=seed)
        for c in colonists:
            for stat_name, val in c["stats"].items():
                assert 0.0 <= val <= 1.0, f"{c['id']}.{stat_name} = {val}"

    @pytest.mark.parametrize("seed", range(10))
    def test_skills_in_zero_one(self, seed: int) -> None:
        colonists = create_all_colonists(seed=seed)
        for c in colonists:
            for skill_name, val in c["skills"].items():
                assert 0.0 <= val <= 1.0, f"{c['id']}.{skill_name} = {val}"


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------


class TestRelationships:
    def test_relationships_initialized(self) -> None:
        colonists = create_all_colonists(seed=42)
        for c in colonists:
            assert len(c["relationships"]) == 9  # 10 - self

    def test_no_self_relationship(self) -> None:
        colonists = create_all_colonists(seed=42)
        for c in colonists:
            assert c["id"] not in c["relationships"]

    def test_relationships_bounded(self) -> None:
        colonists = create_all_colonists(seed=42)
        for c in colonists:
            for other_id, val in c["relationships"].items():
                assert -1.0 <= val <= 1.0

    def test_update_relationship(self) -> None:
        colonists = create_all_colonists(seed=42)
        c = colonists[0]
        other_id = colonists[1]["id"]
        old = c["relationships"][other_id]
        update_relationship(c, other_id, 0.5)
        new = c["relationships"][other_id]
        assert abs(new - (old + 0.5)) < 0.01 or new == 1.0

    def test_relationship_clamped(self) -> None:
        colonists = create_all_colonists(seed=42)
        c = colonists[0]
        other_id = colonists[1]["id"]
        update_relationship(c, other_id, 10.0)
        assert c["relationships"][other_id] == 1.0
        update_relationship(c, other_id, -20.0)
        assert c["relationships"][other_id] == -1.0


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class TestMemory:
    def test_add_memory(self) -> None:
        colonists = create_all_colonists(seed=42)
        c = colonists[0]
        add_memory(c, 1, "Something happened")
        assert len(c["memory"]) == 1
        assert c["memory"][0]["year"] == 1

    def test_memory_bounded(self) -> None:
        colonists = create_all_colonists(seed=42)
        c = colonists[0]
        for i in range(MAX_MEMORY + 20):
            add_memory(c, i, f"Event {i}")
        assert len(c["memory"]) == MAX_MEMORY
        # Oldest should be pruned
        assert c["memory"][0]["year"] == 20


# ---------------------------------------------------------------------------
# Death / archival
# ---------------------------------------------------------------------------


class TestDeath:
    def test_kill_colonist(self) -> None:
        colonists = create_all_colonists(seed=42)
        c = colonists[0]
        soul = kill_colonist(c, year=50, cause="meteor")
        assert c["alive"] is False
        assert c["death_year"] == 50
        assert c["death_cause"] == "meteor"

    def test_soul_file_has_required_fields(self) -> None:
        colonists = create_all_colonists(seed=42)
        c = colonists[0]
        soul = kill_colonist(c, year=50, cause="starvation")
        for field in ("id", "name", "element", "archetype", "death_year",
                      "death_cause", "final_stats", "final_skills", "memory",
                      "epitaph"):
            assert field in soul

    def test_soul_has_epitaph(self) -> None:
        colonists = create_all_colonists(seed=42)
        soul = kill_colonist(colonists[0], year=50, cause="epidemic")
        assert len(soul["epitaph"]) > 10


# ---------------------------------------------------------------------------
# Stat evolution
# ---------------------------------------------------------------------------


class TestStatEvolution:
    def test_evolve_stats_changes_values(self) -> None:
        colonists = create_all_colonists(seed=42)
        c = colonists[0]
        old_stats = dict(c["stats"])
        rng = random.Random(42)
        evolve_stats(c, year=1, event_type="dust_storm", rng=rng)
        # At least one stat should have changed
        changed = sum(1 for k in STAT_NAMES if c["stats"][k] != old_stats[k])
        assert changed > 0

    def test_evolve_stats_bounded(self) -> None:
        colonists = create_all_colonists(seed=42)
        c = colonists[0]
        rng = random.Random(42)
        for year in range(1, 101):
            evolve_stats(c, year, "dust_storm", rng)
        for stat_name, val in c["stats"].items():
            assert 0.0 <= val <= 1.0, f"{stat_name} = {val}"

    @pytest.mark.parametrize("event", [
        "calm", "dust_storm", "resource_strike", "equipment_failure",
        "earth_contact", "alien_signal", "solar_flare", "meteor", "epidemic",
    ])
    def test_evolve_all_events(self, event: str) -> None:
        colonists = create_all_colonists(seed=42)
        c = colonists[0]
        rng = random.Random(42)
        evolve_stats(c, 1, event, rng)
        for val in c["stats"].values():
            assert 0.0 <= val <= 1.0
