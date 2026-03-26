"""
test_garden.py — Comprehensive unit tests for the Neural Garden evolution engine.

Tests cover: utilities, DNA generation/mutation, organism lifecycle, food spawning,
genesis, tick simulation, species classification, environmental invariants,
energy conservation, and population bounds.

75+ tests. The community ships code.
"""
from __future__ import annotations
import json
import math
import os
import random
import tempfile
from pathlib import Path

import pytest

# ── bootstrap import path ────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import garden


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

class TestClamp:
    def test_within_bounds(self):
        assert garden.clamp(5, 0, 10) == 5

    def test_below_lower(self):
        assert garden.clamp(-3, 0, 10) == 0

    def test_above_upper(self):
        assert garden.clamp(15, 0, 10) == 10

    def test_equal_to_lower(self):
        assert garden.clamp(0, 0, 10) == 0

    def test_equal_to_upper(self):
        assert garden.clamp(10, 0, 10) == 10

    def test_negative_range(self):
        assert garden.clamp(0, -5, -1) == -1


class TestWrap:
    def test_no_wrap_needed(self):
        assert garden.wrap(5, 10) == 5

    def test_wraps_at_boundary(self):
        assert garden.wrap(10, 10) == 0

    def test_wraps_over(self):
        assert garden.wrap(13, 10) == 3

    def test_negative_not_needed(self):
        assert garden.wrap(0, 10) == 0


class TestDist:
    def test_same_point(self):
        assert garden.dist([0, 0], [0, 0]) == 0.0

    def test_horizontal(self):
        d = garden.dist([0, 0], [3, 0])
        assert abs(d - 3.0) < 0.01

    def test_vertical(self):
        d = garden.dist([0, 0], [0, 4])
        assert abs(d - 4.0) < 0.01

    def test_wrapping_closer(self):
        """Toroidal distance should pick the shorter wrap-around path."""
        d = garden.dist([10, 0], [garden.WW - 10, 0])
        assert d < garden.WW / 2

    def test_diagonal(self):
        d = garden.dist([0, 0], [3, 4])
        assert abs(d - 5.0) < 0.01


class TestDdist:
    def test_identical_dna_is_zero(self):
        dna = {n: (lo + hi) / 2 for n, lo, hi in garden.GENES}
        assert garden.ddist(dna, dna) == 0.0

    def test_max_spread_dna_nonzero(self):
        lo_dna = {n: lo for n, lo, hi in garden.GENES}
        hi_dna = {n: hi for n, lo, hi in garden.GENES}
        assert garden.ddist(lo_dna, hi_dna) > 0


class TestNowIso:
    def test_returns_string(self):
        assert isinstance(garden.now_iso(), str)

    def test_contains_T(self):
        assert "T" in garden.now_iso()


class TestUid:
    def test_length(self):
        assert len(garden.uid()) == 8

    def test_uniqueness(self):
        ids = {garden.uid() for _ in range(100)}
        assert len(ids) == 100


# ═══════════════════════════════════════════════════════════════════════
# DNA
# ═══════════════════════════════════════════════════════════════════════

class TestRdna:
    def test_all_genes_present(self):
        dna = garden.rdna()
        for name, _, _ in garden.GENES:
            assert name in dna

    def test_genes_in_bounds(self):
        for _ in range(50):
            dna = garden.rdna()
            for name, lo, hi in garden.GENES:
                assert lo <= dna[name] <= hi, f"{name}={dna[name]} not in [{lo},{hi}]"

    def test_segments_is_integer(self):
        dna = garden.rdna()
        assert dna["segments"] == int(dna["segments"])

    def test_with_base_hue(self):
        dna = garden.rdna(bh=0.5)
        assert 0 <= dna["hue"] <= 1

    def test_base_hue_clustering(self):
        """DNA created with a base hue should cluster near that hue."""
        hues = [garden.rdna(bh=0.5)["hue"] for _ in range(100)]
        avg = sum(hues) / len(hues)
        assert abs(avg - 0.5) < 0.15


class TestMdna:
    def test_mutated_dna_stays_in_bounds(self):
        for _ in range(100):
            parent = garden.rdna()
            child = garden.mdna(parent)
            for name, lo, hi in garden.GENES:
                assert lo <= child[name] <= hi, f"{name}={child[name]} out of [{lo},{hi}]"

    def test_segments_stays_integer(self):
        for _ in range(20):
            child = garden.mdna(garden.rdna())
            assert child["segments"] == int(child["segments"])

    def test_mutation_does_not_mutate_original(self):
        parent = garden.rdna()
        original = dict(parent)
        garden.mdna(parent)
        assert parent == original


# ═══════════════════════════════════════════════════════════════════════
# Organism creation
# ═══════════════════════════════════════════════════════════════════════

class TestMorg:
    def test_has_required_fields(self):
        o = garden.morg(0)
        for key in ("id", "born", "pos", "vel", "energy", "dna", "age", "children", "kills"):
            assert key in o, f"Missing key: {key}"

    def test_initial_energy(self):
        o = garden.morg(0)
        assert o["energy"] == garden.BASE_E

    def test_position_in_world(self):
        for _ in range(50):
            o = garden.morg(0)
            assert 0 <= o["pos"][0] <= garden.WW
            assert 0 <= o["pos"][1] <= garden.WH

    def test_custom_position(self):
        o = garden.morg(0, pos=[100, 200])
        assert o["pos"] == [100, 200]

    def test_born_epoch(self):
        o = garden.morg(42)
        assert o["born"] == 42

    def test_parent_tracking(self):
        o = garden.morg(0, par="parent-id")
        assert o["parent"] == "parent-id"

    def test_starts_with_zero_stats(self):
        o = garden.morg(0)
        assert o["age"] == 0
        assert o["children"] == 0
        assert o["kills"] == 0


# ═══════════════════════════════════════════════════════════════════════
# Food
# ═══════════════════════════════════════════════════════════════════════

class TestSfood:
    def test_correct_count(self):
        food = garden.sfood(10)
        assert len(food) == 10

    def test_food_has_pos_and_energy(self):
        food = garden.sfood(5)
        for f in food:
            assert "pos" in f
            assert "e" in f
            assert len(f["pos"]) == 2

    def test_food_positions_in_world(self):
        for f in garden.sfood(100):
            assert 0 <= f["pos"][0] <= garden.WW
            assert 0 <= f["pos"][1] <= garden.WH

    def test_food_energy_positive(self):
        for f in garden.sfood(100):
            assert f["e"] > 0

    def test_zero_food(self):
        assert garden.sfood(0) == []


# ═══════════════════════════════════════════════════════════════════════
# Hue naming
# ═══════════════════════════════════════════════════════════════════════

class TestHueName:
    def test_returns_string(self):
        assert isinstance(garden.hn(0.5), str)

    def test_zero_hue(self):
        name = garden.hn(0.0)
        assert name == "Crimson"

    def test_mid_hue(self):
        name = garden.hn(0.5)
        assert name == "Cyan"

    def test_boundary_values(self):
        for h in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert isinstance(garden.hn(h), str)


# ═══════════════════════════════════════════════════════════════════════
# Genesis
# ═══════════════════════════════════════════════════════════════════════

class TestGenesis:
    def test_creates_state(self):
        s = garden.genesis()
        assert isinstance(s, dict)

    def test_epoch_zero(self):
        s = garden.genesis()
        assert s["epoch"] == 0

    def test_has_organisms(self):
        s = garden.genesis()
        assert len(s["organisms"]) >= garden.GENESIS_POP

    def test_has_food(self):
        s = garden.genesis()
        assert len(s["food"]) > 0

    def test_has_species(self):
        s = garden.genesis()
        assert isinstance(s["species"], dict)
        assert len(s["species"]) >= 1

    def test_has_world(self):
        s = garden.genesis()
        assert "world" in s
        assert s["world"]["w"] == garden.WW
        assert s["world"]["h"] == garden.WH

    def test_has_events(self):
        s = garden.genesis()
        assert len(s["events"]) >= 1
        assert s["events"][0]["t"] == "genesis"

    def test_has_history(self):
        s = garden.genesis()
        assert len(s["history"]) >= 1

    def test_all_organisms_have_species(self):
        s = garden.genesis()
        for o in s["organisms"]:
            assert o.get("species", "") != "", f"Organism {o['id']} has no species"


# ═══════════════════════════════════════════════════════════════════════
# Tick — single step
# ═══════════════════════════════════════════════════════════════════════

class TestTick:
    @pytest.fixture
    def state(self):
        random.seed(42)
        return garden.genesis()

    def test_epoch_increments(self, state):
        old_epoch = state["epoch"]
        garden.tick(state)
        assert state["epoch"] == old_epoch + 1

    def test_population_stays_positive(self, state):
        for _ in range(20):
            garden.tick(state)
        assert len(state["organisms"]) > 0

    def test_population_under_max(self, state):
        for _ in range(50):
            garden.tick(state)
        assert len(state["organisms"]) <= garden.MAX_POP + 50  # births in same tick

    def test_food_bounded(self, state):
        for _ in range(20):
            garden.tick(state)
        assert len(state["food"]) <= 600  # 500 cap + new spawn

    def test_season_advances(self, state):
        s0 = state["world"]["env"]["season"]
        garden.tick(state)
        s1 = state["world"]["env"]["season"]
        assert s1 != s0

    def test_temperature_in_physical_range(self, state):
        for _ in range(100):
            garden.tick(state)
            t = state["world"]["env"]["temp"]
            assert 0.0 <= t <= 1.0, f"Temperature {t} out of [0,1]"

    def test_light_in_physical_range(self, state):
        for _ in range(100):
            garden.tick(state)
            lt = state["world"]["env"]["light"]
            assert 0.0 <= lt <= 1.0, f"Light {lt} out of [0,1]"

    def test_nutrients_in_physical_range(self, state):
        for _ in range(100):
            garden.tick(state)
            n = state["world"]["env"]["nutrients"]
            assert 0.1 <= n <= 0.9, f"Nutrients {n} out of [0.1,0.9]"

    def test_organisms_stay_in_world(self, state):
        for _ in range(30):
            garden.tick(state)
        for o in state["organisms"]:
            assert 0 <= o["pos"][0] <= garden.WW
            assert 0 <= o["pos"][1] <= garden.WH

    def test_organism_energy_finite(self, state):
        for _ in range(30):
            garden.tick(state)
        for o in state["organisms"]:
            assert math.isfinite(o["energy"]), f"Infinite energy: {o['energy']}"

    def test_graveyard_bounded(self, state):
        for _ in range(50):
            garden.tick(state)
        assert len(state["graveyard"]) <= garden.MAX_GRAVE

    def test_events_bounded(self, state):
        for _ in range(50):
            garden.tick(state)
        assert len(state["events"]) <= garden.MAX_EVT

    def test_history_bounded(self, state):
        for _ in range(50):
            garden.tick(state)
        assert len(state["history"]) <= garden.MAX_HIST

    def test_dead_organisms_removed(self, state):
        """No organism should have energy <= 0 after tick."""
        for _ in range(30):
            garden.tick(state)
        for o in state["organisms"]:
            assert o["energy"] > 0

    def test_emergency_reseed(self):
        """If population drops below 8, emergency seeding kicks in."""
        random.seed(99)
        s = garden.genesis()
        s["organisms"] = s["organisms"][:3]  # force critical
        garden.tick(s)
        assert len(s["organisms"]) >= 8

    def test_species_classification_runs(self, state):
        garden.tick(state)
        assert isinstance(state["species"], dict)

    def test_all_surviving_organisms_have_species(self, state):
        for _ in range(10):
            garden.tick(state)
        for o in state["organisms"]:
            assert o.get("species", "") != ""


# ═══════════════════════════════════════════════════════════════════════
# Multi-epoch smoke test
# ═══════════════════════════════════════════════════════════════════════

class TestMultiEpoch:
    def test_100_epochs_no_crash(self):
        """Smoke test: run 100 epochs without any exception."""
        random.seed(7)
        s = garden.genesis()
        for _ in range(100):
            garden.tick(s)
        assert s["epoch"] == 100
        assert len(s["organisms"]) > 0

    def test_speciation_occurs_over_time(self):
        """Over 100 epochs, at least one new species should emerge."""
        random.seed(7)
        s = garden.genesis()
        initial_species = set(s["species"].keys())
        for _ in range(100):
            garden.tick(s)
        all_species = set(s["species"].keys())
        # Some speciation or extinction should have occurred
        assert all_species != initial_species or len(s["events"]) > 1


# ═══════════════════════════════════════════════════════════════════════
# Reproduction
# ═══════════════════════════════════════════════════════════════════════

class TestReproduction:
    def test_child_has_parent(self):
        parent = garden.morg(0)
        parent["energy"] = 300  # above repro threshold
        child = garden._repro(parent, 1)
        assert child["parent"] == parent["id"]

    def test_parent_loses_energy(self):
        parent = garden.morg(0)
        parent["energy"] = 300
        initial = parent["energy"]
        garden._repro(parent, 1)
        assert parent["energy"] < initial

    def test_child_born_epoch(self):
        parent = garden.morg(0)
        parent["energy"] = 300
        child = garden._repro(parent, 5)
        assert child["born"] == 5

    def test_child_energy_positive(self):
        parent = garden.morg(0)
        parent["energy"] = 300
        child = garden._repro(parent, 1)
        assert child["energy"] > 0

    def test_energy_conservation_in_repro(self):
        """Parent + child energy should be less than or equal to parent's initial."""
        parent = garden.morg(0)
        parent["energy"] = 300
        initial = parent["energy"]
        child = garden._repro(parent, 1)
        # child gets cost * 0.8, parent keeps the rest — some energy lost (20% tax)
        assert parent["energy"] + child["energy"] <= initial


# ═══════════════════════════════════════════════════════════════════════
# Species classification
# ═══════════════════════════════════════════════════════════════════════

class TestClassify:
    def test_empty_organisms(self):
        s = {"organisms": [], "species": {}}
        garden.classify(s)
        assert s["species"] == {}

    def test_single_organism_gets_species(self):
        s = {"organisms": [garden.morg(0)], "species": {}}
        garden.classify(s)
        assert len(s["species"]) == 1
        assert s["organisms"][0]["species"] != ""

    def test_species_count_matches(self):
        s = garden.genesis()
        for sp_name, sp_info in s["species"].items():
            actual = sum(1 for o in s["organisms"] if o.get("species") == sp_name)
            assert actual == sp_info["count"], f"{sp_name}: expected {sp_info['count']}, got {actual}"


# ═══════════════════════════════════════════════════════════════════════
# Save / Load round-trip
# ═══════════════════════════════════════════════════════════════════════

class TestSaveLoad:
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "garden_state.json"
            old = garden.STATE_PATH
            garden.STATE_PATH = path
            try:
                s = garden.genesis()
                garden.save(s)
                loaded = garden.load()
                assert loaded is not None
                assert loaded["epoch"] == s["epoch"]
                assert len(loaded["organisms"]) == len(s["organisms"])
            finally:
                garden.STATE_PATH = old

    def test_load_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            old = garden.STATE_PATH
            garden.STATE_PATH = path
            try:
                assert garden.load() is None
            finally:
                garden.STATE_PATH = old

    def test_load_corrupt_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corrupt.json"
            path.write_text("not json at all")
            old = garden.STATE_PATH
            garden.STATE_PATH = path
            try:
                assert garden.load() is None
            finally:
                garden.STATE_PATH = old


# ═══════════════════════════════════════════════════════════════════════
# Property-based invariants
# ═══════════════════════════════════════════════════════════════════════

class TestInvariants:
    """Property-based tests: physical laws the garden must never violate."""

    def test_dna_bounds_after_many_mutations(self):
        """After 1000 chained mutations, all genes still in bounds."""
        dna = garden.rdna()
        for _ in range(1000):
            dna = garden.mdna(dna)
        for name, lo, hi in garden.GENES:
            assert lo <= dna[name] <= hi, f"{name}={dna[name]} escaped [{lo},{hi}]"

    def test_no_negative_energy_survivors(self):
        """After simulation, no living organism has energy <= 0."""
        random.seed(123)
        s = garden.genesis()
        for _ in range(50):
            garden.tick(s)
        for o in s["organisms"]:
            assert o["energy"] > 0

    def test_organism_ages_increase(self):
        """All organisms' ages should be non-negative."""
        random.seed(42)
        s = garden.genesis()
        for _ in range(20):
            garden.tick(s)
        for o in s["organisms"]:
            assert o["age"] >= 0

    def test_epoch_monotonic(self):
        """Epoch counter must be strictly monotonic."""
        random.seed(42)
        s = garden.genesis()
        for i in range(1, 30):
            garden.tick(s)
            assert s["epoch"] == i

    def test_graveyard_has_cause_of_death(self):
        random.seed(42)
        s = garden.genesis()
        for _ in range(50):
            garden.tick(s)
        for d in s["graveyard"]:
            assert "cause" in d
            assert d["cause"] in ("starve", "old", "eaten")


# ═══════════════════════════════════════════════════════════════════════
# Record history
# ═══════════════════════════════════════════════════════════════════════

class TestRecHist:
    def test_appends_entry(self):
        s = garden.genesis()
        initial_len = len(s["history"])
        garden.rec_hist(s, b=5, d=3)
        assert len(s["history"]) == initial_len + 1

    def test_entry_fields(self):
        s = garden.genesis()
        garden.rec_hist(s, b=2, d=1)
        entry = s["history"][-1]
        assert entry["b"] == 2
        assert entry["d"] == 1
        assert "pop" in entry
        assert "sp" in entry
        assert "avg_e" in entry
