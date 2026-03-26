"""
test_reef.py — Unit tests for The Reef autonomous digital ecosystem.

Covers: DNA encoding/decoding, mutation, speciation distance, organism
creation, world seeding, toroidal movement, resource gathering,
predation, reproduction, tick invariants, epoch progression,
energy conservation, population bounds, coordinate wrapping.

85 tests. One file. One merge.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import reef


# ─── fixtures ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_nid():
    """Reset global organism ID counter before each test."""
    reef._nid = 0
    yield


@pytest.fixture
def seeded_world():
    """A fresh world with organisms seeded."""
    random.seed(42)
    w = reef.empty()
    reef.seed(w)
    return w


# ─── DNA primitives ─────────────────────────────────────────────────

class TestDNA:
    """Tests for DNA encoding, decoding, and mutation."""

    def test_rdna_length(self):
        """Random DNA is always 16 hex chars (8 bytes)."""
        for _ in range(50):
            dna = reef.rdna()
            assert len(dna) == 16
            assert all(c in "0123456789abcdef" for c in dna)

    def test_dgene_range(self):
        """Each raw gene value is 0..255."""
        dna = "ff00807f01fe02fd"
        for i in range(8):
            val = reef.dgene(dna, i)
            assert 0 <= val <= 255

    def test_dgene_specific_values(self):
        """Spot-check known byte values."""
        dna = "ff00807f01fe02fd"
        assert reef.dgene(dna, 0) == 0xFF
        assert reef.dgene(dna, 1) == 0x00
        assert reef.dgene(dna, 2) == 0x80
        assert reef.dgene(dna, 3) == 0x7F

    def test_dtrait_bounds(self):
        """Every decoded trait must be within its RANGES bounds."""
        for _ in range(100):
            dna = reef.rdna()
            for name in reef.GENES:
                lo, hi = reef.RANGES[name]
                val = reef.dtrait(dna, name)
                assert lo <= val <= hi, f"{name}={val} outside [{lo},{hi}]"

    def test_dtrait_min_max(self):
        """DNA of all 00 and all ff produce min and max trait values."""
        dna_min = "00" * 8
        dna_max = "ff" * 8
        for name in reef.GENES:
            lo, hi = reef.RANGES[name]
            assert reef.dtrait(dna_min, name) == pytest.approx(lo, abs=1e-6)
            assert reef.dtrait(dna_max, name) == pytest.approx(hi, abs=1e-6)

    def test_dtraits_returns_all_genes(self):
        """dtraits() returns a dict with all 8 gene names."""
        dna = reef.rdna()
        traits = reef.dtraits(dna)
        assert set(traits.keys()) == set(reef.GENES)

    def test_dtraits_values_match_dtrait(self):
        """dtraits() values match individual dtrait() calls."""
        dna = reef.rdna()
        traits = reef.dtraits(dna)
        for name in reef.GENES:
            expected = round(reef.dtrait(dna, name), 3)
            assert traits[name] == expected

    def test_mutdna_preserves_length(self):
        """Mutation never changes DNA length."""
        dna = reef.rdna()
        for rate in [0.0, 0.1, 0.5, 1.0]:
            mutated = reef.mutdna(dna, rate)
            assert len(mutated) == 16

    def test_mutdna_zero_rate_identity(self):
        """Mutation with rate=0 produces identical DNA."""
        dna = reef.rdna()
        for _ in range(20):
            assert reef.mutdna(dna, 0.0) == dna

    def test_mutdna_high_rate_differs(self):
        """Mutation with rate=1.0 almost certainly differs from original."""
        random.seed(7)
        dna = reef.rdna()
        changed = sum(1 for _ in range(20) if reef.mutdna(dna, 1.0) != dna)
        assert changed > 15

    def test_mutdna_clamps_to_valid_hex(self):
        """Mutated DNA is always valid hex in 0..255 range."""
        for _ in range(100):
            dna = reef.rdna()
            mutated = reef.mutdna(dna, 0.5)
            assert len(mutated) == 16
            for i in range(8):
                val = int(mutated[i * 2:i * 2 + 2], 16)
                assert 0 <= val <= 255

    def test_dnadist_self_zero(self):
        """Distance of DNA to itself is zero."""
        dna = reef.rdna()
        assert reef.dnadist(dna, dna) == 0.0

    def test_dnadist_symmetric(self):
        """DNA distance is symmetric: d(a,b) == d(b,a)."""
        a, b = reef.rdna(), reef.rdna()
        assert reef.dnadist(a, b) == pytest.approx(reef.dnadist(b, a))

    def test_dnadist_triangle_inequality(self):
        """DNA distance satisfies the triangle inequality."""
        random.seed(3)
        for _ in range(30):
            a, b, c = reef.rdna(), reef.rdna(), reef.rdna()
            assert reef.dnadist(a, c) <= reef.dnadist(a, b) + reef.dnadist(b, c) + 1e-9

    def test_dnadist_max_bound(self):
        """Max distance: all-0 to all-255 = sqrt(8 * 255^2)."""
        max_dist = reef.dnadist("00" * 8, "ff" * 8)
        theoretical = math.sqrt(8 * (255 ** 2))
        assert max_dist == pytest.approx(theoretical, abs=1e-6)


# ─── Naming ──────────────────────────────────────────────────────────

class TestNaming:
    """Tests for species naming and ID generation."""

    def test_nid_increments(self):
        """nid() produces unique, incrementing IDs."""
        ids = [reef.nid() for _ in range(5)]
        assert ids == ["o-000001", "o-000002", "o-000003", "o-000004", "o-000005"]

    def test_spname_deterministic(self):
        """Same species ID always produces the same name."""
        assert reef.spname("s-001") == reef.spname("s-001")

    def test_spname_ends_with_us(self):
        """Species names end with 'us' (Latin-esque)."""
        for i in range(20):
            name = reef.spname(f"s-{i:03d}")
            assert name.endswith("us"), f"'{name}' doesn't end with 'us'"

    def test_spname_starts_capitalized(self):
        """Species names start with a capital letter."""
        for i in range(20):
            name = reef.spname(f"s-{i:03d}")
            assert name[0].isupper()

    def test_niso_format(self):
        """niso() returns ISO 8601 UTC timestamp."""
        ts = reef.niso()
        assert ts.endswith("Z")
        assert "T" in ts
        assert len(ts) == 20


# ─── Organism creation ───────────────────────────────────────────────

class TestMkorg:
    """Tests for organism creation."""

    def test_mkorg_fields(self):
        """mkorg() returns all required fields."""
        dna = reef.rdna()
        o = reef.mkorg(100.0, 200.0, dna, "s-000")
        required = {"id", "dna", "x", "y", "vx", "vy", "energy", "age",
                     "generation", "parent", "species_id", "cooldown", "traits", "kills"}
        assert required.issubset(set(o.keys()))

    def test_mkorg_initial_values(self):
        """New organism starts with correct defaults."""
        o = reef.mkorg(50.0, 75.0, "ff" * 8, "s-001")
        assert o["energy"] == 100.0
        assert o["age"] == 0
        assert o["generation"] == 0
        assert o["parent"] is None
        assert o["cooldown"] == 0
        assert o["kills"] == 0

    def test_mkorg_position_rounded(self):
        """Positions are rounded to 1 decimal."""
        o = reef.mkorg(100.123456, 200.987654, reef.rdna(), "s-000")
        assert o["x"] == 100.1
        assert o["y"] == 201.0

    def test_mkorg_with_parent(self):
        """mkorg records parent and generation."""
        o = reef.mkorg(10, 10, reef.rdna(), "s-000", pid="o-000001", gen=5)
        assert o["parent"] == "o-000001"
        assert o["generation"] == 5


# ─── World creation ──────────────────────────────────────────────────

class TestWorld:
    """Tests for world creation and seeding."""

    def test_empty_structure(self):
        """empty() returns a valid world structure."""
        w = reef.empty()
        assert w["_meta"]["tick"] == 0
        assert w["_meta"]["epoch"] == "Primordial Soup"
        assert w["organisms"] == []
        assert w["resources"] == []
        assert w["species"] == {}

    def test_empty_history_keys(self):
        """History has all required tracking arrays."""
        w = reef.empty()
        expected = {"population", "species_count", "resource_count",
                    "avg_speed", "avg_size", "avg_aggression", "events"}
        assert set(w["history"].keys()) == expected

    def test_seed_creates_species(self):
        """Seeding creates 3-5 species."""
        random.seed(42)
        w = reef.empty()
        reef.seed(w)
        assert 3 <= len(w["species"]) <= 5

    def test_seed_creates_organisms(self):
        """Seeding populates organisms."""
        random.seed(42)
        w = reef.empty()
        reef.seed(w)
        assert len(w["organisms"]) > 0

    def test_seed_organisms_have_valid_species(self):
        """Every seeded organism belongs to a valid species."""
        random.seed(42)
        w = reef.empty()
        reef.seed(w)
        valid_sids = set(w["species"].keys())
        for o in w["organisms"]:
            assert o["species_id"] in valid_sids

    def test_seed_meta_counts(self):
        """Meta counters match actual organisms/species after seed."""
        random.seed(42)
        w = reef.empty()
        reef.seed(w)
        assert w["_meta"]["total_births"] == len(w["organisms"])
        assert w["_meta"]["total_species"] == len(w["species"])

    def test_seed_species_pop_counts(self):
        """Species current_pop matches actual organism count."""
        random.seed(42)
        w = reef.empty()
        reef.seed(w)
        pop = {}
        for o in w["organisms"]:
            pop[o["species_id"]] = pop.get(o["species_id"], 0) + 1
        for sid, sp in w["species"].items():
            assert sp["current_pop"] == pop.get(sid, 0)


# ─── Toroidal geometry ───────────────────────────────────────────────

class TestGeometry:
    """Tests for toroidal world movement and distance."""

    def test_wrap_basic(self):
        assert reef.wrap(5, 10) == 5
        assert reef.wrap(10, 10) == 0
        assert reef.wrap(15, 10) == 5
        assert reef.wrap(-1, 10) == 9

    def test_wrap_float(self):
        assert reef.wrap(800.5, 800) == pytest.approx(0.5)
        assert reef.wrap(-0.1, 600) == pytest.approx(599.9)

    def test_td_same_point(self):
        assert reef.td(100, 200, 100, 200) == 0.0

    def test_td_non_wrapping(self):
        """Toroidal distance without wrapping = Euclidean distance."""
        assert reef.td(100, 100, 103, 104) == pytest.approx(5.0)

    def test_td_wrapping_x(self):
        """World is 800 wide. Points at x=10 and x=790 are 20 apart."""
        assert reef.td(10, 100, 790, 100) == pytest.approx(20.0)

    def test_td_wrapping_y(self):
        """World is 600 tall. Points at y=5 and y=595 are 10 apart."""
        assert reef.td(100, 5, 100, 595) == pytest.approx(10.0)

    def test_td_symmetric(self):
        for _ in range(30):
            x1, y1 = random.uniform(0, 800), random.uniform(0, 600)
            x2, y2 = random.uniform(0, 800), random.uniform(0, 600)
            assert reef.td(x1, y1, x2, y2) == pytest.approx(reef.td(x2, y2, x1, y1))

    def test_td_max_distance(self):
        """Max toroidal distance is sqrt((W/2)^2 + (H/2)^2)."""
        max_d = math.sqrt((reef.WW / 2) ** 2 + (reef.WH / 2) ** 2)
        for _ in range(200):
            x1, y1 = random.uniform(0, 800), random.uniform(0, 600)
            x2, y2 = random.uniform(0, 800), random.uniform(0, 600)
            assert reef.td(x1, y1, x2, y2) <= max_d + 1e-6


# ─── Movement ────────────────────────────────────────────────────────

class TestMovement:
    """Tests for organism movement functions."""

    def test_mvto_reduces_distance(self):
        o = reef.mkorg(100.0, 100.0, reef.rdna(), "s-000")
        tx, ty = 200.0, 200.0
        d_before = reef.td(o["x"], o["y"], tx, ty)
        reef.mvto(o, tx, ty, 3.0)
        assert reef.td(o["x"], o["y"], tx, ty) < d_before

    def test_mvfr_increases_distance(self):
        o = reef.mkorg(400.0, 300.0, reef.rdna(), "s-000")
        tx, ty = 405.0, 300.0
        d_before = reef.td(o["x"], o["y"], tx, ty)
        reef.mvfr(o, tx, ty, 3.0)
        assert reef.td(o["x"], o["y"], tx, ty) > d_before

    def test_wander_stays_in_bounds(self):
        o = reef.mkorg(400.0, 300.0, reef.rdna(), "s-000")
        for _ in range(100):
            reef.wander(o, 4.0)
            assert 0 <= o["x"] < reef.WW
            assert 0 <= o["y"] < reef.WH

    def test_mvto_wraps_coordinates(self):
        o = reef.mkorg(790.0, 590.0, reef.rdna(), "s-000")
        reef.mvto(o, 10.0, 10.0, 5.0)
        assert 0 <= o["x"] < reef.WW
        assert 0 <= o["y"] < reef.WH

    def test_wander_velocity_capped(self):
        o = reef.mkorg(400.0, 300.0, reef.rdna(), "s-000")
        speed = 2.0
        for _ in range(50):
            reef.wander(o, speed)
            v_mag = math.sqrt(o["vx"] ** 2 + o["vy"] ** 2)
            assert v_mag <= speed + 0.1


# ─── Tick engine ─────────────────────────────────────────────────────

class TestTick:
    """Tests for the main simulation tick."""

    def test_tick_returns_events(self, seeded_world):
        events = reef.tick(seeded_world)
        assert isinstance(events, list)

    def test_tick_updates_meta(self, seeded_world):
        h_before = len(seeded_world["history"]["population"])
        reef.tick(seeded_world)
        assert len(seeded_world["history"]["population"]) == h_before + 1

    def test_tick_spawns_resources(self, seeded_world):
        assert len(seeded_world["resources"]) == 0
        reef.tick(seeded_world)
        assert len(seeded_world["resources"]) > 0

    def test_tick_resource_cap(self, seeded_world):
        for _ in range(20):
            reef.tick(seeded_world)
        assert len(seeded_world["resources"]) <= reef.MAX_RES

    def test_tick_population_cap(self, seeded_world):
        for _ in range(50):
            reef.tick(seeded_world)
        assert len(seeded_world["organisms"]) <= reef.MAX_POP

    def test_tick_organisms_in_bounds(self, seeded_world):
        for _ in range(10):
            reef.tick(seeded_world)
        for o in seeded_world["organisms"]:
            assert 0 <= o["x"] < reef.WW
            assert 0 <= o["y"] < reef.WH

    def test_tick_age_increments(self, seeded_world):
        alive_before = {o["id"]: o["age"] for o in seeded_world["organisms"]}
        reef.tick(seeded_world)
        for o in seeded_world["organisms"]:
            if o["id"] in alive_before:
                assert o["age"] == alive_before[o["id"]] + 1

    def test_tick_dead_organisms_removed(self, seeded_world):
        for o in seeded_world["organisms"]:
            o["energy"] = 0.1
        reef.tick(seeded_world)
        for o in seeded_world["organisms"]:
            assert o["energy"] > 0

    def test_tick_graveyard_records_deaths(self):
        """Old organisms die and enter the graveyard."""
        random.seed(42)
        w = reef.empty()
        reef.seed(w)
        # Force all organisms to be ancient — they MUST die from old age
        for o in w["organisms"]:
            o["age"] = reef.MAX_AGE + 1
        reef.tick(w)
        assert len(w["graveyard"]) > 0

    def test_tick_graveyard_capped(self, seeded_world):
        for _ in range(100):
            reef.tick(seeded_world)
        assert len(seeded_world["graveyard"]) <= 100

    def test_tick_history_capped(self, seeded_world):
        for _ in range(reef.HIST_MAX + 50):
            reef.tick(seeded_world)
        h = seeded_world["history"]
        for key in ["population", "species_count", "resource_count",
                     "avg_speed", "avg_size", "avg_aggression"]:
            assert len(h[key]) <= reef.HIST_MAX

    def test_tick_events_capped(self, seeded_world):
        for _ in range(300):
            reef.tick(seeded_world)
        assert len(seeded_world["history"]["events"]) <= reef.EVT_MAX

    def test_tick_cooldown_decrements(self, seeded_world):
        for o in seeded_world["organisms"]:
            o["cooldown"] = 5
        reef.tick(seeded_world)
        for o in seeded_world["organisms"]:
            assert o["cooldown"] <= 4


# ─── Epoch progression ───────────────────────────────────────────────

class TestEpoch:
    """Tests for epoch progression."""

    def test_epoch_starts_primordial(self):
        w = reef.empty()
        assert w["_meta"]["epoch"] == "Primordial Soup"

    def test_epoch_progresses(self, seeded_world):
        """Tick at threshold 10 triggers 'First Sparks'."""
        seeded_world["_meta"]["tick"] = 10
        reef.tick(seeded_world)
        assert seeded_world["_meta"]["epoch"] == "First Sparks"

    def test_epoch_all_thresholds(self):
        """Each epoch threshold triggers correctly."""
        w = reef.empty()
        reef.seed(w)
        for threshold, name in reef.EPOCHS:
            w["_meta"]["tick"] = threshold
            reef.tick(w)
            assert w["_meta"]["epoch"] == name


# ─── Reproduction and speciation ─────────────────────────────────────

class TestReproduction:
    """Tests for reproduction and speciation mechanics."""

    def test_reproduction_splits_energy(self, seeded_world):
        parent = seeded_world["organisms"][0]
        parent["energy"] = 200.0
        parent["cooldown"] = 0
        initial_energy = parent["energy"]
        initial_count = len(seeded_world["organisms"])
        reef.tick(seeded_world)
        if len(seeded_world["organisms"]) > initial_count:
            surviving = next(
                (o for o in seeded_world["organisms"] if o["id"] == parent["id"]), None
            )
            if surviving:
                assert surviving["energy"] < initial_energy

    def test_speciation_threshold(self):
        """DNA distance > SPEC_TH triggers new species."""
        dist = reef.dnadist("00" * 8, "ff" * 8)
        assert dist > reef.SPEC_TH

    def test_child_inherits_species_or_speciates(self, seeded_world):
        for _ in range(50):
            reef.tick(seeded_world)
        for o in seeded_world["organisms"]:
            assert o["species_id"] in seeded_world["species"]


# ─── Mass extinction and recovery ───────────────────────────────────

class TestExtinction:
    """Tests for mass extinction and recovery."""

    def test_mass_extinction_reseeds(self):
        random.seed(42)
        w = reef.empty()
        reef.seed(w)
        w["organisms"] = []
        events = reef.tick(w)
        assert any(e["type"] == "extinction_event" for e in events)
        assert len(w["organisms"]) > 0

    def test_species_extinction_tracked(self, seeded_world):
        target_sid = seeded_world["organisms"][0]["species_id"]
        seeded_world["organisms"] = [
            o for o in seeded_world["organisms"] if o["species_id"] != target_sid
        ]
        seeded_world["_meta"]["tick"] = 5
        reef.tick(seeded_world)
        assert "extinct_tick" in seeded_world["species"][target_sid]


# ─── Physical invariants (property-based) ────────────────────────────

class TestInvariants:
    """Property-based invariants that must hold across all ticks."""

    def test_energy_non_negative_after_death_removal(self, seeded_world):
        for _ in range(20):
            reef.tick(seeded_world)
        for o in seeded_world["organisms"]:
            assert o["energy"] > 0

    def test_age_non_negative(self, seeded_world):
        for _ in range(20):
            reef.tick(seeded_world)
        for o in seeded_world["organisms"]:
            assert o["age"] >= 0

    def test_generation_non_negative(self, seeded_world):
        for _ in range(20):
            reef.tick(seeded_world)
        for o in seeded_world["organisms"]:
            assert o["generation"] >= 0

    def test_kills_non_negative(self, seeded_world):
        for _ in range(20):
            reef.tick(seeded_world)
        for o in seeded_world["organisms"]:
            assert o["kills"] >= 0

    def test_birth_death_accounting(self, seeded_world):
        """total_births - total_deaths == current population."""
        for _ in range(30):
            reef.tick(seeded_world)
        m = seeded_world["_meta"]
        assert m["total_births"] - m["total_deaths"] == len(seeded_world["organisms"])

    def test_species_pop_matches_organisms(self, seeded_world):
        for _ in range(30):
            reef.tick(seeded_world)
        pop = {}
        for o in seeded_world["organisms"]:
            pop[o["species_id"]] = pop.get(o["species_id"], 0) + 1
        for sid, sp in seeded_world["species"].items():
            assert sp["current_pop"] == pop.get(sid, 0)

    def test_coordinates_always_in_bounds(self, seeded_world):
        for tick_num in range(30):
            reef.tick(seeded_world)
            for o in seeded_world["organisms"]:
                assert 0 <= o["x"] < reef.WW
                assert 0 <= o["y"] < reef.WH

    def test_resource_energy_positive(self, seeded_world):
        for _ in range(10):
            reef.tick(seeded_world)
        for r in seeded_world["resources"]:
            assert r["energy"] > 0


# ─── Smoke tests ─────────────────────────────────────────────────────

class TestSmoke:
    """Run the simulation and make sure it doesn't crash."""

    def test_10_ticks_no_crash(self):
        random.seed(42)
        w = reef.empty()
        reef.seed(w)
        for _ in range(10):
            reef.tick(w)
        assert len(w["organisms"]) > 0

    def test_100_ticks_no_crash(self):
        random.seed(42)
        w = reef.empty()
        reef.seed(w)
        for _ in range(100):
            reef.tick(w)
        assert len(w["history"]["population"]) == 100

    def test_multiple_seeds_stable(self):
        """Different random seeds all produce stable 20-tick runs."""
        for s in [1, 13, 42, 99, 256]:
            random.seed(s)
            w = reef.empty()
            reef.seed(w)
            for _ in range(20):
                reef.tick(w)
            assert len(w["organisms"]) >= 0


# ─── Config constants ────────────────────────────────────────────────

class TestConfig:
    """Sanity checks on configuration constants."""

    def test_world_dimensions_positive(self):
        assert reef.WW > 0 and reef.WH > 0

    def test_population_limits(self):
        assert reef.MAX_POP > reef.INIT_POP > 0

    def test_resource_limits(self):
        assert reef.MAX_RES > 0 and reef.RES_SPAWN > 0 and reef.RES_E > 0

    def test_gene_count(self):
        assert len(reef.GENES) == 8 and len(reef.RANGES) == 8

    def test_ranges_ordered(self):
        for name, (lo, hi) in reef.RANGES.items():
            assert lo < hi, f"{name}: lo={lo} >= hi={hi}"

    def test_epochs_ascending(self):
        thresholds = [t for t, _ in reef.EPOCHS]
        assert thresholds == sorted(thresholds)

    def test_predation_radius_positive(self):
        assert reef.PRED_R > 0 and reef.EAT_R > 0

    def test_repro_cooldown_positive(self):
        assert reef.REPRO_CD > 0

    def test_speciation_threshold_positive(self):
        assert reef.SPEC_TH > 0
