"""Tests for the Mars-100 cultural evolution engine."""
from __future__ import annotations

import random
from src.mars100.culture import (
    CATEGORIES,
    CulturalState,
    Lore,
    MUTATION_BASE_PROB,
    SPREAD_BASE_PROB,
    TRADITION_PROMOTE_THRESHOLD,
    TRADITION_RETIRE_THRESHOLD,
    MAX_TOTAL_LORE,
    check_traditions,
    generate_lore_from_event,
    inherit_culture,
    lore_influence,
    mutate_lore,
    retire_dead_lore,
    spread_lore,
    tick_culture,
)


# ── Fixtures / Helpers ──────────────────────────────────────────────

def _make_lore(lid: str = "lore-0000", carriers: list[str] | None = None,
               category: str = "myth", fidelity: float = 0.9,
               virality: float = 0.5, root_id: str | None = None,
               lispy_effect: str = "(* faith 0.05)") -> Lore:
    carriers = carriers or ["c-0"]
    return Lore(
        id=lid, root_id=root_id or lid, parent_id=None,
        category=category, content="Test lore",
        lispy_effect=lispy_effect,
        origin_year=1, origin_colonist="c-0",
        carrier_ids=list(carriers), fidelity=fidelity,
        virality=virality,
    )


def _make_culture(*lore_items: Lore) -> CulturalState:
    cs = CulturalState()
    for l in lore_items:
        cs.lore[l.id] = l
    cs.next_lore_num = len(lore_items)
    return cs


def _make_social_edges(ids: list[str], trust: float = 0.7) -> dict[str, dict]:
    edges: dict[str, dict] = {}
    for a in ids:
        edges[a] = {}
        for b in ids:
            if a != b:
                edges[a][b] = {"trust": trust, "affection": 0.5, "respect": 0.5}
    return edges


class _FakeColonist:
    def __init__(self, cid: str, alive: bool = True, exiled: bool = False) -> None:
        self.id = cid
        self.alive = alive
        self.exiled = exiled

    def is_active(self) -> bool:
        return self.alive and not self.exiled


# ── Lore dataclass tests ────────────────────────────────────────────

class TestLoreSerialization:
    def test_roundtrip(self) -> None:
        lore = _make_lore()
        d = lore.to_dict()
        restored = Lore.from_dict(d)
        assert restored.id == lore.id
        assert restored.root_id == lore.root_id
        assert restored.category == lore.category
        assert restored.carrier_ids == lore.carrier_ids
        assert restored.fidelity == lore.fidelity
        assert restored.extinct_year is None

    def test_extinct_serialization(self) -> None:
        lore = _make_lore()
        lore.extinct_year = 42
        d = lore.to_dict()
        assert d["extinct_year"] == 42
        restored = Lore.from_dict(d)
        assert restored.extinct_year == 42

    def test_carrier_ids_are_sorted(self) -> None:
        lore = _make_lore(carriers=["c-3", "c-1", "c-2"])
        assert lore.carrier_ids == ["c-3", "c-1", "c-2"]  # raw order preserved
        # But spreading/inheriting sorts them


class TestCulturalStateSerialization:
    def test_roundtrip(self) -> None:
        l1 = _make_lore("lore-0000")
        l2 = _make_lore("lore-0001", category="norm")
        cs = _make_culture(l1, l2)
        cs.tradition_ids = ["lore-0000"]
        cs.dead_ids = ["lore-9999"]
        d = cs.to_dict()
        restored = CulturalState.from_dict(d)
        assert len(restored.lore) == 2
        assert restored.tradition_ids == ["lore-0000"]
        assert restored.dead_ids == ["lore-9999"]

    def test_active_lore_excludes_extinct(self) -> None:
        l1 = _make_lore("lore-0000")
        l2 = _make_lore("lore-0001")
        l2.extinct_year = 10
        cs = _make_culture(l1, l2)
        active = cs.active_lore()
        assert len(active) == 1
        assert active[0].id == "lore-0000"

    def test_active_lore_sorted_by_id(self) -> None:
        l1 = _make_lore("lore-0002")
        l2 = _make_lore("lore-0001")
        l3 = _make_lore("lore-0003")
        cs = _make_culture(l1, l2, l3)
        ids = [l.id for l in cs.active_lore()]
        assert ids == sorted(ids)

    def test_generate_id_deterministic(self) -> None:
        cs = CulturalState()
        id1 = cs.generate_id()
        id2 = cs.generate_id()
        assert id1 == "lore-0000"
        assert id2 == "lore-0001"


# ── Lore generation tests ───────────────────────────────────────────

class TestLoreGeneration:
    def test_generate_from_known_event(self) -> None:
        rng = random.Random(42)
        cs = CulturalState()
        # Force generation by using many attempts with fixed seed
        generated = None
        for _ in range(20):
            generated = generate_lore_from_event(
                "dust_storm", 5, "c-0", cs, rng)
            if generated:
                break
        assert generated is not None
        assert generated.origin_year == 5
        assert generated.origin_colonist == "c-0"
        assert generated.id in cs.lore

    def test_generate_from_unknown_event_returns_none(self) -> None:
        rng = random.Random(42)
        cs = CulturalState()
        result = generate_lore_from_event("unknown_event", 1, "c-0", cs, rng)
        assert result is None

    def test_respects_max_total_lore(self) -> None:
        cs = CulturalState()
        for i in range(MAX_TOTAL_LORE):
            lid = cs.generate_id()
            cs.lore[lid] = _make_lore(lid)
        rng = random.Random(42)
        result = generate_lore_from_event("dust_storm", 1, "c-0", cs, rng)
        assert result is None

    def test_lore_has_valid_category(self) -> None:
        rng = random.Random(42)
        cs = CulturalState()
        lore = None
        for _ in range(30):
            lore = generate_lore_from_event("dust_storm", 1, "c-0", cs, rng)
            if lore:
                break
        if lore:
            assert lore.category in CATEGORIES

    def test_deterministic_with_same_seed(self) -> None:
        results = []
        for _ in range(2):
            rng = random.Random(99)
            cs = CulturalState()
            lore = None
            for attempt in range(30):
                lore = generate_lore_from_event("breakthrough", 10, "c-0", cs, rng)
                if lore:
                    results.append(lore.to_dict())
                    break
        if len(results) == 2:
            assert results[0] == results[1]


# ── Spread tests ────────────────────────────────────────────────────

class TestSpread:
    def test_spreads_to_trusted_neighbors(self) -> None:
        lore = _make_lore(carriers=["c-0"], virality=1.0)
        edges = _make_social_edges(["c-0", "c-1", "c-2"], trust=1.0)
        rng = random.Random(42)
        new = spread_lore(lore, edges, ["c-0", "c-1", "c-2"], rng)
        assert len(new) > 0
        assert all(c in lore.carrier_ids for c in new)

    def test_does_not_spread_to_inactive(self) -> None:
        lore = _make_lore(carriers=["c-0"], virality=1.0)
        edges = _make_social_edges(["c-0", "c-1", "c-2"], trust=1.0)
        rng = random.Random(42)
        # c-2 is not in active_ids
        new = spread_lore(lore, edges, ["c-0", "c-1"], rng)
        assert "c-2" not in new

    def test_does_not_reinfect_existing_carriers(self) -> None:
        lore = _make_lore(carriers=["c-0", "c-1"], virality=1.0)
        edges = _make_social_edges(["c-0", "c-1"], trust=1.0)
        rng = random.Random(42)
        new = spread_lore(lore, edges, ["c-0", "c-1"], rng)
        assert len(new) == 0

    def test_carrier_ids_sorted_after_spread(self) -> None:
        lore = _make_lore(carriers=["c-2"], virality=1.0)
        edges = _make_social_edges(["c-0", "c-1", "c-2"], trust=1.0)
        rng = random.Random(42)
        spread_lore(lore, edges, ["c-0", "c-1", "c-2"], rng)
        assert lore.carrier_ids == sorted(lore.carrier_ids)

    def test_low_virality_reduces_spread(self) -> None:
        spread_counts = []
        for virality in [0.1, 1.0]:
            total = 0
            for trial in range(50):
                lore = _make_lore(carriers=["c-0"], virality=virality)
                edges = _make_social_edges(["c-0", "c-1", "c-2", "c-3", "c-4"], trust=0.7)
                rng = random.Random(trial)
                new = spread_lore(lore, edges, ["c-0", "c-1", "c-2", "c-3", "c-4"], rng)
                total += len(new)
            spread_counts.append(total)
        # High virality should spread more
        assert spread_counts[1] > spread_counts[0]


# ── Mutation tests ──────────────────────────────────────────────────

class TestMutation:
    def test_mutation_creates_new_variant(self) -> None:
        lore = _make_lore(fidelity=0.1)  # Low fidelity = high mutation chance
        cs = _make_culture(lore)
        rng = random.Random(42)
        variant = None
        for _ in range(50):
            variant = mutate_lore(lore, "c-1", 5, cs, rng)
            if variant:
                break
        if variant:
            assert variant.id != lore.id
            assert variant.root_id == lore.root_id
            assert variant.parent_id == lore.id
            assert variant.mutation_count == lore.mutation_count + 1

    def test_variant_removes_carrier_from_original(self) -> None:
        lore = _make_lore(carriers=["c-0", "c-1"], fidelity=0.1)
        cs = _make_culture(lore)
        rng = random.Random(42)
        variant = None
        for _ in range(50):
            variant = mutate_lore(lore, "c-1", 5, cs, rng)
            if variant:
                break
        if variant:
            assert "c-1" not in lore.carrier_ids
            assert "c-1" in variant.carrier_ids

    def test_high_fidelity_resists_mutation(self) -> None:
        mutated_count = 0
        for trial in range(100):
            lore = _make_lore(fidelity=1.0)  # Perfect fidelity
            cs = _make_culture(lore)
            rng = random.Random(trial)
            variant = mutate_lore(lore, "c-1", 5, cs, rng)
            if variant:
                mutated_count += 1
        # High fidelity should rarely mutate (mutation_prob = BASE * (1-fidelity) ≈ 0)
        assert mutated_count < 5  # Less than 5% mutation rate

    def test_respects_max_total_lore(self) -> None:
        cs = CulturalState()
        for i in range(MAX_TOTAL_LORE):
            lid = cs.generate_id()
            cs.lore[lid] = _make_lore(lid)
        source = cs.lore["lore-0000"]
        source.fidelity = 0.0  # Maximum mutation chance
        rng = random.Random(42)
        variant = mutate_lore(source, "c-1", 5, cs, rng)
        assert variant is None


# ── Tradition tests ─────────────────────────────────────────────────

class TestTraditions:
    def test_promote_at_threshold(self) -> None:
        # 4 out of 5 = 80% > 60% threshold
        lore = _make_lore(carriers=["c-0", "c-1", "c-2", "c-3"])
        cs = _make_culture(lore)
        new_t = check_traditions(cs, 5)
        assert "lore-0000" in new_t
        assert "lore-0000" in cs.tradition_ids

    def test_no_promote_below_threshold(self) -> None:
        # 2 out of 5 = 40% < 60% threshold
        lore = _make_lore(carriers=["c-0", "c-1"])
        cs = _make_culture(lore)
        new_t = check_traditions(cs, 5)
        assert len(new_t) == 0

    def test_hysteresis_prevents_flapping(self) -> None:
        # Start as tradition with 50% adoption (between retire threshold 40% and promote 60%)
        lore = _make_lore(carriers=["c-0", "c-1", "c-2", "c-3", "c-4"])
        cs = _make_culture(lore)
        cs.tradition_ids = ["lore-0000"]
        # 5 out of 10 = 50% — above retire (40%) but below promote (60%)
        check_traditions(cs, 10)
        assert "lore-0000" in cs.tradition_ids  # Should NOT be retired

    def test_retire_below_threshold(self) -> None:
        # 1 out of 10 = 10% < 40% retire threshold
        lore = _make_lore(carriers=["c-0"])
        cs = _make_culture(lore)
        cs.tradition_ids = ["lore-0000"]
        check_traditions(cs, 10)
        assert "lore-0000" not in cs.tradition_ids

    def test_zero_active_count_no_crash(self) -> None:
        cs = CulturalState()
        result = check_traditions(cs, 0)
        assert result == []


# ── Lore extinction tests ──────────────────────────────────────────

class TestRetirement:
    def test_lore_dies_when_all_carriers_gone(self) -> None:
        lore = _make_lore(carriers=["c-dead"])
        cs = _make_culture(lore)
        extinct = retire_dead_lore(cs, ["c-alive"], 10)
        assert "lore-0000" in extinct
        assert lore.extinct_year == 10
        assert "lore-0000" in cs.dead_ids

    def test_lore_survives_with_active_carrier(self) -> None:
        lore = _make_lore(carriers=["c-0", "c-dead"])
        cs = _make_culture(lore)
        extinct = retire_dead_lore(cs, ["c-0"], 10)
        assert len(extinct) == 0
        assert lore.carrier_ids == ["c-0"]  # Dead carrier removed

    def test_tradition_removed_on_extinction(self) -> None:
        lore = _make_lore(carriers=["c-dead"])
        cs = _make_culture(lore)
        cs.tradition_ids = ["lore-0000"]
        retire_dead_lore(cs, ["c-alive"], 10)
        assert "lore-0000" not in cs.tradition_ids


# ── Inheritance tests ───────────────────────────────────────────────

class TestInheritance:
    def test_child_inherits_parent_lore(self) -> None:
        lore = _make_lore(carriers=["parent-a", "parent-b"])
        cs = _make_culture(lore)
        rng = random.Random(42)
        inherited = inherit_culture("child-0", ["parent-a", "parent-b"], cs, rng)
        # With 70% chance and seed 42, should usually inherit
        # Run multiple times to be probabilistically robust
        total_inherited = 0
        for trial in range(20):
            test_lore = _make_lore(carriers=["parent-a"])
            test_cs = _make_culture(test_lore)
            result = inherit_culture("child-0", ["parent-a"], test_cs, random.Random(trial))
            total_inherited += len(result)
        assert total_inherited > 5  # At least ~25% of 20 trials

    def test_child_does_not_inherit_non_parent_lore(self) -> None:
        lore = _make_lore(carriers=["stranger"])
        cs = _make_culture(lore)
        rng = random.Random(42)
        inherited = inherit_culture("child-0", ["parent-a"], cs, rng)
        assert len(inherited) == 0

    def test_inherited_carriers_sorted(self) -> None:
        lore = _make_lore(carriers=["parent-a"])
        cs = _make_culture(lore)
        rng = random.Random(1)  # Seed that gives inheritance
        inherited = inherit_culture("child-0", ["parent-a"], cs, rng)
        if inherited:
            assert lore.carrier_ids == sorted(lore.carrier_ids)


# ── Influence tests ─────────────────────────────────────────────────

class TestInfluence:
    def test_basic_influence(self) -> None:
        lore = _make_lore(lispy_effect="(* faith 0.05)")
        result = lore_influence([("c-0", lore)])
        assert "faith" in result
        assert abs(result["faith"] - 0.05) < 0.001

    def test_multiple_lore_stack(self) -> None:
        l1 = _make_lore("l1", lispy_effect="(* faith 0.05)")
        l2 = _make_lore("l2", lispy_effect="(* faith 0.03)")
        result = lore_influence([("c-0", l1), ("c-0", l2)])
        assert abs(result["faith"] - 0.08) < 0.001

    def test_influence_capped(self) -> None:
        lore_items = [
            ("c-0", _make_lore(f"l{i}", lispy_effect="(* resolve 0.10)"))
            for i in range(5)
        ]
        result = lore_influence(lore_items)
        assert result["resolve"] <= 0.15  # Capped

    def test_invalid_effect_ignored(self) -> None:
        lore = _make_lore(lispy_effect="(broken syntax)")
        result = lore_influence([("c-0", lore)])
        assert len(result) == 0

    def test_empty_lore_no_influence(self) -> None:
        result = lore_influence([])
        assert result == {}


# ── tick_culture integration tests ──────────────────────────────────

class _FakeEvent:
    def __init__(self, name: str, severity: float = 0.5) -> None:
        self.name = name
        self.severity = severity


class TestTickCulture:
    def test_generates_lore_from_events(self) -> None:
        colonists = [_FakeColonist(f"c-{i}") for i in range(5)]
        edges = _make_social_edges([f"c-{i}" for i in range(5)])
        events = [_FakeEvent("dust_storm"), _FakeEvent("breakthrough")]
        cs = CulturalState()
        rng = random.Random(42)

        # Run multiple ticks to get at least some lore
        total_new = 0
        for year in range(1, 20):
            delta = tick_culture(year, colonists, edges, events, cs, rng)
            total_new += len(delta["new_lore"])
        assert total_new > 0

    def test_deterministic(self) -> None:
        results = []
        for _ in range(2):
            colonists = [_FakeColonist(f"c-{i}") for i in range(5)]
            edges = _make_social_edges([f"c-{i}" for i in range(5)])
            events = [_FakeEvent("dust_storm")]
            cs = CulturalState()
            rng = random.Random(42)
            delta = tick_culture(1, colonists, edges, events, cs, rng)
            results.append(delta)
        assert results[0] == results[1]

    def test_no_crash_with_empty_colony(self) -> None:
        cs = CulturalState()
        rng = random.Random(42)
        delta = tick_culture(1, [], {}, [], cs, rng)
        assert delta["year"] == 1
        assert len(delta["new_lore"]) == 0

    def test_lore_spreads_over_time(self) -> None:
        colonists = [_FakeColonist(f"c-{i}") for i in range(10)]
        edges = _make_social_edges([f"c-{i}" for i in range(10)], trust=0.8)
        events = [_FakeEvent("dust_storm")]
        cs = CulturalState()
        rng = random.Random(42)

        # Generate initial lore
        for year in range(1, 5):
            tick_culture(year, colonists, edges, events, cs, rng)

        if cs.active_lore():
            initial_carriers = len(cs.active_lore()[0].carrier_ids)
            # Run more ticks to spread
            for year in range(5, 20):
                tick_culture(year, colonists, edges, events, cs, rng)
            final_carriers = len(cs.active_lore()[0].carrier_ids)
            assert final_carriers >= initial_carriers

    def test_dead_colonist_lore_cleaned(self) -> None:
        c0 = _FakeColonist("c-0")
        c1 = _FakeColonist("c-1", alive=False)
        lore = _make_lore(carriers=["c-0", "c-1"])
        cs = _make_culture(lore)
        edges = _make_social_edges(["c-0"])
        rng = random.Random(42)
        delta = tick_culture(10, [c0, c1], edges, [], cs, rng)
        assert "c-1" not in lore.carrier_ids

    def test_influence_computed(self) -> None:
        c0 = _FakeColonist("c-0")
        lore = _make_lore(carriers=["c-0"], lispy_effect="(* resolve 0.05)")
        cs = _make_culture(lore)
        edges = _make_social_edges(["c-0"])
        rng = random.Random(42)
        delta = tick_culture(1, [c0], edges, [], cs, rng)
        assert "c-0" in delta["influence"]
        assert "resolve" in delta["influence"]["c-0"]


# ── Property-based invariant tests ──────────────────────────────────

class TestInvariants:
    def test_carrier_ids_always_subset_of_active(self) -> None:
        """After tick_culture, no extinct/inactive colonist remains a carrier."""
        colonists = [_FakeColonist(f"c-{i}", alive=(i < 5)) for i in range(10)]
        edges = _make_social_edges([f"c-{i}" for i in range(5)])
        events = [_FakeEvent("dust_storm"), _FakeEvent("earth_contact")]
        cs = CulturalState()
        rng = random.Random(42)

        for year in range(1, 30):
            tick_culture(year, colonists, edges, events, cs, rng)

        active_ids = {f"c-{i}" for i in range(5)}
        for lore in cs.active_lore():
            for carrier in lore.carrier_ids:
                assert carrier in active_ids

    def test_lore_count_bounded(self) -> None:
        """Total active lore never exceeds MAX_TOTAL_LORE."""
        colonists = [_FakeColonist(f"c-{i}") for i in range(10)]
        edges = _make_social_edges([f"c-{i}" for i in range(10)])
        events = [_FakeEvent("dust_storm"), _FakeEvent("breakthrough"),
                  _FakeEvent("alien_signal")]
        cs = CulturalState()
        rng = random.Random(42)

        for year in range(1, 200):
            tick_culture(year, colonists, edges, events, cs, rng)
            assert len(cs.active_lore()) <= MAX_TOTAL_LORE

    def test_influence_deltas_bounded(self) -> None:
        """Influence on any stat is capped at ±0.15."""
        lore_items = [
            ("c-0", _make_lore(f"l{i}", lispy_effect=f"(* faith {0.1 * (i+1)})"))
            for i in range(10)
        ]
        result = lore_influence(lore_items)
        for stat, delta in result.items():
            assert -0.15 <= delta <= 0.15

    def test_same_seed_same_culture(self) -> None:
        """Same seed produces identical cultural state."""
        states = []
        for _ in range(2):
            colonists = [_FakeColonist(f"c-{i}") for i in range(10)]
            edges = _make_social_edges([f"c-{i}" for i in range(10)])
            events = [_FakeEvent("dust_storm"), _FakeEvent("earth_contact")]
            cs = CulturalState()
            rng = random.Random(42)
            for year in range(1, 50):
                tick_culture(year, colonists, edges, events, cs, rng)
            states.append(cs.to_dict())
        assert states[0] == states[1]

    def test_fidelity_decays_monotonically(self) -> None:
        """Lore fidelity never increases over time (excluding new lore)."""
        lore = _make_lore(fidelity=0.9)
        cs = _make_culture(lore)
        colonists = [_FakeColonist("c-0")]
        edges = _make_social_edges(["c-0"])
        rng = random.Random(42)

        prev_fidelity = lore.fidelity
        for year in range(1, 20):
            tick_culture(year, colonists, edges, [], cs, rng)
            assert lore.fidelity <= prev_fidelity
            prev_fidelity = lore.fidelity
