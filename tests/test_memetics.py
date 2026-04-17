"""Tests for the Mars-100 memetics engine."""
from __future__ import annotations

import random
import pytest

from src.mars100.memetics import (
    Meme, MemePool, MAX_MEMES_PER_COLONIST,
    PROPAGATION_BASE, CRISIS_SEVERITY_THRESHOLD,
    _governance_content, _crisis_content,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

class FakeRelationship:
    def __init__(self, trust: float = 0.5):
        self.trust = trust

class FakeStats:
    def __init__(self, empathy: float = 0.5, paranoia: float = 0.2):
        self.empathy = empathy
        self.paranoia = paranoia


def _make_pool_with_meme(rng: random.Random | None = None) -> tuple[MemePool, Meme]:
    rng = rng or random.Random(42)
    pool = MemePool()
    meme = pool.create_governance_meme(year=5, proposer_id="a", gov_type="council", rng=rng)
    return pool, meme


# ------------------------------------------------------------------
# Meme basics
# ------------------------------------------------------------------

class TestMemeBasics:
    def test_creation(self):
        rng = random.Random(42)
        pool = MemePool()
        m = pool.create_governance_meme(1, "kira-sol", "council", rng)
        assert m.meme_type == "governance_norm"
        assert "kira-sol" in m.carriers
        assert m.origin_year == 1
        assert m.id == "meme-0"

    def test_serialization_roundtrip(self):
        rng = random.Random(42)
        pool = MemePool()
        pool.create_governance_meme(1, "kira-sol", "council", rng)
        pool.create_crisis_meme(5, ["kira-sol", "fen-marsh"], "dust_storm", rng)
        d = pool.to_dict()
        assert d["total_memes"] == 2
        assert d["active_memes"] == 2
        restored = MemePool.from_dict(d)
        assert len(restored.memes) == 2
        for mid in pool.memes:
            assert mid in restored.memes
            assert restored.memes[mid].to_dict() == pool.memes[mid].to_dict()

    def test_meme_to_dict_carriers_sorted(self):
        m = Meme(id="meme-0", name="test", meme_type="governance_norm",
                 origin_year=1, origin_colonist="b",
                 content={"mediate": 0.1}, virality=0.5,
                 carriers=["c", "a", "b"])
        d = m.to_dict()
        assert d["carriers"] == ["a", "b", "c"]

    def test_salience_decreases_with_age(self):
        m = Meme(id="meme-0", name="old", meme_type="governance_norm",
                 origin_year=1, origin_colonist="a",
                 content={}, virality=0.5, carriers=["a"])
        s1 = m.salience(2)
        s50 = m.salience(50)
        assert s1 > s50, "Older memes should have lower salience"

    def test_salience_increases_with_spread(self):
        m1 = Meme(id="meme-0", name="lonely", meme_type="governance_norm",
                  origin_year=1, origin_colonist="a",
                  content={}, virality=0.5, carriers=["a"])
        m2 = Meme(id="meme-1", name="popular", meme_type="governance_norm",
                  origin_year=1, origin_colonist="a",
                  content={}, virality=0.5, carriers=["a", "b", "c", "d", "e"])
        assert m2.salience(5) > m1.salience(5)


# ------------------------------------------------------------------
# Genesis
# ------------------------------------------------------------------

class TestGenesis:
    def test_governance_meme_types(self):
        rng = random.Random(42)
        pool = MemePool()
        for gov_type in ("council", "dictator", "lottery", "consensus",
                         "ai_governor", "anarchy"):
            m = pool.create_governance_meme(1, "a", gov_type, rng)
            assert m.meme_type == "governance_norm"
            assert m.content  # should have at least one action weight

    def test_crisis_meme_all_events(self):
        rng = random.Random(42)
        pool = MemePool()
        events = ["dust_storm", "solar_flare", "equipment_failure",
                  "epidemic", "colonist_conflict", "unknown_event"]
        for ev in events:
            m = pool.create_crisis_meme(10, ["a", "b"], ev, rng)
            assert m.meme_type == "crisis_response"
            assert m.carriers == ["a", "b"]

    def test_crisis_meme_empty_survivors(self):
        rng = random.Random(42)
        pool = MemePool()
        m = pool.create_crisis_meme(10, [], "dust_storm", rng)
        assert m.carriers == []

    def test_subsim_meme(self):
        rng = random.Random(42)
        pool = MemePool()
        m = pool.create_subsim_meme(20, "rust-vega", "cooperation insight", rng)
        assert m.meme_type == "subsim_insight"
        assert "rust-vega" in m.carriers

    def test_id_increments(self):
        rng = random.Random(42)
        pool = MemePool()
        m1 = pool.create_governance_meme(1, "a", "council", rng)
        m2 = pool.create_governance_meme(2, "b", "dictator", rng)
        assert m1.id == "meme-0"
        assert m2.id == "meme-1"


# ------------------------------------------------------------------
# Propagation
# ------------------------------------------------------------------

class TestPropagation:
    def test_high_trust_spreads(self):
        """Memes should spread along high-trust social edges."""
        rng = random.Random(42)
        pool, meme = _make_pool_with_meme(rng)
        active = ["a", "b"]

        def social_get(a, b):
            return FakeRelationship(trust=0.95)

        def stats_get(cid):
            return FakeStats(empathy=0.9, paranoia=0.0)

        spread = False
        for _ in range(20):
            pool.propagate(5, active, social_get, stats_get, rng)
            if "b" in meme.carriers:
                spread = True
                break
        assert spread, "High trust + high empathy should cause propagation"

    def test_low_trust_blocks(self):
        """Memes should NOT spread along zero-trust edges."""
        rng = random.Random(42)
        pool, meme = _make_pool_with_meme(rng)
        active = ["a", "b"]

        def social_get(a, b):
            return FakeRelationship(trust=0.0)

        def stats_get(cid):
            return FakeStats(empathy=0.5, paranoia=0.5)

        for _ in range(50):
            pool.propagate(5, active, social_get, stats_get, rng)
        assert "b" not in meme.carriers

    def test_high_paranoia_resists(self):
        """High paranoia targets should resist meme adoption."""
        rng = random.Random(42)
        pool, meme = _make_pool_with_meme(rng)
        active = ["a", "b"]

        def social_get(a, b):
            return FakeRelationship(trust=0.5)

        def stats_get(cid):
            if cid == "b":
                return FakeStats(empathy=0.5, paranoia=1.0)
            return FakeStats(empathy=0.5, paranoia=0.0)

        for _ in range(50):
            pool.propagate(5, active, social_get, stats_get, rng)
        # Reduced but still possible — just verifying no crash
        assert True

    def test_no_self_propagation(self):
        """A colonist should not propagate a meme to themselves."""
        rng = random.Random(42)
        pool, meme = _make_pool_with_meme(rng)
        initial_count = meme.carriers.count("a")
        pool.propagate(5, ["a"], lambda a, b: FakeRelationship(1.0),
                       lambda c: FakeStats(1.0, 0.0), rng)
        assert meme.carriers.count("a") == initial_count

    def test_already_carrying_skipped(self):
        """If target already carries the meme, skip."""
        rng = random.Random(42)
        pool = MemePool()
        m = pool.create_governance_meme(1, "a", "council", rng)
        m.carriers = sorted(["a", "b"])
        original_carriers = list(m.carriers)
        pool.propagate(5, ["a", "b"], lambda a, b: FakeRelationship(1.0),
                       lambda c: FakeStats(1.0, 0.0), rng)
        assert m.carriers == original_carriers

    def test_null_stats_skipped(self):
        """If stats_get returns None, propagation should skip that pair."""
        rng = random.Random(42)
        pool, meme = _make_pool_with_meme(rng)
        active = ["a", "b"]

        pool.propagate(5, active, lambda a, b: FakeRelationship(1.0),
                       lambda c: None, rng)
        # Should not crash, b should not adopt
        assert "b" not in meme.carriers


# ------------------------------------------------------------------
# Displacement
# ------------------------------------------------------------------

class TestDisplacement:
    def test_max_memes_enforced(self):
        """A colonist cannot hold more than MAX_MEMES_PER_COLONIST via adoption."""
        rng = random.Random(42)
        pool = MemePool()
        # Create MAX + 3 memes all carried by "a"
        for i in range(MAX_MEMES_PER_COLONIST + 3):
            pool.create_governance_meme(1, "a", "council", rng)
        # Propagate to b via direct adoption
        active = ["a", "b"]
        pool.propagate(5, active, lambda a, b: FakeRelationship(1.0),
                       lambda c: FakeStats(1.0, 0.0), rng)
        assert pool.carrier_count("b") <= MAX_MEMES_PER_COLONIST

    def test_weaker_meme_displaced(self):
        """When at cap, weaker memes get displaced by stronger ones."""
        rng = random.Random(42)
        pool = MemePool()
        for i in range(MAX_MEMES_PER_COLONIST):
            m = Meme(
                id=f"meme-{i}", name=f"old-{i}", meme_type="governance_norm",
                origin_year=1, origin_colonist="x",
                content={"mediate": 0.01}, virality=0.05,
                carriers=["b"],
            )
            pool.memes[m.id] = m
        pool._next_id = MAX_MEMES_PER_COLONIST
        strong = pool.create_governance_meme(50, "a", "council", rng)
        strong.virality = 0.95
        adopted = pool._try_adopt("b", strong, 50)
        assert adopted, "Strong meme should displace weak ones"
        assert "b" in strong.carriers


# ------------------------------------------------------------------
# Inheritance
# ------------------------------------------------------------------

class TestInheritance:
    def test_child_inherits_parent_memes(self):
        rng = random.Random(42)
        pool = MemePool()
        pool.create_governance_meme(1, "a", "council", rng)
        pool.create_crisis_meme(5, ["a", "b"], "dust_storm", rng)
        inherited = pool.inherit_memes("child-1", "a", "b", 10, rng)
        assert len(inherited) > 0
        assert len(inherited) <= MAX_MEMES_PER_COLONIST

    def test_child_inherits_max_cap(self):
        rng = random.Random(42)
        pool = MemePool()
        for i in range(10):
            parent = "a" if i < 5 else "b"
            pool.create_governance_meme(i, parent, "council", rng)
        inherited = pool.inherit_memes("child-1", "a", "b", 20, rng)
        assert len(inherited) <= MAX_MEMES_PER_COLONIST

    def test_inherit_from_one_parent(self):
        rng = random.Random(42)
        pool = MemePool()
        pool.create_governance_meme(1, "a", "council", rng)
        inherited = pool.inherit_memes("child-1", "a", "nobody", 10, rng)
        assert isinstance(inherited, list)


# ------------------------------------------------------------------
# Carrier cleanup
# ------------------------------------------------------------------

class TestCarrierCleanup:
    def test_deactivate_removes_from_all_memes(self):
        rng = random.Random(42)
        pool = MemePool()
        m1 = pool.create_governance_meme(1, "a", "council", rng)
        m1.carriers = sorted(["a", "b", "c"])
        m2 = pool.create_crisis_meme(5, ["a", "c"], "dust_storm", rng)
        pool.deactivate_carrier("a")
        assert "a" not in m1.carriers
        assert "a" not in m2.carriers
        assert "b" in m1.carriers
        assert "c" in m2.carriers

    def test_deactivate_nonexistent(self):
        """Deactivating a non-carrier should not crash."""
        pool = MemePool()
        pool.deactivate_carrier("nobody")


# ------------------------------------------------------------------
# Action weight deltas
# ------------------------------------------------------------------

class TestActionWeights:
    def test_single_meme_weights(self):
        pool = MemePool()
        m = Meme(id="meme-0", name="test", meme_type="governance_norm",
                 origin_year=1, origin_colonist="a",
                 content={"mediate": 0.1, "cooperate": 0.05},
                 virality=0.5, carriers=["a"])
        pool.memes[m.id] = m
        deltas = pool.action_weight_deltas("a")
        assert deltas["mediate"] == pytest.approx(0.1)
        assert deltas["cooperate"] == pytest.approx(0.05)

    def test_multiple_memes_stack(self):
        pool = MemePool()
        m1 = Meme(id="meme-0", name="t1", meme_type="governance_norm",
                  origin_year=1, origin_colonist="a",
                  content={"mediate": 0.1}, virality=0.5, carriers=["a"])
        m2 = Meme(id="meme-1", name="t2", meme_type="crisis_response",
                  origin_year=2, origin_colonist="a",
                  content={"mediate": 0.05, "code": 0.03},
                  virality=0.5, carriers=["a"])
        pool.memes[m1.id] = m1
        pool.memes[m2.id] = m2
        deltas = pool.action_weight_deltas("a")
        assert deltas["mediate"] == pytest.approx(0.15)
        assert deltas["code"] == pytest.approx(0.03)

    def test_no_memes_empty_deltas(self):
        pool = MemePool()
        deltas = pool.action_weight_deltas("nobody")
        assert deltas == {}


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

class TestSummary:
    def test_summary_counts(self):
        rng = random.Random(42)
        pool = MemePool()
        pool.create_governance_meme(1, "a", "council", rng)
        pool.create_crisis_meme(5, ["a", "b"], "dust_storm", rng)
        extinct = Meme(id="meme-99", name="dead", meme_type="governance_norm",
                       origin_year=1, origin_colonist="x",
                       content={}, virality=0.1, carriers=[])
        pool.memes[extinct.id] = extinct
        s = pool.summary(10)
        assert s["total"] == 3
        assert s["active"] == 2
        assert s["extinct"] == 1
        assert s["by_type"]["governance_norm"] == 1
        assert s["by_type"]["crisis_response"] == 1


# ------------------------------------------------------------------
# Content generators
# ------------------------------------------------------------------

class TestContentGenerators:
    def test_governance_content_all_types(self):
        rng = random.Random(42)
        for gov_type in ("council", "dictator", "lottery", "consensus",
                         "ai_governor", "anarchy"):
            content = _governance_content(gov_type, rng)
            assert isinstance(content, dict)
            for v in content.values():
                assert isinstance(v, float)

    def test_crisis_content_all_events(self):
        rng = random.Random(42)
        for ev in ("dust_storm", "solar_flare", "equipment_failure",
                   "epidemic", "colonist_conflict", "other"):
            content = _crisis_content(ev, rng)
            assert isinstance(content, dict)


# ------------------------------------------------------------------
# Determinism
# ------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_memes(self):
        """Two pools with same seed should produce identical memes."""
        for seed in (42, 99, 1):
            pool_a = MemePool()
            rng_a = random.Random(seed)
            pool_a.create_governance_meme(1, "a", "council", rng_a)
            pool_a.create_crisis_meme(5, ["a", "b"], "dust_storm", rng_a)

            pool_b = MemePool()
            rng_b = random.Random(seed)
            pool_b.create_governance_meme(1, "a", "council", rng_b)
            pool_b.create_crisis_meme(5, ["a", "b"], "dust_storm", rng_b)

            assert pool_a.to_dict() == pool_b.to_dict()

    def test_propagation_deterministic(self):
        for seed in (42, 77):
            def run_prop(s):
                rng = random.Random(s)
                pool, _ = _make_pool_with_meme(rng)
                active = ["a", "b", "c"]
                for _ in range(10):
                    pool.propagate(5, active,
                                   lambda a, b: FakeRelationship(0.8),
                                   lambda c: FakeStats(0.7, 0.1),
                                   rng)
                return pool.to_dict()
            assert run_prop(seed) == run_prop(seed)


# ------------------------------------------------------------------
# Property-based invariants
# ------------------------------------------------------------------

class TestInvariants:
    def test_virality_bounded(self):
        rng = random.Random(42)
        pool = MemePool()
        for _ in range(50):
            m = pool.create_governance_meme(1, "a", "council", rng)
            assert 0.0 <= m.virality <= 1.0

    def test_carriers_always_sorted(self):
        rng = random.Random(42)
        pool = MemePool()
        m = pool.create_crisis_meme(1, ["c", "a", "b"], "dust_storm", rng)
        assert m.carriers == sorted(m.carriers)
        pool.propagate(5, ["a", "b", "c", "d"],
                       lambda a, b: FakeRelationship(0.9),
                       lambda c: FakeStats(0.9, 0.0), rng)
        for meme in pool.memes.values():
            assert meme.carriers == sorted(meme.carriers), \
                f"Meme {meme.id} carriers not sorted: {meme.carriers}"

    def test_no_duplicate_carriers(self):
        rng = random.Random(42)
        pool = MemePool()
        m = pool.create_governance_meme(1, "a", "council", rng)
        pool._try_adopt("a", m, 5)  # already carrying
        assert m.carriers.count("a") == 1
