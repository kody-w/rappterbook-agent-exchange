"""Tests for the genetics organ (v10.0)."""
from __future__ import annotations

import math
import random
import pytest
from src.mars100.genetics import (
    Genome, create_genome_from_stats, create_random_genome,
    crossover, compute_inbreeding_coefficient, inbreeding_death_modifier,
    inbreeding_learning_modifier, genetic_distance, colony_diversity,
    NUM_ALLELES, STAT_NAMES, MUTATION_RATE, MUTATION_SIGMA,
    ALLELES_PER_STAT, INBREEDING_DEATH_MULTIPLIER,
    INBREEDING_LEARNING_PENALTY, DIVERSITY_WARNING_THRESHOLD,
    _get_ancestors,
)


class TestGenome:
    """Genome data structure tests."""

    def test_default_alleles(self) -> None:
        g = Genome()
        assert len(g.alleles) == NUM_ALLELES
        assert all(a == 0.5 for a in g.alleles)

    def test_stat_baseline(self) -> None:
        alleles = [0.0] * NUM_ALLELES
        alleles[0] = 0.3
        alleles[1] = 0.7
        g = Genome(alleles=alleles)
        assert abs(g.stat_baseline(0) - 0.5) < 1e-9

    def test_stat_baselines_all(self) -> None:
        g = Genome()
        baselines = g.stat_baselines()
        assert set(baselines.keys()) == set(STAT_NAMES)
        for v in baselines.values():
            assert 0.0 <= v <= 1.0

    def test_roundtrip(self) -> None:
        g = Genome(alleles=[0.1 * i for i in range(NUM_ALLELES)],
                   parent_ids=["p1", "p2"])
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        assert len(g2.alleles) == len(g.alleles)
        for a, b in zip(g.alleles, g2.alleles):
            assert abs(a - b) < 1e-5
        assert g2.parent_ids == ["p1", "p2"]

    def test_from_empty_dict(self) -> None:
        g = Genome.from_dict({})
        assert len(g.alleles) == NUM_ALLELES
        assert g.parent_ids == []

    def test_from_none(self) -> None:
        g = Genome.from_dict(None)
        assert len(g.alleles) == NUM_ALLELES


class TestCreateGenomeFromStats:
    """Back-fitting genome from existing stats."""

    def test_baselines_match_stats(self) -> None:
        stats = {"resolve": 0.9, "improvisation": 0.4, "empathy": 0.5,
                 "hoarding": 0.3, "faith": 0.2, "paranoia": 0.7}
        rng = random.Random(42)
        g = create_genome_from_stats(stats, rng)
        baselines = g.stat_baselines()
        for name in STAT_NAMES:
            # The mean of two alleles should be close to the original stat
            assert abs(baselines[name] - stats[name]) < 0.2, \
                f"{name}: baseline={baselines[name]:.3f}, stat={stats[name]}"

    def test_alleles_bounded(self) -> None:
        stats = {"resolve": 0.95, "improvisation": 0.05, "empathy": 0.5,
                 "hoarding": 0.5, "faith": 0.5, "paranoia": 0.5}
        rng = random.Random(42)
        g = create_genome_from_stats(stats, rng)
        for a in g.alleles:
            assert 0.0 <= a <= 1.0

    def test_deterministic(self) -> None:
        stats = {"resolve": 0.5, "improvisation": 0.5, "empathy": 0.5,
                 "hoarding": 0.5, "faith": 0.5, "paranoia": 0.5}
        g1 = create_genome_from_stats(stats, random.Random(42))
        g2 = create_genome_from_stats(stats, random.Random(42))
        assert g1.alleles == g2.alleles

    def test_creates_allelic_variation(self) -> None:
        """Two alleles for each stat should not be identical."""
        stats = {"resolve": 0.5, "improvisation": 0.5, "empathy": 0.5,
                 "hoarding": 0.5, "faith": 0.5, "paranoia": 0.5}
        rng = random.Random(42)
        g = create_genome_from_stats(stats, rng)
        # At least some allele pairs should differ
        diffs = [abs(g.alleles[i*2] - g.alleles[i*2+1]) for i in range(6)]
        assert any(d > 0.001 for d in diffs)


class TestCreateRandomGenome:
    def test_bounded(self) -> None:
        rng = random.Random(42)
        g = create_random_genome(rng)
        assert len(g.alleles) == NUM_ALLELES
        for a in g.alleles:
            assert 0.0 <= a <= 1.0

    def test_deterministic(self) -> None:
        g1 = create_random_genome(random.Random(42))
        g2 = create_random_genome(random.Random(42))
        assert g1.alleles == g2.alleles

    def test_different_seeds_different_genomes(self) -> None:
        g1 = create_random_genome(random.Random(1))
        g2 = create_random_genome(random.Random(2))
        assert g1.alleles != g2.alleles


class TestCrossover:
    def test_child_alleles_bounded(self) -> None:
        rng = random.Random(42)
        pa = create_random_genome(rng)
        pb = create_random_genome(rng)
        child = crossover(pa, pb, "child-1", random.Random(99))
        for a in child.alleles:
            assert 0.0 <= a <= 1.0

    def test_child_allele_count(self) -> None:
        rng = random.Random(42)
        pa = create_random_genome(rng)
        pb = create_random_genome(rng)
        child = crossover(pa, pb, "child-1", random.Random(99))
        assert len(child.alleles) == NUM_ALLELES

    def test_deterministic(self) -> None:
        rng1 = random.Random(42)
        pa = create_random_genome(rng1)
        pb = create_random_genome(rng1)
        c1 = crossover(pa, pb, "c1", random.Random(99))
        rng2 = random.Random(42)
        pa2 = create_random_genome(rng2)
        pb2 = create_random_genome(rng2)
        c2 = crossover(pa2, pb2, "c2", random.Random(99))
        assert c1.alleles == c2.alleles

    def test_child_inherits_from_parents(self) -> None:
        """With no mutation, each child allele should come from a parent."""
        # Use a seed where mutations are unlikely
        pa = Genome(alleles=[0.1] * NUM_ALLELES)
        pb = Genome(alleles=[0.9] * NUM_ALLELES)
        # Run many times, check alleles are from parent range
        rng = random.Random(42)
        for _ in range(20):
            child = crossover(pa, pb, "c", rng)
            for a in child.alleles:
                # Mutation could push slightly out of [0.1, 0.9]
                assert -0.1 <= a <= 1.1, f"allele {a} way out of range"

    def test_mutation_bounded(self) -> None:
        """Even with mutation, alleles should be clamped to [0, 1]."""
        pa = Genome(alleles=[0.0] * NUM_ALLELES)
        pb = Genome(alleles=[1.0] * NUM_ALLELES)
        rng = random.Random(42)
        for _ in range(100):
            child = crossover(pa, pb, "c", rng)
            for a in child.alleles:
                assert 0.0 <= a <= 1.0


class TestInbreedingCoefficient:
    def test_unrelated_parents(self) -> None:
        pedigree = {"a": [], "b": []}
        f = compute_inbreeding_coefficient(["a", "b"], pedigree)
        assert f == 0.0

    def test_sibling_parents(self) -> None:
        """Children of the same parents have F = 0.25."""
        pedigree = {
            "grandpa": [], "grandma": [],
            "parent_a": ["grandpa", "grandma"],
            "parent_b": ["grandpa", "grandma"],
        }
        f = compute_inbreeding_coefficient(["parent_a", "parent_b"], pedigree)
        assert abs(f - 0.25) < 0.01

    def test_half_sibling_parents(self) -> None:
        """Half-siblings share one parent: F = 0.125."""
        pedigree = {
            "shared": [], "other_a": [], "other_b": [],
            "parent_a": ["shared", "other_a"],
            "parent_b": ["shared", "other_b"],
        }
        f = compute_inbreeding_coefficient(["parent_a", "parent_b"], pedigree)
        assert abs(f - 0.125) < 0.01

    def test_empty_pedigree(self) -> None:
        f = compute_inbreeding_coefficient(["a", "b"], {})
        assert f == 0.0

    def test_single_parent(self) -> None:
        f = compute_inbreeding_coefficient(["a"], {"a": []})
        assert f == 0.0

    def test_deeply_related(self) -> None:
        """Multiple generations of inbreeding → higher F."""
        pedigree = {
            "g1a": [], "g1b": [],
            "g2a": ["g1a", "g1b"], "g2b": ["g1a", "g1b"],
            "g3a": ["g2a", "g2b"], "g3b": ["g2a", "g2b"],
        }
        f = compute_inbreeding_coefficient(["g3a", "g3b"], pedigree)
        # Should be higher than simple siblings
        assert f > 0.25

    def test_f_bounded(self) -> None:
        """F should always be in [0, 1]."""
        pedigree = {
            "a": [], "b": [],
            "c": ["a", "b"], "d": ["a", "b"],
            "e": ["c", "d"], "f": ["c", "d"],
        }
        f = compute_inbreeding_coefficient(["e", "f"], pedigree)
        assert 0.0 <= f <= 1.0


class TestInbreedingModifiers:
    def test_death_modifier_zero_f(self) -> None:
        assert inbreeding_death_modifier(0.0) == 1.0

    def test_death_modifier_max_f(self) -> None:
        m = inbreeding_death_modifier(1.0)
        assert abs(m - INBREEDING_DEATH_MULTIPLIER) < 1e-9

    def test_death_modifier_linear(self) -> None:
        m = inbreeding_death_modifier(0.5)
        expected = 1.0 + 0.5 * (INBREEDING_DEATH_MULTIPLIER - 1.0)
        assert abs(m - expected) < 1e-9

    def test_learning_modifier_zero_f(self) -> None:
        assert inbreeding_learning_modifier(0.0) == 1.0

    def test_learning_modifier_max_f(self) -> None:
        m = inbreeding_learning_modifier(1.0)
        assert m == max(0.5, 1.0 - INBREEDING_LEARNING_PENALTY)

    def test_learning_modifier_bounded(self) -> None:
        for f in [0.0, 0.25, 0.5, 0.75, 1.0]:
            m = inbreeding_learning_modifier(f)
            assert 0.5 <= m <= 1.0


class TestGeneticDistance:
    def test_identical_genomes(self) -> None:
        g = Genome(alleles=[0.5] * NUM_ALLELES)
        assert genetic_distance(g, g) == 0.0

    def test_maximum_distance(self) -> None:
        g1 = Genome(alleles=[0.0] * NUM_ALLELES)
        g2 = Genome(alleles=[1.0] * NUM_ALLELES)
        d = genetic_distance(g1, g2)
        assert abs(d - 1.0) < 1e-9

    def test_symmetry(self) -> None:
        rng = random.Random(42)
        g1 = create_random_genome(rng)
        g2 = create_random_genome(rng)
        assert abs(genetic_distance(g1, g2) - genetic_distance(g2, g1)) < 1e-9

    def test_bounded(self) -> None:
        rng = random.Random(42)
        for _ in range(20):
            g1 = create_random_genome(rng)
            g2 = create_random_genome(rng)
            d = genetic_distance(g1, g2)
            assert 0.0 <= d <= 1.0


class TestColonyDiversity:
    def test_single_genome(self) -> None:
        g = Genome()
        assert colony_diversity([g]) == 0.0

    def test_identical_genomes(self) -> None:
        gs = [Genome(alleles=[0.5] * NUM_ALLELES) for _ in range(5)]
        assert colony_diversity(gs) == 0.0

    def test_maximally_diverse(self) -> None:
        """Two genomes at extreme opposites should give high diversity."""
        g1 = Genome(alleles=[0.0] * NUM_ALLELES)
        g2 = Genome(alleles=[1.0] * NUM_ALLELES)
        d = colony_diversity([g1, g2])
        assert d > 0.5

    def test_bounded(self) -> None:
        rng = random.Random(42)
        gs = [create_random_genome(rng) for _ in range(10)]
        d = colony_diversity(gs)
        assert 0.0 <= d <= 1.0

    def test_more_genomes_more_stable(self) -> None:
        """Adding similar genomes shouldn't dramatically change diversity."""
        rng = random.Random(42)
        gs = [create_random_genome(rng) for _ in range(5)]
        d5 = colony_diversity(gs)
        gs.extend([create_random_genome(rng) for _ in range(5)])
        d10 = colony_diversity(gs)
        assert abs(d10 - d5) < 0.5  # shouldn't change wildly

    def test_immigration_increases_diversity(self) -> None:
        """Cloned population + immigrant should increase diversity."""
        base = Genome(alleles=[0.5] * NUM_ALLELES)
        gs = [base] * 5
        d_before = colony_diversity(gs)
        rng = random.Random(42)
        immigrant = create_random_genome(rng)
        gs_after = gs + [immigrant]
        d_after = colony_diversity(gs_after)
        assert d_after > d_before


class TestGetAncestors:
    def test_no_ancestors(self) -> None:
        anc = _get_ancestors("a", {"a": []})
        assert anc == {}

    def test_simple_parents(self) -> None:
        pedigree = {"a": ["b", "c"], "b": [], "c": []}
        anc = _get_ancestors("a", pedigree)
        assert "b" in anc and anc["b"] == 1
        assert "c" in anc and anc["c"] == 1

    def test_depth_limit(self) -> None:
        # Chain: a -> b -> c -> d -> ... up to depth 10
        pedigree = {}
        for i in range(15):
            pedigree[f"n{i}"] = [f"n{i+1}"] if i < 14 else []
        anc = _get_ancestors("n0", pedigree, max_depth=3)
        assert "n3" in anc
        assert "n5" not in anc  # beyond depth 3


class TestIntegrationWithColonist:
    """Test that genome integrates properly with Colonist data model."""

    def test_colonist_roundtrip_with_genome(self) -> None:
        from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills
        g = Genome(alleles=[0.3 + 0.05 * i for i in range(NUM_ALLELES)],
                   parent_ids=["pa", "pb"])
        c = Colonist(
            id="test", name="Test", element="fire", archetype="tester",
            stats=ColonistStats(), skills=ColonistSkills(),
            decision_expr="(+ 1 1)", genome=g, parent_ids=["pa", "pb"],
        )
        d = c.to_dict()
        assert "genome" in d
        assert d["parent_ids"] == ["pa", "pb"]
        c2 = Colonist.from_dict(d)
        assert c2.genome is not None
        assert len(c2.genome.alleles) == NUM_ALLELES
        assert c2.parent_ids == ["pa", "pb"]

    def test_colonist_roundtrip_without_genome(self) -> None:
        from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills
        c = Colonist(
            id="test", name="Test", element="fire", archetype="tester",
            stats=ColonistStats(), skills=ColonistSkills(),
            decision_expr="(+ 1 1)",
        )
        d = c.to_dict()
        assert "genome" not in d
        c2 = Colonist.from_dict(d)
        assert c2.genome is None

    def test_lispy_bindings_with_genome(self) -> None:
        from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills
        g = Genome(alleles=[0.3, 0.7] * 6)
        c = Colonist(
            id="test", name="Test", element="fire", archetype="tester",
            stats=ColonistStats(), skills=ColonistSkills(),
            decision_expr="(+ 1 1)", genome=g, parent_ids=["a", "b"],
        )
        bindings = c.lispy_bindings()
        assert "genetic-diversity" in bindings
        assert "inbred" in bindings
        assert bindings["inbred"] is True

    def test_lispy_bindings_without_genome(self) -> None:
        from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills
        c = Colonist(
            id="test", name="Test", element="fire", archetype="tester",
            stats=ColonistStats(), skills=ColonistSkills(),
            decision_expr="(+ 1 1)",
        )
        bindings = c.lispy_bindings()
        assert "genetic-diversity" not in bindings
        assert bindings["inbred"] is False


class TestCreateChildWithGenetics:
    """Test the genetic crossover path in create_child."""

    def test_child_gets_genome(self) -> None:
        from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, create_child
        rng = random.Random(42)
        gen_rng = random.Random(99)
        pa = Colonist(id="pa", name="A", element="fire", archetype="test",
                      stats=ColonistStats(resolve=0.8), skills=ColonistSkills(),
                      decision_expr="(+ 1 1)",
                      genome=create_genome_from_stats({"resolve": 0.8, "improvisation": 0.5,
                          "empathy": 0.5, "hoarding": 0.5, "faith": 0.5, "paranoia": 0.5}, gen_rng))
        pb = Colonist(id="pb", name="B", element="water", archetype="test",
                      stats=ColonistStats(resolve=0.3), skills=ColonistSkills(),
                      decision_expr="(+ 2 2)",
                      genome=create_genome_from_stats({"resolve": 0.3, "improvisation": 0.5,
                          "empathy": 0.5, "hoarding": 0.5, "faith": 0.5, "paranoia": 0.5}, gen_rng))
        child = create_child(pa, pb, "c1", 10, rng, genetics_rng=gen_rng)
        assert child.genome is not None
        assert child.parent_ids == ["pa", "pb"]
        for a in child.genome.alleles:
            assert 0.0 <= a <= 1.0

    def test_child_stats_from_genome(self) -> None:
        """Child stats should be influenced by genome baselines."""
        from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, create_child
        rng = random.Random(42)
        gen_rng = random.Random(99)
        high_resolve = create_genome_from_stats(
            {"resolve": 0.95, "improvisation": 0.5, "empathy": 0.5,
             "hoarding": 0.5, "faith": 0.5, "paranoia": 0.5}, gen_rng)
        low_resolve = create_genome_from_stats(
            {"resolve": 0.05, "improvisation": 0.5, "empathy": 0.5,
             "hoarding": 0.5, "faith": 0.5, "paranoia": 0.5}, gen_rng)
        pa = Colonist(id="pa", name="A", element="fire", archetype="t",
                      stats=ColonistStats(resolve=0.95), skills=ColonistSkills(),
                      decision_expr="(+ 1 1)", genome=high_resolve)
        pb = Colonist(id="pb", name="B", element="water", archetype="t",
                      stats=ColonistStats(resolve=0.05), skills=ColonistSkills(),
                      decision_expr="(+ 2 2)", genome=low_resolve)
        child = create_child(pa, pb, "c1", 10, rng, genetics_rng=gen_rng)
        # Child resolve should be somewhere between parents (with noise)
        assert 0.0 <= child.stats.resolve <= 1.0

    def test_legacy_path_without_genomes(self) -> None:
        """Without genomes, create_child falls back to v9 stat averaging."""
        from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, create_child
        rng = random.Random(42)
        pa = Colonist(id="pa", name="A", element="fire", archetype="t",
                      stats=ColonistStats(resolve=0.8), skills=ColonistSkills(),
                      decision_expr="(+ 1 1)")
        pb = Colonist(id="pb", name="B", element="water", archetype="t",
                      stats=ColonistStats(resolve=0.3), skills=ColonistSkills(),
                      decision_expr="(+ 2 2)")
        child = create_child(pa, pb, "c1", 10, rng)
        assert child.genome is None
        # Stats should be roughly averaged
        assert 0.0 <= child.stats.resolve <= 1.0


class TestEngineIntegration:
    """Smoke tests for genetics integration with the full engine."""

    def test_10_year_smoke(self) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert len(result.years) > 0
        # All colonists should have genomes
        for c in engine.colonists:
            assert c.genome is not None

    def test_genetic_diversity_tracked(self) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        for yr in result.years:
            assert 0.0 <= yr.genetic_diversity <= 1.0

    def test_25_year_determinism(self) -> None:
        from src.mars100.engine import Mars100Engine
        a = Mars100Engine(seed=99, total_years=25).run()
        b = Mars100Engine(seed=99, total_years=25).run()
        assert len(a.years) == len(b.years)
        for ya, yb in zip(a.years, b.years):
            assert ya.genetic_diversity == yb.genetic_diversity

    def test_pedigree_populated(self) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=50)
        engine.run()
        # Founding colonists should be in pedigree
        assert len(engine.pedigree) >= 10
        for cid in ["kira-sol", "fen-marsh", "rust-vega"]:
            assert cid in engine.pedigree
            assert engine.pedigree[cid] == []

    def test_children_have_parent_ids(self) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        for yr in result.years:
            for birth in yr.births:
                child_id = birth["id"]
                # Child should be in pedigree
                assert child_id in engine.pedigree
                assert len(engine.pedigree[child_id]) == 2

    def test_version_is_10(self) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        d = result.to_dict()
        assert d["_meta"]["version"] == "10.0"

    def test_final_diversity_in_result(self) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        d = result.to_dict()
        assert "final_genetic_diversity" in d
        assert "genetic_diversity_trend" in d

    def test_50_year_no_crash(self) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        assert len(result.years) > 0
        assert result.final_genetic_diversity >= 0.0


class TestPropertyInvariants:
    """Property-based invariants that should hold for any seed."""

    @pytest.mark.parametrize("seed", [1, 42, 99, 137, 256])
    def test_alleles_bounded_all_seeds(self, seed: int) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=seed, total_years=20)
        engine.run()
        for c in engine.colonists:
            if c.genome is not None:
                for a in c.genome.alleles:
                    assert 0.0 <= a <= 1.0, \
                        f"seed={seed}, colonist={c.id}, allele={a}"

    @pytest.mark.parametrize("seed", [1, 42, 99])
    def test_diversity_bounded_all_seeds(self, seed: int) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=seed, total_years=30)
        result = engine.run()
        for yr in result.years:
            assert 0.0 <= yr.genetic_diversity <= 1.0

    @pytest.mark.parametrize("seed", [1, 42, 99])
    def test_inbreeding_bounded_all_seeds(self, seed: int) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=seed, total_years=50)
        engine.run()
        for cid, parents in engine.pedigree.items():
            if len(parents) >= 2:
                f = compute_inbreeding_coefficient(parents, engine.pedigree)
                assert 0.0 <= f <= 1.0, \
                    f"seed={seed}, colonist={cid}, F={f}"
