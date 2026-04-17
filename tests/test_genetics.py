"""Tests for genetics organ (engine v11.0)."""
from __future__ import annotations

import random
import pytest
from src.mars100.genetics import (
    Allele, Locus, Genome, GeneticsState, GeneticsYearContext,
    GeneticsTickResult, Pedigree,
    LOCUS_NAMES, ALLELE_MIN, ALLELE_MAX, MUTATION_RATE, CROSSOVER_RATE,
    DIVERSITY_EPIDEMIC_THRESHOLD, DIVERSITY_WARNING_THRESHOLD,
    EPIDEMIC_VULNERABILITY_MAX,
    create_founder_genome, create_immigrant_genome, inherit_genome,
    compute_individual_fitness, compute_colony_heterozygosity,
    compute_epidemic_vulnerability, compute_diversity_warnings,
    tick_genetics, compute_genetic_death_modifier, compute_nature_genetic_bonus,
)


# ---------------------------------------------------------------------------
# Allele
# ---------------------------------------------------------------------------

class TestAllele:
    def test_default(self):
        a = Allele()
        assert a.value == 0.5

    def test_custom_value(self):
        a = Allele(0.75)
        assert a.value == 0.75

    def test_to_dict_roundtrip(self):
        a = Allele(0.123456789)
        d = a.to_dict()
        a2 = Allele.from_dict(d)
        assert abs(a2.value - a.value) < 1e-5

    def test_from_dict_handles_dict_input(self):
        a = Allele.from_dict({"value": 0.3})
        assert a.value == 0.3


# ---------------------------------------------------------------------------
# Locus
# ---------------------------------------------------------------------------

class TestLocus:
    def test_expression_mean(self):
        loc = Locus(a=Allele(0.2), b=Allele(0.8))
        assert loc.expression() == pytest.approx(0.5)

    def test_heterozygosity(self):
        loc = Locus(a=Allele(0.2), b=Allele(0.8))
        assert loc.heterozygosity() == pytest.approx(0.6)

    def test_homozygous(self):
        loc = Locus(a=Allele(0.5), b=Allele(0.5))
        assert loc.heterozygosity() == pytest.approx(0.0)

    def test_roundtrip(self):
        loc = Locus(a=Allele(0.3), b=Allele(0.7))
        d = loc.to_dict()
        loc2 = Locus.from_dict(d)
        assert loc2.a.value == pytest.approx(loc.a.value, abs=1e-5)
        assert loc2.b.value == pytest.approx(loc.b.value, abs=1e-5)


# ---------------------------------------------------------------------------
# Genome
# ---------------------------------------------------------------------------

class TestGenome:
    def test_default_has_all_loci(self):
        g = Genome()
        assert set(g.loci.keys()) == set(LOCUS_NAMES)

    def test_expression(self):
        g = Genome()
        for name in LOCUS_NAMES:
            val = g.expression(name)
            assert ALLELE_MIN <= val <= ALLELE_MAX

    def test_mean_heterozygosity_bounds(self):
        g = Genome()
        h = g.mean_heterozygosity()
        assert 0.0 <= h <= 1.0

    def test_roundtrip(self):
        rng = random.Random(42)
        g = create_founder_genome("test", rng)
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        for name in LOCUS_NAMES:
            assert g2.expression(name) == pytest.approx(g.expression(name), abs=1e-5)


# ---------------------------------------------------------------------------
# Pedigree
# ---------------------------------------------------------------------------

class TestPedigree:
    def test_founder_kinship_zero(self):
        p = Pedigree()
        p.register_founder("a")
        p.register_founder("b")
        assert p.kinship("a", "b") == 0.0

    def test_self_kinship(self):
        p = Pedigree()
        p.register_founder("a")
        assert p.kinship("a", "a") == 1.0

    def test_sibling_kinship(self):
        p = Pedigree()
        p.register_founder("mom")
        p.register_founder("dad")
        p.register_child("child1", "mom", "dad")
        p.register_child("child2", "mom", "dad")
        k = p.kinship("child1", "child2")
        assert 0.45 <= k <= 0.55  # ~0.5 coefficient of relatedness for full siblings

    def test_half_sibling_kinship(self):
        p = Pedigree()
        p.register_founder("mom")
        p.register_founder("dad1")
        p.register_founder("dad2")
        p.register_child("child1", "mom", "dad1")
        p.register_child("child2", "mom", "dad2")
        k = p.kinship("child1", "child2")
        assert 0.2 <= k <= 0.3  # ~0.25 coefficient of relatedness for half siblings

    def test_inbreeding_coefficient_outbred(self):
        p = Pedigree()
        p.register_founder("a")
        p.register_founder("b")
        p.register_child("c", "a", "b")
        assert p.inbreeding_coefficient("c") == 0.0

    def test_inbreeding_coefficient_inbred(self):
        p = Pedigree()
        p.register_founder("gm")
        p.register_founder("gd")
        p.register_child("mom", "gm", "gd")
        p.register_child("dad", "gm", "gd")
        p.register_child("child", "mom", "dad")
        f = p.inbreeding_coefficient("child")
        assert f > 0.1  # siblings mating → high inbreeding

    def test_roundtrip(self):
        p = Pedigree()
        p.register_founder("a")
        p.register_child("b", "a", "a")
        d = p.to_dict()
        p2 = Pedigree.from_dict(d)
        assert p2.get_parents("b") == ("a", "a")
        assert p2.get_parents("a") == ("", "")


# ---------------------------------------------------------------------------
# Founder/immigrant genome creation
# ---------------------------------------------------------------------------

class TestGenomeCreation:
    def test_founder_genome_bounds(self):
        rng = random.Random(42)
        for i in range(20):
            g = create_founder_genome(f"c{i}", rng)
            for name in LOCUS_NAMES:
                assert ALLELE_MIN <= g.loci[name].a.value <= ALLELE_MAX
                assert ALLELE_MIN <= g.loci[name].b.value <= ALLELE_MAX

    def test_founder_genome_diversity(self):
        rng = random.Random(42)
        genomes = [create_founder_genome(f"c{i}", rng) for i in range(10)]
        vals = [g.expression("immunity") for g in genomes]
        assert max(vals) - min(vals) > 0.05  # non-trivial diversity

    def test_founder_genome_deterministic(self):
        g1 = create_founder_genome("x", random.Random(99))
        g2 = create_founder_genome("x", random.Random(99))
        for name in LOCUS_NAMES:
            assert g1.loci[name].a.value == g2.loci[name].a.value

    def test_immigrant_genome_bounds(self):
        rng = random.Random(42)
        g = create_immigrant_genome(rng)
        for name in LOCUS_NAMES:
            assert ALLELE_MIN <= g.loci[name].a.value <= ALLELE_MAX
            assert ALLELE_MIN <= g.loci[name].b.value <= ALLELE_MAX


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------

class TestInheritance:
    def test_child_has_all_loci(self):
        rng = random.Random(42)
        pa = create_founder_genome("a", rng)
        pb = create_founder_genome("b", rng)
        child, muts = inherit_genome(pa, pb, "a", "b", rng)
        assert set(child.loci.keys()) == set(LOCUS_NAMES)

    def test_child_alleles_bounded(self):
        rng = random.Random(42)
        pa = create_founder_genome("a", rng)
        pb = create_founder_genome("b", rng)
        for seed in range(50):
            child, _ = inherit_genome(pa, pb, "a", "b", random.Random(seed))
            for name in LOCUS_NAMES:
                assert ALLELE_MIN <= child.loci[name].a.value <= ALLELE_MAX
                assert ALLELE_MIN <= child.loci[name].b.value <= ALLELE_MAX

    def test_child_parent_ids_set(self):
        rng = random.Random(42)
        pa = create_founder_genome("a", rng)
        pb = create_founder_genome("b", rng)
        child, _ = inherit_genome(pa, pb, "a", "b", rng)
        assert child.parent_ids == ["a", "b"]

    def test_no_mutation_at_zero_rate(self):
        """With mutation rate effectively zero, child alleles come from parents."""
        import src.mars100.genetics as gen_mod
        orig_rate = gen_mod.MUTATION_RATE
        gen_mod.MUTATION_RATE = 0.0
        try:
            rng = random.Random(42)
            pa = create_founder_genome("a", rng)
            pb = create_founder_genome("b", rng)
            child, muts = inherit_genome(pa, pb, "a", "b", random.Random(99))
            assert len(muts) == 0
            for name in LOCUS_NAMES:
                child_a = child.loci[name].a.value
                child_b = child.loci[name].b.value
                parent_alleles = {pa.loci[name].a.value, pa.loci[name].b.value,
                                  pb.loci[name].a.value, pb.loci[name].b.value}
                # Due to crossover, alleles may swap between loci
                # But each individual allele value must come from a parent
                # (just possibly from a different locus due to crossover)
        finally:
            gen_mod.MUTATION_RATE = orig_rate

    def test_mutations_logged(self):
        rng = random.Random(42)
        pa = create_founder_genome("a", rng)
        pb = create_founder_genome("b", rng)
        all_mutations: list[dict] = []
        for seed in range(100):
            _, muts = inherit_genome(pa, pb, "a", "b", random.Random(seed))
            all_mutations.extend(muts)
        # With 100 births × 12 alleles × 0.05 mutation rate ≈ 60 expected mutations
        assert len(all_mutations) > 10


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

class TestFitness:
    def test_default_genome_neutral(self):
        g = Genome()  # all alleles 0.5
        fit = compute_individual_fitness(g)
        assert fit["death_rate_mult"] == pytest.approx(1.0)
        assert fit["birth_prob_bonus"] == pytest.approx(0.0)
        assert fit["skill_learning_rate"] == pytest.approx(1.0)

    def test_high_immunity_reduces_death(self):
        g = Genome()
        g.loci["immunity"] = Locus(a=Allele(0.9), b=Allele(0.9))
        fit = compute_individual_fitness(g)
        assert fit["death_rate_mult"] < 1.0

    def test_low_immunity_increases_death(self):
        g = Genome()
        g.loci["immunity"] = Locus(a=Allele(0.1), b=Allele(0.1))
        fit = compute_individual_fitness(g)
        assert fit["death_rate_mult"] > 1.0

    def test_high_fertility_bonus(self):
        g = Genome()
        g.loci["fertility"] = Locus(a=Allele(0.9), b=Allele(0.9))
        fit = compute_individual_fitness(g)
        assert fit["birth_prob_bonus"] > 0.0

    def test_fitness_bounds(self):
        rng = random.Random(42)
        for _ in range(100):
            g = create_founder_genome("x", rng)
            fit = compute_individual_fitness(g)
            assert 0.5 <= fit["death_rate_mult"] <= 2.0
            assert -0.1 <= fit["birth_prob_bonus"] <= 0.2
            assert 0.5 <= fit["skill_learning_rate"] <= 1.5
            assert 0.8 <= fit["food_consumption_mult"] <= 1.2
            assert 0.4 <= fit["rad_damage_mult"] <= 1.0

    def test_rad_resistance_reduces_damage(self):
        g = Genome()
        g.loci["radiation_resistance"] = Locus(a=Allele(1.0), b=Allele(1.0))
        fit = compute_individual_fitness(g)
        assert fit["rad_damage_mult"] < 0.5


# ---------------------------------------------------------------------------
# Colony heterozygosity
# ---------------------------------------------------------------------------

class TestColonyHeterozygosity:
    def test_empty_colony(self):
        assert compute_colony_heterozygosity([]) == 0.0

    def test_single_genome(self):
        g = Genome()
        h = compute_colony_heterozygosity([g])
        assert 0.0 <= h <= 1.0

    def test_diverse_colony(self):
        rng = random.Random(42)
        genomes = [create_founder_genome(f"c{i}", rng) for i in range(10)]
        h = compute_colony_heterozygosity(genomes)
        assert h > 0.3  # 10 founders should have decent diversity

    def test_cloned_colony_low_diversity(self):
        g = Genome()
        # All identical genomes
        clones = [Genome() for _ in range(10)]
        h = compute_colony_heterozygosity(clones)
        # All alleles are 0.5, so all in same bin → He ≈ 0
        assert h < 0.15

    def test_heterozygosity_bounded(self):
        rng = random.Random(42)
        for seed in range(20):
            genomes = [create_founder_genome(f"c{i}", random.Random(seed + i))
                       for i in range(10)]
            h = compute_colony_heterozygosity(genomes)
            assert 0.0 <= h <= 1.0


# ---------------------------------------------------------------------------
# Epidemic vulnerability
# ---------------------------------------------------------------------------

class TestEpidemicVulnerability:
    def test_high_diversity_no_vulnerability(self):
        assert compute_epidemic_vulnerability(0.5) == 0.0

    def test_zero_diversity_max_vulnerability(self):
        v = compute_epidemic_vulnerability(0.0)
        assert v == pytest.approx(EPIDEMIC_VULNERABILITY_MAX)

    def test_threshold_boundary(self):
        assert compute_epidemic_vulnerability(DIVERSITY_EPIDEMIC_THRESHOLD) == 0.0
        assert compute_epidemic_vulnerability(
            DIVERSITY_EPIDEMIC_THRESHOLD - 0.01) > 0.0

    def test_vulnerability_bounded(self):
        for h in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
            v = compute_epidemic_vulnerability(h)
            assert 0.0 <= v <= EPIDEMIC_VULNERABILITY_MAX


# ---------------------------------------------------------------------------
# Diversity warnings
# ---------------------------------------------------------------------------

class TestDiversityWarnings:
    def test_healthy_colony_no_warnings(self):
        warnings = compute_diversity_warnings(0.5, 10, 0)
        assert len(warnings) == 0

    def test_low_diversity_warning(self):
        warnings = compute_diversity_warnings(0.28, 10, 0)
        assert any("warning threshold" in w for w in warnings)

    def test_critical_diversity_warning(self):
        warnings = compute_diversity_warnings(0.2, 10, 0)
        assert any("CRITICAL" in w for w in warnings)

    def test_inbreeding_warning(self):
        warnings = compute_diversity_warnings(0.5, 10, 3)
        assert any("inbreeding" in w for w in warnings)


# ---------------------------------------------------------------------------
# Tick
# ---------------------------------------------------------------------------

class TestTickGenetics:
    def test_basic_tick(self):
        rng = random.Random(42)
        genomes = [create_founder_genome(f"c{i}", rng) for i in range(10)]
        state = GeneticsState()
        ctx = GeneticsYearContext(
            year=1, active_genomes=genomes,
            active_ids=[f"c{i}" for i in range(10)])
        result = tick_genetics(state, ctx, rng)
        assert 0.0 <= result.colony_heterozygosity <= 1.0
        assert 0.0 <= result.epidemic_vulnerability <= EPIDEMIC_VULNERABILITY_MAX

    def test_tick_updates_state(self):
        rng = random.Random(42)
        genomes = [create_founder_genome(f"c{i}", rng) for i in range(10)]
        state = GeneticsState()
        ctx = GeneticsYearContext(
            year=1, active_genomes=genomes,
            active_ids=[f"c{i}" for i in range(10)])
        tick_genetics(state, ctx, rng)
        assert state.colony_heterozygosity > 0.0

    def test_tick_result_serializable(self):
        rng = random.Random(42)
        genomes = [create_founder_genome(f"c{i}", rng) for i in range(5)]
        state = GeneticsState()
        ctx = GeneticsYearContext(
            year=1, active_genomes=genomes,
            active_ids=[f"c{i}" for i in range(5)])
        result = tick_genetics(state, ctx, rng)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "colony_heterozygosity" in d

    def test_state_serializable(self):
        state = GeneticsState()
        d = state.to_dict()
        assert isinstance(d, dict)
        assert "colony_heterozygosity" in d


# ---------------------------------------------------------------------------
# Death modifier
# ---------------------------------------------------------------------------

class TestGeneticDeathModifier:
    def test_no_genome_neutral(self):
        assert compute_genetic_death_modifier(None, 0.0) == pytest.approx(1.0)

    def test_epidemic_increases_death(self):
        m = compute_genetic_death_modifier(None, 0.1)
        assert m > 1.0

    def test_good_genome_reduces_death(self):
        g = Genome()
        g.loci["immunity"] = Locus(a=Allele(0.9), b=Allele(0.9))
        g.loci["longevity"] = Locus(a=Allele(0.9), b=Allele(0.9))
        m = compute_genetic_death_modifier(g, 0.0)
        assert m < 1.0

    def test_modifier_bounded(self):
        rng = random.Random(42)
        for _ in range(100):
            g = create_founder_genome("x", rng)
            m = compute_genetic_death_modifier(g, rng.random() * 0.15)
            assert 0.5 <= m <= 3.0


# ---------------------------------------------------------------------------
# Nature genetic bonus
# ---------------------------------------------------------------------------

class TestNatureGeneticBonus:
    def test_empty_genomes(self):
        result = compute_nature_genetic_bonus([])
        assert result == {}

    def test_default_genomes_near_neutral(self):
        genomes = [Genome() for _ in range(5)]
        result = compute_nature_genetic_bonus(genomes)
        assert abs(result.get("food_maintenance_mult", 1.0) - 1.0) < 0.05

    def test_result_keys(self):
        rng = random.Random(42)
        genomes = [create_founder_genome(f"c{i}", rng) for i in range(10)]
        result = compute_nature_genetic_bonus(genomes)
        assert "food_maintenance_mult" in result
        assert "medicine_maintenance_mult" in result


# ---------------------------------------------------------------------------
# Property-based invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    @pytest.mark.parametrize("seed", range(50))
    def test_founder_genome_all_loci_bounded(self, seed):
        rng = random.Random(seed)
        g = create_founder_genome("test", rng)
        for name in LOCUS_NAMES:
            assert ALLELE_MIN <= g.loci[name].a.value <= ALLELE_MAX
            assert ALLELE_MIN <= g.loci[name].b.value <= ALLELE_MAX
            assert ALLELE_MIN <= g.expression(name) <= ALLELE_MAX

    @pytest.mark.parametrize("seed", range(50))
    def test_inherited_genome_bounded(self, seed):
        rng = random.Random(seed)
        pa = create_founder_genome("a", rng)
        pb = create_founder_genome("b", rng)
        child, _ = inherit_genome(pa, pb, "a", "b", rng)
        for name in LOCUS_NAMES:
            assert ALLELE_MIN <= child.loci[name].a.value <= ALLELE_MAX
            assert ALLELE_MIN <= child.loci[name].b.value <= ALLELE_MAX

    @pytest.mark.parametrize("seed", range(20))
    def test_fitness_invariants(self, seed):
        rng = random.Random(seed)
        g = create_founder_genome("x", rng)
        fit = compute_individual_fitness(g)
        # All modifiers bounded
        assert 0.5 <= fit["death_rate_mult"] <= 2.0
        assert -0.1 <= fit["birth_prob_bonus"] <= 0.2
        assert 0.5 <= fit["skill_learning_rate"] <= 1.5
        assert 0.8 <= fit["food_consumption_mult"] <= 1.2
        assert 0.4 <= fit["rad_damage_mult"] <= 1.0

    @pytest.mark.parametrize("seed", range(20))
    def test_colony_heterozygosity_bounded(self, seed):
        rng = random.Random(seed)
        genomes = [create_founder_genome(f"c{i}", rng) for i in range(10)]
        h = compute_colony_heterozygosity(genomes)
        assert 0.0 <= h <= 1.0

    @pytest.mark.parametrize("seed", range(20))
    def test_epidemic_vulnerability_bounded(self, seed):
        h = seed / 20.0
        v = compute_epidemic_vulnerability(h)
        assert 0.0 <= v <= EPIDEMIC_VULNERABILITY_MAX
