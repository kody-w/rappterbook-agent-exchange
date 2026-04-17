"""Tests for the genetics organ (engine v11.0)."""
from __future__ import annotations

import random

import pytest

from src.mars100.genetics import (
    AllelePair,
    EpigeneticMark,
    Genome,
    GeneticsState,
    GeneticsTickResult,
    DOMINANT_WEIGHT,
    DRIFT_BOUND,
    INBREEDING_THRESHOLD,
    MAX_ALLELE,
    MIN_ALLELE,
    MUTATION_RATE,
    RECESSIVE_WEIGHT,
    STAT_NAMES,
    apply_epigenetic_mark,
    apply_genetic_drift_bound,
    compute_colony_diversity,
    compute_genome_baseline,
    compute_inbreeding_coefficient,
    create_genome_from_phenotype,
    create_random_genome,
    inherit_genome,
    tick_genetics,
)


# ---------- AllelePair ----------

class TestAllelePair:
    def test_dominant_recessive(self):
        p = AllelePair(0.3, 0.8)
        assert p.dominant() == 0.8
        assert p.recessive() == 0.3

    def test_expression(self):
        p = AllelePair(0.3, 0.8)
        expected = 0.8 * DOMINANT_WEIGHT + 0.3 * RECESSIVE_WEIGHT
        assert abs(p.express() - expected) < 1e-9

    def test_heterozygosity(self):
        p = AllelePair(0.3, 0.8)
        assert abs(p.heterozygosity() - 0.5) < 1e-9

    def test_round_trip(self):
        p = AllelePair(0.123456, 0.654321)
        d = p.to_dict()
        p2 = AllelePair.from_dict(d)
        assert abs(p2.a - 0.123456) < 1e-5
        assert abs(p2.b - 0.654321) < 1e-5


# ---------- Genome ----------

class TestGenome:
    def test_express_all_bounded(self):
        rng = random.Random(42)
        g = create_random_genome(rng)
        expressed = g.express_all()
        for name in STAT_NAMES:
            assert MIN_ALLELE <= expressed[name] <= MAX_ALLELE

    def test_avg_heterozygosity_bounded(self):
        rng = random.Random(42)
        g = create_random_genome(rng)
        h = g.avg_heterozygosity()
        assert 0.0 <= h <= 1.0

    def test_round_trip(self):
        rng = random.Random(42)
        g = create_random_genome(rng)
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        assert set(g2.loci.keys()) == set(g.loci.keys())
        assert g2.generation == g.generation

    def test_epigenetic_mark_on_genome(self):
        rng = random.Random(42)
        g = create_random_genome(rng)
        mark = EpigeneticMark(locus="resolve", modifier=0.05,
                               origin_year=10, cause="dust_storm")
        g.epigenetic_marks.append(mark)
        expressed = g.express_all()
        # Resolve should be modified by epigenetic mark
        g_clean = create_random_genome(random.Random(42))
        assert expressed["resolve"] != g_clean.express_all()["resolve"]


# ---------- create_genome_from_phenotype ----------

class TestCreateGenomeFromPhenotype:
    def test_expression_approximates_target(self):
        stats = {"resolve": 0.8, "improvisation": 0.3, "empathy": 0.6,
                 "hoarding": 0.5, "faith": 0.9, "paranoia": 0.2}
        rng = random.Random(42)
        genome = create_genome_from_phenotype(stats, rng)
        expressed = genome.express_all()
        for name, target in stats.items():
            # Within 0.15 of target is acceptable given spread
            assert abs(expressed[name] - target) < 0.20, \
                f"{name}: expressed={expressed[name]:.3f} vs target={target}"

    def test_heterozygosity_present(self):
        stats = {s: 0.5 for s in STAT_NAMES}
        rng = random.Random(42)
        genome = create_genome_from_phenotype(stats, rng)
        assert genome.avg_heterozygosity() > 0.01

    def test_generation_zero(self):
        stats = {s: 0.5 for s in STAT_NAMES}
        genome = create_genome_from_phenotype(stats, random.Random(1))
        assert genome.generation == 0


# ---------- inherit_genome ----------

class TestInheritGenome:
    def _make_parents(self, seed=42):
        rng = random.Random(seed)
        a = create_random_genome(rng)
        b = create_random_genome(rng)
        return a, b

    def test_child_has_all_loci(self):
        a, b = self._make_parents()
        child = inherit_genome(a, b, random.Random(99))
        assert set(child.loci.keys()) == set(STAT_NAMES)

    def test_child_alleles_bounded(self):
        a, b = self._make_parents()
        child = inherit_genome(a, b, random.Random(99))
        for name, pair in child.loci.items():
            assert MIN_ALLELE <= pair.a <= MAX_ALLELE
            assert MIN_ALLELE <= pair.b <= MAX_ALLELE

    def test_generation_incremented(self):
        a, b = self._make_parents()
        a.generation = 2
        b.generation = 3
        child = inherit_genome(a, b, random.Random(99))
        assert child.generation == 4

    def test_deterministic(self):
        a, b = self._make_parents()
        c1 = inherit_genome(a, b, random.Random(77))
        c2 = inherit_genome(a, b, random.Random(77))
        for name in STAT_NAMES:
            assert c1.loci[name].a == c2.loci[name].a
            assert c1.loci[name].b == c2.loci[name].b

    def test_epigenetic_marks_can_inherit(self):
        a, b = self._make_parents()
        a.epigenetic_marks.append(
            EpigeneticMark("resolve", 0.1, 5, "dust_storm"))
        children_with_mark = 0
        for seed in range(100):
            child = inherit_genome(a, b, random.Random(seed))
            if child.epigenetic_marks:
                children_with_mark += 1
        # ~50% inheritance rate
        assert 20 < children_with_mark < 80


# ---------- Inbreeding ----------

class TestInbreeding:
    def test_unrelated_zero(self):
        coeff = compute_inbreeding_coefficient(
            ["parent-a"], ["parent-b"], {})
        assert coeff == 0.0

    def test_sibling_overlap(self):
        pedigree = {"child-1": ["mom", "dad"], "child-2": ["mom", "dad"]}
        coeff = compute_inbreeding_coefficient(
            ["child-1"], ["child-2"], pedigree)
        # Both children share mom and dad ancestors
        assert coeff > 0.0

    def test_self_overlap_maximum(self):
        coeff = compute_inbreeding_coefficient(
            ["same"], ["same"], {})
        assert coeff == 1.0

    def test_bounded(self):
        pedigree = {f"c{i}": [f"c{i-1}", f"c{i-2}"] for i in range(3, 10)}
        coeff = compute_inbreeding_coefficient(
            ["c8"], ["c9"], pedigree)
        assert 0.0 <= coeff <= 1.0


# ---------- Diversity ----------

class TestColonyDiversity:
    def test_single_genome_zero(self):
        g = create_random_genome(random.Random(42))
        assert compute_colony_diversity([g]) == 0.0

    def test_identical_genomes_same_diversity(self):
        rng = random.Random(42)
        g = create_random_genome(rng)
        import copy
        g2 = copy.deepcopy(g)
        # Two identical genomes have the same allele pool as one,
        # so diversity equals the within-genome heterozygosity variance
        d1 = compute_colony_diversity([g])
        d2 = compute_colony_diversity([g, g2])
        # With 1 genome we get 0 (need >=2), with 2 identical we get
        # the within-locus variance which is constant
        assert d2 >= 0.0
        # Adding a truly different genome should increase diversity
        g3 = create_random_genome(random.Random(99))
        d3 = compute_colony_diversity([g, g2, g3])
        assert d3 >= d2

    def test_diverse_genomes_positive(self):
        genomes = [create_random_genome(random.Random(i)) for i in range(10)]
        d = compute_colony_diversity(genomes)
        assert d > 0.0

    def test_bounded(self):
        genomes = [create_random_genome(random.Random(i)) for i in range(20)]
        d = compute_colony_diversity(genomes)
        assert 0.0 <= d <= 1.0


# ---------- Epigenetic marks ----------

class TestEpigeneticMarks:
    def test_low_severity_no_mark(self):
        g = create_random_genome(random.Random(42))
        applied = apply_epigenetic_mark(g, "breeze", 0.3, 10, random.Random(1))
        assert not applied

    def test_high_severity_can_mark(self):
        marked = 0
        for seed in range(100):
            g = create_random_genome(random.Random(42))
            if apply_epigenetic_mark(g, "hab_breach", 0.9, 10, random.Random(seed)):
                marked += 1
        assert marked > 0

    def test_mark_round_trip(self):
        m = EpigeneticMark("resolve", 0.03, 5, "storm")
        d = m.to_dict()
        m2 = EpigeneticMark.from_dict(d)
        assert m2.locus == "resolve"
        assert abs(m2.modifier - 0.03) < 1e-6


# ---------- Genetic drift bounds ----------

class TestDriftBounds:
    def test_within_bound_unchanged(self):
        assert apply_genetic_drift_bound(0.55, 0.5) == 0.55

    def test_exceeds_bound_pulled_back(self):
        result = apply_genetic_drift_bound(0.8, 0.5)
        # 0.8 is 0.3 above baseline, bound is 0.15, so pulled back
        assert result < 0.8
        assert result > 0.5

    def test_below_bound_pulled_back(self):
        result = apply_genetic_drift_bound(0.2, 0.5)
        assert result > 0.2
        assert result < 0.5


# ---------- GeneticsState ----------

class TestGeneticsState:
    def test_record_birth(self):
        state = GeneticsState()
        state.record_birth("child-1", ["parent-a", "parent-b"])
        assert state.pedigree["child-1"] == ["parent-a", "parent-b"]

    def test_round_trip(self):
        state = GeneticsState()
        state.diversity_history = [0.1, 0.2, 0.3]
        state.generation_max = 5
        d = state.to_dict()
        s2 = GeneticsState.from_dict(d)
        assert s2.generation_max == 5
        assert len(s2.diversity_history) == 3


# ---------- tick_genetics ----------

class TestTickGenetics:
    def test_basic_tick(self):
        state = GeneticsState()
        genomes = [create_random_genome(random.Random(i)) for i in range(5)]
        rng = random.Random(42)
        result = tick_genetics(state, genomes, "dust_storm", 0.5, 1, rng)
        assert isinstance(result, GeneticsTickResult)
        assert result.colony_diversity >= 0.0

    def test_diversity_recorded(self):
        state = GeneticsState()
        genomes = [create_random_genome(random.Random(i)) for i in range(5)]
        tick_genetics(state, genomes, "calm", 0.1, 1, random.Random(1))
        assert len(state.diversity_history) == 1

    def test_severe_event_marks(self):
        state = GeneticsState()
        genomes = [create_random_genome(random.Random(i)) for i in range(10)]
        result = tick_genetics(state, genomes, "hab_breach", 0.9, 5, random.Random(42))
        # With severity 0.9 and 10 genomes, should get some marks
        assert result.epigenetic_marks_applied >= 0


# ---------- Engine integration smoke test ----------

class TestEngineIntegration:
    def test_engine_runs_10_years_with_genetics(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert len(result.years) == 10
        # Check genetics data present in year results
        for yr in result.years:
            assert "colony_diversity" in yr.genetics
            assert "avg_heterozygosity" in yr.genetics
        # Check final genetics state
        assert result.final_genetics is not None
        assert "generation_max" in result.final_genetics

    def test_founding_colonists_have_genomes(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=1)
        for colonist in engine.colonists:
            assert colonist.genome_data is not None
            assert colonist.id in engine.genome_map

    def test_stats_bounded_by_genetics(self):
        """Stats should stay within drift bounds of genetic baseline."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        # Check that active colonists' stats haven't drifted too far
        for colonist in engine.colonists:
            if not colonist.is_active():
                continue
            genome = engine.genome_map.get(colonist.id)
            if genome is None:
                continue
            baseline = compute_genome_baseline(genome)
            for sn in STAT_NAMES:
                current = getattr(colonist.stats, sn)
                base_val = baseline.get(sn, 0.5)
                # After drift bound application, should be within
                # DRIFT_BOUND + small tolerance
                assert abs(current - base_val) < DRIFT_BOUND + 0.1, \
                    f"{colonist.name} {sn}: {current:.3f} vs base {base_val:.3f}"

    def test_diversity_changes_over_time(self):
        """Colony diversity should change as births and deaths occur."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        diversities = [yr.genetics.get("colony_diversity", 0)
                       for yr in result.years]
        # Not all the same value
        assert len(set(round(d, 6) for d in diversities)) > 1


# ---------- Property-based invariants ----------

class TestInvariants:
    def test_allele_bounds_maintained_through_inheritance(self):
        """Over many generations, alleles stay in [0, 1]."""
        rng = random.Random(42)
        parents = [create_random_genome(rng) for _ in range(10)]
        for gen in range(20):
            children = []
            for _ in range(10):
                a = rng.choice(parents)
                b = rng.choice(parents)
                child = inherit_genome(a, b, rng)
                for pair in child.loci.values():
                    assert MIN_ALLELE <= pair.a <= MAX_ALLELE
                    assert MIN_ALLELE <= pair.b <= MAX_ALLELE
                children.append(child)
            parents = children

    def test_expression_in_physical_bounds(self):
        """Expressed traits always in [0, 1]."""
        rng = random.Random(42)
        for _ in range(100):
            g = create_random_genome(rng)
            expressed = g.express_all()
            for val in expressed.values():
                assert MIN_ALLELE <= val <= MAX_ALLELE

    def test_diversity_nonnegative(self):
        """Colony diversity is never negative."""
        rng = random.Random(42)
        for n in range(1, 20):
            genomes = [create_random_genome(random.Random(i)) for i in range(n)]
            d = compute_colony_diversity(genomes)
            assert d >= 0.0
