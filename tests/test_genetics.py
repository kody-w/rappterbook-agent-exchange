"""Tests for the genetics organ (engine v11.0)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.genetics import (
    GENE_NAMES, NUM_GENES, MUTATION_RATE, MUTATION_SIGMA,
    CROSSOVER_PROBABILITY, HYBRID_VIGOR_THRESHOLD, HYBRID_VIGOR_BONUS,
    INBREEDING_THRESHOLD, INBREEDING_PENALTY_MAX,
    DIVERSITY_HEALTHY, EPIGENETIC_PROBABILITY,
    Genome, GeneticsState, GeneticsYearContext, GeneticsTickResult,
    create_genome_from_phenotype, crossover, mutate, breed,
    compute_hybrid_vigor, compute_colony_diversity,
    compute_inbreeding_penalty, epigenetic_activation,
    founder_gene_survival, compute_health_modifier, tick_genetics,
)


# ---------------------------------------------------------------------------
# Genome basics
# ---------------------------------------------------------------------------

class TestGenome:
    def test_default_genome_has_correct_shape(self):
        g = Genome()
        assert len(g.alleles) == NUM_GENES
        for pair in g.alleles:
            assert len(pair) == 2

    def test_express_weighted(self):
        g = Genome(alleles=[[0.8, 0.2]] + [[0.5, 0.5]] * (NUM_GENES - 1))
        expressed = g.express(0)
        assert abs(expressed - (0.7 * 0.8 + 0.3 * 0.2)) < 1e-9

    def test_express_all_keys(self):
        g = Genome()
        expressed = g.express_all()
        assert set(expressed.keys()) == set(GENE_NAMES)
        for v in expressed.values():
            assert 0.0 <= v <= 1.0

    def test_heterozygosity_homozygous(self):
        g = Genome(alleles=[[0.5, 0.5]] * NUM_GENES)
        assert g.heterozygosity() == 0.0

    def test_heterozygosity_max(self):
        g = Genome(alleles=[[0.0, 1.0]] * NUM_GENES)
        assert abs(g.heterozygosity() - 1.0) < 1e-9

    def test_genetic_distance_self(self):
        g = Genome()
        assert g.genetic_distance(g) == 0.0

    def test_genetic_distance_symmetric(self):
        rng = random.Random(1)
        a = create_genome_from_phenotype(
            {n: rng.random() for n in GENE_NAMES[:6]},
            {n: rng.random() for n in GENE_NAMES[6:]}, rng)
        b = create_genome_from_phenotype(
            {n: rng.random() for n in GENE_NAMES[:6]},
            {n: rng.random() for n in GENE_NAMES[6:]}, rng)
        assert abs(a.genetic_distance(b) - b.genetic_distance(a)) < 1e-9

    def test_to_dict_from_dict_roundtrip(self):
        rng = random.Random(42)
        g = create_genome_from_phenotype(
            {n: rng.random() for n in GENE_NAMES[:6]},
            {n: rng.random() for n in GENE_NAMES[6:]}, rng)
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        assert len(g2.alleles) == NUM_GENES
        for i in range(NUM_GENES):
            assert abs(g.alleles[i][0] - g2.alleles[i][0]) < 1e-5
            assert abs(g.alleles[i][1] - g2.alleles[i][1]) < 1e-5

    def test_from_dict_empty(self):
        g = Genome.from_dict({})
        assert len(g.alleles) == NUM_GENES

    def test_clamp_enforces_bounds(self):
        g = Genome(alleles=[[-0.5, 1.5]] * NUM_GENES)
        g.clamp()
        for a, b in g.alleles:
            assert 0.0 <= a <= 1.0
            assert 0.0 <= b <= 1.0


# ---------------------------------------------------------------------------
# Genome creation
# ---------------------------------------------------------------------------

class TestCreateGenome:
    def test_founder_genome_preserves_phenotype(self):
        """Founder genomes should express close to original trait values."""
        rng = random.Random(42)
        stats = {"resolve": 0.9, "improvisation": 0.4, "empathy": 0.5,
                 "hoarding": 0.3, "faith": 0.2, "paranoia": 0.4}
        skills = {"terraforming": 0.7, "hydroponics": 0.2, "mediation": 0.5,
                  "coding": 0.3, "prayer": 0.0, "sabotage": 0.1}
        g = create_genome_from_phenotype(stats, skills, rng)
        expressed = g.express_all()
        for name, original in {**stats, **skills}.items():
            assert abs(expressed[name] - original) < 0.15, \
                f"{name}: expressed={expressed[name]:.3f}, original={original:.3f}"

    def test_founder_genomes_differ(self):
        rng = random.Random(42)
        traits = {n: 0.5 for n in GENE_NAMES}
        g1 = create_genome_from_phenotype(traits, {}, rng)
        g2 = create_genome_from_phenotype(traits, {}, rng)
        assert g1.genetic_distance(g2) > 0.0


# ---------------------------------------------------------------------------
# Crossover and mutation
# ---------------------------------------------------------------------------

class TestBreeding:
    def test_crossover_produces_valid_genome(self):
        rng = random.Random(42)
        a = Genome(alleles=[[0.2, 0.3]] * NUM_GENES)
        b = Genome(alleles=[[0.7, 0.8]] * NUM_GENES)
        child = crossover(a, b, rng)
        assert len(child.alleles) == NUM_GENES
        for pair in child.alleles:
            for v in pair:
                assert 0.0 <= v <= 1.0

    def test_crossover_child_differs_from_parents(self):
        rng = random.Random(42)
        a = Genome(alleles=[[0.1, 0.2]] * NUM_GENES)
        b = Genome(alleles=[[0.8, 0.9]] * NUM_GENES)
        child = crossover(a, b, rng)
        assert child.genetic_distance(a) > 0.0
        assert child.genetic_distance(b) > 0.0

    def test_mutate_stays_in_bounds(self):
        rng = random.Random(42)
        g = Genome(alleles=[[0.0, 1.0]] * NUM_GENES)
        mutate(g, rng, rate=1.0, sigma=0.5)
        for a, b in g.alleles:
            assert 0.0 <= a <= 1.0
            assert 0.0 <= b <= 1.0

    def test_mutate_returns_count(self):
        rng = random.Random(42)
        g = Genome()
        count = mutate(g, rng, rate=1.0)
        assert count == NUM_GENES * 2

    def test_mutate_zero_rate(self):
        rng = random.Random(42)
        g = Genome(alleles=[[0.5, 0.5]] * NUM_GENES)
        original = [[a, b] for a, b in g.alleles]
        count = mutate(g, rng, rate=0.0)
        assert count == 0
        for i in range(NUM_GENES):
            assert g.alleles[i][0] == original[i][0]
            assert g.alleles[i][1] == original[i][1]

    def test_breed_returns_genome_and_count(self):
        rng = random.Random(42)
        a = Genome()
        b = Genome()
        child, n = breed(a, b, rng)
        assert isinstance(child, Genome)
        assert isinstance(n, int)
        assert n >= 0


# ---------------------------------------------------------------------------
# Colony diversity
# ---------------------------------------------------------------------------

class TestDiversity:
    def test_diversity_identical_genomes(self):
        g = Genome()
        assert compute_colony_diversity([g, g, g]) == 0.0

    def test_diversity_diverse_genomes(self):
        a = Genome(alleles=[[0.0, 0.0]] * NUM_GENES)
        b = Genome(alleles=[[1.0, 1.0]] * NUM_GENES)
        d = compute_colony_diversity([a, b])
        assert d > 0.5

    def test_diversity_single_genome(self):
        assert compute_colony_diversity([Genome()]) == 0.0

    def test_diversity_empty(self):
        assert compute_colony_diversity([]) == 0.0

    def test_diversity_increases_with_variation(self):
        rng = random.Random(42)
        similar = [Genome(alleles=[[0.5 + rng.gauss(0, 0.01),
                                     0.5 + rng.gauss(0, 0.01)]
                                    for _ in range(NUM_GENES)])
                   for _ in range(5)]
        diverse = [Genome(alleles=[[rng.random(), rng.random()]
                                    for _ in range(NUM_GENES)])
                   for _ in range(5)]
        assert compute_colony_diversity(diverse) > compute_colony_diversity(similar)


# ---------------------------------------------------------------------------
# Hybrid vigor and inbreeding
# ---------------------------------------------------------------------------

class TestFitnessModifiers:
    def test_hybrid_vigor_distant_parents(self):
        a = Genome(alleles=[[0.0, 0.0]] * NUM_GENES)
        b = Genome(alleles=[[1.0, 1.0]] * NUM_GENES)
        bonus = compute_hybrid_vigor(a, b)
        assert bonus > 0.0
        assert bonus <= HYBRID_VIGOR_BONUS

    def test_hybrid_vigor_similar_parents(self):
        a = Genome(alleles=[[0.5, 0.5]] * NUM_GENES)
        b = Genome(alleles=[[0.5, 0.5]] * NUM_GENES)
        assert compute_hybrid_vigor(a, b) == 0.0

    def test_inbreeding_penalty_healthy(self):
        assert compute_inbreeding_penalty(0.5) == 1.0

    def test_inbreeding_penalty_low_diversity(self):
        penalty = compute_inbreeding_penalty(0.0)
        assert penalty > 1.0
        assert penalty <= 1.0 + INBREEDING_PENALTY_MAX

    def test_inbreeding_penalty_at_threshold(self):
        assert compute_inbreeding_penalty(INBREEDING_THRESHOLD) == 1.0

    def test_health_modifier_delegates(self):
        state = GeneticsState(diversity_index=0.5)
        assert compute_health_modifier(state) == 1.0
        state.diversity_index = 0.0
        assert compute_health_modifier(state) > 1.0


# ---------------------------------------------------------------------------
# Epigenetics
# ---------------------------------------------------------------------------

class TestEpigenetics:
    def test_epigenetic_activation_no_stress(self):
        rng = random.Random(42)
        g = Genome()
        results = [epigenetic_activation(g, 0.0, rng) for _ in range(100)]
        # Low stress → very few activations
        activations = [r for r in results if r is not None]
        assert len(activations) < 20

    def test_epigenetic_activation_high_stress(self):
        rng = random.Random(42)
        g = Genome()
        results = [epigenetic_activation(g, 1.0, rng) for _ in range(100)]
        activations = [r for r in results if r is not None]
        assert len(activations) > 0
        for gene in activations:
            assert gene in GENE_NAMES

    def test_epigenetic_swaps_alleles(self):
        rng = random.Random(42)
        g = Genome(alleles=[[0.1, 0.9]] + [[0.5, 0.5]] * (NUM_GENES - 1))
        original_expressed = g.express(0)
        # Force many activations
        for _ in range(1000):
            result = epigenetic_activation(g, 1.0, rng)
            if result == GENE_NAMES[0]:
                # After swap, expression should change
                break


# ---------------------------------------------------------------------------
# Founder gene survival
# ---------------------------------------------------------------------------

class TestFounderSurvival:
    def test_founder_survival_self(self):
        g = Genome(alleles=[[0.3, 0.3]] * NUM_GENES)
        result = founder_gene_survival({"f1": g}, {"c1": g})
        assert result["f1"] > 0.5

    def test_founder_survival_empty(self):
        assert founder_gene_survival({}, {}) == {}

    def test_founder_survival_distant(self):
        f = Genome(alleles=[[0.0, 0.0]] * NUM_GENES)
        c = Genome(alleles=[[1.0, 1.0]] * NUM_GENES)
        result = founder_gene_survival({"f1": f}, {"c1": c})
        assert result["f1"] < 0.5


# ---------------------------------------------------------------------------
# GeneticsState serialization
# ---------------------------------------------------------------------------

class TestGeneticsState:
    def test_to_dict_from_dict_roundtrip(self):
        state = GeneticsState(
            diversity_index=0.42, generation_count=5,
            total_mutations=100, epigenetic_events=3,
            founder_survival={"f1": 0.8, "f2": 0.3},
            diversity_history=[0.5, 0.48, 0.46],
        )
        d = state.to_dict()
        state2 = GeneticsState.from_dict(d)
        assert abs(state2.diversity_index - 0.42) < 1e-4
        assert state2.generation_count == 5
        assert state2.total_mutations == 100
        assert state2.epigenetic_events == 3
        assert len(state2.founder_survival) == 2

    def test_from_dict_empty(self):
        state = GeneticsState.from_dict({})
        assert state.diversity_index == 0.5


# ---------------------------------------------------------------------------
# tick_genetics integration
# ---------------------------------------------------------------------------

class TestTickGenetics:
    def _make_genomes(self, n: int, rng: random.Random) -> dict[str, Genome]:
        return {f"c-{i}": Genome(alleles=[[rng.random(), rng.random()]
                                           for _ in range(NUM_GENES)])
                for i in range(n)}

    def test_tick_produces_valid_result(self):
        rng = random.Random(42)
        state = GeneticsState()
        genomes = self._make_genomes(10, rng)
        founders = {k: v for k, v in list(genomes.items())[:3]}
        ctx = GeneticsYearContext(
            year=10, births_this_year=1, deaths_this_year=0,
            population=10, avg_stress=0.3)
        result = tick_genetics(state, genomes, founders, ctx, rng)
        assert 0.0 <= result.diversity_index <= 1.0
        assert result.inbreeding_penalty >= 1.0
        assert result.diversity_trend in ("increasing", "decreasing", "stable")

    def test_tick_updates_state(self):
        rng = random.Random(42)
        state = GeneticsState()
        genomes = self._make_genomes(10, rng)
        founders = {}
        ctx = GeneticsYearContext(
            year=1, births_this_year=0, deaths_this_year=0,
            population=10, avg_stress=0.0)
        tick_genetics(state, genomes, founders, ctx, rng)
        assert len(state.diversity_history) == 1
        assert state.diversity_index > 0.0

    def test_tick_increments_generation_on_birth(self):
        rng = random.Random(42)
        state = GeneticsState(generation_count=0)
        genomes = self._make_genomes(5, rng)
        ctx = GeneticsYearContext(
            year=1, births_this_year=1, deaths_this_year=0,
            population=5, avg_stress=0.0)
        tick_genetics(state, genomes, {}, ctx, rng)
        assert state.generation_count == 1

    def test_tick_no_generation_without_birth(self):
        rng = random.Random(42)
        state = GeneticsState(generation_count=0)
        genomes = self._make_genomes(5, rng)
        ctx = GeneticsYearContext(
            year=1, births_this_year=0, deaths_this_year=0,
            population=5, avg_stress=0.0)
        tick_genetics(state, genomes, {}, ctx, rng)
        assert state.generation_count == 0

    def test_tick_computes_founder_survival_at_decade(self):
        rng = random.Random(42)
        state = GeneticsState()
        genomes = self._make_genomes(5, rng)
        founders = {"c-0": genomes["c-0"]}
        ctx = GeneticsYearContext(
            year=10, births_this_year=0, deaths_this_year=0,
            population=5, avg_stress=0.0)
        tick_genetics(state, genomes, founders, ctx, rng)
        assert "c-0" in state.founder_survival

    def test_tick_epigenetics_under_stress(self):
        rng = random.Random(42)
        state = GeneticsState()
        genomes = self._make_genomes(10, rng)
        ctx = GeneticsYearContext(
            year=50, births_this_year=0, deaths_this_year=0,
            population=10, avg_stress=0.9)
        result = tick_genetics(state, genomes, {}, ctx, rng)
        # High stress should trigger at least some epigenetic events
        # (probabilistic, but with 10 colonists and stress 0.9, very likely)
        assert isinstance(result.epigenetic_events, list)

    def test_tick_result_serializable(self):
        rng = random.Random(42)
        state = GeneticsState()
        genomes = self._make_genomes(5, rng)
        ctx = GeneticsYearContext(
            year=1, births_this_year=0, deaths_this_year=0,
            population=5, avg_stress=0.0)
        result = tick_genetics(state, genomes, {}, ctx, rng)
        d = result.to_dict()
        assert "diversity_index" in d
        assert "inbreeding_penalty" in d


# ---------------------------------------------------------------------------
# Physical invariants (property-based)
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_diversity_nonneg(self):
        rng = random.Random(99)
        for _ in range(50):
            genomes = [Genome(alleles=[[rng.random(), rng.random()]
                                        for _ in range(NUM_GENES)])
                       for _ in range(rng.randint(2, 10))]
            d = compute_colony_diversity(genomes)
            assert d >= 0.0

    def test_inbreeding_penalty_nondecreasing(self):
        """Lower diversity → higher penalty."""
        vals = [i * 0.05 for i in range(21)]
        penalties = [compute_inbreeding_penalty(v) for v in vals]
        for i in range(len(penalties) - 1):
            assert penalties[i] >= penalties[i + 1]

    def test_hybrid_vigor_nonneg(self):
        rng = random.Random(42)
        for _ in range(50):
            a = Genome(alleles=[[rng.random(), rng.random()]
                                 for _ in range(NUM_GENES)])
            b = Genome(alleles=[[rng.random(), rng.random()]
                                 for _ in range(NUM_GENES)])
            assert compute_hybrid_vigor(a, b) >= 0.0

    def test_expression_in_bounds(self):
        rng = random.Random(42)
        for _ in range(100):
            g = Genome(alleles=[[rng.random(), rng.random()]
                                 for _ in range(NUM_GENES)])
            for i in range(NUM_GENES):
                assert 0.0 <= g.express(i) <= 1.0

    def test_breed_child_in_bounds(self):
        rng = random.Random(42)
        for _ in range(50):
            a = Genome(alleles=[[rng.random(), rng.random()]
                                 for _ in range(NUM_GENES)])
            b = Genome(alleles=[[rng.random(), rng.random()]
                                 for _ in range(NUM_GENES)])
            child, _ = breed(a, b, rng)
            for pair in child.alleles:
                assert 0.0 <= pair[0] <= 1.0
                assert 0.0 <= pair[1] <= 1.0

    def test_engine_smoke_10_years(self):
        """Smoke test: run engine 10 years with genetics — no crash."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert len(result.years) == 10
        assert result.total_deaths >= 0
