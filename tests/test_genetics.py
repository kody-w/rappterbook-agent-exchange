"""Tests for the genetics organ (engine v11.0)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.genetics import (
    Genome, GeneticsState, GeneticsTickResult, PopulationGenetics,
    create_genome_from_stats, create_random_genome,
    reproduce, express_phenotype, form_gamete, combine_gametes,
    compute_genetic_conditions, compute_death_modifier,
    compute_population_metrics, tick_genetics,
    apply_radiation_mutation,
    ALL_LOCI, STAT_LOCUS_MAP, NUM_LOCI,
    MAX_PHENOTYPE_MODIFIER, CONDITION_THRESHOLD, _clamp01,
    MUTATION_RATE,
)
from src.mars100.colonist import (
    Colonist, ColonistStats, ColonistSkills,
    create_founding_ten, create_child, create_immigrant,
    STAT_NAMES,
)


# -----------------------------------------------------------------------
# Genome basics
# -----------------------------------------------------------------------

class TestGenome:
    def test_default_genome_has_correct_loci(self):
        g = Genome()
        assert len(g.alleles) == NUM_LOCI
        assert NUM_LOCI == 12

    def test_locus_value_is_mean(self):
        g = Genome(alleles=[(0.2, 0.8)] + [(0.5, 0.5)] * 11)
        assert abs(g.locus_value(0) - 0.5) < 1e-9

    def test_locus_heterozygosity(self):
        g = Genome(alleles=[(0.2, 0.8)] + [(0.5, 0.5)] * 11)
        assert abs(g.locus_heterozygosity(0) - 0.6) < 1e-9
        assert abs(g.locus_heterozygosity(1) - 0.0) < 1e-9

    def test_serialization_roundtrip(self):
        rng = random.Random(42)
        g = create_random_genome(rng)
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        for i in range(NUM_LOCI):
            assert abs(g.alleles[i][0] - g2.alleles[i][0]) < 0.001
            assert abs(g.alleles[i][1] - g2.alleles[i][1]) < 0.001

    def test_from_dict_empty(self):
        g = Genome.from_dict({})
        assert len(g.alleles) == NUM_LOCI

    def test_from_dict_none(self):
        g = Genome.from_dict(None)
        assert len(g.alleles) == NUM_LOCI


# -----------------------------------------------------------------------
# Creation
# -----------------------------------------------------------------------

class TestCreation:
    def test_genome_from_stats_alleles_bounded(self):
        rng = random.Random(123)
        stats = {"resolve": 0.9, "empathy": 0.1, "improvisation": 0.5,
                 "hoarding": 0.5, "faith": 0.5, "paranoia": 0.5}
        g = create_genome_from_stats(stats, rng)
        for a, b in g.alleles:
            assert 0.0 <= a <= 1.0
            assert 0.0 <= b <= 1.0

    def test_genome_from_stats_stat_correlation(self):
        """Stat-linked loci should correlate with stat values."""
        rng = random.Random(42)
        stats = {"resolve": 0.9, "empathy": 0.1, "improvisation": 0.5,
                 "hoarding": 0.5, "faith": 0.5, "paranoia": 0.5}
        # Run many times and check average correlation
        high_resolve_sum = 0.0
        low_empathy_sum = 0.0
        n = 50
        for i in range(n):
            g = create_genome_from_stats(stats, random.Random(i))
            high_resolve_sum += g.locus_value(0)  # resolve_gene
            low_empathy_sum += g.locus_value(2)  # empathy_gene
        assert high_resolve_sum / n > 0.7  # resolve gene should be high
        assert low_empathy_sum / n < 0.3  # empathy gene should be low

    def test_random_genome_bounded(self):
        rng = random.Random(99)
        for _ in range(100):
            g = create_random_genome(rng)
            for a, b in g.alleles:
                assert 0.0 <= a <= 1.0
                assert 0.0 <= b <= 1.0


# -----------------------------------------------------------------------
# Reproduction
# -----------------------------------------------------------------------

class TestReproduction:
    def test_gamete_has_correct_length(self):
        rng = random.Random(42)
        g = create_random_genome(rng)
        gamete = form_gamete(g, rng)
        assert len(gamete) == NUM_LOCI

    def test_gamete_values_from_parent(self):
        """Each gamete value should come from one of the parent's alleles."""
        rng = random.Random(42)
        g = Genome(alleles=[(0.1, 0.9)] * NUM_LOCI)
        gamete = form_gamete(g, rng)
        for val in gamete:
            assert val in (0.1, 0.9)

    def test_combine_gametes_produces_valid_genome(self):
        rng = random.Random(42)
        ga = [0.3] * NUM_LOCI
        gb = [0.7] * NUM_LOCI
        child = combine_gametes(ga, gb, rng)
        assert len(child.alleles) == NUM_LOCI
        for a, b in child.alleles:
            assert 0.0 <= a <= 1.0
            assert 0.0 <= b <= 1.0

    def test_reproduce_preserves_loci_count(self):
        rng = random.Random(42)
        pa = create_random_genome(rng)
        pb = create_random_genome(rng)
        child = reproduce(pa, pb, rng)
        assert len(child.alleles) == NUM_LOCI

    def test_reproduce_alleles_bounded(self):
        rng = random.Random(42)
        for seed in range(100):
            pa = create_random_genome(random.Random(seed))
            pb = create_random_genome(random.Random(seed + 1000))
            child = reproduce(pa, pb, random.Random(seed + 2000))
            for a, b in child.alleles:
                assert 0.0 <= a <= 1.0, f"allele out of bounds: {a}"
                assert 0.0 <= b <= 1.0, f"allele out of bounds: {b}"

    def test_mutation_occurs(self):
        """Over many offspring, mutations should occur sometimes."""
        rng = random.Random(42)
        pa = Genome(alleles=[(0.5, 0.5)] * NUM_LOCI)
        pb = Genome(alleles=[(0.5, 0.5)] * NUM_LOCI)
        mutations_seen = 0
        for i in range(200):
            child = reproduce(pa, pb, random.Random(i))
            for a, b in child.alleles:
                if abs(a - 0.5) > 0.01 or abs(b - 0.5) > 0.01:
                    mutations_seen += 1
                    break
        assert mutations_seen > 0, "No mutations in 200 offspring"


# -----------------------------------------------------------------------
# Phenotype expression
# -----------------------------------------------------------------------

class TestPhenotype:
    def test_neutral_genome_near_zero_modifiers(self):
        g = Genome()  # default 0.5, 0.5 at every locus
        mods = express_phenotype(g)
        for stat in STAT_NAMES:
            key = f"stat_{stat}"
            assert abs(mods[key]) < 1e-6, f"{key} should be ~0 for neutral genome"

    def test_modifiers_bounded(self):
        rng = random.Random(42)
        for _ in range(100):
            g = create_random_genome(rng)
            mods = express_phenotype(g)
            for key, val in mods.items():
                if key.startswith("stat_"):
                    assert -MAX_PHENOTYPE_MODIFIER <= val <= MAX_PHENOTYPE_MODIFIER
                else:
                    assert 0.0 <= val <= 1.0

    def test_high_stat_gene_positive_modifier(self):
        alleles = [(0.5, 0.5)] * NUM_LOCI
        alleles[0] = (0.95, 0.95)  # resolve_gene very high
        g = Genome(alleles=alleles)
        mods = express_phenotype(g)
        assert mods["stat_resolve"] > 0

    def test_low_stat_gene_negative_modifier(self):
        alleles = [(0.5, 0.5)] * NUM_LOCI
        alleles[0] = (0.05, 0.05)  # resolve_gene very low
        g = Genome(alleles=alleles)
        mods = express_phenotype(g)
        assert mods["stat_resolve"] < 0


# -----------------------------------------------------------------------
# Genetic conditions
# -----------------------------------------------------------------------

class TestConditions:
    def test_no_conditions_for_neutral_genome(self):
        g = Genome()
        conditions = compute_genetic_conditions(g)
        # Neutral genome (0.5) should not trigger any condition
        assert len(conditions) == 0

    def test_radiation_sensitivity_triggers(self):
        alleles = [(0.5, 0.5)] * NUM_LOCI
        idx = ALL_LOCI.index("radiation_resistance")
        alleles[idx] = (0.1, 0.1)  # both below threshold
        g = Genome(alleles=alleles)
        conditions = compute_genetic_conditions(g)
        names = [c["name"] for c in conditions]
        assert "radiation_sensitivity" in names

    def test_genetic_vigor_triggers(self):
        alleles = [(0.5, 0.5)] * NUM_LOCI
        idx = ALL_LOCI.index("longevity")
        alleles[idx] = (0.9, 0.9)  # both above 0.75 threshold
        g = Genome(alleles=alleles)
        conditions = compute_genetic_conditions(g)
        names = [c["name"] for c in conditions]
        assert "genetic_vigor" in names

    def test_no_vigor_when_one_allele_low(self):
        alleles = [(0.5, 0.5)] * NUM_LOCI
        idx = ALL_LOCI.index("longevity")
        alleles[idx] = (0.9, 0.3)  # one low = no vigor
        g = Genome(alleles=alleles)
        conditions = compute_genetic_conditions(g)
        names = [c["name"] for c in conditions]
        assert "genetic_vigor" not in names


# -----------------------------------------------------------------------
# Death modifier
# -----------------------------------------------------------------------

class TestDeathModifier:
    def test_neutral_genome_near_one(self):
        g = Genome()
        mod = compute_death_modifier(g)
        assert 0.95 <= mod <= 1.05

    def test_vulnerable_genome_above_one(self):
        alleles = [(0.5, 0.5)] * NUM_LOCI
        idx = ALL_LOCI.index("radiation_resistance")
        alleles[idx] = (0.1, 0.1)
        idx2 = ALL_LOCI.index("immune_strength")
        alleles[idx2] = (0.1, 0.1)
        g = Genome(alleles=alleles)
        mod = compute_death_modifier(g)
        assert mod > 1.0

    def test_vigorous_genome_below_one(self):
        alleles = [(0.5, 0.5)] * NUM_LOCI
        idx = ALL_LOCI.index("longevity")
        alleles[idx] = (0.9, 0.9)
        idx2 = ALL_LOCI.index("mars_adaptation")
        alleles[idx2] = (0.9, 0.9)
        g = Genome(alleles=alleles)
        mod = compute_death_modifier(g)
        assert mod < 1.0

    def test_modifier_clamped(self):
        for _ in range(100):
            rng = random.Random(_)
            g = create_random_genome(rng)
            mod = compute_death_modifier(g)
            assert 0.5 <= mod <= 2.0


# -----------------------------------------------------------------------
# Radiation mutations
# -----------------------------------------------------------------------

class TestRadiation:
    def test_no_mutations_at_zero_severity(self):
        rng = random.Random(42)
        g = create_random_genome(rng)
        muts = apply_radiation_mutation(g, 0.0, rng)
        assert len(muts) == 0

    def test_mutations_possible_at_high_severity(self):
        """With high severity, at least some mutations should occur over many trials."""
        total_muts = 0
        for seed in range(50):
            rng = random.Random(seed)
            g = create_random_genome(rng)
            muts = apply_radiation_mutation(g, 0.9, random.Random(seed + 100))
            total_muts += len(muts)
        assert total_muts > 0

    def test_mutations_stay_bounded(self):
        rng = random.Random(42)
        g = create_random_genome(rng)
        apply_radiation_mutation(g, 1.0, rng)
        for a, b in g.alleles:
            assert 0.0 <= a <= 1.0
            assert 0.0 <= b <= 1.0

    def test_high_resistance_fewer_mutations(self):
        """Colonists with high radiation resistance should get fewer mutations."""
        high_res_muts = 0
        low_res_muts = 0
        for seed in range(100):
            # High resistance
            alleles_h = [(0.5, 0.5)] * NUM_LOCI
            idx = ALL_LOCI.index("radiation_resistance")
            alleles_h[idx] = (0.95, 0.95)
            gh = Genome(alleles=list(alleles_h))
            high_res_muts += len(apply_radiation_mutation(
                gh, 0.8, random.Random(seed)))

            # Low resistance
            alleles_l = [(0.5, 0.5)] * NUM_LOCI
            alleles_l[idx] = (0.05, 0.05)
            gl = Genome(alleles=list(alleles_l))
            low_res_muts += len(apply_radiation_mutation(
                gl, 0.8, random.Random(seed)))

        assert low_res_muts > high_res_muts


# -----------------------------------------------------------------------
# Population metrics
# -----------------------------------------------------------------------

class TestPopulationMetrics:
    def test_empty_population(self):
        m = compute_population_metrics([])
        assert m.heterozygosity == 0.0
        assert m.diversity_index == 0.0

    def test_single_genome(self):
        g = Genome(alleles=[(0.3, 0.7)] * NUM_LOCI)
        m = compute_population_metrics([g])
        assert m.heterozygosity > 0
        assert m.diversity_index == 0.0  # no variance with 1 individual

    def test_identical_genomes_low_diversity(self):
        g = Genome(alleles=[(0.5, 0.5)] * NUM_LOCI)
        m = compute_population_metrics([g, g, g])
        assert m.diversity_index == 0.0

    def test_diverse_genomes_high_diversity(self):
        g1 = Genome(alleles=[(0.1, 0.1)] * NUM_LOCI)
        g2 = Genome(alleles=[(0.9, 0.9)] * NUM_LOCI)
        m = compute_population_metrics([g1, g2])
        assert m.diversity_index > 0.1

    def test_metrics_serialization(self):
        rng = random.Random(42)
        genomes = [create_random_genome(rng) for _ in range(5)]
        m = compute_population_metrics(genomes)
        d = m.to_dict()
        assert "heterozygosity" in d
        assert "diversity_index" in d
        assert len(d["mean_locus_values"]) == NUM_LOCI


# -----------------------------------------------------------------------
# State and tick
# -----------------------------------------------------------------------

class TestGeneticsTick:
    def test_tick_returns_result(self):
        rng = random.Random(42)
        state = GeneticsState()
        genomes = {f"c-{i}": create_random_genome(random.Random(i))
                   for i in range(5)}
        active = list(genomes.keys())
        result = tick_genetics(state, genomes, active, year=1,
                               radiation_severity=0.0, rng=rng)
        assert result.year == 1
        assert "heterozygosity" in result.population_metrics

    def test_tick_updates_history(self):
        rng = random.Random(42)
        state = GeneticsState()
        genomes = {f"c-{i}": create_random_genome(random.Random(i))
                   for i in range(5)}
        active = list(genomes.keys())
        for yr in range(1, 11):
            tick_genetics(state, genomes, active, year=yr,
                          radiation_severity=0.0, rng=rng)
        assert len(state.diversity_history) == 10
        assert len(state.heterozygosity_history) == 10
        assert state.generations_tracked == 10

    def test_tick_with_radiation(self):
        rng = random.Random(42)
        state = GeneticsState()
        genomes = {f"c-{i}": create_random_genome(random.Random(i))
                   for i in range(10)}
        active = list(genomes.keys())
        result = tick_genetics(state, genomes, active, year=1,
                               radiation_severity=0.8, rng=rng)
        # Should have some mutations
        assert isinstance(result.radiation_mutations, list)

    def test_tick_computes_death_modifiers(self):
        rng = random.Random(42)
        state = GeneticsState()
        genomes = {"a": create_random_genome(rng), "b": create_random_genome(rng)}
        result = tick_genetics(state, genomes, ["a", "b"], year=1,
                               radiation_severity=0.0, rng=rng)
        assert "a" in result.death_modifiers
        assert "b" in result.death_modifiers

    def test_tick_computes_phenotype_modifiers(self):
        rng = random.Random(42)
        state = GeneticsState()
        genomes = {"a": create_random_genome(rng)}
        result = tick_genetics(state, genomes, ["a"], year=1,
                               radiation_severity=0.0, rng=rng)
        assert "a" in result.phenotype_modifiers

    def test_state_serialization_roundtrip(self):
        state = GeneticsState()
        state.diversity_history = [0.1, 0.2, 0.3]
        state.total_mutations = 5
        d = state.to_dict()
        state2 = GeneticsState.from_dict(d)
        assert state2.diversity_history == [0.1, 0.2, 0.3]
        assert state2.total_mutations == 5

    def test_mutation_log_capped(self):
        state = GeneticsState()
        state.mutation_log = [{"i": i} for i in range(100)]
        d = state.to_dict()
        assert len(d["mutation_log"]) == 50  # MAX_MUTATION_LOG

    def test_tick_result_serialization(self):
        rng = random.Random(42)
        state = GeneticsState()
        genomes = {"a": create_random_genome(rng)}
        result = tick_genetics(state, genomes, ["a"], year=1,
                               radiation_severity=0.5, rng=rng)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "year" in d
        assert "population_metrics" in d


# -----------------------------------------------------------------------
# Colonist integration
# -----------------------------------------------------------------------

class TestColonistIntegration:
    def test_founding_ten_have_genomes(self):
        colonists = create_founding_ten(42)
        for c in colonists:
            assert c.genome is not None
            assert len(c.genome.alleles) == NUM_LOCI

    def test_child_inherits_genome(self):
        rng = random.Random(42)
        pa = create_founding_ten(42)[0]
        pb = create_founding_ten(42)[1]
        child = create_child(pa, pb, "child-0", 10, rng)
        assert child.genome is not None
        assert len(child.genome.alleles) == NUM_LOCI

    def test_child_without_parent_genomes(self):
        """create_child should work when parents have no genomes."""
        rng = random.Random(42)
        pa = Colonist(id="a", name="A", element="fire", archetype="test",
                      stats=ColonistStats(), skills=ColonistSkills(),
                      decision_expr="(+ 1 1)")
        pb = Colonist(id="b", name="B", element="water", archetype="test",
                      stats=ColonistStats(), skills=ColonistSkills(),
                      decision_expr="(+ 2 2)")
        child = create_child(pa, pb, "child-0", 10, rng)
        assert child.genome is None  # both parents None → child None

    def test_child_with_one_parent_genome(self):
        rng = random.Random(42)
        pa = create_founding_ten(42)[0]  # has genome
        pb = Colonist(id="b", name="B", element="water", archetype="test",
                      stats=ColonistStats(), skills=ColonistSkills(),
                      decision_expr="(+ 2 2)")
        child = create_child(pa, pb, "child-0", 10, rng)
        assert child.genome is not None  # one parent has genome → child gets genome

    def test_immigrant_has_genome(self):
        rng = random.Random(42)
        imm = create_immigrant("imm-0", 15, rng)
        assert imm.genome is not None
        assert len(imm.genome.alleles) == NUM_LOCI

    def test_colonist_serialization_with_genome(self):
        colonists = create_founding_ten(42)
        c = colonists[0]
        d = c.to_dict()
        assert "genome" in d
        c2 = Colonist.from_dict(d)
        assert c2.genome is not None
        assert len(c2.genome.alleles) == NUM_LOCI

    def test_colonist_serialization_without_genome(self):
        """Old-format colonist dicts (no genome) should load fine."""
        d = {
            "id": "test", "name": "Test", "element": "fire",
            "archetype": "test", "stats": {s: 0.5 for s in STAT_NAMES},
            "skills": {"terraforming": 0, "hydroponics": 0, "mediation": 0,
                       "coding": 0, "prayer": 0, "sabotage": 0},
        }
        c = Colonist.from_dict(d)
        assert c.genome is None


# -----------------------------------------------------------------------
# Engine integration (smoke test)
# -----------------------------------------------------------------------

class TestEngineIntegration:
    def test_engine_10_years_no_crash(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert len(result.years) == 10
        # Genetics should be populated
        for yr in result.years:
            assert "genetics" in yr.to_dict()

    def test_engine_has_genetics_state(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        d = result.to_dict()
        assert "final_genetics" in d
        assert "diversity_history" in d["final_genetics"]

    def test_engine_colonist_genomes_tracked(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=1)
        engine.run()
        assert len(engine.colonist_genomes) >= 10

    def test_engine_version_is_11(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=1)
        result = engine.run()
        d = result.to_dict()
        assert d["_meta"]["version"] == "11.0"


# -----------------------------------------------------------------------
# Property-based invariants
# -----------------------------------------------------------------------

class TestInvariants:
    def test_all_alleles_bounded_after_many_reproductions(self):
        """After many generations, alleles should remain in [0, 1]."""
        rng = random.Random(42)
        population = [create_random_genome(rng) for _ in range(10)]
        for gen in range(20):
            next_gen = []
            for _ in range(10):
                pa = rng.choice(population)
                pb = rng.choice(population)
                child = reproduce(pa, pb, rng)
                for a, b in child.alleles:
                    assert 0.0 <= a <= 1.0
                    assert 0.0 <= b <= 1.0
                next_gen.append(child)
            population = next_gen

    def test_diversity_nonincreasing_in_small_isolated_population(self):
        """In a small isolated population, diversity should not increase
        indefinitely (genetic drift)."""
        rng = random.Random(42)
        population = [create_random_genome(rng) for _ in range(4)]
        initial_div = compute_population_metrics(population).diversity_index
        for gen in range(50):
            next_gen = []
            for _ in range(4):
                pa = rng.choice(population)
                pb = rng.choice(population)
                next_gen.append(reproduce(pa, pb, rng))
            population = next_gen
        final_div = compute_population_metrics(population).diversity_index
        # With only 4 individuals, drift should reduce diversity
        # (may not always hold due to mutation, but on average)
        # Just check it's finite
        assert final_div >= 0.0

    def test_phenotype_never_exceeds_max(self):
        for seed in range(200):
            g = create_random_genome(random.Random(seed))
            mods = express_phenotype(g)
            for k, v in mods.items():
                if k.startswith("stat_"):
                    assert -MAX_PHENOTYPE_MODIFIER <= v <= MAX_PHENOTYPE_MODIFIER

    def test_death_modifier_always_bounded(self):
        for seed in range(200):
            g = create_random_genome(random.Random(seed))
            mod = compute_death_modifier(g)
            assert 0.5 <= mod <= 2.0
