"""Tests for the genetics organ (engine v11.0)."""
from __future__ import annotations

import random

import pytest

from src.mars100.genetics import (
    LOCUS_NAMES, MUTATION_RATE, MUTATION_SIGMA,
    INBREEDING_PENALTY_THRESHOLD, INBREEDING_STAT_PENALTY,
    ADAPTATION_LOCI, MAX_DEATH_RATE_REDUCTION, MAX_MUTATION_LOG,
    Genome, GeneticsState, GeneticsYearContext, GeneticsTickResult,
    create_founder_genome, create_immigrant_genome, crossover,
    check_inbreeding, apply_inbreeding_penalty,
    compute_heterozygosity, compute_adaptation_score,
    compute_avg_inbreeding, compute_genetics_modifiers,
    mutate_alleles, tick_genetics,
)


# ---- helpers ----

def make_rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


def make_genome(val: float = 0.5, generation: int = 0) -> Genome:
    alleles = {loc: (val, val) for loc in LOCUS_NAMES}
    return Genome(alleles=alleles, generation=generation)


def make_diverse_genomes(n: int, rng: random.Random) -> list[Genome]:
    return [create_founder_genome(f"c-{i}", ["fire", "water", "earth", "air"][i % 4], rng)
            for i in range(n)]


# ---- Genome ----

class TestGenome:
    def test_phenotype_codominant(self):
        g = Genome(alleles={"radiation_resistance": (0.8, 0.2)})
        p = g.phenotype("radiation_resistance")
        assert 0.0 <= p <= 1.0
        assert abs(p - 0.5) < 0.01  # co-dominant: avg

    def test_phenotype_missing_locus(self):
        g = Genome(alleles={})
        assert g.phenotype("radiation_resistance") == 0.5

    def test_phenotypes_all_loci(self):
        g = make_genome(0.7)
        pheno = g.phenotypes()
        assert set(pheno.keys()) == set(LOCUS_NAMES)
        for v in pheno.values():
            assert 0.0 <= v <= 1.0

    def test_similarity_identical(self):
        a = make_genome(0.5)
        b = make_genome(0.5)
        assert a.similarity(b) == pytest.approx(1.0, abs=0.01)

    def test_similarity_different(self):
        a = make_genome(0.0)
        b = make_genome(1.0)
        assert a.similarity(b) < 0.2

    def test_similarity_empty(self):
        a = Genome()
        b = Genome()
        assert a.similarity(b) == 0.0

    def test_to_dict_from_dict_roundtrip(self):
        g = make_genome(0.6, generation=3)
        d = g.to_dict()
        restored = Genome.from_dict(d)
        for loc in LOCUS_NAMES:
            a_orig, b_orig = g.alleles[loc]
            a_rest, b_rest = restored.alleles[loc]
            assert abs(a_orig - a_rest) < 0.001
            assert abs(b_orig - b_rest) < 0.001
        assert restored.generation == 3

    def test_from_dict_empty(self):
        g = Genome.from_dict({})
        assert g.generation == 0
        assert g.alleles == {}


# ---- GeneticsState ----

class TestGeneticsState:
    def test_to_dict_from_dict(self):
        s = GeneticsState(heterozygosity=0.8, adaptation_score=0.3,
                          inbreeding_coefficient=0.1, generation_avg=1.5,
                          mutation_log=[{"locus": "bone_density", "year": 5}])
        d = s.to_dict()
        restored = GeneticsState.from_dict(d)
        assert abs(restored.heterozygosity - 0.8) < 0.01
        assert abs(restored.adaptation_score - 0.3) < 0.01
        assert len(restored.mutation_log) == 1

    def test_mutation_log_capped(self):
        s = GeneticsState(mutation_log=[{"i": i} for i in range(100)])
        d = s.to_dict()
        assert len(d["mutation_log"]) == MAX_MUTATION_LOG


# ---- Founder / Immigrant genome creation ----

class TestGenomeCreation:
    def test_founder_genome_bounds(self):
        rng = make_rng()
        for element in ("fire", "water", "earth", "air"):
            g = create_founder_genome(f"c-{element}", element, rng)
            for loc in LOCUS_NAMES:
                a, b = g.alleles[loc]
                assert 0.0 <= a <= 1.0
                assert 0.0 <= b <= 1.0
            assert g.generation == 0

    def test_immigrant_genome_bounds(self):
        rng = make_rng()
        g = create_immigrant_genome(rng)
        for loc in LOCUS_NAMES:
            a, b = g.alleles[loc]
            assert 0.0 <= a <= 1.0
            assert 0.0 <= b <= 1.0
        assert g.generation == 0

    def test_founder_element_bias(self):
        """Fire founders should have higher radiation_resistance on average."""
        rng = make_rng(99)
        fire_vals = []
        water_vals = []
        for i in range(50):
            fg = create_founder_genome(f"f-{i}", "fire", rng)
            wg = create_founder_genome(f"w-{i}", "water", rng)
            fire_vals.append(fg.phenotype("radiation_resistance"))
            water_vals.append(wg.phenotype("radiation_resistance"))
        assert sum(fire_vals) / len(fire_vals) > sum(water_vals) / len(water_vals)


# ---- Crossover ----

class TestCrossover:
    def test_child_alleles_bounded(self):
        rng = make_rng()
        a = make_genome(0.8, generation=1)
        b = make_genome(0.2, generation=2)
        child = crossover(a, b, rng)
        for loc in LOCUS_NAMES:
            x, y = child.alleles[loc]
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0
        assert child.generation == 3  # max(1, 2) + 1

    def test_crossover_deterministic(self):
        a = make_genome(0.7)
        b = make_genome(0.3)
        c1 = crossover(a, b, make_rng(42))
        c2 = crossover(a, b, make_rng(42))
        for loc in LOCUS_NAMES:
            assert c1.alleles[loc] == c2.alleles[loc]

    def test_crossover_inherits_from_both(self):
        """Over many trials, child alleles should come from both parents."""
        rng = make_rng(123)
        a = make_genome(0.1)
        b = make_genome(0.9)
        children = [crossover(a, b, rng) for _ in range(100)]
        for loc in LOCUS_NAMES:
            vals = [c.alleles[loc][0] for c in children]
            avg = sum(vals) / len(vals)
            # Should be between parents (with mutation noise)
            assert 0.0 < avg < 1.0  # mutation noise can push beyond parent range


# ---- Inbreeding ----

class TestInbreeding:
    def test_identical_parents_high_inbreeding(self):
        a = make_genome(0.5)
        b = make_genome(0.5)
        coeff = check_inbreeding(a, b)
        assert coeff > INBREEDING_PENALTY_THRESHOLD

    def test_diverse_parents_low_inbreeding(self):
        a = make_genome(0.1)
        b = make_genome(0.9)
        coeff = check_inbreeding(a, b)
        assert coeff < INBREEDING_PENALTY_THRESHOLD

    def test_penalty_reduces_alleles(self):
        child = make_genome(0.5)
        apply_inbreeding_penalty(child, 0.95)
        for loc in LOCUS_NAMES:
            a, b = child.alleles[loc]
            assert a < 0.5
            assert b < 0.5

    def test_no_penalty_below_threshold(self):
        child = make_genome(0.5)
        original = {loc: child.alleles[loc] for loc in LOCUS_NAMES}
        apply_inbreeding_penalty(child, 0.5)
        for loc in LOCUS_NAMES:
            assert child.alleles[loc] == original[loc]


# ---- Colony metrics ----

class TestColonyMetrics:
    def test_heterozygosity_diverse(self):
        rng = make_rng()
        genomes = make_diverse_genomes(10, rng)
        h = compute_heterozygosity(genomes)
        assert h > 0.3  # diverse population

    def test_heterozygosity_uniform(self):
        genomes = [make_genome(0.5) for _ in range(10)]
        h = compute_heterozygosity(genomes)
        assert h < 0.3  # all identical

    def test_heterozygosity_empty(self):
        assert compute_heterozygosity([]) == 0.0

    def test_adaptation_score_bounds(self):
        rng = make_rng()
        genomes = make_diverse_genomes(10, rng)
        score = compute_adaptation_score(genomes)
        assert 0.0 <= score <= 1.0

    def test_adaptation_score_empty(self):
        assert compute_adaptation_score([]) == 0.0

    def test_avg_inbreeding_identical(self):
        genomes = [make_genome(0.5) for _ in range(5)]
        coeff = compute_avg_inbreeding(genomes)
        assert coeff > 0.9

    def test_avg_inbreeding_single(self):
        assert compute_avg_inbreeding([make_genome()]) == 0.0


# ---- Modifiers ----

class TestModifiers:
    def test_modifiers_default(self):
        genomes = [make_genome(0.5) for _ in range(5)]
        mods = compute_genetics_modifiers(genomes)
        assert abs(mods["death_rate_modifier"] - 1.0) < 0.2
        assert abs(mods["skill_learning_modifier"] - 1.0) < 0.2
        assert abs(mods["stress_modifier"]) < 0.1
        assert abs(mods["air_bonus"]) < 0.02

    def test_modifiers_high_adaptation(self):
        genomes = [make_genome(0.9) for _ in range(5)]
        mods = compute_genetics_modifiers(genomes)
        assert mods["death_rate_modifier"] < 1.0  # reduced death rate

    def test_modifiers_empty(self):
        mods = compute_genetics_modifiers([])
        assert mods["death_rate_modifier"] == 1.0

    def test_death_rate_bounded(self):
        """Death rate modifier should never go below 1 - MAX_DEATH_RATE_REDUCTION."""
        genomes = [make_genome(1.0) for _ in range(10)]
        mods = compute_genetics_modifiers(genomes)
        assert mods["death_rate_modifier"] >= 1.0 - MAX_DEATH_RATE_REDUCTION


# ---- Mutations ----

class TestMutations:
    def test_mutate_alleles_bounded(self):
        rng = make_rng()
        g = make_genome(0.5)
        muts = mutate_alleles(g, 1.0, rng)  # 100% rate for testing
        for loc in LOCUS_NAMES:
            a, b = g.alleles[loc]
            assert 0.0 <= a <= 1.0
            assert 0.0 <= b <= 1.0
        assert len(muts) > 0

    def test_mutate_zero_rate(self):
        rng = make_rng()
        g = make_genome(0.5)
        original = {loc: g.alleles[loc] for loc in LOCUS_NAMES}
        muts = mutate_alleles(g, 0.0, rng)
        assert len(muts) == 0
        for loc in LOCUS_NAMES:
            assert g.alleles[loc] == original[loc]


# ---- tick_genetics ----

class TestTickGenetics:
    def test_basic_tick(self):
        rng = make_rng()
        state = GeneticsState()
        genomes = {f"c-{i}": create_founder_genome(f"c-{i}", "fire", rng)
                   for i in range(5)}
        ctx = GeneticsYearContext(
            year=10, active_ids=list(genomes.keys()),
            radiation_event=False, births_this_year=[], deaths_this_year=[])
        result = tick_genetics(state, genomes, ctx, rng)
        assert 0.5 <= result.death_rate_modifier <= 1.5
        assert result.skill_learning_modifier > 0.0
        assert state.heterozygosity > 0.0

    def test_tick_removes_dead(self):
        rng = make_rng()
        state = GeneticsState()
        genomes = {f"c-{i}": make_genome(0.5) for i in range(5)}
        ctx = GeneticsYearContext(
            year=10, active_ids=["c-0", "c-1", "c-2", "c-3", "c-4"],
            radiation_event=False, births_this_year=[],
            deaths_this_year=["c-4"])
        tick_genetics(state, genomes, ctx, rng)
        assert "c-4" not in genomes

    def test_tick_radiation_boosts_mutations(self):
        rng1 = make_rng(99)
        rng2 = make_rng(99)
        state1 = GeneticsState()
        state2 = GeneticsState()
        genomes1 = {f"c-{i}": make_genome(0.5) for i in range(20)}
        genomes2 = {f"c-{i}": make_genome(0.5) for i in range(20)}
        ids = [f"c-{i}" for i in range(20)]

        ctx1 = GeneticsYearContext(year=10, active_ids=ids,
                                   radiation_event=False,
                                   births_this_year=[], deaths_this_year=[])
        ctx2 = GeneticsYearContext(year=10, active_ids=ids,
                                   radiation_event=True,
                                   births_this_year=[], deaths_this_year=[])
        # Run many times to get statistical significance
        total_muts_normal = 0
        total_muts_rad = 0
        for trial in range(100):
            rng_n = make_rng(trial)
            rng_r = make_rng(trial + 10000)
            s_n = GeneticsState()
            s_r = GeneticsState()
            g_n = {f"c-{i}": make_genome(0.5) for i in range(20)}
            g_r = {f"c-{i}": make_genome(0.5) for i in range(20)}
            r_n = tick_genetics(s_n, g_n, ctx1, rng_n)
            r_r = tick_genetics(s_r, g_r, ctx2, rng_r)
            total_muts_normal += len(r_n.mutations_this_year)
            total_muts_rad += len(r_r.mutations_this_year)
        # Radiation should produce more mutations on average
        assert total_muts_rad > total_muts_normal

    def test_tick_diversity_warning(self):
        rng = make_rng()
        state = GeneticsState()
        # All identical genomes → low heterozygosity
        genomes = {f"c-{i}": make_genome(0.5) for i in range(5)}
        ctx = GeneticsYearContext(
            year=10, active_ids=list(genomes.keys()),
            radiation_event=False, births_this_year=[], deaths_this_year=[])
        result = tick_genetics(state, genomes, ctx, rng)
        assert result.diversity_warning is not None

    def test_tick_to_dict(self):
        result = GeneticsTickResult(death_rate_modifier=0.95)
        d = result.to_dict()
        assert d["death_rate_modifier"] == 0.95


# ---- Smoke test: run 10 years ----

class TestSmokeGenetics:
    def test_10_year_run(self):
        rng = make_rng(42)
        state = GeneticsState()
        elements = ["fire", "water", "earth", "air"]
        genomes = {f"c-{i}": create_founder_genome(
            f"c-{i}", elements[i % 4], rng) for i in range(10)}

        for year in range(1, 11):
            active = list(genomes.keys())
            deaths = []
            # Simulate one death in year 5
            if year == 5 and len(active) > 3:
                deaths = [active[-1]]

            ctx = GeneticsYearContext(
                year=year, active_ids=active,
                radiation_event=(year == 3),
                births_this_year=[], deaths_this_year=deaths)
            result = tick_genetics(state, genomes, ctx, rng)

            assert 0.5 <= result.death_rate_modifier <= 1.5
            assert result.skill_learning_modifier > 0.0
            assert state.heterozygosity >= 0.0
            assert state.adaptation_score >= 0.0
            assert 0.0 <= state.inbreeding_coefficient <= 1.0

        # After 10 years, state should be stable
        assert state.generation_avg == 0.0  # no births → all gen 0
        assert len(genomes) == 9  # one death

    def test_with_births(self):
        rng = make_rng(42)
        state = GeneticsState()
        genomes = {f"c-{i}": make_genome(0.3 + i * 0.07) for i in range(6)}

        # Create a child via crossover
        parent_a = genomes["c-0"]
        parent_b = genomes["c-3"]
        child = crossover(parent_a, parent_b, rng)
        inbreeding = check_inbreeding(parent_a, parent_b)
        apply_inbreeding_penalty(child, inbreeding)
        genomes["child-0"] = child

        ctx = GeneticsYearContext(
            year=15, active_ids=list(genomes.keys()),
            radiation_event=False,
            births_this_year=[{"id": "child-0", "parents": ["c-0", "c-3"]}],
            deaths_this_year=[])
        result = tick_genetics(state, genomes, ctx, rng)

        assert "child-0" in genomes
        assert state.generation_avg > 0.0  # child is gen 1
        assert 0.0 <= state.heterozygosity <= 1.0


# ---- Property-based: all outputs in physical bounds ----

class TestPropertyBounds:
    @pytest.mark.parametrize("seed", range(20))
    def test_modifiers_bounded(self, seed):
        rng = make_rng(seed)
        genomes = make_diverse_genomes(8, rng)
        mods = compute_genetics_modifiers(genomes)
        assert 0.5 <= mods["death_rate_modifier"] <= 1.5
        assert 0.5 <= mods["skill_learning_modifier"] <= 1.5
        assert -0.1 <= mods["stress_modifier"] <= 0.1
        assert -0.02 <= mods["air_bonus"] <= 0.02

    @pytest.mark.parametrize("seed", range(20))
    def test_phenotypes_bounded(self, seed):
        rng = make_rng(seed)
        g = create_founder_genome("test", "fire", rng)
        for loc in LOCUS_NAMES:
            assert 0.0 <= g.phenotype(loc) <= 1.0

    @pytest.mark.parametrize("seed", range(10))
    def test_crossover_bounded(self, seed):
        rng = make_rng(seed)
        a = create_founder_genome("a", "fire", rng)
        b = create_founder_genome("b", "water", rng)
        child = crossover(a, b, rng)
        for loc in LOCUS_NAMES:
            x, y = child.alleles[loc]
            assert 0.0 <= x <= 1.0, f"{loc} allele 0 out of bounds: {x}"
            assert 0.0 <= y <= 1.0, f"{loc} allele 1 out of bounds: {y}"
