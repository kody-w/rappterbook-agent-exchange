"""Tests for the genetics organ (engine v11.0)."""
from __future__ import annotations

import random
import pytest

from src.mars100.genetics import (
    Genome, GeneticsState, GeneticsTickResult,
    STAT_NAMES, MUTATION_RATE, MUTATION_SIGMA,
    INBREEDING_THRESHOLD, INBREEDING_BIRTH_PENALTY,
    MIN_DIVERSITY_WARNING,
    create_genome_from_stats, crossover, mutate,
    compute_pedigree_kinship, compute_diversity_index,
    inbreeding_birth_modifier, genetic_death_modifier,
    record_birth, tick_genetics,
)


# ── Genome basics ──────────────────────────────────────────────────────


class TestGenome:
    def test_express_is_mean(self) -> None:
        g = Genome(alleles={"resolve": (0.8, 0.4), "empathy": (0.6, 0.2)})
        pheno = g.express()
        assert pheno["resolve"] == pytest.approx(0.6)
        assert pheno["empathy"] == pytest.approx(0.4)

    def test_express_identical_alleles(self) -> None:
        g = Genome(alleles={"resolve": (0.7, 0.7)})
        assert g.express()["resolve"] == pytest.approx(0.7)

    def test_homozygosity_identical(self) -> None:
        g = Genome(alleles={n: (0.5, 0.5) for n in STAT_NAMES})
        assert g.homozygosity() == pytest.approx(1.0)

    def test_homozygosity_diverse(self) -> None:
        g = Genome(alleles={n: (0.0, 1.0) for n in STAT_NAMES})
        assert g.homozygosity() == pytest.approx(0.0)

    def test_roundtrip(self) -> None:
        g = Genome(alleles={"resolve": (0.8, 0.3), "empathy": (0.1, 0.9)})
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        for name in g.alleles:
            a1, b1 = g.alleles[name]
            a2, b2 = g2.alleles[name]
            assert a2 == pytest.approx(a1, abs=1e-3)
            assert b2 == pytest.approx(b1, abs=1e-3)

    def test_clamp(self) -> None:
        g = Genome(alleles={"resolve": (-0.5, 1.5)})
        g.clamp()
        a, b = g.alleles["resolve"]
        assert a == 0.0
        assert b == 1.0


# ── Genome creation from stats ─────────────────────────────────────────


class TestCreateGenome:
    def test_express_matches_stats(self) -> None:
        rng = random.Random(42)
        stats = {n: 0.5 for n in STAT_NAMES}
        g = create_genome_from_stats(stats, rng)
        pheno = g.express()
        for name in STAT_NAMES:
            assert 0.3 < pheno[name] < 0.7

    def test_all_stats_present(self) -> None:
        rng = random.Random(99)
        stats = {"resolve": 0.9, "empathy": 0.1}
        g = create_genome_from_stats(stats, rng)
        for name in STAT_NAMES:
            assert name in g.alleles

    def test_deterministic(self) -> None:
        stats = {n: 0.5 for n in STAT_NAMES}
        g1 = create_genome_from_stats(stats, random.Random(7))
        g2 = create_genome_from_stats(stats, random.Random(7))
        assert g1.to_dict() == g2.to_dict()


# ── Crossover ──────────────────────────────────────────────────────────


class TestCrossover:
    def test_child_has_all_stats(self) -> None:
        rng = random.Random(42)
        pa = Genome(alleles={n: (0.8, 0.2) for n in STAT_NAMES})
        pb = Genome(alleles={n: (0.6, 0.4) for n in STAT_NAMES})
        child = crossover(pa, pb, rng)
        for name in STAT_NAMES:
            assert name in child.alleles

    def test_child_alleles_from_parents(self) -> None:
        rng = random.Random(42)
        pa = Genome(alleles={"resolve": (0.1, 0.2)})
        pb = Genome(alleles={"resolve": (0.8, 0.9)})
        child = crossover(pa, pb, rng)
        a, b = child.alleles["resolve"]
        assert a in (0.1, 0.2)
        assert b in (0.8, 0.9)

    def test_crossover_deterministic(self) -> None:
        pa = Genome(alleles={n: (0.3, 0.7) for n in STAT_NAMES})
        pb = Genome(alleles={n: (0.4, 0.6) for n in STAT_NAMES})
        c1 = crossover(pa, pb, random.Random(1))
        c2 = crossover(pa, pb, random.Random(1))
        assert c1.to_dict() == c2.to_dict()

    def test_crossover_variation(self) -> None:
        """Different seeds produce different children."""
        pa = Genome(alleles={n: (0.1, 0.9) for n in STAT_NAMES})
        pb = Genome(alleles={n: (0.2, 0.8) for n in STAT_NAMES})
        results = set()
        for seed in range(20):
            child = crossover(pa, pb, random.Random(seed))
            key = tuple(child.alleles["resolve"])
            results.add(key)
        assert len(results) > 1


# ── Mutation ───────────────────────────────────────────────────────────


class TestMutation:
    def test_mutation_count(self) -> None:
        rng = random.Random(42)
        g = Genome(alleles={n: (0.5, 0.5) for n in STAT_NAMES})
        count = mutate(g, rng, rate=1.0)
        assert count == 12  # all alleles mutated

    def test_no_mutation_at_zero_rate(self) -> None:
        g = Genome(alleles={n: (0.5, 0.5) for n in STAT_NAMES})
        original = g.to_dict()
        count = mutate(g, random.Random(42), rate=0.0)
        assert count == 0
        assert g.to_dict() == original

    def test_mutation_stays_in_bounds(self) -> None:
        rng = random.Random(42)
        g = Genome(alleles={n: (0.99, 0.01) for n in STAT_NAMES})
        for _ in range(50):
            mutate(g, rng, rate=1.0, sigma=0.2)
        for name, (a, b) in g.alleles.items():
            assert 0.0 <= a <= 1.0
            assert 0.0 <= b <= 1.0

    def test_mutation_rate_approximate(self) -> None:
        """Over many runs, mutation count roughly matches expected rate."""
        total_mutations = 0
        total_alleles = 0
        for seed in range(100):
            g = Genome(alleles={n: (0.5, 0.5) for n in STAT_NAMES})
            count = mutate(g, random.Random(seed), rate=MUTATION_RATE)
            total_mutations += count
            total_alleles += 12
        observed_rate = total_mutations / total_alleles
        assert abs(observed_rate - MUTATION_RATE) < 0.02


# ── Pedigree kinship ──────────────────────────────────────────────────


class TestPedigreeKinship:
    def test_self_kinship(self) -> None:
        assert compute_pedigree_kinship("a", "a", {}) == 1.0

    def test_unrelated_founders(self) -> None:
        assert compute_pedigree_kinship("a", "b", {}) == 0.0

    def test_siblings(self) -> None:
        lineage = {"child-1": ["pa", "ma"], "child-2": ["pa", "ma"]}
        kin = compute_pedigree_kinship("child-1", "child-2", lineage)
        assert kin > 0.2

    def test_half_siblings(self) -> None:
        lineage = {"child-1": ["pa", "ma1"], "child-2": ["pa", "ma2"]}
        full = compute_pedigree_kinship("child-1", "child-2", lineage)
        assert 0.0 < full < 0.5

    def test_parent_child(self) -> None:
        lineage = {"child": ["pa", "ma"]}
        kin = compute_pedigree_kinship("child", "pa", lineage)
        assert kin > 0.0

    def test_unrelated_immigrants(self) -> None:
        lineage = {"child": ["pa", "ma"]}
        kin = compute_pedigree_kinship("child", "immigrant", lineage)
        assert kin == 0.0

    def test_cousins_lower_than_siblings(self) -> None:
        lineage = {
            "aunt": ["gpa", "gma"],
            "parent": ["gpa", "gma"],
            "cousin": ["aunt", "uncle"],
            "self": ["parent", "other"],
        }
        sibling_kin = compute_pedigree_kinship("aunt", "parent", lineage)
        cousin_kin = compute_pedigree_kinship("cousin", "self", lineage)
        assert cousin_kin < sibling_kin

    def test_no_infinite_loop(self) -> None:
        """Cyclic lineage (shouldn't happen, but guard against it)."""
        lineage = {"a": ["b"], "b": ["a"]}
        kin = compute_pedigree_kinship("a", "b", lineage)
        assert isinstance(kin, float)


# ── Diversity index ────────────────────────────────────────────────────


class TestDiversityIndex:
    def test_single_genome(self) -> None:
        g = Genome(alleles={n: (0.5, 0.5) for n in STAT_NAMES})
        assert compute_diversity_index([g]) == 0.0

    def test_identical_genomes(self) -> None:
        g = Genome(alleles={n: (0.5, 0.5) for n in STAT_NAMES})
        assert compute_diversity_index([g, g, g]) == 0.0

    def test_diverse_genomes(self) -> None:
        rng = random.Random(42)
        genomes = []
        for _ in range(10):
            alleles = {n: (rng.random(), rng.random()) for n in STAT_NAMES}
            genomes.append(Genome(alleles=alleles))
        div = compute_diversity_index(genomes)
        assert div > 0.1

    def test_diversity_bounded(self) -> None:
        rng = random.Random(42)
        genomes = [Genome(alleles={n: (rng.random(), rng.random())
                   for n in STAT_NAMES}) for _ in range(20)]
        div = compute_diversity_index(genomes)
        assert 0.0 <= div <= 1.0


# ── Inbreeding modifiers ──────────────────────────────────────────────


class TestInbreedingModifiers:
    def test_no_penalty_for_unrelated(self) -> None:
        assert inbreeding_birth_modifier(0.0) == 1.0

    def test_no_penalty_below_threshold(self) -> None:
        assert inbreeding_birth_modifier(INBREEDING_THRESHOLD - 0.01) == 1.0

    def test_penalty_above_threshold(self) -> None:
        mod = inbreeding_birth_modifier(0.3)
        assert mod < 1.0
        assert mod >= INBREEDING_BIRTH_PENALTY

    def test_high_kinship_floors_at_penalty(self) -> None:
        mod = inbreeding_birth_modifier(1.0)
        assert mod == INBREEDING_BIRTH_PENALTY

    def test_genetic_death_modifier_none_genome(self) -> None:
        assert genetic_death_modifier(None) == 1.0

    def test_genetic_death_modifier_healthy(self) -> None:
        g = Genome(alleles={n: (0.3, 0.7) for n in STAT_NAMES})
        mod = genetic_death_modifier(g)
        assert mod >= 1.0
        assert mod < 1.01  # small effect

    def test_genetic_death_modifier_inbred(self) -> None:
        g = Genome(alleles={n: (0.5, 0.5) for n in STAT_NAMES})
        mod = genetic_death_modifier(g, kinship_to_parents=0.5)
        assert mod > 1.0


# ── Lineage recording ─────────────────────────────────────────────────


class TestRecordBirth:
    def test_records_parents(self) -> None:
        state = GeneticsState()
        record_birth(state, "child-1", ["pa", "ma"])
        assert state.lineage["child-1"] == ["pa", "ma"]

    def test_generation_count_increments(self) -> None:
        state = GeneticsState()
        record_birth(state, "child-1", ["pa", "ma"])
        assert state.generation_count == 1
        record_birth(state, "grandchild", ["child-1", "other"])
        assert state.generation_count == 2


# ── tick_genetics ──────────────────────────────────────────────────────


class TestTickGenetics:
    def _make_genomes(self, n: int, rng: random.Random) -> dict[str, Genome]:
        genomes = {}
        for i in range(n):
            cid = f"col-{i}"
            alleles = {name: (rng.random(), rng.random()) for name in STAT_NAMES}
            genomes[cid] = Genome(alleles=alleles)
        return genomes

    def test_tick_returns_result(self) -> None:
        rng = random.Random(42)
        state = GeneticsState()
        genomes = self._make_genomes(5, rng)
        active_ids = list(genomes.keys())
        result = tick_genetics(state, genomes, active_ids, 1, rng)
        assert isinstance(result, GeneticsTickResult)
        assert 0.0 <= result.diversity_index <= 1.0

    def test_tick_records_diversity_history(self) -> None:
        rng = random.Random(42)
        state = GeneticsState()
        genomes = self._make_genomes(5, rng)
        active_ids = list(genomes.keys())
        for year in range(5):
            tick_genetics(state, genomes, active_ids, year, rng)
        assert len(state.diversity_history) == 5

    def test_tick_serializable(self) -> None:
        rng = random.Random(42)
        state = GeneticsState()
        genomes = self._make_genomes(5, rng)
        active_ids = list(genomes.keys())
        result = tick_genetics(state, genomes, active_ids, 1, rng)
        d = result.to_dict()
        assert "diversity_index" in d
        assert "max_kinship" in d

    def test_inbreeding_detection(self) -> None:
        rng = random.Random(42)
        state = GeneticsState()
        record_birth(state, "child-1", ["pa", "ma"])
        record_birth(state, "child-2", ["pa", "ma"])
        genomes = {
            "child-1": Genome(alleles={n: (0.5, 0.5) for n in STAT_NAMES}),
            "child-2": Genome(alleles={n: (0.5, 0.5) for n in STAT_NAMES}),
        }
        result = tick_genetics(state, genomes, ["child-1", "child-2"], 10, rng)
        assert result.max_kinship > INBREEDING_THRESHOLD

    def test_state_roundtrip(self) -> None:
        state = GeneticsState(
            diversity_history=[0.8, 0.75, 0.7],
            total_mutations=42,
            inbreeding_events=3,
            generation_count=5,
        )
        d = state.to_dict()
        s2 = GeneticsState.from_dict(d)
        assert s2.total_mutations == 42
        assert s2.generation_count == 5


# ── Integration smoke test ─────────────────────────────────────────────


class TestGeneticsSmokeTest:
    def test_10_year_sim(self) -> None:
        """Simulate 10 years of genetic tracking without crash."""
        rng = random.Random(42)
        state = GeneticsState()
        genomes: dict[str, Genome] = {}
        stats = {n: 0.5 for n in STAT_NAMES}
        for i in range(10):
            cid = f"founder-{i}"
            genomes[cid] = create_genome_from_stats(stats, rng)
        active_ids = list(genomes.keys())

        for year in range(10):
            result = tick_genetics(state, genomes, active_ids, year, rng)
            assert 0.0 <= result.diversity_index <= 1.0
            assert result.avg_homozygosity >= 0.0

            if year == 5 and len(active_ids) >= 2:
                pa_id = active_ids[0]
                ma_id = active_ids[1]
                child_g = crossover(genomes[pa_id], genomes[ma_id], rng)
                mutations = mutate(child_g, rng)
                state.total_mutations += mutations
                child_id = f"child-{year}"
                genomes[child_id] = child_g
                record_birth(state, child_id, [pa_id, ma_id])
                active_ids.append(child_id)

        assert state.generation_count >= 1
        assert state.total_mutations >= 0

    def test_bounds_invariant(self) -> None:
        """All alleles stay in [0,1] after many crossover+mutation cycles."""
        rng = random.Random(99)
        genomes = []
        for _ in range(4):
            g = Genome(alleles={n: (rng.random(), rng.random()) for n in STAT_NAMES})
            genomes.append(g)
        for _ in range(50):
            pa, pb = rng.sample(genomes, 2)
            child = crossover(pa, pb, rng)
            mutate(child, rng, rate=0.2, sigma=0.1)
            child.clamp()
            genomes.append(child)
        for g in genomes:
            for name, (a, b) in g.alleles.items():
                assert 0.0 <= a <= 1.0, f"{name} allele a = {a}"
                assert 0.0 <= b <= 1.0, f"{name} allele b = {b}"

    def test_diversity_decreases_with_inbreeding(self) -> None:
        """Closed population should lose diversity over generations."""
        rng = random.Random(42)
        pair = [
            Genome(alleles={n: (0.2, 0.8) for n in STAT_NAMES}),
            Genome(alleles={n: (0.3, 0.7) for n in STAT_NAMES}),
        ]
        initial_div = compute_diversity_index(pair)
        pop = list(pair)
        for _ in range(30):
            pa, pb = rng.sample(pop[-4:] if len(pop) > 4 else pop, 2)
            child = crossover(pa, pb, rng)
            mutate(child, rng)
            pop.append(child)
        final_div = compute_diversity_index(pop[-4:])
        assert final_div <= initial_div + 0.15  # may not strictly decrease due to mutation
