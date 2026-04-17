"""Tests for the genetics organ (engine v11.0)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.genetics import (
    Genome,
    GenePool,
    GeneticsState,
    GeneticsYearContext,
    GeneticsTickResult,
    SelectionReport,
    STAT_NAMES,
    MUTATION_SIGMA,
    CROSSOVER_PROB,
    GENOTYPE_PULL_STRENGTH,
    MAX_TRAIT_BONUS,
    EARTH_ALLELE_MEAN,
    EARTH_ALLELE_SIGMA,
    create_founder_genome,
    inherit_genome,
    create_immigrant_genome,
    compute_diversity,
    compute_selection_report,
    compute_genetic_pull,
    compute_gene_pool_stats,
    tick_genetics,
    apply_genetic_pull_to_stat,
    _clamp,
    _allele_distance,
)


# ---------------------------------------------------------------------------
# Genome creation
# ---------------------------------------------------------------------------

class TestGenome:
    def test_founder_genome_from_stats(self):
        rng = random.Random(42)
        stats = {"resolve": 0.8, "improvisation": 0.3, "empathy": 0.5,
                 "hoarding": 0.4, "faith": 0.6, "paranoia": 0.2}
        g = create_founder_genome("kira-sol", stats, rng)
        assert isinstance(g, Genome)
        assert g.generation == 0
        assert g.parent_ids == []
        assert g.mutation_count == 0
        for stat in STAT_NAMES:
            assert 0.0 <= g.alleles[stat] <= 1.0

    def test_founder_genome_close_to_stats(self):
        """Founders' genotypes should be close to their phenotype."""
        rng = random.Random(42)
        stats = {"resolve": 0.9, "improvisation": 0.1, "empathy": 0.5,
                 "hoarding": 0.5, "faith": 0.5, "paranoia": 0.5}
        g = create_founder_genome("test", stats, rng)
        for stat in STAT_NAMES:
            assert abs(g.alleles[stat] - stats[stat]) < 0.2

    def test_founder_genome_deterministic(self):
        stats = {"resolve": 0.5, "improvisation": 0.5, "empathy": 0.5,
                 "hoarding": 0.5, "faith": 0.5, "paranoia": 0.5}
        g1 = create_founder_genome("a", stats, random.Random(99))
        g2 = create_founder_genome("a", stats, random.Random(99))
        assert g1.alleles == g2.alleles

    def test_genome_serialization(self):
        g = Genome(alleles={"resolve": 0.5, "improvisation": 0.6,
                            "empathy": 0.7, "hoarding": 0.3,
                            "faith": 0.4, "paranoia": 0.2},
                   generation=2, parent_ids=["a", "b"], mutation_count=1)
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        assert g2.generation == 2
        assert g2.parent_ids == ["a", "b"]
        assert g2.mutation_count == 1
        for stat in STAT_NAMES:
            assert abs(g2.alleles[stat] - g.alleles[stat]) < 1e-5


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------

class TestInheritance:
    def test_child_genome_has_correct_generation(self):
        rng = random.Random(42)
        p1 = Genome(alleles={s: 0.5 for s in STAT_NAMES}, generation=0)
        p2 = Genome(alleles={s: 0.5 for s in STAT_NAMES}, generation=1)
        child = inherit_genome(p1, p2, rng)
        assert child.generation == 2

    def test_child_alleles_in_bounds(self):
        rng = random.Random(42)
        p1 = Genome(alleles={s: 0.0 for s in STAT_NAMES})
        p2 = Genome(alleles={s: 1.0 for s in STAT_NAMES})
        for _ in range(100):
            child = inherit_genome(p1, p2, rng)
            for stat in STAT_NAMES:
                assert 0.0 <= child.alleles[stat] <= 1.0

    def test_child_inherits_from_parents(self):
        """Over many children, alleles should cluster near parent values."""
        rng = random.Random(42)
        p1 = Genome(alleles={s: 0.2 for s in STAT_NAMES})
        p2 = Genome(alleles={s: 0.8 for s in STAT_NAMES})
        means = {s: 0.0 for s in STAT_NAMES}
        n = 200
        for _ in range(n):
            child = inherit_genome(p1, p2, rng)
            for s in STAT_NAMES:
                means[s] += child.alleles[s]
        for s in STAT_NAMES:
            mean = means[s] / n
            assert 0.3 < mean < 0.7, f"{s} mean {mean} not near parent midpoint"

    def test_mutation_introduces_variation(self):
        """Children from identical parents should still vary."""
        rng = random.Random(42)
        p = Genome(alleles={s: 0.5 for s in STAT_NAMES})
        children = [inherit_genome(p, p, rng) for _ in range(50)]
        for stat in STAT_NAMES:
            values = [c.alleles[stat] for c in children]
            assert max(values) - min(values) > 0.01

    def test_parent_ids_tracked(self):
        p1 = Genome(alleles={s: 0.5 for s in STAT_NAMES}, parent_ids=["gp1"])
        p2 = Genome(alleles={s: 0.5 for s in STAT_NAMES}, parent_ids=["gp2"])
        child = inherit_genome(p1, p2, random.Random(42))
        # Parent IDs should include grandparent IDs (from the inheritance)
        assert len(child.parent_ids) <= 4


# ---------------------------------------------------------------------------
# Immigrant genomes
# ---------------------------------------------------------------------------

class TestImmigrant:
    def test_immigrant_genome_in_bounds(self):
        rng = random.Random(42)
        for _ in range(100):
            g = create_immigrant_genome(rng)
            for stat in STAT_NAMES:
                assert 0.0 <= g.alleles[stat] <= 1.0

    def test_immigrant_is_generation_zero(self):
        g = create_immigrant_genome(random.Random(42))
        assert g.generation == 0
        assert g.parent_ids == []

    def test_immigrants_are_diverse(self):
        """Multiple immigrants should not be clones."""
        rng = random.Random(42)
        immigrants = [create_immigrant_genome(rng) for _ in range(20)]
        for stat in STAT_NAMES:
            values = [g.alleles[stat] for g in immigrants]
            assert max(values) - min(values) > 0.1


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

class TestDiversity:
    def test_identical_genomes_zero_diversity(self):
        g = Genome(alleles={s: 0.5 for s in STAT_NAMES})
        genomes = {"a": g, "b": g, "c": g}
        d = compute_diversity(genomes, ["a", "b", "c"])
        assert d == 0.0

    def test_max_diverse_genomes(self):
        g1 = Genome(alleles={s: 0.0 for s in STAT_NAMES})
        g2 = Genome(alleles={s: 1.0 for s in STAT_NAMES})
        genomes = {"a": g1, "b": g2}
        d = compute_diversity(genomes, ["a", "b"])
        assert d > 0.9

    def test_single_colonist_returns_one(self):
        g = Genome(alleles={s: 0.5 for s in STAT_NAMES})
        assert compute_diversity({"a": g}, ["a"]) == 1.0

    def test_diversity_zero_to_one(self):
        rng = random.Random(42)
        genomes = {f"c{i}": create_immigrant_genome(rng) for i in range(20)}
        d = compute_diversity(genomes, list(genomes.keys()))
        assert 0.0 <= d <= 1.0


# ---------------------------------------------------------------------------
# Selection report
# ---------------------------------------------------------------------------

class TestSelectionReport:
    def test_no_deaths_empty_report(self):
        genomes = {"a": Genome(alleles={s: 0.5 for s in STAT_NAMES})}
        r = compute_selection_report(genomes, ["a"], [], year=10)
        assert r.deaths_this_year == 0
        assert r.pressure_direction == {}

    def test_deaths_produce_pressure(self):
        survivor = Genome(alleles={s: 0.8 for s in STAT_NAMES})
        deceased = Genome(alleles={s: 0.2 for s in STAT_NAMES})
        genomes = {"alive": survivor, "dead": deceased}
        deaths = [{"colonist_id": "dead", "cause": "asphyxiation"}]
        r = compute_selection_report(genomes, ["alive", "dead"], deaths, year=10)
        assert r.deaths_this_year == 1
        for stat in STAT_NAMES:
            assert r.pressure_direction[stat] > 0  # survivors had higher values

    def test_report_serialization(self):
        r = SelectionReport(year=5, deaths_this_year=2,
                           survivor_mean={"resolve": 0.6},
                           deceased_mean={"resolve": 0.3},
                           pressure_direction={"resolve": 0.3})
        d = r.to_dict()
        assert d["year"] == 5
        assert d["deaths_this_year"] == 2


# ---------------------------------------------------------------------------
# Genetic pull
# ---------------------------------------------------------------------------

class TestGeneticPull:
    def test_pull_returns_allele_targets(self):
        g = Genome(alleles={"resolve": 0.8, "improvisation": 0.3,
                            "empathy": 0.5, "hoarding": 0.5,
                            "faith": 0.5, "paranoia": 0.5})
        pull = compute_genetic_pull(g)
        assert pull["resolve"] == 0.8
        assert pull["improvisation"] == 0.3

    def test_apply_pull_moves_toward_target(self):
        # Current stat = 0.3, genotype target = 0.8
        delta = apply_genetic_pull_to_stat(0.3, 0.8)
        assert delta > 0  # should pull upward
        assert delta <= MAX_TRAIT_BONUS

    def test_apply_pull_moves_down(self):
        delta = apply_genetic_pull_to_stat(0.9, 0.2)
        assert delta < 0

    def test_pull_clamped(self):
        delta = apply_genetic_pull_to_stat(0.0, 1.0)
        assert abs(delta) <= MAX_TRAIT_BONUS

    def test_no_pull_when_at_target(self):
        delta = apply_genetic_pull_to_stat(0.5, 0.5)
        assert delta == 0.0


# ---------------------------------------------------------------------------
# Gene pool stats
# ---------------------------------------------------------------------------

class TestGenePoolStats:
    def test_empty_pool(self):
        pool = compute_gene_pool_stats({}, [])
        assert pool.diversity_index == 1.0
        assert pool.generation_count == 0

    def test_pool_with_genomes(self):
        rng = random.Random(42)
        genomes = {f"c{i}": create_immigrant_genome(rng) for i in range(10)}
        pool = compute_gene_pool_stats(genomes, list(genomes.keys()))
        assert 0.0 <= pool.diversity_index <= 1.0
        assert all(s in pool.mean_alleles for s in STAT_NAMES)
        assert all(s in pool.allele_variance for s in STAT_NAMES)

    def test_pool_serialization(self):
        pool = GenePool(diversity_index=0.75,
                       mean_alleles={"resolve": 0.5},
                       allele_variance={"resolve": 0.01})
        d = pool.to_dict()
        p2 = GenePool.from_dict(d)
        assert abs(p2.diversity_index - 0.75) < 1e-5


# ---------------------------------------------------------------------------
# tick_genetics integration
# ---------------------------------------------------------------------------

class TestTickGenetics:
    def _make_state(self, n: int = 10) -> tuple[GeneticsState, list[str]]:
        rng = random.Random(42)
        state = GeneticsState()
        ids = []
        for i in range(n):
            cid = f"c{i}"
            ids.append(cid)
            stats = {s: rng.random() for s in STAT_NAMES}
            state.genomes[cid] = create_founder_genome(cid, stats, rng)
        return state, ids

    def test_tick_no_events(self):
        state, ids = self._make_state()
        ctx = GeneticsYearContext(
            year=1, active_ids=ids, deaths=[], birth_ids=[],
            birth_parent_map={}, immigrant_ids=[],
        )
        result = tick_genetics(state, ctx, random.Random(42))
        assert result.new_genomes == []
        assert 0.0 <= result.diversity_index <= 1.0

    def test_tick_with_births(self):
        state, ids = self._make_state()
        ctx = GeneticsYearContext(
            year=5, active_ids=ids, deaths=[],
            birth_ids=["child-1"],
            birth_parent_map={"child-1": ("c0", "c1")},
            immigrant_ids=[],
        )
        result = tick_genetics(state, ctx, random.Random(42))
        assert "child-1" in result.new_genomes
        assert "child-1" in state.genomes
        assert state.genomes["child-1"].generation == 1
        assert len(result.events) >= 1

    def test_tick_with_immigrants(self):
        state, ids = self._make_state()
        ctx = GeneticsYearContext(
            year=10, active_ids=ids, deaths=[],
            birth_ids=[], birth_parent_map={},
            immigrant_ids=["imm-1", "imm-2"],
        )
        result = tick_genetics(state, ctx, random.Random(42))
        assert "imm-1" in result.new_genomes
        assert "imm-2" in result.new_genomes
        assert "imm-1" in state.genomes
        assert state.genomes["imm-1"].generation == 0

    def test_tick_with_deaths(self):
        state, ids = self._make_state()
        deaths = [{"colonist_id": "c0", "cause": "asphyxiation"}]
        ctx = GeneticsYearContext(
            year=20, active_ids=ids, deaths=deaths,
            birth_ids=[], birth_parent_map={}, immigrant_ids=[],
        )
        result = tick_genetics(state, ctx, random.Random(42))
        assert result.selection_report["deaths_this_year"] == 1

    def test_tick_genetic_pull_computed(self):
        state, ids = self._make_state()
        ctx = GeneticsYearContext(
            year=1, active_ids=ids, deaths=[],
            birth_ids=[], birth_parent_map={}, immigrant_ids=[],
        )
        result = tick_genetics(state, ctx, random.Random(42))
        assert len(result.genetic_pull) == len(ids)
        for cid in ids:
            assert all(s in result.genetic_pull[cid] for s in STAT_NAMES)

    def test_tick_result_serialization(self):
        state, ids = self._make_state()
        ctx = GeneticsYearContext(
            year=1, active_ids=ids, deaths=[],
            birth_ids=[], birth_parent_map={}, immigrant_ids=[],
        )
        result = tick_genetics(state, ctx, random.Random(42))
        d = result.to_dict()
        assert "diversity_index" in d
        assert "selection_report" in d

    def test_state_serialization(self):
        state, _ = self._make_state(5)
        d = state.to_dict()
        s2 = GeneticsState.from_dict(d)
        assert len(s2.genomes) == 5

    def test_low_diversity_warning(self):
        """When all genomes are identical, a warning should appear."""
        state = GeneticsState()
        ids = [f"c{i}" for i in range(5)]
        for cid in ids:
            state.genomes[cid] = Genome(alleles={s: 0.5 for s in STAT_NAMES})
        ctx = GeneticsYearContext(
            year=50, active_ids=ids, deaths=[],
            birth_ids=[], birth_parent_map={}, immigrant_ids=[],
        )
        result = tick_genetics(state, ctx, random.Random(42))
        assert any("diversity" in e.lower() for e in result.events)


# ---------------------------------------------------------------------------
# Property-based invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_alleles_always_bounded(self):
        """All alleles must be in [0, 1] regardless of input."""
        rng = random.Random(42)
        for _ in range(500):
            p1 = Genome(alleles={s: rng.random() for s in STAT_NAMES})
            p2 = Genome(alleles={s: rng.random() for s in STAT_NAMES})
            child = inherit_genome(p1, p2, rng)
            for s in STAT_NAMES:
                assert 0.0 <= child.alleles[s] <= 1.0, f"{s}={child.alleles[s]}"

    def test_diversity_bounded(self):
        rng = random.Random(42)
        for _ in range(50):
            n = rng.randint(2, 20)
            genomes = {f"c{i}": create_immigrant_genome(rng) for i in range(n)}
            d = compute_diversity(genomes, list(genomes.keys()))
            assert 0.0 <= d <= 1.0

    def test_genetic_pull_bounded(self):
        rng = random.Random(42)
        for _ in range(100):
            current = rng.random()
            target = rng.random()
            delta = apply_genetic_pull_to_stat(current, target)
            assert abs(delta) <= MAX_TRAIT_BONUS

    def test_clamp_idempotent(self):
        for v in [-0.1, 0.0, 0.5, 1.0, 1.1]:
            c = _clamp(v)
            assert 0.0 <= c <= 1.0
            assert _clamp(c) == c

    def test_10_year_smoke_run(self):
        """Run tick_genetics for 10 years without crash."""
        state = GeneticsState()
        rng = random.Random(42)
        ids = [f"c{i}" for i in range(10)]
        for cid in ids:
            state.genomes[cid] = create_founder_genome(
                cid, {s: rng.random() for s in STAT_NAMES}, rng)
        for year in range(1, 11):
            births = []
            parent_map = {}
            immigrants = []
            deaths = []
            if year == 3:
                births = ["child-1"]
                parent_map = {"child-1": ("c0", "c1")}
                ids.append("child-1")
            if year == 5:
                immigrants = ["imm-1"]
                ids.append("imm-1")
            if year == 7:
                deaths = [{"colonist_id": "c9", "cause": "dust_storm"}]
                ids = [i for i in ids if i != "c9"]
            ctx = GeneticsYearContext(
                year=year, active_ids=ids, deaths=deaths,
                birth_ids=births, birth_parent_map=parent_map,
                immigrant_ids=immigrants,
            )
            result = tick_genetics(state, ctx, rng)
            assert 0.0 <= result.diversity_index <= 1.0
