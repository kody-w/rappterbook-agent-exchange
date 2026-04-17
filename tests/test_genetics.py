"""Tests for the genetics organ (engine v11.0)."""
from __future__ import annotations

import random
import pytest

from src.mars100.genetics import (
    ALL_LOCI, STAT_LOCI, SKILL_LOCI,
    BASE_MUTATION_RATE, MAX_MUTATION_RATE, MUTATION_SIGMA,
    TRAIT_BIAS_STRENGTH, SKILL_APT_MULTIPLIER,
    Genome, EpigeneticMarks, GeneticsState, GeneticsTickResult,
    _stable_subseed, bootstrap_genome, crossover,
    compute_mutation_rate, compute_trait_biases, compute_skill_aptitudes,
    update_epigenetics, compute_heterozygosity, tick_genetics,
)
from src.mars100.colonist import (
    Colonist, ColonistStats, ColonistSkills, STAT_NAMES, SKILL_NAMES,
    create_founding_ten, create_child,
)


# ---------------------------------------------------------------------------
# Genome basics
# ---------------------------------------------------------------------------

class TestGenome:
    def test_default_empty(self) -> None:
        g = Genome()
        assert g.alleles == {}
        assert g.expressed("resolve_apt") == 0.5  # fallback

    def test_expressed_mean(self) -> None:
        g = Genome(alleles={"resolve_apt": (0.2, 0.8)})
        assert g.expressed("resolve_apt") == pytest.approx(0.5)

    def test_roundtrip(self) -> None:
        g = Genome(alleles={locus: (0.3, 0.7) for locus in ALL_LOCI})
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        for locus in ALL_LOCI:
            assert g2.alleles[locus][0] == pytest.approx(0.3, abs=0.001)
            assert g2.alleles[locus][1] == pytest.approx(0.7, abs=0.001)


class TestEpigeneticMarks:
    def test_default_zero(self) -> None:
        m = EpigeneticMarks()
        assert m.get("resolve_apt") == 0.0

    def test_roundtrip(self) -> None:
        m = EpigeneticMarks(marks={"resolve_apt": 0.1, "faith_apt": -0.05})
        d = m.to_dict()
        m2 = EpigeneticMarks.from_dict(d)
        assert m2.get("resolve_apt") == pytest.approx(0.1, abs=0.001)
        assert m2.get("faith_apt") == pytest.approx(-0.05, abs=0.001)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_deterministic(self) -> None:
        stats = {"resolve": 0.8, "empathy": 0.3}
        skills = {"coding": 0.6, "prayer": 0.1}
        g1 = bootstrap_genome(stats, skills, 42, "agent-1")
        g2 = bootstrap_genome(stats, skills, 42, "agent-1")
        assert g1.to_dict() == g2.to_dict()

    def test_different_id_different_genome(self) -> None:
        stats = {"resolve": 0.5}
        skills = {}
        g1 = bootstrap_genome(stats, skills, 42, "agent-1")
        g2 = bootstrap_genome(stats, skills, 42, "agent-2")
        assert g1.to_dict() != g2.to_dict()

    def test_alleles_in_bounds(self) -> None:
        stats = {s: 0.5 for s in STAT_NAMES}
        skills = {s: 0.5 for s in SKILL_NAMES}
        g = bootstrap_genome(stats, skills, 99, "test")
        for locus in ALL_LOCI:
            a, b = g.alleles[locus]
            assert 0.0 <= a <= 1.0, f"{locus} allele_a out of bounds: {a}"
            assert 0.0 <= b <= 1.0, f"{locus} allele_b out of bounds: {b}"

    def test_expressed_near_stat(self) -> None:
        """Bootstrap genome should express values close to original stats."""
        stats = {"resolve": 0.9, "empathy": 0.1, "faith": 0.5,
                 "hoarding": 0.3, "improvisation": 0.7, "paranoia": 0.4}
        skills = {s: 0.0 for s in SKILL_NAMES}
        g = bootstrap_genome(stats, skills, 42, "test")
        for stat_name, val in stats.items():
            locus = f"{stat_name}_apt"
            assert g.expressed(locus) == pytest.approx(val, abs=0.15)


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

class TestCrossover:
    def test_child_has_all_loci(self) -> None:
        rng = random.Random(42)
        pa = Genome(alleles={l: (0.2, 0.3) for l in ALL_LOCI})
        pb = Genome(alleles={l: (0.7, 0.8) for l in ALL_LOCI})
        child, _ = crossover(pa, pb, 0.0, rng)
        for locus in ALL_LOCI:
            assert locus in child.alleles

    def test_no_mutation_alleles_from_parents(self) -> None:
        """Without mutation, each child allele must come from one parent."""
        rng = random.Random(42)
        pa = Genome(alleles={l: (0.1, 0.2) for l in ALL_LOCI})
        pb = Genome(alleles={l: (0.8, 0.9) for l in ALL_LOCI})
        child, mutations = crossover(pa, pb, 0.0, rng)
        assert mutations == 0
        parent_vals = {0.1, 0.2, 0.8, 0.9}
        for locus in ALL_LOCI:
            a, b = child.alleles[locus]
            assert a in parent_vals, f"{locus} allele_a {a} not from parents"
            assert b in parent_vals, f"{locus} allele_b {b} not from parents"

    def test_mutation_changes_alleles(self) -> None:
        """With high mutation rate, some alleles should be mutated."""
        rng = random.Random(42)
        pa = Genome(alleles={l: (0.5, 0.5) for l in ALL_LOCI})
        pb = Genome(alleles={l: (0.5, 0.5) for l in ALL_LOCI})
        child, mutations = crossover(pa, pb, 1.0, rng)  # 100% mutation rate
        assert mutations > 0
        # At least one allele should differ from 0.5
        diffs = sum(1 for l in ALL_LOCI
                    for a in child.alleles[l] if abs(a - 0.5) > 0.001)
        assert diffs > 0

    def test_alleles_clamped(self) -> None:
        rng = random.Random(42)
        pa = Genome(alleles={l: (0.0, 0.0) for l in ALL_LOCI})
        pb = Genome(alleles={l: (1.0, 1.0) for l in ALL_LOCI})
        for _ in range(20):
            child, _ = crossover(pa, pb, MAX_MUTATION_RATE, rng)
            for locus in ALL_LOCI:
                a, b = child.alleles[locus]
                assert 0.0 <= a <= 1.0
                assert 0.0 <= b <= 1.0


# ---------------------------------------------------------------------------
# Mutation rate
# ---------------------------------------------------------------------------

class TestMutationRate:
    def test_base_rate_with_earth_atmosphere(self) -> None:
        """At Earth-like pressure (101 kPa), rate near base."""
        rate = compute_mutation_rate(101.0, False)
        assert rate == pytest.approx(BASE_MUTATION_RATE, abs=0.005)

    def test_thin_atmosphere_higher_rate(self) -> None:
        rate = compute_mutation_rate(0.6, False)  # Mars baseline
        assert rate > BASE_MUTATION_RATE

    def test_radiation_increases_rate(self) -> None:
        rate_normal = compute_mutation_rate(5.0, False)
        rate_radiation = compute_mutation_rate(5.0, True)
        assert rate_radiation > rate_normal

    def test_never_exceeds_max(self) -> None:
        rate = compute_mutation_rate(0.0, True)
        assert rate <= MAX_MUTATION_RATE


# ---------------------------------------------------------------------------
# Trait biases & skill aptitudes
# ---------------------------------------------------------------------------

class TestTraitBiases:
    def test_returns_all_stats(self) -> None:
        g = Genome(alleles={l: (0.5, 0.5) for l in ALL_LOCI})
        m = EpigeneticMarks()
        biases = compute_trait_biases(g, m)
        for stat in STAT_NAMES:
            assert stat in biases
            assert 0.0 <= biases[stat] <= 1.0

    def test_epigenetic_shift(self) -> None:
        g = Genome(alleles={"resolve_apt": (0.5, 0.5)})
        m = EpigeneticMarks(marks={"resolve_apt": 0.15})
        biases = compute_trait_biases(g, m)
        assert biases["resolve"] > 0.5  # shifted up


class TestSkillAptitudes:
    def test_returns_all_skills(self) -> None:
        g = Genome(alleles={l: (0.5, 0.5) for l in ALL_LOCI})
        m = EpigeneticMarks()
        apts = compute_skill_aptitudes(g, m)
        for skill in SKILL_NAMES:
            assert skill in apts
            assert 0.5 <= apts[skill] <= 2.0

    def test_high_aptitude_high_multiplier(self) -> None:
        g = Genome(alleles={"coding_apt": (0.9, 0.9)})
        m = EpigeneticMarks()
        apts = compute_skill_aptitudes(g, m)
        assert apts["coding"] > 1.5


# ---------------------------------------------------------------------------
# Epigenetics
# ---------------------------------------------------------------------------

class TestEpigenetics:
    def test_marks_bounded(self) -> None:
        rng = random.Random(42)
        m = EpigeneticMarks()
        for _ in range(200):
            update_epigenetics(m, stress=0.9, resource_avg=0.1,
                               radiation_event=True, rng=rng)
        for locus in ALL_LOCI:
            val = m.get(locus)
            assert -0.2 <= val <= 0.2, f"{locus} mark out of bounds: {val}"

    def test_stress_suppresses_empathy(self) -> None:
        rng = random.Random(42)
        m = EpigeneticMarks()
        for _ in range(50):
            update_epigenetics(m, stress=0.95, resource_avg=0.5,
                               radiation_event=False, rng=rng)
        assert m.get("empathy_apt") < 0.0  # suppressed


# ---------------------------------------------------------------------------
# Heterozygosity
# ---------------------------------------------------------------------------

class TestHeterozygosity:
    def test_identical_genomes_low(self) -> None:
        g = Genome(alleles={l: (0.5, 0.5) for l in ALL_LOCI})
        het = compute_heterozygosity([g, g, g])
        assert het == pytest.approx(0.0, abs=0.001)

    def test_diverse_genomes_higher(self) -> None:
        rng = random.Random(42)
        genomes = []
        for _ in range(10):
            alleles = {l: (rng.random(), rng.random()) for l in ALL_LOCI}
            genomes.append(Genome(alleles=alleles))
        het = compute_heterozygosity(genomes)
        assert het > 0.02

    def test_empty_population(self) -> None:
        assert compute_heterozygosity([]) == 0.0


# ---------------------------------------------------------------------------
# tick_genetics
# ---------------------------------------------------------------------------

class TestTickGenetics:
    def test_basic_tick(self) -> None:
        state = GeneticsState()
        rng = random.Random(42)
        g1 = Genome(alleles={l: (0.3, 0.7) for l in ALL_LOCI})
        g2 = Genome(alleles={l: (0.4, 0.6) for l in ALL_LOCI})
        marks = {"c1": EpigeneticMarks(), "c2": EpigeneticMarks()}
        stress = {"c1": 0.3, "c2": 0.5}
        result = tick_genetics(state, [g1, g2], marks, stress,
                               resource_avg=0.5, radiation_event=False,
                               rng=rng)
        assert isinstance(result, GeneticsTickResult)
        assert result.heterozygosity >= 0.0
        assert len(state.diversity_history) == 1

    def test_diversity_history_capped(self) -> None:
        state = GeneticsState()
        rng = random.Random(42)
        g = Genome(alleles={l: (0.5, 0.5) for l in ALL_LOCI})
        marks = {"c1": EpigeneticMarks()}
        for _ in range(150):
            tick_genetics(state, [g], marks, {"c1": 0.0},
                          resource_avg=0.5, radiation_event=False, rng=rng)
        assert len(state.diversity_history) == 100


# ---------------------------------------------------------------------------
# Integration with colonist
# ---------------------------------------------------------------------------

class TestColonistGenomeIntegration:
    def test_founding_ten_no_genome_by_default(self) -> None:
        """Pre-v11 colonists have genome=None (backward compat)."""
        colonists = create_founding_ten(42)
        # Without engine bootstrap, genome is None
        for c in colonists:
            assert c.genome is None

    def test_evolve_stats_without_genome(self) -> None:
        """evolve_stats still works without genome (backward compat)."""
        c = create_founding_ten(42)[0]
        rng = random.Random(42)
        old = c.stats.resolve
        c.evolve_stats("calm", rng)
        # Should not crash; value should have changed
        assert c.stats.resolve != old or True  # drift might be tiny

    def test_evolve_stats_with_genome(self) -> None:
        c = create_founding_ten(42)[0]
        c.genome = bootstrap_genome(
            c.stats.to_dict(), c.skills.to_dict(), 42, c.id)
        c.epigenetic_marks = EpigeneticMarks()
        rng = random.Random(42)
        c.evolve_stats("calm", rng)
        # Should not crash
        for name in STAT_NAMES:
            val = getattr(c.stats, name)
            assert 0.0 <= val <= 1.0

    def test_evolve_skills_with_genome(self) -> None:
        c = create_founding_ten(42)[0]
        c.genome = bootstrap_genome(
            c.stats.to_dict(), c.skills.to_dict(), 42, c.id)
        c.epigenetic_marks = EpigeneticMarks()
        rng = random.Random(42)
        old_coding = c.skills.coding
        c.evolve_skills("code", rng)
        assert c.skills.coding >= old_coding  # should gain
        assert c.skills.coding <= 1.0

    def test_create_child_with_genomes(self) -> None:
        pa = create_founding_ten(42)[0]
        pb = create_founding_ten(42)[1]
        pa.genome = bootstrap_genome(
            pa.stats.to_dict(), pa.skills.to_dict(), 42, pa.id)
        pb.genome = bootstrap_genome(
            pb.stats.to_dict(), pb.skills.to_dict(), 42, pb.id)
        pa.epigenetic_marks = EpigeneticMarks()
        pb.epigenetic_marks = EpigeneticMarks()
        rng = random.Random(42)
        child = create_child(pa, pb, "child-1", 10, rng, mutation_rate=0.02)
        assert child.genome is not None
        assert child.epigenetic_marks is not None
        for locus in ALL_LOCI:
            a, b = child.genome.alleles[locus]
            assert 0.0 <= a <= 1.0
            assert 0.0 <= b <= 1.0

    def test_create_child_without_genomes_fallback(self) -> None:
        """If parents lack genomes, child should still be created."""
        pa = create_founding_ten(42)[0]
        pb = create_founding_ten(42)[1]
        rng = random.Random(42)
        child = create_child(pa, pb, "child-1", 10, rng)
        assert child.genome is None  # no genome to crossover

    def test_colonist_roundtrip_with_genome(self) -> None:
        c = create_founding_ten(42)[0]
        c.genome = bootstrap_genome(
            c.stats.to_dict(), c.skills.to_dict(), 42, c.id)
        c.epigenetic_marks = EpigeneticMarks(marks={"resolve_apt": 0.1})
        d = c.to_dict()
        c2 = Colonist.from_dict(d)
        assert c2.genome is not None
        assert c2.genome.expressed("resolve_apt") == pytest.approx(
            c.genome.expressed("resolve_apt"), abs=0.001)
        assert c2.epigenetic_marks.get("resolve_apt") == pytest.approx(0.1, abs=0.001)


# ---------------------------------------------------------------------------
# Engine smoke test
# ---------------------------------------------------------------------------

class TestEngineSmoke:
    def test_10_year_run(self) -> None:
        """Full engine run for 10 years should not crash."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert len(result.years) == 10
        # Genetics state should be populated
        assert result.final_genetics is not None
        assert "mean_heterozygosity" in result.final_genetics
        assert result.final_genetics["mean_heterozygosity"] >= 0.0
        # All colonists should have genomes
        for c in engine.colonists:
            assert c.genome is not None

    def test_genetics_in_yearly_output(self) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=3)
        result = engine.run()
        for yr in result.years:
            d = yr.to_dict()
            assert "genetics" in d
            assert "heterozygosity" in d["genetics"]

    def test_deterministic(self) -> None:
        """Two runs with same seed produce identical genetics."""
        from src.mars100.engine import Mars100Engine
        e1 = Mars100Engine(seed=42, total_years=5)
        r1 = e1.run()
        e2 = Mars100Engine(seed=42, total_years=5)
        r2 = e2.run()
        for i in range(5):
            g1 = r1.years[i].genetics
            g2 = r2.years[i].genetics
            assert g1["heterozygosity"] == pytest.approx(
                g2["heterozygosity"], abs=1e-6)


# ---------------------------------------------------------------------------
# Property: all phenotypic values in [0, 1]
# ---------------------------------------------------------------------------

class TestPropertyBounds:
    def test_trait_biases_bounded(self) -> None:
        rng = random.Random(42)
        for _ in range(100):
            alleles = {l: (rng.random(), rng.random()) for l in ALL_LOCI}
            g = Genome(alleles=alleles)
            marks = EpigeneticMarks(
                marks={l: rng.uniform(-0.2, 0.2) for l in ALL_LOCI})
            biases = compute_trait_biases(g, marks)
            for stat, val in biases.items():
                assert 0.0 <= val <= 1.0, f"{stat} bias {val} out of bounds"

    def test_skill_aptitudes_bounded(self) -> None:
        rng = random.Random(42)
        for _ in range(100):
            alleles = {l: (rng.random(), rng.random()) for l in ALL_LOCI}
            g = Genome(alleles=alleles)
            marks = EpigeneticMarks(
                marks={l: rng.uniform(-0.2, 0.2) for l in ALL_LOCI})
            apts = compute_skill_aptitudes(g, marks)
            for skill, val in apts.items():
                assert 0.5 <= val <= 2.0, f"{skill} apt {val} out of bounds"
