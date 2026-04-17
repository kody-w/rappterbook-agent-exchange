"""Tests for the genetics organ (engine v11.0)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mars100.genetics import (
    TRAIT_NAMES, EARTH_PRIORS, MUTATION_RATE,
    Genome, GeneticsState, GeneticsYearContext, GeneticsTickResult,
    create_genome, create_genome_deterministic, inherit_genome,
    compute_mars_fitness, compute_diversity, compute_genetic_death_modifier,
    compute_genetic_birth_modifier, compute_bottleneck_penalty,
    tick_genetics,
    MAX_DEATH_MODIFIER, MIN_DEATH_MODIFIER,
    MAX_BIRTH_MODIFIER, MIN_BIRTH_MODIFIER,
    REPRODUCTIVE_AGE_MIN, REPRODUCTIVE_AGE_MAX,
    BOTTLENECK_THRESHOLD,
)
from src.mars100.colonist import (
    create_founding_ten, create_child, create_immigrant, Colonist,
)


# ---- Genome creation ----

class TestGenomeCreation:
    def test_create_genome_has_all_traits(self):
        rng = random.Random(42)
        g = create_genome(rng)
        for t in TRAIT_NAMES:
            assert t in g.alleles
            a, b = g.alleles[t]
            assert 0.0 <= a <= 1.0
            assert 0.0 <= b <= 1.0

    def test_create_genome_earth_priors(self):
        """Founders should cluster around Earth-human means, not uniform."""
        rng = random.Random(99)
        values = {t: [] for t in TRAIT_NAMES}
        for _ in range(100):
            g = create_genome(rng)
            for t in TRAIT_NAMES:
                values[t].append(g.phenotype(t))
        # radiation_resistance mean should be near 0.25, not 0.5
        rad_mean = sum(values["radiation_resistance"]) / 100
        assert rad_mean < 0.40, f"Rad mean {rad_mean} too high for Earth priors"
        # bone_density mean should be near 0.60
        bone_mean = sum(values["bone_density"]) / 100
        assert bone_mean > 0.45, f"Bone mean {bone_mean} too low for Earth priors"

    def test_create_genome_deterministic(self):
        g1 = create_genome_deterministic("kira-sol")
        g2 = create_genome_deterministic("kira-sol")
        assert g1.to_dict() == g2.to_dict()

    def test_different_ids_different_genomes(self):
        g1 = create_genome_deterministic("kira-sol")
        g2 = create_genome_deterministic("fen-marsh")
        assert g1.to_dict() != g2.to_dict()

    def test_seed_reproducibility(self):
        g1 = create_genome(random.Random(42))
        g2 = create_genome(random.Random(42))
        assert g1.to_dict() == g2.to_dict()


# ---- Genome serialization ----

class TestGenomeSerialization:
    def test_round_trip(self):
        rng = random.Random(42)
        g = create_genome(rng)
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        d2 = g2.to_dict()
        for t in TRAIT_NAMES:
            assert d[t][0] == pytest.approx(d2[t][0], abs=1e-4)
            assert d[t][1] == pytest.approx(d2[t][1], abs=1e-4)

    def test_from_dict_defaults(self):
        g = Genome.from_dict({})
        for t in TRAIT_NAMES:
            assert g.phenotype(t) == pytest.approx(0.5)

    def test_clamp(self):
        g = Genome(alleles={"radiation_resistance": (-0.5, 1.5)})
        g.clamp()
        a, b = g.alleles["radiation_resistance"]
        assert a == 0.0
        assert b == 1.0


# ---- Phenotype ----

class TestPhenotype:
    def test_phenotype_is_mean(self):
        g = Genome(alleles={"radiation_resistance": (0.2, 0.8)})
        assert g.phenotype("radiation_resistance") == pytest.approx(0.5)

    def test_all_phenotypes(self):
        rng = random.Random(42)
        g = create_genome(rng)
        pheno = g.all_phenotypes()
        assert len(pheno) == len(TRAIT_NAMES)
        for t in TRAIT_NAMES:
            assert 0.0 <= pheno[t] <= 1.0


# ---- Inheritance ----

class TestInheritance:
    def test_child_gets_mix_of_parents(self):
        rng = random.Random(42)
        pa = create_genome(rng)
        pb = create_genome(rng)
        child = inherit_genome(pa, pb, rng)
        for t in TRAIT_NAMES:
            assert t in child.alleles
            a, b = child.alleles[t]
            assert 0.0 <= a <= 1.0
            assert 0.0 <= b <= 1.0

    def test_child_alleles_from_parents(self):
        """Without mutation, child alleles should come from parent pools."""
        # Use a deterministic test with mutation effectively disabled
        rng = random.Random(42)
        pa = Genome(alleles={t: (0.1, 0.2) for t in TRAIT_NAMES})
        pb = Genome(alleles={t: (0.8, 0.9) for t in TRAIT_NAMES})
        # Run many children, most alleles should be from parent range
        from_a = 0
        from_b = 0
        total = 0
        for _ in range(200):
            child = inherit_genome(pa, pb, rng)
            for t in TRAIT_NAMES:
                ca, cb = child.alleles[t]
                for v in (ca, cb):
                    total += 1
                    if 0.05 <= v <= 0.25:  # close to parent A range
                        from_a += 1
                    elif 0.75 <= v <= 0.95:  # close to parent B range
                        from_b += 1
        # Most should be from parents (mutation is 2% so ~96% unaffected)
        assert from_a > total * 0.3
        assert from_b > total * 0.3

    def test_mutation_stays_in_bounds(self):
        rng = random.Random(42)
        extreme_a = Genome(alleles={t: (0.0, 0.0) for t in TRAIT_NAMES})
        extreme_b = Genome(alleles={t: (1.0, 1.0) for t in TRAIT_NAMES})
        for _ in range(100):
            child = inherit_genome(extreme_a, extreme_b, rng)
            for t in TRAIT_NAMES:
                a, b = child.alleles[t]
                assert 0.0 <= a <= 1.0
                assert 0.0 <= b <= 1.0


# ---- Fitness ----

class TestFitness:
    def test_mars_optimal_high_fitness(self):
        pheno = {t: 1.0 for t in TRAIT_NAMES}
        f = compute_mars_fitness(pheno)
        assert f > 0.9

    def test_earth_typical_low_fitness(self):
        pheno = {t: mean for t, (mean, _) in EARTH_PRIORS.items()}
        f = compute_mars_fitness(pheno)
        assert f < 0.5

    def test_fitness_in_bounds(self):
        rng = random.Random(42)
        for _ in range(50):
            g = create_genome(rng)
            f = compute_mars_fitness(g.all_phenotypes())
            assert 0.0 <= f <= 1.0


# ---- Diversity ----

class TestDiversity:
    def test_clones_zero_diversity(self):
        g = create_genome(random.Random(42))
        d = compute_diversity([g, g, g])
        assert d == pytest.approx(0.0)

    def test_diverse_population(self):
        rng = random.Random(42)
        genomes = [create_genome(rng) for _ in range(10)]
        d = compute_diversity(genomes)
        assert d > 0.0

    def test_single_colonist_zero(self):
        g = create_genome(random.Random(42))
        d = compute_diversity([g])
        assert d == 0.0

    def test_empty_zero(self):
        d = compute_diversity([])
        assert d == 0.0

    def test_extreme_diversity(self):
        a = Genome(alleles={t: (0.0, 0.0) for t in TRAIT_NAMES})
        b = Genome(alleles={t: (1.0, 1.0) for t in TRAIT_NAMES})
        d = compute_diversity([a, b])
        assert d > 0.5


# ---- Death/Birth Modifiers ----

class TestModifiers:
    def test_death_modifier_range(self):
        for _ in range(50):
            rng = random.Random(_)
            g = create_genome(rng)
            m = compute_genetic_death_modifier(g.all_phenotypes())
            assert MIN_DEATH_MODIFIER <= m <= MAX_DEATH_MODIFIER

    def test_high_fitness_lower_death_rate(self):
        good = {t: 0.9 for t in TRAIT_NAMES}
        bad = {t: 0.1 for t in TRAIT_NAMES}
        m_good = compute_genetic_death_modifier(good)
        m_bad = compute_genetic_death_modifier(bad)
        assert m_good < m_bad

    def test_birth_modifier_age_gating(self):
        pheno = {"fertility": 0.8}
        # Too young
        assert compute_genetic_birth_modifier(pheno, 5) == 0.0
        # Reproductive age
        m = compute_genetic_birth_modifier(pheno, 25)
        assert m > 0.0
        # Too old
        assert compute_genetic_birth_modifier(pheno, 60) == 0.0

    def test_birth_modifier_range(self):
        for fert in [0.0, 0.5, 1.0]:
            m = compute_genetic_birth_modifier(
                {"fertility": fert}, 25)
            assert MIN_BIRTH_MODIFIER <= m <= MAX_BIRTH_MODIFIER


# ---- Bottleneck ----

class TestBottleneck:
    def test_no_penalty_above_threshold(self):
        state = GeneticsState(diversity=0.3, bottleneck_years=5)
        p = compute_bottleneck_penalty(state)
        assert p == 0.0

    def test_penalty_below_threshold(self):
        state = GeneticsState(diversity=0.05, bottleneck_years=5)
        p = compute_bottleneck_penalty(state)
        assert p > 0.0
        assert p <= 0.05

    def test_penalty_increases_with_duration(self):
        state_short = GeneticsState(diversity=0.05, bottleneck_years=2)
        state_long = GeneticsState(diversity=0.05, bottleneck_years=8)
        p_short = compute_bottleneck_penalty(state_short)
        p_long = compute_bottleneck_penalty(state_long)
        assert p_long > p_short


# ---- GeneticsState ----

class TestGeneticsState:
    def test_round_trip(self):
        state = GeneticsState(diversity=0.3, avg_fitness=0.5, max_generation=2,
                              mars_adapted_count=3, bottleneck_years=0,
                              adaptation_trend=[0.3, 0.4, 0.5])
        d = state.to_dict()
        s2 = GeneticsState.from_dict(d)
        assert s2.diversity == pytest.approx(0.3, abs=1e-4)
        assert s2.avg_fitness == pytest.approx(0.5, abs=1e-4)
        assert s2.max_generation == 2
        assert s2.mars_adapted_count == 3

    def test_health(self):
        healthy = GeneticsState(diversity=0.4, avg_fitness=0.8)
        assert healthy.health() > 0.5
        unhealthy = GeneticsState(diversity=0.02, avg_fitness=0.2)
        assert unhealthy.health() < 0.5


# ---- Tick ----

class TestTick:
    def _make_colonists(self, n: int, seed: int = 42) -> list:
        rng = random.Random(seed)
        colonists = create_founding_ten(seed)[:n]
        return colonists

    def test_tick_basic(self):
        state = GeneticsState()
        colonists = self._make_colonists(10)
        ctx = GeneticsYearContext(
            year=1, active_colonists=colonists,
            births_this_year=[], deaths_this_year=[])
        rng = random.Random(42)
        result = tick_genetics(state, ctx, rng)
        assert result.diversity > 0.0
        assert result.avg_fitness > 0.0
        assert result.max_generation == 0
        assert state.diversity == result.diversity

    def test_tick_empty_colonists(self):
        state = GeneticsState()
        ctx = GeneticsYearContext(
            year=1, active_colonists=[],
            births_this_year=[], deaths_this_year=[])
        result = tick_genetics(state, ctx, random.Random(42))
        assert result.diversity == 0.0
        assert result.avg_fitness == 0.0

    def test_tick_single_colonist(self):
        state = GeneticsState()
        colonists = self._make_colonists(1)
        ctx = GeneticsYearContext(
            year=1, active_colonists=colonists,
            births_this_year=[], deaths_this_year=[])
        result = tick_genetics(state, ctx, random.Random(42))
        assert result.diversity == 0.0
        assert result.avg_fitness > 0.0

    def test_tick_updates_bottleneck(self):
        state = GeneticsState(diversity=0.05, bottleneck_years=3)
        # All clones → low diversity
        c = self._make_colonists(1)
        colonists = [c[0]] * 5  # clones
        ctx = GeneticsYearContext(
            year=1, active_colonists=colonists,
            births_this_year=[], deaths_this_year=[])
        result = tick_genetics(state, ctx, random.Random(42))
        assert state.bottleneck_years >= 4  # incremented

    def test_tick_result_serialization(self):
        result = GeneticsTickResult(
            diversity=0.3, avg_fitness=0.5, max_generation=2,
            mars_adapted_count=1, bottleneck_penalty=0.01,
            adaptation_event="rapid_adaptation")
        d = result.to_dict()
        assert "diversity" in d
        assert "adaptation_event" in d
        assert d["adaptation_event"] == "rapid_adaptation"

    def test_adaptation_trend_window(self):
        state = GeneticsState()
        colonists = self._make_colonists(10)
        rng = random.Random(42)
        for year in range(1, 15):
            ctx = GeneticsYearContext(
                year=year, active_colonists=colonists,
                births_this_year=[], deaths_this_year=[])
            tick_genetics(state, ctx, rng)
        assert len(state.adaptation_trend) <= 10


# ---- Colonist integration ----

class TestColonistIntegration:
    def test_founding_ten_have_genomes(self):
        colonists = create_founding_ten(42)
        for c in colonists:
            assert c.genome is not None
            assert c.generation == 0
            if isinstance(c.genome, Genome):
                pheno = c.genome.all_phenotypes()
                for t in TRAIT_NAMES:
                    assert 0.0 <= pheno[t] <= 1.0

    def test_child_inherits_genome(self):
        colonists = create_founding_ten(42)
        rng = random.Random(42)
        child = create_child(colonists[0], colonists[1], "child-1", 10, rng)
        assert child.genome is not None
        assert child.generation == 1

    def test_child_generation_increment(self):
        colonists = create_founding_ten(42)
        rng = random.Random(42)
        c1 = create_child(colonists[0], colonists[1], "c1", 10, rng)
        assert c1.generation == 1
        c2 = create_child(c1, colonists[2], "c2", 20, rng)
        assert c2.generation == 2

    def test_immigrant_has_genome(self):
        rng = random.Random(42)
        imm = create_immigrant("imm-1", 15, rng)
        assert imm.genome is not None
        assert imm.generation == 0

    def test_colonist_serialization_with_genome(self):
        colonists = create_founding_ten(42)
        c = colonists[0]
        d = c.to_dict()
        assert "genome" in d
        assert "generation" in d
        assert d["generation"] == 0
        # Verify genome round-trips
        from src.mars100.colonist import Colonist
        c2 = Colonist.from_dict(d)
        assert c2.generation == 0
        assert c2.genome is not None


# ---- Engine smoke test ----

class TestEngineSmoke:
    def test_10_year_smoke(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert len(result.years) == 10
        assert "genetics" in result.years[0].to_dict()
        assert result.final_genetics is not None

    def test_100_year_smoke(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        assert len(result.years) > 50
        genetics = result.final_genetics
        assert genetics["diversity"] >= 0.0
        assert genetics["avg_fitness"] >= 0.0

    def test_genetics_evolves_over_time(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        y1_gen = result.years[0].genetics
        last_gen = result.years[-1].genetics
        # After 50 years, something should have changed
        assert y1_gen != last_gen or len(result.years) < 2

    def test_seed_determinism(self):
        from src.mars100.engine import Mars100Engine
        e1 = Mars100Engine(seed=99, total_years=10)
        r1 = e1.run()
        e2 = Mars100Engine(seed=99, total_years=10)
        r2 = e2.run()
        assert r1.final_genetics == r2.final_genetics
        assert r1.total_deaths == r2.total_deaths


# ---- Physical bounds invariants ----

class TestPhysicalBounds:
    def test_all_phenotypes_bounded(self):
        rng = random.Random(42)
        for _ in range(100):
            g = create_genome(rng)
            for t in TRAIT_NAMES:
                v = g.phenotype(t)
                assert 0.0 <= v <= 1.0, f"{t} out of bounds: {v}"

    def test_diversity_bounded(self):
        rng = random.Random(42)
        genomes = [create_genome(rng) for _ in range(20)]
        d = compute_diversity(genomes)
        assert 0.0 <= d <= 1.0

    def test_fitness_bounded(self):
        rng = random.Random(42)
        for _ in range(100):
            g = create_genome(rng)
            f = compute_mars_fitness(g.all_phenotypes())
            assert 0.0 <= f <= 1.0

    def test_genetics_health_bounded(self):
        for d, f in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]:
            state = GeneticsState(diversity=d, avg_fitness=f)
            h = state.health()
            assert 0.0 <= h <= 1.0
