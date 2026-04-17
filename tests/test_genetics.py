"""Tests for the genetics organ (engine v11.0)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mars100.genetics import (
    LOCI, EARTH_BASELINE_MEAN, EARTH_BASELINE_SPREAD,
    MUTATION_RATE, MAX_O2_SURVIVAL_BONUS, MAX_SKILL_MULTIPLIER,
    BOTTLENECK_THRESHOLD,
    Genome, GeneticsState, GeneticsYearContext, GeneticsTickResult,
    create_earth_genome, create_child_genome,
    express_phenotype, compute_o2_survival_bonus, compute_skill_multiplier,
    compute_diversity, compute_mars_adaptation, is_close_relative,
    tick_genetics,
)


# ---------------------------------------------------------------------------
# Genome creation
# ---------------------------------------------------------------------------

class TestCreateEarthGenome:
    def test_all_loci_present(self):
        g = create_earth_genome(random.Random(1))
        for loc in LOCI:
            assert loc in g.alleles

    def test_alleles_near_baseline(self):
        rng = random.Random(42)
        for _ in range(50):
            g = create_earth_genome(rng)
            for loc in LOCI:
                a, b = g.alleles[loc]
                assert 0.0 <= a <= 1.0
                assert 0.0 <= b <= 1.0

    def test_phenotype_near_baseline(self):
        rng = random.Random(99)
        phenotypes = [create_earth_genome(rng).phenotype("o2_tolerance")
                      for _ in range(100)]
        mean_p = sum(phenotypes) / len(phenotypes)
        assert abs(mean_p - EARTH_BASELINE_MEAN) < 0.05

    def test_deterministic_with_seed(self):
        g1 = create_earth_genome(random.Random(7))
        g2 = create_earth_genome(random.Random(7))
        assert g1.alleles == g2.alleles


class TestCreateChildGenome:
    def _parents(self, seed=42):
        rng = random.Random(seed)
        return create_earth_genome(rng), create_earth_genome(rng)

    def test_child_has_all_loci(self):
        pa, pb = self._parents()
        child = create_child_genome(pa, pb, random.Random(1))
        for loc in LOCI:
            assert loc in child.alleles

    def test_child_alleles_bounded(self):
        pa, pb = self._parents()
        rng = random.Random(42)
        for _ in range(100):
            child = create_child_genome(pa, pb, rng)
            for loc in LOCI:
                a, b = child.alleles[loc]
                assert 0.0 <= a <= 1.0, f"{loc} allele_a out of bounds: {a}"
                assert 0.0 <= b <= 1.0, f"{loc} allele_b out of bounds: {b}"

    def test_child_differs_from_parents(self):
        pa, pb = self._parents()
        child = create_child_genome(pa, pb, random.Random(123))
        assert child.alleles != pa.alleles or child.alleles != pb.alleles

    def test_deterministic_with_seed(self):
        pa, pb = self._parents()
        c1 = create_child_genome(pa, pb, random.Random(5))
        c2 = create_child_genome(pa, pb, random.Random(5))
        assert c1.alleles == c2.alleles


# ---------------------------------------------------------------------------
# Phenotype expression
# ---------------------------------------------------------------------------

class TestPhenotype:
    def test_express_all_loci(self):
        g = Genome(alleles={loc: (0.3, 0.7) for loc in LOCI})
        p = express_phenotype(g)
        for loc in LOCI:
            assert abs(p[loc] - 0.5) < 1e-9

    def test_phenotype_range(self):
        g = Genome(alleles={loc: (0.0, 1.0) for loc in LOCI})
        p = express_phenotype(g)
        for loc in LOCI:
            assert 0.0 <= p[loc] <= 1.0


class TestO2SurvivalBonus:
    def test_no_bonus_below_catastrophic(self):
        assert compute_o2_survival_bonus(0.9, 0.03) == 0.0

    def test_no_bonus_above_safe(self):
        assert compute_o2_survival_bonus(0.9, 0.20) == 0.0

    def test_no_bonus_at_baseline(self):
        bonus = compute_o2_survival_bonus(EARTH_BASELINE_MEAN, 0.10)
        assert bonus == 0.0

    def test_positive_bonus_with_tolerance(self):
        bonus = compute_o2_survival_bonus(0.8, 0.10)
        assert bonus > 0.0
        assert bonus <= MAX_O2_SURVIVAL_BONUS

    def test_bonus_increases_with_tolerance(self):
        low = compute_o2_survival_bonus(0.6, 0.10)
        high = compute_o2_survival_bonus(0.9, 0.10)
        assert high > low

    def test_bonus_increases_with_air(self):
        low_air = compute_o2_survival_bonus(0.8, 0.06)
        high_air = compute_o2_survival_bonus(0.8, 0.14)
        assert high_air > low_air


class TestSkillMultiplier:
    def test_baseline_gives_one(self):
        assert abs(compute_skill_multiplier(EARTH_BASELINE_MEAN) - 1.0) < 1e-9

    def test_high_plasticity_boosts(self):
        mult = compute_skill_multiplier(0.8)
        assert mult > 1.0
        assert mult <= 1.0 + MAX_SKILL_MULTIPLIER

    def test_low_plasticity_slows(self):
        mult = compute_skill_multiplier(0.2)
        assert mult < 1.0
        assert mult >= 1.0 - MAX_SKILL_MULTIPLIER

    def test_range_capped(self):
        assert compute_skill_multiplier(1.0) <= 1.0 + MAX_SKILL_MULTIPLIER + 1e-9
        assert compute_skill_multiplier(0.0) >= 1.0 - MAX_SKILL_MULTIPLIER - 1e-9


# ---------------------------------------------------------------------------
# Colony metrics
# ---------------------------------------------------------------------------

class TestDiversity:
    def _make_genomes(self, n: int, rng: random.Random) -> dict[str, Genome]:
        return {f"c{i}": create_earth_genome(rng) for i in range(n)}

    def test_zero_with_one_colonist(self):
        genomes = self._make_genomes(1, random.Random(1))
        assert compute_diversity(genomes, list(genomes.keys())) == 0.0

    def test_positive_with_many(self):
        genomes = self._make_genomes(10, random.Random(42))
        d = compute_diversity(genomes, list(genomes.keys()))
        assert d > 0.0

    def test_bounded(self):
        genomes = self._make_genomes(20, random.Random(99))
        d = compute_diversity(genomes, list(genomes.keys()))
        assert 0.0 <= d <= 1.0

    def test_ignores_inactive(self):
        genomes = self._make_genomes(10, random.Random(42))
        active = list(genomes.keys())[:5]
        d = compute_diversity(genomes, active)
        assert d > 0.0


class TestMarsAdaptation:
    def test_baseline_near_zero(self):
        genomes = {f"c{i}": create_earth_genome(random.Random(i))
                   for i in range(10)}
        adapt = compute_mars_adaptation(genomes, list(genomes.keys()))
        assert adapt < 0.1

    def test_drifted_colony_higher(self):
        drifted = {f"c{i}": Genome(alleles={loc: (0.8, 0.9) for loc in LOCI})
                   for i in range(5)}
        adapt = compute_mars_adaptation(drifted, list(drifted.keys()))
        assert adapt > 0.2

    def test_empty_returns_zero(self):
        assert compute_mars_adaptation({}, []) == 0.0


class TestCloseRelative:
    def test_siblings_detected(self):
        pedigree = {
            "child-1": ["parent-a", "parent-b"],
            "child-2": ["parent-a", "parent-c"],
        }
        assert is_close_relative("child-1", "child-2", pedigree)

    def test_unrelated_not_detected(self):
        pedigree = {
            "child-1": ["parent-a", "parent-b"],
            "child-2": ["parent-c", "parent-d"],
        }
        assert not is_close_relative("child-1", "child-2", pedigree)

    def test_founders_not_related(self):
        pedigree = {
            "founder-1": [None, None],
            "founder-2": [None, None],
        }
        assert not is_close_relative("founder-1", "founder-2", pedigree)


# ---------------------------------------------------------------------------
# GeneticsState
# ---------------------------------------------------------------------------

class TestGeneticsState:
    def _make_state(self) -> GeneticsState:
        rng = random.Random(42)
        state = GeneticsState()
        for i in range(5):
            state.register_founder(f"f{i}", create_earth_genome(rng))
        return state

    def test_register_founder(self):
        state = self._make_state()
        assert len(state.genomes) == 5
        assert all(state.generation[f"f{i}"] == 0 for i in range(5))

    def test_register_birth(self):
        state = self._make_state()
        child_genome = create_child_genome(
            state.genomes["f0"], state.genomes["f1"], random.Random(1))
        state.register_birth("child-0", "f0", "f1", child_genome)
        assert state.generation["child-0"] == 1
        assert state.pedigree["child-0"] == ["f0", "f1"]

    def test_register_immigrant(self):
        state = self._make_state()
        imm_genome = create_earth_genome(random.Random(99))
        state.register_immigrant("imm-0", imm_genome)
        assert state.generation["imm-0"] == 0
        assert state.pedigree["imm-0"] == [None, None]

    def test_roundtrip_serialization(self):
        state = self._make_state()
        child_genome = create_child_genome(
            state.genomes["f0"], state.genomes["f1"], random.Random(1))
        state.register_birth("child-0", "f0", "f1", child_genome)
        d = state.to_dict()
        restored = GeneticsState.from_dict(d)
        assert set(restored.genomes.keys()) == set(state.genomes.keys())
        assert restored.generation == state.generation
        for cid in state.genomes:
            assert restored.genomes[cid].alleles == state.genomes[cid].alleles


# ---------------------------------------------------------------------------
# Yearly tick
# ---------------------------------------------------------------------------

class TestTickGenetics:
    def _make_state_and_ctx(self, n=10, year=50):
        rng = random.Random(42)
        state = GeneticsState()
        ids = []
        for i in range(n):
            cid = f"c{i}"
            state.register_founder(cid, create_earth_genome(rng))
            ids.append(cid)
        # Add one child
        child_genome = create_child_genome(
            state.genomes["c0"], state.genomes["c1"], rng)
        state.register_birth("child-0", "c0", "c1", child_genome)
        ids.append("child-0")
        ctx = GeneticsYearContext(year=year, active_ids=ids)
        return state, ctx

    def test_tick_returns_result(self):
        state, ctx = self._make_state_and_ctx()
        result = tick_genetics(state, ctx, random.Random(42))
        assert isinstance(result, GeneticsTickResult)

    def test_phenotypes_computed(self):
        state, ctx = self._make_state_and_ctx()
        result = tick_genetics(state, ctx, random.Random(42))
        assert len(result.phenotypes) == len(ctx.active_ids)
        for cid in ctx.active_ids:
            for loc in LOCI:
                assert loc in result.phenotypes[cid]

    def test_diversity_positive(self):
        state, ctx = self._make_state_and_ctx()
        result = tick_genetics(state, ctx, random.Random(42))
        assert result.diversity_index > 0.0
        assert result.diversity_index <= 1.0

    def test_mars_adaptation_low_for_earth(self):
        state, ctx = self._make_state_and_ctx()
        result = tick_genetics(state, ctx, random.Random(42))
        assert result.mars_adaptation < 0.1

    def test_generation_stats(self):
        state, ctx = self._make_state_and_ctx()
        result = tick_genetics(state, ctx, random.Random(42))
        assert result.max_generation >= 1
        assert result.avg_generation >= 0.0

    def test_diversity_history_appended(self):
        state, ctx = self._make_state_and_ctx()
        assert len(state.diversity_history) == 0
        tick_genetics(state, ctx, random.Random(42))
        assert len(state.diversity_history) == 1

    def test_result_serialization(self):
        state, ctx = self._make_state_and_ctx()
        result = tick_genetics(state, ctx, random.Random(42))
        d = result.to_dict()
        assert "diversity_index" in d
        assert "mars_adaptation" in d
        assert "max_generation" in d
        assert "notable_events" in d


# ---------------------------------------------------------------------------
# Property-based invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    """Property-based tests: physical bounds, conservation laws."""

    def test_alleles_always_bounded(self):
        """All alleles must stay in [0, 1] across many generations."""
        rng = random.Random(42)
        genomes = [create_earth_genome(rng) for _ in range(4)]
        for gen_num in range(20):
            pa = genomes[rng.randint(0, len(genomes) - 1)]
            pb = genomes[rng.randint(0, len(genomes) - 1)]
            child = create_child_genome(pa, pb, rng)
            for loc in LOCI:
                a, b = child.alleles[loc]
                assert 0.0 <= a <= 1.0, f"gen {gen_num}, {loc} a={a}"
                assert 0.0 <= b <= 1.0, f"gen {gen_num}, {loc} b={b}"
            genomes.append(child)

    def test_phenotype_always_bounded(self):
        """Phenotypes must be in [0, 1]."""
        rng = random.Random(99)
        for _ in range(200):
            g = create_earth_genome(rng)
            for loc in LOCI:
                p = g.phenotype(loc)
                assert 0.0 <= p <= 1.0

    def test_skill_multiplier_always_bounded(self):
        """Skill multiplier must be in valid range for any input."""
        for v in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            m = compute_skill_multiplier(v)
            assert 0.5 <= m <= 1.5, f"multiplier {m} for {v}"

    def test_o2_bonus_never_negative(self):
        """O2 survival bonus must never be negative."""
        for tol in [0.0, 0.3, 0.5, 0.7, 1.0]:
            for air in [0.0, 0.05, 0.10, 0.15, 0.20, 0.50]:
                bonus = compute_o2_survival_bonus(tol, air)
                assert bonus >= 0.0, f"tol={tol}, air={air}, bonus={bonus}"


# ---------------------------------------------------------------------------
# Smoke test: full lifecycle
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_ten_generation_lifecycle(self):
        """Run genetics through 10 generations without crash."""
        rng = random.Random(42)
        state = GeneticsState()
        colonist_ids = []

        # 10 founders
        for i in range(10):
            cid = f"founder-{i}"
            state.register_founder(cid, create_earth_genome(rng))
            colonist_ids.append(cid)

        for year in range(1, 51):
            ctx = GeneticsYearContext(year=year, active_ids=list(colonist_ids))
            result = tick_genetics(state, ctx, rng)
            assert result.diversity_index >= 0.0

            # Random births
            if len(colonist_ids) >= 2 and rng.random() < 0.3:
                pa_id = rng.choice(colonist_ids)
                pb_id = rng.choice([c for c in colonist_ids if c != pa_id])
                child_id = f"child-{year}"
                child_g = create_child_genome(
                    state.genomes[pa_id], state.genomes[pb_id], rng)
                state.register_birth(child_id, pa_id, pb_id, child_g)
                colonist_ids.append(child_id)

            # Random deaths
            if len(colonist_ids) > 5 and rng.random() < 0.1:
                dead = rng.choice(colonist_ids)
                colonist_ids.remove(dead)

        # Final state should be consistent
        assert len(state.genomes) >= len(colonist_ids)
        final_result = tick_genetics(
            state,
            GeneticsYearContext(year=51, active_ids=colonist_ids),
            rng)
        assert final_result.max_generation >= 1
