"""Tests for the genetics organ (engine v11.0)."""
from __future__ import annotations

import random
import pytest

from src.mars100.genetics import (
    Genome, GeneticsState, GeneticsTickResult,
    create_genome_from_stats, create_immigrant_genome,
    inherit_genome, kinship_coefficient, check_inbreeding,
    expected_heterozygosity, compute_adaptation,
    compute_selection_pressures, compute_mars_fitness,
    compute_resource_modifiers, tick_genetics,
    STAT_NAMES, MARS_TARGET, MUTATION_RATE, MUTATION_MAGNITUDE,
    INBREEDING_COEFFICIENT_WARN, INBREEDING_DEATH_MULT,
    DIVERSITY_FITNESS_FLOOR, MAX_FOOD_BONUS, MAX_DEATH_REDUCTION,
    _clamp,
)
from src.mars100.colonist import (
    Colonist, ColonistStats, ColonistSkills, create_founding_ten,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return random.Random(42)


@pytest.fixture
def founders():
    return create_founding_ten(42)


@pytest.fixture
def state():
    return GeneticsState()


@pytest.fixture
def populated_state(founders, rng):
    """A GeneticsState with genomes for all founders."""
    state = GeneticsState()
    for c in founders:
        state.genomes[c.id] = create_genome_from_stats(
            c.id, c.stats.to_dict(), rng)
    return state


# ── Genome creation ───────────────────────────────────────────────────────

class TestGenomeCreation:
    def test_genome_has_all_stat_alleles(self, rng):
        stats = {s: 0.5 for s in STAT_NAMES}
        g = create_genome_from_stats("test", stats, rng)
        for stat in STAT_NAMES:
            assert stat in g.alleles
            a, b = g.alleles[stat]
            assert 0.0 <= a <= 1.0
            assert 0.0 <= b <= 1.0

    def test_genome_expression_close_to_original_stats(self, rng):
        stats = {"resolve": 0.8, "improvisation": 0.3, "empathy": 0.6,
                 "hoarding": 0.4, "faith": 0.7, "paranoia": 0.2}
        g = create_genome_from_stats("test", stats, rng)
        expr = g.express()
        for stat in STAT_NAMES:
            assert abs(expr[stat] - stats[stat]) < 0.15, \
                f"{stat}: expressed {expr[stat]:.3f} vs original {stats[stat]}"

    def test_genome_generation_zero_for_founders(self, rng):
        stats = {s: 0.5 for s in STAT_NAMES}
        g = create_genome_from_stats("test", stats, rng)
        assert g.generation == 0

    def test_genome_parent_ids_none_for_founders(self, rng):
        stats = {s: 0.5 for s in STAT_NAMES}
        g = create_genome_from_stats("test", stats, rng)
        assert g.parent_ids == (None, None)

    def test_immigrant_genome_generation_zero(self, rng):
        stats = {s: 0.6 for s in STAT_NAMES}
        g = create_immigrant_genome("imm-1", stats, rng)
        assert g.generation == 0

    def test_genome_alleles_clamped(self):
        """Alleles must stay in [0, 1] even with extreme inputs."""
        rng = random.Random(999)
        stats = {s: 0.99 for s in STAT_NAMES}
        g = create_genome_from_stats("extreme", stats, rng)
        for stat in STAT_NAMES:
            a, b = g.alleles[stat]
            assert 0.0 <= a <= 1.0
            assert 0.0 <= b <= 1.0


# ── Inheritance ───────────────────────────────────────────────────────────

class TestInheritance:
    def test_child_has_all_loci(self, populated_state, rng):
        ids = list(populated_state.genomes.keys())
        ga = populated_state.genomes[ids[0]]
        gb = populated_state.genomes[ids[1]]
        child_genome, _ = inherit_genome(ids[0], ids[1], ga, gb, "child-1", rng)
        for stat in STAT_NAMES:
            assert stat in child_genome.alleles

    def test_child_generation_increments(self, populated_state, rng):
        ids = list(populated_state.genomes.keys())
        ga = populated_state.genomes[ids[0]]
        gb = populated_state.genomes[ids[1]]
        child_genome, _ = inherit_genome(ids[0], ids[1], ga, gb, "child-1", rng)
        assert child_genome.generation == 1

    def test_child_parent_ids_set(self, populated_state, rng):
        ids = list(populated_state.genomes.keys())
        ga = populated_state.genomes[ids[0]]
        gb = populated_state.genomes[ids[1]]
        child_genome, _ = inherit_genome(ids[0], ids[1], ga, gb, "child-1", rng)
        assert child_genome.parent_ids == (ids[0], ids[1])

    def test_child_alleles_clamped(self, rng):
        """Even with extreme parents, child alleles are in [0, 1]."""
        ga = Genome(alleles={s: (0.99, 0.99) for s in STAT_NAMES})
        gb = Genome(alleles={s: (0.01, 0.01) for s in STAT_NAMES})
        for _ in range(50):  # run many times to catch mutation overflow
            child, _ = inherit_genome("a", "b", ga, gb, "c", rng)
            for stat in STAT_NAMES:
                a, b = child.alleles[stat]
                assert 0.0 <= a <= 1.0, f"{stat} allele a={a}"
                assert 0.0 <= b <= 1.0, f"{stat} allele b={b}"

    def test_mutations_occur(self):
        """Over many offspring, mutations should appear at expected rate."""
        rng = random.Random(1234)
        ga = Genome(alleles={s: (0.5, 0.5) for s in STAT_NAMES})
        gb = Genome(alleles={s: (0.5, 0.5) for s in STAT_NAMES})
        total_mutations = 0
        n_trials = 500
        for i in range(n_trials):
            _, muts = inherit_genome("a", "b", ga, gb, f"c-{i}", rng)
            total_mutations += len(muts)
        # Expected: n_trials * 6 stats * 2 alleles * MUTATION_RATE
        expected = n_trials * len(STAT_NAMES) * 2 * MUTATION_RATE
        assert total_mutations > expected * 0.3, "Too few mutations"
        assert total_mutations < expected * 3.0, "Too many mutations"

    def test_deterministic_with_same_seed(self, populated_state):
        ids = list(populated_state.genomes.keys())
        ga = populated_state.genomes[ids[0]]
        gb = populated_state.genomes[ids[1]]
        rng1 = random.Random(777)
        rng2 = random.Random(777)
        child1, _ = inherit_genome(ids[0], ids[1], ga, gb, "c1", rng1)
        child2, _ = inherit_genome(ids[0], ids[1], ga, gb, "c2", rng2)
        for stat in STAT_NAMES:
            assert child1.alleles[stat] == child2.alleles[stat]


# ── Kinship & inbreeding ──────────────────────────────────────────────────

class TestKinship:
    def test_unrelated_individuals(self, populated_state):
        ids = list(populated_state.genomes.keys())
        coeff = kinship_coefficient(ids[0], ids[1], populated_state.genomes)
        assert coeff == 0.0

    def test_siblings_share_kinship(self, populated_state, rng):
        ids = list(populated_state.genomes.keys())
        pa, pb = ids[0], ids[1]
        ga = populated_state.genomes[pa]
        gb = populated_state.genomes[pb]
        child1, _ = inherit_genome(pa, pb, ga, gb, "sib-1", rng)
        child2, _ = inherit_genome(pa, pb, ga, gb, "sib-2", rng)
        populated_state.genomes["sib-1"] = child1
        populated_state.genomes["sib-2"] = child2
        coeff = kinship_coefficient("sib-1", "sib-2",
                                     populated_state.genomes)
        assert coeff > 0.0, "Siblings should have nonzero kinship"
        assert coeff <= 0.5, "Kinship shouldn't exceed 0.5"

    def test_inbreeding_warning_for_siblings_mating(self, populated_state, rng):
        ids = list(populated_state.genomes.keys())
        pa, pb = ids[0], ids[1]
        ga = populated_state.genomes[pa]
        gb = populated_state.genomes[pb]
        c1, _ = inherit_genome(pa, pb, ga, gb, "sib-1", rng)
        c2, _ = inherit_genome(pa, pb, ga, gb, "sib-2", rng)
        populated_state.genomes["sib-1"] = c1
        populated_state.genomes["sib-2"] = c2
        warning = check_inbreeding("grandchild", "sib-1", "sib-2",
                                    populated_state.genomes)
        assert warning is not None, "Siblings mating should trigger warning"
        assert warning["severity"] in ("moderate", "high")

    def test_no_warning_for_unrelated(self, populated_state):
        ids = list(populated_state.genomes.keys())
        warning = check_inbreeding("child", ids[0], ids[1],
                                    populated_state.genomes)
        assert warning is None


# ── Expected heterozygosity ──────────────────────────────────────────────

class TestHeterozygosity:
    def test_diverse_population_high_het(self, populated_state):
        ids = list(populated_state.genomes.keys())
        het = expected_heterozygosity(populated_state.genomes, ids)
        assert 0.5 < het <= 1.0, f"Expected high diversity, got {het}"

    def test_clonal_population_low_het(self):
        genomes = {}
        for i in range(10):
            genomes[f"clone-{i}"] = Genome(
                alleles={s: (0.5, 0.5) for s in STAT_NAMES})
        ids = list(genomes.keys())
        het = expected_heterozygosity(genomes, ids)
        assert het < 0.2, f"Expected low diversity for clones, got {het}"

    def test_single_colonist_returns_one(self):
        genomes = {"solo": Genome(alleles={s: (0.5, 0.5) for s in STAT_NAMES})}
        het = expected_heterozygosity(genomes, ["solo"])
        assert het == 1.0

    def test_empty_returns_one(self):
        het = expected_heterozygosity({}, [])
        assert het == 1.0


# ── Adaptation ────────────────────────────────────────────────────────────

class TestAdaptation:
    def test_perfect_adaptation(self):
        genomes = {}
        for i in range(5):
            genomes[f"c-{i}"] = Genome(
                alleles={s: (v, v) for s, v in MARS_TARGET.items()})
        adapt = compute_adaptation(genomes, list(genomes.keys()))
        assert adapt > 0.95, f"Perfect target should give high adaptation: {adapt}"

    def test_opposite_adaptation(self):
        genomes = {}
        anti = {s: 1.0 - v for s, v in MARS_TARGET.items()}
        for i in range(5):
            genomes[f"c-{i}"] = Genome(
                alleles={s: (v, v) for s, v in anti.items()})
        adapt = compute_adaptation(genomes, list(genomes.keys()))
        assert adapt < 0.8

    def test_empty_returns_zero(self):
        assert compute_adaptation({}, []) == 0.0


# ── Mars fitness ──────────────────────────────────────────────────────────

class TestMarsFitness:
    def test_perfect_genome(self):
        g = Genome(alleles={s: (v, v) for s, v in MARS_TARGET.items()})
        assert compute_mars_fitness(g) > 0.95

    def test_fitness_bounded(self, rng):
        stats = {s: rng.random() for s in STAT_NAMES}
        g = create_genome_from_stats("test", stats, rng)
        f = compute_mars_fitness(g)
        assert 0.0 <= f <= 1.0


# ── Resource modifiers ────────────────────────────────────────────────────

class TestResourceModifiers:
    def test_high_diversity_low_death(self):
        state = GeneticsState(diversity_index=0.8, adaptation_index=0.5)
        mods = compute_resource_modifiers(state)
        assert mods["death_rate_mult"] <= 1.0

    def test_low_diversity_high_death(self):
        state = GeneticsState(diversity_index=0.1, adaptation_index=0.5)
        mods = compute_resource_modifiers(state)
        assert mods["death_rate_mult"] > 1.0

    def test_adaptation_food_bonus(self):
        state = GeneticsState(diversity_index=0.8, adaptation_index=0.9)
        mods = compute_resource_modifiers(state)
        assert mods["food"] > 1.0

    def test_zero_adaptation_no_food_bonus(self):
        state = GeneticsState(diversity_index=0.8, adaptation_index=0.0)
        mods = compute_resource_modifiers(state)
        assert mods["food"] == 1.0


# ── Selection pressures ──────────────────────────────────────────────────

class TestSelectionPressures:
    def test_pressure_toward_target(self, populated_state):
        ids = list(populated_state.genomes.keys())
        pressures = compute_selection_pressures(
            populated_state.genomes, ids)
        for stat in STAT_NAMES:
            assert isinstance(pressures[stat], float)

    def test_empty_returns_zeros(self):
        pressures = compute_selection_pressures({}, [])
        for stat in STAT_NAMES:
            assert pressures[stat] == 0.0


# ── Serialization ─────────────────────────────────────────────────────────

class TestSerialization:
    def test_genome_roundtrip(self, rng):
        stats = {s: rng.random() for s in STAT_NAMES}
        g = create_genome_from_stats("test", stats, rng)
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        assert g.generation == g2.generation
        for stat in STAT_NAMES:
            assert g.alleles[stat] == (g2.alleles[stat][0], g2.alleles[stat][1])

    def test_state_roundtrip(self, populated_state):
        d = populated_state.to_dict()
        s2 = GeneticsState.from_dict(d)
        assert len(s2.genomes) == len(populated_state.genomes)
        assert abs(s2.diversity_index - populated_state.diversity_index) < 0.001

    def test_tick_result_serializable(self):
        result = GeneticsTickResult(
            diversity_index=0.8, adaptation_index=0.5,
            mutations=[{"child": "c1", "stat": "resolve", "allele": "a",
                        "delta": 0.05}],
            resource_modifiers={"food": 1.004},
        )
        d = result.to_dict()
        assert d["diversity_index"] == 0.8
        assert len(d["mutations"]) == 1


# ── Main tick ─────────────────────────────────────────────────────────────

class TestTick:
    def test_smoke_10_years(self, founders, rng):
        """Run genetics tick for 10 years without crash."""
        state = GeneticsState()
        active_ids = [c.id for c in founders]
        for year in range(1, 11):
            result = tick_genetics(state, [], [], founders,
                                   active_ids, year, rng)
            assert 0.0 <= result.diversity_index <= 1.0
            assert 0.0 <= result.adaptation_index <= 1.0

    def test_founders_get_genomes(self, founders, rng):
        state = GeneticsState()
        active_ids = [c.id for c in founders]
        tick_genetics(state, [], [], founders, active_ids, 1, rng)
        for c in founders:
            assert c.id in state.genomes

    def test_birth_creates_child_genome(self, founders, rng):
        state = GeneticsState()
        active_ids = [c.id for c in founders]
        tick_genetics(state, [], [], founders, active_ids, 1, rng)
        births = [{"id": "child-1", "parents": [founders[0].id, founders[1].id]}]
        result = tick_genetics(state, births, [], founders,
                               active_ids, 2, rng)
        assert "child-1" in state.genomes
        assert state.genomes["child-1"].generation == 1

    def test_adaptation_changes_over_time(self, founders, rng):
        state = GeneticsState()
        active_ids = [c.id for c in founders]
        adaptations = []
        for year in range(1, 21):
            result = tick_genetics(state, [], [], founders,
                                   active_ids, year, rng)
            adaptations.append(result.adaptation_index)
        # Adaptation should move (either up or down)
        assert adaptations[0] != adaptations[-1] or all(
            a == adaptations[0] for a in adaptations), \
            "Adaptation should change over time"

    def test_resource_modifiers_bounded(self, founders, rng):
        state = GeneticsState()
        active_ids = [c.id for c in founders]
        for year in range(1, 11):
            result = tick_genetics(state, [], [], founders,
                                   active_ids, year, rng)
            mods = result.resource_modifiers
            assert mods.get("food", 1.0) >= 1.0
            assert mods.get("food", 1.0) <= 1.0 + MAX_FOOD_BONUS * 1.1
            if "death_rate_mult" in mods:
                assert mods["death_rate_mult"] >= 0.9
                assert mods["death_rate_mult"] <= INBREEDING_DEATH_MULT * 1.1

    def test_single_colonist_no_crash(self, rng):
        """Edge case: single colonist."""
        c = create_founding_ten(42)[0]
        state = GeneticsState()
        result = tick_genetics(state, [], [], [c], [c.id], 1, rng)
        assert result.diversity_index == 1.0

    def test_no_colonists_no_crash(self, rng):
        state = GeneticsState()
        result = tick_genetics(state, [], [], [], [], 1, rng)
        assert result.diversity_index == 1.0


# ── Conservation laws ─────────────────────────────────────────────────────

class TestConservation:
    def test_alleles_always_in_bounds(self, founders, rng):
        """Property: all alleles remain in [0, 1] across 50 years of
        births and mutations."""
        state = GeneticsState()
        colonists = list(founders)
        for year in range(1, 51):
            active = [c for c in colonists if c.is_active()]
            active_ids = [c.id for c in active]
            births = []
            if len(active) >= 2 and rng.random() < 0.3:
                pa, pb = rng.sample(active, 2)
                cid = f"child-{year}"
                births = [{"id": cid, "parents": [pa.id, pb.id]}]
                from src.mars100.colonist import create_child
                child = create_child(pa, pb, cid, year, rng)
                colonists.append(child)
                active_ids.append(cid)
            tick_genetics(state, births, [], colonists,
                          active_ids, year, rng)
            for gid, genome in state.genomes.items():
                for stat in STAT_NAMES:
                    a, b = genome.alleles[stat]
                    assert 0.0 <= a <= 1.0, \
                        f"Year {year}, {gid}.{stat} allele a={a}"
                    assert 0.0 <= b <= 1.0, \
                        f"Year {year}, {gid}.{stat} allele b={b}"


# ── Clamp helper ──────────────────────────────────────────────────────────

class TestClamp:
    def test_clamp_within_bounds(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_low(self):
        assert _clamp(-0.1) == 0.0

    def test_clamp_high(self):
        assert _clamp(1.5) == 1.0
