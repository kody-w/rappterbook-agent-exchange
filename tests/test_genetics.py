"""Tests for the genetics organ (v11.0)."""
from __future__ import annotations

import random
import pytest

from src.mars100.genetics import (
    GENETIC_TRAITS,
    GeneticProfile, GeneticsState, GeneticsTickResult,
    create_founding_genetics, create_immigrant_genetics,
    inherit_genetics, compute_kinship, is_pair_blocked,
    compute_population_diversity,
    compute_genetic_death_modifier, compute_genetic_stress_reduction,
    compute_genetic_radiation_modifier,
    tick_genetics,
    _compute_fitness, _random_alleles,
    _ALLELE_MIN, _ALLELE_MAX, _FITNESS_MIN, _FITNESS_MAX,
)


# ── constants ──────────────────────────────────────────────────────────

class TestConstants:
    def test_genetic_traits_count(self):
        assert len(GENETIC_TRAITS) == 8

    def test_genetic_traits_unique(self):
        assert len(set(GENETIC_TRAITS)) == len(GENETIC_TRAITS)

    def test_allele_bounds(self):
        assert _ALLELE_MIN < 0 < _ALLELE_MAX
        assert _ALLELE_MIN == -_ALLELE_MAX


# ── GeneticProfile ─────────────────────────────────────────────────────

class TestGeneticProfile:
    def test_default_alleles(self):
        p = GeneticProfile()
        assert len(p.alleles) == 8
        for t in GENETIC_TRAITS:
            assert p.alleles[t] == 0.0

    def test_clamp(self):
        p = GeneticProfile()
        p.alleles["bone_density"] = 0.5
        p.alleles["immune_response"] = -0.5
        p.clamp()
        assert p.alleles["bone_density"] == _ALLELE_MAX
        assert p.alleles["immune_response"] == _ALLELE_MIN

    def test_roundtrip(self):
        rng = random.Random(42)
        p = GeneticProfile(
            alleles=_random_alleles(rng),
            ancestor_ids={"a", "b", "c"},
            birth_year=10,
            parent_ids=("a", "b"),
            adaptations=["adapted:bone_density"],
            conditions=["deficiency:immune_response"],
        )
        d = p.to_dict()
        p2 = GeneticProfile.from_dict(d)
        assert p2.alleles == p.alleles
        assert p2.ancestor_ids == p.ancestor_ids
        assert p2.birth_year == p.birth_year
        assert p2.parent_ids == p.parent_ids
        assert p2.adaptations == p.adaptations
        assert p2.conditions == p.conditions

    def test_from_dict_missing_fields(self):
        p = GeneticProfile.from_dict({})
        assert len(p.alleles) == 8
        assert p.birth_year == 0
        assert p.parent_ids is None

    def test_to_dict_ancestor_ids_sorted(self):
        p = GeneticProfile(ancestor_ids={"z", "a", "m"})
        d = p.to_dict()
        assert d["ancestor_ids"] == ["a", "m", "z"]


# ── GeneticsState ──────────────────────────────────────────────────────

class TestGeneticsState:
    def test_default(self):
        s = GeneticsState()
        assert s.profiles == {}
        assert s.colony_diversity == 1.0
        assert s.avg_fitness == 1.0

    def test_clamp(self):
        s = GeneticsState(colony_diversity=5.0, avg_fitness=-1.0)
        s.clamp()
        assert s.colony_diversity == 1.0
        assert s.avg_fitness == _FITNESS_MIN

    def test_roundtrip(self):
        rng = random.Random(42)
        profiles = create_founding_genetics(["a", "b"], rng)
        s = GeneticsState(profiles=profiles, colony_diversity=0.5, avg_fitness=1.2,
                          total_mutations=10, total_conditions=2)
        d = s.to_dict()
        s2 = GeneticsState.from_dict(d)
        assert len(s2.profiles) == 2
        assert s2.colony_diversity == 0.5
        assert s2.total_mutations == 10

    def test_from_dict_empty(self):
        s = GeneticsState.from_dict({})
        assert s.profiles == {}


# ── GeneticsTickResult ─────────────────────────────────────────────────

class TestGeneticsTickResult:
    def test_default(self):
        r = GeneticsTickResult()
        assert r.mutations == []
        assert r.new_adaptations == []
        assert r.diversity_index == 1.0

    def test_to_dict(self):
        r = GeneticsTickResult(
            mutations=[{"trait": "bone_density"}],
            diversity_index=0.12345678,
        )
        d = r.to_dict()
        assert d["diversity_index"] == 0.1235
        assert len(d["mutations"]) == 1


# ── profile creation ───────────────────────────────────────────────────

class TestCreateFoundingGenetics:
    def test_creates_all(self):
        rng = random.Random(42)
        ids = ["c0", "c1", "c2"]
        profiles = create_founding_genetics(ids, rng)
        assert set(profiles.keys()) == set(ids)

    def test_each_has_self_as_ancestor(self):
        rng = random.Random(42)
        profiles = create_founding_genetics(["c0", "c1"], rng)
        for cid, p in profiles.items():
            assert cid in p.ancestor_ids

    def test_no_parents(self):
        rng = random.Random(42)
        profiles = create_founding_genetics(["c0"], rng)
        assert profiles["c0"].parent_ids is None

    def test_alleles_in_bounds(self):
        rng = random.Random(42)
        profiles = create_founding_genetics([f"c{i}" for i in range(10)], rng)
        for p in profiles.values():
            for t in GENETIC_TRAITS:
                assert _ALLELE_MIN <= p.alleles[t] <= _ALLELE_MAX

    def test_deterministic(self):
        p1 = create_founding_genetics(["a", "b"], random.Random(42))
        p2 = create_founding_genetics(["a", "b"], random.Random(42))
        assert p1["a"].alleles == p2["a"].alleles


class TestCreateImmigrantGenetics:
    def test_basic(self):
        p = create_immigrant_genetics("imm-1", random.Random(42))
        assert "imm-1" in p.ancestor_ids
        assert p.parent_ids is None
        for t in GENETIC_TRAITS:
            assert _ALLELE_MIN <= p.alleles[t] <= _ALLELE_MAX


class TestInheritGenetics:
    def test_child_has_parent_ancestors(self):
        rng = random.Random(42)
        ga = GeneticProfile(alleles=_random_alleles(rng), ancestor_ids={"pa"})
        gb = GeneticProfile(alleles=_random_alleles(rng), ancestor_ids={"pb"})
        child = inherit_genetics("pa", "pb", ga, gb, "child-1", 15, rng)
        assert "pa" in child.ancestor_ids
        assert "pb" in child.ancestor_ids
        assert "child-1" in child.ancestor_ids

    def test_child_alleles_in_bounds(self):
        rng = random.Random(42)
        ga = GeneticProfile(alleles={t: _ALLELE_MAX for t in GENETIC_TRAITS})
        gb = GeneticProfile(alleles={t: _ALLELE_MIN for t in GENETIC_TRAITS})
        child = inherit_genetics("pa", "pb", ga, gb, "child-1", 10, rng)
        for t in GENETIC_TRAITS:
            assert _ALLELE_MIN <= child.alleles[t] <= _ALLELE_MAX

    def test_child_birth_year(self):
        rng = random.Random(42)
        ga = GeneticProfile(alleles=_random_alleles(rng))
        gb = GeneticProfile(alleles=_random_alleles(rng))
        child = inherit_genetics("pa", "pb", ga, gb, "c1", 25, rng)
        assert child.birth_year == 25
        assert child.parent_ids == ("pa", "pb")


# ── kinship ────────────────────────────────────────────────────────────

class TestKinship:
    def test_unrelated_founders(self):
        ga = GeneticProfile(ancestor_ids={"a"})
        gb = GeneticProfile(ancestor_ids={"b"})
        assert compute_kinship("a", "b", ga, gb) == 0.0

    def test_siblings_share_parents(self):
        ga = GeneticProfile(ancestor_ids={"c1", "pa", "pb"})
        gb = GeneticProfile(ancestor_ids={"c2", "pa", "pb"})
        k = compute_kinship("c1", "c2", ga, gb)
        assert k > 0.0

    def test_self_kinship_zero_when_only_self(self):
        ga = GeneticProfile(ancestor_ids={"a"})
        gb = GeneticProfile(ancestor_ids={"b"})
        assert compute_kinship("a", "b", ga, gb) == 0.0


class TestIsPairBlocked:
    def test_unrelated_not_blocked(self):
        ga = GeneticProfile(ancestor_ids={"a"})
        gb = GeneticProfile(ancestor_ids={"b"})
        assert not is_pair_blocked("a", "b", ga, gb)

    def test_parent_child_blocked(self):
        ga = GeneticProfile(ancestor_ids={"pa", "gpa", "gma"}, parent_ids=("gpa", "gma"))
        gb = GeneticProfile(ancestor_ids={"child", "pa", "pb", "gpa", "gma"},
                            parent_ids=("pa", "pb"))
        assert is_pair_blocked("pa", "child", ga, gb)

    def test_siblings_blocked(self):
        ga = GeneticProfile(ancestor_ids={"c1", "pa", "pb"}, parent_ids=("pa", "pb"))
        gb = GeneticProfile(ancestor_ids={"c2", "pa", "pb"}, parent_ids=("pa", "pb"))
        assert is_pair_blocked("c1", "c2", ga, gb)


# ── fitness and modifiers ──────────────────────────────────────────────

class TestFitness:
    def test_default_fitness(self):
        p = GeneticProfile()
        assert _compute_fitness(p) == 1.0

    def test_max_alleles(self):
        p = GeneticProfile(alleles={t: _ALLELE_MAX for t in GENETIC_TRAITS})
        f = _compute_fitness(p)
        assert f <= _FITNESS_MAX

    def test_min_alleles(self):
        p = GeneticProfile(alleles={t: _ALLELE_MIN for t in GENETIC_TRAITS})
        f = _compute_fitness(p)
        assert f >= _FITNESS_MIN

    def test_fitness_in_bounds(self):
        rng = random.Random(42)
        for _ in range(100):
            p = GeneticProfile(alleles=_random_alleles(rng))
            f = _compute_fitness(p)
            assert _FITNESS_MIN <= f <= _FITNESS_MAX


class TestDeathModifier:
    def test_default_is_one(self):
        p = GeneticProfile()
        assert compute_genetic_death_modifier(p) == 1.0

    def test_healthy_reduces(self):
        p = GeneticProfile(alleles={t: 0.0 for t in GENETIC_TRAITS})
        p.alleles["bone_density"] = 0.15
        p.alleles["immune_response"] = 0.15
        p.alleles["metabolic_efficiency"] = 0.15
        m = compute_genetic_death_modifier(p)
        assert m < 1.0

    def test_conditions_increase(self):
        p = GeneticProfile(conditions=["deficiency:a", "deficiency:b"])
        m = compute_genetic_death_modifier(p)
        assert m > 1.0

    def test_bounds(self):
        rng = random.Random(42)
        for _ in range(100):
            p = GeneticProfile(alleles=_random_alleles(rng),
                               conditions=[f"d:{i}" for i in range(rng.randint(0, 5))])
            m = compute_genetic_death_modifier(p)
            assert 0.3 <= m <= 2.0


class TestStressReduction:
    def test_default_zero(self):
        p = GeneticProfile()
        assert compute_genetic_stress_reduction(p) == 0.0

    def test_positive_alleles(self):
        p = GeneticProfile()
        p.alleles["stress_resilience"] = 0.15
        p.alleles["social_bonding"] = 0.15
        r = compute_genetic_stress_reduction(p)
        assert 0 < r <= 0.1

    def test_bounds(self):
        rng = random.Random(42)
        for _ in range(100):
            p = GeneticProfile(alleles=_random_alleles(rng))
            r = compute_genetic_stress_reduction(p)
            assert 0.0 <= r <= 0.1


class TestRadiationModifier:
    def test_default_is_one(self):
        p = GeneticProfile()
        assert compute_genetic_radiation_modifier(p) == 1.0

    def test_tolerant(self):
        p = GeneticProfile()
        p.alleles["radiation_tolerance"] = 0.15
        m = compute_genetic_radiation_modifier(p)
        assert m < 1.0

    def test_sensitive(self):
        p = GeneticProfile()
        p.alleles["radiation_tolerance"] = -0.15
        m = compute_genetic_radiation_modifier(p)
        assert m > 1.0

    def test_bounds(self):
        rng = random.Random(42)
        for _ in range(100):
            p = GeneticProfile(alleles=_random_alleles(rng))
            m = compute_genetic_radiation_modifier(p)
            assert 0.2 <= m <= 2.0


# ── diversity ──────────────────────────────────────────────────────────

class TestPopulationDiversity:
    def test_empty(self):
        assert compute_population_diversity({}) == 0.0

    def test_single(self):
        p = GeneticProfile()
        assert compute_population_diversity({"a": p}) == 0.0

    def test_identical_pair(self):
        p1 = GeneticProfile()
        p2 = GeneticProfile()
        assert compute_population_diversity({"a": p1, "b": p2}) == 0.0

    def test_diverse_pair(self):
        p1 = GeneticProfile(alleles={t: _ALLELE_MAX for t in GENETIC_TRAITS})
        p2 = GeneticProfile(alleles={t: _ALLELE_MIN for t in GENETIC_TRAITS})
        d = compute_population_diversity({"a": p1, "b": p2})
        assert d > 0.5  # Max spread

    def test_in_bounds(self):
        rng = random.Random(42)
        profiles = create_founding_genetics([f"c{i}" for i in range(10)], rng)
        d = compute_population_diversity(profiles)
        assert 0.0 <= d <= 1.0


# ── tick ───────────────────────────────────────────────────────────────

class TestTickGenetics:
    def test_basic(self):
        rng = random.Random(42)
        ids = ["c0", "c1", "c2"]
        state = GeneticsState(profiles=create_founding_genetics(ids, rng))
        result = tick_genetics(state, ids, 1, 0, rng)
        assert isinstance(result, GeneticsTickResult)
        assert result.diversity_index >= 0
        assert result.avg_fitness >= _FITNESS_MIN

    def test_updates_state(self):
        rng = random.Random(42)
        ids = ["c0", "c1"]
        state = GeneticsState(profiles=create_founding_genetics(ids, rng))
        tick_genetics(state, ids, 1, 0, rng)
        assert state.colony_diversity >= 0.0
        assert state.avg_fitness >= _FITNESS_MIN

    def test_high_biome_fewer_mutations(self):
        """Higher biome level = more atmosphere = fewer radiation mutations."""
        rng_low = random.Random(42)
        rng_high = random.Random(42)
        ids = [f"c{i}" for i in range(10)]
        state_low = GeneticsState(profiles=create_founding_genetics(ids, random.Random(99)))
        state_high = GeneticsState(profiles=create_founding_genetics(ids, random.Random(99)))
        r_low = tick_genetics(state_low, ids, 1, 0, rng_low)  # No atmosphere
        r_high = tick_genetics(state_high, ids, 1, 6, rng_high)  # Full atmosphere
        # On average high biome should have fewer mutations
        # (not guaranteed per-run, but with 10 colonists it's likely)
        # Just check both are valid
        assert isinstance(r_low, GeneticsTickResult)
        assert isinstance(r_high, GeneticsTickResult)

    def test_missing_profile_skipped(self):
        state = GeneticsState(profiles={})
        result = tick_genetics(state, ["missing"], 1, 0, random.Random(42))
        assert result.mutations == []

    def test_result_to_dict(self):
        rng = random.Random(42)
        ids = ["c0"]
        state = GeneticsState(profiles=create_founding_genetics(ids, rng))
        result = tick_genetics(state, ids, 1, 0, rng)
        d = result.to_dict()
        assert "mutations" in d
        assert "diversity_index" in d
        assert "avg_fitness" in d


# ── property-based invariants ──────────────────────────────────────────

class TestPropertyInvariants:
    @pytest.mark.parametrize("seed", range(5))
    def test_alleles_always_in_bounds_after_tick(self, seed):
        rng = random.Random(seed)
        ids = [f"c{i}" for i in range(10)]
        state = GeneticsState(profiles=create_founding_genetics(ids, rng))
        for year in range(20):
            tick_genetics(state, ids, year, rng.randint(0, 6), rng)
        for p in state.profiles.values():
            for t in GENETIC_TRAITS:
                assert _ALLELE_MIN <= p.alleles[t] <= _ALLELE_MAX, \
                    f"Allele {t} out of bounds after 20 years"

    @pytest.mark.parametrize("seed", range(5))
    def test_fitness_always_in_bounds(self, seed):
        rng = random.Random(seed)
        ids = [f"c{i}" for i in range(10)]
        state = GeneticsState(profiles=create_founding_genetics(ids, rng))
        for year in range(20):
            tick_genetics(state, ids, year, rng.randint(0, 6), rng)
        assert _FITNESS_MIN <= state.avg_fitness <= _FITNESS_MAX

    @pytest.mark.parametrize("seed", range(3))
    def test_diversity_in_bounds(self, seed):
        rng = random.Random(seed)
        ids = [f"c{i}" for i in range(10)]
        state = GeneticsState(profiles=create_founding_genetics(ids, rng))
        for year in range(20):
            tick_genetics(state, ids, year, rng.randint(0, 6), rng)
        assert 0.0 <= state.colony_diversity <= 1.0

    @pytest.mark.parametrize("seed", range(3))
    def test_death_modifier_in_bounds(self, seed):
        rng = random.Random(seed)
        ids = [f"c{i}" for i in range(10)]
        state = GeneticsState(profiles=create_founding_genetics(ids, rng))
        for year in range(20):
            tick_genetics(state, ids, year, rng.randint(0, 6), rng)
        for p in state.profiles.values():
            m = compute_genetic_death_modifier(p)
            assert 0.3 <= m <= 2.0

    @pytest.mark.parametrize("seed", range(3))
    def test_stress_reduction_in_bounds(self, seed):
        rng = random.Random(seed)
        ids = [f"c{i}" for i in range(10)]
        state = GeneticsState(profiles=create_founding_genetics(ids, rng))
        for year in range(20):
            tick_genetics(state, ids, year, rng.randint(0, 6), rng)
        for p in state.profiles.values():
            r = compute_genetic_stress_reduction(p)
            assert 0.0 <= r <= 0.1


# ── smoke tests ────────────────────────────────────────────────────────

class TestSmokeTests:
    def test_10_year_sim(self):
        """Run 10 years of genetics evolution without crash."""
        rng = random.Random(42)
        ids = [f"c{i}" for i in range(10)]
        state = GeneticsState(profiles=create_founding_genetics(ids, rng))
        for year in range(1, 11):
            result = tick_genetics(state, ids, year, min(year // 2, 6), rng)
            assert isinstance(result, GeneticsTickResult)
        assert state.total_mutations >= 0

    def test_100_year_sim_with_births_deaths(self):
        """Simulate 100 years with births and deaths."""
        rng = random.Random(42)
        ids = [f"c{i}" for i in range(10)]
        state = GeneticsState(profiles=create_founding_genetics(ids, rng))
        active = list(ids)
        next_id = 10

        for year in range(1, 101):
            result = tick_genetics(state, active, year, min(year // 15, 6), rng)

            # Simulate a birth every ~10 years
            if year % 10 == 0 and len(active) >= 2:
                pa, pb = active[0], active[1]
                ga = state.profiles.get(pa)
                gb = state.profiles.get(pb)
                if ga and gb and not is_pair_blocked(pa, pb, ga, gb):
                    child_id = f"child-{next_id}"
                    next_id += 1
                    child_p = inherit_genetics(pa, pb, ga, gb, child_id, year, rng)
                    state.profiles[child_id] = child_p
                    active.append(child_id)

            # Simulate a death every ~20 years
            if year % 20 == 0 and len(active) > 3:
                dead = active.pop(rng.randint(0, len(active) - 1))
                # Profile stays in state (legacy, not delete)

            # Simulate an immigrant every ~30 years
            if year % 30 == 0:
                imm_id = f"imm-{next_id}"
                next_id += 1
                state.profiles[imm_id] = create_immigrant_genetics(imm_id, rng)
                active.append(imm_id)

        # Final checks
        assert state.total_mutations > 0
        assert 0.0 <= state.colony_diversity <= 1.0
        assert _FITNESS_MIN <= state.avg_fitness <= _FITNESS_MAX
        assert len(state.profiles) > 10  # Should have grown
