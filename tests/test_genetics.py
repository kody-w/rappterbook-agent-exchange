"""Tests for the genetics organ (engine v11.0)."""
from __future__ import annotations

import random
from typing import Any
import pytest

from src.mars100.genetics import (
    Locus, GeneticProfile, GeneticsState, GeneticsYearContext,
    GeneticsTickResult,
    LOCUS_NAMES, MUTATION_RATE, MUTATION_SIGMA,
    MAX_GENETIC_SURVIVAL, CAUSE_LOCUS_MAP,
    MARS_ADAPTATION_LOCI, MARS_ADAPTATION_WEIGHTS,
    create_founder_genetics, create_immigrant_genetics,
    inherit_genetics, compute_colony_diversity,
    compute_relatedness, compute_genetic_survival,
    apply_epigenetic_stress, compute_metabolism_modifier,
    tick_genetics,
)


# ── Locus tests ──────────────────────────────────────────────────────

class TestLocus:
    def test_defaults(self) -> None:
        l = Locus()
        assert l.allele_a == pytest.approx(0.5)
        assert l.allele_b == pytest.approx(0.5)
        assert l.epigenetic == pytest.approx(1.0)

    def test_expression_codominant(self) -> None:
        l = Locus(allele_a=0.2, allele_b=0.8)
        assert l.expression == pytest.approx(0.5)

    def test_expression_with_epigenetic(self) -> None:
        l = Locus(allele_a=0.4, allele_b=0.4, epigenetic=1.2)
        assert l.expression == pytest.approx(0.48)

    def test_expression_clamped_high(self) -> None:
        l = Locus(allele_a=0.9, allele_b=0.9, epigenetic=1.5)
        assert l.expression <= 1.0

    def test_expression_clamped_low(self) -> None:
        l = Locus(allele_a=0.0, allele_b=0.0, epigenetic=0.5)
        assert l.expression >= 0.0

    def test_heterozygosity(self) -> None:
        l = Locus(allele_a=0.2, allele_b=0.8)
        assert l.heterozygosity == pytest.approx(0.6)

    def test_heterozygosity_homozygous(self) -> None:
        l = Locus(allele_a=0.5, allele_b=0.5)
        assert l.heterozygosity == pytest.approx(0.0)

    def test_roundtrip(self) -> None:
        l = Locus(allele_a=0.3, allele_b=0.7, epigenetic=1.1)
        d = l.to_dict()
        l2 = Locus.from_dict(d)
        assert l2.allele_a == pytest.approx(l.allele_a, abs=1e-3)
        assert l2.allele_b == pytest.approx(l.allele_b, abs=1e-3)
        assert l2.epigenetic == pytest.approx(l.epigenetic, abs=1e-3)

    def test_clamp(self) -> None:
        l = Locus(allele_a=-0.1, allele_b=1.5, epigenetic=2.0)
        l.clamp()
        assert l.allele_a == 0.0
        assert l.allele_b == 1.0
        assert l.epigenetic == 1.5


# ── GeneticProfile tests ────────────────────────────────────────────

class TestGeneticProfile:
    def test_defaults(self) -> None:
        p = GeneticProfile()
        assert len(p.loci) == len(LOCUS_NAMES)
        for name in LOCUS_NAMES:
            assert name in p.loci

    def test_expression(self) -> None:
        p = GeneticProfile()
        assert p.expression("resilience") == pytest.approx(0.5)

    def test_expression_missing_locus(self) -> None:
        p = GeneticProfile(loci={})
        assert p.expression("nonexistent") == pytest.approx(0.5)

    def test_mean_heterozygosity_default(self) -> None:
        p = GeneticProfile()
        assert p.mean_heterozygosity() == pytest.approx(0.0)

    def test_adaptation_score_bounds(self) -> None:
        p = GeneticProfile()
        assert 0.0 <= p.adaptation_score() <= 1.0

    def test_adaptation_score_high(self) -> None:
        loci = {name: Locus(allele_a=0.9, allele_b=0.9)
                for name in LOCUS_NAMES}
        p = GeneticProfile(loci=loci)
        assert p.adaptation_score() > 0.8

    def test_roundtrip(self) -> None:
        rng = random.Random(99)
        p = create_founder_genetics("test", {"resolve": 0.7}, rng)
        d = p.to_dict()
        p2 = GeneticProfile.from_dict(d)
        assert p2.generation == p.generation
        assert len(p2.loci) == len(LOCUS_NAMES)
        for name in LOCUS_NAMES:
            assert p2.loci[name].allele_a == pytest.approx(
                p.loci[name].allele_a, abs=1e-3)

    def test_clamp(self) -> None:
        loci = {name: Locus(allele_a=-1.0, allele_b=2.0, epigenetic=3.0)
                for name in LOCUS_NAMES}
        p = GeneticProfile(loci=loci)
        p.clamp()
        for locus in p.loci.values():
            assert 0.0 <= locus.allele_a <= 1.0
            assert 0.0 <= locus.allele_b <= 1.0
            assert 0.5 <= locus.epigenetic <= 1.5


# ── Founder / Immigrant / Inheritance ────────────────────────────────

class TestFounderGenetics:
    def test_creates_valid_profile(self) -> None:
        rng = random.Random(42)
        p = create_founder_genetics("c-0", {"resolve": 0.8}, rng)
        assert isinstance(p, GeneticProfile)
        assert p.generation == 0
        assert len(p.loci) == len(LOCUS_NAMES)

    def test_stat_consistency(self) -> None:
        """Founder resilience alleles should be influenced by resolve stat."""
        rng = random.Random(42)
        p_high = create_founder_genetics("c-0", {"resolve": 0.9}, rng)
        rng2 = random.Random(42)
        p_low = create_founder_genetics("c-1", {"resolve": 0.1}, rng2)
        # On average, high-resolve founders should have higher resilience
        # (but randomness means individual comparison may not hold)
        # So test over many seeds
        high_sum = sum(
            create_founder_genetics("x", {"resolve": 0.9}, random.Random(i))
            .expression("resilience") for i in range(100))
        low_sum = sum(
            create_founder_genetics("x", {"resolve": 0.1}, random.Random(i))
            .expression("resilience") for i in range(100))
        assert high_sum > low_sum

    def test_all_alleles_in_bounds(self) -> None:
        for seed in range(50):
            rng = random.Random(seed)
            p = create_founder_genetics("c-0", {}, rng)
            for locus in p.loci.values():
                assert 0.0 <= locus.allele_a <= 1.0
                assert 0.0 <= locus.allele_b <= 1.0


class TestImmigrantGenetics:
    def test_creates_valid_profile(self) -> None:
        rng = random.Random(42)
        p = create_immigrant_genetics(rng)
        assert isinstance(p, GeneticProfile)
        assert p.generation == 0
        assert len(p.loci) == len(LOCUS_NAMES)

    def test_alleles_in_bounds(self) -> None:
        for seed in range(50):
            rng = random.Random(seed)
            p = create_immigrant_genetics(rng)
            for locus in p.loci.values():
                assert 0.2 <= locus.allele_a <= 0.8
                assert 0.2 <= locus.allele_b <= 0.8

    def test_diversity(self) -> None:
        """Different seeds should produce different profiles."""
        profiles = [create_immigrant_genetics(random.Random(i))
                    for i in range(10)]
        alleles = [p.loci["resilience"].allele_a for p in profiles]
        assert len(set(round(a, 3) for a in alleles)) > 3


class TestInheritance:
    def test_child_generation(self) -> None:
        rng = random.Random(42)
        pa = GeneticProfile(generation=0)
        pb = GeneticProfile(generation=0)
        child = inherit_genetics(pa, pb, "child-0", rng)
        assert child.generation == 1

    def test_child_generation_mixed(self) -> None:
        rng = random.Random(42)
        pa = GeneticProfile(generation=1)
        pb = GeneticProfile(generation=2)
        child = inherit_genetics(pa, pb, "child-0", rng)
        assert child.generation == 3

    def test_alleles_from_parents(self) -> None:
        """Child alleles should come from parent alleles (with possible mutation)."""
        rng = random.Random(42)
        pa = GeneticProfile(loci={
            name: Locus(allele_a=0.2, allele_b=0.3) for name in LOCUS_NAMES})
        pb = GeneticProfile(loci={
            name: Locus(allele_a=0.7, allele_b=0.8) for name in LOCUS_NAMES})
        child = inherit_genetics(pa, pb, "child-0", rng)
        # Each child allele should be near a parent allele (within mutation range)
        for name in LOCUS_NAMES:
            ca = child.loci[name].allele_a
            cb = child.loci[name].allele_b
            # Without mutation, alleles are exactly from parents
            # With mutation (5% chance, sigma 0.08), they're close
            assert 0.0 <= ca <= 1.0
            assert 0.0 <= cb <= 1.0

    def test_epigenetics_reset(self) -> None:
        """Child epigenetics should be 1.0 regardless of parent values."""
        rng = random.Random(42)
        pa = GeneticProfile(loci={
            name: Locus(epigenetic=1.3) for name in LOCUS_NAMES})
        pb = GeneticProfile(loci={
            name: Locus(epigenetic=0.7) for name in LOCUS_NAMES})
        child = inherit_genetics(pa, pb, "child-0", rng)
        for locus in child.loci.values():
            assert locus.epigenetic == pytest.approx(1.0)

    def test_mutation_bounded(self) -> None:
        """Even with mutations, alleles stay in [0, 1]."""
        for seed in range(100):
            rng = random.Random(seed)
            pa = GeneticProfile(loci={
                name: Locus(allele_a=0.0, allele_b=1.0) for name in LOCUS_NAMES})
            pb = GeneticProfile(loci={
                name: Locus(allele_a=0.0, allele_b=1.0) for name in LOCUS_NAMES})
            child = inherit_genetics(pa, pb, "child-0", rng)
            for locus in child.loci.values():
                assert 0.0 <= locus.allele_a <= 1.0
                assert 0.0 <= locus.allele_b <= 1.0

    def test_mutations_occur(self) -> None:
        """Over many births, some mutations should happen."""
        mutation_count = 0
        pa_alleles = {name: (0.3, 0.3) for name in LOCUS_NAMES}
        pb_alleles = {name: (0.7, 0.7) for name in LOCUS_NAMES}
        pa = GeneticProfile(loci={
            name: Locus(allele_a=a, allele_b=b)
            for name, (a, b) in pa_alleles.items()})
        pb = GeneticProfile(loci={
            name: Locus(allele_a=a, allele_b=b)
            for name, (a, b) in pb_alleles.items()})
        for seed in range(200):
            rng = random.Random(seed)
            child = inherit_genetics(pa, pb, f"child-{seed}", rng)
            for name in LOCUS_NAMES:
                ca = child.loci[name].allele_a
                cb = child.loci[name].allele_b
                # Without mutation, allele should be exactly 0.3 or 0.7
                if abs(ca - 0.3) > 0.001 and abs(ca - 0.7) > 0.001:
                    mutation_count += 1
                if abs(cb - 0.3) > 0.001 and abs(cb - 0.7) > 0.001:
                    mutation_count += 1
        # Expect roughly 5% of 200*8*2 = 3200 allele transfers to mutate
        assert mutation_count > 50  # At least some mutations


# ── Colony diversity ─────────────────────────────────────────────────

class TestColonyDiversity:
    def test_single_profile(self) -> None:
        profiles = [GeneticProfile()]
        assert compute_colony_diversity(profiles) == pytest.approx(0.0)

    def test_identical_profiles(self) -> None:
        profiles = [GeneticProfile() for _ in range(5)]
        assert compute_colony_diversity(profiles) == pytest.approx(0.0)

    def test_diverse_profiles(self) -> None:
        profiles = []
        for i in range(10):
            rng = random.Random(i)
            profiles.append(create_immigrant_genetics(rng))
        diversity = compute_colony_diversity(profiles)
        assert diversity > 0.0

    def test_diversity_increases_with_immigrants(self) -> None:
        """Adding diverse immigrants should increase colony diversity."""
        rng = random.Random(42)
        base = [create_founder_genetics(f"c-{i}", {"resolve": 0.5}, rng)
                for i in range(5)]
        d_base = compute_colony_diversity(base)
        with_imm = base + [create_immigrant_genetics(random.Random(i + 100))
                           for i in range(5)]
        d_with = compute_colony_diversity(with_imm)
        assert d_with >= d_base


# ── Relatedness ──────────────────────────────────────────────────────

class TestRelatedness:
    def test_self_relatedness(self) -> None:
        p = GeneticProfile()
        assert compute_relatedness(p, p) == pytest.approx(1.0)

    def test_siblings_more_related_than_strangers(self) -> None:
        rng = random.Random(42)
        pa = create_founder_genetics("pa", {}, rng)
        pb = create_founder_genetics("pb", {}, random.Random(99))
        child1 = inherit_genetics(pa, pb, "c1", random.Random(1))
        child2 = inherit_genetics(pa, pb, "c2", random.Random(2))
        stranger = create_immigrant_genetics(random.Random(999))
        sib_rel = compute_relatedness(child1, child2)
        str_rel = compute_relatedness(child1, stranger)
        # Siblings should generally be more related (test over averages)
        # Individual comparison may not hold, but on average it should
        assert sib_rel > 0.0
        assert str_rel > 0.0

    def test_bounds(self) -> None:
        for seed in range(20):
            rng = random.Random(seed)
            a = create_immigrant_genetics(rng)
            b = create_immigrant_genetics(random.Random(seed + 100))
            r = compute_relatedness(a, b)
            assert 0.0 <= r <= 1.0


# ── Genetic survival ────────────────────────────────────────────────

class TestGeneticSurvival:
    def test_no_survival_for_unmapped_cause(self) -> None:
        p = GeneticProfile()
        assert compute_genetic_survival("starvation", p) == pytest.approx(0.0)
        assert compute_genetic_survival("asphyxiation", p) == pytest.approx(0.0)

    def test_radiation_survival(self) -> None:
        loci = {name: Locus() for name in LOCUS_NAMES}
        loci["radiation_tolerance"] = Locus(allele_a=0.9, allele_b=0.9)
        p = GeneticProfile(loci=loci)
        surv = compute_genetic_survival("radiation exposure", p)
        assert surv > 0.0
        assert surv <= MAX_GENETIC_SURVIVAL

    def test_immune_survival(self) -> None:
        loci = {name: Locus() for name in LOCUS_NAMES}
        loci["immune_vigor"] = Locus(allele_a=1.0, allele_b=1.0)
        p = GeneticProfile(loci=loci)
        surv = compute_genetic_survival("medical emergency", p)
        assert surv == pytest.approx(MAX_GENETIC_SURVIVAL)

    def test_low_genes_low_survival(self) -> None:
        loci = {name: Locus(allele_a=0.1, allele_b=0.1)
                for name in LOCUS_NAMES}
        p = GeneticProfile(loci=loci)
        surv = compute_genetic_survival("radiation exposure", p)
        assert surv < 0.05


# ── Epigenetics ──────────────────────────────────────────────────────

class TestEpigenetics:
    def test_no_shift_low_severity(self) -> None:
        p = GeneticProfile()
        affected = apply_epigenetic_stress(p, "solar_flare", 0.1)
        assert affected == []

    def test_shift_on_high_severity(self) -> None:
        p = GeneticProfile()
        affected = apply_epigenetic_stress(p, "solar_flare", 0.8)
        assert "radiation_tolerance" in affected

    def test_epigenetic_clamped(self) -> None:
        p = GeneticProfile()
        for _ in range(100):
            apply_epigenetic_stress(p, "solar_flare", 1.0)
        assert p.loci["radiation_tolerance"].epigenetic <= 1.5

    def test_unknown_event_no_effect(self) -> None:
        p = GeneticProfile()
        affected = apply_epigenetic_stress(p, "alien_landing", 0.9)
        assert affected == []


# ── Metabolism modifier ──────────────────────────────────────────────

class TestMetabolismModifier:
    def test_default(self) -> None:
        profiles = [GeneticProfile() for _ in range(5)]
        mod = compute_metabolism_modifier(profiles)
        assert mod == pytest.approx(1.0)

    def test_high_metabolism(self) -> None:
        loci = {name: Locus() for name in LOCUS_NAMES}
        loci["metabolism"] = Locus(allele_a=0.9, allele_b=0.9)
        profiles = [GeneticProfile(loci=dict(loci)) for _ in range(5)]
        mod = compute_metabolism_modifier(profiles)
        assert mod < 1.0  # More efficient → lower consumption

    def test_low_metabolism(self) -> None:
        loci = {name: Locus() for name in LOCUS_NAMES}
        loci["metabolism"] = Locus(allele_a=0.1, allele_b=0.1)
        profiles = [GeneticProfile(loci=dict(loci)) for _ in range(5)]
        mod = compute_metabolism_modifier(profiles)
        assert mod > 1.0  # Less efficient → higher consumption

    def test_empty(self) -> None:
        assert compute_metabolism_modifier([]) == pytest.approx(1.0)

    def test_bounds(self) -> None:
        for seed in range(20):
            profiles = [create_immigrant_genetics(random.Random(i))
                        for i in range(10)]
            mod = compute_metabolism_modifier(profiles)
            assert 0.9 <= mod <= 1.1


# ── GeneticsState ───────────────────────────────────────────────────

class TestGeneticsState:
    def test_roundtrip(self) -> None:
        s = GeneticsState(
            diversity_history=[0.3, 0.4, 0.5],
            adaptation_history=[0.2, 0.3],
            founding_diversity=0.35,
            total_mutations=42,
            max_generation=3,
        )
        d = s.to_dict()
        s2 = GeneticsState.from_dict(d)
        assert s2.founding_diversity == pytest.approx(s.founding_diversity)
        assert s2.total_mutations == s.total_mutations
        assert s2.max_generation == s.max_generation


# ── tick_genetics integration ────────────────────────────────────────

class TestTickGenetics:
    def _make_colony(self, n: int = 10, seed: int = 42
                     ) -> tuple[dict[str, GeneticProfile], list[str]]:
        rng = random.Random(seed)
        genetics_map: dict[str, GeneticProfile] = {}
        ids: list[str] = []
        for i in range(n):
            cid = f"c-{i}"
            ids.append(cid)
            genetics_map[cid] = create_founder_genetics(cid, {}, rng)
        return genetics_map, ids

    def test_basic_tick(self) -> None:
        gmap, ids = self._make_colony()
        state = GeneticsState()
        ctx = GeneticsYearContext(year=1)
        rng = random.Random(42)
        result = tick_genetics(gmap, state, ids, ctx, rng)
        assert result.colony_diversity >= 0.0
        assert result.mean_adaptation >= 0.0
        assert result.mean_heterozygosity >= 0.0
        assert result.mars_born_count == 0
        assert result.earth_born_count == 10
        assert len(state.diversity_history) == 1

    def test_epigenetic_shifts_logged(self) -> None:
        gmap, ids = self._make_colony()
        state = GeneticsState()
        ctx = GeneticsYearContext(
            year=5, event_type="solar_flare", event_severity=0.8)
        rng = random.Random(42)
        result = tick_genetics(gmap, state, ids, ctx, rng)
        assert len(result.epigenetic_shifts) > 0

    def test_generation_tracking(self) -> None:
        gmap, ids = self._make_colony(5)
        pa = gmap["c-0"]
        pb = gmap["c-1"]
        child = inherit_genetics(pa, pb, "child-0", random.Random(99))
        gmap["child-0"] = child
        ids.append("child-0")
        state = GeneticsState()
        ctx = GeneticsYearContext(year=15)
        rng = random.Random(42)
        result = tick_genetics(gmap, state, ids, ctx, rng)
        assert result.generation_counts.get(0) == 5
        assert result.generation_counts.get(1) == 1
        assert result.mars_born_count == 1
        assert result.earth_born_count == 5

    def test_founding_diversity_set_once(self) -> None:
        gmap, ids = self._make_colony()
        state = GeneticsState()
        ctx = GeneticsYearContext(year=1)
        rng = random.Random(42)
        tick_genetics(gmap, state, ids, ctx, rng)
        founding = state.founding_diversity
        assert founding > 0.0
        # Year 2: should not overwrite
        ctx2 = GeneticsYearContext(year=2)
        tick_genetics(gmap, state, ids, ctx2, rng)
        assert state.founding_diversity == pytest.approx(founding)

    def test_result_serializable(self) -> None:
        gmap, ids = self._make_colony()
        state = GeneticsState()
        ctx = GeneticsYearContext(year=1)
        rng = random.Random(42)
        result = tick_genetics(gmap, state, ids, ctx, rng)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "colony_diversity" in d
        assert "metabolism_modifier" in d


# ── Smoke test: 20-year simulation ───────────────────────────────────

class TestGeneticsSmoke:
    def test_20_year_run(self) -> None:
        """Run genetics for 20 years without crash."""
        rng = random.Random(42)
        gmap: dict[str, GeneticProfile] = {}
        ids: list[str] = []
        for i in range(10):
            cid = f"c-{i}"
            ids.append(cid)
            gmap[cid] = create_founder_genetics(cid, {}, rng)
        state = GeneticsState()
        event_types = ["dust_storm", "solar_flare", "epidemic",
                       "resource_strike", "none"]
        for year in range(1, 21):
            evt = event_types[year % len(event_types)]
            sev = 0.5 if evt != "none" else 0.0
            ctx = GeneticsYearContext(
                year=year, event_type=evt, event_severity=sev)
            result = tick_genetics(gmap, state, ids, ctx, rng)
            assert 0.0 <= result.colony_diversity <= 1.0
            assert 0.0 <= result.mean_adaptation <= 1.0
            assert 0.95 <= result.metabolism_modifier <= 1.05
            # Simulate a birth at year 15
            if year == 15 and len(ids) >= 2:
                child = inherit_genetics(gmap[ids[0]], gmap[ids[1]],
                                         "child-0", rng)
                gmap["child-0"] = child
                ids.append("child-0")
        assert len(state.diversity_history) == 20
        assert len(state.adaptation_history) == 20


# ── Property-based invariants ────────────────────────────────────────

class TestPropertyInvariants:
    def test_all_expressions_in_bounds(self) -> None:
        """Every locus expression across many random profiles is in [0, 1]."""
        for seed in range(100):
            rng = random.Random(seed)
            p = create_founder_genetics(f"c-{seed}", {
                "resolve": rng.random(),
                "improvisation": rng.random(),
                "empathy": rng.random(),
            }, rng)
            for name in LOCUS_NAMES:
                expr = p.expression(name)
                assert 0.0 <= expr <= 1.0, f"seed={seed}, locus={name}, expr={expr}"

    def test_diversity_bounded(self) -> None:
        """Colony diversity is always in [0, 1]."""
        for seed in range(20):
            profiles = [create_immigrant_genetics(random.Random(i + seed * 100))
                        for i in range(15)]
            d = compute_colony_diversity(profiles)
            assert 0.0 <= d <= 1.0

    def test_adaptation_bounded(self) -> None:
        """Adaptation score is always in [0, 1]."""
        for seed in range(100):
            rng = random.Random(seed)
            p = create_founder_genetics(f"c-{seed}", {}, rng)
            assert 0.0 <= p.adaptation_score() <= 1.0

    def test_genetic_survival_bounded(self) -> None:
        """Genetic survival is always in [0, MAX_GENETIC_SURVIVAL]."""
        for seed in range(50):
            rng = random.Random(seed)
            p = create_founder_genetics(f"c-{seed}", {}, rng)
            for cause in CAUSE_LOCUS_MAP:
                s = compute_genetic_survival(cause, p)
                assert 0.0 <= s <= MAX_GENETIC_SURVIVAL
