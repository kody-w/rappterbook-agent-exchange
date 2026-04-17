"""Tests for the genetics organ (engine v11.0)."""
from __future__ import annotations

import random
import pytest

from src.mars100.genetics import (
    ALL_LOCI, DIVERSITY_DEATH_PENALTY, DIVERSITY_PENALTY_THRESHOLD,
    SKILL_LOCI, SKILL_LOCUS_TO_SKILL, STAT_LOCI, TRAIT_DEFS,
    Genome, GeneticsTickResult, Locus,
    blend_child_stats, compute_allele_drift, compute_diversity_index,
    compute_relatedness, compute_skill_learning_modifier,
    compute_trait_cohesion_modifier, compute_trait_death_modifier,
    count_inbreeding_pairs, create_founding_genome, create_immigrant_genome,
    determine_traits, inherit_genome, tick_genetics,
)


class TestLocus:
    def test_defaults(self) -> None:
        loc = Locus()
        assert loc.allele_a == 0.5
        assert loc.allele_b == 0.5

    def test_expression_mean(self) -> None:
        loc = Locus(allele_a=0.2, allele_b=0.8)
        assert loc.expression() == pytest.approx(0.5)

    def test_roundtrip(self) -> None:
        loc = Locus(allele_a=0.3, allele_b=0.7)
        d = loc.to_dict()
        loc2 = Locus.from_dict(d)
        assert loc2.allele_a == pytest.approx(loc.allele_a, abs=1e-3)
        assert loc2.allele_b == pytest.approx(loc.allele_b, abs=1e-3)

    def test_clamp(self) -> None:
        loc = Locus(allele_a=-0.5, allele_b=1.5)
        loc.clamp()
        assert loc.allele_a == 0.0
        assert loc.allele_b == 1.0

    def test_expression_bounds(self) -> None:
        for a, b in [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0)]:
            loc = Locus(allele_a=a, allele_b=b)
            assert 0.0 <= loc.expression() <= 1.0


class TestGenome:
    def test_defaults(self) -> None:
        g = Genome()
        assert g.loci == {}
        assert g.generation == 0

    def test_expression_missing_locus(self) -> None:
        g = Genome()
        assert g.expression("resolve") == 0.5

    def test_all_expressions(self) -> None:
        g = Genome(loci={"resolve": Locus(0.3, 0.7), "empathy": Locus(0.8, 0.8)})
        expr = g.all_expressions()
        assert expr["resolve"] == pytest.approx(0.5)
        assert expr["empathy"] == pytest.approx(0.8)

    def test_heterozygosity(self) -> None:
        g = Genome(loci={"resolve": Locus(0.3, 0.7), "empathy": Locus(0.5, 0.5)})
        assert g.heterozygosity() == pytest.approx(0.2)

    def test_roundtrip(self) -> None:
        g = Genome(loci={"resolve": Locus(0.3, 0.7)}, parent_a_id="a",
                   parent_b_id="b", generation=3, traits=["resilient"])
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        assert g2.generation == 3
        assert g2.parent_a_id == "a"
        assert "resilient" in g2.traits


class TestFoundingGenome:
    def test_creates_all_loci(self) -> None:
        rng = random.Random(42)
        stats = {n: 0.5 for n in STAT_LOCI}
        skills = {SKILL_LOCUS_TO_SKILL[s]: 0.3 for s in SKILL_LOCI}
        g = create_founding_genome(stats, skills, rng)
        assert set(g.loci.keys()) == set(ALL_LOCI)

    def test_allele_values_bounded(self) -> None:
        rng = random.Random(99)
        stats = {n: 0.9 for n in STAT_LOCI}
        skills = {SKILL_LOCUS_TO_SKILL[s]: 0.9 for s in SKILL_LOCI}
        g = create_founding_genome(stats, skills, rng)
        for loc in g.loci.values():
            assert 0.0 <= loc.allele_a <= 1.0
            assert 0.0 <= loc.allele_b <= 1.0

    def test_generation_zero(self) -> None:
        g = create_founding_genome({n: 0.5 for n in STAT_LOCI},
                                    {SKILL_LOCUS_TO_SKILL[s]: 0.5 for s in SKILL_LOCI},
                                    random.Random(42))
        assert g.generation == 0

    def test_stat_loci_near_initial(self) -> None:
        stats = {"resolve": 0.9, "improvisation": 0.1, "empathy": 0.5,
                 "hoarding": 0.5, "faith": 0.5, "paranoia": 0.5}
        skills = {SKILL_LOCUS_TO_SKILL[s]: 0.0 for s in SKILL_LOCI}
        g = create_founding_genome(stats, skills, random.Random(42))
        assert g.expression("resolve") > 0.6
        assert g.expression("improvisation") < 0.4

    def test_deterministic(self) -> None:
        stats = {n: 0.5 for n in STAT_LOCI}
        skills = {SKILL_LOCUS_TO_SKILL[s]: 0.5 for s in SKILL_LOCI}
        g1 = create_founding_genome(stats, skills, random.Random(42))
        g2 = create_founding_genome(stats, skills, random.Random(42))
        for name in ALL_LOCI:
            assert g1.loci[name].allele_a == g2.loci[name].allele_a


class TestImmigrantGenome:
    def test_creates_all_loci(self) -> None:
        g = create_immigrant_genome(random.Random(42))
        assert set(g.loci.keys()) == set(ALL_LOCI)

    def test_diverse_alleles(self) -> None:
        rng = random.Random(42)
        genomes = [create_immigrant_genome(rng) for _ in range(10)]
        values = [g.expression("resolve") for g in genomes]
        assert max(values) - min(values) > 0.1


class TestInheritance:
    def test_child_has_all_loci(self) -> None:
        pa = Genome(loci={n: Locus(0.3, 0.7) for n in ALL_LOCI})
        pb = Genome(loci={n: Locus(0.2, 0.8) for n in ALL_LOCI})
        child, _ = inherit_genome(pa, pb, "a", "b", random.Random(42))
        assert set(child.loci.keys()) == set(ALL_LOCI)

    def test_parent_ids_recorded(self) -> None:
        pa = Genome(loci={n: Locus(0.5, 0.5) for n in ALL_LOCI})
        pb = Genome(loci={n: Locus(0.5, 0.5) for n in ALL_LOCI})
        child, _ = inherit_genome(pa, pb, "parent-a", "parent-b", random.Random(42))
        assert child.parent_a_id == "parent-a"
        assert child.parent_b_id == "parent-b"

    def test_generation_incremented(self) -> None:
        pa = Genome(loci={n: Locus(0.5, 0.5) for n in ALL_LOCI}, generation=2)
        pb = Genome(loci={n: Locus(0.5, 0.5) for n in ALL_LOCI}, generation=3)
        child, _ = inherit_genome(pa, pb, "a", "b", random.Random(42))
        assert child.generation == 4

    def test_alleles_bounded(self) -> None:
        pa = Genome(loci={n: Locus(0.0, 1.0) for n in ALL_LOCI})
        pb = Genome(loci={n: Locus(0.0, 1.0) for n in ALL_LOCI})
        for seed in range(50):
            child, _ = inherit_genome(pa, pb, "a", "b", random.Random(seed), True)
            for loc in child.loci.values():
                assert 0.0 <= loc.allele_a <= 1.0
                assert 0.0 <= loc.allele_b <= 1.0

    def test_radiation_increases_mutations(self) -> None:
        pa = Genome(loci={n: Locus(0.5, 0.5) for n in ALL_LOCI})
        pb = Genome(loci={n: Locus(0.5, 0.5) for n in ALL_LOCI})
        normal = sum(inherit_genome(pa, pb, "a", "b", random.Random(s), False)[1]
                     for s in range(100))
        rad = sum(inherit_genome(pa, pb, "a", "b", random.Random(s), True)[1]
                  for s in range(100))
        assert rad > normal

    def test_deterministic(self) -> None:
        pa = Genome(loci={n: Locus(0.3, 0.7) for n in ALL_LOCI})
        pb = Genome(loci={n: Locus(0.2, 0.8) for n in ALL_LOCI})
        c1, m1 = inherit_genome(pa, pb, "a", "b", random.Random(42))
        c2, m2 = inherit_genome(pa, pb, "a", "b", random.Random(42))
        assert m1 == m2
        for name in ALL_LOCI:
            assert c1.loci[name].allele_a == c2.loci[name].allele_a


class TestTraitDetermination:
    def test_resilient_high_resolve(self) -> None:
        g = Genome(loci={"resolve": Locus(0.8, 0.8)})
        assert "resilient" in determine_traits(g, [])

    def test_fragile_low_resolve(self) -> None:
        g = Genome(loci={"resolve": Locus(0.1, 0.1)})
        assert "fragile" in determine_traits(g, [])

    def test_no_resilient_mid_resolve(self) -> None:
        g = Genome(loci={"resolve": Locus(0.5, 0.5)})
        traits = determine_traits(g, [])
        assert "resilient" not in traits
        assert "fragile" not in traits

    def test_hysteresis_keeps_trait(self) -> None:
        g = Genome(loci={"resolve": Locus(0.65, 0.65)})
        assert "resilient" in determine_traits(g, ["resilient"])

    def test_hysteresis_loses_trait(self) -> None:
        g = Genome(loci={"resolve": Locus(0.55, 0.55)})
        assert "resilient" not in determine_traits(g, ["resilient"])

    def test_multiple_traits(self) -> None:
        g = Genome(loci={"resolve": Locus(0.8, 0.8), "empathy": Locus(0.9, 0.9),
                         "paranoia": Locus(0.9, 0.9)})
        traits = determine_traits(g, [])
        assert "resilient" in traits
        assert "empathic_bond" in traits
        assert "paranoid_gene" in traits


class TestDiversity:
    def test_homogeneous_low(self) -> None:
        genomes = [Genome(loci={n: Locus(0.5, 0.5) for n in ALL_LOCI})
                   for _ in range(5)]
        assert compute_diversity_index(genomes) < 0.01

    def test_diverse_population(self) -> None:
        rng = random.Random(42)
        genomes = [create_immigrant_genome(rng) for _ in range(10)]
        assert compute_diversity_index(genomes) > 0.01

    def test_too_few(self) -> None:
        assert compute_diversity_index([Genome()]) == 0.0


class TestRelatedness:
    def test_full_siblings(self) -> None:
        ga = Genome(parent_a_id="mom", parent_b_id="dad")
        gb = Genome(parent_a_id="mom", parent_b_id="dad")
        assert compute_relatedness(ga, gb, {}) == pytest.approx(0.5)

    def test_half_siblings(self) -> None:
        ga = Genome(parent_a_id="mom", parent_b_id="d1")
        gb = Genome(parent_a_id="mom", parent_b_id="d2")
        assert compute_relatedness(ga, gb, {}) == pytest.approx(0.25)

    def test_unrelated(self) -> None:
        ga = Genome(parent_a_id="a", parent_b_id="b")
        gb = Genome(parent_a_id="c", parent_b_id="d")
        assert compute_relatedness(ga, gb, {}) == 0.0

    def test_founders(self) -> None:
        assert compute_relatedness(Genome(), Genome(), {}) == 0.0


class TestInbreedingPairs:
    def test_siblings_counted(self) -> None:
        gmap = {"c1": Genome(parent_a_id="m", parent_b_id="d"),
                "c2": Genome(parent_a_id="m", parent_b_id="d")}
        assert count_inbreeding_pairs(["c1", "c2"], gmap) == 1

    def test_unrelated_zero(self) -> None:
        gmap = {"c1": Genome(parent_a_id="a", parent_b_id="b"),
                "c2": Genome(parent_a_id="c", parent_b_id="d")}
        assert count_inbreeding_pairs(["c1", "c2"], gmap) == 0


class TestTraitModifiers:
    def test_resilient(self) -> None:
        assert compute_trait_death_modifier(Genome(traits=["resilient"])) == pytest.approx(0.85)

    def test_fragile(self) -> None:
        assert compute_trait_death_modifier(Genome(traits=["fragile"])) == pytest.approx(1.2)

    def test_neutral(self) -> None:
        assert compute_trait_death_modifier(Genome(traits=[])) == pytest.approx(1.0)

    def test_empathic_cohesion(self) -> None:
        assert compute_trait_cohesion_modifier(Genome(traits=["empathic_bond"])) > 0

    def test_paranoid_cohesion(self) -> None:
        assert compute_trait_cohesion_modifier(Genome(traits=["paranoid_gene"])) < 0


class TestSkillLearning:
    def test_high_aptitude(self) -> None:
        g = Genome(loci={"terraforming_apt": Locus(0.9, 0.9)})
        assert compute_skill_learning_modifier(g)["terraforming"] > 1.5

    def test_low_aptitude(self) -> None:
        g = Genome(loci={"terraforming_apt": Locus(0.1, 0.1)})
        assert compute_skill_learning_modifier(g)["terraforming"] < 0.8

    def test_all_skills(self) -> None:
        g = Genome(loci={n: Locus(0.5, 0.5) for n in SKILL_LOCI})
        mods = compute_skill_learning_modifier(g)
        for skill in SKILL_LOCUS_TO_SKILL.values():
            assert skill in mods


class TestTickGenetics:
    def test_empty(self) -> None:
        r = tick_genetics({}, [], 1, random.Random(42))
        assert r.diversity_index == 0.0

    def test_homogeneous_penalty(self) -> None:
        gmap = {f"c{i}": Genome(loci={n: Locus(0.5, 0.5) for n in ALL_LOCI})
                for i in range(10)}
        r = tick_genetics(gmap, list(gmap.keys()), 1, random.Random(42))
        assert r.death_rate_modifier == pytest.approx(DIVERSITY_DEATH_PENALTY)

    def test_trait_census(self) -> None:
        gmap = {"c1": Genome(traits=["resilient"]), "c2": Genome(traits=["resilient"]),
                "c3": Genome(traits=["fragile"])}
        r = tick_genetics(gmap, ["c1", "c2", "c3"], 1, random.Random(42))
        assert r.trait_counts.get("resilient") == 2

    def test_result_serializable(self) -> None:
        r = GeneticsTickResult(diversity_index=0.15, mean_generation=2.5)
        d = r.to_dict()
        assert d["diversity_index"] == pytest.approx(0.15, abs=1e-3)


class TestBlendChildStats:
    def test_nature_dominates(self) -> None:
        result = blend_child_stats(
            {"resolve": 0.9, "empathy": 0.1},
            {"resolve": 0.5, "empathy": 0.5},
            random.Random(42))
        assert result["resolve"] > 0.65
        assert result["empathy"] < 0.4

    def test_bounded(self) -> None:
        result = blend_child_stats(
            {n: 1.0 for n in STAT_LOCI},
            {n: 1.0 for n in STAT_LOCI},
            random.Random(42))
        for v in result.values():
            assert 0.0 <= v <= 1.0


class TestAlleleDrift:
    def test_empty(self) -> None:
        assert compute_allele_drift([]) == {}

    def test_uniform(self) -> None:
        genomes = [Genome(loci={n: Locus(0.6, 0.6) for n in ALL_LOCI})
                   for _ in range(5)]
        drift = compute_allele_drift(genomes)
        for name in ALL_LOCI:
            assert drift[name] == pytest.approx(0.6)


class TestMultiYearSmoke:
    def test_10_generations(self) -> None:
        rng = random.Random(42)
        stats = {n: 0.5 for n in STAT_LOCI}
        skills = {SKILL_LOCUS_TO_SKILL[s]: 0.5 for s in SKILL_LOCI}
        gmap: dict[str, Genome] = {}
        for i in range(10):
            gmap[f"founder-{i}"] = create_founding_genome(stats, skills, rng)
        active_ids = list(gmap.keys())
        for year in range(1, 11):
            result = tick_genetics(gmap, active_ids, year, rng)
            assert result.diversity_index >= 0.0
            if len(active_ids) >= 2:
                pa_id = active_ids[0]
                pb_id = active_ids[1]
                child_id = f"child-{year}"
                child, muts = inherit_genome(
                    gmap[pa_id], gmap[pb_id], pa_id, pb_id, rng)
                gmap[child_id] = child
                active_ids.append(child_id)
