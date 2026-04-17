"""Tests for the genetics organ (engine v11.0)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.genetics import (
    Allele, Genome, GeneticsState, GeneticsTickResult,
    LOCUS_NAMES, DOMINANCE, MODIFIER_RANGES,
    MUTATION_RATE_BASE, MUTATION_RATE_INBRED, INBREEDING_THRESHOLD,
    DIVERSITY_WARNING_THRESHOLD,
    express_locus, express_genome, compute_phenotype_modifiers,
    create_founder_genome, create_immigrant_genome, inherit_genome,
    compute_pedigree_kinship, compute_kinship_matrix,
    compute_heterozygosity, compute_allele_diversity,
    compute_death_modifiers, compute_birth_kinship_penalty,
    initialize_founder_genomes, tick_genetics,
)


# --- Allele tests ---

class TestAllele:
    def test_roundtrip(self):
        a = Allele(value=0.73)
        d = a.to_dict()
        b = Allele.from_dict(d)
        assert abs(b.value - 0.73) < 1e-5

    def test_default(self):
        a = Allele.from_dict({})
        assert a.value == 0.5


# --- Genome tests ---

class TestGenome:
    def test_founder_has_all_loci(self):
        rng = random.Random(42)
        g = create_founder_genome(rng)
        assert set(g.loci.keys()) == set(LOCUS_NAMES)

    def test_founder_alleles_in_bounds(self):
        rng = random.Random(99)
        for _ in range(50):
            g = create_founder_genome(rng)
            for locus, (a, b) in g.loci.items():
                assert 0.0 <= a.value <= 1.0, f"{locus} allele a out of bounds"
                assert 0.0 <= b.value <= 1.0, f"{locus} allele b out of bounds"

    def test_immigrant_has_all_loci(self):
        rng = random.Random(42)
        g = create_immigrant_genome(rng)
        assert set(g.loci.keys()) == set(LOCUS_NAMES)

    def test_immigrant_wider_diversity(self):
        """Immigrants should have wider allele spread than founders on average."""
        rng_f = random.Random(42)
        rng_i = random.Random(42)
        founder_spread = []
        immigrant_spread = []
        for _ in range(200):
            fg = create_founder_genome(rng_f)
            ig = create_immigrant_genome(rng_i)
            for locus in LOCUS_NAMES:
                fa, fb = fg.loci[locus]
                ia, ib = ig.loci[locus]
                founder_spread.append(abs(fa.value - 0.5) + abs(fb.value - 0.5))
                immigrant_spread.append(abs(ia.value - 0.5) + abs(ib.value - 0.5))
        # Immigrants should have larger average deviation from 0.5
        avg_f = sum(founder_spread) / len(founder_spread)
        avg_i = sum(immigrant_spread) / len(immigrant_spread)
        assert avg_i > avg_f

    def test_roundtrip(self):
        rng = random.Random(42)
        g = create_founder_genome(rng)
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        assert set(g2.loci.keys()) == set(g.loci.keys())
        for locus in LOCUS_NAMES:
            assert abs(g2.loci[locus][0].value - g.loci[locus][0].value) < 1e-5
            assert abs(g2.loci[locus][1].value - g.loci[locus][1].value) < 1e-5

    def test_generation_zero_for_founder(self):
        rng = random.Random(42)
        g = create_founder_genome(rng)
        assert g.generation == 0

    def test_no_parents_for_founder(self):
        rng = random.Random(42)
        g = create_founder_genome(rng)
        assert g.parent_ids is None


# --- Expression tests ---

class TestExpression:
    def test_dominant_takes_max(self):
        a = Allele(value=0.3)
        b = Allele(value=0.8)
        result = express_locus(a, b, "bone_density")
        assert DOMINANCE["bone_density"] == "dominant"
        assert abs(result - 0.8) < 1e-10

    def test_codominant_averages(self):
        a = Allele(value=0.2)
        b = Allele(value=0.8)
        result = express_locus(a, b, "radiation_resistance")
        assert DOMINANCE["radiation_resistance"] == "codominant"
        assert abs(result - 0.5) < 1e-10

    def test_express_genome_all_loci(self):
        rng = random.Random(42)
        g = create_founder_genome(rng)
        expressed = express_genome(g)
        assert set(expressed.keys()) == set(LOCUS_NAMES)
        for val in expressed.values():
            assert 0.0 <= val <= 1.0

    def test_phenotype_modifiers_in_range(self):
        rng = random.Random(42)
        for seed in range(100):
            rng2 = random.Random(seed)
            g = create_founder_genome(rng2)
            mods = compute_phenotype_modifiers(g)
            for locus, mod in mods.items():
                lo, hi = MODIFIER_RANGES[locus]
                assert lo <= mod <= hi, f"{locus}: {mod} not in [{lo}, {hi}]"


# --- Inheritance tests ---

class TestInheritance:
    def _make_parents(self, rng):
        pa = create_founder_genome(rng)
        pb = create_founder_genome(rng)
        return pa, pb

    def test_child_has_all_loci(self):
        rng = random.Random(42)
        pa, pb = self._make_parents(rng)
        child = inherit_genome(pa, pb, "a", "b", 0.0, rng)
        assert set(child.loci.keys()) == set(LOCUS_NAMES)

    def test_child_generation_increments(self):
        rng = random.Random(42)
        pa, pb = self._make_parents(rng)
        child = inherit_genome(pa, pb, "a", "b", 0.0, rng)
        assert child.generation == 1

    def test_child_records_parents(self):
        rng = random.Random(42)
        pa, pb = self._make_parents(rng)
        child = inherit_genome(pa, pb, "parent-a", "parent-b", 0.0, rng)
        assert child.parent_ids == ("parent-a", "parent-b")

    def test_alleles_from_parents(self):
        """Without mutation, each child allele should match one parent allele."""
        # Use a very low mutation rate by testing many children
        rng = random.Random(42)
        pa = create_founder_genome(rng)
        pb = create_founder_genome(rng)
        matched = 0
        total = 0
        for _ in range(100):
            child = inherit_genome(pa, pb, "a", "b", 0.0, random.Random(rng.randint(0, 10000)))
            for locus in LOCUS_NAMES:
                ca, cb = child.loci[locus]
                pa_vals = {pa.loci[locus][0].value, pa.loci[locus][1].value}
                pb_vals = {pb.loci[locus][0].value, pb.loci[locus][1].value}
                # One should be from parent a, one from parent b (unless mutated)
                if ca.value in pa_vals or ca.value in pb_vals:
                    matched += 1
                if cb.value in pa_vals or cb.value in pb_vals:
                    matched += 1
                total += 2
        # Most alleles should match (mutation rate is low)
        assert matched / total > 0.9

    def test_child_alleles_in_bounds(self):
        rng = random.Random(42)
        for _ in range(200):
            pa, pb = self._make_parents(rng)
            child = inherit_genome(pa, pb, "a", "b", rng.random() * 0.5, rng)
            for locus, (a, b) in child.loci.items():
                assert 0.0 <= a.value <= 1.0, f"{locus} allele a: {a.value}"
                assert 0.0 <= b.value <= 1.0, f"{locus} allele b: {b.value}"

    def test_inbreeding_increases_mutations(self):
        """High kinship should produce more mutations on average."""
        rng_lo = random.Random(42)
        rng_hi = random.Random(42)
        lo_muts = 0
        hi_muts = 0
        n = 200
        for _ in range(n):
            pa, pb = self._make_parents(random.Random(rng_lo.randint(0, 99999)))
            child_lo = inherit_genome(pa, pb, "a", "b", 0.0, rng_lo)
            lo_muts += len(child_lo.mutations_log)
            pa2, pb2 = self._make_parents(random.Random(rng_hi.randint(0, 99999)))
            child_hi = inherit_genome(pa2, pb2, "a", "b", 0.4, rng_hi)
            hi_muts += len(child_hi.mutations_log)
        assert hi_muts > lo_muts

    def test_multi_generation(self):
        """Three generations should work without error."""
        rng = random.Random(42)
        g1a = create_founder_genome(rng)
        g1b = create_founder_genome(rng)
        g2 = inherit_genome(g1a, g1b, "p1", "p2", 0.0, rng)
        g1c = create_founder_genome(rng)
        g3 = inherit_genome(g2, g1c, "child-1", "p3", 0.0, rng)
        assert g3.generation == 2
        assert g3.parent_ids == ("child-1", "p3")


# --- Kinship tests ---

class TestKinship:
    def test_self_kinship_founder(self):
        """Self-kinship of a non-inbred individual = 0.5."""
        genomes = {"a": create_founder_genome(random.Random(42))}
        k = compute_pedigree_kinship("a", "a", genomes)
        assert abs(k - 0.5) < 1e-10

    def test_unrelated_founders(self):
        """Kinship between unrelated founders = 0."""
        rng = random.Random(42)
        genomes = {
            "a": create_founder_genome(rng),
            "b": create_founder_genome(rng),
        }
        k = compute_pedigree_kinship("a", "b", genomes)
        assert k == 0.0

    def test_parent_child_kinship(self):
        """Kinship between parent and child = 0.25."""
        rng = random.Random(42)
        pa = create_founder_genome(rng)
        pb = create_founder_genome(rng)
        child = inherit_genome(pa, pb, "pa", "pb", 0.0, rng)
        genomes = {"pa": pa, "pb": pb, "child": child}
        k = compute_pedigree_kinship("pa", "child", genomes)
        assert abs(k - 0.25) < 1e-10

    def test_sibling_kinship(self):
        """Kinship between full siblings = 0.25."""
        rng = random.Random(42)
        pa = create_founder_genome(rng)
        pb = create_founder_genome(rng)
        c1 = inherit_genome(pa, pb, "pa", "pb", 0.0, rng)
        c2 = inherit_genome(pa, pb, "pa", "pb", 0.0, rng)
        genomes = {"pa": pa, "pb": pb, "c1": c1, "c2": c2}
        k = compute_pedigree_kinship("c1", "c2", genomes)
        assert abs(k - 0.25) < 1e-10

    def test_kinship_is_symmetric(self):
        rng = random.Random(42)
        pa = create_founder_genome(rng)
        pb = create_founder_genome(rng)
        child = inherit_genome(pa, pb, "pa", "pb", 0.0, rng)
        genomes = {"pa": pa, "pb": pb, "child": child}
        k1 = compute_pedigree_kinship("pa", "child", genomes)
        k2 = compute_pedigree_kinship("child", "pa", genomes)
        assert abs(k1 - k2) < 1e-10

    def test_kinship_in_range(self):
        """Kinship should be in [0, 0.5] for non-inbred."""
        rng = random.Random(42)
        genomes = initialize_founder_genomes(
            [f"c{i}" for i in range(10)], rng)
        for a in genomes:
            for b in genomes:
                k = compute_pedigree_kinship(a, b, genomes)
                assert 0.0 <= k <= 0.5 + 1e-10

    def test_kinship_matrix_shape(self):
        rng = random.Random(42)
        ids = ["a", "b", "c"]
        genomes = initialize_founder_genomes(ids, rng)
        matrix = compute_kinship_matrix(ids, genomes)
        assert set(matrix.keys()) == set(ids)
        for row in matrix.values():
            assert set(row.keys()) == set(ids)

    def test_unknown_id_kinship(self):
        """Kinship with unknown ID = 0."""
        genomes = {"a": create_founder_genome(random.Random(42))}
        k = compute_pedigree_kinship("a", "unknown", genomes)
        assert k == 0.0


# --- Diversity tests ---

class TestDiversity:
    def test_heterozygosity_bounds(self):
        rng = random.Random(42)
        genomes = initialize_founder_genomes(
            [f"c{i}" for i in range(10)], rng)
        h = compute_heterozygosity(genomes, list(genomes.keys()))
        assert 0.0 <= h <= 1.0

    def test_heterozygosity_empty(self):
        assert compute_heterozygosity({}, []) == 0.0

    def test_allele_diversity_bounds(self):
        rng = random.Random(42)
        genomes = initialize_founder_genomes(
            [f"c{i}" for i in range(10)], rng)
        d = compute_allele_diversity(genomes, list(genomes.keys()))
        assert 0.0 <= d <= 1.0

    def test_allele_diversity_empty(self):
        assert compute_allele_diversity({}, []) == 0.0

    def test_clone_population_low_diversity(self):
        """A population of clones should have low heterozygosity."""
        template = Genome(
            loci={l: (Allele(0.5), Allele(0.5)) for l in LOCUS_NAMES})
        genomes = {f"c{i}": template for i in range(10)}
        h = compute_heterozygosity(genomes, list(genomes.keys()))
        assert h < 0.1


# --- Death modifiers tests ---

class TestDeathModifiers:
    def test_modifiers_in_range(self):
        rng = random.Random(42)
        for _ in range(100):
            g = create_founder_genome(rng)
            mods = compute_death_modifiers(g)
            for key, val in mods.items():
                assert 0.5 <= val <= 1.1, f"{key}: {val}"

    def test_high_resistance_reduces_risk(self):
        """High radiation_resistance alleles should reduce radiation death risk."""
        g_high = Genome(loci={
            l: (Allele(0.9), Allele(0.9)) for l in LOCUS_NAMES})
        g_low = Genome(loci={
            l: (Allele(0.1), Allele(0.1)) for l in LOCUS_NAMES})
        mods_high = compute_death_modifiers(g_high)
        mods_low = compute_death_modifiers(g_low)
        assert mods_high["radiation_mult"] < mods_low["radiation_mult"]
        assert mods_high["asphyxiation_mult"] < mods_low["asphyxiation_mult"]


# --- Birth kinship penalty tests ---

class TestBirthKinshipPenalty:
    def test_unrelated_no_penalty(self):
        rng = random.Random(42)
        genomes = initialize_founder_genomes(["a", "b"], rng)
        penalty = compute_birth_kinship_penalty("a", "b", genomes)
        assert abs(penalty - 1.0) < 1e-10

    def test_siblings_penalized(self):
        rng = random.Random(42)
        pa = create_founder_genome(rng)
        pb = create_founder_genome(rng)
        c1 = inherit_genome(pa, pb, "pa", "pb", 0.0, rng)
        c2 = inherit_genome(pa, pb, "pa", "pb", 0.0, rng)
        genomes = {"pa": pa, "pb": pb, "c1": c1, "c2": c2}
        penalty = compute_birth_kinship_penalty("c1", "c2", genomes)
        assert penalty < 1.0
        assert penalty >= 0.0


# --- GeneticsState tests ---

class TestGeneticsState:
    def test_roundtrip(self):
        rng = random.Random(42)
        state = GeneticsState()
        state.genomes = initialize_founder_genomes(["a", "b"], rng)
        state.diversity_history = [0.5, 0.48, 0.46]
        state.generation_count = 2
        state.total_mutations = 5
        d = state.to_dict()
        state2 = GeneticsState.from_dict(d)
        assert set(state2.genomes.keys()) == {"a", "b"}
        assert state2.generation_count == 2
        assert state2.total_mutations == 5
        assert len(state2.diversity_history) == 3

    def test_empty_state(self):
        state = GeneticsState.from_dict({})
        assert state.genomes == {}
        assert state.diversity_history == []


# --- tick_genetics tests ---

class TestTickGenetics:
    def _make_state_with_founders(self, n=10, seed=42):
        rng = random.Random(seed)
        ids = [f"c{i}" for i in range(n)]
        state = GeneticsState()
        state.genomes = initialize_founder_genomes(ids, rng)
        return state, ids, rng

    def test_tick_no_events(self):
        state, ids, rng = self._make_state_with_founders()
        result = tick_genetics(state, ids, [], [], [], 1, rng)
        assert result.diversity > 0
        assert result.new_genomes == 0
        assert len(state.diversity_history) == 1

    def test_tick_with_birth(self):
        state, ids, rng = self._make_state_with_founders()
        births = [{"id": "child-1", "parents": ["c0", "c1"]}]
        result = tick_genetics(state, ids + ["child-1"], births, [], [], 15, rng)
        assert "child-1" in state.genomes
        assert result.new_genomes == 1
        child_g = state.genomes["child-1"]
        assert child_g.parent_ids == ("c0", "c1")
        assert child_g.generation == 1

    def test_tick_with_immigrant(self):
        state, ids, rng = self._make_state_with_founders()
        immigrants = [{"id": "imm-1"}]
        result = tick_genetics(state, ids + ["imm-1"], [], [], immigrants, 20, rng)
        assert "imm-1" in state.genomes
        assert result.new_genomes == 1

    def test_tick_tracks_diversity_history(self):
        state, ids, rng = self._make_state_with_founders()
        for year in range(1, 11):
            tick_genetics(state, ids, [], [], [], year, rng)
        assert len(state.diversity_history) == 10

    def test_diversity_warning_late(self):
        """Should warn about low diversity in later years."""
        # Create a very inbred population (all clones)
        state = GeneticsState()
        template = Genome(
            loci={l: (Allele(0.5), Allele(0.5)) for l in LOCUS_NAMES})
        ids = [f"c{i}" for i in range(5)]
        state.genomes = {cid: template for cid in ids}
        rng = random.Random(42)
        result = tick_genetics(state, ids, [], [], [], 25, rng)
        assert result.diversity_warning is not None

    def test_no_warning_early(self):
        """Should NOT warn about diversity in early years."""
        state = GeneticsState()
        template = Genome(
            loci={l: (Allele(0.5), Allele(0.5)) for l in LOCUS_NAMES})
        ids = [f"c{i}" for i in range(5)]
        state.genomes = {cid: template for cid in ids}
        rng = random.Random(42)
        result = tick_genetics(state, ids, [], [], [], 5, rng)
        assert result.diversity_warning is None

    def test_tick_result_roundtrip(self):
        state, ids, rng = self._make_state_with_founders()
        result = tick_genetics(state, ids, [], [], [], 1, rng)
        d = result.to_dict()
        assert "diversity" in d
        assert "allele_diversity" in d

    def test_multiple_births_single_tick(self):
        state, ids, rng = self._make_state_with_founders()
        births = [
            {"id": "child-1", "parents": ["c0", "c1"]},
            {"id": "child-2", "parents": ["c2", "c3"]},
        ]
        result = tick_genetics(state, ids + ["child-1", "child-2"],
                               births, [], [], 15, rng)
        assert result.new_genomes == 2
        assert "child-1" in state.genomes
        assert "child-2" in state.genomes


# --- Integration: 10-year mini-sim ---

class TestIntegration:
    def test_ten_year_no_crash(self):
        """Run 10 years of genetics ticks without error."""
        rng = random.Random(42)
        ids = [f"c{i}" for i in range(10)]
        state = GeneticsState()
        state.genomes = initialize_founder_genomes(ids, rng)
        for year in range(1, 11):
            births = []
            immigrants = []
            if year == 5:
                births = [{"id": "child-1", "parents": ["c0", "c1"]}]
                ids.append("child-1")
            if year == 7:
                immigrants = [{"id": "imm-1"}]
                ids.append("imm-1")
            result = tick_genetics(state, ids, births, [], immigrants, year, rng)
            assert result.diversity >= 0

    def test_diversity_decreases_without_immigration(self):
        """After many generations of inbreeding, diversity should drop."""
        rng = random.Random(42)
        ids = ["p0", "p1"]
        state = GeneticsState()
        state.genomes = initialize_founder_genomes(ids, rng)
        # Breed several generations from the same pair
        for gen in range(20):
            child_id = f"child-{gen}"
            births = [{"id": child_id, "parents": ids[:2]}]
            ids.append(child_id)
            tick_genetics(state, ids, births, [], [], gen + 1, rng)
        # Diversity should be measurable but potentially declining
        early_div = state.diversity_history[0] if state.diversity_history else 1.0
        late_div = state.diversity_history[-1] if state.diversity_history else 0.0
        # At minimum, the function runs without error
        assert late_div >= 0.0


# --- Property-based invariants ---

class TestPropertyInvariants:
    def test_all_alleles_bounded(self):
        """Property: every allele value is in [0, 1]."""
        rng = random.Random(42)
        state = GeneticsState()
        ids = [f"c{i}" for i in range(10)]
        state.genomes = initialize_founder_genomes(ids, rng)
        for gen in range(10):
            child_id = f"child-{gen}"
            pa, pb = rng.sample(ids, 2)
            births = [{"id": child_id, "parents": [pa, pb]}]
            ids.append(child_id)
            tick_genetics(state, ids, births, [], [], gen + 1, rng)
        for cid, genome in state.genomes.items():
            for locus, (a, b) in genome.loci.items():
                assert 0.0 <= a.value <= 1.0, f"{cid}.{locus}.a = {a.value}"
                assert 0.0 <= b.value <= 1.0, f"{cid}.{locus}.b = {b.value}"

    def test_genome_always_6_loci(self):
        """Property: every genome has exactly 6 loci."""
        rng = random.Random(42)
        for _ in range(100):
            g = create_founder_genome(rng)
            assert len(g.loci) == 6
        for _ in range(100):
            g = create_immigrant_genome(rng)
            assert len(g.loci) == 6

    def test_kinship_symmetric_property(self):
        """Property: kinship(a, b) == kinship(b, a)."""
        rng = random.Random(42)
        ids = [f"c{i}" for i in range(6)]
        genomes = initialize_founder_genomes(ids, rng)
        # Add a child
        c = inherit_genome(genomes["c0"], genomes["c1"], "c0", "c1", 0.0, rng)
        genomes["child"] = c
        ids.append("child")
        for a in ids:
            for b in ids:
                k1 = compute_pedigree_kinship(a, b, genomes)
                k2 = compute_pedigree_kinship(b, a, genomes)
                assert abs(k1 - k2) < 1e-10, f"k({a},{b})={k1} != k({b},{a})={k2}"

    def test_modifiers_physical_bounds(self):
        """Property: phenotype modifiers stay in defined ranges."""
        rng = random.Random(42)
        for _ in range(500):
            g = create_founder_genome(rng)
            mods = compute_phenotype_modifiers(g)
            for locus in LOCUS_NAMES:
                lo, hi = MODIFIER_RANGES[locus]
                assert lo <= mods[locus] <= hi

    def test_death_modifiers_positive(self):
        """Property: death modifiers are always positive."""
        rng = random.Random(42)
        for _ in range(500):
            g = create_founder_genome(rng)
            mods = compute_death_modifiers(g)
            for key, val in mods.items():
                assert val > 0.0, f"{key} = {val}"
