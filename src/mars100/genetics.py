"""
Genetics organ for Mars-100 (engine v11.0).

Models population genetics in a small Mars colony: diploid genomes with
6 Mars-relevant loci, Mendelian inheritance, mutation under radiation,
pedigree-based kinship tracking, and genetic diversity monitoring.

Phenotype modifiers affect survival traits (death rate, resource consumption)
but NOT cognitive/social stats (those belong to psychology/behavior organs).

RNG offset: seed + 13337
"""
from __future__ import annotations

import math
import random as _random_module
from dataclasses import dataclass, field
from typing import Any

# --- Loci ---
# 6 physiological traits relevant to Mars survival.
# Each locus has two alleles (diploid). Values in [0, 1].
LOCUS_NAMES: tuple[str, ...] = (
    "radiation_resistance",
    "bone_density",
    "o2_efficiency",
    "stress_tolerance",
    "metabolic_rate",
    "immune_vigor",
)

# Dominance model per locus: "dominant" = max(a,b), "codominant" = avg(a,b)
DOMINANCE: dict[str, str] = {
    "radiation_resistance": "codominant",
    "bone_density": "dominant",
    "o2_efficiency": "codominant",
    "stress_tolerance": "codominant",
    "metabolic_rate": "dominant",
    "immune_vigor": "dominant",
}

# Phenotype modifier ranges: how much the expressed value scales a trait.
# modifier = lerp(min_mod, max_mod, expressed_value)
MODIFIER_RANGES: dict[str, tuple[float, float]] = {
    "radiation_resistance": (0.85, 1.15),
    "bone_density": (0.90, 1.10),
    "o2_efficiency": (0.85, 1.15),
    "stress_tolerance": (0.90, 1.10),
    "metabolic_rate": (0.90, 1.10),
    "immune_vigor": (0.85, 1.15),
}

MUTATION_RATE_BASE = 0.02
MUTATION_RATE_INBRED = 0.04
MUTATION_MAGNITUDE = 0.08
INBREEDING_THRESHOLD = 0.25
INBREEDING_PENALTY = 0.10
EARTH_DIVERSITY_SPREAD = 0.15
FOUNDER_DIVERSITY_SPREAD = 0.10
DIVERSITY_WARNING_THRESHOLD = 0.3


@dataclass
class Allele:
    """One allele at a genetic locus."""
    value: float  # 0.0-1.0

    def to_dict(self) -> dict:
        return {"value": round(self.value, 6)}

    @classmethod
    def from_dict(cls, d: dict) -> Allele:
        return cls(value=d.get("value", 0.5))


@dataclass
class Genome:
    """Diploid genome with one pair of alleles per locus."""
    loci: dict[str, tuple[Allele, Allele]]
    generation: int = 0
    parent_ids: tuple[str, str] | None = None
    mutations_log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        loci_d: dict[str, list[dict]] = {}
        for name, (a, b) in self.loci.items():
            loci_d[name] = [a.to_dict(), b.to_dict()]
        d: dict[str, Any] = {
            "loci": loci_d,
            "generation": self.generation,
            "mutations_log": list(self.mutations_log),
        }
        if self.parent_ids is not None:
            d["parent_ids"] = list(self.parent_ids)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Genome:
        loci: dict[str, tuple[Allele, Allele]] = {}
        for name, pair in d.get("loci", {}).items():
            if len(pair) == 2:
                loci[name] = (Allele.from_dict(pair[0]), Allele.from_dict(pair[1]))
        pids = d.get("parent_ids")
        parent_ids = tuple(pids) if pids and len(pids) == 2 else None
        return cls(
            loci=loci,
            generation=d.get("generation", 0),
            parent_ids=parent_ids,
            mutations_log=list(d.get("mutations_log", [])),
        )


def express_locus(allele_a: Allele, allele_b: Allele, locus: str) -> float:
    """Compute expressed phenotype value for one locus.

    Returns a value in [0, 1] based on the dominance model.
    """
    dom = DOMINANCE.get(locus, "codominant")
    if dom == "dominant":
        return max(allele_a.value, allele_b.value)
    # codominant: average
    return (allele_a.value + allele_b.value) / 2.0


def express_genome(genome: Genome) -> dict[str, float]:
    """Express all loci into phenotype values in [0, 1]."""
    expressed: dict[str, float] = {}
    for locus in LOCUS_NAMES:
        pair = genome.loci.get(locus)
        if pair is None:
            expressed[locus] = 0.5
        else:
            expressed[locus] = express_locus(pair[0], pair[1], locus)
    return expressed


def compute_phenotype_modifiers(genome: Genome) -> dict[str, float]:
    """Compute phenotype modifiers from genome.

    Each modifier is a multiplier in the locus-specific range.
    These are applied once (at birth/immigration) and stored,
    NOT recomputed/compounded each year.
    """
    expressed = express_genome(genome)
    modifiers: dict[str, float] = {}
    for locus in LOCUS_NAMES:
        lo, hi = MODIFIER_RANGES.get(locus, (0.9, 1.1))
        val = expressed.get(locus, 0.5)
        modifiers[locus] = lo + (hi - lo) * val
    return modifiers


# --- Genome creation ---

def create_founder_genome(rng: _random_module.Random) -> Genome:
    """Create a genome for a founding colonist.

    Founders have moderate diversity — drawn from normal distribution
    centered at 0.5. No parents.
    """
    loci: dict[str, tuple[Allele, Allele]] = {}
    for locus in LOCUS_NAMES:
        a = Allele(value=_clamp01(rng.gauss(0.5, FOUNDER_DIVERSITY_SPREAD)))
        b = Allele(value=_clamp01(rng.gauss(0.5, FOUNDER_DIVERSITY_SPREAD)))
        loci[locus] = (a, b)
    return Genome(loci=loci, generation=0, parent_ids=None)


def create_immigrant_genome(rng: _random_module.Random) -> Genome:
    """Create a genome for an immigrant from Earth.

    Immigrants have WIDER genetic diversity than colony-born colonists,
    representing Earth's much larger gene pool.
    """
    loci: dict[str, tuple[Allele, Allele]] = {}
    for locus in LOCUS_NAMES:
        a = Allele(value=_clamp01(rng.gauss(0.5, EARTH_DIVERSITY_SPREAD)))
        b = Allele(value=_clamp01(rng.gauss(0.5, EARTH_DIVERSITY_SPREAD)))
        loci[locus] = (a, b)
    return Genome(loci=loci, generation=0, parent_ids=None)


def inherit_genome(
    parent_a: Genome,
    parent_b: Genome,
    parent_a_id: str,
    parent_b_id: str,
    kinship: float,
    rng: _random_module.Random,
) -> Genome:
    """Create a child genome via Mendelian inheritance.

    For each locus, child gets one random allele from each parent.
    Mutation rate increases under inbreeding or radiation.
    """
    generation = max(parent_a.generation, parent_b.generation) + 1
    mutation_rate = (MUTATION_RATE_INBRED if kinship > INBREEDING_THRESHOLD
                     else MUTATION_RATE_BASE)

    loci: dict[str, tuple[Allele, Allele]] = {}
    mutations: list[dict] = []

    for locus in LOCUS_NAMES:
        # Select one allele from each parent
        pair_a = parent_a.loci.get(locus)
        pair_b = parent_b.loci.get(locus)

        if pair_a is None:
            allele_from_a = Allele(value=0.5)
        else:
            allele_from_a = Allele(value=rng.choice(pair_a).value)

        if pair_b is None:
            allele_from_b = Allele(value=0.5)
        else:
            allele_from_b = Allele(value=rng.choice(pair_b).value)

        # Mutation
        for allele, source in [(allele_from_a, "a"), (allele_from_b, "b")]:
            if rng.random() < mutation_rate:
                old_val = allele.value
                allele.value = _clamp01(allele.value + rng.gauss(0, MUTATION_MAGNITUDE))
                mutations.append({
                    "locus": locus,
                    "source": source,
                    "old": round(old_val, 6),
                    "new": round(allele.value, 6),
                    "generation": generation,
                })

        loci[locus] = (allele_from_a, allele_from_b)

    # Inbreeding penalty: shift all alleles slightly toward 0.5 (reduced fitness)
    if kinship > INBREEDING_THRESHOLD:
        penalty = INBREEDING_PENALTY * (kinship - INBREEDING_THRESHOLD)
        for locus in LOCUS_NAMES:
            a, b = loci[locus]
            a.value = _clamp01(a.value - penalty * (a.value - 0.5))
            b.value = _clamp01(b.value - penalty * (b.value - 0.5))

    return Genome(
        loci=loci,
        generation=generation,
        parent_ids=(parent_a_id, parent_b_id),
        mutations_log=mutations,
    )


# --- Pedigree kinship ---

def compute_pedigree_kinship(
    id_a: str,
    id_b: str,
    genomes: dict[str, Genome],
    _cache: dict[tuple[str, str], float] | None = None,
) -> float:
    """Compute Wright's coefficient of kinship from pedigree.

    Uses the recursive definition:
      k(A, B) = 0.5 * (k(parent_a_of_A, B) + k(parent_b_of_A, B))
    with k(X, X) = 0.5 * (1 + k(parent_a_of_X, parent_b_of_X))
    and k(unknown, Y) = 0.

    Memoized to avoid exponential blowup.
    """
    if _cache is None:
        _cache = {}

    key = (min(id_a, id_b), max(id_a, id_b))
    if key in _cache:
        return _cache[key]

    # Base cases
    if id_a not in genomes or id_b not in genomes:
        _cache[key] = 0.0
        return 0.0

    if id_a == id_b:
        genome = genomes[id_a]
        if genome.parent_ids is None:
            _cache[key] = 0.5
        else:
            pa, pb = genome.parent_ids
            _cache[key] = 0.5 * (1.0 + compute_pedigree_kinship(pa, pb, genomes, _cache))
        return _cache[key]

    # Recursive: decompose via whichever individual has known parents
    genome_a = genomes[id_a]
    genome_b = genomes[id_b]

    if genome_a.parent_ids is not None:
        pa, pb = genome_a.parent_ids
        result = 0.5 * (
            compute_pedigree_kinship(pa, id_b, genomes, _cache)
            + compute_pedigree_kinship(pb, id_b, genomes, _cache)
        )
    elif genome_b.parent_ids is not None:
        pa, pb = genome_b.parent_ids
        result = 0.5 * (
            compute_pedigree_kinship(id_a, pa, genomes, _cache)
            + compute_pedigree_kinship(id_a, pb, genomes, _cache)
        )
    else:
        # Both are founders with no known parents — unrelated
        result = 0.0

    _cache[key] = result
    return result


def compute_kinship_matrix(
    active_ids: list[str],
    genomes: dict[str, Genome],
) -> dict[str, dict[str, float]]:
    """Compute pairwise kinship for all active colonists."""
    cache: dict[tuple[str, str], float] = {}
    matrix: dict[str, dict[str, float]] = {}
    for a in active_ids:
        row: dict[str, float] = {}
        for b in active_ids:
            if a == b:
                row[b] = compute_pedigree_kinship(a, a, genomes, cache)
            else:
                row[b] = compute_pedigree_kinship(a, b, genomes, cache)
        matrix[a] = row
    return matrix


# --- Population diversity ---

def compute_heterozygosity(genomes: dict[str, Genome],
                           active_ids: list[str]) -> float:
    """Compute average observed heterozygosity across all loci.

    For each locus, for each individual: heterozygous if alleles differ
    by more than 0.1. Average across individuals and loci.
    Returns 0-1 where 1 = maximally diverse.
    """
    if not active_ids:
        return 0.0

    total = 0.0
    count = 0
    for cid in active_ids:
        genome = genomes.get(cid)
        if genome is None:
            continue
        for locus in LOCUS_NAMES:
            pair = genome.loci.get(locus)
            if pair is None:
                continue
            diff = abs(pair[0].value - pair[1].value)
            total += 1.0 if diff > 0.1 else 0.0
            count += 1

    return total / max(1, count)


def compute_allele_diversity(genomes: dict[str, Genome],
                             active_ids: list[str]) -> float:
    """Compute allele frequency diversity (expected heterozygosity).

    Uses Nei's gene diversity: H = 1 - sum(p_i^2) averaged across loci.
    Alleles are binned into 10 frequency classes.
    """
    if not active_ids:
        return 0.0

    n_bins = 10
    locus_diversities: list[float] = []

    for locus in LOCUS_NAMES:
        bins: list[int] = [0] * n_bins
        total_alleles = 0
        for cid in active_ids:
            genome = genomes.get(cid)
            if genome is None:
                continue
            pair = genome.loci.get(locus)
            if pair is None:
                continue
            for allele in pair:
                bin_idx = min(n_bins - 1, int(allele.value * n_bins))
                bins[bin_idx] += 1
                total_alleles += 1
        if total_alleles == 0:
            continue
        freqs = [b / total_alleles for b in bins]
        h = 1.0 - sum(f * f for f in freqs)
        locus_diversities.append(h)

    return sum(locus_diversities) / max(1, len(locus_diversities))


# --- Death rate modifiers ---

def compute_death_modifiers(genome: Genome) -> dict[str, float]:
    """Compute death-rate modifiers from genome expression.

    Returns multipliers < 1.0 that reduce specific death risks.
    """
    expressed = express_genome(genome)
    return {
        "radiation_mult": 1.0 - expressed.get("radiation_resistance", 0.5) * 0.3,
        "asphyxiation_mult": 1.0 - expressed.get("o2_efficiency", 0.5) * 0.3,
        "starvation_mult": 1.0 - expressed.get("metabolic_rate", 0.5) * 0.15,
        "illness_mult": 1.0 - expressed.get("immune_vigor", 0.5) * 0.25,
    }


def compute_birth_kinship_penalty(
    parent_a_id: str,
    parent_b_id: str,
    genomes: dict[str, Genome],
) -> float:
    """Compute birth probability penalty from kinship.

    Returns a multiplier in [0.0, 1.0]. High kinship → lower birth chance.
    """
    kinship = compute_pedigree_kinship(parent_a_id, parent_b_id, genomes)
    if kinship <= 0.0:
        return 1.0
    if kinship >= 0.5:
        return 0.0
    # Linear decay from 1.0 at kinship=0 to 0.0 at kinship=0.5
    return max(0.0, 1.0 - 2.0 * kinship)


# --- Main tick ---

@dataclass
class GeneticsState:
    """Colony-wide genetics state."""
    genomes: dict[str, Genome] = field(default_factory=dict)
    diversity_history: list[float] = field(default_factory=list)
    generation_count: int = 0
    total_mutations: int = 0
    diversity_warnings: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "genomes": {k: v.to_dict() for k, v in self.genomes.items()},
            "diversity_history": [round(d, 6) for d in self.diversity_history],
            "generation_count": self.generation_count,
            "total_mutations": self.total_mutations,
            "diversity_warnings": list(self.diversity_warnings),
        }

    @classmethod
    def from_dict(cls, d: dict) -> GeneticsState:
        genomes = {k: Genome.from_dict(v) for k, v in d.get("genomes", {}).items()}
        return cls(
            genomes=genomes,
            diversity_history=list(d.get("diversity_history", [])),
            generation_count=d.get("generation_count", 0),
            total_mutations=d.get("total_mutations", 0),
            diversity_warnings=list(d.get("diversity_warnings", [])),
        )


@dataclass
class GeneticsTickResult:
    """Output of one genetics tick."""
    diversity: float = 0.0
    allele_diversity: float = 0.0
    new_genomes: int = 0
    mutations_this_year: int = 0
    diversity_warning: str | None = None
    kinship_pairs: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "diversity": round(self.diversity, 6),
            "allele_diversity": round(self.allele_diversity, 6),
            "new_genomes": self.new_genomes,
            "mutations_this_year": self.mutations_this_year,
        }
        if self.diversity_warning:
            d["diversity_warning"] = self.diversity_warning
        if self.kinship_pairs:
            d["kinship_pairs"] = self.kinship_pairs
        return d


def initialize_founder_genomes(
    colonist_ids: list[str],
    rng: _random_module.Random,
) -> dict[str, Genome]:
    """Create genomes for all founding colonists."""
    genomes: dict[str, Genome] = {}
    for cid in colonist_ids:
        genomes[cid] = create_founder_genome(rng)
    return genomes


def tick_genetics(
    state: GeneticsState,
    active_ids: list[str],
    births: list[dict],
    deaths: list[dict],
    immigrants: list[dict],
    year: int,
    rng: _random_module.Random,
) -> GeneticsTickResult:
    """Advance genetics by one year.

    Processes births (inherit genomes), immigrants (new genomes),
    and deaths (archive genomes). Computes diversity metrics.
    """
    result = GeneticsTickResult()

    # Create genomes for births
    for birth in births:
        child_id = birth.get("id", "")
        parents = birth.get("parents", [])
        if len(parents) == 2 and child_id:
            pa_id, pb_id = parents[0], parents[1]
            pa_genome = state.genomes.get(pa_id)
            pb_genome = state.genomes.get(pb_id)
            if pa_genome is not None and pb_genome is not None:
                kinship = compute_pedigree_kinship(pa_id, pb_id, state.genomes)
                child_genome = inherit_genome(
                    pa_genome, pb_genome, pa_id, pb_id, kinship, rng)
                state.genomes[child_id] = child_genome
                result.new_genomes += 1
                result.mutations_this_year += len(child_genome.mutations_log)
                state.total_mutations += len(child_genome.mutations_log)
                state.generation_count = max(
                    state.generation_count, child_genome.generation)

    # Create genomes for immigrants
    for imm in immigrants:
        imm_id = imm.get("id", "")
        if imm_id and imm_id not in state.genomes:
            state.genomes[imm_id] = create_immigrant_genome(rng)
            result.new_genomes += 1

    # Compute diversity
    result.diversity = compute_heterozygosity(state.genomes, active_ids)
    result.allele_diversity = compute_allele_diversity(state.genomes, active_ids)
    state.diversity_history.append(result.diversity)

    # Diversity warning
    if result.diversity < DIVERSITY_WARNING_THRESHOLD and year > 20:
        warning = (f"Year {year}: genetic diversity critically low "
                   f"({result.diversity:.3f})")
        result.diversity_warning = warning
        state.diversity_warnings.append({"year": year, "diversity": result.diversity})

    # Compute notable kinship pairs (for the year report)
    if len(active_ids) <= 50:
        cache: dict[tuple[str, str], float] = {}
        for i, a in enumerate(active_ids):
            for b in active_ids[i + 1:]:
                k = compute_pedigree_kinship(a, b, state.genomes, cache)
                if k > 0.1:
                    result.kinship_pairs.append({
                        "a": a, "b": b, "kinship": round(k, 4)})

    return result


def _clamp01(v: float) -> float:
    """Clamp value to [0, 1]."""
    return max(0.0, min(1.0, v))
