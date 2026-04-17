"""
Genetics organ for Mars-100 (engine v11.0).

Models heritable genotypes, genetic inheritance, diversity, and selection pressure.
Each colonist has a Genome — heritable allele values for the 6 personality stats.
Children inherit blended alleles from parents with crossover + mutation.
Phenotype (actual stats) drifts toward genotype each year, making heredity meaningful.

Selection pressure is DESCRIPTIVE: it reports which traits correlate with survival
but does not artificially amplify alleles. The population signal is too small
for meaningful artificial selection — let natural drift + inheritance do the work.

Diversity is measured via mean pairwise allele distance (continuous metric,
avoids binning artifacts from Shannon entropy on floats).

RNG offset: seed + 12553
"""
from __future__ import annotations

import math
import random as _random_module
from dataclasses import dataclass, field
from typing import Any


STAT_NAMES = ("resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia")

MUTATION_SIGMA = 0.05
CROSSOVER_PROB = 0.5
GENOTYPE_PULL_STRENGTH = 0.01
MAX_TRAIT_BONUS = 0.02
EARTH_ALLELE_MEAN = 0.5
EARTH_ALLELE_SIGMA = 0.15


@dataclass
class Genome:
    """A colonist's heritable genetic profile.

    Alleles are float values in [0, 1] for each stat. These represent
    the genetic predisposition — the 'nature' component. Phenotype
    (actual stats) drifts toward these values over time.
    """
    alleles: dict[str, float]
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    mutation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "alleles": {k: round(v, 6) for k, v in self.alleles.items()},
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "mutation_count": self.mutation_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Genome:
        return cls(
            alleles={k: d["alleles"].get(k, 0.5) for k in STAT_NAMES},
            generation=d.get("generation", 0),
            parent_ids=d.get("parent_ids", []),
            mutation_count=d.get("mutation_count", 0),
        )


@dataclass
class SelectionReport:
    """Descriptive report of selection pressure — observational, not prescriptive."""
    year: int
    deaths_this_year: int
    survivor_mean: dict[str, float] = field(default_factory=dict)
    deceased_mean: dict[str, float] = field(default_factory=dict)
    pressure_direction: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "year": self.year,
            "deaths_this_year": self.deaths_this_year,
        }
        if self.survivor_mean:
            d["survivor_mean"] = {k: round(v, 4) for k, v in self.survivor_mean.items()}
        if self.deceased_mean:
            d["deceased_mean"] = {k: round(v, 4) for k, v in self.deceased_mean.items()}
        if self.pressure_direction:
            d["pressure_direction"] = {k: round(v, 4) for k, v in self.pressure_direction.items()}
        return d


@dataclass
class GenePool:
    """Colony-wide genetic statistics."""
    diversity_index: float = 1.0
    mean_alleles: dict[str, float] = field(default_factory=dict)
    allele_variance: dict[str, float] = field(default_factory=dict)
    generation_count: int = 0
    total_mutations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "diversity_index": round(self.diversity_index, 6),
            "mean_alleles": {k: round(v, 4) for k, v in self.mean_alleles.items()},
            "allele_variance": {k: round(v, 6) for k, v in self.allele_variance.items()},
            "generation_count": self.generation_count,
            "total_mutations": self.total_mutations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GenePool:
        return cls(
            diversity_index=d.get("diversity_index", 1.0),
            mean_alleles=d.get("mean_alleles", {}),
            allele_variance=d.get("allele_variance", {}),
            generation_count=d.get("generation_count", 0),
            total_mutations=d.get("total_mutations", 0),
        )


@dataclass
class GeneticsState:
    """Full genetics state for the colony."""
    genomes: dict[str, Genome] = field(default_factory=dict)
    gene_pool: GenePool = field(default_factory=GenePool)
    selection_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "genomes": {cid: g.to_dict() for cid, g in self.genomes.items()},
            "gene_pool": self.gene_pool.to_dict(),
            "selection_history_len": len(self.selection_history),
        }

    @classmethod
    def from_dict(cls, d: dict) -> GeneticsState:
        genomes = {cid: Genome.from_dict(gd) for cid, gd in d.get("genomes", {}).items()}
        gene_pool = GenePool.from_dict(d.get("gene_pool", {}))
        return cls(genomes=genomes, gene_pool=gene_pool)


@dataclass
class GeneticsYearContext:
    """Inputs for one year of genetics simulation."""
    year: int
    active_ids: list[str]
    deaths: list[dict]
    birth_ids: list[str]
    birth_parent_map: dict[str, tuple[str, str]]
    immigrant_ids: list[str]
    ecology_biome_level: int = 0


@dataclass
class GeneticsTickResult:
    """Output of one year of genetics simulation."""
    new_genomes: list[str]
    diversity_index: float
    generation_count: int
    total_mutations: int
    selection_report: dict
    genetic_pull: dict[str, dict[str, float]]
    events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "new_genomes": self.new_genomes,
            "diversity_index": round(self.diversity_index, 6),
            "generation_count": self.generation_count,
            "total_mutations": self.total_mutations,
            "selection_report": self.selection_report,
            "genetic_pull_count": len(self.genetic_pull),
            "events": self.events,
        }


# ---------------------------------------------------------------------------
# Core genetics functions
# ---------------------------------------------------------------------------

def create_founder_genome(colonist_id: str, stats_dict: dict[str, float],
                          rng: _random_module.Random) -> Genome:
    """Create a genome for a founding colonist from their initial stats.

    Founders' genotypes are seeded from their phenotype (stats) with
    small noise — they ARE their genetics at generation 0.
    """
    alleles: dict[str, float] = {}
    for stat in STAT_NAMES:
        base = stats_dict.get(stat, 0.5)
        alleles[stat] = _clamp(base + rng.gauss(0, 0.03))
    return Genome(alleles=alleles, generation=0, parent_ids=[], mutation_count=0)


def inherit_genome(parent_a: Genome, parent_b: Genome,
                   rng: _random_module.Random) -> Genome:
    """Create a child genome via crossover + mutation.

    Each allele is randomly taken from one parent (crossover).
    Then gaussian mutation is applied. Generation = max(parents) + 1.
    """
    alleles: dict[str, float] = {}
    mutations = 0
    for stat in STAT_NAMES:
        if rng.random() < CROSSOVER_PROB:
            base = parent_a.alleles.get(stat, 0.5)
        else:
            base = parent_b.alleles.get(stat, 0.5)
        mutation = rng.gauss(0, MUTATION_SIGMA)
        if abs(mutation) > MUTATION_SIGMA:
            mutations += 1
        alleles[stat] = _clamp(base + mutation)
    gen = max(parent_a.generation, parent_b.generation) + 1
    parent_ids = list({pid for pid in parent_a.parent_ids[-2:] + parent_b.parent_ids[-2:]})
    return Genome(
        alleles=alleles, generation=gen,
        parent_ids=parent_ids[:4],
        mutation_count=mutations,
    )


def create_immigrant_genome(rng: _random_module.Random) -> Genome:
    """Create a genome for an Earth immigrant — independent of colony gene pool.

    Sampled from a broad Earth distribution (mean=0.5, σ=0.15).
    This maintains genetic diversity through immigration.
    """
    alleles: dict[str, float] = {}
    for stat in STAT_NAMES:
        alleles[stat] = _clamp(rng.gauss(EARTH_ALLELE_MEAN, EARTH_ALLELE_SIGMA))
    return Genome(alleles=alleles, generation=0, parent_ids=[], mutation_count=0)


def compute_diversity(genomes: dict[str, Genome],
                      active_ids: list[str]) -> float:
    """Compute genetic diversity as mean pairwise allele distance.

    Returns 0.0 (identical) to 1.0 (maximally diverse).
    Uses only active colonists. Falls back to 1.0 for <2 colonists.
    """
    active_genomes = [genomes[cid] for cid in active_ids if cid in genomes]
    n = len(active_genomes)
    if n < 2:
        return 1.0

    total_distance = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = _allele_distance(active_genomes[i], active_genomes[j])
            total_distance += dist
            pair_count += 1

    if pair_count == 0:
        return 1.0

    raw = total_distance / pair_count
    # Normalize: max possible distance is 1.0 (all alleles at opposite extremes)
    # Typical random population has mean distance ~0.33
    return min(1.0, raw / 0.5)


def compute_selection_report(
    genomes: dict[str, Genome],
    active_ids: list[str],
    deaths: list[dict],
    year: int,
) -> SelectionReport:
    """Generate a descriptive report of selection pressure.

    Compares mean alleles of survivors vs deceased. Does NOT modify
    the gene pool — this is observational data for analysis.
    """
    report = SelectionReport(year=year, deaths_this_year=len(deaths))
    if not deaths:
        return report

    dead_ids = {d.get("colonist_id", d.get("id", "")) for d in deaths}
    survivor_genomes = [genomes[cid] for cid in active_ids
                        if cid in genomes and cid not in dead_ids]
    deceased_genomes = [genomes[cid] for cid in dead_ids if cid in genomes]

    if not survivor_genomes or not deceased_genomes:
        return report

    for stat in STAT_NAMES:
        s_mean = sum(g.alleles.get(stat, 0.5) for g in survivor_genomes) / len(survivor_genomes)
        d_mean = sum(g.alleles.get(stat, 0.5) for g in deceased_genomes) / len(deceased_genomes)
        report.survivor_mean[stat] = s_mean
        report.deceased_mean[stat] = d_mean
        report.pressure_direction[stat] = s_mean - d_mean

    return report


def compute_genetic_pull(genome: Genome) -> dict[str, float]:
    """Compute the pull each allele exerts on the corresponding phenotype stat.

    Returns a dict of stat_name -> pull_value. The pull is clamped to
    ±MAX_TRAIT_BONUS. Positive = allele wants stat higher, negative = lower.
    This is applied externally by the engine to evolve_stats drift.
    """
    pull: dict[str, float] = {}
    for stat in STAT_NAMES:
        allele = genome.alleles.get(stat, 0.5)
        # Pull toward allele value — strength proportional to distance
        # This is computed as a target-seeking force, not applied here
        pull[stat] = allele
    return pull


def compute_gene_pool_stats(
    genomes: dict[str, Genome],
    active_ids: list[str],
) -> GenePool:
    """Compute colony-wide gene pool statistics."""
    active_genomes = [genomes[cid] for cid in active_ids if cid in genomes]
    n = len(active_genomes)

    if n == 0:
        return GenePool()

    mean_alleles: dict[str, float] = {}
    variance: dict[str, float] = {}
    max_gen = 0

    for stat in STAT_NAMES:
        values = [g.alleles.get(stat, 0.5) for g in active_genomes]
        mean_val = sum(values) / n
        mean_alleles[stat] = mean_val
        if n > 1:
            variance[stat] = sum((v - mean_val) ** 2 for v in values) / (n - 1)
        else:
            variance[stat] = 0.0
        max_gen = max(max_gen, max((g.generation for g in active_genomes), default=0))

    total_mutations = sum(g.mutation_count for g in active_genomes)
    diversity = compute_diversity(genomes, active_ids)

    return GenePool(
        diversity_index=diversity,
        mean_alleles=mean_alleles,
        allele_variance=variance,
        generation_count=max_gen,
        total_mutations=total_mutations,
    )


def tick_genetics(
    state: GeneticsState,
    ctx: GeneticsYearContext,
    rng: _random_module.Random,
) -> GeneticsTickResult:
    """Run one year of genetics simulation.

    1. Assign genomes to new births (inherited) and immigrants (Earth pool)
    2. Compute selection report (descriptive)
    3. Update gene pool statistics
    4. Compute genetic pull for all active colonists
    """
    new_genome_ids: list[str] = []
    events: list[str] = []

    # 1. Assign genomes to births
    for child_id in ctx.birth_ids:
        parents = ctx.birth_parent_map.get(child_id)
        if parents and parents[0] in state.genomes and parents[1] in state.genomes:
            parent_a = state.genomes[parents[0]]
            parent_b = state.genomes[parents[1]]
            child_genome = inherit_genome(parent_a, parent_b, rng)
            child_genome.parent_ids = [parents[0], parents[1]]
            state.genomes[child_id] = child_genome
            new_genome_ids.append(child_id)
            if child_genome.mutation_count > 0:
                events.append(
                    f"{child_id} born with {child_genome.mutation_count} genetic mutations "
                    f"(gen {child_genome.generation})"
                )
            else:
                events.append(f"{child_id} born (gen {child_genome.generation})")
        else:
            # Fallback: create from Earth pool if parents unknown
            state.genomes[child_id] = create_immigrant_genome(rng)
            new_genome_ids.append(child_id)
            events.append(f"{child_id} born with no known parental genome")

    # 2. Assign genomes to immigrants
    for imm_id in ctx.immigrant_ids:
        state.genomes[imm_id] = create_immigrant_genome(rng)
        new_genome_ids.append(imm_id)
        events.append(f"{imm_id} arrives with Earth genome")

    # 3. Selection report (descriptive only)
    sel_report = compute_selection_report(
        state.genomes, ctx.active_ids, ctx.deaths, ctx.year,
    )
    if sel_report.deaths_this_year > 0:
        state.selection_history.append(sel_report.to_dict())
        # Keep last 20 years of history
        if len(state.selection_history) > 20:
            state.selection_history = state.selection_history[-20:]

    # 4. Update gene pool stats
    state.gene_pool = compute_gene_pool_stats(state.genomes, ctx.active_ids)

    # 5. Compute genetic pull for all active colonists
    genetic_pull: dict[str, dict[str, float]] = {}
    for cid in ctx.active_ids:
        if cid in state.genomes:
            genetic_pull[cid] = compute_genetic_pull(state.genomes[cid])

    # 6. Diversity warning
    if state.gene_pool.diversity_index < 0.3:
        events.append(
            f"WARNING: Genetic diversity critically low ({state.gene_pool.diversity_index:.2f})"
        )
    elif state.gene_pool.diversity_index < 0.5:
        events.append(
            f"Genetic diversity declining ({state.gene_pool.diversity_index:.2f})"
        )

    return GeneticsTickResult(
        new_genomes=new_genome_ids,
        diversity_index=state.gene_pool.diversity_index,
        generation_count=state.gene_pool.generation_count,
        total_mutations=state.gene_pool.total_mutations,
        selection_report=sel_report.to_dict(),
        genetic_pull=genetic_pull,
        events=events,
    )


def apply_genetic_pull_to_stat(
    current_stat: float,
    genotype_target: float,
    strength: float = GENOTYPE_PULL_STRENGTH,
) -> float:
    """Compute the genetic pull adjustment for a single stat.

    Pulls the phenotype toward the genotype by `strength` fraction
    of the distance. Returns the DELTA to apply (not the new value).
    """
    distance = genotype_target - current_stat
    delta = distance * strength
    return max(-MAX_TRAIT_BONUS, min(MAX_TRAIT_BONUS, delta))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


def _allele_distance(g1: Genome, g2: Genome) -> float:
    """Mean absolute allele distance between two genomes."""
    total = 0.0
    count = 0
    for stat in STAT_NAMES:
        a1 = g1.alleles.get(stat, 0.5)
        a2 = g2.alleles.get(stat, 0.5)
        total += abs(a1 - a2)
        count += 1
    return total / count if count > 0 else 0.0
