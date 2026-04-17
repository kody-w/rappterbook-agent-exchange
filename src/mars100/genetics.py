"""
Genetics organ for Mars-100 (engine v11.0).

Trait heredity for colonist reproduction. Each colonist carries a Genome —
a set of allele pairs for each stat. Children inherit alleles via Mendelian
crossover with mutation. Colony-wide genetic diversity is tracked using
expected heterozygosity.

No I/O. Pure computation. Deterministic given RNG state.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


STAT_NAMES = ("resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia")

MUTATION_RATE = 0.05
MUTATION_SIGMA = 0.08
INBREEDING_THRESHOLD = 0.25
INBREEDING_PENALTY = 0.15
MIN_ALLELE = 0.0
MAX_ALLELE = 1.0
DOMINANT_WEIGHT = 0.7
RECESSIVE_WEIGHT = 0.3


@dataclass
class AllelePair:
    """Two alleles for one trait locus."""
    a: float
    b: float

    def dominant(self) -> float:
        return max(self.a, self.b)

    def recessive(self) -> float:
        return min(self.a, self.b)

    def express(self) -> float:
        """Phenotypic expression: weighted blend of dominant and recessive."""
        return self.dominant() * DOMINANT_WEIGHT + self.recessive() * RECESSIVE_WEIGHT

    def heterozygosity(self) -> float:
        """Per-locus heterozygosity: distance between alleles."""
        return abs(self.a - self.b)

    def to_dict(self) -> dict[str, float]:
        return {"a": round(self.a, 6), "b": round(self.b, 6)}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> AllelePair:
        return cls(a=d.get("a", 0.5), b=d.get("b", 0.5))


@dataclass
class EpigeneticMark:
    """A mark on a specific locus caused by an extreme event."""
    locus: str
    modifier: float
    origin_year: int
    cause: str

    def to_dict(self) -> dict:
        return {"locus": self.locus, "modifier": round(self.modifier, 6),
                "origin_year": self.origin_year, "cause": self.cause}

    @classmethod
    def from_dict(cls, d: dict) -> EpigeneticMark:
        return cls(locus=d["locus"], modifier=d["modifier"],
                   origin_year=d["origin_year"], cause=d["cause"])


@dataclass
class Genome:
    """Complete genome for a colonist: allele pairs for each stat locus."""
    loci: dict[str, AllelePair] = field(default_factory=dict)
    epigenetic_marks: list[EpigeneticMark] = field(default_factory=list)
    generation: int = 0

    def express_all(self) -> dict[str, float]:
        """Express all loci into phenotypic stat modifiers."""
        result: dict[str, float] = {}
        for name, pair in self.loci.items():
            base = pair.express()
            epi_mod = sum(m.modifier for m in self.epigenetic_marks
                          if m.locus == name)
            result[name] = max(MIN_ALLELE, min(MAX_ALLELE, base + epi_mod))
        return result

    def avg_heterozygosity(self) -> float:
        """Average heterozygosity across all loci."""
        if not self.loci:
            return 0.0
        return sum(p.heterozygosity() for p in self.loci.values()) / len(self.loci)

    def to_dict(self) -> dict:
        return {
            "loci": {k: v.to_dict() for k, v in self.loci.items()},
            "epigenetic_marks": [m.to_dict() for m in self.epigenetic_marks],
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Genome:
        loci = {k: AllelePair.from_dict(v)
                for k, v in d.get("loci", {}).items()}
        marks = [EpigeneticMark.from_dict(m)
                 for m in d.get("epigenetic_marks", [])]
        return cls(loci=loci, epigenetic_marks=marks,
                   generation=d.get("generation", 0))


def create_genome_from_phenotype(stats: dict[str, float],
                                 rng: random.Random) -> Genome:
    """Bootstrap a genome from existing phenotypic stats.

    Generates two DISTINCT alleles whose weighted expression matches
    the target stat value, preserving heterozygosity.
    """
    loci: dict[str, AllelePair] = {}
    for name in STAT_NAMES:
        target = stats.get(name, 0.5)
        spread = rng.uniform(0.05, 0.3)
        # Solve: dom * 0.7 + rec * 0.3 = target
        # with dom = target + offset, rec = target - offset * (0.7/0.3)
        # Simplified: generate two alleles around target
        a = max(MIN_ALLELE, min(MAX_ALLELE, target + spread * rng.uniform(-1, 1)))
        b = max(MIN_ALLELE, min(MAX_ALLELE, target + spread * rng.uniform(-1, 1)))
        # Adjust so expression ≈ target
        expressed = max(a, b) * DOMINANT_WEIGHT + min(a, b) * RECESSIVE_WEIGHT
        correction = target - expressed
        a = max(MIN_ALLELE, min(MAX_ALLELE, a + correction * 0.5))
        b = max(MIN_ALLELE, min(MAX_ALLELE, b + correction * 0.5))
        loci[name] = AllelePair(a=a, b=b)
    return Genome(loci=loci, generation=0)


def create_random_genome(rng: random.Random) -> Genome:
    """Create a genome with random alleles (for immigrants)."""
    loci: dict[str, AllelePair] = {}
    for name in STAT_NAMES:
        a = rng.uniform(0.1, 0.9)
        b = rng.uniform(0.1, 0.9)
        loci[name] = AllelePair(a=a, b=b)
    return Genome(loci=loci, generation=0)


def inherit_genome(parent_a: Genome, parent_b: Genome,
                   rng: random.Random) -> Genome:
    """Mendelian inheritance: one allele from each parent per locus,
    with crossover and mutation."""
    child_loci: dict[str, AllelePair] = {}
    child_marks: list[EpigeneticMark] = []
    parent_gen = max(parent_a.generation, parent_b.generation)

    for name in STAT_NAMES:
        pair_a = parent_a.loci.get(name, AllelePair(0.5, 0.5))
        pair_b = parent_b.loci.get(name, AllelePair(0.5, 0.5))

        # Each parent donates one allele randomly
        from_a = rng.choice([pair_a.a, pair_a.b])
        from_b = rng.choice([pair_b.a, pair_b.b])

        # Mutation
        if rng.random() < MUTATION_RATE:
            from_a += rng.gauss(0, MUTATION_SIGMA)
            from_a = max(MIN_ALLELE, min(MAX_ALLELE, from_a))
        if rng.random() < MUTATION_RATE:
            from_b += rng.gauss(0, MUTATION_SIGMA)
            from_b = max(MIN_ALLELE, min(MAX_ALLELE, from_b))

        child_loci[name] = AllelePair(a=from_a, b=from_b)

    # Inherit epigenetic marks (50% chance each, decaying over generations)
    all_marks = parent_a.epigenetic_marks + parent_b.epigenetic_marks
    for mark in all_marks:
        if rng.random() < 0.5:
            faded = EpigeneticMark(
                locus=mark.locus,
                modifier=mark.modifier * 0.6,
                origin_year=mark.origin_year,
                cause=mark.cause,
            )
            if abs(faded.modifier) > 0.005:
                child_marks.append(faded)

    return Genome(loci=child_loci, epigenetic_marks=child_marks,
                  generation=parent_gen + 1)


def compute_inbreeding_coefficient(parent_a_ids: list[str],
                                   parent_b_ids: list[str],
                                   pedigree: dict[str, list[str]]) -> float:
    """Estimate inbreeding coefficient from pedigree overlap.

    Uses ancestor overlap: fraction of shared ancestors in the last
    3 generations. Returns 0.0 (unrelated) to 1.0 (identical ancestry).
    """
    def ancestors(cid: str, depth: int) -> set[str]:
        if depth <= 0 or cid not in pedigree:
            return set()
        parents = pedigree.get(cid, [])
        result = set(parents)
        for p in parents:
            result |= ancestors(p, depth - 1)
        return result

    # Get ancestor IDs for the parents of the child
    anc_a: set[str] = set()
    for pid in parent_a_ids:
        anc_a.add(pid)
        anc_a |= ancestors(pid, 3)
    anc_b: set[str] = set()
    for pid in parent_b_ids:
        anc_b.add(pid)
        anc_b |= ancestors(pid, 3)

    if not anc_a or not anc_b:
        return 0.0

    overlap = anc_a & anc_b
    union = anc_a | anc_b
    if not union:
        return 0.0
    return len(overlap) / len(union)


def compute_colony_diversity(genomes: list[Genome]) -> float:
    """Expected heterozygosity across the colony.

    Average of per-locus allele variance across all individuals.
    Range: 0.0 (all identical) to ~0.5 (maximally diverse).
    """
    if len(genomes) < 2:
        return 0.0

    diversities: list[float] = []
    for name in STAT_NAMES:
        all_alleles: list[float] = []
        for g in genomes:
            pair = g.loci.get(name)
            if pair:
                all_alleles.extend([pair.a, pair.b])
        if len(all_alleles) < 2:
            continue
        mean = sum(all_alleles) / len(all_alleles)
        variance = sum((x - mean) ** 2 for x in all_alleles) / len(all_alleles)
        diversities.append(variance)

    return sum(diversities) / len(diversities) if diversities else 0.0


def apply_epigenetic_mark(genome: Genome, event_type: str,
                          severity: float, year: int,
                          rng: random.Random) -> bool:
    """Apply an epigenetic mark from an extreme event.

    Only triggers for severity > 0.6. Marks a random locus.
    Returns True if a mark was applied.
    """
    if severity <= 0.6:
        return False
    if rng.random() > severity * 0.5:
        return False

    locus = rng.choice(list(STAT_NAMES))
    # Negative events increase paranoia/hoarding genes,
    # decrease empathy/faith genes
    if locus in ("paranoia", "hoarding"):
        modifier = rng.uniform(0.01, 0.04)
    elif locus in ("empathy", "faith"):
        modifier = rng.uniform(-0.04, -0.01)
    else:
        modifier = rng.uniform(-0.02, 0.02)

    mark = EpigeneticMark(locus=locus, modifier=modifier,
                          origin_year=year, cause=event_type)
    genome.epigenetic_marks.append(mark)
    # Prune old/faded marks
    genome.epigenetic_marks = [
        m for m in genome.epigenetic_marks if abs(m.modifier) > 0.005
    ]
    return True


def compute_genome_baseline(genome: Genome) -> dict[str, float]:
    """Compute the genetic baseline for stats.

    This is the set point that yearly environmental drift is bounded
    relative to. Stats can drift ±0.15 from baseline before snapping back.
    """
    return genome.express_all()


DRIFT_BOUND = 0.15


def apply_genetic_drift_bound(current_stat: float, baseline: float) -> float:
    """Bound stat drift relative to genetic baseline.

    Stats can deviate up to DRIFT_BOUND from their genetic baseline.
    Beyond that, they're pulled back toward baseline.
    """
    deviation = current_stat - baseline
    if abs(deviation) <= DRIFT_BOUND:
        return current_stat
    # Pull back toward baseline
    overshoot = abs(deviation) - DRIFT_BOUND
    direction = 1.0 if deviation > 0 else -1.0
    pullback = overshoot * 0.3
    return current_stat - direction * pullback


@dataclass
class GeneticsTickResult:
    """Result of one year of genetics processing."""
    colony_diversity: float = 0.0
    avg_heterozygosity: float = 0.0
    max_generation: int = 0
    epigenetic_marks_applied: int = 0
    inbreeding_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "colony_diversity": round(self.colony_diversity, 6),
            "avg_heterozygosity": round(self.avg_heterozygosity, 6),
            "max_generation": self.max_generation,
            "epigenetic_marks_applied": self.epigenetic_marks_applied,
            "inbreeding_warnings": self.inbreeding_warnings,
        }


@dataclass
class GeneticsState:
    """Colony-wide genetics tracking."""
    diversity_history: list[float] = field(default_factory=list)
    generation_max: int = 0
    total_marks_applied: int = 0
    pedigree: dict[str, list[str]] = field(default_factory=dict)

    def record_birth(self, child_id: str, parent_ids: list[str]) -> None:
        """Record parent-child relationship in pedigree."""
        self.pedigree[child_id] = list(parent_ids)

    def to_dict(self) -> dict:
        return {
            "diversity_history": [round(d, 6) for d in self.diversity_history[-20:]],
            "generation_max": self.generation_max,
            "total_marks_applied": self.total_marks_applied,
            "pedigree_size": len(self.pedigree),
        }

    @classmethod
    def from_dict(cls, d: dict) -> GeneticsState:
        return cls(
            diversity_history=d.get("diversity_history", []),
            generation_max=d.get("generation_max", 0),
            total_marks_applied=d.get("total_marks_applied", 0),
        )


def tick_genetics(
    state: GeneticsState,
    genomes: list[Genome],
    event_type: str,
    event_severity: float,
    year: int,
    rng: random.Random,
) -> GeneticsTickResult:
    """Per-year genetics update.

    Applies epigenetic marks from events, computes diversity,
    tracks generational metrics.
    """
    result = GeneticsTickResult()

    # Apply epigenetic marks from severe events
    marks_applied = 0
    for genome in genomes:
        if apply_epigenetic_mark(genome, event_type, event_severity, year, rng):
            marks_applied += 1
    result.epigenetic_marks_applied = marks_applied
    state.total_marks_applied += marks_applied

    # Compute colony diversity
    result.colony_diversity = compute_colony_diversity(genomes)
    state.diversity_history.append(result.colony_diversity)
    if len(state.diversity_history) > 100:
        state.diversity_history = state.diversity_history[-100:]

    # Average heterozygosity
    if genomes:
        result.avg_heterozygosity = (
            sum(g.avg_heterozygosity() for g in genomes) / len(genomes))
    result.max_generation = max((g.generation for g in genomes), default=0)
    state.generation_max = max(state.generation_max, result.max_generation)

    return result
