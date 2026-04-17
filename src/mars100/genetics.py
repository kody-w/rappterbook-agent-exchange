"""
Genetics organ for Mars-100 (engine v11.0).

Diploid genomes with per-locus independent assortment.  Genomes encode
trait *aptitude* (bias direction for stat drift) and skill *learning rate*
(ceiling for skill gain), not raw stat/skill values.

Epigenetic marks are per-colonist, non-heritable, affected by yearly
environment.  Mutation happens only at reproduction.

RNG offset: seed + 12553
"""
from __future__ import annotations

import hashlib
import math
import random as _random_module
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Locus registry (stable ordering — never reorder, only append)
# ---------------------------------------------------------------------------

STAT_LOCI = (
    "resolve_apt", "improvisation_apt", "empathy_apt",
    "hoarding_apt", "faith_apt", "paranoia_apt",
)
SKILL_LOCI = (
    "terraforming_apt", "hydroponics_apt", "mediation_apt",
    "coding_apt", "prayer_apt", "sabotage_apt",
)
ALL_LOCI: tuple[str, ...] = STAT_LOCI + SKILL_LOCI

_STAT_LOCUS_TO_STAT = {
    "resolve_apt": "resolve", "improvisation_apt": "improvisation",
    "empathy_apt": "empathy", "hoarding_apt": "hoarding",
    "faith_apt": "faith", "paranoia_apt": "paranoia",
}
_SKILL_LOCUS_TO_SKILL = {
    "terraforming_apt": "terraforming", "hydroponics_apt": "hydroponics",
    "mediation_apt": "mediation", "coding_apt": "coding",
    "prayer_apt": "prayer", "sabotage_apt": "sabotage",
}

BASE_MUTATION_RATE = 0.02
MAX_MUTATION_RATE = 0.12
MUTATION_SIGMA = 0.06
EPIGENETIC_DRIFT = 0.015
TRAIT_BIAS_STRENGTH = 0.004
SKILL_APT_MULTIPLIER = 1.5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Genome:
    """Diploid genome: each locus has two float alleles in [0, 1]."""
    alleles: dict[str, tuple[float, float]] = field(default_factory=dict)

    def expressed(self, locus: str) -> float:
        """Phenotypic expression = mean of two alleles (additive dominance)."""
        pair = self.alleles.get(locus, (0.5, 0.5))
        return (pair[0] + pair[1]) / 2.0

    def to_dict(self) -> dict[str, list[float]]:
        return {k: [round(a, 4), round(b, 4)]
                for k, (a, b) in self.alleles.items()}

    @classmethod
    def from_dict(cls, d: dict[str, list[float]]) -> Genome:
        alleles = {k: (v[0], v[1]) for k, v in d.items()
                   if isinstance(v, list) and len(v) == 2}
        return cls(alleles=alleles)


@dataclass
class EpigeneticMarks:
    """Per-colonist expression modifiers.  Not heritable.

    Each mark is a float in [-0.2, 0.2] — additive shift to expressed value.
    """
    marks: dict[str, float] = field(default_factory=dict)

    def get(self, locus: str) -> float:
        return self.marks.get(locus, 0.0)

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in self.marks.items()}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> EpigeneticMarks:
        return cls(marks=dict(d))


@dataclass
class GeneticsState:
    """Population-level genetics tracking."""
    mean_heterozygosity: float = 0.5
    diversity_history: list[float] = field(default_factory=list)
    generation_count: int = 0
    total_mutations: int = 0
    mutation_rate_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_heterozygosity": round(self.mean_heterozygosity, 4),
            "diversity_history": [round(d, 4) for d in self.diversity_history],
            "generation_count": self.generation_count,
            "total_mutations": self.total_mutations,
            "mutation_rate_history": [round(r, 4)
                                      for r in self.mutation_rate_history],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GeneticsState:
        return cls(
            mean_heterozygosity=d.get("mean_heterozygosity", 0.5),
            diversity_history=d.get("diversity_history", []),
            generation_count=d.get("generation_count", 0),
            total_mutations=d.get("total_mutations", 0),
            mutation_rate_history=d.get("mutation_rate_history", []),
        )


@dataclass
class GeneticsTickResult:
    """Aggregate genetics metrics for one year."""
    heterozygosity: float = 0.5
    mutation_rate: float = BASE_MUTATION_RATE
    mean_trait_bias: dict[str, float] = field(default_factory=dict)
    epigenetic_shifts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "heterozygosity": round(self.heterozygosity, 4),
            "mutation_rate": round(self.mutation_rate, 4),
            "mean_trait_bias": {k: round(v, 4)
                                for k, v in self.mean_trait_bias.items()},
            "epigenetic_shifts": self.epigenetic_shifts,
        }


# ---------------------------------------------------------------------------
# Bootstrap — create genome from existing stats/skills
# ---------------------------------------------------------------------------

def _stable_subseed(base_seed: int, colonist_id: str) -> int:
    """Deterministic sub-seed from base + colonist ID (no main RNG consumed)."""
    h = hashlib.md5(f"{base_seed}:{colonist_id}".encode()).hexdigest()
    return int(h[:8], 16)


def bootstrap_genome(stats_dict: dict[str, float],
                     skills_dict: dict[str, float],
                     base_seed: int,
                     colonist_id: str) -> Genome:
    """Create a genome that approximately expresses the given phenotype.

    Alleles are set so mean ≈ trait value, with small noise for heterozygosity.
    Uses a dedicated sub-seed — never consumes the main simulation RNG.
    """
    subseed = _stable_subseed(base_seed, colonist_id)
    rng = _random_module.Random(subseed)
    alleles: dict[str, tuple[float, float]] = {}
    for locus in STAT_LOCI:
        stat_name = _STAT_LOCUS_TO_STAT[locus]
        val = stats_dict.get(stat_name, 0.5)
        noise = rng.gauss(0, 0.04)
        a = max(0.0, min(1.0, val + noise))
        b = max(0.0, min(1.0, val - noise))
        alleles[locus] = (a, b)
    for locus in SKILL_LOCI:
        skill_name = _SKILL_LOCUS_TO_SKILL[locus]
        val = skills_dict.get(skill_name, 0.1)
        # Aptitude centered slightly above current skill (room to grow)
        apt = min(1.0, val + 0.15)
        noise = rng.gauss(0, 0.04)
        a = max(0.0, min(1.0, apt + noise))
        b = max(0.0, min(1.0, apt - noise))
        alleles[locus] = (a, b)
    return Genome(alleles=alleles)


# ---------------------------------------------------------------------------
# Reproduction — per-locus independent assortment + mutation
# ---------------------------------------------------------------------------

def compute_mutation_rate(atmosphere_pressure_kpa: float,
                          radiation_event: bool) -> float:
    """Mutation rate rises with UV exposure (low atmosphere) and radiation.

    Thin Mars atmosphere ≈ high UV → higher mutations.
    As terraforming increases pressure, rate drops toward Earth baseline.
    """
    pressure_factor = max(0.0, 1.0 - atmosphere_pressure_kpa / 20.0)
    rate = BASE_MUTATION_RATE + pressure_factor * 0.04
    if radiation_event:
        rate += 0.03
    return min(MAX_MUTATION_RATE, rate)


def crossover(parent_a: Genome, parent_b: Genome,
              mutation_rate: float,
              rng: _random_module.Random) -> tuple[Genome, int]:
    """Per-locus independent assortment with point mutation.

    For each locus: child inherits one random allele from each parent.
    Then each allele may mutate with probability ``mutation_rate``.

    Returns (child_genome, mutation_count).
    """
    child_alleles: dict[str, tuple[float, float]] = {}
    mutations = 0
    for locus in ALL_LOCI:
        pa = parent_a.alleles.get(locus, (0.5, 0.5))
        pb = parent_b.alleles.get(locus, (0.5, 0.5))
        # Independent assortment: one allele from each parent
        a = rng.choice(pa)
        b = rng.choice(pb)
        # Point mutation
        if rng.random() < mutation_rate:
            a = max(0.0, min(1.0, a + rng.gauss(0, MUTATION_SIGMA)))
            mutations += 1
        if rng.random() < mutation_rate:
            b = max(0.0, min(1.0, b + rng.gauss(0, MUTATION_SIGMA)))
            mutations += 1
        child_alleles[locus] = (a, b)
    return Genome(alleles=child_alleles), mutations


# ---------------------------------------------------------------------------
# Phenotype → trait bias & skill aptitude
# ---------------------------------------------------------------------------

def compute_trait_biases(genome: Genome,
                         marks: EpigeneticMarks) -> dict[str, float]:
    """Genome → per-stat bias target for stat drift direction.

    Returns dict mapping stat name → target value in [0, 1].
    The engine uses this to nudge stat evolution toward the target.
    """
    biases: dict[str, float] = {}
    for locus, stat_name in _STAT_LOCUS_TO_STAT.items():
        raw = genome.expressed(locus) + marks.get(locus)
        biases[stat_name] = max(0.0, min(1.0, raw))
    return biases


def compute_skill_aptitudes(genome: Genome,
                            marks: EpigeneticMarks) -> dict[str, float]:
    """Genome → per-skill learning rate multiplier.

    Returns dict mapping skill name → multiplier in [0.5, 2.0].
    Higher aptitude = faster skill gain.
    """
    aptitudes: dict[str, float] = {}
    for locus, skill_name in _SKILL_LOCUS_TO_SKILL.items():
        raw = genome.expressed(locus) + marks.get(locus)
        raw = max(0.0, min(1.0, raw))
        # Map [0, 1] → [0.5, 2.0]
        aptitudes[skill_name] = 0.5 + raw * SKILL_APT_MULTIPLIER
    return aptitudes


# ---------------------------------------------------------------------------
# Epigenetics — yearly environment modifiers (non-heritable)
# ---------------------------------------------------------------------------

def update_epigenetics(marks: EpigeneticMarks,
                       stress: float,
                       resource_avg: float,
                       radiation_event: bool,
                       rng: _random_module.Random) -> int:
    """Drift epigenetic marks based on environment.

    High stress suppresses empathy/faith aptitude expression.
    Low resources boost hoarding/resolve aptitude expression.
    Radiation adds noise to all loci.

    Returns count of loci shifted beyond threshold.
    """
    shifts = 0
    for locus in ALL_LOCI:
        drift = rng.gauss(0, EPIGENETIC_DRIFT)
        # Environmental pressures
        if stress > 0.6 and locus in ("empathy_apt", "faith_apt"):
            drift -= 0.005 * stress
        if resource_avg < 0.35 and locus in ("hoarding_apt", "resolve_apt"):
            drift += 0.005 * (1.0 - resource_avg)
        if radiation_event:
            drift += rng.gauss(0, 0.008)
        old = marks.marks.get(locus, 0.0)
        new = max(-0.2, min(0.2, old + drift))
        if abs(new - old) > 0.005:
            shifts += 1
        marks.marks[locus] = new
    return shifts


# ---------------------------------------------------------------------------
# Population genetics — heterozygosity
# ---------------------------------------------------------------------------

def compute_heterozygosity(genomes: list[Genome]) -> float:
    """Expected heterozygosity (mean pairwise distance) across all loci.

    For each locus: collect all alleles, compute variance.
    Mean variance across loci = heterozygosity proxy.
    Small populations with similar alleles → low heterozygosity.
    """
    if not genomes:
        return 0.0
    locus_variances: list[float] = []
    for locus in ALL_LOCI:
        all_alleles: list[float] = []
        for g in genomes:
            pair = g.alleles.get(locus, (0.5, 0.5))
            all_alleles.extend(pair)
        if len(all_alleles) < 2:
            locus_variances.append(0.0)
            continue
        mean = sum(all_alleles) / len(all_alleles)
        var = sum((a - mean) ** 2 for a in all_alleles) / len(all_alleles)
        locus_variances.append(var)
    return sum(locus_variances) / max(1, len(locus_variances))


# ---------------------------------------------------------------------------
# Yearly tick — aggregate metrics + epigenetic update
# ---------------------------------------------------------------------------

def tick_genetics(
    state: GeneticsState,
    genomes: list[Genome],
    marks_map: dict[str, EpigeneticMarks],
    colonist_stress: dict[str, float],
    resource_avg: float,
    radiation_event: bool,
    rng: _random_module.Random,
) -> GeneticsTickResult:
    """Per-year genetics tick.

    Updates epigenetic marks for each colonist and computes population
    heterozygosity.  Does NOT mutate genomes (mutation is reproduction-only).
    """
    # Update epigenetics
    total_shifts = 0
    for cid, marks in marks_map.items():
        stress = colonist_stress.get(cid, 0.0)
        shifts = update_epigenetics(marks, stress, resource_avg,
                                    radiation_event, rng)
        total_shifts += shifts

    # Population heterozygosity
    het = compute_heterozygosity(genomes)
    state.mean_heterozygosity = het
    state.diversity_history.append(het)
    # Keep last 100 entries
    if len(state.diversity_history) > 100:
        state.diversity_history = state.diversity_history[-100:]

    # Compute current mutation rate (for reporting — applied at reproduction)
    # Use a nominal pressure (stored externally, passed as resource proxy)
    mut_rate = compute_mutation_rate(resource_avg * 20.0, radiation_event)
    state.mutation_rate_history.append(mut_rate)
    if len(state.mutation_rate_history) > 100:
        state.mutation_rate_history = state.mutation_rate_history[-100:]

    # Mean trait bias across population
    mean_biases: dict[str, float] = {s: 0.0 for s in
                                      _STAT_LOCUS_TO_STAT.values()}
    if genomes:
        for g in genomes:
            for locus, stat_name in _STAT_LOCUS_TO_STAT.items():
                mean_biases[stat_name] += g.expressed(locus)
        for k in mean_biases:
            mean_biases[k] /= len(genomes)

    return GeneticsTickResult(
        heterozygosity=het,
        mutation_rate=mut_rate,
        mean_trait_bias=mean_biases,
        epigenetic_shifts=total_shifts,
    )
