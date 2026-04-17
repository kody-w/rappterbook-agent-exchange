"""
Genetics organ for Mars-100 colony simulation (engine v10.0).

Models diploid genomes with 12 alleles (2 per personality stat),
crossover + mutation for reproduction, pedigree-based inbreeding
coefficients, and colony-level genetic diversity tracking.

Key dynamics:
  - Each colonist carries a Genome: 12 floats in [0, 1]
  - Stats are influenced by genome baseline (mean of 2 alleles)
  - Children are produced via single-point crossover + Gaussian mutation
  - Pedigree (parent_ids) is stored to compute Wright's F
  - High inbreeding coefficient increases death rate
  - Colony genetic diversity (expected heterozygosity) is tracked per year
  - Immigrants carry fresh random genomes, boosting diversity

Design decisions:
  - Genome does NOT override lived stats — it provides a baseline that
    stat evolution drifts around.  The expressed phenotype is always the
    colonist's actual stat values (which evolve via events, culture, etc.)
  - Skills remain learned-only (no genetic basis)
  - Founding 10 get genomes back-fitted from their hand-authored stats
  - Separate RNG stream (seed + 11213) to avoid disturbing v9 paths
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

# -- constants ---------------------------------------------------------------

ALLELES_PER_STAT = 2
STAT_NAMES = ("resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia")
NUM_ALLELES = len(STAT_NAMES) * ALLELES_PER_STAT  # 12

MUTATION_RATE = 0.02       # probability of mutation per allele per generation
MUTATION_SIGMA = 0.08      # Gaussian stddev when mutation occurs

INBREEDING_DEATH_MULTIPLIER = 1.5   # max death-rate multiplier at F=1.0
INBREEDING_LEARNING_PENALTY = 0.15  # skill learning reduction at F=1.0

DIVERSITY_WARNING_THRESHOLD = 0.20  # colony diversity below this is concerning


# -- data classes ------------------------------------------------------------

@dataclass
class Genome:
    """Diploid genome: 2 alleles per personality stat (12 total).

    Alleles are indexed by stat:
      resolve: alleles[0], alleles[1]
      improvisation: alleles[2], alleles[3]
      ...etc.
    """
    alleles: list[float] = field(default_factory=lambda: [0.5] * NUM_ALLELES)
    parent_ids: list[str] = field(default_factory=list)

    def stat_baseline(self, stat_index: int) -> float:
        """Mean of the two alleles for a given stat (diploid expression)."""
        i = stat_index * ALLELES_PER_STAT
        return (self.alleles[i] + self.alleles[i + 1]) / 2.0

    def stat_baselines(self) -> dict[str, float]:
        """Compute all stat baselines from alleles."""
        return {
            name: self.stat_baseline(i)
            for i, name in enumerate(STAT_NAMES)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "alleles": [round(a, 6) for a in self.alleles],
            "parent_ids": list(self.parent_ids),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Genome:
        if not d:
            return cls()
        return cls(
            alleles=list(d.get("alleles", [0.5] * NUM_ALLELES)),
            parent_ids=list(d.get("parent_ids", [])),
        )


# -- genome creation ---------------------------------------------------------

def create_genome_from_stats(stats: dict[str, float],
                             rng: random.Random) -> Genome:
    """Back-fit a genome from existing stat values.

    Each stat value becomes the mean of two alleles.  We add small noise
    to create allelic variation while preserving the expressed phenotype.
    """
    alleles: list[float] = []
    for name in STAT_NAMES:
        base = stats.get(name, 0.5)
        # Split the stat into two alleles that average to ~base
        spread = rng.uniform(0.0, min(base, 1.0 - base, 0.15))
        a1 = max(0.0, min(1.0, base + spread))
        a2 = max(0.0, min(1.0, base - spread))
        alleles.extend([a1, a2])
    return Genome(alleles=alleles, parent_ids=[])


def create_random_genome(rng: random.Random) -> Genome:
    """Create a fresh random genome (for immigrants)."""
    alleles = [max(0.0, min(1.0, rng.gauss(0.5, 0.15)))
               for _ in range(NUM_ALLELES)]
    return Genome(alleles=alleles, parent_ids=[])


# -- reproduction ------------------------------------------------------------

def crossover(parent_a: Genome, parent_b: Genome,
              child_id: str, rng: random.Random) -> Genome:
    """Single-point crossover with per-allele mutation.

    Each parent donates one allele per stat (randomly chosen).
    Mutation: each allele has MUTATION_RATE chance of Gaussian perturbation.
    """
    child_alleles: list[float] = []

    for stat_idx in range(len(STAT_NAMES)):
        base = stat_idx * ALLELES_PER_STAT

        # Parent A donates one allele (randomly pick which)
        a_pick = rng.randint(0, ALLELES_PER_STAT - 1)
        allele_a = parent_a.alleles[base + a_pick]

        # Parent B donates one allele (randomly pick which)
        b_pick = rng.randint(0, ALLELES_PER_STAT - 1)
        allele_b = parent_b.alleles[base + b_pick]

        # Apply mutation
        if rng.random() < MUTATION_RATE:
            allele_a = max(0.0, min(1.0, allele_a + rng.gauss(0, MUTATION_SIGMA)))
        if rng.random() < MUTATION_RATE:
            allele_b = max(0.0, min(1.0, allele_b + rng.gauss(0, MUTATION_SIGMA)))

        child_alleles.extend([allele_a, allele_b])

    # Parent IDs are set by the caller (engine knows colonist IDs, genome doesn't)
    return Genome(alleles=child_alleles, parent_ids=[])


# -- inbreeding --------------------------------------------------------------

def compute_inbreeding_coefficient(
    child_parent_ids: list[str],
    pedigree: dict[str, list[str]],
) -> float:
    """Compute Wright's inbreeding coefficient F from full pedigree.

    F = probability that two alleles at a locus are identical by descent.
    Uses ancestor counting: F = sum over common ancestors of (1/2)^(La+Lb+1)
    where La and Lb are the path lengths from parent A and B to the common
    ancestor.

    Returns F in [0.0, 1.0].
    """
    if len(child_parent_ids) < 2:
        return 0.0

    parent_a_id, parent_b_id = child_parent_ids[0], child_parent_ids[1]

    # Build ancestor sets with path lengths
    ancestors_a = _get_ancestors(parent_a_id, pedigree, max_depth=8)
    ancestors_b = _get_ancestors(parent_b_id, pedigree, max_depth=8)

    common = set(ancestors_a.keys()) & set(ancestors_b.keys())
    if not common:
        return 0.0

    f = 0.0
    for ancestor_id in common:
        la = ancestors_a[ancestor_id]  # path length from parent A
        lb = ancestors_b[ancestor_id]  # path length from parent B
        # Wright's formula: (1/2)^(La+Lb+1) * (1 + Fa)
        # We approximate Fa=0 for simplicity (no recursive F computation)
        f += math.pow(0.5, la + lb + 1)

    return min(1.0, f)


def _get_ancestors(
    individual_id: str,
    pedigree: dict[str, list[str]],
    max_depth: int = 8,
) -> dict[str, int]:
    """Get all ancestors of an individual with path lengths.

    Returns {ancestor_id: min_path_length}.
    """
    ancestors: dict[str, int] = {}
    frontier: list[tuple[str, int]] = [(individual_id, 0)]
    visited: set[str] = set()

    while frontier:
        current_id, depth = frontier.pop(0)
        if depth > max_depth:
            continue
        if current_id in visited:
            continue
        visited.add(current_id)

        parents = pedigree.get(current_id, [])
        for pid in parents:
            new_depth = depth + 1
            if pid not in ancestors or ancestors[pid] > new_depth:
                ancestors[pid] = new_depth
            frontier.append((pid, new_depth))

    return ancestors


def inbreeding_death_modifier(f_coefficient: float) -> float:
    """Death rate multiplier from inbreeding coefficient.

    Returns 1.0 (no effect) at F=0, up to INBREEDING_DEATH_MULTIPLIER at F=1.
    Linear interpolation.
    """
    return 1.0 + f_coefficient * (INBREEDING_DEATH_MULTIPLIER - 1.0)


def inbreeding_learning_modifier(f_coefficient: float) -> float:
    """Skill learning rate modifier from inbreeding.

    Returns 1.0 (normal) at F=0, reduced at high F.
    """
    return max(0.5, 1.0 - f_coefficient * INBREEDING_LEARNING_PENALTY)


# -- genetic distance & diversity --------------------------------------------

def genetic_distance(genome_a: Genome, genome_b: Genome) -> float:
    """Euclidean distance between two genomes, normalized to [0, 1].

    Max possible distance = sqrt(NUM_ALLELES) when all alleles differ by 1.0.
    """
    sq_sum = sum(
        (a - b) ** 2
        for a, b in zip(genome_a.alleles, genome_b.alleles)
    )
    max_dist = math.sqrt(NUM_ALLELES)
    return math.sqrt(sq_sum) / max_dist


def colony_diversity(genomes: list[Genome]) -> float:
    """Expected heterozygosity: mean per-locus allele variance.

    High diversity → values near 0.5 (max variance).
    Low diversity → values near 0.0 (all alleles identical).
    Returns value in [0.0, 1.0].
    """
    if len(genomes) < 2:
        return 0.0

    n = len(genomes)
    total_het = 0.0

    for locus in range(NUM_ALLELES):
        allele_values = [g.alleles[locus] for g in genomes]
        mean = sum(allele_values) / n
        variance = sum((a - mean) ** 2 for a in allele_values) / n
        # Normalize variance: max possible variance for [0,1] uniform is 0.25
        total_het += min(1.0, variance / 0.25)

    return total_het / NUM_ALLELES
