"""
Genetics organ for Mars-100 (engine v11.0).

Models heredity: each colonist carries a diploid genome (12 genes × 2 alleles).
Children inherit alleles through crossover + mutation.  Colony-wide diversity
index tracks allelic variation; low diversity triggers health penalties.

One-year lag: LAST year's diversity drives THIS year's health modifier.

RNG offset: seed + 13337
"""
from __future__ import annotations

import math
import random as _random_module
from dataclasses import dataclass, field
from typing import Any

GENE_NAMES: tuple[str, ...] = (
    "resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia",
    "terraforming", "hydroponics", "mediation", "coding", "prayer", "sabotage",
)
NUM_GENES = len(GENE_NAMES)

MUTATION_RATE = 0.02
MUTATION_SIGMA = 0.05
CROSSOVER_PROBABILITY = 0.7
HYBRID_VIGOR_THRESHOLD = 0.25
HYBRID_VIGOR_BONUS = 0.08
INBREEDING_THRESHOLD = 0.15
INBREEDING_PENALTY_MAX = 0.3
EPIGENETIC_PROBABILITY = 0.03
EPIGENETIC_SHIFT = 0.04
DIVERSITY_HEALTHY = 0.3


@dataclass
class Genome:
    """Diploid genome: 12 genes, each with two alleles in [0, 1]."""
    alleles: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.alleles:
            self.alleles = [[0.5, 0.5] for _ in range(NUM_GENES)]

    def express(self, gene_index: int) -> float:
        """Compute expressed phenotype: 0.7 * max + 0.3 * min."""
        a, b = self.alleles[gene_index]
        return 0.7 * max(a, b) + 0.3 * min(a, b)

    def express_all(self) -> dict[str, float]:
        """Return expressed values for all genes."""
        return {GENE_NAMES[i]: self.express(i) for i in range(NUM_GENES)}

    def heterozygosity(self) -> float:
        """Mean absolute difference between allele pairs."""
        if not self.alleles:
            return 0.0
        diffs = [abs(a - b) for a, b in self.alleles]
        return sum(diffs) / len(diffs)

    def genetic_distance(self, other: Genome) -> float:
        """Euclidean distance between expressed phenotypes."""
        total = 0.0
        for i in range(NUM_GENES):
            diff = self.express(i) - other.express(i)
            total += diff * diff
        return math.sqrt(total / NUM_GENES)

    def to_dict(self) -> dict:
        return {"alleles": [[round(a, 6), round(b, 6)] for a, b in self.alleles]}

    @classmethod
    def from_dict(cls, d: dict) -> Genome:
        if not d or "alleles" not in d:
            return cls()
        return cls(alleles=[list(pair) for pair in d["alleles"]])

    def clamp(self) -> None:
        """Enforce [0, 1] bounds on all alleles."""
        for pair in self.alleles:
            pair[0] = max(0.0, min(1.0, pair[0]))
            pair[1] = max(0.0, min(1.0, pair[1]))


def create_genome_from_phenotype(
    stats: dict[str, float],
    skills: dict[str, float],
    rng: _random_module.Random,
    homozygous_noise: float = 0.02,
) -> Genome:
    """Create a near-homozygous genome matching observed phenotype.

    Founders get alleles close to their current trait values so existing
    behavior is preserved.  Small noise prevents perfect clones.
    """
    alleles: list[list[float]] = []
    for name in GENE_NAMES:
        value = stats.get(name, skills.get(name, 0.5))
        a = max(0.0, min(1.0, value + rng.gauss(0, homozygous_noise)))
        b = max(0.0, min(1.0, value + rng.gauss(0, homozygous_noise)))
        alleles.append([a, b])
    return Genome(alleles=alleles)


def crossover(
    parent_a: Genome,
    parent_b: Genome,
    rng: _random_module.Random,
) -> Genome:
    """Single-point crossover between two parent genomes.

    Each gene: pick one allele from each parent.  With CROSSOVER_PROBABILITY,
    a crossover point splits which parent contributes which allele.
    """
    child_alleles: list[list[float]] = []
    if rng.random() < CROSSOVER_PROBABILITY:
        xpoint = rng.randint(1, NUM_GENES - 1)
        for i in range(NUM_GENES):
            if i < xpoint:
                a = rng.choice(parent_a.alleles[i])
                b = rng.choice(parent_b.alleles[i])
            else:
                a = rng.choice(parent_b.alleles[i])
                b = rng.choice(parent_a.alleles[i])
            child_alleles.append([a, b])
    else:
        for i in range(NUM_GENES):
            a = rng.choice(parent_a.alleles[i])
            b = rng.choice(parent_b.alleles[i])
            child_alleles.append([a, b])
    return Genome(alleles=child_alleles)


def mutate(
    genome: Genome,
    rng: _random_module.Random,
    rate: float = MUTATION_RATE,
    sigma: float = MUTATION_SIGMA,
) -> int:
    """Apply gaussian mutation to alleles. Returns count of mutations."""
    mutations = 0
    for pair in genome.alleles:
        for idx in range(2):
            if rng.random() < rate:
                pair[idx] += rng.gauss(0, sigma)
                mutations += 1
    genome.clamp()
    return mutations


def breed(
    parent_a: Genome,
    parent_b: Genome,
    rng: _random_module.Random,
) -> tuple[Genome, int]:
    """Full breeding: crossover + mutation. Returns (child_genome, mutation_count)."""
    child = crossover(parent_a, parent_b, rng)
    n_mutations = mutate(child, rng)
    return child, n_mutations


def compute_hybrid_vigor(parent_a: Genome, parent_b: Genome) -> float:
    """Compute fitness bonus from genetic distance between parents.

    High distance (> threshold) → bonus to child fitness.
    Returns bonus in [0, HYBRID_VIGOR_BONUS].
    """
    dist = parent_a.genetic_distance(parent_b)
    if dist < HYBRID_VIGOR_THRESHOLD:
        return 0.0
    excess = min(1.0, (dist - HYBRID_VIGOR_THRESHOLD) / 0.5)
    return excess * HYBRID_VIGOR_BONUS


def compute_colony_diversity(genomes: list[Genome]) -> float:
    """Compute colony-wide genetic diversity as mean pairwise distance.

    Returns value in [0, ~1].  Higher = more diverse.
    """
    if len(genomes) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(genomes)):
        for j in range(i + 1, len(genomes)):
            total += genomes[i].genetic_distance(genomes[j])
            count += 1
    return total / count if count > 0 else 0.0


def compute_inbreeding_penalty(diversity: float) -> float:
    """Compute health penalty from low genetic diversity.

    Returns multiplier > 1.0 that increases death rate.
    """
    if diversity >= INBREEDING_THRESHOLD:
        return 1.0
    shortfall = INBREEDING_THRESHOLD - diversity
    penalty = (shortfall / INBREEDING_THRESHOLD) * INBREEDING_PENALTY_MAX
    return 1.0 + penalty


def epigenetic_activation(
    genome: Genome,
    stress: float,
    rng: _random_module.Random,
) -> str | None:
    """Stress-triggered epigenetic shift.  May activate latent alleles.

    High stress increases probability of gene expression shift.
    Returns name of affected gene or None.
    """
    prob = EPIGENETIC_PROBABILITY * (1.0 + stress)
    if rng.random() >= prob:
        return None
    gene_idx = rng.randint(0, NUM_GENES - 1)
    pair = genome.alleles[gene_idx]
    pair[0], pair[1] = pair[1], pair[0]
    genome.clamp()
    return GENE_NAMES[gene_idx]


def founder_gene_survival(
    founder_genomes: dict[str, Genome],
    current_genomes: dict[str, Genome],
) -> dict[str, float]:
    """Compute how much of each founder's genetic signature persists.

    Returns dict mapping founder_id → fraction [0, 1] of their alleles
    still detectable in the current population.
    """
    if not founder_genomes or not current_genomes:
        return {}
    result: dict[str, float] = {}
    for fid, fgenome in founder_genomes.items():
        total_signal = 0.0
        total_possible = 0.0
        for cid, cgenome in current_genomes.items():
            for i in range(NUM_GENES):
                fa, fb = fgenome.alleles[i]
                ca, cb = cgenome.alleles[i]
                for f_allele in (fa, fb):
                    closest = min(abs(ca - f_allele), abs(cb - f_allele))
                    total_signal += max(0.0, 1.0 - closest * 5.0)
                    total_possible += 1.0
        result[fid] = total_signal / total_possible if total_possible > 0 else 0.0
    return result


@dataclass
class GeneticsState:
    """Colony-wide genetics state."""
    diversity_index: float = 0.5
    generation_count: int = 0
    total_mutations: int = 0
    epigenetic_events: int = 0
    founder_survival: dict[str, float] = field(default_factory=dict)
    diversity_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "diversity_index": round(self.diversity_index, 6),
            "generation_count": self.generation_count,
            "total_mutations": self.total_mutations,
            "epigenetic_events": self.epigenetic_events,
            "founder_survival": {k: round(v, 4) for k, v in self.founder_survival.items()},
            "diversity_history": [round(d, 4) for d in self.diversity_history[-20:]],
        }

    @classmethod
    def from_dict(cls, d: dict) -> GeneticsState:
        if not d:
            return cls()
        return cls(
            diversity_index=d.get("diversity_index", 0.5),
            generation_count=d.get("generation_count", 0),
            total_mutations=d.get("total_mutations", 0),
            epigenetic_events=d.get("epigenetic_events", 0),
            founder_survival=dict(d.get("founder_survival", {})),
            diversity_history=list(d.get("diversity_history", [])),
        )


@dataclass
class GeneticsYearContext:
    """Input context for one genetics tick."""
    year: int
    births_this_year: int
    deaths_this_year: int
    population: int
    avg_stress: float


@dataclass
class GeneticsTickResult:
    """Output of one genetics tick."""
    diversity_index: float = 0.5
    inbreeding_penalty: float = 1.0
    epigenetic_events: list[dict] = field(default_factory=list)
    diversity_trend: str = "stable"
    mutations_this_year: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "diversity_index": round(self.diversity_index, 6),
            "inbreeding_penalty": round(self.inbreeding_penalty, 4),
            "epigenetic_events": self.epigenetic_events,
            "diversity_trend": self.diversity_trend,
            "mutations_this_year": self.mutations_this_year,
        }


def compute_health_modifier(state: GeneticsState) -> float:
    """Compute death-rate multiplier from genetic diversity.

    One-year lag: caller should use LAST year's state.
    """
    return compute_inbreeding_penalty(state.diversity_index)


def tick_genetics(
    state: GeneticsState,
    genomes: dict[str, Genome],
    founder_genomes: dict[str, Genome],
    ctx: GeneticsYearContext,
    rng: _random_module.Random,
) -> GeneticsTickResult:
    """Advance genetics by one year.  Mutates state in place.

    Diversity is computed from current genomes.  Inbreeding penalty uses
    LAST year's diversity (already stored in state before this call).
    Epigenetic events may fire for stressed colonists.
    """
    result = GeneticsTickResult()

    # Inbreeding penalty from LAST year's diversity (one-year lag)
    result.inbreeding_penalty = compute_inbreeding_penalty(state.diversity_index)

    # Compute current diversity
    genome_list = list(genomes.values())
    current_diversity = compute_colony_diversity(genome_list)
    result.diversity_index = current_diversity

    # Diversity trend
    if state.diversity_history:
        prev = state.diversity_history[-1]
        diff = current_diversity - prev
        if diff > 0.02:
            result.diversity_trend = "increasing"
        elif diff < -0.02:
            result.diversity_trend = "decreasing"
        else:
            result.diversity_trend = "stable"

    # Epigenetic events under stress
    if ctx.avg_stress > 0.4:
        for cid, genome in genomes.items():
            gene = epigenetic_activation(genome, ctx.avg_stress, rng)
            if gene:
                result.epigenetic_events.append({
                    "colonist_id": cid, "gene": gene, "year": ctx.year,
                })
                state.epigenetic_events += 1

    # Update state
    state.diversity_index = current_diversity
    state.diversity_history.append(current_diversity)
    if len(state.diversity_history) > 100:
        state.diversity_history = state.diversity_history[-100:]
    if ctx.births_this_year > 0:
        state.generation_count += 1

    # Founder gene survival (computed every 10 years for performance)
    if ctx.year % 10 == 0 and founder_genomes:
        state.founder_survival = founder_gene_survival(founder_genomes, genomes)

    return result
