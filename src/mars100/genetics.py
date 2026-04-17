"""
Genetics organ for Mars-100 (engine v11.0).

Models trait inheritance, mutation, and adaptive drift in a small colony.
Continuous effect alleles (not classical Mendelian) — phenotype = mean(a, b).
Founders start from Earth-human priors; Mars selection pressure favors
radiation resistance, bone density, O2 efficiency, and cold tolerance.

RNG offset: seed + 12553
"""
from __future__ import annotations

import math
import random as _random_module
from dataclasses import dataclass, field
from typing import Any

# --- Trait definitions ---

TRAIT_NAMES = (
    "radiation_resistance",
    "bone_density",
    "o2_efficiency",
    "cold_tolerance",
    "social_aptitude",
    "risk_taking",
    "longevity",
    "fertility",
)

# Earth-human prior: (mean, stddev) for each allele.
# Mars-optimal is 1.0 for physical traits; 0.5 for behavioral traits.
EARTH_PRIORS: dict[str, tuple[float, float]] = {
    "radiation_resistance": (0.25, 0.12),
    "bone_density":         (0.60, 0.10),
    "o2_efficiency":        (0.30, 0.10),
    "cold_tolerance":       (0.20, 0.12),
    "social_aptitude":      (0.50, 0.15),
    "risk_taking":          (0.50, 0.15),
    "longevity":            (0.50, 0.12),
    "fertility":            (0.50, 0.12),
}

MUTATION_RATE = 0.02           # per allele per generation
MUTATION_MAGNITUDE = 0.08      # stddev of mutation shift

# Mars fitness weights (physical traits matter more)
MARS_FITNESS_WEIGHTS: dict[str, float] = {
    "radiation_resistance": 0.30,
    "bone_density":         0.20,
    "o2_efficiency":        0.25,
    "cold_tolerance":       0.20,
    "social_aptitude":      0.02,
    "risk_taking":          0.01,
    "longevity":            0.01,
    "fertility":            0.01,
}

# Modifier caps (genetics never dominates)
MAX_DEATH_MODIFIER = 1.3       # worst case: +30% death rate
MIN_DEATH_MODIFIER = 0.85      # best case: -15% death rate
MAX_BIRTH_MODIFIER = 1.2
MIN_BIRTH_MODIFIER = 0.7

# Diversity thresholds
BOTTLENECK_THRESHOLD = 0.10    # below → health penalty
HEALTHY_DIVERSITY = 0.25       # above → no penalty

REPRODUCTIVE_AGE_MIN = 15      # Mars-years since birth
REPRODUCTIVE_AGE_MAX = 50


# --- Genome ---

@dataclass
class Genome:
    """Colonist genome: continuous-effect alleles for each trait."""
    alleles: dict[str, tuple[float, float]] = field(default_factory=dict)

    def phenotype(self, trait: str) -> float:
        """Express a trait: mean of two alleles."""
        pair = self.alleles.get(trait, (0.5, 0.5))
        return (pair[0] + pair[1]) / 2.0

    def all_phenotypes(self) -> dict[str, float]:
        """Express all traits."""
        return {t: self.phenotype(t) for t in TRAIT_NAMES}

    def to_dict(self) -> dict:
        return {t: [round(a, 4), round(b, 4)]
                for t, (a, b) in self.alleles.items()}

    @classmethod
    def from_dict(cls, d: dict) -> Genome:
        alleles = {}
        for t in TRAIT_NAMES:
            pair = d.get(t, [0.5, 0.5])
            if isinstance(pair, list) and len(pair) == 2:
                alleles[t] = (float(pair[0]), float(pair[1]))
            else:
                alleles[t] = (0.5, 0.5)
        return cls(alleles=alleles)

    def clamp(self) -> None:
        """Enforce [0, 1] bounds on all alleles."""
        for t in list(self.alleles):
            a, b = self.alleles[t]
            self.alleles[t] = (max(0.0, min(1.0, a)),
                               max(0.0, min(1.0, b)))


def create_genome(rng: _random_module.Random) -> Genome:
    """Create a genome from Earth-human priors."""
    alleles: dict[str, tuple[float, float]] = {}
    for trait in TRAIT_NAMES:
        mean, sd = EARTH_PRIORS[trait]
        a = max(0.0, min(1.0, rng.gauss(mean, sd)))
        b = max(0.0, min(1.0, rng.gauss(mean, sd)))
        alleles[trait] = (a, b)
    return Genome(alleles=alleles)


def create_genome_deterministic(colonist_id: str) -> Genome:
    """Create a deterministic genome from a colonist ID (for legacy saves)."""
    seed = sum(ord(c) for c in colonist_id)
    rng = _random_module.Random(seed)
    return create_genome(rng)


def inherit_genome(
    parent_a: Genome,
    parent_b: Genome,
    rng: _random_module.Random,
) -> Genome:
    """Create a child genome by inheriting one allele from each parent.

    Each trait: child gets one randomly chosen allele from parent A
    and one from parent B, with possible mutation.
    """
    alleles: dict[str, tuple[float, float]] = {}
    for trait in TRAIT_NAMES:
        pa = parent_a.alleles.get(trait, (0.5, 0.5))
        pb = parent_b.alleles.get(trait, (0.5, 0.5))
        # Pick one allele from each parent
        from_a = pa[rng.randint(0, 1)]
        from_b = pb[rng.randint(0, 1)]
        # Apply mutation
        if rng.random() < MUTATION_RATE:
            from_a += rng.gauss(0, MUTATION_MAGNITUDE)
        if rng.random() < MUTATION_RATE:
            from_b += rng.gauss(0, MUTATION_MAGNITUDE)
        from_a = max(0.0, min(1.0, from_a))
        from_b = max(0.0, min(1.0, from_b))
        alleles[trait] = (from_a, from_b)
    return Genome(alleles=alleles)


# --- Colony-level genetics state ---

@dataclass
class GeneticsState:
    """Colony-wide genetics metrics."""
    diversity: float = 0.5          # allele distance-based diversity proxy
    avg_fitness: float = 0.3        # average Mars fitness
    max_generation: int = 0
    mars_adapted_count: int = 0     # colonists with fitness > 0.7
    bottleneck_years: int = 0       # consecutive years below diversity threshold
    adaptation_trend: list[float] = field(default_factory=list)  # last 10 fitness values

    def to_dict(self) -> dict:
        return {
            "diversity": round(self.diversity, 4),
            "avg_fitness": round(self.avg_fitness, 4),
            "max_generation": self.max_generation,
            "mars_adapted_count": self.mars_adapted_count,
            "bottleneck_years": self.bottleneck_years,
            "adaptation_trend": [round(v, 4) for v in self.adaptation_trend[-10:]],
        }

    @classmethod
    def from_dict(cls, d: dict) -> GeneticsState:
        return cls(
            diversity=d.get("diversity", 0.5),
            avg_fitness=d.get("avg_fitness", 0.3),
            max_generation=d.get("max_generation", 0),
            mars_adapted_count=d.get("mars_adapted_count", 0),
            bottleneck_years=d.get("bottleneck_years", 0),
            adaptation_trend=list(d.get("adaptation_trend", [])),
        )

    def health(self) -> float:
        """Overall genetic health [0, 1]."""
        diversity_score = min(1.0, self.diversity / HEALTHY_DIVERSITY)
        fitness_score = self.avg_fitness
        return diversity_score * 0.5 + fitness_score * 0.5


# --- Computation functions ---

def compute_mars_fitness(phenotype: dict[str, float]) -> float:
    """Compute how Mars-adapted a phenotype is [0, 1]."""
    score = 0.0
    for trait, weight in MARS_FITNESS_WEIGHTS.items():
        val = phenotype.get(trait, 0.5)
        score += val * weight
    return max(0.0, min(1.0, score))


def compute_diversity(genomes: list[Genome]) -> float:
    """Compute genetic diversity as mean pairwise allele distance.

    Returns a value in [0, 1]. Higher = more diverse.
    """
    n = len(genomes)
    if n < 2:
        return 0.0
    total_distance = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = _genome_distance(genomes[i], genomes[j])
            total_distance += dist
            pair_count += 1
    return total_distance / pair_count if pair_count > 0 else 0.0


def _genome_distance(a: Genome, b: Genome) -> float:
    """Mean absolute allele difference across all traits."""
    total = 0.0
    count = 0
    for trait in TRAIT_NAMES:
        pa = a.alleles.get(trait, (0.5, 0.5))
        pb = b.alleles.get(trait, (0.5, 0.5))
        # Compare sorted alleles to ignore allele ordering
        sa = sorted(pa)
        sb = sorted(pb)
        total += abs(sa[0] - sb[0]) + abs(sa[1] - sb[1])
        count += 2
    return total / count if count > 0 else 0.0


def compute_genetic_death_modifier(phenotype: dict[str, float]) -> float:
    """Compute death rate multiplier from genetics.

    High radiation_resistance, bone_density → lower death rate.
    High longevity → lower death rate.
    """
    rad = phenotype.get("radiation_resistance", 0.25)
    bone = phenotype.get("bone_density", 0.60)
    longevity = phenotype.get("longevity", 0.50)

    # Base modifier: 1.0. Better genetics → lower modifier.
    # Each trait can reduce by up to ~10%
    modifier = 1.0
    modifier -= (rad - 0.25) * 0.15        # from Earth-prior mean
    modifier -= (bone - 0.60) * 0.10
    modifier -= (longevity - 0.50) * 0.12

    return max(MIN_DEATH_MODIFIER, min(MAX_DEATH_MODIFIER, modifier))


def compute_genetic_birth_modifier(
    phenotype: dict[str, float],
    colonist_age: int,
) -> float:
    """Compute birth probability multiplier from genetics.

    Only applies to reproductive-age colonists.
    """
    if colonist_age < REPRODUCTIVE_AGE_MIN or colonist_age > REPRODUCTIVE_AGE_MAX:
        return 0.0  # not reproductive age

    fert = phenotype.get("fertility", 0.50)
    # Center around 1.0: high fertility → higher chance
    modifier = 0.7 + fert * 0.6  # range [0.7, 1.3] → clamped to [0.7, 1.2]
    return max(MIN_BIRTH_MODIFIER, min(MAX_BIRTH_MODIFIER, modifier))


def compute_bottleneck_penalty(state: GeneticsState) -> float:
    """Compute health penalty from genetic bottleneck.

    Returns a stress increase value [0, 0.05].
    Kicks in after consecutive low-diversity years.
    """
    if state.diversity >= BOTTLENECK_THRESHOLD:
        return 0.0
    severity = (BOTTLENECK_THRESHOLD - state.diversity) / BOTTLENECK_THRESHOLD
    years_factor = min(1.0, state.bottleneck_years / 10.0)
    return severity * years_factor * 0.05


# --- Tick ---

@dataclass
class GeneticsYearContext:
    """Input context for one genetics tick."""
    year: int
    active_colonists: list[Any]    # Colonist objects with .genome, .generation, .year_born
    births_this_year: list[dict]   # birth records
    deaths_this_year: list[dict]   # death records


@dataclass
class GeneticsTickResult:
    """Output of one genetics tick."""
    diversity: float = 0.0
    avg_fitness: float = 0.0
    max_generation: int = 0
    mars_adapted_count: int = 0
    bottleneck_penalty: float = 0.0
    adaptation_event: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "diversity": round(self.diversity, 4),
            "avg_fitness": round(self.avg_fitness, 4),
            "max_generation": self.max_generation,
            "mars_adapted_count": self.mars_adapted_count,
            "bottleneck_penalty": round(self.bottleneck_penalty, 4),
        }
        if self.adaptation_event:
            d["adaptation_event"] = self.adaptation_event
        return d


def tick_genetics(
    state: GeneticsState,
    ctx: GeneticsYearContext,
    rng: _random_module.Random,
) -> GeneticsTickResult:
    """Advance genetics by one year. Mutates state in place."""
    result = GeneticsTickResult()

    colonists = ctx.active_colonists
    if not colonists:
        return result

    # Gather genomes from active colonists
    genomes: list[Genome] = []
    fitnesses: list[float] = []
    max_gen = 0
    adapted = 0

    for c in colonists:
        genome = getattr(c, 'genome', None)
        if genome is None:
            genome = create_genome_deterministic(c.id)
        if isinstance(genome, dict):
            genome = Genome.from_dict(genome)
        genomes.append(genome)

        pheno = genome.all_phenotypes()
        fitness = compute_mars_fitness(pheno)
        fitnesses.append(fitness)

        gen = getattr(c, 'generation', 0)
        max_gen = max(max_gen, gen)
        if fitness > 0.7:
            adapted += 1

    # Diversity
    diversity = compute_diversity(genomes)
    state.diversity = diversity
    result.diversity = diversity

    # Fitness
    avg_fit = sum(fitnesses) / len(fitnesses)
    state.avg_fitness = avg_fit
    result.avg_fitness = avg_fit

    # Generation tracking
    state.max_generation = max_gen
    result.max_generation = max_gen

    # Adapted count
    state.mars_adapted_count = adapted
    result.mars_adapted_count = adapted

    # Bottleneck tracking
    if diversity < BOTTLENECK_THRESHOLD:
        state.bottleneck_years += 1
    else:
        state.bottleneck_years = 0

    penalty = compute_bottleneck_penalty(state)
    result.bottleneck_penalty = penalty

    # Adaptation trend (rolling window of 10)
    state.adaptation_trend.append(avg_fit)
    if len(state.adaptation_trend) > 10:
        state.adaptation_trend = state.adaptation_trend[-10:]

    # Check for adaptation milestone events
    if len(state.adaptation_trend) >= 5:
        recent = state.adaptation_trend[-5:]
        early = state.adaptation_trend[:5] if len(state.adaptation_trend) >= 10 else [0.3]
        trend = sum(recent) / len(recent) - sum(early) / len(early)
        if trend > 0.1 and adapted >= 3:
            result.adaptation_event = "rapid_adaptation"
        elif diversity < 0.05 and len(colonists) > 5:
            result.adaptation_event = "genetic_bottleneck"
        elif max_gen >= 3 and avg_fit > 0.6:
            result.adaptation_event = "martian_phenotype_emerging"

    return result
