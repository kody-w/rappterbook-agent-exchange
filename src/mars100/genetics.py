"""
Genetics organ for Mars-100 (engine v11.0).

Heritable traits across colonist generations.  Each colonist carries a diploid
genome: 5 loci with two alleles each (0.0-1.0).  Children inherit via crossover
with gaussian mutation.  Colony-level diversity and adaptation are tracked.

One-year lag: LAST year's genetics modifiers feed THIS year's resource effects.
Genomes themselves are immediate — a child born in year N has its genome from
year N, but colony-level bonuses are staged for year N+1.

RNG offset: seed + 13397
"""
from __future__ import annotations

import math
import random as _random_module
from dataclasses import dataclass, field
from typing import Any

# --- Loci ---

LOCUS_NAMES = (
    "radiation_resistance",
    "o2_efficiency",
    "bone_density",
    "cognitive_plasticity",
    "stress_resilience",
)

# Dominance weights: 0.5 = co-dominant (avg of alleles),
# higher = first allele biased.  All co-dominant for simplicity.
DOMINANCE: dict[str, float] = {name: 0.5 for name in LOCUS_NAMES}

# --- Constants ---

MUTATION_RATE: float = 0.005       # per allele per year
MUTATION_SIGMA: float = 0.06       # gaussian noise magnitude
RADIATION_MUTATION_BOOST: float = 0.01  # extra mutation rate during radiation
INBREEDING_PENALTY_THRESHOLD: float = 0.85  # genome similarity above this penalizes children
INBREEDING_STAT_PENALTY: float = 0.03
ADAPTATION_LOCI = ("radiation_resistance", "o2_efficiency")
MAX_DEATH_RATE_REDUCTION: float = 0.15
MAX_SKILL_LEARNING_BONUS: float = 0.20
MAX_STRESS_REDUCTION: float = 0.04
MAX_AIR_BONUS: float = 0.010
MAX_MUTATION_LOG: int = 50


# --- Data classes ---

@dataclass
class Genome:
    """Diploid genome: each locus has two alleles in [0, 1]."""
    alleles: dict[str, tuple[float, float]] = field(default_factory=dict)
    generation: int = 0

    def phenotype(self, locus: str) -> float:
        """Express phenotype for a locus.  Co-dominant: weighted average."""
        a, b = self.alleles.get(locus, (0.5, 0.5))
        d = DOMINANCE.get(locus, 0.5)
        return a * d + b * (1.0 - d)

    def phenotypes(self) -> dict[str, float]:
        """All phenotype values."""
        return {loc: self.phenotype(loc) for loc in LOCUS_NAMES}

    def similarity(self, other: Genome) -> float:
        """Genetic similarity [0-1].  1.0 = identical."""
        if not self.alleles or not other.alleles:
            return 0.0
        diffs = 0.0
        count = 0
        for loc in LOCUS_NAMES:
            a1, a2 = self.alleles.get(loc, (0.5, 0.5))
            b1, b2 = other.alleles.get(loc, (0.5, 0.5))
            diffs += abs(a1 - b1) + abs(a2 - b2)
            count += 2
        if count == 0:
            return 0.0
        return max(0.0, 1.0 - diffs / count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alleles": {k: [round(a, 4), round(b, 4)]
                        for k, (a, b) in self.alleles.items()},
            "generation": self.generation,
            "phenotypes": {k: round(v, 4) for k, v in self.phenotypes().items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Genome:
        alleles = {}
        for k, v in d.get("alleles", {}).items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                alleles[k] = (float(v[0]), float(v[1]))
        return cls(alleles=alleles, generation=d.get("generation", 0))


@dataclass
class GeneticsState:
    """Colony-level genetics tracking."""
    heterozygosity: float = 1.0
    adaptation_score: float = 0.0
    inbreeding_coefficient: float = 0.0
    generation_avg: float = 0.0
    mutation_log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "heterozygosity": round(self.heterozygosity, 4),
            "adaptation_score": round(self.adaptation_score, 4),
            "inbreeding_coefficient": round(self.inbreeding_coefficient, 4),
            "generation_avg": round(self.generation_avg, 2),
            "mutation_log": list(self.mutation_log[-MAX_MUTATION_LOG:]),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GeneticsState:
        return cls(
            heterozygosity=d.get("heterozygosity", 1.0),
            adaptation_score=d.get("adaptation_score", 0.0),
            inbreeding_coefficient=d.get("inbreeding_coefficient", 0.0),
            generation_avg=d.get("generation_avg", 0.0),
            mutation_log=list(d.get("mutation_log", [])),
        )


@dataclass
class GeneticsYearContext:
    """Input context for one genetics tick."""
    year: int
    active_ids: list[str]
    radiation_event: bool
    births_this_year: list[dict]  # [{id, parents: [a, b]}]
    deaths_this_year: list[str]   # colonist IDs


@dataclass
class GeneticsTickResult:
    """Output of one genetics tick."""
    death_rate_modifier: float = 1.0
    skill_learning_modifier: float = 1.0
    stress_modifier: float = 0.0
    air_bonus: float = 0.0
    mutations_this_year: list[dict] = field(default_factory=list)
    diversity_warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "death_rate_modifier": round(self.death_rate_modifier, 4),
            "skill_learning_modifier": round(self.skill_learning_modifier, 4),
            "stress_modifier": round(self.stress_modifier, 4),
            "air_bonus": round(self.air_bonus, 6),
            "mutations_this_year": self.mutations_this_year,
        }
        if self.diversity_warning:
            d["diversity_warning"] = self.diversity_warning
        return d


# --- Pure functions ---

def create_founder_genome(colonist_id: str, element: str,
                          rng: _random_module.Random) -> Genome:
    """Create a genome for a founding colonist.

    Founders get diverse starting alleles with element-based biases.
    """
    element_bias: dict[str, dict[str, float]] = {
        "fire": {"radiation_resistance": 0.1, "stress_resilience": 0.05},
        "water": {"o2_efficiency": 0.05, "stress_resilience": 0.1},
        "earth": {"bone_density": 0.1, "radiation_resistance": 0.05},
        "air": {"o2_efficiency": 0.1, "cognitive_plasticity": 0.05},
    }
    bias = element_bias.get(element, {})
    alleles: dict[str, tuple[float, float]] = {}
    for loc in LOCUS_NAMES:
        base = 0.4 + bias.get(loc, 0.0)
        a = max(0.0, min(1.0, base + rng.gauss(0, 0.12)))
        b = max(0.0, min(1.0, base + rng.gauss(0, 0.12)))
        alleles[loc] = (a, b)
    return Genome(alleles=alleles, generation=0)


def create_immigrant_genome(rng: _random_module.Random) -> Genome:
    """Create a genome for an immigrant colonist (fresh Earth genetics)."""
    alleles: dict[str, tuple[float, float]] = {}
    for loc in LOCUS_NAMES:
        a = max(0.0, min(1.0, rng.gauss(0.5, 0.15)))
        b = max(0.0, min(1.0, rng.gauss(0.5, 0.15)))
        alleles[loc] = (a, b)
    return Genome(alleles=alleles, generation=0)


def crossover(parent_a: Genome, parent_b: Genome,
              rng: _random_module.Random) -> Genome:
    """Create child genome via crossover + mutation.

    For each locus, child inherits one random allele from each parent.
    Then applies gaussian mutation noise.
    """
    child_alleles: dict[str, tuple[float, float]] = {}
    for loc in LOCUS_NAMES:
        a_alleles = parent_a.alleles.get(loc, (0.5, 0.5))
        b_alleles = parent_b.alleles.get(loc, (0.5, 0.5))
        # Inherit one allele from each parent
        from_a = a_alleles[rng.randint(0, 1)]
        from_b = b_alleles[rng.randint(0, 1)]
        # Mutation noise
        from_a = max(0.0, min(1.0, from_a + rng.gauss(0, MUTATION_SIGMA)))
        from_b = max(0.0, min(1.0, from_b + rng.gauss(0, MUTATION_SIGMA)))
        child_alleles[loc] = (from_a, from_b)
    child_gen = max(parent_a.generation, parent_b.generation) + 1
    return Genome(alleles=child_alleles, generation=child_gen)


def check_inbreeding(parent_a: Genome, parent_b: Genome) -> float:
    """Return inbreeding coefficient for a potential pairing.

    Returns similarity score; above INBREEDING_PENALTY_THRESHOLD causes
    stat penalties in offspring.
    """
    return parent_a.similarity(parent_b)


def apply_inbreeding_penalty(child_genome: Genome,
                             inbreeding_coeff: float) -> None:
    """Apply fitness penalty to child alleles if parents are too similar.

    Reduces all alleles slightly when inbreeding is high.
    """
    if inbreeding_coeff < INBREEDING_PENALTY_THRESHOLD:
        return
    excess = inbreeding_coeff - INBREEDING_PENALTY_THRESHOLD
    penalty = excess * INBREEDING_STAT_PENALTY * 10  # scale: 0..0.045
    for loc in LOCUS_NAMES:
        a, b = child_genome.alleles.get(loc, (0.5, 0.5))
        child_genome.alleles[loc] = (
            max(0.0, a - penalty),
            max(0.0, b - penalty),
        )


def compute_heterozygosity(genomes: list[Genome]) -> float:
    """Compute average heterozygosity across loci.

    For each locus: expected heterozygosity = 1 - sum(p_i^2) where p_i are
    allele frequency bins.  Averaged across loci.  Higher = more diverse.
    """
    if not genomes:
        return 0.0
    n_bins = 10
    locus_hets: list[float] = []
    for loc in LOCUS_NAMES:
        all_alleles: list[float] = []
        for g in genomes:
            a, b = g.alleles.get(loc, (0.5, 0.5))
            all_alleles.extend([a, b])
        if not all_alleles:
            locus_hets.append(0.0)
            continue
        # Bin alleles into n_bins buckets
        bins = [0] * n_bins
        for val in all_alleles:
            idx = min(n_bins - 1, int(val * n_bins))
            bins[idx] += 1
        total = len(all_alleles)
        het = 1.0 - sum((c / total) ** 2 for c in bins)
        locus_hets.append(het)
    return sum(locus_hets) / len(locus_hets) if locus_hets else 0.0


def compute_adaptation_score(genomes: list[Genome]) -> float:
    """Compute colony-wide Mars adaptation score.

    Average phenotype of adaptation-relevant loci across all colonists.
    """
    if not genomes:
        return 0.0
    scores: list[float] = []
    for g in genomes:
        vals = [g.phenotype(loc) for loc in ADAPTATION_LOCI]
        scores.append(sum(vals) / len(vals))
    return sum(scores) / len(scores)


def compute_avg_inbreeding(genomes: list[Genome]) -> float:
    """Compute average pairwise genetic similarity (inbreeding proxy)."""
    if len(genomes) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(genomes)):
        for j in range(i + 1, len(genomes)):
            total += genomes[i].similarity(genomes[j])
            count += 1
    return total / count if count > 0 else 0.0


def compute_genetics_modifiers(genomes: list[Genome]) -> dict[str, float]:
    """Compute resource/rate modifiers from colony genetics.

    Returns multipliers/bonuses to be staged for NEXT year (one-year lag).
    """
    if not genomes:
        return {
            "death_rate_modifier": 1.0,
            "skill_learning_modifier": 1.0,
            "stress_modifier": 0.0,
            "air_bonus": 0.0,
        }
    avg_pheno: dict[str, float] = {}
    for loc in LOCUS_NAMES:
        vals = [g.phenotype(loc) for loc in [loc] for g in genomes]
        avg_pheno[loc] = sum(vals) / len(vals)

    # Higher radiation_resistance + bone_density → lower death rate
    rad = avg_pheno.get("radiation_resistance", 0.5)
    bone = avg_pheno.get("bone_density", 0.5)
    death_mod = max(1.0 - MAX_DEATH_RATE_REDUCTION,
                    1.0 - (rad * 0.6 + bone * 0.4) * MAX_DEATH_RATE_REDUCTION * 2)

    # Higher cognitive_plasticity → faster skill learning
    cog = avg_pheno.get("cognitive_plasticity", 0.5)
    skill_mod = 1.0 + (cog - 0.5) * MAX_SKILL_LEARNING_BONUS * 2

    # Higher stress_resilience → stress reduction
    stress_res = avg_pheno.get("stress_resilience", 0.5)
    stress_mod = (stress_res - 0.5) * MAX_STRESS_REDUCTION * 2

    # Higher o2_efficiency → air bonus
    o2_eff = avg_pheno.get("o2_efficiency", 0.5)
    air_bonus = (o2_eff - 0.5) * MAX_AIR_BONUS * 2

    return {
        "death_rate_modifier": round(death_mod, 4),
        "skill_learning_modifier": round(skill_mod, 4),
        "stress_modifier": round(stress_mod, 4),
        "air_bonus": round(air_bonus, 6),
    }


def mutate_alleles(genome: Genome, mutation_rate: float,
                   rng: _random_module.Random) -> list[dict]:
    """Apply random mutations to a genome.  Returns list of mutation records."""
    mutations: list[dict] = []
    for loc in LOCUS_NAMES:
        a, b = genome.alleles.get(loc, (0.5, 0.5))
        if rng.random() < mutation_rate:
            old_a = a
            a = max(0.0, min(1.0, a + rng.gauss(0, MUTATION_SIGMA * 2)))
            mutations.append({"locus": loc, "allele": 0,
                              "old": round(old_a, 4), "new": round(a, 4)})
        if rng.random() < mutation_rate:
            old_b = b
            b = max(0.0, min(1.0, b + rng.gauss(0, MUTATION_SIGMA * 2)))
            mutations.append({"locus": loc, "allele": 1,
                              "old": round(old_b, 4), "new": round(b, 4)})
        genome.alleles[loc] = (a, b)
    return mutations


def tick_genetics(
    state: GeneticsState,
    genome_map: dict[str, Genome],
    ctx: GeneticsYearContext,
    rng: _random_module.Random,
) -> GeneticsTickResult:
    """Advance genetics by one year.  Mutates state and genome_map in place.

    Modifiers are computed from state BEFORE mutation (one-year lag).
    """
    result = GeneticsTickResult()

    # Compute modifiers from LAST year's state (before we mutate)
    active_genomes = [genome_map[cid] for cid in ctx.active_ids
                      if cid in genome_map]
    mods = compute_genetics_modifiers(active_genomes)
    result.death_rate_modifier = mods["death_rate_modifier"]
    result.skill_learning_modifier = mods["skill_learning_modifier"]
    result.stress_modifier = mods["stress_modifier"]
    result.air_bonus = mods["air_bonus"]

    # --- Mutations ---
    rate = MUTATION_RATE
    if ctx.radiation_event:
        rate += RADIATION_MUTATION_BOOST
    all_mutations: list[dict] = []
    for cid in ctx.active_ids:
        genome = genome_map.get(cid)
        if genome is None:
            continue
        muts = mutate_alleles(genome, rate, rng)
        for m in muts:
            m["colonist_id"] = cid
            m["year"] = ctx.year
        all_mutations.extend(muts)
    result.mutations_this_year = all_mutations
    state.mutation_log.extend(all_mutations)
    if len(state.mutation_log) > MAX_MUTATION_LOG:
        state.mutation_log = state.mutation_log[-MAX_MUTATION_LOG:]

    # --- Handle births (genomes already created by engine) ---
    # Engine creates child genomes via crossover() and stores in genome_map

    # --- Remove dead colonists from genome_map ---
    for dead_id in ctx.deaths_this_year:
        genome_map.pop(dead_id, None)

    # --- Recompute colony metrics ---
    active_genomes = [genome_map[cid] for cid in ctx.active_ids
                      if cid in genome_map and cid not in ctx.deaths_this_year]
    state.heterozygosity = compute_heterozygosity(active_genomes)
    state.adaptation_score = compute_adaptation_score(active_genomes)
    state.inbreeding_coefficient = compute_avg_inbreeding(active_genomes)
    if active_genomes:
        state.generation_avg = (sum(g.generation for g in active_genomes)
                                / len(active_genomes))
    else:
        state.generation_avg = 0.0

    # Diversity warning
    if state.heterozygosity < 0.3 and len(active_genomes) >= 3:
        result.diversity_warning = (
            f"Genetic bottleneck: heterozygosity {state.heterozygosity:.2f}")

    return result
