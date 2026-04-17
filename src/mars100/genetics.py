"""
Genetics organ for Mars-100 (engine v11.0).

Models heredity and adaptation: diploid genomes, crossover + mutation,
phenotype expression, and colony-level diversity tracking.

Phase 1 wires two live effects:
  - o2_tolerance: softens marginal asphyxiation risk
  - cognitive_plasticity: modulates skill learning speed

Other loci (radiation_resistance, bone_density, perchlorate_tolerance,
social_binding) are tracked in genomes but deferred to future organs.

RNG offset: seed + 12553
"""
from __future__ import annotations

import math
import random as _random_module
from dataclasses import dataclass, field
from typing import Any

LOCI = (
    "o2_tolerance",
    "radiation_resistance",
    "bone_density",
    "perchlorate_tolerance",
    "cognitive_plasticity",
    "social_binding",
)

# --- Earth baseline parameters ---
EARTH_BASELINE_MEAN = 0.50
EARTH_BASELINE_SPREAD = 0.05  # alleles in [0.45, 0.55]

# --- Mutation parameters ---
MUTATION_RATE = 0.10         # per-locus chance of mutation during reproduction
MUTATION_SIGMA = 0.06        # gaussian std-dev of mutation magnitude

# --- Phenotype effect caps ---
MAX_O2_SURVIVAL_BONUS = 0.15  # max extra survival chance at marginal air
MAX_SKILL_MULTIPLIER = 0.10   # cognitive_plasticity: ±10% on skill gain

# --- Diversity thresholds ---
BOTTLENECK_THRESHOLD = 0.20   # diversity below this triggers an event
HEALTHY_DIVERSITY = 0.40


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Genome:
    """Diploid genome: each locus has two alleles in [0, 1]."""

    alleles: dict[str, tuple[float, float]]

    def phenotype(self, locus: str) -> float:
        """Express phenotype as mean of the two alleles."""
        a, b = self.alleles[locus]
        return (a + b) / 2.0

    def phenotypes(self) -> dict[str, float]:
        """All phenotype values."""
        return {loc: self.phenotype(loc) for loc in LOCI}

    def to_dict(self) -> dict:
        return {
            loc: list(self.alleles[loc]) for loc in LOCI
        }

    @classmethod
    def from_dict(cls, d: dict) -> Genome:
        alleles: dict[str, tuple[float, float]] = {}
        for loc in LOCI:
            pair = d.get(loc, [EARTH_BASELINE_MEAN, EARTH_BASELINE_MEAN])
            alleles[loc] = (float(pair[0]), float(pair[1]))
        return cls(alleles=alleles)


@dataclass
class GeneticsState:
    """Colony-wide genetics state."""

    genomes: dict[str, Genome] = field(default_factory=dict)
    pedigree: dict[str, list[str | None]] = field(default_factory=dict)
    generation: dict[str, int] = field(default_factory=dict)
    diversity_history: list[float] = field(default_factory=list)

    # --- Registration helpers (centralized bookkeeping) ---

    def register_founder(self, colonist_id: str, genome: Genome) -> None:
        """Register a founding colonist with an Earth-baseline genome."""
        self.genomes[colonist_id] = genome
        self.pedigree[colonist_id] = [None, None]
        self.generation[colonist_id] = 0

    def register_birth(self, child_id: str, parent_a_id: str,
                       parent_b_id: str, genome: Genome) -> None:
        """Register a Mars-born child."""
        self.genomes[child_id] = genome
        self.pedigree[child_id] = [parent_a_id, parent_b_id]
        gen_a = self.generation.get(parent_a_id, 0)
        gen_b = self.generation.get(parent_b_id, 0)
        self.generation[child_id] = max(gen_a, gen_b) + 1

    def register_immigrant(self, colonist_id: str, genome: Genome) -> None:
        """Register an Earth immigrant (generation 0, no parents)."""
        self.genomes[colonist_id] = genome
        self.pedigree[colonist_id] = [None, None]
        self.generation[colonist_id] = 0

    def to_dict(self) -> dict:
        return {
            "genomes": {cid: g.to_dict() for cid, g in self.genomes.items()},
            "pedigree": dict(self.pedigree),
            "generation": dict(self.generation),
            "diversity_history": list(self.diversity_history),
        }

    @classmethod
    def from_dict(cls, d: dict) -> GeneticsState:
        genomes = {cid: Genome.from_dict(gd)
                   for cid, gd in d.get("genomes", {}).items()}
        return cls(
            genomes=genomes,
            pedigree={k: list(v) for k, v in d.get("pedigree", {}).items()},
            generation=dict(d.get("generation", {})),
            diversity_history=list(d.get("diversity_history", [])),
        )


@dataclass
class GeneticsYearContext:
    """Input context for the yearly genetics tick."""
    year: int
    active_ids: list[str]


@dataclass
class GeneticsTickResult:
    """Output of a yearly genetics tick."""
    phenotypes: dict[str, dict[str, float]] = field(default_factory=dict)
    diversity_index: float = 1.0
    mars_adaptation: float = 0.0
    max_generation: int = 0
    avg_generation: float = 0.0
    notable_events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "diversity_index": round(self.diversity_index, 4),
            "mars_adaptation": round(self.mars_adaptation, 4),
            "max_generation": self.max_generation,
            "avg_generation": round(self.avg_generation, 2),
            "notable_events": list(self.notable_events),
        }


# ---------------------------------------------------------------------------
# Genome creation
# ---------------------------------------------------------------------------

def create_earth_genome(rng: _random_module.Random) -> Genome:
    """Create an Earth-baseline genome with slight individual variation."""
    alleles: dict[str, tuple[float, float]] = {}
    for loc in LOCI:
        a = max(0.0, min(1.0, rng.gauss(EARTH_BASELINE_MEAN, EARTH_BASELINE_SPREAD)))
        b = max(0.0, min(1.0, rng.gauss(EARTH_BASELINE_MEAN, EARTH_BASELINE_SPREAD)))
        alleles[loc] = (round(a, 4), round(b, 4))
    return Genome(alleles=alleles)


def create_child_genome(parent_a: Genome, parent_b: Genome,
                        rng: _random_module.Random) -> Genome:
    """Create a child genome via crossover + mutation.

    For each locus: randomly pick one allele from each parent (crossover),
    then apply mutation with probability MUTATION_RATE.
    """
    alleles: dict[str, tuple[float, float]] = {}
    for loc in LOCI:
        # Crossover: one allele from each parent
        from_a = parent_a.alleles[loc][rng.randint(0, 1)]
        from_b = parent_b.alleles[loc][rng.randint(0, 1)]
        # Mutation
        if rng.random() < MUTATION_RATE:
            from_a = max(0.0, min(1.0, from_a + rng.gauss(0, MUTATION_SIGMA)))
        if rng.random() < MUTATION_RATE:
            from_b = max(0.0, min(1.0, from_b + rng.gauss(0, MUTATION_SIGMA)))
        alleles[loc] = (round(from_a, 4), round(from_b, 4))
    return Genome(alleles=alleles)


# ---------------------------------------------------------------------------
# Phenotype expression
# ---------------------------------------------------------------------------

def express_phenotype(genome: Genome) -> dict[str, float]:
    """Express all loci to phenotype values (0-1 range)."""
    return genome.phenotypes()


def compute_o2_survival_bonus(phenotype_o2: float, air_level: float) -> float:
    """Compute survival bonus from O2 tolerance in marginal air conditions.

    Only effective when air is between 0.05 and 0.15 (marginal zone).
    Returns a probability reduction for death (0 to MAX_O2_SURVIVAL_BONUS).
    Below 0.05 (catastrophic), no individual adaptation helps.
    Above 0.15 (safe), no bonus needed.
    """
    if air_level <= 0.05 or air_level >= 0.15:
        return 0.0
    # Marginal zone: bonus scales with tolerance above baseline
    excess = max(0.0, phenotype_o2 - EARTH_BASELINE_MEAN)
    # Scale by how marginal the air is (closer to 0.05 = less effective)
    marginality = (air_level - 0.05) / 0.10  # 0 at 0.05, 1 at 0.15
    return min(MAX_O2_SURVIVAL_BONUS, excess * 0.3 * marginality)


def compute_skill_multiplier(phenotype_cog: float) -> float:
    """Compute skill learning rate multiplier from cognitive plasticity.

    Returns a multiplier in [1 - MAX_SKILL_MULTIPLIER, 1 + MAX_SKILL_MULTIPLIER].
    Earth baseline (0.5) gives 1.0x.
    """
    deviation = phenotype_cog - EARTH_BASELINE_MEAN
    return 1.0 + (deviation / 0.5) * MAX_SKILL_MULTIPLIER


# ---------------------------------------------------------------------------
# Colony-level metrics
# ---------------------------------------------------------------------------

def compute_diversity(genomes: dict[str, Genome],
                      active_ids: list[str]) -> float:
    """Compute genetic diversity as mean heterozygosity across active colonists.

    Heterozygosity per locus per individual = |allele_a - allele_b|.
    Colony diversity = mean heterozygosity across all loci and individuals.
    Higher values indicate more genetic variation.
    """
    active_genomes = [genomes[cid] for cid in active_ids if cid in genomes]
    if len(active_genomes) < 2:
        return 0.0
    total_het = 0.0
    count = 0
    for genome in active_genomes:
        for loc in LOCI:
            a, b = genome.alleles[loc]
            total_het += abs(a - b)
            count += 1
    individual_het = total_het / max(1, count)

    # Also compute allele frequency variance across the colony
    allele_variance = 0.0
    for loc in LOCI:
        all_alleles = []
        for g in active_genomes:
            all_alleles.extend(g.alleles[loc])
        if len(all_alleles) >= 2:
            mean_a = sum(all_alleles) / len(all_alleles)
            var = sum((x - mean_a) ** 2 for x in all_alleles) / len(all_alleles)
            allele_variance += var
    allele_variance /= len(LOCI)

    # Combine: weight individual heterozygosity and population variance
    return min(1.0, individual_het * 0.5 + allele_variance * 5.0)


def compute_mars_adaptation(genomes: dict[str, Genome],
                            active_ids: list[str]) -> float:
    """Compute how far the colony has drifted from Earth baseline.

    Returns 0.0 for pure Earth-baseline, higher for Mars-adapted.
    """
    active_genomes = [genomes[cid] for cid in active_ids if cid in genomes]
    if not active_genomes:
        return 0.0
    total_drift = 0.0
    count = 0
    for genome in active_genomes:
        for loc in LOCI:
            phenotype = genome.phenotype(loc)
            total_drift += abs(phenotype - EARTH_BASELINE_MEAN)
            count += 1
    return total_drift / max(1, count)


def is_close_relative(colonist_a: str, colonist_b: str,
                      pedigree: dict[str, list[str | None]]) -> bool:
    """Check if two colonists share a parent (half-siblings or closer)."""
    parents_a = pedigree.get(colonist_a, [None, None])
    parents_b = pedigree.get(colonist_b, [None, None])
    known_a = {p for p in parents_a if p is not None}
    known_b = {p for p in parents_b if p is not None}
    if not known_a or not known_b:
        return False
    return bool(known_a & known_b)


# ---------------------------------------------------------------------------
# Yearly tick
# ---------------------------------------------------------------------------

def tick_genetics(state: GeneticsState, ctx: GeneticsYearContext,
                  rng: _random_module.Random) -> GeneticsTickResult:
    """Compute yearly genetics metrics for the colony.

    This is a bookkeeping/analysis tick — genome creation for births and
    immigrants is handled inline by the engine during those events.
    """
    result = GeneticsTickResult()

    # Compute phenotypes for all active colonists
    for cid in ctx.active_ids:
        genome = state.genomes.get(cid)
        if genome:
            result.phenotypes[cid] = express_phenotype(genome)

    # Colony-level diversity
    result.diversity_index = compute_diversity(state.genomes, ctx.active_ids)
    state.diversity_history.append(round(result.diversity_index, 4))

    # Mars adaptation score
    result.mars_adaptation = compute_mars_adaptation(state.genomes, ctx.active_ids)

    # Generation statistics
    gens = [state.generation.get(cid, 0) for cid in ctx.active_ids]
    if gens:
        result.max_generation = max(gens)
        result.avg_generation = sum(gens) / len(gens)
    else:
        result.max_generation = 0
        result.avg_generation = 0.0

    # Notable events
    if result.max_generation >= 2 and ctx.year > 1:
        third_gen = [cid for cid in ctx.active_ids
                     if state.generation.get(cid, 0) >= 2]
        if len(third_gen) == 1:
            result.notable_events.append(
                f"First third-generation colonist: {third_gen[0]}")

    if result.max_generation >= 3:
        fourth_gen = [cid for cid in ctx.active_ids
                      if state.generation.get(cid, 0) >= 3]
        if len(fourth_gen) == 1:
            result.notable_events.append(
                f"First fourth-generation colonist: {fourth_gen[0]}")

    if result.diversity_index < BOTTLENECK_THRESHOLD and len(ctx.active_ids) >= 4:
        result.notable_events.append(
            f"Genetic bottleneck detected — diversity {result.diversity_index:.3f}")

    # Detect rising adaptation
    if result.mars_adaptation > 0.08:
        result.notable_events.append(
            f"Mars adaptation diverging from Earth baseline: {result.mars_adaptation:.3f}")

    return result
