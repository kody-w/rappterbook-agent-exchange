"""
Genetics organ for Mars-100 (engine v11.0).

Models diploid genomes with 12 loci. Each locus has two alleles (0.0-1.0).
Reproduction uses gamete formation (one allele per locus from each parent)
plus per-allele mutation.  Phenotype expression modifies effective stats
without mutating base stats.

Population-level metrics track diversity, heterozygosity, and genetic drift
across generations.  Genetic conditions (recessive traits) express when
both alleles at a locus fall below a threshold.

RNG offset: seed + 13337
"""
from __future__ import annotations

import math
import random as _random_module
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Locus definitions
# ---------------------------------------------------------------------------

STAT_LOCI = (
    "resolve_gene", "improvisation_gene", "empathy_gene",
    "hoarding_gene", "faith_gene", "paranoia_gene",
)
PHYSICAL_LOCI = (
    "radiation_resistance", "bone_density", "immune_strength",
    "longevity", "mars_adaptation", "creativity",
)
ALL_LOCI = STAT_LOCI + PHYSICAL_LOCI
NUM_LOCI = len(ALL_LOCI)

STAT_LOCUS_MAP = {
    "resolve_gene": "resolve",
    "improvisation_gene": "improvisation",
    "empathy_gene": "empathy",
    "hoarding_gene": "hoarding",
    "faith_gene": "faith",
    "paranoia_gene": "paranoia",
}

MUTATION_RATE = 0.02
MUTATION_SIGMA = 0.06
MAX_PHENOTYPE_MODIFIER = 0.08
CONDITION_THRESHOLD = 0.20

MAX_MUTATION_LOG = 50


# ---------------------------------------------------------------------------
# Genome
# ---------------------------------------------------------------------------

@dataclass
class Genome:
    """Diploid genome: 12 loci, 2 alleles each."""
    alleles: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.alleles:
            self.alleles = [(0.5, 0.5)] * NUM_LOCI

    def locus_value(self, index: int) -> float:
        """Mean of allele pair — the expressed value at a locus."""
        a, b = self.alleles[index]
        return (a + b) / 2.0

    def locus_heterozygosity(self, index: int) -> float:
        """Within-locus allele difference (0 = homozygous, 1 = max heterozygous)."""
        a, b = self.alleles[index]
        return abs(a - b)

    def to_dict(self) -> dict:
        return {
            "alleles": [[round(a, 4), round(b, 4)] for a, b in self.alleles],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Genome:
        if not d:
            return cls()
        raw = d.get("alleles", [])
        alleles = [(float(pair[0]), float(pair[1])) for pair in raw]
        if len(alleles) != NUM_LOCI:
            alleles = [(0.5, 0.5)] * NUM_LOCI
        return cls(alleles=alleles)


def _clamp01(v: float) -> float:
    """Clamp value to [0, 1]."""
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

def create_genome_from_stats(stats_dict: dict[str, float],
                             rng: _random_module.Random) -> Genome:
    """Create a genome correlated with existing stats.

    Stat loci are centered on the stat value with small noise.
    Physical loci are randomized around 0.5.  Alleles are split from
    the locus value so the mean equals the target.
    """
    alleles: list[tuple[float, float]] = []
    for locus_name in ALL_LOCI:
        stat_name = STAT_LOCUS_MAP.get(locus_name)
        if stat_name and stat_name in stats_dict:
            center = stats_dict[stat_name]
        else:
            center = 0.5
        spread = rng.uniform(0.02, 0.15)
        a = _clamp01(center + rng.gauss(0, 0.03) + spread / 2)
        b = _clamp01(center + rng.gauss(0, 0.03) - spread / 2)
        alleles.append((a, b))
    return Genome(alleles=alleles)


def create_random_genome(rng: _random_module.Random) -> Genome:
    """Create a fully random genome for immigrants."""
    alleles = [(_clamp01(rng.gauss(0.5, 0.15)),
                _clamp01(rng.gauss(0.5, 0.15))) for _ in range(NUM_LOCI)]
    return Genome(alleles=alleles)


# ---------------------------------------------------------------------------
# Reproduction
# ---------------------------------------------------------------------------

def form_gamete(genome: Genome, rng: _random_module.Random) -> list[float]:
    """Form a haploid gamete: choose one allele per locus."""
    return [rng.choice(pair) for pair in genome.alleles]


def combine_gametes(gamete_a: list[float], gamete_b: list[float],
                    rng: _random_module.Random) -> Genome:
    """Combine two gametes into a diploid genome with mutation."""
    alleles: list[tuple[float, float]] = []
    for a_val, b_val in zip(gamete_a, gamete_b):
        if rng.random() < MUTATION_RATE:
            a_val = _clamp01(a_val + rng.gauss(0, MUTATION_SIGMA))
        if rng.random() < MUTATION_RATE:
            b_val = _clamp01(b_val + rng.gauss(0, MUTATION_SIGMA))
        alleles.append((a_val, b_val))
    return Genome(alleles=alleles)


def reproduce(parent_a: Genome, parent_b: Genome,
              rng: _random_module.Random) -> Genome:
    """Produce offspring genome from two parents via gamete formation."""
    gamete_a = form_gamete(parent_a, rng)
    gamete_b = form_gamete(parent_b, rng)
    return combine_gametes(gamete_a, gamete_b, rng)


# ---------------------------------------------------------------------------
# Phenotype expression
# ---------------------------------------------------------------------------

def express_phenotype(genome: Genome) -> dict[str, float]:
    """Compute phenotype modifiers from genome.

    Returns stat modifiers (positive or negative, bounded by
    MAX_PHENOTYPE_MODIFIER) and physical trait values.
    Stat modifiers are centered so a locus value of 0.5 yields 0 modifier.
    """
    modifiers: dict[str, float] = {}
    for i, locus_name in enumerate(ALL_LOCI):
        value = genome.locus_value(i)
        stat_name = STAT_LOCUS_MAP.get(locus_name)
        if stat_name:
            modifier = (value - 0.5) * 2 * MAX_PHENOTYPE_MODIFIER
            modifiers[f"stat_{stat_name}"] = max(-MAX_PHENOTYPE_MODIFIER,
                                                  min(MAX_PHENOTYPE_MODIFIER, modifier))
        else:
            modifiers[locus_name] = value
    return modifiers


# ---------------------------------------------------------------------------
# Genetic conditions
# ---------------------------------------------------------------------------

GENETIC_CONDITIONS = {
    "radiation_sensitivity": {
        "locus": "radiation_resistance",
        "threshold": CONDITION_THRESHOLD,
        "description": "Both radiation resistance alleles are weak",
        "death_modifier": 1.15,
    },
    "brittle_bones": {
        "locus": "bone_density",
        "threshold": CONDITION_THRESHOLD,
        "description": "Low bone density under Mars gravity",
        "death_modifier": 1.10,
    },
    "immune_deficiency": {
        "locus": "immune_strength",
        "threshold": CONDITION_THRESHOLD,
        "description": "Weakened immune system in closed habitat",
        "death_modifier": 1.20,
    },
    "genetic_vigor": {
        "locus": "longevity",
        "threshold": 0.75,
        "description": "Both longevity alleles are strong",
        "death_modifier": 0.85,
        "invert": True,
    },
    "mars_native": {
        "locus": "mars_adaptation",
        "threshold": 0.70,
        "description": "Genetically adapted to Martian conditions",
        "death_modifier": 0.90,
        "invert": True,
    },
}


def compute_genetic_conditions(genome: Genome) -> list[dict[str, Any]]:
    """Detect active genetic conditions from genome.

    Standard conditions activate when BOTH alleles < threshold (recessive).
    Inverted conditions activate when BOTH alleles >= threshold (dominant beneficial).
    """
    active: list[dict[str, Any]] = []
    for cond_name, spec in GENETIC_CONDITIONS.items():
        locus_idx = ALL_LOCI.index(spec["locus"])
        a, b = genome.alleles[locus_idx]
        invert = spec.get("invert", False)
        if invert:
            if a >= spec["threshold"] and b >= spec["threshold"]:
                active.append({"name": cond_name, **spec})
        else:
            if a < spec["threshold"] and b < spec["threshold"]:
                active.append({"name": cond_name, **spec})
    return active


def compute_death_modifier(genome: Genome) -> float:
    """Compute combined death rate modifier from genetic conditions.

    Returns a multiplier (< 1.0 = healthier, > 1.0 = more vulnerable).
    """
    modifier = 1.0
    for cond in compute_genetic_conditions(genome):
        modifier *= cond["death_modifier"]
    longevity_idx = ALL_LOCI.index("longevity")
    longevity_val = genome.locus_value(longevity_idx)
    modifier *= 1.0 + (0.5 - longevity_val) * 0.1
    return max(0.5, min(2.0, modifier))


# ---------------------------------------------------------------------------
# Radiation mutations
# ---------------------------------------------------------------------------

def apply_radiation_mutation(genome: Genome, severity: float,
                             rng: _random_module.Random) -> list[dict]:
    """Apply radiation-induced mutations.  Higher severity = more mutations."""
    resistance_idx = ALL_LOCI.index("radiation_resistance")
    resistance = genome.locus_value(resistance_idx)
    mutation_chance = severity * 0.15 * (1.0 - resistance * 0.7)
    mutations: list[dict] = []
    for i in range(NUM_LOCI):
        if rng.random() < mutation_chance:
            slot = rng.randint(0, 1)
            old_val = genome.alleles[i][slot]
            delta = rng.gauss(0, severity * 0.1)
            new_val = _clamp01(old_val + delta)
            pair = list(genome.alleles[i])
            pair[slot] = new_val
            genome.alleles[i] = (pair[0], pair[1])
            mutations.append({
                "locus": ALL_LOCI[i], "slot": slot,
                "old": round(old_val, 4), "new": round(new_val, 4),
                "cause": "radiation",
            })
    return mutations


# ---------------------------------------------------------------------------
# Population metrics
# ---------------------------------------------------------------------------

@dataclass
class PopulationGenetics:
    """Population-level genetic metrics."""
    heterozygosity: float = 0.0
    diversity_index: float = 0.0
    mean_locus_values: list[float] = field(default_factory=list)
    locus_variance: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "heterozygosity": round(self.heterozygosity, 4),
            "diversity_index": round(self.diversity_index, 4),
            "mean_locus_values": [round(v, 4) for v in self.mean_locus_values],
            "locus_variance": [round(v, 6) for v in self.locus_variance],
        }


def compute_population_metrics(genomes: list[Genome]) -> PopulationGenetics:
    """Compute population-level genetic metrics.

    heterozygosity = mean within-individual allele difference across all loci
    diversity_index = mean across-population locus variance (genetic diversity)
    """
    if not genomes:
        return PopulationGenetics()

    n = len(genomes)
    het_sum = 0.0
    mean_values = [0.0] * NUM_LOCI
    all_values: list[list[float]] = [[] for _ in range(NUM_LOCI)]

    for g in genomes:
        for i in range(NUM_LOCI):
            het_sum += g.locus_heterozygosity(i)
            val = g.locus_value(i)
            mean_values[i] += val
            all_values[i].append(val)

    heterozygosity = het_sum / (n * NUM_LOCI)
    mean_values = [v / n for v in mean_values]

    variances = []
    for i in range(NUM_LOCI):
        if n < 2:
            variances.append(0.0)
        else:
            mean = mean_values[i]
            var = sum((v - mean) ** 2 for v in all_values[i]) / (n - 1)
            variances.append(var)

    diversity = sum(variances) / NUM_LOCI if variances else 0.0

    return PopulationGenetics(
        heterozygosity=heterozygosity,
        diversity_index=diversity,
        mean_locus_values=mean_values,
        locus_variance=variances,
    )


# ---------------------------------------------------------------------------
# State and tick
# ---------------------------------------------------------------------------

@dataclass
class GeneticsState:
    """Colony-wide genetics tracking."""
    diversity_history: list[float] = field(default_factory=list)
    heterozygosity_history: list[float] = field(default_factory=list)
    mutation_log: list[dict] = field(default_factory=list)
    total_mutations: int = 0
    generations_tracked: int = 0

    def to_dict(self) -> dict:
        return {
            "diversity_history": [round(v, 4) for v in self.diversity_history],
            "heterozygosity_history": [round(v, 4) for v in self.heterozygosity_history],
            "mutation_log": self.mutation_log[-MAX_MUTATION_LOG:],
            "total_mutations": self.total_mutations,
            "generations_tracked": self.generations_tracked,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GeneticsState:
        if not d:
            return cls()
        return cls(
            diversity_history=d.get("diversity_history", []),
            heterozygosity_history=d.get("heterozygosity_history", []),
            mutation_log=d.get("mutation_log", [])[-MAX_MUTATION_LOG:],
            total_mutations=d.get("total_mutations", 0),
            generations_tracked=d.get("generations_tracked", 0),
        )


@dataclass
class GeneticsTickResult:
    """Result of one year's genetics tick."""
    year: int = 0
    population_metrics: dict = field(default_factory=dict)
    conditions_by_colonist: dict[str, list[dict]] = field(default_factory=dict)
    radiation_mutations: list[dict] = field(default_factory=list)
    death_modifiers: dict[str, float] = field(default_factory=dict)
    phenotype_modifiers: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "population_metrics": self.population_metrics,
            "conditions_by_colonist": self.conditions_by_colonist,
            "radiation_mutations": self.radiation_mutations,
            "death_modifiers": {k: round(v, 4) for k, v in self.death_modifiers.items()},
            "phenotype_modifiers": {
                cid: {k: round(v, 4) for k, v in mods.items()}
                for cid, mods in self.phenotype_modifiers.items()
            },
        }


def tick_genetics(
    genetics_state: GeneticsState,
    colonist_genomes: dict[str, Genome],
    active_ids: list[str],
    year: int,
    radiation_severity: float,
    rng: _random_module.Random,
) -> GeneticsTickResult:
    """Run one year of genetics processing.

    1. Compute population metrics from active colonists
    2. Detect genetic conditions per colonist
    3. Apply radiation mutations if radiation event occurred
    4. Compute phenotype modifiers and death modifiers
    5. Update state history
    """
    result = GeneticsTickResult(year=year)

    active_genomes = [colonist_genomes[cid] for cid in active_ids
                      if cid in colonist_genomes]

    # Population metrics
    metrics = compute_population_metrics(active_genomes)
    result.population_metrics = metrics.to_dict()

    genetics_state.diversity_history.append(metrics.diversity_index)
    genetics_state.heterozygosity_history.append(metrics.heterozygosity)
    genetics_state.generations_tracked += 1

    # Per-colonist analysis
    for cid in active_ids:
        genome = colonist_genomes.get(cid)
        if genome is None:
            continue

        # Conditions
        conditions = compute_genetic_conditions(genome)
        if conditions:
            result.conditions_by_colonist[cid] = [
                {"name": c["name"], "description": c["description"]}
                for c in conditions
            ]

        # Death modifier
        result.death_modifiers[cid] = compute_death_modifier(genome)

        # Phenotype
        result.phenotype_modifiers[cid] = express_phenotype(genome)

    # Radiation mutations
    if radiation_severity > 0.3:
        for cid in active_ids:
            genome = colonist_genomes.get(cid)
            if genome is None:
                continue
            mutations = apply_radiation_mutation(genome, radiation_severity, rng)
            if mutations:
                for m in mutations:
                    m["colonist_id"] = cid
                    m["year"] = year
                result.radiation_mutations.extend(mutations)
                genetics_state.total_mutations += len(mutations)
                genetics_state.mutation_log.extend(mutations)

    # Trim log
    if len(genetics_state.mutation_log) > MAX_MUTATION_LOG:
        genetics_state.mutation_log = genetics_state.mutation_log[-MAX_MUTATION_LOG:]

    return result
