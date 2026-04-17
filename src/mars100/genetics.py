"""
Genetics organ for Mars-100 (engine v11.0).

Models heredity, alleles, genetic drift, inbreeding, and founder effects
for a small Mars colony.  A 10-person founder bottleneck is the most
genetically precarious scenario short of total isolation.

Each colonist carries a diploid genome: 6 loci, 2 alleles per locus.
Trait expression = mean of alleles (additive / codominant).
Children inherit via Mendelian crossover + small mutation.
Inbreeding coefficient computed from pedigree (up to 3 generations).

One-year lag: LAST year's colony diversity drives THIS year's
epidemic vulnerability and fitness modifiers.

RNG offset: seed + 12553
"""
from __future__ import annotations

import math
import random as _random_module
from dataclasses import dataclass, field
from typing import Any


# --- Loci and constants ---------------------------------------------------

LOCUS_NAMES = (
    "metabolism",         # energy efficiency
    "immunity",           # disease resistance
    "radiation_resistance",  # rad tolerance
    "cognitive",          # learning speed
    "longevity",          # lifespan modifier
    "fertility",          # reproductive fitness
)

ALLELE_MIN = 0.0
ALLELE_MAX = 1.0
MUTATION_RATE = 0.05         # probability of mutation per allele per birth
MUTATION_SIGMA = 0.08        # gaussian noise when mutation fires
CROSSOVER_RATE = 0.3         # probability of crossover between loci

# Fitness effect magnitudes
IMMUNITY_DEATH_MULT_LOW = 1.25     # death rate multiplier when immunity < 0.3
IMMUNITY_DEATH_MULT_HIGH = 0.90    # death rate multiplier when immunity > 0.7
RAD_RESIST_DAMAGE_FACTOR = 0.6     # fraction of rad damage absorbed at max
LONGEVITY_DEATH_MULT_LOW = 1.15
LONGEVITY_DEATH_MULT_HIGH = 0.85
FERTILITY_BONUS = 0.12             # added to birth prob when fertility > 0.7
FERTILITY_PENALTY = -0.06          # added to birth prob when fertility < 0.3
COGNITIVE_SKILL_BONUS = 0.015      # extra skill gain per year per 0.1 cognitive
METABOLISM_FOOD_MULT = 0.02        # food consumption modifier per 0.1 metabolism deviation

# Colony-wide diversity
DIVERSITY_EPIDEMIC_THRESHOLD = 0.25  # below this heterozygosity, epidemic risk rises
EPIDEMIC_VULNERABILITY_MAX = 0.15    # max added death rate from low diversity
DIVERSITY_WARNING_THRESHOLD = 0.30   # warn when heterozygosity drops below this


# --- Data types -----------------------------------------------------------

@dataclass
class Allele:
    """One allele value, a float in [0, 1]."""
    value: float = 0.5

    def to_dict(self) -> float:
        return round(self.value, 6)

    @classmethod
    def from_dict(cls, v: float | dict) -> Allele:
        if isinstance(v, dict):
            return cls(value=v.get("value", 0.5))
        return cls(value=float(v))


@dataclass
class Locus:
    """A diploid locus: two alleles."""
    a: Allele = field(default_factory=Allele)
    b: Allele = field(default_factory=Allele)

    def expression(self) -> float:
        """Phenotypic expression (codominant = mean)."""
        return (self.a.value + self.b.value) / 2.0

    def heterozygosity(self) -> float:
        """Absolute difference between alleles; higher = more heterozygous."""
        return abs(self.a.value - self.b.value)

    def to_dict(self) -> dict:
        return {"a": self.a.to_dict(), "b": self.b.to_dict(),
                "expression": round(self.expression(), 6)}

    @classmethod
    def from_dict(cls, d: dict) -> Locus:
        return cls(a=Allele.from_dict(d["a"]), b=Allele.from_dict(d["b"]))


@dataclass
class Genome:
    """A colonist's diploid genome across all loci."""
    loci: dict[str, Locus] = field(default_factory=dict)
    parent_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in LOCUS_NAMES:
            if name not in self.loci:
                self.loci[name] = Locus()

    def expression(self, locus_name: str) -> float:
        """Get phenotypic expression for a specific locus."""
        return self.loci[locus_name].expression()

    def mean_heterozygosity(self) -> float:
        """Average heterozygosity across all loci."""
        if not self.loci:
            return 0.0
        return sum(l.heterozygosity() for l in self.loci.values()) / len(self.loci)

    def to_dict(self) -> dict:
        return {
            "loci": {name: locus.to_dict() for name, locus in self.loci.items()},
            "parent_ids": list(self.parent_ids),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Genome:
        loci = {name: Locus.from_dict(ld) for name, ld in d.get("loci", {}).items()}
        return cls(loci=loci, parent_ids=d.get("parent_ids", []))


@dataclass
class GeneticsState:
    """Colony-wide genetics state (tracked per year)."""
    colony_heterozygosity: float = 0.5
    inbreeding_events: int = 0
    epidemic_vulnerability: float = 0.0
    diversity_warnings: list[str] = field(default_factory=list)
    mutation_log: list[dict] = field(default_factory=list)
    mean_fitness: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "colony_heterozygosity": round(self.colony_heterozygosity, 6),
            "inbreeding_events": self.inbreeding_events,
            "epidemic_vulnerability": round(self.epidemic_vulnerability, 6),
            "diversity_warnings": list(self.diversity_warnings),
            "mutation_log": list(self.mutation_log[-20:]),
            "mean_fitness": {k: round(v, 6) for k, v in self.mean_fitness.items()},
        }


@dataclass
class GeneticsYearContext:
    """Inputs for the genetics tick."""
    year: int
    active_genomes: list[Genome]
    active_ids: list[str]

    def to_dict(self) -> dict:
        return {"year": self.year, "active_count": len(self.active_genomes)}


@dataclass
class GeneticsTickResult:
    """Output from one year of genetics computation."""
    colony_heterozygosity: float = 0.5
    epidemic_vulnerability: float = 0.0
    fitness_modifiers: dict[str, float] = field(default_factory=dict)
    diversity_warnings: list[str] = field(default_factory=list)
    mutations_this_year: int = 0

    def to_dict(self) -> dict:
        return {
            "colony_heterozygosity": round(self.colony_heterozygosity, 6),
            "epidemic_vulnerability": round(self.epidemic_vulnerability, 6),
            "fitness_modifiers": {k: round(v, 6) for k, v in self.fitness_modifiers.items()},
            "diversity_warnings": list(self.diversity_warnings),
            "mutations_this_year": self.mutations_this_year,
        }


# --- Pedigree & Inbreeding ------------------------------------------------

class Pedigree:
    """Tracks parent-child relationships for kinship computation.

    Stores mapping: colonist_id -> (parent_a_id, parent_b_id).
    Founders have no parents (empty tuple).
    """

    def __init__(self) -> None:
        self._parents: dict[str, tuple[str, str]] = {}

    def register_founder(self, colonist_id: str) -> None:
        """Register a founder (no parents in the pedigree)."""
        self._parents[colonist_id] = ("", "")

    def register_child(self, child_id: str, parent_a: str, parent_b: str) -> None:
        """Register a child with two parents."""
        self._parents[child_id] = (parent_a, parent_b)

    def get_parents(self, colonist_id: str) -> tuple[str, str]:
        """Return (parent_a, parent_b) or ('', '') for founders/unknown."""
        return self._parents.get(colonist_id, ("", ""))

    def kinship(self, id_a: str, id_b: str, max_depth: int = 3) -> float:
        """Compute kinship coefficient between two individuals.

        Uses recursive Wright's path coefficient method, limited to
        max_depth generations to avoid exponential blowup.  Returns
        a value in [0.0, 1.0] where 0.5 = full siblings, 0.25 = half
        siblings, 0.125 = cousins, etc.
        """
        if id_a == id_b:
            return 1.0
        return self._kinship_recursive(id_a, id_b, max_depth, {})

    def _kinship_recursive(self, id_a: str, id_b: str, depth: int,
                           cache: dict[tuple[str, str], float]) -> float:
        """Recursive kinship with memoization."""
        if depth <= 0:
            return 0.0
        key = (min(id_a, id_b), max(id_a, id_b))
        if key in cache:
            return cache[key]
        if id_a == id_b:
            cache[key] = 1.0
            return 1.0
        pa_a, pb_a = self.get_parents(id_a)
        pa_b, pb_b = self.get_parents(id_b)
        # If either has no parents (founder), no shared ancestry
        if not pa_a or not pa_b:
            cache[key] = 0.0
            return 0.0
        # Kinship = 0.25 * (K(pa_a, pa_b) + K(pa_a, pb_b) + K(pb_a, pa_b) + K(pb_a, pb_b))
        k = 0.25 * (
            self._kinship_recursive(pa_a, pa_b, depth - 1, cache) +
            self._kinship_recursive(pa_a, pb_b, depth - 1, cache) +
            self._kinship_recursive(pb_a, pa_b, depth - 1, cache) +
            self._kinship_recursive(pb_a, pb_b, depth - 1, cache)
        )
        cache[key] = k
        return k

    def inbreeding_coefficient(self, child_id: str) -> float:
        """Compute inbreeding coefficient F for a child.

        F = kinship(parent_a, parent_b).  For outbred individuals F=0.
        """
        pa, pb = self.get_parents(child_id)
        if not pa or not pb:
            return 0.0
        return self.kinship(pa, pb)

    def to_dict(self) -> dict:
        return dict(self._parents)

    @classmethod
    def from_dict(cls, d: dict) -> Pedigree:
        p = cls()
        for cid, parents in d.items():
            if isinstance(parents, (list, tuple)) and len(parents) == 2:
                p._parents[cid] = (str(parents[0]), str(parents[1]))
            else:
                p._parents[cid] = ("", "")
        return p


# --- Genome creation and inheritance --------------------------------------

def create_founder_genome(colonist_id: str, rng: _random_module.Random) -> Genome:
    """Create a heterozygous genome for a founding colonist.

    Founders get alleles drawn from a broad distribution to maximize
    initial diversity — the colony's genetic future depends on it.
    """
    loci: dict[str, Locus] = {}
    for name in LOCUS_NAMES:
        a_val = max(ALLELE_MIN, min(ALLELE_MAX, rng.gauss(0.5, 0.2)))
        b_val = max(ALLELE_MIN, min(ALLELE_MAX, rng.gauss(0.5, 0.2)))
        loci[name] = Locus(a=Allele(a_val), b=Allele(b_val))
    return Genome(loci=loci, parent_ids=[])


def create_immigrant_genome(rng: _random_module.Random) -> Genome:
    """Create a genome for an Earth immigrant (fresh genetic material)."""
    loci: dict[str, Locus] = {}
    for name in LOCUS_NAMES:
        a_val = max(ALLELE_MIN, min(ALLELE_MAX, rng.gauss(0.55, 0.18)))
        b_val = max(ALLELE_MIN, min(ALLELE_MAX, rng.gauss(0.55, 0.18)))
        loci[name] = Locus(a=Allele(a_val), b=Allele(b_val))
    return Genome(loci=loci, parent_ids=[])


def inherit_genome(parent_a: Genome, parent_b: Genome,
                   parent_a_id: str, parent_b_id: str,
                   rng: _random_module.Random) -> tuple[Genome, list[dict]]:
    """Create a child genome from two parents via Mendelian inheritance.

    Returns (child_genome, mutation_log_entries).
    Each locus: child gets one allele from each parent (random choice).
    Crossover: with probability CROSSOVER_RATE, alleles swap between
    adjacent loci.  Mutation: with probability MUTATION_RATE per allele,
    gaussian noise is added.
    """
    child_loci: dict[str, Locus] = {}
    mutations: list[dict] = []

    # Collect allele pairs from parents
    parent_a_gamete: list[float] = []
    parent_b_gamete: list[float] = []
    for name in LOCUS_NAMES:
        la = parent_a.loci.get(name, Locus())
        lb = parent_b.loci.get(name, Locus())
        # Meiosis: pick one allele from each parent
        a_allele = rng.choice([la.a.value, la.b.value])
        b_allele = rng.choice([lb.a.value, lb.b.value])
        parent_a_gamete.append(a_allele)
        parent_b_gamete.append(b_allele)

    # Crossover: swap segments between gametes
    if len(LOCUS_NAMES) > 1 and rng.random() < CROSSOVER_RATE:
        crossover_point = rng.randint(1, len(LOCUS_NAMES) - 1)
        parent_a_gamete[crossover_point:], parent_b_gamete[crossover_point:] = (
            parent_b_gamete[crossover_point:], parent_a_gamete[crossover_point:])

    # Assemble child loci with possible mutation
    for i, name in enumerate(LOCUS_NAMES):
        a_val = parent_a_gamete[i]
        b_val = parent_b_gamete[i]
        # Mutation on allele from parent A
        if rng.random() < MUTATION_RATE:
            delta = rng.gauss(0, MUTATION_SIGMA)
            a_val = max(ALLELE_MIN, min(ALLELE_MAX, a_val + delta))
            mutations.append({"locus": name, "allele": "a",
                              "delta": round(delta, 6)})
        # Mutation on allele from parent B
        if rng.random() < MUTATION_RATE:
            delta = rng.gauss(0, MUTATION_SIGMA)
            b_val = max(ALLELE_MIN, min(ALLELE_MAX, b_val + delta))
            mutations.append({"locus": name, "allele": "b",
                              "delta": round(delta, 6)})
        child_loci[name] = Locus(a=Allele(a_val), b=Allele(b_val))

    child = Genome(loci=child_loci, parent_ids=[parent_a_id, parent_b_id])
    return child, mutations


# --- Fitness computation --------------------------------------------------

def compute_individual_fitness(genome: Genome) -> dict[str, float]:
    """Compute fitness modifiers from an individual's genome.

    Returns a dict of named modifiers that downstream systems use:
    - death_rate_mult: multiplier on base death rate
    - birth_prob_bonus: additive bonus to birth probability
    - skill_learning_rate: multiplier on skill gain
    - food_consumption_mult: multiplier on food consumption
    - rad_damage_mult: multiplier on radiation event damage
    """
    immunity = genome.expression("immunity")
    rad_resist = genome.expression("radiation_resistance")
    longevity = genome.expression("longevity")
    fertility = genome.expression("fertility")
    cognitive = genome.expression("cognitive")
    metabolism = genome.expression("metabolism")

    # Death rate multiplier from immunity + longevity
    death_mult = 1.0
    if immunity < 0.3:
        death_mult *= IMMUNITY_DEATH_MULT_LOW
    elif immunity > 0.7:
        death_mult *= IMMUNITY_DEATH_MULT_HIGH
    if longevity < 0.3:
        death_mult *= LONGEVITY_DEATH_MULT_LOW
    elif longevity > 0.7:
        death_mult *= LONGEVITY_DEATH_MULT_HIGH

    # Birth probability bonus
    birth_bonus = 0.0
    if fertility > 0.7:
        birth_bonus = FERTILITY_BONUS
    elif fertility < 0.3:
        birth_bonus = FERTILITY_PENALTY

    # Skill learning rate
    skill_rate = 1.0 + (cognitive - 0.5) * COGNITIVE_SKILL_BONUS * 10

    # Food consumption modifier (high metabolism = more food needed)
    food_mult = 1.0 + (metabolism - 0.5) * METABOLISM_FOOD_MULT * 10

    # Radiation damage multiplier
    rad_mult = 1.0 - rad_resist * RAD_RESIST_DAMAGE_FACTOR

    return {
        "death_rate_mult": round(max(0.5, min(2.0, death_mult)), 6),
        "birth_prob_bonus": round(max(-0.1, min(0.2, birth_bonus)), 6),
        "skill_learning_rate": round(max(0.5, min(1.5, skill_rate)), 6),
        "food_consumption_mult": round(max(0.8, min(1.2, food_mult)), 6),
        "rad_damage_mult": round(max(0.4, min(1.0, rad_mult)), 6),
    }


# --- Colony-wide diversity metrics ----------------------------------------

def compute_colony_heterozygosity(genomes: list[Genome]) -> float:
    """Compute mean heterozygosity across the colony.

    Uses expected heterozygosity (gene diversity): for each locus,
    compute allele frequencies and He = 1 - sum(p_i^2).
    Then average across loci.
    """
    if not genomes:
        return 0.0
    he_per_locus: list[float] = []
    for name in LOCUS_NAMES:
        alleles: list[float] = []
        for g in genomes:
            locus = g.loci.get(name, Locus())
            alleles.extend([locus.a.value, locus.b.value])
        if not alleles:
            he_per_locus.append(0.0)
            continue
        # Bin alleles into 10 frequency classes for He calculation
        n_bins = 10
        bins = [0] * n_bins
        for val in alleles:
            idx = min(n_bins - 1, int(val * n_bins))
            bins[idx] += 1
        total = len(alleles)
        freq_sq_sum = sum((count / total) ** 2 for count in bins)
        he = 1.0 - freq_sq_sum
        he_per_locus.append(he)
    return sum(he_per_locus) / len(he_per_locus) if he_per_locus else 0.0


def compute_epidemic_vulnerability(heterozygosity: float) -> float:
    """Compute epidemic vulnerability from colony heterozygosity.

    Low diversity = high vulnerability.  Above threshold = 0.
    """
    if heterozygosity >= DIVERSITY_EPIDEMIC_THRESHOLD:
        return 0.0
    shortfall = DIVERSITY_EPIDEMIC_THRESHOLD - heterozygosity
    normalized = shortfall / DIVERSITY_EPIDEMIC_THRESHOLD
    return min(EPIDEMIC_VULNERABILITY_MAX, normalized * EPIDEMIC_VULNERABILITY_MAX)


def compute_diversity_warnings(heterozygosity: float, year: int,
                               inbreeding_events: int) -> list[str]:
    """Generate warnings about genetic diversity issues."""
    warnings: list[str] = []
    if heterozygosity < DIVERSITY_WARNING_THRESHOLD:
        warnings.append(
            f"Year {year}: Colony heterozygosity ({heterozygosity:.3f}) "
            f"below warning threshold ({DIVERSITY_WARNING_THRESHOLD})")
    if heterozygosity < DIVERSITY_EPIDEMIC_THRESHOLD:
        warnings.append(
            f"Year {year}: CRITICAL — heterozygosity ({heterozygosity:.3f}) "
            f"below epidemic threshold ({DIVERSITY_EPIDEMIC_THRESHOLD})")
    if inbreeding_events > 0:
        warnings.append(
            f"Year {year}: {inbreeding_events} inbreeding event(s) detected "
            f"in colony history")
    return warnings


# --- Main tick function ---------------------------------------------------

def tick_genetics(state: GeneticsState, ctx: GeneticsYearContext,
                  rng: _random_module.Random) -> GeneticsTickResult:
    """Advance genetics by one year.

    Computes colony-wide diversity metrics and per-colonist fitness.
    Does NOT handle births — that's driven by the engine; this organ
    only measures and reports.
    """
    genomes = ctx.active_genomes
    het = compute_colony_heterozygosity(genomes)
    state.colony_heterozygosity = het

    vuln = compute_epidemic_vulnerability(het)
    state.epidemic_vulnerability = vuln

    warnings = compute_diversity_warnings(het, ctx.year, state.inbreeding_events)
    state.diversity_warnings = warnings

    # Compute mean fitness across colony
    fitness_sums: dict[str, float] = {}
    fitness_count = 0
    for g in genomes:
        fit = compute_individual_fitness(g)
        for k, v in fit.items():
            fitness_sums[k] = fitness_sums.get(k, 0.0) + v
        fitness_count += 1
    state.mean_fitness = {
        k: v / max(1, fitness_count) for k, v in fitness_sums.items()
    }

    return GeneticsTickResult(
        colony_heterozygosity=het,
        epidemic_vulnerability=vuln,
        fitness_modifiers=dict(state.mean_fitness),
        diversity_warnings=warnings,
        mutations_this_year=len(state.mutation_log),
    )


def compute_genetic_death_modifier(genome: Genome | None,
                                   epidemic_vulnerability: float) -> float:
    """Compute combined death rate multiplier from genetics.

    Combines individual genetic fitness with colony-wide epidemic risk.
    Returns a multiplier on the base death rate (1.0 = no change).
    """
    mult = 1.0
    if genome is not None:
        fitness = compute_individual_fitness(genome)
        mult *= fitness["death_rate_mult"]
    mult += epidemic_vulnerability
    return max(0.5, min(3.0, mult))


def compute_nature_genetic_bonus(genomes: list[Genome]) -> dict[str, float]:
    """Compute resource bonuses from colony-wide genetic fitness.

    High mean metabolism → slightly less food needed colony-wide.
    High mean immunity → less medicine needed.
    Returns resource modifiers (multipliers around 1.0).
    """
    if not genomes:
        return {}
    mean_metabolism = sum(g.expression("metabolism") for g in genomes) / len(genomes)
    mean_immunity = sum(g.expression("immunity") for g in genomes) / len(genomes)
    return {
        "food_maintenance_mult": round(1.0 + (0.5 - mean_metabolism) * 0.1, 6),
        "medicine_maintenance_mult": round(1.0 + (0.5 - mean_immunity) * 0.1, 6),
    }
