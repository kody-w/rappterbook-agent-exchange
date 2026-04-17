"""
Genetics organ for Mars-100 (engine v11.0).

Models hereditable traits via simplified diploid genetics.
Each colonist carries a genome of allelic loci that influence stat
expression and skill learning rates.  Children inherit via Mendelian
per-locus selection + point mutation.

Tracks colony-wide genetic diversity (allele variance), pedigree-based
relatedness, and emergent hereditary traits with hysteresis.

RNG offset: seed + 13001
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
import random as _random_module

# --- locus names -----------------------------------------------------------

STAT_LOCI = ("resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia")
SKILL_LOCI = ("terraforming_apt", "hydroponics_apt", "mediation_apt",
              "coding_apt", "prayer_apt", "sabotage_apt")
ALL_LOCI = STAT_LOCI + SKILL_LOCI

SKILL_LOCUS_TO_SKILL = {
    "terraforming_apt": "terraforming",
    "hydroponics_apt": "hydroponics",
    "mediation_apt": "mediation",
    "coding_apt": "coding",
    "prayer_apt": "prayer",
    "sabotage_apt": "sabotage",
}

# --- constants --------------------------------------------------------------

BASE_MUTATION_RATE = 0.02
RADIATION_MUTATION_BOOST = 0.06
MUTATION_SIGMA = 0.08
SKILL_LEARNING_MULT_MIN = 0.5
SKILL_LEARNING_MULT_MAX = 2.0
NATURE_NURTURE_BLEND = 0.6
DIVERSITY_BONUS_THRESHOLD = 0.12
DIVERSITY_PENALTY_THRESHOLD = 0.04
DIVERSITY_DEATH_BONUS = 0.9
DIVERSITY_DEATH_PENALTY = 1.15
DIVERSITY_SKILL_BONUS = 0.05
MIN_ACTIVE_FOR_DIVERSITY = 3


@dataclass(frozen=True)
class TraitDef:
    """Definition of a hereditary trait with hysteresis thresholds."""
    name: str
    locus: str
    acquire_threshold: float
    lose_threshold: float
    high: bool
    description: str


TRAIT_DEFS: tuple[TraitDef, ...] = (
    TraitDef("resilient", "resolve", 0.72, 0.60, True,
             "Hardened against adversity -- lower death risk"),
    TraitDef("fragile", "resolve", 0.22, 0.32, False,
             "Constitution weakened -- higher death risk"),
    TraitDef("green_thumb", "terraforming_apt", 0.65, 0.50, True,
             "Natural affinity for growing things -- farming bonus"),
    TraitDef("empathic_bond", "empathy", 0.75, 0.60, True,
             "Deep emotional attunement -- cohesion bonus"),
    TraitDef("innovation_spark", "coding_apt", 0.65, 0.50, True,
             "Born problem-solver -- research speed bonus"),
    TraitDef("paranoid_gene", "paranoia", 0.78, 0.65, True,
             "Genetic predisposition to distrust -- trust penalty"),
    TraitDef("faithful_soul", "faith", 0.80, 0.65, True,
             "Deep spiritual resonance -- morale buffer"),
    TraitDef("hoarder_instinct", "hoarding", 0.75, 0.60, True,
             "Compulsive resource accumulation -- trade penalty"),
)


@dataclass
class Locus:
    """Diploid locus with two allele values in [0, 1]."""
    allele_a: float = 0.5
    allele_b: float = 0.5

    def expression(self) -> float:
        """Phenotypic expression: co-dominance (mean)."""
        return (self.allele_a + self.allele_b) / 2.0

    def to_dict(self) -> dict[str, float]:
        return {"a": round(self.allele_a, 4), "b": round(self.allele_b, 4)}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> Locus:
        return cls(allele_a=d.get("a", 0.5), allele_b=d.get("b", 0.5))

    def clamp(self) -> None:
        self.allele_a = max(0.0, min(1.0, self.allele_a))
        self.allele_b = max(0.0, min(1.0, self.allele_b))


@dataclass
class Genome:
    """Complete genome for a colonist."""
    loci: dict[str, Locus] = field(default_factory=dict)
    parent_a_id: str | None = None
    parent_b_id: str | None = None
    generation: int = 0
    traits: list[str] = field(default_factory=list)

    def expression(self, locus_name: str) -> float:
        loc = self.loci.get(locus_name)
        return loc.expression() if loc else 0.5

    def all_expressions(self) -> dict[str, float]:
        return {name: loc.expression() for name, loc in self.loci.items()}

    def heterozygosity(self) -> float:
        if not self.loci:
            return 0.0
        diffs = [abs(loc.allele_a - loc.allele_b) for loc in self.loci.values()]
        return sum(diffs) / len(diffs)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "loci": {name: loc.to_dict() for name, loc in self.loci.items()},
            "generation": self.generation,
            "traits": list(self.traits),
        }
        if self.parent_a_id is not None:
            d["parent_a_id"] = self.parent_a_id
        if self.parent_b_id is not None:
            d["parent_b_id"] = self.parent_b_id
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Genome:
        loci = {name: Locus.from_dict(v) for name, v in d.get("loci", {}).items()}
        return cls(
            loci=loci,
            parent_a_id=d.get("parent_a_id"),
            parent_b_id=d.get("parent_b_id"),
            generation=d.get("generation", 0),
            traits=list(d.get("traits", [])),
        )


@dataclass
class GeneticsTickResult:
    """Output of one genetics tick."""
    diversity_index: float = 0.0
    mean_generation: float = 0.0
    trait_counts: dict[str, int] = field(default_factory=dict)
    inbreeding_pairs: int = 0
    mutations_this_year: int = 0
    allele_drift: dict[str, float] = field(default_factory=dict)
    death_rate_modifier: float = 1.0
    skill_learning_modifiers: dict[str, float] = field(default_factory=dict)
    cohesion_modifier: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "diversity_index": round(self.diversity_index, 4),
            "mean_generation": round(self.mean_generation, 2),
            "trait_counts": dict(self.trait_counts),
            "inbreeding_pairs": self.inbreeding_pairs,
            "mutations_this_year": self.mutations_this_year,
            "allele_drift": {k: round(v, 4) for k, v in self.allele_drift.items()},
            "death_rate_modifier": round(self.death_rate_modifier, 4),
            "skill_learning_modifiers": {
                k: round(v, 4) for k, v in self.skill_learning_modifiers.items()
            },
            "cohesion_modifier": round(self.cohesion_modifier, 4),
        }


def create_founding_genome(
    stat_values: dict[str, float],
    skill_values: dict[str, float],
    rng: _random_module.Random,
) -> Genome:
    """Create a founding colonist genome from their initial stats/skills."""
    loci: dict[str, Locus] = {}
    for stat_name in STAT_LOCI:
        base = stat_values.get(stat_name, 0.5)
        a = max(0.0, min(1.0, base + rng.gauss(0, 0.05)))
        b = max(0.0, min(1.0, base + rng.gauss(0, 0.05)))
        loci[stat_name] = Locus(allele_a=a, allele_b=b)
    for skill_locus in SKILL_LOCI:
        skill_name = SKILL_LOCUS_TO_SKILL[skill_locus]
        base = skill_values.get(skill_name, 0.0) * 0.5
        a = max(0.0, min(1.0, base + rng.gauss(0, 0.05)))
        b = max(0.0, min(1.0, base + rng.gauss(0, 0.05)))
        loci[skill_locus] = Locus(allele_a=a, allele_b=b)
    genome = Genome(loci=loci, generation=0)
    genome.traits = determine_traits(genome, existing_traits=[])
    return genome


def create_immigrant_genome(rng: _random_module.Random) -> Genome:
    """Create a genome for an immigrant from Earth (diverse gene pool)."""
    loci: dict[str, Locus] = {}
    for name in ALL_LOCI:
        a = max(0.0, min(1.0, rng.gauss(0.5, 0.15)))
        b = max(0.0, min(1.0, rng.gauss(0.5, 0.15)))
        loci[name] = Locus(allele_a=a, allele_b=b)
    genome = Genome(loci=loci, generation=0)
    genome.traits = determine_traits(genome, existing_traits=[])
    return genome


def inherit_genome(
    parent_a: Genome,
    parent_b: Genome,
    parent_a_id: str,
    parent_b_id: str,
    rng: _random_module.Random,
    radiation_event: bool = False,
) -> tuple[Genome, int]:
    """Create a child genome via Mendelian per-locus inheritance.

    For each locus: randomly pick one allele from parent A's locus,
    one from parent B's locus.  Point-mutate at BASE_MUTATION_RATE.
    Returns (child_genome, mutation_count).
    """
    mutation_rate = BASE_MUTATION_RATE
    if radiation_event:
        mutation_rate += RADIATION_MUTATION_BOOST
    mutations = 0
    child_loci: dict[str, Locus] = {}
    for name in ALL_LOCI:
        pa = parent_a.loci.get(name, Locus())
        pb = parent_b.loci.get(name, Locus())
        a = rng.choice([pa.allele_a, pa.allele_b])
        b = rng.choice([pb.allele_a, pb.allele_b])
        if rng.random() < mutation_rate:
            a = max(0.0, min(1.0, a + rng.gauss(0, MUTATION_SIGMA)))
            mutations += 1
        if rng.random() < mutation_rate:
            b = max(0.0, min(1.0, b + rng.gauss(0, MUTATION_SIGMA)))
            mutations += 1
        child_loci[name] = Locus(allele_a=a, allele_b=b)
    gen = max(parent_a.generation, parent_b.generation) + 1
    child = Genome(
        loci=child_loci,
        parent_a_id=parent_a_id,
        parent_b_id=parent_b_id,
        generation=gen,
    )
    child.traits = determine_traits(child, existing_traits=[])
    return child, mutations


def determine_traits(genome: Genome, existing_traits: list[str]) -> list[str]:
    """Evaluate trait conditions with hysteresis."""
    new_traits: list[str] = []
    for td in TRAIT_DEFS:
        expr = genome.expression(td.locus)
        has_trait = td.name in existing_traits
        if td.high:
            if has_trait:
                keep = expr >= td.lose_threshold
            else:
                keep = expr >= td.acquire_threshold
        else:
            if has_trait:
                keep = expr <= td.lose_threshold
            else:
                keep = expr <= td.acquire_threshold
        if keep:
            new_traits.append(td.name)
    return new_traits


def compute_diversity_index(genomes: list[Genome]) -> float:
    """Compute allele-variance-based diversity across active colonists."""
    if len(genomes) < MIN_ACTIVE_FOR_DIVERSITY:
        return 0.0
    per_locus_variance: list[float] = []
    for name in ALL_LOCI:
        all_alleles: list[float] = []
        for g in genomes:
            loc = g.loci.get(name)
            if loc:
                all_alleles.extend([loc.allele_a, loc.allele_b])
        if len(all_alleles) < 2:
            continue
        mean = sum(all_alleles) / len(all_alleles)
        variance = sum((v - mean) ** 2 for v in all_alleles) / len(all_alleles)
        per_locus_variance.append(variance)
    if not per_locus_variance:
        return 0.0
    return sum(per_locus_variance) / len(per_locus_variance)


def compute_relatedness(genome_a: Genome, genome_b: Genome,
                        genome_map: dict[str, Genome]) -> float:
    """Compute pedigree-based relatedness (2-generation check)."""
    if genome_a.parent_a_id is None or genome_b.parent_a_id is None:
        return 0.0
    parents_a = {genome_a.parent_a_id, genome_a.parent_b_id} - {None}
    parents_b = {genome_b.parent_a_id, genome_b.parent_b_id} - {None}
    shared = parents_a & parents_b
    if len(shared) == 2:
        return 0.5
    if len(shared) == 1:
        return 0.25
    return 0.0


def count_inbreeding_pairs(
    active_ids: list[str],
    genome_map: dict[str, Genome],
) -> int:
    """Count pairs with relatedness >= 0.25 among active colonists."""
    count = 0
    for i, id_a in enumerate(active_ids):
        ga = genome_map.get(id_a)
        if ga is None:
            continue
        for id_b in active_ids[i + 1:]:
            gb = genome_map.get(id_b)
            if gb is None:
                continue
            if compute_relatedness(ga, gb, genome_map) >= 0.25:
                count += 1
    return count


def compute_allele_drift(genomes: list[Genome]) -> dict[str, float]:
    """Compute mean allele expression per locus across all active genomes."""
    if not genomes:
        return {}
    drift: dict[str, float] = {}
    for name in ALL_LOCI:
        values: list[float] = []
        for g in genomes:
            loc = g.loci.get(name)
            if loc:
                values.append(loc.expression())
        drift[name] = sum(values) / len(values) if values else 0.5
    return drift


def compute_trait_death_modifier(genome: Genome) -> float:
    """Per-colonist death rate modifier from genetic traits."""
    mult = 1.0
    if "resilient" in genome.traits:
        mult *= 0.85
    if "fragile" in genome.traits:
        mult *= 1.2
    return mult


def compute_skill_learning_modifier(genome: Genome) -> dict[str, float]:
    """Per-colonist skill learning rate modifiers from aptitude loci."""
    mods: dict[str, float] = {}
    for locus_name, skill_name in SKILL_LOCUS_TO_SKILL.items():
        expr = genome.expression(locus_name)
        mult = SKILL_LEARNING_MULT_MIN + (
            (SKILL_LEARNING_MULT_MAX - SKILL_LEARNING_MULT_MIN) * expr
        )
        mods[skill_name] = mult
    return mods


def compute_trait_cohesion_modifier(genome: Genome) -> float:
    """Cohesion modifier from traits."""
    mod = 0.0
    if "empathic_bond" in genome.traits:
        mod += 0.02
    if "paranoid_gene" in genome.traits:
        mod -= 0.03
    return mod


def tick_genetics(
    genome_map: dict[str, Genome],
    active_ids: list[str],
    year: int,
    rng: _random_module.Random,
) -> GeneticsTickResult:
    """Compute colony-wide genetic state for one year.

    Recomputed fresh each year from living colonists.
    """
    result = GeneticsTickResult()
    active_genomes = [genome_map[cid] for cid in active_ids if cid in genome_map]
    if not active_genomes:
        return result

    result.diversity_index = compute_diversity_index(active_genomes)
    result.mean_generation = (
        sum(g.generation for g in active_genomes) / len(active_genomes)
    )

    for g in active_genomes:
        for trait in g.traits:
            result.trait_counts[trait] = result.trait_counts.get(trait, 0) + 1

    result.inbreeding_pairs = count_inbreeding_pairs(active_ids, genome_map)
    result.allele_drift = compute_allele_drift(active_genomes)

    if result.diversity_index >= DIVERSITY_BONUS_THRESHOLD:
        result.death_rate_modifier = DIVERSITY_DEATH_BONUS
        for skill in SKILL_LOCUS_TO_SKILL.values():
            result.skill_learning_modifiers[skill] = 1.0 + DIVERSITY_SKILL_BONUS
    elif result.diversity_index <= DIVERSITY_PENALTY_THRESHOLD:
        result.death_rate_modifier = DIVERSITY_DEATH_PENALTY
        for skill in SKILL_LOCUS_TO_SKILL.values():
            result.skill_learning_modifiers[skill] = 1.0 - DIVERSITY_SKILL_BONUS
    else:
        result.death_rate_modifier = 1.0
        for skill in SKILL_LOCUS_TO_SKILL.values():
            result.skill_learning_modifiers[skill] = 1.0

    empathic_count = result.trait_counts.get("empathic_bond", 0)
    paranoid_count = result.trait_counts.get("paranoid_gene", 0)
    pop = len(active_genomes)
    result.cohesion_modifier = (empathic_count - paranoid_count) * 0.01 / max(1, pop)

    return result


def blend_child_stats(
    genetic_expression: dict[str, float],
    parent_avg_stats: dict[str, float],
    rng: _random_module.Random,
) -> dict[str, float]:
    """Blend genetic expression with parent average for child starting stats."""
    result: dict[str, float] = {}
    for stat_name in STAT_LOCI:
        genetic = genetic_expression.get(stat_name, 0.5)
        nurture = parent_avg_stats.get(stat_name, 0.5)
        blended = NATURE_NURTURE_BLEND * genetic + (1 - NATURE_NURTURE_BLEND) * nurture
        blended += rng.gauss(0, 0.03)
        result[stat_name] = max(0.0, min(1.0, blended))
    return result
