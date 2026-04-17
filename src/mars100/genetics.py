"""
Genetics organ for Mars-100 (engine v11.0).

Models heredity, mutation, and genetic drift in a small colony.
Each colonist has a diploid genome — pairs of alleles for each stat.
Children inherit one allele from each parent with crossover and mutation.
Genetic diversity and inbreeding are tracked colony-wide via pedigree.

Stats are heritable biology; skills are learned behavior (not genetic).
Element is inherited from one parent (simple dominance, not diploid).

RNG offset: seed + 12553
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import random as _random_module

STAT_NAMES = ("resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia")

MUTATION_RATE = 0.03
MUTATION_SIGMA = 0.05
INBREEDING_THRESHOLD = 0.125
INBREEDING_BIRTH_PENALTY = 0.5
INBREEDING_FITNESS_PENALTY = 0.002
HOMOZYGOSITY_FITNESS_COST = 0.001
MIN_DIVERSITY_WARNING = 0.3


@dataclass
class Genome:
    """Diploid genome — two alleles per stat trait, each 0.0-1.0."""
    alleles: dict[str, tuple[float, float]]

    def express(self) -> dict[str, float]:
        """Phenotype = simple mean of two alleles (no directional dominance)."""
        return {name: (a + b) / 2.0 for name, (a, b) in self.alleles.items()}

    def homozygosity(self) -> float:
        """Mean squared difference between alleles (0 = fully heterozygous)."""
        if not self.alleles:
            return 0.0
        total = sum((a - b) ** 2 for a, b in self.alleles.values())
        return 1.0 - total / len(self.alleles)

    def to_dict(self) -> dict:
        return {name: [round(a, 4), round(b, 4)]
                for name, (a, b) in self.alleles.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        alleles = {}
        for name, pair in d.items():
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                alleles[name] = (float(pair[0]), float(pair[1]))
        return cls(alleles=alleles)

    def clamp(self) -> None:
        """Clamp all alleles to [0, 1]."""
        for name, (a, b) in self.alleles.items():
            self.alleles[name] = (max(0.0, min(1.0, a)),
                                  max(0.0, min(1.0, b)))


@dataclass
class GeneticsTickResult:
    """Result of one year of genetic tracking."""
    diversity_index: float = 1.0
    avg_homozygosity: float = 0.0
    max_kinship: float = 0.0
    inbreeding_warnings: list[dict] = field(default_factory=list)
    mutations_this_year: int = 0
    generation: int = 0

    def to_dict(self) -> dict:
        return {
            "diversity_index": round(self.diversity_index, 4),
            "avg_homozygosity": round(self.avg_homozygosity, 4),
            "max_kinship": round(self.max_kinship, 4),
            "inbreeding_warnings": self.inbreeding_warnings,
            "mutations_this_year": self.mutations_this_year,
            "generation": self.generation,
        }


@dataclass
class GeneticsState:
    """Colony-wide genetic tracking state."""
    diversity_history: list[float] = field(default_factory=list)
    total_mutations: int = 0
    inbreeding_events: int = 0
    generation_count: int = 0
    lineage: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "diversity_history": [round(d, 4) for d in self.diversity_history[-20:]],
            "total_mutations": self.total_mutations,
            "inbreeding_events": self.inbreeding_events,
            "generation_count": self.generation_count,
            "lineage_size": len(self.lineage),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GeneticsState":
        return cls(
            diversity_history=d.get("diversity_history", []),
            total_mutations=d.get("total_mutations", 0),
            inbreeding_events=d.get("inbreeding_events", 0),
            generation_count=d.get("generation_count", 0),
            lineage={},
        )


def create_genome_from_stats(stats_dict: dict[str, float],
                             rng: _random_module.Random) -> Genome:
    """Create a diploid genome from existing stat values.

    Each stat value is split into two alleles with slight noise,
    so that express() ≈ original stat value.
    """
    alleles: dict[str, tuple[float, float]] = {}
    for name in STAT_NAMES:
        val = stats_dict.get(name, 0.5)
        noise = rng.gauss(0, 0.03)
        a = max(0.0, min(1.0, val + noise))
        b = max(0.0, min(1.0, val - noise))
        alleles[name] = (a, b)
    genome = Genome(alleles=alleles)
    genome.clamp()
    return genome


def crossover(parent_a: Genome, parent_b: Genome,
              rng: _random_module.Random) -> Genome:
    """Produce a child genome via meiosis.

    For each stat, the child inherits one allele from each parent.
    Which allele is chosen from each parent is random.
    """
    child_alleles: dict[str, tuple[float, float]] = {}
    for name in STAT_NAMES:
        a_pair = parent_a.alleles.get(name, (0.5, 0.5))
        b_pair = parent_b.alleles.get(name, (0.5, 0.5))
        from_a = a_pair[rng.randint(0, 1)]
        from_b = b_pair[rng.randint(0, 1)]
        child_alleles[name] = (from_a, from_b)
    return Genome(alleles=child_alleles)


def mutate(genome: Genome, rng: _random_module.Random,
           rate: float = MUTATION_RATE,
           sigma: float = MUTATION_SIGMA) -> int:
    """Apply random mutations to a genome in-place.

    Each allele has ``rate`` probability of mutating (gaussian shift).
    Returns count of mutations applied.
    """
    count = 0
    for name in list(genome.alleles.keys()):
        a, b = genome.alleles[name]
        if rng.random() < rate:
            a = max(0.0, min(1.0, a + rng.gauss(0, sigma)))
            count += 1
        if rng.random() < rate:
            b = max(0.0, min(1.0, b + rng.gauss(0, sigma)))
            count += 1
        genome.alleles[name] = (a, b)
    return count


def compute_pedigree_kinship(id_a: str, id_b: str,
                             lineage: dict[str, list[str]],
                             max_depth: int = 8) -> float:
    """Compute relatedness coefficient from pedigree (lineage tree).

    Uses BFS ancestor collection with generation distance.  Each individual
    is included as its own ancestor at distance 0 so that parent-child
    kinship is detected (the parent IS the shared ancestor).
    Founders (no lineage entry) are assumed unrelated.
    """
    def _ancestors(cid: str, max_d: int) -> dict[str, int]:
        """Collect ancestors with generation distance via BFS."""
        result: dict[str, int] = {cid: 0}
        queue: list[tuple[str, int]] = [(cid, 0)]
        visited: set[str] = {cid}
        while queue:
            current, dist = queue.pop(0)
            if dist >= max_d:
                continue
            for p in lineage.get(current, []):
                gen_dist = dist + 1
                if p not in result or gen_dist < result[p]:
                    result[p] = gen_dist
                if p not in visited:
                    visited.add(p)
                    queue.append((p, gen_dist))
        return result

    if id_a == id_b:
        return 1.0
    anc_a = _ancestors(id_a, max_depth)
    anc_b = _ancestors(id_b, max_depth)
    shared = set(anc_a.keys()) & set(anc_b.keys())
    if not shared:
        return 0.0
    kinship = sum(0.5 ** (anc_a[s] + anc_b[s]) for s in shared)
    return min(1.0, kinship)


def compute_diversity_index(genomes: list[Genome]) -> float:
    """Colony-wide genetic diversity based on allele heterozygosity.

    Measures mean expected heterozygosity across all loci and individuals.
    Returns 0.0 (monomorphic) to 1.0 (maximally diverse).
    """
    if len(genomes) < 2:
        return 0.0
    diversities: list[float] = []
    for name in STAT_NAMES:
        all_alleles: list[float] = []
        for g in genomes:
            a, b = g.alleles.get(name, (0.5, 0.5))
            all_alleles.extend([a, b])
        if not all_alleles:
            continue
        mean = sum(all_alleles) / len(all_alleles)
        variance = sum((x - mean) ** 2 for x in all_alleles) / len(all_alleles)
        diversities.append(min(1.0, variance * 4.0))
    return sum(diversities) / max(1, len(diversities))


def inbreeding_birth_modifier(kinship: float) -> float:
    """Reduce birth probability for closely related pairs.

    Returns a multiplier in [INBREEDING_BIRTH_PENALTY, 1.0].
    Pairs below INBREEDING_THRESHOLD are not penalized.
    """
    if kinship < INBREEDING_THRESHOLD:
        return 1.0
    penalty = 1.0 - (kinship - INBREEDING_THRESHOLD) * 2.0
    return max(INBREEDING_BIRTH_PENALTY, min(1.0, penalty))


def genetic_death_modifier(genome: Genome | None,
                           kinship_to_parents: float = 0.0) -> float:
    """Compute death rate multiplier from genetic factors.

    High homozygosity and inbreeding slightly increase death rate.
    Returns a multiplier >= 1.0.
    """
    if genome is None:
        return 1.0
    homo_cost = genome.homozygosity() * HOMOZYGOSITY_FITNESS_COST
    inbreeding_cost = max(0.0, kinship_to_parents - INBREEDING_THRESHOLD) * INBREEDING_FITNESS_PENALTY
    return 1.0 + homo_cost + inbreeding_cost


def record_birth(state: GeneticsState, child_id: str,
                 parent_ids: list[str]) -> None:
    """Record a birth in the pedigree lineage."""
    state.lineage[child_id] = list(parent_ids)
    depth = _lineage_depth(child_id, state.lineage)
    if depth > state.generation_count:
        state.generation_count = depth


def _lineage_depth(cid: str, lineage: dict[str, list[str]],
                   seen: set | None = None) -> int:
    """Compute generation depth of a colonist."""
    if seen is None:
        seen = set()
    if cid in seen:
        return 0
    seen.add(cid)
    parents = lineage.get(cid, [])
    if not parents:
        return 0
    return 1 + max(_lineage_depth(p, lineage, seen) for p in parents)


def tick_genetics(
    state: GeneticsState,
    genomes: dict[str, Genome],
    active_ids: list[str],
    year: int,
    rng: _random_module.Random,
) -> GeneticsTickResult:
    """Yearly genetics update: recompute diversity, check inbreeding.

    Called after births/deaths/immigrants are processed.
    """
    result = GeneticsTickResult()
    active_genomes = [genomes[cid] for cid in active_ids if cid in genomes]

    result.diversity_index = compute_diversity_index(active_genomes)
    state.diversity_history.append(result.diversity_index)
    if len(state.diversity_history) > 100:
        state.diversity_history = state.diversity_history[-100:]

    if active_genomes:
        result.avg_homozygosity = (
            sum(g.homozygosity() for g in active_genomes) / len(active_genomes))

    max_kin = 0.0
    for i, id_a in enumerate(active_ids):
        for id_b in active_ids[i + 1:]:
            kin = compute_pedigree_kinship(id_a, id_b, state.lineage)
            if kin > max_kin:
                max_kin = kin
    result.max_kinship = max_kin

    if result.diversity_index < MIN_DIVERSITY_WARNING:
        result.inbreeding_warnings.append({
            "year": year,
            "diversity": round(result.diversity_index, 4),
            "message": "Colony genetic diversity critically low",
        })

    for i, id_a in enumerate(active_ids):
        for id_b in active_ids[i + 1:]:
            kin = compute_pedigree_kinship(id_a, id_b, state.lineage)
            if kin >= INBREEDING_THRESHOLD:
                state.inbreeding_events += 1
                result.inbreeding_warnings.append({
                    "year": year,
                    "pair": [id_a, id_b],
                    "kinship": round(kin, 4),
                })

    result.generation = state.generation_count
    return result
