"""
Genetics organ for Mars-100 (engine v11.0).

Models population genetics: diploid alleles, Mendelian inheritance,
mutation, expected heterozygosity, kinship/inbreeding detection, and
Mars adaptation pressure.

No crossover or chromosome model — each stat locus is independent.
Colony-level fitness bonuses use a one-year lag; genome state updates
(births, deaths, diversity) are immediate.

RNG offset: seed + 13217
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import STAT_NAMES

# ── Tuning constants ─────────────────────────────────────────────────────

MUTATION_RATE: float = 0.02          # per allele per generation
MUTATION_MAGNITUDE: float = 0.08     # std-dev of mutation effect
INBREEDING_COEFFICIENT_WARN: float = 0.125  # 1st-cousin level
INBREEDING_DEATH_MULT: float = 1.25  # death rate multiplier from inbreeding
DIVERSITY_FITNESS_FLOOR: float = 0.3 # heterozygosity below this → penalty
MAX_FOOD_BONUS: float = 0.008       # max food bonus from adaptation
MAX_DEATH_REDUCTION: float = 0.002  # max death-rate reduction from fitness
MAX_ADAPTATION_RATE: float = 0.012  # adaptation index change per year
EXPRESSION_NOISE: float = 0.03      # environmental noise on phenotype

# Mars-optimal stat profile (what Mars selects FOR)
MARS_TARGET: dict[str, float] = {
    "resolve": 0.75,
    "improvisation": 0.65,
    "empathy": 0.50,
    "hoarding": 0.40,
    "faith": 0.35,
    "paranoia": 0.30,
}


# ── Data types ────────────────────────────────────────────────────────────

@dataclass
class Genome:
    """Diploid genome: two alleles per stat locus."""
    alleles: dict[str, tuple[float, float]] = field(default_factory=dict)
    parent_ids: tuple[str | None, str | None] = (None, None)
    generation: int = 0
    mars_fitness: float = 0.5

    def express(self) -> dict[str, float]:
        """Phenotype: mean of alleles per locus."""
        return {
            stat: (a + b) / 2.0
            for stat, (a, b) in self.alleles.items()
        }

    def to_dict(self) -> dict:
        return {
            "alleles": {k: list(v) for k, v in self.alleles.items()},
            "parent_ids": list(self.parent_ids),
            "generation": self.generation,
            "mars_fitness": round(self.mars_fitness, 6),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Genome:
        alleles = {k: (v[0], v[1]) for k, v in d.get("alleles", {}).items()}
        pids = d.get("parent_ids", [None, None])
        return cls(
            alleles=alleles,
            parent_ids=(pids[0] if len(pids) > 0 else None,
                        pids[1] if len(pids) > 1 else None),
            generation=d.get("generation", 0),
            mars_fitness=d.get("mars_fitness", 0.5),
        )


@dataclass
class GeneticsState:
    """Colony-wide genetics state."""
    genomes: dict[str, Genome] = field(default_factory=dict)
    diversity_index: float = 1.0
    adaptation_index: float = 0.0
    mutations_log: list[dict] = field(default_factory=list)
    generation_counts: dict[int, int] = field(default_factory=dict)
    inbreeding_warnings: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "genomes": {k: v.to_dict() for k, v in self.genomes.items()},
            "diversity_index": round(self.diversity_index, 6),
            "adaptation_index": round(self.adaptation_index, 6),
            "mutations_total": len(self.mutations_log),
            "generation_counts": self.generation_counts,
            "inbreeding_warnings": len(self.inbreeding_warnings),
        }

    @classmethod
    def from_dict(cls, d: dict) -> GeneticsState:
        genomes = {k: Genome.from_dict(v) for k, v in d.get("genomes", {}).items()}
        return cls(
            genomes=genomes,
            diversity_index=d.get("diversity_index", 1.0),
            adaptation_index=d.get("adaptation_index", 0.0),
        )


@dataclass
class GeneticsTickResult:
    """Result of one year of genetics processing."""
    diversity_index: float = 1.0
    adaptation_index: float = 0.0
    mutations: list[dict] = field(default_factory=list)
    inbreeding_warnings: list[dict] = field(default_factory=list)
    resource_modifiers: dict[str, float] = field(default_factory=dict)
    offspring_frailty: dict[str, float] = field(default_factory=dict)
    selection_pressures: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "diversity_index": round(self.diversity_index, 6),
            "adaptation_index": round(self.adaptation_index, 6),
            "mutations": self.mutations,
            "inbreeding_warnings": self.inbreeding_warnings,
            "resource_modifiers": {k: round(v, 6)
                                   for k, v in self.resource_modifiers.items()},
            "offspring_frailty": {k: round(v, 6)
                                  for k, v in self.offspring_frailty.items()},
            "selection_pressures": {k: round(v, 6)
                                    for k, v in self.selection_pressures.items()},
        }


# ── Genome creation ──────────────────────────────────────────────────────

def create_genome_from_stats(colonist_id: str,
                             stats: dict[str, float],
                             rng: Any) -> Genome:
    """Bootstrap a genome from existing colonist stats.

    Each stat becomes two alleles that average to (approximately) the
    current stat value plus a tiny noise term.
    """
    alleles: dict[str, tuple[float, float]] = {}
    for stat in STAT_NAMES:
        base = stats.get(stat, 0.5)
        spread = rng.uniform(0.02, 0.12)
        a1 = _clamp(base + spread / 2 + rng.gauss(0, 0.01))
        a2 = _clamp(base - spread / 2 + rng.gauss(0, 0.01))
        alleles[stat] = (a1, a2)
    return Genome(alleles=alleles, parent_ids=(None, None), generation=0)


def create_immigrant_genome(colonist_id: str,
                            stats: dict[str, float],
                            rng: Any) -> Genome:
    """Create a genome for an Earth immigrant.

    Immigrants are generation 0 (unrelated to colony lineage) and add
    fresh alleles to the gene pool.
    """
    genome = create_genome_from_stats(colonist_id, stats, rng)
    genome.generation = 0
    return genome


# ── Inheritance ───────────────────────────────────────────────────────────

def inherit_genome(parent_a_id: str, parent_b_id: str,
                   genome_a: Genome, genome_b: Genome,
                   child_id: str, rng: Any) -> Genome:
    """Mendelian inheritance: one allele from each parent, with mutation.

    For each stat locus, the child receives one randomly chosen allele
    from each parent.  Each allele has a MUTATION_RATE chance of being
    mutated by a gaussian offset.
    """
    child_alleles: dict[str, tuple[float, float]] = {}
    mutations: list[dict] = []

    for stat in STAT_NAMES:
        # Pick one allele from each parent
        a_alleles = genome_a.alleles.get(stat, (0.5, 0.5))
        b_alleles = genome_b.alleles.get(stat, (0.5, 0.5))
        from_a = rng.choice(a_alleles)
        from_b = rng.choice(b_alleles)

        # Mutation
        if rng.random() < MUTATION_RATE:
            delta = rng.gauss(0, MUTATION_MAGNITUDE)
            from_a = _clamp(from_a + delta)
            mutations.append({"child": child_id, "stat": stat,
                              "allele": "a", "delta": round(delta, 6)})
        if rng.random() < MUTATION_RATE:
            delta = rng.gauss(0, MUTATION_MAGNITUDE)
            from_b = _clamp(from_b + delta)
            mutations.append({"child": child_id, "stat": stat,
                              "allele": "b", "delta": round(delta, 6)})

        child_alleles[stat] = (from_a, from_b)

    gen = max(genome_a.generation, genome_b.generation) + 1
    genome = Genome(
        alleles=child_alleles,
        parent_ids=(parent_a_id, parent_b_id),
        generation=gen,
    )
    return genome, mutations


# ── Kinship & inbreeding ──────────────────────────────────────────────────

def kinship_coefficient(id_a: str, id_b: str,
                        genomes: dict[str, Genome],
                        max_depth: int = 4) -> float:
    """Compute kinship coefficient by common-ancestor depth search.

    Returns 0.0 for unrelated individuals, 0.25 for siblings,
    0.125 for first cousins, etc.  Bounded by max_depth to keep
    computation tractable.
    """
    ancestors_a = _collect_ancestors(id_a, genomes, max_depth)
    ancestors_b = _collect_ancestors(id_b, genomes, max_depth)

    if not ancestors_a or not ancestors_b:
        return 0.0

    shared = set(ancestors_a.keys()) & set(ancestors_b.keys())
    if not shared:
        return 0.0

    coeff = 0.0
    for ancestor_id in shared:
        depth_a = ancestors_a[ancestor_id]
        depth_b = ancestors_b[ancestor_id]
        coeff += 0.5 ** (depth_a + depth_b)

    return min(1.0, coeff)


def _collect_ancestors(colonist_id: str, genomes: dict[str, Genome],
                       max_depth: int) -> dict[str, int]:
    """BFS to collect ancestor IDs with their depth."""
    ancestors: dict[str, int] = {}
    queue: list[tuple[str, int]] = [(colonist_id, 0)]

    while queue:
        current_id, depth = queue.pop(0)
        if depth > max_depth:
            continue
        genome = genomes.get(current_id)
        if genome is None:
            continue
        for pid in genome.parent_ids:
            if pid is not None and pid not in ancestors:
                ancestors[pid] = depth + 1
                queue.append((pid, depth + 1))

    return ancestors


def check_inbreeding(child_id: str, parent_a_id: str, parent_b_id: str,
                     genomes: dict[str, Genome]) -> dict | None:
    """Check if parents are too closely related."""
    coeff = kinship_coefficient(parent_a_id, parent_b_id, genomes)
    if coeff >= INBREEDING_COEFFICIENT_WARN:
        return {
            "child_id": child_id,
            "parent_a": parent_a_id,
            "parent_b": parent_b_id,
            "kinship_coefficient": round(coeff, 4),
            "severity": "high" if coeff >= 0.25 else "moderate",
        }
    return None


# ── Expected heterozygosity ──────────────────────────────────────────────

def expected_heterozygosity(genomes: dict[str, Genome],
                            active_ids: list[str]) -> float:
    """Compute mean expected heterozygosity across all loci.

    For each locus, compute allele frequencies across the population,
    then H_e = 1 - sum(p_i^2).  Average across all loci.
    """
    active_genomes = [genomes[cid] for cid in active_ids if cid in genomes]
    if len(active_genomes) < 2:
        return 1.0  # too few to measure

    locus_het: list[float] = []
    for stat in STAT_NAMES:
        # Collect all alleles at this locus, bucket into bins of width 0.05
        all_alleles: list[float] = []
        for g in active_genomes:
            pair = g.alleles.get(stat, (0.5, 0.5))
            all_alleles.extend(pair)

        if not all_alleles:
            locus_het.append(1.0)
            continue

        # Bucket into 20 bins of width 0.05
        n_bins = 20
        counts = [0] * n_bins
        for val in all_alleles:
            idx = min(n_bins - 1, int(val * n_bins))
            counts[idx] += 1

        total = len(all_alleles)
        sum_p_sq = sum((c / total) ** 2 for c in counts if c > 0)
        locus_het.append(1.0 - sum_p_sq)

    return sum(locus_het) / len(locus_het) if locus_het else 1.0


# ── Mars adaptation ──────────────────────────────────────────────────────

def compute_adaptation(genomes: dict[str, Genome],
                       active_ids: list[str]) -> float:
    """How close the population's mean phenotype is to the Mars-optimal
    target profile.  Returns 0.0 (no match) to 1.0 (perfect match)."""
    active_genomes = [genomes[cid] for cid in active_ids if cid in genomes]
    if not active_genomes:
        return 0.0

    mean_phenotype: dict[str, float] = {s: 0.0 for s in STAT_NAMES}
    for g in active_genomes:
        expr = g.express()
        for s in STAT_NAMES:
            mean_phenotype[s] += expr.get(s, 0.5)
    for s in STAT_NAMES:
        mean_phenotype[s] /= len(active_genomes)

    # Euclidean distance from target, normalized to [0, 1]
    max_dist = math.sqrt(len(STAT_NAMES))  # worst case: all stats off by 1.0
    dist = math.sqrt(sum(
        (mean_phenotype[s] - MARS_TARGET[s]) ** 2
        for s in STAT_NAMES
    ))
    return max(0.0, 1.0 - dist / max_dist)


def compute_selection_pressures(genomes: dict[str, Genome],
                                active_ids: list[str]) -> dict[str, float]:
    """Compute per-stat selection pressure: how far the population mean
    is from the Mars-optimal target.  Positive = need more, negative = have too much."""
    active_genomes = [genomes[cid] for cid in active_ids if cid in genomes]
    if not active_genomes:
        return {s: 0.0 for s in STAT_NAMES}

    mean_phenotype: dict[str, float] = {s: 0.0 for s in STAT_NAMES}
    for g in active_genomes:
        expr = g.express()
        for s in STAT_NAMES:
            mean_phenotype[s] += expr.get(s, 0.5)
    for s in STAT_NAMES:
        mean_phenotype[s] /= len(active_genomes)

    return {s: round(MARS_TARGET[s] - mean_phenotype[s], 6)
            for s in STAT_NAMES}


# ── Mars fitness per genome ───────────────────────────────────────────────

def compute_mars_fitness(genome: Genome) -> float:
    """Individual Mars fitness: closeness of phenotype to target."""
    expr = genome.express()
    dist = math.sqrt(sum(
        (expr.get(s, 0.5) - MARS_TARGET[s]) ** 2
        for s in STAT_NAMES
    ))
    max_dist = math.sqrt(len(STAT_NAMES))
    return max(0.0, 1.0 - dist / max_dist)


# ── Resource modifiers ────────────────────────────────────────────────────

def compute_resource_modifiers(state: GeneticsState) -> dict[str, float]:
    """Colony-level resource modifiers from genetic fitness.

    High diversity and adaptation provide small bonuses.
    Low diversity incurs a death-rate penalty.
    """
    mods: dict[str, float] = {}

    # Adaptation → food bonus
    food_bonus = state.adaptation_index * MAX_FOOD_BONUS
    mods["food"] = 1.0 + food_bonus

    # Diversity → death rate modifier
    if state.diversity_index < DIVERSITY_FITNESS_FLOOR:
        penalty = (DIVERSITY_FITNESS_FLOOR - state.diversity_index) / DIVERSITY_FITNESS_FLOOR
        mods["death_rate_mult"] = 1.0 + penalty * (INBREEDING_DEATH_MULT - 1.0)
    else:
        # High diversity → slight death rate reduction
        bonus = min(MAX_DEATH_REDUCTION,
                    (state.diversity_index - DIVERSITY_FITNESS_FLOOR) * 0.005)
        mods["death_rate_mult"] = max(0.95, 1.0 - bonus)

    return mods


# ── Main tick ─────────────────────────────────────────────────────────────

def tick_genetics(state: GeneticsState,
                  births: list[dict],
                  deaths: list[dict],
                  colonists: list[Any],
                  active_ids: list[str],
                  year: int,
                  rng: Any) -> GeneticsTickResult:
    """Advance genetics by one year.

    1. Ensure all colonists have genomes (bootstrap founders/immigrants).
    2. Process births: inherit genomes, check inbreeding.
    3. Process deaths: remove from active gene pool tracking.
    4. Recompute diversity, adaptation, selection pressures.
    5. Update per-genome Mars fitness.
    6. Compute resource modifiers (uses PREVIOUS year's adaptation — caller
       stages these for next year via pending_genetics_mods pattern).
    """
    result = GeneticsTickResult()
    year_mutations: list[dict] = []

    # 1. Bootstrap genomes for any colonist missing one
    for c in colonists:
        cid = c.id if hasattr(c, "id") else c.get("id", "")
        if cid and cid not in state.genomes:
            stats = (c.stats.to_dict() if hasattr(c, "stats")
                     else c.get("stats", {}))
            if isinstance(stats, dict):
                genome = create_genome_from_stats(cid, stats, rng)
            else:
                genome = create_genome_from_stats(cid, stats.to_dict(), rng)
            state.genomes[cid] = genome

    # 2. Process births — inherit genomes
    for birth in births:
        child_id = birth.get("id", "")
        parents = birth.get("parents", [])
        if len(parents) >= 2 and child_id:
            pa_id, pb_id = parents[0], parents[1]
            ga = state.genomes.get(pa_id)
            gb = state.genomes.get(pb_id)
            if ga and gb:
                child_genome, muts = inherit_genome(
                    pa_id, pb_id, ga, gb, child_id, rng)
                state.genomes[child_id] = child_genome
                year_mutations.extend(muts)

                # Check inbreeding
                warning = check_inbreeding(child_id, pa_id, pb_id,
                                           state.genomes)
                if warning:
                    warning["year"] = year
                    result.inbreeding_warnings.append(warning)
                    state.inbreeding_warnings.append(warning)
                    # Offspring frailty: inbred children have higher death risk
                    coeff = warning["kinship_coefficient"]
                    result.offspring_frailty[child_id] = coeff

    # 3. Update generation counts
    gen_counts: dict[int, int] = {}
    for cid in active_ids:
        g = state.genomes.get(cid)
        if g:
            gen_counts[g.generation] = gen_counts.get(g.generation, 0) + 1
    state.generation_counts = gen_counts

    # 4. Recompute population metrics
    state.diversity_index = expected_heterozygosity(state.genomes, active_ids)
    new_adaptation = compute_adaptation(state.genomes, active_ids)
    # Smooth adaptation change
    delta = new_adaptation - state.adaptation_index
    state.adaptation_index += max(-MAX_ADAPTATION_RATE,
                                  min(MAX_ADAPTATION_RATE, delta))
    state.adaptation_index = max(0.0, min(1.0, state.adaptation_index))

    # 5. Update per-genome fitness
    for cid in active_ids:
        g = state.genomes.get(cid)
        if g:
            g.mars_fitness = compute_mars_fitness(g)

    # 6. Compute result
    result.diversity_index = state.diversity_index
    result.adaptation_index = state.adaptation_index
    result.mutations = year_mutations
    result.resource_modifiers = compute_resource_modifiers(state)
    result.selection_pressures = compute_selection_pressures(
        state.genomes, active_ids)

    # Log mutations
    state.mutations_log.extend(year_mutations)
    # Trim log to last 200 entries
    if len(state.mutations_log) > 200:
        state.mutations_log = state.mutations_log[-200:]

    return result


# ── Helpers ───────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, v))
