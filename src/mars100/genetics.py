"""Genetics organ (v11.0) — hereditary traits, diversity, inbreeding,
radiation-driven mutation, and emergent adaptations across generations.

Each colonist carries a *GeneticProfile* of eight loci.  Allele values
drift through crossover, mutation, and environmental pressure.  Colony-
level metrics (diversity, avg fitness) feed back into psychology and
death checks.

RNG stream: seed + 12553.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# ── constants ──────────────────────────────────────────────────────────

GENETIC_TRAITS: tuple[str, ...] = (
    "stress_resilience",
    "learning_rate",
    "bone_density",
    "radiation_tolerance",
    "oxygen_efficiency",
    "social_bonding",
    "metabolic_efficiency",
    "immune_response",
)

_ALLELE_MIN = -0.15
_ALLELE_MAX = 0.15
_FITNESS_MIN = 0.5
_FITNESS_MAX = 1.5
_KINSHIP_BLOCK_THRESHOLD = 0.25
_MUTATION_BASE_RATE = 0.03
_MUTATION_MAGNITUDE = 0.04
_ADAPTATION_THRESHOLD = 0.10
_CONDITION_THRESHOLD = -0.10


# ── dataclasses ────────────────────────────────────────────────────────

@dataclass
class GeneticProfile:
    """Per-colonist genetic state."""
    alleles: dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in GENETIC_TRAITS})
    ancestor_ids: set[str] = field(default_factory=set)
    birth_year: int = 0
    parent_ids: tuple[str, str] | None = None
    adaptations: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)

    def clamp(self) -> None:
        for t in GENETIC_TRAITS:
            self.alleles[t] = max(_ALLELE_MIN, min(_ALLELE_MAX, self.alleles.get(t, 0.0)))

    def to_dict(self) -> dict:
        return {
            "alleles": dict(self.alleles),
            "ancestor_ids": sorted(self.ancestor_ids),
            "birth_year": self.birth_year,
            "parent_ids": list(self.parent_ids) if self.parent_ids else None,
            "adaptations": list(self.adaptations),
            "conditions": list(self.conditions),
        }

    @classmethod
    def from_dict(cls, d: dict) -> GeneticProfile:
        alleles = {t: d.get("alleles", {}).get(t, 0.0) for t in GENETIC_TRAITS}
        ancestor_ids = set(d.get("ancestor_ids", []))
        parent_ids = tuple(d["parent_ids"]) if d.get("parent_ids") else None
        p = cls(
            alleles=alleles,
            ancestor_ids=ancestor_ids,
            birth_year=d.get("birth_year", 0),
            parent_ids=parent_ids,
            adaptations=list(d.get("adaptations", [])),
            conditions=list(d.get("conditions", [])),
        )
        p.clamp()
        return p


@dataclass
class GeneticsState:
    """Colony-wide genetics state."""
    profiles: dict[str, GeneticProfile] = field(default_factory=dict)
    colony_diversity: float = 1.0
    avg_fitness: float = 1.0
    total_mutations: int = 0
    total_conditions: int = 0

    def clamp(self) -> None:
        self.colony_diversity = max(0.0, min(1.0, self.colony_diversity))
        self.avg_fitness = max(_FITNESS_MIN, min(_FITNESS_MAX, self.avg_fitness))
        self.total_mutations = max(0, self.total_mutations)
        self.total_conditions = max(0, self.total_conditions)
        for p in self.profiles.values():
            p.clamp()

    def to_dict(self) -> dict:
        return {
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()},
            "colony_diversity": round(self.colony_diversity, 4),
            "avg_fitness": round(self.avg_fitness, 4),
            "total_mutations": self.total_mutations,
            "total_conditions": self.total_conditions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GeneticsState:
        profiles = {k: GeneticProfile.from_dict(v) for k, v in d.get("profiles", {}).items()}
        s = cls(
            profiles=profiles,
            colony_diversity=d.get("colony_diversity", 1.0),
            avg_fitness=d.get("avg_fitness", 1.0),
            total_mutations=d.get("total_mutations", 0),
            total_conditions=d.get("total_conditions", 0),
        )
        s.clamp()
        return s


@dataclass
class GeneticsTickResult:
    """Output of one year of genetic evolution."""
    mutations: list[dict] = field(default_factory=list)
    new_adaptations: list[dict] = field(default_factory=list)
    new_conditions: list[dict] = field(default_factory=list)
    diversity_index: float = 1.0
    avg_fitness: float = 1.0

    def to_dict(self) -> dict:
        return {
            "mutations": self.mutations,
            "new_adaptations": self.new_adaptations,
            "new_conditions": self.new_conditions,
            "diversity_index": round(self.diversity_index, 4),
            "avg_fitness": round(self.avg_fitness, 4),
        }


# ── profile creation ───────────────────────────────────────────────────

def _random_alleles(rng: random.Random) -> dict[str, float]:
    return {t: max(_ALLELE_MIN, min(_ALLELE_MAX, rng.gauss(0.0, 0.05))) for t in GENETIC_TRAITS}


def create_founding_genetics(
    active_ids: list[str], rng: random.Random,
) -> dict[str, GeneticProfile]:
    """Create genetic profiles for the original colonists."""
    profiles: dict[str, GeneticProfile] = {}
    for cid in active_ids:
        profiles[cid] = GeneticProfile(
            alleles=_random_alleles(rng),
            ancestor_ids={cid},
            birth_year=0,
            parent_ids=None,
        )
        profiles[cid].clamp()
    return profiles


def create_immigrant_genetics(
    imm_id: str, rng: random.Random,
) -> GeneticProfile:
    """Create a genetic profile for an immigrant (fresh lineage)."""
    p = GeneticProfile(
        alleles=_random_alleles(rng),
        ancestor_ids={imm_id},
        birth_year=0,
        parent_ids=None,
    )
    p.clamp()
    return p


def inherit_genetics(
    parent_a_id: str, parent_b_id: str,
    ga: GeneticProfile, gb: GeneticProfile,
    child_id: str, year: int,
    rng: random.Random,
) -> GeneticProfile:
    """Create a child profile by crossing two parents."""
    alleles: dict[str, float] = {}
    for t in GENETIC_TRAITS:
        if rng.random() < 0.5:
            alleles[t] = ga.alleles.get(t, 0.0)
        else:
            alleles[t] = gb.alleles.get(t, 0.0)
        # Small random drift at birth
        alleles[t] += rng.gauss(0.0, 0.01)
        alleles[t] = max(_ALLELE_MIN, min(_ALLELE_MAX, alleles[t]))

    ancestors = ga.ancestor_ids | gb.ancestor_ids | {child_id, parent_a_id, parent_b_id}
    p = GeneticProfile(
        alleles=alleles,
        ancestor_ids=ancestors,
        birth_year=year,
        parent_ids=(parent_a_id, parent_b_id),
    )
    p.clamp()
    return p


# ── kinship and inbreeding ────────────────────────────────────────────

def compute_kinship(
    a_id: str, b_id: str,
    ga: GeneticProfile, gb: GeneticProfile,
) -> float:
    """Compute kinship coefficient from shared ancestry (0-1).

    Uses Jaccard similarity of ancestor sets, excluding self-IDs.
    """
    a_anc = ga.ancestor_ids - {a_id}
    b_anc = gb.ancestor_ids - {b_id}
    if not a_anc and not b_anc:
        return 0.0
    union = a_anc | b_anc
    if not union:
        return 0.0
    return len(a_anc & b_anc) / len(union)


def is_pair_blocked(
    a_id: str, b_id: str,
    ga: GeneticProfile, gb: GeneticProfile,
) -> bool:
    """Return True if this pair should be blocked from breeding.

    Blocks: parent-child, siblings, or kinship above threshold.
    """
    # Parent-child
    if gb.parent_ids and a_id in gb.parent_ids:
        return True
    if ga.parent_ids and b_id in ga.parent_ids:
        return True
    if ga.parent_ids and gb.parent_ids and ga.parent_ids == gb.parent_ids:
        return True  # Siblings
    return compute_kinship(a_id, b_id, ga, gb) > _KINSHIP_BLOCK_THRESHOLD


# ── fitness and modifiers ──────────────────────────────────────────────

def _compute_fitness(profile: GeneticProfile) -> float:
    """Aggregate fitness from allele values (0.5-1.5)."""
    total = sum(profile.alleles.get(t, 0.0) for t in GENETIC_TRAITS)
    fitness = 1.0 + total
    return max(_FITNESS_MIN, min(_FITNESS_MAX, fitness))


def compute_genetic_death_modifier(profile: GeneticProfile) -> float:
    """Multiply base death rate. <1 = more survivable, >1 = more fragile.

    Uses bone_density, immune_response, and metabolic_efficiency.
    """
    bd = profile.alleles.get("bone_density", 0.0)
    ir = profile.alleles.get("immune_response", 0.0)
    me = profile.alleles.get("metabolic_efficiency", 0.0)
    modifier = 1.0 - (bd + ir + me) * 2.0
    condition_penalty = len(profile.conditions) * 0.1
    modifier += condition_penalty
    return max(0.3, min(2.0, modifier))


def compute_genetic_stress_reduction(profile: GeneticProfile) -> float:
    """How much baseline stress is reduced per year (0-0.1)."""
    sr = profile.alleles.get("stress_resilience", 0.0)
    sb = profile.alleles.get("social_bonding", 0.0)
    reduction = (sr + sb) * 0.3
    return max(0.0, min(0.1, reduction))


def compute_genetic_radiation_modifier(profile: GeneticProfile) -> float:
    """Multiplier on radiation damage. <1 = more tolerant."""
    rt = profile.alleles.get("radiation_tolerance", 0.0)
    modifier = 1.0 - rt * 4.0
    return max(0.2, min(2.0, modifier))


def compute_population_diversity(
    profiles: dict[str, GeneticProfile],
) -> float:
    """Colony genetic diversity index (0-1).

    Uses average pairwise allelic distance across all traits.
    """
    ids = list(profiles.keys())
    n = len(ids)
    if n < 2:
        return 0.0
    total_dist = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            pa = profiles[ids[i]]
            pb = profiles[ids[j]]
            dist = sum(
                abs(pa.alleles.get(t, 0.0) - pb.alleles.get(t, 0.0))
                for t in GENETIC_TRAITS
            )
            total_dist += dist / len(GENETIC_TRAITS)
            pairs += 1
    avg_dist = total_dist / pairs if pairs else 0.0
    # Normalise: max possible distance is 0.30 (from -0.15 to +0.15)
    return min(1.0, avg_dist / 0.30)


# ── tick ───────────────────────────────────────────────────────────────

def _apply_mutations(
    profile: GeneticProfile, biome_level: int, year: int,
    rng: random.Random,
) -> list[dict]:
    """Apply random mutations driven by radiation and age."""
    mutations: list[dict] = []
    # Higher biome_level = more atmosphere = less radiation
    radiation_factor = max(0.0, 1.0 - biome_level * 0.15)
    rad_mod = compute_genetic_radiation_modifier(profile)
    mutation_rate = _MUTATION_BASE_RATE * (1.0 + radiation_factor * rad_mod)
    for trait in GENETIC_TRAITS:
        if rng.random() < mutation_rate:
            old = profile.alleles.get(trait, 0.0)
            delta = rng.gauss(0.0, _MUTATION_MAGNITUDE)
            profile.alleles[trait] = max(_ALLELE_MIN, min(_ALLELE_MAX, old + delta))
            mutations.append({
                "trait": trait, "old": round(old, 4),
                "new": round(profile.alleles[trait], 4),
                "cause": "radiation" if radiation_factor > 0.5 else "spontaneous",
            })
    return mutations


def _check_adaptations(profile: GeneticProfile) -> list[str]:
    """Detect new adaptations (positive trait thresholds)."""
    new: list[str] = []
    for trait in GENETIC_TRAITS:
        label = f"adapted:{trait}"
        if profile.alleles.get(trait, 0.0) >= _ADAPTATION_THRESHOLD:
            if label not in profile.adaptations:
                profile.adaptations.append(label)
                new.append(label)
    return new


def _check_conditions(profile: GeneticProfile) -> list[str]:
    """Detect new genetic conditions (negative trait thresholds)."""
    new: list[str] = []
    for trait in GENETIC_TRAITS:
        label = f"deficiency:{trait}"
        if profile.alleles.get(trait, 0.0) <= _CONDITION_THRESHOLD:
            if label not in profile.conditions:
                profile.conditions.append(label)
                new.append(label)
    return new


def tick_genetics(
    state: GeneticsState,
    active_ids: list[str],
    year: int,
    biome_level: int,
    rng: random.Random,
) -> GeneticsTickResult:
    """Advance genetics by one year.

    Applies mutations, checks for adaptations/conditions, updates
    colony-level diversity and fitness metrics.
    """
    all_mutations: list[dict] = []
    all_adaptations: list[dict] = []
    all_conditions: list[dict] = []

    for cid in active_ids:
        profile = state.profiles.get(cid)
        if profile is None:
            continue

        muts = _apply_mutations(profile, biome_level, year, rng)
        for m in muts:
            m["colonist"] = cid
            m["year"] = year
        all_mutations.extend(muts)

        new_adapt = _check_adaptations(profile)
        for a in new_adapt:
            all_adaptations.append({"colonist": cid, "adaptation": a, "year": year})

        new_cond = _check_conditions(profile)
        for c in new_cond:
            all_conditions.append({"colonist": cid, "condition": c, "year": year})

        profile.clamp()

    # Update colony metrics
    state.total_mutations += len(all_mutations)
    state.total_conditions += len(all_conditions)

    active_profiles = {cid: state.profiles[cid] for cid in active_ids if cid in state.profiles}
    state.colony_diversity = compute_population_diversity(active_profiles)

    fitnesses = [_compute_fitness(p) for p in active_profiles.values()]
    state.avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 1.0

    state.clamp()

    return GeneticsTickResult(
        mutations=all_mutations,
        new_adaptations=all_adaptations,
        new_conditions=all_conditions,
        diversity_index=state.colony_diversity,
        avg_fitness=state.avg_fitness,
    )
