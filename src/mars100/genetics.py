"""
Genetics organ for Mars-100 (engine v11.0).

Models heritable traits across colonist generations: 8 diploid loci,
Mendelian crossover at birth with mutation, epigenetic modifiers from
environment, colony-wide diversity and adaptation tracking.

Survival coupling: cause-specific genetic resistance gives colonists a
second chance after a lethal roll — high radiation_tolerance reduces death
from radiation exposure, high immune_vigor reduces disease deaths.

RNG offset: seed + 11239
"""
from __future__ import annotations

import math
import random as _random_module
from dataclasses import dataclass, field
from typing import Any

LOCUS_NAMES = (
    "resilience",            # resolve stat affinity
    "adaptability",          # improvisation stat affinity
    "sociability",           # empathy stat affinity
    "metabolism",            # food/water efficiency
    "radiation_tolerance",   # survival vs radiation deaths
    "bone_density",          # low-gravity Mars adaptation
    "cognitive_flex",        # coding/research skill affinity
    "immune_vigor",          # survival vs disease deaths
)

MUTATION_RATE = 0.05
MUTATION_SIGMA = 0.08
FOUNDER_NOISE_SIGMA = 0.12

# Cause-specific survival loci mapping
CAUSE_LOCUS_MAP: dict[str, str] = {
    "radiation exposure": "radiation_tolerance",
    "medical emergency": "immune_vigor",
    "untreated illness": "immune_vigor",
    "equipment malfunction": "adaptability",
    "habitat breach": "bone_density",
}
MAX_GENETIC_SURVIVAL = 0.35  # max 35% chance to survive a lethal cause

# Adaptation: which loci are "Mars-adapted" (higher = better for Mars)
MARS_ADAPTATION_LOCI = ("radiation_tolerance", "bone_density", "metabolism")
MARS_ADAPTATION_WEIGHTS = (0.4, 0.35, 0.25)


@dataclass
class Locus:
    """A single diploid locus with two alleles and an epigenetic modifier."""
    allele_a: float = 0.5
    allele_b: float = 0.5
    epigenetic: float = 1.0

    @property
    def expression(self) -> float:
        """Co-dominant expression: mean of alleles, scaled by epigenetics."""
        raw = (self.allele_a + self.allele_b) / 2.0
        return max(0.0, min(1.0, raw * self.epigenetic))

    @property
    def heterozygosity(self) -> float:
        """Per-locus diversity measure."""
        return abs(self.allele_a - self.allele_b)

    def to_dict(self) -> dict[str, float]:
        return {
            "allele_a": round(self.allele_a, 4),
            "allele_b": round(self.allele_b, 4),
            "epigenetic": round(self.epigenetic, 4),
            "expression": round(self.expression, 4),
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> Locus:
        return cls(
            allele_a=d.get("allele_a", 0.5),
            allele_b=d.get("allele_b", 0.5),
            epigenetic=d.get("epigenetic", 1.0),
        )

    def clamp(self) -> None:
        """Enforce physical bounds."""
        self.allele_a = max(0.0, min(1.0, self.allele_a))
        self.allele_b = max(0.0, min(1.0, self.allele_b))
        self.epigenetic = max(0.5, min(1.5, self.epigenetic))


@dataclass
class GeneticProfile:
    """Complete genotype for one colonist."""
    loci: dict[str, Locus] = field(default_factory=lambda: {
        name: Locus() for name in LOCUS_NAMES
    })
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)

    def expression(self, locus_name: str) -> float:
        """Get expressed value of a locus."""
        locus = self.loci.get(locus_name)
        if locus is None:
            return 0.5
        return locus.expression

    def mean_heterozygosity(self) -> float:
        """Mean heterozygosity across all loci."""
        if not self.loci:
            return 0.0
        return sum(l.heterozygosity for l in self.loci.values()) / len(self.loci)

    def adaptation_score(self) -> float:
        """How Mars-adapted this genotype is (0.0–1.0)."""
        score = 0.0
        for locus_name, weight in zip(MARS_ADAPTATION_LOCI, MARS_ADAPTATION_WEIGHTS):
            score += self.expression(locus_name) * weight
        return max(0.0, min(1.0, score))

    def to_dict(self) -> dict[str, Any]:
        return {
            "loci": {name: locus.to_dict() for name, locus in self.loci.items()},
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "heterozygosity": round(self.mean_heterozygosity(), 4),
            "adaptation": round(self.adaptation_score(), 4),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GeneticProfile:
        loci_data = d.get("loci", {})
        loci = {}
        for name in LOCUS_NAMES:
            if name in loci_data:
                loci[name] = Locus.from_dict(loci_data[name])
            else:
                loci[name] = Locus()
        return cls(
            loci=loci,
            generation=d.get("generation", 0),
            parent_ids=list(d.get("parent_ids", [])),
        )

    def clamp(self) -> None:
        """Enforce physical bounds on all loci."""
        for locus in self.loci.values():
            locus.clamp()


def create_founder_genetics(colonist_id: str, stats_dict: dict[str, float],
                            rng: _random_module.Random) -> GeneticProfile:
    """Create a genetic profile for a founding colonist.

    Alleles are seeded from the colonist's stats to ensure consistency,
    then jittered with noise so alleles don't perfectly equal stats.
    """
    stat_locus_map = {
        "resilience": "resolve",
        "adaptability": "improvisation",
        "sociability": "empathy",
    }
    loci: dict[str, Locus] = {}
    for name in LOCUS_NAMES:
        if name in stat_locus_map:
            stat_val = stats_dict.get(stat_locus_map[name], 0.5)
            a = max(0.0, min(1.0, stat_val + rng.gauss(0, FOUNDER_NOISE_SIGMA)))
            b = max(0.0, min(1.0, stat_val + rng.gauss(0, FOUNDER_NOISE_SIGMA)))
        else:
            a = max(0.0, min(1.0, rng.gauss(0.5, 0.2)))
            b = max(0.0, min(1.0, rng.gauss(0.5, 0.2)))
        loci[name] = Locus(allele_a=a, allele_b=b)
    return GeneticProfile(loci=loci, generation=0, parent_ids=[])


def create_immigrant_genetics(rng: _random_module.Random) -> GeneticProfile:
    """Create a genetic profile for an immigrant from Earth.

    Immigrants bring genetic diversity — alleles are uniformly random.
    """
    loci: dict[str, Locus] = {}
    for name in LOCUS_NAMES:
        a = max(0.0, min(1.0, rng.uniform(0.2, 0.8)))
        b = max(0.0, min(1.0, rng.uniform(0.2, 0.8)))
        loci[name] = Locus(allele_a=a, allele_b=b)
    return GeneticProfile(loci=loci, generation=0, parent_ids=[])


def inherit_genetics(parent_a_profile: GeneticProfile,
                     parent_b_profile: GeneticProfile,
                     child_id: str,
                     rng: _random_module.Random) -> GeneticProfile:
    """Create a child's genetic profile via Mendelian crossover + mutation.

    For each locus: child gets one random allele from each parent.
    Mutation: MUTATION_RATE chance per locus of gaussian noise on inherited allele.
    Epigenetics reset to 1.0 at birth.
    """
    parent_gen = max(parent_a_profile.generation, parent_b_profile.generation)
    loci: dict[str, Locus] = {}
    for name in LOCUS_NAMES:
        pa = parent_a_profile.loci.get(name, Locus())
        pb = parent_b_profile.loci.get(name, Locus())
        # Mendelian: pick one allele from each parent
        a = rng.choice([pa.allele_a, pa.allele_b])
        b = rng.choice([pb.allele_a, pb.allele_b])
        # Mutation
        if rng.random() < MUTATION_RATE:
            a = max(0.0, min(1.0, a + rng.gauss(0, MUTATION_SIGMA)))
        if rng.random() < MUTATION_RATE:
            b = max(0.0, min(1.0, b + rng.gauss(0, MUTATION_SIGMA)))
        loci[name] = Locus(allele_a=a, allele_b=b, epigenetic=1.0)
    return GeneticProfile(
        loci=loci,
        generation=parent_gen + 1,
        parent_ids=[
            *parent_a_profile.parent_ids[-1:],
            *parent_b_profile.parent_ids[-1:],
        ] if parent_a_profile.parent_ids or parent_b_profile.parent_ids
        else [],
    )


def compute_colony_diversity(profiles: list[GeneticProfile]) -> float:
    """Compute colony-wide genetic diversity via allele variance.

    Higher variance = more diversity = healthier gene pool.
    Returns 0.0–1.0 (normalized).
    """
    if len(profiles) < 2:
        return 0.0
    diversity_sum = 0.0
    for name in LOCUS_NAMES:
        alleles: list[float] = []
        for p in profiles:
            locus = p.loci.get(name, Locus())
            alleles.append(locus.allele_a)
            alleles.append(locus.allele_b)
        if len(alleles) < 2:
            continue
        mean_val = sum(alleles) / len(alleles)
        variance = sum((a - mean_val) ** 2 for a in alleles) / len(alleles)
        # Max theoretical variance for [0,1] uniform is 1/12 ≈ 0.083
        diversity_sum += min(1.0, variance / 0.083)
    return diversity_sum / len(LOCUS_NAMES)


def compute_relatedness(profile_a: GeneticProfile,
                        profile_b: GeneticProfile) -> float:
    """Estimate genetic relatedness between two colonists (0.0–1.0).

    Uses allele similarity across loci as a proxy. Higher = more related.
    """
    if not profile_a.loci or not profile_b.loci:
        return 0.0
    similarity_sum = 0.0
    count = 0
    for name in LOCUS_NAMES:
        la = profile_a.loci.get(name)
        lb = profile_b.loci.get(name)
        if la is None or lb is None:
            continue
        # Compare all allele pairs, take minimum distance
        diffs = [
            abs(la.allele_a - lb.allele_a),
            abs(la.allele_a - lb.allele_b),
            abs(la.allele_b - lb.allele_a),
            abs(la.allele_b - lb.allele_b),
        ]
        min_diff = min(diffs)
        similarity_sum += 1.0 - min_diff
        count += 1
    if count == 0:
        return 0.0
    return similarity_sum / count


def compute_genetic_survival(cause: str, profile: GeneticProfile) -> float:
    """Compute probability of genetically surviving a lethal cause.

    Returns 0.0–MAX_GENETIC_SURVIVAL. Higher = more likely to survive.
    Only applies to causes with a matching genetic locus.
    """
    locus_name = CAUSE_LOCUS_MAP.get(cause)
    if locus_name is None:
        return 0.0
    expression = profile.expression(locus_name)
    return expression * MAX_GENETIC_SURVIVAL


def apply_epigenetic_stress(profile: GeneticProfile, event_type: str,
                            severity: float) -> list[str]:
    """Apply epigenetic modifications based on environmental stress.

    Severe events shift epigenetic markers on relevant loci.
    Returns list of affected locus names.
    """
    affected: list[str] = []
    if severity < 0.3:
        return affected

    stress_map: dict[str, list[str]] = {
        "dust_storm": ["bone_density", "adaptability"],
        "solar_flare": ["radiation_tolerance"],
        "epidemic": ["immune_vigor"],
        "resource_strike": ["metabolism"],
        "equipment_failure": ["cognitive_flex", "adaptability"],
        "alien_signal": ["cognitive_flex", "sociability"],
        "earthquake": ["bone_density"],
        "cold_snap": ["metabolism", "resilience"],
    }
    target_loci = stress_map.get(event_type, [])
    if not target_loci:
        return affected

    # Stress up-regulates relevant genes (mild Lamarckian pressure)
    shift = severity * 0.05
    for locus_name in target_loci:
        locus = profile.loci.get(locus_name)
        if locus is not None:
            locus.epigenetic = max(0.5, min(1.5, locus.epigenetic + shift))
            affected.append(locus_name)
    return affected


def compute_metabolism_modifier(profiles: list[GeneticProfile]) -> float:
    """Compute colony-wide metabolism modifier from active colonists.

    Higher mean metabolism expression → lower resource consumption.
    Returns multiplier near 1.0 (0.95–1.05 range).
    """
    if not profiles:
        return 1.0
    mean_met = sum(p.expression("metabolism") for p in profiles) / len(profiles)
    # Center around 0.5 default → 1.0 modifier
    return 1.0 - (mean_met - 0.5) * 0.1


@dataclass
class GeneticsYearContext:
    """Input context for genetics tick."""
    year: int
    event_type: str = "none"
    event_severity: float = 0.0
    birth_records: list[dict] = field(default_factory=list)
    death_ids: list[str] = field(default_factory=list)
    immigration_ids: list[str] = field(default_factory=list)


@dataclass
class GeneticsTickResult:
    """Output of one year's genetics computation."""
    colony_diversity: float = 0.0
    mean_adaptation: float = 0.0
    mean_heterozygosity: float = 0.0
    generation_counts: dict[int, int] = field(default_factory=dict)
    epigenetic_shifts: list[dict] = field(default_factory=list)
    mars_born_count: int = 0
    earth_born_count: int = 0
    metabolism_modifier: float = 1.0
    genetic_survivals: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "colony_diversity": round(self.colony_diversity, 4),
            "mean_adaptation": round(self.mean_adaptation, 4),
            "mean_heterozygosity": round(self.mean_heterozygosity, 4),
            "generation_counts": dict(self.generation_counts),
            "epigenetic_shifts": self.epigenetic_shifts,
            "mars_born_count": self.mars_born_count,
            "earth_born_count": self.earth_born_count,
            "metabolism_modifier": round(self.metabolism_modifier, 4),
            "genetic_survivals": self.genetic_survivals,
        }


@dataclass
class GeneticsState:
    """Colony-wide genetics bookkeeping."""
    diversity_history: list[float] = field(default_factory=list)
    adaptation_history: list[float] = field(default_factory=list)
    founding_diversity: float = 0.0
    total_mutations: int = 0
    max_generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "diversity_history": [round(d, 4) for d in self.diversity_history[-20:]],
            "adaptation_history": [round(a, 4) for a in self.adaptation_history[-20:]],
            "founding_diversity": round(self.founding_diversity, 4),
            "total_mutations": self.total_mutations,
            "max_generation": self.max_generation,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GeneticsState:
        return cls(
            diversity_history=list(d.get("diversity_history", [])),
            adaptation_history=list(d.get("adaptation_history", [])),
            founding_diversity=d.get("founding_diversity", 0.0),
            total_mutations=d.get("total_mutations", 0),
            max_generation=d.get("max_generation", 0),
        )


def tick_genetics(
    genetics_map: dict[str, GeneticProfile],
    genetics_state: GeneticsState,
    active_ids: list[str],
    ctx: GeneticsYearContext,
    rng: _random_module.Random,
) -> GeneticsTickResult:
    """Advance genetics by one year.

    1. Apply epigenetic stress from events.
    2. Compute colony-wide diversity and adaptation.
    3. Update state history.
    """
    result = GeneticsTickResult()

    # 1. Epigenetic stress from events
    if ctx.event_severity > 0.3:
        for cid in active_ids:
            profile = genetics_map.get(cid)
            if profile is None:
                continue
            affected = apply_epigenetic_stress(
                profile, ctx.event_type, ctx.event_severity)
            if affected:
                result.epigenetic_shifts.append({
                    "colonist_id": cid,
                    "loci": affected,
                    "event": ctx.event_type,
                })

    # 2. Colony-wide metrics
    active_profiles = [genetics_map[cid] for cid in active_ids
                       if cid in genetics_map]
    result.colony_diversity = compute_colony_diversity(active_profiles)
    if active_profiles:
        result.mean_adaptation = sum(
            p.adaptation_score() for p in active_profiles) / len(active_profiles)
        result.mean_heterozygosity = sum(
            p.mean_heterozygosity() for p in active_profiles) / len(active_profiles)

    # 3. Generation counts
    gen_counts: dict[int, int] = {}
    mars_born = 0
    earth_born = 0
    for cid in active_ids:
        p = genetics_map.get(cid)
        if p is None:
            continue
        gen_counts[p.generation] = gen_counts.get(p.generation, 0) + 1
        if p.generation > 0:
            mars_born += 1
        else:
            earth_born += 1
        if p.generation > genetics_state.max_generation:
            genetics_state.max_generation = p.generation
    result.generation_counts = gen_counts
    result.mars_born_count = mars_born
    result.earth_born_count = earth_born

    # 4. Metabolism modifier
    result.metabolism_modifier = compute_metabolism_modifier(active_profiles)

    # 5. Update state history
    genetics_state.diversity_history.append(result.colony_diversity)
    genetics_state.adaptation_history.append(result.mean_adaptation)
    if ctx.year == 1 and genetics_state.founding_diversity == 0.0:
        genetics_state.founding_diversity = result.colony_diversity

    # Clamp all profiles
    for profile in active_profiles:
        profile.clamp()

    return result
