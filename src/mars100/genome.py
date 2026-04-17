"""Cultural Genome Engine — mines Mars-100 year files into executable LisPy genes.

Retrospective compilation: reads the colony's 100-year fossil record and extracts
recurring patterns as portable s-expression policy fragments.

Each gene encodes: "in situation X, response Y produced outcome Z" — compiled from
actual colony history, validated against the LisPy VM, and evolvable via mutation
and crossover.

Six extraction strategies:
  1. Crisis genes — event→action correlations via resource deltas
  2. Governance genes — election/amendment cadences and thresholds
  3. Resource genes — per-capita baselines and crisis frequencies
  4. Social genes — birth thresholds, cooperation scaling
  5. Subsim genes — activation thresholds, per-proposal patterns
  6. Amendment genes — constitutional amendments as temporal conditionals
"""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.mars100.lispy_vm import run as lispy_run, LispyError, LispyRuntimeError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GENE_CATEGORIES = ("crisis", "governance", "resource", "social", "subsim", "amendment")

_ACTION_VERB_RE = re.compile(r"works on (\w+)")
_PROPOSE_RE = re.compile(r"proposes\b")
_ELECTION_RE = re.compile(r"Leadership election:.*\(ratio (\d+)%\)")
_AMENDMENT_RE = re.compile(r"Amendment approved \(ratio (\d+)%\):\s*(.*)")


def _extract_action_verb(action_str: str) -> str:
    """Extract the verb from an action string like 'Ares works on terraforming (output: 27.9)'."""
    m = _ACTION_VERB_RE.search(action_str)
    if m:
        return m.group(1)
    if _PROPOSE_RE.search(action_str):
        return "propose"
    return "unknown"


def _resource_delta(before: dict[str, float] | None, after: dict[str, float] | None) -> dict[str, float]:
    """Compute resource changes between two snapshots."""
    if not before or not after:
        return {}
    return {k: after.get(k, 0) - before.get(k, 0) for k in before if k in after}


def _default_bindings() -> dict[str, Any]:
    """Return default LisPy bindings for gene validation."""
    return {
        "population": 10, "food": 1000, "water": 2000, "power": 500,
        "oxygen": 1500, "materials": 800, "morale": 65,
        "event-id": "dust_storm", "event-severity": 0.5,
        "year": 50, "cohesion": 0.6, "colony-morale": 65,
        "colony-population": 10, "sim-depth": 0, "sim-year": 50,
        "last-amendment-year": 30, "surplus": 200, "confidence": 0.7,
    }


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Gene:
    """A single cultural gene — an executable LisPy s-expression with metadata."""
    id: str
    category: str
    s_expr: str
    description: str
    source_years: list[int] = field(default_factory=list)
    confidence: float = 0.5
    support_count: int = 0

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "category": self.category,
            "s_expr": self.s_expr, "description": self.description,
            "source_years": self.source_years,
            "confidence": round(self.confidence, 4),
            "support_count": self.support_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Gene:
        return cls(
            id=d["id"], category=d["category"],
            s_expr=d["s_expr"], description=d["description"],
            source_years=d.get("source_years", []),
            confidence=d.get("confidence", 0.5),
            support_count=d.get("support_count", 0),
        )


@dataclass
class CulturalGenome:
    """The complete genome compiled from the colony fossil record."""
    genes: list[Gene]
    total_years: int = 0
    total_subsims: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "_meta": {
                "engine": "cultural-genome", "version": "1.0",
                "total_years_mined": self.total_years,
                "total_subsims_analyzed": self.total_subsims,
                "total_genes": len(self.genes),
                "categories": {c: sum(1 for g in self.genes if g.category == c) for c in GENE_CATEGORIES},
            },
            "genes": [g.to_dict() for g in self.genes],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CulturalGenome:
        meta = d.get("_meta", {})
        return cls(
            genes=[Gene.from_dict(g) for g in d.get("genes", [])],
            total_years=meta.get("total_years_mined", 0),
            total_subsims=meta.get("total_subsims_analyzed", 0),
        )


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
def load_year_files(data_dir: Path) -> list[dict[str, Any]]:
    """Load and sort all year-*.json files from the given directory."""
    years = []
    for p in sorted(data_dir.glob("year-*.json")):
        try:
            years.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(years, key=lambda y: y.get("year", 0))


# ---------------------------------------------------------------------------
# Extraction strategies
# ---------------------------------------------------------------------------
def extract_crisis_genes(years: list[dict[str, Any]]) -> list[Gene]:
    """Extract crisis-response genes: event→action correlations."""
    if not years:
        return []

    event_actions: dict[str, dict[str, int]] = {}
    event_years: dict[str, list[int]] = {}

    for yr in years:
        ev = yr.get("event", {})
        eid = ev.get("id", "unknown")
        ynum = yr.get("year", 0)
        for ca in yr.get("colonist_actions", []):
            verb = _extract_action_verb(ca.get("action", ""))
            if verb == "unknown":
                continue
            event_actions.setdefault(eid, {})
            event_actions[eid][verb] = event_actions[eid].get(verb, 0) + 1
            event_years.setdefault(eid, []).append(ynum)

    genes = []
    for eid, verbs in sorted(event_actions.items()):
        if not verbs:
            continue
        top_verb = max(verbs, key=lambda v: verbs[v])
        total = sum(verbs.values())
        conf = min(verbs[top_verb] / max(total, 1), 1.0)
        src_years = sorted(set(event_years.get(eid, [])))
        s_expr = f"(if (= event-id '{eid}) '{top_verb} 'cooperate)"
        genes.append(Gene(
            id=f"crisis-{eid}-{top_verb}",
            category="crisis", s_expr=s_expr,
            description=f"When {eid} occurs, dominant response is {top_verb} ({conf:.0%} of actions)",
            source_years=src_years, confidence=conf,
            support_count=verbs[top_verb],
        ))
    return genes


def extract_governance_genes(years: list[dict[str, Any]]) -> list[Gene]:
    """Extract governance genes: election cadence, amendment thresholds."""
    genes = []
    election_years = []
    amendment_years = []
    avg_ratio = []

    for yr in years:
        ynum = yr.get("year", 0)
        for g in yr.get("governance_results", []):
            m = _ELECTION_RE.search(g)
            if m:
                election_years.append(ynum)
                avg_ratio.append(int(m.group(1)))
            m2 = _AMENDMENT_RE.search(g)
            if m2:
                amendment_years.append(ynum)

    if len(election_years) >= 2:
        gaps = [election_years[i+1] - election_years[i] for i in range(len(election_years)-1)]
        cadence = sum(gaps) / len(gaps)
        mean_ratio = sum(avg_ratio) / len(avg_ratio) if avg_ratio else 60
        genes.append(Gene(
            id="gov-election-cadence",
            category="governance",
            s_expr=f"(if (> (- year last-amendment-year) {int(cadence)}) 'hold-election 'wait)",
            description=f"Elections every ~{cadence:.1f} years, avg support {mean_ratio:.0f}%",
            source_years=election_years,
            confidence=min(len(election_years) / max(len(years), 1), 1.0),
            support_count=len(election_years),
        ))

    if len(amendment_years) >= 2:
        gaps = [amendment_years[i+1] - amendment_years[i] for i in range(len(amendment_years)-1)]
        cadence = sum(gaps) / len(gaps)
        genes.append(Gene(
            id="gov-amendment-cadence",
            category="governance",
            s_expr=f"(if (> (- year last-amendment-year) {int(cadence)}) 'propose-amendment 'wait)",
            description=f"Amendments every ~{cadence:.1f} years",
            source_years=amendment_years,
            confidence=min(len(amendment_years) / max(len(years), 1), 1.0),
            support_count=len(amendment_years),
        ))

    if avg_ratio:
        threshold = sum(avg_ratio) / len(avg_ratio)
        genes.append(Gene(
            id="gov-democracy-threshold",
            category="governance",
            s_expr=f"(if (> confidence {threshold / 100:.2f}) 'approve 'reject)",
            description=f"Governance threshold: {threshold:.0f}% average approval",
            source_years=election_years,
            confidence=min(len(avg_ratio) / 20, 1.0),
            support_count=len(avg_ratio),
        ))

    return genes


def extract_resource_genes(years: list[dict[str, Any]]) -> list[Gene]:
    """Extract resource genes: per-capita baselines, crisis frequencies."""
    if not years:
        return []

    genes = []
    resource_keys = ["food", "water", "power", "oxygen", "materials", "morale"]

    per_capita: dict[str, list[float]] = {k: [] for k in resource_keys}
    for yr in years:
        pop = yr.get("population", 10)
        res = yr.get("resources_snapshot", {})
        if not res or pop <= 0:
            continue
        for k in resource_keys:
            v = res.get(k, 0)
            if k == "morale":
                per_capita[k].append(v)
            else:
                per_capita[k].append(v / pop)

    for k in resource_keys:
        vals = per_capita[k]
        if len(vals) < 5:
            continue
        mean_val = sum(vals) / len(vals)
        threshold = int(mean_val * 0.6)
        genes.append(Gene(
            id=f"resource-{k}-baseline",
            category="resource",
            s_expr=f"(if (< {k} {threshold}) 'crisis-mode 'normal)",
            description=f"Per-capita {k} baseline: {mean_val:.0f}; crisis below {threshold}",
            source_years=list(range(1, len(years) + 1)),
            confidence=min(len(vals) / 50, 1.0),
            support_count=len(vals),
        ))

    crisis_years = sum(1 for yr in years
                       if yr.get("event", {}).get("severity", 0) > 0.6)
    if len(years) > 0:
        crisis_freq = crisis_years / len(years)
        genes.append(Gene(
            id="resource-crisis-frequency",
            category="resource",
            s_expr=f"(if (> event-severity 0.6) 'conserve 'normal)",
            description=f"Crisis years: {crisis_years}/{len(years)} ({crisis_freq:.0%})",
            source_years=[yr.get("year", 0) for yr in years if yr.get("event", {}).get("severity", 0) > 0.6],
            confidence=min(crisis_years / 20, 1.0),
            support_count=crisis_years,
        ))

    return genes


def extract_social_genes(years: list[dict[str, Any]]) -> list[Gene]:
    """Extract social genes: birth thresholds, cooperation scaling."""
    if not years:
        return []

    genes = []
    birth_pops = []

    for yr in years:
        births = yr.get("births", [])
        if births:
            birth_pops.append(yr.get("population", 10))

    if birth_pops:
        avg_pop = sum(birth_pops) / len(birth_pops)
        genes.append(Gene(
            id="social-birth-threshold",
            category="social",
            s_expr=f"(if (> colony-population {int(avg_pop * 0.8)}) 'allow-birth 'delay)",
            description=f"Births occur at avg pop {avg_pop:.0f}; threshold set at {int(avg_pop * 0.8)}",
            source_years=[yr.get("year", 0) for yr in years if yr.get("births")],
            confidence=min(len(birth_pops) / 20, 1.0),
            support_count=len(birth_pops),
        ))

    coop_count = 0
    total_actions = 0
    for yr in years:
        for ca in yr.get("colonist_actions", []):
            total_actions += 1
            verb = _extract_action_verb(ca.get("action", ""))
            if verb in ("cooperate", "mediate", "farm", "terraform"):
                coop_count += 1

    if total_actions > 0:
        coop_ratio = coop_count / total_actions
        genes.append(Gene(
            id="social-cooperation-ratio",
            category="social",
            s_expr=f"(if (> colony-morale {int(50 * (1 + coop_ratio))}) 'cooperate 'self-preserve)",
            description=f"Cooperation rate: {coop_ratio:.0%} of all actions",
            source_years=list(range(1, len(years) + 1)),
            confidence=min(coop_ratio, 1.0),
            support_count=coop_count,
        ))

    return genes


def extract_subsim_genes(years: list[dict[str, Any]]) -> list[Gene]:
    """Extract subsim genes: activation threshold, per-proposal-type patterns."""
    genes = []
    subsim_years = []
    proposal_types: dict[str, int] = {}

    for yr in years:
        subs = yr.get("sub_sims", [])
        if subs:
            subsim_years.append(yr.get("year", 0))
        for s in subs:
            pt = s.get("proposal", "unknown")
            proposal_types[pt] = proposal_types.get(pt, 0) + 1

    if not subsim_years:
        return []

    first_subsim_year = min(subsim_years)
    genes.append(Gene(
        id="subsim-activation-threshold",
        category="subsim",
        s_expr=f"(if (> sim-year {first_subsim_year}) 'allow-subsim 'wait)",
        description=f"Sub-simulations first appeared in year {first_subsim_year}",
        source_years=subsim_years,
        confidence=min(len(subsim_years) / max(len(years), 1), 1.0),
        support_count=len(subsim_years),
    ))

    for pt, count in sorted(proposal_types.items(), key=lambda x: -x[1]):
        genes.append(Gene(
            id=f"subsim-{pt.replace(' ', '-')}",
            category="subsim",
            s_expr=f"(if (= event-id '{pt}) '(sub-sim 1) 'skip)",
            description=f"Subsim type '{pt}' used {count} times",
            source_years=subsim_years,
            confidence=min(count / max(sum(proposal_types.values()), 1), 1.0),
            support_count=count,
        ))

    return genes


def extract_amendment_genes(years: list[dict[str, Any]]) -> list[Gene]:
    """Extract amendment genes: each constitutional amendment as a temporal conditional."""
    genes = []

    for yr in years:
        ynum = yr.get("year", 0)
        pop = yr.get("population", 10)
        for g in yr.get("governance_results", []):
            m = _AMENDMENT_RE.search(g)
            if m:
                ratio = int(m.group(1))
                text = m.group(2).strip()
                slug = re.sub(r"\W+", "-", text.lower())[:40].rstrip("-")
                genes.append(Gene(
                    id=f"amendment-y{ynum}-{slug}",
                    category="amendment",
                    s_expr=f"(if (> year {ynum - 1}) '(apply-amendment '{slug}) 'pre-amendment)",
                    description=f"Year {ynum} (pop {pop}, {ratio}% approval): {text}",
                    source_years=[ynum],
                    confidence=ratio / 100.0,
                    support_count=1,
                ))

    return genes


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------
def extract_genome(years: list[dict[str, Any]]) -> list[Gene]:
    """Run all extraction strategies and return deduplicated genes."""
    extractors = [
        extract_crisis_genes, extract_governance_genes,
        extract_resource_genes, extract_social_genes,
        extract_subsim_genes, extract_amendment_genes,
    ]
    genes = []
    seen_ids: set[str] = set()
    for fn in extractors:
        for g in fn(years):
            if g.id not in seen_ids:
                seen_ids.add(g.id)
                genes.append(g)
    return genes


# ---------------------------------------------------------------------------
# Validation / evaluation
# ---------------------------------------------------------------------------
def validate_gene(gene: Gene, extra_bindings: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Validate a gene by running its s-expression in the LisPy VM."""
    bindings = _default_bindings()
    if extra_bindings:
        bindings.update(extra_bindings)
    try:
        result = lispy_run(gene.s_expr, extra_bindings=bindings, max_steps=200, max_depth=10)
        return True, f"OK: {result}"
    except (LispyError, LispyRuntimeError, Exception) as exc:
        return False, f"FAIL: {exc}"


def evaluate_gene(gene: Gene, colony_history: list[dict[str, Any]]) -> float:
    """Evaluate a gene's fitness score. Returns 0.0 for invalid genes."""
    valid, _ = validate_gene(gene)
    if not valid:
        return 0.0
    score = gene.confidence
    year_span = max(gene.source_years) - min(gene.source_years) if gene.source_years else 0
    if year_span >= 50:
        score = min(score + 0.1, 1.0)
    if gene.support_count > 20:
        score = min(score + 0.05, 1.0)
    return round(score, 4)


# ---------------------------------------------------------------------------
# Mutation / crossover
# ---------------------------------------------------------------------------
_THRESHOLD_RE = re.compile(r"(\(>\s+\w+\s+)(\d+)(\))")


def mutate_gene(gene: Gene, rng: random.Random | None = None) -> Gene:
    """Mutate a gene by adjusting numerical thresholds in its s-expression."""
    if rng is None:
        rng = random.Random()

    s_expr = gene.s_expr
    m = _THRESHOLD_RE.search(s_expr)
    if m:
        old_val = int(m.group(2))
        delta = rng.choice([-2, -1, 1, 2])
        new_val = max(0, old_val + delta)
        s_expr = s_expr[:m.start(2)] + str(new_val) + s_expr[m.end(2):]

    return Gene(
        id=f"{gene.id}-mut",
        category=gene.category,
        s_expr=s_expr,
        description=f"Mutated from {gene.id}",
        source_years=gene.source_years,
        confidence=max(0.0, gene.confidence * 0.9),
        support_count=gene.support_count,
    )


def crossover_genes(a: Gene, b: Gene) -> Gene:
    """Crossover two genes of the same category into a conditional composite."""
    if a.category != b.category:
        raise ValueError(f"Cannot crossover different categories: {a.category} vs {b.category}")

    s_expr = f"(if (> year 50) {a.s_expr} {b.s_expr})"
    return Gene(
        id=f"{a.id}+{b.id}",
        category=a.category,
        s_expr=s_expr,
        description=f"Crossover of {a.id} and {b.id}",
        source_years=sorted(set(a.source_years + b.source_years)),
        confidence=round((a.confidence + b.confidence) / 2, 4),
        support_count=a.support_count + b.support_count,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def run_genome_extraction(data_dir: Path, output_path: Path) -> CulturalGenome:
    """Load year files, extract genome, validate, and write output."""
    years = load_year_files(data_dir)
    genes = extract_genome(years)

    total_subsims = sum(len(yr.get("sub_sims", [])) for yr in years)

    valid_genes = []
    for g in genes:
        ok, msg = validate_gene(g)
        if ok:
            valid_genes.append(g)

    genome = CulturalGenome(
        genes=valid_genes,
        total_years=len(years),
        total_subsims=total_subsims,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(genome.to_dict(), indent=2))
    return genome


if __name__ == "__main__":
    import sys
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/mars-100")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir / "genome.json"
    genome = run_genome_extraction(data_dir, out)
    print(f"Extracted {len(genome.genes)} genes from {genome.total_years} years")
    for cat in GENE_CATEGORIES:
        n = sum(1 for g in genome.genes if g.category == cat)
        if n:
            print(f"  {cat}: {n}")
