"""
Counterfactual engine for Mars-100.

Identifies critical "hinge points" in the colony's history, forks
alternate timelines with real interventions, and measures how sensitive
the colony's fate was to each decision.

The Counterfactual Principle: any irreversible decision must be preceded
by alternate-timeline modeling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.mars100.engine import Mars100Engine, YearResult
from src.mars100.colony import RESOURCE_NAMES

HINGE_CATEGORIES = (
    "governance_change", "death", "resource_crisis",
    "birth", "exile", "meta_awareness",
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class HingePoint:
    """A moment where the colony's trajectory could have diverged."""
    year: int
    category: str
    description: str
    severity: float  # 0.0-1.0
    actual_outcome: str

    def to_dict(self) -> dict:
        return {
            "year": self.year, "category": self.category,
            "description": self.description, "severity": self.severity,
            "actual_outcome": self.actual_outcome,
        }


@dataclass
class AlternateTimeline:
    """Result of running an alternate history from a hinge point."""
    hinge_year: int
    intervention: str
    years_simulated: int
    final_population: int
    final_resources: dict[str, float]
    final_governance: str
    total_deaths: int
    final_cohesion: float
    governance_changes: int
    meta_events: int

    def to_dict(self) -> dict:
        return {
            "hinge_year": self.hinge_year, "intervention": self.intervention,
            "years_simulated": self.years_simulated,
            "final_population": self.final_population,
            "final_resources": self.final_resources,
            "final_governance": self.final_governance,
            "total_deaths": self.total_deaths,
            "final_cohesion": self.final_cohesion,
            "governance_changes": self.governance_changes,
            "meta_events": self.meta_events,
        }


@dataclass
class TimelineDivergence:
    """Comparison between actual and alternate timeline."""
    hinge: HingePoint
    alternate: AlternateTimeline
    population_delta: int
    resource_divergence: float
    governance_same: bool
    cohesion_delta: float
    divergence_score: float  # 0.0-1.0 composite metric

    def to_dict(self) -> dict:
        return {
            "hinge": self.hinge.to_dict(),
            "alternate": self.alternate.to_dict(),
            "population_delta": self.population_delta,
            "resource_divergence": self.resource_divergence,
            "governance_same": self.governance_same,
            "cohesion_delta": self.cohesion_delta,
            "divergence_score": self.divergence_score,
        }


@dataclass
class CounterfactualAnalysis:
    """Complete what-if analysis of the colony's history."""
    hinge_points: list[HingePoint]
    divergences: list[TimelineDivergence]
    most_consequential: HingePoint | None
    fragility_score: float  # 0.0-1.0, severity-weighted mean divergence
    max_divergence: float
    proposed_amendment: dict

    def to_dict(self) -> dict:
        return {
            "hinge_points": [h.to_dict() for h in self.hinge_points],
            "divergences": [d.to_dict() for d in self.divergences],
            "most_consequential": self.most_consequential.to_dict() if self.most_consequential else None,
            "fragility_score": self.fragility_score,
            "max_divergence": self.max_divergence,
            "proposed_amendment": self.proposed_amendment,
        }


# ---------------------------------------------------------------------------
# Hinge point detection
# ---------------------------------------------------------------------------

def find_hinge_points(years: list[YearResult]) -> list[HingePoint]:
    """Scan simulation results for critical decision moments."""
    hinges: list[HingePoint] = []
    prev_gov_type = "anarchy"
    for yr in years:
        prev_gov_type = _check_governance_hinge(yr, hinges, prev_gov_type)
        _check_death_hinge(yr, hinges)
        _check_resource_crisis_hinge(yr, hinges)
        _check_birth_hinge(yr, hinges)
        _check_exile_hinge(yr, hinges)
        _check_meta_hinge(yr, hinges)
    return sorted(hinges, key=lambda h: (-h.severity, h.year))


def _check_governance_hinge(yr: YearResult, hinges: list[HingePoint],
                             prev_gov_type: str) -> str:
    """Governance changes are high-impact hinge points.

    Returns the current governance type for tracking across years.
    """
    gov = yr.governance
    if gov and isinstance(gov, dict) and gov.get("passed"):
        new_type = gov.get("gov_type", "unknown")
        hinges.append(HingePoint(
            year=yr.year, category="governance_change",
            description=f"Governance changed to {new_type}",
            severity=0.8,
            actual_outcome=f"Proposal passed: {new_type}",
        ))
        return new_type
    # Check governance_state for changes not captured by proposal
    gs = yr.governance_state
    if isinstance(gs, dict):
        current_type = gs.get("gov_type", "anarchy")
        if current_type != prev_gov_type:
            return current_type
    return prev_gov_type


def _check_death_hinge(yr: YearResult, hinges: list[HingePoint]) -> None:
    """Deaths are moderate-to-high impact depending on population."""
    for death in yr.deaths:
        alive_count = sum(1 for c in yr.colonist_snapshots if c.get("alive"))
        severity = min(1.0, 0.5 + (1.0 / max(1, alive_count)))
        name = death.get("name", death.get("id", "unknown"))
        cause = death.get("cause", "unknown")
        hinges.append(HingePoint(
            year=yr.year, category="death",
            description=f"{name} died: {cause}",
            severity=severity,
            actual_outcome=f"Death of {name}",
        ))


def _check_resource_crisis_hinge(yr: YearResult, hinges: list[HingePoint]) -> None:
    """Resource crises (any resource below 0.15) are hinge points."""
    for name in RESOURCE_NAMES:
        val = yr.resources_after.get(name, 0.5)
        if val < 0.15:
            severity = min(1.0, 0.6 + (0.15 - val) * 3)
            hinges.append(HingePoint(
                year=yr.year, category="resource_crisis",
                description=f"{name} critically low at {val:.0%}",
                severity=severity,
                actual_outcome=f"{name} dropped to {val:.0%}",
            ))


def _check_birth_hinge(yr: YearResult, hinges: list[HingePoint]) -> None:
    """Births in early colony years are significant."""
    for birth in yr.births:
        severity = max(0.3, 0.6 - yr.year * 0.005)
        name = birth.get("name", birth.get("id", "unknown"))
        parents = birth.get("parents", [])
        hinges.append(HingePoint(
            year=yr.year, category="birth",
            description=f"Birth of {name}",
            severity=severity,
            actual_outcome=f"{name} born to {', '.join(parents)}",
        ))


def _check_exile_hinge(yr: YearResult, hinges: list[HingePoint]) -> None:
    """Exiles are high-impact social events."""
    for exile in yr.exiles:
        name = exile.get("name", exile.get("id", "unknown"))
        hinges.append(HingePoint(
            year=yr.year, category="exile",
            description=f"{name} exiled from colony",
            severity=0.7,
            actual_outcome=f"Exile of {name}",
        ))


def _check_meta_hinge(yr: YearResult, hinges: list[HingePoint]) -> None:
    """First meta-awareness event in a decade is a philosophical hinge."""
    if yr.meta_awareness and yr.year <= 35:
        insight = "meta-awareness"
        if isinstance(yr.meta_awareness[0], dict):
            insight = yr.meta_awareness[0].get("insight", "meta-awareness")[:100]
        hinges.append(HingePoint(
            year=yr.year, category="meta_awareness",
            description="Colonist questions the nature of reality",
            severity=0.5,
            actual_outcome=insight,
        ))


# ---------------------------------------------------------------------------
# Intervention functions — modify a snapshot to test "what if"
# ---------------------------------------------------------------------------

def _apply_intervention(engine: Mars100Engine, hinge: HingePoint) -> str:
    """Apply a category-specific intervention to a forked engine.

    Returns a description of the intervention applied.
    """
    if hinge.category == "governance_change":
        return _intervene_governance(engine)
    elif hinge.category == "death":
        return _intervene_death(engine, hinge)
    elif hinge.category == "resource_crisis":
        return _intervene_resource(engine, hinge)
    elif hinge.category == "birth":
        return _intervene_no_birth(engine)
    elif hinge.category == "exile":
        return _intervene_no_exile(engine)
    elif hinge.category == "meta_awareness":
        return _intervene_meta_boost(engine)
    return "no_intervention"


def _intervene_governance(engine: Mars100Engine) -> str:
    """Revert governance to anarchy — test what if the proposal never passed."""
    old_type = engine.governance.gov_type
    engine.governance.gov_type = "anarchy"
    engine.governance.leader_id = None
    engine.governance.council_ids = []
    engine.governance.term_end_year = None
    return f"reverted_governance_from_{old_type}_to_anarchy"


def _intervene_death(engine: Mars100Engine, hinge: HingePoint) -> str:
    """Resurrect the dead colonist — test what if they survived."""
    # Find the colonist who died
    desc = hinge.description
    for colonist in engine.colonists:
        if colonist.name in desc and not colonist.is_active():
            colonist.death_year = None
            colonist.death_cause = None
            colonist.alive = True
            colonist.stats.resolve = min(1.0, colonist.stats.resolve + 0.1)
            # Boost medicine to represent prevention
            engine.resources.medicine = min(1.0, engine.resources.medicine + 0.2)
            return f"resurrected_{colonist.name}"
    # If we can't find the exact colonist, boost medicine anyway
    engine.resources.medicine = min(1.0, engine.resources.medicine + 0.2)
    return "boosted_medicine_no_specific_target"


def _intervene_resource(engine: Mars100Engine, hinge: HingePoint) -> str:
    """Inject resources — test what if the crisis was averted."""
    desc = hinge.description.lower()
    boosted: list[str] = []
    for name in RESOURCE_NAMES:
        if name in desc:
            val = getattr(engine.resources, name)
            setattr(engine.resources, name, min(1.0, val + 0.4))
            boosted.append(name)
    if not boosted:
        # Boost the worst resource
        worst = min(RESOURCE_NAMES, key=lambda n: getattr(engine.resources, n))
        val = getattr(engine.resources, worst)
        setattr(engine.resources, worst, min(1.0, val + 0.4))
        boosted.append(worst)
    return f"injected_resources_{'+'.join(boosted)}"


def _intervene_no_birth(engine: Mars100Engine) -> str:
    """Remove the last-born colonist — test what if the birth didn't happen."""
    if engine.colonists and len(engine.colonists) > 10:
        last = engine.colonists[-1]
        engine.colonists = engine.colonists[:-1]
        return f"suppressed_birth_of_{last.name}"
    return "no_birth_to_suppress"


def _intervene_no_exile(engine: Mars100Engine) -> str:
    """Reverse an exile — test what if they stayed."""
    for colonist in engine.colonists:
        if getattr(colonist, "exile_year", None) is not None:
            colonist.exile_year = None
            return f"reversed_exile_of_{colonist.name}"
    return "no_exile_to_reverse"


def _intervene_meta_boost(engine: Mars100Engine) -> str:
    """Boost faith and improvisation — accelerate meta-awareness."""
    boosted = 0
    for colonist in engine.colonists:
        if colonist.is_active():
            colonist.stats.faith = min(1.0, colonist.stats.faith + 0.1)
            colonist.stats.improvisation = min(1.0, colonist.stats.improvisation + 0.1)
            boosted += 1
    return f"boosted_meta_capacity_for_{boosted}_colonists"


# ---------------------------------------------------------------------------
# Timeline forking
# ---------------------------------------------------------------------------

def fork_timeline(snapshots: dict[int, dict], hinge: HingePoint,
                  total_years: int = 100) -> AlternateTimeline | None:
    """Fork reality at a hinge point and run the alternate timeline.

    Takes a snapshot dict (year -> engine snapshot), restores the engine
    state at hinge_year, applies an intervention, and continues the
    simulation to completion.
    """
    snap_year = hinge.year
    if snap_year not in snapshots:
        closest = min(snapshots.keys(), key=lambda y: abs(y - snap_year),
                      default=None)
        if closest is None:
            return None
        snap_year = closest

    engine = Mars100Engine.from_snapshot(snapshots[snap_year])
    intervention = _apply_intervention(engine, hinge)
    remaining = total_years - engine.year
    if remaining <= 0:
        return None

    engine.total_years = engine.year + remaining
    result = _run_remaining(engine, remaining)

    active = [c for c in engine.colonists if c.is_active()]
    return AlternateTimeline(
        hinge_year=hinge.year,
        intervention=intervention,
        years_simulated=remaining,
        final_population=len(active),
        final_resources=engine.resources.to_dict(),
        final_governance=engine.governance.gov_type,
        total_deaths=result["deaths"],
        final_cohesion=result["cohesion"],
        governance_changes=result["gov_changes"],
        meta_events=result["meta_events"],
    )


def _run_remaining(engine: Mars100Engine, years: int) -> dict[str, Any]:
    """Run the engine for N more years and collect summary stats."""
    deaths = 0
    gov_changes = 0
    meta_events = 0
    for _ in range(years):
        if not engine._active_colonists():
            break
        yr = engine.tick()
        deaths += len(yr.deaths)
        if yr.governance and isinstance(yr.governance, dict) and yr.governance.get("passed"):
            gov_changes += 1
        meta_events += len(yr.meta_awareness)

    active_ids = engine._active_ids()
    cohesion = engine.social.colony_cohesion(active_ids) if active_ids else 0.0
    return {"deaths": deaths, "gov_changes": gov_changes,
            "meta_events": meta_events, "cohesion": cohesion}


# ---------------------------------------------------------------------------
# Divergence computation
# ---------------------------------------------------------------------------

def compute_divergence(hinge: HingePoint, alternate: AlternateTimeline,
                       actual_population: int, actual_resources: dict[str, float],
                       actual_governance: str, actual_cohesion: float) -> TimelineDivergence:
    """Measure how different the alternate timeline is from reality."""
    pop_delta = alternate.final_population - actual_population
    pop_score = min(1.0, abs(pop_delta) / max(1, actual_population))

    resource_diffs = []
    for name in RESOURCE_NAMES:
        actual_val = actual_resources.get(name, 0.5)
        alt_val = alternate.final_resources.get(name, 0.5)
        resource_diffs.append(abs(actual_val - alt_val))
    resource_divergence = sum(resource_diffs) / max(1, len(resource_diffs))
    resource_score = min(1.0, resource_divergence * 2)

    gov_same = alternate.final_governance == actual_governance
    gov_score = 0.0 if gov_same else 0.5

    cohesion_delta = alternate.final_cohesion - actual_cohesion
    cohesion_score = min(1.0, abs(cohesion_delta) * 2)

    divergence = (pop_score * 0.3 + resource_score * 0.25 +
                  gov_score * 0.25 + cohesion_score * 0.2)

    return TimelineDivergence(
        hinge=hinge, alternate=alternate,
        population_delta=pop_delta,
        resource_divergence=resource_divergence,
        governance_same=gov_same,
        cohesion_delta=cohesion_delta,
        divergence_score=max(0.0, min(1.0, divergence)),
    )


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def run_counterfactual_analysis(
    engine_seed: int = 42,
    total_years: int = 100,
    max_hinges: int = 5,
) -> CounterfactualAnalysis:
    """Run the full counterfactual analysis.

    1. Run the canonical simulation capturing snapshots.
    2. Identify hinge points from the results.
    3. Fork timelines at the top hinge points with category-specific interventions.
    4. Compute divergence between actual and alternate outcomes.
    5. Propose an amendment based on the findings.
    """
    # Phase 1: Run canonical sim with snapshots
    engine = Mars100Engine(seed=engine_seed, total_years=total_years)
    snapshots: dict[int, dict] = {}
    years: list[YearResult] = []

    for _ in range(total_years):
        if not engine._active_colonists():
            break
        snapshots[engine.year + 1] = engine.snapshot()
        yr = engine.tick()
        years.append(yr)

    actual_population = len([c for c in engine.colonists if c.is_active()])
    actual_resources = engine.resources.to_dict()
    actual_governance = engine.governance.gov_type
    active_ids = engine._active_ids()
    actual_cohesion = engine.social.colony_cohesion(active_ids) if active_ids else 0.0

    # Phase 2: Find hinge points
    all_hinges = find_hinge_points(years)
    # Diverse selection: round-robin across categories
    selected = _select_diverse_hinges(all_hinges, max_hinges)

    # Phase 3: Fork timelines
    divergences: list[TimelineDivergence] = []
    for hinge in selected:
        alt = fork_timeline(snapshots, hinge, total_years)
        if alt is None:
            continue
        div = compute_divergence(hinge, alt, actual_population,
                                 actual_resources, actual_governance,
                                 actual_cohesion)
        divergences.append(div)

    # Phase 4: Compute fragility
    if divergences:
        weighted_sum = sum(d.divergence_score * d.hinge.severity for d in divergences)
        weight_total = sum(d.hinge.severity for d in divergences)
        fragility = weighted_sum / max(0.001, weight_total)
        max_div = max(d.divergence_score for d in divergences)
        most_consequential = max(divergences, key=lambda d: d.divergence_score).hinge
    else:
        fragility = 0.0
        max_div = 0.0
        most_consequential = None

    # Phase 5: Propose amendment
    amendment = propose_amendment(fragility, most_consequential, divergences)

    return CounterfactualAnalysis(
        hinge_points=selected,
        divergences=divergences,
        most_consequential=most_consequential,
        fragility_score=max(0.0, min(1.0, fragility)),
        max_divergence=max(0.0, min(1.0, max_div)),
        proposed_amendment=amendment,
    )


def _select_diverse_hinges(hinges: list[HingePoint],
                           max_count: int) -> list[HingePoint]:
    """Select diverse hinge points using round-robin across categories.

    Ensures category diversity first, then decade diversity within each
    category. This prevents a single high-severity category from
    dominating all slots.
    """
    if not hinges or max_count <= 0:
        return []

    # Group by category, keeping severity order within each group
    by_category: dict[str, list[HingePoint]] = {}
    for h in hinges:
        by_category.setdefault(h.category, []).append(h)

    # Within each category, deduplicate by decade (keep highest severity)
    for cat in by_category:
        deduped: list[HingePoint] = []
        seen_decades: set[int] = set()
        for h in by_category[cat]:
            decade = h.year // 10
            if decade not in seen_decades:
                seen_decades.add(decade)
                deduped.append(h)
        by_category[cat] = deduped

    # Round-robin across categories sorted by best severity
    categories = sorted(by_category.keys(),
                        key=lambda c: by_category[c][0].severity if by_category[c] else 0,
                        reverse=True)
    selected: list[HingePoint] = []
    cat_indices: dict[str, int] = {c: 0 for c in categories}

    while len(selected) < max_count:
        added = False
        for cat in categories:
            if len(selected) >= max_count:
                break
            idx = cat_indices[cat]
            if idx < len(by_category[cat]):
                selected.append(by_category[cat][idx])
                cat_indices[cat] = idx + 1
                added = True
        if not added:
            break

    return selected


# ---------------------------------------------------------------------------
# Amendment proposal
# ---------------------------------------------------------------------------

def propose_amendment(fragility: float, most_consequential: HingePoint | None,
                      divergences: list[TimelineDivergence]) -> dict:
    """Generate a proposed constitutional amendment from counterfactual findings.

    The Counterfactual Principle: any irreversible decision must be preceded
    by alternate-timeline modeling.
    """
    evidence: list[str] = []
    for div in divergences[:3]:
        evidence.append(
            f"Year {div.hinge.year} ({div.hinge.category}): "
            f"divergence {div.divergence_score:.0%}, "
            f"population delta {div.population_delta:+d}"
        )

    confidence = "high" if fragility > 0.4 else "moderate" if fragility > 0.2 else "low"

    return {
        "name": "The Counterfactual Principle",
        "text": (
            "Any irreversible community decision — exile, constitutional amendment, "
            "resource allocation above 30% of reserves — must be preceded by at least "
            "two alternate-timeline simulations exploring different outcomes. "
            "The simulation results become part of the decision's public record. "
            "Decisions made without counterfactual evidence are advisory only and "
            "may be revisited within one governance cycle."
        ),
        "rationale": (
            f"Colony fragility score: {fragility:.0%}. "
            f"Most consequential decision: year {most_consequential.year} "
            f"({most_consequential.category}). "
            if most_consequential else
            f"Colony fragility score: {fragility:.0%}. "
        ) + (
            "Alternate timelines showed decisions without counterfactual modeling "
            "produced divergent outcomes that could have been anticipated."
        ),
        "evidence": evidence,
        "confidence": confidence,
        "fragility_score": fragility,
    }
