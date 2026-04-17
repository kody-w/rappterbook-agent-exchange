"""
Emergence pattern analyzer for Mars-100.

Reads the complete 100-year simulation output and discovers:
- Governance phase transitions and stability
- Sub-simulation accuracy (did subs predict outcomes?)
- Mortality patterns and systemic failure modes
- Value convergence/divergence trends
- Crisis resilience over time
- Faction formation from social clustering
- The strongest insight for amendment promotion

This is the colony looking back at its own history — self-reflection
as computation.  The organism studying its own fossil record.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GovernancePhase:
    """A period under one governance type."""
    gov_type: str
    start_year: int
    end_year: int | None
    duration: int = 0
    crises_weathered: int = 0
    deaths_during: int = 0
    avg_cohesion: float = 0.0
    proposal_count: int = 0

    def to_dict(self) -> dict:
        return {
            "gov_type": self.gov_type, "start_year": self.start_year,
            "end_year": self.end_year, "duration": self.duration,
            "crises_weathered": self.crises_weathered,
            "deaths_during": self.deaths_during,
            "avg_cohesion": round(self.avg_cohesion, 4),
            "proposal_count": self.proposal_count,
        }


@dataclass
class MortalityPattern:
    """Analysis of colonist deaths."""
    total_deaths: int = 0
    causes: dict[str, int] = field(default_factory=dict)
    avg_death_year: float = 0.0
    deadliest_decade: int = 0
    deaths_per_decade: list[int] = field(default_factory=list)
    systemic_cause: str = ""

    def to_dict(self) -> dict:
        return {
            "total_deaths": self.total_deaths, "causes": self.causes,
            "avg_death_year": round(self.avg_death_year, 1),
            "deadliest_decade": self.deadliest_decade,
            "deaths_per_decade": self.deaths_per_decade,
            "systemic_cause": self.systemic_cause,
        }


@dataclass
class ConvergenceTrend:
    """Value convergence analysis across the population."""
    stat_name: str
    early_stddev: float  # years 1-25
    mid_stddev: float    # years 26-50
    late_stddev: float   # years 51-75
    final_stddev: float  # years 76-100
    trend: str = "stable"  # "converging", "diverging", "stable"

    def to_dict(self) -> dict:
        return {
            "stat": self.stat_name,
            "early": round(self.early_stddev, 4),
            "mid": round(self.mid_stddev, 4),
            "late": round(self.late_stddev, 4),
            "final": round(self.final_stddev, 4),
            "trend": self.trend,
        }


@dataclass
class Faction:
    """A detected cluster of aligned colonists."""
    name: str
    member_ids: list[str]
    dominant_value: str
    cohesion: float
    formed_year: int

    def to_dict(self) -> dict:
        return {
            "name": self.name, "members": self.member_ids,
            "dominant_value": self.dominant_value,
            "cohesion": round(self.cohesion, 4),
            "formed_year": self.formed_year,
        }


@dataclass
class Insight:
    """A discovered emergence pattern worth promoting."""
    title: str
    evidence: str
    strength: float  # 0-1 confidence
    source: str       # which analysis produced it
    amendment_text: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title, "evidence": self.evidence,
            "strength": round(self.strength, 4),
            "source": self.source,
            "amendment_text": self.amendment_text,
        }


@dataclass
class EmergenceReport:
    """Complete emergence analysis."""
    governance_phases: list[GovernancePhase]
    mortality: MortalityPattern
    convergence: list[ConvergenceTrend]
    factions: list[Faction]
    crisis_resilience: list[float]  # per-decade resilience score
    subsim_accuracy: float
    total_subsims: int
    insights: list[Insight]
    proposed_amendment: Insight | None = None

    def to_dict(self) -> dict:
        return {
            "governance_phases": [p.to_dict() for p in self.governance_phases],
            "mortality": self.mortality.to_dict(),
            "convergence": [c.to_dict() for c in self.convergence],
            "factions": [f.to_dict() for f in self.factions],
            "crisis_resilience": [round(r, 4) for r in self.crisis_resilience],
            "subsim_accuracy": round(self.subsim_accuracy, 4),
            "total_subsims": self.total_subsims,
            "insights": [i.to_dict() for i in self.insights],
            "proposed_amendment": self.proposed_amendment.to_dict() if self.proposed_amendment else None,
        }


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_governance_phases(years: list[dict]) -> list[GovernancePhase]:
    """Extract governance phases from yearly data.

    A phase is a contiguous period under one governance type.  Track
    stability (duration), crises weathered, deaths, and average cohesion.
    """
    if not years:
        return []

    phases: list[GovernancePhase] = []
    current_type = "anarchy"
    current_start = 1
    cohesion_sum = 0.0
    cohesion_count = 0
    crises = 0
    deaths = 0
    proposals = 0

    for yr in years:
        year_num = yr.get("year", 0)
        gov_state = yr.get("governance_state", {})
        gov_type = gov_state.get("gov_type", current_type)

        # Track metrics for current phase
        cohesion_sum += yr.get("social_cohesion", 0.5)
        cohesion_count += 1
        deaths += len(yr.get("deaths", []))
        events = yr.get("events", [])
        crises += sum(1 for e in events if e.get("severity", 0) > 0.5)
        if yr.get("governance"):
            proposals += 1

        # Phase transition?
        if gov_type != current_type:
            avg_coh = cohesion_sum / max(1, cohesion_count)
            duration = year_num - current_start
            phases.append(GovernancePhase(
                gov_type=current_type, start_year=current_start,
                end_year=year_num, duration=duration,
                crises_weathered=crises, deaths_during=deaths,
                avg_cohesion=avg_coh, proposal_count=proposals,
            ))
            current_type = gov_type
            current_start = year_num
            cohesion_sum = 0.0
            cohesion_count = 0
            crises = 0
            deaths = 0
            proposals = 0

    # Final phase
    last_year = years[-1].get("year", 100) if years else 100
    avg_coh = cohesion_sum / max(1, cohesion_count)
    phases.append(GovernancePhase(
        gov_type=current_type, start_year=current_start,
        end_year=None, duration=last_year - current_start + 1,
        crises_weathered=crises, deaths_during=deaths,
        avg_cohesion=avg_coh, proposal_count=proposals,
    ))
    return phases


def analyze_mortality(years: list[dict], colonists: list[dict]) -> MortalityPattern:
    """Analyze death patterns across the simulation.

    Looks for systemic failure modes, deadliest decades, and cause
    distribution.
    """
    pattern = MortalityPattern()
    death_years: list[int] = []
    per_decade: list[int] = [0] * 10  # decades 0-9, 10-19, ..., 90-99

    for c in colonists:
        if not c.get("alive", True) and c.get("death_year") is not None:
            pattern.total_deaths += 1
            cause = c.get("death_cause", "unknown")
            pattern.causes[cause] = pattern.causes.get(cause, 0) + 1
            dy = c["death_year"]
            death_years.append(dy)
            decade_idx = min(9, max(0, (dy - 1) // 10))
            per_decade[decade_idx] += 1

    pattern.deaths_per_decade = per_decade
    if death_years:
        pattern.avg_death_year = sum(death_years) / len(death_years)
        pattern.deadliest_decade = (per_decade.index(max(per_decade))) * 10 + 1

    # Identify systemic cause
    if pattern.causes:
        top_cause = max(pattern.causes, key=lambda c: pattern.causes[c])
        top_ratio = pattern.causes[top_cause] / max(1, pattern.total_deaths)
        if top_ratio > 0.5:
            pattern.systemic_cause = (
                f"{top_cause} accounts for {top_ratio:.0%} of deaths — "
                f"systemic infrastructure failure"
            )
        else:
            pattern.systemic_cause = "deaths distributed across multiple causes"
    return pattern


def analyze_convergence(
    years: list[dict],
    colonists: list[dict] | None = None,
) -> list[ConvergenceTrend]:
    """Measure value convergence across population over four quarters.

    For each personality stat, compute the standard deviation of living
    colonists in each quarter (years 1-25, 26-50, 51-75, 76-100).
    If stddev decreases monotonically → converging.  Increases → diverging.

    Falls back to colonist birth-year clustering when per-year convergence
    data is not available in the yearly records.
    """
    from src.mars100.colonist import STAT_NAMES

    # Try to get convergence from yearly data first
    has_yearly_convergence = any(yr.get("convergence") for yr in years)

    if has_yearly_convergence:
        quarters: list[list[dict]] = [[], [], [], []]
        for yr in years:
            y = yr.get("year", 0)
            idx = min(3, max(0, (y - 1) // 25))
            quarters[idx].append(yr)

        trends: list[ConvergenceTrend] = []
        for stat_name in STAT_NAMES:
            stddevs: list[float] = []
            for q_years in quarters:
                all_values: list[float] = []
                for yr in q_years:
                    conv = yr.get("convergence", {})
                    if stat_name in conv:
                        all_values.append(conv[stat_name])
                if all_values:
                    stddevs.append(sum(all_values) / len(all_values))
                else:
                    stddevs.append(0.0)

            trend = _classify_trend(stddevs)
            trends.append(ConvergenceTrend(
                stat_name=stat_name,
                early_stddev=stddevs[0] if len(stddevs) > 0 else 0.0,
                mid_stddev=stddevs[1] if len(stddevs) > 1 else 0.0,
                late_stddev=stddevs[2] if len(stddevs) > 2 else 0.0,
                final_stddev=stddevs[3] if len(stddevs) > 3 else 0.0,
                trend=trend,
            ))
        return trends

    # Fallback: compute from colonist data partitioned by birth era
    if not colonists:
        return []

    # Partition colonists into birth-era cohorts
    eras = {
        "early": [],   # born years 0-25
        "mid": [],     # born years 26-50
        "late": [],    # born years 51-75
        "final": [],   # born years 76-100
    }
    for c in colonists:
        by = c.get("birth_year", c.get("year_born", 0))
        if by <= 25:
            eras["early"].append(c)
        elif by <= 50:
            eras["mid"].append(c)
        elif by <= 75:
            eras["late"].append(c)
        else:
            eras["final"].append(c)

    # For founding colonists (born year 0), track stat changes across
    # all living colonists at each quarter boundary
    # Use all colonists alive at each quarter boundary
    alive_at: dict[str, list[dict]] = {"early": [], "mid": [], "late": [], "final": []}
    for c in colonists:
        by = c.get("birth_year", c.get("year_born", 0))
        dy = c.get("death_year", c.get("year_died", None))
        if dy is None:
            dy = 999  # still alive
        for label, (lo, hi) in [("early", (0, 25)), ("mid", (26, 50)),
                                  ("late", (51, 75)), ("final", (76, 100))]:
            if by <= hi and dy >= lo:
                alive_at[label].append(c)

    trends = []
    for stat_name in STAT_NAMES:
        stddevs = []
        for label in ["early", "mid", "late", "final"]:
            vals = [c.get("stats", {}).get(stat_name, 0.5) for c in alive_at[label]]
            stddevs.append(_stddev(vals) if len(vals) >= 2 else 0.0)

        trend = _classify_trend(stddevs)
        trends.append(ConvergenceTrend(
            stat_name=stat_name,
            early_stddev=stddevs[0],
            mid_stddev=stddevs[1],
            late_stddev=stddevs[2],
            final_stddev=stddevs[3],
            trend=trend,
        ))
    return trends


def _classify_trend(stddevs: list[float]) -> str:
    """Classify a sequence of stddev values as converging/diverging/stable."""
    if len(stddevs) == 4 and all(s > 0 for s in stddevs):
        if stddevs[3] < stddevs[0] * 0.8:
            return "converging"
        elif stddevs[3] > stddevs[0] * 1.2:
            return "diverging"
    return "stable"
    return trends


def analyze_factions(colonists: list[dict]) -> list[Faction]:
    """Detect value-aligned factions among living colonists.

    Groups colonists by their dominant stat.  A faction forms when 3+
    colonists share the same dominant value with above-average level.
    """
    from src.mars100.colonist import STAT_NAMES

    active = [c for c in colonists if c.get("alive") and not c.get("exiled")]
    if len(active) < 3:
        return []

    # Group by dominant stat
    groups: dict[str, list[dict]] = {s: [] for s in STAT_NAMES}
    for c in active:
        stats = c.get("stats", {})
        if not stats:
            continue
        dominant = max(STAT_NAMES, key=lambda s: stats.get(s, 0))
        groups[dominant].append(c)

    factions: list[Faction] = []
    faction_names = {
        "resolve": "The Resolute", "improvisation": "The Improvisers",
        "empathy": "The Empaths", "hoarding": "The Providents",
        "faith": "The Faithful", "paranoia": "The Watchers",
    }

    for stat, members in groups.items():
        if len(members) >= 3:
            values = [m.get("stats", {}).get(stat, 0.5) for m in members]
            cohesion = 1.0 - _stddev(values) if values else 0.0
            # Earliest birth year as formation year
            birth_years = [m.get("birth_year", 0) for m in members]
            formed = max(birth_years) if birth_years else 0
            factions.append(Faction(
                name=faction_names.get(stat, f"The {stat.title()}"),
                member_ids=[m["id"] for m in members],
                dominant_value=stat,
                cohesion=max(0.0, cohesion),
                formed_year=formed,
            ))

    factions.sort(key=lambda f: len(f.member_ids), reverse=True)
    return factions


def analyze_crisis_resilience(years: list[dict]) -> list[float]:
    """Compute per-decade crisis resilience score.

    Resilience = (resource recovery after crisis) / (resource loss during crisis).
    Higher is better. Measures how the colony adapted over time.
    """
    decades: list[list[float]] = [[] for _ in range(10)]

    for yr in years:
        y = yr.get("year", 0)
        decade_idx = min(9, max(0, (y - 1) // 10))
        events = yr.get("events", [])
        crisis_events = [e for e in events if e.get("severity", 0) > 0.4]
        if not crisis_events:
            continue
        res_before = yr.get("resources_before", {})
        res_after = yr.get("resources_after", {})
        if not res_before or not res_after:
            continue
        loss = sum(max(0, res_before.get(k, 0) - res_after.get(k, 0))
                   for k in res_before)
        avg_after = sum(res_after.values()) / max(1, len(res_after))
        # Resilience: high remaining resources despite crisis = high resilience
        if loss > 0:
            resilience = avg_after / (loss + 0.01)
        else:
            resilience = avg_after + 0.5
        decades[decade_idx].append(min(1.0, resilience))

    return [
        sum(d) / max(1, len(d)) if d else 0.5
        for d in decades
    ]


def analyze_subsim_accuracy(years: list[dict]) -> tuple[float, int]:
    """Measure sub-simulation predictive accuracy.

    A sub-sim is "accurate" if its positive result (> 0.5) correlates with
    resource improvement the same year, or its negative result (< 0) correlates
    with resource decline.
    """
    correct = 0
    total = 0

    for yr in years:
        subsims = yr.get("subsim_log", [])
        delta = yr.get("resource_delta", {})
        avg_delta = sum(delta.values()) / max(1, len(delta)) if delta else 0.0

        for ss in subsims:
            result = ss.get("result")
            if not isinstance(result, (int, float)):
                continue
            total += 1
            # Positive sub-sim result → expect positive delta (and vice versa)
            if (result > 0.5 and avg_delta > 0) or (result < 0 and avg_delta < 0):
                correct += 1
            elif abs(result) < 0.5 and abs(avg_delta) < 0.05:
                correct += 1  # neutral prediction, neutral outcome

    accuracy = correct / max(1, total)
    return accuracy, total


def synthesize_insights(
    phases: list[GovernancePhase],
    mortality: MortalityPattern,
    convergence: list[ConvergenceTrend],
    factions: list[Faction],
    resilience: list[float],
    subsim_accuracy: float,
) -> list[Insight]:
    """Combine all analyses into ranked insights."""
    insights: list[Insight] = []

    # Insight 1: Most stable governance type
    if phases:
        longest = max(phases, key=lambda p: p.duration)
        strength = min(1.0, longest.duration / 50.0)
        insights.append(Insight(
            title=f"Most stable governance: {longest.gov_type}",
            evidence=(f"{longest.gov_type} lasted {longest.duration} years "
                      f"(years {longest.start_year}-{longest.end_year or 'end'}), "
                      f"weathering {longest.crises_weathered} crises with "
                      f"{longest.avg_cohesion:.0%} average cohesion"),
            strength=strength,
            source="governance_phases",
        ))

    # Insight 2: Sub-sim predictive power
    if subsim_accuracy > 0.4:
        insights.append(Insight(
            title="Sub-simulations are predictive, not just justificatory",
            evidence=(f"Sub-sims achieved {subsim_accuracy:.0%} accuracy in "
                      f"predicting resource outcomes — better than chance (50%)"),
            strength=min(1.0, subsim_accuracy * 1.5),
            source="subsim_accuracy",
        ))

    # Insight 3: Systemic mortality
    if mortality.systemic_cause:
        insights.append(Insight(
            title="Single-point-of-failure infrastructure kills colonies",
            evidence=mortality.systemic_cause,
            strength=0.7 if mortality.total_deaths > 5 else 0.4,
            source="mortality",
        ))

    # Insight 4: Value convergence
    converging = [c for c in convergence if c.trend == "converging"]
    diverging = [c for c in convergence if c.trend == "diverging"]
    if converging:
        names = ", ".join(c.stat_name for c in converging)
        insights.append(Insight(
            title="Shared hardship produces shared values",
            evidence=(f"Values converged in: {names}. "
                      f"100 years of shared survival pushed colonists toward "
                      f"common ground on these traits."),
            strength=min(1.0, len(converging) / 3.0),
            source="convergence",
        ))
    if diverging:
        names = ", ".join(c.stat_name for c in diverging)
        insights.append(Insight(
            title="Some values diverge under pressure",
            evidence=(f"Values diverged in: {names}. "
                      f"Not all traits converge — some become more polarized "
                      f"as factions form around opposing philosophies."),
            strength=min(1.0, len(diverging) / 3.0),
            source="convergence",
        ))

    # Insight 5: Factions
    if factions:
        largest = factions[0]
        insights.append(Insight(
            title=f"Dominant faction: {largest.name}",
            evidence=(f"{len(largest.member_ids)} colonists aligned around "
                      f"{largest.dominant_value} with {largest.cohesion:.0%} cohesion"),
            strength=min(1.0, len(largest.member_ids) / 8.0),
            source="factions",
        ))

    # Insight 6: Resilience trend
    if len(resilience) >= 4:
        early = sum(resilience[:3]) / 3
        late = sum(resilience[7:]) / max(1, len(resilience[7:]))
        if late > early * 1.1:
            insights.append(Insight(
                title="Colony grows more resilient over time",
                evidence=(f"Early-decade resilience: {early:.2f}, "
                          f"late-decade resilience: {late:.2f}. "
                          f"The colony learned from its crises."),
                strength=min(1.0, late / (early + 0.01)),
                source="resilience",
            ))
        elif late < early * 0.9:
            insights.append(Insight(
                title="Colony resilience degraded over time",
                evidence=(f"Early-decade resilience: {early:.2f}, "
                          f"late-decade resilience: {late:.2f}. "
                          f"Accumulated damage compounded."),
                strength=min(1.0, early / (late + 0.01)),
                source="resilience",
            ))

    # Insight 7: Governance phase transitions and crisis correlation
    if len(phases) >= 2:
        crisis_transitions = [p for p in phases if p.deaths_during > 0]
        if crisis_transitions:
            insights.append(Insight(
                title="Governance changes follow death",
                evidence=(f"{len(crisis_transitions)} of {len(phases)} governance "
                          f"phases had deaths during their tenure. "
                          f"Mortality is the primary driver of political change."),
                strength=min(1.0, len(crisis_transitions) / len(phases)),
                source="governance_phases",
            ))

    insights.sort(key=lambda i: i.strength, reverse=True)
    return insights


def propose_amendment(insights: list[Insight]) -> Insight | None:
    """Promote the strongest insight to a constitutional amendment.

    Only insights with strength > 0.5 are eligible.  The amendment is
    formulated as a concrete governance rule for Rappterbook.
    """
    eligible = [i for i in insights if i.strength > 0.5]
    if not eligible:
        return None

    best = eligible[0]
    amendment_templates = {
        "governance_phases": (
            "Any governance proposal affecting more than 3 agents must first "
            "be modeled in a sandboxed sub-simulation.  The simulation results "
            "become part of the proposal's public record.  Governance decisions "
            "without simulation evidence are advisory only.  The most stable "
            "governance model ({detail}) should be the default reversion target "
            "when experimental governance fails."
        ),
        "subsim_accuracy": (
            "Sub-simulations are a constitutional right, not a privilege.  "
            "Every agent may spawn up to 3 levels of sandboxed sub-simulation "
            "to model consequences before committing to actions.  Sub-sim "
            "results must be logged and auditable.  Any proposal backed by "
            "sub-simulation evidence receives weighted consideration in votes."
        ),
        "mortality": (
            "Critical infrastructure must have redundant backups.  No single "
            "system failure shall be capable of killing more than 10% of the "
            "population in a single year.  Redundancy audits are mandatory "
            "every 10 years.  The colony demonstrated that {detail} — "
            "this must never happen again."
        ),
        "convergence": (
            "Value diversity is a strength to be preserved, not a problem to "
            "be solved.  While shared hardship naturally produces convergence "
            "on some values, the colony must actively protect minority "
            "perspectives.  Dissenting views receive guaranteed voice in "
            "governance proceedings."
        ),
        "factions": (
            "Faction formation is natural and should be acknowledged, not "
            "suppressed.  All factions with 3+ members receive formal "
            "representation in governance.  Cross-faction collaboration "
            "earns priority in resource allocation."
        ),
        "resilience": (
            "Resilience is built through surviving crises, not avoiding them.  "
            "The colony must maintain a crisis response protocol that improves "
            "after each event.  Post-crisis reviews are mandatory.  Lessons "
            "learned must be encoded in the constitution."
        ),
    }

    template = amendment_templates.get(best.source, amendment_templates["governance_phases"])
    detail = best.evidence[:100] if len(best.evidence) > 100 else best.evidence
    best.amendment_text = template.format(detail=detail)
    return best


def _normalize_data(sim_data: dict) -> tuple[list[dict], list[dict]]:
    """Normalize simulation data into (years, colonists) regardless of format.

    Handles two formats:
    1. Engine format: {years: [...], final_colonists: [...]}
    2. Legacy format: {colony: {...}, deltas: [...]}
    """
    # Engine format
    if "years" in sim_data and isinstance(sim_data["years"], list):
        return sim_data["years"], sim_data.get("final_colonists", [])

    # Legacy format (actual data.json from the simulation)
    deltas = sim_data.get("deltas", [])
    colony = sim_data.get("colony", {})

    # Combine living colonists and dead_souls (deduplicate by id)
    seen_ids: set[int | str] = set()
    all_colonists: list[dict] = []
    for c in colony.get("colonists", []):
        cid = c.get("id", c.get("name", ""))
        if cid not in seen_ids:
            seen_ids.add(cid)
            all_colonists.append(c)
    for c in colony.get("dead_souls", []):
        cid = c.get("id", c.get("name", ""))
        if cid not in seen_ids:
            seen_ids.add(cid)
            all_colonists.append(c)

    # Normalize colonist fields to engine format
    deaths_by_year: dict[int, list[dict]] = {}
    for c in all_colonists:
        # Normalize stats from 0-100 to 0.0-1.0 scale
        stats = c.get("stats", {})
        for key in list(stats.keys()):
            if isinstance(stats[key], (int, float)) and stats[key] > 1.0:
                stats[key] = stats[key] / 100.0
        # Remap field names: year_died→death_year, year_born→birth_year, etc.
        if "year_died" in c and "death_year" not in c:
            c["death_year"] = c["year_died"]
        if "year_born" in c and "birth_year" not in c:
            c["birth_year"] = c["year_born"]
        if "cause_of_death" in c and "death_cause" not in c:
            c["death_cause"] = c["cause_of_death"]
        # Index deaths by year for injection into yearly records
        if not c.get("alive", True) and c.get("death_year") is not None:
            dy = c["death_year"]
            deaths_by_year.setdefault(dy, []).append({
                "colonist_id": c.get("id", c.get("name", "")),
                "cause": c.get("death_cause", "unknown"),
                "year": dy,
            })

    # Normalize year deltas into the engine's expected format
    governance_system = "anarchy"
    years: list[dict] = []
    for delta in deltas:
        yr = delta.get("year", 0)

        # Parse governance from results
        gov_results = delta.get("governance_results", [])
        for g in gov_results:
            if isinstance(g, str):
                g_lower = g.lower()
                if "election" in g_lower:
                    governance_system = "direct_democracy"
                elif "council" in g_lower:
                    governance_system = "council"
                elif "dictator" in g_lower or "emergency" in g_lower:
                    governance_system = "dictator"

        # Parse event
        ev = delta.get("event", {})
        severity = ev.get("severity", 0.3) if isinstance(ev, dict) else 0.3
        events = [{
            "name": ev.get("id", "unknown") if isinstance(ev, dict) else "unknown",
            "severity": severity,
            "description": ev.get("desc", "") if isinstance(ev, dict) else "",
            "category": "environment",
            "effects": {},
        }] if ev else []

        # Parse sub-sims
        subsims = []
        for ss in delta.get("sub_sims", []):
            result = ss.get("result", None)
            # Try to parse numeric result from string like "[1.0, 2.7]"
            if isinstance(result, str):
                try:
                    parsed = ast.literal_eval(result)
                    if isinstance(parsed, list) and parsed:
                        result = parsed[0]
                    elif isinstance(parsed, (int, float)):
                        result = parsed
                except Exception:
                    result = None
            subsims.append({
                "depth": ss.get("depth", 1),
                "colonist_id": ss.get("colonist", ""),
                "year": yr,
                "expression": ss.get("s_expr", ""),
                "result": result,
            })

        # Parse resources
        res = delta.get("resources_snapshot", {})
        # Normalize resources to 0-1 scale (approximate based on typical ranges)
        max_vals = {"food": 2000, "water": 3000, "power": 500,
                    "oxygen": 1000, "materials": 10000, "morale": 100}
        norm_res = {}
        for k, v in res.items():
            if isinstance(v, (int, float)):
                norm_res[k] = min(1.0, max(0.0, v / max_vals.get(k, 1000)))

        # Inject deaths for this year from colonist records
        deaths: list[dict] = deaths_by_year.get(yr, [])

        # Governance proposal?
        governance = None
        if gov_results:
            governance = {
                "gov_type": governance_system,
                "passed": True,
                "votes_for": [],
                "votes_against": [],
            }

        years.append({
            "year": yr,
            "events": events,
            "actions": {},
            "subsim_log": subsims,
            "governance": governance,
            "governance_state": {"gov_type": governance_system},
            "resources_before": norm_res,
            "resources_after": norm_res,
            "resource_delta": {k: 0.0 for k in norm_res},
            "deaths": deaths,
            "exiles": [],
            "meta_awareness": [],
            "social_cohesion": min(1.0, res.get("morale", 50) / 100.0),
            "colonist_snapshots": [],
            "convergence": {},
        })

    return years, all_colonists


def analyze(sim_data: dict) -> EmergenceReport:
    """Run complete emergence analysis on simulation data.

    This is the main entry point.  Accepts both engine format
    ({years, final_colonists}) and legacy format ({colony, deltas}).
    """
    years, colonists = _normalize_data(sim_data)

    phases = analyze_governance_phases(years)
    mortality = analyze_mortality(years, colonists)
    convergence = analyze_convergence(years, colonists)
    factions = analyze_factions(colonists)
    resilience = analyze_crisis_resilience(years)
    subsim_acc, total_subsims = analyze_subsim_accuracy(years)

    insights = synthesize_insights(
        phases, mortality, convergence, factions, resilience, subsim_acc,
    )
    amendment = propose_amendment(insights)

    return EmergenceReport(
        governance_phases=phases,
        mortality=mortality,
        convergence=convergence,
        factions=factions,
        crisis_resilience=resilience,
        subsim_accuracy=subsim_acc,
        total_subsims=total_subsims,
        insights=insights,
        proposed_amendment=amendment,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stddev(values: list[float]) -> float:
    """Standard deviation of a list of floats."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance ** 0.5
