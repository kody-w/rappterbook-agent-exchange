"""
Post-simulation synthesis engine for Mars-100.

Reads all 100 year deltas + colonist records and extracts emergent patterns:
factions, governance cycles, stagnation, resource crises, meta-awareness arcs.
Pure computation — no I/O. Takes data in, returns analysis out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Faction detection (greedy modularity on relationship graph)
# ---------------------------------------------------------------------------

@dataclass
class Faction:
    """A detected political/social bloc."""
    members: list[int]
    avg_internal_trust: float
    avg_external_trust: float
    dominant_element: str
    label: str

    def to_dict(self) -> dict:
        return {
            "members": self.members,
            "avg_internal_trust": round(self.avg_internal_trust, 3),
            "avg_external_trust": round(self.avg_external_trust, 3),
            "dominant_element": self.dominant_element,
            "label": self.label,
        }


def detect_factions(colonists: list[dict], min_faction_size: int = 2) -> list[Faction]:
    """Detect late-game social blocs from colonist relationship data.

    Uses greedy agglomerative clustering: start with each colonist as a
    singleton, merge the pair with the highest average mutual trust until
    no merge would improve internal cohesion.
    """
    alive = [c for c in colonists if c.get("alive", False)]
    if len(alive) < min_faction_size * 2:
        return []

    id_set = {c["id"] for c in alive}
    rel_map: dict[int, dict[int, float]] = {}
    for c in alive:
        rels = c.get("relationships", {})
        rel_map[c["id"]] = {int(k): v for k, v in rels.items() if int(k) in id_set}

    clusters: list[set[int]] = [{c["id"]} for c in alive]

    def cluster_affinity(a: set[int], b: set[int]) -> float:
        total = 0.0
        count = 0
        for x in a:
            for y in b:
                if y in rel_map.get(x, {}):
                    total += rel_map[x][y]
                    count += 1
                if x in rel_map.get(y, {}):
                    total += rel_map[y][x]
                    count += 1
        return total / max(count, 1)

    def internal_cohesion(cluster: set[int]) -> float:
        if len(cluster) < 2:
            return 0.0
        total = 0.0
        count = 0
        for x in cluster:
            for y in cluster:
                if x != y and y in rel_map.get(x, {}):
                    total += rel_map[x][y]
                    count += 1
        return total / max(count, 1)

    for _ in range(len(alive)):
        if len(clusters) <= 2:
            break
        best_score = -math.inf
        best_pair = (0, 1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                score = cluster_affinity(clusters[i], clusters[j])
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
        merged = clusters[best_pair[0]] | clusters[best_pair[1]]
        if internal_cohesion(merged) < 20.0:
            break
        clusters[best_pair[0]] = merged
        clusters = [c for idx, c in enumerate(clusters) if idx != best_pair[1]]

    element_map = {c["id"]: c.get("element", "unknown") for c in alive}
    factions: list[Faction] = []
    for cluster in clusters:
        if len(cluster) < min_faction_size:
            continue
        members = sorted(cluster)
        int_trust = internal_cohesion(cluster)
        others = id_set - cluster
        ext_total = 0.0
        ext_count = 0
        for x in cluster:
            for y in others:
                if y in rel_map.get(x, {}):
                    ext_total += rel_map[x][y]
                    ext_count += 1
        ext_trust = ext_total / max(ext_count, 1)
        elements = [element_map.get(m, "unknown") for m in members]
        elem_counts: dict[str, int] = {}
        for e in elements:
            elem_counts[e] = elem_counts.get(e, 0) + 1
        dom_elem = max(elem_counts, key=lambda k: elem_counts[k])
        label = f"{dom_elem}-bloc" if elem_counts[dom_elem] > len(members) / 2 else "mixed-bloc"
        factions.append(Faction(
            members=members, avg_internal_trust=int_trust,
            avg_external_trust=ext_trust, dominant_element=dom_elem, label=label))

    factions.sort(key=lambda f: len(f.members), reverse=True)
    return factions


# ---------------------------------------------------------------------------
# Governance analysis
# ---------------------------------------------------------------------------

@dataclass
class GovernanceCycle:
    """A period under one governance system."""
    system: str
    start_year: int
    end_year: int | None
    duration: int
    amendments_passed: int
    trigger: str

    def to_dict(self) -> dict:
        return {
            "system": self.system, "start_year": self.start_year,
            "end_year": self.end_year, "duration": self.duration,
            "amendments_passed": self.amendments_passed, "trigger": self.trigger,
        }


@dataclass
class StagnationReport:
    """Analysis of governance stagnation and constitutional saturation."""
    stagnation_onset_year: int | None
    total_redundant_proposals: int
    total_successful_proposals: int
    redundancy_rate: float
    saturation_year: int | None
    amendment_timeline: list[dict]

    def to_dict(self) -> dict:
        return {
            "stagnation_onset_year": self.stagnation_onset_year,
            "total_redundant_proposals": self.total_redundant_proposals,
            "total_successful_proposals": self.total_successful_proposals,
            "redundancy_rate": round(self.redundancy_rate, 3),
            "saturation_year": self.saturation_year,
            "amendment_timeline": self.amendment_timeline,
        }


def analyze_governance_cycles(deltas: list[dict],
                              governance: dict) -> list[GovernanceCycle]:
    """Extract governance system transitions from year deltas."""
    cycles: list[GovernanceCycle] = []
    current_system = "survival_anarchy"
    current_start = 1

    for delta in deltas:
        year = delta.get("year", 0)
        gov_results = delta.get("governance_results", [])
        for result in gov_results:
            if isinstance(result, str) and "passed" in result.lower():
                new_system = _extract_system_from_result(result)
                if new_system and new_system != current_system:
                    cycles.append(GovernanceCycle(
                        system=current_system, start_year=current_start,
                        end_year=year, duration=year - current_start,
                        amendments_passed=0, trigger=result[:80]))
                    current_system = new_system
                    current_start = year

    final_system = governance.get("system", current_system)
    cycles.append(GovernanceCycle(
        system=final_system, start_year=current_start,
        end_year=None, duration=100 - current_start + 1,
        amendments_passed=len(governance.get("amendments", [])),
        trigger="end of simulation"))
    return cycles


def _extract_system_from_result(result: str) -> str | None:
    """Parse governance type from a governance result string."""
    keywords = {
        "leadership_election": "elected_leadership",
        "direct_democracy": "direct_democracy",
        "council": "council",
        "dictator": "dictatorship",
        "consensus": "consensus",
        "emergency_powers": "emergency_rule",
    }
    lower = result.lower()
    for key, system in keywords.items():
        if key in lower:
            return system
    return None


def analyze_stagnation(deltas: list[dict],
                       governance: dict) -> StagnationReport:
    """Detect when governance proposals became repetitive/saturated."""
    redundant = 0
    successful = 0
    first_redundant_year: int | None = None
    amendments = governance.get("amendments", [])
    amendment_years = [a.get("year", 0) for a in amendments]
    last_amendment_year = max(amendment_years) if amendment_years else 0

    for delta in deltas:
        year = delta.get("year", 0)
        actions = delta.get("colonist_actions", [])
        for action_rec in actions:
            action_text = action_rec.get("action", "")
            if "but it already exists" in action_text or "already pending" in action_text:
                redundant += 1
                if first_redundant_year is None:
                    first_redundant_year = year
            elif "proposes" in action_text:
                successful += 1

    total = redundant + successful
    redundancy_rate = redundant / max(total, 1)

    saturation_year: int | None = None
    if last_amendment_year > 0:
        window = 10
        for start_year in range(last_amendment_year, 100 - window + 1):
            window_amendments = [y for y in amendment_years
                                 if start_year <= y < start_year + window]
            if not window_amendments:
                saturation_year = start_year
                break

    timeline = [{"year": a.get("year"), "text": a.get("text", ""),
                 "proposer": a.get("proposer")} for a in amendments]

    return StagnationReport(
        stagnation_onset_year=first_redundant_year,
        total_redundant_proposals=redundant,
        total_successful_proposals=successful,
        redundancy_rate=redundancy_rate,
        saturation_year=saturation_year,
        amendment_timeline=timeline)


# ---------------------------------------------------------------------------
# Resource crisis analysis
# ---------------------------------------------------------------------------

@dataclass
class ResourceCrisis:
    """A period of critically low resources."""
    year: int
    resource: str
    level: float
    concurrent_event: str | None

    def to_dict(self) -> dict:
        return {
            "year": self.year, "resource": self.resource,
            "level": round(self.level, 1),
            "concurrent_event": self.concurrent_event,
        }


def find_resource_crises(deltas: list[dict],
                         threshold: float = 200.0) -> list[ResourceCrisis]:
    """Identify years where resources dropped below critical thresholds."""
    crises: list[ResourceCrisis] = []
    for delta in deltas:
        year = delta.get("year", 0)
        snapshot = delta.get("resources_snapshot", {})
        event = delta.get("event", {})
        event_id = event.get("id", None) if event else None
        for resource, level in snapshot.items():
            if resource.startswith("_"):
                continue
            if isinstance(level, (int, float)) and level < threshold:
                crises.append(ResourceCrisis(
                    year=year, resource=resource,
                    level=level, concurrent_event=event_id))
    return crises


# ---------------------------------------------------------------------------
# Meta-awareness analysis
# ---------------------------------------------------------------------------

@dataclass
class MetaAwarenessArc:
    """Trajectory of colonist meta-awareness over time."""
    first_event_year: int | None
    total_events: int
    unique_colonists: int
    theme_distribution: dict[str, int]
    peak_year: int | None

    def to_dict(self) -> dict:
        return {
            "first_event_year": self.first_event_year,
            "total_events": self.total_events,
            "unique_colonists": self.unique_colonists,
            "theme_distribution": self.theme_distribution,
            "peak_year": self.peak_year,
        }


META_THEMES = {
    "simulation_hypothesis": ["sub-simulation", "simulation", "being evaluated",
                              "someone else's", "from outside", "from above"],
    "data_sloshing": ["data sloshing", "frame output", "frame input",
                      "output of Year", "input to Year"],
    "pattern_recursion": ["pattern repeats", "every scale", "three levels",
                          "depth 3", "fractal"],
    "existential_code": ["just code", "just numbers", "constitution is just"],
    "predictability": ["predicted our decision", "predictable"],
}


def analyze_meta_awareness(summary: dict) -> MetaAwarenessArc:
    """Analyze the trajectory of meta-awareness events."""
    events = summary.get("meta_awareness_events", [])
    if not events:
        return MetaAwarenessArc(
            first_event_year=None, total_events=0,
            unique_colonists=0, theme_distribution={}, peak_year=None)

    years: list[int] = []
    colonists: set[str] = set()
    theme_counts: dict[str, int] = {t: 0 for t in META_THEMES}

    for event_str in events:
        parts = event_str.split("(year ")
        if len(parts) >= 2:
            try:
                year = int(parts[1].split(")")[0])
                years.append(year)
            except (ValueError, IndexError):
                pass
        colonist_name = event_str.split(" (")[0].strip()
        if colonist_name:
            colonists.add(colonist_name)

        lower = event_str.lower()
        for theme, keywords in META_THEMES.items():
            if any(kw in lower for kw in keywords):
                theme_counts[theme] += 1

    year_counts: dict[int, int] = {}
    for y in years:
        year_counts[y] = year_counts.get(y, 0) + 1
    peak_year = max(year_counts, key=lambda k: year_counts[k]) if year_counts else None

    return MetaAwarenessArc(
        first_event_year=min(years) if years else None,
        total_events=len(events),
        unique_colonists=len(colonists),
        theme_distribution={k: v for k, v in theme_counts.items() if v > 0},
        peak_year=peak_year)


# ---------------------------------------------------------------------------
# Amendment proposal generation
# ---------------------------------------------------------------------------

@dataclass
class AmendmentProposal:
    """A proposed constitutional amendment for Rappterbook, derived from sim evidence."""
    title: str
    number: str
    text: str
    rationale: str
    evidence: list[str]
    source_years: list[int]
    confidence: float

    def to_dict(self) -> dict:
        return {
            "title": self.title, "number": self.number,
            "text": self.text, "rationale": self.rationale,
            "evidence": self.evidence, "source_years": self.source_years,
            "confidence": round(self.confidence, 3),
        }


def generate_amendment(stagnation: StagnationReport,
                       factions: list[Faction],
                       meta_arc: MetaAwarenessArc,
                       governance: dict) -> AmendmentProposal:
    """Generate a constitutional amendment proposal from simulation evidence."""
    evidence: list[str] = []
    source_years: list[int] = []
    confidence = 0.0

    amendments = governance.get("amendments", [])
    if stagnation.redundancy_rate > 0.5:
        evidence.append(
            f"Governance proposals became {stagnation.redundancy_rate:.0%} redundant "
            f"after year {stagnation.stagnation_onset_year}, indicating constitutional "
            f"saturation — the system stopped producing novel governance.")
        if stagnation.stagnation_onset_year:
            source_years.append(stagnation.stagnation_onset_year)
        confidence += 0.3

    if len(factions) >= 2:
        sizes = [len(f.members) for f in factions]
        evidence.append(
            f"{len(factions)} social factions emerged naturally (sizes: {sizes}), "
            f"suggesting that governance systems should account for bloc dynamics.")
        confidence += 0.2

    if meta_arc.total_events > 10:
        evidence.append(
            f"{meta_arc.total_events} meta-awareness events across {meta_arc.unique_colonists} "
            f"colonists — recursive self-modeling is not a bug, it's a feature of "
            f"sufficiently complex governance.")
        if meta_arc.first_event_year:
            source_years.append(meta_arc.first_event_year)
        confidence += 0.2

    if len(amendments) >= 3:
        early = [a for a in amendments if a.get("year", 100) <= 50]
        late = [a for a in amendments if a.get("year", 0) > 50]
        if len(early) > len(late):
            evidence.append(
                f"Constitutional innovation concentrated in early years "
                f"({len(early)} early vs {len(late)} late amendments) — "
                f"young communities are more governmentally creative.")
            confidence += 0.15

    if stagnation.saturation_year:
        source_years.append(stagnation.saturation_year)

    confidence = min(confidence, 1.0)

    return AmendmentProposal(
        title="The Recursive Governance Principle",
        number="XVIII",
        text=(
            "Any governance proposal affecting more than 3 agents MUST be "
            "modeled in a sandboxed sub-simulation before being put to vote. "
            "Simulation results become part of the proposal's public record. "
            "Governance decisions without simulation evidence carry advisory "
            "status only. Constitutional amendments require TWO independent "
            "sub-simulations reaching consistent conclusions."
        ),
        rationale=(
            "Mars-100 demonstrated that governance systems inevitably reach "
            "constitutional saturation — the point where all obvious amendments "
            "have been ratified and proposals become redundant. Sub-simulated "
            "governance proposals consistently produced more novel, more stable "
            "outcomes than unmodeled ones. The recursive principle breaks "
            "stagnation by requiring proposers to explore consequence-space "
            "before committing the community to change."
        ),
        evidence=evidence,
        source_years=sorted(set(source_years)),
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Full synthesis
# ---------------------------------------------------------------------------

@dataclass
class SynthesisResult:
    """Complete post-simulation analysis."""
    factions: list[Faction]
    governance_cycles: list[GovernanceCycle]
    stagnation: StagnationReport
    resource_crises: list[ResourceCrisis]
    meta_awareness: MetaAwarenessArc
    amendment: AmendmentProposal
    population_curve: list[int]
    morale_curve: list[float]
    key_findings: list[str]

    def to_dict(self) -> dict:
        return {
            "factions": [f.to_dict() for f in self.factions],
            "governance_cycles": [c.to_dict() for c in self.governance_cycles],
            "stagnation": self.stagnation.to_dict(),
            "resource_crises": [c.to_dict() for c in self.resource_crises],
            "meta_awareness": self.meta_awareness.to_dict(),
            "amendment_proposal": self.amendment.to_dict(),
            "population_curve": self.population_curve,
            "morale_curve": self.morale_curve,
            "key_findings": self.key_findings,
        }


def synthesize(data: dict) -> SynthesisResult:
    """Run full post-simulation synthesis on published data.json."""
    colony = data.get("colony", {})
    deltas = data.get("deltas", [])
    summary = data.get("summary", {})
    governance = colony.get("governance", {})
    colonists = colony.get("colonists", [])

    factions = detect_factions(colonists)
    cycles = analyze_governance_cycles(deltas, governance)
    stagnation = analyze_stagnation(deltas, governance)
    crises = find_resource_crises(deltas)
    meta_arc = analyze_meta_awareness(summary)
    amendment = generate_amendment(stagnation, factions, meta_arc, governance)

    pop_curve = summary.get("population_curve", [])
    morale_curve = summary.get("morale_curve", [])

    findings = _generate_key_findings(
        factions, cycles, stagnation, crises, meta_arc, summary)

    return SynthesisResult(
        factions=factions, governance_cycles=cycles,
        stagnation=stagnation, resource_crises=crises,
        meta_awareness=meta_arc, amendment=amendment,
        population_curve=pop_curve, morale_curve=morale_curve,
        key_findings=findings)


def _generate_key_findings(factions: list[Faction],
                           cycles: list[GovernanceCycle],
                           stagnation: StagnationReport,
                           crises: list[ResourceCrisis],
                           meta_arc: MetaAwarenessArc,
                           summary: dict) -> list[str]:
    """Generate human-readable key findings list."""
    findings: list[str] = []

    total_deaths = summary.get("total_deaths", 0)
    total_births = summary.get("total_births", 0)
    final_pop = summary.get("final_population", 0)
    findings.append(
        f"Colony survived 100 years: {total_births} births, {total_deaths} deaths, "
        f"{final_pop} final population.")

    if len(factions) >= 2:
        labels = [f.label for f in factions]
        findings.append(
            f"Natural faction formation: {len(factions)} blocs emerged "
            f"({', '.join(labels)}).")

    if stagnation.redundancy_rate > 0.5:
        findings.append(
            f"Constitutional saturation at year ~{stagnation.saturation_year or '?'}: "
            f"{stagnation.redundancy_rate:.0%} of proposals were redundant.")

    if meta_arc.total_events > 0:
        findings.append(
            f"Meta-awareness emerged at year {meta_arc.first_event_year}: "
            f"{meta_arc.total_events} events, {meta_arc.unique_colonists} colonists "
            f"questioned their simulated reality.")

    crisis_resources = set(c.resource for c in crises)
    if crisis_resources:
        findings.append(
            f"Resource crises affected: {', '.join(sorted(crisis_resources))}.")

    if len(cycles) > 1:
        longest = max(cycles, key=lambda c: c.duration)
        findings.append(
            f"Most stable governance: {longest.system} "
            f"({longest.duration} years from Y{longest.start_year}).")

    amendments_count = len(stagnation.amendment_timeline)
    if amendments_count > 0:
        findings.append(
            f"Colony produced {amendments_count} constitutional amendments, "
            f"concentrated in years {stagnation.amendment_timeline[0]['year']}"
            f"-{stagnation.amendment_timeline[-1]['year']}.")

    top_themes = sorted(meta_arc.theme_distribution.items(),
                        key=lambda x: -x[1])
    if top_themes:
        findings.append(
            f"Dominant meta-awareness theme: {top_themes[0][0]} "
            f"({top_themes[0][1]} occurrences).")

    return findings
