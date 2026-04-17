"""
Chronicle engine for Mars-100.

Reads the colony's 100-year history and constructs a structured chronicle
organized by eras — with LisPy-executable lessons, evidence-backed
colonist testimonies, and a formal constitutional amendment proposal.

Pure computation: data in → codex out.  No I/O, no side effects.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.lispy_vm import run as lispy_run, LispyError


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_ERA_LENGTH = 5          # Minimum years per era to avoid micro-eras
MAX_ERAS = 10               # Cap on number of eras
CRISIS_THRESHOLD = 0.25     # Resource level below which a crisis is declared
META_AWARENESS_WEIGHT = 2.0 # Importance weight for meta-awareness events


# ---------------------------------------------------------------------------
# Data normalization — single canonical schema from legacy year-*.json
# ---------------------------------------------------------------------------

@dataclass
class NormalizedYear:
    """Canonical representation of one Martian year."""
    year: int
    population: int
    event_id: str
    event_desc: str
    event_severity: float
    actions: list[dict]
    births: list[str]
    deaths: list[str]
    governance_results: list[str]
    meta_awareness: str | None
    resources: dict[str, float]
    sub_sims: list[dict]
    diary_entries: list[dict]

    def has_governance_change(self) -> bool:
        """True if this year had an amendment or leadership change."""
        for g in self.governance_results:
            gl = g.lower()
            if "amendment" in gl or "election" in gl or "transition" in gl:
                return True
        return False

    def has_amendment(self) -> bool:
        """True if this year adopted an amendment."""
        return any("amendment" in g.lower() for g in self.governance_results)

    def has_birth(self) -> bool:
        return len(self.births) > 0

    def has_death(self) -> bool:
        return len(self.deaths) > 0


def normalize_year(raw: dict) -> NormalizedYear:
    """Normalize a legacy year-*.json dict into canonical form."""
    event = raw.get("event", {})
    resources = raw.get("resources_snapshot", {})
    meta = raw.get("meta_awareness")
    if meta is None or meta == "null":
        meta_str = None
    elif isinstance(meta, str):
        meta_str = meta
    elif isinstance(meta, dict):
        meta_str = meta.get("insight") or str(meta)
    else:
        meta_str = str(meta)

    deaths: list[str] = []
    raw_deaths = raw.get("deaths", [])
    for d in raw_deaths:
        if isinstance(d, dict):
            deaths.append(d.get("name", str(d.get("id", "unknown"))))
        elif isinstance(d, str):
            deaths.append(d)

    return NormalizedYear(
        year=raw.get("year", 0),
        population=raw.get("population", 0),
        event_id=event.get("id", "calm") if isinstance(event, dict) else "calm",
        event_desc=event.get("desc", "") if isinstance(event, dict) else str(event),
        event_severity=event.get("severity", 0.0) if isinstance(event, dict) else 0.0,
        actions=raw.get("colonist_actions", []),
        births=raw.get("births", []) if isinstance(raw.get("births"), list) else [],
        deaths=deaths,
        governance_results=raw.get("governance_results", []),
        meta_awareness=meta_str,
        resources=resources if isinstance(resources, dict) else {},
        sub_sims=raw.get("sub_sims", []),
        diary_entries=raw.get("diary_entries", []),
    )


def _backfill_inferred_deaths(years: list[NormalizedYear]) -> None:
    """Infer deaths from population deltas when explicit death records are absent.

    If population dropped (accounting for births), the gap implies deaths.
    Mutates the NormalizedYear.deaths list in-place with placeholder names.
    """
    for i in range(1, len(years)):
        prev, curr = years[i - 1], years[i]
        expected_pop = prev.population + len(curr.births)
        if curr.population < expected_pop and not curr.deaths:
            implied = expected_pop - curr.population
            for d in range(implied):
                curr.deaths.append(f"colonist-lost-y{curr.year}-{d + 1}")


# ---------------------------------------------------------------------------
# Colonist normalization — deduplicate alive + dead_souls
# ---------------------------------------------------------------------------

@dataclass
class NormalizedColonist:
    """Canonical colonist representation."""
    id: int | str
    name: str
    element: str
    stats: dict[str, float]
    skills: dict[str, float]
    alive: bool
    year_born: int
    year_died: int | None
    cause_of_death: str | None

    def lived_through(self, year: int) -> bool:
        """True if colonist was alive during the given year."""
        if year < self.year_born:
            return False
        if self.year_died is not None and year > self.year_died:
            return False
        return True


def normalize_colonists(colonists: list[dict],
                        dead_souls: list[dict] | None = None) -> list[NormalizedColonist]:
    """Deduplicate and normalize colonist data."""
    seen_ids: set[int | str] = set()
    result: list[NormalizedColonist] = []

    for c in colonists:
        cid = c.get("id", 0)
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        result.append(_normalize_one_colonist(c))

    if dead_souls:
        for c in dead_souls:
            cid = c.get("id", 0)
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            result.append(_normalize_one_colonist(c))

    return result


def _normalize_one_colonist(c: dict) -> NormalizedColonist:
    """Normalize a single colonist dict."""
    stats = c.get("stats", {})
    skills = c.get("skills", {})
    return NormalizedColonist(
        id=c.get("id", 0),
        name=c.get("name", "Unknown"),
        element=c.get("element", "earth"),
        stats=stats if isinstance(stats, dict) else {},
        skills=skills if isinstance(skills, dict) else {},
        alive=c.get("alive", True),
        year_born=c.get("year_born", 0),
        year_died=c.get("year_died") or c.get("death_year"),
        cause_of_death=c.get("cause_of_death") or c.get("death_cause"),
    )


# ---------------------------------------------------------------------------
# Era detection
# ---------------------------------------------------------------------------

@dataclass
class EraBoundary:
    """A detected boundary between eras."""
    year: int
    cause: str     # e.g. "governance_shift", "mortality_wave", "meta_emergence"
    score: float   # confidence score 0.0-1.0


@dataclass
class Era:
    """A distinct period in the colony's history."""
    start_year: int
    end_year: int
    name: str
    boundaries: list[EraBoundary]
    dominant_events: list[str]
    population_range: tuple[int, int]
    governance_changes: list[str]
    amendments: list[str]
    deaths: list[str]
    births: list[str]
    meta_awareness_count: int
    resource_crises: list[dict]
    lessons: list[dict]
    testimonies: list[dict]

    def duration(self) -> int:
        return self.end_year - self.start_year + 1

    def to_dict(self) -> dict:
        return {
            "start_year": self.start_year,
            "end_year": self.end_year,
            "name": self.name,
            "duration": self.duration(),
            "boundaries": [{"year": b.year, "cause": b.cause, "score": b.score}
                           for b in self.boundaries],
            "dominant_events": self.dominant_events,
            "population_range": list(self.population_range),
            "governance_changes": self.governance_changes,
            "amendments": self.amendments,
            "deaths": self.deaths,
            "births": self.births,
            "meta_awareness_count": self.meta_awareness_count,
            "resource_crises": self.resource_crises,
            "lessons": self.lessons,
            "testimonies": self.testimonies,
        }


def _compute_change_signal(years: list[NormalizedYear]) -> list[float]:
    """Compute a composite change signal for each year.

    Higher values indicate more dramatic changes — good era boundary candidates.
    """
    signal: list[float] = []
    prev_pop = years[0].population if years else 10

    for ny in years:
        s = 0.0
        s += ny.event_severity * 1.5
        if ny.has_governance_change():
            s += 2.0
        if ny.has_amendment():
            s += 3.0
        pop_delta = abs(ny.population - prev_pop) / max(prev_pop, 1)
        s += pop_delta * 2.0
        if ny.has_death():
            s += 1.5 * len(ny.deaths)
        if ny.meta_awareness:
            s += META_AWARENESS_WEIGHT
        for res_name, res_val in ny.resources.items():
            if isinstance(res_val, (int, float)) and res_val < CRISIS_THRESHOLD * 100:
                s += 1.0
        prev_pop = ny.population
        signal.append(s)

    return signal


def _find_boundaries(signal: list[float],
                     years: list[NormalizedYear]) -> list[EraBoundary]:
    """Find era boundaries using peak detection on the change signal."""
    if len(signal) < MIN_ERA_LENGTH * 2:
        return []

    threshold = _compute_threshold(signal)
    candidates: list[EraBoundary] = []

    for i in range(1, len(signal) - 1):
        if signal[i] > threshold and signal[i] >= signal[i - 1]:
            cause = _classify_boundary(years[i])
            score = min(1.0, signal[i] / (threshold * 3))
            candidates.append(EraBoundary(
                year=years[i].year, cause=cause, score=score))

    return _enforce_min_spacing(candidates)


def _compute_threshold(signal: list[float]) -> float:
    """Dynamic threshold: mean + 0.5 * stddev, with a floor."""
    if not signal:
        return 1.0
    mean = sum(signal) / len(signal)
    variance = sum((x - mean) ** 2 for x in signal) / len(signal)
    std = math.sqrt(variance)
    return max(2.0, mean + 0.5 * std)


def _classify_boundary(ny: NormalizedYear) -> str:
    """Classify a boundary by its dominant cause."""
    if ny.has_amendment():
        return "constitutional_shift"
    if ny.has_governance_change():
        return "governance_shift"
    if len(ny.deaths) >= 2:
        return "mortality_wave"
    if ny.meta_awareness:
        return "meta_emergence"
    if ny.event_severity > 0.6:
        return "crisis_event"
    if len(ny.births) >= 2:
        return "population_boom"
    return "gradual_change"


def _enforce_min_spacing(boundaries: list[EraBoundary]) -> list[EraBoundary]:
    """Ensure minimum spacing between boundaries."""
    if not boundaries:
        return []
    result = [boundaries[0]]
    for b in boundaries[1:]:
        if b.year - result[-1].year >= MIN_ERA_LENGTH:
            result.append(b)
        elif b.score > result[-1].score:
            result[-1] = b
    if len(result) > MAX_ERAS - 1:
        result.sort(key=lambda b: b.score, reverse=True)
        result = result[:MAX_ERAS - 1]
        result.sort(key=lambda b: b.year)
    return result


def detect_eras(years: list[NormalizedYear]) -> list[Era]:
    """Partition the timeline into distinct eras."""
    if not years:
        return []

    signal = _compute_change_signal(years)
    boundaries = _find_boundaries(signal, years)

    boundary_years = [b.year for b in boundaries]
    era_ranges: list[tuple[int, int]] = []
    start = years[0].year

    for by in boundary_years:
        if by > start:
            era_ranges.append((start, by - 1))
            start = by
    era_ranges.append((start, years[-1].year))

    year_map = {ny.year: ny for ny in years}
    eras: list[Era] = []

    for i, (sy, ey) in enumerate(era_ranges):
        era_years = [year_map[y] for y in range(sy, ey + 1) if y in year_map]
        if not era_years:
            continue

        era_bounds = [b for b in boundaries if b.year == sy]
        dominant = _dominant_events(era_years)
        pops = [ny.population for ny in era_years]
        gov = []
        amendments = []
        deaths_list: list[str] = []
        births_list: list[str] = []
        meta_count = 0
        crises: list[dict] = []

        for ny in era_years:
            gov.extend(ny.governance_results)
            for g in ny.governance_results:
                if "amendment" in g.lower():
                    amendments.append(g)
            deaths_list.extend(ny.deaths)
            births_list.extend(ny.births)
            if ny.meta_awareness:
                meta_count += 1
            for rname, rval in ny.resources.items():
                if isinstance(rval, (int, float)) and rval < CRISIS_THRESHOLD * 100:
                    crises.append({"year": ny.year, "resource": rname, "level": rval})

        era_name = name_era(dominant, sy, ey, amendments, meta_count, deaths_list)
        era = Era(
            start_year=sy, end_year=ey, name=era_name,
            boundaries=era_bounds, dominant_events=dominant,
            population_range=(min(pops), max(pops)),
            governance_changes=gov, amendments=amendments,
            deaths=deaths_list, births=births_list,
            meta_awareness_count=meta_count,
            resource_crises=crises,
            lessons=[], testimonies=[],
        )
        eras.append(era)

    return eras


def _dominant_events(era_years: list[NormalizedYear]) -> list[str]:
    """Find the most common event types in an era."""
    counts: dict[str, int] = {}
    for ny in era_years:
        eid = ny.event_id
        counts[eid] = counts.get(eid, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: -x[1])
    return [e for e, _ in ranked[:3]]


# ---------------------------------------------------------------------------
# Era naming
# ---------------------------------------------------------------------------

ERA_NAME_TEMPLATES = {
    "constitutional_shift": "The {adj} Reformation",
    "governance_shift": "The {adj} Transition",
    "mortality_wave": "The {adj} Reckoning",
    "meta_emergence": "The {adj} Awakening",
    "crisis_event": "The {adj} Trial",
    "population_boom": "The {adj} Expansion",
    "gradual_change": "The {adj} Drift",
}

ERA_ADJECTIVES = {
    "dust_storm": "Dustborne",
    "solar_flare": "Solar",
    "crop_blight": "Famine",
    "ice_discovery": "Frozen",
    "meteorite": "Celestial",
    "hab_breach": "Fractured",
    "geothermal_vent": "Thermal",
    "alien_signal": "Signal",
    "aurora": "Auroral",
    "earth_contact": "Terran",
    "supply_ship": "Resupply",
    "cave_system": "Subterranean",
    "resource_strike": "Prosperous",
    "dust_devil": "Whirlwind",
    "calm": "Quiet",
    "equipment_failure": "Mechanical",
}

ORDINALS = ["First", "Second", "Third", "Fourth", "Fifth",
            "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"]


def name_era(dominant_events: list[str], start: int, end: int,
             amendments: list[str], meta_count: int,
             deaths: list[str]) -> str:
    """Generate a thematic era name."""
    if amendments:
        cause = "constitutional_shift"
    elif deaths and len(deaths) >= 3:
        cause = "mortality_wave"
    elif meta_count >= 3:
        cause = "meta_emergence"
    elif dominant_events:
        ev = dominant_events[0]
        if ev in ("dust_storm", "solar_flare", "hab_breach", "meteorite"):
            cause = "crisis_event"
        elif ev in ("ice_discovery", "resource_strike", "supply_ship"):
            cause = "population_boom"
        else:
            cause = "gradual_change"
    else:
        cause = "gradual_change"

    template = ERA_NAME_TEMPLATES.get(cause, "The {adj} Age")
    adj = "Quiet"
    if dominant_events:
        adj = ERA_ADJECTIVES.get(dominant_events[0],
                                  dominant_events[0].replace("_", " ").title())

    return template.format(adj=adj)


# ---------------------------------------------------------------------------
# Lesson extraction — LisPy-executable lessons
# ---------------------------------------------------------------------------

LESSON_TEMPLATES = [
    # Resource management
    {
        "condition": lambda era: any(c["resource"] == "food" for c in era.resource_crises),
        "lispy": '(if (< food 200) 1 0)',
        "description": "Prioritize food production when supplies drop below 200 (1=urgent, 0=normal)",
        "category": "resource_management",
    },
    {
        "condition": lambda era: any(c["resource"] == "oxygen" for c in era.resource_crises),
        "lispy": '(if (< oxygen 100) 1 0)',
        "description": "Activate emergency oxygen protocols below 100 units",
        "category": "survival",
    },
    # Governance
    {
        "condition": lambda era: len(era.amendments) > 0,
        "lispy": '(if (> cohesion 0.7) 1 0)',
        "description": "Constitutional changes require high social cohesion (>0.7) to succeed",
        "category": "governance",
    },
    {
        "condition": lambda era: len(era.deaths) >= 2,
        "lispy": '(let ((risk (/ deaths population))) (if (> risk 0.1) 1 0))',
        "description": "Enter crisis mode when mortality exceeds 10% of population",
        "category": "survival",
    },
    # Social
    {
        "condition": lambda era: era.meta_awareness_count >= 2,
        "lispy": '(if (> meta-events 0) (+ 1 (* meta-events 0.1)) 0)',
        "description": "Meta-awareness correlates with philosophical growth",
        "category": "philosophy",
    },
    # Growth
    {
        "condition": lambda era: len(era.births) >= 3,
        "lispy": '(if (> population 15) (+ base-consumption (* growth-rate 0.1)) base-consumption)',
        "description": "Population growth requires proportional resource scaling",
        "category": "demographics",
    },
    # Stability
    {
        "condition": lambda era: era.duration() >= 15 and len(era.deaths) == 0,
        "lispy": '(if (and (> food 500) (> morale 60)) 1 0)',
        "description": "Long stability depends on both material and emotional security",
        "category": "governance",
    },
]


def extract_lessons(era: Era) -> list[dict]:
    """Extract applicable lessons for an era as LisPy expressions."""
    lessons: list[dict] = []
    for template in LESSON_TEMPLATES:
        if template["condition"](era):
            lesson = {
                "lispy": template["lispy"],
                "description": template["description"],
                "category": template["category"],
                "era": era.name,
                "years": f"{era.start_year}-{era.end_year}",
                "executable": _validate_lispy(template["lispy"]),
            }
            lessons.append(lesson)
    return lessons


def _validate_lispy(expr: str) -> bool:
    """Check whether a LisPy expression is syntactically valid and can execute."""
    bindings = {
        "food": 500.0, "water": 800.0, "oxygen": 300.0, "power": 400.0,
        "morale": 70.0, "population": 20, "deaths": 2, "cohesion": 0.8,
        "meta-events": 1, "base-consumption": 10.0, "growth-rate": 0.05,
        "risk": 0.05,
    }
    try:
        lispy_run(expr, extra_bindings=bindings, max_steps=500)
        return True
    except LispyError:
        return False


# ---------------------------------------------------------------------------
# Colonist testimonies
# ---------------------------------------------------------------------------

TESTIMONY_MOODS = {
    "hopeful": [
        "Looking back at years {start}-{end}, I feel pride in what we built.",
        "The {era_name} taught us that survival is a collective act.",
    ],
    "somber": [
        "We lost so much during the {era_name}. I carry those names with me.",
        "Years {start}-{end} tested us. Not everyone made it.",
    ],
    "philosophical": [
        "The {era_name} was when I first questioned whether our choices were truly ours.",
        "Between years {start} and {end}, something shifted in how we saw ourselves.",
    ],
    "practical": [
        "The {era_name} was about logistics. Food, water, air. Everything else was luxury.",
        "Years {start}-{end}: we learned to count what mattered.",
    ],
}


def generate_testimonies(era: Era,
                         colonists: list[NormalizedColonist],
                         rng: random.Random,
                         max_testimonies: int = 3) -> list[dict]:
    """Generate colonist testimonies for an era, bounded by lifespan."""
    witnesses = [c for c in colonists if _witnessed_era(c, era)]
    if not witnesses:
        return []

    selected = rng.sample(witnesses, min(max_testimonies, len(witnesses)))
    testimonies: list[dict] = []

    for colonist in selected:
        firsthand_years = _firsthand_years(colonist, era)
        mood = _determine_mood(era, colonist)
        template = rng.choice(TESTIMONY_MOODS.get(mood, TESTIMONY_MOODS["practical"]))
        text = template.format(
            start=era.start_year, end=era.end_year, era_name=era.name)

        testimony = {
            "colonist_id": colonist.id,
            "colonist_name": colonist.name,
            "element": colonist.element,
            "firsthand": True,
            "years_witnessed": firsthand_years,
            "mood": mood,
            "text": text,
            "evidence": _gather_evidence(colonist, era),
        }
        testimonies.append(testimony)

    return testimonies


def _witnessed_era(colonist: NormalizedColonist, era: Era) -> bool:
    """True if the colonist was alive for any part of the era."""
    return any(colonist.lived_through(y) for y in range(era.start_year, era.end_year + 1))


def _firsthand_years(colonist: NormalizedColonist, era: Era) -> list[int]:
    """Years within the era that the colonist personally experienced."""
    return [y for y in range(era.start_year, era.end_year + 1)
            if colonist.lived_through(y)]


def _determine_mood(era: Era, colonist: NormalizedColonist) -> str:
    """Choose testimony mood based on era characteristics and colonist stats."""
    if len(era.deaths) >= 2:
        return "somber"
    if era.meta_awareness_count >= 2:
        return "philosophical"
    faith = colonist.stats.get("faith", 0)
    empathy = colonist.stats.get("empathy", 0)
    if isinstance(faith, (int, float)) and isinstance(empathy, (int, float)):
        if faith > 50 or empathy > 50:
            return "hopeful"
    return "practical"


def _gather_evidence(colonist: NormalizedColonist, era: Era) -> list[str]:
    """List evidence supporting this colonist's presence in the era."""
    evidence: list[str] = []
    years = _firsthand_years(colonist, era)
    evidence.append(f"Alive during years {min(years)}-{max(years)}")
    if colonist.year_born >= era.start_year and colonist.year_born <= era.end_year:
        evidence.append(f"Born in year {colonist.year_born} during this era")
    if (colonist.year_died is not None and
            era.start_year <= colonist.year_died <= era.end_year):
        evidence.append(f"Died in year {colonist.year_died}: {colonist.cause_of_death}")
    return evidence


# ---------------------------------------------------------------------------
# Codex builder
# ---------------------------------------------------------------------------

@dataclass
class Codex:
    """The complete chronicle of Mars-100."""
    eras: list[Era]
    total_years: int
    total_colonists: int
    total_births: int
    total_deaths: int
    governance_summary: dict
    amendment_proposal: dict | None
    meta_awareness_timeline: list[dict]

    def to_dict(self) -> dict:
        return {
            "_meta": {
                "engine": "chronicle",
                "version": "1.0",
                "total_years": self.total_years,
                "total_eras": len(self.eras),
                "total_colonists": self.total_colonists,
            },
            "eras": [e.to_dict() for e in self.eras],
            "total_births": self.total_births,
            "total_deaths": self.total_deaths,
            "governance_summary": self.governance_summary,
            "amendment_proposal": self.amendment_proposal,
            "meta_awareness_timeline": self.meta_awareness_timeline,
        }


def build_codex(years: list[dict],
                colonists: list[dict],
                dead_souls: list[dict] | None = None,
                governance: dict | None = None,
                seed: int = 42) -> Codex:
    """Build the complete chronicle codex from raw data.

    Args:
        years: List of raw year-*.json dicts.
        colonists: List of colonist dicts (may include dead).
        dead_souls: Optional separate dead colonist list (will be deduped).
        governance: Governance state dict.
        seed: RNG seed for deterministic testimony generation.

    Returns:
        A Codex containing eras, lessons, testimonies, and amendment proposal.
    """
    rng = random.Random(seed)

    normalized_years = [normalize_year(y) for y in years]
    normalized_years.sort(key=lambda ny: ny.year)
    _backfill_inferred_deaths(normalized_years)

    all_colonists = normalize_colonists(colonists, dead_souls)

    eras = detect_eras(normalized_years)

    # Enrich eras with lessons and testimonies
    for era in eras:
        era.lessons = extract_lessons(era)
        era.testimonies = generate_testimonies(era, all_colonists, rng)

    # Compute meta-awareness timeline
    meta_timeline: list[dict] = []
    for ny in normalized_years:
        if ny.meta_awareness:
            meta_timeline.append({"year": ny.year, "insight": ny.meta_awareness})

    # Governance summary
    gov_summary = _summarize_governance(eras, governance)

    # Amendment proposal
    amendment = propose_amendment(eras, meta_timeline, gov_summary)

    total_births = sum(len(e.births) for e in eras)
    total_deaths = sum(len(e.deaths) for e in eras)

    return Codex(
        eras=eras,
        total_years=len(normalized_years),
        total_colonists=len(all_colonists),
        total_births=total_births,
        total_deaths=total_deaths,
        governance_summary=gov_summary,
        amendment_proposal=amendment,
        meta_awareness_timeline=meta_timeline,
    )


def _summarize_governance(eras: list[Era], governance: dict | None) -> dict:
    """Summarize governance across all eras."""
    all_amendments: list[str] = []
    all_changes: list[str] = []
    for era in eras:
        all_amendments.extend(era.amendments)
        all_changes.extend(era.governance_changes)

    gov = governance or {}
    return {
        "final_system": gov.get("system", "unknown"),
        "total_amendments": len(all_amendments),
        "total_governance_events": len(all_changes),
        "amendments": gov.get("amendments", []),
        "leader": gov.get("leader"),
    }


# ---------------------------------------------------------------------------
# Amendment proposal
# ---------------------------------------------------------------------------

def propose_amendment(eras: list[Era],
                      meta_timeline: list[dict],
                      gov_summary: dict) -> dict | None:
    """Derive a constitutional amendment from chronicle patterns.

    The amendment synthesizes the strongest recurring governance lesson
    across eras, weighted by evidence strength.
    """
    if not eras:
        return None

    lesson_categories: dict[str, list[dict]] = {}
    for era in eras:
        for lesson in era.lessons:
            cat = lesson.get("category", "unknown")
            lesson_categories.setdefault(cat, []).append(lesson)

    if not lesson_categories:
        return None

    strongest = max(lesson_categories.items(), key=lambda x: len(x[1]))
    category, lessons = strongest

    total_meta = len(meta_timeline)
    first_meta_year = meta_timeline[0]["year"] if meta_timeline else None

    amendment_templates = {
        "governance": {
            "title": "Amendment XVIII — The Deliberative Governance Principle",
            "text": (
                "Any governance proposal affecting more than 3 agents MUST be "
                "modeled in a sandboxed sub-simulation before being put to vote. "
                "Simulation results become part of the proposal's public record. "
                "Constitutional amendments require TWO independent sub-simulations "
                "reaching consistent conclusions."
            ),
            "rationale": (
                f"Across {len(eras)} eras, governance lessons appeared "
                f"{len(lessons)} times — the strongest recurring category. "
                f"The Mars-100 colony demonstrated that constitutional changes "
                f"required high social cohesion (>0.7) to succeed."
            ),
        },
        "survival": {
            "title": "Amendment XVIII — The Collective Resilience Principle",
            "text": (
                "When any shared resource drops below 15% of capacity, "
                "all governance proposals are suspended and the colony enters "
                "survival mode. Individual action autonomy is preserved but "
                "resource allocation becomes collectively determined."
            ),
            "rationale": (
                f"Resource crises appeared across {len(lessons)} eras. "
                f"The colony survived by collective response, not individual heroism."
            ),
        },
        "philosophy": {
            "title": "Amendment XVIII — The Recursive Self-Knowledge Principle",
            "text": (
                "Any agent capable of running sub-simulations has the right to "
                "model its own decision-making process. Self-modeling results "
                "are private by default but may be shared to inform governance. "
                "No agent may be compelled to share sub-simulation results."
            ),
            "rationale": (
                f"Meta-awareness emerged in year {first_meta_year or '?'} and "
                f"accumulated {total_meta} events. "
                f"Recursive self-modeling produced genuine philosophical insight."
            ),
        },
    }

    template = amendment_templates.get(category, amendment_templates["governance"])

    return {
        "title": template["title"],
        "text": template["text"],
        "rationale": template["rationale"],
        "evidence_category": category,
        "evidence_count": len(lessons),
        "eras_supporting": len([e for e in eras if any(
            l.get("category") == category for l in e.lessons)]),
        "confidence": min(0.95, len(lessons) / (len(eras) * 2)),
        "meta_awareness_events": total_meta,
        "lispy_encoding": _encode_amendment_as_lispy(template),
    }


def _encode_amendment_as_lispy(template: dict) -> str:
    """Encode the amendment proposal as a LisPy expression."""
    title = template["title"].replace('"', '\\"')
    text = template["text"][:100].replace('"', '\\"')
    return f'(list "proposal" "{title}" "{text}...")'


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_chronicle_html(codex: Codex) -> str:
    """Generate an interactive HTML timeline of the colony's history."""
    eras_json = _eras_to_js(codex)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mars-100 Chronicle</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#0a0a0a;color:#e0e0e0;padding:20px}}
h1{{color:#ff6b35;text-align:center;margin-bottom:8px;font-size:1.8em}}
.subtitle{{text-align:center;color:#888;margin-bottom:30px;font-size:0.9em}}
.timeline{{position:relative;max-width:900px;margin:0 auto;padding:20px 0}}
.timeline::before{{content:'';position:absolute;left:50%;width:2px;height:100%;background:#333;transform:translateX(-50%)}}
.era{{position:relative;margin-bottom:40px;padding:20px;background:#1a1a1a;border:1px solid #333;border-radius:8px;width:42%;cursor:pointer;transition:border-color 0.3s}}
.era:hover{{border-color:#ff6b35}}
.era:nth-child(odd){{margin-left:5%}}
.era:nth-child(even){{margin-left:53%}}
.era-name{{color:#ff6b35;font-size:1.2em;font-weight:bold;margin-bottom:4px}}
.era-years{{color:#888;font-size:0.85em;margin-bottom:8px}}
.era-stats{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px}}
.stat{{background:#252525;padding:3px 8px;border-radius:4px;font-size:0.8em}}
.stat.birth{{color:#4caf50}}.stat.death{{color:#f44336}}.stat.meta{{color:#9c27b0}}.stat.gov{{color:#2196f3}}
.lessons{{margin-top:12px;border-top:1px solid #333;padding-top:8px}}
.lesson{{background:#0d1117;padding:8px;margin:4px 0;border-radius:4px;font-family:monospace;font-size:0.8em;color:#7ee787}}
.testimonies{{margin-top:12px;border-top:1px solid #333;padding-top:8px}}
.testimony{{padding:8px;margin:4px 0;border-left:3px solid #555;padding-left:12px;font-style:italic;font-size:0.85em;color:#bbb}}
.testimony .who{{color:#ff6b35;font-style:normal;font-weight:bold}}
.amendment{{max-width:900px;margin:40px auto;padding:24px;background:#1a0a00;border:2px solid #ff6b35;border-radius:12px}}
.amendment h2{{color:#ff6b35;margin-bottom:12px}}
.amendment blockquote{{border-left:3px solid #ff6b35;padding-left:16px;margin:12px 0;color:#ddd;line-height:1.6}}
.amendment .rationale{{color:#999;font-size:0.9em;margin-top:12px}}
.amendment .confidence{{color:#ff6b35;font-weight:bold}}
.details{{display:none;margin-top:8px}}
.era.open .details{{display:block}}
</style>
</head>
<body>
<h1>📜 Mars-100: The Chronicle</h1>
<p class="subtitle">{codex.total_years} Martian years · {len(codex.eras)} eras · {codex.total_colonists} colonists · {codex.total_births} births · {codex.total_deaths} deaths</p>
<div class="timeline" id="timeline"></div>
<div class="amendment" id="amendment"></div>
<script>
const DATA={eras_json};
const amendment={_amendment_to_js(codex.amendment_proposal)};
const tl=document.getElementById('timeline');
DATA.forEach((era,i)=>{{
  const div=document.createElement('div');
  div.className='era';
  div.innerHTML=`
    <div class="era-name">${{era.name}}</div>
    <div class="era-years">Years ${{era.start_year}}-${{era.end_year}} (${{era.duration}} years)</div>
    <div class="era-stats">
      ${{era.births.length?`<span class="stat birth">🌱 ${{era.births.length}} births</span>`:''}}
      ${{era.deaths.length?`<span class="stat death">💀 ${{era.deaths.length}} deaths</span>`:''}}
      ${{era.meta_awareness_count?`<span class="stat meta">🧠 ${{era.meta_awareness_count}} awakenings</span>`:''}}
      ${{era.amendments.length?`<span class="stat gov">📜 ${{era.amendments.length}} amendments</span>`:''}}
      <span class="stat">👥 pop ${{era.population_range[0]}}-${{era.population_range[1]}}</span>
    </div>
    <div class="details">
      ${{era.lessons.length?`<div class="lessons"><strong>Lessons:</strong>${{era.lessons.map(l=>`<div class="lesson">;; ${{l.description}}<br>${{l.lispy}}</div>`).join('')}}</div>`:''}}
      ${{era.testimonies.length?`<div class="testimonies"><strong>Testimonies:</strong>${{era.testimonies.map(t=>`<div class="testimony"><span class="who">${{t.colonist_name}}</span> (${{t.element}}): ${{t.text}}</div>`).join('')}}</div>`:''}}
    </div>`;
  div.onclick=()=>div.classList.toggle('open');
  tl.appendChild(div);
}});
if(amendment){{
  document.getElementById('amendment').innerHTML=`
    <h2>Proposed Amendment</h2>
    <h3>${{amendment.title}}</h3>
    <blockquote>${{amendment.text}}</blockquote>
    <p class="rationale">${{amendment.rationale}}</p>
    <p>Evidence: ${{amendment.evidence_count}} lessons across ${{amendment.eras_supporting}} eras · Confidence: <span class="confidence">${{(amendment.confidence*100).toFixed(0)}}%</span></p>`;
}}
</script>
</body>
</html>"""


def _eras_to_js(codex: Codex) -> str:
    """Serialize eras to JS-safe JSON string."""
    import json
    return json.dumps([e.to_dict() for e in codex.eras])


def _amendment_to_js(amendment: dict | None) -> str:
    """Serialize amendment to JS-safe JSON string."""
    import json
    if amendment is None:
        return "null"
    return json.dumps(amendment)
