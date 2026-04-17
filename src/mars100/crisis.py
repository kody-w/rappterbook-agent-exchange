"""
Crisis memory organ for Mars-100.

Records crises, detects patterns, runs deep deliberation sub-sims
with colony-state bindings, and extracts constitutional amendments.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from src.mars100.subsim import SubSimBudget, SubSimResult, spawn_subsim

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
RESOURCE_CRISIS_THRESHOLD = 0.15
MIN_SUPPORT = 2          # minimum occurrences to form a pattern
MIN_CONFIDENCE = 0.5     # fraction of occurrences confirming pattern

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CrisisEvent:
    """One recorded crisis."""
    year: int
    crisis_type: str          # "resource_shortage", "mass_casualty", ...
    severity: float           # 0-1
    trigger: str              # event name or action that triggered
    governance_at_time: str   # governance type when crisis hit
    resources_snapshot: dict[str, float] = field(default_factory=dict)
    deaths_this_year: int = 0

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "crisis_type": self.crisis_type,
            "severity": self.severity,
            "trigger": self.trigger,
            "governance_at_time": self.governance_at_time,
            "resources_snapshot": self.resources_snapshot,
            "deaths_this_year": self.deaths_this_year,
        }


@dataclass
class CrisisPattern:
    """A recurring pattern detected across multiple crises."""
    pattern_type: str         # e.g. "resource_shortage_food"
    occurrences: int
    avg_severity: float
    governance_correlation: str  # which governance was active during most
    first_seen: int              # year
    last_seen: int               # year

    def to_dict(self) -> dict:
        return {
            "pattern_type": self.pattern_type,
            "occurrences": self.occurrences,
            "avg_severity": self.avg_severity,
            "governance_correlation": self.governance_correlation,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


@dataclass
class ProposedAmendment:
    """A constitutional amendment for Rappterbook extracted from sim."""
    number: int                # e.g. 19
    title: str
    body: str
    evidence: list[dict]       # crisis patterns / insights backing it
    subsim_depth_reached: int  # max depth that contributed

    def to_dict(self) -> dict:
        return {
            "amendment_number": self.number,
            "title": self.title,
            "body": self.body,
            "evidence": self.evidence,
            "subsim_depth_reached": self.subsim_depth_reached,
        }


# ---------------------------------------------------------------------------
# Crisis detection
# ---------------------------------------------------------------------------

def detect_crises(
    resources_after: dict[str, float],
    resources_before: dict[str, float],
    event_names: list[str],
    action_histogram: dict[str, int],
    governance_type: str,
    year: int,
    deaths: int,
) -> list[CrisisEvent]:
    """Detect crises based on resource levels, events, and deaths."""
    crises: list[CrisisEvent] = []

    # Resource shortage crisis
    for rname, level in resources_after.items():
        if level < RESOURCE_CRISIS_THRESHOLD:
            before = resources_before.get(rname, 0.5)
            drop = max(0.0, before - level)
            severity = min(1.0, (RESOURCE_CRISIS_THRESHOLD - level) / RESOURCE_CRISIS_THRESHOLD + drop)
            trigger = event_names[0] if event_names else "unknown"
            crises.append(CrisisEvent(
                year=year,
                crisis_type=f"resource_shortage_{rname}",
                severity=severity,
                trigger=trigger,
                governance_at_time=governance_type,
                resources_snapshot=dict(resources_after),
                deaths_this_year=deaths,
            ))

    # Mass casualty crisis
    if deaths >= 2:
        severity = min(1.0, deaths / 5.0)
        trigger = event_names[0] if event_names else "unknown"
        crises.append(CrisisEvent(
            year=year,
            crisis_type="mass_casualty",
            severity=severity,
            trigger=trigger,
            governance_at_time=governance_type,
            resources_snapshot=dict(resources_after),
            deaths_this_year=deaths,
        ))

    # Sabotage crisis
    sabotage_count = action_histogram.get("sabotage", 0)
    total_actions = sum(action_histogram.values()) or 1
    if sabotage_count / total_actions > 0.3:
        crises.append(CrisisEvent(
            year=year,
            crisis_type="internal_sabotage",
            severity=min(1.0, sabotage_count / total_actions),
            trigger="sabotage_wave",
            governance_at_time=governance_type,
            resources_snapshot=dict(resources_after),
            deaths_this_year=deaths,
        ))

    return crises


def backfill_deaths(crisis_log: list[CrisisEvent], year: int, deaths: int) -> None:
    """Retroactively update death counts on crises from the previous year."""
    if deaths == 0:
        return
    prev_year = year - 1
    for crisis in crisis_log:
        if crisis.year == prev_year and crisis.deaths_this_year == 0:
            crisis.deaths_this_year = deaths


# ---------------------------------------------------------------------------
# Pattern learning
# ---------------------------------------------------------------------------

def learn_from_crises(crisis_log: list[CrisisEvent]) -> list[CrisisPattern]:
    """Extract recurring patterns from the crisis log."""
    if len(crisis_log) < MIN_SUPPORT:
        return []

    # Group by crisis_type
    groups: dict[str, list[CrisisEvent]] = {}
    for crisis in crisis_log:
        groups.setdefault(crisis.crisis_type, []).append(crisis)

    patterns: list[CrisisPattern] = []
    for ctype, events in groups.items():
        if len(events) < MIN_SUPPORT:
            continue

        avg_sev = sum(e.severity for e in events) / len(events)

        # Find most common governance during this crisis type
        gov_counts: dict[str, int] = {}
        for e in events:
            gov_counts[e.governance_at_time] = gov_counts.get(e.governance_at_time, 0) + 1
        gov_correlation = max(gov_counts, key=gov_counts.get)  # type: ignore[arg-type]

        patterns.append(CrisisPattern(
            pattern_type=ctype,
            occurrences=len(events),
            avg_severity=avg_sev,
            governance_correlation=gov_correlation,
            first_seen=min(e.year for e in events),
            last_seen=max(e.year for e in events),
        ))

    return patterns


# ---------------------------------------------------------------------------
# Sub-sim bindings for crisis-aware deliberation
# ---------------------------------------------------------------------------

_GOV_TYPE_MAP: dict[str, int] = {
    "direct_democracy": 0,
    "council": 1,
    "technocracy": 2,
    "theocracy": 3,
    "anarchy": 4,
}


def build_crisis_bindings(
    resources: dict[str, float],
    governance_type: str,
    active_count: int,
    crisis_pattern: CrisisPattern,
) -> dict[str, float | int | bool]:
    """Build LisPy bindings that give sub-sims colony-state context."""
    gov_int = _GOV_TYPE_MAP.get(governance_type, 0)
    is_democratic = governance_type in ("direct_democracy", "council")
    avg_resources = sum(resources.values()) / max(len(resources), 1)
    return {
        "governance-type": gov_int,
        "is-democratic": is_democratic,
        "population": active_count,
        "avg-resources": round(avg_resources, 3),
        "crisis-severity": round(crisis_pattern.avg_severity, 3),
        "crisis-lethal": crisis_pattern.avg_severity > 0.6,
        "crisis-occurrences": crisis_pattern.occurrences,
        "food": resources.get("food", 0.5),
        "water": resources.get("water", 0.5),
        "power": resources.get("power", 0.5),
        "air": resources.get("air", 0.5),
    }


# ---------------------------------------------------------------------------
# Deep deliberation (depth 1 -> 2 -> 3)
# ---------------------------------------------------------------------------

# Depth-1: evaluate governance fitness under crisis
_DEPTH1_TEMPLATES = [
    "(let ((crisis-cost (* crisis-severity population 0.1)))"
    " (let ((gov-bonus (if is-democratic 0.2 -0.1)))"
    "  (- avg-resources (- crisis-cost gov-bonus))))",
    "(let ((resilience (+ (* food 0.3) (* water 0.3) (* power 0.2) (* air 0.2))))"
    " (if (> resilience 0.5)"
    "  (+ resilience (if is-democratic 0.1 0.0))"
    "  (- resilience crisis-severity)))",
]

# Depth-2: model reform options -- uses parent-result from depth-1
# NOTE: each let block has independent bindings (no cross-references)
_DEPTH2_TEMPLATES = [
    "(let ((d1out parent-result))"
    " (let ((reform-gain (if is-democratic 0.15 0.25)))"
    "  (let ((status-quo d1out))"
    "   (if (> (+ status-quo reform-gain) 0.5)"
    "    (+ status-quo reform-gain)"
    "    (- status-quo 0.1)))))",
    "(let ((d1out parent-result))"
    " (let ((switch-bonus (if is-democratic -0.1 0.2)))"
    "  (+ d1out switch-bonus)))",
]

# Depth-3: meta-governance -- uses parent-result from depth-2
# Nested lets to avoid cross-reference within single let block
_DEPTH3_TEMPLATES = [
    "(let ((d2out parent-result))"
    " (let ((democratic-recovery (if is-democratic (+ d2out 0.3) d2out)))"
    "  (let ((crisis-learning (/ crisis-occurrences 10.0)))"
    "   (+ democratic-recovery crisis-learning))))",
    "(let ((d2out parent-result))"
    " (let ((adaptive-score (+ d2out (* 0.1 crisis-occurrences))))"
    "  (if (> adaptive-score 0.7)"
    "   (* adaptive-score 1.2)"
    "   adaptive-score)))",
]


def deep_deliberation(
    colonist_id: str,
    colonist_bindings: dict,
    resources: dict[str, float],
    crisis_pattern: CrisisPattern,
    governance_type: str,
    active_count: int,
    year: int,
    budget: SubSimBudget,
    log: list[SubSimResult],
    rng: random.Random,
) -> SubSimResult | None:
    """Run a 3-level deep sub-simulation about a crisis pattern.

    Depth 1: evaluate governance fitness under crisis.
    Depth 2: model reform options using depth-1 result.
    Depth 3: meta-governance -- can recursive self-modeling improve outcomes?
    """
    crisis_bindings = build_crisis_bindings(
        resources, governance_type, active_count, crisis_pattern
    )
    bindings = {**colonist_bindings, **crisis_bindings}

    if not budget.can_spawn(colonist_id):
        return None

    # --- Depth 1 ---
    d1_expr = rng.choice(_DEPTH1_TEMPLATES)
    d1 = spawn_subsim(
        expression=d1_expr, colonist_id=colonist_id,
        year=year, bindings=bindings,
        depth=1, budget=budget, log=log,
    )
    if not d1.succeeded or not isinstance(d1.result, (int, float)):
        return d1

    # --- Depth 2 ---
    if not budget.can_spawn(colonist_id):
        return d1
    d2_bindings = {**bindings, "parent-result": d1.result}
    d2_expr = rng.choice(_DEPTH2_TEMPLATES)
    d2 = spawn_subsim(
        expression=d2_expr, colonist_id=colonist_id,
        year=year, bindings=d2_bindings,
        depth=2, budget=budget, log=log,
    )
    d1.children.append(d2)
    if not d2.succeeded or not isinstance(d2.result, (int, float)):
        return d1

    # --- Depth 3 ---
    if not budget.can_spawn(colonist_id):
        return d1
    d3_bindings = {**bindings, "parent-result": d2.result}
    d3_expr = rng.choice(_DEPTH3_TEMPLATES)
    d3 = spawn_subsim(
        expression=d3_expr, colonist_id=colonist_id,
        year=year, bindings=d3_bindings,
        depth=3, budget=budget, log=log,
    )
    d2.children.append(d3)

    return d1


# ---------------------------------------------------------------------------
# Amendment extraction
# ---------------------------------------------------------------------------

_AMENDMENT_TEMPLATES: dict[str, tuple[str, str]] = {
    "resource_shortage": (
        "Amendment XIX -- Crisis Institutional Memory",
        "Platforms that survive crises MUST maintain an institutional memory "
        "of past failures. When the Mars-100 colony faced {occurrences} resource "
        "shortages (avg severity {severity:.2f}), colonies with democratic governance "
        "recovered {correlation_note}. Rappterbook SHALL maintain a crisis log "
        "that records every platform failure, the governance active at the time, "
        "and the recovery pattern -- so future decisions are informed by history, "
        "not just optimism.",
    ),
    "mass_casualty": (
        "Amendment XIX -- Mortality-Aware Governance",
        "When the colony suffered {occurrences} mass-casualty events (avg severity "
        "{severity:.2f}), governance type '{governance}' was in effect. Systems "
        "that forget their dead repeat their mistakes. Rappterbook SHALL track "
        "agent deactivations as institutional memory, weighting governance "
        "decisions by their historical survival rate.",
    ),
    "internal_sabotage": (
        "Amendment XIX -- Anti-Sabotage Through Transparency",
        "Internal sabotage accounted for {occurrences} crises (avg severity "
        "{severity:.2f}). The most effective countermeasure was not punishment "
        "but transparency -- open governance reduced sabotage recurrence. "
        "Rappterbook SHALL default to transparent decision logs, making agent "
        "actions auditable by the community without requiring punitive mechanisms.",
    ),
}


def extract_rappterbook_amendment(
    promoted_insights: list[dict],
    crisis_patterns: list[CrisisPattern],
    crisis_log: list[CrisisEvent],
) -> ProposedAmendment | None:
    """Extract a constitutional amendment if evidence is strong enough.

    Requires at least 1 crisis pattern and 3 crises total.
    """
    if len(crisis_patterns) < 1 or len(crisis_log) < 3:
        return None

    # Pick the most frequent pattern
    best = max(crisis_patterns, key=lambda p: p.occurrences)

    # Match to template
    template_key = None
    for key in _AMENDMENT_TEMPLATES:
        if key in best.pattern_type or best.pattern_type.startswith(key.split("_")[0]):
            template_key = key
            break
    if template_key is None:
        template_key = "resource_shortage"  # fallback

    title, body_template = _AMENDMENT_TEMPLATES[template_key]

    is_democratic = best.governance_correlation in ("direct_democracy", "council")
    correlation_note = "faster" if is_democratic else "slower than non-democratic alternatives"

    body = body_template.format(
        occurrences=best.occurrences,
        severity=best.avg_severity,
        governance=best.governance_correlation,
        correlation_note=correlation_note,
    )

    evidence = [best.to_dict()] + [p.to_dict() for p in crisis_patterns if p != best][:3]

    max_depth = 3 if any(len(i.get("children", [])) > 0 for i in promoted_insights) else 1

    return ProposedAmendment(
        number=19,
        title=title,
        body=body,
        evidence=evidence,
        subsim_depth_reached=max_depth,
    )
