"""
Prophecy Engine for Mars-100 colony simulation.

Every 10 years the colony's most analytically gifted colonist analyses
accumulated history and produces predictions using depth-2 sub-simulations.
Prophecies feed back into colonist decision-making, creating a reflexive
intelligence loop — the colony studying itself mid-flight.

Resolution uses three outcomes:
  hit     — prediction matched reality
  averted — prediction was correct but colony response prevented it
  miss    — prediction was wrong

Accurate prophets gain influence; useful warnings (averted) count as
partial credit so the system doesn't punish successful prevention.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.subsim import (
    SubSimBudget, SubSimResult, spawn_subsim, MAX_SUBSIM_DEPTH,
)

PROPHECY_INTERVAL = 10
PROPHECY_LOOKAHEAD_MIN = 5
PROPHECY_LOOKAHEAD_MAX = 10
MAX_PROPHECIES = 20
INFLUENCE_BOUND = 0.3
AVERTED_CREDIT = 0.5

PREDICTION_TYPES = [
    "crisis", "prosperity", "death_wave",
    "governance_shift", "breakthrough",
]


# ---- Resolution predicates --------------------------------------------------

def _is_crisis(year_summary: dict) -> bool:
    """Any resource below 0.2."""
    res = year_summary.get("resources_after", {})
    return any(v < 0.2 for v in res.values()) if res else False


def _is_prosperity(year_summary: dict) -> bool:
    """All resources above 0.6."""
    res = year_summary.get("resources_after", {})
    return all(v > 0.6 for v in res.values()) if res else False


def _is_death_wave(year_summary: dict) -> bool:
    """Two or more deaths in one year."""
    return len(year_summary.get("deaths", [])) >= 2


def _is_governance_shift(year_summary: dict) -> bool:
    """A governance proposal passed."""
    gov = year_summary.get("governance")
    if isinstance(gov, dict):
        return gov.get("passed", False)
    return False


def _is_breakthrough(year_summary: dict) -> bool:
    """An infrastructure project completed."""
    infra = year_summary.get("infrastructure", {})
    return bool(infra.get("just_completed"))


RESOLUTION_PREDICATES: dict[str, Any] = {
    "crisis": _is_crisis,
    "prosperity": _is_prosperity,
    "death_wave": _is_death_wave,
    "governance_shift": _is_governance_shift,
    "breakthrough": _is_breakthrough,
}


# ---- Data types --------------------------------------------------------------

@dataclass
class Prophecy:
    """A single prediction made by a colonist-prophet."""
    prophet_id: str
    year_made: int
    year_target: int
    prediction_type: str
    confidence: float
    subsim_depth: int = 1
    evidence_expr: str = ""
    evidence_result: Any = None
    resolved: bool = False
    outcome: str = ""          # "hit", "averted", "miss", or ""
    colony_response: str = ""  # brief description of colony reaction

    def to_dict(self) -> dict:
        return {
            "prophet_id": self.prophet_id,
            "year_made": self.year_made,
            "year_target": self.year_target,
            "prediction_type": self.prediction_type,
            "confidence": round(self.confidence, 4),
            "subsim_depth": self.subsim_depth,
            "evidence_expr": self.evidence_expr,
            "evidence_result": _safe(self.evidence_result),
            "resolved": self.resolved,
            "outcome": self.outcome,
            "colony_response": self.colony_response,
        }


@dataclass
class ProphecyState:
    """Persistent prophecy state across the simulation."""
    prophecies: list[Prophecy] = field(default_factory=list)
    track_record: dict[str, dict] = field(default_factory=dict)
    current_influence: float = 0.0

    def active_prophecies(self, year: int) -> list[Prophecy]:
        """Prophecies not yet resolved whose target year hasn't passed."""
        return [p for p in self.prophecies
                if not p.resolved and p.year_target >= year]

    def pending_for_year(self, year: int) -> list[Prophecy]:
        """Prophecies targeting exactly this year, not yet resolved."""
        return [p for p in self.prophecies
                if not p.resolved and p.year_target == year]

    def prophet_accuracy(self, colonist_id: str) -> float:
        """Weighted accuracy: hit=1.0, averted=AVERTED_CREDIT, miss=0."""
        rec = self.track_record.get(colonist_id, {})
        total = rec.get("total", 0)
        if total == 0:
            return 0.5
        score = rec.get("hits", 0) + rec.get("averted", 0) * AVERTED_CREDIT
        return min(1.0, score / total)

    def record_outcome(self, colonist_id: str, outcome: str) -> None:
        """Record a prophecy resolution in the track record."""
        rec = self.track_record.setdefault(colonist_id, {
            "hits": 0, "averted": 0, "misses": 0, "total": 0,
        })
        rec["total"] += 1
        if outcome == "hit":
            rec["hits"] += 1
        elif outcome == "averted":
            rec["averted"] += 1
        else:
            rec["misses"] += 1

    def _update_influence(self) -> None:
        """Recompute prophecy influence from track records."""
        if not self.track_record:
            self.current_influence = 0.0
            return
        accuracies = [self.prophet_accuracy(cid)
                      for cid in self.track_record]
        avg = sum(accuracies) / len(accuracies)
        self.current_influence = min(INFLUENCE_BOUND, max(0.0, avg - 0.3))

    def to_dict(self) -> dict:
        return {
            "prophecies": [p.to_dict() for p in self.prophecies],
            "track_record": dict(self.track_record),
            "current_influence": round(self.current_influence, 4),
        }

    def summary(self) -> dict:
        """Compact summary for year results."""
        active = [p for p in self.prophecies if not p.resolved]
        resolved = [p for p in self.prophecies if p.resolved]
        return {
            "total": len(self.prophecies),
            "active": len(active),
            "resolved": len(resolved),
            "influence": round(self.current_influence, 4),
            "latest": self.prophecies[-1].to_dict() if self.prophecies else None,
        }


# ---- Core functions ----------------------------------------------------------

def _safe(value: Any) -> Any:
    """JSON-safe serialization."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_safe(v) for v in value]
    return str(value)


def select_prophet(
    colonists: list,
    track_record: dict[str, dict],
    rng: random.Random,
) -> Any | None:
    """Select the colonist best suited to prophecy.

    Weighted by coding skill + improvisation stat + historical accuracy.
    Returns None if no eligible colonist exists.
    """
    active = [c for c in colonists if c.is_active()]
    if not active:
        return None

    weights: list[float] = []
    for c in active:
        skill_score = c.skills.coding * 0.4 + c.stats.improvisation * 0.3
        rec = track_record.get(c.id, {})
        total = rec.get("total", 0)
        if total > 0:
            hits = rec.get("hits", 0) + rec.get("averted", 0) * AVERTED_CREDIT
            accuracy_bonus = (hits / total) * 0.3
        else:
            accuracy_bonus = 0.15
        weights.append(max(0.01, skill_score + accuracy_bonus))

    total_w = sum(weights)
    r = rng.random() * total_w
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return active[i]
    return active[-1]


def _analyse_trend(history: list[dict], resource: str) -> float:
    """Compute average change per year for a resource over recent history."""
    values = []
    for h in history:
        res = h.get("resources_after", {})
        if resource in res:
            values.append(res[resource])
    if len(values) < 2:
        return 0.0
    deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    return sum(deltas) / len(deltas)


def _build_projection_expr(trends: dict[str, float],
                           current: dict[str, float]) -> str:
    """Build a LisPy expression projecting resources forward."""
    bindings: list[str] = []
    projections: list[str] = []
    for res, trend in trends.items():
        cur = current.get(res, 0.5)
        safe_trend = round(trend, 4)
        safe_cur = round(cur, 4)
        bindings.append(f"({res}-trend {safe_trend})")
        bindings.append(f"({res}-now {safe_cur})")
        projections.append(f"(+ {res}-now (* {res}-trend 5))")

    if not projections:
        return "(+ 0.5 0.0)"

    binding_str = " ".join(bindings)
    min_expr = projections[0]
    for p in projections[1:]:
        min_expr = f"(if (< {p} {min_expr}) {p} {min_expr})"
    return f"(let ({binding_str}) {min_expr})"


def _build_counter_expr(primary_result: float,
                        prophet_skill: float) -> str:
    """Build a LisPy expression for the counter-scenario sub-sim."""
    safe_result = round(primary_result if isinstance(primary_result, (int, float)) else 0.5, 4)
    safe_skill = round(prophet_skill, 4)
    return (
        f"(let ((baseline {safe_result}) (skill {safe_skill})) "
        f"(if (< baseline 0.3) "
        f"(+ baseline (* skill 0.2)) "
        f"(- baseline (* (- 1 skill) 0.1))))"
    )


def generate_prophecy(
    prophet: Any,
    year: int,
    history: list[dict],
    resources: dict[str, float],
    total_years: int,
    subsim_log: list[SubSimResult],
    rng: random.Random,
) -> Prophecy | None:
    """Generate a prophecy using depth-2 sub-simulations.

    Returns None if year_target would exceed total_years.
    """
    lookahead = rng.randint(PROPHECY_LOOKAHEAD_MIN, PROPHECY_LOOKAHEAD_MAX)
    year_target = year + lookahead
    if year_target > total_years:
        return None

    # Prophecy gets its own dedicated budget (doesn't compete with regular sims)
    budget = SubSimBudget(year=year)

    # Analyse resource trends from history
    resource_names = list(resources.keys())
    trends = {r: _analyse_trend(history, r) for r in resource_names}

    # Depth-1: project current trends forward
    projection_expr = _build_projection_expr(trends, resources)
    primary = spawn_subsim(
        expression=projection_expr,
        colonist_id=prophet.id,
        year=year,
        bindings=prophet.lispy_bindings(),
        depth=1,
        budget=budget,
        log=subsim_log,
    )

    primary_value = primary.result if isinstance(primary.result, (int, float)) else 0.5

    # Depth-2: counter-scenario — what if the colony responds?
    counter_expr = _build_counter_expr(primary_value, prophet.skills.coding)
    counter = spawn_subsim(
        expression=counter_expr,
        colonist_id=prophet.id,
        year=year,
        bindings=prophet.lispy_bindings(),
        depth=2,
        budget=budget,
        log=subsim_log,
    )
    primary.children.append(counter)

    counter_value = counter.result if isinstance(counter.result, (int, float)) else 0.5

    # Determine prediction type from projections
    prediction_type = _classify_prediction(
        primary_value, counter_value, trends, rng)
    confidence = _compute_confidence(prophet, primary, counter)

    prophecy = Prophecy(
        prophet_id=prophet.id,
        year_made=year,
        year_target=year_target,
        prediction_type=prediction_type,
        confidence=confidence,
        subsim_depth=2 if counter.succeeded else 1,
        evidence_expr=projection_expr,
        evidence_result={"primary": _safe(primary.result),
                         "counter": _safe(counter.result)},
    )
    return prophecy


def _classify_prediction(
    primary: float, counter: float,
    trends: dict[str, float],
    rng: random.Random,
) -> str:
    """Choose prediction type based on sub-sim results and trends."""
    if primary < 0.2:
        return "crisis"
    if primary > 0.7 and counter > 0.6:
        return "prosperity"
    neg_trends = sum(1 for v in trends.values() if v < -0.02)
    if neg_trends >= 3:
        return "death_wave"
    if abs(primary - counter) > 0.3:
        return "governance_shift"
    if primary > 0.5 and counter > primary:
        return "breakthrough"
    return rng.choice(PREDICTION_TYPES)


def _compute_confidence(
    prophet: Any,
    primary: SubSimResult,
    counter: SubSimResult,
) -> float:
    """Compute prophecy confidence based on prophet skill and sub-sim quality."""
    base = prophet.skills.coding * 0.4 + prophet.stats.improvisation * 0.3
    if primary.succeeded:
        base += 0.15
    if counter.succeeded:
        base += 0.15
    return min(1.0, max(0.1, base))


def resolve_prophecy(
    prophecy: Prophecy,
    year_summary: dict,
    prev_year_summary: dict | None = None,
) -> str:
    """Resolve a prophecy against actual year results.

    Returns "hit", "averted", or "miss".

    Averted detection: if the prediction type matches a resource trend
    that was heading toward the prediction but the colony responded
    (e.g., crisis predicted, resources were low but recovered).
    """
    predicate = RESOLUTION_PREDICATES.get(prophecy.prediction_type)
    if predicate is None:
        prophecy.resolved = True
        prophecy.outcome = "miss"
        return "miss"

    matched = predicate(year_summary)

    if matched:
        prophecy.resolved = True
        prophecy.outcome = "hit"
        return "hit"

    # Check for averted: was the prediction on track to happen?
    if prev_year_summary is not None and _was_trending_toward(
        prophecy.prediction_type, prev_year_summary, year_summary
    ):
        prophecy.resolved = True
        prophecy.outcome = "averted"
        return "averted"

    prophecy.resolved = True
    prophecy.outcome = "miss"
    return "miss"


def _was_trending_toward(
    prediction_type: str,
    prev: dict,
    current: dict,
) -> bool:
    """Heuristic: was the colony trending toward the predicted outcome?"""
    if prediction_type == "crisis":
        prev_res = prev.get("resources_after", {})
        cur_res = current.get("resources_after", {})
        if not prev_res or not cur_res:
            return False
        prev_min = min(prev_res.values())
        cur_min = min(cur_res.values())
        return prev_min < 0.3 and cur_min >= 0.2

    if prediction_type == "death_wave":
        prev_deaths = len(prev.get("deaths", []))
        return prev_deaths >= 1

    return False


def compute_prophecy_influence(state: ProphecyState, year: int) -> dict[str, float]:
    """Compute action weight adjustments from active prophecies.

    Returns a dict mapping action names to weight deltas, bounded by
    INFLUENCE_BOUND.
    """
    adjustments: dict[str, float] = {}
    active = state.active_prophecies(year)
    if not active or state.current_influence <= 0:
        return adjustments

    for prophecy in active:
        proximity = 1.0 / max(1, prophecy.year_target - year)
        strength = prophecy.confidence * state.current_influence * proximity

        if prophecy.prediction_type == "crisis":
            adjustments["farm"] = adjustments.get("farm", 0.0) + strength
            adjustments["cooperate"] = adjustments.get("cooperate", 0.0) + strength * 0.5
            adjustments["hoard"] = adjustments.get("hoard", 0.0) - strength * 0.3
        elif prophecy.prediction_type == "prosperity":
            adjustments["explore"] = adjustments.get("explore", 0.0) + strength
            adjustments["research"] = adjustments.get("research", 0.0) + strength * 0.5
        elif prophecy.prediction_type == "death_wave":
            adjustments["mediate"] = adjustments.get("mediate", 0.0) + strength
            adjustments["cooperate"] = adjustments.get("cooperate", 0.0) + strength
            adjustments["sabotage"] = adjustments.get("sabotage", 0.0) - strength
        elif prophecy.prediction_type == "governance_shift":
            adjustments["mediate"] = adjustments.get("mediate", 0.0) + strength * 0.5
        elif prophecy.prediction_type == "breakthrough":
            adjustments["research"] = adjustments.get("research", 0.0) + strength
            adjustments["code"] = adjustments.get("code", 0.0) + strength * 0.5

    # Clamp all adjustments to INFLUENCE_BOUND
    for action in adjustments:
        adjustments[action] = max(-INFLUENCE_BOUND,
                                  min(INFLUENCE_BOUND, adjustments[action]))

    return adjustments
