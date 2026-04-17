"""
Oral tradition / cultural memory for Mars-100.

When colonists die, their decision-making wisdom is distilled into
*ancestral teachings* — LisPy expressions stored in a colony-wide
cultural memory.  Living colonists can study teachings to gain small
stat/skill boosts.  Teachings decay each year unless actively studied.

This is inter-generational, colony-wide knowledge transfer — distinct
from parent→child genetic inheritance (create_child copies decision_expr)
and individual self-modification (dreaming engine, PR #347).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.lispy_vm import (
    LispyError,
    parse_all,
    _count_nodes,
    MAX_AST_NODES,
)

SKILL_KEYWORDS: dict[str, str] = {
    "terraforming": "terraforming",
    "hydroponics": "hydroponics",
    "mediation": "mediation",
    "coding": "coding",
    "prayer": "prayer",
    "sabotage": "sabotage",
    "resolve": "terraforming",
    "empathy": "mediation",
    "faith": "prayer",
    "improvisation": "coding",
    "hoarding": "hydroponics",
    "paranoia": "sabotage",
}

MAX_ACTIVE_TEACHINGS = 20
STUDY_BOOST = 0.015
DECAY_PER_YEAR = 0.06
MIN_POTENCY = 0.05
STUDY_AGE_GATE = 2


def _extract_symbols(expr: Any) -> set[str]:
    """Extract all symbol names from a parsed LisPy expression."""
    if isinstance(expr, str) and not expr.startswith('"'):
        return {expr}
    if isinstance(expr, list):
        result: set[str] = set()
        for item in expr:
            result |= _extract_symbols(item)
        return result
    return set()


def _infer_skill_focus(decision_expr: str) -> str:
    """Infer the primary skill a teaching emphasises from its LisPy expression."""
    try:
        parsed = parse_all(decision_expr)
    except LispyError:
        return "mediation"
    symbols = set()
    for expr in parsed:
        symbols |= _extract_symbols(expr)
    best: str | None = None
    best_count = 0
    skill_hits: dict[str, int] = {}
    for sym in symbols:
        mapped = SKILL_KEYWORDS.get(sym)
        if mapped:
            skill_hits[mapped] = skill_hits.get(mapped, 0) + 1
    if skill_hits:
        best = max(sorted(skill_hits), key=lambda k: skill_hits[k])
    return best or "mediation"


def _validate_teaching_expr(expr_str: str) -> bool:
    """Check that a teaching expression is valid, safe, and not oversized."""
    try:
        parsed = parse_all(expr_str)
        if not parsed:
            return False
        total_nodes = sum(_count_nodes(e) for e in parsed)
        return total_nodes <= MAX_AST_NODES
    except LispyError:
        return False


@dataclass
class AncestralTeaching:
    """A teaching left by a dead colonist in cultural memory."""
    source_id: str
    source_name: str
    year_created: int
    teaching_expr: str
    skill_focus: str
    potency: float
    decay_rate: float
    times_studied: int = 0
    studied_by: list[str] = field(default_factory=list)
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "year_created": self.year_created,
            "teaching_expr": self.teaching_expr,
            "skill_focus": self.skill_focus,
            "potency": round(self.potency, 4),
            "decay_rate": round(self.decay_rate, 4),
            "times_studied": self.times_studied,
            "studied_by": list(self.studied_by),
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AncestralTeaching:
        return cls(
            source_id=d["source_id"],
            source_name=d["source_name"],
            year_created=d["year_created"],
            teaching_expr=d["teaching_expr"],
            skill_focus=d["skill_focus"],
            potency=d["potency"],
            decay_rate=d["decay_rate"],
            times_studied=d.get("times_studied", 0),
            studied_by=list(d.get("studied_by", [])),
            active=d.get("active", True),
        )


@dataclass
class CulturalMemory:
    """Colony-wide cultural memory holding ancestral teachings."""
    teachings: list[AncestralTeaching] = field(default_factory=list)

    def distill_teaching(self, colonist_dict: dict[str, Any], year: int) -> AncestralTeaching | None:
        """Extract a teaching from a dying colonist.

        Returns None if the colonist's expression is invalid or if
        the teaching cap has been reached.
        """
        expr = colonist_dict.get("decision_expr", "")
        if not expr or not _validate_teaching_expr(expr):
            return None

        active_count = sum(1 for t in self.teachings if t.active)
        if active_count >= MAX_ACTIVE_TEACHINGS:
            weakest = min(
                (t for t in self.teachings if t.active),
                key=lambda t: t.potency,
                default=None,
            )
            if weakest:
                weakest.active = False

        skills = colonist_dict.get("skills", {})
        best_skill_val = max(skills.values()) if skills else 0.3
        potency = max(MIN_POTENCY, min(1.0, best_skill_val * 0.8))
        decay_rate = max(0.02, DECAY_PER_YEAR * (1.0 - potency * 0.5))

        teaching = AncestralTeaching(
            source_id=colonist_dict["id"],
            source_name=colonist_dict["name"],
            year_created=year,
            teaching_expr=expr,
            skill_focus=_infer_skill_focus(expr),
            potency=potency,
            decay_rate=decay_rate,
        )
        self.teachings.append(teaching)
        return teaching

    def study(self, colonist_id: str, teaching: AncestralTeaching) -> bool:
        """A colonist studies an ancestral teaching. Returns True if new study."""
        if not teaching.active:
            return False
        if colonist_id in teaching.studied_by:
            return False
        teaching.studied_by.append(colonist_id)
        teaching.times_studied += 1
        return True

    def decay(self) -> list[str]:
        """Decay all active teachings. Returns IDs of teachings that died."""
        decayed: list[str] = []
        for teaching in self.teachings:
            if not teaching.active:
                continue
            reinforcement = min(teaching.times_studied * 0.01, 0.03)
            teaching.potency -= (teaching.decay_rate - reinforcement)
            teaching.potency = max(0.0, teaching.potency)
            if teaching.potency < MIN_POTENCY:
                teaching.active = False
                decayed.append(teaching.source_id)
            teaching.times_studied = 0
        return decayed

    def active_teachings(self) -> list[AncestralTeaching]:
        """Return currently active teachings."""
        return [t for t in self.teachings if t.active]

    def to_dict(self) -> dict[str, Any]:
        return {
            "teachings": [t.to_dict() for t in self.teachings],
            "active_count": len(self.active_teachings()),
            "total_created": len(self.teachings),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CulturalMemory:
        teachings = [AncestralTeaching.from_dict(t)
                     for t in d.get("teachings", [])]
        return cls(teachings=teachings)


def cultural_phase(
    culture: CulturalMemory,
    dead_this_year: list[dict[str, Any]],
    active_colonists: list[dict[str, Any]],
    social_graph_dict: dict,
    year: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Run the cultural phase for one year.

    1. Distill teachings from newly dead colonists.
    2. Living colonists may study a teaching (weighted by empathy, faith,
       trust toward the source).
    3. Decay all teachings.

    Returns a list of cultural event dicts for the year log.
    """
    events: list[dict[str, Any]] = []

    for dead in dead_this_year:
        teaching = culture.distill_teaching(dead, year)
        if teaching:
            events.append({
                "type": "teaching_distilled",
                "year": year,
                "source": teaching.source_id,
                "source_name": teaching.source_name,
                "skill_focus": teaching.skill_focus,
                "potency": round(teaching.potency, 3),
            })

    available = culture.active_teachings()
    if available:
        for colonist in active_colonists:
            cid = colonist["id"]
            birth_year = colonist.get("birth_year", 0)
            if year - birth_year < STUDY_AGE_GATE:
                continue
            empathy = colonist.get("stats", {}).get("empathy", 0.5)
            faith = colonist.get("stats", {}).get("faith", 0.5)
            study_chance = empathy * 0.3 + faith * 0.2
            if rng.random() > study_chance:
                continue

            scored: list[tuple[AncestralTeaching, float]] = []
            colonist_rels = social_graph_dict.get(cid, {})
            for teaching in available:
                if cid in teaching.studied_by:
                    continue
                trust = 0.5
                source_rel = colonist_rels.get(teaching.source_id, {})
                if isinstance(source_rel, dict):
                    trust = source_rel.get("trust", 0.5)
                score = teaching.potency * 0.5 + trust * 0.3 + rng.random() * 0.2
                scored.append((teaching, score))

            if not scored:
                continue
            scored.sort(key=lambda x: -x[1])
            chosen, _ = scored[0]
            if culture.study(cid, chosen):
                events.append({
                    "type": "teaching_studied",
                    "year": year,
                    "colonist_id": cid,
                    "teaching_source": chosen.source_id,
                    "skill_focus": chosen.skill_focus,
                    "potency": round(chosen.potency, 3),
                })

    decayed = culture.decay()
    for source_id in decayed:
        events.append({
            "type": "teaching_decayed",
            "year": year,
            "source": source_id,
        })

    return events


def apply_study_boost(colonist_skills: dict[str, float],
                      skill_focus: str, potency: float) -> float:
    """Apply a small skill boost from studying a teaching.

    Returns the actual boost applied (may be less than STUDY_BOOST if
    the skill is near cap).
    """
    current = colonist_skills.get(skill_focus, 0.0)
    boost = STUDY_BOOST * potency
    headroom = 1.0 - current
    actual = min(boost, headroom * 0.5)
    colonist_skills[skill_focus] = min(1.0, current + actual)
    return actual
