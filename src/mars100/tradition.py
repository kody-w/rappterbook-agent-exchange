"""
Oral Tradition engine for Mars-100.

Cultural memory transmission between colonists: teaching, legacy
inheritance, and knowledge-crisis detection.  Addresses the colony's
terminal-decline failure mode where expert knowledge dies with its
holders.

Teaching is snapshot-then-apply (two-phase) for determinism.
Legacy pools are fixed-budget (no skill inflation on death).
Crisis detection uses bus-factor counting, not average skill.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import Colonist, SKILL_NAMES

TEACHING_RATE = 0.08
MAX_STUDENTS_PER_TEACHER = 2
MAX_TEACHERS_PER_STUDENT = 1
TRUST_RESPECT_THRESHOLD = 1.0
TEACHER_SKILL_FLOOR = 0.3
LEGACY_FRACTION = 0.60
LEGACY_MAX_HEIRS = 3
LEGACY_AFFECTION_THRESHOLD = 0.5
EXPERT_THRESHOLD = 0.4
MIN_EXPERTS_SAFE = 2
CRITICAL_SKILLS = ("terraforming", "hydroponics", "coding")


@dataclass
class TeachingRecord:
    """One teaching event: teacher passes skill to student."""
    year: int
    teacher_id: str
    student_id: str
    skill: str
    amount: float

    def to_dict(self) -> dict:
        return {"year": self.year, "teacher_id": self.teacher_id,
                "student_id": self.student_id, "skill": self.skill,
                "amount": round(self.amount, 4)}


@dataclass
class LegacyRecord:
    """Knowledge left by a dead colonist, split among heirs."""
    year: int
    deceased_id: str
    deceased_name: str
    skill: str
    original_value: float
    heirs: list[dict]
    decision_expr: str

    def to_dict(self) -> dict:
        return {"year": self.year, "deceased_id": self.deceased_id,
                "deceased_name": self.deceased_name, "skill": self.skill,
                "original_value": round(self.original_value, 4),
                "heirs": self.heirs, "decision_expr": self.decision_expr}


@dataclass
class KnowledgeCrisis:
    """Alert: too few experts in a critical skill area."""
    year: int
    skill: str
    expert_count: int
    threshold: int

    def to_dict(self) -> dict:
        return {"year": self.year, "skill": self.skill,
                "expert_count": self.expert_count,
                "threshold": self.threshold}


@dataclass
class TraditionResult:
    """All tradition events for one year."""
    teachings: list[TeachingRecord] = field(default_factory=list)
    legacies: list[LegacyRecord] = field(default_factory=list)
    crises: list[KnowledgeCrisis] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "teachings": [t.to_dict() for t in self.teachings],
            "legacies": [l.to_dict() for l in self.legacies],
            "crises": [c.to_dict() for c in self.crises],
        }


def compute_expert_coverage(active: list[Colonist]) -> dict[str, int]:
    """Count how many active colonists qualify as experts per skill."""
    coverage: dict[str, int] = {}
    for skill in CRITICAL_SKILLS:
        count = sum(1 for c in active
                    if getattr(c.skills, skill) >= EXPERT_THRESHOLD)
        coverage[skill] = count
    return coverage


def detect_crises(year: int, active: list[Colonist]) -> list[KnowledgeCrisis]:
    """Flag knowledge crises where expert bus-factor is dangerously low."""
    coverage = compute_expert_coverage(active)
    crises: list[KnowledgeCrisis] = []
    for skill, count in coverage.items():
        if count < MIN_EXPERTS_SAFE:
            crises.append(KnowledgeCrisis(
                year=year, skill=skill,
                expert_count=count, threshold=MIN_EXPERTS_SAFE))
    return crises


def _find_best_teacher_pairs(
    active: list[Colonist],
    social_get: Any,
    crises: list[KnowledgeCrisis],
    rng: random.Random,
) -> list[tuple[Colonist, Colonist, str]]:
    """Find (teacher, student, skill) tuples respecting budgets.

    Returns pairs sorted by priority: crisis skills first, then by
    skill gap (largest gap = most to learn).
    """
    crisis_skills = {c.skill for c in crises}
    sorted_active = sorted(active, key=lambda c: c.id)

    candidates: list[tuple[float, Colonist, Colonist, str]] = []
    for teacher in sorted_active:
        for skill in SKILL_NAMES:
            teacher_val = getattr(teacher.skills, skill)
            if teacher_val < TEACHER_SKILL_FLOOR:
                continue
            for student in sorted_active:
                if student.id == teacher.id:
                    continue
                student_val = getattr(student.skills, skill)
                gap = teacher_val - student_val
                if gap <= 0.05:
                    continue
                rel = social_get(student.id, teacher.id)
                quality = rel.trust + rel.respect
                if quality < TRUST_RESPECT_THRESHOLD:
                    continue
                priority = gap * quality
                if skill in crisis_skills:
                    priority += 10.0
                candidates.append((priority, teacher, student, skill))

    candidates.sort(key=lambda x: -x[0])

    teacher_slots: dict[str, int] = {}
    student_slots: dict[str, int] = {}
    selected: list[tuple[Colonist, Colonist, str]] = []

    for _, teacher, student, skill in candidates:
        t_used = teacher_slots.get(teacher.id, 0)
        s_used = student_slots.get(student.id, 0)
        if t_used >= MAX_STUDENTS_PER_TEACHER:
            continue
        if s_used >= MAX_TEACHERS_PER_STUDENT:
            continue
        selected.append((teacher, student, skill))
        teacher_slots[teacher.id] = t_used + 1
        student_slots[student.id] = s_used + 1

    return selected


def run_teaching(
    year: int,
    active: list[Colonist],
    social_get: Any,
    crises: list[KnowledgeCrisis],
    rng: random.Random,
) -> list[TeachingRecord]:
    """Execute the teaching phase: snapshot skills, compute transfers, apply.

    Two-phase for determinism: all gains computed from start-of-year
    skill values, then applied together.
    """
    pairs = _find_best_teacher_pairs(active, social_get, crises, rng)

    transfers: list[tuple[Colonist, str, float]] = []
    records: list[TeachingRecord] = []

    for teacher, student, skill in pairs:
        teacher_val = getattr(teacher.skills, skill)
        student_val = getattr(student.skills, skill)
        gap = max(0.0, teacher_val - student_val)
        rel = social_get(student.id, teacher.id)
        quality = (rel.trust + rel.respect) / 2.0
        amount = TEACHING_RATE * quality * gap
        amount = min(amount, 1.0 - student_val)
        if amount < 0.001:
            continue
        transfers.append((student, skill, amount))
        records.append(TeachingRecord(
            year=year, teacher_id=teacher.id,
            student_id=student.id, skill=skill, amount=amount))

    for student, skill, amount in transfers:
        current = getattr(student.skills, skill)
        setattr(student.skills, skill, min(1.0, current + amount))

    return records


def create_legacies(
    year: int,
    dead_this_year: list[Colonist],
    active: list[Colonist],
    social_get: Any,
) -> list[LegacyRecord]:
    """Generate legacy records for colonists who died this year.

    Each dead colonist's top 2 skills are pooled and split among the
    heirs with highest affection (max LEGACY_MAX_HEIRS).  The pool is
    fixed-budget: total inherited = LEGACY_FRACTION * original skill.
    """
    records: list[LegacyRecord] = []
    sorted_dead = sorted(dead_this_year, key=lambda c: c.id)

    for deceased in sorted_dead:
        skill_values = [(sk, getattr(deceased.skills, sk)) for sk in SKILL_NAMES]
        skill_values.sort(key=lambda x: -x[1])
        top_skills = [(sk, val) for sk, val in skill_values[:2] if val > 0.1]

        heir_scores: list[tuple[Colonist, float]] = []
        for candidate in sorted(active, key=lambda c: c.id):
            rel = social_get(candidate.id, deceased.id)
            if rel.affection >= LEGACY_AFFECTION_THRESHOLD:
                heir_scores.append((candidate, rel.affection))
        heir_scores.sort(key=lambda x: -x[1])
        heirs = [h for h, _ in heir_scores[:LEGACY_MAX_HEIRS]]

        if not heirs:
            continue

        for skill, original_val in top_skills:
            pool = original_val * LEGACY_FRACTION
            per_heir = pool / len(heirs)
            heir_dicts: list[dict] = []
            for heir in heirs:
                current = getattr(heir.skills, skill)
                gain = min(per_heir, 1.0 - current)
                setattr(heir.skills, skill, min(1.0, current + gain))
                heir_dicts.append({"id": heir.id, "name": heir.name,
                                   "gain": round(gain, 4)})
            records.append(LegacyRecord(
                year=year, deceased_id=deceased.id,
                deceased_name=deceased.name, skill=skill,
                original_value=original_val, heirs=heir_dicts,
                decision_expr=deceased.decision_expr))

    return records


def run_tradition_phase(
    year: int,
    all_colonists: list[Colonist],
    dead_this_year: list[Colonist],
    social_get: Any,
    rng: random.Random,
) -> TraditionResult:
    """Execute the full tradition phase for one year.

    Order: detect crises → teach → process legacies → re-check crises.
    """
    active = sorted(
        [c for c in all_colonists if c.is_active()],
        key=lambda c: c.id,
    )

    crises_before = detect_crises(year, active)
    teachings = run_teaching(year, active, social_get, crises_before, rng)
    legacies = create_legacies(year, dead_this_year, active, social_get)

    active_after = [c for c in all_colonists if c.is_active()]
    crises_after = detect_crises(year, active_after)

    return TraditionResult(
        teachings=teachings,
        legacies=legacies,
        crises=crises_after,
    )
