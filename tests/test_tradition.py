"""Tests for the Oral Tradition engine (src/mars100/tradition.py)."""
from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from src.mars100.colonist import (
    Colonist, ColonistSkills, ColonistStats, SKILL_NAMES, create_founding_ten,
)
from src.mars100.colony import Relationship, SocialGraph
from src.mars100.tradition import (
    CRITICAL_SKILLS,
    EXPERT_THRESHOLD,
    LEGACY_AFFECTION_THRESHOLD,
    LEGACY_FRACTION,
    LEGACY_MAX_HEIRS,
    MAX_STUDENTS_PER_TEACHER,
    MAX_TEACHERS_PER_STUDENT,
    MIN_EXPERTS_SAFE,
    TEACHER_SKILL_FLOOR,
    TRUST_RESPECT_THRESHOLD,
    KnowledgeCrisis,
    LegacyRecord,
    TeachingRecord,
    TraditionResult,
    compute_expert_coverage,
    create_legacies,
    detect_crises,
    run_teaching,
    run_tradition_phase,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_colonist(
    cid: str = "c0",
    name: str = "Test",
    skills: dict[str, float] | None = None,
    stats: dict[str, float] | None = None,
    alive: bool = True,
    exiled: bool = False,
) -> Colonist:
    sk = skills or {}
    st = stats or {}
    return Colonist(
        id=cid, name=name, element="fire", archetype="test",
        stats=ColonistStats.from_dict(st),
        skills=ColonistSkills.from_dict(sk),
        decision_expr="(+ resolve empathy)",
        alive=alive, exiled=exiled,
    )


def _high_trust_graph(ids: list[str]) -> SocialGraph:
    """Social graph where everyone trusts each other."""
    g = SocialGraph()
    for a in ids:
        g.edges[a] = {}
        for b in ids:
            if a != b:
                g.edges[a][b] = Relationship(trust=0.8, affection=0.7, respect=0.7)
    return g


def _low_trust_graph(ids: list[str]) -> SocialGraph:
    """Social graph where no one trusts anyone."""
    g = SocialGraph()
    for a in ids:
        g.edges[a] = {}
        for b in ids:
            if a != b:
                g.edges[a][b] = Relationship(trust=0.1, affection=0.1, respect=0.1)
    return g


# ---------------------------------------------------------------------------
# TeachingRecord serialization
# ---------------------------------------------------------------------------

class TestTeachingRecordSerialization:
    def test_to_dict(self) -> None:
        r = TeachingRecord(year=5, teacher_id="t", student_id="s",
                           skill="coding", amount=0.04321)
        d = r.to_dict()
        assert d["year"] == 5
        assert d["teacher_id"] == "t"
        assert d["amount"] == 0.0432

    def test_round_trip_values(self) -> None:
        r = TeachingRecord(year=1, teacher_id="a", student_id="b",
                           skill="hydroponics", amount=0.0)
        d = r.to_dict()
        assert d["amount"] == 0.0


# ---------------------------------------------------------------------------
# LegacyRecord serialization
# ---------------------------------------------------------------------------

class TestLegacyRecordSerialization:
    def test_to_dict(self) -> None:
        r = LegacyRecord(year=50, deceased_id="d", deceased_name="Dead",
                         skill="terraforming", original_value=0.8,
                         heirs=[{"id": "h1", "name": "Heir", "gain": 0.12}],
                         decision_expr="(+ 1 2)")
        d = r.to_dict()
        assert d["deceased_id"] == "d"
        assert d["original_value"] == 0.8
        assert len(d["heirs"]) == 1


# ---------------------------------------------------------------------------
# TraditionResult serialization
# ---------------------------------------------------------------------------

class TestTraditionResultSerialization:
    def test_empty_result(self) -> None:
        tr = TraditionResult()
        d = tr.to_dict()
        assert d["teachings"] == []
        assert d["legacies"] == []
        assert d["crises"] == []


# ---------------------------------------------------------------------------
# Expert coverage & crisis detection
# ---------------------------------------------------------------------------

class TestExpertCoverage:
    def test_all_experts(self) -> None:
        colonists = [
            _make_colonist(f"c{i}", skills={sk: 0.8 for sk in CRITICAL_SKILLS})
            for i in range(5)
        ]
        cov = compute_expert_coverage(colonists)
        for sk in CRITICAL_SKILLS:
            assert cov[sk] == 5

    def test_no_experts(self) -> None:
        colonists = [_make_colonist(f"c{i}", skills={}) for i in range(5)]
        cov = compute_expert_coverage(colonists)
        for sk in CRITICAL_SKILLS:
            assert cov[sk] == 0

    def test_threshold_boundary(self) -> None:
        c1 = _make_colonist("c1", skills={"coding": EXPERT_THRESHOLD})
        c2 = _make_colonist("c2", skills={"coding": EXPERT_THRESHOLD - 0.01})
        cov = compute_expert_coverage([c1, c2])
        assert cov["coding"] == 1

    def test_only_counts_active(self) -> None:
        c1 = _make_colonist("c1", skills={"coding": 0.9})
        c2 = _make_colonist("c2", skills={"coding": 0.9}, alive=False)
        c3 = _make_colonist("c3", skills={"coding": 0.9}, exiled=True)
        # c2 and c3 are filtered out before calling compute_expert_coverage
        active = [c for c in [c1, c2, c3] if c.is_active()]
        cov = compute_expert_coverage(active)
        assert cov["coding"] == 1


class TestCrisisDetection:
    def test_no_crisis_with_enough_experts(self) -> None:
        colonists = [
            _make_colonist(f"c{i}", skills={sk: 0.8 for sk in CRITICAL_SKILLS})
            for i in range(3)
        ]
        crises = detect_crises(10, colonists)
        assert len(crises) == 0

    def test_crisis_when_below_threshold(self) -> None:
        colonists = [
            _make_colonist("c0", skills={"coding": 0.9}),
            _make_colonist("c1", skills={"coding": 0.1}),
        ]
        crises = detect_crises(10, colonists)
        coding_crises = [c for c in crises if c.skill == "coding"]
        assert len(coding_crises) == 1
        assert coding_crises[0].expert_count == 1

    def test_crisis_serializes(self) -> None:
        c = KnowledgeCrisis(year=20, skill="hydroponics",
                            expert_count=0, threshold=2)
        d = c.to_dict()
        assert d["skill"] == "hydroponics"
        assert d["expert_count"] == 0


# ---------------------------------------------------------------------------
# Teaching
# ---------------------------------------------------------------------------

class TestTeaching:
    def test_basic_transfer(self) -> None:
        teacher = _make_colonist("t", skills={"coding": 0.9})
        student = _make_colonist("s", skills={"coding": 0.1})
        graph = _high_trust_graph(["t", "s"])
        records = run_teaching(1, [teacher, student], graph.get, [], random.Random(42))
        assert len(records) >= 1
        assert records[0].skill == "coding"
        assert records[0].amount > 0
        assert student.skills.coding > 0.1

    def test_no_transfer_low_trust(self) -> None:
        teacher = _make_colonist("t", skills={"coding": 0.9})
        student = _make_colonist("s", skills={"coding": 0.1})
        graph = _low_trust_graph(["t", "s"])
        records = run_teaching(1, [teacher, student], graph.get, [], random.Random(42))
        assert len(records) == 0
        assert student.skills.coding == pytest.approx(0.1, abs=0.001)

    def test_teacher_skill_floor(self) -> None:
        teacher = _make_colonist("t", skills={"coding": TEACHER_SKILL_FLOOR - 0.01})
        student = _make_colonist("s", skills={"coding": 0.0})
        graph = _high_trust_graph(["t", "s"])
        records = run_teaching(1, [teacher, student], graph.get, [], random.Random(42))
        coding_recs = [r for r in records if r.skill == "coding"]
        assert len(coding_recs) == 0

    def test_max_students_per_teacher(self) -> None:
        teacher = _make_colonist("t", skills={"coding": 0.9, "hydroponics": 0.9, "terraforming": 0.9})
        students = [_make_colonist(f"s{i}", skills={}) for i in range(5)]
        all_c = [teacher] + students
        graph = _high_trust_graph([c.id for c in all_c])
        records = run_teaching(1, all_c, graph.get, [], random.Random(42))
        assert len(records) <= MAX_STUDENTS_PER_TEACHER

    def test_max_teachers_per_student(self) -> None:
        teachers = [_make_colonist(f"t{i}", skills={"coding": 0.9}) for i in range(5)]
        student = _make_colonist("s", skills={"coding": 0.0})
        all_c = teachers + [student]
        graph = _high_trust_graph([c.id for c in all_c])
        records = run_teaching(1, all_c, graph.get, [], random.Random(42))
        student_records = [r for r in records if r.student_id == "s"]
        assert len(student_records) <= MAX_TEACHERS_PER_STUDENT

    def test_skill_stays_in_bounds(self) -> None:
        teacher = _make_colonist("t", skills={"coding": 1.0})
        student = _make_colonist("s", skills={"coding": 0.95})
        graph = _high_trust_graph(["t", "s"])
        records = run_teaching(1, [teacher, student], graph.get, [], random.Random(42))
        assert student.skills.coding <= 1.0

    def test_no_self_teaching(self) -> None:
        c = _make_colonist("c", skills={"coding": 0.9})
        graph = _high_trust_graph(["c"])
        records = run_teaching(1, [c], graph.get, [], random.Random(42))
        assert len(records) == 0

    def test_crisis_skill_prioritized(self) -> None:
        teacher = _make_colonist("t", skills={"coding": 0.8, "terraforming": 0.8})
        student = _make_colonist("s", skills={"coding": 0.0, "terraforming": 0.0})
        graph = _high_trust_graph(["t", "s"])
        crisis = [KnowledgeCrisis(year=1, skill="terraforming",
                                  expert_count=1, threshold=2)]
        records = run_teaching(1, [teacher, student], graph.get, crisis, random.Random(42))
        assert len(records) >= 1
        assert records[0].skill == "terraforming"

    def test_deterministic_same_seed(self) -> None:
        def _run(seed: int) -> list[dict]:
            teacher = _make_colonist("t", skills={"coding": 0.8})
            student = _make_colonist("s", skills={"coding": 0.1})
            graph = _high_trust_graph(["t", "s"])
            return [r.to_dict() for r in
                    run_teaching(1, [teacher, student], graph.get, [], random.Random(seed))]
        assert _run(99) == _run(99)


# ---------------------------------------------------------------------------
# Legacy
# ---------------------------------------------------------------------------

class TestLegacy:
    def test_basic_legacy_inheritance(self) -> None:
        dead = _make_colonist("d", name="Dead", skills={"coding": 0.9, "terraforming": 0.8})
        dead.die(50, "asphyxiation")
        heir = _make_colonist("h", skills={"coding": 0.1})
        graph = _high_trust_graph(["d", "h"])
        records = create_legacies(50, [dead], [heir], graph.get)
        assert len(records) >= 1
        assert heir.skills.coding > 0.1

    def test_legacy_no_inflation(self) -> None:
        """Total inherited across heirs must not exceed LEGACY_FRACTION of original."""
        dead = _make_colonist("d", skills={"coding": 0.8})
        dead.die(50, "cause")
        heirs = [_make_colonist(f"h{i}", skills={"coding": 0.0}) for i in range(5)]
        graph = _high_trust_graph(["d"] + [h.id for h in heirs])
        records = create_legacies(50, [dead], heirs, graph.get)
        coding_recs = [r for r in records if r.skill == "coding"]
        if coding_recs:
            total_gain = sum(h["gain"] for h in coding_recs[0].heirs)
            assert total_gain <= 0.8 * LEGACY_FRACTION + 0.01

    def test_legacy_max_heirs(self) -> None:
        dead = _make_colonist("d", skills={"coding": 0.8})
        dead.die(50, "cause")
        heirs = [_make_colonist(f"h{i}", skills={}) for i in range(10)]
        graph = _high_trust_graph(["d"] + [h.id for h in heirs])
        records = create_legacies(50, [dead], heirs, graph.get)
        for rec in records:
            assert len(rec.heirs) <= LEGACY_MAX_HEIRS

    def test_legacy_requires_affection(self) -> None:
        dead = _make_colonist("d", skills={"coding": 0.8})
        dead.die(50, "cause")
        heir = _make_colonist("h", skills={})
        graph = _low_trust_graph(["d", "h"])
        records = create_legacies(50, [dead], [heir], graph.get)
        assert len(records) == 0

    def test_legacy_does_not_resurrect(self) -> None:
        dead = _make_colonist("d", skills={"coding": 0.8})
        dead.die(50, "cause")
        heir = _make_colonist("h", skills={})
        graph = _high_trust_graph(["d", "h"])
        create_legacies(50, [dead], [heir], graph.get)
        assert not dead.alive
        assert dead.death_year == 50

    def test_legacy_skill_clamped(self) -> None:
        dead = _make_colonist("d", skills={"coding": 1.0})
        dead.die(50, "cause")
        heir = _make_colonist("h", skills={"coding": 0.9})
        graph = _high_trust_graph(["d", "h"])
        create_legacies(50, [dead], [heir], graph.get)
        assert heir.skills.coding <= 1.0

    def test_legacy_stores_decision_expr(self) -> None:
        dead = _make_colonist("d", skills={"coding": 0.8})
        dead.decision_expr = "(if (> faith 0.5) 1 0)"
        dead.die(50, "cause")
        heir = _make_colonist("h", skills={})
        graph = _high_trust_graph(["d", "h"])
        records = create_legacies(50, [dead], [heir], graph.get)
        assert any(r.decision_expr == "(if (> faith 0.5) 1 0)" for r in records)

    def test_no_legacy_from_low_skill(self) -> None:
        dead = _make_colonist("d", skills={"coding": 0.05})
        dead.die(50, "cause")
        heir = _make_colonist("h", skills={})
        graph = _high_trust_graph(["d", "h"])
        records = create_legacies(50, [dead], [heir], graph.get)
        assert len(records) == 0


# ---------------------------------------------------------------------------
# Full tradition phase
# ---------------------------------------------------------------------------

class TestTraditionPhase:
    def test_smoke_no_crash(self) -> None:
        colonists = create_founding_ten(42)
        graph = SocialGraph()
        graph.initialize([c.id for c in colonists], random.Random(42))
        result = run_tradition_phase(
            year=5, all_colonists=colonists, dead_this_year=[],
            social_get=graph.get, rng=random.Random(42),
        )
        assert isinstance(result, TraditionResult)
        assert isinstance(result.teachings, list)
        assert isinstance(result.legacies, list)
        assert isinstance(result.crises, list)

    def test_with_death(self) -> None:
        colonists = create_founding_ten(42)
        graph = SocialGraph()
        graph.initialize([c.id for c in colonists], random.Random(42))
        victim = colonists[0]
        victim.die(20, "asphyxiation")
        result = run_tradition_phase(
            year=20, all_colonists=colonists, dead_this_year=[victim],
            social_get=graph.get, rng=random.Random(42),
        )
        # Should have legacy records since founding colonists have high skills
        if victim.skills.terraforming > 0.1 or victim.skills.coding > 0.1:
            assert len(result.legacies) >= 1

    def test_serialization_round_trip(self) -> None:
        colonists = create_founding_ten(42)
        graph = SocialGraph()
        graph.initialize([c.id for c in colonists], random.Random(42))
        result = run_tradition_phase(
            year=5, all_colonists=colonists, dead_this_year=[],
            social_get=graph.get, rng=random.Random(42),
        )
        d = result.to_dict()
        assert "teachings" in d
        assert "legacies" in d
        assert "crises" in d

    def test_deterministic(self) -> None:
        def _run(seed: int) -> dict:
            colonists = create_founding_ten(seed)
            graph = SocialGraph()
            graph.initialize([c.id for c in colonists], random.Random(seed))
            return run_tradition_phase(
                year=5, all_colonists=colonists, dead_this_year=[],
                social_get=graph.get, rng=random.Random(seed),
            ).to_dict()
        assert _run(77) == _run(77)


# ---------------------------------------------------------------------------
# Invariant tests
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_skills_always_bounded(self) -> None:
        """After any tradition phase, all skills remain in [0, 1]."""
        rng = random.Random(123)
        colonists = create_founding_ten(123)
        graph = SocialGraph()
        graph.initialize([c.id for c in colonists], rng)
        for year in range(1, 30):
            if year == 15:
                colonists[0].die(year, "test")
                dead = [colonists[0]]
            else:
                dead = []
            run_tradition_phase(year, colonists, dead, graph.get, rng)
            for c in colonists:
                for sk in SKILL_NAMES:
                    val = getattr(c.skills, sk)
                    assert 0.0 <= val <= 1.0, f"{c.id}.{sk} = {val} out of bounds"

    def test_teaching_uses_start_of_year_snapshot(self) -> None:
        """Teaching amounts should not cascade within one year.

        If A teaches B coding, B should NOT then teach C coding with the
        newly gained skill in the same year.  We verify this by checking
        that a student who becomes eligible to teach doesn't produce a
        second transfer in the same call.
        """
        teacher = _make_colonist("t", skills={"coding": 0.9})
        middleman = _make_colonist("m", skills={"coding": 0.28})
        student = _make_colonist("s", skills={"coding": 0.0})
        graph = _high_trust_graph(["t", "m", "s"])
        records = run_teaching(1, [teacher, middleman, student], graph.get, [], random.Random(42))
        # middleman may learn from teacher, but middleman should NOT teach student
        # because middleman's start-of-year skill is below TEACHER_SKILL_FLOOR (0.3)
        middleman_teaches = [r for r in records if r.teacher_id == "m"]
        assert len(middleman_teaches) == 0

    def test_legacy_skill_mass_conservation(self) -> None:
        """Total skill mass created by legacy must not exceed budget."""
        dead = _make_colonist("d", skills={"coding": 0.6, "terraforming": 0.7})
        dead.die(50, "cause")
        heirs = [_make_colonist(f"h{i}", skills={}) for i in range(LEGACY_MAX_HEIRS)]
        graph = _high_trust_graph(["d"] + [h.id for h in heirs])

        before_sums: dict[str, float] = {}
        for sk in SKILL_NAMES:
            before_sums[sk] = sum(getattr(h.skills, sk) for h in heirs)

        create_legacies(50, [dead], heirs, graph.get)

        for sk in ("coding", "terraforming"):
            after_sum = sum(getattr(h.skills, sk) for h in heirs)
            added = after_sum - before_sums[sk]
            original = getattr(dead.skills, sk)
            assert added <= original * LEGACY_FRACTION + 0.01


# ---------------------------------------------------------------------------
# Integration: 10-year smoke test with full engine
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_10_year_engine_with_tradition(self) -> None:
        """Run the engine for 10 years and verify tradition data appears."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert len(result.years) == 10
        # Tradition data should be present in year results
        for yr in result.years:
            d = yr.to_dict()
            assert "tradition" in d

    def test_100_year_tradition_coverage(self) -> None:
        """Full 100-year sim: tradition should produce teaching events."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        total_teachings = 0
        total_legacies = 0
        for yr in result.years:
            td = yr.to_dict().get("tradition", {})
            total_teachings += len(td.get("teachings", []))
            total_legacies += len(td.get("legacies", []))
        assert total_teachings > 0, "No teaching happened in 100 years"
