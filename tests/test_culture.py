"""Tests for the oral tradition / cultural memory system."""
from __future__ import annotations

import random

import pytest

from src.mars100.culture import (
    AncestralTeaching,
    CulturalMemory,
    DECAY_PER_YEAR,
    MAX_ACTIVE_TEACHINGS,
    MIN_POTENCY,
    STUDY_AGE_GATE,
    STUDY_BOOST,
    _extract_symbols,
    _infer_skill_focus,
    _validate_teaching_expr,
    apply_study_boost,
    cultural_phase,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_colonist_dict(
    cid: str = "kira-sol",
    name: str = "Kira Sol",
    element: str = "fire",
    archetype: str = "commander",
    decision_expr: str = "(if (> resolve 0.7) (+ resolve improvisation) (* empathy 2))",
    alive: bool = True,
    birth_year: int = 0,
    **overrides,
) -> dict:
    """Create a minimal colonist dict for testing."""
    d = {
        "id": cid,
        "name": name,
        "element": element,
        "archetype": archetype,
        "decision_expr": decision_expr,
        "alive": alive,
        "birth_year": birth_year,
        "stats": {"resolve": 0.8, "improvisation": 0.5, "empathy": 0.5,
                  "hoarding": 0.3, "faith": 0.3, "paranoia": 0.3},
        "skills": {"terraforming": 0.7, "hydroponics": 0.2, "mediation": 0.5,
                   "coding": 0.3, "prayer": 0.1, "sabotage": 0.1},
    }
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# _extract_symbols
# ---------------------------------------------------------------------------

class TestExtractSymbols:
    def test_simple_expr(self):
        syms = _extract_symbols(["if", [">", "resolve", 0.7], "empathy", "faith"])
        assert "resolve" in syms
        assert "empathy" in syms
        assert "faith" in syms

    def test_numbers_excluded(self):
        syms = _extract_symbols(["+", 1, 2])
        assert 1 not in syms
        assert "+" in syms

    def test_nested(self):
        syms = _extract_symbols(["let", [["x", "coding"]], ["+", "x", "prayer"]])
        assert "coding" in syms
        assert "prayer" in syms

    def test_atom(self):
        assert _extract_symbols("resolve") == {"resolve"}

    def test_empty_list(self):
        assert _extract_symbols([]) == set()


# ---------------------------------------------------------------------------
# _infer_skill_focus
# ---------------------------------------------------------------------------

class TestInferSkillFocus:
    def test_faith_expr(self):
        assert _infer_skill_focus("(* faith faith)") == "prayer"

    def test_coding_expr(self):
        assert _infer_skill_focus("(+ coding improvisation)") == "coding"

    def test_resolve_expr(self):
        assert _infer_skill_focus("(+ resolve terraforming)") == "terraforming"

    def test_invalid_expr_fallback(self):
        assert _infer_skill_focus("((((") == "mediation"

    def test_empty_expr_fallback(self):
        assert _infer_skill_focus("") == "mediation"


# ---------------------------------------------------------------------------
# _validate_teaching_expr
# ---------------------------------------------------------------------------

class TestValidateTeachingExpr:
    def test_valid(self):
        assert _validate_teaching_expr("(+ 1 2)") is True

    def test_complex_valid(self):
        assert _validate_teaching_expr("(if (> resolve 0.5) (+ faith empathy) paranoia)") is True

    def test_empty(self):
        assert _validate_teaching_expr("") is False

    def test_malformed(self):
        assert _validate_teaching_expr("(((") is False


# ---------------------------------------------------------------------------
# AncestralTeaching serialization
# ---------------------------------------------------------------------------

class TestAncestralTeaching:
    def test_to_dict_round_trip(self):
        t = AncestralTeaching(
            source_id="kira-sol", source_name="Kira Sol",
            year_created=42, teaching_expr="(+ resolve faith)",
            skill_focus="terraforming", potency=0.7, decay_rate=0.04,
            times_studied=3, studied_by=["fen-marsh"],
        )
        d = t.to_dict()
        restored = AncestralTeaching.from_dict(d)
        assert restored.source_id == t.source_id
        assert restored.potency == pytest.approx(t.potency, abs=0.001)
        assert restored.studied_by == t.studied_by
        assert restored.active is True

    def test_inactive_round_trip(self):
        t = AncestralTeaching(
            source_id="a", source_name="A", year_created=1,
            teaching_expr="(+ 1 2)", skill_focus="coding",
            potency=0.1, decay_rate=0.05, active=False,
        )
        assert AncestralTeaching.from_dict(t.to_dict()).active is False


# ---------------------------------------------------------------------------
# CulturalMemory.distill_teaching
# ---------------------------------------------------------------------------

class TestDistillTeaching:
    def test_basic_distill(self):
        mem = CulturalMemory()
        colonist = _make_colonist_dict()
        teaching = mem.distill_teaching(colonist, year=50)
        assert teaching is not None
        assert teaching.source_id == "kira-sol"
        assert teaching.potency > 0
        assert teaching.active is True
        assert len(mem.teachings) == 1

    def test_invalid_expr_returns_none(self):
        mem = CulturalMemory()
        colonist = _make_colonist_dict(decision_expr="(((")
        assert mem.distill_teaching(colonist, year=10) is None

    def test_empty_expr_returns_none(self):
        mem = CulturalMemory()
        colonist = _make_colonist_dict(decision_expr="")
        assert mem.distill_teaching(colonist, year=10) is None

    def test_cap_enforcement(self):
        mem = CulturalMemory()
        for i in range(MAX_ACTIVE_TEACHINGS + 5):
            c = _make_colonist_dict(cid=f"c-{i}", name=f"C{i}")
            mem.distill_teaching(c, year=i)
        active = mem.active_teachings()
        assert len(active) <= MAX_ACTIVE_TEACHINGS

    def test_potency_from_skills(self):
        mem = CulturalMemory()
        expert = _make_colonist_dict(
            skills={"terraforming": 0.95, "hydroponics": 0.1,
                    "mediation": 0.1, "coding": 0.1,
                    "prayer": 0.1, "sabotage": 0.1},
        )
        novice = _make_colonist_dict(
            cid="novice", name="Novice",
            skills={"terraforming": 0.1, "hydroponics": 0.1,
                    "mediation": 0.1, "coding": 0.1,
                    "prayer": 0.1, "sabotage": 0.1},
        )
        t_expert = mem.distill_teaching(expert, year=1)
        t_novice = mem.distill_teaching(novice, year=2)
        assert t_expert is not None and t_novice is not None
        assert t_expert.potency > t_novice.potency


# ---------------------------------------------------------------------------
# CulturalMemory.study
# ---------------------------------------------------------------------------

class TestStudy:
    def test_basic_study(self):
        mem = CulturalMemory()
        t = AncestralTeaching(
            source_id="a", source_name="A", year_created=1,
            teaching_expr="(+ 1 2)", skill_focus="coding",
            potency=0.5, decay_rate=0.04,
        )
        mem.teachings.append(t)
        assert mem.study("b", t) is True
        assert t.times_studied == 1
        assert "b" in t.studied_by

    def test_no_double_study(self):
        mem = CulturalMemory()
        t = AncestralTeaching(
            source_id="a", source_name="A", year_created=1,
            teaching_expr="(+ 1 2)", skill_focus="coding",
            potency=0.5, decay_rate=0.04,
        )
        mem.teachings.append(t)
        mem.study("b", t)
        assert mem.study("b", t) is False
        assert t.times_studied == 1

    def test_inactive_teaching_rejected(self):
        mem = CulturalMemory()
        t = AncestralTeaching(
            source_id="a", source_name="A", year_created=1,
            teaching_expr="(+ 1 2)", skill_focus="coding",
            potency=0.5, decay_rate=0.04, active=False,
        )
        mem.teachings.append(t)
        assert mem.study("b", t) is False


# ---------------------------------------------------------------------------
# CulturalMemory.decay
# ---------------------------------------------------------------------------

class TestDecay:
    def test_potency_decreases(self):
        mem = CulturalMemory()
        t = AncestralTeaching(
            source_id="a", source_name="A", year_created=1,
            teaching_expr="(+ 1 2)", skill_focus="coding",
            potency=0.5, decay_rate=0.1,
        )
        mem.teachings.append(t)
        mem.decay()
        assert t.potency < 0.5

    def test_teaching_dies_at_min_potency(self):
        mem = CulturalMemory()
        t = AncestralTeaching(
            source_id="a", source_name="A", year_created=1,
            teaching_expr="(+ 1 2)", skill_focus="coding",
            potency=0.06, decay_rate=0.1,
        )
        mem.teachings.append(t)
        decayed = mem.decay()
        assert "a" in decayed
        assert t.active is False

    def test_study_slows_decay(self):
        mem = CulturalMemory()
        t1 = AncestralTeaching(
            source_id="a", source_name="A", year_created=1,
            teaching_expr="(+ 1 2)", skill_focus="coding",
            potency=0.5, decay_rate=0.06,
        )
        t2 = AncestralTeaching(
            source_id="b", source_name="B", year_created=1,
            teaching_expr="(+ 3 4)", skill_focus="coding",
            potency=0.5, decay_rate=0.06, times_studied=5,
        )
        mem.teachings.extend([t1, t2])
        mem.decay()
        assert t2.potency > t1.potency

    def test_study_count_resets_after_decay(self):
        mem = CulturalMemory()
        t = AncestralTeaching(
            source_id="a", source_name="A", year_created=1,
            teaching_expr="(+ 1 2)", skill_focus="coding",
            potency=0.5, decay_rate=0.04, times_studied=3,
        )
        mem.teachings.append(t)
        mem.decay()
        assert t.times_studied == 0


# ---------------------------------------------------------------------------
# CulturalMemory serialization
# ---------------------------------------------------------------------------

class TestCulturalMemorySerialization:
    def test_round_trip(self):
        mem = CulturalMemory()
        colonist = _make_colonist_dict()
        mem.distill_teaching(colonist, year=10)
        d = mem.to_dict()
        restored = CulturalMemory.from_dict(d)
        assert len(restored.teachings) == 1
        assert restored.teachings[0].source_id == "kira-sol"

    def test_empty_round_trip(self):
        mem = CulturalMemory()
        d = mem.to_dict()
        assert d["active_count"] == 0
        restored = CulturalMemory.from_dict(d)
        assert len(restored.teachings) == 0


# ---------------------------------------------------------------------------
# apply_study_boost
# ---------------------------------------------------------------------------

class TestApplyStudyBoost:
    def test_basic_boost(self):
        skills = {"coding": 0.3, "prayer": 0.1}
        boost = apply_study_boost(skills, "coding", potency=0.8)
        assert skills["coding"] > 0.3
        assert boost > 0

    def test_near_cap(self):
        skills = {"coding": 0.99}
        boost = apply_study_boost(skills, "coding", potency=1.0)
        assert skills["coding"] <= 1.0
        assert boost < STUDY_BOOST

    def test_zero_potency(self):
        skills = {"coding": 0.5}
        boost = apply_study_boost(skills, "coding", potency=0.0)
        assert boost == 0.0

    def test_missing_skill(self):
        skills = {}
        boost = apply_study_boost(skills, "coding", potency=0.5)
        assert "coding" in skills
        assert skills["coding"] > 0


# ---------------------------------------------------------------------------
# cultural_phase integration
# ---------------------------------------------------------------------------

class TestCulturalPhase:
    def test_distills_from_dead(self):
        culture = CulturalMemory()
        dead = [_make_colonist_dict(alive=False)]
        active = [_make_colonist_dict(cid="fen-marsh", name="Fen Marsh",
                                       birth_year=0,
                                       stats={"empathy": 0.9, "faith": 0.8,
                                              "resolve": 0.4, "improvisation": 0.5,
                                              "hoarding": 0.2, "paranoia": 0.1})]
        events = cultural_phase(culture, dead, active, {}, year=50,
                                rng=random.Random(42))
        distilled = [e for e in events if e["type"] == "teaching_distilled"]
        assert len(distilled) == 1
        assert distilled[0]["source"] == "kira-sol"

    def test_age_gate_blocks_newborns(self):
        culture = CulturalMemory()
        t = AncestralTeaching(
            source_id="a", source_name="A", year_created=1,
            teaching_expr="(+ 1 2)", skill_focus="coding",
            potency=0.8, decay_rate=0.04,
        )
        culture.teachings.append(t)
        newborn = _make_colonist_dict(cid="baby", name="Baby", birth_year=49,
                                       stats={"empathy": 1.0, "faith": 1.0,
                                              "resolve": 0.5, "improvisation": 0.5,
                                              "hoarding": 0.5, "paranoia": 0.5})
        events = cultural_phase(culture, [], [newborn], {}, year=50,
                                rng=random.Random(42))
        studied = [e for e in events if e["type"] == "teaching_studied"]
        assert len(studied) == 0

    def test_old_enough_can_study(self):
        culture = CulturalMemory()
        t = AncestralTeaching(
            source_id="a", source_name="A", year_created=1,
            teaching_expr="(+ 1 2)", skill_focus="coding",
            potency=0.8, decay_rate=0.04,
        )
        culture.teachings.append(t)
        elder = _make_colonist_dict(cid="elder", name="Elder", birth_year=0,
                                     stats={"empathy": 1.0, "faith": 1.0,
                                            "resolve": 0.5, "improvisation": 0.5,
                                            "hoarding": 0.5, "paranoia": 0.5})
        events = cultural_phase(culture, [], [elder], {}, year=50,
                                rng=random.Random(1))
        studied = [e for e in events if e["type"] == "teaching_studied"]
        assert len(studied) == 1

    def test_social_graph_weighting(self):
        culture = CulturalMemory()
        t1 = AncestralTeaching(
            source_id="trusted", source_name="Trusted", year_created=1,
            teaching_expr="(+ 1 2)", skill_focus="coding",
            potency=0.5, decay_rate=0.04,
        )
        t2 = AncestralTeaching(
            source_id="distrusted", source_name="Distrusted", year_created=1,
            teaching_expr="(+ 3 4)", skill_focus="prayer",
            potency=0.5, decay_rate=0.04,
        )
        culture.teachings.extend([t1, t2])
        colonist = _make_colonist_dict(
            cid="learner", name="Learner", birth_year=0,
            stats={"empathy": 1.0, "faith": 1.0,
                   "resolve": 0.5, "improvisation": 0.5,
                   "hoarding": 0.5, "paranoia": 0.5},
        )
        social = {
            "learner": {
                "trusted": {"trust": 0.95, "affection": 0.5, "respect": 0.5},
                "distrusted": {"trust": 0.05, "affection": 0.1, "respect": 0.1},
            }
        }
        events = cultural_phase(culture, [], [colonist], social, year=50,
                                rng=random.Random(1))
        studied = [e for e in events if e["type"] == "teaching_studied"]
        if studied:
            assert studied[0]["teaching_source"] == "trusted"

    def test_decay_events(self):
        culture = CulturalMemory()
        t = AncestralTeaching(
            source_id="dying-teach", source_name="DT", year_created=1,
            teaching_expr="(+ 1 2)", skill_focus="coding",
            potency=0.06, decay_rate=0.1,
        )
        culture.teachings.append(t)
        events = cultural_phase(culture, [], [], {}, year=50,
                                rng=random.Random(42))
        decayed = [e for e in events if e["type"] == "teaching_decayed"]
        assert len(decayed) == 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_result(self):
        """Cultural phase must be deterministic for the same RNG seed."""
        culture1 = CulturalMemory()
        culture2 = CulturalMemory()
        dead = [_make_colonist_dict(alive=False)]
        active = [_make_colonist_dict(cid="learner", name="Learner",
                                       birth_year=0,
                                       stats={"empathy": 0.9, "faith": 0.8,
                                              "resolve": 0.5, "improvisation": 0.5,
                                              "hoarding": 0.3, "paranoia": 0.2})]
        e1 = cultural_phase(culture1, dead, active, {}, year=50,
                            rng=random.Random(99))
        e2 = cultural_phase(culture2, dead, active, {}, year=50,
                            rng=random.Random(99))
        assert e1 == e2
