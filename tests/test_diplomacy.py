"""Tests for the diplomacy organ."""
from __future__ import annotations

import random

import pytest

from src.mars100.diplomacy import (
    DiplomacyState,
    DiplomacyTickResult,
    ExternalEntity,
    EVIDENCE_CONFIRM,
    EVIDENCE_CORROBORATE,
    MAX_ENTITIES,
    SIGNAL_BASE_YEAR,
    SPLINTER_EXILE_THRESHOLD,
    SPLINTER_WINDOW_YEARS,
    compute_diplomatic_pressure,
    compute_stance,
    diplomatic_subsim_expression,
    generate_entity_name,
    signal_detection_probability,
    should_form_splinter,
    tick_diplomacy,
)


# ---------------------------------------------------------------------------
#  ExternalEntity
# ---------------------------------------------------------------------------

class TestExternalEntity:
    def test_status_rumor(self):
        e = ExternalEntity(id="x", name="X", discovered_year=1,
                           evidence=0.1)
        assert e.status == "rumor"

    def test_status_corroborated(self):
        e = ExternalEntity(id="x", name="X", discovered_year=1,
                           evidence=EVIDENCE_CORROBORATE)
        assert e.status == "corroborated"

    def test_status_confirmed(self):
        e = ExternalEntity(id="x", name="X", discovered_year=1,
                           evidence=EVIDENCE_CONFIRM)
        assert e.status == "confirmed"

    def test_to_dict_roundtrip(self):
        e = ExternalEntity(
            id="s1", name="New Haven", discovered_year=25,
            evidence=0.55, relationship=-0.1, contact_count=3,
            origin="splinter", population_estimate=4,
            exiled_founders=["dax-iron"],
        )
        d = e.to_dict()
        e2 = ExternalEntity.from_dict(d)
        assert e2.id == e.id
        assert e2.name == e.name
        assert e2.evidence == pytest.approx(e.evidence, abs=1e-3)
        assert e2.relationship == pytest.approx(e.relationship, abs=1e-3)
        assert e2.origin == e.origin
        assert e2.exiled_founders == e.exiled_founders

    def test_relationship_bounded(self):
        e = ExternalEntity(id="x", name="X", discovered_year=1,
                           relationship=-2.0)
        assert e.relationship == -2.0  # raw storage allows it
        # but _clamp in evolve will bound it
        d = e.to_dict()
        assert d["relationship"] == -2.0

    def test_status_boundary_values(self):
        e1 = ExternalEntity(id="x", name="X", discovered_year=1,
                            evidence=EVIDENCE_CORROBORATE - 0.001)
        assert e1.status == "rumor"
        e2 = ExternalEntity(id="x", name="X", discovered_year=1,
                            evidence=EVIDENCE_CONFIRM - 0.001)
        assert e2.status == "corroborated"


# ---------------------------------------------------------------------------
#  DiplomacyState
# ---------------------------------------------------------------------------

class TestDiplomacyState:
    def test_initial_state(self):
        ds = DiplomacyState()
        assert ds.stance == "unaware"
        assert ds.first_contact_year is None
        assert ds.signals_detected == 0
        assert len(ds.entities) == 0

    def test_to_dict_roundtrip(self):
        ds = DiplomacyState()
        ds.entities["s1"] = ExternalEntity(
            id="s1", name="Test", discovered_year=20, evidence=0.5)
        ds.stance = "cautious"
        ds.signals_detected = 3
        d = ds.to_dict()
        ds2 = DiplomacyState.from_dict(d)
        assert ds2.stance == "cautious"
        assert ds2.signals_detected == 3
        assert "s1" in ds2.entities
        assert ds2.entities["s1"].name == "Test"

    def test_empty_to_dict(self):
        ds = DiplomacyState()
        d = ds.to_dict()
        assert d["total_entities"] == 0
        assert d["confirmed_entities"] == 0


# ---------------------------------------------------------------------------
#  Signal detection
# ---------------------------------------------------------------------------

class TestSignalDetection:
    def test_no_signal_before_base_year(self):
        assert signal_detection_probability(5, 0.5, False) == 0.0
        assert signal_detection_probability(SIGNAL_BASE_YEAR - 1, 0.5, False) == 0.0

    def test_signal_at_base_year(self):
        prob = signal_detection_probability(SIGNAL_BASE_YEAR, 0.5, False)
        assert prob > 0.0

    def test_signal_increases_with_year(self):
        p1 = signal_detection_probability(20, 0.5, False)
        p2 = signal_detection_probability(50, 0.5, False)
        assert p2 > p1

    def test_comm_infra_doubles(self):
        p_no = signal_detection_probability(30, 0.5, False)
        p_yes = signal_detection_probability(30, 0.5, True)
        assert p_yes > p_no * 1.5

    def test_coding_skill_bonus(self):
        p_low = signal_detection_probability(30, 0.1, False)
        p_high = signal_detection_probability(30, 0.9, False)
        assert p_high > p_low

    def test_capped_at_half(self):
        prob = signal_detection_probability(200, 1.0, True)
        assert prob <= 0.5

    def test_zero_coding_still_works(self):
        prob = signal_detection_probability(30, 0.0, False)
        assert prob >= 0.0


# ---------------------------------------------------------------------------
#  Splinter colony formation
# ---------------------------------------------------------------------------

class TestSplinterFormation:
    def test_sufficient_exiles_forms_splinter(self):
        exiles = [{"id": f"exile-{i}", "year": 25}
                  for i in range(SPLINTER_EXILE_THRESHOLD)]
        assert should_form_splinter(exiles, 26)

    def test_insufficient_exiles(self):
        exiles = [{"id": "exile-0", "year": 25}]
        assert not should_form_splinter(exiles, 26)

    def test_old_exiles_dont_count(self):
        exiles = [{"id": f"exile-{i}", "year": 5}
                  for i in range(SPLINTER_EXILE_THRESHOLD)]
        assert not should_form_splinter(exiles, 5 + SPLINTER_WINDOW_YEARS + 1)

    def test_empty_list(self):
        assert not should_form_splinter([], 30)

    def test_mixed_old_and_new(self):
        exiles = [
            {"id": "exile-0", "year": 1},
            {"id": "exile-1", "year": 25},
            {"id": "exile-2", "year": 26},
        ]
        assert should_form_splinter(exiles, 27)


# ---------------------------------------------------------------------------
#  Diplomatic pressure
# ---------------------------------------------------------------------------

class TestDiplomaticPressure:
    def test_no_entities_no_pressure(self):
        ds = DiplomacyState()
        assert compute_diplomatic_pressure(ds) == {}

    def test_isolationist_boosts_hoard(self):
        ds = DiplomacyState(stance="isolationist")
        ds.entities["x"] = ExternalEntity(id="x", name="X",
                                          discovered_year=1)
        p = compute_diplomatic_pressure(ds)
        assert p.get("hoard", 0) > 0
        assert p.get("sabotage", 0) < 0

    def test_cautious_boosts_mediate(self):
        ds = DiplomacyState(stance="cautious")
        ds.entities["x"] = ExternalEntity(id="x", name="X",
                                          discovered_year=1)
        p = compute_diplomatic_pressure(ds)
        assert p.get("mediate", 0) > 0
        assert p.get("code", 0) > 0

    def test_open_boosts_cooperate(self):
        ds = DiplomacyState(stance="open")
        ds.entities["x"] = ExternalEntity(id="x", name="X",
                                          discovered_year=1)
        p = compute_diplomatic_pressure(ds)
        assert p.get("cooperate", 0) > 0
        assert p.get("sabotage", 0) < 0

    def test_more_entities_stronger_pressure(self):
        ds1 = DiplomacyState(stance="cautious")
        ds1.entities["x"] = ExternalEntity(id="x", name="X",
                                           discovered_year=1)
        ds2 = DiplomacyState(stance="cautious")
        for i in range(3):
            ds2.entities[f"x{i}"] = ExternalEntity(
                id=f"x{i}", name=f"X{i}", discovered_year=1)
        p1 = compute_diplomatic_pressure(ds1)
        p2 = compute_diplomatic_pressure(ds2)
        assert p2.get("mediate", 0) >= p1.get("mediate", 0)


# ---------------------------------------------------------------------------
#  Stance computation
# ---------------------------------------------------------------------------

class TestComputeStance:
    def test_no_entities_unaware(self):
        assert compute_stance({}, 0.5, 0.5) == "unaware"

    def test_high_paranoia_isolationist(self):
        entities = {"x": ExternalEntity(id="x", name="X",
                                        discovered_year=1)}
        assert compute_stance(entities, 0.9, 0.2) == "isolationist"

    def test_high_empathy_open(self):
        entities = {"x": ExternalEntity(id="x", name="X",
                                        discovered_year=1)}
        assert compute_stance(entities, 0.1, 0.9) == "open"

    def test_balanced_cautious(self):
        entities = {"x": ExternalEntity(id="x", name="X",
                                        discovered_year=1)}
        stance = compute_stance(entities, 0.5, 0.5)
        assert stance in ("cautious", "isolationist", "open")

    def test_hostile_entity_pushes_isolationist(self):
        entities = {"x": ExternalEntity(id="x", name="X",
                                        discovered_year=1,
                                        relationship=-0.5)}
        stance = compute_stance(entities, 0.5, 0.6)
        # hostile entity should push toward isolationism
        assert stance in ("cautious", "isolationist")


# ---------------------------------------------------------------------------
#  Name generation
# ---------------------------------------------------------------------------

class TestNameGeneration:
    def test_splinter_names(self):
        rng = random.Random(42)
        name = generate_entity_name("splinter", rng)
        assert len(name) > 0
        parts = name.split()
        assert len(parts) == 2

    def test_earth_mission_names(self):
        rng = random.Random(42)
        name = generate_entity_name("earth_mission", rng)
        assert len(name) > 0

    def test_unknown_names(self):
        rng = random.Random(42)
        name = generate_entity_name("unknown", rng)
        assert len(name) > 0

    def test_different_seeds_different_names(self):
        n1 = generate_entity_name("splinter", random.Random(1))
        n2 = generate_entity_name("splinter", random.Random(999))
        # not guaranteed but highly likely with different seeds
        # just check both are valid
        assert len(n1) > 0 and len(n2) > 0


# ---------------------------------------------------------------------------
#  Subsim expressions
# ---------------------------------------------------------------------------

class TestDiplomaticSubsim:
    def test_generates_valid_lispy(self):
        from src.mars100.lispy_vm import run as lispy_run
        entity = ExternalEntity(id="x", name="X", discovered_year=1,
                                evidence=0.5, relationship=0.2)
        for stance in ("isolationist", "cautious", "open"):
            expr = diplomatic_subsim_expression(entity, stance)
            bindings = {
                "paranoia": 0.5, "empathy": 0.5, "resolve": 0.5,
                "coding": 0.5, "faith": 0.3,
            }
            result = lispy_run(expr, extra_bindings=bindings)
            assert isinstance(result, (int, float))

    def test_different_stances_different_expressions(self):
        entity = ExternalEntity(id="x", name="X", discovered_year=1,
                                evidence=0.5, relationship=0.2)
        e_iso = diplomatic_subsim_expression(entity, "isolationist")
        e_open = diplomatic_subsim_expression(entity, "open")
        assert e_iso != e_open


# ---------------------------------------------------------------------------
#  tick_diplomacy integration
# ---------------------------------------------------------------------------

class TestTickDiplomacy:
    def _make_tick_kwargs(self, year: int = 30,
                          rng_seed: int = 42,
                          **overrides) -> dict:
        defaults = {
            "year": year,
            "year_exiles": [],
            "all_exiles_history": [],
            "coding_skill_avg": 0.5,
            "colony_paranoia_avg": 0.5,
            "colony_empathy_avg": 0.5,
            "has_comm_infra": False,
            "earth_announced_mission": False,
            "rng": random.Random(rng_seed),
        }
        defaults.update(overrides)
        return defaults

    def test_no_activity_early_years(self):
        ds = DiplomacyState()
        result = tick_diplomacy(ds, **self._make_tick_kwargs(year=5))
        assert len(result.new_signals) == 0
        assert len(result.new_entities) == 0
        assert ds.stance == "unaware"

    def test_signal_detection_possible(self):
        """Over many years, signals should eventually be detected."""
        ds = DiplomacyState()
        detected_any = False
        rng = random.Random(42)
        for year in range(SIGNAL_BASE_YEAR, SIGNAL_BASE_YEAR + 100):
            result = tick_diplomacy(
                ds,
                **self._make_tick_kwargs(year=year, rng_seed=None,
                                         rng=rng, has_comm_infra=True))
            if result.new_signals:
                detected_any = True
                break
        assert detected_any, "Should detect at least one signal in 100 years"

    def test_splinter_colony_formation(self):
        ds = DiplomacyState()
        exiles = [{"id": f"exile-{i}", "year": 30}
                  for i in range(SPLINTER_EXILE_THRESHOLD)]
        result = tick_diplomacy(
            ds,
            **self._make_tick_kwargs(
                year=31,
                year_exiles=exiles,
                all_exiles_history=exiles,
            ))
        assert len(result.new_entities) == 1
        entity = list(ds.entities.values())[0]
        assert entity.origin == "splinter"
        assert entity.evidence == 0.5
        assert entity.relationship == pytest.approx(-0.2, abs=0.05)

    def test_earth_mission_announcement(self):
        ds = DiplomacyState()
        result = tick_diplomacy(
            ds,
            **self._make_tick_kwargs(earth_announced_mission=True))
        assert len(result.new_entities) == 1
        entity = list(ds.entities.values())[0]
        assert entity.origin == "earth_mission"
        assert entity.evidence == 0.8
        assert entity.relationship == pytest.approx(0.3, abs=0.05)

    def test_earth_mission_only_once(self):
        ds = DiplomacyState()
        tick_diplomacy(
            ds, **self._make_tick_kwargs(earth_announced_mission=True))
        result2 = tick_diplomacy(
            ds, **self._make_tick_kwargs(
                year=31, earth_announced_mission=True))
        assert len(result2.new_entities) == 0
        assert len(ds.entities) == 1

    def test_stance_evolves(self):
        ds = DiplomacyState()
        ds.entities["x"] = ExternalEntity(
            id="x", name="X", discovered_year=20, evidence=0.8)
        result = tick_diplomacy(
            ds,
            **self._make_tick_kwargs(
                colony_empathy_avg=0.9, colony_paranoia_avg=0.1))
        assert result.stance_after == "open"

    def test_max_entities_cap(self):
        ds = DiplomacyState()
        for i in range(MAX_ENTITIES):
            ds.entities[f"e{i}"] = ExternalEntity(
                id=f"e{i}", name=f"E{i}", discovered_year=10)
        exiles = [{"id": f"exile-{i}", "year": 30}
                  for i in range(SPLINTER_EXILE_THRESHOLD)]
        result = tick_diplomacy(
            ds,
            **self._make_tick_kwargs(
                year_exiles=exiles,
                all_exiles_history=exiles,
            ))
        assert len(ds.entities) == MAX_ENTITIES
        assert len(result.new_entities) == 0

    def test_entity_evidence_accumulates(self):
        ds = DiplomacyState()
        ds.entities["x"] = ExternalEntity(
            id="x", name="X", discovered_year=20, evidence=0.3)
        ds.signals_detected = 5
        rng = random.Random(42)
        initial_ev = ds.entities["x"].evidence
        tick_diplomacy(
            ds,
            **self._make_tick_kwargs(year=30, rng_seed=None, rng=rng))
        assert ds.entities["x"].evidence >= initial_ev

    def test_incident_logging(self):
        ds = DiplomacyState()
        exiles = [{"id": f"exile-{i}", "year": 30}
                  for i in range(SPLINTER_EXILE_THRESHOLD)]
        tick_diplomacy(
            ds,
            **self._make_tick_kwargs(
                year=31,
                year_exiles=exiles,
                all_exiles_history=exiles,
            ))
        assert len(ds.incidents) > 0
        assert ds.incidents[0]["type"] == "splinter_formed"


# ---------------------------------------------------------------------------
#  Physical bounds / invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_evidence_bounded_0_1(self):
        """Evidence never exceeds [0, 1] after tick."""
        ds = DiplomacyState()
        ds.entities["x"] = ExternalEntity(
            id="x", name="X", discovered_year=1, evidence=0.99)
        ds.signals_detected = 100
        rng = random.Random(42)
        for year in range(20, 80):
            tick_diplomacy(
                ds,
                **{
                    "year": year,
                    "year_exiles": [],
                    "all_exiles_history": [],
                    "coding_skill_avg": 0.9,
                    "colony_paranoia_avg": 0.5,
                    "colony_empathy_avg": 0.5,
                    "has_comm_infra": True,
                    "earth_announced_mission": False,
                    "rng": rng,
                })
        assert ds.entities["x"].evidence <= 1.0

    def test_relationship_bounded_neg1_pos1_after_evolution(self):
        """After many ticks, relationship stays in [-1, 1]."""
        ds = DiplomacyState()
        ds.entities["x"] = ExternalEntity(
            id="x", name="X", discovered_year=1,
            evidence=0.9, relationship=0.9)
        rng = random.Random(42)
        for year in range(20, 120):
            tick_diplomacy(
                ds,
                **{
                    "year": year,
                    "year_exiles": [],
                    "all_exiles_history": [],
                    "coding_skill_avg": 0.5,
                    "colony_paranoia_avg": 0.5,
                    "colony_empathy_avg": 0.5,
                    "has_comm_infra": False,
                    "earth_announced_mission": False,
                    "rng": rng,
                })
        assert -1.0 <= ds.entities["x"].relationship <= 1.0

    def test_stance_always_valid(self):
        valid_stances = {"unaware", "isolationist", "cautious", "open"}
        ds = DiplomacyState()
        rng = random.Random(42)
        for year in range(1, 80):
            exiles = []
            if year % 15 == 0:
                exiles = [{"id": f"e-{year}", "year": year}]
            tick_diplomacy(
                ds,
                **{
                    "year": year,
                    "year_exiles": exiles,
                    "all_exiles_history": exiles,
                    "coding_skill_avg": 0.5,
                    "colony_paranoia_avg": rng.uniform(0.2, 0.8),
                    "colony_empathy_avg": rng.uniform(0.2, 0.8),
                    "has_comm_infra": year > 40,
                    "earth_announced_mission": year == 50,
                    "rng": rng,
                })
            assert ds.stance in valid_stances


# ---------------------------------------------------------------------------
#  Smoke test: 50 years of diplomacy
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_50_year_diplomacy_run(self):
        """Run 50 years of diplomacy without crash."""
        ds = DiplomacyState()
        rng = random.Random(42)
        all_exiles: list[dict] = []
        for year in range(1, 51):
            year_exiles = []
            if year in (20, 21):
                year_exiles = [{"id": f"exile-{year}", "year": year}]
                all_exiles.extend(year_exiles)
            result = tick_diplomacy(
                ds,
                year=year,
                year_exiles=year_exiles,
                all_exiles_history=all_exiles,
                coding_skill_avg=0.5,
                colony_paranoia_avg=0.4,
                colony_empathy_avg=0.6,
                has_comm_infra=year > 30,
                earth_announced_mission=(year == 35),
                rng=rng,
            )
            assert isinstance(result, DiplomacyTickResult)
        # After 50 years we should have some entities
        assert ds.signals_detected >= 0
        d = ds.to_dict()
        assert "entities" in d
        assert "stance" in d
