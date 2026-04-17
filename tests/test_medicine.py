"""Tests for the medicine / health organ (engine v9.0)."""
from __future__ import annotations

import random
import pytest

from src.mars100.medicine import (
    HealthState, MedicalState, ColonistHealthContext, MedicineTickResult,
    tick_medicine, check_health_death, _age_decay_rate, _compute_vitality,
    _check_epidemic_start,
    BASE_RADIATION_PER_YEAR, RADIATION_DUST_STORM_BONUS,
    RADIATION_SOLAR_FLARE_BONUS, OLD_AGE_VITALITY_THRESHOLD,
    OLD_AGE_MIN_BIO_AGE, RADIATION_DEATH_THRESHOLD,
    INJURY_DEATH_THRESHOLD, EPIDEMIC_DEATH_THRESHOLD,
    FOUNDER_AGE_OFFSET, IMMIGRANT_AGE_OFFSET, CHILD_AGE_OFFSET,
    BASE_MEDICAL_CAPACITY, MED_BAY_CAPACITY_BONUS,
    EPIDEMIC_CROWDING_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(colonist_id: str = "c-0", birth_year: int = 0,
                  action: str = "rest", event_severity: float = 0.0,
                  event_name: str = "calm", has_med_bay: bool = False,
                  population: int = 10, food: float = 0.8,
                  water: float = 0.8, medicine: float = 0.5) -> ColonistHealthContext:
    return ColonistHealthContext(
        colonist_id=colonist_id, birth_year=birth_year, action=action,
        event_severity=event_severity, event_name=event_name,
        has_med_bay=has_med_bay, population=population,
        food_level=food, water_level=water, medicine_level=medicine)


def _make_contexts(n: int = 5, **kwargs) -> list[ColonistHealthContext]:
    return [_make_context(colonist_id=f"c-{i}", **kwargs) for i in range(n)]


# ---------------------------------------------------------------------------
# HealthState basics
# ---------------------------------------------------------------------------

class TestHealthState:
    def test_defaults(self):
        hs = HealthState()
        assert hs.vitality == 1.0
        assert hs.injury_load == 0.0
        assert hs.disease_load == 0.0
        assert hs.radiation == 0.0
        assert hs.age_offset == FOUNDER_AGE_OFFSET

    def test_biological_age_founder(self):
        hs = HealthState(age_offset=30)
        assert hs.biological_age(50, 0) == 80  # 30 + (50 - 0)

    def test_biological_age_child(self):
        hs = HealthState(age_offset=0)
        assert hs.biological_age(50, 20) == 30  # 0 + (50 - 20)

    def test_biological_age_immigrant(self):
        hs = HealthState(age_offset=28)
        assert hs.biological_age(60, 30) == 58  # 28 + (60 - 30)

    def test_round_trip(self):
        hs = HealthState(vitality=0.75, injury_load=0.1, disease_load=0.2,
                         radiation=0.3, age_offset=28)
        d = hs.to_dict()
        hs2 = HealthState.from_dict(d)
        assert abs(hs2.vitality - 0.75) < 0.001
        assert abs(hs2.radiation - 0.3) < 0.001
        assert hs2.age_offset == 28


# ---------------------------------------------------------------------------
# MedicalState basics
# ---------------------------------------------------------------------------

class TestMedicalState:
    def test_defaults(self):
        ms = MedicalState()
        assert ms.medical_capacity == BASE_MEDICAL_CAPACITY
        assert not ms.epidemic_active
        assert ms.treatments_given == 0

    def test_round_trip(self):
        ms = MedicalState(epidemic_active=True, epidemic_year_started=10,
                          epidemic_severity=0.6, epidemic_duration=3,
                          treatments_given=42)
        d = ms.to_dict()
        ms2 = MedicalState.from_dict(d)
        assert ms2.epidemic_active is True
        assert ms2.epidemic_year_started == 10
        assert ms2.treatments_given == 42


# ---------------------------------------------------------------------------
# Age decay
# ---------------------------------------------------------------------------

class TestAgeDecay:
    def test_young_decays_slowly(self):
        rate = _age_decay_rate(10)
        assert rate == 0.001

    def test_adult_moderate(self):
        rate = _age_decay_rate(35)
        assert 0.002 < rate < 0.005

    def test_elder_faster(self):
        rate = _age_decay_rate(60)
        assert rate > _age_decay_rate(40)

    def test_ancient_fastest(self):
        rate = _age_decay_rate(80)
        assert rate > _age_decay_rate(65)
        assert rate > 0.013

    def test_monotonically_increasing(self):
        """Decay rate must always increase with age."""
        rates = [_age_decay_rate(a) for a in range(0, 120)]
        for i in range(1, len(rates)):
            assert rates[i] >= rates[i - 1], f"Non-monotonic at age {i}"


# ---------------------------------------------------------------------------
# Vitality computation
# ---------------------------------------------------------------------------

class TestVitalityFormula:
    def test_perfect_health(self):
        v = _compute_vitality(1.0, 0.0, 0.0, 0.0)
        assert v == 1.0

    def test_full_injury(self):
        v = _compute_vitality(1.0, 1.0, 0.0, 0.0)
        assert v == pytest.approx(0.7, abs=0.01)

    def test_full_disease(self):
        v = _compute_vitality(1.0, 0.0, 1.0, 0.0)
        assert v == pytest.approx(0.6, abs=0.01)

    def test_full_radiation(self):
        v = _compute_vitality(1.0, 0.0, 0.0, 1.0)
        assert v == pytest.approx(0.8, abs=0.01)

    def test_all_stressors_compound(self):
        v = _compute_vitality(1.0, 0.5, 0.5, 0.5)
        # (1 - 0.15) * (1 - 0.2) * (1 - 0.1) = 0.85 * 0.8 * 0.9 = 0.612
        assert v == pytest.approx(0.612, abs=0.01)

    def test_never_negative(self):
        v = _compute_vitality(0.1, 1.0, 1.0, 1.0)
        assert v >= 0.0

    def test_clamped_to_one(self):
        v = _compute_vitality(1.5, 0.0, 0.0, 0.0)
        assert v <= 1.0


# ---------------------------------------------------------------------------
# Epidemic trigger
# ---------------------------------------------------------------------------

class TestEpidemicTrigger:
    def test_no_epidemic_small_population(self):
        ms = MedicalState()
        rng = random.Random(42)
        # population below threshold
        assert not _check_epidemic_start(ms, 10, 0.1, 0.1, 0.1, rng)

    def test_no_epidemic_good_resources(self):
        ms = MedicalState()
        rng = random.Random(42)
        # resources above 0.3
        result = _check_epidemic_start(ms, 20, 0.8, 0.8, 0.8, rng)
        assert not result

    def test_no_epidemic_already_active(self):
        ms = MedicalState(epidemic_active=True)
        rng = random.Random(42)
        assert not _check_epidemic_start(ms, 30, 0.1, 0.1, 0.1, rng)

    def test_epidemic_possible_with_scarcity_and_crowding(self):
        """With enough trials, an epidemic should eventually start."""
        triggered = False
        for seed in range(200):
            ms = MedicalState()
            rng = random.Random(seed)
            if _check_epidemic_start(ms, 25, 0.1, 0.1, 0.1, rng):
                triggered = True
                break
        assert triggered, "Epidemic never triggered despite conditions"


# ---------------------------------------------------------------------------
# Death cause checks
# ---------------------------------------------------------------------------

class TestHealthDeath:
    def test_old_age(self):
        hs = HealthState(vitality=0.05, age_offset=50)
        cause = check_health_death(hs, 50, 0)  # bio_age = 100
        assert cause == "old_age"

    def test_no_old_age_if_young(self):
        hs = HealthState(vitality=0.05, age_offset=0)
        cause = check_health_death(hs, 20, 10)  # bio_age = 10
        assert cause is None  # too young

    def test_radiation_sickness(self):
        hs = HealthState(radiation=0.95)
        cause = check_health_death(hs, 50, 0)
        assert cause == "radiation_sickness"

    def test_untreated_injury(self):
        hs = HealthState(injury_load=0.95)
        cause = check_health_death(hs, 50, 0)
        assert cause == "untreated_injury"

    def test_epidemic_death(self):
        hs = HealthState(disease_load=0.85)
        cause = check_health_death(hs, 50, 0)
        assert cause == "epidemic"

    def test_healthy_no_death(self):
        hs = HealthState()
        cause = check_health_death(hs, 50, 0)
        assert cause is None

    def test_priority_old_age_over_radiation(self):
        """Old age checked first."""
        hs = HealthState(vitality=0.03, radiation=0.95, age_offset=50)
        cause = check_health_death(hs, 50, 0)
        assert cause == "old_age"


# ---------------------------------------------------------------------------
# tick_medicine integration
# ---------------------------------------------------------------------------

class TestTickMedicine:
    def test_basic_tick(self):
        health_map: dict[str, HealthState] = {}
        med_state = MedicalState()
        contexts = _make_contexts(5, birth_year=0)
        rng = random.Random(42)
        result = tick_medicine(health_map, med_state, contexts, year=10, rng=rng)
        assert isinstance(result, MedicineTickResult)
        assert len(health_map) == 5
        assert result.avg_vitality > 0.0
        assert result.avg_vitality <= 1.0

    def test_vitality_decreases_with_time(self):
        """Running many years should reduce vitality."""
        health_map: dict[str, HealthState] = {}
        med_state = MedicalState()
        rng = random.Random(42)
        for year in range(1, 51):
            contexts = _make_contexts(3, birth_year=0)
            tick_medicine(health_map, med_state, contexts, year=year, rng=rng)
        for hs in health_map.values():
            assert hs.vitality < 0.95, "Vitality should decrease over 50 years"

    def test_radiation_accumulates(self):
        """Radiation should monotonically increase over time."""
        health_map: dict[str, HealthState] = {}
        med_state = MedicalState()
        rng = random.Random(42)
        prev_rad = 0.0
        for year in range(1, 30):
            contexts = [_make_context(colonist_id="c-0", birth_year=0)]
            tick_medicine(health_map, med_state, contexts, year=year, rng=rng)
            current_rad = health_map["c-0"].radiation
            assert current_rad >= prev_rad, f"Radiation decreased at year {year}"
            prev_rad = current_rad
        assert prev_rad > 0.1, "Radiation should accumulate significantly"

    def test_event_causes_injury(self):
        """High-severity events should cause injuries (probabilistically)."""
        injured = False
        for seed in range(50):
            health_map: dict[str, HealthState] = {}
            med_state = MedicalState()
            rng = random.Random(seed)
            contexts = _make_contexts(10, birth_year=0, event_severity=0.8,
                                      event_name="dust_storm")
            result = tick_medicine(health_map, med_state, contexts, year=5, rng=rng)
            if result.injuries_this_year > 0:
                injured = True
                break
        assert injured, "No injuries from high-severity events across 50 seeds"

    def test_med_bay_increases_capacity(self):
        health_map: dict[str, HealthState] = {}
        med_state = MedicalState()
        contexts = _make_contexts(5, has_med_bay=True)
        rng = random.Random(42)
        tick_medicine(health_map, med_state, contexts, year=1, rng=rng)
        assert med_state.medical_capacity == BASE_MEDICAL_CAPACITY + MED_BAY_CAPACITY_BONUS

    def test_treatment_reduces_injury(self):
        """Injured colonists who get treated should heal partially."""
        health_map = {"c-0": HealthState(injury_load=0.5)}
        med_state = MedicalState()
        contexts = [_make_context(colonist_id="c-0", birth_year=0)]
        rng = random.Random(42)
        tick_medicine(health_map, med_state, contexts, year=10, rng=rng)
        assert health_map["c-0"].injury_load < 0.5

    def test_treatment_capacity_limits(self):
        """Only medical_capacity colonists should be treated per year."""
        n_colonists = 10
        health_map = {f"c-{i}": HealthState(injury_load=0.5) for i in range(n_colonists)}
        med_state = MedicalState()  # capacity = 2
        contexts = _make_contexts(n_colonists, birth_year=0)
        rng = random.Random(42)
        result = tick_medicine(health_map, med_state, contexts, year=10, rng=rng)
        assert len(result.treatments) <= med_state.medical_capacity

    def test_empty_contexts(self):
        health_map: dict[str, HealthState] = {}
        med_state = MedicalState()
        rng = random.Random(42)
        result = tick_medicine(health_map, med_state, [], year=10, rng=rng)
        assert result.avg_vitality == 1.0

    def test_epidemic_lifecycle(self):
        """Epidemic should start, persist, then end."""
        health_map: dict[str, HealthState] = {}
        med_state = MedicalState()
        epidemic_seen = False
        epidemic_ended = False
        for seed in range(100):
            health_map.clear()
            med_state = MedicalState()
            rng = random.Random(seed)
            for year in range(1, 40):
                contexts = _make_contexts(
                    20, birth_year=0, food=0.15, water=0.15, medicine=0.1,
                    population=20)
                result = tick_medicine(health_map, med_state, contexts,
                                       year=year, rng=rng)
                if result.epidemic_started:
                    epidemic_seen = True
                if result.epidemic_ended:
                    epidemic_ended = True
                    break
            if epidemic_ended:
                break
        assert epidemic_seen, "No epidemic triggered across 100 seeds"
        assert epidemic_ended, "No epidemic resolved across 100 seeds"


# ---------------------------------------------------------------------------
# Property-based invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    @pytest.mark.parametrize("seed", range(20))
    def test_all_values_bounded(self, seed):
        """All health values remain in [0, 1] after 100 ticks."""
        health_map: dict[str, HealthState] = {}
        med_state = MedicalState()
        rng = random.Random(seed)
        for year in range(1, 101):
            contexts = _make_contexts(
                5, birth_year=0, event_severity=rng.uniform(0, 0.9),
                event_name=rng.choice(["calm", "dust_storm", "solar_flare"]),
                population=15 + seed, food=rng.uniform(0.1, 0.9),
                water=rng.uniform(0.1, 0.9), medicine=rng.uniform(0.1, 0.9))
            tick_medicine(health_map, med_state, contexts, year=year, rng=rng)
        for cid, hs in health_map.items():
            assert 0.0 <= hs.vitality <= 1.0, f"{cid} vitality out of bounds"
            assert 0.0 <= hs.injury_load <= 1.0, f"{cid} injury out of bounds"
            assert 0.0 <= hs.disease_load <= 1.0, f"{cid} disease out of bounds"
            assert 0.0 <= hs.radiation <= 1.0, f"{cid} radiation out of bounds"

    @pytest.mark.parametrize("seed", range(10))
    def test_radiation_monotonic(self, seed):
        """Radiation must never decrease."""
        health_map: dict[str, HealthState] = {}
        med_state = MedicalState()
        rng = random.Random(seed)
        prev_rads: dict[str, float] = {}
        for year in range(1, 51):
            contexts = _make_contexts(3, birth_year=0,
                                      event_name="dust_storm",
                                      event_severity=0.5)
            tick_medicine(health_map, med_state, contexts, year=year, rng=rng)
            for cid, hs in health_map.items():
                prev = prev_rads.get(cid, 0.0)
                assert hs.radiation >= prev, f"{cid} radiation decreased at year {year}"
                prev_rads[cid] = hs.radiation
