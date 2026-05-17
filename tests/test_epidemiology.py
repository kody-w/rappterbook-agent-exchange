"""Tests for the epidemiology organ (engine v12.0)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.epidemiology import (
    BASE_RECOVERY_RATE, DISEASE_DEATH_BASE, DEFAULT_BASE_R0,
    EXPOSURE_TRUST_THRESHOLD, IMMUNITY_DECAY_PER_YEAR,
    MAX_COHESION_PENALTY, MAX_STRESS_BUMP,
    PANDEMIC_PREVALENCE, QUARANTINE_PREVALENCE,
    Disease, EpidemiologyState, EpidemiologyTickResult,
    EpidemiologyYearContext, HealthRecord, InfectionEvent,
    behavior_action_bias, colonist_health_summary, colonist_is_infected,
    colonist_strain, compute_effective_r0, compute_emergence_probability,
    diplomacy_cohesion_penalty, stress_bump, tick_epidemiology,
    E, I, R, S,
)


class FakeRel:
    def __init__(self, trust: float = 0.6):
        self.trust = trust


def make_social_get(default_trust: float = 0.6):
    def _get(a: str, b: str) -> FakeRel:
        return FakeRel(trust=default_trust)
    return _get


def make_colonist(cid: str, age: int = 30,
                  resolve: float = 0.5, faith: float = 0.5) -> dict:
    return {
        "id": cid, "age": age,
        "stats": {"resolve": resolve, "faith": faith,
                  "empathy": 0.5, "paranoia": 0.3},
    }


def make_population(n: int, ages: list[int] | None = None) -> list[dict]:
    if ages is None:
        ages = [30] * n
    return [make_colonist(f"c-{i}", age=ages[i]) for i in range(n)]


def make_context(active, *, year: int = 1, medicine: float = 0.6,
                 actions: dict | None = None, trust: float = 0.6,
                 infra: list[str] | None = None, pollution: float = 0.0,
                 immigrants: int = 0, quarantine: bool = False
                 ) -> EpidemiologyYearContext:
    actions = actions or {c["id"]: "rest" for c in active}
    return EpidemiologyYearContext(
        year=year, active_colonists=active, actions=actions,
        social_get=make_social_get(trust),
        medicine_level=medicine, population=len(active),
        infrastructure_completed=infra or [],
        ecology_pollution=pollution,
        immigrants_this_year=immigrants,
        quarantine_active=quarantine,
    )


class HighRng:
    """RNG that always rolls high: no recoveries, no deaths, no emergences."""
    def random(self): return 0.999
    def randint(self, a, b): return a
    def choice(self, seq): return seq[0]
    def choices(self, seq, weights=None, k=1): return [seq[0]] * k


# ---------------------------------------------------------------------------
# Roundtrips
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_disease_roundtrip(self) -> None:
        d = Disease(id="x", name="Test", origin_year=3, r0=1.5,
                    mortality=0.1, duration=1.2, immunity_duration=4.0,
                    vector="contact", extinct=False,
                    total_infections=5, total_deaths=1, total_recoveries=4)
        d2 = Disease.from_dict(d.to_dict())
        assert d2 == d

    def test_health_record_roundtrip(self) -> None:
        rec = HealthRecord(status=I, active_strain="s1",
                           years_in_status=2,
                           immunities={"s0": 3.0, "s2": 1.5})
        rec2 = HealthRecord.from_dict(rec.to_dict())
        assert rec2.status == rec.status
        assert rec2.active_strain == rec.active_strain
        assert rec2.years_in_status == rec.years_in_status
        assert rec2.immunities == rec.immunities

    def test_state_roundtrip(self) -> None:
        st = EpidemiologyState()
        st.diseases["s1"] = Disease(id="s1", name="A", origin_year=1)
        st.health["c-0"] = HealthRecord(status=E, active_strain="s1")
        st.history.append({"year": 1, "event": "strain_emerged"})
        st.last_emergence_year = 1
        st2 = EpidemiologyState.from_dict(st.to_dict())
        assert "s1" in st2.diseases
        assert st2.health["c-0"].status == E
        assert st2.last_emergence_year == 1


# ---------------------------------------------------------------------------
# Emergence
# ---------------------------------------------------------------------------

class TestEmergence:
    def test_cooldown_blocks_emergence(self) -> None:
        assert compute_emergence_probability(0.5, 0.5, 0, 0) == 0.0
        assert compute_emergence_probability(0.5, 0.5, 0, 1) == 0.0

    def test_probability_bounded(self) -> None:
        for med in (0.0, 0.5, 1.0):
            for poll in (0.0, 0.5, 1.0):
                for imm in (0, 5, 50):
                    p = compute_emergence_probability(med, poll, imm, 100)
                    assert 0.0 <= p <= 0.20

    def test_low_medicine_raises_emergence(self) -> None:
        assert (compute_emergence_probability(0.0, 0.0, 0, 100)
                > compute_emergence_probability(1.0, 0.0, 0, 100))

    def test_pollution_raises_emergence(self) -> None:
        assert (compute_emergence_probability(0.5, 1.0, 0, 100)
                > compute_emergence_probability(0.5, 0.0, 0, 100))


# ---------------------------------------------------------------------------
# Effective R0
# ---------------------------------------------------------------------------

class TestEffectiveR0:
    def test_quarantine_cuts_r0(self) -> None:
        d = Disease(id="s", name="X", origin_year=1, r0=2.0)
        pop = make_population(10)
        assert (compute_effective_r0(d, make_context(pop, quarantine=True))
                < compute_effective_r0(d, make_context(pop, quarantine=False)))

    def test_med_bay_cuts_r0(self) -> None:
        d = Disease(id="s", name="X", origin_year=1, r0=2.0)
        pop = make_population(10)
        assert (compute_effective_r0(d, make_context(pop, infra=["med_bay"]))
                < compute_effective_r0(d, make_context(pop)))

    def test_rest_actions_lower_r0(self) -> None:
        d = Disease(id="s", name="X", origin_year=1, r0=2.0)
        pop = make_population(10)
        all_rest = make_context(pop, actions={c["id"]: "rest" for c in pop})
        all_work = make_context(pop, actions={c["id"]: "code" for c in pop})
        assert compute_effective_r0(d, all_rest) < compute_effective_r0(d, all_work)

    def test_r0_always_positive(self) -> None:
        d = Disease(id="s", name="X", origin_year=1, r0=0.5)
        pop = make_population(5)
        ctx = make_context(pop, quarantine=True,
                           infra=["med_bay", "air_recycler", "research_lab"])
        assert compute_effective_r0(d, ctx) > 0.0


# ---------------------------------------------------------------------------
# Tick mechanics
# ---------------------------------------------------------------------------

class TestTick:
    def test_no_disease_no_transmission(self) -> None:
        rng = random.Random(1)
        state = EpidemiologyState()
        pop = make_population(8)
        ctx = make_context(pop, medicine=0.9)
        result = tick_epidemiology(state, ctx, rng)
        assert all(e.via != "transmission" for e in result.new_infections)

    def test_strain_emerges_over_time(self) -> None:
        rng = random.Random(7)
        state = EpidemiologyState()
        pop = make_population(20)
        for year in range(1, 40):
            ctx = make_context(pop, year=year, medicine=0.2, pollution=0.8)
            tick_epidemiology(state, ctx, rng)
        assert len(state.diseases) >= 1

    def test_infected_can_recover(self) -> None:
        rng = random.Random(2)
        state = EpidemiologyState()
        d = Disease(id="s1", name="Test", origin_year=0,
                    r0=1.0, mortality=0.0, duration=1.0, immunity_duration=5.0)
        state.diseases["s1"] = d
        state.health["c-0"] = HealthRecord(status=I, active_strain="s1")
        pop = [make_colonist("c-0", resolve=0.9)]
        recovered = False
        for year in range(1, 10):
            tick_epidemiology(state, make_context(pop, year=year), rng)
            if state.health["c-0"].status == R:
                recovered = True
                break
        assert recovered

    def test_recovery_grants_immunity(self) -> None:
        rng = random.Random(3)
        state = EpidemiologyState()
        d = Disease(id="s1", name="Test", origin_year=0,
                    mortality=0.0, immunity_duration=5.0)
        state.diseases["s1"] = d
        state.health["c-0"] = HealthRecord(status=I, active_strain="s1")
        pop = [make_colonist("c-0")]
        for year in range(1, 8):
            tick_epidemiology(state, make_context(pop, year=year), rng)
            if state.health["c-0"].status == R:
                break
        assert state.health["c-0"].immunities.get("s1", 0) > 0

    def test_immunity_decays(self) -> None:
        rng = random.Random(4)
        state = EpidemiologyState()
        d = Disease(id="s1", name="Test", origin_year=0,
                    mortality=0.0, immunity_duration=2.0)
        state.diseases["s1"] = d
        state.health["c-0"] = HealthRecord(status=R, immunities={"s1": 0.05})
        pop = [make_colonist("c-0")]
        tick_epidemiology(state, make_context(pop, year=1), rng)
        assert "s1" not in state.health["c-0"].immunities

    def test_transmission_to_trusted_contacts(self) -> None:
        rng = random.Random(5)
        state = EpidemiologyState()
        d = Disease(id="s1", name="Plague", origin_year=0,
                    r0=3.0, mortality=0.0, duration=10.0)
        state.diseases["s1"] = d
        pop = make_population(10)
        state.health[pop[0]["id"]] = HealthRecord(status=I, active_strain="s1")
        result = tick_epidemiology(state, make_context(pop, year=1, trust=0.9, medicine=0.5), rng)
        assert len(result.new_infections) >= 1

    def test_no_transmission_when_isolated(self) -> None:
        rng = random.Random(6)
        state = EpidemiologyState()
        d = Disease(id="s1", name="Plague", origin_year=0,
                    r0=3.0, mortality=0.0, duration=10.0)
        state.diseases["s1"] = d
        pop = make_population(10)
        state.health[pop[0]["id"]] = HealthRecord(status=I, active_strain="s1")
        ctx = make_context(pop, year=1, trust=EXPOSURE_TRUST_THRESHOLD - 0.05)
        result = tick_epidemiology(state, ctx, rng)
        transmissions = [e for e in result.new_infections if e.via == "transmission"]
        assert transmissions == []

    def test_immune_colonists_not_reinfected(self) -> None:
        rng = random.Random(7)
        state = EpidemiologyState()
        d = Disease(id="s1", name="Plague", origin_year=0,
                    r0=3.0, mortality=0.0, duration=10.0)
        state.diseases["s1"] = d
        pop = make_population(10)
        state.health[pop[0]["id"]] = HealthRecord(status=I, active_strain="s1")
        for c in pop[1:]:
            state.health[c["id"]] = HealthRecord(status=R, immunities={"s1": 10.0})
        result = tick_epidemiology(state, make_context(pop, year=1, trust=0.9), rng)
        new = [e for e in result.new_infections
               if e.via == "transmission" and e.strain_id == "s1"]
        assert new == []


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_seir_status_always_valid(self) -> None:
        rng = random.Random(11)
        state = EpidemiologyState()
        pop = make_population(15)
        for year in range(1, 30):
            tick_epidemiology(state, make_context(pop, year=year,
                                                  medicine=0.3, pollution=0.4), rng)
            for rec in state.health.values():
                assert rec.status in (S, E, I, R)

    def test_seir_count_equals_active_population(self) -> None:
        rng = random.Random(12)
        state = EpidemiologyState()
        pop = make_population(12)
        for year in range(1, 20):
            tick_epidemiology(state, make_context(pop, year=year, medicine=0.3), rng)
            counts = sum(1 for r in state.health.values() if r.status in (S, E, I, R))
            assert counts == len(pop)

    def test_prevalence_in_unit_interval(self) -> None:
        rng = random.Random(13)
        state = EpidemiologyState()
        pop = make_population(20)
        for year in range(1, 25):
            result = tick_epidemiology(state, make_context(
                pop, year=year, medicine=0.2, pollution=0.7), rng)
            assert 0.0 <= result.prevalence <= 1.0
            assert 0.0 <= result.susceptible_fraction <= 1.0
            assert 0.0 <= result.recovered_fraction <= 1.0
            exposed = sum(1 for r in state.health.values() if r.status == E)
            total = (result.prevalence + result.susceptible_fraction
                     + result.recovered_fraction + exposed / len(pop))
            assert abs(total - 1.0) < 1e-9

    def test_cohesion_and_stress_caps(self) -> None:
        rng = random.Random(14)
        state = EpidemiologyState()
        d = Disease(id="s1", name="Doom", origin_year=0,
                    r0=3.5, mortality=0.0, duration=20.0, immunity_duration=10.0)
        state.diseases["s1"] = d
        pop = make_population(20)
        for c in pop:
            state.health[c["id"]] = HealthRecord(status=I, active_strain="s1")
        result = tick_epidemiology(state, make_context(
            pop, year=1, trust=0.95, medicine=0.0), rng)
        assert result.cohesion_penalty <= MAX_COHESION_PENALTY + 1e-9
        assert result.stress_bump <= MAX_STRESS_BUMP + 1e-9

    def test_quarantine_flag_triggers_at_threshold(self) -> None:
        rng = HighRng()
        state = EpidemiologyState()
        d = Disease(id="s1", name="Doom", origin_year=0,
                    r0=0.1, mortality=0.0, duration=500.0, immunity_duration=500.0)
        state.diseases["s1"] = d
        pop = make_population(10)
        for c in pop[:3]:
            state.health[c["id"]] = HealthRecord(status=I, active_strain="s1")
        for c in pop[3:]:
            state.health[c["id"]] = HealthRecord(status=S)
        result = tick_epidemiology(state, make_context(pop, year=1, trust=0.0), rng)
        assert result.quarantine_needed
        assert result.prevalence >= QUARANTINE_PREVALENCE

    def test_pandemic_flag(self) -> None:
        rng = HighRng()
        state = EpidemiologyState()
        d = Disease(id="s1", name="Doom", origin_year=0,
                    r0=0.1, mortality=0.0, duration=500.0)
        state.diseases["s1"] = d
        pop = make_population(10)
        for c in pop[:5]:
            state.health[c["id"]] = HealthRecord(status=I, active_strain="s1")
        for c in pop[5:]:
            state.health[c["id"]] = HealthRecord(status=S)
        result = tick_epidemiology(state, make_context(pop, year=1, trust=0.0), rng)
        assert result.pandemic

    def test_gc_removes_departed_colonists(self) -> None:
        rng = random.Random(17)
        state = EpidemiologyState()
        state.health["ghost"] = HealthRecord(status=I, active_strain="s1")
        pop = make_population(5)
        tick_epidemiology(state, make_context(pop, year=1), rng)
        assert "ghost" not in state.health

    def test_disease_counts_monotonic(self) -> None:
        rng = random.Random(18)
        state = EpidemiologyState()
        d = Disease(id="s1", name="X", origin_year=0,
                    r0=2.5, mortality=0.1, duration=2.0, immunity_duration=4.0)
        state.diseases["s1"] = d
        pop = make_population(15)
        state.health[pop[0]["id"]] = HealthRecord(status=I, active_strain="s1")
        prev_inf = prev_rec = prev_dead = 0
        for year in range(1, 15):
            tick_epidemiology(state, make_context(pop, year=year, trust=0.7), rng)
            assert d.total_infections >= prev_inf
            assert d.total_recoveries >= prev_rec
            assert d.total_deaths >= prev_dead
            prev_inf, prev_rec, prev_dead = (
                d.total_infections, d.total_recoveries, d.total_deaths)

    def test_extinct_strains_marked(self) -> None:
        rng = random.Random(19)
        state = EpidemiologyState()
        d = Disease(id="s1", name="Burnout", origin_year=0,
                    r0=0.0, mortality=0.0, duration=1.0)
        state.diseases["s1"] = d
        pop = make_population(8)
        state.health[pop[0]["id"]] = HealthRecord(status=I, active_strain="s1")
        for year in range(1, 10):
            tick_epidemiology(state, make_context(pop, year=year, trust=0.0), rng)
        assert state.diseases["s1"].extinct


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

class TestHooks:
    def test_behavior_bias_pushes_rest(self) -> None:
        state = EpidemiologyState()
        state.health["c-0"] = HealthRecord(status=I, active_strain="s1")
        state.health["c-1"] = HealthRecord(status=E, active_strain="s1")
        state.health["c-2"] = HealthRecord(status=R)
        bias = behavior_action_bias(state)
        assert bias["c-0"]["rest"] > 0
        assert bias["c-0"]["sabotage"] < 0
        assert bias["c-1"]["rest"] > 0
        assert "c-2" not in bias

    def test_colonist_is_infected(self) -> None:
        state = EpidemiologyState()
        state.health["c-0"] = HealthRecord(status=I, active_strain="s1")
        state.health["c-1"] = HealthRecord(status=S)
        assert colonist_is_infected(state, "c-0") is True
        assert colonist_is_infected(state, "c-1") is False
        assert colonist_is_infected(state, "missing") is False

    def test_colonist_strain_lookup(self) -> None:
        state = EpidemiologyState()
        state.health["c-0"] = HealthRecord(status=I, active_strain="s7")
        assert colonist_strain(state, "c-0") == "s7"
        assert colonist_strain(state, "missing") is None

    def test_health_summary_default(self) -> None:
        summary = colonist_health_summary(EpidemiologyState(), "missing")
        assert summary["status"] == S
        assert summary["active_strain"] is None
        assert summary["immunities"] == {}

    def test_cohesion_and_stress_hooks(self) -> None:
        r = EpidemiologyTickResult(year=1, cohesion_penalty=0.1, stress_bump=0.05)
        assert diplomacy_cohesion_penalty(r) == 0.1
        assert stress_bump(r) == 0.05


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_100_year_smoke(self) -> None:
        rng = random.Random(42)
        state = EpidemiologyState()
        pop = make_population(25, ages=[20 + (i * 3) % 40 for i in range(25)])
        for year in range(1, 101):
            ctx = make_context(
                pop, year=year, medicine=0.3 + (year % 5) * 0.1,
                pollution=0.2 + (year % 7) * 0.05,
                infra=["med_bay"] if year > 20 else [],
                trust=0.55,
                actions={c["id"]: ("rest" if i % 3 == 0 else "code")
                         for i, c in enumerate(pop)},
                quarantine=(year % 17 == 0))
            result = tick_epidemiology(state, ctx, rng)
            assert 0.0 <= result.prevalence <= 1.0
            assert all(r.status in (S, E, I, R) for r in state.health.values())
        assert len(state.to_dict()["history"]) <= 200

    def test_roundtrip_after_long_run(self) -> None:
        rng = random.Random(99)
        state = EpidemiologyState()
        pop = make_population(15)
        for year in range(1, 50):
            tick_epidemiology(state, make_context(
                pop, year=year, medicine=0.3, pollution=0.4, trust=0.6), rng)
        clone = EpidemiologyState.from_dict(state.to_dict())
        assert set(clone.diseases) == set(state.diseases)
        assert set(clone.health) == set(state.health)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
