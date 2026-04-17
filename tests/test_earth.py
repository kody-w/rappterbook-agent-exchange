"""Tests for src/mars100/earth.py — Earth parallel evolution organ."""
from __future__ import annotations

import random

import pytest

from src.mars100.earth import (
    COLLAPSE_THRESHOLD,
    SIGNAL_TYPES,
    EarthSignal,
    EarthState,
    _clamp,
    apply_signal_effects,
    generate_signal,
    tick_earth,
)


# ── helpers ──────────────────────────────────────────────────────────

def make_rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


# ── EarthState ───────────────────────────────────────────────────────

class TestEarthState:

    def test_defaults_in_range(self) -> None:
        s = EarthState()
        assert 0 <= s.tech_level <= 1
        assert 0 <= s.political_stability <= 1
        assert 0 <= s.mars_interest <= 1
        assert 0 <= s.climate_crisis <= 1
        assert s.contact_active is True
        assert s.signals == []

    def test_to_dict_keys(self) -> None:
        d = EarthState().to_dict()
        for key in ("tech_level", "political_stability", "mars_interest",
                     "climate_crisis", "contact_active", "total_signals"):
            assert key in d

    def test_snapshot_compact(self) -> None:
        snap = EarthState().snapshot()
        assert set(snap.keys()) == {"tech", "stability", "interest",
                                     "climate", "contact"}

    def test_signals_not_shared(self) -> None:
        """Ensure default_factory prevents shared mutable default."""
        a, b = EarthState(), EarthState()
        a.signals.append({"test": 1})
        assert b.signals == []


# ── tick_earth ───────────────────────────────────────────────────────

class TestTickEarth:

    def test_tech_grows(self) -> None:
        s = EarthState(tech_level=0.5)
        rng = make_rng()
        for _ in range(20):
            tick_earth(s, 1, colony_pop=10, colony_resource_avg=0.5, rng=rng)
        assert s.tech_level > 0.5

    def test_climate_accumulates(self) -> None:
        s = EarthState(climate_crisis=0.15)
        rng = make_rng()
        for _ in range(30):
            tick_earth(s, 1, colony_pop=10, colony_resource_avg=0.5, rng=rng)
        assert s.climate_crisis > 0.15

    def test_values_clamped(self) -> None:
        s = EarthState(tech_level=0.99, climate_crisis=0.99,
                       political_stability=0.01, mars_interest=0.01)
        rng = make_rng()
        for _ in range(50):
            tick_earth(s, 1, colony_pop=10, colony_resource_avg=0.5, rng=rng)
        assert 0.0 <= s.tech_level <= 1.0
        assert 0.0 <= s.political_stability <= 1.0
        assert 0.0 <= s.mars_interest <= 1.0
        assert 0.0 <= s.climate_crisis <= 1.0

    def test_collapse_severs_contact(self) -> None:
        s = EarthState(climate_crisis=COLLAPSE_THRESHOLD - 0.01)
        rng = random.Random(999)
        s.climate_crisis = COLLAPSE_THRESHOLD
        tick_earth(s, 50, colony_pop=10, colony_resource_avg=0.5, rng=rng)
        assert s.contact_active is False

    def test_no_tick_after_collapse(self) -> None:
        s = EarthState(contact_active=False, tech_level=0.5)
        rng = make_rng()
        tick_earth(s, 50, colony_pop=10, colony_resource_avg=0.5, rng=rng)
        assert s.tech_level == 0.5  # unchanged

    def test_colony_success_boosts_interest(self) -> None:
        rng1, rng2 = make_rng(1), make_rng(1)
        struggling = EarthState()
        thriving = EarthState()
        for _ in range(15):
            tick_earth(struggling, 1, colony_pop=5,
                       colony_resource_avg=0.2, rng=rng1)
            tick_earth(thriving, 1, colony_pop=30,
                       colony_resource_avg=0.7, rng=rng2)
        assert thriving.mars_interest > struggling.mars_interest

    def test_deterministic(self) -> None:
        s1, s2 = EarthState(), EarthState()
        for s, rng in [(s1, make_rng(7)), (s2, make_rng(7))]:
            for yr in range(1, 51):
                tick_earth(s, yr, colony_pop=10,
                           colony_resource_avg=0.5, rng=rng)
        assert s1.to_dict() == s2.to_dict()


# ── generate_signal ──────────────────────────────────────────────────

class TestGenerateSignal:

    def test_returns_none_sometimes(self) -> None:
        s = EarthState(mars_interest=0.1, political_stability=0.1)
        nones = sum(
            1 for seed in range(200)
            if generate_signal(s, 10, 10, 0.5, make_rng(seed)) is None
        )
        assert nones > 0, "should sometimes return None"

    def test_returns_signal_sometimes(self) -> None:
        s = EarthState(mars_interest=0.9, political_stability=0.9)
        signals = sum(
            1 for seed in range(200)
            if generate_signal(s, 10, 10, 0.5, make_rng(seed)) is not None
        )
        assert signals > 0, "should sometimes return a signal"

    def test_signal_types_valid(self) -> None:
        s = EarthState(mars_interest=0.9, political_stability=0.9)
        for seed in range(300):
            sig = generate_signal(s, 10, 10, 0.5, make_rng(seed))
            if sig is not None:
                assert sig.signal_type in SIGNAL_TYPES

    def test_no_signal_after_collapse(self) -> None:
        s = EarthState(contact_active=False)
        for seed in range(100):
            assert generate_signal(s, 10, 10, 0.5, make_rng(seed)) is None

    def test_signal_to_dict(self) -> None:
        sig = EarthSignal(year=10, signal_type="supply_drop",
                          magnitude=0.5, description="Test")
        d = sig.to_dict()
        assert d["year"] == 10
        assert d["signal_type"] == "supply_drop"
        assert isinstance(d["magnitude"], float)

    def test_signal_appended_to_state(self) -> None:
        s = EarthState(mars_interest=0.95, political_stability=0.95)
        before = len(s.signals)
        found = False
        for seed in range(200):
            sig = generate_signal(s, 10, 10, 0.5, make_rng(seed))
            if sig is not None:
                found = True
                break
        if found:
            assert len(s.signals) == before + 1


# ── apply_signal_effects ─────────────────────────────────────────────

class TestApplySignalEffects:

    def test_none_returns_empty(self) -> None:
        assert apply_signal_effects(None) == {}

    def test_supply_drop_positive(self) -> None:
        sig = EarthSignal(1, "supply_drop", 0.8, "Supplies!")
        effects = apply_signal_effects(sig)
        assert effects.get("food", 0) > 0
        assert effects.get("medicine", 0) > 0

    def test_demand_tribute_negative(self) -> None:
        sig = EarthSignal(1, "demand_tribute", 0.6, "Pay up.")
        effects = apply_signal_effects(sig)
        assert effects.get("food", 0) < 0

    def test_embargo_negative(self) -> None:
        sig = EarthSignal(1, "embargo", 0.7, "Embargo!")
        effects = apply_signal_effects(sig)
        for v in effects.values():
            assert v <= 0

    def test_tech_transfer_positive(self) -> None:
        sig = EarthSignal(1, "tech_transfer", 0.5, "Blueprints")
        effects = apply_signal_effects(sig)
        assert effects.get("power", 0) > 0

    def test_immigration_no_resource_effect(self) -> None:
        sig = EarthSignal(1, "immigration_offer", 0.5, "New person")
        assert apply_signal_effects(sig) == {}

    def test_radio_silence_no_resource_effect(self) -> None:
        sig = EarthSignal(1, "radio_silence", 0.5, "Static")
        assert apply_signal_effects(sig) == {}

    def test_magnitude_scales_effects(self) -> None:
        low = EarthSignal(1, "supply_drop", 0.1, "Small")
        high = EarthSignal(1, "supply_drop", 0.9, "Big")
        e_low = apply_signal_effects(low)
        e_high = apply_signal_effects(high)
        assert e_high["food"] > e_low["food"]


# ── clamp helper ─────────────────────────────────────────────────────

class TestClamp:

    def test_in_range(self) -> None:
        assert _clamp(0.5) == 0.5

    def test_floor(self) -> None:
        assert _clamp(-0.1) == 0.0

    def test_ceiling(self) -> None:
        assert _clamp(1.5) == 1.0


# ── 100-year smoke test ─────────────────────────────────────────────

class TestSmoke:

    def test_100_year_no_crash(self) -> None:
        """Earth evolves for 100 years without raising."""
        s = EarthState()
        rng = make_rng(42)
        signals_received = 0
        for yr in range(1, 101):
            tick_earth(s, yr, colony_pop=10 + yr // 5,
                       colony_resource_avg=0.5, rng=rng)
            sig = generate_signal(s, yr, 10 + yr // 5, 0.5, rng)
            if sig is not None:
                effects = apply_signal_effects(sig)
                signals_received += 1
                for v in effects.values():
                    assert isinstance(v, (int, float))
        assert signals_received > 0, "should have received at least one signal"

    def test_collapse_happens_eventually(self) -> None:
        """Climate crisis should trigger collapse within 200 years."""
        s = EarthState()
        rng = make_rng(42)
        for yr in range(1, 201):
            if not s.contact_active:
                break
            tick_earth(s, yr, colony_pop=10, colony_resource_avg=0.5, rng=rng)
        assert not s.contact_active, "Earth should collapse from climate crisis"

    def test_contact_once_lost_stays_lost(self) -> None:
        """Contact is a one-way transition."""
        s = EarthState(contact_active=False, climate_crisis=1.0)
        rng = make_rng()
        for yr in range(1, 50):
            tick_earth(s, yr, colony_pop=10, colony_resource_avg=0.5, rng=rng)
        assert not s.contact_active


# ── property-based invariants ────────────────────────────────────────

class TestInvariants:

    @pytest.mark.parametrize("seed", range(10))
    def test_earth_state_bounded(self, seed: int) -> None:
        """All earth state values stay in [0, 1] over 100 years."""
        s = EarthState()
        rng = make_rng(seed)
        for yr in range(1, 101):
            tick_earth(s, yr, colony_pop=10, colony_resource_avg=0.5, rng=rng)
            assert 0 <= s.tech_level <= 1
            assert 0 <= s.political_stability <= 1
            assert 0 <= s.mars_interest <= 1
            assert 0 <= s.climate_crisis <= 1

    @pytest.mark.parametrize("seed", range(10))
    def test_signal_magnitude_bounded(self, seed: int) -> None:
        s = EarthState(mars_interest=0.9, political_stability=0.9)
        rng = make_rng(seed)
        for yr in range(1, 101):
            sig = generate_signal(s, yr, 10, 0.5, rng)
            if sig is not None:
                assert 0 < sig.magnitude < 1
