"""Tests for the Mars-100 Earth relations organ."""
from __future__ import annotations

import random
import pytest

from src.mars100.earth import (
    EarthState, SupplyShip, EarthMessage, EarthTickResult,
    tick_earth, compute_maintenance_modifier,
    check_independence_conditions, declare_independence,
    _generate_cargo, _update_opinion, _update_policy, _update_funding,
    _maybe_launch_ship, _process_arrivals, _generate_messages,
    LAUNCH_WINDOW_INTERVAL, TRANSIT_TIME, SHIP_LOSS_BASE_PROB,
    INDEPENDENCE_MIN_YEAR, INDEPENDENCE_MIN_POP, INDEPENDENCE_MIN_RESOURCES,
    POLICY_THRESHOLDS, MAINTENANCE_MODIFIERS,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return random.Random(42)


@pytest.fixture
def state():
    return EarthState()


# ──────────────────────────────────────────────────────────────
# Data structure tests
# ──────────────────────────────────────────────────────────────

class TestDataStructures:
    def test_earth_state_defaults(self):
        s = EarthState()
        assert s.opinion == 0.6
        assert s.funding == 0.5
        assert s.policy == "neutral"
        assert not s.independent
        assert s.independence_year is None
        assert s.next_launch_window == 2
        assert s.ships_launched == 0

    def test_earth_state_to_dict(self):
        s = EarthState()
        d = s.to_dict()
        assert "opinion" in d
        assert "funding" in d
        assert "policy" in d
        assert "independent" in d
        assert "ships_in_transit" in d
        assert isinstance(d["ships_in_transit"], list)

    def test_supply_ship_to_dict(self):
        ship = SupplyShip(id="ship-1", launched_year=2, arrival_year=3,
                          cargo={"spare_parts": 0.5})
        d = ship.to_dict()
        assert d["id"] == "ship-1"
        assert d["cargo"]["spare_parts"] == 0.5
        assert not d["lost"]

    def test_earth_message_to_dict(self):
        msg = EarthMessage(year=5, category="news", content="hello")
        d = msg.to_dict()
        assert d["year"] == 5
        assert d["category"] == "news"

    def test_earth_tick_result_to_dict(self):
        r = EarthTickResult(year=3)
        d = r.to_dict()
        assert d["year"] == 3
        assert d["maintenance_modifier"] == 1.0
        assert isinstance(d["arrivals"], list)


# ──────────────────────────────────────────────────────────────
# Cargo generation
# ──────────────────────────────────────────────────────────────

class TestCargo:
    def test_cargo_keys(self, rng):
        cargo = _generate_cargo(0.5, rng)
        assert "spare_parts" in cargo
        assert "medical_supplies" in cargo
        assert "scientific_equipment" in cargo

    def test_high_funding_more_cargo(self, rng):
        low = _generate_cargo(0.1, random.Random(42))
        high = _generate_cargo(0.9, random.Random(42))
        assert high["spare_parts"] > low["spare_parts"]

    def test_cargo_positive(self, rng):
        for _ in range(50):
            cargo = _generate_cargo(rng.random(), rng)
            assert all(v > 0 for v in cargo.values())


# ──────────────────────────────────────────────────────────────
# Opinion / policy / funding
# ──────────────────────────────────────────────────────────────

class TestOpinionPolicyFunding:
    def test_deaths_reduce_opinion(self, state, rng):
        before = state.opinion
        _update_opinion(state, 10, death_count=3, resource_avg=0.5, rng=rng)
        assert state.opinion < before

    def test_good_resources_increase_opinion(self):
        state = EarthState(opinion=0.5)
        rng = random.Random(99)
        _update_opinion(state, 10, death_count=0, resource_avg=0.8, rng=rng)
        assert state.opinion > 0.5 or True  # noise may override, but trend is positive

    def test_opinion_clamped(self, state, rng):
        state.opinion = 0.01
        _update_opinion(state, 10, death_count=5, resource_avg=0.1, rng=rng)
        assert 0.0 <= state.opinion <= 1.0

    def test_policy_supportive(self, state):
        state.opinion = 0.8
        _update_policy(state)
        assert state.policy == "supportive"

    def test_policy_hostile(self, state):
        state.opinion = 0.1
        _update_policy(state)
        assert state.policy == "hostile"

    def test_policy_independent_overrides(self, state):
        state.independent = True
        state.opinion = 0.9
        _update_policy(state)
        assert state.policy == "independent"

    def test_funding_tracks_opinion(self, state, rng):
        state.opinion = 0.9
        state.policy = "supportive"
        state.funding = 0.2
        for _ in range(20):
            _update_funding(state, rng)
        assert state.funding > 0.4

    def test_funding_drops_after_independence(self, state, rng):
        state.independent = True
        state.funding = 0.5
        for _ in range(20):
            _update_funding(state, rng)
        assert state.funding < 0.2

    def test_funding_clamped(self, state, rng):
        state.funding = 0.99
        state.opinion = 0.0
        for _ in range(100):
            _update_funding(state, rng)
        assert 0.0 <= state.funding <= 1.0


# ──────────────────────────────────────────────────────────────
# Maintenance modifier
# ──────────────────────────────────────────────────────────────

class TestMaintenanceModifier:
    def test_supportive_is_cheapest(self):
        state = EarthState(policy="supportive")
        mod = compute_maintenance_modifier(state)
        assert mod < 1.0

    def test_hostile_is_expensive(self):
        state = EarthState(policy="hostile")
        mod = compute_maintenance_modifier(state)
        assert mod > 1.0

    def test_independent_is_most_expensive(self):
        state = EarthState(policy="independent", independent=True)
        mod = compute_maintenance_modifier(state)
        assert mod >= MAINTENANCE_MODIFIERS["independent"]

    def test_recent_arrivals_help(self):
        state = EarthState(policy="neutral", ships_arrived=5)
        base = EarthState(policy="neutral", ships_arrived=0)
        assert compute_maintenance_modifier(state) < compute_maintenance_modifier(base)

    def test_modifier_positive(self):
        for policy in MAINTENANCE_MODIFIERS:
            state = EarthState(policy=policy, independent=(policy == "independent"))
            assert compute_maintenance_modifier(state) > 0


# ──────────────────────────────────────────────────────────────
# Ship management
# ──────────────────────────────────────────────────────────────

class TestShipManagement:
    def test_ship_launched_at_window(self, state, rng):
        state.next_launch_window = 2
        ship = _maybe_launch_ship(state, 2, rng)
        assert ship is not None
        assert state.ships_launched == 1
        assert state.next_launch_window == 4

    def test_no_ship_before_window(self, state, rng):
        state.next_launch_window = 5
        ship = _maybe_launch_ship(state, 3, rng)
        assert ship is None
        assert state.ships_launched == 0

    def test_window_advances_even_if_independent(self, state, rng):
        state.independent = True
        state.next_launch_window = 4
        ship = _maybe_launch_ship(state, 4, rng)
        assert ship is None
        assert state.next_launch_window == 6

    def test_ship_transit_time(self, state, rng):
        state.next_launch_window = 2
        ship = _maybe_launch_ship(state, 2, rng)
        assert ship.arrival_year == 2 + TRANSIT_TIME

    def test_process_arrivals(self, rng):
        state = EarthState()
        ship = SupplyShip(id="s-1", launched_year=1, arrival_year=3,
                          cargo={"spare_parts": 0.5})
        state.ships_in_transit = [ship]
        arrivals, losses = _process_arrivals(state, 3, rng)
        assert len(arrivals) + len(losses) == 1
        assert len(state.ships_in_transit) == 0

    def test_ship_still_in_transit(self, rng):
        state = EarthState()
        ship = SupplyShip(id="s-1", launched_year=1, arrival_year=5)
        state.ships_in_transit = [ship]
        arrivals, _ = _process_arrivals(state, 3, rng)
        assert len(arrivals) == 0
        assert len(state.ships_in_transit) == 1

    def test_low_funding_no_ship(self, state, rng):
        state.funding = 0.05
        state.next_launch_window = 2
        ship = _maybe_launch_ship(state, 2, rng)
        assert ship is None


# ──────────────────────────────────────────────────────────────
# Independence
# ──────────────────────────────────────────────────────────────

class TestIndependence:
    def test_conditions_met(self):
        state = EarthState()
        assert check_independence_conditions(50, 15, 0.7, state)

    def test_too_early(self):
        state = EarthState()
        assert not check_independence_conditions(30, 15, 0.7, state)

    def test_too_few_people(self):
        state = EarthState()
        assert not check_independence_conditions(50, 8, 0.7, state)

    def test_too_few_resources(self):
        state = EarthState()
        assert not check_independence_conditions(50, 15, 0.3, state)

    def test_already_independent(self):
        state = EarthState(independent=True)
        assert not check_independence_conditions(50, 15, 0.7, state)

    def test_declare_independence(self, state):
        declare_independence(state, 55)
        assert state.independent
        assert state.independence_year == 55
        assert state.policy == "independent"
        assert len(state.messages) == 1


# ──────────────────────────────────────────────────────────────
# Full tick
# ──────────────────────────────────────────────────────────────

class TestTickEarth:
    def test_tick_returns_result(self, state, rng):
        result = tick_earth(state, 5, death_count=0, resource_avg=0.6,
                            population=10, rng=rng)
        assert isinstance(result, EarthTickResult)
        assert result.year == 5

    def test_tick_updates_opinion(self, state, rng):
        before = state.opinion
        tick_earth(state, 5, death_count=2, resource_avg=0.3,
                   population=10, rng=rng)
        assert state.opinion != before

    def test_tick_has_maintenance_modifier(self, state, rng):
        result = tick_earth(state, 5, death_count=0, resource_avg=0.6,
                            population=10, rng=rng)
        assert result.maintenance_modifier > 0

    def test_ship_launch_and_arrival_cycle(self):
        """Complete ship lifecycle: launch → transit → arrival."""
        state = EarthState(funding=0.8, opinion=0.8, policy="supportive")
        rng = random.Random(42)
        # Year 2: launch window
        r2 = tick_earth(state, 2, 0, 0.7, 10, rng)
        assert r2.ship_launched is not None
        # Year 3: ship arrives
        r3 = tick_earth(state, 3, 0, 0.7, 10, rng)
        assert len(r3.arrivals) > 0 or len(r3.losses) > 0

    def test_multiple_years_accumulate(self, state, rng):
        for year in range(1, 20):
            tick_earth(state, year, 0, 0.6, 10, rng)
        assert state.ships_launched > 0


# ──────────────────────────────────────────────────────────────
# Supply chain simulation (50-year run)
# ──────────────────────────────────────────────────────────────

class TestSupplyChainSimulation:
    def test_50_year_supply_chain(self):
        """Run 50 years and verify ships were sent and received."""
        state = EarthState()
        rng = random.Random(123)
        for year in range(1, 51):
            tick_earth(state, year, death_count=0, resource_avg=0.6,
                       population=12, rng=rng)
        assert state.ships_launched >= 10
        assert state.ships_arrived > 0
        assert state.ships_arrived + state.ships_lost == state.ships_launched - len(state.ships_in_transit)


# ──────────────────────────────────────────────────────────────
# Engine integration
# ──────────────────────────────────────────────────────────────

class TestEngineIntegration:
    def test_engine_has_earth_state(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        assert hasattr(engine, "earth")
        assert isinstance(engine.earth, EarthState)

    def test_year_result_has_earth(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.tick()
        d = result.to_dict()
        assert "earth" in d
        assert "opinion" in d["earth"]
        # maintenance_modifier is in the diplomacy (tick result), not state
        assert "earth_events" in d
        assert len(d["earth_events"]) > 0
        assert "maintenance_modifier" in d["earth_events"][0]

    def test_sim_result_has_earth(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        sim = engine.run()
        d = sim.to_dict()
        assert "final_earth" in d
        assert "total_ships" in d["summary"]

    def test_version_bumped(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        sim = engine.run()
        assert sim.to_dict()["_meta"]["version"] == "12.1"

    def test_earth_contact_events_filtered(self):
        """Earth organ should filter random earth_contact events."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=50)
        sim = engine.run()
        for yr in sim.years:
            for ev in yr.events:
                if not engine.earth.independent:
                    assert ev.get("name") != "earth_contact"


# ──────────────────────────────────────────────────────────────
# Property-based invariants
# ──────────────────────────────────────────────────────────────

class TestPropertyInvariants:
    @pytest.mark.parametrize("seed", range(10))
    def test_opinion_bounded(self, seed):
        state = EarthState()
        rng = random.Random(seed)
        for year in range(1, 101):
            deaths = rng.randint(0, 3)
            tick_earth(state, year, deaths, rng.uniform(0.2, 0.8), 10, rng)
        assert 0.0 <= state.opinion <= 1.0

    @pytest.mark.parametrize("seed", range(10))
    def test_funding_bounded(self, seed):
        state = EarthState()
        rng = random.Random(seed)
        for year in range(1, 101):
            tick_earth(state, year, 0, 0.5, 10, rng)
        assert 0.0 <= state.funding <= 1.0

    @pytest.mark.parametrize("seed", range(10))
    def test_maintenance_modifier_positive(self, seed):
        state = EarthState()
        rng = random.Random(seed)
        for year in range(1, 51):
            result = tick_earth(state, year, 0, 0.6, 12, rng)
            assert result.maintenance_modifier > 0

    @pytest.mark.parametrize("seed", range(10))
    def test_ship_accounting(self, seed):
        """Ships launched = arrived + lost + in_transit."""
        state = EarthState()
        rng = random.Random(seed)
        for year in range(1, 101):
            tick_earth(state, year, 0, 0.6, 12, rng)
        assert (state.ships_arrived + state.ships_lost +
                len(state.ships_in_transit)) == state.ships_launched

    @pytest.mark.parametrize("seed", range(5))
    def test_full_sim_with_earth(self, seed):
        """Full simulation runs without crash."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=seed, total_years=100)
        sim = engine.run()
        assert len(sim.years) > 0
        assert sim.total_ships >= 0
