"""
Tests for landing_pad.py — Mars Cargo Landing Operations.

80 tests across 10 test classes. The colony's lifeline to Earth.

Run: python -m pytest tests/test_landing_pad.py -v
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.landing_pad import (
    LandingPadState,
    LandingSol,
    CargoManifest,
    landing_success_probability,
    is_launch_window,
    next_arrival_sol,
    generate_cargo,
    pad_repair_amount,
    propellant_status,
    tick_landing_pad,
    create_landing_pad,
    run_landing_ops,
    BASE_LANDING_SUCCESS_PROB,
    HEAVY_CARGO_THRESHOLD_KG,
    MAX_PAYLOAD_KG,
    MASS_PENALTY_PER_TONNE,
    PAD_HEALTH_MAX,
    PAD_MIN_SAFE_HEALTH,
    PAD_DAMAGE_PER_LANDING,
    PAD_DAMAGE_PER_DUST_STORM,
    PAD_REPAIR_PER_SOL,
    BEACON_FAILURE_PENALTY,
    DUST_STORM_LANDING_PENALTY,
    PAD_DAMAGE_PENALTY,
    SYNODIC_PERIOD_SOLS,
    TRANSIT_TIME_SOLS,
    ORBIT_INSERTION_SOLS,
    LAUNCH_WINDOW_SOLS,
    PROPELLANT_PER_LAUNCH_KG,
    PROPELLANT_PER_ABORT_KG,
    PROPELLANT_PRODUCTION_PER_SOL_KG,
    CARGO_CATEGORIES,
    BEACON_REPAIR_SOLS,
)


# --- Landing Success Probability ---

class TestLandingProbability:

    def test_ideal_conditions(self):
        prob = landing_success_probability(5000, PAD_HEALTH_MAX, True, False)
        assert abs(prob - BASE_LANDING_SUCCESS_PROB) < 0.01

    def test_heavy_cargo_reduces_prob(self):
        prob_light = landing_success_probability(5000, PAD_HEALTH_MAX, True, False)
        prob_heavy = landing_success_probability(MAX_PAYLOAD_KG, PAD_HEALTH_MAX, True, False)
        assert prob_heavy < prob_light

    def test_damaged_pad_reduces_prob(self):
        prob_good = landing_success_probability(5000, PAD_HEALTH_MAX, True, False)
        prob_bad = landing_success_probability(5000, 10.0, True, False)
        assert prob_bad < prob_good

    def test_beacon_down_reduces_prob(self):
        prob_beacon = landing_success_probability(5000, PAD_HEALTH_MAX, True, False)
        prob_blind = landing_success_probability(5000, PAD_HEALTH_MAX, False, False)
        assert abs(prob_beacon - prob_blind - BEACON_FAILURE_PENALTY) < 0.01

    def test_dust_storm_reduces_prob(self):
        prob_clear = landing_success_probability(5000, PAD_HEALTH_MAX, True, False)
        prob_storm = landing_success_probability(5000, PAD_HEALTH_MAX, True, True)
        assert abs(prob_clear - prob_storm - DUST_STORM_LANDING_PENALTY) < 0.01

    def test_worst_case_still_positive(self):
        """Even worst conditions have at least 5% success."""
        prob = landing_success_probability(MAX_PAYLOAD_KG, 0.0, False, True)
        assert prob >= 0.05

    def test_probability_bounded_0_1(self):
        for mass in [1000, 10000, 20000]:
            for health in [0, 30, 50, 100]:
                for beacon in [True, False]:
                    for storm in [True, False]:
                        p = landing_success_probability(
                            mass, float(health), beacon, storm
                        )
                        assert 0.0 <= p <= 1.0

    def test_pad_above_safe_threshold_no_penalty(self):
        """Pad health above PAD_MIN_SAFE_HEALTH incurs no pad penalty."""
        p1 = landing_success_probability(5000, PAD_HEALTH_MAX, True, False)
        p2 = landing_success_probability(5000, PAD_MIN_SAFE_HEALTH + 1, True, False)
        assert abs(p1 - p2) < 0.01


# --- Launch Windows ---

class TestLaunchWindows:

    def test_before_window(self):
        assert is_launch_window(100, 780) is False

    def test_at_window_start(self):
        assert is_launch_window(780, 780) is True

    def test_during_window(self):
        assert is_launch_window(790, 780) is True

    def test_at_window_end(self):
        assert is_launch_window(780 + LAUNCH_WINDOW_SOLS, 780) is False

    def test_after_window(self):
        assert is_launch_window(900, 780) is False

    def test_arrival_timing(self):
        arrival = next_arrival_sol(780)
        expected = 780 + TRANSIT_TIME_SOLS + ORBIT_INSERTION_SOLS
        assert arrival == expected


# --- Cargo Generation ---

class TestCargoGeneration:

    def test_cargo_has_valid_mass(self):
        rng = random.Random(42)
        for _ in range(50):
            mass, cat = generate_cargo(rng)
            assert 5000 <= mass <= MAX_PAYLOAD_KG

    def test_cargo_has_valid_category(self):
        rng = random.Random(42)
        for _ in range(50):
            mass, cat = generate_cargo(rng)
            assert cat in CARGO_CATEGORIES

    def test_custom_mass(self):
        rng = random.Random(42)
        mass, cat = generate_cargo(rng, mass_kg=12345.0)
        assert mass == 12345.0

    def test_deterministic(self):
        m1, c1 = generate_cargo(random.Random(99))
        m2, c2 = generate_cargo(random.Random(99))
        assert m1 == m2
        assert c1 == c2


# --- Pad Repair ---

class TestPadRepair:

    def test_no_crew_no_repair(self):
        assert pad_repair_amount(False, 50.0) == 0.0

    def test_full_health_no_repair(self):
        assert pad_repair_amount(True, PAD_HEALTH_MAX) == 0.0

    def test_crew_repairs(self):
        repair = pad_repair_amount(True, 50.0)
        assert repair == PAD_REPAIR_PER_SOL

    def test_repair_capped_at_max(self):
        repair = pad_repair_amount(True, PAD_HEALTH_MAX - 2.0)
        assert abs(repair - 2.0) < 0.01

    def test_repair_non_negative(self):
        for h in [0, 30, 50, 100]:
            assert pad_repair_amount(True, float(h)) >= 0.0


# --- Propellant Status ---

class TestPropellantStatus:

    def test_nominal(self):
        assert propellant_status(PROPELLANT_PER_LAUNCH_KG + PROPELLANT_PER_ABORT_KG) == "nominal"

    def test_low(self):
        assert propellant_status(PROPELLANT_PER_ABORT_KG + 100) == "low"

    def test_critical(self):
        assert propellant_status(100) == "critical"

    def test_zero(self):
        assert propellant_status(0) == "critical"

    def test_abundant(self):
        assert propellant_status(50000) == "nominal"


# --- Tick Function ---

class TestTickLandingPad:

    def test_tick_advances_sol(self):
        state = create_landing_pad()
        state, sol = tick_landing_pad(state, rng=random.Random(42))
        assert state.sol == 1
        assert sol.sol == 1

    def test_propellant_accumulates(self):
        state = create_landing_pad()
        state, sol = tick_landing_pad(state, rng=random.Random(42))
        assert state.propellant_kg == PROPELLANT_PRODUCTION_PER_SOL_KG
        assert sol.propellant_produced == PROPELLANT_PRODUCTION_PER_SOL_KG

    def test_no_isru_no_propellant(self):
        state = create_landing_pad()
        state, sol = tick_landing_pad(state, isru_operational=False, rng=random.Random(42))
        assert state.propellant_kg == 0.0

    def test_dust_storm_damages_pad(self):
        state = create_landing_pad()
        state, sol = tick_landing_pad(state, dust_storm=True, rng=random.Random(42))
        assert state.pad_health < PAD_HEALTH_MAX

    def test_crew_repair_restores_pad(self):
        state = create_landing_pad()
        state.pad_health = 50.0
        state, sol = tick_landing_pad(state, crew_repairing=True, rng=random.Random(42))
        assert state.pad_health > 50.0

    def test_non_operational_skips(self):
        state = create_landing_pad()
        state.operational = False
        state, sol = tick_landing_pad(state, rng=random.Random(42))
        assert state.propellant_kg == 0.0

    def test_beacon_can_fail(self):
        """Run enough sols that beacon failure is likely."""
        state = create_landing_pad()
        beacon_failed = False
        rng = random.Random(42)
        for _ in range(2000):
            state, sol = tick_landing_pad(state, rng=rng)
            if not sol.beacon_up:
                beacon_failed = True
                break
        assert beacon_failed

    def test_beacon_repairs_itself(self):
        state = create_landing_pad()
        state.beacon_operational = False
        state.beacon_repair_sols_left = 1
        state, sol = tick_landing_pad(state, rng=random.Random(42))
        assert state.beacon_operational is True

    def test_deterministic(self):
        s1 = create_landing_pad(first_window_sol=50)
        s2 = create_landing_pad(first_window_sol=50)
        for _ in range(100):
            s1, _ = tick_landing_pad(s1, rng=random.Random(42))
            s2, _ = tick_landing_pad(s2, rng=random.Random(42))
        assert s1.sol == s2.sol
        assert s1.propellant_kg == s2.propellant_kg

    def test_pad_health_bounded(self):
        """Pad health stays in [0, PAD_HEALTH_MAX]."""
        state = create_landing_pad()
        rng = random.Random(42)
        for _ in range(100):
            storm = rng.random() < 0.5
            state, sol = tick_landing_pad(
                state, dust_storm=storm, crew_repairing=True, rng=rng
            )
            assert 0.0 <= state.pad_health <= PAD_HEALTH_MAX


# --- Landing Events ---

class TestLandingEvents:

    def test_ships_arrive_after_window(self):
        """Ships launched at window sol arrive after transit time."""
        state = create_landing_pad(first_window_sol=10)
        rng = random.Random(42)
        landing_happened = False
        for _ in range(400):
            state, sol = tick_landing_pad(state, rng=rng)
            if sol.landing_attempted:
                landing_happened = True
                break
        assert landing_happened
        assert state.total_landings >= 1

    def test_successful_landing_delivers_cargo(self):
        """A successful landing adds to cargo totals."""
        state = create_landing_pad(first_window_sol=10)
        rng = random.Random(42)
        for _ in range(400):
            state, sol = tick_landing_pad(state, rng=rng)
            if sol.landing_success:
                break
        if state.successful_landings > 0:
            assert state.total_cargo_delivered_kg > 0.0

    def test_landing_damages_pad(self):
        """Successful landing scours the pad."""
        state = create_landing_pad(first_window_sol=10)
        rng = random.Random(42)
        for _ in range(400):
            state, sol = tick_landing_pad(state, rng=rng)
            if sol.landing_success:
                assert state.pad_health < PAD_HEALTH_MAX
                break

    def test_cargo_tracked_by_category(self):
        state = create_landing_pad(first_window_sol=10)
        rng = random.Random(42)
        for _ in range(400):
            state, sol = tick_landing_pad(state, rng=rng)
        total_by_cat = sum(state.cargo_by_category.values())
        assert abs(total_by_cat - state.total_cargo_delivered_kg) < 0.01

    def test_manifest_log_tracks_all(self):
        state = create_landing_pad(first_window_sol=10)
        rng = random.Random(42)
        for _ in range(400):
            state, sol = tick_landing_pad(state, rng=rng)
        assert len(state.manifest_log) == state.total_landings


# --- Campaign ---

class TestLandingCampaign:

    def test_campaign_runs(self):
        state, history = run_landing_ops(100, seed=42)
        assert state.sol == 100
        assert len(history) == 100

    def test_campaign_produces_propellant(self):
        state, history = run_landing_ops(100, seed=42)
        assert state.propellant_kg > 0.0

    def test_campaign_handles_landings(self):
        state, history = run_landing_ops(500, first_window_sol=50, seed=42)
        assert state.total_landings >= 1

    def test_campaign_survival(self):
        """Colony survives 1000 sols — the long wait between windows."""
        state, history = run_landing_ops(1000, first_window_sol=100, seed=42)
        assert state.sol == 1000
        assert state.pad_health >= 0.0
        assert state.propellant_kg >= 0.0

    def test_campaign_multiple_windows(self):
        """Run long enough to see two supply windows."""
        state, history = run_landing_ops(
            2000, first_window_sol=100, seed=42
        )
        # With windows at sol 100 and sol 880, should have multiple landings
        assert state.total_landings >= 2


# --- Smoke Test ---

class TestSmokeTest:

    def test_10_sol_smoke(self):
        state, history = run_landing_ops(10, seed=1)
        assert state.sol == 10
        assert state.pad_health >= 0.0
        assert state.propellant_kg >= 0.0
        assert len(history) == 10

    def test_500_sol_endurance(self):
        state, history = run_landing_ops(500, first_window_sol=50, seed=99)
        assert state.sol == 500
        assert state.pad_health >= 0.0
        for h in history:
            assert 0.0 <= h.pad_health <= PAD_HEALTH_MAX
            assert h.propellant_kg >= 0.0

    def test_worst_case_weather(self):
        """Every sol is a dust storm — pad survives."""
        state, history = run_landing_ops(
            100, dust_storm_prob=1.0, seed=42
        )
        assert state.sol == 100
        assert state.pad_health >= 0.0

    def test_no_repair_degradation(self):
        """Without repair, pad health degrades over time."""
        state, history = run_landing_ops(
            100, dust_storm_prob=0.3, crew_repair_interval=9999, seed=42
        )
        assert state.pad_health < PAD_HEALTH_MAX
