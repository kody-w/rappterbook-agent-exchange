"""Tests for landing_pad.py — Mars Colony Landing Pad & Supply Reception.

Covers: SupplyPod clamping, LandingPadState defaults/invariants,
Tsiolkovsky fuel equation, approach go/no-go, landing, unloading,
pad clearing, resurfacing, beacon maintenance, thermal cooling,
multi-sol integration, and property-based physics bounds.

87 tests. The colony receives its first delivery.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.landing_pad import (
    SupplyPod, LandingPadState, PadSol,
    fuel_required_kg, approach_go_nogo, land_pod, unload_cargo,
    clear_pad, resurface_pad, maintain_beacons, tick_pad,
    create_landing_pad,
    MARS_GRAVITY_M_S2, EARTH_GRAVITY_M_S2,
    LANDING_DELTA_V_M_S, ENGINE_ISP_S, DEFAULT_POD_DRY_MASS_KG,
    MAX_PAYLOAD_KG, PAD_RADIUS_M,
    SURFACE_STRENGTH_NEW, SURFACE_DEGRADATION_PER_LANDING,
    SURFACE_FOD_THRESHOLD, SURFACE_MIN, RESURFACE_RESTORE,
    WIND_SAFE_LIMIT_M_S, DUST_TAU_SAFE_LIMIT, VISIBILITY_MIN_M,
    PAD_AMBIENT_TEMP_K, PAD_TEMP_SPIKE_PER_LANDING_K, PAD_COOLING_RATE,
    BEACON_HEALTH_NEW, BEACON_MIN_HEALTH,
    BEACON_DEGRADATION_PER_SOL, BEACON_DEGRADATION_PER_LANDING,
    BEACON_MAINTENANCE_RESTORE,
    UNLOAD_RATE_KG_PER_SOL,
)


# ═══════════════════════════════════════════════════════════════════
#  SupplyPod dataclass
# ═══════════════════════════════════════════════════════════════════

class TestSupplyPod:
    def test_defaults(self):
        pod = SupplyPod()
        assert pod.payload_kg == 0.0
        assert pod.landed is False
        assert pod.unloaded_kg == 0.0
        assert pod.manifest == []

    def test_payload_clamped_high(self):
        pod = SupplyPod(payload_kg=99999.0)
        assert pod.payload_kg == MAX_PAYLOAD_KG

    def test_payload_clamped_low(self):
        pod = SupplyPod(payload_kg=-500.0)
        assert pod.payload_kg == 0.0

    def test_unloaded_clamped_to_payload(self):
        pod = SupplyPod(payload_kg=1000.0, unloaded_kg=5000.0)
        assert pod.unloaded_kg == 1000.0

    def test_unloaded_clamped_low(self):
        pod = SupplyPod(payload_kg=1000.0, unloaded_kg=-100.0)
        assert pod.unloaded_kg == 0.0

    def test_remaining_kg(self):
        pod = SupplyPod(payload_kg=3000.0, unloaded_kg=1000.0)
        assert abs(pod.remaining_kg - 2000.0) < 0.01

    def test_remaining_never_negative(self):
        pod = SupplyPod(payload_kg=100.0, unloaded_kg=100.0)
        assert pod.remaining_kg >= 0.0

    def test_fully_unloaded_true(self):
        pod = SupplyPod(payload_kg=1000.0, unloaded_kg=1000.0)
        assert pod.fully_unloaded is True

    def test_fully_unloaded_false(self):
        pod = SupplyPod(payload_kg=1000.0, unloaded_kg=500.0)
        assert pod.fully_unloaded is False

    def test_empty_pod_is_fully_unloaded(self):
        pod = SupplyPod(payload_kg=0.0)
        assert pod.fully_unloaded is True

    def test_manifest_independent(self):
        p1 = SupplyPod(manifest=["food"])
        p2 = SupplyPod(manifest=["tools"])
        assert p1.manifest != p2.manifest


# ═══════════════════════════════════════════════════════════════════
#  LandingPadState dataclass
# ═══════════════════════════════════════════════════════════════════

class TestLandingPadState:
    def test_defaults(self):
        s = create_landing_pad()
        assert s.sol == 0
        assert s.surface_strength == SURFACE_STRENGTH_NEW
        assert s.beacon_health == BEACON_HEALTH_NEW
        assert s.active_pod is None
        assert s.pad_temp_k == PAD_AMBIENT_TEMP_K
        assert s.total_landings == 0

    def test_surface_clamped_high(self):
        s = LandingPadState(surface_strength=5.0)
        assert s.surface_strength == 1.0

    def test_surface_clamped_low(self):
        s = LandingPadState(surface_strength=-1.0)
        assert s.surface_strength == SURFACE_MIN

    def test_beacon_clamped_high(self):
        s = LandingPadState(beacon_health=2.0)
        assert s.beacon_health == 1.0

    def test_beacon_clamped_low(self):
        s = LandingPadState(beacon_health=-1.0)
        assert s.beacon_health == BEACON_MIN_HEALTH

    def test_temp_clamped_to_ambient(self):
        s = LandingPadState(pad_temp_k=50.0)
        assert s.pad_temp_k == PAD_AMBIENT_TEMP_K

    def test_negative_landings_clamped(self):
        s = LandingPadState(total_landings=-5)
        assert s.total_landings == 0

    def test_negative_fuel_clamped(self):
        s = LandingPadState(total_fuel_spent_kg=-100.0)
        assert s.total_fuel_spent_kg == 0.0

    def test_negative_cargo_clamped(self):
        s = LandingPadState(total_cargo_received_kg=-200.0)
        assert s.total_cargo_received_kg == 0.0

    def test_pad_clear_no_pod(self):
        s = LandingPadState()
        assert s.pad_clear is True

    def test_pad_not_clear_with_pod(self):
        s = LandingPadState(active_pod=SupplyPod(payload_kg=1000.0))
        assert s.pad_clear is False

    def test_fod_risk_below_threshold(self):
        s = LandingPadState(surface_strength=SURFACE_FOD_THRESHOLD - 0.01)
        assert s.fod_risk is True

    def test_fod_risk_above_threshold(self):
        s = LandingPadState(surface_strength=SURFACE_FOD_THRESHOLD + 0.01)
        assert s.fod_risk is False

    def test_landing_safe_all_nominal(self):
        s = LandingPadState(wind_speed_m_s=5.0, dust_tau=0.3,
                            visibility_m=5000.0, beacon_health=0.9)
        assert s.landing_safe is True

    def test_landing_unsafe_high_wind(self):
        s = LandingPadState(wind_speed_m_s=30.0, dust_tau=0.3,
                            visibility_m=5000.0, beacon_health=0.9)
        assert s.landing_safe is False

    def test_landing_unsafe_low_beacon(self):
        s = LandingPadState(wind_speed_m_s=5.0, dust_tau=0.3,
                            visibility_m=5000.0, beacon_health=BEACON_MIN_HEALTH)
        assert s.landing_safe is False


# ═══════════════════════════════════════════════════════════════════
#  Tsiolkovsky fuel equation
# ═══════════════════════════════════════════════════════════════════

class TestFuelRequired:
    def test_zero_payload(self):
        fuel = fuel_required_kg(0.0)
        assert fuel > 0.0  # dry mass still needs fuel

    def test_max_payload(self):
        fuel = fuel_required_kg(MAX_PAYLOAD_KG)
        assert fuel > 0.0

    def test_more_payload_more_fuel(self):
        f_light = fuel_required_kg(1000.0)
        f_heavy = fuel_required_kg(5000.0)
        assert f_heavy > f_light

    def test_negative_payload(self):
        assert fuel_required_kg(-100.0) == 0.0

    def test_zero_delta_v(self):
        assert fuel_required_kg(1000.0, delta_v_m_s=0.0) == 0.0

    def test_negative_isp(self):
        assert fuel_required_kg(1000.0, isp_s=-1.0) == 0.0

    def test_tsiolkovsky_manual_check(self):
        """Verify against hand-calculated Tsiolkovsky result."""
        payload = 3000.0
        m_f = DEFAULT_POD_DRY_MASS_KG + payload  # 5000 kg
        v_e = ENGINE_ISP_S * EARTH_GRAVITY_M_S2  # 3433.5 m/s
        mass_ratio = math.exp(LANDING_DELTA_V_M_S / v_e)
        expected_fuel = m_f * (mass_ratio - 1.0)
        actual_fuel = fuel_required_kg(payload)
        assert abs(actual_fuel - expected_fuel) < 0.01

    def test_fuel_always_nonnegative(self):
        for p in [0.0, 100.0, 1000.0, 5000.0]:
            assert fuel_required_kg(p) >= 0.0

    def test_higher_delta_v_more_fuel(self):
        f_low = fuel_required_kg(2000.0, delta_v_m_s=600.0)
        f_high = fuel_required_kg(2000.0, delta_v_m_s=2400.0)
        assert f_high > f_low

    def test_higher_isp_less_fuel(self):
        f_low_isp = fuel_required_kg(2000.0, isp_s=200.0)
        f_high_isp = fuel_required_kg(2000.0, isp_s=400.0)
        assert f_low_isp > f_high_isp


# ═══════════════════════════════════════════════════════════════════
#  Approach Go/No-Go
# ═══════════════════════════════════════════════════════════════════

class TestApproachGoNogo:
    def test_all_nominal_is_go(self):
        result = approach_go_nogo(5.0, 0.3, 5000.0, True, 0.9)
        assert result["go"] is True
        assert all(result["criteria"].values())

    def test_high_wind_is_nogo(self):
        result = approach_go_nogo(30.0, 0.3, 5000.0, True, 0.9)
        assert result["go"] is False
        assert result["criteria"]["wind"] is False

    def test_high_dust_is_nogo(self):
        result = approach_go_nogo(5.0, 4.0, 5000.0, True, 0.9)
        assert result["go"] is False
        assert result["criteria"]["dust"] is False

    def test_low_visibility_is_nogo(self):
        result = approach_go_nogo(5.0, 0.3, 100.0, True, 0.9)
        assert result["go"] is False
        assert result["criteria"]["visibility"] is False

    def test_pad_occupied_is_nogo(self):
        result = approach_go_nogo(5.0, 0.3, 5000.0, False, 0.9)
        assert result["go"] is False
        assert result["criteria"]["pad_clear"] is False

    def test_dead_beacons_is_nogo(self):
        result = approach_go_nogo(5.0, 0.3, 5000.0, True, BEACON_MIN_HEALTH)
        assert result["go"] is False
        assert result["criteria"]["beacons"] is False

    def test_boundary_wind_exact_limit(self):
        result = approach_go_nogo(WIND_SAFE_LIMIT_M_S, 0.3, 5000.0, True, 0.9)
        assert result["criteria"]["wind"] is True

    def test_boundary_dust_exact_limit(self):
        result = approach_go_nogo(5.0, DUST_TAU_SAFE_LIMIT, 5000.0, True, 0.9)
        assert result["criteria"]["dust"] is True

    def test_boundary_visibility_exact_limit(self):
        result = approach_go_nogo(5.0, 0.3, VISIBILITY_MIN_M, True, 0.9)
        assert result["criteria"]["visibility"] is True

    def test_five_criteria_present(self):
        result = approach_go_nogo(5.0, 0.3, 5000.0, True, 0.9)
        expected_keys = {"wind", "dust", "visibility", "pad_clear", "beacons"}
        assert set(result["criteria"].keys()) == expected_keys


# ═══════════════════════════════════════════════════════════════════
#  Landing
# ═══════════════════════════════════════════════════════════════════

class TestLandPod:
    def test_successful_landing(self):
        state = create_landing_pad()
        pod = SupplyPod(payload_kg=3000.0, manifest=["food", "tools"])
        result = land_pod(state, pod)
        assert result["success"] is True
        assert result["fuel_spent_kg"] > 0
        assert state.active_pod is pod
        assert pod.landed is True
        assert state.total_landings == 1

    def test_landing_degrades_surface(self):
        state = create_landing_pad()
        initial_surface = state.surface_strength
        pod = SupplyPod(payload_kg=1000.0)
        land_pod(state, pod)
        assert state.surface_strength < initial_surface
        expected = initial_surface - SURFACE_DEGRADATION_PER_LANDING
        assert abs(state.surface_strength - expected) < 0.001

    def test_landing_heats_pad(self):
        state = create_landing_pad()
        initial_temp = state.pad_temp_k
        pod = SupplyPod(payload_kg=1000.0)
        land_pod(state, pod)
        assert state.pad_temp_k == initial_temp + PAD_TEMP_SPIKE_PER_LANDING_K

    def test_landing_degrades_beacons(self):
        state = create_landing_pad()
        initial_beacon = state.beacon_health
        pod = SupplyPod(payload_kg=1000.0)
        land_pod(state, pod)
        expected = initial_beacon - BEACON_DEGRADATION_PER_LANDING
        assert abs(state.beacon_health - expected) < 0.001

    def test_landing_tracks_fuel(self):
        state = create_landing_pad()
        pod = SupplyPod(payload_kg=2000.0)
        result = land_pod(state, pod)
        assert state.total_fuel_spent_kg == result["fuel_spent_kg"]

    def test_nogo_high_wind(self):
        state = create_landing_pad()
        state.wind_speed_m_s = 30.0
        pod = SupplyPod(payload_kg=1000.0)
        result = land_pod(state, pod)
        assert result["success"] is False
        assert result["reason"] == "no_go"
        assert state.active_pod is None

    def test_nogo_pad_occupied(self):
        state = create_landing_pad()
        pod1 = SupplyPod(payload_kg=1000.0)
        land_pod(state, pod1)
        pod2 = SupplyPod(payload_kg=2000.0)
        result = land_pod(state, pod2)
        assert result["success"] is False

    def test_surface_never_below_min(self):
        state = create_landing_pad()
        state.surface_strength = SURFACE_MIN + 0.01
        pod = SupplyPod(payload_kg=1000.0)
        land_pod(state, pod)
        assert state.surface_strength >= SURFACE_MIN

    def test_beacon_never_below_min(self):
        state = create_landing_pad()
        state.beacon_health = BEACON_MIN_HEALTH + 0.03
        pod = SupplyPod(payload_kg=1000.0)
        land_pod(state, pod)
        assert state.beacon_health >= BEACON_MIN_HEALTH


# ═══════════════════════════════════════════════════════════════════
#  Cargo unloading
# ═══════════════════════════════════════════════════════════════════

class TestUnloadCargo:
    def test_no_pod_on_pad(self):
        state = create_landing_pad()
        result = unload_cargo(state)
        assert result["success"] is False
        assert result["reason"] == "no_pod_on_pad"

    def test_default_unload_rate(self):
        state = create_landing_pad()
        pod = SupplyPod(payload_kg=5000.0)
        pod.landed = True
        state.active_pod = pod
        result = unload_cargo(state)
        assert result["success"] is True
        assert abs(result["unloaded_kg"] - UNLOAD_RATE_KG_PER_SOL) < 0.01

    def test_partial_unload(self):
        state = create_landing_pad()
        pod = SupplyPod(payload_kg=3000.0)
        pod.landed = True
        state.active_pod = pod
        result = unload_cargo(state, kg_to_unload=500.0)
        assert abs(result["unloaded_kg"] - 500.0) < 0.01
        assert abs(result["remaining_kg"] - 2500.0) < 0.01

    def test_unload_capped_at_remaining(self):
        state = create_landing_pad()
        pod = SupplyPod(payload_kg=100.0)
        pod.landed = True
        state.active_pod = pod
        result = unload_cargo(state, kg_to_unload=500.0)
        assert abs(result["unloaded_kg"] - 100.0) < 0.01
        assert result["fully_unloaded"] is True

    def test_already_unloaded(self):
        state = create_landing_pad()
        pod = SupplyPod(payload_kg=100.0, unloaded_kg=100.0)
        pod.landed = True
        state.active_pod = pod
        result = unload_cargo(state)
        assert result["fully_unloaded"] is True
        assert result["unloaded_kg"] == 0.0

    def test_cargo_accounting(self):
        state = create_landing_pad()
        pod = SupplyPod(payload_kg=4000.0)
        pod.landed = True
        state.active_pod = pod
        unload_cargo(state, kg_to_unload=1500.0)
        unload_cargo(state, kg_to_unload=1500.0)
        assert abs(state.total_cargo_received_kg - 3000.0) < 0.01

    def test_negative_unload_clamped(self):
        state = create_landing_pad()
        pod = SupplyPod(payload_kg=1000.0)
        pod.landed = True
        state.active_pod = pod
        result = unload_cargo(state, kg_to_unload=-100.0)
        assert result["unloaded_kg"] == 0.0


# ═══════════════════════════════════════════════════════════════════
#  Pad clearing
# ═══════════════════════════════════════════════════════════════════

class TestClearPad:
    def test_clear_with_pod(self):
        state = create_landing_pad()
        pod = SupplyPod(payload_kg=2000.0, unloaded_kg=2000.0)
        state.active_pod = pod
        result = clear_pad(state)
        assert result["success"] is True
        assert result["cargo_remaining_kg"] == 0.0
        assert state.pad_clear is True

    def test_clear_with_remaining_cargo(self):
        state = create_landing_pad()
        pod = SupplyPod(payload_kg=2000.0, unloaded_kg=500.0)
        state.active_pod = pod
        result = clear_pad(state)
        assert result["success"] is True
        assert abs(result["cargo_remaining_kg"] - 1500.0) < 0.01

    def test_clear_empty_pad(self):
        state = create_landing_pad()
        result = clear_pad(state)
        assert result["success"] is False


# ═══════════════════════════════════════════════════════════════════
#  Resurfacing
# ═══════════════════════════════════════════════════════════════════

class TestResurfacePad:
    def test_resurface_improves_strength(self):
        state = create_landing_pad()
        state.surface_strength = 0.5
        result = resurface_pad(state)
        assert result["success"] is True
        assert state.surface_strength > 0.5

    def test_resurface_occupied_pad_fails(self):
        state = create_landing_pad()
        state.surface_strength = 0.5
        state.active_pod = SupplyPod(payload_kg=100.0)
        result = resurface_pad(state)
        assert result["success"] is False
        assert state.surface_strength == 0.5

    def test_resurface_never_exceeds_one(self):
        state = create_landing_pad()
        state.surface_strength = 0.99
        resurface_pad(state)
        assert state.surface_strength <= 1.0

    def test_resurface_fraction_correct(self):
        state = create_landing_pad()
        state.surface_strength = 0.4
        gap = 1.0 - 0.4
        expected = 0.4 + gap * RESURFACE_RESTORE
        resurface_pad(state)
        assert abs(state.surface_strength - expected) < 0.001


# ═══════════════════════════════════════════════════════════════════
#  Beacon maintenance
# ═══════════════════════════════════════════════════════════════════

class TestMaintainBeacons:
    def test_maintenance_improves_health(self):
        state = create_landing_pad()
        state.beacon_health = 0.5
        result = maintain_beacons(state)
        assert result["success"] is True
        assert state.beacon_health > 0.5

    def test_maintenance_never_exceeds_one(self):
        state = create_landing_pad()
        state.beacon_health = 0.99
        maintain_beacons(state)
        assert state.beacon_health <= 1.0

    def test_maintenance_fraction_correct(self):
        state = create_landing_pad()
        state.beacon_health = 0.3
        gap = 1.0 - 0.3
        expected = 0.3 + gap * BEACON_MAINTENANCE_RESTORE
        maintain_beacons(state)
        assert abs(state.beacon_health - expected) < 0.001


# ═══════════════════════════════════════════════════════════════════
#  Main tick
# ═══════════════════════════════════════════════════════════════════

class TestTickPad:
    def test_idle_sol(self):
        state = create_landing_pad()
        sol = PadSol(sol=1)
        result = tick_pad(state, sol)
        assert result["sol"] == 1
        assert result["landing"] is None
        assert result["unload"] is None
        assert state.sol == 1

    def test_landing_sol(self):
        state = create_landing_pad()
        pod = SupplyPod(payload_kg=3000.0)
        sol = PadSol(sol=1, incoming_pod=pod)
        result = tick_pad(state, sol)
        assert result["landing"]["success"] is True
        assert state.total_landings == 1

    def test_thermal_cooling(self):
        state = create_landing_pad()
        state.pad_temp_k = PAD_AMBIENT_TEMP_K + 100.0
        sol = PadSol(sol=1)
        tick_pad(state, sol)
        # Should cool by PAD_COOLING_RATE fraction of excess
        expected = PAD_AMBIENT_TEMP_K + 100.0 * (1.0 - PAD_COOLING_RATE)
        assert abs(state.pad_temp_k - expected) < 0.1

    def test_thermal_never_below_ambient(self):
        state = create_landing_pad()
        state.pad_temp_k = PAD_AMBIENT_TEMP_K
        sol = PadSol(sol=1)
        tick_pad(state, sol)
        assert state.pad_temp_k >= PAD_AMBIENT_TEMP_K

    def test_beacon_daily_degradation(self):
        state = create_landing_pad()
        initial = state.beacon_health
        sol = PadSol(sol=1)
        tick_pad(state, sol)
        assert state.beacon_health < initial
        expected = initial - BEACON_DEGRADATION_PER_SOL
        assert abs(state.beacon_health - expected) < 0.0001

    def test_weather_updates(self):
        state = create_landing_pad()
        sol = PadSol(sol=1, wind_m_s=20.0, dust_tau=2.5, visibility_m=800.0)
        tick_pad(state, sol)
        assert state.wind_speed_m_s == 20.0
        assert state.dust_tau == 2.5
        assert state.visibility_m == 800.0

    def test_full_lifecycle_land_unload_clear(self):
        """Complete lifecycle: land → unload → clear."""
        state = create_landing_pad()
        # Sol 1: land
        pod = SupplyPod(payload_kg=1500.0)
        sol1 = PadSol(sol=1, incoming_pod=pod)
        r1 = tick_pad(state, sol1)
        assert r1["landing"]["success"] is True
        assert state.pad_clear is False

        # Sol 2: unload (1500 < 2000 rate, so all in one sol)
        sol2 = PadSol(sol=2, unload=True)
        r2 = tick_pad(state, sol2)
        assert r2["unload"]["fully_unloaded"] is True

        # Sol 3: clear
        sol3 = PadSol(sol=3, clear_pod=True)
        r3 = tick_pad(state, sol3)
        assert r3["clear"]["success"] is True
        assert state.pad_clear is True
        assert abs(state.total_cargo_received_kg - 1500.0) < 0.01


# ═══════════════════════════════════════════════════════════════════
#  Multi-sol integration & property-based invariants
# ═══════════════════════════════════════════════════════════════════

class TestMultiSolInvariants:
    def test_10_sol_smoke(self):
        """Smoke test: 10 sols without crash."""
        state = create_landing_pad()
        for i in range(10):
            sol = PadSol(sol=i, wind_m_s=5.0, dust_tau=0.3, visibility_m=5000.0)
            tick_pad(state, sol)
        assert state.sol == 9

    def test_surface_monotonically_decreases_under_landings(self):
        """Surface degrades with each landing, never increases without resurfacing."""
        state = create_landing_pad()
        prev = state.surface_strength
        for i in range(8):
            pod = SupplyPod(payload_kg=1000.0)
            sol = PadSol(sol=i, incoming_pod=pod, clear_pod=True,
                         wind_m_s=5.0, dust_tau=0.3, visibility_m=5000.0)
            tick_pad(state, sol)
            assert state.surface_strength <= prev
            prev = state.surface_strength

    def test_beacon_never_below_min(self):
        """Beacon health never drops below minimum even after many sols."""
        state = create_landing_pad()
        for i in range(1000):
            sol = PadSol(sol=i)
            tick_pad(state, sol)
        assert state.beacon_health >= BEACON_MIN_HEALTH

    def test_temperature_bounded(self):
        """Pad temperature always >= ambient, converges back after landing."""
        state = create_landing_pad()
        pod = SupplyPod(payload_kg=5000.0)
        sol = PadSol(sol=0, incoming_pod=pod)
        tick_pad(state, sol)
        assert state.pad_temp_k > PAD_AMBIENT_TEMP_K
        # Cool for 50 sols
        for i in range(1, 51):
            sol = PadSol(sol=i)
            tick_pad(state, sol)
        # Should be very close to ambient
        assert state.pad_temp_k < PAD_AMBIENT_TEMP_K + 1.0
        assert state.pad_temp_k >= PAD_AMBIENT_TEMP_K

    def test_total_cargo_conservation(self):
        """Total cargo received matches sum of all unloads."""
        state = create_landing_pad()
        expected_total = 0.0
        for i in range(5):
            pod = SupplyPod(payload_kg=2000.0)
            sol_land = PadSol(sol=i * 3, incoming_pod=pod,
                              wind_m_s=5.0, dust_tau=0.3, visibility_m=5000.0)
            tick_pad(state, sol_land)
            sol_unload = PadSol(sol=i * 3 + 1, unload=True)
            r = tick_pad(state, sol_unload)
            if r["unload"] and r["unload"]["success"]:
                expected_total += r["unload"]["unloaded_kg"]
            sol_clear = PadSol(sol=i * 3 + 2, clear_pod=True)
            tick_pad(state, sol_clear)
        assert abs(state.total_cargo_received_kg - expected_total) < 0.01

    def test_landing_count_monotonic(self):
        """Total landings only ever increases."""
        state = create_landing_pad()
        prev = 0
        for i in range(5):
            pod = SupplyPod(payload_kg=1000.0)
            sol = PadSol(sol=i * 2, incoming_pod=pod, clear_pod=True,
                         wind_m_s=5.0, dust_tau=0.3, visibility_m=5000.0)
            tick_pad(state, sol)
            assert state.total_landings >= prev
            prev = state.total_landings

    def test_fuel_accounting_positive(self):
        """Fuel spent is always non-negative and monotonically increasing."""
        state = create_landing_pad()
        for i in range(5):
            pod = SupplyPod(payload_kg=3000.0)
            sol = PadSol(sol=i * 2, incoming_pod=pod, clear_pod=True,
                         wind_m_s=5.0, dust_tau=0.3, visibility_m=5000.0)
            tick_pad(state, sol)
        assert state.total_fuel_spent_kg > 0.0

    def test_resurface_restores_after_damage(self):
        """Resurfacing after many landings restores pad above FOD threshold."""
        state = create_landing_pad()
        # 12 landings to degrade surface
        for i in range(12):
            pod = SupplyPod(payload_kg=1000.0)
            sol = PadSol(sol=i * 2, incoming_pod=pod, clear_pod=True,
                         wind_m_s=5.0, dust_tau=0.3, visibility_m=5000.0)
            tick_pad(state, sol)
        assert state.fod_risk is True
        # Resurface
        sol_fix = PadSol(sol=99, resurface=True)
        tick_pad(state, sol_fix)
        assert state.surface_strength > SURFACE_FOD_THRESHOLD

    def test_beacon_maintenance_restores_after_degradation(self):
        """Beacon maintenance after many sols restores health."""
        state = create_landing_pad()
        for i in range(200):
            sol = PadSol(sol=i)
            tick_pad(state, sol)
        degraded = state.beacon_health
        sol_fix = PadSol(sol=200, maintain_beacons=True)
        tick_pad(state, sol_fix)
        assert state.beacon_health > degraded
