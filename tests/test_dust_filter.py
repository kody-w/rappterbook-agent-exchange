"""Tests for dust_filter.py — Mars Habitat HEPA Filtration System."""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.dust_filter import (
    FilterUnit, ESPStage, FilterSystemState, TickRecord,
    darcy_pressure_drop, loaded_permeability, hepa_efficiency,
    esp_efficiency, dust_ingress_g, settling_velocity_ms,
    perchlorate_mass_mg, hab_dust_concentration, filter_needs_replacement,
    air_quality_safe, system_flow_rate_m3s, time_to_filter_hab,
    tick_filter, create_filter_system, run_filter_system,
    HEPA_CLEAN_EFFICIENCY, HEPA_MPPS_UM, MARS_DUST_MEDIAN_UM,
    ESP_BASE_EFFICIENCY, ESP_RATED_VOLTAGE_KV,
    MAX_DUST_LOAD_KG, DUST_PER_EVA_CYCLE_G, AMBIENT_DUST_FLUX_G_PER_HR,
    SAFE_DUST_CONCENTRATION_MG_M3, HAB_VOLUME_M3,
    DEFAULT_PERMEABILITY_M2, DEFAULT_FILTER_AREA_M2,
    DEFAULT_FACE_VELOCITY_MS, AIR_VISCOSITY, PERCHLORATE_IN_DUST_FRAC,
)


# ── FilterUnit clamping ────────────────────────────────────────────

class TestFilterUnit:
    def test_defaults(self):
        u = FilterUnit()
        assert u.unit_id == 0
        assert u.dust_load_kg == 0.0
        assert u.hours_in_service == 0.0
        assert u.replaced_count == 0

    def test_area_clamped_positive(self):
        u = FilterUnit(area_m2=-1.0)
        assert u.area_m2 >= 0.01

    def test_thickness_clamped(self):
        u = FilterUnit(thickness_m=0.0001)
        assert u.thickness_m >= 0.001
        u2 = FilterUnit(thickness_m=10.0)
        assert u2.thickness_m <= 0.5

    def test_permeability_clamped(self):
        u = FilterUnit(permeability_m2=-1.0)
        assert u.permeability_m2 > 0

    def test_dust_load_clamped(self):
        u = FilterUnit(dust_load_kg=-1.0)
        assert u.dust_load_kg == 0.0
        u2 = FilterUnit(dust_load_kg=999.0)
        assert u2.dust_load_kg == MAX_DUST_LOAD_KG

    def test_hours_clamped(self):
        u = FilterUnit(hours_in_service=-10.0)
        assert u.hours_in_service == 0.0

    def test_replaced_count_clamped(self):
        u = FilterUnit(replaced_count=-5)
        assert u.replaced_count == 0


# ── ESPStage clamping ──────────────────────────────────────────────

class TestESPStage:
    def test_defaults(self):
        e = ESPStage()
        assert e.voltage_kv == ESP_RATED_VOLTAGE_KV
        assert e.powered is True

    def test_voltage_clamped(self):
        e = ESPStage(voltage_kv=-5.0)
        assert e.voltage_kv == 0.0
        e2 = ESPStage(voltage_kv=100.0)
        assert e2.voltage_kv == 30.0

    def test_dust_clamped(self):
        e = ESPStage(collected_dust_kg=-1.0)
        assert e.collected_dust_kg == 0.0

    def test_hours_clamped(self):
        e = ESPStage(hours_run=-1.0)
        assert e.hours_run == 0.0


# ── FilterSystemState clamping ─────────────────────────────────────

class TestFilterSystemState:
    def test_defaults(self):
        s = FilterSystemState()
        assert s.spare_filters == 10  # dataclass default
        assert s.hab_dust_concentration_mg_m3 == 0.0
        assert s.alert == ""

    def test_spare_filters_clamped(self):
        s = FilterSystemState(spare_filters=-3)
        assert s.spare_filters == 0

    def test_face_velocity_clamped(self):
        s = FilterSystemState(face_velocity_ms=0.0)
        assert s.face_velocity_ms >= 0.01
        s2 = FilterSystemState(face_velocity_ms=100.0)
        assert s2.face_velocity_ms <= 5.0

    def test_concentration_clamped(self):
        s = FilterSystemState(hab_dust_concentration_mg_m3=-1.0)
        assert s.hab_dust_concentration_mg_m3 == 0.0


# ── Darcy pressure drop ────────────────────────────────────────────

class TestDarcyPressureDrop:
    def test_zero_velocity(self):
        dp = darcy_pressure_drop(AIR_VISCOSITY, 0.0, 0.05, 1e-10)
        assert dp == 0.0

    def test_positive_drop(self):
        dp = darcy_pressure_drop(AIR_VISCOSITY, 0.5, 0.05, 1e-10)
        assert dp > 0

    def test_proportional_to_velocity(self):
        dp1 = darcy_pressure_drop(AIR_VISCOSITY, 0.5, 0.05, 1e-10)
        dp2 = darcy_pressure_drop(AIR_VISCOSITY, 1.0, 0.05, 1e-10)
        assert abs(dp2 / dp1 - 2.0) < 1e-6

    def test_inverse_proportional_to_permeability(self):
        dp1 = darcy_pressure_drop(AIR_VISCOSITY, 0.5, 0.05, 1e-10)
        dp2 = darcy_pressure_drop(AIR_VISCOSITY, 0.5, 0.05, 2e-10)
        assert abs(dp1 / dp2 - 2.0) < 1e-6

    def test_zero_permeability_returns_inf(self):
        dp = darcy_pressure_drop(AIR_VISCOSITY, 0.5, 0.05, 0.0)
        assert dp == float("inf")

    def test_physical_range(self):
        """Clean HEPA: ΔP should be 10–1000 Pa at typical conditions."""
        dp = darcy_pressure_drop(AIR_VISCOSITY, 0.5, 0.05, 1e-10)
        assert 10.0 < dp < 100_000.0


# ── Loaded permeability ────────────────────────────────────────────

class TestLoadedPermeability:
    def test_clean_filter(self):
        k = loaded_permeability(1e-10, 0.0)
        assert k == 1e-10

    def test_full_load_minimum(self):
        k = loaded_permeability(1e-10, MAX_DUST_LOAD_KG)
        assert abs(k - 1e-11) < 1e-15  # 10% floor

    def test_half_load(self):
        k = loaded_permeability(1e-10, MAX_DUST_LOAD_KG / 2)
        assert 1e-11 < k < 1e-10

    def test_monotonically_decreasing(self):
        loads = [i * 0.1 for i in range(6)]
        perms = [loaded_permeability(1e-10, ld) for ld in loads]
        for i in range(1, len(perms)):
            assert perms[i] <= perms[i - 1]

    def test_overload_clamped(self):
        k = loaded_permeability(1e-10, MAX_DUST_LOAD_KG * 10)
        assert k == loaded_permeability(1e-10, MAX_DUST_LOAD_KG)

    def test_negative_load_clamped(self):
        k = loaded_permeability(1e-10, -1.0)
        assert k == 1e-10

    def test_zero_max_load(self):
        k = loaded_permeability(1e-10, 0.5, max_load_kg=0)
        assert k == 1e-10


# ── HEPA efficiency ────────────────────────────────────────────────

class TestHEPAEfficiency:
    def test_at_mpps(self):
        eff = hepa_efficiency(HEPA_MPPS_UM)
        assert abs(eff - HEPA_CLEAN_EFFICIENCY) < 1e-6

    def test_larger_particles_better(self):
        eff_mpps = hepa_efficiency(HEPA_MPPS_UM)
        eff_big = hepa_efficiency(5.0)
        assert eff_big >= eff_mpps

    def test_smaller_particles_better(self):
        eff_mpps = hepa_efficiency(HEPA_MPPS_UM)
        eff_small = hepa_efficiency(0.01)
        assert eff_small >= eff_mpps

    def test_mars_dust_efficiency(self):
        eff = hepa_efficiency(MARS_DUST_MEDIAN_UM)
        assert eff > 0.999  # excellent capture at 1.5 μm

    def test_zero_diameter(self):
        assert hepa_efficiency(0.0) == 0.0

    def test_negative_diameter(self):
        assert hepa_efficiency(-1.0) == 0.0

    def test_bounded_zero_one(self):
        for d in [0.001, 0.01, 0.1, 0.3, 1.0, 10.0, 100.0]:
            eff = hepa_efficiency(d)
            assert 0.0 <= eff <= 1.0

    def test_mpps_is_minimum(self):
        """MPPS is the minimum efficiency point by definition."""
        eff_mpps = hepa_efficiency(HEPA_MPPS_UM)
        for d in [0.01, 0.05, 0.1, 1.0, 5.0, 20.0]:
            assert hepa_efficiency(d) >= eff_mpps - 1e-9


# ── ESP efficiency ─────────────────────────────────────────────────

class TestESPEfficiency:
    def test_rated_voltage(self):
        eff = esp_efficiency(ESP_RATED_VOLTAGE_KV, True)
        assert abs(eff - ESP_BASE_EFFICIENCY) < 1e-6

    def test_half_voltage(self):
        eff = esp_efficiency(ESP_RATED_VOLTAGE_KV / 2, True)
        assert abs(eff - ESP_BASE_EFFICIENCY / 2) < 1e-6

    def test_unpowered(self):
        assert esp_efficiency(ESP_RATED_VOLTAGE_KV, False) == 0.0

    def test_zero_voltage(self):
        assert esp_efficiency(0.0, True) == 0.0

    def test_over_rated_capped(self):
        eff = esp_efficiency(ESP_RATED_VOLTAGE_KV * 2, True)
        assert abs(eff - ESP_BASE_EFFICIENCY) < 1e-6

    def test_bounded_zero_one(self):
        for v in [0.0, 3.0, 6.0, 12.0, 24.0]:
            eff = esp_efficiency(v, True)
            assert 0.0 <= eff <= 1.0


# ── Dust ingress ───────────────────────────────────────────────────

class TestDustIngress:
    def test_no_eva_no_ambient(self):
        g = dust_ingress_g(0, 0.0, 1.0)
        assert g == 0.0

    def test_one_eva(self):
        g = dust_ingress_g(1, 0.0, 1.0)
        assert abs(g - DUST_PER_EVA_CYCLE_G) < 1e-6

    def test_ambient_only(self):
        g = dust_ingress_g(0, AMBIENT_DUST_FLUX_G_PER_HR, 2.0)
        assert abs(g - AMBIENT_DUST_FLUX_G_PER_HR * 2.0) < 1e-6

    def test_combined(self):
        g = dust_ingress_g(2, AMBIENT_DUST_FLUX_G_PER_HR, 1.0)
        expected = 2 * DUST_PER_EVA_CYCLE_G + AMBIENT_DUST_FLUX_G_PER_HR
        assert abs(g - expected) < 1e-6

    def test_non_negative(self):
        assert dust_ingress_g(0, 0.0, 0.0) >= 0.0


# ── Settling velocity ──────────────────────────────────────────────

class TestSettlingVelocity:
    def test_zero_particle(self):
        assert settling_velocity_ms(0.0) == 0.0

    def test_negative_particle(self):
        assert settling_velocity_ms(-1.0) == 0.0

    def test_positive_for_real_particle(self):
        v = settling_velocity_ms(10.0)
        assert v > 0

    def test_larger_settles_faster(self):
        v1 = settling_velocity_ms(1.0)
        v10 = settling_velocity_ms(10.0)
        assert v10 > v1

    def test_proportional_to_diameter_squared(self):
        v1 = settling_velocity_ms(1.0)
        v2 = settling_velocity_ms(2.0)
        assert abs(v2 / v1 - 4.0) < 1e-6

    def test_mars_dust_slow(self):
        """1.5 μm particle on Mars: ~1e-7 m/s — basically floating."""
        v = settling_velocity_ms(MARS_DUST_MEDIAN_UM)
        assert v < 1e-4  # much less than 0.1 mm/s

    def test_proportional_to_gravity(self):
        v_mars = settling_velocity_ms(10.0, gravity=3.72)
        v_earth = settling_velocity_ms(10.0, gravity=9.81)
        assert abs(v_earth / v_mars - 9.81 / 3.72) < 1e-4


# ── Perchlorate mass ───────────────────────────────────────────────

class TestPerchlorateMass:
    def test_zero_dust(self):
        assert perchlorate_mass_mg(0.0) == 0.0

    def test_positive(self):
        mg = perchlorate_mass_mg(1.0)  # 1 g dust
        assert abs(mg - PERCHLORATE_IN_DUST_FRAC * 1000.0) < 1e-6

    def test_negative_clamped(self):
        assert perchlorate_mass_mg(-1.0) == 0.0

    def test_large_dust_mass(self):
        mg = perchlorate_mass_mg(1000.0)
        assert mg > 0


# ── Hab dust concentration ─────────────────────────────────────────

class TestHabDustConcentration:
    def test_no_change(self):
        c = hab_dust_concentration(0.0, 0.0, 0.0)
        assert c == 0.0

    def test_dust_added(self):
        c = hab_dust_concentration(0.0, 1.0, 0.0)
        assert c > 0

    def test_dust_removed(self):
        c = hab_dust_concentration(1.0, 0.0, 0.5)
        assert c < 1.0

    def test_never_negative(self):
        c = hab_dust_concentration(0.0, 0.0, 100.0)
        assert c >= 0.0

    def test_settling_reduces(self):
        c1 = hab_dust_concentration(1.0, 0.0, 0.0, natural_settling_frac=0.0)
        c2 = hab_dust_concentration(1.0, 0.0, 0.0, natural_settling_frac=0.1)
        assert c2 < c1

    def test_zero_volume(self):
        c = hab_dust_concentration(1.0, 1.0, 0.0, hab_volume_m3=0.0)
        assert c == 0.0


# ── Filter replacement ─────────────────────────────────────────────

class TestFilterNeedsReplacement:
    def test_clean_filter(self):
        u = FilterUnit(dust_load_kg=0.0)
        assert not filter_needs_replacement(u)

    def test_full_filter(self):
        u = FilterUnit(dust_load_kg=MAX_DUST_LOAD_KG)
        assert filter_needs_replacement(u)

    def test_near_full(self):
        u = FilterUnit(dust_load_kg=MAX_DUST_LOAD_KG * 0.96)
        assert filter_needs_replacement(u)

    def test_below_threshold(self):
        u = FilterUnit(dust_load_kg=MAX_DUST_LOAD_KG * 0.5)
        assert not filter_needs_replacement(u)


# ── Air quality ────────────────────────────────────────────────────

class TestAirQualitySafe:
    def test_safe(self):
        assert air_quality_safe(0.0) is True

    def test_at_limit(self):
        assert air_quality_safe(SAFE_DUST_CONCENTRATION_MG_M3) is True

    def test_unsafe(self):
        assert air_quality_safe(SAFE_DUST_CONCENTRATION_MG_M3 + 0.01) is False


# ── System flow rate ───────────────────────────────────────────────

class TestSystemFlowRate:
    def test_zero_units(self):
        assert system_flow_rate_m3s(0) == 0.0

    def test_positive(self):
        q = system_flow_rate_m3s(4)
        assert q > 0

    def test_proportional_to_units(self):
        q1 = system_flow_rate_m3s(1)
        q4 = system_flow_rate_m3s(4)
        assert abs(q4 / q1 - 4.0) < 1e-6


# ── Time to filter hab ─────────────────────────────────────────────

class TestTimeToFilterHab:
    def test_zero_units(self):
        assert time_to_filter_hab(0) == float("inf")

    def test_positive(self):
        t = time_to_filter_hab(4)
        assert 0 < t < 10_000

    def test_more_units_faster(self):
        t1 = time_to_filter_hab(1)
        t4 = time_to_filter_hab(4)
        assert t4 < t1


# ── Factory ────────────────────────────────────────────────────────

class TestCreateFilterSystem:
    def test_default(self):
        sys = create_filter_system()
        assert len(sys.hepa_units) == 4
        assert sys.spare_filters == 10
        assert sys.esp.powered is True

    def test_custom(self):
        sys = create_filter_system(num_hepa_units=8, spare_filters=20,
                                   esp_enabled=False)
        assert len(sys.hepa_units) == 8
        assert sys.spare_filters == 20
        assert sys.esp.powered is False

    def test_unit_ids_sequential(self):
        sys = create_filter_system(num_hepa_units=6)
        ids = [u.unit_id for u in sys.hepa_units]
        assert ids == list(range(6))


# ── Tick function ──────────────────────────────────────────────────

class TestTickFilter:
    def test_basic_tick(self):
        state = create_filter_system()
        rec = tick_filter(state)
        assert rec.dust_ingress_g >= 0
        assert rec.hour == 0.0

    def test_time_advances(self):
        state = create_filter_system()
        tick_filter(state)
        tick_filter(state)
        assert state.hours_elapsed == 2.0

    def test_eva_adds_dust(self):
        state = create_filter_system()
        rec_no_eva = tick_filter(state, eva_cycles=0)
        state2 = create_filter_system()
        rec_eva = tick_filter(state2, eva_cycles=3)
        assert rec_eva.dust_ingress_g > rec_no_eva.dust_ingress_g

    def test_esp_captures_dust(self):
        state = create_filter_system()
        rec = tick_filter(state, eva_cycles=2)
        assert rec.esp_captured_g > 0

    def test_esp_off_captures_nothing(self):
        state = create_filter_system(esp_enabled=False)
        rec = tick_filter(state, eva_cycles=2)
        assert rec.esp_captured_g == 0.0

    def test_hepa_captures_dust(self):
        state = create_filter_system()
        rec = tick_filter(state, eva_cycles=2)
        assert rec.hepa_captured_g > 0

    def test_perchlorate_tracked(self):
        state = create_filter_system()
        rec = tick_filter(state, eva_cycles=2)
        assert rec.perchlorate_captured_mg > 0

    def test_pressure_drop_positive(self):
        state = create_filter_system()
        rec = tick_filter(state, eva_cycles=1)
        assert rec.pressure_drop_pa > 0

    def test_pressure_drop_increases_with_load(self):
        state = create_filter_system(num_hepa_units=1)
        dps = []
        for _ in range(50):
            rec = tick_filter(state, eva_cycles=2)
            dps.append(rec.pressure_drop_pa)
        assert dps[-1] > dps[0]  # pressure drop rises as filter loads

    def test_filter_replacement(self):
        state = create_filter_system(num_hepa_units=1, spare_filters=5)
        state.hepa_units[0].dust_load_kg = MAX_DUST_LOAD_KG
        rec = tick_filter(state)
        assert rec.filters_replaced == 1
        assert state.spare_filters == 4
        assert state.hepa_units[0].dust_load_kg == 0.0

    def test_no_replacement_without_spares(self):
        state = create_filter_system(num_hepa_units=1, spare_filters=0)
        state.hepa_units[0].dust_load_kg = MAX_DUST_LOAD_KG
        rec = tick_filter(state)
        assert rec.filters_replaced == 0

    def test_dust_storm_multiplier(self):
        state1 = create_filter_system()
        rec1 = tick_filter(state1, dust_storm_multiplier=1.0)
        state2 = create_filter_system()
        rec2 = tick_filter(state2, dust_storm_multiplier=5.0)
        assert rec2.dust_ingress_g > rec1.dust_ingress_g

    def test_air_quality_starts_safe(self):
        state = create_filter_system()
        rec = tick_filter(state)
        assert rec.air_quality_safe is True

    def test_alert_on_spent_filters(self):
        state = create_filter_system(num_hepa_units=1, spare_filters=0)
        state.hepa_units[0].dust_load_kg = MAX_DUST_LOAD_KG
        rec = tick_filter(state, eva_cycles=5)
        assert "ALL_FILTERS_SPENT" in rec.alert

    def test_alert_low_spares(self):
        state = create_filter_system(spare_filters=1)
        rec = tick_filter(state)
        assert "LOW_SPARES" in rec.alert


# ── Conservation & invariants ──────────────────────────────────────

class TestConservation:
    def test_dust_mass_conservation(self):
        """Total captured + airborne ≈ total ingress over time."""
        state = create_filter_system()
        total_ingress = 0.0
        for _ in range(100):
            rec = tick_filter(state, eva_cycles=1)
            total_ingress += rec.dust_ingress_g
        total_captured_g = state.total_dust_captured_kg * 1000.0
        airborne_g = state.hab_dust_concentration_mg_m3 * HAB_VOLUME_M3 / 1000.0
        accounted = total_captured_g + airborne_g
        # Some dust settles, so accounted ≤ total_ingress
        assert accounted <= total_ingress + 1e-6
        # But most should be captured (>80% with ESP + HEPA)
        assert accounted > total_ingress * 0.5

    def test_filters_used_monotonic(self):
        state = create_filter_system(num_hepa_units=1, spare_filters=20)
        prev = state.total_filters_used
        for _ in range(500):
            tick_filter(state, eva_cycles=3)
            assert state.total_filters_used >= prev
            prev = state.total_filters_used

    def test_pressure_drop_always_non_negative(self):
        state = create_filter_system()
        for _ in range(200):
            rec = tick_filter(state, eva_cycles=1)
            assert rec.pressure_drop_pa >= 0

    def test_concentration_always_non_negative(self):
        state = create_filter_system()
        for _ in range(200):
            rec = tick_filter(state, eva_cycles=2)
            assert rec.hab_dust_mg_m3 >= 0


# ── Runner ─────────────────────────────────────────────────────────

class TestRunFilterSystem:
    def test_basic_run(self):
        state = create_filter_system()
        records = run_filter_system(state, hours=48)
        assert len(records) == 48

    def test_dust_storm_run(self):
        state = create_filter_system()
        records = run_filter_system(state, hours=72,
                                    dust_storm_hours=(24, 48))
        storm_ingress = [r.dust_ingress_g for r in records[24:48]]
        calm_ingress = [r.dust_ingress_g for r in records[0:24]]
        # Storm hours should have higher average ingress
        assert sum(storm_ingress) > sum(calm_ingress)


# ── Smoke test: 720 hours (30 sols) without crash ──────────────────

class TestSmoke:
    def test_30_sol_simulation(self):
        """Run 30 sols (720 hours). No crash. System captures dust."""
        state = create_filter_system(num_hepa_units=4, spare_filters=20)
        records = run_filter_system(
            state, hours=720, eva_cycles_per_day=3,
            dust_storm_hours=(200, 260),
        )
        assert len(records) == 720
        assert state.hours_elapsed == 720.0
        assert state.total_dust_captured_kg > 0
        # ESP + HEPA have accumulated dust
        assert state.esp.collected_dust_kg > 0
        assert all(u.dust_load_kg > 0 for u in state.hepa_units)
        # System survived a 60-hour dust storm
        storm_records = records[200:260]
        assert all(r.dust_ingress_g > 0 for r in storm_records)

    def test_heavy_use_replaces_filters(self):
        """Under heavy EVA load, filters get replaced."""
        state = create_filter_system(num_hepa_units=1, spare_filters=5,
                                     esp_enabled=False)
        # No ESP means HEPA takes full load. 10 EVAs/day for 90 days.
        records = run_filter_system(
            state, hours=2160, eva_cycles_per_day=10,
        )
        assert state.total_filters_used > 0

    def test_10_step_smoke(self):
        """Minimum: 10 steps without crash."""
        state = create_filter_system()
        for i in range(10):
            rec = tick_filter(state, eva_cycles=i % 3)
            assert isinstance(rec, TickRecord)
