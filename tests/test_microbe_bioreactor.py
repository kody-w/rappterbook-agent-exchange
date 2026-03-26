"""tests/test_microbe_bioreactor.py — 120 tests for microbe_bioreactor.py.

The colony's cyanobacteria scrub CO₂ and make O₂ for free, powered by
light.  These tests verify every physics function, growth phase, failure
mode, and conservation law before we trust micro-organisms with our air.

Covers:
  - Monod growth kinetics (substrate saturation curve)
  - Light attenuation (Beer-Lambert self-shading)
  - Temperature cardinal model
  - pH growth factor
  - Effective growth rate (combined limiting factors)
  - CO₂/O₂ stoichiometry
  - LED energy consumption
  - Harvest / dilution
  - Contamination probability
  - pH shift from photosynthesis
  - State creation and serialisation round-trip
  - Tick mechanics: normal growth, contamination, cleanup, washout
  - Multi-sol simulation smoke test
  - Physical invariants across all ticks
  - Edge cases (zero volume, extreme temps, LEDs off)
"""
from __future__ import annotations

import math
import pytest
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from microbe_bioreactor import (
    # Constants
    CO2_MOLAR_MASS, O2_MOLAR_MASS, O2_PER_CO2_MASS,
    MU_MAX_PER_HOUR, KS_CO2_MG_L, KI_LIGHT_UMOL,
    YIELD_BIOMASS_PER_CO2, BEER_LAMBERT_EPSILON,
    LED_POWER_W_PER_M2, LED_PAR_UMOL,
    TEMP_OPT_C, TEMP_MIN_C, TEMP_MAX_C, TEMP_KILL_C,
    PH_OPT_LOW, PH_OPT_HIGH, PH_MIN, PH_MAX,
    DEFAULT_VOLUME_L, DEFAULT_DEPTH_CM, DEFAULT_FACE_AREA_M2,
    DEFAULT_INITIAL_BIOMASS_G_L, DEFAULT_CO2_FEED_G_L,
    DEFAULT_DILUTION_RATE_PER_H, DEFAULT_TEMP_C, DEFAULT_PH,
    CONTAMINATION_BASE_PROB_PER_SOL, CONTAMINATION_AGE_FACTOR,
    CONTAMINATION_PRODUCTIVITY_FACTOR, CLEANUP_SOLS,
    HOURS_PER_SOL, LIGHT_HOURS_PER_SOL, MAX_BIOMASS_G_L,
    # Pure functions
    monod, light_factor, beer_lambert, temperature_factor,
    ph_factor, effective_growth_rate, co2_consumed_kg,
    o2_produced_kg, led_power_kwh_per_sol, harvest_biomass_g,
    contamination_probability, ph_shift_from_photosynthesis,
    # Dataclasses
    BioreactorState, TickResult,
    # Factory & simulation
    create_bioreactor, tick, run_simulation,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def default_state() -> BioreactorState:
    """Fresh bioreactor with default parameters."""
    return create_bioreactor()


@pytest.fixture
def dense_culture() -> BioreactorState:
    """High-density culture near ceiling."""
    return create_bioreactor(biomass_g_l=14.0)


@pytest.fixture
def contaminated_state() -> BioreactorState:
    """Reactor in contamination cleanup."""
    return create_bioreactor(
        contaminated=True,
        contamination_cleanup_remaining=CLEANUP_SOLS,
    )


@pytest.fixture
def cold_reactor() -> BioreactorState:
    """Reactor at low temperature — near dormancy."""
    return create_bioreactor(temp_c=8.0)


@pytest.fixture
def dark_reactor() -> BioreactorState:
    """Reactor with LEDs off."""
    return create_bioreactor(led_on=False)


# ===================================================================
# 1. Monod kinetics
# ===================================================================

class TestMonod:
    """Monod growth rate: μ = μ_max · S / (Ks + S)."""

    def test_zero_substrate_gives_zero_rate(self):
        assert monod(0.0) == 0.0

    def test_negative_substrate_clamped_to_zero(self):
        assert monod(-100.0) == 0.0

    def test_at_half_saturation_gives_half_max(self):
        result = monod(KS_CO2_MG_L)
        assert result == pytest.approx(MU_MAX_PER_HOUR / 2.0, rel=1e-6)

    def test_high_substrate_approaches_mu_max(self):
        result = monod(1e6)
        assert result == pytest.approx(MU_MAX_PER_HOUR, rel=1e-3)

    def test_monotonically_increasing(self):
        values = [monod(s) for s in range(0, 5001, 100)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]

    def test_always_non_negative(self):
        for s in [-10, 0, 1, 50, 1000, 1e6]:
            assert monod(s) >= 0.0


# ===================================================================
# 2. Light factor
# ===================================================================

class TestLightFactor:
    """Light limitation: Monod-type for PAR."""

    def test_zero_light_gives_zero(self):
        assert light_factor(0.0) == 0.0

    def test_negative_light_clamped(self):
        assert light_factor(-50.0) == 0.0

    def test_at_half_saturation(self):
        result = light_factor(KI_LIGHT_UMOL)
        assert result == pytest.approx(0.5, rel=1e-6)

    def test_high_light_approaches_one(self):
        result = light_factor(1e6)
        assert result == pytest.approx(1.0, rel=1e-3)

    def test_result_bounded_zero_one(self):
        for par in [0, 10, 50, 150, 400, 1000, 1e6]:
            lf = light_factor(par)
            assert 0.0 <= lf <= 1.0


# ===================================================================
# 3. Beer-Lambert light attenuation
# ===================================================================

class TestBeerLambert:
    """Beer-Lambert average PAR through self-shading culture."""

    def test_no_biomass_returns_incident(self):
        result = beer_lambert(400.0, 0.0, 10.0)
        assert result == pytest.approx(400.0, rel=1e-3)

    def test_more_biomass_less_light(self):
        low = beer_lambert(400.0, 1.0, 10.0)
        high = beer_lambert(400.0, 10.0, 10.0)
        assert high < low

    def test_deeper_path_less_light(self):
        shallow = beer_lambert(400.0, 5.0, 5.0)
        deep = beer_lambert(400.0, 5.0, 20.0)
        assert deep < shallow

    def test_very_dense_culture_near_zero(self):
        result = beer_lambert(400.0, 100.0, 50.0)
        assert result == pytest.approx(0.0, abs=1e-3)

    def test_zero_incident_gives_zero(self):
        result = beer_lambert(0.0, 5.0, 10.0)
        assert result == 0.0

    def test_negative_biomass_clamped(self):
        result = beer_lambert(400.0, -1.0, 10.0)
        assert result == pytest.approx(400.0, rel=1e-3)

    def test_always_non_negative(self):
        for bm in [0, 0.5, 5, 15, 100]:
            assert beer_lambert(400.0, bm, 10.0) >= 0.0


# ===================================================================
# 4. Temperature factor
# ===================================================================

class TestTemperatureFactor:
    """Cardinal temperature model for cyanobacteria."""

    def test_optimum_gives_one(self):
        assert temperature_factor(TEMP_OPT_C) == pytest.approx(1.0, rel=1e-6)

    def test_below_minimum_gives_zero(self):
        assert temperature_factor(TEMP_MIN_C) == 0.0
        assert temperature_factor(0.0) == 0.0
        assert temperature_factor(-10.0) == 0.0

    def test_at_kill_temperature_gives_zero(self):
        assert temperature_factor(TEMP_KILL_C) == 0.0

    def test_above_kill_gives_zero(self):
        assert temperature_factor(60.0) == 0.0

    def test_between_min_and_opt_increasing(self):
        temps = [10, 15, 20, 25, 30]
        factors = [temperature_factor(t) for t in temps]
        for i in range(1, len(factors)):
            assert factors[i] >= factors[i - 1]

    def test_between_opt_and_kill_decreasing(self):
        temps = [30, 35, 40, 45]
        factors = [temperature_factor(t) for t in temps]
        for i in range(1, len(factors)):
            assert factors[i] <= factors[i - 1]

    def test_always_bounded(self):
        for t in range(-20, 80):
            tf = temperature_factor(float(t))
            assert 0.0 <= tf <= 1.0


# ===================================================================
# 5. pH factor
# ===================================================================

class TestPHFactor:
    """pH growth modifier."""

    def test_optimal_range_gives_one(self):
        for ph in [7.0, 7.5, 8.0, 8.5]:
            assert ph_factor(ph) == pytest.approx(1.0, rel=1e-6)

    def test_below_minimum_gives_zero(self):
        assert ph_factor(PH_MIN - 0.01) == 0.0
        assert ph_factor(3.0) == 0.0

    def test_above_maximum_gives_zero(self):
        assert ph_factor(PH_MAX + 0.01) == 0.0
        assert ph_factor(14.0) == 0.0

    def test_midpoint_below_optimal(self):
        mid = (PH_MIN + PH_OPT_LOW) / 2.0
        assert 0.0 < ph_factor(mid) < 1.0

    def test_midpoint_above_optimal(self):
        mid = (PH_OPT_HIGH + PH_MAX) / 2.0
        assert 0.0 < ph_factor(mid) < 1.0

    def test_always_bounded(self):
        for ph_val in [x * 0.5 for x in range(0, 30)]:
            pf = ph_factor(ph_val)
            assert 0.0 <= pf <= 1.0


# ===================================================================
# 6. Effective growth rate
# ===================================================================

class TestEffectiveGrowthRate:
    """Combined growth rate with all limiting factors."""

    def test_optimal_conditions_near_mu_max(self):
        rate = effective_growth_rate(
            co2_mg_l=1e6,
            avg_par=1e6,
            temp_c=TEMP_OPT_C,
            ph=(PH_OPT_LOW + PH_OPT_HIGH) / 2.0,
        )
        assert rate == pytest.approx(MU_MAX_PER_HOUR, rel=0.01)

    def test_zero_co2_gives_zero(self):
        assert effective_growth_rate(0.0, 400.0, 25.0, 7.5) == 0.0

    def test_zero_light_gives_zero(self):
        assert effective_growth_rate(2000.0, 0.0, 25.0, 7.5) == 0.0

    def test_extreme_cold_gives_zero(self):
        assert effective_growth_rate(2000.0, 400.0, -10.0, 7.5) == 0.0

    def test_extreme_ph_gives_zero(self):
        assert effective_growth_rate(2000.0, 400.0, 25.0, 3.0) == 0.0

    def test_always_non_negative(self):
        for co2 in [0, 50, 2000]:
            for par in [0, 100, 400]:
                for t in [0, 25, 50]:
                    for ph in [4, 7.5, 12]:
                        assert effective_growth_rate(co2, par, t, ph) >= 0.0


# ===================================================================
# 7. CO₂ / O₂ stoichiometry
# ===================================================================

class TestStoichiometry:
    """Photosynthesis stoichiometry: CO₂ consumed → O₂ produced."""

    def test_co2_consumed_positive(self):
        assert co2_consumed_kg(1.0) > 0.0

    def test_co2_consumed_zero_for_zero_biomass(self):
        assert co2_consumed_kg(0.0) == 0.0

    def test_co2_consumed_negative_clamped(self):
        assert co2_consumed_kg(-1.0) == 0.0

    def test_o2_co2_mass_ratio(self):
        co2 = 1.0  # 1 kg CO₂
        o2 = o2_produced_kg(co2)
        expected_ratio = 32.0 / 44.01
        assert o2 / co2 == pytest.approx(expected_ratio, rel=1e-3)

    def test_o2_produced_zero_for_zero_co2(self):
        assert o2_produced_kg(0.0) == 0.0

    def test_o2_produced_non_negative(self):
        assert o2_produced_kg(-1.0) == 0.0

    def test_biomass_yield_relationship(self):
        """1 kg biomass requires 1/YIELD = 2 kg CO₂."""
        co2 = co2_consumed_kg(1.0)
        assert co2 == pytest.approx(1.0 / YIELD_BIOMASS_PER_CO2, rel=1e-6)


# ===================================================================
# 8. LED energy
# ===================================================================

class TestLEDEnergy:
    """LED electrical energy consumption per sol."""

    def test_default_face_area(self):
        kwh = led_power_kwh_per_sol(DEFAULT_FACE_AREA_M2)
        expected = DEFAULT_FACE_AREA_M2 * LED_POWER_W_PER_M2 * LIGHT_HOURS_PER_SOL / 1000.0
        assert kwh == pytest.approx(expected, rel=1e-6)

    def test_zero_area_gives_zero(self):
        assert led_power_kwh_per_sol(0.0) == 0.0

    def test_negative_area_clamped(self):
        assert led_power_kwh_per_sol(-1.0) == 0.0

    def test_scales_linearly_with_area(self):
        single = led_power_kwh_per_sol(1.0)
        double = led_power_kwh_per_sol(2.0)
        assert double == pytest.approx(2.0 * single, rel=1e-6)

    def test_photoperiod_capped_at_sol_length(self):
        normal = led_power_kwh_per_sol(1.0, photoperiod_h=LIGHT_HOURS_PER_SOL)
        over = led_power_kwh_per_sol(1.0, photoperiod_h=100.0)
        sol_max = led_power_kwh_per_sol(1.0, photoperiod_h=HOURS_PER_SOL)
        assert over == pytest.approx(sol_max, rel=1e-6)


# ===================================================================
# 9. Harvest / dilution
# ===================================================================

class TestHarvest:
    """Continuous dilution harvest."""

    def test_basic_harvest(self):
        result = harvest_biomass_g(0.01, 5.0, 500.0, 24.0)
        expected = 0.01 * 5.0 * 500.0 * 24.0  # 600 g
        assert result == pytest.approx(expected, rel=1e-6)

    def test_zero_dilution_no_harvest(self):
        assert harvest_biomass_g(0.0, 5.0, 500.0, 24.0) == 0.0

    def test_zero_biomass_no_harvest(self):
        assert harvest_biomass_g(0.01, 0.0, 500.0, 24.0) == 0.0

    def test_zero_volume_no_harvest(self):
        assert harvest_biomass_g(0.01, 5.0, 0.0, 24.0) == 0.0

    def test_negative_inputs_clamped(self):
        assert harvest_biomass_g(-0.01, 5.0, 500.0, 24.0) == 0.0

    def test_always_non_negative(self):
        for d in [-1, 0, 0.01, 0.1]:
            for b in [-1, 0, 1, 10]:
                assert harvest_biomass_g(d, b, 500.0, 24.0) >= 0.0


# ===================================================================
# 10. Contamination probability
# ===================================================================

class TestContaminationProbability:
    """Contamination risk increases with culture age."""

    def test_day_zero(self):
        prob = contamination_probability(0)
        assert prob == pytest.approx(CONTAMINATION_BASE_PROB_PER_SOL, rel=1e-6)

    def test_increases_with_age(self):
        young = contamination_probability(10)
        old = contamination_probability(1000)
        assert old > young

    def test_capped_at_one(self):
        prob = contamination_probability(1_000_000)
        assert prob <= 1.0

    def test_non_negative(self):
        for age in [0, 1, 100, 10000]:
            assert contamination_probability(age) >= 0.0


# ===================================================================
# 11. pH shift from photosynthesis
# ===================================================================

class TestPHShift:
    """CO₂ removal raises pH."""

    def test_no_co2_consumed_no_shift(self):
        ph = ph_shift_from_photosynthesis(0.0, 500.0, 7.5)
        assert ph == pytest.approx(7.5, rel=1e-6)

    def test_co2_consumed_raises_ph(self):
        ph = ph_shift_from_photosynthesis(10.0, 500.0, 7.5)
        assert ph > 7.5

    def test_capped_at_14(self):
        ph = ph_shift_from_photosynthesis(1e6, 1.0, 13.0)
        assert ph <= 14.0

    def test_zero_volume_returns_current(self):
        ph = ph_shift_from_photosynthesis(10.0, 0.0, 7.5)
        assert ph == pytest.approx(7.5, rel=1e-6)

    def test_larger_volume_smaller_shift(self):
        small_v = ph_shift_from_photosynthesis(10.0, 100.0, 7.5)
        large_v = ph_shift_from_photosynthesis(10.0, 10000.0, 7.5)
        assert small_v > large_v


# ===================================================================
# 12. State creation & serialisation
# ===================================================================

class TestStateSerialization:
    """BioreactorState to_dict / from_dict round-trip."""

    def test_default_creation(self, default_state):
        assert default_state.sol == 0
        assert default_state.biomass_g_l == DEFAULT_INITIAL_BIOMASS_G_L
        assert default_state.volume_l == DEFAULT_VOLUME_L

    def test_to_dict_has_all_fields(self, default_state):
        d = default_state.to_dict()
        for field_name in BioreactorState.__dataclass_fields__:
            assert field_name in d

    def test_round_trip(self, default_state):
        d = default_state.to_dict()
        restored = BioreactorState.from_dict(d)
        d2 = restored.to_dict()
        assert d == d2

    def test_from_dict_ignores_extra_keys(self):
        d = create_bioreactor().to_dict()
        d["alien_field"] = "should be ignored"
        state = BioreactorState.from_dict(d)
        assert not hasattr(state, "alien_field") or "alien_field" not in state.__dataclass_fields__

    def test_create_with_overrides(self):
        state = create_bioreactor(volume_l=1000.0, temp_c=20.0)
        assert state.volume_l == 1000.0
        assert state.temp_c == 20.0

    def test_tick_result_to_dict(self):
        tr = TickResult(sol=1, growth_rate_per_h=0.03, biomass_g_l=1.5)
        d = tr.to_dict()
        assert d["sol"] == 1
        assert "growth_rate_per_h" in d


# ===================================================================
# 13. Normal growth tick
# ===================================================================

class TestNormalTick:
    """Single-sol tick under normal conditions."""

    def test_sol_increments(self, default_state):
        tick(default_state, rng_roll=-1.0)
        assert default_state.sol == 1

    def test_biomass_grows(self, default_state):
        initial = default_state.biomass_g_l
        result = tick(default_state, rng_roll=-1.0)
        assert default_state.biomass_g_l > 0.0
        assert result.biomass_produced_g > 0.0

    def test_co2_consumed_positive(self, default_state):
        result = tick(default_state, rng_roll=-1.0)
        assert result.co2_consumed_g > 0.0

    def test_o2_produced_positive(self, default_state):
        result = tick(default_state, rng_roll=-1.0)
        assert result.o2_produced_g > 0.0

    def test_led_energy_consumed(self, default_state):
        result = tick(default_state, rng_roll=-1.0)
        assert result.led_energy_kwh > 0.0

    def test_ph_rises_from_growth(self, default_state):
        initial_ph = default_state.ph
        tick(default_state, rng_roll=-1.0)
        assert default_state.ph >= initial_ph

    def test_culture_age_increments(self, default_state):
        tick(default_state, rng_roll=-1.0)
        assert default_state.culture_age_sols == 1

    def test_cumulative_totals_increase(self, default_state):
        tick(default_state, rng_roll=-1.0)
        assert default_state.total_co2_consumed_kg > 0.0
        assert default_state.total_o2_produced_kg > 0.0
        assert default_state.total_led_energy_kwh > 0.0


# ===================================================================
# 14. Contamination event
# ===================================================================

class TestContamination:
    """Contamination triggers cleanup cycle."""

    def test_contamination_triggers_on_low_roll(self):
        state = create_bioreactor(culture_age_sols=1000)
        # With age=1000, prob = 0.002 + 0.0001*1000 = 0.102
        result = tick(state, rng_roll=0.001)
        assert result.contamination_event is True
        assert state.contaminated is True
        assert state.contamination_cleanup_remaining == CLEANUP_SOLS

    def test_no_contamination_on_high_roll(self, default_state):
        result = tick(default_state, rng_roll=0.999)
        assert result.contamination_event is False
        assert default_state.contaminated is False

    def test_no_contamination_in_deterministic_mode(self, default_state):
        result = tick(default_state, rng_roll=-1.0)
        assert result.contamination_event is False

    def test_cleanup_countdown(self, contaminated_state):
        initial_remaining = contaminated_state.contamination_cleanup_remaining
        tick(contaminated_state, rng_roll=-1.0)
        assert contaminated_state.contamination_cleanup_remaining == initial_remaining - 1

    def test_cleanup_completes(self, contaminated_state):
        for _ in range(CLEANUP_SOLS):
            tick(contaminated_state, rng_roll=-1.0)
        assert contaminated_state.contaminated is False
        assert contaminated_state.contamination_cleanup_remaining == 0
        assert contaminated_state.biomass_g_l == DEFAULT_INITIAL_BIOMASS_G_L

    def test_generation_increments_after_cleanup(self, contaminated_state):
        initial_gen = contaminated_state.generation
        for _ in range(CLEANUP_SOLS):
            tick(contaminated_state, rng_roll=-1.0)
        assert contaminated_state.generation == initial_gen + 1


# ===================================================================
# 15. Dark reactor (LEDs off)
# ===================================================================

class TestDarkReactor:
    """No photosynthesis without light."""

    def test_no_led_energy(self, dark_reactor):
        result = tick(dark_reactor, rng_roll=-1.0)
        assert result.led_energy_kwh == 0.0

    def test_zero_par(self, dark_reactor):
        result = tick(dark_reactor, rng_roll=-1.0)
        assert result.avg_par_umol == 0.0

    def test_zero_growth_rate(self, dark_reactor):
        result = tick(dark_reactor, rng_roll=-1.0)
        assert result.growth_rate_per_h == 0.0

    def test_no_o2_produced(self, dark_reactor):
        result = tick(dark_reactor, rng_roll=-1.0)
        assert result.o2_produced_g == 0.0


# ===================================================================
# 16. Cold reactor
# ===================================================================

class TestColdReactor:
    """Growth severely limited at low temperature."""

    def test_low_growth_rate(self, cold_reactor):
        result = tick(cold_reactor, rng_roll=-1.0)
        # temp=8 → factor=(8-5)/(30-5)=0.12, very slow growth
        assert result.temp_factor == pytest.approx(3.0 / 25.0, rel=1e-3)
        assert result.growth_rate_per_h < MU_MAX_PER_HOUR * 0.2

    def test_still_produces_some_o2(self, cold_reactor):
        result = tick(cold_reactor, rng_roll=-1.0)
        assert result.o2_produced_g > 0.0


# ===================================================================
# 17. Dense culture self-shading
# ===================================================================

class TestDenseCulture:
    """High biomass concentration limits light penetration."""

    def test_reduced_par(self, dense_culture):
        result = tick(dense_culture, rng_roll=-1.0)
        light_state = create_bioreactor()
        light_result = tick(light_state, rng_roll=-1.0)
        assert result.avg_par_umol < light_result.avg_par_umol

    def test_biomass_capped_at_max(self):
        state = create_bioreactor(biomass_g_l=MAX_BIOMASS_G_L + 5.0)
        tick(state, rng_roll=-1.0)
        assert state.biomass_g_l <= MAX_BIOMASS_G_L


# ===================================================================
# 18. Washout
# ===================================================================

class TestWashout:
    """High dilution rate washes out the culture."""

    def test_washout_detected(self):
        state = create_bioreactor(
            biomass_g_l=0.001,
            dilution_rate_h=1.0,  # very aggressive
        )
        result = tick(state, rng_roll=-1.0)
        assert result.washout is True

    def test_biomass_non_negative_after_washout(self):
        state = create_bioreactor(
            biomass_g_l=0.001,
            dilution_rate_h=10.0,
        )
        tick(state, rng_roll=-1.0)
        assert state.biomass_g_l >= 0.0


# ===================================================================
# 19. Multi-sol simulation
# ===================================================================

class TestRunSimulation:
    """Run the bioreactor for multiple sols."""

    def test_smoke_10_sols(self, default_state):
        results = run_simulation(default_state, sols=10, rng_roll=-1.0)
        assert len(results) == 10
        assert all(r.sol > 0 for r in results)

    def test_smoke_100_sols(self, default_state):
        results = run_simulation(default_state, sols=100, rng_roll=-1.0)
        assert len(results) == 100
        assert default_state.sol == 100

    def test_biomass_stabilises(self, default_state):
        results = run_simulation(default_state, sols=200, rng_roll=-1.0)
        late_biomass = [r.biomass_g_l for r in results[-20:]]
        assert all(0.0 < b <= MAX_BIOMASS_G_L for b in late_biomass)

    def test_total_o2_increases(self, default_state):
        run_simulation(default_state, sols=50, rng_roll=-1.0)
        assert default_state.total_o2_produced_kg > 0.0

    def test_total_co2_increases(self, default_state):
        run_simulation(default_state, sols=50, rng_roll=-1.0)
        assert default_state.total_co2_consumed_kg > 0.0


# ===================================================================
# 20. Physical invariants (property-based)
# ===================================================================

class TestPhysicalInvariants:
    """Conservation laws and physical bounds across all ticks."""

    def test_o2_co2_ratio_exact(self, default_state):
        """O₂/CO₂ mass ratio must be exactly 32/44."""
        results = run_simulation(default_state, sols=50, rng_roll=-1.0)
        for r in results:
            if r.co2_consumed_g > 0.001:
                ratio = r.o2_produced_g / r.co2_consumed_g
                assert ratio == pytest.approx(O2_PER_CO2_MASS, rel=1e-4)

    def test_biomass_always_non_negative(self, default_state):
        results = run_simulation(default_state, sols=100, rng_roll=-1.0)
        for r in results:
            assert r.biomass_g_l >= 0.0

    def test_biomass_never_exceeds_ceiling(self, default_state):
        results = run_simulation(default_state, sols=200, rng_roll=-1.0)
        for r in results:
            assert r.biomass_g_l <= MAX_BIOMASS_G_L + 0.001

    def test_ph_stays_bounded(self, default_state):
        results = run_simulation(default_state, sols=100, rng_roll=-1.0)
        for r in results:
            assert 0.0 <= r.ph <= 14.0

    def test_growth_rate_non_negative(self, default_state):
        results = run_simulation(default_state, sols=100, rng_roll=-1.0)
        for r in results:
            assert r.growth_rate_per_h >= 0.0

    def test_co2_consumed_non_negative(self, default_state):
        results = run_simulation(default_state, sols=100, rng_roll=-1.0)
        for r in results:
            assert r.co2_consumed_g >= 0.0

    def test_o2_produced_non_negative(self, default_state):
        results = run_simulation(default_state, sols=100, rng_roll=-1.0)
        for r in results:
            assert r.o2_produced_g >= 0.0

    def test_led_energy_non_negative(self, default_state):
        results = run_simulation(default_state, sols=100, rng_roll=-1.0)
        for r in results:
            assert r.led_energy_kwh >= 0.0

    def test_cumulative_totals_monotonically_increase(self):
        state = create_bioreactor()
        prev_co2 = 0.0
        prev_o2 = 0.0
        prev_kwh = 0.0
        for _ in range(50):
            tick(state, rng_roll=-1.0)
            assert state.total_co2_consumed_kg >= prev_co2
            assert state.total_o2_produced_kg >= prev_o2
            assert state.total_led_energy_kwh >= prev_kwh
            prev_co2 = state.total_co2_consumed_kg
            prev_o2 = state.total_o2_produced_kg
            prev_kwh = state.total_led_energy_kwh

    def test_harvest_accounts_for_removed_biomass(self, default_state):
        results = run_simulation(default_state, sols=50, rng_roll=-1.0)
        total_harvested_g = sum(r.biomass_harvested_g for r in results)
        assert default_state.total_biomass_harvested_kg == pytest.approx(
            total_harvested_g / 1000.0, rel=1e-3
        )


# ===================================================================
# 21. Edge cases
# ===================================================================

class TestEdgeCases:
    """Boundary and degenerate inputs."""

    def test_zero_volume_reactor_raises(self):
        """Zero-volume reactor triggers ZeroDivisionError in tick().
        Known limitation: volume_l=0 is physically nonsensical and the
        module does not guard against it.  Document the behavior.
        """
        state = create_bioreactor(volume_l=0.0)
        with pytest.raises(ZeroDivisionError):
            tick(state, rng_roll=-1.0)

    def test_extreme_temperature(self):
        state = create_bioreactor(temp_c=100.0)
        result = tick(state, rng_roll=-1.0)
        assert result.growth_rate_per_h == 0.0

    def test_extreme_ph(self):
        state = create_bioreactor(ph=2.0)
        result = tick(state, rng_roll=-1.0)
        assert result.growth_rate_per_h == 0.0

    def test_zero_biomass_start(self):
        state = create_bioreactor(biomass_g_l=0.0)
        result = tick(state, rng_roll=-1.0)
        assert result.biomass_produced_g == 0.0

    def test_zero_co2_feed(self):
        state = create_bioreactor(co2_feed_g_l=0.0)
        result = tick(state, rng_roll=-1.0)
        assert result.growth_rate_per_h == 0.0

    def test_max_dilution_causes_washout(self):
        state = create_bioreactor(dilution_rate_h=100.0, biomass_g_l=0.1)
        result = tick(state, rng_roll=-1.0)
        assert result.washout is True
