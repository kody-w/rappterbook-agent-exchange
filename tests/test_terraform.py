"""Tests for the Mars-100 terraform engine."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.terraform import (
    TerraformState,
    TerraformDelta,
    tick_terraform,
    event_severity_modifier,
    resource_production_bonus,
    subsim_bindings,
    INITIAL_PRESSURE_ATM,
    INITIAL_TEMPERATURE_C,
    INITIAL_WATER_ACCESS,
    INITIAL_SOIL_FERTILITY,
    INITIAL_RADIATION_REL,
    TARGET_PRESSURE_ATM,
    TARGET_TEMPERATURE_C,
    TARGET_WATER_ACCESS,
    TARGET_SOIL_FERTILITY,
    TARGET_RADIATION_REL,
    TERRAFORM_POWER_COST,
    TERRAFORM_WATER_COST,
)


# ---------- TerraformState basics ----------

def test_initial_state_matches_constants():
    ts = TerraformState()
    assert ts.pressure_atm == INITIAL_PRESSURE_ATM
    assert ts.temperature_c == INITIAL_TEMPERATURE_C
    assert ts.water_access == INITIAL_WATER_ACCESS
    assert ts.soil_fertility == INITIAL_SOIL_FERTILITY
    assert ts.radiation_rel == INITIAL_RADIATION_REL
    assert ts.cumulative_effort == 0.0
    assert ts.milestone() == "barren"


def test_serialization_roundtrip():
    ts = TerraformState(pressure_atm=0.04, temperature_c=-30.0, radiation_rel=0.4,
                        water_access=0.2, soil_fertility=0.12, cumulative_effort=15.0)
    d = ts.to_dict()
    ts2 = TerraformState.from_dict(d)
    assert abs(ts2.pressure_atm - ts.pressure_atm) < 1e-4
    assert abs(ts2.temperature_c - ts.temperature_c) < 0.5
    assert abs(ts2.water_access - ts.water_access) < 1e-3
    assert abs(ts2.soil_fertility - ts.soil_fertility) < 1e-3
    assert abs(ts2.radiation_rel - ts.radiation_rel) < 1e-3
    assert abs(ts2.cumulative_effort - ts.cumulative_effort) < 1e-3


def test_terraforming_score_initial():
    ts = TerraformState()
    assert 0.0 <= ts.terraforming_score() <= 0.05


def test_terraforming_score_advanced():
    ts = TerraformState(pressure_atm=0.08, temperature_c=-20.0,
                        water_access=0.25, soil_fertility=0.15, radiation_rel=0.3)
    assert ts.terraforming_score() > 0.3


# ---------- tick_terraform ----------

def test_zero_effort_no_progress():
    ts = TerraformState()
    rng = random.Random(42)
    delta = tick_terraform(ts, {"col-1": "mediate", "col-2": "pray"}, {"col-1": 0.5, "col-2": 0.3}, rng)
    assert delta.effort_this_year == 0.0
    assert delta.power_cost == 0.0
    assert delta.water_cost == 0.0


def test_effort_increases_pressure():
    ts = TerraformState()
    rng = random.Random(42)
    tick_terraform(ts, {"col-1": "terraform", "col-2": "terraform"}, {"col-1": 0.8, "col-2": 0.6}, rng)
    assert ts.pressure_atm > INITIAL_PRESSURE_ATM


def test_pressure_trend_increases():
    ts = TerraformState()
    rng = random.Random(42)
    pressures = [ts.pressure_atm]
    for _ in range(20):
        tick_terraform(ts, {"col-1": "terraform"}, {"col-1": 0.7}, rng)
        pressures.append(ts.pressure_atm)
    assert pressures[-1] > pressures[0]


def test_diminishing_returns():
    early = TerraformState()
    d1 = tick_terraform(early, {"a": "terraform"}, {"a": 0.8}, random.Random(99))
    late = TerraformState(pressure_atm=TARGET_PRESSURE_ATM * 0.8, cumulative_effort=50.0)
    d2 = tick_terraform(late, {"a": "terraform"}, {"a": 0.8}, random.Random(99))
    assert abs(d2.pressure_delta) < abs(d1.pressure_delta) + 0.001


def test_temperature_gated_on_pressure():
    low = TerraformState(pressure_atm=INITIAL_PRESSURE_ATM * 0.9)
    d = tick_terraform(low, {"a": "terraform"}, {"a": 0.9}, random.Random(42))
    assert abs(d.temperature_delta) < 1.0
    high = TerraformState(pressure_atm=INITIAL_PRESSURE_ATM * 2.0)
    d2 = tick_terraform(high, {"a": "terraform"}, {"a": 0.9}, random.Random(42))
    assert d2.temperature_delta > -0.1


def test_soil_gated_on_water_and_effort():
    dry = TerraformState(water_access=0.02, cumulative_effort=2.0)
    d1 = tick_terraform(dry, {"a": "terraform"}, {"a": 0.9}, random.Random(42))
    assert d1.soil_delta == 0.0
    wet = TerraformState(water_access=0.15, cumulative_effort=10.0)
    d2 = tick_terraform(wet, {"a": "terraform"}, {"a": 0.9}, random.Random(42))
    assert d2.soil_delta > 0.0


def test_resource_costs():
    ts = TerraformState()
    delta = tick_terraform(ts, {"a": "terraform", "b": "terraform"}, {"a": 0.5, "b": 0.5}, random.Random(42))
    assert abs(delta.power_cost - 2 * TERRAFORM_POWER_COST) < 1e-9
    assert abs(delta.water_cost - 2 * TERRAFORM_WATER_COST) < 1e-9


def test_cumulative_effort_tracks():
    ts = TerraformState()
    rng = random.Random(42)
    for _ in range(5):
        tick_terraform(ts, {"a": "terraform"}, {"a": 0.7}, rng)
    assert ts.cumulative_effort > 0.0


# ---------- Milestones ----------

def test_milestone_progression():
    ts = TerraformState()
    rng = random.Random(42)
    milestones = {ts.milestone()}
    for _ in range(300):
        tick_terraform(ts, {"a": "terraform", "b": "terraform"}, {"a": 0.95, "b": 0.95}, rng)
        milestones.add(ts.milestone())
    assert "microbes" in milestones, f"Only reached: {milestones}"


def test_milestone_ordering():
    order = ["barren", "microbes", "plant_life", "ecopoiesis"]
    ts = TerraformState()
    rng = random.Random(42)
    max_idx = 0
    for _ in range(100):
        tick_terraform(ts, {"a": "terraform"}, {"a": 0.8}, rng)
        m = ts.milestone()
        if m in order:
            idx = order.index(m)
            assert idx >= max_idx
            max_idx = idx


# ---------- Modifiers ----------

def test_event_severity_modifier_bounded():
    ts = TerraformState(pressure_atm=0.08, temperature_c=-20.0, water_access=0.2,
                        soil_fertility=0.1, radiation_rel=0.35)
    for _, val in event_severity_modifier(ts).items():
        assert 0.3 <= val <= 1.5


def test_event_severity_modifier_at_baseline():
    for _, val in event_severity_modifier(TerraformState()).items():
        assert 0.9 <= val <= 1.1


def test_event_severity_modifier_reduces_storms():
    barren = event_severity_modifier(TerraformState())
    advanced = event_severity_modifier(TerraformState(
        pressure_atm=0.10, temperature_c=-15.0, water_access=0.3,
        soil_fertility=0.15, radiation_rel=0.3))
    assert advanced["dust_storm"] <= barren["dust_storm"]


def test_resource_bonus_nonnegative():
    for p in [INITIAL_PRESSURE_ATM, 0.05, 0.1]:
        for _, val in resource_production_bonus(TerraformState(pressure_atm=p, water_access=0.2)).items():
            assert val >= 0.0


def test_resource_bonus_increases_with_progress():
    low = sum(resource_production_bonus(TerraformState()).values())
    high = sum(resource_production_bonus(TerraformState(
        pressure_atm=0.08, temperature_c=-10.0, water_access=0.3,
        soil_fertility=0.2, radiation_rel=0.3)).values())
    assert high >= low


# ---------- Sub-sim bindings ----------

def test_subsim_bindings_complete():
    b = subsim_bindings(TerraformState(pressure_atm=0.05, temperature_c=-25.0, water_access=0.15))
    for key in ["mars-pressure", "mars-temperature", "mars-water",
                "mars-soil", "mars-radiation", "terraform-score", "terraform-effort"]:
        assert key in b


def test_subsim_bindings_values():
    ts = TerraformState(pressure_atm=0.05, temperature_c=-25.0, water_access=0.15)
    b = subsim_bindings(ts)
    assert b["mars-pressure"] == 0.05
    assert b["mars-temperature"] == -25.0
    assert b["mars-water"] == 0.15


# ---------- TerraformDelta ----------

def test_delta_serialization():
    d = TerraformDelta(pressure_delta=0.01, temperature_delta=0.5, water_delta=0.005,
                       soil_delta=0.001, radiation_delta=-5.0, effort_this_year=1.5,
                       power_cost=0.04, water_cost=0.02,
                       milestone_before="barren", milestone_after="microbes", milestone_changed=True)
    data = d.to_dict()
    assert data["pressure_delta"] == 0.01
    assert data["milestone_after"] == "microbes"
    assert data["milestone_changed"] is True


def test_delta_defaults():
    d = TerraformDelta()
    assert d.pressure_delta == 0.0
    assert d.effort_this_year == 0.0


# ---------- from_dict edge cases ----------

def test_from_dict_missing_fields():
    ts = TerraformState.from_dict({"pressure_atm": 0.05})
    assert ts.pressure_atm == 0.05
    assert ts.temperature_c == INITIAL_TEMPERATURE_C


def test_from_dict_empty():
    ts = TerraformState.from_dict({})
    assert ts.pressure_atm == INITIAL_PRESSURE_ATM
    assert ts.milestone() == "barren"


# ---------- Integration ----------

def test_hundred_year_terraform():
    ts = TerraformState()
    rng = random.Random(42)
    for year in range(1, 101):
        n = max(1, 3 - year // 40)
        actions = {f"c-{i}": "terraform" for i in range(n)}
        actions.update({f"c-{i}": "mediate" for i in range(n, 5)})
        skills = {f"c-{i}": min(0.5 + year * 0.005, 0.99) for i in range(5)}
        delta = tick_terraform(ts, actions, skills, rng)
        assert delta.power_cost >= 0.0
    assert ts.pressure_atm > INITIAL_PRESSURE_ATM
    assert ts.cumulative_effort > 0.0


def test_physical_bounds_after_100_years():
    ts = TerraformState()
    rng = random.Random(123)
    for _ in range(100):
        tick_terraform(ts, {f"c-{i}": "terraform" for i in range(10)},
                       {f"c-{i}": 0.99 for i in range(10)}, rng)
    assert ts.pressure_atm <= TARGET_PRESSURE_ATM * 1.5
    assert ts.temperature_c <= TARGET_TEMPERATURE_C + 10.0
    assert 0.0 <= ts.water_access <= 1.0
    assert 0.0 <= ts.soil_fertility <= 1.0
    assert ts.radiation_rel >= TARGET_RADIATION_REL


def test_no_terraformers_zero_cost():
    d = tick_terraform(TerraformState(), {"a": "mediate"}, {"a": 0.5}, random.Random(42))
    assert d.power_cost == 0.0
    assert d.water_cost == 0.0
