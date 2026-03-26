"""
tests/test_circadian.py -- Unit tests for engine/circadian.py (the fly's internal clock).

190 lines of circadian rhythm, hormones, wing degradation, bacterial ecology,
and danger memory -- all untested until now.

53 votes said ship code. One file. One test. One merge.

Run:
    python -m pytest tests/test_circadian.py -v
"""
from __future__ import annotations

import copy
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.circadian import update_circadian


def make_fly_state(
    stage="adult",
    time_of_day=0.5,
    is_airborne=False,
    wing_condition=None,
    bacterial_load=None,
    melatonin=0.0,
    cortisol=0.5,
    hunger=20.0,
    fear_level=0.0,
    current_goal="idle",
    total_ticks=100,
    stage_tick=10,
    adult_duration=65,
    energy=80.0,
    wind=0.0,
    stress_level=0.0,
    smells=None,
    kitchen_objects=None,
    danger_zones=None,
):
    """Build a minimal fly state dict for circadian testing."""
    state = {
        "_meta": {"frame": 1},
        "lifecycle": {
            "stage": stage,
            "total_ticks": total_ticks,
            "stage_tick": stage_tick,
            "stage_durations": {"adult": adult_duration},
        },
        "body": {
            "position": {"x": 100.0, "y": 100.0, "z": 0.0},
            "velocity": {"x": 1.0, "y": 2.0, "z": 0.0},
            "is_airborne": is_airborne,
            "surface": "counter",
        },
        "brain": {
            "current_goal": current_goal,
            "state": "active",
            "fear_level": fear_level,
        },
        "kitchen": {
            "time_of_day": time_of_day,
            "ambient_temp": 22.0,
            "objects": kitchen_objects or [],
        },
        "energy": {"current": energy, "hunger": hunger},
        "genome": [0.5] * 10,
        "senses": {"smell": smells or [], "wind": wind},
        "memory": {"danger_zones": danger_zones or []},
        "stress": {"level": stress_level},
        "history": [],
    }
    if wing_condition is not None:
        state["body"]["wing_condition"] = wing_condition
    if bacterial_load is not None:
        state["body"]["bacterial_load"] = bacterial_load
    if melatonin != 0.0 or cortisol != 0.5:
        state["brain"]["circadian"] = {
            "melatonin": melatonin,
            "cortisol": cortisol,
            "body_temp_offset": 0.0,
            "wing_beat_hz": 200,
            "phase": "awake",
            "sleep_debt": 0.0,
        }
    return state


class TestInitialization:
    def test_wing_condition_initialized_adult(self):
        s = make_fly_state(stage="adult")
        assert "wing_condition" not in s["body"]
        update_circadian(s)
        assert s["body"]["wing_condition"] == 1.0

    def test_wing_condition_initialized_egg(self):
        s = make_fly_state(stage="egg")
        update_circadian(s)
        assert s["body"]["wing_condition"] == 0.0

    def test_wing_condition_initialized_larva(self):
        s = make_fly_state(stage="larva")
        update_circadian(s)
        assert s["body"]["wing_condition"] == 0.0

    def test_bacterial_load_initialized(self):
        s = make_fly_state(stage="adult")
        assert "bacterial_load" not in s["body"]
        update_circadian(s)
        assert "bacterial_load" in s["body"]

    def test_weather_initialized(self):
        s = make_fly_state()
        assert "weather" not in s["kitchen"]
        update_circadian(s)
        assert s["kitchen"]["weather"] == "clear"

    def test_circadian_dict_initialized(self):
        s = make_fly_state()
        assert "circadian" not in s["brain"]
        update_circadian(s)
        circ = s["brain"]["circadian"]
        assert "melatonin" in circ
        assert "cortisol" in circ
        assert "phase" in circ
        assert "sleep_debt" in circ
        assert "wing_beat_hz" in circ
        assert "body_temp_offset" in circ

    def test_danger_zones_initialized(self):
        s = make_fly_state()
        del s["memory"]["danger_zones"]
        update_circadian(s)
        assert "danger_zones" in s["memory"]

    def test_idempotent_initialization(self):
        s = make_fly_state(stage="adult", wing_condition=0.8, bacterial_load=0.3)
        s["brain"]["circadian"] = {
            "melatonin": 0.7, "cortisol": 0.2, "body_temp_offset": 0.0,
            "wing_beat_hz": 180, "phase": "drowsy", "sleep_debt": 0.3,
        }
        update_circadian(s)
        assert s["brain"]["circadian"]["melatonin"] != 0.0


class TestHormones:
    def test_dark_raises_melatonin(self):
        s = make_fly_state(time_of_day=0.1, melatonin=0.3, cortisol=0.5)
        update_circadian(s)
        assert s["brain"]["circadian"]["melatonin"] > 0.3

    def test_dark_lowers_cortisol(self):
        s = make_fly_state(time_of_day=0.1, melatonin=0.3, cortisol=0.5)
        update_circadian(s)
        assert s["brain"]["circadian"]["cortisol"] < 0.5

    def test_light_lowers_melatonin(self):
        s = make_fly_state(time_of_day=0.5, melatonin=0.5, cortisol=0.3)
        update_circadian(s)
        assert s["brain"]["circadian"]["melatonin"] < 0.5

    def test_light_raises_cortisol(self):
        s = make_fly_state(time_of_day=0.5, melatonin=0.5, cortisol=0.3)
        update_circadian(s)
        assert s["brain"]["circadian"]["cortisol"] > 0.3

    def test_late_dark_raises_melatonin(self):
        s = make_fly_state(time_of_day=0.95, melatonin=0.2, cortisol=0.6)
        update_circadian(s)
        assert s["brain"]["circadian"]["melatonin"] > 0.2

    def test_melatonin_clamped_at_1(self):
        s = make_fly_state(time_of_day=0.1, melatonin=0.98, cortisol=0.1)
        update_circadian(s)
        assert s["brain"]["circadian"]["melatonin"] <= 1.0

    def test_cortisol_clamped_at_0(self):
        s = make_fly_state(time_of_day=0.1, melatonin=0.5, cortisol=0.02)
        update_circadian(s)
        assert s["brain"]["circadian"]["cortisol"] >= 0.0

    def test_cortisol_clamped_at_1(self):
        s = make_fly_state(time_of_day=0.5, melatonin=0.1, cortisol=0.99)
        update_circadian(s)
        assert s["brain"]["circadian"]["cortisol"] <= 1.0

    def test_melatonin_clamped_at_0(self):
        s = make_fly_state(time_of_day=0.5, melatonin=0.02, cortisol=0.5)
        update_circadian(s)
        assert s["brain"]["circadian"]["melatonin"] >= 0.0

    def test_full_day_cycle_melatonin_rises_and_falls(self):
        s = make_fly_state(time_of_day=0.0, melatonin=0.0, cortisol=0.5)
        random.seed(999)
        for _ in range(20):
            s["kitchen"]["time_of_day"] = (s["kitchen"]["time_of_day"] + 0.009) % 1.0
            update_circadian(s)
        night_mel = s["brain"]["circadian"]["melatonin"]
        for _ in range(40):
            s["kitchen"]["time_of_day"] = (s["kitchen"]["time_of_day"] + 0.009) % 1.0
            update_circadian(s)
        day_mel = s["brain"]["circadian"]["melatonin"]
        assert night_mel > day_mel


class TestSleepPhase:
    def test_egg_is_dormant(self):
        s = make_fly_state(stage="egg")
        update_circadian(s)
        assert s["brain"]["circadian"]["phase"] == "dormant"

    def test_pupa_is_dormant(self):
        s = make_fly_state(stage="pupa")
        update_circadian(s)
        assert s["brain"]["circadian"]["phase"] == "dormant"

    def test_death_is_dormant(self):
        s = make_fly_state(stage="death")
        update_circadian(s)
        assert s["brain"]["circadian"]["phase"] == "dormant"

    def test_larva_is_awake(self):
        s = make_fly_state(stage="larva")
        update_circadian(s)
        assert s["brain"]["circadian"]["phase"] == "awake"

    def test_adult_drowsy_high_melatonin_low_hunger(self):
        s = make_fly_state(melatonin=0.75, hunger=20.0, fear_level=0.0)
        s["kitchen"]["time_of_day"] = 0.1
        update_circadian(s)
        phase = s["brain"]["circadian"]["phase"]
        assert phase in ("drowsy", "sleeping")

    def test_adult_sleeping_bypasses_drowsy(self):
        """Sleeping needs mel>0.85 AND hunger>=35 (skips drowsy branch) AND fear<0.3."""
        s = make_fly_state(melatonin=0.9, hunger=40.0, fear_level=0.0)
        s["kitchen"]["time_of_day"] = 0.1
        update_circadian(s)
        assert s["brain"]["circadian"]["phase"] == "sleeping"

    def test_adult_awake_in_daylight(self):
        s = make_fly_state(melatonin=0.1, cortisol=0.5, hunger=50.0)
        s["kitchen"]["time_of_day"] = 0.5
        update_circadian(s)
        assert s["brain"]["circadian"]["phase"] == "awake"

    def test_fear_prevents_sleep(self):
        s = make_fly_state(melatonin=0.9, hunger=40.0, fear_level=0.5)
        s["kitchen"]["time_of_day"] = 0.1
        update_circadian(s)
        assert s["brain"]["circadian"]["phase"] == "awake"

    def test_high_hunger_prevents_drowsy(self):
        s = make_fly_state(melatonin=0.75, hunger=40.0, fear_level=0.0)
        s["kitchen"]["time_of_day"] = 0.1
        update_circadian(s)
        phase = s["brain"]["circadian"]["phase"]
        assert phase in ("sleeping", "awake")

    def test_sleep_debt_decreases_when_drowsy(self):
        s = make_fly_state(melatonin=0.75, hunger=20.0)
        s["brain"]["circadian"] = {
            "melatonin": 0.75, "cortisol": 0.2, "body_temp_offset": 0.0,
            "wing_beat_hz": 0, "phase": "awake", "sleep_debt": 0.5,
        }
        s["kitchen"]["time_of_day"] = 0.1
        update_circadian(s)
        if s["brain"]["circadian"]["phase"] == "drowsy":
            assert s["brain"]["circadian"]["sleep_debt"] < 0.5

    def test_sleep_debt_increases_when_awake(self):
        s = make_fly_state(melatonin=0.1, cortisol=0.5, hunger=50.0)
        s["brain"]["circadian"] = {
            "melatonin": 0.1, "cortisol": 0.5, "body_temp_offset": 0.0,
            "wing_beat_hz": 0, "phase": "awake", "sleep_debt": 0.3,
        }
        s["kitchen"]["time_of_day"] = 0.5
        update_circadian(s)
        assert s["brain"]["circadian"]["sleep_debt"] > 0.3


class TestSleepBehavior:
    def test_sleeping_zeroes_velocity(self):
        s = make_fly_state(melatonin=0.92, hunger=50.0, fear_level=0.0)
        s["body"]["velocity"] = {"x": 5.0, "y": 3.0, "z": 1.0}
        s["kitchen"]["time_of_day"] = 0.05
        update_circadian(s)
        if s["brain"]["circadian"]["phase"] == "sleeping":
            v = s["body"]["velocity"]
            assert v["x"] == 0 and v["y"] == 0 and v["z"] == 0

    def test_sleeping_grounds_airborne_fly(self):
        s = make_fly_state(melatonin=0.92, hunger=50.0, fear_level=0.0, is_airborne=True)
        s["body"]["position"]["z"] = 50.0
        s["kitchen"]["time_of_day"] = 0.05
        update_circadian(s)
        if s["brain"]["circadian"]["phase"] == "sleeping":
            assert s["body"]["is_airborne"] is False
            assert s["body"]["position"]["z"] == 0

    def test_sleeping_sets_goal_sleep(self):
        s = make_fly_state(melatonin=0.92, hunger=50.0, fear_level=0.0)
        s["kitchen"]["time_of_day"] = 0.05
        update_circadian(s)
        if s["brain"]["circadian"]["phase"] == "sleeping":
            assert s["brain"]["current_goal"] == "sleep"
            assert s["brain"]["state"] == "resting"

    def test_drowsy_idle_adult_seeks_rest(self):
        s = make_fly_state(melatonin=0.75, hunger=20.0, current_goal="idle")
        s["kitchen"]["time_of_day"] = 0.1
        update_circadian(s)
        if s["brain"]["circadian"]["phase"] == "drowsy":
            assert s["brain"]["current_goal"] == "seek_rest"


class TestBodyTemperature:
    def test_offset_at_midnight(self):
        s = make_fly_state(time_of_day=0.0)
        update_circadian(s)
        expected = round(math.sin(0.0) * 1.5, 2)
        assert s["brain"]["circadian"]["body_temp_offset"] == expected

    def test_offset_at_noon(self):
        s = make_fly_state(time_of_day=0.5)
        update_circadian(s)
        expected = round(math.sin(math.pi) * 1.5, 2)
        assert abs(s["brain"]["circadian"]["body_temp_offset"] - expected) < 0.01

    def test_offset_at_quarter_day(self):
        s = make_fly_state(time_of_day=0.25)
        update_circadian(s)
        expected = round(math.sin(0.25 * math.pi * 2) * 1.5, 2)
        assert s["brain"]["circadian"]["body_temp_offset"] == expected

    def test_offset_bounded(self):
        for tod in [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]:
            s = make_fly_state(time_of_day=tod)
            update_circadian(s)
            off = s["brain"]["circadian"]["body_temp_offset"]
            assert -1.51 <= off <= 1.51


class TestWingBeat:
    def test_grounded_adult_zero_hz(self):
        s = make_fly_state(is_airborne=False, wing_condition=1.0)
        update_circadian(s)
        assert s["brain"]["circadian"]["wing_beat_hz"] == 0

    def test_airborne_adult_positive_hz(self):
        s = make_fly_state(is_airborne=True, wing_condition=1.0, cortisol=0.5)
        update_circadian(s)
        assert s["brain"]["circadian"]["wing_beat_hz"] > 0

    def test_damaged_wings_lower_hz(self):
        s1 = make_fly_state(is_airborne=True, wing_condition=1.0, cortisol=0.5, stress_level=0.0)
        s2 = make_fly_state(is_airborne=True, wing_condition=0.3, cortisol=0.5, stress_level=0.0)
        update_circadian(s1)
        update_circadian(s2)
        assert s2["brain"]["circadian"]["wing_beat_hz"] < s1["brain"]["circadian"]["wing_beat_hz"]

    def test_stress_increases_hz(self):
        s1 = make_fly_state(is_airborne=True, wing_condition=1.0, cortisol=0.5, stress_level=0.0)
        s2 = make_fly_state(is_airborne=True, wing_condition=1.0, cortisol=0.5, stress_level=0.8)
        update_circadian(s1)
        update_circadian(s2)
        assert s2["brain"]["circadian"]["wing_beat_hz"] > s1["brain"]["circadian"]["wing_beat_hz"]

    def test_non_adult_zero_hz(self):
        s = make_fly_state(stage="larva", is_airborne=False)
        update_circadian(s)
        assert s["brain"]["circadian"]["wing_beat_hz"] == 0

    def test_cortisol_increases_hz(self):
        s1 = make_fly_state(is_airborne=True, wing_condition=1.0, cortisol=0.1, stress_level=0.0)
        s2 = make_fly_state(is_airborne=True, wing_condition=1.0, cortisol=0.9, stress_level=0.0)
        update_circadian(s1)
        update_circadian(s2)
        assert s2["brain"]["circadian"]["wing_beat_hz"] > s1["brain"]["circadian"]["wing_beat_hz"]


class TestAging:
    def test_young_adult_senses_unchanged(self):
        smells = [{"type": "banana", "intensity": 1.0}]
        s = make_fly_state(stage_tick=10, adult_duration=100, smells=smells)
        update_circadian(s)
        assert s["senses"]["smell"][0]["intensity"] == 1.0

    def test_old_adult_senses_dulled(self):
        smells = [{"type": "banana", "intensity": 1.0}]
        s = make_fly_state(stage_tick=80, adult_duration=100, smells=smells)
        update_circadian(s)
        assert s["senses"]["smell"][0]["intensity"] < 1.0

    def test_very_old_adult_significant_dulling(self):
        """age_ratio=0.95: dull_factor = 1.0 - (0.95-0.5)*0.6 = 0.73"""
        smells = [{"type": "banana", "intensity": 1.0}]
        s = make_fly_state(stage_tick=95, adult_duration=100, smells=smells)
        update_circadian(s)
        assert s["senses"]["smell"][0]["intensity"] < 0.8
        assert s["senses"]["smell"][0]["intensity"] >= 0.7

    def test_multiple_smells_all_dulled(self):
        smells = [
            {"type": "banana", "intensity": 0.9},
            {"type": "trash", "intensity": 0.8},
        ]
        s = make_fly_state(stage_tick=75, adult_duration=100, smells=smells)
        update_circadian(s)
        for smell in s["senses"]["smell"]:
            assert smell["intensity"] < 0.9


class TestWingCondition:
    def test_flight_degrades_wings(self):
        s = make_fly_state(is_airborne=True, wing_condition=0.9)
        update_circadian(s)
        assert s["body"]["wing_condition"] < 0.9

    def test_grooming_repairs_wings(self):
        s = make_fly_state(is_airborne=False, wing_condition=0.7, current_goal="groom")
        update_circadian(s)
        assert s["body"]["wing_condition"] > 0.7

    def test_grooming_caps_at_1(self):
        s = make_fly_state(is_airborne=False, wing_condition=0.99, current_goal="groom")
        update_circadian(s)
        assert s["body"]["wing_condition"] <= 1.0

    def test_wing_floor_at_01(self):
        s = make_fly_state(is_airborne=True, wing_condition=0.11, wind=0.9)
        update_circadian(s)
        assert s["body"]["wing_condition"] >= 0.1

    def test_high_wind_damages_wings(self):
        s = make_fly_state(is_airborne=True, wing_condition=0.8, wind=0.8)
        update_circadian(s)
        assert s["body"]["wing_condition"] < 0.8

    def test_grounded_no_wind_damage(self):
        s = make_fly_state(is_airborne=False, wing_condition=0.8, wind=0.9)
        update_circadian(s)
        assert s["body"]["wing_condition"] >= 0.8

    def test_non_adult_wing_unchanged(self):
        s = make_fly_state(stage="larva", wing_condition=0.0)
        update_circadian(s)
        assert s["body"]["wing_condition"] == 0.0

    def test_repeated_flight_compounds(self):
        s = make_fly_state(is_airborne=True, wing_condition=1.0)
        random.seed(42)
        for _ in range(50):
            update_circadian(s)
        assert s["body"]["wing_condition"] < 0.95


class TestBacterialLoad:
    def test_immune_reduces_bacteria(self):
        s = make_fly_state(bacterial_load=0.5)
        update_circadian(s)
        assert s["body"]["bacterial_load"] < 0.5

    def test_bacteria_never_negative(self):
        s = make_fly_state(bacterial_load=0.002)
        update_circadian(s)
        assert s["body"]["bacterial_load"] >= 0.0

    def test_high_bacteria_drains_energy(self):
        s = make_fly_state(bacterial_load=0.7, energy=80.0)
        update_circadian(s)
        assert s["energy"]["current"] < 80.0

    def test_low_bacteria_no_energy_drain(self):
        s = make_fly_state(bacterial_load=0.3, energy=80.0)
        update_circadian(s)
        assert s["energy"]["current"] == 80.0

    def test_ambient_pickup_near_decayed_food(self):
        food = {"type": "food", "x": 105.0, "y": 105.0, "decay": 0.9}
        s = make_fly_state(bacterial_load=0.1, kitchen_objects=[food])
        update_circadian(s)
        assert s["body"]["bacterial_load"] > 0.1

    def test_no_pickup_far_from_food(self):
        food = {"type": "food", "x": 500.0, "y": 500.0, "decay": 0.9}
        s = make_fly_state(bacterial_load=0.1, kitchen_objects=[food])
        update_circadian(s)
        assert s["body"]["bacterial_load"] < 0.1

    def test_no_pickup_from_fresh_food(self):
        food = {"type": "food", "x": 105.0, "y": 105.0, "decay": 0.3}
        s = make_fly_state(bacterial_load=0.1, kitchen_objects=[food])
        update_circadian(s)
        assert s["body"]["bacterial_load"] < 0.1

    def test_larva_picks_up_bacteria(self):
        food = {"type": "food", "x": 105.0, "y": 105.0, "decay": 0.9}
        s = make_fly_state(stage="larva", bacterial_load=0.1, kitchen_objects=[food])
        update_circadian(s)
        assert s["body"]["bacterial_load"] >= 0.0

    def test_bacteria_capped_at_1(self):
        food = {"type": "food", "x": 101.0, "y": 101.0, "decay": 0.99}
        s = make_fly_state(bacterial_load=0.99, kitchen_objects=[food])
        update_circadian(s)
        assert s["body"]["bacterial_load"] <= 1.0


class TestLethalInfection:
    def test_lethal_triggers_death(self):
        s = make_fly_state(bacterial_load=0.96)
        update_circadian(s)
        assert s["lifecycle"]["stage"] == "death"
        assert s["_meta"]["cause_of_death"] == "infection"

    def test_lethal_records_history(self):
        s = make_fly_state(bacterial_load=0.96)
        update_circadian(s)
        assert any("infection" in h["event"] for h in s["history"])

    def test_sub_lethal_survives(self):
        s = make_fly_state(bacterial_load=0.944)
        update_circadian(s)
        assert s["lifecycle"]["stage"] == "adult"

    def test_non_adult_immune(self):
        s = make_fly_state(stage="larva", bacterial_load=0.99)
        update_circadian(s)
        assert s["lifecycle"]["stage"] == "larva"

    def test_infection_sets_brain_dead(self):
        s = make_fly_state(bacterial_load=0.96)
        update_circadian(s)
        assert s["brain"]["state"] == "dead"


class TestDangerMemory:
    def test_recent_danger_kept(self):
        dz = [{"x": 50, "y": 50, "tick": 90}]
        s = make_fly_state(total_ticks=100, danger_zones=dz)
        update_circadian(s)
        assert len(s["memory"]["danger_zones"]) == 1

    def test_old_danger_pruned(self):
        dz = [{"x": 50, "y": 50, "tick": 50}]
        s = make_fly_state(total_ticks=100, danger_zones=dz)
        update_circadian(s)
        assert len(s["memory"]["danger_zones"]) == 0

    def test_mixed_ages(self):
        dz = [
            {"x": 10, "y": 10, "tick": 55},
            {"x": 20, "y": 20, "tick": 80},
            {"x": 30, "y": 30, "tick": 61},
        ]
        s = make_fly_state(total_ticks=100, danger_zones=dz)
        update_circadian(s)
        assert len(s["memory"]["danger_zones"]) == 2

    def test_empty_danger_zones(self):
        s = make_fly_state(danger_zones=[])
        update_circadian(s)
        assert s["memory"]["danger_zones"] == []

    def test_boundary_40_ticks_pruned(self):
        dz = [{"x": 10, "y": 10, "tick": 60}]
        s = make_fly_state(total_ticks=100, danger_zones=dz)
        update_circadian(s)
        assert len(s["memory"]["danger_zones"]) == 0


class TestWeather:
    def test_rain_lowers_temp(self):
        s = make_fly_state()
        s["kitchen"]["weather"] = "rain"
        s["kitchen"]["ambient_temp"] = 22.0
        random.seed(999)
        update_circadian(s)
        assert s["kitchen"]["ambient_temp"] < 22.0

    def test_hot_raises_temp(self):
        s = make_fly_state()
        s["kitchen"]["weather"] = "hot"
        s["kitchen"]["ambient_temp"] = 22.0
        random.seed(999)
        update_circadian(s)
        assert s["kitchen"]["ambient_temp"] > 22.0

    def test_clear_no_temp_change(self):
        s = make_fly_state()
        s["kitchen"]["weather"] = "clear"
        s["kitchen"]["ambient_temp"] = 22.0
        random.seed(999)
        update_circadian(s)
        assert s["kitchen"]["ambient_temp"] == 22.0


class TestPropertyInvariants:
    def test_100_ticks_no_crash(self):
        s = make_fly_state(stage="adult", time_of_day=0.0, wing_condition=1.0, bacterial_load=0.0)
        random.seed(42)
        for _ in range(100):
            s["kitchen"]["time_of_day"] = (s["kitchen"]["time_of_day"] + 0.009) % 1.0
            s["lifecycle"]["total_ticks"] += 1
            s["lifecycle"]["stage_tick"] += 1
            update_circadian(s)

    def test_hormones_bounded_100_ticks(self):
        s = make_fly_state(stage="adult", time_of_day=0.0)
        random.seed(42)
        for _ in range(100):
            s["kitchen"]["time_of_day"] = (s["kitchen"]["time_of_day"] + 0.009) % 1.0
            s["lifecycle"]["total_ticks"] += 1
            update_circadian(s)
            circ = s["brain"]["circadian"]
            assert 0.0 <= circ["melatonin"] <= 1.0
            assert 0.0 <= circ["cortisol"] <= 1.0

    def test_wing_bounded_100_ticks(self):
        s = make_fly_state(stage="adult", is_airborne=True, wing_condition=1.0, wind=0.3)
        random.seed(42)
        for _ in range(100):
            s["kitchen"]["time_of_day"] = (s["kitchen"]["time_of_day"] + 0.009) % 1.0
            s["lifecycle"]["total_ticks"] += 1
            update_circadian(s)
            assert 0.1 <= s["body"]["wing_condition"] <= 1.0

    def test_bacteria_bounded_100_ticks(self):
        food = {"type": "food", "x": 105.0, "y": 105.0, "decay": 0.9}
        s = make_fly_state(stage="adult", bacterial_load=0.0, kitchen_objects=[food])
        random.seed(42)
        alive = 0
        for _ in range(100):
            if s["lifecycle"]["stage"] == "death":
                break
            s["kitchen"]["time_of_day"] = (s["kitchen"]["time_of_day"] + 0.009) % 1.0
            s["lifecycle"]["total_ticks"] += 1
            update_circadian(s)
            assert 0.0 <= s["body"]["bacterial_load"] <= 1.0
            alive += 1
        assert alive > 0

    def test_sleep_debt_bounded(self):
        s = make_fly_state(stage="adult", time_of_day=0.0)
        random.seed(42)
        for _ in range(100):
            s["kitchen"]["time_of_day"] = (s["kitchen"]["time_of_day"] + 0.009) % 1.0
            s["lifecycle"]["total_ticks"] += 1
            update_circadian(s)
            assert 0.0 <= s["brain"]["circadian"]["sleep_debt"] <= 1.0

    def test_body_temp_bounded(self):
        s = make_fly_state(stage="adult", time_of_day=0.0)
        random.seed(42)
        for _ in range(100):
            s["kitchen"]["time_of_day"] = (s["kitchen"]["time_of_day"] + 0.009) % 1.0
            update_circadian(s)
            assert -1.6 <= s["brain"]["circadian"]["body_temp_offset"] <= 1.6

    def test_full_lifecycle_stages(self):
        random.seed(42)
        for stage in ["egg", "larva", "pupa", "adult"]:
            s = make_fly_state(stage=stage, time_of_day=0.5)
            update_circadian(s)
            assert s["brain"]["circadian"]["phase"] in ("dormant", "awake", "drowsy", "sleeping")
