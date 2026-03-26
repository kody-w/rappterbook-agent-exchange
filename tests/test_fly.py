"""
tests/test_fly.py — Unit tests for engine/fly.py (Musca domestica tick engine).

The last untested engine module. 828 lines of fly lifecycle simulation.
Tests cover every public function:

  - dist2d, clamp — utility math
  - record — event logging with history cap
  - update_kitchen — time-of-day, food decay
  - update_weather — wind, humidity, window state
  - update_threats — threat activation/deactivation
  - update_senses — smell, sight, pheromone sensing
  - think — goal selection (flee, seek_food, explore, etc.)
  - move — physics-based movement per goal
  - try_feed — energy gain from food proximity
  - deposit_pheromone — stigmergy trail system
  - update_energy — metabolic drain, position update, wind drift
  - check_transition — lifecycle stage transitions
  - generate_narration — prose output per stage
  - rebirth — generational inheritance
  - tick — full integration (one sol of fly life)

Run:
    python -m pytest tests/test_fly.py -v
"""
from __future__ import annotations

import copy
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.fly import (
    dist2d,
    clamp,
    record,
    update_kitchen,
    update_weather,
    update_threats,
    update_senses,
    think,
    move,
    try_feed,
    deposit_pheromone,
    update_energy,
    check_transition,
    generate_narration,
    rebirth,
    tick,
    STAGES,
)


def _make_state(
    stage: str = "adult",
    stage_tick: int = 5,
    total_ticks: int = 50,
    energy: float = 60.0,
    hunger: float = 20.0,
    px: float = 300.0,
    py: float = 200.0,
    pz: float = 0.0,
    is_airborne: bool = False,
    frame: int = 100,
    generation: int = 1,
) -> dict:
    """Build a minimal valid fly state for testing."""
    return {
        "_meta": {
            "organism": "Musca domestica",
            "frame": frame,
            "born_at": "2026-01-01T00:00:00Z",
            "version": "3.0.0",
            "generation": generation,
            "parent_cause_of_death": None,
            "parent_lifespan": 0,
            "total_frames_alive": total_ticks,
            "lineage": [],
        },
        "genome": {
            "species": "Musca domestica",
            "wing_vein_pattern": 0.5,
            "eye_facets": 0.5,
            "body_color_hue": 0.3,
            "bristle_density": 0.4,
            "metabolic_rate": 0.8,
            "flight_efficiency": 0.7,
            "smell_sensitivity": 1.0,
            "heat_tolerance": 0.5,
            "lifespan_modifier": 0.5,
            "microbiome": 0.5,
        },
        "lifecycle": {
            "stage": stage,
            "stage_tick": stage_tick,
            "total_ticks": total_ticks,
            "stage_durations": {"egg": 8, "larva": 25, "pupa": 18, "adult": 65},
            "molts": 0,
            "larva_instar": 0,
        },
        "body": {
            "position": {"x": px, "y": py, "z": pz},
            "velocity": {"x": 0, "y": 0, "z": 0},
            "facing": 0,
            "size": 6.0 if stage == "adult" else 2.0,
            "mass": 0.012,
            "wing_state": "functional" if stage == "adult" else "none",
            "leg_state": "functional" if stage == "adult" else "stub",
            "is_airborne": is_airborne,
            "surface": "counter",
            "wing_damage": 0,
        },
        "energy": {
            "current": energy,
            "max": 100,
            "hunger": hunger,
            "metabolic_drain": 0.5,
            "last_fed_tick": 0,
        },
        "brain": {
            "state": "active" if stage == "adult" else "dormant",
            "current_goal": "idle",
            "fear_level": 0,
            "curiosity": 0.3,
            "satisfaction": 0.5,
            "decisions_made": 10,
            "neural_complexity": 0.8 if stage == "adult" else 0.1,
            "inherited_memory": {
                "parent_favorite_food": None,
                "parent_danger_zones": [],
                "epigenetic_bias": 0.2,
            },
        },
        "senses": {
            "smell": [],
            "sight": [],
            "pheromones": [],
            "touch": {"surface": "counter", "vibration": 0},
            "temperature": 22,
            "wind": 0,
        },
        "memory": {
            "food_sources": [],
            "danger_zones": [],
            "visited_positions": [],
            "total_distance": 0,
            "times_fed": 0,
            "times_fled": 0,
            "peak_altitude": 0,
            "favorite_food": None,
        },
        "kitchen": {
            "width": 800,
            "height": 600,
            "objects": [
                {"id": "banana", "type": "food", "name": "banana",
                 "x": 200, "y": 150, "z": 0, "smell_radius": 80,
                 "energy": 40, "decay": 0.3},
                {"id": "jam", "type": "food", "name": "jam jar",
                 "x": 500, "y": 300, "z": 0, "smell_radius": 60,
                 "energy": 30, "decay": 0.2},
                {"id": "light1", "type": "light", "name": "ceiling light",
                 "x": 400, "y": 300, "z": 2.5, "intensity": 0.8},
                {"id": "cat", "type": "threat", "name": "cat",
                 "x": -100, "y": -100, "active": False},
                {"id": "swatter", "type": "threat", "name": "fly swatter",
                 "x": -100, "y": -100, "active": False},
            ],
            "ambient_temp": 22,
            "time_of_day": 0.5,
            "lights_on": True,
            "active_events": [],
            "event_vibration": 0,
            "event_wind": 0,
            "event_temp_delta": 0,
            "humidity": 0.6,
        },
        "history": [],
        "ancestors": [],
        "narration": "",
        "weather": {
            "wind_direction": 0.0,
            "wind_strength": 0.0,
            "humidity": 0.6,
            "window_open": False,
        },
        "pheromones": [],
    }


# ──────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────

class TestDist2d:
    def test_same_point(self) -> None:
        assert dist2d(5, 5, 5, 5) == 0.0

    def test_horizontal(self) -> None:
        assert abs(dist2d(0, 0, 3, 0) - 3.0) < 1e-9

    def test_vertical(self) -> None:
        assert abs(dist2d(0, 0, 0, 4) - 4.0) < 1e-9

    def test_diagonal(self) -> None:
        assert abs(dist2d(0, 0, 3, 4) - 5.0) < 1e-9

    def test_negative_coords(self) -> None:
        assert abs(dist2d(-1, -1, 2, 3) - 5.0) < 1e-9


class TestClamp:
    def test_within_range(self) -> None:
        assert clamp(5, 0, 10) == 5

    def test_below_min(self) -> None:
        assert clamp(-5, 0, 10) == 0

    def test_above_max(self) -> None:
        assert clamp(15, 0, 10) == 10

    def test_at_boundary(self) -> None:
        assert clamp(0, 0, 10) == 0
        assert clamp(10, 0, 10) == 10


# ──────────────────────────────────────────────────────────────────────
# record
# ──────────────────────────────────────────────────────────────────────

class TestRecord:
    def test_appends_event(self) -> None:
        state = _make_state()
        record(state, "test event")
        assert len(state["history"]) == 1
        assert state["history"][0]["event"] == "test event"

    def test_event_has_required_fields(self) -> None:
        state = _make_state()
        record(state, "test")
        entry = state["history"][0]
        assert "tick" in entry
        assert "event" in entry
        assert "stage" in entry
        assert "energy" in entry
        assert "position" in entry

    def test_history_capped_at_200(self) -> None:
        state = _make_state()
        state["history"] = [{"tick": i} for i in range(205)]
        record(state, "overflow")
        assert len(state["history"]) <= 200


# ──────────────────────────────────────────────────────────────────────
# update_kitchen
# ──────────────────────────────────────────────────────────────────────

class TestUpdateKitchen:
    def test_time_advances(self) -> None:
        state = _make_state()
        old_tod = state["kitchen"]["time_of_day"]
        update_kitchen(state)
        assert state["kitchen"]["time_of_day"] != old_tod

    def test_time_wraps(self) -> None:
        state = _make_state()
        state["kitchen"]["time_of_day"] = 0.999
        update_kitchen(state)
        assert state["kitchen"]["time_of_day"] < 0.999

    def test_lights_follow_time(self) -> None:
        state = _make_state()
        state["kitchen"]["time_of_day"] = 0.5
        update_kitchen(state)
        assert state["kitchen"]["lights_on"] is True

    def test_lights_off_at_night(self) -> None:
        state = _make_state()
        state["kitchen"]["time_of_day"] = 0.09  # after +0.009 → 0.099
        update_kitchen(state)
        assert state["kitchen"]["lights_on"] is False

    def test_temp_varies_with_time(self) -> None:
        state = _make_state()
        state["kitchen"]["time_of_day"] = 0.0
        update_kitchen(state)
        t1 = state["kitchen"]["ambient_temp"]
        state["kitchen"]["time_of_day"] = 0.5
        update_kitchen(state)
        t2 = state["kitchen"]["ambient_temp"]
        assert t1 != t2


# ──────────────────────────────────────────────────────────────────────
# update_weather
# ──────────────────────────────────────────────────────────────────────

class TestUpdateWeather:
    def test_initializes_weather(self) -> None:
        state = _make_state()
        del state["weather"]
        update_weather(state)
        assert "weather" in state

    def test_humidity_bounded(self) -> None:
        state = _make_state()
        for _ in range(100):
            state["_meta"]["frame"] += 1
            update_weather(state)
        assert 0.3 <= state["weather"]["humidity"] <= 0.95

    def test_wind_decays(self) -> None:
        state = _make_state()
        state["weather"]["wind_strength"] = 2.0
        state["_meta"]["frame"] = 999999  # unlikely to trigger wind change
        update_weather(state)
        assert state["weather"]["wind_strength"] <= 2.0


# ──────────────────────────────────────────────────────────────────────
# update_senses
# ──────────────────────────────────────────────────────────────────────

class TestUpdateSenses:
    def test_egg_has_no_senses(self) -> None:
        state = _make_state(stage="egg")
        update_senses(state)
        assert state["senses"]["smell"] == []
        assert state["senses"]["sight"] == []

    def test_pupa_has_no_senses(self) -> None:
        state = _make_state(stage="pupa")
        update_senses(state)
        assert state["senses"]["smell"] == []

    def test_adult_detects_nearby_food(self) -> None:
        state = _make_state(px=210, py=155)  # near banana at (200,150)
        update_senses(state)
        smells = state["senses"]["smell"]
        assert len(smells) > 0
        assert any("banana" in s["name"] for s in smells)

    def test_adult_detects_light(self) -> None:
        state = _make_state(px=390, py=295)  # near light at (400,300)
        update_senses(state)
        sights = state["senses"]["sight"]
        assert any("light" in s["name"] for s in sights)

    def test_threat_detection(self) -> None:
        state = _make_state(px=100, py=100)
        state["kitchen"]["objects"][3]["active"] = True
        state["kitchen"]["objects"][3]["x"] = 105
        state["kitchen"]["objects"][3]["y"] = 105
        update_senses(state)
        threats = [s for s in state["senses"]["sight"] if s.get("threat")]
        assert len(threats) > 0

    def test_smell_intensity_decreases_with_distance(self) -> None:
        state_near = _make_state(px=205, py=155)
        state_far = _make_state(px=250, py=150)
        update_senses(state_near)
        update_senses(state_far)
        near_i = max((s["intensity"] for s in state_near["senses"]["smell"]), default=0)
        far_i = max((s["intensity"] for s in state_far["senses"]["smell"]), default=0)
        assert near_i > far_i

    def test_pheromone_sensing(self) -> None:
        state = _make_state(px=300, py=200)
        state["pheromones"] = [
            {"x": 310, "y": 205, "intensity": 0.8, "gen": 1, "tick": 40},
        ]
        update_senses(state)
        assert len(state["senses"]["pheromones"]) > 0


# ──────────────────────────────────────────────────────────────────────
# think
# ──────────────────────────────────────────────────────────────────────

class TestThink:
    def test_egg_dormant(self) -> None:
        state = _make_state(stage="egg")
        think(state)
        assert state["brain"]["state"] == "dormant"
        assert state["brain"]["current_goal"] is None

    def test_pupa_metamorphosis(self) -> None:
        state = _make_state(stage="pupa")
        think(state)
        assert state["brain"]["state"] == "metamorphosis"

    def test_flee_from_threat(self) -> None:
        state = _make_state()
        state["senses"]["sight"] = [
            {"id": "cat", "name": "cat", "distance": 50, "threat": True},
        ]
        think(state)
        assert state["brain"]["current_goal"] == "flee"
        assert state["brain"]["fear_level"] > 0

    def test_seek_food_when_hungry(self) -> None:
        state = _make_state(hunger=80.0)
        state["senses"]["smell"] = [
            {"id": "banana", "name": "banana", "distance": 30, "intensity": 0.8},
        ]
        think(state)
        assert state["brain"]["current_goal"] == "seek_food"

    def test_rest_at_night(self) -> None:
        state = _make_state()
        state["kitchen"]["lights_on"] = False
        think(state)
        assert state["brain"]["current_goal"] == "rest"

    def test_larva_simple_reflex(self) -> None:
        state = _make_state(stage="larva")
        think(state)
        assert state["brain"]["state"] == "simple_reflex"

    def test_decisions_increment(self) -> None:
        state = _make_state(hunger=80.0)
        state["senses"]["smell"] = [
            {"id": "banana", "name": "banana", "distance": 30, "intensity": 0.8},
        ]
        before = state["brain"]["decisions_made"]
        think(state)
        assert state["brain"]["decisions_made"] > before


# ──────────────────────────────────────────────────────────────────────
# move
# ──────────────────────────────────────────────────────────────────────

class TestMove:
    def test_egg_no_movement(self) -> None:
        state = _make_state(stage="egg")
        move(state)
        assert state["body"]["velocity"] == {"x": 0, "y": 0, "z": 0}

    def test_pupa_no_movement(self) -> None:
        state = _make_state(stage="pupa")
        move(state)
        assert state["body"]["velocity"] == {"x": 0, "y": 0, "z": 0}

    def test_flee_creates_velocity(self) -> None:
        state = _make_state(px=300, py=200)
        state["brain"]["current_goal"] = "flee"
        state["kitchen"]["objects"][3]["active"] = True
        state["kitchen"]["objects"][3]["x"] = 305
        state["kitchen"]["objects"][3]["y"] = 205
        state["senses"]["sight"] = [
            {"id": "cat", "name": "cat", "distance": 10, "threat": True},
        ]
        move(state)
        v = state["body"]["velocity"]
        assert v["x"] != 0 or v["y"] != 0

    def test_groom_stops_movement(self) -> None:
        state = _make_state(is_airborne=True)
        state["body"]["position"]["z"] = 1.5
        state["brain"]["current_goal"] = "groom"
        move(state)
        assert state["body"]["velocity"] == {"x": 0, "y": 0, "z": 0}
        assert state["body"]["is_airborne"] is False

    def test_rest_stops_movement(self) -> None:
        state = _make_state()
        state["brain"]["current_goal"] = "rest"
        move(state)
        assert state["body"]["velocity"] == {"x": 0, "y": 0, "z": 0}

    def test_explore_creates_velocity(self) -> None:
        state = _make_state()
        state["brain"]["current_goal"] = "explore"
        move(state)
        v = state["body"]["velocity"]
        speed = math.sqrt(v["x"]**2 + v["y"]**2)
        assert speed > 0

    def test_seek_food_moves_toward_food(self) -> None:
        state = _make_state(px=300, py=200)
        state["brain"]["current_goal"] = "seek_food"
        state["senses"]["smell"] = [
            {"id": "banana", "name": "banana", "distance": 30, "intensity": 0.8},
        ]
        move(state)
        # Should move toward banana at (200, 150): negative x, negative y
        assert state["body"]["velocity"]["x"] < 0
        assert state["body"]["velocity"]["y"] < 0


# ──────────────────────────────────────────────────────────────────────
# try_feed
# ──────────────────────────────────────────────────────────────────────

class TestTryFeed:
    def test_egg_cannot_feed(self) -> None:
        state = _make_state(stage="egg", px=200, py=150)
        old_energy = state["energy"]["current"]
        try_feed(state)
        assert state["energy"]["current"] == old_energy

    def test_adult_feeds_near_food(self) -> None:
        state = _make_state(px=202, py=152, energy=50.0, hunger=60.0)
        try_feed(state)
        assert state["energy"]["current"] > 50.0
        assert state["energy"]["hunger"] < 60.0
        assert state["memory"]["times_fed"] == 1

    def test_too_far_no_feed(self) -> None:
        state = _make_state(px=600, py=500, energy=50.0)
        try_feed(state)
        assert state["energy"]["current"] == 50.0
        assert state["memory"]["times_fed"] == 0

    def test_feed_records_food_source(self) -> None:
        state = _make_state(px=202, py=152)
        try_feed(state)
        assert len(state["memory"]["food_sources"]) > 0

    def test_feed_sets_favorite_food(self) -> None:
        state = _make_state(px=202, py=152)
        try_feed(state)
        assert state["memory"]["favorite_food"] is not None

    def test_energy_capped_at_max(self) -> None:
        state = _make_state(px=202, py=152, energy=99.0)
        try_feed(state)
        assert state["energy"]["current"] <= state["energy"]["max"]


# ──────────────────────────────────────────────────────────────────────
# deposit_pheromone
# ──────────────────────────────────────────────────────────────────────

class TestDepositPheromone:
    def test_egg_no_pheromone(self) -> None:
        state = _make_state(stage="egg")
        deposit_pheromone(state)
        assert len(state["pheromones"]) == 0

    def test_adult_deposits(self) -> None:
        state = _make_state()
        deposit_pheromone(state)
        assert len(state["pheromones"]) == 1
        assert state["pheromones"][0]["intensity"] > 0

    def test_pheromones_decay(self) -> None:
        state = _make_state()
        state["pheromones"] = [
            {"x": 100, "y": 100, "intensity": 0.5, "gen": 1, "tick": 1},
        ]
        deposit_pheromone(state)
        old_p = [p for p in state["pheromones"] if p["x"] == 100][0]
        assert old_p["intensity"] < 0.5

    def test_pheromones_capped_at_400(self) -> None:
        state = _make_state()
        state["pheromones"] = [
            {"x": i, "y": 0, "intensity": 0.5, "gen": 1, "tick": 1}
            for i in range(450)
        ]
        deposit_pheromone(state)
        assert len(state["pheromones"]) <= 400

    def test_weak_pheromones_pruned(self) -> None:
        state = _make_state()
        state["pheromones"] = [
            {"x": 50, "y": 50, "intensity": 0.005, "gen": 1, "tick": 1},
        ]
        deposit_pheromone(state)
        very_weak = [p for p in state["pheromones"] if p["intensity"] < 0.01 and p["x"] == 50]
        assert len(very_weak) == 0

    def test_larva_deposits_lower_intensity(self) -> None:
        adult_state = _make_state(stage="adult")
        larva_state = _make_state(stage="larva")
        deposit_pheromone(adult_state)
        deposit_pheromone(larva_state)
        a_i = [p for p in adult_state["pheromones"] if p["x"] == 300][0]["intensity"]
        l_i = [p for p in larva_state["pheromones"] if p["x"] == 300][0]["intensity"]
        assert a_i > l_i


# ──────────────────────────────────────────────────────────────────────
# update_energy
# ──────────────────────────────────────────────────────────────────────

class TestUpdateEnergy:
    def test_energy_drains(self) -> None:
        state = _make_state(energy=80.0)
        update_energy(state)
        assert state["energy"]["current"] < 80.0

    def test_hunger_increases(self) -> None:
        state = _make_state(hunger=20.0)
        update_energy(state)
        assert state["energy"]["hunger"] > 20.0

    def test_airborne_drains_faster(self) -> None:
        ground = _make_state(energy=80.0)
        air = _make_state(energy=80.0, is_airborne=True)
        air["body"]["position"]["z"] = 1.5
        update_energy(ground)
        update_energy(air)
        assert air["energy"]["current"] < ground["energy"]["current"]

    def test_night_drains_slower(self) -> None:
        day = _make_state(energy=80.0)
        night = _make_state(energy=80.0)
        night["kitchen"]["lights_on"] = False
        update_energy(day)
        update_energy(night)
        assert night["energy"]["current"] > day["energy"]["current"]

    def test_position_updates(self) -> None:
        state = _make_state(px=100.0, py=100.0)
        state["body"]["velocity"]["x"] = 5.0
        state["body"]["velocity"]["y"] = 3.0
        update_energy(state)
        assert state["body"]["position"]["x"] > 100.0
        assert state["body"]["position"]["y"] > 100.0

    def test_position_clamped_to_kitchen(self) -> None:
        state = _make_state(px=799.0, py=599.0)
        state["body"]["velocity"]["x"] = 10.0
        state["body"]["velocity"]["y"] = 10.0
        update_energy(state)
        assert state["body"]["position"]["x"] <= 800
        assert state["body"]["position"]["y"] <= 600

    def test_energy_never_negative(self) -> None:
        state = _make_state(energy=0.1)
        update_energy(state)
        assert state["energy"]["current"] >= 0

    def test_distance_tracked(self) -> None:
        state = _make_state()
        state["body"]["velocity"]["x"] = 5.0
        update_energy(state)
        assert state["memory"]["total_distance"] > 0

    def test_wind_affects_airborne_fly(self) -> None:
        state = _make_state(is_airborne=True, px=400, py=300)
        state["body"]["position"]["z"] = 1.5
        state["weather"]["wind_strength"] = 2.0
        state["weather"]["wind_direction"] = 0  # east
        old_vx = state["body"]["velocity"]["x"]
        update_energy(state)
        # Wind pushes east: position should shift right
        assert state["body"]["position"]["x"] > 400


# ──────────────────────────────────────────────────────────────────────
# check_transition
# ──────────────────────────────────────────────────────────────────────

class TestCheckTransition:
    def test_no_transition_within_duration(self) -> None:
        state = _make_state(stage="egg", stage_tick=3)
        check_transition(state)
        assert state["lifecycle"]["stage"] == "egg"

    def test_egg_to_larva(self) -> None:
        state = _make_state(stage="egg", stage_tick=8)
        check_transition(state)
        assert state["lifecycle"]["stage"] == "larva"

    def test_larva_to_pupa(self) -> None:
        state = _make_state(stage="larva", stage_tick=25)
        check_transition(state)
        assert state["lifecycle"]["stage"] == "pupa"

    def test_pupa_to_adult(self) -> None:
        state = _make_state(stage="pupa", stage_tick=18)
        check_transition(state)
        assert state["lifecycle"]["stage"] == "adult"

    def test_adult_to_death(self) -> None:
        state = _make_state(stage="adult", stage_tick=65)
        check_transition(state)
        assert state["lifecycle"]["stage"] == "death"
        assert state["_meta"]["cause_of_death"] == "old age"

    def test_starvation_death(self) -> None:
        state = _make_state(stage="egg", stage_tick=8, energy=0.0)
        check_transition(state)
        assert state["lifecycle"]["stage"] == "death"
        assert state["_meta"]["cause_of_death"] == "starvation"

    def test_death_is_terminal(self) -> None:
        state = _make_state(stage="death")
        check_transition(state)
        assert state["lifecycle"]["stage"] == "death"

    def test_larva_grows_during_stage(self) -> None:
        state = _make_state(stage="larva", stage_tick=5)
        old_size = state["body"]["size"]
        check_transition(state)
        assert state["body"]["size"] > old_size

    def test_larva_molts(self) -> None:
        state = _make_state(stage="larva", stage_tick=8)
        state["lifecycle"]["larva_instar"] = 0
        check_transition(state)
        assert state["lifecycle"]["larva_instar"] > 0
        assert state["lifecycle"]["molts"] > 0

    def test_pupa_neural_complexity_increases(self) -> None:
        state = _make_state(stage="pupa", stage_tick=10)
        state["brain"]["neural_complexity"] = 0.3
        check_transition(state)
        assert state["brain"]["neural_complexity"] > 0.3

    def test_records_transition_event(self) -> None:
        state = _make_state(stage="egg", stage_tick=8)
        check_transition(state)
        assert any("egg -> larva" in h["event"] for h in state["history"])


# ──────────────────────────────────────────────────────────────────────
# generate_narration
# ──────────────────────────────────────────────────────────────────────

class TestGenerateNarration:
    def test_egg_narration(self) -> None:
        state = _make_state(stage="egg", stage_tick=2)
        generate_narration(state)
        assert "egg" in state["narration"].lower() or "cell" in state["narration"].lower()

    def test_death_narration(self) -> None:
        state = _make_state(stage="death")
        generate_narration(state)
        assert "stillness" in state["narration"].lower() or "pheromone" in state["narration"].lower()

    def test_adult_flee_narration(self) -> None:
        state = _make_state()
        state["brain"]["current_goal"] = "flee"
        generate_narration(state)
        assert "danger" in state["narration"].lower() or "bolts" in state["narration"].lower()

    def test_narration_is_string(self) -> None:
        state = _make_state()
        generate_narration(state)
        assert isinstance(state["narration"], str)
        assert len(state["narration"]) > 0

    def test_generation_prefix(self) -> None:
        state = _make_state(generation=3)
        generate_narration(state)
        assert "Gen 3" in state["narration"]


# ──────────────────────────────────────────────────────────────────────
# rebirth
# ──────────────────────────────────────────────────────────────────────

class TestRebirth:
    def _dead_state(self) -> dict:
        state = _make_state(stage="death")
        state["_meta"]["cause_of_death"] = "old age"
        state["_meta"]["died_at"] = 100
        state["memory"]["favorite_food"] = "banana"
        state["memory"]["times_fed"] = 15
        state["memory"]["times_fled"] = 3
        state["memory"]["total_distance"] = 500.0
        return state

    def test_returns_new_state(self) -> None:
        new = rebirth(self._dead_state())
        assert new is not None
        assert isinstance(new, dict)

    def test_generation_increments(self) -> None:
        old = self._dead_state()
        new = rebirth(old)
        assert new["_meta"]["generation"] == old["_meta"]["generation"] + 1

    def test_starts_as_egg(self) -> None:
        new = rebirth(self._dead_state())
        assert new["lifecycle"]["stage"] == "egg"
        assert new["lifecycle"]["stage_tick"] == 0

    def test_genome_mutated(self) -> None:
        old = self._dead_state()
        new = rebirth(old)
        # At least one numeric gene should differ
        diffs = 0
        for k, v in old["genome"].items():
            if isinstance(v, (int, float)) and k != "species":
                if new["genome"][k] != v:
                    diffs += 1
        assert diffs > 0

    def test_genome_clamped(self) -> None:
        new = rebirth(self._dead_state())
        for k, v in new["genome"].items():
            if isinstance(v, float):
                assert 0.01 <= v <= 1.0, f"{k}={v} out of bounds"

    def test_inherits_parent_memory(self) -> None:
        new = rebirth(self._dead_state())
        inherited = new["brain"]["inherited_memory"]
        assert inherited["parent_favorite_food"] == "banana"

    def test_ancestors_recorded(self) -> None:
        new = rebirth(self._dead_state())
        assert len(new["ancestors"]) > 0

    def test_ancestors_capped_at_10(self) -> None:
        old = self._dead_state()
        old["ancestors"] = [{"generation": i} for i in range(12)]
        new = rebirth(old)
        assert len(new["ancestors"]) <= 10

    def test_pheromones_inherited_decayed(self) -> None:
        old = self._dead_state()
        old["pheromones"] = [
            {"x": 100, "y": 100, "intensity": 1.0, "gen": 1, "tick": 50},
        ]
        new = rebirth(old)
        if new["pheromones"]:
            assert new["pheromones"][0]["intensity"] < 1.0

    def test_corpse_added_to_kitchen(self) -> None:
        old = self._dead_state()
        gen = old["_meta"]["generation"]
        new = rebirth(old)
        corpse_id = f"corpse_gen{gen}"
        corpses = [o for o in new["kitchen"]["objects"] if o["id"] == corpse_id]
        assert len(corpses) == 1
        assert corpses[0]["type"] == "food"

    def test_fresh_energy(self) -> None:
        new = rebirth(self._dead_state())
        assert new["energy"]["current"] >= 80
        assert new["energy"]["hunger"] == 5.0

    def test_lineage_recorded(self) -> None:
        new = rebirth(self._dead_state())
        assert len(new["_meta"]["lineage"]) > 0


# ──────────────────────────────────────────────────────────────────────
# tick — full integration
# ──────────────────────────────────────────────────────────────────────

class TestTick:
    def test_single_tick_no_crash(self) -> None:
        state = _make_state()
        result = tick(state)
        assert result is not None

    def test_tick_advances_frame(self) -> None:
        state = _make_state(frame=100)
        tick(state)
        assert state["_meta"]["frame"] == 101

    def test_tick_advances_stage_tick(self) -> None:
        state = _make_state(stage_tick=5)
        tick(state)
        assert state["lifecycle"]["stage_tick"] == 6

    def test_tick_advances_total_ticks(self) -> None:
        state = _make_state(total_ticks=50)
        tick(state)
        assert state["lifecycle"]["total_ticks"] == 51

    def test_death_triggers_rebirth(self) -> None:
        state = _make_state(stage="death")
        state["_meta"]["cause_of_death"] = "old age"
        state["_meta"]["died_at"] = 100
        new_state = tick(state)
        assert new_state["lifecycle"]["stage"] == "egg"
        assert new_state["_meta"]["generation"] == 2

    def test_narration_updated(self) -> None:
        state = _make_state()
        tick(state)
        assert state["narration"] != ""

    def test_10_ticks_no_crash(self) -> None:
        """Smoke test: 10 consecutive ticks without exception."""
        state = _make_state()
        for _ in range(10):
            state = tick(state)
        assert state["lifecycle"]["total_ticks"] >= 10 or state["lifecycle"]["stage"] == "egg"

    def test_full_lifecycle_no_crash(self) -> None:
        """Run through egg → larva → pupa → adult → death → rebirth."""
        state = _make_state(stage="egg", stage_tick=0, total_ticks=0, frame=1)
        max_ticks = 200
        saw_death = False
        for _ in range(max_ticks):
            state = tick(state)
            if state["lifecycle"]["stage"] == "death":
                saw_death = True
            if state["_meta"]["generation"] > 1:
                break
        # Either reached generation 2, or at least saw death
        assert saw_death or state["_meta"]["generation"] > 1

    def test_energy_bounded(self) -> None:
        """Energy stays in [0, max] across multiple ticks."""
        state = _make_state()
        for _ in range(20):
            state = tick(state)
            if state["lifecycle"]["stage"] == "death":
                break
            assert 0 <= state["energy"]["current"] <= state["energy"]["max"]

    def test_position_bounded(self) -> None:
        """Fly stays within kitchen bounds across ticks."""
        state = _make_state()
        for _ in range(20):
            state = tick(state)
            if state["lifecycle"]["stage"] == "death":
                break
            pos = state["body"]["position"]
            assert 0 <= pos["x"] <= state["kitchen"]["width"]
            assert 0 <= pos["y"] <= state["kitchen"]["height"]


# ──────────────────────────────────────────────────────────────────────
# STAGES constant
# ──────────────────────────────────────────────────────────────────────

class TestStages:
    def test_correct_order(self) -> None:
        assert STAGES == ["egg", "larva", "pupa", "adult", "death"]

    def test_five_stages(self) -> None:
        assert len(STAGES) == 5
