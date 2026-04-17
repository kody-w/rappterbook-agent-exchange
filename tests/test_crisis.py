"""Tests for the Crisis Protocol module — the colony's immune system."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.crisis import (
    CrisisState, CrisisEpisode, LegacyWarning,
    detect_crises, update_crisis_state, compute_consumption_modifier,
    crisis_action_weights, compute_trust_boost, harvest_legacy,
    format_crisis_year_data, _cause_to_resource,
    CRISIS_THRESHOLD, MAX_PREPAREDNESS, RATIONING_MULT, TRUST_BOOST_PER_YEAR,
    RESOURCE_RESPONSE_MAP,
)
from src.mars100.colony import Resources, RESOURCE_NAMES


# ---------- detect_crises ----------

def test_detect_crises_none_critical():
    """No crises when all resources are healthy."""
    r = Resources()
    assert detect_crises(r) == []


def test_detect_crises_one_critical():
    """One resource below threshold triggers one crisis."""
    r = Resources()
    r.air = 0.05
    crises = detect_crises(r)
    assert crises == ["air"]


def test_detect_crises_multiple_critical():
    """Multiple resources below threshold."""
    r = Resources()
    r.food = 0.10
    r.water = 0.12
    crises = detect_crises(r)
    assert "food" in crises and "water" in crises
    assert len(crises) == 2


def test_detect_crises_boundary():
    """Resource exactly at threshold is NOT in crisis."""
    r = Resources()
    r.air = CRISIS_THRESHOLD
    assert detect_crises(r) == []
    r.air = CRISIS_THRESHOLD - 0.001
    assert detect_crises(r) == ["air"]


# ---------- update_crisis_state ----------

def test_update_starts_episode():
    """New crisis starts an episode."""
    state = CrisisState()
    update_crisis_state(state, ["air"], year=50)
    assert len(state.episodes) == 1
    assert state.episodes[0].resource == "air"
    assert state.episodes[0].start_year == 50
    assert state.episodes[0].is_active()
    assert state.active == ["air"]
    assert state.total_crisis_years == 1


def test_update_ends_episode():
    """Resolved crisis ends the episode."""
    state = CrisisState()
    update_crisis_state(state, ["air"], year=50)
    update_crisis_state(state, [], year=51)
    assert not state.episodes[0].is_active()
    assert state.episodes[0].end_year == 51
    assert state.active == []


def test_update_preparedness_increases():
    """Each new episode bumps preparedness."""
    state = CrisisState()
    initial = state.preparedness
    update_crisis_state(state, ["air"], year=50)
    assert abs(state.preparedness - (initial + 0.1)) < 1e-9
    update_crisis_state(state, ["air", "food"], year=51)
    assert abs(state.preparedness - (initial + 0.2)) < 1e-9


def test_preparedness_capped():
    """Preparedness doesn't exceed MAX_PREPAREDNESS."""
    state = CrisisState()
    state.preparedness = MAX_PREPAREDNESS - 0.05
    update_crisis_state(state, ["air"], year=50)
    assert state.preparedness == MAX_PREPAREDNESS


def test_continuing_crisis_increments_years():
    """Ongoing crisis increments total_crisis_years."""
    state = CrisisState()
    update_crisis_state(state, ["air"], year=50)
    update_crisis_state(state, ["air"], year=51)
    assert state.total_crisis_years == 2


# ---------- compute_consumption_modifier ----------

def test_consumption_no_crisis():
    """No crisis means normal consumption."""
    state = CrisisState()
    assert compute_consumption_modifier(state) == 1.0


def test_consumption_during_crisis():
    """Active crisis triggers rationing."""
    state = CrisisState()
    state.active = ["air"]
    assert compute_consumption_modifier(state) == RATIONING_MULT


# ---------- crisis_action_weights ----------

def test_action_weights_no_crisis():
    """No crisis means no bonuses."""
    state = CrisisState()
    assert crisis_action_weights(state) == {}


def test_action_weights_air_crisis():
    """Air crisis boosts terraform and code."""
    state = CrisisState()
    state.active = ["air"]
    bonuses = crisis_action_weights(state)
    assert "terraform" in bonuses
    assert "code" in bonuses
    assert bonuses["terraform"] > 0


def test_action_weights_multi_crisis():
    """Multiple crises stack bonuses."""
    state = CrisisState()
    state.active = ["air", "food"]
    bonuses = crisis_action_weights(state)
    # terraform gets bonus from both air and water response maps
    assert bonuses.get("terraform", 0) > 0
    assert bonuses.get("farm", 0) > 0


def test_action_weights_scaled_by_preparedness():
    """Higher preparedness scales bonuses up."""
    state = CrisisState()
    state.active = ["air"]
    state.preparedness = 1.0
    base = crisis_action_weights(state)
    state.preparedness = 2.0
    scaled = crisis_action_weights(state)
    for action in base:
        assert scaled[action] == base[action] * 2.0


# ---------- compute_trust_boost ----------

def test_trust_boost_no_crisis():
    """No crisis means no trust boost."""
    state = CrisisState()
    assert compute_trust_boost(state) == 0.0


def test_trust_boost_single_crisis():
    """Single crisis gives base trust boost."""
    state = CrisisState()
    state.active = ["air"]
    assert compute_trust_boost(state) == TRUST_BOOST_PER_YEAR


def test_trust_boost_multi_crisis():
    """Multiple crises scale trust boost."""
    state = CrisisState()
    state.active = ["air", "food", "water"]
    assert compute_trust_boost(state) == TRUST_BOOST_PER_YEAR * 3


# ---------- harvest_legacy ----------

def test_harvest_legacy_basic():
    """Harvesting creates a warning."""
    state = CrisisState()
    colonist = {"id": "col-1", "name": "Aria", "memories": []}
    harvest_legacy(colonist, state, year=80, cause="asphyxiation")
    assert len(state.legacy_warnings) == 1
    w = state.legacy_warnings[0]
    assert w.colonist_id == "col-1"
    assert w.colonist_name == "Aria"
    assert "asphyxiation" in w.warning
    assert w.year == 80


def test_harvest_legacy_with_memory():
    """Legacy includes strongest memory when available."""
    state = CrisisState()
    colonist = {
        "id": "col-2", "name": "Kael",
        "memories": [
            {"event": "A beautiful sunset on Mars", "emotional_valence": 0.3},
            {"event": "The air ran out and I couldn't breathe", "emotional_valence": -0.9},
        ]
    }
    harvest_legacy(colonist, state, year=85, cause="asphyxiation")
    w = state.legacy_warnings[0]
    assert "air ran out" in w.warning


def test_harvest_legacy_counts_deaths():
    """Deaths during active air crisis increment episode counter."""
    state = CrisisState()
    state.episodes.append(CrisisEpisode(resource="air", start_year=80))
    colonist = {"id": "col-1", "name": "Aria", "memories": []}
    harvest_legacy(colonist, state, year=82, cause="asphyxiation")
    assert state.episodes[0].deaths_during == 1


# ---------- format_crisis_year_data ----------

def test_format_crisis_year_data():
    """Formatted data includes all expected keys."""
    state = CrisisState()
    state.active = ["air"]
    state.preparedness = 1.5
    state.total_crisis_years = 3
    data = format_crisis_year_data(state, 0.75, 0.08)
    assert data["active_crises"] == ["air"]
    assert data["consumption_mult"] == 0.75
    assert data["min_resource_level"] == 0.08
    assert data["preparedness"] == 1.5
    assert data["total_crisis_years"] == 3


# ---------- _cause_to_resource ----------

def test_cause_to_resource_air():
    assert _cause_to_resource("asphyxiation") == "air"
    assert _cause_to_resource("suffocation") == "air"


def test_cause_to_resource_food():
    assert _cause_to_resource("starvation") == "food"


def test_cause_to_resource_unknown():
    assert _cause_to_resource("meteor_impact") == "unknown"


# ---------- CrisisEpisode ----------

def test_episode_duration():
    """Duration calculation."""
    ep = CrisisEpisode(resource="air", start_year=50, end_year=55)
    assert ep.duration() == 6  # years 50-55 inclusive

    active_ep = CrisisEpisode(resource="food", start_year=80)
    assert active_ep.duration() == 1  # still active, min duration


def test_episode_to_dict():
    """Serialization includes all fields."""
    ep = CrisisEpisode(resource="air", start_year=50, deaths_during=3)
    d = ep.to_dict()
    assert d["resource"] == "air"
    assert d["deaths_during"] == 3


# ---------- CrisisState ----------

def test_crisis_state_to_dict():
    """Full state serialization."""
    state = CrisisState()
    state.episodes.append(CrisisEpisode(resource="air", start_year=50))
    state.active = ["air"]
    state.preparedness = 1.3
    d = state.to_dict()
    assert d["active_episodes"] == 1
    assert d["total_episodes"] == 1
    assert d["preparedness"] == 1.3


# ---------- Resource bounds invariant ----------

def test_resources_stay_in_bounds_with_rationing():
    """Consumption multiplier keeps resources in [0, 1]."""
    from src.mars100.colony import tick_resources
    r = Resources()
    r.air = 0.10
    bonuses = {name: 0.0 for name in RESOURCE_NAMES}
    effects = {name: 0.0 for name in RESOURCE_NAMES}
    # With rationing (0.75x consumption)
    tick_resources(r, 5, bonuses, effects, consumption_mult=RATIONING_MULT)
    for name in RESOURCE_NAMES:
        val = getattr(r, name)
        assert 0.0 <= val <= 1.0, f"{name}={val} out of bounds"


def test_tick_resources_default_consumption_mult():
    """Default consumption_mult=1.0 preserves existing behavior."""
    from src.mars100.colony import tick_resources
    r1 = Resources()
    r2 = Resources()
    bonuses = {name: 0.0 for name in RESOURCE_NAMES}
    effects = {name: 0.0 for name in RESOURCE_NAMES}
    tick_resources(r1, 5, bonuses, effects)
    tick_resources(r2, 5, bonuses, effects, consumption_mult=1.0)
    for name in RESOURCE_NAMES:
        assert getattr(r1, name) == getattr(r2, name)
