"""Tests for the Vigil organ (engine v12.0)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars100.vigil import (
    VigilState, ChannelVitals,
    tick_vigil, compute_revival_pressure,
    DORMANCY_THRESHOLD, EXTINCTION_THRESHOLD,
    REVIVAL_NUDGE_BASE, REVIVAL_NUDGE_MAX, EXTINCTION_NUDGE_DECAY,
    SKILL_TO_ACTION, DEFAULT_ACTIONS,
)


def _colonist(cid: str, skills: dict[str, float]) -> dict:
    return {"id": cid, "skills": skills}


def _diplomacy(faction_specs: list[tuple[str, int]]) -> dict:
    return {"factions": [{"id": fid, "size": size} for fid, size in faction_specs]}


def _culture(traditions: list[tuple[str, float]]) -> dict:
    return {
        "traditions": [
            {"name": n, "importance": imp, "source_year": 0}
            for n, imp in traditions
        ],
    }


def test_smoke_ten_years():
    state = VigilState()
    rng = random.Random(13591)
    for year in range(1, 11):
        colonists = [_colonist("c1", {"terraforming": 0.5, "hydroponics": 0.2}),
                     _colonist("c2", {"terraforming": 0.1, "hydroponics": 0.6})]
        actions = {"c1": "terraform", "c2": "farm"}
        diplo = _diplomacy([("red", 4)])
        culture = _culture([("Common Table", 0.9)])
        result = tick_vigil(state, year, colonists, actions, DEFAULT_ACTIONS,
                            diplomacy=diplo, culture=culture, rng=rng)
        assert result.year == year
    assert "skill:terraforming" in state.channels
    assert "action:farm" in state.channels


def test_pressure_within_bounds():
    state = VigilState()
    for act in ("terraform", "farm", "mediate", "code", "pray"):
        ch = ChannelVitals(kind="action", name=act, last_spark_year=0,
                           silent_for=200, status="dormant", born_year=0)
        state.channels[f"action:{act}"] = ch
    pressure = compute_revival_pressure(state, DEFAULT_ACTIONS)
    for act, p in pressure.items():
        assert -REVIVAL_NUDGE_MAX <= p <= REVIVAL_NUDGE_MAX, \
            f"pressure[{act}]={p} out of bounds"


def test_extinct_pressure_is_decayed():
    s_dormant = VigilState()
    s_dormant.channels["action:pray"] = ChannelVitals(
        kind="action", name="pray", last_spark_year=0,
        silent_for=DORMANCY_THRESHOLD, status="dormant", born_year=0)
    s_extinct = VigilState()
    s_extinct.channels["action:pray"] = ChannelVitals(
        kind="action", name="pray", last_spark_year=0,
        silent_for=DORMANCY_THRESHOLD, status="extinct", born_year=0)

    p_dormant = compute_revival_pressure(s_dormant, DEFAULT_ACTIONS)["pray"]
    p_extinct = compute_revival_pressure(s_extinct, DEFAULT_ACTIONS)["pray"]
    assert p_extinct == pytest.approx(p_dormant * EXTINCTION_NUDGE_DECAY, abs=1e-6)


def test_channels_never_deleted():
    state = VigilState()
    counts: list[int] = []
    rng = random.Random(0)
    for year in range(1, 30):
        colonists = [_colonist("c1", {"terraforming": 0.5})]
        actions = {"c1": "rest"}
        tick_vigil(state, year, colonists, actions, DEFAULT_ACTIONS, rng=rng)
        counts.append(len(state.channels))
    assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1)), \
        f"channel count decreased: {counts}"


def test_spark_resets_silence():
    state = VigilState()
    state.channels["action:terraform"] = ChannelVitals(
        kind="action", name="terraform", last_spark_year=0,
        silent_for=5, status="alive", born_year=0)
    tick_vigil(state, year=10,
               active_colonists=[_colonist("c1", {})],
               actions={"c1": "terraform"},
               action_channel_list=("terraform",))
    assert state.channels["action:terraform"].silent_for == 0
    assert state.channels["action:terraform"].last_spark_year == 10


def test_flatline_after_threshold():
    state = VigilState()
    # Use empty skills dict so only the action channel is tracked.
    tick_vigil(state, 1, [_colonist("c1", {})],
               {"c1": "sabotage"}, ("sabotage",))
    assert state.channels["action:sabotage"].status == "alive"
    for y in range(2, 2 + DORMANCY_THRESHOLD + 1):
        tick_vigil(state, y, [_colonist("c1", {})], {"c1": "rest"},
                   ("sabotage", "rest"))
    assert state.channels["action:sabotage"].status == "dormant"
    assert state.channels["action:sabotage"].flatlines == 1
    assert state.total_flatlines == 1


def test_extinction_after_long_silence():
    state = VigilState()
    tick_vigil(state, 1, [_colonist("c1", {})],
               {"c1": "pray"}, ("pray",))
    for y in range(2, 2 + EXTINCTION_THRESHOLD + 1):
        tick_vigil(state, y, [_colonist("c1", {})], {"c1": "rest"},
                   ("pray", "rest"))
    assert state.channels["action:pray"].status == "extinct"
    assert state.total_extinctions == 1


def test_revival_increments_counter():
    state = VigilState()
    state.channels["action:mediate"] = ChannelVitals(
        kind="action", name="mediate", last_spark_year=0,
        silent_for=15, status="dormant", born_year=0)
    result = tick_vigil(state, 16,
                        [_colonist("c1", {})],
                        {"c1": "mediate"},
                        ("mediate",))
    assert state.channels["action:mediate"].status == "revived"
    assert state.channels["action:mediate"].revivals == 1
    assert state.total_revivals == 1
    assert any(e["name"] == "mediate" for e in result.newly_revived)


def test_revival_prompts_have_required_shape():
    state = VigilState()
    state.channels["skill:terraforming"] = ChannelVitals(
        kind="skill", name="terraforming", last_spark_year=0,
        silent_for=DORMANCY_THRESHOLD, status="dormant", born_year=0)
    result = tick_vigil(state, DORMANCY_THRESHOLD + 1,
                        [_colonist("c1", {})], {"c1": "rest"},
                        ("terraform", "rest"))
    prompts = [p for p in result.revival_prompts if p["channel_name"] == "terraforming"]
    assert prompts, "expected a revival prompt for dormant terraforming skill"
    p = prompts[0]
    for key in ("channel_kind", "channel_name", "action", "strength",
                "status", "silent_for", "prompt"):
        assert key in p, f"missing key {key}"
    assert p["action"] == SKILL_TO_ACTION["terraforming"]
    assert 0 < p["strength"] <= REVIVAL_NUDGE_MAX


def test_sabotage_and_hoard_are_not_revived():
    state = VigilState()
    state.channels["action:sabotage"] = ChannelVitals(
        kind="action", name="sabotage", last_spark_year=0,
        silent_for=50, status="dormant", born_year=0)
    state.channels["action:hoard"] = ChannelVitals(
        kind="action", name="hoard", last_spark_year=0,
        silent_for=50, status="dormant", born_year=0)
    pressure = compute_revival_pressure(state, ("sabotage", "hoard", "mediate"))
    assert "sabotage" not in pressure
    assert "hoard" not in pressure


def test_faction_dormancy_nudges_mediate():
    state = VigilState()
    state.channels["faction:red"] = ChannelVitals(
        kind="faction", name="red", last_spark_year=0,
        silent_for=DORMANCY_THRESHOLD + 1, status="dormant", born_year=0)
    pressure = compute_revival_pressure(state, ("mediate", "cooperate"))
    assert "mediate" in pressure
    assert pressure["mediate"] > 0


def test_tradition_dormancy_nudges_cooperate():
    state = VigilState()
    state.channels["tradition:Common Table"] = ChannelVitals(
        kind="tradition", name="Common Table", last_spark_year=0,
        silent_for=DORMANCY_THRESHOLD + 1, status="dormant", born_year=0)
    pressure = compute_revival_pressure(state, ("cooperate", "mediate"))
    assert "cooperate" in pressure


def test_snapshot_serialises_cleanly():
    state = VigilState()
    tick_vigil(state, 1, [_colonist("c1", {"hydroponics": 0.4})],
               {"c1": "farm"}, DEFAULT_ACTIONS,
               diplomacy=_diplomacy([("blue", 3)]),
               culture=_culture([("Common Table", 0.9)]))
    snapshot = state.to_dict()
    encoded = json.dumps(snapshot)
    decoded = json.loads(encoded)
    assert "channels" in decoded
    assert "totals" in decoded


def test_pressure_only_for_actions_in_play():
    state = VigilState()
    state.channels["action:terraform"] = ChannelVitals(
        kind="action", name="terraform", last_spark_year=0,
        silent_for=DORMANCY_THRESHOLD + 1, status="dormant", born_year=0)
    pressure = compute_revival_pressure(state, ("rest",))
    assert pressure == {}


def test_integration_with_engine_tick():
    from src.mars100.engine import Mars100Engine
    engine = Mars100Engine(seed=42, total_years=15)
    result = engine.run()
    assert any(
        isinstance(y.to_dict().get("vigil"), dict)
        for y in result.years
    ), "engine output must include vigil snapshot"
