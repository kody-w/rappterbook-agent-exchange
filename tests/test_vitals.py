"""Tests for the vitals organ (engine v12.0)."""
from __future__ import annotations

import copy
import json

import pytest

from src.mars100.vitals import (
    CHANNEL_NAMES, DYING_THRESHOLD, FLATLINE_THRESHOLD, MAX_ACTION_NUDGE,
    ChannelVitals, VitalsState, VitalsTickResult,
    colony_health_report, compute_action_nudges, tick_vitals,
)


ACTIONS = ["terraform", "farm", "mediate", "code", "pray",
           "cooperate", "rest", "explore", "sabotage", "hoard"]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _live_year(year: int = 1) -> dict:
    """A year_result with activity in every channel.

    Social cohesion and biosphere drift slightly year-over-year so that
    delta-based detectors (social, ecology) register signal when the
    caller threads the baseline properly.
    """
    cohesion = 0.50 + 0.05 * ((year % 3) - 1)
    biosphere = 0.10 + 0.02 * (year % 4)
    return {
        "year": year,
        "resource_delta": {"food": 3.0, "water": -2.0, "oxygen": 1.5},
        "births": [{"id": f"c-{year + 100}"}],
        "immigrants": [],
        "governance": {"proposal_id": f"p{year}", "passed": True},
        "infrastructure": {
            "active_projects": [{"id": "greenhouse"}],
            "completed_this_year": 1,
        },
        "culture": {"new_traditions": 1, "new_martyrs": 0, "new_taboos": 1},
        "economics": {
            "trades": [{"a": "c-1", "b": "c-2"}],
            "total_labor_income": 10.0,
        },
        "diplomacy": {
            "factions_formed": [{"id": f"f-{year}"}],
            "factions_dissolved": [],
            "alliances_formed": [], "alliances_broken": [],
            "schisms": [], "leader_changes": [],
            "faction_count": 1,
        },
        "ecology": {"biosphere_index": biosphere},
        "psychology": {"crises": [{"colonist_id": "c-3"}], "morale_delta": 0.1},
        "earth_events": [{"ship_arrived": True, "messages": ["hi"]}],
        "social_cohesion": cohesion,
    }


def _dead_year(year: int) -> dict:
    return {
        "year": year,
        "resource_delta": {},
        "births": [],
        "immigrants": [],
        "governance": None,
        "infrastructure": {"active_projects": [], "completed_this_year": 0},
        "culture": {"new_traditions": 0, "new_martyrs": 0, "new_taboos": 0},
        "economics": {"trades": [], "total_labor_income": 0},
        "diplomacy": {"factions_formed": [], "factions_dissolved": [],
                      "alliances_formed": [], "alliances_broken": [],
                      "schisms": [], "leader_changes": [],
                      "faction_count": 0},
        "ecology": {"biosphere_index": 0.0},
        "psychology": {"crises": [], "morale_delta": 0.0},
        "earth_events": [],
        "social_cohesion": 0.0,
    }


def _tick_with_baseline(state: VitalsState, year: int,
                         year_result: dict, prev: dict) -> VitalsTickResult:
    """Run tick_vitals while threading ecology/social baselines (mutates prev)."""
    r = tick_vitals(
        state, year=year, year_result=year_result,
        prev_ecology_biosphere=prev.get("biosphere"),
        prev_social_cohesion=prev.get("cohesion"),
    )
    prev["biosphere"] = year_result.get("ecology", {}).get("biosphere_index", 0.0)
    prev["cohesion"] = year_result.get("social_cohesion", 0.0)
    return r


# ---------------------------------------------------------------------------
# Smoke / shape
# ---------------------------------------------------------------------------

def test_state_initializes_all_channels():
    state = VitalsState()
    assert set(state.channels.keys()) == set(CHANNEL_NAMES)
    for ch in state.channels.values():
        assert ch.pulse == 0.0
        assert ch.silent_years == 0


def test_tick_returns_pulses_for_every_channel():
    state = VitalsState()
    result = tick_vitals(state, year=1, year_result=_live_year())
    assert isinstance(result, VitalsTickResult)
    assert set(result.pulses.keys()) == set(CHANNEL_NAMES)


def test_tick_result_serializes_cleanly():
    state = VitalsState()
    result = tick_vitals(state, year=1, year_result=_live_year())
    json.dumps(result.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# Invariants — physical bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("year_factory", [_live_year, _dead_year])
def test_pulses_stay_in_unit_interval(year_factory):
    state = VitalsState()
    biosphere = 0.0
    cohesion = 0.0
    for year in range(1, 25):
        yr = year_factory(year)
        r = tick_vitals(state, year=year, year_result=yr,
                        prev_ecology_biosphere=biosphere,
                        prev_social_cohesion=cohesion)
        biosphere = yr.get("ecology", {}).get("biosphere_index", 0.0)
        cohesion = yr.get("social_cohesion", 0.0)
        for name, pulse in r.pulses.items():
            assert 0.0 <= pulse <= 1.0, f"{name} pulse out of bounds: {pulse}"
        assert 0.0 <= r.overall_vitality <= 1.0


def test_overall_vitality_is_mean_of_channel_pulses():
    state = VitalsState()
    r = tick_vitals(state, year=1, year_result=_live_year())
    expected = sum(r.pulses.values()) / len(r.pulses)
    assert r.overall_vitality == pytest.approx(expected, abs=1e-9)


def test_decay_monotonic_when_no_signal():
    """With zero signal every year, pulse must not rise."""
    state = VitalsState()
    tick_vitals(state, year=1, year_result=_live_year(1))
    last_pulses = {n: state.channels[n].pulse for n in CHANNEL_NAMES}
    for year in range(2, 10):
        tick_vitals(state, year=year, year_result=_dead_year(year))
        for name in CHANNEL_NAMES:
            current = state.channels[name].pulse
            assert current <= last_pulses[name] + 1e-9, (
                f"{name} pulse rose without signal: "
                f"{last_pulses[name]} -> {current}")
            last_pulses[name] = current


# ---------------------------------------------------------------------------
# Flatline detection
# ---------------------------------------------------------------------------

def test_silent_channel_flatlines_at_threshold():
    state = VitalsState()
    revivals_by_year: dict[int, list[str]] = {}
    for year in range(1, FLATLINE_THRESHOLD + 3):
        r = tick_vitals(state, year=year, year_result=_dead_year(year))
        revivals_by_year[year] = [p["channel"] for p in r.revivals]
    for year in range(1, FLATLINE_THRESHOLD):
        assert revivals_by_year[year] == [], (
            f"unexpected revival at year {year}")
    fired = revivals_by_year[FLATLINE_THRESHOLD]
    assert set(fired) == set(CHANNEL_NAMES), (
        f"expected all channels to flatline together; got {fired}")


def test_chronic_flatline_re_emits_every_threshold_period():
    state = VitalsState()
    counts: dict[str, int] = {n: 0 for n in CHANNEL_NAMES}
    years_to_run = FLATLINE_THRESHOLD * 3 + 1
    for year in range(1, years_to_run + 1):
        r = tick_vitals(state, year=year, year_result=_dead_year(year))
        for p in r.revivals:
            counts[p["channel"]] += 1
    for name in CHANNEL_NAMES:
        assert counts[name] == 3, f"{name} emitted {counts[name]} times"


def test_signal_resets_silent_counter():
    state = VitalsState()
    prev: dict = {}
    for year in range(1, FLATLINE_THRESHOLD):
        _tick_with_baseline(state, year, _dead_year(year), prev)
    _tick_with_baseline(state, FLATLINE_THRESHOLD,
                        _live_year(FLATLINE_THRESHOLD), prev)
    r = _tick_with_baseline(state, FLATLINE_THRESHOLD + 1,
                            _live_year(FLATLINE_THRESHOLD + 1), prev)
    assert r.revivals == []
    for name in CHANNEL_NAMES:
        assert state.channels[name].silent_years == 0, (
            f"{name} still silent after live year")


# ---------------------------------------------------------------------------
# Revival prompt content
# ---------------------------------------------------------------------------

def test_revival_prompt_severity_escalates():
    state = VitalsState()
    severities: list[str] = []
    for year in range(1, FLATLINE_THRESHOLD * 2 + 2):
        r = tick_vitals(state, year=year, year_result=_dead_year(year))
        severities.extend(p["severity"] for p in r.revivals)
    assert "warn" in severities
    assert "critical" in severities
    first_warn = severities.index("warn")
    first_critical = severities.index("critical")
    assert first_warn < first_critical


def test_revival_prompts_have_actionable_suggestions():
    state = VitalsState()
    r = None
    for year in range(1, FLATLINE_THRESHOLD + 1):
        r = tick_vitals(state, year=year, year_result=_dead_year(year))
    assert r is not None
    for p in r.revivals:
        assert isinstance(p["suggestion"], str)
        assert len(p["suggestion"]) > 10
        assert p["channel"] in CHANNEL_NAMES


# ---------------------------------------------------------------------------
# Action nudges — feedback loop
# ---------------------------------------------------------------------------

def test_action_nudges_zero_when_all_healthy():
    state = VitalsState()
    prev: dict = {}
    for year in range(1, 20):
        _tick_with_baseline(state, year, _live_year(year), prev)
    nudges = compute_action_nudges(state, ACTIONS)
    for action, weight in nudges.items():
        assert weight == 0.0, f"{action} got nudge {weight} despite health"


def test_action_nudges_capped_at_max():
    state = VitalsState()
    for year in range(1, FLATLINE_THRESHOLD + 5):
        tick_vitals(state, year=year, year_result=_dead_year(year))
    nudges = compute_action_nudges(state, ACTIONS)
    for action, weight in nudges.items():
        assert 0.0 <= weight <= MAX_ACTION_NUDGE + 1e-9, (
            f"{action} nudge {weight} exceeded cap")
    assert any(w > 0 for w in nudges.values())


def test_action_nudges_only_touch_known_actions():
    state = VitalsState()
    for year in range(1, FLATLINE_THRESHOLD + 1):
        tick_vitals(state, year=year, year_result=_dead_year(year))
    nudges = compute_action_nudges(state, ACTIONS)
    assert set(nudges.keys()) == set(ACTIONS)


# ---------------------------------------------------------------------------
# Status / serialization / report
# ---------------------------------------------------------------------------

def test_channel_status_transitions():
    ch = ChannelVitals(name="resources")
    assert ch.status() == "dying"
    ch.pulse = 0.5
    assert ch.status() == "stable"
    ch.pulse = 0.9
    assert ch.status() == "healthy"
    ch.silent_years = FLATLINE_THRESHOLD
    assert ch.status() == "flatlined"


def test_state_serialization_round_trip():
    state = VitalsState()
    for year in range(1, 8):
        tick_vitals(state, year=year, year_result=_live_year(year))
    d = state.to_dict()
    restored = VitalsState.from_dict(d)
    for name in CHANNEL_NAMES:
        a, b = state.channels[name], restored.channels[name]
        assert a.pulse == pytest.approx(b.pulse, abs=1e-4)
        assert a.silent_years == b.silent_years
        assert a.total_signals == b.total_signals


def test_health_report_categorizes_channels():
    state = VitalsState()
    prev: dict = {}
    for year in range(1, 15):
        _tick_with_baseline(state, year, _live_year(year), prev)
    report = colony_health_report(state)
    assert "overall_vitality" in report
    assert "flatlined" in report
    assert "dying" in report
    assert "healthy" in report
    assert isinstance(report["channels"], dict)
    assert report["flatlined"] == []


def test_health_report_after_long_silence_flags_everything():
    state = VitalsState()
    for year in range(1, FLATLINE_THRESHOLD + 5):
        tick_vitals(state, year=year, year_result=_dead_year(year))
    report = colony_health_report(state)
    assert set(report["flatlined"]) == set(CHANNEL_NAMES)
    assert report["overall_vitality"] < DYING_THRESHOLD


def test_tick_does_not_mutate_year_result():
    state = VitalsState()
    yr = _live_year(1)
    snapshot = copy.deepcopy(yr)
    tick_vitals(state, year=1, year_result=yr)
    assert yr == snapshot, "vitals organ must be read-only over year_result"


# ---------------------------------------------------------------------------
# Engine integration smoke
# ---------------------------------------------------------------------------

def test_engine_includes_vitals_in_year_result():
    from src.mars100.engine import Mars100Engine
    engine = Mars100Engine(seed=42, total_years=3)
    result = engine.tick()
    assert hasattr(result, "vitals")
    assert isinstance(result.vitals, dict)
    assert "pulses" in result.vitals
    assert set(result.vitals["pulses"].keys()) == set(CHANNEL_NAMES)


def test_engine_runs_ten_ticks_without_crash():
    """The mandatory 10-step smoke test from the framing."""
    from src.mars100.engine import Mars100Engine
    engine = Mars100Engine(seed=7, total_years=10)
    for _ in range(10):
        if not engine._active_colonists():
            break
        engine.tick()
    assert 0.0 <= engine.vitals.last_overall <= 1.0
