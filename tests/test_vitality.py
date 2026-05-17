"""Tests for the vitality organ (engine v12.0)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.vitality import (
    CRITICAL_THRESHOLD, DORMANCY_THRESHOLD, MAX_PROMPTS_PER_TICK,
    PROMPT_COOLDOWN_YEARS, RevivalPrompt, SUBJECT_TYPES,
    TRACKED_ACTIONS, TRACKED_IDEOLOGIES, VitalityState, VitalitySubject,
    VitalityTickResult, _make_prompt, compute_colony_vitality_index,
    gather_activity, tick_vitality,
)


class TestConstants:
    def test_thresholds_ordered(self):
        assert 0 < DORMANCY_THRESHOLD < CRITICAL_THRESHOLD

    def test_seed_doctrine(self):
        assert DORMANCY_THRESHOLD == 10  # "0 posts in 10+ frames"

    def test_cooldown_sane(self):
        assert PROMPT_COOLDOWN_YEARS > 0
        assert MAX_PROMPTS_PER_TICK > 0

    def test_tracked_actions_match_engine(self):
        from src.mars100 import engine as eng
        for a in TRACKED_ACTIONS:
            assert a in eng.ACTIONS, f"action {a} missing from engine.ACTIONS"

    def test_tracked_ideologies_match_diplomacy(self):
        from src.mars100.diplomacy import IDEOLOGY_NAMES
        assert set(TRACKED_IDEOLOGIES) == set(IDEOLOGY_NAMES)

    def test_subject_types_complete(self):
        for t in ("faction", "action", "tradition", "project", "ideology"):
            assert t in SUBJECT_TYPES


class TestVitalitySubject:
    def test_default_status_alive(self):
        s = VitalitySubject(subject_type="action", subject_id="farm")
        assert s.status() == "alive"
        assert s.vitality_score() == 1.0

    def test_fading_status(self):
        s = VitalitySubject(subject_type="action", subject_id="farm",
                            years_dormant=DORMANCY_THRESHOLD // 2)
        assert s.status() == "fading"

    def test_dormant_status(self):
        s = VitalitySubject(subject_type="action", subject_id="farm",
                            years_dormant=DORMANCY_THRESHOLD)
        assert s.status() == "dormant"

    def test_critical_status(self):
        s = VitalitySubject(subject_type="action", subject_id="farm",
                            years_dormant=CRITICAL_THRESHOLD)
        assert s.status() == "critical"

    def test_vitality_score_bounds(self):
        for d in range(0, CRITICAL_THRESHOLD + 5):
            s = VitalitySubject(subject_type="action", subject_id="farm",
                                years_dormant=d)
            assert 0.0 <= s.vitality_score() <= 1.0

    def test_vitality_score_monotonic(self):
        scores = [
            VitalitySubject(subject_type="x", subject_id="y",
                            years_dormant=d).vitality_score()
            for d in range(0, CRITICAL_THRESHOLD + 1)
        ]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_to_dict_fields(self):
        s = VitalitySubject(subject_type="faction", subject_id="f1",
                            label="Cooperators", years_dormant=3)
        d = s.to_dict()
        assert d["subject_type"] == "faction"
        assert d["subject_id"] == "f1"
        assert d["label"] == "Cooperators"
        assert d["years_dormant"] == 3
        assert d["status"] == "alive"
        assert "vitality_score" in d


class TestGatherActivity:
    def test_empty_inputs(self):
        assert gather_activity(year=1) == {}

    def test_actions_counted(self):
        actions = {"c1": "farm", "c2": "farm", "c3": "code"}
        act = gather_activity(year=1, actions=actions)
        assert act["action:farm"] == 2
        assert act["action:code"] == 1

    def test_unknown_action_ignored(self):
        actions = {"c1": "tap-dance"}
        act = gather_activity(year=1, actions=actions)
        assert act == {}

    def test_factions_from_dataclass_like(self):
        class Fac:
            def __init__(self, id, members):
                self.id = id
                self.members = members
        facs = [Fac("f1", ["a", "b", "c"]), Fac("f2", [])]
        act = gather_activity(year=1, factions=facs)
        assert act["faction:f1"] == 3
        assert "faction:f2" not in act

    def test_factions_from_dict(self):
        facs = [{"id": "fx", "members": ["a", "b"]}]
        act = gather_activity(year=1, factions=facs)
        assert act["faction:fx"] == 2

    def test_traditions_counted(self):
        act = gather_activity(year=1, cultural_traditions=["harvest_song", "founders_day"])
        assert act["tradition:harvest_song"] == 1
        assert act["tradition:founders_day"] == 1

    def test_projects_counted(self):
        class Proj:
            tech_id = "greenhouse_v2"
        act = gather_activity(year=1, active_projects=[Proj()])
        assert act["project:greenhouse_v2"] == 1

    def test_ideologies_counted(self):
        act = gather_activity(year=1,
                              ideology_counts={"cooperative": 4, "survivalist": 0})
        assert act["ideology:cooperative"] == 4
        assert "ideology:survivalist" not in act

    def test_unknown_ideology_ignored(self):
        act = gather_activity(year=1, ideology_counts={"anarchist": 5})
        assert act == {}


class TestTickVitality:
    def test_tick_returns_result(self):
        state = VitalityState()
        result = tick_vitality(state, year=1, activity={})
        assert isinstance(result, VitalityTickResult)
        assert result.year == 1

    def test_active_subject_resets_dormancy(self):
        state = VitalityState()
        tick_vitality(state, year=1, activity={"action:farm": 3})
        subj = state.subjects["action:farm"]
        assert subj.years_dormant == 0
        assert subj.total_activity == 3

        tick_vitality(state, year=2, activity={})
        assert state.subjects["action:farm"].years_dormant == 1

        tick_vitality(state, year=3, activity={"action:farm": 1})
        assert state.subjects["action:farm"].years_dormant == 0
        assert state.subjects["action:farm"].total_activity == 4

    def test_known_subjects_register_even_when_silent(self):
        state = VitalityState()
        known = {"action": [(a, a) for a in TRACKED_ACTIONS]}
        tick_vitality(state, year=1, activity={}, known_subjects=known)
        for a in TRACKED_ACTIONS:
            assert f"action:{a}" in state.subjects

    def test_dormancy_threshold_emits_prompt(self):
        state = VitalityState()
        known = {"action": [("farm", "farm")]}
        tick_vitality(state, year=0, activity={}, known_subjects=known)
        all_prompts = []
        for year in range(1, DORMANCY_THRESHOLD + 2):
            result = tick_vitality(state, year=year, activity={},
                                   known_subjects=known)
            all_prompts.extend(result.prompts)
        assert len(all_prompts) >= 1
        prompt = all_prompts[0]
        assert prompt.subject_id == "farm"
        assert prompt.severity == "dormant"
        assert prompt.years_dormant >= DORMANCY_THRESHOLD

    def test_no_prompt_before_threshold(self):
        state = VitalityState()
        known = {"action": [("farm", "farm")]}
        tick_vitality(state, year=0, activity={}, known_subjects=known)
        for year in range(1, DORMANCY_THRESHOLD):
            result = tick_vitality(state, year=year, activity={},
                                   known_subjects=known)
            assert result.prompts == [], f"unexpected prompt at year {year}"

    def test_critical_severity_triggers(self):
        state = VitalityState()
        known = {"action": [("pray", "pray")]}
        tick_vitality(state, year=0, activity={}, known_subjects=known)
        all_prompts = []
        for year in range(1, CRITICAL_THRESHOLD + 2):
            result = tick_vitality(state, year=year, activity={},
                                   known_subjects=known)
            all_prompts.extend(result.prompts)
        assert any(p.severity == "critical" for p in all_prompts),             f"no critical prompts; severities: {[p.severity for p in all_prompts]}"

    def test_cooldown_suppresses_repeat_prompts(self):
        state = VitalityState()
        known = {"action": [("farm", "farm")]}
        tick_vitality(state, year=0, activity={}, known_subjects=known)
        first_prompt_year = None
        for year in range(1, DORMANCY_THRESHOLD + 3):
            result = tick_vitality(state, year=year, activity={},
                                   known_subjects=known)
            if result.prompts and first_prompt_year is None:
                first_prompt_year = year

        assert first_prompt_year is not None

        for offset in range(1, PROMPT_COOLDOWN_YEARS):
            year = first_prompt_year + offset
            result = tick_vitality(state, year=year, activity={},
                                   known_subjects=known)
            farm_prompts = [p for p in result.prompts if p.subject_id == "farm"]
            assert farm_prompts == [], (
                f"prompt repeated within cooldown at year {year}")

        fired_again = False
        for offset in range(PROMPT_COOLDOWN_YEARS, PROMPT_COOLDOWN_YEARS + 2):
            year = first_prompt_year + offset
            result = tick_vitality(state, year=year, activity={},
                                   known_subjects=known)
            if any(p.subject_id == "farm" for p in result.prompts):
                fired_again = True
                break
        assert fired_again, "prompt did not fire again after cooldown"

    def test_revival_recorded(self):
        state = VitalityState()
        known = {"action": [("farm", "farm")]}
        tick_vitality(state, year=0, activity={}, known_subjects=known)
        for year in range(1, DORMANCY_THRESHOLD + 2):
            tick_vitality(state, year=year, activity={}, known_subjects=known)
        result = tick_vitality(state, year=DORMANCY_THRESHOLD + 3,
                               activity={"action:farm": 1},
                               known_subjects=known)
        assert "action:farm" in result.revived
        assert state.subjects["action:farm"].revival_count == 1
        assert state.subjects["action:farm"].years_dormant == 0
        assert state.revival_history, "revival history not appended"

    def test_newly_dormant_event_fires_once(self):
        state = VitalityState()
        known = {"action": [("explore", "explore")]}
        tick_vitality(state, year=0, activity={}, known_subjects=known)
        fired_years = []
        for year in range(1, DORMANCY_THRESHOLD + 5):
            result = tick_vitality(state, year=year, activity={},
                                   known_subjects=known)
            if "action:explore" in result.newly_dormant:
                fired_years.append(year)
        assert len(fired_years) == 1, (
            f"newly_dormant should fire exactly once, fired: {fired_years}")

    def test_max_prompts_per_tick_capped(self):
        state = VitalityState()
        known = {"action": [(a, a) for a in TRACKED_ACTIONS]}
        tick_vitality(state, year=0, activity={}, known_subjects=known)
        max_in_a_tick = 0
        for year in range(1, DORMANCY_THRESHOLD * 3):
            result = tick_vitality(state, year=year, activity={},
                                   known_subjects=known)
            max_in_a_tick = max(max_in_a_tick, len(result.prompts))
        assert max_in_a_tick <= MAX_PROMPTS_PER_TICK
        assert max_in_a_tick > 0, "expected at least one tick to emit prompts"

    def test_summary_invariants(self):
        state = VitalityState()
        known = {"action": [(a, a) for a in TRACKED_ACTIONS[:5]]}
        tick_vitality(state, year=0, activity={}, known_subjects=known)
        result = None
        for year in range(1, 15):
            result = tick_vitality(state, year=year, activity={},
                                   known_subjects=known)
        s = result.summary
        total = s["alive"] + s["fading"] + s["dormant"] + s["critical"]
        assert total == len(state.subjects)


class TestPromptContent:
    def test_action_prompt_event(self):
        subj = VitalitySubject(subject_type="action", subject_id="farm",
                               years_dormant=DORMANCY_THRESHOLD)
        p = _make_prompt(subj, 50)
        assert p.suggested_event["kind"] == "event"
        assert p.suggested_event["action"] == "farm"
        assert "farm" in p.suggestion
        assert p.severity == "dormant"

    def test_faction_prompt(self):
        subj = VitalitySubject(subject_type="faction", subject_id="cooperators",
                               years_dormant=DORMANCY_THRESHOLD)
        p = _make_prompt(subj, 50)
        assert p.suggested_event["kind"] == "diplomacy"
        assert p.suggested_event["faction_id"] == "cooperators"
        assert p.suggested_event["action"] == "recruit"

    def test_tradition_critical_prompt(self):
        subj = VitalitySubject(subject_type="tradition", subject_id="harvest_song",
                               years_dormant=CRITICAL_THRESHOLD)
        p = _make_prompt(subj, 50)
        assert p.severity == "critical"
        assert p.suggested_event["kind"] == "culture"
        assert p.suggested_event["action"] == "archive"

    def test_project_prompt(self):
        subj = VitalitySubject(subject_type="project", subject_id="greenhouse_v2",
                               years_dormant=DORMANCY_THRESHOLD)
        p = _make_prompt(subj, 50)
        assert p.suggested_event["kind"] == "infrastructure"
        assert p.suggested_event["action"] == "reallocate"

    def test_ideology_prompt(self):
        subj = VitalitySubject(subject_type="ideology", subject_id="spiritual",
                               years_dormant=DORMANCY_THRESHOLD + 3)
        p = _make_prompt(subj, 50)
        assert p.suggested_event["kind"] == "immigration"
        assert p.suggested_event["ideology"] == "spiritual"

    def test_unknown_subject_type_safe(self):
        subj = VitalitySubject(subject_type="weather", subject_id="dust",
                               years_dormant=DORMANCY_THRESHOLD)
        p = _make_prompt(subj, 50)
        assert isinstance(p, RevivalPrompt)
        assert p.suggested_event["kind"] == "generic"


class TestAggregates:
    def test_colony_vitality_index_empty(self):
        assert compute_colony_vitality_index(VitalityState()) == 1.0

    def test_colony_vitality_index_bounds(self):
        state = VitalityState()
        known = {"action": [(a, a) for a in TRACKED_ACTIONS]}
        tick_vitality(state, year=0, activity={}, known_subjects=known)
        for year in range(1, CRITICAL_THRESHOLD + 5):
            tick_vitality(state, year=year, activity={}, known_subjects=known)
        idx = compute_colony_vitality_index(state)
        assert 0.0 <= idx <= 1.0

    def test_colony_vitality_index_monotone_decay(self):
        state = VitalityState()
        known = {"action": [(a, a) for a in TRACKED_ACTIONS]}
        tick_vitality(state, year=0, activity={}, known_subjects=known)
        prev = compute_colony_vitality_index(state)
        for year in range(1, CRITICAL_THRESHOLD + 1):
            tick_vitality(state, year=year, activity={}, known_subjects=known)
            cur = compute_colony_vitality_index(state)
            assert cur <= prev + 1e-9
            prev = cur

    def test_to_dict_structure(self):
        state = VitalityState()
        known = {"action": [("farm", "farm"), ("code", "code")]}
        tick_vitality(state, year=0, activity={"action:farm": 2},
                      known_subjects=known)
        d = state.to_dict()
        assert "subjects" in d
        assert "by_type" in d
        assert "status_counts" in d
        assert d["total_subjects"] == 2
        assert d["dormancy_threshold"] == DORMANCY_THRESHOLD
        for entries in d["by_type"].values():
            yds = [e["years_dormant"] for e in entries]
            assert yds == sorted(yds, reverse=True)

    def test_tick_result_to_dict(self):
        state = VitalityState()
        result = tick_vitality(state, year=1, activity={"action:farm": 1})
        d = result.to_dict()
        assert d["year"] == 1
        assert "prompts" in d
        assert "summary" in d


class TestSmoke:
    def test_fifteen_year_run_no_crash(self):
        state = VitalityState()
        known = {
            "action": [(a, a) for a in TRACKED_ACTIONS],
            "ideology": [(i, i) for i in TRACKED_IDEOLOGIES],
        }
        for year in range(0, 15):
            actions = {"c1": "farm", "c2": "code"}
            ideos = {"cooperative": 3, "technocratic": 2}
            activity = gather_activity(year=year, actions=actions,
                                       ideology_counts=ideos)
            result = tick_vitality(state, year=year, activity=activity,
                                   known_subjects=known)
            assert isinstance(result, VitalityTickResult)

        assert state.subjects["action:farm"].status() == "alive"
        assert state.subjects["action:pray"].status() in ("fading", "dormant")

    def test_hundred_year_run_bounded_memory(self):
        state = VitalityState()
        known = {"action": [(a, a) for a in TRACKED_ACTIONS]}
        for year in range(0, 100):
            actions = {"c1": "farm"} if year % 2 == 0 else {"c1": "code"}
            activity = gather_activity(year=year, actions=actions)
            tick_vitality(state, year=year, activity=activity,
                          known_subjects=known)
        assert len(state.subjects) == len(TRACKED_ACTIONS)
        assert len(state.revival_history) < 1000
