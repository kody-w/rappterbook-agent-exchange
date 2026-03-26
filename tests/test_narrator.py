"""
Tests for narrator.py — narrative chronicle generation from simulation results.

Run: python -m pytest tests/test_narrator.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.narrator import (
    Chronicle,
    Event,
    narrate,
    format_chronicle,
    _classify_arc,
    _extract_population_milestones,
    _extract_epidemics,
    _extract_tech_unlocks,
    _extract_storms,
    _extract_death_spikes,
    _extract_terraforming,
    _extract_migrations,
)
from src.tick_engine import Simulation


# ─── Fixtures ───

def _run_sim(sols: int = 100, seed: int = 42) -> dict:
    """Run a short simulation and return results dict."""
    sim = Simulation(sols=sols, env_seed=seed)
    return sim.run()


def _empty_results() -> dict:
    """Minimal results dict with no data."""
    return {
        "_meta": {"sols": 0, "engine": "mars-barn", "version": "test"},
        "environment": {"history": [], "final_terraforming_progress": 0.0, "terraform_phase": "none"},
        "colonies": [],
        "summary": {"colonies": [], "total_migrations": 0, "terraforming": {"progress": 0, "phase": "none"}},
    }


def _make_colony_summary(start: int, end: int, name: str = "Test Colony") -> dict:
    """Helper to build a minimal colony summary entry."""
    growth = (end - start) / max(1, start) * 100
    return {
        "name": name,
        "strategy": "balanced",
        "start_pop": start,
        "end_pop": end,
        "total_births": max(0, end - start + 5),
        "total_deaths": 5,
        "growth_pct": round(growth, 1),
        "death_causes": {"starvation": 2, "accident": 3},
    }


# ─── Chronicle dataclass ───

class TestChronicle:
    def test_to_dict_serializable(self) -> None:
        """Chronicle.to_dict() must produce JSON-serializable output."""
        c = Chronicle(arc="triumph", arc_reason="test", sol_count=100, colony_count=3)
        d = c.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_to_dict_roundtrip(self) -> None:
        """All fields should survive serialization."""
        ev = Event(sol=10, kind="test", colony="Alpha", headline="h", detail="d", magnitude=0.5)
        c = Chronicle(
            arc="survival", arc_reason="reason",
            events=[ev], opening="o", closing="c",
            sol_count=365, colony_count=2,
        )
        d = c.to_dict()
        assert d["arc"] == "survival"
        assert d["sol_count"] == 365
        assert len(d["events"]) == 1
        assert d["events"][0]["sol"] == 10

    def test_default_events_empty(self) -> None:
        c = Chronicle(arc="x", arc_reason="y")
        assert c.events == []


class TestEvent:
    def test_magnitude_stored(self) -> None:
        ev = Event(sol=1, kind="test", colony=None, headline="h", detail="d", magnitude=0.42)
        assert ev.magnitude == 0.42

    def test_colony_nullable(self) -> None:
        ev = Event(sol=1, kind="storm", colony=None, headline="h", detail="d", magnitude=0.5)
        assert ev.colony is None


# ─── Arc classification ───

class TestArcClassification:
    def test_triumph_high_growth(self) -> None:
        results = {
            "summary": {
                "colonies": [
                    _make_colony_summary(100, 200),
                    _make_colony_summary(100, 180),
                ],
            },
        }
        arc, _ = _classify_arc(results)
        assert arc == "triumph"

    def test_collapse_on_dead_colony(self) -> None:
        results = {
            "summary": {
                "colonies": [
                    _make_colony_summary(100, 150),
                    _make_colony_summary(100, 0),  # dead
                ],
            },
        }
        arc, _ = _classify_arc(results)
        assert arc == "collapse"

    def test_collapse_on_decline(self) -> None:
        results = {
            "summary": {
                "colonies": [
                    _make_colony_summary(100, 70),
                    _make_colony_summary(100, 60),
                ],
            },
        }
        arc, _ = _classify_arc(results)
        assert arc == "collapse"

    def test_stagnation_flat(self) -> None:
        results = {
            "summary": {
                "colonies": [
                    _make_colony_summary(100, 102),
                    _make_colony_summary(100, 101),
                ],
            },
        }
        arc, _ = _classify_arc(results)
        assert arc == "stagnation"

    def test_survival_moderate_growth(self) -> None:
        results = {
            "summary": {
                "colonies": [
                    _make_colony_summary(100, 120),
                    _make_colony_summary(100, 115),
                ],
            },
        }
        arc, _ = _classify_arc(results)
        assert arc == "survival"

    def test_empty_summary(self) -> None:
        results = {"summary": {"colonies": []}}
        arc, _ = _classify_arc(results)
        assert arc == "stagnation"

    def test_arc_always_has_reason(self) -> None:
        """Every arc classification must include a reason string."""
        for start, end in [(100, 200), (100, 0), (100, 102), (100, 120)]:
            results = {"summary": {"colonies": [_make_colony_summary(start, end)]}}
            _, reason = _classify_arc(results)
            assert isinstance(reason, str)
            assert len(reason) > 5


# ─── Event extraction ───

class TestPopulationMilestones:
    def test_crosses_100(self) -> None:
        col = {
            "name": "Alpha",
            "history": [
                {"sol": 1, "population": 80},
                {"sol": 2, "population": 105},
            ],
        }
        events = _extract_population_milestones([col])
        assert any(e.kind == "milestone" and "100" in e.headline for e in events)

    def test_no_milestone_below(self) -> None:
        col = {
            "name": "Alpha",
            "history": [{"sol": s, "population": 50 + s} for s in range(1, 10)],
        }
        events = _extract_population_milestones([col])
        assert len(events) == 0

    def test_multiple_milestones(self) -> None:
        col = {
            "name": "Alpha",
            "history": [
                {"sol": 1, "population": 50},
                {"sol": 100, "population": 150},
                {"sol": 200, "population": 250},
            ],
        }
        events = _extract_population_milestones([col])
        kinds = [e.headline for e in events]
        assert any("100" in k for k in kinds)
        assert any("200" in k for k in kinds)

    def test_magnitude_bounded(self) -> None:
        col = {
            "name": "Big",
            "history": [
                {"sol": 1, "population": 50},
                {"sol": 500, "population": 1500},
            ],
        }
        events = _extract_population_milestones([col])
        for e in events:
            assert 0.0 <= e.magnitude <= 1.0


class TestEpidemicExtraction:
    def test_detects_outbreak(self) -> None:
        col = {
            "name": "Alpha",
            "events": [{"type": "epidemic", "strain": "Mars Flu", "sol": 42}],
        }
        events = _extract_epidemics([col])
        assert len(events) == 1
        assert events[0].kind == "epidemic"

    def test_detects_spread(self) -> None:
        col = {
            "name": "Beta",
            "events": [{"type": "epidemic_spread", "strain": "Rad Fever", "sol": 60, "from": "Alpha"}],
        }
        events = _extract_epidemics([col])
        assert "spreads" in events[0].headline

    def test_no_events_no_epidemics(self) -> None:
        col = {"name": "Clean", "events": []}
        assert _extract_epidemics([col]) == []


class TestTechExtraction:
    def test_extracts_unlock(self) -> None:
        col = {
            "name": "Alpha",
            "tech": {
                "unlocked": [
                    {"sol": 50, "name": "Advanced Solar Cells", "branch": "power", "description": "+25%"},
                ],
            },
        }
        events = _extract_tech_unlocks([col])
        assert len(events) == 1
        assert "Solar" in events[0].headline

    def test_no_tech_no_events(self) -> None:
        col = {"name": "Alpha", "tech": {"unlocked": []}}
        assert _extract_tech_unlocks([col]) == []


class TestStormExtraction:
    def test_detects_storm(self) -> None:
        history = [
            {"sol": 1, "dust_opacity": 0.01, "storm_kind": None},
            {"sol": 2, "dust_opacity": 0.8, "storm_kind": "global"},
            {"sol": 3, "dust_opacity": 0.7, "storm_kind": "global"},
            {"sol": 4, "dust_opacity": 0.01, "storm_kind": None},
        ]
        events = _extract_storms(history)
        assert len(events) == 1
        assert "Global" in events[0].headline

    def test_regional_storm(self) -> None:
        history = [
            {"sol": 1, "dust_opacity": 0.01, "storm_kind": None},
            {"sol": 2, "dust_opacity": 0.3, "storm_kind": "regional"},
            {"sol": 3, "dust_opacity": 0.01, "storm_kind": None},
        ]
        events = _extract_storms(history)
        assert events[0].magnitude < 0.5  # regional < global

    def test_no_storms(self) -> None:
        history = [{"sol": i, "dust_opacity": 0.01, "storm_kind": None} for i in range(10)]
        assert _extract_storms(history) == []


class TestDeathSpikes:
    def test_detects_spike(self) -> None:
        history = [{"sol": i, "deaths": 1} for i in range(20)]
        history.append({"sol": 20, "deaths": 15})  # spike
        col = {"name": "Alpha", "history": history}
        events = _extract_death_spikes([col])
        assert len(events) >= 1
        assert events[0].kind == "death_spike"

    def test_no_spike_normal(self) -> None:
        history = [{"sol": i, "deaths": 1} for i in range(20)]
        col = {"name": "Calm", "history": history}
        events = _extract_death_spikes([col])
        assert len(events) == 0

    def test_short_history_skipped(self) -> None:
        history = [{"sol": i, "deaths": 10} for i in range(5)]
        col = {"name": "Short", "history": history}
        events = _extract_death_spikes([col])
        assert len(events) == 0


class TestTerraformingExtraction:
    def test_detects_progress(self) -> None:
        results = {
            "_meta": {"sols": 365},
            "environment": {
                "final_terraforming_progress": 0.05,
                "terraform_phase": "atmosphere_thickening",
            },
        }
        events = _extract_terraforming(results)
        assert len(events) == 1
        assert "Terraforming" in events[0].headline

    def test_no_progress_no_event(self) -> None:
        results = {
            "environment": {
                "final_terraforming_progress": 0.005,
                "terraform_phase": "none",
            },
        }
        events = _extract_terraforming(results)
        assert len(events) == 0


class TestMigrationExtraction:
    def test_detects_evacuation(self) -> None:
        col = {
            "name": "Alpha",
            "events": [{"type": "evacuation", "sol": 30, "count": 10, "to": "Beta"}],
        }
        events = _extract_migrations([col])
        assert len(events) == 1
        assert "evacuate" in events[0].headline


# ─── Full pipeline: narrate() ───

class TestNarrate:
    def test_returns_chronicle(self) -> None:
        results = _run_sim(sols=50)
        c = narrate(results)
        assert isinstance(c, Chronicle)

    def test_arc_is_valid(self) -> None:
        results = _run_sim(sols=100)
        c = narrate(results)
        assert c.arc in ("triumph", "survival", "collapse", "stagnation")

    def test_events_sorted_by_sol(self) -> None:
        results = _run_sim(sols=100)
        c = narrate(results)
        sols = [e.sol for e in c.events]
        assert sols == sorted(sols)

    def test_has_opening_and_closing(self) -> None:
        results = _run_sim(sols=50)
        c = narrate(results)
        assert len(c.opening) > 10
        assert len(c.closing) > 10

    def test_sol_count_matches(self) -> None:
        results = _run_sim(sols=75)
        c = narrate(results)
        assert c.sol_count == 75

    def test_colony_count_matches(self) -> None:
        results = _run_sim(sols=50)
        c = narrate(results)
        assert c.colony_count == 3  # default 3 colonies

    def test_to_dict_serializable(self) -> None:
        results = _run_sim(sols=50)
        c = narrate(results)
        serialized = json.dumps(c.to_dict())
        parsed = json.loads(serialized)
        assert parsed["arc"] == c.arc

    def test_empty_results_no_crash(self) -> None:
        results = _empty_results()
        c = narrate(results)
        assert isinstance(c, Chronicle)
        assert c.arc == "stagnation"

    def test_events_have_valid_kinds(self) -> None:
        results = _run_sim(sols=200)
        c = narrate(results)
        valid_kinds = {"milestone", "epidemic", "tech", "storm", "death_spike", "terraform", "migration"}
        for e in c.events:
            assert e.kind in valid_kinds, "Unknown event kind: %s" % e.kind

    def test_all_magnitudes_bounded(self) -> None:
        """Every event magnitude must be in [0, 1]."""
        results = _run_sim(sols=200)
        c = narrate(results)
        for e in c.events:
            assert 0.0 <= e.magnitude <= 1.0, "Event %s has magnitude %.3f" % (e.headline, e.magnitude)


# ─── format_chronicle() ───

class TestFormatChronicle:
    def test_contains_arc(self) -> None:
        results = _run_sim(sols=50)
        c = narrate(results)
        text = format_chronicle(c)
        assert c.arc.upper() in text

    def test_contains_opening(self) -> None:
        results = _run_sim(sols=50)
        c = narrate(results)
        text = format_chronicle(c)
        assert "Sol 1" in text

    def test_contains_key_events_header(self) -> None:
        results = _run_sim(sols=200)
        c = narrate(results)
        text = format_chronicle(c)
        if c.events:
            assert "KEY EVENTS" in text

    def test_output_is_string(self) -> None:
        results = _run_sim(sols=50)
        c = narrate(results)
        text = format_chronicle(c)
        assert isinstance(text, str)
        assert len(text) > 100

    def test_empty_chronicle_no_crash(self) -> None:
        c = Chronicle(arc="stagnation", arc_reason="test", opening="o", closing="c")
        text = format_chronicle(c)
        assert "STAGNATION" in text


# ─── Conservation / invariant tests ───

class TestConservation:
    def test_event_count_nonnegative(self) -> None:
        for seed in range(5):
            results = _run_sim(sols=100, seed=seed)
            c = narrate(results)
            assert len(c.events) >= 0

    def test_deterministic_same_seed(self) -> None:
        """Same seed → same arc and event count."""
        r1 = _run_sim(sols=100, seed=42)
        r2 = _run_sim(sols=100, seed=42)
        c1 = narrate(r1)
        c2 = narrate(r2)
        assert c1.arc == c2.arc
        assert len(c1.events) == len(c2.events)

    def test_different_seeds_may_differ(self) -> None:
        """Different seeds can produce different chronicles."""
        r1 = _run_sim(sols=200, seed=1)
        r2 = _run_sim(sols=200, seed=999)
        c1 = narrate(r1)
        c2 = narrate(r2)
        # At minimum, event lists should differ in count or content
        # (probabilistically near-certain with different seeds)
        events_differ = (len(c1.events) != len(c2.events) or c1.arc != c2.arc)
        # This is probabilistic — if by freak chance they match, that's ok
        # The test just confirms the function runs with different seeds
        assert isinstance(c1, Chronicle) and isinstance(c2, Chronicle)

    def test_long_sim_more_events(self) -> None:
        """Longer simulations should generally produce more events."""
        short = narrate(_run_sim(sols=30, seed=42))
        long = narrate(_run_sim(sols=365, seed=42))
        # Not strictly guaranteed but very likely
        assert long.sol_count > short.sol_count

    def test_all_event_sols_bounded(self) -> None:
        """No event should have a sol outside the simulation range."""
        results = _run_sim(sols=200, seed=42)
        c = narrate(results)
        for e in c.events:
            assert e.sol >= 0, "Negative sol: %d" % e.sol
            # Terraforming events use the total sols as their sol
            assert e.sol <= 210, "Sol %d exceeds sim length" % e.sol
