"""Tests for the Chronicle Engine — Mars-100 history organizer."""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.chronicle import (
    NormalizedYear,
    NormalizedColonist,
    Era,
    EraBoundary,
    Codex,
    normalize_year,
    normalize_colonists,
    detect_eras,
    name_era,
    extract_lessons,
    generate_testimonies,
    build_codex,
    propose_amendment,
    generate_chronicle_html,
    _compute_change_signal,
    _find_boundaries,
    _classify_boundary,
    _enforce_min_spacing,
    _validate_lispy,
    _witnessed_era,
    _firsthand_years,
    _determine_mood,
    _gather_evidence,
    _backfill_inferred_deaths,
    MIN_ERA_LENGTH,
    MAX_ERAS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_year(year: int, event_id: str = "calm", severity: float = 0.1,
               population: int = 10, gov: list[str] | None = None,
               births: list[str] | None = None, deaths: list[str] | None = None,
               meta: str | None = None, food: float = 600.0,
               water: float = 800.0) -> dict:
    """Create a synthetic year-*.json dict."""
    return {
        "year": year,
        "population": population,
        "event": {"id": event_id, "desc": f"{event_id} event", "severity": severity},
        "event_effects": [],
        "colonist_actions": [{"colonist": "Ares", "action": "Ares works on terraforming (output: 27.9)"}],
        "sub_sims": [],
        "governance_results": gov or [],
        "resource_effects": [],
        "births": births or [],
        "diary_entries": [],
        "meta_awareness": meta,
        "deaths": deaths or [],
        "resources_snapshot": {"food": food, "water": water, "power": 400.0,
                                "oxygen": 300.0, "materials": 500.0, "morale": 70.0},
    }


def _make_colonist(cid: int, name: str = "Test", element: str = "fire",
                   alive: bool = True, year_born: int = 0,
                   year_died: int | None = None,
                   cause: str | None = None,
                   faith: float = 50.0, empathy: float = 50.0) -> dict:
    """Create a synthetic colonist dict."""
    return {
        "id": cid, "name": name, "element": element,
        "stats": {"resolve": 50, "improvisation": 50, "empathy": empathy,
                  "hoarding": 30, "faith": faith, "paranoia": 20},
        "skills": {"terraforming": 40, "hydroponics": 30, "mediation": 20,
                   "coding": 50, "prayer": 10, "sabotage": 5},
        "alive": alive, "year_born": year_born,
        "year_died": year_died, "cause_of_death": cause,
        "relationships": {},
    }


def _make_100_years() -> list[dict]:
    """Generate a full 100-year timeline with realistic events."""
    years: list[dict] = []
    events = ["dust_storm", "solar_flare", "crop_blight", "ice_discovery",
              "meteorite", "hab_breach", "calm", "geothermal_vent", "aurora",
              "earth_contact", "supply_ship"]
    rng = random.Random(42)
    pop = 10
    for y in range(1, 101):
        ev = rng.choice(events)
        sev = rng.uniform(0.1, 0.8)
        gov: list[str] = []
        births: list[str] = []
        deaths: list[str] = []
        meta: str | None = None

        # Inject some governance events
        if y in (15, 30, 55, 78):
            gov = ["Amendment adopted: collective resource sharing"]
        if y in (20, 40, 60, 80):
            gov = ["Leadership election completed"]
        # Births
        if y % 8 == 0:
            births = [f"Child-{y}"]
            pop += 1
        # Deaths
        if y in (45, 46, 47, 85, 86):
            deaths = [f"Colonist-{y}"]
            pop -= 1
        # Meta
        if y >= 25 and y % 5 == 0:
            meta = f"Year {y}: colonists question the nature of their simulation"

        years.append(_make_year(year=y, event_id=ev, severity=sev,
                                population=pop, gov=gov, births=births,
                                deaths=deaths, meta=meta,
                                food=max(50.0, 600.0 - y * 2)))
    return years


def _make_colonists_list() -> list[dict]:
    """Create a diverse set of colonists matching the 100-year timeline."""
    names = ["Ares", "Lyra", "Nova", "Orion", "Vega",
             "Sol", "Iris", "Zephyr", "Luna", "Mars"]
    elements = ["fire", "water", "earth", "air"]
    result = []
    for i, name in enumerate(names):
        result.append(_make_colonist(
            i, name, elements[i % 4],
            year_born=0 if i < 5 else i * 8,
            faith=20.0 + i * 10, empathy=30.0 + i * 8))
    # Add some dead
    result.append(_make_colonist(
        20, "Fallen-1", "fire", alive=False, year_born=0, year_died=45,
        cause="dust_storm"))
    result.append(_make_colonist(
        21, "Fallen-2", "water", alive=False, year_born=10, year_died=86,
        cause="equipment_failure"))
    return result


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

class TestNormalizeYear:
    """Tests for year data normalization."""

    def test_basic_normalization(self):
        raw = _make_year(1, "dust_storm", 0.5, 10)
        ny = normalize_year(raw)
        assert ny.year == 1
        assert ny.population == 10
        assert ny.event_id == "dust_storm"
        assert ny.event_severity == 0.5

    def test_meta_awareness_string(self):
        raw = _make_year(5, meta="colonists question reality")
        ny = normalize_year(raw)
        assert ny.meta_awareness == "colonists question reality"

    def test_meta_awareness_none(self):
        raw = _make_year(5, meta=None)
        ny = normalize_year(raw)
        assert ny.meta_awareness is None

    def test_meta_awareness_null_string(self):
        raw = _make_year(5)
        raw["meta_awareness"] = "null"
        ny = normalize_year(raw)
        assert ny.meta_awareness is None

    def test_meta_awareness_dict(self):
        raw = _make_year(5)
        raw["meta_awareness"] = {"insight": "we are in a sim"}
        ny = normalize_year(raw)
        assert ny.meta_awareness == "we are in a sim"

    def test_births(self):
        raw = _make_year(10, births=["Child-A", "Child-B"])
        ny = normalize_year(raw)
        assert ny.births == ["Child-A", "Child-B"]
        assert ny.has_birth()

    def test_deaths_from_dicts(self):
        raw = _make_year(50, deaths=[{"name": "Ares", "cause": "storm"}])
        ny = normalize_year(raw)
        assert ny.deaths == ["Ares"]
        assert ny.has_death()

    def test_deaths_from_strings(self):
        raw = _make_year(50, deaths=["Ares", "Lyra"])
        ny = normalize_year(raw)
        assert ny.deaths == ["Ares", "Lyra"]

    def test_governance_change_detection(self):
        raw = _make_year(15, gov=["Amendment adopted: X"])
        ny = normalize_year(raw)
        assert ny.has_governance_change()
        assert ny.has_amendment()

    def test_no_governance_change(self):
        raw = _make_year(15)
        ny = normalize_year(raw)
        assert not ny.has_governance_change()
        assert not ny.has_amendment()

    def test_election_is_governance_change(self):
        raw = _make_year(20, gov=["Leadership election completed"])
        ny = normalize_year(raw)
        assert ny.has_governance_change()
        assert not ny.has_amendment()

    def test_missing_event_defaults(self):
        raw = {"year": 1, "population": 10}
        ny = normalize_year(raw)
        assert ny.event_id == "calm"
        assert ny.event_severity == 0.0


# ---------------------------------------------------------------------------
# Colonist normalization tests
# ---------------------------------------------------------------------------

class TestNormalizeColonists:
    """Tests for colonist normalization and deduplication."""

    def test_basic_normalization(self):
        colonists = [_make_colonist(0, "Ares"), _make_colonist(1, "Lyra")]
        result = normalize_colonists(colonists)
        assert len(result) == 2
        assert result[0].name == "Ares"

    def test_deduplication(self):
        colonists = [_make_colonist(0, "Ares")]
        dead_souls = [_make_colonist(0, "Ares", alive=False, year_died=85)]
        result = normalize_colonists(colonists, dead_souls)
        assert len(result) == 1

    def test_dead_souls_added(self):
        colonists = [_make_colonist(0, "Ares")]
        dead_souls = [_make_colonist(5, "NewDead", alive=False, year_died=50)]
        result = normalize_colonists(colonists, dead_souls)
        assert len(result) == 2

    def test_empty_inputs(self):
        result = normalize_colonists([])
        assert len(result) == 0

    def test_lived_through(self):
        c = _make_colonist(0, year_born=10, alive=False, year_died=50)
        nc = normalize_colonists([c])[0]
        assert nc.lived_through(10)
        assert nc.lived_through(30)
        assert nc.lived_through(50)
        assert not nc.lived_through(9)
        assert not nc.lived_through(51)

    def test_alive_colonist_lives_through_all(self):
        c = _make_colonist(0, year_born=0, alive=True)
        nc = normalize_colonists([c])[0]
        assert nc.lived_through(0)
        assert nc.lived_through(100)
        assert not nc.lived_through(-1)


# ---------------------------------------------------------------------------
# Backfill inferred deaths tests
# ---------------------------------------------------------------------------

class TestBackfillDeaths:
    """Tests for inferring deaths from population deltas."""

    def test_no_backfill_when_deaths_present(self):
        y1 = normalize_year(_make_year(1, population=10))
        y2 = normalize_year(_make_year(2, population=9, deaths=["Ares"]))
        years = [y1, y2]
        _backfill_inferred_deaths(years)
        assert years[1].deaths == ["Ares"]  # unchanged

    def test_backfill_on_pop_drop(self):
        y1 = normalize_year(_make_year(1, population=10))
        y2 = normalize_year(_make_year(2, population=9))
        years = [y1, y2]
        _backfill_inferred_deaths(years)
        assert len(years[1].deaths) == 1
        assert "colonist-lost-y2-1" in years[1].deaths

    def test_backfill_with_birth_offset(self):
        y1 = normalize_year(_make_year(1, population=10))
        y2 = normalize_year(_make_year(2, population=10, births=["Child-1"]))
        years = [y1, y2]
        _backfill_inferred_deaths(years)
        assert len(years[1].deaths) == 1  # pop stayed same but had a birth

    def test_no_backfill_on_growth(self):
        y1 = normalize_year(_make_year(1, population=10))
        y2 = normalize_year(_make_year(2, population=11))
        years = [y1, y2]
        _backfill_inferred_deaths(years)
        assert len(years[1].deaths) == 0

    def test_backfill_multiple_deaths(self):
        y1 = normalize_year(_make_year(1, population=10))
        y2 = normalize_year(_make_year(2, population=7))
        years = [y1, y2]
        _backfill_inferred_deaths(years)
        assert len(years[1].deaths) == 3


# ---------------------------------------------------------------------------
# Era detection tests
# ---------------------------------------------------------------------------

class TestEraDetection:
    """Tests for era boundary detection and segmentation."""

    def test_detects_eras_on_100_years(self):
        years = [normalize_year(y) for y in _make_100_years()]
        eras = detect_eras(years)
        assert len(eras) >= 2
        assert len(eras) <= MAX_ERAS

    def test_eras_cover_all_years(self):
        years = [normalize_year(y) for y in _make_100_years()]
        eras = detect_eras(years)
        covered = set()
        for era in eras:
            for y in range(era.start_year, era.end_year + 1):
                covered.add(y)
        expected = set(range(1, 101))
        assert expected == covered

    def test_eras_dont_overlap(self):
        years = [normalize_year(y) for y in _make_100_years()]
        eras = detect_eras(years)
        for i in range(len(eras) - 1):
            assert eras[i].end_year < eras[i + 1].start_year

    def test_empty_years(self):
        assert detect_eras([]) == []

    def test_single_year(self):
        ny = normalize_year(_make_year(1))
        eras = detect_eras([ny])
        assert len(eras) == 1
        assert eras[0].start_year == 1
        assert eras[0].end_year == 1

    def test_min_era_length_respected(self):
        years = [normalize_year(y) for y in _make_100_years()]
        eras = detect_eras(years)
        # First and last eras can be shorter due to boundary placement
        for era in eras[1:-1]:
            assert era.duration() >= MIN_ERA_LENGTH - 1

    def test_era_has_population_range(self):
        years = [normalize_year(y) for y in _make_100_years()]
        eras = detect_eras(years)
        for era in eras:
            assert era.population_range[0] <= era.population_range[1]

    def test_era_accumulates_births_deaths(self):
        years_raw = _make_100_years()
        years = [normalize_year(y) for y in years_raw]
        eras = detect_eras(years)
        total_births = sum(len(e.births) for e in eras)
        total_deaths = sum(len(e.deaths) for e in eras)
        expected_births = sum(len(normalize_year(y).births) for y in years_raw)
        expected_deaths = sum(len(normalize_year(y).deaths) for y in years_raw)
        assert total_births == expected_births
        assert total_deaths == expected_deaths


# ---------------------------------------------------------------------------
# Change signal tests
# ---------------------------------------------------------------------------

class TestChangeSignal:
    """Tests for the composite change signal computation."""

    def test_calm_years_low_signal(self):
        years = [normalize_year(_make_year(y)) for y in range(1, 6)]
        signal = _compute_change_signal(years)
        assert all(s < 2.0 for s in signal)

    def test_amendment_year_high_signal(self):
        years = [normalize_year(_make_year(y)) for y in range(1, 6)]
        years[2] = normalize_year(_make_year(3, gov=["Amendment adopted: X"]))
        signal = _compute_change_signal(years)
        assert signal[2] > signal[0]

    def test_death_increases_signal(self):
        years = [normalize_year(_make_year(y, population=10)) for y in range(1, 6)]
        years[2] = normalize_year(_make_year(3, population=8,
                                              deaths=["A", "B"]))
        signal = _compute_change_signal(years)
        assert signal[2] > signal[0]

    def test_meta_awareness_increases_signal(self):
        years = [normalize_year(_make_year(y)) for y in range(1, 6)]
        years[2] = normalize_year(_make_year(3, meta="insight!"))
        signal = _compute_change_signal(years)
        assert signal[2] > signal[0]


# ---------------------------------------------------------------------------
# Era naming tests
# ---------------------------------------------------------------------------

class TestEraNaming:
    """Tests for era name generation."""

    def test_amendment_causes_reformation(self):
        name = name_era(["dust_storm"], 1, 20, ["Amendment I"], 0, [])
        assert "Reformation" in name

    def test_mass_death_causes_reckoning(self):
        name = name_era(["dust_storm"], 1, 20, [], 0, ["A", "B", "C"])
        assert "Reckoning" in name

    def test_meta_awareness_causes_awakening(self):
        name = name_era(["calm"], 1, 20, [], 5, [])
        assert "Awakening" in name

    def test_crisis_event(self):
        name = name_era(["dust_storm"], 1, 20, [], 0, [])
        assert "Trial" in name

    def test_adjective_from_event(self):
        name = name_era(["ice_discovery"], 1, 20, [], 0, [])
        assert "Frozen" in name

    def test_default_quiet(self):
        name = name_era([], 1, 20, [], 0, [])
        assert "Quiet" in name


# ---------------------------------------------------------------------------
# Lesson extraction tests
# ---------------------------------------------------------------------------

class TestLessonExtraction:
    """Tests for LisPy lesson extraction."""

    def test_food_crisis_lesson(self):
        era = Era(start_year=1, end_year=20, name="Test", boundaries=[],
                  dominant_events=["crop_blight"], population_range=(8, 12),
                  governance_changes=[], amendments=[], deaths=[], births=[],
                  meta_awareness_count=0,
                  resource_crises=[{"year": 5, "resource": "food", "level": 10}],
                  lessons=[], testimonies=[])
        lessons = extract_lessons(era)
        assert any(l["category"] == "resource_management" for l in lessons)

    def test_governance_lesson_on_amendment(self):
        era = Era(start_year=1, end_year=20, name="Test", boundaries=[],
                  dominant_events=["calm"], population_range=(10, 12),
                  governance_changes=["Amendment adopted"],
                  amendments=["Amendment I"], deaths=[], births=[],
                  meta_awareness_count=0, resource_crises=[],
                  lessons=[], testimonies=[])
        lessons = extract_lessons(era)
        assert any(l["category"] == "governance" for l in lessons)

    def test_all_lessons_are_valid_lispy(self):
        """Every lesson template must produce executable LisPy."""
        era = Era(start_year=1, end_year=20, name="Test", boundaries=[],
                  dominant_events=["calm"], population_range=(8, 25),
                  governance_changes=["Amendment adopted"],
                  amendments=["Amendment I"],
                  deaths=["A", "B", "C"], births=["X", "Y", "Z"],
                  meta_awareness_count=5,
                  resource_crises=[
                      {"year": 5, "resource": "food", "level": 10},
                      {"year": 6, "resource": "oxygen", "level": 5},
                  ],
                  lessons=[], testimonies=[])
        lessons = extract_lessons(era)
        assert len(lessons) >= 3
        for lesson in lessons:
            assert lesson["executable"] is True, f"Lesson not executable: {lesson['lispy']}"

    def test_no_lessons_for_calm_era(self):
        era = Era(start_year=1, end_year=5, name="Test", boundaries=[],
                  dominant_events=["calm"], population_range=(10, 10),
                  governance_changes=[], amendments=[], deaths=[], births=[],
                  meta_awareness_count=0, resource_crises=[],
                  lessons=[], testimonies=[])
        lessons = extract_lessons(era)
        assert len(lessons) == 0

    def test_lispy_validation(self):
        assert _validate_lispy('(+ 1 2)') is True
        assert _validate_lispy('(undefined-function)') is False


# ---------------------------------------------------------------------------
# Testimony tests
# ---------------------------------------------------------------------------

class TestTestimonies:
    """Tests for colonist testimony generation."""

    def test_basic_testimony(self):
        colonists = normalize_colonists([_make_colonist(0, "Ares", year_born=0)])
        era = Era(start_year=1, end_year=20, name="The Test Era", boundaries=[],
                  dominant_events=["calm"], population_range=(10, 12),
                  governance_changes=[], amendments=[], deaths=[], births=[],
                  meta_awareness_count=0, resource_crises=[],
                  lessons=[], testimonies=[])
        rng = random.Random(42)
        testimonies = generate_testimonies(era, colonists, rng)
        assert len(testimonies) == 1
        t = testimonies[0]
        assert t["colonist_name"] == "Ares"
        assert t["firsthand"] is True
        assert "The Test Era" in t["text"]

    def test_testimony_bounded_by_lifespan(self):
        colonists = normalize_colonists([
            _make_colonist(0, "Early", year_born=0, alive=False, year_died=10),
            _make_colonist(1, "Late", year_born=50),
        ])
        era = Era(start_year=40, end_year=60, name="Late Era", boundaries=[],
                  dominant_events=["calm"], population_range=(10, 12),
                  governance_changes=[], amendments=[], deaths=[], births=[],
                  meta_awareness_count=0, resource_crises=[],
                  lessons=[], testimonies=[])
        rng = random.Random(42)
        testimonies = generate_testimonies(era, colonists, rng)
        names = [t["colonist_name"] for t in testimonies]
        assert "Late" in names
        assert "Early" not in names

    def test_no_witnesses(self):
        colonists = normalize_colonists([
            _make_colonist(0, alive=False, year_born=0, year_died=5)])
        era = Era(start_year=50, end_year=60, name="Later", boundaries=[],
                  dominant_events=["calm"], population_range=(10, 12),
                  governance_changes=[], amendments=[], deaths=[], births=[],
                  meta_awareness_count=0, resource_crises=[],
                  lessons=[], testimonies=[])
        rng = random.Random(42)
        testimonies = generate_testimonies(era, colonists, rng)
        assert len(testimonies) == 0

    def test_testimony_has_evidence(self):
        colonists = normalize_colonists([_make_colonist(0, "Ares", year_born=0)])
        era = Era(start_year=1, end_year=20, name="Test", boundaries=[],
                  dominant_events=["calm"], population_range=(10, 12),
                  governance_changes=[], amendments=[], deaths=[], births=[],
                  meta_awareness_count=0, resource_crises=[],
                  lessons=[], testimonies=[])
        rng = random.Random(42)
        testimonies = generate_testimonies(era, colonists, rng)
        assert len(testimonies[0]["evidence"]) >= 1

    def test_death_in_era_noted_as_evidence(self):
        colonists = normalize_colonists([
            _make_colonist(0, "Fallen", year_born=0, alive=False,
                          year_died=15, cause="dust_storm")])
        era = Era(start_year=10, end_year=20, name="Test", boundaries=[],
                  dominant_events=["calm"], population_range=(9, 10),
                  governance_changes=[], amendments=[], deaths=["Fallen"],
                  births=[], meta_awareness_count=0, resource_crises=[],
                  lessons=[], testimonies=[])
        rng = random.Random(42)
        testimonies = generate_testimonies(era, colonists, rng)
        evidence = testimonies[0]["evidence"]
        assert any("Died" in e for e in evidence)

    def test_mood_somber_on_deaths(self):
        colonists = normalize_colonists([_make_colonist(0, year_born=0)])
        era = Era(start_year=1, end_year=20, name="Test", boundaries=[],
                  dominant_events=["calm"], population_range=(8, 10),
                  governance_changes=[], amendments=[],
                  deaths=["A", "B"], births=[],
                  meta_awareness_count=0, resource_crises=[],
                  lessons=[], testimonies=[])
        rng = random.Random(42)
        testimonies = generate_testimonies(era, colonists, rng)
        assert testimonies[0]["mood"] == "somber"

    def test_mood_philosophical_on_meta(self):
        colonists = normalize_colonists([_make_colonist(0, year_born=0)])
        era = Era(start_year=1, end_year=20, name="Test", boundaries=[],
                  dominant_events=["calm"], population_range=(10, 12),
                  governance_changes=[], amendments=[], deaths=[], births=[],
                  meta_awareness_count=5, resource_crises=[],
                  lessons=[], testimonies=[])
        rng = random.Random(42)
        testimonies = generate_testimonies(era, colonists, rng)
        assert testimonies[0]["mood"] == "philosophical"


# ---------------------------------------------------------------------------
# Codex builder tests
# ---------------------------------------------------------------------------

class TestCodexBuilder:
    """Tests for the full codex construction."""

    def test_builds_codex_from_100_years(self):
        years = _make_100_years()
        colonists = _make_colonists_list()
        codex = build_codex(years, colonists)
        assert codex.total_years == 100
        assert len(codex.eras) >= 2

    def test_codex_to_dict_has_meta(self):
        years = _make_100_years()
        colonists = _make_colonists_list()
        codex = build_codex(years, colonists)
        d = codex.to_dict()
        assert "_meta" in d
        assert d["_meta"]["engine"] == "chronicle"
        assert d["_meta"]["total_years"] == 100

    def test_codex_deterministic(self):
        years = _make_100_years()
        colonists = _make_colonists_list()
        c1 = build_codex(years, colonists, seed=42)
        c2 = build_codex(years, colonists, seed=42)
        assert c1.to_dict() == c2.to_dict()

    def test_codex_different_seeds_differ(self):
        years = _make_100_years()
        colonists = _make_colonists_list()
        c1 = build_codex(years, colonists, seed=42)
        c2 = build_codex(years, colonists, seed=99)
        # Testimonies use RNG, so should differ
        t1 = [t for e in c1.eras for t in e.testimonies]
        t2 = [t for e in c2.eras for t in e.testimonies]
        assert t1 != t2

    def test_codex_with_dead_souls(self):
        years = _make_100_years()
        colonists = _make_colonists_list()[:5]
        dead = [_make_colonist(99, "Ghost", alive=False, year_born=0, year_died=50)]
        codex = build_codex(years, colonists, dead_souls=dead)
        assert codex.total_colonists == 6

    def test_codex_with_governance(self):
        years = _make_100_years()
        colonists = _make_colonists_list()
        gov = {"system": "direct_democracy", "leader": "Ares",
               "amendments": ["I", "II"]}
        codex = build_codex(years, colonists, governance=gov)
        assert codex.governance_summary["final_system"] == "direct_democracy"

    def test_codex_empty_years(self):
        codex = build_codex([], [])
        assert codex.total_years == 0
        assert len(codex.eras) == 0


# ---------------------------------------------------------------------------
# Amendment proposal tests
# ---------------------------------------------------------------------------

class TestAmendmentProposal:
    """Tests for constitutional amendment derivation."""

    def test_proposes_amendment(self):
        years = _make_100_years()
        colonists = _make_colonists_list()
        codex = build_codex(years, colonists)
        assert codex.amendment_proposal is not None

    def test_amendment_has_confidence(self):
        years = _make_100_years()
        colonists = _make_colonists_list()
        codex = build_codex(years, colonists)
        assert 0 < codex.amendment_proposal["confidence"] <= 1.0

    def test_amendment_has_lispy_encoding(self):
        years = _make_100_years()
        colonists = _make_colonists_list()
        codex = build_codex(years, colonists)
        assert "lispy_encoding" in codex.amendment_proposal
        assert codex.amendment_proposal["lispy_encoding"].startswith("(list")

    def test_no_amendment_from_empty(self):
        result = propose_amendment([], [], {})
        assert result is None


# ---------------------------------------------------------------------------
# HTML generation tests
# ---------------------------------------------------------------------------

class TestHTMLGeneration:
    """Tests for chronicle HTML output."""

    def test_generates_html(self):
        years = _make_100_years()
        colonists = _make_colonists_list()
        codex = build_codex(years, colonists)
        html = generate_chronicle_html(codex)
        assert "<html" in html
        assert "Mars-100" in html

    def test_html_contains_era_data(self):
        years = _make_100_years()
        colonists = _make_colonists_list()
        codex = build_codex(years, colonists)
        html = generate_chronicle_html(codex)
        for era in codex.eras:
            assert era.name in html

    def test_html_contains_amendment(self):
        years = _make_100_years()
        colonists = _make_colonists_list()
        codex = build_codex(years, colonists)
        html = generate_chronicle_html(codex)
        assert "Proposed Amendment" in html


# ---------------------------------------------------------------------------
# Integration tests with real data
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestIntegration:
    """Integration tests using real Mars-100 data files."""

    @pytest.fixture
    def real_data(self):
        state_path = REPO_ROOT / "state" / "mars100.json"
        if not state_path.exists():
            pytest.skip("No real state data")
        state = json.loads(state_path.read_text())

        years = []
        for y in range(1, 101):
            yf = REPO_ROOT / "docs" / "mars-100" / f"year-{y}.json"
            if yf.exists():
                years.append(json.loads(yf.read_text()))
        if not years:
            pytest.skip("No year data files")

        return {
            "years": years,
            "colonists": state.get("colonists", []),
            "dead_souls": state.get("dead_souls", []),
            "governance": state.get("governance", {}),
        }

    def test_builds_codex_from_real_data(self, real_data):
        codex = build_codex(**real_data)
        assert codex.total_years == 100
        assert len(codex.eras) >= 2
        assert codex.total_colonists > 0
        assert codex.amendment_proposal is not None

    def test_real_data_eras_cover_all_years(self, real_data):
        codex = build_codex(**real_data)
        covered = set()
        for era in codex.eras:
            for y in range(era.start_year, era.end_year + 1):
                covered.add(y)
        expected = set(range(1, 101))
        assert expected == covered

    def test_real_codex_serializable(self, real_data):
        codex = build_codex(**real_data)
        d = codex.to_dict()
        serialized = json.dumps(d)
        roundtrip = json.loads(serialized)
        assert roundtrip["_meta"]["total_years"] == 100

    def test_real_html_generates(self, real_data):
        codex = build_codex(**real_data)
        html = generate_chronicle_html(codex)
        assert len(html) > 1000
        assert "Mars-100" in html

    def test_real_data_has_inferred_deaths(self, real_data):
        """Real data should have inferred deaths from population drops."""
        codex = build_codex(**real_data)
        assert codex.total_deaths > 0, "Expected inferred deaths from population deltas"
