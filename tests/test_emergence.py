"""Tests for emergence pattern analyzer."""
from __future__ import annotations

import pytest
from src.mars100.emergence import (
    GovernancePhase, MortalityPattern, ConvergenceTrend, Faction, Insight,
    EmergenceReport,
    analyze_governance_phases, analyze_mortality, analyze_convergence,
    analyze_factions, analyze_crisis_resilience, analyze_subsim_accuracy,
    synthesize_insights, propose_amendment, analyze,
    _stddev,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_year(year: int, **overrides) -> dict:
    """Create a minimal year record for testing."""
    base = {
        "year": year,
        "events": [{"name": "dust_storm", "severity": 0.5,
                     "description": "storm", "category": "environment",
                     "effects": {"power": -0.1}}],
        "actions": {"c1": "farm", "c2": "terraform"},
        "subsim_log": [{"depth": 1, "colonist_id": "c1", "year": year,
                         "expression": "(+ 1 2)", "result": 0.6}],
        "governance": None,
        "governance_state": {"gov_type": "anarchy"},
        "resources_before": {"food": 0.6, "water": 0.7, "power": 0.8,
                              "air": 0.9, "medicine": 0.5},
        "resources_after": {"food": 0.55, "water": 0.65, "power": 0.75,
                             "air": 0.85, "medicine": 0.45},
        "resource_delta": {"food": -0.05, "water": -0.05, "power": -0.05,
                            "air": -0.05, "medicine": -0.05},
        "deaths": [],
        "exiles": [],
        "meta_awareness": [],
        "social_cohesion": 0.55,
        "colonist_snapshots": [],
        "convergence": {"resolve": 0.1, "improvisation": 0.12,
                         "empathy": 0.08, "hoarding": 0.15,
                         "faith": 0.11, "paranoia": 0.09},
    }
    base.update(overrides)
    return base


def _make_colonist(cid: str, alive: bool = True, **kwargs) -> dict:
    """Create a minimal colonist dict for testing."""
    base = {
        "id": cid, "name": f"Colonist-{cid}", "element": "fire",
        "archetype": "commander", "alive": alive, "exiled": False,
        "birth_year": 0,
        "stats": {"resolve": 0.5, "improvisation": 0.5, "empathy": 0.5,
                  "hoarding": 0.5, "faith": 0.5, "paranoia": 0.5},
        "skills": {"terraforming": 0.5, "hydroponics": 0.5, "mediation": 0.5,
                   "coding": 0.5, "prayer": 0.5, "sabotage": 0.0},
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Tests: _stddev
# ---------------------------------------------------------------------------

class TestStddev:
    def test_single_value(self):
        assert _stddev([5.0]) == 0.0

    def test_identical_values(self):
        assert _stddev([3.0, 3.0, 3.0]) == 0.0

    def test_known_stddev(self):
        # [1, 3] → mean=2, var=1, stddev=1
        assert abs(_stddev([1.0, 3.0]) - 1.0) < 0.001

    def test_empty_list(self):
        assert _stddev([]) == 0.0


# ---------------------------------------------------------------------------
# Tests: Governance phases
# ---------------------------------------------------------------------------

class TestGovernancePhases:
    def test_empty_years(self):
        assert analyze_governance_phases([]) == []

    def test_single_phase(self):
        years = [_make_year(y) for y in range(1, 11)]
        phases = analyze_governance_phases(years)
        assert len(phases) == 1
        assert phases[0].gov_type == "anarchy"
        assert phases[0].start_year == 1

    def test_phase_transition(self):
        years = [_make_year(y) for y in range(1, 6)]
        # Transition at year 6
        years += [_make_year(y, governance_state={"gov_type": "council"})
                  for y in range(6, 11)]
        phases = analyze_governance_phases(years)
        assert len(phases) == 2
        assert phases[0].gov_type == "anarchy"
        assert phases[0].end_year == 6
        assert phases[1].gov_type == "council"
        assert phases[1].start_year == 6

    def test_multiple_transitions(self):
        years = (
            [_make_year(y) for y in range(1, 4)]
            + [_make_year(y, governance_state={"gov_type": "council"})
               for y in range(4, 7)]
            + [_make_year(y, governance_state={"gov_type": "consensus"})
               for y in range(7, 11)]
        )
        phases = analyze_governance_phases(years)
        assert len(phases) == 3
        assert [p.gov_type for p in phases] == ["anarchy", "council", "consensus"]

    def test_crisis_tracking(self):
        years = [_make_year(y, events=[{"name": "solar_flare", "severity": 0.8,
                                         "description": "big", "category": "cosmic",
                                         "effects": {}}])
                 for y in range(1, 6)]
        phases = analyze_governance_phases(years)
        assert phases[0].crises_weathered == 5

    def test_death_tracking(self):
        years = [_make_year(1, deaths=[{"id": "c1", "name": "A", "cause": "x"}])]
        phases = analyze_governance_phases(years)
        assert phases[0].deaths_during == 1

    def test_to_dict(self):
        phase = GovernancePhase(gov_type="council", start_year=1, end_year=10,
                                duration=9, avg_cohesion=0.5)
        d = phase.to_dict()
        assert d["gov_type"] == "council"
        assert d["duration"] == 9


# ---------------------------------------------------------------------------
# Tests: Mortality
# ---------------------------------------------------------------------------

class TestMortality:
    def test_no_deaths(self):
        colonists = [_make_colonist("c1"), _make_colonist("c2")]
        m = analyze_mortality([], colonists)
        assert m.total_deaths == 0

    def test_single_cause(self):
        colonists = [
            _make_colonist("c1", alive=False, death_year=50, death_cause="asphyxiation"),
            _make_colonist("c2", alive=False, death_year=60, death_cause="asphyxiation"),
        ]
        m = analyze_mortality([], colonists)
        assert m.total_deaths == 2
        assert m.causes == {"asphyxiation": 2}
        assert m.avg_death_year == 55.0
        assert "asphyxiation" in m.systemic_cause

    def test_mixed_causes(self):
        colonists = [
            _make_colonist("c1", alive=False, death_year=30, death_cause="radiation"),
            _make_colonist("c2", alive=False, death_year=50, death_cause="starvation"),
            _make_colonist("c3", alive=False, death_year=70, death_cause="disease"),
        ]
        m = analyze_mortality([], colonists)
        assert m.total_deaths == 3
        assert "distributed" in m.systemic_cause

    def test_deaths_per_decade(self):
        colonists = [
            _make_colonist("c1", alive=False, death_year=85, death_cause="x"),
            _make_colonist("c2", alive=False, death_year=88, death_cause="x"),
            _make_colonist("c3", alive=False, death_year=92, death_cause="x"),
        ]
        m = analyze_mortality([], colonists)
        assert m.deaths_per_decade[8] == 2  # decade 81-90
        assert m.deaths_per_decade[9] == 1  # decade 91-100

    def test_to_dict(self):
        m = MortalityPattern(total_deaths=3, causes={"x": 3})
        d = m.to_dict()
        assert d["total_deaths"] == 3


# ---------------------------------------------------------------------------
# Tests: Convergence
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_empty_years(self):
        trends = analyze_convergence([], [])
        # No colonists and no yearly data → empty
        assert len(trends) == 0

    def test_converging_trend(self):
        years = []
        for y in range(1, 101):
            # Early: high stddev, Late: low stddev
            stddev = 0.2 - (y / 100) * 0.15
            years.append(_make_year(y, convergence={
                "resolve": stddev, "improvisation": stddev,
                "empathy": stddev, "hoarding": stddev,
                "faith": stddev, "paranoia": stddev,
            }))
        trends = analyze_convergence(years)
        for t in trends:
            assert t.early_stddev > t.final_stddev
            assert t.trend == "converging"

    def test_diverging_trend(self):
        years = []
        for y in range(1, 101):
            stddev = 0.05 + (y / 100) * 0.20
            years.append(_make_year(y, convergence={
                "resolve": stddev, "improvisation": stddev,
                "empathy": stddev, "hoarding": stddev,
                "faith": stddev, "paranoia": stddev,
            }))
        trends = analyze_convergence(years)
        for t in trends:
            assert t.trend == "diverging"

    def test_to_dict(self):
        t = ConvergenceTrend(stat_name="resolve", early_stddev=0.1,
                              mid_stddev=0.09, late_stddev=0.08,
                              final_stddev=0.07, trend="converging")
        d = t.to_dict()
        assert d["stat"] == "resolve"
        assert d["trend"] == "converging"


# ---------------------------------------------------------------------------
# Tests: Factions
# ---------------------------------------------------------------------------

class TestFactions:
    def test_no_factions(self):
        colonists = [_make_colonist("c1"), _make_colonist("c2")]
        factions = analyze_factions(colonists)
        assert factions == []

    def test_single_faction(self):
        colonists = [
            _make_colonist("c1", stats={"resolve": 0.9, "improvisation": 0.3,
                                         "empathy": 0.3, "hoarding": 0.3,
                                         "faith": 0.3, "paranoia": 0.3}),
            _make_colonist("c2", stats={"resolve": 0.85, "improvisation": 0.4,
                                         "empathy": 0.3, "hoarding": 0.3,
                                         "faith": 0.3, "paranoia": 0.3}),
            _make_colonist("c3", stats={"resolve": 0.8, "improvisation": 0.3,
                                         "empathy": 0.3, "hoarding": 0.3,
                                         "faith": 0.3, "paranoia": 0.3}),
        ]
        factions = analyze_factions(colonists)
        assert len(factions) == 1
        assert factions[0].name == "The Resolute"
        assert len(factions[0].member_ids) == 3

    def test_excludes_dead(self):
        colonists = [
            _make_colonist("c1", alive=False, stats={"resolve": 0.9,
                "improvisation": 0.3, "empathy": 0.3, "hoarding": 0.3,
                "faith": 0.3, "paranoia": 0.3}),
            _make_colonist("c2", stats={"resolve": 0.85,
                "improvisation": 0.4, "empathy": 0.3, "hoarding": 0.3,
                "faith": 0.3, "paranoia": 0.3}),
            _make_colonist("c3", stats={"resolve": 0.8,
                "improvisation": 0.3, "empathy": 0.3, "hoarding": 0.3,
                "faith": 0.3, "paranoia": 0.3}),
        ]
        factions = analyze_factions(colonists)
        # Only 2 alive with same dominant → no faction (need 3)
        assert factions == []

    def test_to_dict(self):
        f = Faction(name="The Resolute", member_ids=["c1", "c2", "c3"],
                    dominant_value="resolve", cohesion=0.85, formed_year=10)
        d = f.to_dict()
        assert d["name"] == "The Resolute"
        assert len(d["members"]) == 3


# ---------------------------------------------------------------------------
# Tests: Crisis resilience
# ---------------------------------------------------------------------------

class TestCrisisResilience:
    def test_no_crises(self):
        years = [_make_year(y, events=[]) for y in range(1, 11)]
        r = analyze_crisis_resilience(years)
        assert len(r) == 10

    def test_with_crises(self):
        years = [_make_year(y, events=[{"name": "solar_flare", "severity": 0.8,
                                         "description": "big", "category": "cosmic",
                                         "effects": {"power": -0.2}}])
                 for y in range(1, 11)]
        r = analyze_crisis_resilience(years)
        assert r[0] > 0  # first decade should have some resilience score

    def test_returns_10_decades(self):
        years = [_make_year(y) for y in range(1, 101)]
        r = analyze_crisis_resilience(years)
        assert len(r) == 10


# ---------------------------------------------------------------------------
# Tests: Subsim accuracy
# ---------------------------------------------------------------------------

class TestSubsimAccuracy:
    def test_no_subsims(self):
        years = [_make_year(y, subsim_log=[]) for y in range(1, 6)]
        acc, total = analyze_subsim_accuracy(years)
        assert total == 0

    def test_all_correct(self):
        years = [_make_year(y,
            subsim_log=[{"depth": 1, "colonist_id": "c1", "year": y,
                          "expression": "(+ 1)", "result": -0.5}],
            resource_delta={"food": -0.1, "water": -0.05, "power": -0.02,
                             "air": -0.01, "medicine": -0.01})
                 for y in range(1, 6)]
        acc, total = analyze_subsim_accuracy(years)
        assert total == 5
        assert acc > 0.5

    def test_non_numeric_results_skipped(self):
        years = [_make_year(1,
            subsim_log=[{"depth": 1, "colonist_id": "c1", "year": 1,
                          "expression": "(list 1 2)", "result": [1, 2]}])]
        acc, total = analyze_subsim_accuracy(years)
        assert total == 0


# ---------------------------------------------------------------------------
# Tests: Insight synthesis
# ---------------------------------------------------------------------------

class TestSynthesizeInsights:
    def _default_phases(self):
        return [GovernancePhase("anarchy", 1, 20, 19, 5, 1, 0.5),
                GovernancePhase("council", 20, None, 81, 15, 3, 0.6)]

    def test_produces_insights(self):
        insights = synthesize_insights(
            self._default_phases(),
            MortalityPattern(total_deaths=6, causes={"asphyxiation": 6},
                             systemic_cause="asphyxiation accounts for 100% of deaths"),
            [], [], [0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.7], 0.6,
        )
        assert len(insights) > 0

    def test_sorted_by_strength(self):
        insights = synthesize_insights(
            self._default_phases(),
            MortalityPattern(total_deaths=6, causes={"asphyxiation": 6},
                             systemic_cause="asphyxiation is systemic"),
            [], [], [0.3] * 10, 0.6,
        )
        for i in range(len(insights) - 1):
            assert insights[i].strength >= insights[i + 1].strength

    def test_governance_insight(self):
        phases = [GovernancePhase("council", 1, None, 100, 20, 2, 0.7)]
        insights = synthesize_insights(phases, MortalityPattern(), [], [], [], 0.0)
        gov_insights = [i for i in insights if "governance" in i.source]
        assert len(gov_insights) > 0

    def test_resilience_improving(self):
        resilience = [0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8]
        insights = synthesize_insights([], MortalityPattern(), [], [],
                                        resilience, 0.0)
        res_insights = [i for i in insights if "resilience" in i.source]
        assert len(res_insights) > 0
        assert "more resilient" in res_insights[0].title.lower()


# ---------------------------------------------------------------------------
# Tests: Amendment proposal
# ---------------------------------------------------------------------------

class TestProposeAmendment:
    def test_no_eligible(self):
        insights = [Insight("weak", "evidence", 0.3, "test")]
        assert propose_amendment(insights) is None

    def test_selects_strongest(self):
        insights = [
            Insight("strong", "evidence", 0.9, "governance_phases"),
            Insight("medium", "evidence", 0.6, "mortality"),
        ]
        result = propose_amendment(insights)
        assert result is not None
        assert result.title == "strong"
        assert len(result.amendment_text) > 0

    def test_amendment_text_populated(self):
        insights = [Insight("test", "evidence", 0.8, "subsim_accuracy")]
        result = propose_amendment(insights)
        assert result is not None
        assert "sub-simulation" in result.amendment_text.lower()


# ---------------------------------------------------------------------------
# Tests: Full analysis pipeline
# ---------------------------------------------------------------------------

class TestFullAnalysis:
    def _make_sim_data(self, num_years: int = 20) -> dict:
        years = []
        for y in range(1, num_years + 1):
            gov = "anarchy" if y < 10 else "council"
            years.append(_make_year(y, governance_state={"gov_type": gov}))
        colonists = [
            _make_colonist("c1"),
            _make_colonist("c2", alive=False, death_year=15,
                           death_cause="asphyxiation"),
        ]
        return {"years": years, "final_colonists": colonists}

    def test_analyze_returns_report(self):
        report = analyze(self._make_sim_data())
        assert isinstance(report, EmergenceReport)
        assert len(report.governance_phases) > 0
        assert report.mortality.total_deaths >= 0

    def test_report_to_dict(self):
        report = analyze(self._make_sim_data())
        d = report.to_dict()
        assert "governance_phases" in d
        assert "mortality" in d
        assert "convergence" in d
        assert "factions" in d
        assert "insights" in d

    def test_empty_sim_data(self):
        report = analyze({})
        assert isinstance(report, EmergenceReport)
        assert report.governance_phases == []

    def test_insights_generated(self):
        report = analyze(self._make_sim_data(50))
        assert len(report.insights) > 0

    def test_convergence_has_all_stats(self):
        report = analyze(self._make_sim_data())
        assert len(report.convergence) == 6

    def test_physical_bounds_resources(self):
        """Property: all resilience scores should be non-negative."""
        data = self._make_sim_data(100)
        report = analyze(data)
        for r in report.crisis_resilience:
            assert r >= 0.0

    def test_smoke_100_years(self):
        """Smoke test: 100 years without crash."""
        data = self._make_sim_data(100)
        report = analyze(data)
        assert isinstance(report, EmergenceReport)
        d = report.to_dict()
        assert isinstance(d, dict)


# ---------------------------------------------------------------------------
# Tests: _normalize_data adapter (legacy format)
# ---------------------------------------------------------------------------

from src.mars100.emergence import _normalize_data


class TestNormalizeData:
    def _make_legacy_data(self) -> dict:
        """Create a minimal legacy-format simulation data dict."""
        return {
            "_meta": {"total_years": 5},
            "colony": {
                "colonists": [
                    {"id": 0, "name": "Ares", "element": "fire", "alive": True,
                     "year_born": 0,
                     "stats": {"resolve": 80, "improvisation": 50, "empathy": 30,
                               "hoarding": 40, "faith": 20, "paranoia": 10},
                     "skills": {"terraforming": 70}},
                    {"id": 1, "name": "Lyra", "element": "air", "alive": False,
                     "year_born": 0, "year_died": 3, "cause_of_death": "asphyxiation",
                     "stats": {"resolve": 20, "improvisation": 60, "empathy": 80,
                               "hoarding": 40, "faith": 30, "paranoia": 50},
                     "skills": {"mediation": 50}},
                ],
                "dead_souls": [
                    {"id": 1, "name": "Lyra", "element": "air", "alive": False,
                     "year_born": 0, "year_died": 3, "cause_of_death": "asphyxiation",
                     "stats": {"resolve": 20, "improvisation": 60, "empathy": 80,
                               "hoarding": 40, "faith": 30, "paranoia": 50},
                     "skills": {"mediation": 50}},
                ],
                "resources": {"food": 500, "water": 800, "power": 100, "oxygen": 300},
                "governance": {"system": "direct_democracy", "leader": 0, "amendments": []},
            },
            "deltas": [
                {"year": y, "population": 2,
                 "event": {"id": "dust_storm", "desc": "A dust storm", "severity": 0.6},
                 "colonist_actions": [{"colonist": "Ares", "action": "farm"}],
                 "sub_sims": [{"colonist": "Ares", "depth": 1,
                               "proposal": "test", "result": "[1.5, 2.0]",
                               "s_expr": "(+ 1 2)"}],
                 "governance_results": ["Leadership election: Ares elected (ratio 80%)"],
                 "resources_snapshot": {"food": 400 + y * 20, "water": 700, "power": 80,
                                         "oxygen": 200, "materials": 1000, "morale": 90},
                 "births": [], "diary_entries": [], "meta_awareness": []}
                for y in range(1, 6)
            ],
        }

    def test_engine_format_passthrough(self):
        """Engine format should pass through unchanged."""
        data = {"years": [{"year": 1}], "final_colonists": [{"id": "c1"}]}
        years, colonists = _normalize_data(data)
        assert len(years) == 1
        assert len(colonists) == 1

    def test_legacy_deduplicates_colonists(self):
        """Dead colonists in both lists should not be double-counted."""
        data = self._make_legacy_data()
        _, colonists = _normalize_data(data)
        ids = [c.get("id") for c in colonists]
        assert len(ids) == len(set(ids))
        assert len(colonists) == 2  # Ares + Lyra (once each)

    def test_legacy_normalizes_stats(self):
        """Stats 0-100 should become 0.0-1.0."""
        data = self._make_legacy_data()
        _, colonists = _normalize_data(data)
        for c in colonists:
            for v in c.get("stats", {}).values():
                assert 0.0 <= v <= 1.0, f"stat {v} not normalized"

    def test_legacy_remaps_field_names(self):
        """year_died→death_year, year_born→birth_year, cause_of_death→death_cause."""
        data = self._make_legacy_data()
        _, colonists = _normalize_data(data)
        dead = [c for c in colonists if not c.get("alive", True)]
        assert len(dead) == 1
        assert dead[0]["death_year"] == 3
        assert dead[0]["death_cause"] == "asphyxiation"
        assert dead[0]["birth_year"] == 0

    def test_legacy_injects_deaths_into_years(self):
        """Deaths should appear in the yearly record for the year they died."""
        data = self._make_legacy_data()
        years, _ = _normalize_data(data)
        year3 = [y for y in years if y["year"] == 3][0]
        assert len(year3["deaths"]) == 1
        assert year3["deaths"][0]["cause"] == "asphyxiation"

    def test_legacy_parses_events(self):
        """Single event dict should become a list of event dicts."""
        data = self._make_legacy_data()
        years, _ = _normalize_data(data)
        assert len(years[0]["events"]) == 1
        assert years[0]["events"][0]["name"] == "dust_storm"
        assert years[0]["events"][0]["severity"] == 0.6

    def test_legacy_parses_subsims(self):
        """Sub-sim results should be parsed from string."""
        data = self._make_legacy_data()
        years, _ = _normalize_data(data)
        subsims = years[0]["subsim_log"]
        assert len(subsims) == 1
        assert subsims[0]["depth"] == 1
        # Result "[1.5, 2.0]" → first element = 1.5
        assert subsims[0]["result"] == 1.5

    def test_legacy_detects_governance(self):
        """Election results should set governance type to direct_democracy."""
        data = self._make_legacy_data()
        years, _ = _normalize_data(data)
        assert years[-1]["governance_state"]["gov_type"] == "direct_democracy"

    def test_legacy_normalizes_resources(self):
        """Resource snapshots should be normalized to 0-1."""
        data = self._make_legacy_data()
        years, _ = _normalize_data(data)
        for k, v in years[0]["resources_before"].items():
            assert 0.0 <= v <= 1.0, f"resource {k}={v} out of range"

    def test_full_analyze_legacy_format(self):
        """Full analysis should work end-to-end on legacy format."""
        data = self._make_legacy_data()
        report = analyze(data)
        assert isinstance(report, EmergenceReport)
        assert report.mortality.total_deaths == 1
        assert "asphyxiation" in report.mortality.causes
        assert report.total_subsims == 5
