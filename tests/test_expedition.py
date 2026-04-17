"""Tests for the Mars-100 expedition & discovery engine."""
from __future__ import annotations

import random

import pytest

from src.mars100.expedition import (
    BONUS_DECAY_RATE,
    EXPEDITION_BASE_SUCCESS,
    EXPEDITION_DEATH_RATE,
    EXPEDITION_FOOD_COST,
    EXPEDITION_POWER_COST,
    MAX_SITES,
    MIN_TEAM_SIZE,
    SITE_NAMES_POOL,
    SITE_TYPES,
    ExpeditionResult,
    Site,
    age_sites,
    can_form_expedition,
    compute_meta_boost,
    compute_site_bonuses,
    compute_success_probability,
    generate_site,
    run_expedition,
    select_team,
)
from src.mars100.colonist import (
    Colonist,
    ColonistSkills,
    ColonistStats,
    create_founding_ten,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_colonist(cid: str = "test-1", resolve: float = 0.6,
                   improvisation: float = 0.5, **kwargs: float) -> Colonist:
    stats = ColonistStats(resolve=resolve, improvisation=improvisation,
                          empathy=kwargs.get("empathy", 0.5),
                          hoarding=kwargs.get("hoarding", 0.3),
                          faith=kwargs.get("faith", 0.4),
                          paranoia=kwargs.get("paranoia", 0.3))
    return Colonist(id=cid, name=f"Col-{cid}", element="fire",
                    archetype="explorer", stats=stats,
                    skills=ColonistSkills(), decision_expr="(+ resolve 1)")


def _make_team(n: int = 3, seed: int = 42) -> list[Colonist]:
    return [_make_colonist(f"t-{i}", resolve=0.5 + i * 0.1) for i in range(n)]


def _make_site(year: int = 10, site_type: str = "ice_field",
               name: str = "Test Site") -> Site:
    return Site(id=f"site-{year}-{site_type}", name=name,
                site_type=site_type, label="Test",
                description="A test site.", discovered_year=year,
                discovered_by=["t-0"], lat=10.0, lon=20.0,
                bonus={"water": 0.03}, meta_boost=0.0)


# ── Site tests ───────────────────────────────────────────────────────────

class TestSite:
    def test_current_bonus_fresh(self) -> None:
        site = _make_site()
        assert site.current_bonus() == {"water": 0.03}

    def test_current_bonus_decays(self) -> None:
        site = _make_site()
        site.years_active = 5
        expected = 0.03 * (BONUS_DECAY_RATE ** 5)
        assert abs(site.current_bonus()["water"] - expected) < 1e-6

    def test_to_dict_keys(self) -> None:
        site = _make_site()
        d = site.to_dict()
        expected_keys = {"id", "name", "type", "label", "description",
                         "discovered_year", "discovered_by", "lat", "lon",
                         "bonus", "meta_boost", "years_active", "current_bonus"}
        assert set(d.keys()) == expected_keys

    def test_to_dict_roundtrip_lat_lon(self) -> None:
        site = _make_site()
        site.lat = 12.345678
        site.lon = -67.891234
        d = site.to_dict()
        assert d["lat"] == 12.35
        assert d["lon"] == -67.89

    def test_meta_boost_site(self) -> None:
        site = Site(id="s-anomaly", name="X", site_type="ancient_anomaly",
                    label="Anomaly", description="Strange.",
                    discovered_year=20, discovered_by=["a"],
                    lat=0, lon=0, bonus={}, meta_boost=0.02)
        assert site.meta_boost == 0.02
        assert site.current_bonus() == {}


# ── can_form_expedition tests ────────────────────────────────────────────

class TestCanFormExpedition:
    def test_enough_explorers_and_resources(self) -> None:
        team = _make_team(3)
        assert can_form_expedition(team, food=0.5, power=0.3) is True

    def test_too_few_explorers(self) -> None:
        team = _make_team(1)
        assert can_form_expedition(team, food=0.5, power=0.3) is False

    def test_insufficient_food(self) -> None:
        team = _make_team(3)
        assert can_form_expedition(team, food=0.01, power=0.3) is False

    def test_insufficient_power(self) -> None:
        team = _make_team(3)
        assert can_form_expedition(team, food=0.5, power=0.01) is False

    def test_exactly_two_explorers(self) -> None:
        team = _make_team(2)
        assert can_form_expedition(team, food=0.5, power=0.3) is True


# ── select_team tests ────────────────────────────────────────────────────

class TestSelectTeam:
    def test_returns_at_least_min_size(self) -> None:
        team = _make_team(5)
        rng = random.Random(42)
        selected = select_team(team, rng)
        assert len(selected) >= MIN_TEAM_SIZE

    def test_returns_at_most_max_size(self) -> None:
        team = _make_team(10)
        rng = random.Random(42)
        selected = select_team(team, rng, max_size=4)
        assert len(selected) <= 4

    def test_prefers_high_resolve_improv(self) -> None:
        weak = _make_colonist("weak", resolve=0.1, improvisation=0.1)
        strong = _make_colonist("strong", resolve=0.9, improvisation=0.9)
        team = [weak, strong, _make_colonist("mid", resolve=0.5, improvisation=0.5)]
        rng = random.Random(42)
        selected = select_team(team, rng, max_size=2)
        assert strong in selected


# ── compute_success_probability tests ────────────────────────────────────

class TestSuccessProbability:
    def test_base_range(self) -> None:
        team = _make_team(3)
        prob = compute_success_probability(team, event_severity=0.3,
                                           num_known_sites=0)
        assert 0.05 <= prob <= 0.95

    def test_higher_with_better_stats(self) -> None:
        weak = [_make_colonist(f"w{i}", resolve=0.2, improvisation=0.2)
                for i in range(3)]
        strong = [_make_colonist(f"s{i}", resolve=0.9, improvisation=0.9)
                  for i in range(3)]
        p_weak = compute_success_probability(weak, 0.3, 0)
        p_strong = compute_success_probability(strong, 0.3, 0)
        assert p_strong > p_weak

    def test_lower_with_high_severity(self) -> None:
        team = _make_team(3)
        p_calm = compute_success_probability(team, 0.1, 0)
        p_storm = compute_success_probability(team, 0.9, 0)
        assert p_calm > p_storm

    def test_lower_with_more_sites(self) -> None:
        team = _make_team(3)
        p_early = compute_success_probability(team, 0.3, 0)
        p_late = compute_success_probability(team, 0.3, 10)
        assert p_early > p_late

    def test_clamped(self) -> None:
        team = _make_team(3)
        prob = compute_success_probability(team, 1.0, 50)
        assert prob >= 0.05
        prob = compute_success_probability(team, 0.0, 0)
        assert prob <= 0.95


# ── generate_site tests ──────────────────────────────────────────────────

class TestGenerateSite:
    def test_returns_site(self) -> None:
        rng = random.Random(42)
        site = generate_site(10, _make_team(2), [], rng)
        assert isinstance(site, Site)
        assert site.discovered_year == 10
        assert len(site.discovered_by) == 2

    def test_avoids_duplicate_types(self) -> None:
        rng = random.Random(42)
        existing = [_make_site(5, "ice_field", "Ice Place")]
        site = generate_site(15, _make_team(2), existing, rng)
        assert site.site_type != "ice_field"

    def test_avoids_duplicate_names(self) -> None:
        rng = random.Random(42)
        # Use up most names
        existing: list[Site] = []
        for i, name in enumerate(SITE_NAMES_POOL[:15]):
            existing.append(_make_site(i, "mineral_vein", name))
        site = generate_site(20, _make_team(2), existing, rng)
        used_names = {s.name for s in existing}
        assert site.name not in used_names

    def test_lat_lon_in_bounds(self) -> None:
        rng = random.Random(123)
        for _ in range(20):
            site = generate_site(10, _make_team(2), [], rng)
            assert -60.0 <= site.lat <= 60.0
            assert -180.0 <= site.lon <= 180.0

    def test_wraps_when_all_types_used(self) -> None:
        """Still generates a site even when all types have been used."""
        rng = random.Random(42)
        existing = [_make_site(i, t["type"], f"N-{i}")
                    for i, t in enumerate(SITE_TYPES)]
        site = generate_site(50, _make_team(2), existing, rng)
        assert isinstance(site, Site)


# ── run_expedition tests ─────────────────────────────────────────────────

class TestRunExpedition:
    def test_success_produces_site(self) -> None:
        rng = random.Random(1)
        team = [_make_colonist(f"e{i}", resolve=0.9, improvisation=0.9)
                for i in range(3)]
        sites: list[Site] = []
        result = run_expedition(10, team, sites, 0.1, rng)
        assert isinstance(result, ExpeditionResult)
        if result.success:
            assert result.site is not None
            assert len(sites) == 1

    def test_failure_no_site(self) -> None:
        """Force failure by setting terrible conditions."""
        rng = random.Random(999)
        team = [_make_colonist(f"f{i}", resolve=0.05, improvisation=0.05)
                for i in range(2)]
        sites: list[Site] = [_make_site(y, "mineral_vein", f"S{y}")
                             for y in range(15)]
        result = run_expedition(50, team, sites, 0.95, rng)
        # Either way, result should be valid
        assert isinstance(result, ExpeditionResult)
        assert isinstance(result.narrative, str)
        assert len(result.narrative) > 0

    def test_deaths_use_expedition_cause(self) -> None:
        rng = random.Random(42)
        team = _make_team(4)
        sites: list[Site] = []
        # Run many times to get at least one death
        any_death = False
        for seed in range(100):
            rng = random.Random(seed)
            fresh_team = _make_team(4)
            result = run_expedition(20, fresh_team, sites, 0.8, rng)
            for d in result.deaths:
                assert "expedition" in d["cause"]
                any_death = True
        assert any_death, "Expected at least one death in 100 trials"

    def test_bonds_between_survivors(self) -> None:
        rng = random.Random(42)
        team = [_make_colonist(f"b{i}", resolve=0.9, improvisation=0.9)
                for i in range(3)]
        sites: list[Site] = []
        result = run_expedition(10, team, sites, 0.1, rng)
        surviving_ids = {c.id for c in team if c.is_active()}
        for a, b in result.bonds:
            assert a in surviving_ids
            assert b in surviving_ids

    def test_to_dict(self) -> None:
        rng = random.Random(42)
        team = _make_team(3)
        sites: list[Site] = []
        result = run_expedition(10, team, sites, 0.2, rng)
        d = result.to_dict()
        assert "year" in d
        assert "team" in d
        assert "success" in d
        assert "narrative" in d

    def test_expedition_marks_dead_colonists(self) -> None:
        """Dead colonists should have alive=False after expedition."""
        for seed in range(200):
            rng = random.Random(seed)
            team = _make_team(4)
            sites: list[Site] = []
            result = run_expedition(30, team, sites, 0.7, rng)
            for d in result.deaths:
                member = next(c for c in team if c.id == d["id"])
                assert not member.alive
                assert member.death_year == 30
                assert "expedition" in member.death_cause


# ── compute_site_bonuses tests ───────────────────────────────────────────

class TestComputeSiteBonuses:
    def test_empty(self) -> None:
        assert compute_site_bonuses([]) == {}

    def test_single_fresh_site(self) -> None:
        sites = [_make_site()]
        bonuses = compute_site_bonuses(sites)
        assert abs(bonuses["water"] - 0.03) < 1e-6

    def test_diminishing_returns(self) -> None:
        site = _make_site()
        site.years_active = 10
        bonuses = compute_site_bonuses([site])
        assert bonuses["water"] < 0.03

    def test_multiple_sites_stack(self) -> None:
        sites = [_make_site(1, "ice_field", "A"), _make_site(2, "ice_field", "B")]
        bonuses = compute_site_bonuses(sites)
        assert bonuses["water"] > 0.03  # two fresh sites


# ── compute_meta_boost tests ─────────────────────────────────────────────

class TestComputeMetaBoost:
    def test_no_anomalies(self) -> None:
        sites = [_make_site()]  # ice_field, meta_boost=0
        assert compute_meta_boost(sites) == 0.0

    def test_with_anomaly(self) -> None:
        anomaly = Site(id="s-a", name="X", site_type="ancient_anomaly",
                       label="Anomaly", description="Strange.",
                       discovered_year=20, discovered_by=["a"],
                       lat=0, lon=0, bonus={}, meta_boost=0.02)
        assert compute_meta_boost([anomaly]) == pytest.approx(0.02)

    def test_decays_with_age(self) -> None:
        anomaly = Site(id="s-a", name="X", site_type="ancient_anomaly",
                       label="Anomaly", description="Strange.",
                       discovered_year=20, discovered_by=["a"],
                       lat=0, lon=0, bonus={}, meta_boost=0.02,
                       years_active=5)
        expected = 0.02 * (BONUS_DECAY_RATE ** 5)
        assert compute_meta_boost([anomaly]) == pytest.approx(expected)


# ── age_sites tests ──────────────────────────────────────────────────────

class TestAgeSites:
    def test_increments_years_active(self) -> None:
        sites = [_make_site(), _make_site(5, "lava_tube", "Other")]
        age_sites(sites)
        assert all(s.years_active == 1 for s in sites)
        age_sites(sites)
        assert all(s.years_active == 2 for s in sites)

    def test_empty_list_ok(self) -> None:
        age_sites([])  # no crash


# ── Simulation-level calibration tests ───────────────────────────────────

class TestCalibration:
    """Run across multiple seeds to verify expedition statistics stay sane."""

    def _run_100_years(self, seed: int) -> dict:
        """Simulate 100 years of expedition attempts."""
        rng = random.Random(seed)
        colonists = create_founding_ten(seed)
        sites: list[Site] = []
        expedition_count = 0
        total_deaths = 0

        for year in range(1, 101):
            # Simulate some colonists choosing explore
            explorers = [c for c in colonists
                         if c.is_active() and rng.random() < 0.3]
            if can_form_expedition(explorers, food=0.5, power=0.5):
                team = select_team(explorers, rng)
                severity = rng.uniform(0.1, 0.6)
                result = run_expedition(year, team, sites, severity, rng)
                expedition_count += 1
                total_deaths += len(result.deaths)
            age_sites(sites)

        return {
            "expeditions": expedition_count,
            "sites": len(sites),
            "deaths": total_deaths,
        }

    def test_expedition_count_reasonable(self) -> None:
        """Across 10 seeds, average expeditions stay in a reasonable range."""
        counts = [self._run_100_years(s)["expeditions"] for s in range(10)]
        avg = sum(counts) / len(counts)
        assert 1 <= avg <= 50, f"avg expeditions={avg}"

    def test_sites_monotonic_and_bounded(self) -> None:
        for seed in range(5):
            result = self._run_100_years(seed)
            assert result["sites"] <= result["expeditions"]

    def test_no_excessive_deaths(self) -> None:
        """Expedition deaths should be a small fraction of team-years."""
        for seed in range(5):
            result = self._run_100_years(seed)
            if result["expeditions"] > 0:
                death_rate = result["deaths"] / (result["expeditions"] * 3)
                assert death_rate < 0.5, f"death_rate={death_rate:.2f}"

    def test_site_bonuses_dont_explode(self) -> None:
        """Total site bonuses should stay bounded."""
        rng = random.Random(42)
        sites: list[Site] = []
        for i in range(MAX_SITES):
            site = generate_site(i, _make_team(2), sites, rng)
            sites.append(site)
        bonuses = compute_site_bonuses(sites)
        for resource, val in bonuses.items():
            assert val < 0.5, f"{resource}={val}"


# ── ExpeditionResult serialization ───────────────────────────────────────

class TestExpeditionResultSerialization:
    def test_to_dict_all_fields(self) -> None:
        result = ExpeditionResult(
            year=15, team=["a", "b"], success=True,
            site={"id": "s-1", "name": "Test"},
            deaths=[], bonds=[("a", "b")],
            narrative="They went. They came back."
        )
        d = result.to_dict()
        assert d["year"] == 15
        assert d["team"] == ["a", "b"]
        assert d["success"] is True
        assert d["site"]["name"] == "Test"
        assert d["bonds"] == [("a", "b")]
