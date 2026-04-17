"""Tests for the diplomacy organ — factions, alliances, schisms, vote bias."""
from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, create_founding_ten
from src.mars100.colony import SocialGraph, Relationship
from src.mars100.diplomacy import (
    Faction, Alliance, DiplomacyState,
    detect_factions, reconcile_factions, check_schism,
    assign_platform, compute_faction_cohesion,
    try_form_alliance, check_alliance_breakups,
    faction_vote_bias, tick_diplomacy,
    MIN_FACTION_SIZE, TRUST_THRESHOLD, VOTE_BIAS_CAP,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_colonist(cid: str, **stat_kw) -> Colonist:
    """Create a minimal colonist for testing."""
    stats = ColonistStats(**{k: stat_kw.get(k, 0.5) for k in [
        "resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia",
    ]})
    skills = ColonistSkills()
    return Colonist(
        id=cid, name=cid.title(), element="fire", archetype="pioneer",
        stats=stats, skills=skills,
        decision_expr="(+ resolve empathy)",
    )


def _high_trust_graph(ids: list[str], trust: float = 0.8) -> SocialGraph:
    """Create a social graph where everyone trusts everyone."""
    sg = SocialGraph()
    sg.edges = {}
    for a in ids:
        sg.edges[a] = {}
        for b in ids:
            if a != b:
                sg.edges[a][b] = Relationship(trust=trust, affection=0.5, respect=0.5)
    return sg


def _two_cluster_graph(
    group_a: list[str], group_b: list[str],
    intra_trust: float = 0.8, inter_trust: float = 0.2,
) -> SocialGraph:
    """Two clusters with high internal trust, low cross-cluster trust."""
    all_ids = group_a + group_b
    sg = SocialGraph()
    sg.edges = {}
    for a in all_ids:
        sg.edges[a] = {}
        for b in all_ids:
            if a == b:
                continue
            same_group = (a in group_a and b in group_a) or (a in group_b and b in group_b)
            t = intra_trust if same_group else inter_trust
            sg.edges[a][b] = Relationship(trust=t, affection=0.5, respect=0.5)
    return sg


# ── Faction detection ──────────────────────────────────────────────────────

class TestDetectFactions:
    def test_single_cluster_all_trusting(self) -> None:
        ids = ["c-0", "c-1", "c-2", "c-3"]
        sg = _high_trust_graph(ids)
        factions = detect_factions(sg, ids)
        assert len(factions) == 1
        assert factions[0] == set(ids)

    def test_two_clusters(self) -> None:
        ga = ["c-0", "c-1", "c-2"]
        gb = ["c-3", "c-4", "c-5"]
        sg = _two_cluster_graph(ga, gb)
        factions = detect_factions(sg, ga + gb)
        assert len(factions) == 2
        faction_sets = [set(f) for f in factions]
        assert set(ga) in faction_sets
        assert set(gb) in faction_sets

    def test_isolated_individuals_not_factions(self) -> None:
        ids = ["c-0", "c-1", "c-2"]
        sg = SocialGraph()  # empty edges → default trust 0.5 < TRUST_THRESHOLD
        factions = detect_factions(sg, ids)
        assert len(factions) == 0

    def test_respects_min_faction_size(self) -> None:
        ids = ["c-0", "c-1"]
        sg = _high_trust_graph(ids)
        factions = detect_factions(sg, ids)
        assert len(factions) == 1

    def test_below_min_size_excluded(self) -> None:
        ids = ["c-0"]
        sg = SocialGraph()
        factions = detect_factions(sg, ids)
        assert len(factions) == 0


# ── Faction reconciliation ─────────────────────────────────────────────────

class TestReconcileFactions:
    def test_new_factions_get_ids(self) -> None:
        clusters = [{"c-0", "c-1"}, {"c-2", "c-3"}]
        result = reconcile_factions([], clusters, year=1)
        assert len(result) == 2
        assert all(f.id.startswith("faction-") for f in result)

    def test_stable_ids_across_reconciliation(self) -> None:
        old = [Faction(id="faction-1", member_ids=["c-0", "c-1"], formed_year=1)]
        new_clusters = [{"c-0", "c-1", "c-2"}]
        result = reconcile_factions(old, new_clusters, year=2)
        assert len(result) == 1
        assert result[0].id == "faction-1"

    def test_completely_new_cluster_gets_new_id(self) -> None:
        old = [Faction(id="faction-1", member_ids=["c-0", "c-1"], formed_year=1)]
        new_clusters = [{"c-5", "c-6", "c-7"}]
        result = reconcile_factions(old, new_clusters, year=2)
        assert result[0].id != "faction-1"

    def test_preserves_formed_year_for_matched(self) -> None:
        old = [Faction(id="faction-1", member_ids=["c-0", "c-1"], formed_year=5)]
        new_clusters = [{"c-0", "c-1"}]
        result = reconcile_factions(old, new_clusters, year=10)
        assert result[0].formed_year == 5


# ── Platform assignment ────────────────────────────────────────────────────

class TestAssignPlatform:
    def test_high_resolve_gets_technocracy(self) -> None:
        colonists = [_make_colonist("c-0", resolve=0.9), _make_colonist("c-1", resolve=0.8)]
        f = Faction(id="f", member_ids=["c-0", "c-1"])
        assert assign_platform(f, colonists) == "technocracy"

    def test_high_empathy_gets_democracy(self) -> None:
        colonists = [_make_colonist("c-0", empathy=0.9), _make_colonist("c-1", empathy=0.8)]
        f = Faction(id="f", member_ids=["c-0", "c-1"])
        assert assign_platform(f, colonists) == "direct_democracy"

    def test_high_improvisation_gets_anarchy(self) -> None:
        colonists = [_make_colonist("c-0", improvisation=0.9), _make_colonist("c-1", improvisation=0.8)]
        f = Faction(id="f", member_ids=["c-0", "c-1"])
        assert assign_platform(f, colonists) == "anarchy"

    def test_balanced_gets_council_or_meritocracy(self) -> None:
        colonists = [_make_colonist("c-0")]
        f = Faction(id="f", member_ids=["c-0"])
        result = assign_platform(f, colonists)
        assert result in ("council", "meritocracy")


# ── Cohesion ───────────────────────────────────────────────────────────────

class TestFactionCohesion:
    def test_high_trust_high_cohesion(self) -> None:
        f = Faction(id="f", member_ids=["c-0", "c-1", "c-2"])
        sg = _high_trust_graph(["c-0", "c-1", "c-2"], trust=0.9)
        assert compute_faction_cohesion(f, sg) == pytest.approx(0.9)

    def test_single_member_zero_cohesion(self) -> None:
        f = Faction(id="f", member_ids=["c-0"])
        sg = SocialGraph()
        assert compute_faction_cohesion(f, sg) == 0.0


# ── Schisms ────────────────────────────────────────────────────────────────

class TestCheckSchism:
    def test_small_faction_no_schism(self) -> None:
        f = Faction(id="f", member_ids=["c-0", "c-1"])
        sg = SocialGraph()
        result = check_schism(f, sg, year=5, rng=random.Random(42))
        assert result is None

    def test_high_trust_no_schism(self) -> None:
        members = ["c-0", "c-1", "c-2", "c-3", "c-4", "c-5"]
        f = Faction(id="f", member_ids=members)
        sg = _high_trust_graph(members, trust=0.9)
        result = check_schism(f, sg, year=5, rng=random.Random(42))
        assert result is None

    def test_low_trust_may_schism(self) -> None:
        members = ["c-0", "c-1", "c-2", "c-3", "c-4", "c-5"]
        ga, gb = members[:3], members[3:]
        sg = _two_cluster_graph(ga, gb, intra_trust=0.7, inter_trust=0.1)
        f = Faction(id="f", member_ids=members)

        schism_happened = False
        for seed in range(100):
            result = check_schism(f, sg, year=5, rng=random.Random(seed))
            if result is not None:
                schism_happened = True
                assert len(result) == 2
                all_members = set()
                for child in result:
                    all_members.update(child.member_ids)
                assert all_members == set(members)
                break
        assert schism_happened, "No schism after 100 seeds"


# ── Alliances ──────────────────────────────────────────────────────────────

class TestAlliances:
    def test_two_factions_can_ally(self) -> None:
        ga = ["c-0", "c-1"]
        gb = ["c-2", "c-3"]
        sg = _two_cluster_graph(ga, gb, intra_trust=0.8, inter_trust=0.6)
        state = DiplomacyState(factions=[
            Faction(id="f-1", member_ids=ga),
            Faction(id="f-2", member_ids=gb),
        ])
        alliance = try_form_alliance(state, sg, year=5, rng=random.Random(42))
        assert alliance is not None
        assert {alliance.faction_a, alliance.faction_b} == {"f-1", "f-2"}

    def test_low_inter_trust_no_alliance(self) -> None:
        ga = ["c-0", "c-1"]
        gb = ["c-2", "c-3"]
        sg = _two_cluster_graph(ga, gb, intra_trust=0.8, inter_trust=0.1)
        state = DiplomacyState(factions=[
            Faction(id="f-1", member_ids=ga),
            Faction(id="f-2", member_ids=gb),
        ])
        alliance = try_form_alliance(state, sg, year=5, rng=random.Random(42))
        assert alliance is None

    def test_alliance_breakup_on_trust_decay(self) -> None:
        ga = ["c-0", "c-1"]
        gb = ["c-2", "c-3"]
        sg = _two_cluster_graph(ga, gb, intra_trust=0.8, inter_trust=0.1)
        state = DiplomacyState(
            factions=[
                Faction(id="f-1", member_ids=ga),
                Faction(id="f-2", member_ids=gb),
            ],
            alliances=[Alliance(faction_a="f-1", faction_b="f-2", formed_year=1)],
        )
        broken = check_alliance_breakups(state, sg)
        assert len(broken) == 1
        assert len(state.alliances) == 0


# ── Vote bias ──────────────────────────────────────────────────────────────

class TestVoteBias:
    def test_matching_platform_positive_bias(self) -> None:
        state = DiplomacyState(factions=[
            Faction(id="f-1", member_ids=["c-0"], platform="technocracy"),
        ])
        c = _make_colonist("c-0")
        bias = faction_vote_bias(state, c, "technocracy")
        assert bias == pytest.approx(VOTE_BIAS_CAP)

    def test_opposing_platform_negative_bias(self) -> None:
        state = DiplomacyState(factions=[
            Faction(id="f-1", member_ids=["c-0"], platform="technocracy"),
        ])
        c = _make_colonist("c-0")
        bias = faction_vote_bias(state, c, "direct_democracy")
        assert bias == pytest.approx(-VOTE_BIAS_CAP * 0.5)

    def test_no_faction_no_bias(self) -> None:
        state = DiplomacyState()
        c = _make_colonist("c-0")
        assert faction_vote_bias(state, c, "anything") == 0.0

    def test_bias_bounded(self) -> None:
        state = DiplomacyState(factions=[
            Faction(id="f-1", member_ids=["c-0"], platform="council"),
        ])
        c = _make_colonist("c-0")
        for gov in ["council", "technocracy", "direct_democracy", "anarchy", "meritocracy"]:
            b = faction_vote_bias(state, c, gov)
            assert abs(b) <= VOTE_BIAS_CAP


# ── Serialization ──────────────────────────────────────────────────────────

class TestSerialization:
    def test_faction_roundtrip(self) -> None:
        f = Faction(id="f-1", member_ids=["a", "b"], formed_year=3, platform="council", cohesion=0.7)
        d = f.to_dict()
        f2 = Faction.from_dict(d)
        assert f2.id == "f-1"
        assert f2.member_ids == ["a", "b"]
        assert f2.platform == "council"

    def test_alliance_roundtrip(self) -> None:
        a = Alliance(faction_a="f-1", faction_b="f-2", formed_year=5, strength=0.6)
        d = a.to_dict()
        a2 = Alliance.from_dict(d)
        assert a2.faction_a == "f-1"
        assert a2.strength == 0.6

    def test_diplomacy_state_roundtrip(self) -> None:
        state = DiplomacyState(
            factions=[Faction(id="f-1", member_ids=["a"])],
            alliances=[Alliance(faction_a="f-1", faction_b="f-2")],
            history=[{"type": "test"}],
        )
        d = state.to_dict()
        state2 = DiplomacyState.from_dict(d)
        assert len(state2.factions) == 1
        assert len(state2.alliances) == 1


# ── Tick integration ───────────────────────────────────────────────────────

class TestTickDiplomacy:
    def test_tick_returns_events(self) -> None:
        colonists = [_make_colonist(f"c-{i}") for i in range(6)]
        ids = [c.id for c in colonists]
        ga, gb = ids[:3], ids[3:]
        sg = _two_cluster_graph(ga, gb, intra_trust=0.8, inter_trust=0.2)
        state = DiplomacyState()
        events = tick_diplomacy(state, sg, colonists, ids, year=1, rng=random.Random(42))
        assert isinstance(events, list)
        assert state.active_faction_count == 2

    def test_tick_over_10_years_no_crash(self) -> None:
        colonists = [_make_colonist(f"c-{i}") for i in range(8)]
        ids = [c.id for c in colonists]
        sg = _high_trust_graph(ids, trust=0.7)
        state = DiplomacyState()
        rng = random.Random(42)
        for year in range(1, 11):
            for a in ids:
                for b in ids:
                    if a != b and a in sg.edges and b in sg.edges.get(a, {}):
                        rel = sg.edges[a][b]
                        rel.trust = max(0.0, min(1.0, rel.trust + rng.gauss(0, 0.05)))
            events = tick_diplomacy(state, sg, colonists, ids, year=year, rng=rng)
            assert isinstance(events, list)
        assert state.active_faction_count >= 0


# ── Engine integration ─────────────────────────────────────────────────────

class TestEngineIntegration:
    def test_engine_runs_with_diplomacy(self) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        assert result.diplomacy is not None
        d = result.to_dict()
        assert "diplomacy" in d
        assert "active_faction_count" in d["diplomacy"]

    def test_year_results_have_diplomacy(self) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=3)
        result = engine.run()
        for yr in result.years:
            d = yr.to_dict()
            assert "diplomacy" in d
            assert "self_modifications" in d
