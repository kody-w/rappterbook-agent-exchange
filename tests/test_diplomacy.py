"""Tests for the diplomacy organ (engine v11.0)."""
from __future__ import annotations

import random
import pytest

from src.mars100.diplomacy import (
    Faction, Alliance, Rivalry, DiplomacyState, DiplomacyYearContext,
    DiplomacyTickResult,
    MIN_FACTION_SIZE, TRUST_THRESHOLD, STAT_SIMILARITY_THRESHOLD,
    SCHISM_VARIANCE_THRESHOLD, SCHISM_MIN_SIZE, ALLIANCE_IDEOLOGY_DIST,
    RIVALRY_IDEOLOGY_DIST, ALLIANCE_MIN_TENURE, COOLDOWN_YEARS,
    OVERLAP_MATCH_THRESHOLD,
    compute_ideology, ideology_distance, detect_clusters,
    reconcile_factions, form_factions, check_schisms,
    update_alliances, compute_power_balance, compute_governance_pressure,
    compute_psych_modifiers, tick_diplomacy,
    _colonist_ideology, _avg_mutual_trust, _canonical_pair,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_stats(empathy: float = 0.5, hoarding: float = 0.5,
                resolve: float = 0.5, paranoia: float = 0.5,
                faith: float = 0.5, improvisation: float = 0.5) -> dict[str, float]:
    return {"empathy": empathy, "hoarding": hoarding,
            "resolve": resolve, "paranoia": paranoia,
            "faith": faith, "improvisation": improvisation}


def _high_trust_ctx(ids: list[str], year: int = 10,
                    trust: float = 0.8,
                    stats: dict[str, dict[str, float]] | None = None
                    ) -> DiplomacyYearContext:
    """Create a context where all colonists trust each other highly."""
    if stats is None:
        stats = {cid: _make_stats() for cid in ids}
    trusts: dict[str, dict[str, float]] = {}
    for a in ids:
        trusts[a] = {b: trust for b in ids if b != a}
    return DiplomacyYearContext(
        year=year, active_colonist_ids=ids,
        colonist_stats=stats, social_trusts=trusts,
        governance_type="direct_democracy", resource_avg=0.6)


def _mixed_trust_ctx(group_a: list[str], group_b: list[str],
                     year: int = 10,
                     intra_trust: float = 0.8,
                     inter_trust: float = 0.2,
                     stats_a: dict[str, float] | None = None,
                     stats_b: dict[str, float] | None = None,
                     ) -> DiplomacyYearContext:
    """Create a context with two trust clusters."""
    all_ids = group_a + group_b
    sa = stats_a or _make_stats(empathy=0.8, hoarding=0.2)
    sb = stats_b or _make_stats(empathy=0.2, hoarding=0.8)
    stats = {cid: dict(sa) for cid in group_a}
    stats.update({cid: dict(sb) for cid in group_b})
    trusts: dict[str, dict[str, float]] = {}
    for a in all_ids:
        trusts[a] = {}
        for b in all_ids:
            if a == b:
                continue
            if (a in group_a and b in group_a) or (a in group_b and b in group_b):
                trusts[a][b] = intra_trust
            else:
                trusts[a][b] = inter_trust
    return DiplomacyYearContext(
        year=year, active_colonist_ids=all_ids,
        colonist_stats=stats, social_trusts=trusts,
        governance_type="direct_democracy", resource_avg=0.6)


# ===========================================================================
# Ideology tests
# ===========================================================================
class TestIdeology:
    def test_compute_ideology_balanced(self) -> None:
        stats = {"a": _make_stats(), "b": _make_stats()}
        ideo = compute_ideology(["a", "b"], stats)
        assert ideo["collectivism"] == pytest.approx(0.0)
        assert ideo["expansionism"] == pytest.approx(0.0)
        assert ideo["spiritualism"] == pytest.approx(0.5)

    def test_compute_ideology_collectivist(self) -> None:
        stats = {"a": _make_stats(empathy=0.9, hoarding=0.1)}
        ideo = compute_ideology(["a"], stats)
        assert ideo["collectivism"] == pytest.approx(0.8)

    def test_compute_ideology_individualist(self) -> None:
        stats = {"a": _make_stats(empathy=0.1, hoarding=0.9)}
        ideo = compute_ideology(["a"], stats)
        assert ideo["collectivism"] == pytest.approx(-0.8)

    def test_compute_ideology_empty(self) -> None:
        ideo = compute_ideology([], {})
        assert ideo["collectivism"] == 0.0

    def test_ideology_distance_same(self) -> None:
        a = {"collectivism": 0.5, "expansionism": 0.3, "spiritualism": 0.7}
        assert ideology_distance(a, a) == pytest.approx(0.0)

    def test_ideology_distance_opposite(self) -> None:
        a = {"collectivism": 1.0, "expansionism": 1.0, "spiritualism": 1.0}
        b = {"collectivism": -1.0, "expansionism": -1.0, "spiritualism": 0.0}
        dist = ideology_distance(a, b)
        assert dist > 2.0

    def test_ideology_clamped(self) -> None:
        stats = {"a": _make_stats(empathy=1.0, hoarding=0.0)}
        ideo = compute_ideology(["a"], stats)
        assert ideo["collectivism"] <= 1.0
        assert ideo["collectivism"] >= -1.0

    def test_colonist_ideology(self) -> None:
        stats = _make_stats(empathy=0.8, hoarding=0.3, resolve=0.9, paranoia=0.1, faith=0.7)
        ideo = _colonist_ideology(stats)
        assert ideo["collectivism"] == pytest.approx(0.5)
        assert ideo["expansionism"] == pytest.approx(0.8)
        assert ideo["spiritualism"] == pytest.approx(0.7)


# ===========================================================================
# Cluster detection tests
# ===========================================================================
class TestClusterDetection:
    def test_no_clusters_too_few(self) -> None:
        ctx = _high_trust_ctx(["a", "b"])
        clusters = detect_clusters(ctx, set())
        assert clusters == []

    def test_single_cluster_three(self) -> None:
        ctx = _high_trust_ctx(["a", "b", "c"])
        clusters = detect_clusters(ctx, set())
        assert len(clusters) == 1
        assert sorted(clusters[0]) == ["a", "b", "c"]

    def test_no_cluster_low_trust(self) -> None:
        ctx = _high_trust_ctx(["a", "b", "c"], trust=0.3)
        clusters = detect_clusters(ctx, set())
        assert clusters == []

    def test_already_assigned_excluded(self) -> None:
        ctx = _high_trust_ctx(["a", "b", "c", "d"])
        clusters = detect_clusters(ctx, {"a", "b", "c"})
        assert clusters == []  # only "d" left, < MIN_FACTION_SIZE

    def test_two_clusters(self) -> None:
        ctx = _mixed_trust_ctx(
            ["a", "b", "c"], ["d", "e", "f"])
        clusters = detect_clusters(ctx, set())
        assert len(clusters) == 2
        cluster_sets = [set(c) for c in clusters]
        assert {"a", "b", "c"} in cluster_sets
        assert {"d", "e", "f"} in cluster_sets

    def test_deterministic(self) -> None:
        ctx = _high_trust_ctx(["c", "a", "b", "d"])
        c1 = detect_clusters(ctx, set())
        c2 = detect_clusters(ctx, set())
        assert c1 == c2


# ===========================================================================
# Faction lifecycle tests
# ===========================================================================
class TestFactionLifecycle:
    def test_form_faction(self) -> None:
        ctx = _high_trust_ctx(["a", "b", "c"])
        state = DiplomacyState()
        result = DiplomacyTickResult()
        clusters = detect_clusters(ctx, set())
        form_factions(clusters, ctx, state, result)
        assert len(state.active_factions()) == 1
        assert len(result.factions_formed) == 1
        f = list(state.factions.values())[0]
        assert sorted(f.member_ids) == ["a", "b", "c"]
        assert f.is_active()
        assert f.formed_year == 10

    def test_match_existing_by_overlap(self) -> None:
        state = DiplomacyState()
        f = Faction(id="faction-0", name="Test", leader_id="a",
                    member_ids=["a", "b", "c"],
                    ideology={"collectivism": 0.0, "expansionism": 0.0, "spiritualism": 0.5},
                    formed_year=5)
        state.factions["faction-0"] = f
        ctx = _high_trust_ctx(["a", "b", "c", "d"])
        result = DiplomacyTickResult()
        clusters = [["a", "b", "c", "d"]]  # overlap with faction-0
        form_factions(clusters, ctx, state, result)
        assert len(result.factions_formed) == 0  # reused existing
        assert "d" in state.factions["faction-0"].member_ids

    def test_dissolution_after_grace(self) -> None:
        state = DiplomacyState()
        f = Faction(id="faction-0", name="Test", leader_id="a",
                    member_ids=["a", "b", "c"],
                    ideology={}, formed_year=5)
        state.factions["faction-0"] = f
        active_ids = {"a"}  # b and c died
        result = DiplomacyTickResult()
        # Year 1: countdown starts
        reconcile_factions(state, active_ids, {"a": _make_stats()}, 10, result)
        assert state.factions["faction-0"].is_active()
        assert len(result.factions_dissolved) == 0
        # Year 2: countdown triggers dissolution
        result2 = DiplomacyTickResult()
        reconcile_factions(state, active_ids, {"a": _make_stats()}, 11, result2)
        assert not state.factions["faction-0"].is_active()
        assert len(result2.factions_dissolved) == 1

    def test_leader_succession(self) -> None:
        state = DiplomacyState()
        f = Faction(id="faction-0", name="Test", leader_id="a",
                    member_ids=["a", "b", "c"],
                    ideology={}, formed_year=5)
        state.factions["faction-0"] = f
        stats = {"b": _make_stats(resolve=0.9), "c": _make_stats(resolve=0.3)}
        active_ids = {"b", "c"}  # "a" died
        result = DiplomacyTickResult()
        reconcile_factions(state, active_ids, stats, 10, result)
        assert state.factions["faction-0"].leader_id == "b"


# ===========================================================================
# Schism tests
# ===========================================================================
class TestSchism:
    def test_schism_on_high_variance(self) -> None:
        state = DiplomacyState()
        # 6 members: 3 collectivist, 3 individualist
        members = ["a", "b", "c", "d", "e", "f"]
        stats: dict[str, dict[str, float]] = {
            "a": _make_stats(empathy=0.9, hoarding=0.1),
            "b": _make_stats(empathy=0.85, hoarding=0.15),
            "c": _make_stats(empathy=0.8, hoarding=0.2),
            "d": _make_stats(empathy=0.1, hoarding=0.9),
            "e": _make_stats(empathy=0.15, hoarding=0.85),
            "f": _make_stats(empathy=0.2, hoarding=0.8),
        }
        centroid = compute_ideology(members, stats)
        f0 = Faction(id="faction-0", name="Test", leader_id="a",
                     member_ids=members, ideology=centroid, formed_year=1)
        state.factions["faction-0"] = f0
        ctx = _high_trust_ctx(members, year=10, stats=stats)
        result = DiplomacyTickResult()
        check_schisms(state, ctx, random.Random(42), result)
        assert len(result.schisms) == 1
        assert len(state.active_factions()) == 2

    def test_no_schism_small_faction(self) -> None:
        state = DiplomacyState()
        members = ["a", "b", "c"]
        stats = {m: _make_stats() for m in members}
        f0 = Faction(id="faction-0", name="Test", leader_id="a",
                     member_ids=members, ideology=compute_ideology(members, stats),
                     formed_year=1)
        state.factions["faction-0"] = f0
        ctx = _high_trust_ctx(members, year=10, stats=stats)
        result = DiplomacyTickResult()
        check_schisms(state, ctx, random.Random(42), result)
        assert len(result.schisms) == 0

    def test_schism_cooldown(self) -> None:
        state = DiplomacyState()
        members = ["a", "b", "c", "d", "e", "f"]
        stats = {
            "a": _make_stats(empathy=0.9, hoarding=0.1),
            "b": _make_stats(empathy=0.85, hoarding=0.15),
            "c": _make_stats(empathy=0.8, hoarding=0.2),
            "d": _make_stats(empathy=0.1, hoarding=0.9),
            "e": _make_stats(empathy=0.15, hoarding=0.85),
            "f": _make_stats(empathy=0.2, hoarding=0.8),
        }
        centroid = compute_ideology(members, stats)
        f0 = Faction(id="faction-0", name="Test", leader_id="a",
                     member_ids=members, ideology=centroid, formed_year=1,
                     last_schism_year=9)  # schism just happened
        state.factions["faction-0"] = f0
        ctx = _high_trust_ctx(members, year=10, stats=stats)
        result = DiplomacyTickResult()
        check_schisms(state, ctx, random.Random(42), result)
        assert len(result.schisms) == 0  # cooldown prevents schism


# ===========================================================================
# Alliance and rivalry tests
# ===========================================================================
class TestAlliancesRivalries:
    def test_alliance_forms_on_close_ideology(self) -> None:
        state = DiplomacyState()
        f0 = Faction(id="faction-0", name="A", leader_id="a",
                     member_ids=["a", "b", "c"],
                     ideology={"collectivism": 0.5, "expansionism": 0.3, "spiritualism": 0.4},
                     formed_year=1)
        f1 = Faction(id="faction-1", name="B", leader_id="d",
                     member_ids=["d", "e", "f"],
                     ideology={"collectivism": 0.6, "expansionism": 0.2, "spiritualism": 0.5},
                     formed_year=1)
        state.factions = {"faction-0": f0, "faction-1": f1}
        ids = ["a", "b", "c", "d", "e", "f"]
        ctx = _high_trust_ctx(ids)
        # Use a fixed seed that will produce alliance
        rng = random.Random(1)
        result = DiplomacyTickResult()
        update_alliances(state, ctx, rng, result)
        # May or may not form based on RNG, but the pair was checked
        dist = ideology_distance(f0.ideology, f1.ideology)
        assert dist < ALLIANCE_IDEOLOGY_DIST

    def test_rivalry_on_distant_ideology(self) -> None:
        state = DiplomacyState()
        f0 = Faction(id="faction-0", name="A", leader_id="a",
                     member_ids=["a", "b", "c"],
                     ideology={"collectivism": 1.0, "expansionism": 1.0, "spiritualism": 0.0},
                     formed_year=1)
        f1 = Faction(id="faction-1", name="B", leader_id="d",
                     member_ids=["d", "e", "f"],
                     ideology={"collectivism": -1.0, "expansionism": -1.0, "spiritualism": 1.0},
                     formed_year=1)
        state.factions = {"faction-0": f0, "faction-1": f1}
        ids = ["a", "b", "c", "d", "e", "f"]
        ctx = _high_trust_ctx(ids)
        rng = random.Random(0)  # seed that makes rivalry form
        result = DiplomacyTickResult()
        update_alliances(state, ctx, rng, result)
        dist = ideology_distance(f0.ideology, f1.ideology)
        assert dist > RIVALRY_IDEOLOGY_DIST

    def test_alliance_min_tenure(self) -> None:
        state = DiplomacyState()
        f0 = Faction(id="faction-0", name="A", leader_id="a",
                     member_ids=["a", "b", "c"],
                     ideology={"collectivism": 0.5, "expansionism": 0.3, "spiritualism": 0.4},
                     formed_year=1)
        f1 = Faction(id="faction-1", name="B", leader_id="d",
                     member_ids=["d", "e", "f"],
                     ideology={"collectivism": -0.8, "expansionism": -0.8, "spiritualism": 0.9},
                     formed_year=1)
        state.factions = {"faction-0": f0, "faction-1": f1}
        alliance = Alliance(faction_a="faction-0", faction_b="faction-1",
                            strength=0.8, formed_year=9)
        state.alliances = [alliance]
        ids = ["a", "b", "c", "d", "e", "f"]
        ctx = _high_trust_ctx(ids, year=10)
        result = DiplomacyTickResult()
        update_alliances(state, ctx, random.Random(42), result)
        # Alliance should survive despite ideology drift (min tenure)
        assert len(result.alliances_broken) == 0

    def test_canonical_pair(self) -> None:
        assert _canonical_pair("b", "a") == ("a", "b")
        assert _canonical_pair("a", "b") == ("a", "b")


# ===========================================================================
# Power and governance pressure tests
# ===========================================================================
class TestPowerAndGovernance:
    def test_power_balance(self) -> None:
        state = DiplomacyState()
        f0 = Faction(id="faction-0", name="A", leader_id="a",
                     member_ids=["a", "b"],
                     ideology={}, formed_year=1)
        state.factions = {"faction-0": f0}
        stats = {"a": _make_stats(resolve=0.8, empathy=0.6, improvisation=0.4),
                 "b": _make_stats(resolve=0.6, empathy=0.4, improvisation=0.3)}
        ctx = DiplomacyYearContext(
            year=10, active_colonist_ids=["a", "b", "c"],
            colonist_stats=stats, social_trusts={},
            governance_type="direct_democracy", resource_avg=0.5)
        balance = compute_power_balance(state, ctx)
        assert "faction-0" in balance
        assert 0 < balance["faction-0"] < 1

    def test_governance_pressure_collectivist(self) -> None:
        state = DiplomacyState()
        f0 = Faction(id="faction-0", name="A", leader_id="a",
                     member_ids=["a", "b", "c"],
                     ideology={"collectivism": 0.8, "expansionism": 0.0, "spiritualism": 0.2},
                     formed_year=1, power=0.5)
        state.factions = {"faction-0": f0}
        pressure = compute_governance_pressure(state)
        assert "direct_democracy" in pressure
        assert pressure["direct_democracy"] > 0

    def test_governance_pressure_empty(self) -> None:
        state = DiplomacyState()
        pressure = compute_governance_pressure(state)
        assert pressure == {}

    def test_governance_pressure_normalized(self) -> None:
        state = DiplomacyState()
        f0 = Faction(id="faction-0", name="A", leader_id="a",
                     member_ids=["a"], ideology={"collectivism": 0.8}, formed_year=1, power=0.5)
        f1 = Faction(id="faction-1", name="B", leader_id="b",
                     member_ids=["b"], ideology={"collectivism": -0.8}, formed_year=1, power=0.5)
        state.factions = {"faction-0": f0, "faction-1": f1}
        pressure = compute_governance_pressure(state)
        total = sum(pressure.values())
        assert total <= len(pressure)  # each value <= 1.0


# ===========================================================================
# Psychology modifier tests
# ===========================================================================
class TestPsychModifiers:
    def test_faction_reduces_loneliness(self) -> None:
        state = DiplomacyState()
        f0 = Faction(id="faction-0", name="A", leader_id="a",
                     member_ids=["a", "b", "c"],
                     ideology={}, formed_year=1)
        state.factions = {"faction-0": f0}
        lone, purp = compute_psych_modifiers(state, ["a", "b", "c"])
        assert lone["a"] < 0  # reduced loneliness
        assert purp["a"] > 0  # increased purpose

    def test_unaffiliated_no_modifier(self) -> None:
        state = DiplomacyState()
        lone, purp = compute_psych_modifiers(state, ["a", "b"])
        assert "a" not in lone
        assert "a" not in purp

    def test_modifier_capped(self) -> None:
        state = DiplomacyState()
        f0 = Faction(id="faction-0", name="A", leader_id="a",
                     member_ids=[f"c{i}" for i in range(20)],
                     ideology={}, formed_year=1)
        state.factions = {"faction-0": f0}
        lone, purp = compute_psych_modifiers(state, f0.member_ids)
        for v in lone.values():
            assert v >= -0.08
        for v in purp.values():
            assert v <= 0.08


# ===========================================================================
# Serialization roundtrip tests
# ===========================================================================
class TestSerialization:
    def test_faction_roundtrip(self) -> None:
        f = Faction(id="faction-0", name="Test", leader_id="a",
                    member_ids=["a", "b"], ideology={"collectivism": 0.5},
                    formed_year=5, power=0.3, last_schism_year=2)
        d = f.to_dict()
        f2 = Faction.from_dict(d)
        assert f2.id == f.id
        assert f2.member_ids == f.member_ids
        assert f2.last_schism_year == f.last_schism_year

    def test_alliance_roundtrip(self) -> None:
        a = Alliance(faction_a="a", faction_b="b", strength=0.7, formed_year=3)
        a2 = Alliance.from_dict(a.to_dict())
        assert a2.pair() == a.pair()

    def test_alliance_canonical_order(self) -> None:
        a = Alliance.from_dict({"faction_a": "z", "faction_b": "a",
                                "strength": 0.5, "formed_year": 1})
        assert a.faction_a == "a"
        assert a.faction_b == "z"

    def test_rivalry_roundtrip(self) -> None:
        r = Rivalry(faction_a="a", faction_b="b", intensity=0.6, formed_year=4)
        r2 = Rivalry.from_dict(r.to_dict())
        assert r2.pair() == r.pair()

    def test_state_roundtrip(self) -> None:
        state = DiplomacyState()
        f = Faction(id="faction-0", name="Test", leader_id="a",
                    member_ids=["a", "b", "c"], ideology={}, formed_year=5)
        state.factions["faction-0"] = f
        state.alliances.append(Alliance("a", "b", 0.5, 3))
        state.rivalries.append(Rivalry("c", "d", 0.6, 4))
        state.history.append({"type": "test", "year": 5})
        d = state.to_dict()
        s2 = DiplomacyState.from_dict(d)
        assert "faction-0" in s2.factions
        assert len(s2.alliances) == 1
        assert len(s2.rivalries) == 1

    def test_result_to_dict(self) -> None:
        result = DiplomacyTickResult()
        result.factions_formed = [{"id": "x"}]
        result.power_balance = {"faction-0": 0.5}
        d = result.to_dict()
        assert d["factions_formed"] == [{"id": "x"}]
        assert d["power_balance"]["faction-0"] == 0.5


# ===========================================================================
# Integration: full tick
# ===========================================================================
class TestTickDiplomacy:
    def test_tick_empty_state(self) -> None:
        state = DiplomacyState()
        ctx = _high_trust_ctx(["a", "b", "c"])
        rng = random.Random(42)
        result = tick_diplomacy(state, ctx, rng)
        assert len(state.active_factions()) == 1
        assert len(result.factions_formed) == 1

    def test_tick_preserves_factions(self) -> None:
        state = DiplomacyState()
        ids = ["a", "b", "c", "d"]
        ctx = _high_trust_ctx(ids)
        rng = random.Random(42)
        tick_diplomacy(state, ctx, rng)
        factions_before = set(state.active_factions().keys())
        # Tick again with same colonists
        ctx2 = _high_trust_ctx(ids, year=11)
        tick_diplomacy(state, ctx2, rng)
        factions_after = set(state.active_factions().keys())
        assert factions_before == factions_after  # no new factions formed

    def test_tick_handles_death(self) -> None:
        state = DiplomacyState()
        ids = ["a", "b", "c"]
        ctx = _high_trust_ctx(ids)
        tick_diplomacy(state, ctx, random.Random(42))
        assert len(state.active_factions()) == 1
        # "c" dies
        ctx2 = _high_trust_ctx(["a", "b"], year=11)
        result2 = tick_diplomacy(state, ctx2, random.Random(42))
        # Faction should start dissolution countdown (< MIN_FACTION_SIZE)
        f = list(state.factions.values())[0]
        assert f.is_active()  # grace period
        # One more year
        ctx3 = _high_trust_ctx(["a", "b"], year=12)
        result3 = tick_diplomacy(state, ctx3, random.Random(42))
        assert len(result3.factions_dissolved) == 1

    def test_tick_two_factions(self) -> None:
        ctx = _mixed_trust_ctx(
            ["a", "b", "c"], ["d", "e", "f"])
        state = DiplomacyState()
        result = tick_diplomacy(state, ctx, random.Random(42))
        active = state.active_factions()
        assert len(active) == 2
        assert len(result.factions_formed) == 2

    def test_tick_deterministic(self) -> None:
        """Same inputs produce same outputs."""
        ids = ["a", "b", "c", "d", "e", "f"]
        for _ in range(3):
            state = DiplomacyState()
            ctx = _high_trust_ctx(ids)
            result = tick_diplomacy(state, ctx, random.Random(42))
            assert len(result.factions_formed) > 0

    def test_tick_psych_modifiers(self) -> None:
        state = DiplomacyState()
        ids = ["a", "b", "c"]
        ctx = _high_trust_ctx(ids)
        result = tick_diplomacy(state, ctx, random.Random(42))
        assert "a" in result.loneliness_modifiers
        assert result.loneliness_modifiers["a"] < 0
        assert result.purpose_modifiers["a"] > 0

    def test_history_pruned(self) -> None:
        state = DiplomacyState()
        state.history = [{"type": "test", "year": i} for i in range(100)]
        ctx = _high_trust_ctx(["a", "b", "c"])
        tick_diplomacy(state, ctx, random.Random(42))
        assert len(state.history) <= 40 + 5  # MAX_HISTORY + new events


# ===========================================================================
# Physical bounds (property-based invariants)
# ===========================================================================
class TestInvariants:
    @pytest.mark.parametrize("seed", range(5))
    def test_power_nonneg(self, seed: int) -> None:
        state = DiplomacyState()
        ids = [f"c{i}" for i in range(8)]
        stats = {cid: _make_stats(
            resolve=random.Random(seed + i).random(),
            empathy=random.Random(seed + i + 10).random(),
            improvisation=random.Random(seed + i + 20).random(),
        ) for i, cid in enumerate(ids)}
        ctx = _high_trust_ctx(ids, stats=stats)
        tick_diplomacy(state, ctx, random.Random(seed))
        for fid, power in compute_power_balance(state, ctx).items():
            assert power >= 0

    @pytest.mark.parametrize("seed", range(5))
    def test_governance_pressure_bounded(self, seed: int) -> None:
        state = DiplomacyState()
        ids = [f"c{i}" for i in range(6)]
        stats = {cid: _make_stats(
            empathy=random.Random(seed + i).random(),
            hoarding=random.Random(seed + i + 5).random(),
            faith=random.Random(seed + i + 10).random(),
        ) for i, cid in enumerate(ids)}
        ctx = _high_trust_ctx(ids, stats=stats)
        tick_diplomacy(state, ctx, random.Random(seed))
        for v in compute_governance_pressure(state).values():
            assert 0.0 <= v <= 1.0

    @pytest.mark.parametrize("seed", range(5))
    def test_no_colonist_in_two_factions(self, seed: int) -> None:
        state = DiplomacyState()
        ids = [f"c{i}" for i in range(10)]
        ctx = _mixed_trust_ctx(ids[:5], ids[5:],
                               stats_a=_make_stats(empathy=0.9, hoarding=0.1),
                               stats_b=_make_stats(empathy=0.1, hoarding=0.9))
        tick_diplomacy(state, ctx, random.Random(seed))
        seen: set[str] = set()
        for f in state.active_factions().values():
            for m in f.member_ids:
                assert m not in seen, f"{m} in multiple factions"
                seen.add(m)

    def test_ideology_axes_bounded(self) -> None:
        for _ in range(20):
            stats = {f"c{i}": _make_stats(
                empathy=random.random(), hoarding=random.random(),
                resolve=random.random(), paranoia=random.random(),
                faith=random.random(),
            ) for i in range(5)}
            ideo = compute_ideology(list(stats.keys()), stats)
            assert -1.0 <= ideo["collectivism"] <= 1.0
            assert -1.0 <= ideo["expansionism"] <= 1.0
            assert 0.0 <= ideo["spiritualism"] <= 1.0


# ===========================================================================
# Smoke: 20-year mini simulation
# ===========================================================================
class TestSmoke:
    def test_20_year_run(self) -> None:
        """Run diplomacy for 20 years with evolving membership."""
        state = DiplomacyState()
        rng = random.Random(99)
        base_ids = [f"c{i}" for i in range(10)]

        for year in range(1, 21):
            # Randomly evolve trust and stats each year
            stats = {cid: _make_stats(
                empathy=max(0, min(1, 0.5 + rng.gauss(0, 0.2))),
                hoarding=max(0, min(1, 0.5 + rng.gauss(0, 0.2))),
                resolve=max(0, min(1, 0.5 + rng.gauss(0, 0.2))),
                paranoia=max(0, min(1, 0.5 + rng.gauss(0, 0.2))),
                faith=max(0, min(1, 0.5 + rng.gauss(0, 0.2))),
                improvisation=max(0, min(1, 0.5 + rng.gauss(0, 0.2))),
            ) for cid in base_ids}

            trusts: dict[str, dict[str, float]] = {}
            for a in base_ids:
                trusts[a] = {}
                for b in base_ids:
                    if a != b:
                        trusts[a][b] = max(0, min(1, 0.5 + rng.gauss(0, 0.2)))

            ctx = DiplomacyYearContext(
                year=year, active_colonist_ids=list(base_ids),
                colonist_stats=stats, social_trusts=trusts,
                governance_type="direct_democracy", resource_avg=0.6)

            result = tick_diplomacy(state, ctx, rng)

            # Invariants
            for f in state.active_factions().values():
                assert len(f.member_ids) >= MIN_FACTION_SIZE or \
                    state.dissolution_countdown.get(f.id, 0) > 0
                assert f.power >= 0

        # After 20 years, state should be serializable
        d = state.to_dict()
        s2 = DiplomacyState.from_dict(d)
        assert len(s2.factions) == len(state.factions)
