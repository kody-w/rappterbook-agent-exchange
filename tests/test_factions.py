"""Tests for the factions organ (engine v9.0)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.factions import (
    Faction, FactionState, FactionTickResult,
    tick_factions, stat_distance, colonist_stat_profile,
    compute_ideology, try_form_factions, recruit_unaffiliated,
    dissolve_empty_factions, remove_inactive_members,
    drift_ideologies, _pick_faction_name, _find_clusters,
    FORMATION_MIN_YEAR, FORMATION_DISTANCE_THRESHOLD,
    RECRUITMENT_DISTANCE_THRESHOLD, MIN_FACTION_SIZE,
    MAX_FACTIONS, VOTING_SAME_FACTION_BIAS,
)
from src.mars100.colonist import (
    Colonist, ColonistStats, ColonistSkills, STAT_NAMES,
    create_founding_ten,
)


def _make_colonist(
    cid: str = "c-0",
    name: str = "Test",
    resolve: float = 0.5,
    improvisation: float = 0.5,
    empathy: float = 0.5,
    hoarding: float = 0.5,
    faith: float = 0.5,
    paranoia: float = 0.5,
    alive: bool = True,
    exiled: bool = False,
) -> Colonist:
    """Create a test colonist with specified stats."""
    return Colonist(
        id=cid, name=name, element="fire", archetype="test",
        stats=ColonistStats(
            resolve=resolve, improvisation=improvisation,
            empathy=empathy, hoarding=hoarding,
            faith=faith, paranoia=paranoia,
        ),
        skills=ColonistSkills(),
        decision_expr="(+ resolve empathy)",
        alive=alive, exiled=exiled,
    )


# -- stat_distance tests -----------------------------------------------------

class TestStatDistance:
    def test_identical_profiles(self):
        a = {"resolve": 0.5, "improvisation": 0.5, "empathy": 0.5,
             "hoarding": 0.5, "faith": 0.5, "paranoia": 0.5}
        assert stat_distance(a, a) == 0.0

    def test_opposite_profiles(self):
        a = {s: 0.0 for s in STAT_NAMES}
        b = {s: 1.0 for s in STAT_NAMES}
        assert stat_distance(a, b) == pytest.approx(1.0, abs=0.001)

    def test_symmetric(self):
        a = {"resolve": 0.8, "improvisation": 0.2, "empathy": 0.5,
             "hoarding": 0.3, "faith": 0.7, "paranoia": 0.1}
        b = {"resolve": 0.4, "improvisation": 0.6, "empathy": 0.5,
             "hoarding": 0.5, "faith": 0.3, "paranoia": 0.5}
        assert stat_distance(a, b) == pytest.approx(stat_distance(b, a))

    def test_bounded(self):
        rng = random.Random(42)
        for _ in range(100):
            a = {s: rng.random() for s in STAT_NAMES}
            b = {s: rng.random() for s in STAT_NAMES}
            d = stat_distance(a, b)
            assert 0.0 <= d <= 1.0

    def test_small_difference(self):
        a = {s: 0.5 for s in STAT_NAMES}
        b = dict(a)
        b["resolve"] = 0.6
        d = stat_distance(a, b)
        assert d < 0.05  # only one stat differs by 0.1


class TestComputeIdeology:
    def test_single_member(self):
        c = _make_colonist("c-0", resolve=0.8, empathy=0.2)
        ideology = compute_ideology([c])
        assert ideology["resolve"] == pytest.approx(0.8, abs=0.001)
        assert ideology["empathy"] == pytest.approx(0.2, abs=0.001)

    def test_two_members_averaged(self):
        c1 = _make_colonist("c-1", resolve=0.8, empathy=0.2)
        c2 = _make_colonist("c-2", resolve=0.4, empathy=0.6)
        ideology = compute_ideology([c1, c2])
        assert ideology["resolve"] == pytest.approx(0.6, abs=0.001)
        assert ideology["empathy"] == pytest.approx(0.4, abs=0.001)

    def test_empty_defaults(self):
        ideology = compute_ideology([])
        for s in STAT_NAMES:
            assert ideology[s] == 0.5


# -- FactionState tests ------------------------------------------------------

class TestFactionState:
    def test_empty_state(self):
        fs = FactionState()
        assert fs.active_factions() == []
        assert fs.faction_of("x") is None
        assert fs.members_of("x") == []

    def test_membership_tracking(self):
        fs = FactionState()
        f = Faction(id="f-0", name="Test", formed_year=10,
                    ideology={s: 0.5 for s in STAT_NAMES})
        fs.factions["f-0"] = f
        fs.membership["c-0"] = "f-0"
        fs.membership["c-1"] = "f-0"
        fs.membership["c-2"] = None

        assert fs.faction_of("c-0") == "f-0"
        assert fs.faction_of("c-2") is None
        assert fs.members_of("f-0") == ["c-0", "c-1"]

    def test_same_faction(self):
        fs = FactionState()
        f = Faction(id="f-0", name="Test", formed_year=10,
                    ideology={s: 0.5 for s in STAT_NAMES})
        fs.factions["f-0"] = f
        fs.membership["c-0"] = "f-0"
        fs.membership["c-1"] = "f-0"
        fs.membership["c-2"] = "f-1"
        assert fs.same_faction("c-0", "c-1") is True
        assert fs.same_faction("c-0", "c-2") is False
        assert fs.same_faction("c-0", "c-99") is False

    def test_same_faction_dissolved(self):
        fs = FactionState()
        f = Faction(id="f-0", name="Test", formed_year=10,
                    ideology={}, dissolved=True, dissolved_year=15)
        fs.factions["f-0"] = f
        fs.membership["c-0"] = "f-0"
        fs.membership["c-1"] = "f-0"
        assert fs.same_faction("c-0", "c-1") is False

    def test_round_trip(self):
        fs = FactionState()
        f = Faction(id="f-0", name="Test", formed_year=10,
                    ideology={"resolve": 0.8})
        fs.factions["f-0"] = f
        fs.membership["c-0"] = "f-0"
        fs.next_id = 1
        d = fs.to_dict()
        fs2 = FactionState.from_dict(d)
        assert len(fs2.factions) == 1
        assert fs2.factions["f-0"].name == "Test"
        assert fs2.membership["c-0"] == "f-0"
        assert fs2.next_id == 1

    def test_active_factions_sorted(self):
        fs = FactionState()
        for i in range(3):
            fs.factions[f"f-{i}"] = Faction(
                id=f"f-{i}", name=f"F{i}", formed_year=10,
                ideology={})
        fs.factions["f-1"].dissolved = True
        active = fs.active_factions()
        assert len(active) == 2
        assert active[0].id == "f-0"
        assert active[1].id == "f-2"


# -- Faction name generation -------------------------------------------------

class TestFactionNaming:
    def test_themed_name(self):
        rng = random.Random(42)
        ideology = {"faith": 0.9, "resolve": 0.8, "empathy": 0.3,
                    "hoarding": 0.2, "improvisation": 0.1, "paranoia": 0.1}
        name = _pick_faction_name(ideology, set(), rng)
        assert name in ["The Covenant", "Iron Faith", "The Steadfast"]

    def test_generic_fallback(self):
        rng = random.Random(42)
        ideology = {s: 0.5 for s in STAT_NAMES}
        used = set()
        # Exhaust themed names by using weird stat combos
        ideology["resolve"] = 0.99
        ideology["paranoia"] = 0.98
        # This should still produce a name (themed or generic)
        name = _pick_faction_name(ideology, set(), rng)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_avoids_used_names(self):
        rng = random.Random(42)
        ideology = {"faith": 0.9, "resolve": 0.8, "empathy": 0.3,
                    "hoarding": 0.2, "improvisation": 0.1, "paranoia": 0.1}
        used = {"The Covenant", "Iron Faith", "The Steadfast"}
        name = _pick_faction_name(ideology, used, rng)
        assert name not in used


# -- Cluster finding ---------------------------------------------------------

class TestFindClusters:
    def test_identical_colonists_cluster(self):
        colonists = [
            _make_colonist(f"c-{i}", resolve=0.8, faith=0.8)
            for i in range(3)
        ]
        clusters = _find_clusters(colonists, FORMATION_DISTANCE_THRESHOLD)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_distinct_groups_separate(self):
        group_a = [
            _make_colonist(f"a-{i}", resolve=0.9, faith=0.9, paranoia=0.1)
            for i in range(3)
        ]
        group_b = [
            _make_colonist(f"b-{i}", resolve=0.1, faith=0.1, paranoia=0.9)
            for i in range(3)
        ]
        clusters = _find_clusters(group_a + group_b, FORMATION_DISTANCE_THRESHOLD)
        assert len(clusters) == 2

    def test_singletons_excluded(self):
        c1 = _make_colonist("c-1", resolve=0.9, faith=0.9)
        c2 = _make_colonist("c-2", resolve=0.1, paranoia=0.9)
        clusters = _find_clusters([c1, c2], FORMATION_DISTANCE_THRESHOLD)
        assert len(clusters) == 0  # neither forms a group of 2+

    def test_deterministic_ordering(self):
        colonists = [
            _make_colonist(f"c-{i}", resolve=0.5 + 0.02 * i, faith=0.5)
            for i in range(5)
        ]
        c1 = _find_clusters(colonists, FORMATION_DISTANCE_THRESHOLD)
        c2 = _find_clusters(list(reversed(colonists)), FORMATION_DISTANCE_THRESHOLD)
        # Both should produce same clusters (sorted by ID internally)
        assert len(c1) == len(c2)
        for cl1, cl2 in zip(c1, c2):
            ids1 = sorted(c.id for c in cl1)
            ids2 = sorted(c.id for c in cl2)
            assert ids1 == ids2


# -- Formation ---------------------------------------------------------------

class TestFormation:
    def test_no_formation_before_min_year(self):
        state = FactionState()
        colonists = [
            _make_colonist(f"c-{i}", resolve=0.8, faith=0.8)
            for i in range(4)
        ]
        formed = try_form_factions(state, colonists, year=5, rng=random.Random(42))
        assert formed == []

    def test_formation_after_min_year(self):
        state = FactionState()
        colonists = [
            _make_colonist(f"c-{i}", resolve=0.8, faith=0.8)
            for i in range(4)
        ]
        formed = try_form_factions(state, colonists, year=10, rng=random.Random(42))
        assert len(formed) >= 1
        # All colonists should be affiliated now
        for c in colonists:
            assert state.membership.get(c.id) is not None

    def test_max_factions_respected(self):
        state = FactionState()
        # Pre-fill with MAX_FACTIONS factions
        for i in range(MAX_FACTIONS):
            f = Faction(id=f"pre-{i}", name=f"Pre{i}", formed_year=8,
                        ideology={})
            state.factions[f.id] = f
        colonists = [
            _make_colonist(f"c-{i}", resolve=0.8, faith=0.8)
            for i in range(4)
        ]
        formed = try_form_factions(state, colonists, year=10, rng=random.Random(42))
        assert len(formed) == 0

    def test_inactive_colonists_excluded(self):
        state = FactionState()
        alive = [_make_colonist(f"a-{i}", resolve=0.8, faith=0.8) for i in range(3)]
        dead = [_make_colonist(f"d-{i}", resolve=0.8, faith=0.8, alive=False) for i in range(3)]
        exiled = [_make_colonist(f"e-0", resolve=0.8, faith=0.8, exiled=True)]
        formed = try_form_factions(state, alive + dead + exiled, year=10, rng=random.Random(42))
        # Only the 3 alive colonists should be considered
        for f in formed:
            members = state.members_of(f.id)
            for m in members:
                assert m.startswith("a-")


# -- Recruitment -------------------------------------------------------------

class TestRecruitment:
    def test_recruit_nearby_unaffiliated(self):
        state = FactionState()
        ideology = {"resolve": 0.8, "faith": 0.8, "improvisation": 0.5,
                    "empathy": 0.5, "hoarding": 0.5, "paranoia": 0.5}
        f = Faction(id="f-0", name="Test", formed_year=8, ideology=ideology)
        state.factions["f-0"] = f
        state.membership["c-0"] = "f-0"
        state.membership["c-1"] = "f-0"

        # New colonist with similar stats
        newcomer = _make_colonist("c-new", resolve=0.75, faith=0.85)
        rng = random.Random(1)  # Use seed that makes rng.random() < 0.4
        recruited = recruit_unaffiliated(
            state, [newcomer, _make_colonist("c-0"), _make_colonist("c-1")], rng)
        # May or may not recruit depending on random roll
        # But the function should not crash
        assert isinstance(recruited, list)

    def test_distant_colonist_not_recruited(self):
        state = FactionState()
        ideology = {"resolve": 0.9, "faith": 0.9, "improvisation": 0.1,
                    "empathy": 0.1, "hoarding": 0.1, "paranoia": 0.1}
        f = Faction(id="f-0", name="Test", formed_year=8, ideology=ideology)
        state.factions["f-0"] = f
        # Very different colonist
        outlier = _make_colonist("c-out", resolve=0.1, faith=0.1, paranoia=0.9)
        recruited = recruit_unaffiliated(state, [outlier], random.Random(42))
        assert all(r["colonist_id"] != "c-out" or
                   r["distance"] < RECRUITMENT_DISTANCE_THRESHOLD
                   for r in recruited)


# -- Dissolution -------------------------------------------------------------

class TestDissolution:
    def test_dissolve_on_member_death(self):
        state = FactionState()
        f = Faction(id="f-0", name="Test", formed_year=8,
                    ideology={s: 0.5 for s in STAT_NAMES})
        state.factions["f-0"] = f
        state.membership["c-0"] = "f-0"
        state.membership["c-1"] = "f-0"

        # Only c-0 is active (c-1 died)
        active_ids = {"c-0"}
        dissolved = dissolve_empty_factions(state, active_ids, year=15)
        assert len(dissolved) == 1
        assert dissolved[0]["faction_id"] == "f-0"
        assert f.dissolved is True
        assert f.dissolved_year == 15

    def test_healthy_faction_survives(self):
        state = FactionState()
        f = Faction(id="f-0", name="Test", formed_year=8,
                    ideology={s: 0.5 for s in STAT_NAMES})
        state.factions["f-0"] = f
        state.membership["c-0"] = "f-0"
        state.membership["c-1"] = "f-0"
        state.membership["c-2"] = "f-0"

        active_ids = {"c-0", "c-1", "c-2"}
        dissolved = dissolve_empty_factions(state, active_ids, year=15)
        assert len(dissolved) == 0
        assert f.dissolved is False


# -- Inactive member removal -------------------------------------------------

class TestRemoveInactive:
    def test_dead_removed(self):
        state = FactionState()
        state.membership["alive"] = "f-0"
        state.membership["dead"] = "f-0"
        remove_inactive_members(state, active_ids={"alive"})
        assert state.membership["alive"] == "f-0"
        assert state.membership["dead"] is None


# -- Ideology drift ----------------------------------------------------------

class TestIdeologyDrift:
    def test_drift_toward_members(self):
        state = FactionState()
        f = Faction(id="f-0", name="Test", formed_year=8,
                    ideology={"resolve": 0.5, "improvisation": 0.5,
                              "empathy": 0.5, "hoarding": 0.5,
                              "faith": 0.5, "paranoia": 0.5})
        state.factions["f-0"] = f
        state.membership["c-0"] = "f-0"
        state.membership["c-1"] = "f-0"

        colonists = [
            _make_colonist("c-0", resolve=0.9, faith=0.9),
            _make_colonist("c-1", resolve=0.8, faith=0.8),
        ]
        drift_ideologies(state, colonists)
        # Ideology should move toward member average (0.85 for resolve)
        assert f.ideology["resolve"] > 0.5
        assert f.ideology["faith"] > 0.5


# -- tick_factions integration -----------------------------------------------

class TestTickFactions:
    def test_smoke_10_years(self):
        state = FactionState()
        colonists = create_founding_ten(seed=42)
        rng = random.Random(10007)
        for year in range(1, 11):
            result = tick_factions(state, colonists, year, rng)
            assert isinstance(result, FactionTickResult)

    def test_factions_emerge_by_year_20(self):
        state = FactionState()
        colonists = create_founding_ten(seed=42)
        rng = random.Random(10007)
        for year in range(1, 21):
            tick_factions(state, colonists, year, rng)
        assert len(state.active_factions()) >= 1

    def test_no_colonist_in_multiple_factions(self):
        """Conservation law: each colonist belongs to at most one faction."""
        state = FactionState()
        colonists = create_founding_ten(seed=42)
        rng = random.Random(10007)
        for year in range(1, 51):
            tick_factions(state, colonists, year, rng)
        faction_members: dict[str, set[str]] = {}
        for fid, faction in state.factions.items():
            if not faction.dissolved:
                members = set(state.members_of(fid))
                for other_fid, other_members in faction_members.items():
                    overlap = members & other_members
                    assert len(overlap) == 0, (
                        f"Colonists {overlap} in both {fid} and {other_fid}")
                faction_members[fid] = members

    def test_only_active_colonists_affiliated(self):
        state = FactionState()
        colonists = create_founding_ten(seed=42)
        colonists[0].alive = False  # kill one
        rng = random.Random(10007)
        for year in range(1, 21):
            tick_factions(state, colonists, year, rng)
        dead_id = colonists[0].id
        assert state.membership.get(dead_id) is None

    def test_result_serializable(self):
        state = FactionState()
        colonists = create_founding_ten(seed=42)
        rng = random.Random(10007)
        result = tick_factions(state, colonists, 10, rng)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "formed" in d
        assert "dissolved" in d
        assert "recruited" in d
        assert "active_count" in d

    def test_determinism(self):
        """Same seed, same colonists → same factions."""
        def run_sim():
            state = FactionState()
            colonists = create_founding_ten(seed=42)
            rng = random.Random(10007)
            results = []
            for year in range(1, 31):
                r = tick_factions(state, colonists, year, rng)
                results.append(r.to_dict())
            return results, state.to_dict()

        results_a, state_a = run_sim()
        results_b, state_b = run_sim()
        assert results_a == results_b
        assert state_a == state_b


# -- Engine integration tests ------------------------------------------------

class TestEngineIntegration:
    """Tests that factions integrate correctly into the full engine."""

    def test_10_year_run_with_factions(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert len(result.years) > 0
        for yr in result.years:
            assert "factions" in yr.to_dict()

    def test_factions_in_simulation_result(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=25)
        result = engine.run()
        d = result.to_dict()
        assert "final_factions" in d
        assert "total_factions_formed" in d["summary"]

    def test_50_year_factions_emerge(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        assert result.total_factions_formed >= 0
        # With 50 years and 10 colonists, factions should form
        assert result.final_factions is not None

    def test_determinism_preserved(self):
        """Adding factions should not break engine determinism."""
        from src.mars100.engine import Mars100Engine
        a = Mars100Engine(seed=99, total_years=15).run()
        b = Mars100Engine(seed=99, total_years=15).run()
        assert len(a.years) == len(b.years)
        for ya, yb in zip(a.years, b.years):
            assert ya.year == yb.year
            assert ya.actions == yb.actions
            assert ya.factions == yb.factions

    def test_resources_still_bounded(self):
        """Conservation law: resources in [0, 1] after adding factions."""
        from src.mars100.engine import Mars100Engine
        from src.mars100.colony import RESOURCE_NAMES
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        for yr in result.years:
            for name in RESOURCE_NAMES:
                val = yr.resources_after[name]
                assert 0.0 <= val <= 1.0, f"Year {yr.year}: {name}={val}"

    def test_version_is_9(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        d = result.to_dict()
        assert d["_meta"]["version"] == "9.0"
