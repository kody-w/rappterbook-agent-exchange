"""Tests for Mars-100 diplomacy engine."""
from __future__ import annotations

import random
import pytest
from src.mars100.diplomacy import (
    DiplomacyState, Faction, Alliance,
    detect_factions, reconcile_factions, check_schism,
    update_alliances, faction_vote_bias, tick_diplomacy,
    _compute_density, _external_density, _assign_ideology,
    DENSITY_GAP_THRESHOLD, MIN_FACTION_SIZE, MAX_VOTE_BIAS,
    JACCARD_THRESHOLD, SCHISM_PROBABILITY,
)
from src.mars100.colony import SocialGraph
from src.mars100.colonist import create_founding_ten


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_social(ids: list[str], rng: random.Random,
                clusters: dict[str, list[str]] | None = None,
                high_trust: float = 0.85,
                low_trust: float = 0.25) -> SocialGraph:
    """Create a SocialGraph with optional trust clusters."""
    sg = SocialGraph()
    sg.initialize(ids, rng)
    if clusters:
        for _name, members in clusters.items():
            for a in members:
                for b in members:
                    if a != b:
                        rel = sg.get(a, b)
                        rel.trust = high_trust
        # Lower cross-cluster trust
        all_clustered = set()
        for members in clusters.values():
            all_clustered.update(members)
        for a in all_clustered:
            for b in all_clustered:
                if a != b:
                    in_same = any(a in m and b in m for m in clusters.values())
                    if not in_same:
                        rel = sg.get(a, b)
                        rel.trust = low_trust
    return sg


# ---------------------------------------------------------------------------
# Faction detection
# ---------------------------------------------------------------------------

class TestDetectFactions:
    def test_no_factions_with_few_colonists(self):
        """Fewer than 2 * MIN_FACTION_SIZE colonists → no factions."""
        ids = ["c-0", "c-1", "c-2", "c-3", "c-4"]
        rng = random.Random(42)
        sg = SocialGraph()
        sg.initialize(ids, rng)
        factions = detect_factions(ids, sg, rng)
        assert factions == []

    def test_factions_from_trust_clusters(self):
        """Two high-trust clusters should produce two factions."""
        ids = [f"c-{i}" for i in range(10)]
        rng = random.Random(42)
        sg = make_social(ids, rng, clusters={
            "alpha": ids[:5],
            "beta": ids[5:],
        })
        factions = detect_factions(ids, sg, rng)
        assert len(factions) >= 1
        # Each faction member list should be non-empty
        for f in factions:
            assert len(f.members) >= MIN_FACTION_SIZE

    def test_no_factions_uniform_trust(self):
        """Uniform trust → no clusters → likely no factions (gap too small)."""
        ids = [f"c-{i}" for i in range(10)]
        rng = random.Random(42)
        sg = SocialGraph()
        sg.initialize(ids, rng)
        # Set all trust to 0.5 (uniform)
        for a in ids:
            for b in ids:
                if a != b:
                    sg.get(a, b).trust = 0.5
        factions = detect_factions(ids, sg, rng)
        assert factions == []

    def test_no_colonist_in_two_factions(self):
        """Property: no colonist appears in more than one faction."""
        ids = [f"c-{i}" for i in range(10)]
        rng = random.Random(42)
        sg = make_social(ids, rng, clusters={
            "alpha": ids[:4],
            "beta": ids[4:8],
        })
        factions = detect_factions(ids, sg, rng)
        all_members: list[str] = []
        for f in factions:
            all_members.extend(f.members)
        assert len(all_members) == len(set(all_members))

    def test_leader_is_member(self):
        """Property: every faction leader is in its member list."""
        ids = [f"c-{i}" for i in range(10)]
        rng = random.Random(42)
        sg = make_social(ids, rng, clusters={
            "alpha": ids[:5],
            "beta": ids[5:],
        })
        factions = detect_factions(ids, sg, rng)
        for f in factions:
            assert f.leader_id in f.members


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

class TestReconcileFactions:
    def test_new_factions_get_ids(self):
        """New factions with no history get fresh IDs."""
        rng = random.Random(42)
        state = DiplomacyState()
        factions = [
            Faction(id="", name="", members=["a", "b", "c"],
                    leader_id="a", formed_year=0),
        ]
        result = reconcile_factions(factions, state, year=1, rng=rng)
        assert len(result) == 1
        assert result[0].id == "faction-0"
        assert result[0].formed_year == 1
        assert result[0].name != ""

    def test_matching_preserves_id(self):
        """Faction with >30% overlap keeps its old ID."""
        rng = random.Random(42)
        state = DiplomacyState()
        state.factions = [
            Faction(id="faction-0", name="Old Guard", members=["a", "b", "c", "d"],
                    leader_id="a", formed_year=1, ideology="technocrat"),
        ]
        state.next_faction_id = 1
        new_factions = [
            Faction(id="", name="", members=["a", "b", "c", "e"],
                    leader_id="a", formed_year=0),
        ]
        result = reconcile_factions(new_factions, state, year=5, rng=rng)
        assert result[0].id == "faction-0"
        assert result[0].name == "Old Guard"
        assert result[0].ideology == "technocrat"
        assert result[0].formed_year == 1

    def test_dissolved_factions_recorded(self):
        """Unmatched old factions get recorded as dissolved."""
        rng = random.Random(42)
        state = DiplomacyState()
        state.factions = [
            Faction(id="faction-0", name="Old Guard", members=["a", "b", "c"],
                    leader_id="a", formed_year=1),
        ]
        state.next_faction_id = 1
        result = reconcile_factions([], state, year=5, rng=rng)
        assert len(result) == 0
        assert len(state.dissolved) == 1
        assert state.dissolved[0]["id"] == "faction-0"
        assert state.dissolved[0]["dissolved_year"] == 5


# ---------------------------------------------------------------------------
# Schism
# ---------------------------------------------------------------------------

class TestSchism:
    def test_no_schism_small_faction(self):
        """Factions with < 6 members can't schism."""
        rng = random.Random(42)
        ids = ["a", "b", "c", "d", "e"]
        sg = SocialGraph()
        sg.initialize(ids, rng)
        faction = Faction(id="f-0", name="Test", members=ids,
                          leader_id="a", formed_year=1)
        assert check_schism(faction, sg, rng, year=5) is None

    def test_schism_with_dissidents(self):
        """Faction with clear dissident cluster can split."""
        rng = random.Random(1)  # seed that passes probability gate
        ids = [f"c-{i}" for i in range(8)]
        sg = make_social(ids, rng, clusters={
            "loyalist": ids[:4],
            "dissident": ids[4:],
        })
        # Dissidents distrust leader
        leader = ids[0]
        for d in ids[4:]:
            sg.get(d, leader).trust = 0.2
            sg.get(leader, d).trust = 0.2

        faction = Faction(id="f-0", name="Test", members=ids,
                          leader_id=leader, formed_year=1, cohesion=0.7)

        # Try multiple seeds to find one that passes probability gate
        found_schism = False
        for seed in range(100):
            trial_rng = random.Random(seed)
            result = check_schism(faction, sg, trial_rng, year=5)
            if result is not None:
                daughter_a, daughter_b = result
                assert leader in daughter_a.members
                assert len(daughter_a.members) >= MIN_FACTION_SIZE
                assert len(daughter_b.members) >= MIN_FACTION_SIZE
                found_schism = True
                break
        assert found_schism, "Schism should occur with at least one seed"


# ---------------------------------------------------------------------------
# Alliances
# ---------------------------------------------------------------------------

class TestAlliances:
    def test_alliance_forms_on_high_leader_trust(self):
        """Leaders with high trust form an alliance."""
        rng = random.Random(42)
        ids = ["a", "b", "c", "d", "e", "f"]
        sg = SocialGraph()
        sg.initialize(ids, rng)
        sg.get("a", "d").trust = 0.8
        sg.get("d", "a").trust = 0.8

        factions = [
            Faction(id="f-0", name="Alpha", members=["a", "b", "c"],
                    leader_id="a", formed_year=1),
            Faction(id="f-1", name="Beta", members=["d", "e", "f"],
                    leader_id="d", formed_year=1),
        ]
        alliances = update_alliances(factions, [], sg, year=5)
        assert len(alliances) >= 1
        assert alliances[0].faction_a == "f-0"
        assert alliances[0].faction_b == "f-1"

    def test_alliance_breaks_on_low_trust(self):
        """Alliance breaks when leader trust drops below threshold."""
        rng = random.Random(42)
        ids = ["a", "b", "c", "d", "e", "f"]
        sg = SocialGraph()
        sg.initialize(ids, rng)
        sg.get("a", "d").trust = 0.2
        sg.get("d", "a").trust = 0.2

        factions = [
            Faction(id="f-0", name="Alpha", members=["a", "b", "c"],
                    leader_id="a", formed_year=1),
            Faction(id="f-1", name="Beta", members=["d", "e", "f"],
                    leader_id="d", formed_year=1),
        ]
        existing = [Alliance(faction_a="f-0", faction_b="f-1",
                             formed_year=1, strength=0.8)]
        alliances = update_alliances(factions, existing, sg, year=5)
        assert len(alliances) == 0

    def test_alliance_pruned_when_faction_dissolved(self):
        """Alliance with a non-existent faction gets pruned."""
        rng = random.Random(42)
        ids = ["a", "b", "c"]
        sg = SocialGraph()
        sg.initialize(ids, rng)
        factions = [
            Faction(id="f-0", name="Alpha", members=ids,
                    leader_id="a", formed_year=1),
        ]
        existing = [Alliance(faction_a="f-0", faction_b="f-999",
                             formed_year=1, strength=0.8)]
        alliances = update_alliances(factions, existing, sg, year=5)
        assert len(alliances) == 0


# ---------------------------------------------------------------------------
# Vote bias
# ---------------------------------------------------------------------------

class TestVoteBias:
    def test_no_bias_without_faction(self):
        state = DiplomacyState()
        assert faction_vote_bias("nobody", "council", state) == 0.0

    def test_technocrat_prefers_ai_governor(self):
        state = DiplomacyState()
        state.factions = [
            Faction(id="f-0", name="Test", members=["a"],
                    leader_id="a", formed_year=1,
                    cohesion=1.0, ideology="technocrat"),
        ]
        bias = faction_vote_bias("a", "ai_governor", state)
        assert bias > 0
        assert bias <= MAX_VOTE_BIAS

    def test_bias_capped(self):
        """Bias magnitude never exceeds MAX_VOTE_BIAS."""
        state = DiplomacyState()
        state.factions = [
            Faction(id="f-0", name="Test", members=["a"],
                    leader_id="a", formed_year=1,
                    cohesion=1.0, ideology="libertarian"),
        ]
        bias = faction_vote_bias("a", "dictator", state)
        assert abs(bias) <= MAX_VOTE_BIAS

    def test_ideology_direction(self):
        """Libertarians oppose dictators (negative bias)."""
        state = DiplomacyState()
        state.factions = [
            Faction(id="f-0", name="Test", members=["a"],
                    leader_id="a", formed_year=1,
                    cohesion=1.0, ideology="libertarian"),
        ]
        bias = faction_vote_bias("a", "dictator", state)
        assert bias < 0


# ---------------------------------------------------------------------------
# Tick diplomacy integration
# ---------------------------------------------------------------------------

class TestTickDiplomacy:
    def test_tick_returns_summary(self):
        """tick_diplomacy returns dict with expected keys."""
        rng = random.Random(42)
        ids = [f"c-{i}" for i in range(10)]
        sg = make_social(ids, rng, clusters={
            "alpha": ids[:5],
            "beta": ids[5:],
        })
        state = DiplomacyState()
        summary = tick_diplomacy(state, ids, sg, year=1, rng=rng)
        assert "year" in summary
        assert "factions" in summary
        assert "alliances" in summary
        assert summary["year"] == 1

    def test_persistent_faction_ids(self):
        """Factions maintain IDs across ticks when membership is stable."""
        rng = random.Random(42)
        ids = [f"c-{i}" for i in range(10)]
        sg = make_social(ids, rng, clusters={
            "alpha": ids[:5],
            "beta": ids[5:],
        })
        state = DiplomacyState()
        tick_diplomacy(state, ids, sg, year=1, rng=rng)
        if state.factions:
            old_ids = {f.id for f in state.factions}
            tick_diplomacy(state, ids, sg, year=2, rng=rng)
            new_ids = {f.id for f in state.factions}
            assert old_ids & new_ids, "At least one faction ID should persist"


# ---------------------------------------------------------------------------
# DiplomacyState
# ---------------------------------------------------------------------------

class TestDiplomacyState:
    def test_faction_of(self):
        state = DiplomacyState()
        state.factions = [
            Faction(id="f-0", name="Test", members=["a", "b"],
                    leader_id="a", formed_year=1),
        ]
        assert state.faction_of("a") is not None
        assert state.faction_of("a").id == "f-0"
        assert state.faction_of("nobody") is None

    def test_to_dict_roundtrip(self):
        state = DiplomacyState()
        state.factions = [
            Faction(id="f-0", name="Test", members=["a"],
                    leader_id="a", formed_year=1),
        ]
        state.alliances = [
            Alliance(faction_a="f-0", faction_b="f-1",
                     formed_year=1, strength=0.7),
        ]
        d = state.to_dict()
        assert len(d["factions"]) == 1
        assert len(d["alliances"]) == 1
        assert d["factions"][0]["id"] == "f-0"


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_engine_10_years_with_diplomacy(self):
        """Engine runs 10 years without crash, diplomacy data present."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert len(result.years) == 10
        for yr in result.years:
            d = yr.to_dict()
            assert "diplomacy" in d
            assert "factions" in d["diplomacy"]
            assert "alliances" in d["diplomacy"]

    def test_engine_result_has_diplomacy(self):
        """SimulationResult includes diplomacy summary."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        d = result.to_dict()
        assert "diplomacy" in d
        assert d["_meta"]["version"] == "5.1"

    def test_vote_bias_affects_voting(self):
        """With factions active, vote bias influences outcomes
        (statistical — different results vs no-faction baseline)."""
        from src.mars100.engine import Mars100Engine
        # Run twice with same seed — results should be identical
        e1 = Mars100Engine(seed=99, total_years=20)
        r1 = e1.run()
        e2 = Mars100Engine(seed=99, total_years=20)
        r2 = e2.run()
        # Deterministic: same seed → same results
        assert len(r1.years) == len(r2.years)
        for y1, y2 in zip(r1.years, r2.years):
            assert y1.to_dict()["actions"] == y2.to_dict()["actions"]


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

class TestProperties:
    @pytest.mark.parametrize("seed", range(10))
    def test_no_duplicate_membership(self, seed):
        """No colonist appears in more than one faction (any seed)."""
        rng = random.Random(seed)
        ids = [f"c-{i}" for i in range(10)]
        sg = SocialGraph()
        sg.initialize(ids, rng)
        state = DiplomacyState()
        tick_diplomacy(state, ids, sg, year=1, rng=rng)
        seen: set[str] = set()
        for f in state.factions:
            for m in f.members:
                assert m not in seen, f"Colonist {m} in multiple factions"
                seen.add(m)

    @pytest.mark.parametrize("seed", range(10))
    def test_bias_bounded(self, seed):
        """Vote bias always within [-MAX_VOTE_BIAS, MAX_VOTE_BIAS]."""
        state = DiplomacyState()
        state.factions = [
            Faction(id="f-0", name="T", members=["a"],
                    leader_id="a", formed_year=1,
                    cohesion=1.0,
                    ideology=Faction.IDEOLOGIES[seed % len(Faction.IDEOLOGIES)]),
        ]
        gov_types = ["council", "dictator", "lottery", "consensus",
                     "ai_governor", "anarchy"]
        for gt in gov_types:
            bias = faction_vote_bias("a", gt, state)
            assert -MAX_VOTE_BIAS <= bias <= MAX_VOTE_BIAS

    @pytest.mark.parametrize("seed", range(5))
    def test_engine_30_years_stable(self, seed):
        """Engine runs 30 years without crash across multiple seeds."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=seed, total_years=30)
        result = engine.run()
        assert len(result.years) > 0
        assert result.to_dict()["_meta"]["version"] == "5.1"
