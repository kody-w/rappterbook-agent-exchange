"""Tests for the Mars-100 diplomacy organ (engine v11.0)."""
from __future__ import annotations

import random
import pytest
from dataclasses import dataclass, field
from typing import Any

from src.mars100.diplomacy import (
    DiplomacyState,
    DiplomacyTickResult,
    Faction,
    Treaty,
    FACTION_RECOMPUTE_INTERVAL,
    MIN_FACTION_SIZE,
    TREATY_TYPES,
    TREATY_VIOLATION_LIMIT,
    compute_bloc_vote_bias,
    compute_diplomatic_pressure,
    compute_resource_modifiers,
    check_crises,
    check_violations,
    detect_factions,
    maintain_factions,
    propose_treaties,
    tick_diplomacy,
)


# ---------------------------------------------------------------------------
# Minimal stubs — avoids importing full colonist/colony modules
# ---------------------------------------------------------------------------


@dataclass
class _Stats:
    resolve: float = 0.5
    improvisation: float = 0.3
    empathy: float = 0.4
    hoarding: float = 0.2
    faith: float = 0.6
    paranoia: float = 0.1


@dataclass
class _Skills:
    terraforming: float = 0.0
    hydroponics: float = 0.0
    mediation: float = 0.5
    coding: float = 0.0
    prayer: float = 0.0
    sabotage: float = 0.0


@dataclass
class _Rel:
    trust: float = 0.5
    affection: float = 0.5
    respect: float = 0.5


@dataclass
class _Colonist:
    id: str
    name: str = ""
    stats: _Stats = field(default_factory=_Stats)
    skills: _Skills = field(default_factory=_Skills)
    _active: bool = True

    def is_active(self) -> bool:
        return self._active


class _Social:
    """Minimal social-graph stub."""

    def __init__(self) -> None:
        self._edges: dict[str, dict[str, _Rel]] = {}

    def set_trust(self, a: str, b: str, trust: float) -> None:
        self._edges.setdefault(a, {})[b] = _Rel(trust=trust)

    def get(self, a: str, b: str) -> _Rel:
        return self._edges.get(a, {}).get(b, _Rel())

    def update_from_conflict(self, a: str, b: str, rng: random.Random) -> None:
        rel = self._edges.setdefault(a, {}).setdefault(b, _Rel())
        rel.trust = max(0.0, rel.trust - 0.05)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_colonists(n: int, stat_overrides: dict[str, dict] | None = None) -> list[_Colonist]:
    """Create n colonists with sequential IDs."""
    cols = []
    for i in range(n):
        cid = str(i)
        stats_kw = (stat_overrides or {}).get(cid, {})
        cols.append(_Colonist(id=cid, name=f"Col-{i}",
                              stats=_Stats(**stats_kw)))
    return cols


def _make_social(colonist_ids: list[str], base_trust: float = 0.5) -> _Social:
    social = _Social()
    for a in colonist_ids:
        for b in colonist_ids:
            if a != b:
                social.set_trust(a, b, base_trust)
    return social


# ---------------------------------------------------------------------------
# Faction detection
# ---------------------------------------------------------------------------


class TestDetectFactions:
    """Test organic faction formation from social graph."""

    def test_no_factions_below_minimum_population(self):
        cols = _make_colonists(4)
        social = _make_social([c.id for c in cols])
        state = DiplomacyState()
        formed = detect_factions(cols, social, state, year=10, rng=random.Random(42))
        assert formed == []

    def test_factions_form_from_stat_clusters(self):
        """Colonists with same dominant stat should cluster together."""
        overrides = {
            "0": {"faith": 0.9, "resolve": 0.1},
            "1": {"faith": 0.8, "resolve": 0.2},
            "2": {"faith": 0.85, "resolve": 0.15},
            "3": {"resolve": 0.9, "faith": 0.1},
            "4": {"resolve": 0.85, "faith": 0.15},
            "5": {"resolve": 0.8, "faith": 0.2},
            "6": {"empathy": 0.9, "faith": 0.1},
            "7": {"empathy": 0.85, "faith": 0.15},
            "8": {"empathy": 0.8, "faith": 0.2},
        }
        cols = _make_colonists(9, overrides)
        social = _make_social([c.id for c in cols])
        state = DiplomacyState()
        formed = detect_factions(cols, social, state, year=10, rng=random.Random(42))
        assert len(state.factions) >= 2
        # Every active colonist should be in exactly one faction
        all_members = set()
        for f in state.factions.values():
            for m in f.member_ids:
                assert m not in all_members, f"{m} in multiple factions"
                all_members.add(m)
        assert all_members == {str(i) for i in range(9)}

    def test_faction_id_stability(self):
        """Re-detection with same colonists preserves faction IDs."""
        overrides = {
            str(i): {"faith": 0.9} if i < 4 else {"resolve": 0.9}
            for i in range(8)
        }
        cols = _make_colonists(8, overrides)
        social = _make_social([c.id for c in cols])
        state = DiplomacyState()
        detect_factions(cols, social, state, year=10, rng=random.Random(42))
        old_ids = set(state.factions.keys())
        # Re-detect — IDs should be stable
        detect_factions(cols, social, state, year=20, rng=random.Random(99))
        new_ids = set(state.factions.keys())
        assert old_ids == new_ids

    def test_membership_is_partition(self):
        """No colonist appears in two factions."""
        overrides = {str(i): {"faith": 0.9} if i < 5 else {"resolve": 0.9}
                     for i in range(10)}
        cols = _make_colonists(10, overrides)
        social = _make_social([c.id for c in cols])
        state = DiplomacyState()
        detect_factions(cols, social, state, year=10, rng=random.Random(42))
        seen: set[str] = set()
        for f in state.factions.values():
            for m in f.member_ids:
                assert m not in seen
                seen.add(m)


class TestMaintainFactions:
    """Test annual faction maintenance."""

    def test_prune_dead_members(self):
        cols = _make_colonists(8, {str(i): {"faith": 0.9} if i < 4
                                   else {"resolve": 0.9} for i in range(8)})
        social = _make_social([c.id for c in cols])
        state = DiplomacyState()
        detect_factions(cols, social, state, year=10, rng=random.Random(42))
        # Kill colonist 0
        cols[0]._active = False
        maintain_factions(cols, social, state, year=11, rng=random.Random(42))
        all_members = set()
        for f in state.factions.values():
            all_members.update(f.member_ids)
        assert "0" not in all_members

    def test_assign_newcomer(self):
        cols = _make_colonists(8, {str(i): {"faith": 0.9} if i < 4
                                   else {"resolve": 0.9} for i in range(8)})
        social = _make_social([c.id for c in cols])
        state = DiplomacyState()
        detect_factions(cols, social, state, year=10, rng=random.Random(42))
        # Add new colonist
        new_col = _Colonist(id="99", name="Newcomer",
                            stats=_Stats(faith=0.9))
        cols.append(new_col)
        for cid in [c.id for c in cols if c.id != "99"]:
            social.set_trust("99", cid, 0.5)
            social.set_trust(cid, "99", 0.5)
        maintain_factions(cols, social, state, year=11, rng=random.Random(42))
        assert "99" in state.faction_membership


# ---------------------------------------------------------------------------
# Treaties
# ---------------------------------------------------------------------------


class TestTreaties:
    """Test treaty proposal, evaluation, and violation."""

    def _setup_two_factions(self):
        state = DiplomacyState()
        fa = Faction(id="f-a", name="Alpha", formed_year=10,
                     member_ids=["0", "1", "2"], dominant_value="faith")
        fb = Faction(id="f-b", name="Beta", formed_year=10,
                     member_ids=["3", "4", "5"], dominant_value="resolve")
        state.factions = {"f-a": fa, "f-b": fb}
        state.faction_membership = {
            "0": "f-a", "1": "f-a", "2": "f-a",
            "3": "f-b", "4": "f-b", "5": "f-b",
        }
        social = _make_social(["0", "1", "2", "3", "4", "5"], base_trust=0.6)
        return state, social

    def test_treaty_proposal_with_high_trust(self):
        state, social = self._setup_two_factions()
        rng = random.Random(42)
        # Run multiple times to hit the 25% chance
        proposed = []
        for yr in range(10, 40):
            proposed.extend(propose_treaties(state, social, yr, rng))
        assert len(proposed) > 0
        assert proposed[0]["treaty_type"] in TREATY_TYPES

    def test_no_duplicate_treaties(self):
        state, social = self._setup_two_factions()
        rng = random.Random(42)
        for yr in range(10, 50):
            propose_treaties(state, social, yr, rng)
        # At most 1 treaty between the same pair
        pairs = [(t.party_a, t.party_b) for t in state.treaties.values()]
        assert len(pairs) == len(set(pairs))

    def test_non_aggression_violation(self):
        state, social = self._setup_two_factions()
        # Create an active non-aggression treaty
        treaty = Treaty(id="t1", treaty_type="non_aggression",
                        party_a="f-a", party_b="f-b", formed_year=10)
        state.treaties["t1"] = treaty
        # Colonist 0 (faction a) sabotages
        actions = {"0": "sabotage", "1": "farm", "2": "code",
                   "3": "farm", "4": "code", "5": "rest"}
        violations, dissolved = check_violations(state, actions, social, year=15)
        assert len(violations) == 1
        assert violations[0]["treaty_id"] == "t1"

    def test_treaty_dissolves_after_limit(self):
        state, social = self._setup_two_factions()
        treaty = Treaty(id="t1", treaty_type="non_aggression",
                        party_a="f-a", party_b="f-b", formed_year=10,
                        violations=TREATY_VIOLATION_LIMIT - 1)
        state.treaties["t1"] = treaty
        actions = {"0": "sabotage", "1": "farm", "2": "code",
                   "3": "farm", "4": "code", "5": "rest"}
        violations, dissolved = check_violations(state, actions, social, year=15)
        assert len(dissolved) == 1
        assert dissolved[0]["status"] == "dissolved_violations"

    def test_treaty_expires(self):
        state, social = self._setup_two_factions()
        treaty = Treaty(id="t1", treaty_type="resource_sharing",
                        party_a="f-a", party_b="f-b", formed_year=10,
                        expires_year=15)
        state.treaties["t1"] = treaty
        actions = {"0": "farm", "1": "farm", "2": "code",
                   "3": "farm", "4": "code", "5": "rest"}
        violations, dissolved = check_violations(state, actions, social, year=15)
        assert any(d["status"] == "expired" for d in dissolved)


# ---------------------------------------------------------------------------
# Diplomatic pressure
# ---------------------------------------------------------------------------


class TestDiplomaticPressure:
    """Test action-weight modifiers from treaties."""

    def test_no_treaties_no_pressure(self):
        state = DiplomacyState()
        p = compute_diplomatic_pressure(state)
        assert p == {}

    def test_non_aggression_suppresses_sabotage(self):
        state = DiplomacyState()
        state.treaties["t1"] = Treaty(
            id="t1", treaty_type="non_aggression",
            party_a="f-a", party_b="f-b", formed_year=10)
        p = compute_diplomatic_pressure(state)
        assert p.get("sabotage", 0) < 0
        assert p.get("mediate", 0) > 0

    def test_resource_sharing_boosts_cooperation(self):
        state = DiplomacyState()
        state.treaties["t1"] = Treaty(
            id="t1", treaty_type="resource_sharing",
            party_a="f-a", party_b="f-b", formed_year=10)
        p = compute_diplomatic_pressure(state)
        assert p.get("cooperate", 0) > 0
        assert p.get("hoard", 0) < 0

    def test_resource_modifiers_reduce_spoilage(self):
        state = DiplomacyState()
        state.treaties["t1"] = Treaty(
            id="t1", treaty_type="resource_sharing",
            party_a="f-a", party_b="f-b", formed_year=10)
        mods = compute_resource_modifiers(state)
        assert mods.get("food_spoilage_mult", 1.0) < 1.0


# ---------------------------------------------------------------------------
# Bloc voting
# ---------------------------------------------------------------------------


class TestBlocVoting:
    """Test faction-based voting bias."""

    def test_same_faction_positive_bias(self):
        state = DiplomacyState()
        state.faction_membership = {"a": "f1", "b": "f1"}
        bias = compute_bloc_vote_bias("a", "b", state)
        assert bias > 0

    def test_allied_factions_mild_bias(self):
        state = DiplomacyState()
        state.faction_membership = {"a": "f1", "b": "f2"}
        state.treaties["t1"] = Treaty(
            id="t1", treaty_type="mutual_defense",
            party_a="f1", party_b="f2", formed_year=10)
        state.factions["f1"] = Faction(id="f1", name="Alpha", formed_year=10)
        state.factions["f2"] = Faction(id="f2", name="Beta", formed_year=10)
        bias = compute_bloc_vote_bias("a", "b", state)
        assert bias > 0

    def test_rival_factions_negative_bias(self):
        state = DiplomacyState()
        state.faction_membership = {"a": "f1", "b": "f2"}
        state.crises_log.append({
            "year": 15, "type": "resource_scarcity",
            "factions": ["f1", "f2"],
        })
        bias = compute_bloc_vote_bias("a", "b", state)
        assert bias < 0


# ---------------------------------------------------------------------------
# Crises
# ---------------------------------------------------------------------------


class TestCrises:
    def test_crisis_on_low_resources(self):
        state = DiplomacyState()
        state.factions = {
            "f1": Faction(id="f1", name="A", formed_year=5),
            "f2": Faction(id="f2", name="B", formed_year=5),
        }
        rng = random.Random(42)
        # Run many trials to hit the 30% chance
        crises = []
        for yr in range(10, 50):
            crises.extend(check_crises(state, resource_avg=0.1,
                                       event_severity=0.3, year=yr, rng=rng))
        assert len(crises) > 0


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_diplomacy_state_round_trip(self):
        state = DiplomacyState()
        state.factions["f1"] = Faction(
            id="f1", name="Alpha", formed_year=10,
            member_ids=["0", "1"], dominant_value="faith")
        state.treaties["t1"] = Treaty(
            id="t1", treaty_type="non_aggression",
            party_a="f1", party_b="f2", formed_year=10)
        state.faction_membership = {"0": "f1", "1": "f1"}
        d = state.to_dict()
        restored = DiplomacyState.from_dict(d)
        assert restored.to_dict() == d

    def test_tick_result_round_trip(self):
        result = DiplomacyTickResult(
            factions_formed=[{"id": "f1"}],
            treaties_proposed=[{"id": "t1"}],
            pressure={"cooperate": 0.05},
        )
        d = result.to_dict()
        assert d["factions_formed"] == [{"id": "f1"}]
        assert d["pressure"]["cooperate"] == 0.05


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_result(self):
        overrides = {str(i): {"faith": 0.9} if i < 5 else {"resolve": 0.9}
                     for i in range(10)}
        cols = _make_colonists(10, overrides)
        social = _make_social([c.id for c in cols])

        results = []
        for _ in range(3):
            state = DiplomacyState()
            rng = random.Random(42)
            actions = {str(i): "farm" for i in range(10)}
            r = tick_diplomacy(state, cols, social, actions,
                               resource_avg=0.5, event_severity=0.3,
                               year=10, rng=rng)
            results.append(r.to_dict())

        assert results[0] == results[1] == results[2]


# ---------------------------------------------------------------------------
# Full tick
# ---------------------------------------------------------------------------


class TestTickDiplomacy:
    def test_tick_runs_without_error(self):
        overrides = {str(i): {"faith": 0.9} if i < 5 else {"resolve": 0.9}
                     for i in range(10)}
        cols = _make_colonists(10, overrides)
        social = _make_social([c.id for c in cols])
        state = DiplomacyState()
        actions = {str(i): "farm" for i in range(10)}
        result = tick_diplomacy(
            state, cols, social, actions,
            resource_avg=0.6, event_severity=0.2,
            year=10, rng=random.Random(42))
        assert isinstance(result, DiplomacyTickResult)

    def test_factions_form_after_year_5(self):
        overrides = {str(i): {"faith": 0.9} if i < 5 else {"resolve": 0.9}
                     for i in range(10)}
        cols = _make_colonists(10, overrides)
        social = _make_social([c.id for c in cols])
        state = DiplomacyState()
        actions = {str(i): "farm" for i in range(10)}
        # Year 3 — no factions yet
        tick_diplomacy(state, cols, social, actions, 0.6, 0.2, year=3,
                       rng=random.Random(42))
        assert len(state.factions) == 0
        # Year 10 — factions form
        tick_diplomacy(state, cols, social, actions, 0.6, 0.2, year=10,
                       rng=random.Random(42))
        assert len(state.factions) >= 2


# ---------------------------------------------------------------------------
# Engine integration smoke test
# ---------------------------------------------------------------------------


class TestEngineSmoke:
    """Verify the full engine runs for 100 years with diplomacy."""

    def test_full_sim_100_years(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        assert len(result.years) == 100
        summary = result.to_dict()
        assert summary["_meta"]["version"] == "11.0"
        assert "final_diplomacy" in summary
