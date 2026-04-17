"""Tests for Mars-100 cultural traditions."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, create_founding_ten
from src.mars100.colony import SocialGraph
from src.mars100.factions import Faction
from src.mars100.culture import (
    Tradition, maybe_create_tradition, apply_traditions,
    decay_traditions, reinforce_traditions,
    TRADITION_EFFECT_MAGNITUDE, DECAY_RATE, MIN_COHESION_FOR_TRADITION,
    MAX_TRADITIONS, MIN_AGE_TO_PARTICIPATE,
)


def _make_colonist(cid: str, birth_year: int = 0, alive: bool = True,
                   resolve: float = 0.5, empathy: float = 0.5) -> Colonist:
    """Helper to create a test colonist."""
    c = Colonist(
        id=cid, name=f"Test-{cid}", element="fire",
        archetype="pioneer", decision_expr="(+ resolve empathy)",
        stats=ColonistStats(resolve=resolve, empathy=empathy),
        skills=ColonistSkills(), birth_year=birth_year,
    )
    c.alive = alive
    return c


def _make_faction(fid: str, members: list[str], cohesion: float = 0.8,
                  dominant_stat: str = "resolve") -> Faction:
    return Faction(
        id=fid, name=f"Faction-{fid}", members=members,
        dominant_stat=dominant_stat, centroid={"resolve": 0.7},
        cohesion=cohesion, founded_year=1,
    )


def _make_tradition(tid: str = "t1", faction_origin: str = "f1",
                    target_stat: str = "resolve", strength: float = 1.0,
                    participants: list[str] | None = None) -> Tradition:
    return Tradition(
        id=tid, name="Test Rite", description="A test tradition.",
        faction_origin=faction_origin, founding_year=1,
        target_stat=target_stat, strength=strength,
        participants=participants or [],
    )


class TestTraditionCreation:
    def test_high_cohesion_creates_tradition(self) -> None:
        # prob = (0.9 - 0.65) * 0.3 = 0.075 per attempt
        # Try 100 seeds to be robust
        created = False
        for seed in range(100):
            r = maybe_create_tradition(
                faction_id="f1", faction_name="Hawks",
                dominant_stat="resolve", cohesion=0.9,
                existing_traditions=[], year=5,
                rng=random.Random(seed),
            )
            if r is not None:
                created = True
                break
        assert created, "Should create a tradition with high cohesion over 100 attempts"

    def test_low_cohesion_unlikely(self) -> None:
        count = 0
        for seed in range(100):
            r = maybe_create_tradition(
                faction_id="f1", faction_name="Hawks",
                dominant_stat="resolve", cohesion=0.3,
                existing_traditions=[], year=5,
                rng=random.Random(seed),
            )
            if r is not None:
                count += 1
        assert count == 0, "Below MIN_COHESION should never create traditions"

    def test_max_traditions_cap(self) -> None:
        traditions = [
            _make_tradition(tid=f"t{i}", faction_origin=f"f{i}",
                            target_stat=s)
            for i, s in enumerate(["resolve", "empathy", "faith", "paranoia",
                                    "improvisation", "hoarding",
                                    "resolve", "empathy", "faith", "paranoia"])
        ]
        assert len(traditions) == MAX_TRADITIONS
        result = maybe_create_tradition(
            faction_id="f99", faction_name="X",
            dominant_stat="resolve", cohesion=0.99,
            existing_traditions=traditions, year=5,
            rng=random.Random(42),
        )
        assert result is None, "Should not exceed MAX_TRADITIONS"

    def test_tradition_structure(self) -> None:
        for seed in range(100):
            t = maybe_create_tradition(
                faction_id="f1", faction_name="Hawks",
                dominant_stat="resolve", cohesion=0.9,
                existing_traditions=[], year=5,
                rng=random.Random(seed),
            )
            if t is not None:
                assert t.id
                assert t.name
                assert t.faction_origin == "f1"
                assert t.target_stat == "resolve"
                assert t.strength == 1.0
                assert t.founding_year == 5
                return
        assert False, "No tradition created in 100 tries"


class TestApplyTraditions:
    def test_stat_drift(self) -> None:
        colonists = [_make_colonist("a", resolve=0.5)]
        tradition = _make_tradition(participants=["a"])
        faction_members = {"f1": ["a"]}
        original = colonists[0].stats.resolve
        apply_traditions([tradition], colonists, faction_members, year=10)
        assert colonists[0].stats.resolve != original
        assert abs(colonists[0].stats.resolve - original) <= TRADITION_EFFECT_MAGNITUDE + 1e-9

    def test_stat_clamped_0_1(self) -> None:
        colonists = [_make_colonist("a", resolve=0.999)]
        tradition = _make_tradition(target_stat="resolve", strength=1.0)
        faction_members = {"f1": ["a"]}
        apply_traditions([tradition], colonists, faction_members, year=10)
        assert colonists[0].stats.resolve <= 1.0

    def test_dead_colonists_excluded(self) -> None:
        colonists = [_make_colonist("a", alive=False, resolve=0.5)]
        tradition = _make_tradition()
        faction_members = {"f1": ["a"]}
        original = colonists[0].stats.resolve
        apply_traditions([tradition], colonists, faction_members, year=10)
        assert colonists[0].stats.resolve == original

    def test_young_colonists_excluded(self) -> None:
        colonists = [_make_colonist("a", birth_year=8, resolve=0.5)]
        tradition = _make_tradition()
        faction_members = {"f1": ["a"]}
        original = colonists[0].stats.resolve
        apply_traditions([tradition], colonists, faction_members, year=10)
        assert colonists[0].stats.resolve == original

    def test_non_participants_excluded(self) -> None:
        colonists = [_make_colonist("a"), _make_colonist("b")]
        tradition = _make_tradition()
        faction_members = {"f1": ["a"]}  # only a in faction
        orig_b = colonists[1].stats.resolve
        apply_traditions([tradition], colonists, faction_members, year=10)
        assert colonists[1].stats.resolve == orig_b


class TestDecayTraditions:
    def test_decay_reduces_strength(self) -> None:
        t = _make_tradition(strength=1.0, participants=["a"])
        died = decay_traditions([t], year=5)
        assert t.strength < 1.0
        assert len(died) == 0  # should still be alive

    def test_faster_decay_without_participants(self) -> None:
        t_with = _make_tradition(tid="tw", strength=1.0, participants=["a"])
        t_without = _make_tradition(tid="tno", strength=1.0, participants=[])
        decay_traditions([t_with], year=5)
        decay_traditions([t_without], year=5)
        assert t_without.strength < t_with.strength

    def test_tradition_removal_at_zero(self) -> None:
        t = _make_tradition(strength=0.3, participants=[])
        for yr in range(100):
            died = decay_traditions([t], year=yr)
            if not t.alive:
                assert t.id in died
                return
        assert False, "Should eventually decay below alive threshold"


class TestReinforceTraitions:
    def test_reinforcement_boosts_strength(self) -> None:
        t = _make_tradition(strength=0.5, faction_origin="f1")
        faction_members = {"f1": ["a", "b"]}  # 2+ members triggers reinforce
        reinforce_traditions([t], faction_members)
        assert t.strength > 0.5

    def test_no_reinforcement_without_faction(self) -> None:
        t = _make_tradition(strength=0.5, faction_origin="f1")
        faction_members = {"f2": ["a", "b"]}  # different faction
        reinforce_traditions([t], faction_members)
        assert t.strength == 0.5

    def test_capped_at_1(self) -> None:
        t = _make_tradition(strength=0.99, faction_origin="f1")
        faction_members = {"f1": ["a", "b"]}
        reinforce_traditions([t], faction_members)
        assert t.strength <= 1.0


class TestTraditionSerialization:
    def test_to_dict(self) -> None:
        t = Tradition(
            id="t1", name="Resolve Rite", description="Annual ceremony.",
            faction_origin="f1", founding_year=5, target_stat="resolve",
            strength=0.8, participants=["a", "b"],
        )
        d = t.to_dict()
        assert d["id"] == "t1"
        assert d["name"] == "Resolve Rite"
        assert d["faction_origin"] == "f1"
        assert d["target_stat"] == "resolve"
        assert d["strength"] == 0.8
        assert d["founding_year"] == 5
        assert d["participants"] == ["a", "b"]


class TestIntegration:
    def test_full_cycle(self) -> None:
        """End-to-end: detect factions -> create tradition -> apply -> decay."""
        colonists = create_founding_ten(42)
        sg = SocialGraph()
        sg.initialize([c.id for c in colonists], random.Random(42))

        from src.mars100.factions import detect_factions
        factions = detect_factions(colonists, sg, year=5, rng=random.Random(42))

        # Build faction_members map
        faction_members: dict[str, list[str]] = {}
        for f in factions:
            faction_members[f.id] = f.members

        # Try to create traditions from factions
        traditions: list[Tradition] = []
        for f in factions:
            t = maybe_create_tradition(
                faction_id=f.id, faction_name=f.name,
                dominant_stat=f.dominant_stat, cohesion=f.cohesion,
                existing_traditions=traditions, year=5,
                rng=random.Random(42),
            )
            if t is not None:
                traditions.append(t)

        # Apply traditions
        apply_traditions(traditions, colonists, faction_members, year=5)

        # Reinforce
        reinforce_traditions(traditions, faction_members)

        # Decay
        decay_traditions(traditions, year=5)

        # Everything should be valid
        for t in traditions:
            assert 0.0 <= t.strength <= 1.0
        for c in colonists:
            for stat_name in ("resolve", "improvisation", "empathy",
                              "hoarding", "faith", "paranoia"):
                val = getattr(c.stats, stat_name)
                assert 0.0 <= val <= 1.0, f"{c.id}.{stat_name} = {val}"
