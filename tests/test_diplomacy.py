"""Tests for the diplomacy organ (engine v11.0)."""
from __future__ import annotations

import random
import pytest
from src.mars100.diplomacy import (
    Faction, Treaty, DiplomacyState, DiplomacyTickResult,
    detect_factions, update_faction_membership,
    negotiate_treaty, check_treaty_violations, expire_treaties,
    compute_diplomatic_modifiers, compute_faction_pressure,
    tick_diplomacy,
    FORM_THRESHOLD, DISSOLVE_THRESHOLD, DISSOLVE_GRACE_YEARS,
    MIN_FACTION_SIZE, MAX_FACTIONS, MAX_NEGOTIATIONS_PER_YEAR,
    MAX_TREATIES_PER_PAIR, TREATY_DURATION_RANGE,
    IDEOLOGY_TYPES, TREATY_TYPES,
    _compute_affinity, _determine_ideology, _generate_treaty_terms,
    _build_negotiation_expr,
)
from src.mars100.subsim import SubSimBudget, SubSimResult
from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, create_founding_ten
from src.mars100.colony import SocialGraph, Relationship


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return random.Random(42)


@pytest.fixture
def colonists():
    return create_founding_ten(42)


@pytest.fixture
def social(colonists):
    sg = SocialGraph()
    rng = random.Random(42)
    sg.initialize([c.id for c in colonists], rng)
    return sg


@pytest.fixture
def high_trust_social(colonists):
    """Social graph where first 4 colonists have very high mutual trust."""
    sg = SocialGraph()
    rng = random.Random(42)
    sg.initialize([c.id for c in colonists], rng)
    first_four = [c.id for c in colonists[:4]]
    for a in first_four:
        for b in first_four:
            if a != b:
                sg.edges[a][b] = Relationship(trust=0.9, affection=0.8, respect=0.8)
    return sg


@pytest.fixture
def diplo_state():
    return DiplomacyState()


@pytest.fixture
def budget():
    return SubSimBudget(year=20)


# ── Faction ──────────────────────────────────────────────────────────

class TestFaction:
    def test_create(self):
        f = Faction(id="f-0", name="Iron Circle", ideology="cooperative",
                    member_ids=["c0", "c1", "c2"], influence=0.3,
                    formed_year=10)
        assert f.is_active()
        assert f.ideology == "cooperative"

    def test_dissolved(self):
        f = Faction(id="f-0", name="Test", ideology="militant",
                    member_ids=["c0"], influence=0.1,
                    formed_year=10, dissolved_year=15)
        assert not f.is_active()

    def test_to_dict(self):
        f = Faction(id="f-0", name="Test", ideology="spiritual",
                    member_ids=["c0", "c1"], influence=0.5,
                    formed_year=10)
        d = f.to_dict()
        assert d["id"] == "f-0"
        assert d["ideology"] == "spiritual"
        assert "dissolved_year" not in d

    def test_to_dict_dissolved(self):
        f = Faction(id="f-0", name="Test", ideology="spiritual",
                    member_ids=[], influence=0.0,
                    formed_year=10, dissolved_year=20)
        d = f.to_dict()
        assert d["dissolved_year"] == 20


class TestTreaty:
    def test_active(self):
        t = Treaty(id="t-0", faction_a="f-0", faction_b="f-1",
                   treaty_type="alliance", terms={"cohesion_bonus": 0.03},
                   signed_year=10, expires_year=20)
        assert t.is_active(15)
        assert not t.is_active(20)
        assert not t.is_active(25)

    def test_broken(self):
        t = Treaty(id="t-0", faction_a="f-0", faction_b="f-1",
                   treaty_type="alliance", terms={},
                   signed_year=10, expires_year=20, broken=True)
        assert not t.is_active(15)

    def test_to_dict(self):
        t = Treaty(id="t-0", faction_a="f-0", faction_b="f-1",
                   treaty_type="trade_pact", terms={"trade_bonus": 0.02},
                   signed_year=10, expires_year=20, subsim_score=0.75)
        d = t.to_dict()
        assert d["treaty_type"] == "trade_pact"
        assert d["subsim_score"] == 0.75


# ── DiplomacyState ───────────────────────────────────────────────────

class TestDiplomacyState:
    def test_empty(self):
        s = DiplomacyState()
        assert s.active_factions() == []
        assert s.active_treaties(10) == []

    def test_active_factions(self):
        s = DiplomacyState(factions=[
            Faction(id="f-0", name="A", ideology="cooperative",
                    member_ids=["c0"], influence=0.1, formed_year=5),
            Faction(id="f-1", name="B", ideology="militant",
                    member_ids=["c1"], influence=0.1, formed_year=5,
                    dissolved_year=10),
        ])
        assert len(s.active_factions()) == 1
        assert s.active_factions()[0].id == "f-0"

    def test_treaties_between(self):
        t = Treaty(id="t-0", faction_a="f-0", faction_b="f-1",
                   treaty_type="alliance", terms={},
                   signed_year=10, expires_year=20)
        s = DiplomacyState(treaties=[t])
        assert len(s.treaties_between("f-0", "f-1", 15)) == 1
        assert len(s.treaties_between("f-1", "f-0", 15)) == 1
        assert len(s.treaties_between("f-0", "f-2", 15)) == 0

    def test_to_dict_roundtrip(self):
        s = DiplomacyState()
        s.factions.append(Faction(id="f-0", name="X", ideology="spiritual",
                                  member_ids=["c0", "c1"], influence=0.2,
                                  formed_year=10))
        s.treaties.append(Treaty(id="t-0", faction_a="f-0", faction_b="f-1",
                                 treaty_type="alliance", terms={"cohesion_bonus": 0.03},
                                 signed_year=10, expires_year=20))
        d = s.to_dict()
        s2 = DiplomacyState.from_dict(d)
        assert len(s2.factions) == 1
        assert s2.factions[0].ideology == "spiritual"
        assert len(s2.treaties) == 1
        assert s2.treaties[0].treaty_type == "alliance"


# ── Affinity ─────────────────────────────────────────────────────────

class TestAffinity:
    def test_high_trust_high_similarity(self, colonists, social):
        a, b = colonists[0], colonists[1]
        social.edges[a.id][b.id] = Relationship(trust=0.9, affection=0.8, respect=0.8)
        social.edges[b.id][a.id] = Relationship(trust=0.9, affection=0.8, respect=0.8)
        by_id = {c.id: c for c in colonists}
        aff = _compute_affinity(social, a.id, b.id, by_id)
        assert 0.0 <= aff <= 1.0
        assert aff > 0.5

    def test_low_trust(self, colonists, social):
        a, b = colonists[0], colonists[1]
        social.edges[a.id][b.id] = Relationship(trust=0.1, affection=0.1, respect=0.1)
        social.edges[b.id][a.id] = Relationship(trust=0.1, affection=0.1, respect=0.1)
        by_id = {c.id: c for c in colonists}
        aff = _compute_affinity(social, a.id, b.id, by_id)
        assert aff < 0.5

    def test_missing_colonist(self, social):
        aff = _compute_affinity(social, "missing-a", "missing-b", {})
        assert isinstance(aff, float)


# ── Ideology Detection ───────────────────────────────────────────────

class TestIdeology:
    def test_cooperative(self):
        c1 = Colonist(id="c1", name="A", element="water", archetype="test",
                      stats=ColonistStats(empathy=0.9, faith=0.9),
                      skills=ColonistSkills(), decision_expr="(+ 1 1)")
        c2 = Colonist(id="c2", name="B", element="water", archetype="test",
                      stats=ColonistStats(empathy=0.8, faith=0.8),
                      skills=ColonistSkills(), decision_expr="(+ 1 1)")
        ideology = _determine_ideology(["c1", "c2"], {"c1": c1, "c2": c2})
        assert ideology in IDEOLOGY_TYPES

    def test_militant(self):
        c1 = Colonist(id="c1", name="A", element="fire", archetype="test",
                      stats=ColonistStats(resolve=0.9, paranoia=0.9),
                      skills=ColonistSkills(), decision_expr="(+ 1 1)")
        c2 = Colonist(id="c2", name="B", element="fire", archetype="test",
                      stats=ColonistStats(resolve=0.8, paranoia=0.8),
                      skills=ColonistSkills(), decision_expr="(+ 1 1)")
        ideology = _determine_ideology(["c1", "c2"], {"c1": c1, "c2": c2})
        assert ideology == "militant"


# ── Faction Detection ────────────────────────────────────────────────

class TestDetectFactions:
    def test_no_factions_early(self, colonists, social, diplo_state, rng):
        formed = detect_factions(social, colonists, diplo_state, 10, rng)
        # With default random trust ~0.5, factions may or may not form
        assert isinstance(formed, list)

    def test_high_trust_forms_faction(self, colonists, high_trust_social,
                                      diplo_state, rng):
        formed = detect_factions(high_trust_social, colonists, diplo_state,
                                 10, rng)
        assert len(formed) >= 1
        assert len(diplo_state.active_factions()) >= 1
        faction = diplo_state.active_factions()[0]
        assert len(faction.member_ids) >= MIN_FACTION_SIZE

    def test_max_factions_cap(self, colonists, social, diplo_state, rng):
        # Pre-fill factions to near max
        for i in range(MAX_FACTIONS):
            diplo_state.factions.append(Faction(
                id=f"f-{i}", name=f"F{i}", ideology="cooperative",
                member_ids=[f"x{i}a", f"x{i}b", f"x{i}c"],
                influence=0.1, formed_year=5))
        diplo_state.next_faction_id = MAX_FACTIONS
        formed = detect_factions(social, colonists, diplo_state, 10, rng)
        assert len(formed) == 0

    def test_too_few_colonists(self, diplo_state, rng):
        sg = SocialGraph()
        small = [Colonist(id="c0", name="A", element="fire", archetype="test",
                              stats=ColonistStats(), skills=ColonistSkills(),
                              decision_expr="(+ 1 1)")]
        formed = detect_factions(sg, small, diplo_state, 10, rng)
        assert formed == []


class TestUpdateFactionMembership:
    def test_prune_dead_members(self, colonists, social):
        state = DiplomacyState()
        ids = [c.id for c in colonists[:4]]
        state.factions.append(Faction(
            id="f-0", name="Test", ideology="cooperative",
            member_ids=ids, influence=0.4, formed_year=5))
        colonists[0].die(10, "test")
        dissolved = update_faction_membership(state, social, colonists, 10)
        f = state.factions[0]
        assert colonists[0].id not in f.member_ids
        assert f.is_active()

    def test_dissolve_insufficient_members(self, colonists, social):
        state = DiplomacyState()
        state.factions.append(Faction(
            id="f-0", name="Test", ideology="cooperative",
            member_ids=[colonists[0].id], influence=0.1, formed_year=5))
        colonists[0].die(10, "test")
        dissolved = update_faction_membership(state, social, colonists, 10)
        assert len(dissolved) == 1
        assert dissolved[0]["reason"] == "insufficient_members"

    def test_hysteresis_grace_period(self, colonists, social):
        """Low cohesion doesn't dissolve immediately."""
        state = DiplomacyState()
        ids = [c.id for c in colonists[:3]]
        state.factions.append(Faction(
            id="f-0", name="Test", ideology="cooperative",
            member_ids=ids, influence=0.3, formed_year=5))
        # Set very low trust
        for a in ids:
            for b in ids:
                if a != b:
                    social.edges[a][b] = Relationship(trust=0.1, affection=0.1, respect=0.1)
        dissolved = update_faction_membership(state, social, colonists, 10)
        f = state.factions[0]
        # First year below threshold — should NOT dissolve yet
        assert f.is_active()
        assert f.below_threshold_years == 1


# ── Treaty Negotiation ───────────────────────────────────────────────

class TestNegotiateTreaty:
    def test_basic_negotiation(self, diplo_state, budget, rng):
        fa = Faction(id="f-0", name="A", ideology="cooperative",
                     member_ids=["c0", "c1", "c2"], influence=0.3,
                     formed_year=5)
        fb = Faction(id="f-1", name="B", ideology="technophile",
                     member_ids=["c3", "c4", "c5"], influence=0.3,
                     formed_year=5)
        diplo_state.factions = [fa, fb]
        log: list[SubSimResult] = []
        treaty = negotiate_treaty(fa, fb, diplo_state, 20, budget, log, rng)
        # May or may not succeed depending on subsim result
        assert isinstance(treaty, Treaty) or treaty is None
        assert len(log) >= 1  # At least one subsim was run

    def test_max_treaties_per_pair(self, diplo_state, budget, rng):
        fa = Faction(id="f-0", name="A", ideology="cooperative",
                     member_ids=["c0", "c1", "c2"], influence=0.3,
                     formed_year=5)
        fb = Faction(id="f-1", name="B", ideology="militant",
                     member_ids=["c3", "c4", "c5"], influence=0.3,
                     formed_year=5)
        diplo_state.factions = [fa, fb]
        diplo_state.treaties.append(Treaty(
            id="t-0", faction_a="f-0", faction_b="f-1",
            treaty_type="alliance", terms={},
            signed_year=15, expires_year=30))
        log: list[SubSimResult] = []
        treaty = negotiate_treaty(fa, fb, diplo_state, 20, budget, log, rng)
        assert treaty is None


class TestTreatyTerms:
    @pytest.mark.parametrize("ttype", TREATY_TYPES)
    def test_generate_terms(self, rng, ttype):
        terms = _generate_treaty_terms(ttype, rng)
        assert isinstance(terms, dict)
        assert all(isinstance(v, float) for v in terms.values())

    @pytest.mark.parametrize("ttype", TREATY_TYPES)
    def test_negotiation_expression(self, ttype):
        fa = Faction(id="f-0", name="A", ideology="cooperative",
                     member_ids=["c0", "c1"], influence=0.3, formed_year=5)
        fb = Faction(id="f-1", name="B", ideology="militant",
                     member_ids=["c3", "c4"], influence=0.3, formed_year=5)
        expr = _build_negotiation_expr(ttype, fa, fb)
        assert isinstance(expr, str)
        assert expr.startswith("(")


# ── Treaty Violations ────────────────────────────────────────────────

class TestTreatyViolations:
    def test_sabotage_breaks_non_aggression(self, rng):
        state = DiplomacyState()
        state.factions = [
            Faction(id="f-0", name="A", ideology="cooperative",
                    member_ids=["c0", "c1"], influence=0.2, formed_year=5),
            Faction(id="f-1", name="B", ideology="militant",
                    member_ids=["c2", "c3"], influence=0.2, formed_year=5),
        ]
        state.treaties.append(Treaty(
            id="t-0", faction_a="f-0", faction_b="f-1",
            treaty_type="non_aggression", terms={"paranoia_reduction": 0.02},
            signed_year=10, expires_year=25))
        # Run many times to account for random chance
        any_broken = False
        for seed in range(100):
            r = random.Random(seed)
            actions = {"c0": "farm", "c1": "sabotage", "c2": "farm", "c3": "farm"}
            broken = check_treaty_violations(state, actions, 15, r)
            if broken:
                any_broken = True
                state.treaties[0].broken = False  # Reset for next iteration
                state.treaties[0].broken_year = None
        assert any_broken

    def test_no_violation_without_sabotage(self, rng):
        state = DiplomacyState()
        state.factions = [
            Faction(id="f-0", name="A", ideology="cooperative",
                    member_ids=["c0"], influence=0.1, formed_year=5),
            Faction(id="f-1", name="B", ideology="spiritual",
                    member_ids=["c1"], influence=0.1, formed_year=5),
        ]
        state.treaties.append(Treaty(
            id="t-0", faction_a="f-0", faction_b="f-1",
            treaty_type="non_aggression", terms={},
            signed_year=10, expires_year=25))
        actions = {"c0": "farm", "c1": "pray"}
        broken = check_treaty_violations(state, actions, 15, rng)
        assert broken == []


# ── Treaty Expiration ────────────────────────────────────────────────

class TestExpireTreaties:
    def test_expire(self):
        state = DiplomacyState(treaties=[
            Treaty(id="t-0", faction_a="f-0", faction_b="f-1",
                   treaty_type="alliance", terms={},
                   signed_year=10, expires_year=15),
        ])
        expired = expire_treaties(state, 15)
        assert len(expired) == 1

    def test_not_expired(self):
        state = DiplomacyState(treaties=[
            Treaty(id="t-0", faction_a="f-0", faction_b="f-1",
                   treaty_type="alliance", terms={},
                   signed_year=10, expires_year=25),
        ])
        expired = expire_treaties(state, 15)
        assert len(expired) == 0


# ── Modifiers ────────────────────────────────────────────────────────

class TestDiplomaticModifiers:
    def test_empty_state(self):
        state = DiplomacyState()
        mods = compute_diplomatic_modifiers(state, 10)
        assert mods["cohesion_bonus"] == 0.0
        assert mods["broken_treaty_paranoia"] == 0.0

    def test_alliance_bonus(self):
        state = DiplomacyState(treaties=[
            Treaty(id="t-0", faction_a="f-0", faction_b="f-1",
                   treaty_type="alliance", terms={"cohesion_bonus": 0.04},
                   signed_year=10, expires_year=25),
        ])
        mods = compute_diplomatic_modifiers(state, 15)
        assert mods["cohesion_bonus"] == pytest.approx(0.04)

    def test_broken_treaty_paranoia(self):
        state = DiplomacyState(treaties=[
            Treaty(id="t-0", faction_a="f-0", faction_b="f-1",
                   treaty_type="alliance", terms={},
                   signed_year=10, expires_year=25,
                   broken=True, broken_year=12),
        ])
        mods = compute_diplomatic_modifiers(state, 14)
        assert mods["broken_treaty_paranoia"] > 0

    def test_old_broken_treaty_no_paranoia(self):
        state = DiplomacyState(treaties=[
            Treaty(id="t-0", faction_a="f-0", faction_b="f-1",
                   treaty_type="alliance", terms={},
                   signed_year=5, expires_year=15,
                   broken=True, broken_year=5),
        ])
        mods = compute_diplomatic_modifiers(state, 20)
        assert mods["broken_treaty_paranoia"] == 0.0


# ── Faction Pressure ─────────────────────────────────────────────────

class TestFactionPressure:
    def test_no_faction(self):
        state = DiplomacyState()
        pressure = compute_faction_pressure("c0", state, 10)
        assert pressure == {}

    @pytest.mark.parametrize("ideology", IDEOLOGY_TYPES)
    def test_ideology_pressure(self, ideology):
        state = DiplomacyState(factions=[
            Faction(id="f-0", name="Test", ideology=ideology,
                    member_ids=["c0", "c1", "c2"], influence=0.3,
                    formed_year=5),
        ])
        pressure = compute_faction_pressure("c0", state, 10)
        assert isinstance(pressure, dict)
        assert any(v > 0 for v in pressure.values())


# ── tick_diplomacy ───────────────────────────────────────────────────

class TestTickDiplomacy:
    def test_no_diplomacy_before_year_8(self, colonists, social,
                                         diplo_state, budget, rng):
        log: list[SubSimResult] = []
        actions = {c.id: "farm" for c in colonists}
        result = tick_diplomacy(diplo_state, social, colonists, actions,
                                5, budget, log, rng)
        assert result.factions_formed == []
        assert result.treaties_signed == []

    def test_full_tick_with_high_trust(self, colonists, high_trust_social,
                                        diplo_state, budget, rng):
        actions = {c.id: "farm" for c in colonists}
        log: list[SubSimResult] = []
        result = tick_diplomacy(diplo_state, high_trust_social, colonists,
                                actions, 20, budget, log, rng)
        assert isinstance(result, DiplomacyTickResult)
        # With high trust among first 4, at least one faction should form
        assert len(result.factions_formed) >= 1

    def test_result_to_dict(self, colonists, social, diplo_state, budget, rng):
        log: list[SubSimResult] = []
        actions = {c.id: "rest" for c in colonists}
        result = tick_diplomacy(diplo_state, social, colonists, actions,
                                20, budget, log, rng)
        d = result.to_dict()
        assert "factions_formed" in d
        assert "treaties_signed" in d
        assert "treaties_broken" in d


# ── Property Tests ───────────────────────────────────────────────────

class TestProperties:
    def test_influence_bounds(self, colonists, high_trust_social,
                               diplo_state, rng):
        detect_factions(high_trust_social, colonists, diplo_state, 10, rng)
        for f in diplo_state.factions:
            assert 0.0 <= f.influence <= 1.0

    def test_faction_members_are_active(self, colonists, high_trust_social,
                                         diplo_state, rng):
        detect_factions(high_trust_social, colonists, diplo_state, 10, rng)
        active_ids = {c.id for c in colonists if c.is_active()}
        for f in diplo_state.active_factions():
            for m in f.member_ids:
                assert m in active_ids

    def test_treaty_duration_in_range(self, rng):
        for _ in range(100):
            d = rng.randint(*TREATY_DURATION_RANGE)
            assert TREATY_DURATION_RANGE[0] <= d <= TREATY_DURATION_RANGE[1]

    def test_all_treaty_types_have_terms(self, rng):
        for ttype in TREATY_TYPES:
            terms = _generate_treaty_terms(ttype, rng)
            assert len(terms) > 0

    def test_all_ideologies_have_pressure(self):
        for ideology in IDEOLOGY_TYPES:
            state = DiplomacyState(factions=[
                Faction(id="f-0", name="T", ideology=ideology,
                        member_ids=["c0"], influence=0.1, formed_year=5),
            ])
            p = compute_faction_pressure("c0", state, 10)
            assert len(p) > 0


# ── Integration: 50-year smoke test ──────────────────────────────────

class TestIntegration:
    def test_50_year_smoke(self):
        """Run diplomacy through 50 ticks to verify stability."""
        from src.mars100.colonist import create_founding_ten
        from src.mars100.colony import SocialGraph
        rng = random.Random(99)
        colonists = create_founding_ten(99)
        social = SocialGraph()
        social.initialize([c.id for c in colonists], rng)
        state = DiplomacyState()

        for year in range(1, 51):
            budget = SubSimBudget(year=year)
            log: list[SubSimResult] = []
            actions = {c.id: rng.choice(["farm", "code", "cooperate", "sabotage", "rest"])
                       for c in colonists if c.is_active()}
            # Evolve social graph slightly each year
            active_ids = [c.id for c in colonists if c.is_active()]
            social.update_from_event(active_ids, rng.gauss(0, 0.5), rng)

            result = tick_diplomacy(state, social, colonists, actions,
                                    year, budget, log, rng)

            # Invariants
            for f in state.active_factions():
                assert 0.0 <= f.influence <= 1.0
                assert len(f.member_ids) >= 1  # may drop below MIN after updates
            for t in state.treaties:
                assert t.signed_year <= t.expires_year

        # After 50 years, some factions should have formed
        assert len(state.factions) > 0 or len(colonists) < MIN_FACTION_SIZE
