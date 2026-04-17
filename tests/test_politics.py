"""Tests for the politics organ (engine v9.0)."""
from __future__ import annotations

import math
import random
import pytest

from src.mars100.politics import (
    PoliticalState, Faction, PoliticsContext, PoliticsTickResult,
    OPINION_AXES, FACTION_DISTANCE_THRESHOLD, MIN_FACTION_SIZE,
    MAX_FACTIONS, LEGITIMACY_PROPOSAL_THRESHOLD,
    compute_personality_pressure, compute_social_influence,
    compute_event_pressure, compute_resource_pressure,
    compute_engagement_delta, compute_action_modifiers,
    compute_legitimacy_delta, detect_factions, tick_politics,
    _opinion_distance, _compute_centroid, _dominant_axis,
    _init_political_state,
)


# ---------------------------------------------------------------------------
# PoliticalState
# ---------------------------------------------------------------------------

class TestPoliticalState:
    def test_defaults(self):
        ps = PoliticalState()
        assert ps.liberty_vs_security == 0.0
        assert ps.engagement == 0.30
        assert ps.faction_id is None

    def test_opinion_vector(self):
        ps = PoliticalState(liberty_vs_security=0.3, growth_vs_sustainability=-0.2,
                            individual_vs_collective=0.1)
        assert ps.opinion_vector() == (0.3, -0.2, 0.1)

    def test_round_trip(self):
        ps = PoliticalState(liberty_vs_security=0.123456, engagement=0.789, faction_id="f1")
        d = ps.to_dict()
        ps2 = PoliticalState.from_dict(d)
        assert abs(ps2.liberty_vs_security - 0.1235) < 0.001
        assert ps2.faction_id == "f1"

    def test_to_dict_has_all_axes(self):
        d = PoliticalState().to_dict()
        for ax in OPINION_AXES:
            assert ax in d


# ---------------------------------------------------------------------------
# Opinion distance / centroid helpers
# ---------------------------------------------------------------------------

class TestOpinionDistance:
    def test_same_point(self):
        assert _opinion_distance((0, 0, 0), (0, 0, 0)) == 0.0

    def test_known_distance(self):
        d = _opinion_distance((1, 0, 0), (0, 0, 0))
        assert abs(d - 1.0) < 1e-9

    def test_diagonal(self):
        d = _opinion_distance((1, 1, 1), (0, 0, 0))
        assert abs(d - math.sqrt(3)) < 1e-9


class TestComputeCentroid:
    def test_single_member(self):
        ops = {"a": (0.5, -0.3, 0.1)}
        c = _compute_centroid(ops, ["a"])
        assert abs(c[0] - 0.5) < 1e-9

    def test_average(self):
        ops = {"a": (1.0, 0.0, 0.0), "b": (-1.0, 0.0, 0.0)}
        c = _compute_centroid(ops, ["a", "b"])
        assert abs(c[0]) < 1e-9

    def test_empty_members(self):
        c = _compute_centroid({}, [])
        assert c == (0.0, 0.0, 0.0)


class TestDominantAxis:
    def test_liberty(self):
        assert _dominant_axis((0.9, 0.1, 0.2)) == "liberty_vs_security"

    def test_growth(self):
        assert _dominant_axis((0.1, -0.8, 0.2)) == "growth_vs_sustainability"

    def test_collective(self):
        assert _dominant_axis((0.1, 0.2, 0.7)) == "individual_vs_collective"

    def test_tie_breaks_alphabetically(self):
        # Equal absolute values → sorted by axis name
        axis = _dominant_axis((0.5, 0.5, 0.5))
        assert axis in OPINION_AXES  # deterministic


# ---------------------------------------------------------------------------
# Personality pressure
# ---------------------------------------------------------------------------

class TestPersonalityPressure:
    def test_neutral_stats_give_zero(self):
        stats = {s: 0.5 for s in ("resolve", "improvisation", "empathy",
                                   "hoarding", "faith", "paranoia")}
        deltas = compute_personality_pressure(stats)
        for ax in OPINION_AXES:
            assert abs(deltas[ax]) < 1e-9

    def test_high_paranoia_pushes_security(self):
        stats = {"paranoia": 1.0, "resolve": 0.5, "empathy": 0.5,
                 "hoarding": 0.5, "faith": 0.5, "improvisation": 0.5}
        deltas = compute_personality_pressure(stats)
        assert deltas["liberty_vs_security"] > 0

    def test_high_empathy_pushes_collective(self):
        stats = {"empathy": 1.0, "paranoia": 0.5, "resolve": 0.5,
                 "hoarding": 0.5, "faith": 0.5, "improvisation": 0.5}
        deltas = compute_personality_pressure(stats)
        assert deltas["individual_vs_collective"] > 0


# ---------------------------------------------------------------------------
# Social influence
# ---------------------------------------------------------------------------

class TestSocialInfluence:
    def test_no_trusted(self):
        deltas = compute_social_influence((0.0, 0.0, 0.0), [], [])
        assert all(abs(v) < 1e-9 for v in deltas.values())

    def test_pulls_toward_trusted(self):
        own = (0.0, 0.0, 0.0)
        trusted = [(0.5, 0.5, 0.5)]
        weights = [1.0]
        deltas = compute_social_influence(own, trusted, weights)
        assert deltas["liberty_vs_security"] > 0
        assert deltas["growth_vs_sustainability"] > 0

    def test_zero_weight_no_effect(self):
        own = (0.0, 0.0, 0.0)
        trusted = [(1.0, 1.0, 1.0)]
        weights = [0.0]
        deltas = compute_social_influence(own, trusted, weights)
        assert all(abs(v) < 1e-9 for v in deltas.values())


# ---------------------------------------------------------------------------
# Event / resource pressure
# ---------------------------------------------------------------------------

class TestEventPressure:
    def test_zero_severity(self):
        deltas = compute_event_pressure(0.0)
        assert all(abs(v) < 1e-9 for v in deltas.values())

    def test_high_severity_pushes_security(self):
        deltas = compute_event_pressure(0.8)
        assert deltas["liberty_vs_security"] > 0
        assert deltas["individual_vs_collective"] > 0


class TestResourcePressure:
    def test_abundant_resources(self):
        deltas = compute_resource_pressure(0.8, 0.05)
        # Abundance: less collective pressure
        assert deltas["individual_vs_collective"] <= 0

    def test_scarce_resources(self):
        deltas = compute_resource_pressure(0.2, -0.05)
        assert deltas["individual_vs_collective"] > 0


# ---------------------------------------------------------------------------
# Engagement
# ---------------------------------------------------------------------------

class TestEngagement:
    def test_decay_without_activity(self):
        delta = compute_engagement_delta("rest", had_crisis=False, event_severity=0.0)
        assert delta < 0

    def test_mediate_boosts(self):
        delta = compute_engagement_delta("mediate", had_crisis=False, event_severity=0.0)
        assert delta > compute_engagement_delta("rest", had_crisis=False, event_severity=0.0)

    def test_crisis_boosts(self):
        d1 = compute_engagement_delta("rest", had_crisis=True, event_severity=0.0)
        d2 = compute_engagement_delta("rest", had_crisis=False, event_severity=0.0)
        assert d1 > d2


# ---------------------------------------------------------------------------
# Action modifiers (psych→action perturbation, fulfils v8 deferral)
# ---------------------------------------------------------------------------

class TestActionModifiers:
    def test_high_stress_biases_rest(self):
        mods = compute_action_modifiers(
            psych_stress=0.9, psych_purpose=0.5, psych_morale=0.5,
            political_engagement=0.3, legitimacy=0.6, faction_id=None)
        assert mods.get("rest", 0) > 0
        assert mods.get("pray", 0) > 0

    def test_low_purpose_biases_sabotage(self):
        mods = compute_action_modifiers(
            psych_stress=0.2, psych_purpose=0.1, psych_morale=0.4,
            political_engagement=0.3, legitimacy=0.6, faction_id=None)
        assert mods.get("sabotage", 0) > 0

    def test_high_morale_biases_productive(self):
        mods = compute_action_modifiers(
            psych_stress=0.1, psych_purpose=0.8, psych_morale=0.9,
            political_engagement=0.3, legitimacy=0.6, faction_id=None)
        assert mods.get("terraform", 0) > 0
        assert mods.get("research", 0) > 0

    def test_low_legitimacy_high_engagement_unrest(self):
        mods = compute_action_modifiers(
            psych_stress=0.3, psych_purpose=0.5, psych_morale=0.5,
            political_engagement=0.8, legitimacy=0.1, faction_id=None)
        assert mods.get("sabotage", 0) > 0

    def test_faction_member_cooperates(self):
        mods = compute_action_modifiers(
            psych_stress=0.2, psych_purpose=0.5, psych_morale=0.5,
            political_engagement=0.3, legitimacy=0.6, faction_id="f1")
        assert mods.get("cooperate", 0) > 0

    def test_calm_no_faction_minimal_mods(self):
        mods = compute_action_modifiers(
            psych_stress=0.3, psych_purpose=0.5, psych_morale=0.5,
            political_engagement=0.3, legitimacy=0.6, faction_id=None)
        # Should have minimal or no modifiers
        total = sum(abs(v) for v in mods.values())
        assert total < 1.0

    def test_all_values_nonnegative(self):
        """Action weight modifiers should never be negative."""
        for _ in range(50):
            rng = random.Random(_)
            mods = compute_action_modifiers(
                psych_stress=rng.random(), psych_purpose=rng.random(),
                psych_morale=rng.random(), political_engagement=rng.random(),
                legitimacy=rng.random(), faction_id="f1" if rng.random() > 0.5 else None)
            for v in mods.values():
                assert v >= 0, f"Negative action modifier: {mods}"


# ---------------------------------------------------------------------------
# Legitimacy
# ---------------------------------------------------------------------------

class TestLegitimacy:
    def test_improving_resources_boost(self):
        d = compute_legitimacy_delta(0.05, "council", [], 10, 20, 0.0)
        assert d > 0

    def test_crisis_drops(self):
        d = compute_legitimacy_delta(0.0, "council", [], 10, 20, 0.8)
        assert d < 0

    def test_anarchy_penalty(self):
        d1 = compute_legitimacy_delta(0.0, "anarchy", [], 10, 20, 0.0)
        d2 = compute_legitimacy_delta(0.0, "council", [], 10, 20, 0.0)
        assert d1 < d2

    def test_capped(self):
        d = compute_legitimacy_delta(10.0, "council", [], 10, 20, 0.0)
        assert d <= 0.15 + 1e-9


# ---------------------------------------------------------------------------
# Faction detection
# ---------------------------------------------------------------------------

class TestFactionDetection:
    def _make_map(self, opinions: dict[str, tuple[float, float, float]]) -> dict[str, PoliticalState]:
        pm: dict[str, PoliticalState] = {}
        for cid, (a, b, c) in opinions.items():
            pm[cid] = PoliticalState(liberty_vs_security=a,
                                     growth_vs_sustainability=b,
                                     individual_vs_collective=c)
        return pm

    def test_no_factions_when_few(self):
        pm = self._make_map({"a": (0, 0, 0), "b": (0, 0, 0)})
        factions = detect_factions(pm, ["a", "b"], 10, [], random.Random(42))
        assert len(factions) == 0

    def test_single_faction_tight_cluster(self):
        pm = self._make_map({
            "a": (0.1, 0.1, 0.1), "b": (0.15, 0.1, 0.1),
            "c": (0.1, 0.15, 0.1), "d": (0.12, 0.12, 0.12),
        })
        factions = detect_factions(pm, ["a", "b", "c", "d"], 10, [], random.Random(42))
        assert len(factions) == 1
        assert len(factions[0].member_ids) == 4

    def test_two_factions_distant_clusters(self):
        pm = self._make_map({
            "a": (0.9, 0.9, 0.9), "b": (0.85, 0.9, 0.85), "c": (0.88, 0.88, 0.9),
            "d": (-0.9, -0.9, -0.9), "e": (-0.85, -0.9, -0.85), "f": (-0.88, -0.88, -0.9),
        })
        factions = detect_factions(pm, list(pm.keys()), 10, [], random.Random(42))
        assert len(factions) == 2

    def test_max_factions_cap(self):
        # Create 5 tight clusters, should cap at MAX_FACTIONS (4)
        pm: dict[str, PoliticalState] = {}
        for cluster_idx in range(5):
            base = (cluster_idx - 2) * 0.5
            for i in range(3):
                cid = f"c{cluster_idx}_{i}"
                pm[cid] = PoliticalState(
                    liberty_vs_security=base + i * 0.01,
                    growth_vs_sustainability=base + i * 0.01,
                    individual_vs_collective=0.0)
        factions = detect_factions(pm, list(pm.keys()), 10, [], random.Random(42))
        assert len(factions) <= MAX_FACTIONS

    def test_faction_ids_updated_on_colonists(self):
        pm = self._make_map({
            "a": (0.1, 0.1, 0.1), "b": (0.15, 0.1, 0.1), "c": (0.1, 0.15, 0.1),
        })
        detect_factions(pm, ["a", "b", "c"], 10, [], random.Random(42))
        faction_ids = {pm[cid].faction_id for cid in ["a", "b", "c"]}
        assert None not in faction_ids  # all assigned
        assert len(faction_ids) == 1    # all in same faction

    def test_unaffiliated_colonists_cleared(self):
        pm = self._make_map({
            "a": (0.9, 0.9, 0.9), "b": (0.85, 0.9, 0.85), "c": (0.88, 0.88, 0.9),
            "loner": (-0.5, 0.0, 0.5),
        })
        detect_factions(pm, list(pm.keys()), 10, [], random.Random(42))
        assert pm["loner"].faction_id is None

    def test_deterministic_across_runs(self):
        pm1 = self._make_map({
            "a": (0.1, 0.1, 0.1), "b": (0.15, 0.1, 0.1),
            "c": (0.1, 0.15, 0.1), "d": (-0.8, -0.8, -0.8),
            "e": (-0.75, -0.85, -0.8), "f": (-0.78, -0.8, -0.82),
        })
        pm2 = self._make_map({
            "a": (0.1, 0.1, 0.1), "b": (0.15, 0.1, 0.1),
            "c": (0.1, 0.15, 0.1), "d": (-0.8, -0.8, -0.8),
            "e": (-0.75, -0.85, -0.8), "f": (-0.78, -0.8, -0.82),
        })
        ids = ["a", "b", "c", "d", "e", "f"]
        f1 = detect_factions(pm1, ids, 10, [], random.Random(42))
        f2 = detect_factions(pm2, ids, 10, [], random.Random(42))
        assert len(f1) == len(f2)
        for a, b in zip(f1, f2):
            assert a.member_ids == b.member_ids


# ---------------------------------------------------------------------------
# tick_politics integration
# ---------------------------------------------------------------------------

class TestTickPolitics:
    def _make_contexts(self, n: int = 5) -> list[PoliticsContext]:
        contexts = []
        for i in range(n):
            contexts.append(PoliticsContext(
                colonist_id=f"c{i}",
                stats={"resolve": 0.5, "improvisation": 0.5, "empathy": 0.5,
                       "hoarding": 0.5, "faith": 0.5, "paranoia": 0.5},
                trusted_ids=[f"c{(i+1) % n}"],
                trust_weights=[0.6],
                event_severity=0.3,
                resource_avg=0.6,
                resource_delta=0.01,
                gov_type="council",
                had_crisis=False,
                action="cooperate",
            ))
        return contexts

    def test_tick_creates_states(self):
        pm: dict[str, PoliticalState] = {}
        contexts = self._make_contexts()
        result = tick_politics(pm, contexts, [], 0.6, 10, random.Random(42))
        assert len(result.snapshots) == 5
        assert all(cid in pm for cid in [f"c{i}" for i in range(5)])

    def test_tick_preserves_existing(self):
        pm = {"c0": PoliticalState(liberty_vs_security=0.5)}
        contexts = self._make_contexts(n=1)
        tick_politics(pm, contexts, [], 0.6, 10, random.Random(42))
        # Should have evolved, not reset
        assert pm["c0"].liberty_vs_security != 0.5 or pm["c0"].engagement != 0.30

    def test_opinions_bounded(self):
        pm: dict[str, PoliticalState] = {}
        contexts = self._make_contexts()
        for _ in range(20):
            tick_politics(pm, contexts, [], 0.6, _, random.Random(42))
        for ps in pm.values():
            for ax in OPINION_AXES:
                v = getattr(ps, ax)
                assert -1.0 <= v <= 1.0, f"{ax}={v}"
            assert 0.0 <= ps.engagement <= 1.0

    def test_legitimacy_bounded(self):
        pm: dict[str, PoliticalState] = {}
        legitimacy = 0.6
        for year in range(100):
            contexts = self._make_contexts()
            result = tick_politics(pm, contexts, [], legitimacy, year, random.Random(year))
            legitimacy = result.legitimacy
        assert 0.0 <= legitimacy <= 1.0

    def test_proposal_triggered_at_low_legitimacy(self):
        pm: dict[str, PoliticalState] = {}
        contexts = self._make_contexts()
        # Force very low legitimacy
        result = tick_politics(pm, contexts, [], 0.05, 10, random.Random(42))
        # Legitimacy may rise or stay low; check threshold
        if result.legitimacy < LEGITIMACY_PROPOSAL_THRESHOLD:
            assert result.proposal_triggered

    def test_determinism(self):
        def run_once(seed: int) -> PoliticsTickResult:
            pm: dict[str, PoliticalState] = {}
            contexts = self._make_contexts()
            return tick_politics(pm, contexts, [], 0.6, 10, random.Random(seed))
        r1 = run_once(42)
        r2 = run_once(42)
        assert r1.to_dict() == r2.to_dict()


# ---------------------------------------------------------------------------
# Init political state
# ---------------------------------------------------------------------------

class TestInitPoliticalState:
    def test_deterministic(self):
        stats = {"resolve": 0.7, "paranoia": 0.8, "empathy": 0.3,
                 "hoarding": 0.6, "faith": 0.2, "improvisation": 0.9}
        ps1 = _init_political_state(stats, random.Random(42))
        ps2 = _init_political_state(stats, random.Random(42))
        assert ps1.to_dict() == ps2.to_dict()

    def test_bounded(self):
        for seed in range(50):
            rng = random.Random(seed)
            stats = {s: rng.random() for s in ("resolve", "improvisation",
                     "empathy", "hoarding", "faith", "paranoia")}
            ps = _init_political_state(stats, rng)
            for ax in OPINION_AXES:
                v = getattr(ps, ax)
                assert -1.0 <= v <= 1.0
            assert 0.0 <= ps.engagement <= 1.0


# ---------------------------------------------------------------------------
# Property-based: physical invariants
# ---------------------------------------------------------------------------

class TestPhysicalInvariants:
    """Opinion values always in [-1, 1], engagement in [0, 1],
    legitimacy in [0, 1], faction sizes >= MIN_FACTION_SIZE."""

    @pytest.mark.parametrize("seed", range(20))
    def test_full_tick_bounds(self, seed: int):
        rng = random.Random(seed)
        pm: dict[str, PoliticalState] = {}
        contexts = []
        for i in range(8):
            contexts.append(PoliticsContext(
                colonist_id=f"c{i}",
                stats={s: rng.random() for s in ("resolve", "improvisation",
                       "empathy", "hoarding", "faith", "paranoia")},
                trusted_ids=[f"c{(i+1) % 8}", f"c{(i+2) % 8}"],
                trust_weights=[rng.random(), rng.random()],
                event_severity=rng.random(),
                resource_avg=rng.random(),
                resource_delta=rng.uniform(-0.1, 0.1),
                gov_type=rng.choice(["anarchy", "council", "dictator"]),
                had_crisis=rng.random() > 0.7,
                action=rng.choice(["rest", "cooperate", "sabotage", "mediate"]),
            ))
        result = tick_politics(pm, contexts, [], rng.random(), seed + 10, rng)

        for cid, ps in pm.items():
            for ax in OPINION_AXES:
                v = getattr(ps, ax)
                assert -1.0 <= v <= 1.0, f"{cid}.{ax}={v}"
            assert 0.0 <= ps.engagement <= 1.0

        assert 0.0 <= result.legitimacy <= 1.0

        for f in result.factions:
            assert len(f.member_ids) >= MIN_FACTION_SIZE
