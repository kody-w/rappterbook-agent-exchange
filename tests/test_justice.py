"""Tests for Mars-100 justice engine."""
from __future__ import annotations

import random
from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, create_founding_ten
from src.mars100.colony import Resources, SocialGraph, RESOURCE_NAMES
from src.mars100.justice import (
    HarmRecord, Accusation, TrialResult,
    detect_harm, file_accusations, run_trial, apply_verdict,
    CRISIS_THRESHOLD, SABOTAGE_HARM_FLOOR,
)


# --------------- Helpers ---------------

def _make_colonist(cid: str, **stat_overrides: float) -> Colonist:
    stats = ColonistStats(resolve=0.5, improvisation=0.5, empathy=0.5,
                          hoarding=0.5, faith=0.5, paranoia=0.5)
    skills = ColonistSkills(terraforming=0.5, hydroponics=0.5, mediation=0.5,
                            coding=0.5, prayer=0.5, sabotage=0.5)
    for k, v in stat_overrides.items():
        if hasattr(stats, k):
            setattr(stats, k, v)
        elif hasattr(skills, k):
            setattr(skills, k, v)
    return Colonist(id=cid, name=cid.title(), element="fire",
                    archetype="settler", stats=stats, skills=skills,
                    decision_expr="(+ resolve empathy)")


def _healthy_resources() -> Resources:
    """Resources well above any threshold."""
    return Resources(food=0.8, water=0.8, power=0.8, air=0.8, medicine=0.8)


def _crisis_resources() -> Resources:
    """Resources below critical threshold."""
    return Resources(food=0.08, water=0.10, power=0.5, air=0.9, medicine=0.5)


def _social_graph(colonists: list[Colonist], rng: random.Random) -> SocialGraph:
    sg = SocialGraph()
    sg.initialize([c.id for c in colonists], rng)
    return sg


# --------------- HarmRecord ---------------

class TestHarmRecord:
    def test_to_dict(self) -> None:
        hr = HarmRecord(year=10, accused_id="a", action="sabotage",
                        resource_deltas={"food": -0.05}, crisis_resources=["food"],
                        harm_score=0.6)
        d = hr.to_dict()
        assert d["year"] == 10
        assert d["harm_score"] == 0.6
        assert "food" in d["crisis_resources"]

    def test_harm_score_bounded(self) -> None:
        hr = HarmRecord(year=1, accused_id="x", action="sabotage",
                        resource_deltas={}, crisis_resources=["food", "water"],
                        harm_score=0.0)
        assert 0.0 <= hr.harm_score <= 1.0


# --------------- detect_harm ---------------

class TestDetectHarm:
    def test_no_harm_healthy_colony_no_negative_deltas(self) -> None:
        """No harm when colony is healthy AND no negative resource deltas."""
        colonists = [_make_colonist("a", sabotage=0.8)]
        actions = {"a": "sabotage"}
        deltas = {"food": 0.05, "water": 0.03, "power": 0.01, "air": 0.0, "medicine": 0.0}
        resources = _healthy_resources()
        harms = detect_harm(10, actions, deltas, resources, colonists, random.Random(1))
        assert harms == []

    def test_sabotage_during_crisis(self) -> None:
        colonists = [_make_colonist("sab", sabotage=0.7)]
        actions = {"sab": "sabotage"}
        deltas = {"food": -0.05, "water": -0.04, "power": 0.01, "air": 0.0, "medicine": 0.0}
        resources = _crisis_resources()
        harms = detect_harm(10, actions, deltas, resources, colonists, random.Random(1))
        assert len(harms) == 1
        assert harms[0].accused_id == "sab"
        assert harms[0].action == "sabotage"
        assert 0.0 < harms[0].harm_score <= 1.0

    def test_hoard_during_food_crisis(self) -> None:
        colonists = [_make_colonist("hoarder", hoarding=0.8)]
        actions = {"hoarder": "hoard"}
        deltas = {"food": -0.03, "water": 0.0, "power": 0.0, "air": 0.0, "medicine": 0.0}
        resources = _crisis_resources()
        harms = detect_harm(10, actions, deltas, resources, colonists, random.Random(1))
        assert len(harms) == 1
        assert harms[0].action == "hoard"

    def test_cooperative_action_never_flagged(self) -> None:
        colonists = [_make_colonist("c")]
        actions = {"c": "cooperate"}
        deltas = {"food": -0.1, "water": -0.1, "power": -0.1, "air": -0.1, "medicine": -0.1}
        harms = detect_harm(10, actions, deltas, _crisis_resources(), colonists, random.Random(1))
        assert harms == []

    def test_inactive_colonist_ignored(self) -> None:
        c = _make_colonist("dead", sabotage=0.9)
        c.die(5, "test")
        actions = {"dead": "sabotage"}
        deltas = {"food": -0.05, "water": -0.05, "power": 0.0, "air": 0.0, "medicine": 0.0}
        harms = detect_harm(10, actions, deltas, _crisis_resources(), [c], random.Random(1))
        assert harms == []

    def test_sabotage_no_crisis_no_negative_deltas(self) -> None:
        """Sabotage with no crisis AND no negative deltas → no harm."""
        colonists = [_make_colonist("sab", sabotage=0.7)]
        actions = {"sab": "sabotage"}
        deltas = {"food": 0.01, "water": 0.0, "power": 0.01, "air": 0.0, "medicine": 0.0}
        resources = _healthy_resources()
        harms = detect_harm(10, actions, deltas, resources, colonists, random.Random(1))
        assert harms == []

    def test_sabotage_during_crisis_always_harmful(self) -> None:
        """Sabotage during crisis is always harmful regardless of deltas."""
        colonists = [_make_colonist("sab", sabotage=0.7)]
        actions = {"sab": "sabotage"}
        deltas = {"food": 0.01, "water": 0.0, "power": 0.01, "air": 0.0, "medicine": 0.0}
        resources = _crisis_resources()
        harms = detect_harm(10, actions, deltas, resources, colonists, random.Random(1))
        assert len(harms) == 1
        assert harms[0].accused_id == "sab"

    def test_sabotage_with_negative_delta_no_crisis(self) -> None:
        """Sabotage with negative resource deltas but no crisis triggers harm."""
        colonists = [_make_colonist("sab", sabotage=0.7)]
        actions = {"sab": "sabotage"}
        deltas = {"food": -0.05, "water": 0.0, "power": 0.01, "air": 0.0, "medicine": 0.0}
        resources = _healthy_resources()
        harms = detect_harm(10, actions, deltas, resources, colonists, random.Random(1))
        assert len(harms) == 1
        assert harms[0].accused_id == "sab"


# --------------- file_accusations ---------------

class TestFileAccusations:
    def test_files_accusation(self) -> None:
        rng = random.Random(42)
        accused = _make_colonist("villain", sabotage=0.8)
        others = [_make_colonist(f"c{i}") for i in range(3)]
        all_c = [accused] + others
        sg = _social_graph(all_c, rng)
        harm = HarmRecord(year=10, accused_id="villain", action="sabotage",
                          resource_deltas={"food": -0.05}, crisis_resources=["food"],
                          harm_score=0.6)
        accs = file_accusations([harm], all_c, sg, rng)
        assert len(accs) == 1
        assert accs[0].accused_id == "villain"
        assert accs[0].accuser_id != "villain"

    def test_too_few_colonists(self) -> None:
        rng = random.Random(42)
        c = [_make_colonist("a"), _make_colonist("b")]
        sg = _social_graph(c, rng)
        harm = HarmRecord(year=10, accused_id="a", action="sabotage",
                          resource_deltas={}, crisis_resources=["food"],
                          harm_score=0.5)
        accs = file_accusations([harm], c, sg, rng)
        assert accs == []

    def test_one_accusation_per_colonist(self) -> None:
        rng = random.Random(42)
        colonists = [_make_colonist(f"c{i}") for i in range(5)]
        sg = _social_graph(colonists, rng)
        harms = [
            HarmRecord(year=10, accused_id="c0", action="sabotage",
                       resource_deltas={}, crisis_resources=["food"], harm_score=0.5),
            HarmRecord(year=10, accused_id="c0", action="hoard",
                       resource_deltas={}, crisis_resources=["food"], harm_score=0.3),
        ]
        accs = file_accusations(harms, colonists, sg, rng)
        assert len(accs) == 1


# --------------- run_trial ---------------

class TestRunTrial:
    def test_trial_produces_verdict(self) -> None:
        rng = random.Random(42)
        colonists = create_founding_ten(42)
        sg = _social_graph(colonists, rng)
        harm = HarmRecord(year=5, accused_id=colonists[0].id, action="sabotage",
                          resource_deltas={"food": -0.1}, crisis_resources=["food"],
                          harm_score=0.5)
        acc = Accusation(year_filed=5, accused_id=colonists[0].id,
                         accuser_id=colonists[1].id, harm=harm)
        trial = run_trial(acc, 6, colonists, sg, rng)
        assert trial.verdict in ("acquit", "rehabilitate", "exile")
        assert len(trial.juror_ids) > 0

    def test_dead_accused_acquitted(self) -> None:
        rng = random.Random(42)
        colonists = create_founding_ten(42)
        sg = _social_graph(colonists, rng)
        colonists[0].die(5, "test")
        harm = HarmRecord(year=5, accused_id=colonists[0].id, action="sabotage",
                          resource_deltas={}, crisis_resources=["food"],
                          harm_score=0.9)
        acc = Accusation(year_filed=5, accused_id=colonists[0].id,
                         accuser_id=colonists[1].id, harm=harm)
        trial = run_trial(acc, 6, colonists, sg, rng)
        assert trial.verdict == "acquit"

    def test_too_few_jurors_acquitted(self) -> None:
        rng = random.Random(42)
        c = [_make_colonist("accused"), _make_colonist("accuser")]
        sg = _social_graph(c, rng)
        harm = HarmRecord(year=5, accused_id="accused", action="sabotage",
                          resource_deltas={}, crisis_resources=["food"],
                          harm_score=0.9)
        acc = Accusation(year_filed=5, accused_id="accused",
                         accuser_id="accuser", harm=harm)
        trial = run_trial(acc, 6, c, sg, rng)
        assert trial.verdict == "acquit"

    def test_serialization(self) -> None:
        rng = random.Random(42)
        colonists = create_founding_ten(42)
        sg = _social_graph(colonists, rng)
        harm = HarmRecord(year=5, accused_id=colonists[0].id, action="sabotage",
                          resource_deltas={"food": -0.1}, crisis_resources=["food"],
                          harm_score=0.5)
        acc = Accusation(year_filed=5, accused_id=colonists[0].id,
                         accuser_id=colonists[1].id, harm=harm)
        trial = run_trial(acc, 6, colonists, sg, rng)
        d = trial.to_dict()
        assert "verdict" in d
        assert "accusation" in d
        assert isinstance(d["juror_ids"], list)

    def test_subsim_evidence_included(self) -> None:
        rng = random.Random(42)
        colonists = create_founding_ten(42)
        sg = _social_graph(colonists, rng)
        harm = HarmRecord(year=5, accused_id=colonists[0].id, action="sabotage",
                          resource_deltas={"food": -0.1}, crisis_resources=["food"],
                          harm_score=0.7)
        acc = Accusation(year_filed=5, accused_id=colonists[0].id,
                         accuser_id=colonists[1].id, harm=harm)
        trial = run_trial(acc, 6, colonists, sg, rng)
        d = trial.to_dict()
        # At least one sub-sim should have been attempted
        has_subsim = d.get("subsim_prosecution") or d.get("subsim_defense")
        assert has_subsim

    def test_high_harm_leads_to_conviction(self) -> None:
        """With high harm score and paranoid jurors, conviction should happen."""
        rng = random.Random(42)
        # Create colonists with high paranoia (conviction-prone)
        colonists = [_make_colonist(f"c{i}", paranoia=0.9, empathy=0.1) for i in range(6)]
        sg = _social_graph(colonists, rng)
        harm = HarmRecord(year=5, accused_id="c0", action="sabotage",
                          resource_deltas={"food": -0.2}, crisis_resources=["food", "water", "power"],
                          harm_score=0.9)
        acc = Accusation(year_filed=5, accused_id="c0",
                         accuser_id="c1", harm=harm)
        trial = run_trial(acc, 6, colonists, sg, rng)
        assert trial.verdict in ("rehabilitate", "exile")

    def test_deterministic(self) -> None:
        colonists1 = create_founding_ten(42)
        colonists2 = create_founding_ten(42)
        rng1, rng2 = random.Random(99), random.Random(99)
        sg1 = _social_graph(colonists1, random.Random(99))
        sg2 = _social_graph(colonists2, random.Random(99))
        harm = HarmRecord(year=5, accused_id=colonists1[0].id, action="sabotage",
                          resource_deltas={"food": -0.1}, crisis_resources=["food"],
                          harm_score=0.5)
        acc1 = Accusation(year_filed=5, accused_id=colonists1[0].id,
                          accuser_id=colonists1[1].id, harm=harm)
        acc2 = Accusation(year_filed=5, accused_id=colonists2[0].id,
                          accuser_id=colonists2[1].id, harm=harm)
        t1 = run_trial(acc1, 6, colonists1, sg1, rng1)
        t2 = run_trial(acc2, 6, colonists2, sg2, rng2)
        assert t1.verdict == t2.verdict


# --------------- apply_verdict ---------------

class TestApplyVerdict:
    def test_exile_deactivates(self) -> None:
        rng = random.Random(42)
        colonists = create_founding_ten(42)
        sg = _social_graph(colonists, rng)
        harm = HarmRecord(year=5, accused_id=colonists[0].id, action="sabotage",
                          resource_deltas={}, crisis_resources=["food"],
                          harm_score=0.9)
        acc = Accusation(year_filed=5, accused_id=colonists[0].id,
                         accuser_id=colonists[1].id, harm=harm)
        trial = TrialResult(year=6, accused_id=colonists[0].id,
                            accusation=acc, juror_ids=["j1"],
                            verdict="exile")
        apply_verdict(trial, colonists, sg, 6, rng)
        assert not colonists[0].is_active()

    def test_rehabilitate_adjusts_stats(self) -> None:
        rng = random.Random(42)
        c = _make_colonist("rehab", paranoia=0.8, hoarding=0.7, empathy=0.3)
        colonists = [c]
        sg = _social_graph(colonists, rng)
        harm = HarmRecord(year=5, accused_id="rehab", action="sabotage",
                          resource_deltas={}, crisis_resources=["food"],
                          harm_score=0.5)
        acc = Accusation(year_filed=5, accused_id="rehab",
                         accuser_id="nobody", harm=harm)
        trial = TrialResult(year=6, accused_id="rehab",
                            accusation=acc, juror_ids=["j1"],
                            verdict="rehabilitate")
        old_p = c.stats.paranoia
        old_e = c.stats.empathy
        apply_verdict(trial, colonists, sg, 6, rng)
        assert c.stats.paranoia < old_p
        assert c.stats.empathy > old_e

    def test_acquit_boosts_sympathy(self) -> None:
        rng = random.Random(42)
        colonists = create_founding_ten(42)
        sg = _social_graph(colonists, rng)
        old_trust = sg.get(colonists[2].id, colonists[0].id).trust
        harm = HarmRecord(year=5, accused_id=colonists[0].id, action="sabotage",
                          resource_deltas={}, crisis_resources=["food"],
                          harm_score=0.3)
        acc = Accusation(year_filed=5, accused_id=colonists[0].id,
                         accuser_id=colonists[1].id, harm=harm)
        trial = TrialResult(year=6, accused_id=colonists[0].id,
                            accusation=acc, juror_ids=[colonists[2].id],
                            votes_innocent=[colonists[2].id],
                            verdict="acquit")
        apply_verdict(trial, colonists, sg, 6, rng)
        new_trust = sg.get(colonists[2].id, colonists[0].id).trust
        assert new_trust >= old_trust

    def test_trial_creates_memories(self) -> None:
        rng = random.Random(42)
        colonists = create_founding_ten(42)
        sg = _social_graph(colonists, rng)
        harm = HarmRecord(year=5, accused_id=colonists[0].id, action="sabotage",
                          resource_deltas={}, crisis_resources=["food"],
                          harm_score=0.5)
        acc = Accusation(year_filed=5, accused_id=colonists[0].id,
                         accuser_id=colonists[1].id, harm=harm)
        trial = TrialResult(year=6, accused_id=colonists[0].id,
                            accusation=acc, juror_ids=[colonists[2].id],
                            verdict="acquit")
        mem_before = len(colonists[0].memories)
        apply_verdict(trial, colonists, sg, 6, rng)
        assert len(colonists[0].memories) > mem_before
        assert any("trial" in m.event.lower() for m in colonists[0].memories)

    def test_exile_trust_drops(self) -> None:
        rng = random.Random(42)
        colonists = create_founding_ten(42)
        sg = _social_graph(colonists, rng)
        harm = HarmRecord(year=5, accused_id=colonists[0].id, action="sabotage",
                          resource_deltas={}, crisis_resources=["food"],
                          harm_score=0.9)
        acc = Accusation(year_filed=5, accused_id=colonists[0].id,
                         accuser_id=colonists[1].id, harm=harm)
        old_trusts = [sg.get(c.id, colonists[0].id).trust
                      for c in colonists[1:] if c.is_active()]
        trial = TrialResult(year=6, accused_id=colonists[0].id,
                            accusation=acc, juror_ids=[colonists[2].id],
                            verdict="exile")
        apply_verdict(trial, colonists, sg, 6, rng)
        new_trusts = [sg.get(c.id, colonists[0].id).trust
                      for c in colonists[1:] if c.is_active()]
        assert sum(new_trusts) < sum(old_trusts)

    def test_dead_accused_noop(self) -> None:
        rng = random.Random(42)
        colonists = create_founding_ten(42)
        sg = _social_graph(colonists, rng)
        colonists[0].die(5, "test")
        harm = HarmRecord(year=5, accused_id=colonists[0].id, action="sabotage",
                          resource_deltas={}, crisis_resources=["food"],
                          harm_score=0.9)
        acc = Accusation(year_filed=5, accused_id=colonists[0].id,
                         accuser_id=colonists[1].id, harm=harm)
        trial = TrialResult(year=6, accused_id=colonists[0].id,
                            accusation=acc, juror_ids=[],
                            verdict="exile")
        apply_verdict(trial, colonists, sg, 6, rng)


# --------------- Integration ---------------

class TestIntegration:
    def test_full_pipeline(self) -> None:
        """End-to-end: detect → accuse → trial → verdict."""
        rng = random.Random(42)
        colonists = create_founding_ten(42)
        sg = _social_graph(colonists, rng)
        # Force sabotage action
        actions = {colonists[0].id: "sabotage"}
        for c in colonists[1:]:
            actions[c.id] = "farm"
        deltas = {"food": -0.08, "water": -0.05, "power": 0.0, "air": 0.0, "medicine": 0.0}
        resources = _crisis_resources()
        harms = detect_harm(10, actions, deltas, resources, colonists, rng)
        if harms:
            accusations = file_accusations(harms, colonists, sg, rng)
            for acc in accusations:
                trial = run_trial(acc, 11, colonists, sg, rng)
                assert trial.verdict in ("acquit", "rehabilitate", "exile")
                apply_verdict(trial, colonists, sg, 11, rng)

    def test_no_trials_in_healthy_colony(self) -> None:
        """No sabotage/hoard + healthy resources → zero harms."""
        rng = random.Random(42)
        colonists = create_founding_ten(42)
        actions = {c.id: "farm" for c in colonists}
        deltas = {r: 0.05 for r in RESOURCE_NAMES}
        harms = detect_harm(10, actions, deltas, _healthy_resources(), colonists, rng)
        assert harms == []

    def test_multiple_saboteurs(self) -> None:
        rng = random.Random(42)
        colonists = create_founding_ten(42)
        sg = _social_graph(colonists, rng)
        actions = {c.id: "sabotage" for c in colonists[:3]}
        for c in colonists[3:]:
            actions[c.id] = "farm"
        deltas = {"food": -0.1, "water": -0.1, "power": -0.05, "air": 0.0, "medicine": 0.0}
        harms = detect_harm(10, actions, deltas, _crisis_resources(), colonists, rng)
        assert len(harms) == 3
        accusations = file_accusations(harms, colonists, sg, rng)
        assert len(accusations) == 3


# --------------- Invariants ---------------

class TestInvariants:
    def test_harm_score_always_bounded(self) -> None:
        rng = random.Random(42)
        for seed in range(20):
            rng = random.Random(seed)
            colonists = [_make_colonist(f"c{seed}", sabotage=rng.random(), hoarding=rng.random())]
            actions = {f"c{seed}": rng.choice(["sabotage", "hoard"])}
            deltas = {r: rng.uniform(-0.2, 0.1) for r in RESOURCE_NAMES}
            resources = Resources(food=rng.random(), water=rng.random(),
                                  power=rng.random(), air=rng.random(),
                                  medicine=rng.random())
            harms = detect_harm(10, actions, deltas, resources, colonists, rng)
            for h in harms:
                assert 0.0 <= h.harm_score <= 1.0

    def test_verdict_always_valid(self) -> None:
        rng = random.Random(42)
        colonists = create_founding_ten(42)
        sg = _social_graph(colonists, rng)
        for seed in range(20):
            rng_inner = random.Random(seed)
            harm = HarmRecord(year=5, accused_id=colonists[0].id, action="sabotage",
                              resource_deltas={"food": -0.1},
                              crisis_resources=["food"], harm_score=rng_inner.random())
            acc = Accusation(year_filed=5, accused_id=colonists[0].id,
                             accuser_id=colonists[1].id, harm=harm)
            trial = run_trial(acc, 6, colonists, sg, rng_inner)
            assert trial.verdict in ("acquit", "rehabilitate", "exile")
