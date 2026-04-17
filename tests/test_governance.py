"""Tests for governance system."""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.governance import (
    Proposal, Constitution, GovernanceState,
    colonist_votes, compute_fitness, _gini, _top_n_share,
)
from src.colonist import Colonist, create_colony, STATS, SKILLS


def make_colonist(cid: str, empathy: float = 0.5, paranoia: float = 0.3,
                  faith: float = 0.3, hoarding: float = 0.3,
                  trust_map: dict | None = None) -> Colonist:
    """Helper to create a colonist with specific stats."""
    return Colonist(
        id=cid, name=f"Test-{cid}", element="earth",
        stats={
            "resolve": 0.5, "improvisation": 0.5,
            "empathy": empathy, "paranoia": paranoia,
            "faith": faith, "hoarding": hoarding,
        },
        skills={s: 0.3 for s in SKILLS},
        relationships=trust_map or {},
        memory=[],
    )


class TestProposal:
    def test_vote_result_passed(self):
        p = Proposal(id="p1", year=1, proposer_id="a", kind="policy",
                     description="test")
        p.votes_for = ["a", "b", "c"]
        p.votes_against = ["d"]
        assert p.vote_result() == "passed"

    def test_vote_result_rejected(self):
        p = Proposal(id="p1", year=1, proposer_id="a", kind="policy",
                     description="test")
        p.votes_for = ["a"]
        p.votes_against = ["b", "c", "d"]
        assert p.vote_result() == "rejected"

    def test_vote_result_tabled(self):
        p = Proposal(id="p1", year=1, proposer_id="a", kind="policy",
                     description="test")
        assert p.vote_result() == "tabled"

    def test_vote_result_custom_threshold(self):
        p = Proposal(id="p1", year=1, proposer_id="a", kind="policy",
                     description="test")
        p.votes_for = ["a", "b"]
        p.votes_against = ["c"]
        # 2/3 = 0.667, below 0.75 threshold
        assert p.vote_result(threshold=0.75) == "rejected"

    def test_serialization(self):
        p = Proposal(id="p1", year=5, proposer_id="aria", kind="leader",
                     description="test proposal")
        d = p.to_dict()
        assert d["id"] == "p1"
        assert d["kind"] == "leader"
        assert isinstance(d["votes_for"], list)


class TestConstitution:
    def test_defaults(self):
        c = Constitution()
        assert c.decision_threshold == 0.5
        assert c.exile_threshold == 0.67

    def test_from_dict_clamps(self):
        c = Constitution.from_dict({"decision_threshold": 2.0, "exile_threshold": 0.01})
        assert c.decision_threshold == 0.9
        assert c.exile_threshold == 0.5

    def test_round_trip(self):
        c = Constitution(decision_threshold=0.6, leader_term_years=7)
        d = c.to_dict()
        c2 = Constitution.from_dict(d)
        assert c2.decision_threshold == c.decision_threshold
        assert c2.leader_term_years == c.leader_term_years


class TestVoting:
    def test_proposer_votes_for(self):
        c = make_colonist("aria")
        p = Proposal(id="p1", year=1, proposer_id="aria", kind="policy",
                     description="test")
        assert colonist_votes(c, p, None) == "for"

    def test_dead_colonist_abstains(self):
        c = make_colonist("aria")
        c.alive = False
        p = Proposal(id="p1", year=1, proposer_id="kael", kind="policy",
                     description="test")
        assert colonist_votes(c, p, None) == "abstain"

    def test_high_empathy_tends_for(self):
        c = make_colonist("aria", empathy=0.9, paranoia=0.1,
                         trust_map={"kael": 0.3})
        p = Proposal(id="p1", year=1, proposer_id="kael", kind="policy",
                     description="test")
        assert colonist_votes(c, p, None) == "for"

    def test_high_paranoia_tends_against(self):
        c = make_colonist("aria", empathy=0.1, paranoia=0.9,
                         trust_map={"kael": -0.3})
        p = Proposal(id="p1", year=1, proposer_id="kael", kind="policy",
                     description="test")
        assert colonist_votes(c, p, None) == "against"

    def test_exile_target_votes_against(self):
        c = make_colonist("aria")
        p = Proposal(id="p1", year=1, proposer_id="kael", kind="exile",
                     description="exile aria", target="aria")
        assert colonist_votes(c, p, None) == "against"

    def test_faith_bonus_for_leader(self):
        c = make_colonist("aria", faith=0.9, empathy=0.3, paranoia=0.1,
                         trust_map={"kael": 0.0})
        p = Proposal(id="p1", year=1, proposer_id="kael", kind="policy",
                     description="test")
        # With kael as leader, faith should boost disposition
        vote_with = colonist_votes(c, p, "kael")
        assert vote_with == "for"

    def test_subsim_evidence_sways(self):
        c = make_colonist("aria", empathy=0.5, paranoia=0.5,
                         trust_map={"kael": 0.0})
        p = Proposal(id="p1", year=1, proposer_id="kael", kind="policy",
                     description="test", subsim_evidence={"fitness": 0.9})
        assert colonist_votes(c, p, None) == "for"


class TestGovernanceState:
    def test_initial_state(self):
        gs = GovernanceState()
        assert gs.leader_id is None
        assert gs.infer_governance_type(1) == "nascent"

    def test_power_snapshot(self):
        gs = GovernanceState()
        colony = create_colony(seed=42)
        gs.record_power_snapshot(1, colony)
        assert len(gs.power_history) == 1

    def test_governance_inference_with_data(self):
        gs = GovernanceState()
        colony = create_colony(seed=42)
        # Simulate 15 years of snapshots
        for year in range(1, 16):
            gs.record_power_snapshot(year, colony)
        label = gs.infer_governance_type(15)
        assert label in ("nascent", "democracy", "anarchy", "republic",
                         "oligarchy", "autocracy", "tyranny", "mixed")

    def test_serialization(self):
        gs = GovernanceState()
        gs.leader_id = "aria"
        gs.leader_since = 5
        d = gs.to_dict()
        gs2 = GovernanceState.from_dict(d)
        assert gs2.leader_id == "aria"
        assert gs2.leader_since == 5


class TestMetrics:
    def test_gini_equal(self):
        assert _gini([1, 1, 1, 1]) == pytest.approx(0.0, abs=0.01)

    def test_gini_unequal(self):
        g = _gini([0, 0, 0, 100])
        assert g > 0.5

    def test_gini_empty(self):
        assert _gini([]) == 0.0

    def test_top_n_share(self):
        share = _top_n_share([10, 20, 30, 40], 2)
        assert share == pytest.approx(0.7, rel=0.01)

    def test_top_n_share_empty(self):
        assert _top_n_share([], 3) == 0.0


class TestFitness:
    def test_perfect_colony(self):
        state = {
            "alive_count": 10, "total_count": 10,
            "resources": {"food": 1.5, "water": 1.5, "power": 1.5, "oxygen": 1.5},
            "avg_trust": 0.5, "power_gini": 0.1, "total_exiles": 0,
        }
        f = compute_fitness(state)
        assert f > 0.7

    def test_collapsed_colony(self):
        state = {
            "alive_count": 2, "total_count": 10,
            "resources": {"food": 0.1, "water": 0.1, "power": 0.1, "oxygen": 0.1},
            "avg_trust": -0.5, "power_gini": 0.9, "total_exiles": 5,
        }
        f = compute_fitness(state)
        assert f < 0.3

    def test_fitness_bounded(self):
        for scenario in [
            {"alive_count": 0, "total_count": 10, "resources": {},
             "avg_trust": -1.0, "power_gini": 1.0, "total_exiles": 10},
            {"alive_count": 10, "total_count": 10,
             "resources": {"food": 2.0, "water": 2.0, "power": 2.0, "oxygen": 2.0},
             "avg_trust": 1.0, "power_gini": 0.0, "total_exiles": 0},
        ]:
            f = compute_fitness(scenario)
            assert 0.0 <= f <= 1.0
