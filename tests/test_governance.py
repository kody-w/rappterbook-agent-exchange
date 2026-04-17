"""Tests for governance system."""
from __future__ import annotations

import random
import pytest
from src.mars100.governance import (
    GovernanceProposal, GovernanceState, GOVERNANCE_TYPES,
    should_propose, generate_proposal, resolve_vote, apply_governance,
)


class TestGovernanceState:
    def test_initial_state(self):
        gs = GovernanceState()
        assert gs.gov_type == "anarchy"

    def test_to_dict(self):
        gs = GovernanceState()
        d = gs.to_dict()
        assert d["gov_type"] == "anarchy"
        assert "history" in d
        assert "constitution" in d


class TestShouldPropose:
    def test_never_year_1(self):
        gs = GovernanceState()
        rng = random.Random(42)
        assert not should_propose(1, gs, rng)

    def test_possible_later(self):
        found = False
        for s in range(200):
            if should_propose(10, GovernanceState(), random.Random(s)):
                found = True
                break
        assert found, "should_propose should be True at least once in 200 seeds"


class TestGenerateProposal:
    def test_basic(self):
        gs = GovernanceState()
        rng = random.Random(42)
        p = generate_proposal(year=10, proposer_id="c1", current_gov=gs, rng=rng)
        assert p.year == 10
        assert p.proposer_id == "c1"
        assert p.gov_type in GOVERNANCE_TYPES
        assert p.gov_type != gs.gov_type

    def test_different_from_current(self):
        gs = GovernanceState(gov_type="council")
        rng = random.Random(42)
        p = generate_proposal(year=10, proposer_id="c1", current_gov=gs, rng=rng)
        assert p.gov_type != "council"


class TestResolveVote:
    def _make_proposal(self, **kwargs) -> GovernanceProposal:
        defaults = {"id": "p1", "year": 1, "proposer_id": "c1", "gov_type": "council",
                    "parameters": {}, "rationale": "test"}
        defaults.update(kwargs)
        return GovernanceProposal(**defaults)

    def test_pass(self):
        p = self._make_proposal(votes_for=["a", "b", "c", "d"],
                                votes_against=["e"])
        assert resolve_vote(p, active_count=5)

    def test_fail(self):
        p = self._make_proposal(votes_for=["a"],
                                votes_against=["b", "c", "d", "e"])
        assert not resolve_vote(p, active_count=5)

    def test_supermajority_needed(self):
        p = self._make_proposal(votes_for=["a", "b", "c"],
                                votes_against=["d", "e"])
        assert resolve_vote(p, active_count=5)

    def test_just_under_threshold(self):
        p = self._make_proposal(votes_for=["a", "b"],
                                votes_against=["c", "d", "e"])
        assert not resolve_vote(p, active_count=5)


class TestApplyGovernance:
    def _make_proposal(self, **kwargs) -> GovernanceProposal:
        defaults = {"id": "p1", "year": 5, "proposer_id": "c1", "gov_type": "council",
                    "parameters": {}, "rationale": "test", "passed": True}
        defaults.update(kwargs)
        return GovernanceProposal(**defaults)

    def test_transition(self):
        p = self._make_proposal(gov_type="council")
        gs = GovernanceState(gov_type="anarchy")
        active_ids = ["c1", "c2", "c3"]
        rng = random.Random(42)
        apply_governance(p, gs, active_ids, rng)
        assert gs.gov_type == "council"
        assert len(gs.history) == 1
        assert gs.history[0]["from"] == "anarchy"
        assert gs.history[0]["to"] == "council"

    def test_dictator_sets_leader(self):
        p = self._make_proposal(gov_type="dictator")
        gs = GovernanceState()
        active_ids = ["c1", "c2"]
        rng = random.Random(42)
        apply_governance(p, gs, active_ids, rng)
        assert gs.gov_type == "dictator"
        assert gs.leader_id is not None

    def test_constitution_grows(self):
        gs = GovernanceState()
        rng = random.Random(42)
        active_ids = ["c1", "c2", "c3"]
        for i, gt in enumerate(["council", "consensus", "lottery"]):
            p = self._make_proposal(id=f"p{i}", year=i+5, gov_type=gt)
            apply_governance(p, gs, active_ids, rng)
        assert len(gs.constitution) >= 3


class TestGovernanceTypes:
    def test_all_types_exist(self):
        expected = {"anarchy", "council", "dictator", "lottery", "consensus", "ai_governor"}
        assert set(GOVERNANCE_TYPES) == expected
