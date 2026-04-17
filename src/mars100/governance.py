"""
Emergent governance for Mars-100.

Governance structures emerge from colonist proposals and votes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

GOVERNANCE_TYPES = ("anarchy", "council", "dictator", "lottery", "consensus", "ai_governor")


@dataclass
class GovernanceProposal:
    """A proposal for changing colony governance."""
    id: str
    year: int
    proposer_id: str
    gov_type: str
    parameters: dict[str, Any]
    rationale: str
    subsim_result: dict | None = None
    votes_for: list[str] = field(default_factory=list)
    votes_against: list[str] = field(default_factory=list)
    passed: bool | None = None

    def to_dict(self) -> dict:
        return {"id": self.id, "year": self.year, "proposer_id": self.proposer_id,
                "gov_type": self.gov_type, "parameters": self.parameters,
                "rationale": self.rationale, "subsim_result": self.subsim_result,
                "votes_for": self.votes_for, "votes_against": self.votes_against,
                "passed": self.passed}


@dataclass
class GovernanceState:
    """Current governance model of the colony."""
    gov_type: str = "anarchy"
    leader_id: str | None = None
    council_ids: list[str] = field(default_factory=list)
    term_end_year: int | None = None
    ai_program: str | None = None
    constitution: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"gov_type": self.gov_type, "leader_id": self.leader_id,
                "council_ids": self.council_ids, "term_end_year": self.term_end_year,
                "ai_program": self.ai_program, "constitution": self.constitution,
                "history": self.history}

    @classmethod
    def from_dict(cls, d: dict) -> GovernanceState:
        return cls(gov_type=d.get("gov_type", "anarchy"), leader_id=d.get("leader_id"),
                   council_ids=d.get("council_ids", []), term_end_year=d.get("term_end_year"),
                   ai_program=d.get("ai_program"), constitution=d.get("constitution", []),
                   history=d.get("history", []))


def should_propose(year: int, gov_state: GovernanceState, rng: random.Random) -> bool:
    """Determine if conditions are ripe for a governance proposal."""
    if gov_state.gov_type == "anarchy" and year >= 5:
        return rng.random() < 0.4
    if gov_state.term_end_year is not None and year >= gov_state.term_end_year:
        return True
    return rng.random() < 0.08 + (year / 500)


def generate_proposal(year: int, proposer_id: str, current_gov: GovernanceState,
                      rng: random.Random) -> GovernanceProposal:
    """Generate a governance proposal."""
    options = [g for g in GOVERNANCE_TYPES if g != current_gov.gov_type]
    gov_type = rng.choice(options)
    params: dict[str, Any] = {}
    rationale = ""
    if gov_type == "council":
        cs = rng.choice([3, 5])
        ty = rng.choice([5, 10])
        params["council_size"] = cs
        params["term_years"] = ty
        rationale = f"Elect a council of {cs} for {ty} years."
    elif gov_type == "dictator":
        params["term_years"] = rng.choice([5, 10, 0])
        rationale = "Appoint a strong leader to guide us through crisis."
    elif gov_type == "lottery":
        ry = rng.choice([1, 2, 3])
        params["rotation_years"] = ry
        rationale = f"Random leader selection every {ry} year(s)."
    elif gov_type == "consensus":
        qf = rng.choice([0.6, 0.75, 1.0])
        params["quorum_fraction"] = qf
        rationale = f"All decisions require {int(qf * 100)}% agreement."
    elif gov_type == "ai_governor":
        params["source"] = "(if (> food 0.5) (+ morale 0.1) (- morale 0.05))"
        rationale = "Let a LisPy program govern — transparent, deterministic."
    elif gov_type == "anarchy":
        rationale = "Return to self-governance. No rulers, no masters."
    return GovernanceProposal(id=f"prop-y{year}-{gov_type}", year=year,
                              proposer_id=proposer_id, gov_type=gov_type,
                              parameters=params, rationale=rationale)


def resolve_vote(proposal: GovernanceProposal, active_count: int) -> bool:
    """Determine if a proposal passes. Simple majority."""
    total_votes = len(proposal.votes_for) + len(proposal.votes_against)
    if total_votes == 0:
        return False
    return len(proposal.votes_for) > len(proposal.votes_against)


def apply_governance(proposal: GovernanceProposal, state: GovernanceState,
                     active_ids: list[str], rng: random.Random) -> None:
    """Apply a passed governance proposal."""
    state.history.append({"year": proposal.year, "from": state.gov_type,
                          "to": proposal.gov_type, "proposal_id": proposal.id})
    state.gov_type = proposal.gov_type
    state.leader_id = None
    state.council_ids = []
    state.ai_program = None
    params = proposal.parameters
    if proposal.gov_type == "council":
        size = min(params.get("council_size", 3), len(active_ids))
        state.council_ids = rng.sample(active_ids, size)
        state.term_end_year = proposal.year + params.get("term_years", 5)
    elif proposal.gov_type == "dictator":
        state.leader_id = proposal.proposer_id
        term = params.get("term_years", 0)
        state.term_end_year = proposal.year + term if term > 0 else None
    elif proposal.gov_type == "lottery":
        state.leader_id = rng.choice(active_ids)
        state.term_end_year = proposal.year + params.get("rotation_years", 1)
    elif proposal.gov_type == "consensus":
        state.term_end_year = None
    elif proposal.gov_type == "ai_governor":
        state.ai_program = params.get("source", "(+ 0 0)")
        state.term_end_year = proposal.year + 10
    elif proposal.gov_type == "anarchy":
        state.term_end_year = None
    state.constitution.append(f"Year {proposal.year}: governance changed to {proposal.gov_type}")
