"""
governance.py — Emergent governance for Mars-100.

Governance structures are NOT hardcoded. Instead, we track behavioral
metrics and INFER the governance type from observed patterns:
  - Power concentration (Gini coefficient of leadership scores)
  - Vote entropy (how spread vs. concentrated votes are)
  - Leadership persistence (how long the same leader holds power)
  - Amendment frequency (how often rules change)
  - Exile count (coercion level)

Governance labels are read AFTER the fact, never written into vote rules.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Proposal:
    """A governance proposal by a colonist."""
    id: str
    year: int
    proposer_id: str
    kind: str  # "leader", "resource", "exile", "amendment", "policy"
    description: str
    target: str | None = None  # target colonist for exile, etc.
    parameter: str | None = None  # what rule to change
    value: float | None = None  # proposed value
    votes_for: list[str] = field(default_factory=list)
    votes_against: list[str] = field(default_factory=list)
    abstentions: list[str] = field(default_factory=list)
    outcome: str | None = None  # "passed", "rejected", "tabled"
    subsim_evidence: dict | None = None

    def vote_result(self, threshold: float = 0.5) -> str:
        """Determine outcome: passed if for/(for+against) >= threshold."""
        total_cast = len(self.votes_for) + len(self.votes_against)
        if total_cast == 0:
            return "tabled"
        ratio = len(self.votes_for) / total_cast
        return "passed" if ratio >= threshold else "rejected"

    def to_dict(self) -> dict:
        """JSON-safe serialization."""
        return {
            "id": self.id,
            "year": self.year,
            "proposer_id": self.proposer_id,
            "kind": self.kind,
            "description": self.description,
            "target": self.target,
            "parameter": self.parameter,
            "value": self.value,
            "votes_for": self.votes_for,
            "votes_against": self.votes_against,
            "abstentions": self.abstentions,
            "outcome": self.outcome,
            "subsim_evidence": self.subsim_evidence,
        }


@dataclass
class Constitution:
    """The colony's evolving rule set."""
    decision_threshold: float = 0.5  # majority needed to pass
    exile_threshold: float = 0.67  # supermajority for exile
    resource_share_pct: float = 0.7  # % of resources shared equally
    leader_term_years: int = 5  # years before re-election
    max_proposals_per_year: int = 3
    amendments: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-safe serialization."""
        return {
            "decision_threshold": self.decision_threshold,
            "exile_threshold": self.exile_threshold,
            "resource_share_pct": self.resource_share_pct,
            "leader_term_years": self.leader_term_years,
            "max_proposals_per_year": self.max_proposals_per_year,
            "amendments": self.amendments,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Constitution:
        """Deserialize with bounds checking."""
        return cls(
            decision_threshold=max(0.3, min(0.9, data.get("decision_threshold", 0.5))),
            exile_threshold=max(0.5, min(0.95, data.get("exile_threshold", 0.67))),
            resource_share_pct=max(0.0, min(1.0, data.get("resource_share_pct", 0.7))),
            leader_term_years=max(1, min(20, data.get("leader_term_years", 5))),
            max_proposals_per_year=max(1, min(10, data.get("max_proposals_per_year", 3))),
            amendments=data.get("amendments", []),
        )


@dataclass
class GovernanceState:
    """Tracks governance metrics over time for post-hoc analysis."""
    leader_id: str | None = None
    leader_since: int = 0
    constitution: Constitution = field(default_factory=Constitution)
    proposals_history: list[dict] = field(default_factory=list)
    power_history: list[dict] = field(default_factory=list)
    year_labels: list[dict] = field(default_factory=list)
    exile_log: list[dict] = field(default_factory=list)

    def record_power_snapshot(self, year: int, colonists: list) -> None:
        """Record a power distribution snapshot for post-hoc labeling."""
        active = [c for c in colonists if c.alive]
        if not active:
            return

        scores = [c.leadership_score for c in active]
        gini = _gini(scores) if len(scores) > 1 else 0.0
        leader_tenure = year - self.leader_since if self.leader_id else 0
        top_3_share = _top_n_share(scores, 3)

        self.power_history.append({
            "year": year,
            "gini": round(gini, 4),
            "leader_id": self.leader_id,
            "leader_tenure": leader_tenure,
            "top_3_share": round(top_3_share, 4),
            "active_count": len(active),
        })

    def infer_governance_type(self, year: int) -> str:
        """Infer governance label from recent behavioral metrics."""
        recent = [p for p in self.power_history if p["year"] >= year - 10]
        if not recent:
            return "nascent"

        avg_gini = sum(p["gini"] for p in recent) / len(recent)
        avg_tenure = sum(p["leader_tenure"] for p in recent) / len(recent)
        avg_top3 = sum(p["top_3_share"] for p in recent) / len(recent)

        recent_proposals = [p for p in self.proposals_history
                           if p.get("year", 0) >= year - 10]
        proposal_rate = len(recent_proposals) / max(1, min(10, year))
        recent_exiles = [e for e in self.exile_log if e.get("year", 0) >= year - 10]
        coercion = len(recent_exiles) / max(1, len(recent))

        # Post-hoc inference from behavioral signals
        if avg_gini > 0.7 and avg_tenure > 15:
            return "autocracy"
        if avg_gini > 0.5 and avg_top3 > 0.7:
            return "oligarchy"
        if avg_gini < 0.3 and proposal_rate > 0.5:
            return "democracy"
        if coercion > 0.3:
            return "tyranny"
        if proposal_rate < 0.1 and avg_gini < 0.2:
            return "anarchy"
        if avg_gini < 0.4 and avg_tenure < 8:
            return "republic"
        return "mixed"

    def to_dict(self) -> dict:
        """JSON-safe serialization."""
        return {
            "leader_id": self.leader_id,
            "leader_since": self.leader_since,
            "constitution": self.constitution.to_dict(),
            "proposals_history": self.proposals_history[-50:],
            "power_history": self.power_history[-20:],
            "year_labels": self.year_labels[-20:],
            "exile_log": self.exile_log,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GovernanceState:
        """Deserialize governance state."""
        gs = cls()
        gs.leader_id = data.get("leader_id")
        gs.leader_since = data.get("leader_since", 0)
        gs.constitution = Constitution.from_dict(data.get("constitution", {}))
        gs.proposals_history = data.get("proposals_history", [])
        gs.power_history = data.get("power_history", [])
        gs.year_labels = data.get("year_labels", [])
        gs.exile_log = data.get("exile_log", [])
        return gs


# ---------------------------------------------------------------------------
# Voting mechanics
# ---------------------------------------------------------------------------

def colonist_votes(colonist, proposal: Proposal, leader_id: str | None) -> str:
    """Determine how a colonist votes based on their stats (not hardcoded alignment)."""
    if not colonist.alive:
        return "abstain"
    if colonist.id == proposal.proposer_id:
        return "for"  # proposers always vote for their own

    # Base disposition from relationship with proposer
    trust = colonist.trust(proposal.proposer_id)

    # Stat influences on voting tendency
    empathy_bias = colonist.stat("empathy") * 0.2
    paranoia_bias = -colonist.stat("paranoia") * 0.25
    faith_bias = 0.0
    if leader_id and proposal.proposer_id == leader_id:
        faith_bias = colonist.stat("faith") * 0.3

    # Proposal-type modifiers
    kind_bias = 0.0
    if proposal.kind == "exile" and proposal.target:
        target_trust = colonist.trust(proposal.target)
        kind_bias = -target_trust * 0.3  # distrust target → support exile
        if proposal.target == colonist.id:
            return "against"  # always vote against own exile
    elif proposal.kind == "resource":
        hoarding_bias = -colonist.stat("hoarding") * 0.2
        kind_bias = hoarding_bias  # hoarders resist sharing

    # Sub-sim evidence can sway votes
    evidence_bias = 0.0
    if proposal.subsim_evidence:
        fitness = proposal.subsim_evidence.get("fitness", 0.5)
        evidence_bias = (fitness - 0.5) * 0.4

    disposition = trust + empathy_bias + paranoia_bias + faith_bias + kind_bias + evidence_bias

    if disposition > 0.1:
        return "for"
    elif disposition < -0.1:
        return "against"
    return "abstain"


# ---------------------------------------------------------------------------
# Post-hoc metrics
# ---------------------------------------------------------------------------

def _gini(values: list[float]) -> float:
    """Gini coefficient: 0 = perfect equality, 1 = total inequality."""
    if not values or all(v == 0 for v in values):
        return 0.0
    n = len(values)
    sorted_v = sorted(values)
    total = sum(sorted_v)
    if total == 0:
        return 0.0
    cumsum = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(sorted_v):
        cumsum += v
        weighted_sum += (2 * (i + 1) - n - 1) * v
    return weighted_sum / (n * total)


def _top_n_share(values: list[float], n: int) -> float:
    """Fraction of total held by top N entries."""
    if not values:
        return 0.0
    total = sum(values)
    if total == 0:
        return 1.0 / max(len(values), 1)
    top = sum(sorted(values, reverse=True)[:n])
    return top / total


def compute_fitness(colony_state: dict) -> float:
    """Weighted fitness function for amendment evaluation.

    Components: survival + resource_stability + trust_cohesion + fairness - coercion
    """
    alive_count = colony_state.get("alive_count", 0)
    total_count = colony_state.get("total_count", 10)
    survival = alive_count / max(total_count, 1)

    resources = colony_state.get("resources", {})
    res_values = [resources.get(k, 0.5) for k in ("food", "water", "power", "oxygen")]
    resource_stability = sum(min(1.0, v) for v in res_values) / len(res_values) if res_values else 0.5

    avg_trust = colony_state.get("avg_trust", 0.0)
    trust_cohesion = (avg_trust + 1.0) / 2.0  # normalize -1..1 to 0..1

    gini = colony_state.get("power_gini", 0.5)
    fairness = 1.0 - gini

    exiles = colony_state.get("total_exiles", 0)
    coercion = min(1.0, exiles / max(total_count, 1))

    fitness = (
        survival * 0.3 +
        resource_stability * 0.2 +
        trust_cohesion * 0.2 +
        fairness * 0.2 -
        coercion * 0.1
    )
    return max(0.0, min(1.0, fitness))
