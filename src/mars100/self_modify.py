"""Self-modification engine — colonists rewrite their own LisPy decision expressions.

Meta-aware colonists (those who have realized they may be in a simulation) can
propose mutations to their `decision_expr`, evaluate them in sandboxed contexts,
and adopt improvements. This is the homoiconic property of LisPy made concrete:
colonists are both data AND programs that can rewrite themselves.

Constitutional basis: Amendment XIII (Turtles All the Way Down) — self-modifying
agents as a concrete instance of recursive self-modeling.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import Colonist
from src.mars100.lispy_vm import run as lispy_run, LispyError


# ── Configuration ──────────────────────────────────────────────────────────

MAX_MUTATIONS_PER_YEAR = 3
EVAL_CONTEXTS = 5
ACCEPTANCE_MARGIN = 0.05   # new expr must outperform old by this margin
MIN_META_AWARENESS = 0.6   # minimum meta_awareness stat to attempt mutation

# Whitelist of safe mutation operators and constants
SAFE_OPERATORS = ["+", "-", "*", "min", "max", "if"]
SAFE_CONSTANTS = ["0", "0.1", "0.2", "0.5", "1", "2"]
SAFE_VARIABLES = [
    "water", "oxygen", "food", "energy", "materials",
    "resolve", "empathy", "improvisation", "faith",
    "trust", "cooperation", "year",
]


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class MutationProposal:
    """A proposed change to a colonist's decision expression."""
    colonist_id: str
    year: int
    strategy: str
    old_expr: str
    new_expr: str
    old_score: float = 0.0
    new_score: float = 0.0
    accepted: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "colonist_id": self.colonist_id, "year": self.year,
            "strategy": self.strategy,
            "old_expr": self.old_expr, "new_expr": self.new_expr,
            "old_score": round(self.old_score, 4),
            "new_score": round(self.new_score, 4),
            "accepted": self.accepted, "reason": self.reason,
        }


@dataclass
class MutationLog:
    """Accumulates all mutation proposals and acceptances."""
    proposals: list[MutationProposal] = field(default_factory=list)

    @property
    def total_proposals(self) -> int:
        return len(self.proposals)

    @property
    def total_accepted(self) -> int:
        return sum(1 for p in self.proposals if p.accepted)

    def lineage(self, colonist_id: str) -> list[MutationProposal]:
        """Return all mutations for a given colonist, chronologically."""
        return [p for p in self.proposals if p.colonist_id == colonist_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_proposals": self.total_proposals,
            "total_accepted": self.total_accepted,
            "proposals": [p.to_dict() for p in self.proposals[-100:]],
        }


# ── Tokenization helpers ──────────────────────────────────────────────────

def tokenize_expr(expr: str) -> list[str]:
    """Simple s-expression tokenizer."""
    return expr.replace("(", " ( ").replace(")", " ) ").split()


def detokenize(tokens: list[str]) -> str:
    """Reconstruct s-expression from tokens."""
    result = " ".join(tokens)
    # Clean up spacing around parens
    result = result.replace("( ", "(").replace(" )", ")")
    return result


# ── Mutation strategies ────────────────────────────────────────────────────

def _find_token_indices(tokens: list[str], predicate) -> list[int]:
    """Find indices of tokens matching a predicate."""
    return [i for i, t in enumerate(tokens) if predicate(t)]


def _is_number(s: str) -> bool:
    """Check if a string looks like a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def propose_mutation(
    colonist: Colonist,
    year: int,
    rng: random.Random,
) -> MutationProposal | None:
    """Generate a mutation proposal using one of three strategies.

    Strategies:
    - constant_swap: replace a numeric constant with another
    - operator_swap: replace an operator with a compatible one
    - variable_swap: replace a variable reference with another

    Returns None if no valid mutation can be found.
    """
    expr = colonist.decision_expr
    tokens = tokenize_expr(expr)

    strategies = ["constant_swap", "operator_swap", "variable_swap"]
    rng.shuffle(strategies)

    for strategy in strategies:
        if strategy == "constant_swap":
            indices = _find_token_indices(tokens, _is_number)
            if indices:
                idx = rng.choice(indices)
                old_val = tokens[idx]
                new_val = rng.choice([c for c in SAFE_CONSTANTS if c != old_val] or SAFE_CONSTANTS)
                new_tokens = list(tokens)
                new_tokens[idx] = new_val
                return MutationProposal(
                    colonist_id=colonist.id, year=year,
                    strategy=strategy,
                    old_expr=expr, new_expr=detokenize(new_tokens),
                )

        elif strategy == "operator_swap":
            indices = _find_token_indices(
                tokens, lambda t: t in SAFE_OPERATORS)
            if indices:
                idx = rng.choice(indices)
                old_op = tokens[idx]
                new_op = rng.choice([o for o in SAFE_OPERATORS if o != old_op] or SAFE_OPERATORS)
                new_tokens = list(tokens)
                new_tokens[idx] = new_op
                return MutationProposal(
                    colonist_id=colonist.id, year=year,
                    strategy=strategy,
                    old_expr=expr, new_expr=detokenize(new_tokens),
                )

        elif strategy == "variable_swap":
            indices = _find_token_indices(
                tokens, lambda t: t in SAFE_VARIABLES)
            if indices:
                idx = rng.choice(indices)
                old_var = tokens[idx]
                new_var = rng.choice([v for v in SAFE_VARIABLES if v != old_var] or SAFE_VARIABLES)
                new_tokens = list(tokens)
                new_tokens[idx] = new_var
                return MutationProposal(
                    colonist_id=colonist.id, year=year,
                    strategy=strategy,
                    old_expr=expr, new_expr=detokenize(new_tokens),
                )

    return None


# ── Evaluation ─────────────────────────────────────────────────────────────

def _make_eval_context(rng: random.Random) -> dict[str, float]:
    """Generate a random evaluation context (resource/stat scenario)."""
    return {
        "water": rng.uniform(0.1, 0.9),
        "oxygen": rng.uniform(0.1, 0.9),
        "food": rng.uniform(0.1, 0.9),
        "energy": rng.uniform(0.1, 0.9),
        "materials": rng.uniform(0.1, 0.9),
        "resolve": rng.uniform(0.2, 0.8),
        "empathy": rng.uniform(0.2, 0.8),
        "improvisation": rng.uniform(0.2, 0.8),
        "faith": rng.uniform(0.2, 0.8),
        "trust": rng.uniform(0.0, 1.0),
        "cooperation": rng.uniform(0.0, 1.0),
        "year": float(rng.randint(1, 100)),
    }


def _safe_eval_expr(expr: str, context: dict[str, float]) -> float | None:
    """Evaluate a LisPy expression safely, returning None on error."""
    try:
        result = lispy_run(expr, context)
        if isinstance(result, (int, float)):
            return float(result)
        return None
    except (LispyError, Exception):
        return None


def evaluate_mutation(
    proposal: MutationProposal,
    rng: random.Random,
    num_contexts: int = EVAL_CONTEXTS,
) -> MutationProposal:
    """Evaluate a mutation across multiple random contexts.

    The new expression must produce valid numeric results in ALL contexts
    and must outscore the old expression by at least ACCEPTANCE_MARGIN
    on average to be accepted.
    """
    old_scores: list[float] = []
    new_scores: list[float] = []

    for _ in range(num_contexts):
        ctx = _make_eval_context(rng)
        old_result = _safe_eval_expr(proposal.old_expr, ctx)
        new_result = _safe_eval_expr(proposal.new_expr, ctx)

        if old_result is None or new_result is None:
            proposal.accepted = False
            proposal.reason = "eval_error"
            return proposal

        old_scores.append(old_result)
        new_scores.append(new_result)

    avg_old = sum(old_scores) / len(old_scores) if old_scores else 0.0
    avg_new = sum(new_scores) / len(new_scores) if new_scores else 0.0

    proposal.old_score = avg_old
    proposal.new_score = avg_new

    if avg_new > avg_old + ACCEPTANCE_MARGIN:
        proposal.accepted = True
        proposal.reason = "outperformed"
    else:
        proposal.accepted = False
        proposal.reason = "insufficient_improvement"

    return proposal


# ── Application ────────────────────────────────────────────────────────────

def apply_mutation(colonist: Colonist, proposal: MutationProposal) -> None:
    """Apply an accepted mutation to a colonist's decision expression.

    Only call this for accepted proposals. Updates the colonist's
    decision_expr in place.
    """
    if not proposal.accepted:
        return
    colonist.decision_expr = proposal.new_expr


# ── Tick function (main entry point) ──────────────────────────────────────

def tick_self_modify(
    colonists: list[Colonist],
    meta_aware_ids: set[str],
    year: int,
    mutation_log: MutationLog,
    rng: random.Random,
) -> list[MutationProposal]:
    """Run one year of self-modification for eligible colonists.

    Returns list of all proposals (accepted and rejected).
    Mutations are NOT applied here — caller must apply accepted mutations
    using apply_mutation() to support staged end-of-tick application.
    """
    proposals: list[MutationProposal] = []
    mutations_this_year = 0

    eligible = [
        c for c in colonists
        if c.alive and c.id in meta_aware_ids
        and getattr(c.stats, "meta_awareness", 0.0) >= MIN_META_AWARENESS
    ]

    rng.shuffle(eligible)

    for colonist in eligible:
        if mutations_this_year >= MAX_MUTATIONS_PER_YEAR:
            break

        proposal = propose_mutation(colonist, year, rng)
        if proposal is None:
            continue

        proposal = evaluate_mutation(proposal, rng)
        proposals.append(proposal)
        mutation_log.proposals.append(proposal)
        mutations_this_year += 1

    return proposals
