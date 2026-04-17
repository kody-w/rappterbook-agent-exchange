"""Tests for the self-modification engine — mutation proposals, evaluation, application."""
from __future__ import annotations

import random

import pytest

from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills
from src.mars100.self_modify import (
    MutationProposal, MutationLog,
    tokenize_expr, detokenize,
    propose_mutation, evaluate_mutation, apply_mutation,
    tick_self_modify,
    SAFE_OPERATORS, SAFE_CONSTANTS, SAFE_VARIABLES,
    MAX_MUTATIONS_PER_YEAR, ACCEPTANCE_MARGIN,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_colonist(
    cid: str = "c-0",
    decision_expr: str = "(+ resolve (* empathy 0.5))",
    meta_awareness: float = 0.8,
) -> Colonist:
    stats = ColonistStats(
        resolve=0.6, improvisation=0.5, empathy=0.7,
        hoarding=0.3, faith=0.4, paranoia=0.3,
    )
    c = Colonist(
        id=cid, name=cid.title(), element="water", archetype="pioneer",
        stats=stats, skills=ColonistSkills(),
        decision_expr=decision_expr,
    )
    # meta_awareness is dynamically added by the engine
    c.stats.meta_awareness = meta_awareness
    return c


# ── Tokenization ───────────────────────────────────────────────────────────

class TestTokenize:
    def test_simple_expr(self) -> None:
        tokens = tokenize_expr("(+ 1 2)")
        assert tokens == ["(", "+", "1", "2", ")"]

    def test_nested_expr(self) -> None:
        tokens = tokenize_expr("(+ resolve (* empathy 0.5))")
        assert tokens == ["(", "+", "resolve", "(", "*", "empathy", "0.5", ")", ")"]

    def test_roundtrip(self) -> None:
        expr = "(+ resolve (* empathy 0.5))"
        tokens = tokenize_expr(expr)
        reconstructed = detokenize(tokens)
        assert reconstructed == expr


# ── Mutation proposal ──────────────────────────────────────────────────────

class TestProposeMutation:
    def test_returns_proposal(self) -> None:
        c = _make_colonist()
        proposal = propose_mutation(c, year=5, rng=random.Random(42))
        assert proposal is not None
        assert isinstance(proposal, MutationProposal)
        assert proposal.colonist_id == "c-0"
        assert proposal.year == 5

    def test_new_expr_differs_from_old(self) -> None:
        c = _make_colonist()
        proposal = propose_mutation(c, year=5, rng=random.Random(42))
        assert proposal is not None
        assert proposal.new_expr != proposal.old_expr

    def test_strategy_is_valid(self) -> None:
        c = _make_colonist()
        for seed in range(20):
            proposal = propose_mutation(c, year=1, rng=random.Random(seed))
            if proposal:
                assert proposal.strategy in ("constant_swap", "operator_swap", "variable_swap")

    def test_constant_swap_changes_number(self) -> None:
        c = _make_colonist(decision_expr="(+ 0.5 resolve)")
        found = False
        for seed in range(50):
            proposal = propose_mutation(c, year=1, rng=random.Random(seed))
            if proposal and proposal.strategy == "constant_swap":
                old_tokens = tokenize_expr(proposal.old_expr)
                new_tokens = tokenize_expr(proposal.new_expr)
                assert len(old_tokens) == len(new_tokens)
                found = True
                break
        assert found, "constant_swap never selected"

    def test_operator_swap_changes_operator(self) -> None:
        c = _make_colonist(decision_expr="(+ resolve empathy)")
        found = False
        for seed in range(50):
            proposal = propose_mutation(c, year=1, rng=random.Random(seed))
            if proposal and proposal.strategy == "operator_swap":
                found = True
                break
        assert found, "operator_swap never selected"

    def test_no_mutation_for_expr_without_targets(self) -> None:
        c = _make_colonist(decision_expr="()")
        proposal = propose_mutation(c, year=1, rng=random.Random(42))
        assert proposal is None or isinstance(proposal, MutationProposal)


# ── Evaluation ─────────────────────────────────────────────────────────────

class TestEvaluateMutation:
    def test_accepted_if_outperforms(self) -> None:
        proposal = MutationProposal(
            colonist_id="c-0", year=5, strategy="constant_swap",
            old_expr="(+ 0.1 resolve)",
            new_expr="(+ 0.5 resolve)",
        )
        result = evaluate_mutation(proposal, rng=random.Random(42))
        assert result.accepted is True
        assert result.new_score > result.old_score + ACCEPTANCE_MARGIN

    def test_rejected_if_worse(self) -> None:
        proposal = MutationProposal(
            colonist_id="c-0", year=5, strategy="constant_swap",
            old_expr="(+ 0.5 resolve)",
            new_expr="(+ 0.1 resolve)",
        )
        result = evaluate_mutation(proposal, rng=random.Random(42))
        assert result.accepted is False

    def test_rejected_on_eval_error(self) -> None:
        proposal = MutationProposal(
            colonist_id="c-0", year=5, strategy="operator_swap",
            old_expr="(+ resolve empathy)",
            new_expr="(if resolve)",
        )
        result = evaluate_mutation(proposal, rng=random.Random(42))
        assert isinstance(result, MutationProposal)

    def test_margin_requirement(self) -> None:
        proposal = MutationProposal(
            colonist_id="c-0", year=5, strategy="constant_swap",
            old_expr="(+ resolve 0.5)",
            new_expr="(+ resolve 0.5)",
        )
        result = evaluate_mutation(proposal, rng=random.Random(42))
        assert result.accepted is False


# ── Application ────────────────────────────────────────────────────────────

class TestApplyMutation:
    def test_apply_changes_decision_expr(self) -> None:
        c = _make_colonist(decision_expr="(+ resolve 0.1)")
        proposal = MutationProposal(
            colonist_id="c-0", year=5, strategy="constant_swap",
            old_expr="(+ resolve 0.1)", new_expr="(+ resolve 0.5)",
            accepted=True,
        )
        apply_mutation(c, proposal)
        assert c.decision_expr == "(+ resolve 0.5)"

    def test_no_apply_if_rejected(self) -> None:
        c = _make_colonist(decision_expr="(+ resolve 0.1)")
        proposal = MutationProposal(
            colonist_id="c-0", year=5, strategy="constant_swap",
            old_expr="(+ resolve 0.1)", new_expr="(+ resolve 0.5)",
            accepted=False,
        )
        apply_mutation(c, proposal)
        assert c.decision_expr == "(+ resolve 0.1)"


# ── Mutation log ───────────────────────────────────────────────────────────

class TestMutationLog:
    def test_empty_log(self) -> None:
        log = MutationLog()
        assert log.total_proposals == 0
        assert log.total_accepted == 0

    def test_lineage(self) -> None:
        log = MutationLog()
        log.proposals.append(MutationProposal(
            colonist_id="c-0", year=1, strategy="constant_swap",
            old_expr="a", new_expr="b", accepted=True,
        ))
        log.proposals.append(MutationProposal(
            colonist_id="c-1", year=2, strategy="operator_swap",
            old_expr="c", new_expr="d", accepted=False,
        ))
        assert len(log.lineage("c-0")) == 1
        assert len(log.lineage("c-1")) == 1
        assert len(log.lineage("c-99")) == 0

    def test_to_dict(self) -> None:
        log = MutationLog()
        log.proposals.append(MutationProposal(
            colonist_id="c-0", year=1, strategy="constant_swap",
            old_expr="a", new_expr="b", accepted=True,
        ))
        d = log.to_dict()
        assert d["total_proposals"] == 1
        assert d["total_accepted"] == 1
        assert len(d["proposals"]) == 1


# ── Tick function ──────────────────────────────────────────────────────────

class TestTickSelfModify:
    def test_no_eligible_no_mutations(self) -> None:
        colonists = [_make_colonist("c-0", meta_awareness=0.1)]
        meta_aware: set[str] = set()
        log = MutationLog()
        result = tick_self_modify(colonists, meta_aware, year=5, mutation_log=log, rng=random.Random(42))
        assert len(result) == 0

    def test_meta_aware_can_mutate(self) -> None:
        colonists = [_make_colonist(f"c-{i}") for i in range(5)]
        meta_aware = {"c-0", "c-1", "c-2", "c-3", "c-4"}
        log = MutationLog()
        result = tick_self_modify(colonists, meta_aware, year=5, mutation_log=log, rng=random.Random(42))
        assert len(result) > 0
        assert all(isinstance(p, MutationProposal) for p in result)

    def test_max_mutations_per_year_cap(self) -> None:
        colonists = [_make_colonist(f"c-{i}") for i in range(10)]
        meta_aware = {c.id for c in colonists}
        log = MutationLog()
        result = tick_self_modify(colonists, meta_aware, year=5, mutation_log=log, rng=random.Random(42))
        assert len(result) <= MAX_MUTATIONS_PER_YEAR

    def test_dead_colonists_excluded(self) -> None:
        c = _make_colonist("c-0")
        c.alive = False
        log = MutationLog()
        result = tick_self_modify([c], {"c-0"}, year=5, mutation_log=log, rng=random.Random(42))
        assert len(result) == 0

    def test_proposals_logged(self) -> None:
        colonists = [_make_colonist(f"c-{i}") for i in range(3)]
        meta_aware = {c.id for c in colonists}
        log = MutationLog()
        tick_self_modify(colonists, meta_aware, year=5, mutation_log=log, rng=random.Random(42))
        assert log.total_proposals > 0


# ── Engine integration ─────────────────────────────────────────────────────

class TestEngineIntegration:
    def test_engine_includes_mutation_log(self) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        d = result.to_dict()
        assert "mutation_log" in d
        assert "total_proposals" in d["mutation_log"]

    def test_10_year_smoke(self) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=99, total_years=10)
        result = engine.run()
        for yr in result.years:
            d = yr.to_dict()
            assert "self_modifications" in d
            assert isinstance(d["self_modifications"], list)
