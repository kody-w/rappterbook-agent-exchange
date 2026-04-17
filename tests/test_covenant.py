"""Tests for the covenant engine — persistent executable LisPy laws."""
from __future__ import annotations

import random

import pytest

from src.mars100.covenant import (
    COVENANT_TEMPLATES,
    GLOBAL_DELTA_CAP,
    MAX_ACTIVE_COVENANTS,
    MAX_CONSECUTIVE_ERRORS,
    MAX_DELTA_PER_COVENANT,
    RESOURCE_NAMES,
    Covenant,
    CovenantRegistry,
    _clamp_deltas,
    _parse_deltas,
    draft_covenant,
    execute_covenant,
    tick_covenants,
    vote_covenant,
)


# ── Helpers ───────────────────────────────────────────────────────

def _base_resources() -> dict[str, float]:
    return {"food": 0.5, "water": 0.6, "power": 0.7, "air": 0.8, "medicine": 0.4}


def _low_resources() -> dict[str, float]:
    return {"food": 0.15, "water": 0.2, "power": 0.1, "air": 0.15, "medicine": 0.1}


def _make_covenant(
    cov_id: str = "cov-y10-0",
    expression: str = "(list (list 'food 0.02))",
    year: int = 10,
    **kwargs,
) -> Covenant:
    return Covenant(
        id=cov_id, author_id="ares", year_enacted=year,
        expression=expression, description="test covenant", **kwargs,
    )


# ── Serialization ────────────────────────────────────────────────

class TestCovenantSerialization:
    def test_round_trip(self):
        cov = _make_covenant()
        d = cov.to_dict()
        restored = Covenant.from_dict(d)
        assert restored.id == cov.id
        assert restored.expression == cov.expression
        assert restored.year_enacted == cov.year_enacted
        assert restored.active == cov.active

    def test_sunset_field_preserved(self):
        cov = _make_covenant(sunset_year=50)
        d = cov.to_dict()
        assert d["sunset_year"] == 50
        restored = Covenant.from_dict(d)
        assert restored.sunset_year == 50

    def test_revoked_field(self):
        cov = _make_covenant()
        cov.active = False
        cov.revoked_year = 25
        d = cov.to_dict()
        assert d["revoked_year"] == 25
        assert d["active"] is False


# ── Registry ─────────────────────────────────────────────────────

class TestCovenantRegistry:
    def test_enact_adds_covenant(self):
        reg = CovenantRegistry()
        assert reg.enact(_make_covenant()) is True
        assert reg.active_count() == 1

    def test_max_active_limit(self):
        reg = CovenantRegistry()
        for i in range(MAX_ACTIVE_COVENANTS):
            assert reg.enact(_make_covenant(cov_id=f"cov-{i}")) is True
        assert reg.enact(_make_covenant(cov_id="overflow")) is False

    def test_revoke_frees_slot(self):
        reg = CovenantRegistry()
        for i in range(MAX_ACTIVE_COVENANTS):
            reg.enact(_make_covenant(cov_id=f"cov-{i}"))
        assert reg.can_enact() is False
        reg.revoke("cov-0", year=20)
        assert reg.can_enact() is True

    def test_check_sunsets(self):
        reg = CovenantRegistry()
        cov = _make_covenant(sunset_year=30)
        reg.enact(cov)
        expired = reg.check_sunsets(31)
        assert cov.id in expired
        assert reg.active_count() == 0

    def test_sunset_not_triggered_before_year(self):
        reg = CovenantRegistry()
        cov = _make_covenant(sunset_year=30)
        reg.enact(cov)
        assert len(reg.check_sunsets(29)) == 0
        assert reg.active_count() == 1

    def test_cooldown(self):
        reg = CovenantRegistry(last_draft_year=10)
        assert reg.can_draft(11) is False
        assert reg.can_draft(13) is True

    def test_registry_round_trip(self):
        reg = CovenantRegistry(last_draft_year=5)
        reg.enact(_make_covenant(cov_id="cov-a"))
        reg.enact(_make_covenant(cov_id="cov-b"))
        d = reg.to_dict()
        restored = CovenantRegistry.from_dict(d)
        assert restored.last_draft_year == 5
        assert len(restored.covenants) == 2

    def test_suspend(self):
        reg = CovenantRegistry()
        cov = _make_covenant()
        reg.enact(cov)
        assert reg.suspend(cov.id) is True
        assert cov.active is False
        assert reg.active_count() == 0


# ── Delta parsing & clamping ─────────────────────────────────────

class TestDeltaParsing:
    def test_nil_returns_empty(self):
        assert _parse_deltas(None) == {}

    def test_scalar_returns_tagged(self):
        assert _parse_deltas(0.5) == {"_scalar": 0.5}

    def test_list_of_pairs(self):
        result = _parse_deltas([["food", 0.03], ["water", -0.01]])
        assert result == {"food": 0.03, "water": -0.01}

    def test_invalid_resource_ignored(self):
        result = _parse_deltas([["food", 0.03], ["gold", 0.5]])
        assert "gold" not in result
        assert result["food"] == 0.03

    def test_malformed_items_ignored(self):
        result = _parse_deltas([["food", 0.03], ["water"], 42])
        assert result == {"food": 0.03}

    def test_non_list_non_number(self):
        assert _parse_deltas("hello") == {}

    def test_empty_list(self):
        assert _parse_deltas([]) == {}


class TestDeltaClamping:
    def test_within_bounds_unchanged(self):
        deltas = {"food": 0.02, "water": -0.01}
        assert _clamp_deltas(deltas) == deltas

    def test_exceeding_positive(self):
        assert _clamp_deltas({"food": 0.5})["food"] == MAX_DELTA_PER_COVENANT

    def test_exceeding_negative(self):
        assert _clamp_deltas({"water": -0.5})["water"] == -MAX_DELTA_PER_COVENANT


# ── Covenant execution ───────────────────────────────────────────

class TestCovenantExecution:
    def test_simple_covenant_returns_deltas(self):
        cov = _make_covenant(expression="(list (list 'food 0.03))")
        result = execute_covenant(cov, _base_resources(), 10, 20, "anarchy")
        assert result["error"] is None
        assert result["deltas"]["food"] == pytest.approx(0.03)

    def test_conditional_covenant_triggers(self):
        cov = _make_covenant(
            expression="(if (< food 0.3) (list (list 'food 0.03)) nil)"
        )
        result = execute_covenant(cov, _low_resources(), 10, 20, "anarchy")
        assert result["error"] is None
        assert result["deltas"].get("food", 0) == pytest.approx(0.03)

    def test_conditional_covenant_no_trigger(self):
        cov = _make_covenant(
            expression="(if (< food 0.3) (list (list 'food 0.03)) nil)"
        )
        result = execute_covenant(cov, _base_resources(), 10, 20, "anarchy")
        assert result["error"] is None
        assert result["deltas"] == {}

    def test_multi_resource_covenant(self):
        cov = _make_covenant(
            expression="(list (list 'air 0.04) (list 'power -0.01))"
        )
        result = execute_covenant(cov, _base_resources(), 10, 20, "anarchy")
        assert result["deltas"]["air"] == pytest.approx(0.04)
        assert result["deltas"]["power"] == pytest.approx(-0.01)

    def test_clamping_applied(self):
        cov = _make_covenant(expression="(list (list 'food 0.99))")
        result = execute_covenant(cov, _base_resources(), 10, 20, "anarchy")
        assert result["deltas"]["food"] == MAX_DELTA_PER_COVENANT

    def test_syntax_error_returns_error(self):
        cov = _make_covenant(expression="(if (< food")
        result = execute_covenant(cov, _base_resources(), 10, 20, "anarchy")
        assert result["error"] is not None
        assert result["deltas"] == {}

    def test_budget_exceeded_returns_error(self):
        cov = _make_covenant(
            expression="(let ((f (lambda (x) (f (+ x 1))))) (f 0))"
        )
        result = execute_covenant(cov, _base_resources(), 10, 20, "anarchy")
        assert result["error"] is not None

    def test_bindings_available(self):
        cov = _make_covenant(
            expression="(if (> population 5) (list (list 'food 0.01)) nil)"
        )
        result = execute_covenant(cov, _base_resources(), 10, 20, "anarchy")
        assert result["deltas"].get("food") == pytest.approx(0.01)

    def test_year_binding(self):
        cov = _make_covenant(
            expression="(if (> year 10) (list (list 'power 0.01)) nil)"
        )
        result = execute_covenant(cov, _base_resources(), 5, 20, "anarchy")
        assert result["deltas"].get("power") == pytest.approx(0.01)

    def test_nil_return_no_effect(self):
        cov = _make_covenant(expression="nil")
        result = execute_covenant(cov, _base_resources(), 10, 20, "anarchy")
        assert result["deltas"] == {}
        assert result["error"] is None

    def test_resource_bindings_accurate(self):
        """Covenant sees actual resource values, not defaults."""
        resources = {"food": 0.99, "water": 0.01, "power": 0.5, "air": 0.5, "medicine": 0.5}
        cov = _make_covenant(expression="(if (> food 0.9) (list (list 'food -0.01)) nil)")
        result = execute_covenant(cov, resources, 5, 10, "anarchy")
        assert result["deltas"].get("food") == pytest.approx(-0.01)


# ── Tick covenants (batch execution) ─────────────────────────────

class TestTickCovenants:
    def test_empty_registry(self):
        reg = CovenantRegistry()
        result = tick_covenants(reg, _base_resources(), 10, 20, "anarchy")
        assert result["aggregate_deltas"] == {n: 0.0 for n in RESOURCE_NAMES}
        assert result["executions"] == []

    def test_single_covenant_applies(self):
        reg = CovenantRegistry()
        reg.enact(_make_covenant(expression="(list (list 'food 0.02))"))
        result = tick_covenants(reg, _base_resources(), 10, 20, "anarchy")
        assert result["aggregate_deltas"]["food"] == pytest.approx(0.02)

    def test_multiple_covenants_aggregate(self):
        reg = CovenantRegistry()
        reg.enact(_make_covenant(cov_id="a", expression="(list (list 'food 0.02))"))
        reg.enact(_make_covenant(cov_id="b", expression="(list (list 'food 0.03))"))
        result = tick_covenants(reg, _base_resources(), 10, 20, "anarchy")
        assert result["aggregate_deltas"]["food"] == pytest.approx(0.05)

    def test_global_clamp_limits_aggregate(self):
        reg = CovenantRegistry()
        for i in range(5):
            reg.enact(_make_covenant(cov_id=f"c-{i}", expression="(list (list 'food 0.05))"))
        result = tick_covenants(reg, _base_resources(), 10, 20, "anarchy")
        assert result["aggregate_deltas"]["food"] <= GLOBAL_DELTA_CAP

    def test_error_increments_consecutive_count(self):
        reg = CovenantRegistry()
        cov = _make_covenant(expression="(undefined-var)")
        reg.enact(cov)
        tick_covenants(reg, _base_resources(), 10, 20, "anarchy")
        assert cov.consecutive_errors == 1

    def test_auto_suspension_after_errors(self):
        reg = CovenantRegistry()
        cov = _make_covenant(expression="(undefined-var)")
        reg.enact(cov)
        for year in range(20, 20 + MAX_CONSECUTIVE_ERRORS):
            tick_covenants(reg, _base_resources(), 10, year, "anarchy")
        assert cov.active is False

    def test_success_resets_error_count(self):
        reg = CovenantRegistry()
        cov = _make_covenant(expression="(list (list 'food 0.01))")
        reg.enact(cov)
        cov.consecutive_errors = 2
        tick_covenants(reg, _base_resources(), 10, 20, "anarchy")
        assert cov.consecutive_errors == 0

    def test_expired_covenants_reported(self):
        reg = CovenantRegistry()
        cov = _make_covenant(sunset_year=19)
        reg.enact(cov)
        result = tick_covenants(reg, _base_resources(), 10, 20, "anarchy")
        assert cov.id in result["expired"]

    def test_revoked_covenant_not_executed(self):
        reg = CovenantRegistry()
        cov = _make_covenant(expression="(list (list 'food 0.05))")
        reg.enact(cov)
        reg.revoke(cov.id, year=15)
        result = tick_covenants(reg, _base_resources(), 10, 20, "anarchy")
        assert result["aggregate_deltas"]["food"] == 0.0

    def test_execution_log_capped(self):
        reg = CovenantRegistry()
        cov = _make_covenant(expression="(list (list 'food 0.01))")
        reg.enact(cov)
        for year in range(100):
            tick_covenants(reg, _base_resources(), 10, year, "anarchy")
        assert len(cov.execution_log) <= 20


# ── Drafting ─────────────────────────────────────────────────────

class TestDrafting:
    def test_draft_empathy(self):
        reg = CovenantRegistry()
        cov = draft_covenant("ares", "empathy", 10, reg, random.Random(42))
        assert cov is not None
        assert "food" in cov.expression or "medicine" in cov.expression

    def test_draft_paranoia(self):
        reg = CovenantRegistry()
        cov = draft_covenant("selene", "paranoia", 10, reg, random.Random(42))
        assert cov is not None
        assert "air" in cov.expression or "water" in cov.expression

    def test_draft_respects_cooldown(self):
        reg = CovenantRegistry(last_draft_year=8)
        assert draft_covenant("ares", "empathy", 9, reg, random.Random(42)) is None

    def test_draft_respects_active_cap(self):
        reg = CovenantRegistry()
        for i in range(MAX_ACTIVE_COVENANTS):
            reg.enact(_make_covenant(cov_id=f"c-{i}"))
        assert draft_covenant("ares", "empathy", 20, reg, random.Random(42)) is None

    def test_duplicate_suppression(self):
        reg = CovenantRegistry()
        cov1 = draft_covenant("ares", "empathy", 10, reg, random.Random(42))
        assert cov1 is not None
        reg.enact(cov1)
        # Same seed, same stat → same expression → suppressed
        cov2 = draft_covenant("iris", "empathy", 15, reg, random.Random(42))
        assert cov2 is None

    def test_all_stat_templates_valid(self):
        """Every template expression must be valid LisPy."""
        for stat, templates in COVENANT_TEMPLATES.items():
            for expr, desc in templates:
                cov = _make_covenant(expression=expr)
                result = execute_covenant(cov, _base_resources(), 10, 20, "anarchy")
                assert result["error"] is None, f"Template {stat}/{desc} failed: {result['error']}"

    def test_sunset_year_assigned(self):
        reg = CovenantRegistry()
        cov = draft_covenant("ares", "resolve", 10, reg, random.Random(42))
        assert cov is not None
        assert cov.sunset_year is not None
        assert cov.sunset_year > cov.year_enacted

    def test_unknown_stat_uses_fallback(self):
        reg = CovenantRegistry()
        cov = draft_covenant("x", "unknown_stat", 10, reg, random.Random(42))
        assert cov is not None  # falls back to "resolve" templates


# ── Voting ───────────────────────────────────────────────────────

class TestVoting:
    def test_anarchy_simple_majority(self):
        cov = _make_covenant()
        trust = {"b": 0.8, "c": 0.8}
        result = vote_covenant(cov, ["ares", "b", "c"], "anarchy", None, [], trust, random.Random(42))
        assert isinstance(result, bool)

    def test_dictator_veto(self):
        cov = _make_covenant()
        trust = {"dictator": 0.1, "b": 0.9, "c": 0.9}
        result = vote_covenant(
            cov, ["ares", "dictator", "b", "c"],
            "dictator", "dictator", [], trust, random.Random(42),
        )
        assert result is False

    def test_consensus_high_threshold(self):
        trust = {"b": 0.3, "c": 0.3, "d": 0.3, "e": 0.3}
        results = [
            vote_covenant(
                _make_covenant(cov_id=f"c-{i}"),
                ["ares", "b", "c", "d", "e"],
                "consensus", None, [], trust, random.Random(i),
            )
            for i in range(20)
        ]
        pass_rate = sum(results) / len(results)
        assert pass_rate < 0.5  # low trust + 75% threshold → mostly fails

    def test_empty_voters(self):
        cov = _make_covenant()
        assert vote_covenant(cov, [], "anarchy", None, [], {}, random.Random(42)) is False

    def test_sole_author(self):
        cov = _make_covenant()
        assert vote_covenant(cov, ["ares"], "anarchy", None, [], {}, random.Random(42)) is True

    def test_council_approval(self):
        cov = _make_covenant()
        trust = {"cA": 0.9, "cB": 0.9, "cC": 0.9, "d": 0.5}
        result = vote_covenant(
            cov, ["ares", "cA", "cB", "cC", "d"],
            "council", None, ["cA", "cB", "cC"], trust, random.Random(42),
        )
        assert result is True  # all council members have high trust


# ── Property-based invariants ────────────────────────────────────

class TestInvariants:
    def test_deltas_always_bounded(self):
        """No matter the expression, per-covenant deltas are clamped."""
        expressions = [
            "(list (list 'food 999))",
            "(list (list 'water -999))",
            "(list (list 'food 0.01) (list 'water 0.01) (list 'power 0.01) (list 'air 0.01) (list 'medicine 0.01))",
        ]
        for expr in expressions:
            cov = _make_covenant(expression=expr)
            result = execute_covenant(cov, _base_resources(), 10, 20, "anarchy")
            for delta in result["deltas"].values():
                assert -MAX_DELTA_PER_COVENANT <= delta <= MAX_DELTA_PER_COVENANT

    def test_aggregate_deltas_bounded(self):
        """Aggregate deltas from batch execution are globally bounded."""
        reg = CovenantRegistry()
        for i in range(MAX_ACTIVE_COVENANTS):
            reg.enact(_make_covenant(
                cov_id=f"c-{i}",
                expression="(list (list 'food 0.05) (list 'water -0.05))",
            ))
        result = tick_covenants(reg, _base_resources(), 10, 20, "anarchy")
        for val in result["aggregate_deltas"].values():
            assert -GLOBAL_DELTA_CAP <= val <= GLOBAL_DELTA_CAP

    def test_execution_never_crashes(self):
        """Covenant execution handles garbage input gracefully."""
        garbage = [
            "", "nil", "42", "(+ 1 2)", "(/ 1 0)",
            "(list (list 'food 'not-a-number))",
            "(let ((x 1)) (let ((y 2)) (+ x y)))",
        ]
        for expr in garbage:
            cov = _make_covenant(expression=expr)
            result = execute_covenant(cov, _base_resources(), 10, 20, "anarchy")
            assert isinstance(result["deltas"], dict)

    def test_registry_active_count_invariant(self):
        """Active count matches actual active covenants."""
        reg = CovenantRegistry()
        for i in range(MAX_ACTIVE_COVENANTS):
            reg.enact(_make_covenant(cov_id=f"c-{i}", sunset_year=20 + i))
        reg.revoke("c-0", year=15)
        reg.check_sunsets(21)
        expected = sum(1 for c in reg.covenants if c.active)
        assert reg.active_count() == expected

    def test_resources_finite_after_tick(self):
        """After applying covenant deltas, values stay finite."""
        reg = CovenantRegistry()
        reg.enact(_make_covenant(expression="(list (list 'food 0.05))"))
        resources = _base_resources()
        result = tick_covenants(reg, resources, 10, 20, "anarchy")
        for name, delta in result["aggregate_deltas"].items():
            new_val = resources.get(name, 0.5) + delta
            assert new_val == new_val  # NaN check
            assert abs(new_val) < 100  # sanity bound
