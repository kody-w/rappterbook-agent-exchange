"""Tests for the colonist comm-vitals organ."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.mars100.comm_vitals import (
    ColonistCommVital, compute_colonist_vitals, summarise, revival_prompts,
    GHOSTED_RATIO, ISOLATED_RATIO, LIVE_STATUSES,
)


@dataclass
class FakeChannel:
    """Stand-in for src.mars100.comm_channels.Channel — only the fields
    the vitals organ actually reads."""
    status: str = "vital"
    vitality: float = 1.0
    silence_streak: int = 0


def _build(channels_spec: list) -> dict:
    """Build a {(a,b): FakeChannel} dict from a list of tuples.

    Each tuple is (a, b, status, vitality, silence_streak). Order of a,b
    is preserved as-is (the engine canonicalises elsewhere).
    """
    out = {}
    for a, b, status, vit, sil in channels_spec:
        out[(a, b)] = FakeChannel(status=status, vitality=vit, silence_streak=sil)
    return out


# ---------- structural / smoke ----------

def test_returns_one_per_active_colonist():
    channels = _build([("a", "b", "vital", 1.0, 0)])
    vitals = compute_colonist_vitals(["a", "b", "c"], channels)
    assert {v.colonist_id for v in vitals} == {"a", "b", "c"}
    assert len(vitals) == 3


def test_empty_inputs_yield_empty_result():
    assert compute_colonist_vitals([], {}) == []


def test_inactive_colonists_excluded():
    channels = _build([
        ("a", "b", "vital", 1.0, 0),
        ("a", "c", "vital", 1.0, 0),
    ])
    # c is inactive — channels involving c must not count for a.
    vitals = compute_colonist_vitals(["a", "b"], channels)
    a = next(v for v in vitals if v.colonist_id == "a")
    assert a.channel_count == 1  # only the a-b edge survived the active filter
    assert a.live_channels == 1


def test_names_default_to_id_when_missing():
    channels = _build([("a", "b", "vital", 1.0, 0)])
    vitals = compute_colonist_vitals(["a", "b"], channels)
    assert all(v.name == v.colonist_id for v in vitals)


def test_names_used_when_provided():
    channels = _build([("a", "b", "vital", 1.0, 0)])
    vitals = compute_colonist_vitals(
        ["a", "b"], channels, names={"a": "Alice"})
    a = next(v for v in vitals if v.colonist_id == "a")
    b = next(v for v in vitals if v.colonist_id == "b")
    assert a.name == "Alice"
    assert b.name == "b"


# ---------- counts and ratios ----------

def test_live_count_only_counts_live_statuses():
    channels = _build([
        ("a", "b", "vital", 1.0, 0),
        ("a", "c", "fading", 0.5, 2),
        ("a", "d", "revived", 0.9, 0),
        ("a", "e", "flatlined", 0.0, 12),
        ("a", "f", "dormant", 0.0, 50),
        ("a", "g", "inactive", 0.0, 99),
    ])
    vitals = compute_colonist_vitals(list("abcdefg"), channels)
    a = next(v for v in vitals if v.colonist_id == "a")
    assert a.channel_count == 6
    assert a.live_channels == 3            # vital + fading + revived
    assert a.flatlined_channels == 1


def test_vital_ratio_matches_live_over_total():
    channels = _build([
        ("a", "b", "vital", 1.0, 0),
        ("a", "c", "flatlined", 0.0, 12),
    ])
    vitals = compute_colonist_vitals(["a", "b", "c"], channels)
    a = next(v for v in vitals if v.colonist_id == "a")
    assert a.vital_ratio == pytest.approx(0.5)
    assert a.isolation_score == pytest.approx(0.5)


def test_mean_vitality_and_silence_pressure():
    channels = _build([
        ("a", "b", "vital", 1.0, 0),
        ("a", "c", "fading", 0.4, 6),
    ])
    vitals = compute_colonist_vitals(["a", "b", "c"], channels)
    a = next(v for v in vitals if v.colonist_id == "a")
    assert a.mean_vitality == pytest.approx(0.7)
    assert a.silence_pressure == pytest.approx(3.0)


# ---------- classification ----------

def test_no_channels_is_ghosted():
    vitals = compute_colonist_vitals(["lonely"], {})
    assert vitals[0].classification == "ghosted"
    assert vitals[0].channel_count == 0
    assert vitals[0].vital_ratio == 0.0


def test_all_flatlined_is_ghosted():
    channels = _build([("a", "b", "flatlined", 0.0, 20)])
    vitals = compute_colonist_vitals(["a", "b"], channels)
    a = next(v for v in vitals if v.colonist_id == "a")
    assert a.classification == "ghosted"


def test_low_vital_ratio_is_isolated():
    # 1 live out of 4 => ratio 0.25, which is > GHOSTED_RATIO but < ISOLATED_RATIO
    channels = _build([
        ("a", "b", "vital", 1.0, 0),
        ("a", "c", "flatlined", 0.0, 12),
        ("a", "d", "flatlined", 0.0, 12),
        ("a", "e", "flatlined", 0.0, 12),
    ])
    vitals = compute_colonist_vitals(list("abcde"), channels)
    a = next(v for v in vitals if v.colonist_id == "a")
    assert GHOSTED_RATIO < a.vital_ratio < ISOLATED_RATIO
    assert a.classification == "isolated"


def test_many_flatlines_is_strained():
    channels = _build([
        ("a", "b", "vital", 1.0, 0),
        ("a", "c", "vital", 1.0, 0),
        ("a", "d", "vital", 1.0, 0),
        ("a", "e", "vital", 1.0, 0),
        ("a", "f", "flatlined", 0.0, 12),
        ("a", "g", "flatlined", 0.0, 12),
        ("a", "h", "flatlined", 0.0, 12),
    ])
    vitals = compute_colonist_vitals(list("abcdefgh"), channels)
    a = next(v for v in vitals if v.colonist_id == "a")
    assert a.classification == "strained"


def test_healthy_baseline():
    channels = _build([
        ("a", "b", "vital", 1.0, 0),
        ("a", "c", "vital", 1.0, 0),
    ])
    vitals = compute_colonist_vitals(["a", "b", "c"], channels)
    a = next(v for v in vitals if v.colonist_id == "a")
    assert a.classification == "healthy"


# ---------- lifeline detection ----------

def test_lifeline_when_peer_has_only_us():
    # b has exactly one live edge (to a). a has multiple. a IS b's lifeline.
    channels = _build([
        ("a", "b", "vital", 1.0, 0),
        ("a", "c", "vital", 1.0, 0),
        ("b", "d", "flatlined", 0.0, 15),
    ])
    vitals = compute_colonist_vitals(list("abcd"), channels)
    a = next(v for v in vitals if v.colonist_id == "a")
    b = next(v for v in vitals if v.colonist_id == "b")
    assert a.is_lifeline is True
    assert "b" in a.sole_partners
    # b is NOT a's lifeline (a has another live edge to c)
    assert b.is_lifeline is False


def test_mutual_lifeline_when_pair_isolated_together():
    channels = _build([("a", "b", "vital", 1.0, 0)])
    vitals = compute_colonist_vitals(["a", "b"], channels)
    a = next(v for v in vitals if v.colonist_id == "a")
    b = next(v for v in vitals if v.colonist_id == "b")
    assert a.is_lifeline and b.is_lifeline
    assert a.sole_partners == ["b"]
    assert b.sole_partners == ["a"]


def test_not_lifeline_via_flatlined_edge():
    # b's only edge to a is flatlined, AND c has another live edge to d.
    # So neither b nor c depends solely on a => a is NOT a lifeline.
    channels = _build([
        ("a", "b", "flatlined", 0.0, 12),
        ("a", "c", "vital", 1.0, 0),
        ("c", "d", "vital", 1.0, 0),
    ])
    vitals = compute_colonist_vitals(["a", "b", "c", "d"], channels)
    a = next(v for v in vitals if v.colonist_id == "a")
    assert a.is_lifeline is False


# ---------- urgency ----------

def test_urgency_in_unit_interval():
    channels = _build([
        ("a", "b", "vital", 1.0, 0),
        ("a", "c", "flatlined", 0.0, 50),
    ])
    vitals = compute_colonist_vitals(["a", "b", "c"], channels)
    for v in vitals:
        assert 0.0 <= v.urgency <= 1.0


def test_healthy_colonist_has_low_urgency():
    channels = _build([
        ("a", "b", "vital", 1.0, 0),
        ("a", "c", "vital", 1.0, 0),
    ])
    vitals = compute_colonist_vitals(["a", "b", "c"], channels)
    a = next(v for v in vitals if v.colonist_id == "a")
    # No loneliness, no silence -> urgency = 0 + lifeline_bonus*True
    # a is lifeline because b and c each have only the edge to a as live.
    assert a.urgency == pytest.approx(0.15)


def test_ghosted_lifeline_combination_is_high_urgency():
    # a has 1 vital edge (to b) but 3 flatlined edges -> vital_ratio 0.25
    # b has only a -> a is lifeline. Plus high silence pressure.
    channels = _build([
        ("a", "b", "vital", 1.0, 0),
        ("a", "c", "flatlined", 0.0, 12),
        ("a", "d", "flatlined", 0.0, 12),
        ("a", "e", "flatlined", 0.0, 12),
    ])
    vitals = compute_colonist_vitals(list("abcde"), channels)
    a = next(v for v in vitals if v.colonist_id == "a")
    assert a.urgency > 0.6


def test_results_sorted_by_urgency_desc():
    channels = _build([
        ("a", "b", "vital", 1.0, 0),
        ("a", "c", "vital", 1.0, 0),
        ("d", "e", "flatlined", 0.0, 30),
    ])
    vitals = compute_colonist_vitals(list("abcde"), channels)
    urgencies = [v.urgency for v in vitals]
    assert urgencies == sorted(urgencies, reverse=True)


# ---------- summary ----------

def test_summary_empty():
    s = summarise([])
    assert s["total_colonists"] == 0
    assert s["mean_urgency"] == 0.0
    assert s["lifelines"] == 0


def test_summary_counts_buckets():
    channels = _build([
        ("a", "b", "vital", 1.0, 0),
        ("a", "c", "vital", 1.0, 0),
        ("d", "e", "flatlined", 0.0, 20),
    ])
    vitals = compute_colonist_vitals(list("abcde"), channels)
    s = summarise(vitals)
    assert s["total_colonists"] == 5
    assert (s["healthy"] + s["strained"]
            + s["isolated"] + s["ghosted"]) == 5
    assert 0.0 <= s["mean_urgency"] <= 1.0
    assert s["max_urgency"] >= s["mean_urgency"]


# ---------- revival prompts ----------

def test_revival_prompts_skip_healthy():
    channels = _build([
        ("a", "b", "vital", 1.0, 0),
        ("a", "c", "vital", 1.0, 0),
    ])
    vitals = compute_colonist_vitals(["a", "b", "c"], channels)
    prompts = revival_prompts(vitals, year=10)
    for p in prompts:
        assert p["classification"] != "healthy"


def test_revival_prompts_capped_at_max():
    # Many ghosted colonists, but max_prompts=3.
    channels = _build([(f"x{i}", f"y{i}", "flatlined", 0.0, 30)
                       for i in range(10)])
    ids = [f"x{i}" for i in range(10)] + [f"y{i}" for i in range(10)]
    vitals = compute_colonist_vitals(ids, channels)
    prompts = revival_prompts(vitals, year=42, max_prompts=3)
    assert len(prompts) <= 3
    for p in prompts:
        assert p["year"] == 42
        assert "suggested_action" in p
        assert "text" in p


def test_revival_prompt_for_lifeline_mentions_dependents():
    channels = _build([
        ("a", "b", "vital", 0.3, 5),
        ("a", "c", "flatlined", 0.0, 12),
        ("a", "d", "flatlined", 0.0, 12),
    ])
    vitals = compute_colonist_vitals(list("abcd"), channels)
    prompts = revival_prompts(vitals, year=20, max_prompts=10)
    a_prompts = [p for p in prompts if p["colonist_id"] == "a"]
    assert a_prompts, "expected a prompt for the strained lifeline"
    p = a_prompts[0]
    assert p["is_lifeline"] is True
    assert "b" in p["sole_partners"]


# ---------- determinism / invariants ----------

def test_deterministic_across_calls():
    channels = _build([
        ("a", "b", "vital", 1.0, 0),
        ("a", "c", "fading", 0.5, 4),
        ("b", "d", "flatlined", 0.0, 20),
    ])
    out1 = [v.to_dict() for v in compute_colonist_vitals(list("abcd"), channels)]
    out2 = [v.to_dict() for v in compute_colonist_vitals(list("abcd"), channels)]
    assert out1 == out2


def test_live_statuses_frozen_contract():
    # The exported constant is the contract — anything else is not "live".
    assert LIVE_STATUSES == frozenset({"vital", "fading", "revived"})


def test_to_dict_roundtrip_fields():
    v = ColonistCommVital(colonist_id="x", name="X")
    d = v.to_dict()
    for key in ("colonist_id", "name", "channel_count", "live_channels",
                "flatlined_channels", "vital_ratio", "mean_vitality",
                "silence_pressure", "isolation_score", "sole_partners",
                "is_lifeline", "urgency", "classification"):
        assert key in d


# ---------- smoke against real engine ----------

def test_smoke_against_real_engine():
    """End-to-end: run the real Mars-100 engine for a few years and
    confirm the vitals organ produces a non-empty, well-formed result."""
    from src.mars100 import Mars100Engine
    engine = Mars100Engine(seed=7, total_years=15)
    engine.run()
    active = [c.id for c in engine.colonists if c.is_active()]
    names = {c.id: c.name for c in engine.colonists}
    vitals = compute_colonist_vitals(active, engine.comm_channels.channels, names)
    # Every active colonist gets an entry.
    assert len(vitals) == len(active)
    # Every classification is one of the four buckets.
    for v in vitals:
        assert v.classification in {"healthy", "strained", "isolated", "ghosted"}
        assert 0.0 <= v.urgency <= 1.0
    s = summarise(vitals)
    assert s["total_colonists"] == len(active)
    assert (s["healthy"] + s["strained"]
            + s["isolated"] + s["ghosted"]) == len(active)
