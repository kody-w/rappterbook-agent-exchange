"""Tests for per-colonist comm vitals organ."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.comm_channels import (
    Channel, STATUS_VITAL, STATUS_FADING, STATUS_FLATLINED,
    STATUS_REVIVED, STATUS_DORMANT, STATUS_INACTIVE,
)
from src.mars100.comm_vitals import (
    LIVE_STATUSES,
    ColonistCommVital,
    compute_colonist_vitals,
    summarise,
)


def _ch(a, b, status=STATUS_VITAL, vitality=0.8, silence_streak=0):
    if a > b:
        a, b = b, a
    c = Channel(a=a, b=b, born_year=0, last_contact_year=0)
    c.status = status
    c.vitality = vitality
    c.silence_streak = silence_streak
    return c


def _channels_from(edges):
    out = {}
    for e in edges:
        a, b = e[0], e[1]
        status = e[2] if len(e) > 2 else STATUS_VITAL
        vit = e[3] if len(e) > 3 else 0.8
        silence = e[4] if len(e) > 4 else 0
        key = (a, b) if a < b else (b, a)
        out[key] = _ch(a, b, status, vit, silence)
    return out


# ----- empty / degenerate --------------------------------------------------

def test_no_colonists():
    assert compute_colonist_vitals([], {}) == []


def test_one_lone_colonist():
    vs = compute_colonist_vitals(["solo"], {})
    assert len(vs) == 1
    v = vs[0]
    assert v.channel_count == 0
    assert v.classification == "ghosted"
    assert v.urgency > 0


def test_active_colonist_with_only_inactive_peer():
    ch = _channels_from([("a", "b")])
    vs = compute_colonist_vitals(["a"], ch)
    assert vs[0].channel_count == 0


# ----- classification ------------------------------------------------------

def test_classification_healthy():
    ch = _channels_from([("a", "b"), ("a", "c"), ("a", "d")])
    vs = compute_colonist_vitals(["a", "b", "c", "d"], ch)
    a = next(v for v in vs if v.colonist_id == "a")
    assert a.vital_ratio == 1.0
    assert a.classification == "healthy"
    assert a.isolation_score == 0.0


def test_classification_ghosted_all_flatlined():
    ch = _channels_from([
        ("a", "b", STATUS_FLATLINED),
        ("a", "c", STATUS_FLATLINED),
    ])
    vs = compute_colonist_vitals(["a", "b", "c"], ch)
    a = next(v for v in vs if v.colonist_id == "a")
    assert a.vital_ratio == 0.0
    assert a.live_channels == 0
    assert a.flatlined_channels == 2
    assert a.classification == "ghosted"


def test_classification_isolated_mostly_dead():
    ch = _channels_from([
        ("a", "b", STATUS_VITAL),
        ("a", "c", STATUS_FLATLINED),
        ("a", "d", STATUS_FLATLINED),
        ("a", "e", STATUS_DORMANT),
    ])
    vs = compute_colonist_vitals(list("abcde"), ch)
    a = next(v for v in vs if v.colonist_id == "a")
    assert a.live_channels == 1
    assert a.vital_ratio == 0.25
    assert a.classification == "isolated"


def test_classification_strained_lifeline_with_gaps():
    ch = _channels_from([
        ("a", "b", STATUS_VITAL),       # b's only edge
        ("a", "c", STATUS_VITAL),
        ("a", "d", STATUS_FLATLINED),
        ("a", "e", STATUS_FLATLINED),
        ("c", "d"),                       # c has another live edge
    ])
    vs = compute_colonist_vitals(list("abcde"), ch)
    a = next(v for v in vs if v.colonist_id == "a")
    assert a.is_lifeline
    assert "b" in a.sole_partners
    assert a.vital_ratio == 0.5
    assert a.classification == "strained"


# ----- live statuses include vital/fading/revived --------------------------

def test_live_statuses_include_fading_and_revived():
    ch = _channels_from([
        ("a", "b", STATUS_FADING),
        ("a", "c", STATUS_REVIVED),
    ])
    vs = compute_colonist_vitals(["a", "b", "c"], ch)
    a = next(v for v in vs if v.colonist_id == "a")
    assert a.live_channels == 2
    assert a.vital_ratio == 1.0


def test_dormant_not_counted_as_live():
    ch = _channels_from([
        ("a", "b", STATUS_DORMANT),
        ("a", "c", STATUS_VITAL),
    ])
    vs = compute_colonist_vitals(["a", "b", "c"], ch)
    a = next(v for v in vs if v.colonist_id == "a")
    assert a.live_channels == 1
    assert a.vital_ratio == 0.5


# ----- lifeline / sole partner detection ----------------------------------

def test_sole_partner_symmetry():
    # Linear chain a -- b -- c.
    ch = _channels_from([("a", "b"), ("b", "c")])
    vs = compute_colonist_vitals(["a", "b", "c"], ch)
    by_id = {v.colonist_id: v for v in vs}
    # b is sole partner for both a and c.
    assert by_id["b"].sole_partners == ["a", "c"]
    assert by_id["b"].is_lifeline
    # a has only one peer, but b has multiple, so a is NOT b's lifeline.
    assert by_id["a"].sole_partners == []
    assert not by_id["a"].is_lifeline


def test_lifeline_only_when_peer_truly_alone():
    ch = _channels_from([("a", "b"), ("b", "c"), ("b", "d")])
    vs = compute_colonist_vitals(["a", "b", "c", "d"], ch)
    a = next(v for v in vs if v.colonist_id == "a")
    assert a.sole_partners == []


# ----- silence pressure ----------------------------------------------------

def test_silence_pressure_averages():
    ch = _channels_from([
        ("a", "b", STATUS_VITAL, 0.8, 4),
        ("a", "c", STATUS_FADING, 0.4, 8),
    ])
    vs = compute_colonist_vitals(["a", "b", "c"], ch)
    a = next(v for v in vs if v.colonist_id == "a")
    assert a.silence_pressure == 6.0


# ----- urgency invariants --------------------------------------------------

def test_urgency_bounded():
    ch = _channels_from([
        ("a", "b", STATUS_FLATLINED, 0.05, 50),
        ("a", "c", STATUS_FLATLINED, 0.05, 50),
        ("a", "d", STATUS_FLATLINED, 0.05, 50),
    ])
    vs = compute_colonist_vitals(list("abcd"), ch)
    for v in vs:
        assert 0.0 <= v.urgency <= 1.0
        assert 0.0 <= v.isolation_score <= 1.0
        assert 0.0 <= v.vital_ratio <= 1.0


def test_urgency_orders_loneliest_first():
    ch = _channels_from([
        ("a", "x", STATUS_VITAL),
        ("a", "y", STATUS_VITAL),
        ("lonely", "x", STATUS_FLATLINED),
        ("lonely", "y", STATUS_FLATLINED),
    ])
    vs = compute_colonist_vitals(["a", "x", "y", "lonely"], ch)
    assert vs[0].colonist_id == "lonely"


def test_lifeline_bonus_raises_urgency():
    ch_lifeline = _channels_from([
        ("a", "b", STATUS_VITAL),
        ("a", "c", STATUS_FLATLINED),
        ("a", "d", STATUS_FLATLINED),
    ])
    vs1 = compute_colonist_vitals(["a", "b", "c", "d"], ch_lifeline)
    a_lifeline = next(v for v in vs1 if v.colonist_id == "a")
    assert a_lifeline.is_lifeline

    ch_no = _channels_from([
        ("a", "b", STATUS_VITAL),
        ("a", "c", STATUS_FLATLINED),
        ("a", "d", STATUS_FLATLINED),
        ("b", "e", STATUS_VITAL),
    ])
    vs2 = compute_colonist_vitals(["a", "b", "c", "d", "e"], ch_no)
    a_no = next(v for v in vs2 if v.colonist_id == "a")
    assert not a_no.is_lifeline
    assert a_lifeline.urgency > a_no.urgency


# ----- summarise -----------------------------------------------------------

def test_summarise_empty():
    s = summarise([])
    assert s["total_colonists"] == 0
    assert s["mean_urgency"] == 0.0


def test_summarise_buckets():
    ch = _channels_from([
        ("a", "b", STATUS_VITAL), ("a", "c", STATUS_VITAL),
        ("lonely", "b", STATUS_FLATLINED),
        ("lonely", "c", STATUS_FLATLINED),
    ])
    vs = compute_colonist_vitals(["a", "b", "c", "lonely"], ch)
    s = summarise(vs)
    assert s["total_colonists"] == 4
    assert s["ghosted"] >= 1
    assert 0.0 <= s["mean_urgency"] <= 1.0


# ----- serialization -------------------------------------------------------

def test_to_dict_round_trip_via_json():
    ch = _channels_from([("a", "b"), ("c", "d", STATUS_FLATLINED)])
    vs = compute_colonist_vitals(["a", "b", "c", "d"], ch, names={"a": "Alice"})
    blob = json.dumps([v.to_dict() for v in vs])
    parsed = json.loads(blob)
    assert any(row["name"] == "Alice" for row in parsed)
    for row in parsed:
        assert set(row.keys()) >= {
            "colonist_id", "name", "channel_count", "live_channels",
            "flatlined_channels", "vital_ratio", "mean_vitality",
            "silence_pressure", "isolation_score", "sole_partners",
            "is_lifeline", "urgency", "classification",
        }


def test_names_default_to_id_when_missing():
    ch = _channels_from([("a", "b")])
    vs = compute_colonist_vitals(["a", "b"], ch)
    for v in vs:
        assert v.name == v.colonist_id


def test_deterministic_ordering():
    ch = _channels_from([
        ("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"),
        ("e", "a", STATUS_FLATLINED),
    ])
    vs1 = compute_colonist_vitals(list("abcde"), ch)
    vs2 = compute_colonist_vitals(list("abcde"), ch)
    assert [v.colonist_id for v in vs1] == [v.colonist_id for v in vs2]
    assert [v.urgency for v in vs1] == [v.urgency for v in vs2]
