"""Tests for the rolling-history feature in scripts/channel_health.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "channel_health.py"

sys.path.insert(0, str(REPO))
from scripts import channel_health as ch  # noqa: E402


# ---------- unit: pure helpers ---------------------------------------------

def _mk_entry(t, health, flat, total=10, vital=5, fading=2,
              revived=1, dormant=1, inactive=1, prompts=0):
    return {
        "generated": f"2026-01-{t:02d}T00:00:00+00:00",
        "year_snapshot": t,
        "overall_health": health,
        "total": total, "vital": vital, "fading": fading,
        "flatlined": flat, "revived": revived,
        "dormant": dormant, "inactive": inactive,
        "revival_prompt_count": prompts,
    }


def test_compute_trend_unknown_when_single_entry():
    trend = ch._compute_trend([_mk_entry(1, 0.5, 0)])
    assert trend["direction"] == "unknown"
    assert trend["delta_health"] == 0.0
    assert trend["delta_flatlined"] == 0
    assert trend["baseline_count"] == 0


def test_compute_trend_rising():
    # baseline ~0.40, latest 0.80 — clear rise + flatlined drop
    hist = [_mk_entry(i, 0.40, 3) for i in range(1, 12)]
    hist.append(_mk_entry(12, 0.80, 1))
    trend = ch._compute_trend(hist)
    assert trend["direction"] == "rising"
    assert trend["delta_health"] > 0.02
    assert trend["delta_flatlined"] <= 0


def test_compute_trend_falling_on_health_drop():
    hist = [_mk_entry(i, 0.80, 0) for i in range(1, 12)]
    hist.append(_mk_entry(12, 0.50, 0))
    trend = ch._compute_trend(hist)
    assert trend["direction"] == "falling"
    assert trend["delta_health"] < -0.02


def test_compute_trend_falling_on_flatlined_spike():
    hist = [_mk_entry(i, 0.70, 0) for i in range(1, 12)]
    hist.append(_mk_entry(12, 0.70, 5))
    trend = ch._compute_trend(hist)
    assert trend["direction"] == "falling"
    assert trend["delta_flatlined"] >= 2


def test_compute_trend_stable_within_noise():
    hist = [_mk_entry(i, 0.70, 1) for i in range(1, 12)]
    hist.append(_mk_entry(12, 0.71, 1))
    trend = ch._compute_trend(hist)
    assert trend["direction"] == "stable"


def test_history_snapshot_shape():
    fake_report = {
        "_meta": {"generated": "2026-01-01T00:00:00+00:00", "year_snapshot": 30},
        "overall_health_score": 0.42,
        "summary": {"vital": 4, "fading": 3, "flatlined": 2,
                     "revived": 1, "dormant": 0, "inactive": 5,
                     "total": 15},
        "revival_prompts": [{"text": "x"}] * 3,
    }
    snap = ch._history_snapshot(fake_report)
    for k in ("generated", "year_snapshot", "overall_health", "total",
              "vital", "fading", "flatlined", "revived", "dormant",
              "inactive", "revival_prompt_count"):
        assert k in snap
    assert snap["flatlined"] == 2
    assert snap["revival_prompt_count"] == 3


def test_append_history_creates_file_and_caps(tmp_path):
    hp = tmp_path / "channel_health_history.json"
    # cap at 5 — push 7, expect only last 5 retained
    for i in range(1, 8):
        payload = ch.append_history(
            _mk_entry(i, 0.5 + i * 0.01, i % 3),
            hp, max_entries=5)
    assert hp.exists()
    assert len(payload["entries"]) == 5
    assert payload["entries"][0]["year_snapshot"] == 3
    assert payload["entries"][-1]["year_snapshot"] == 7
    assert payload["_meta"]["max_entries"] == 5
    assert payload["_meta"]["kind"] == "history"


def test_append_history_recovers_from_corrupt_file(tmp_path):
    hp = tmp_path / "history.json"
    hp.write_text("not-json{{{")
    payload = ch.append_history(_mk_entry(1, 0.5, 0), hp)
    assert len(payload["entries"]) == 1


def test_append_history_appends_to_existing(tmp_path):
    hp = tmp_path / "h.json"
    ch.append_history(_mk_entry(1, 0.5, 0), hp)
    payload = ch.append_history(_mk_entry(2, 0.6, 0), hp)
    assert [e["year_snapshot"] for e in payload["entries"]] == [1, 2]
    assert payload["trend"]["direction"] in ("rising", "stable", "falling",
                                              "unknown")


def test_append_history_writes_valid_json(tmp_path):
    hp = tmp_path / "h.json"
    ch.append_history(_mk_entry(1, 0.5, 0), hp)
    data = json.loads(hp.read_text())
    assert "entries" in data and "trend" in data and "_meta" in data


# ---------- integration: subprocess writes history file --------------------

def _run(state_dir: Path, docs_dir: Path, *, years=6, seed=11,
          extra=None):
    args = [sys.executable, str(SCRIPT),
            "--years", str(years), "--seed", str(seed),
            "--state-dir", str(state_dir),
            "--docs-dir", str(docs_dir),
            "--quiet"]
    if extra: args.extend(extra)
    return subprocess.run(args, cwd=str(REPO), capture_output=True,
                           text=True, timeout=300)


def test_script_writes_history_by_default(tmp_path):
    state = tmp_path / "state"
    docs = tmp_path / "docs" / "mars-100"
    result = _run(state, docs)
    assert result.returncode == 0, result.stderr
    hist = state / "channel_health_history.json"
    assert hist.exists()
    payload = json.loads(hist.read_text())
    assert payload["_meta"]["organ"] == "comm-channels"
    assert len(payload["entries"]) == 1
    # mirrored to docs/
    assert (docs / "channel_health_history.json").exists()


def test_history_appends_across_runs(tmp_path):
    state = tmp_path / "state"
    docs = tmp_path / "docs" / "mars-100"
    _run(state, docs, seed=1)
    _run(state, docs, seed=2)
    _run(state, docs, seed=3)
    payload = json.loads((state / "channel_health_history.json").read_text())
    assert len(payload["entries"]) == 3
    # after 3 entries, trend should not be 'unknown' anymore
    assert payload["trend"]["direction"] in ("rising", "falling", "stable")
    assert payload["trend"]["baseline_count"] >= 1


def test_no_history_flag_skips_file(tmp_path):
    state = tmp_path / "state"
    docs = tmp_path / "docs"
    result = _run(state, docs, extra=["--no-history", "--no-docs"])
    assert result.returncode == 0, result.stderr
    assert not (state / "channel_health_history.json").exists()


def test_custom_history_path(tmp_path):
    state = tmp_path / "state"
    docs = tmp_path / "docs"
    custom = tmp_path / "custom" / "h.json"
    result = _run(state, docs, extra=["--no-docs", "--history-path", str(custom)])
    assert result.returncode == 0, result.stderr
    assert custom.exists()
    assert not (state / "channel_health_history.json").exists()


def test_dashboard_html_published():
    page = REPO / "docs" / "mars-100" / "channels.html"
    assert page.exists()
    txt = page.read_text()
    assert "channel_health.json" in txt
    assert "channel_health_history.json" in txt
    assert "Comm Channels Health" in txt


def test_workflow_wires_channel_health():
    wf = (REPO / ".github" / "workflows" / "evolve.yml").read_text()
    assert "scripts/channel_health.py" in wf


# ---------- property invariant ---------------------------------------------

@pytest.mark.parametrize("seed", [1, 7, 42])
def test_overall_health_in_unit_interval(tmp_path, seed):
    state = tmp_path / "state"
    docs = tmp_path / "docs"
    res = _run(state, docs, years=10, seed=seed, extra=["--no-docs"])
    assert res.returncode == 0, res.stderr
    rep = json.loads((state / "channel_health.json").read_text())
    assert 0.0 <= rep["overall_health_score"] <= 1.0
    # summary counts must reconcile with total
    s = rep["summary"]
    statuses = ("vital", "fading", "flatlined", "revived",
                "dormant", "inactive")
    assert sum(s.get(k, 0) for k in statuses) == s["total"]
