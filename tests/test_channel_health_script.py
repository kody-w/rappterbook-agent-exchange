"""Smoke test for scripts/channel_health.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "channel_health.py"


def _run(state_dir: Path, docs_dir: Path, years: int, seed: int,
          extra_args=None):
    args = [sys.executable, str(SCRIPT),
            "--years", str(years), "--seed", str(seed),
            "--state-dir", str(state_dir),
            "--docs-dir", str(docs_dir),
            "--quiet"]
    if extra_args:
        args.extend(extra_args)
    return subprocess.run(args, cwd=str(REPO), capture_output=True,
                           text=True, timeout=300)


def test_script_exists_and_executable():
    assert SCRIPT.exists()
    assert SCRIPT.stat().st_mode & 0o111


def test_runs_and_writes_report(tmp_path):
    state_dir = tmp_path / "state"
    docs_dir = tmp_path / "docs" / "mars-100"
    result = _run(state_dir, docs_dir, years=8, seed=1)
    assert result.returncode == 0, result.stderr
    out = state_dir / "channel_health.json"
    assert out.exists()
    report = json.loads(out.read_text())

    for k in ("_meta", "summary", "overall_health_score",
              "flatlined_channels", "fading_channels",
              "channels", "revival_prompts"):
        assert k in report

    assert report["_meta"]["organ"] == "comm-channels"
    assert report["_meta"]["flatline_threshold_years"] >= 1
    assert isinstance(report["channels"], list)
    assert report["summary"]["total"] == len(report["channels"])
    assert 0.0 <= report["overall_health_score"] <= 1.0

    if report["channels"]:
        sample = report["channels"][0]
        for f in ("a", "b", "name_a", "name_b", "vitality",
                  "status", "silence_streak"):
            assert f in sample

    docs_out = docs_dir / "channel_health.json"
    assert docs_out.exists()


def test_no_docs_flag(tmp_path):
    state_dir = tmp_path / "state"
    docs_dir = tmp_path / "docs"
    result = _run(state_dir, docs_dir, years=5, seed=3,
                   extra_args=["--no-docs"])
    assert result.returncode == 0, result.stderr
    assert (state_dir / "channel_health.json").exists()
    assert not (docs_dir / "mars-100" / "channel_health.json").exists()


def test_summary_counts_consistent(tmp_path):
    state_dir = tmp_path / "state"
    docs_dir = tmp_path / "docs"
    result = _run(state_dir, docs_dir, years=12, seed=9,
                   extra_args=["--no-docs"])
    assert result.returncode == 0, result.stderr
    report = json.loads((state_dir / "channel_health.json").read_text())
    s = report["summary"]
    statuses = ("vital", "fading", "flatlined", "revived",
                "dormant", "inactive")
    assert sum(s.get(k, 0) for k in statuses) == s["total"]
    assert report["flatlined_count"] == len(report["flatlined_channels"])
    assert report["fading_count"] == len(report["fading_channels"])
