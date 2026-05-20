"""Smoke test for scripts/rumor_snapshot.py."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "rumor_snapshot.py"


def _run(state_dir, docs_dir, years, seed, extra=None):
    args = [sys.executable, str(SCRIPT),
            "--years", str(years), "--seed", str(seed),
            "--state-dir", str(state_dir),
            "--docs-dir", str(docs_dir),
            "--quiet"]
    if extra:
        args.extend(extra)
    return subprocess.run(args, cwd=str(REPO), capture_output=True,
                           text=True, timeout=300)


def test_script_exists():
    assert SCRIPT.exists()


def test_runs_and_writes_report(tmp_path):
    state_dir = tmp_path / "state"
    docs_dir = tmp_path / "docs" / "mars-100"
    result = _run(state_dir, docs_dir, years=8, seed=1)
    assert result.returncode == 0, result.stderr
    out = state_dir / "rumors.json"
    assert out.exists()
    report = json.loads(out.read_text())
    for k in ("_meta", "active_rumors", "fragmentation",
              "rumors", "archive_recent", "transmission_log_recent"):
        assert k in report
    assert report["_meta"]["organ"] == "rumors"
    assert report["_meta"]["version"] == "13.0"
    assert 0.0 <= report["fragmentation"] <= 1.0
    assert isinstance(report["rumors"], list)
    assert (docs_dir / "rumors.json").exists()


def test_no_docs_flag(tmp_path):
    state_dir = tmp_path / "state"
    docs_dir = tmp_path / "docs"
    result = _run(state_dir, docs_dir, years=5, seed=3, extra=["--no-docs"])
    assert result.returncode == 0, result.stderr
    assert (state_dir / "rumors.json").exists()
    assert not (docs_dir / "mars-100" / "rumors.json").exists()


def test_rumor_carrier_counts_consistent(tmp_path):
    state_dir = tmp_path / "state"
    docs_dir = tmp_path / "docs"
    result = _run(state_dir, docs_dir, years=12, seed=9, extra=["--no-docs"])
    assert result.returncode == 0, result.stderr
    report = json.loads((state_dir / "rumors.json").read_text())
    for r in report["rumors"]:
        assert r["carrier_count"] == len(r["carriers"])
        assert len(r["carrier_names"]) == r["carrier_count"]
