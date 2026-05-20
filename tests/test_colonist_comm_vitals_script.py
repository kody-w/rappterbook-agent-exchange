"""End-to-end smoke tests for the colonist_comm_vitals CLI script."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "colonist_comm_vitals.py"


def test_script_runs_and_writes_state(tmp_path):
    state_dir = tmp_path / "state"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--years", "10", "--seed", "42",
         "--state-dir", str(state_dir),
         "--no-docs", "--quiet"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    out = state_dir / "colonist_comm_vitals.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["_meta"]["organ"] == "comm-vitals"
    assert data["_meta"]["engine"] == "mars-100"
    assert "colonists" in data and "summary" in data
    assert data["summary"]["total_colonists"] >= 1
    for row in data["colonists"]:
        assert 0.0 <= row["urgency"] <= 1.0
        assert row["classification"] in (
            "healthy", "strained", "isolated", "ghosted"
        )


def test_script_writes_docs_when_not_suppressed(tmp_path):
    state_dir = tmp_path / "state"
    docs_dir = tmp_path / "docs"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--years", "5", "--seed", "1",
         "--state-dir", str(state_dir),
         "--docs-dir", str(docs_dir),
         "--quiet"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert (state_dir / "colonist_comm_vitals.json").exists()
    assert (docs_dir / "colonist_comm_vitals.json").exists()


def test_script_deterministic_with_same_seed(tmp_path):
    def run(out_dir):
        subprocess.run(
            [sys.executable, str(SCRIPT),
             "--years", "8", "--seed", "999",
             "--state-dir", str(out_dir),
             "--no-docs", "--quiet"],
            cwd=REPO_ROOT, check=True, capture_output=True, timeout=120,
        )
        return json.loads(
            (out_dir / "colonist_comm_vitals.json").read_text()
        )
    a = run(tmp_path / "a")
    b = run(tmp_path / "b")
    a["_meta"].pop("generated")
    b["_meta"].pop("generated")
    assert a == b
