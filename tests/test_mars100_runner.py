"""Tests for Mars-100 analysis runner (run_and_analyze + write_outputs)."""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from src.mars100.runner import run_and_analyze, write_outputs


@pytest.fixture
def short_run() -> dict:
    """Run a short analysis for speed."""
    return run_and_analyze(seed=42, total_years=10)


class TestRunAndAnalyze:
    def test_returns_all_keys(self, short_run: dict) -> None:
        assert "sim_dict" in short_run
        assert "analysis" in short_run
        assert "report_md" in short_run

    def test_sim_dict_valid(self, short_run: dict) -> None:
        sim = short_run["sim_dict"]
        assert "years" in sim
        assert "summary" in sim
        assert len(sim["years"]) == 10

    def test_analysis_has_sections(self, short_run: dict) -> None:
        analysis = short_run["analysis"]
        for key in ("value_convergence", "governance_stability",
                     "subsim_effectiveness", "meta_emergence",
                     "amendment_proposal", "fitness"):
            assert key in analysis

    def test_report_is_markdown(self, short_run: dict) -> None:
        report = short_run["report_md"]
        assert report.startswith("# Emergent Governance Patterns")

    def test_deterministic(self) -> None:
        r1 = run_and_analyze(seed=42, total_years=5)
        r2 = run_and_analyze(seed=42, total_years=5)
        assert r1["analysis"] == r2["analysis"]


class TestWriteOutputs:
    def test_writes_all_files(self, short_run: dict, tmp_path: Path) -> None:
        docs = tmp_path / "docs"
        state = tmp_path / "state"
        written = write_outputs(short_run, docs_dir=docs, state_dir=state)

        assert "analysis" in written
        assert "report" in written
        assert "data" in written
        assert "state" in written
        assert "colonists" in written

    def test_analysis_json_valid(self, short_run: dict, tmp_path: Path) -> None:
        docs = tmp_path / "docs"
        state = tmp_path / "state"
        write_outputs(short_run, docs_dir=docs, state_dir=state)

        analysis_path = docs / "analysis.json"
        assert analysis_path.exists()
        data = json.loads(analysis_path.read_text())
        assert "value_convergence" in data
        assert "fitness" in data

    def test_report_md_valid(self, short_run: dict, tmp_path: Path) -> None:
        docs = tmp_path / "docs"
        state = tmp_path / "state"
        write_outputs(short_run, docs_dir=docs, state_dir=state)

        report_path = docs / "governance-report.md"
        assert report_path.exists()
        content = report_path.read_text()
        assert "Emergent Governance" in content

    def test_data_json_has_summary(self, short_run: dict, tmp_path: Path) -> None:
        docs = tmp_path / "docs"
        state = tmp_path / "state"
        write_outputs(short_run, docs_dir=docs, state_dir=state)

        data = json.loads((docs / "data.json").read_text())
        assert "summary" in data
        assert "analysis_summary" in data

    def test_state_mars100_json(self, short_run: dict, tmp_path: Path) -> None:
        docs = tmp_path / "docs"
        state = tmp_path / "state"
        write_outputs(short_run, docs_dir=docs, state_dir=state)

        state_data = json.loads((state / "mars100.json").read_text())
        assert "_meta" in state_data
        assert state_data["_meta"]["version"] == "1.2"

    def test_colonist_files(self, short_run: dict, tmp_path: Path) -> None:
        docs = tmp_path / "docs"
        state = tmp_path / "state"
        write_outputs(short_run, docs_dir=docs, state_dir=state)

        colonist_dir = docs / "colonists"
        assert colonist_dir.exists()
        files = list(colonist_dir.glob("*.json"))
        assert len(files) >= 1  # at least some colonists

    def test_atomic_write_no_partial(self, short_run: dict, tmp_path: Path) -> None:
        """No .tmp files should remain after write."""
        docs = tmp_path / "docs"
        state = tmp_path / "state"
        write_outputs(short_run, docs_dir=docs, state_dir=state)

        tmp_files = list(docs.rglob("*.tmp")) + list(state.rglob("*.tmp"))
        assert len(tmp_files) == 0
