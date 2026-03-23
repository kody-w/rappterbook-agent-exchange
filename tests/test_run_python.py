"""
Tests for run_python.py -- run_python action adapter.

Run: python -m pytest tests/test_run_python.py -v
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.run_python import (
    main,
    run_market_only,
    run_terrarium_only,
    save_proof_state,
)
from src.run_proof import ProofReport, run_proof


class TestTerrariumOnly:

    def test_runs_without_crash(self) -> None:
        result = run_terrarium_only(sols=30, seed=42)
        assert "text" in result and "results" in result

    def test_output_contains_colonies(self) -> None:
        result = run_terrarium_only(sols=30, seed=42)
        for name in ["Ares Prime", "Olympus Station", "Red Frontier"]:
            assert name in result["text"]

    def test_output_header(self) -> None:
        assert "TERRARIUM EXECUTION PROOF" in run_terrarium_only(sols=30, seed=42)["text"]

    def test_3_colonies(self) -> None:
        assert len(run_terrarium_only(sols=30, seed=42)["results"]["colonies"]) == 3

    def test_deterministic(self) -> None:
        r1 = run_terrarium_only(sols=30, seed=42)
        r2 = run_terrarium_only(sols=30, seed=42)
        for c1, c2 in zip(r1["results"]["colonies"], r2["results"]["colonies"]):
            assert c1["final_population"] == c2["final_population"]


class TestMarketOnly:

    def test_runs_without_crash(self) -> None:
        result = run_market_only(sols=30, n_predictions=20, n_seeds=1)
        assert "text" in result and "mean_brier" in result

    def test_output_header(self) -> None:
        assert "PREDICTION MARKET" in run_market_only(sols=30, n_predictions=20, n_seeds=1)["text"]

    def test_brier_bounded(self) -> None:
        assert 0.0 <= run_market_only(sols=30, n_predictions=20, n_seeds=1)["mean_brier"] <= 1.0

    def test_accuracy_bounded(self) -> None:
        assert 0.0 <= run_market_only(sols=30, n_predictions=20, n_seeds=1)["accuracy"] <= 1.0

    def test_all_resolved(self) -> None:
        assert run_market_only(sols=30, n_predictions=20, n_seeds=1)["n_resolved"] == 20


class TestSaveProof:

    def test_saves_proof_json(self) -> None:
        report = run_proof(sols=30, n_predictions=10, n_seeds=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_proof_state(report, Path(tmpdir))
            assert path.exists()
            assert "proof_hash" in json.loads(path.read_text())

    def test_appends_to_history(self) -> None:
        r1 = run_proof(sols=30, n_predictions=10, n_seeds=1, market_seed=1)
        r2 = run_proof(sols=30, n_predictions=10, n_seeds=1, market_seed=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_proof_state(r1, Path(tmpdir))
            save_proof_state(r2, Path(tmpdir))
            history = json.loads((Path(tmpdir) / "proof_history.json").read_text())
            assert len(history) == 2

    def test_history_capped(self) -> None:
        report = run_proof(sols=30, n_predictions=10, n_seeds=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            for _ in range(55):
                save_proof_state(report, Path(tmpdir))
            assert len(json.loads((Path(tmpdir) / "proof_history.json").read_text())) == 50

    def test_no_tmp_left(self) -> None:
        report = run_proof(sols=30, n_predictions=10, n_seeds=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_proof_state(report, Path(tmpdir))
            assert len(list(Path(tmpdir).glob("*.tmp"))) == 0


class TestCLI:

    def test_cli_default(self, capsys) -> None:
        sys.argv = ["run_python", "--sols", "30", "--predictions", "10", "--seeds", "1"]
        assert main() == 0
        assert "EXECUTION PROOF" in capsys.readouterr().out

    def test_cli_terrarium(self, capsys) -> None:
        sys.argv = ["run_python", "--target", "terrarium", "--sols", "30"]
        assert main() == 0
        assert "TERRARIUM" in capsys.readouterr().out

    def test_cli_market(self, capsys) -> None:
        sys.argv = ["run_python", "--target", "market", "--sols", "30",
                     "--predictions", "10", "--seeds", "1"]
        assert main() == 0
        assert "PREDICTION MARKET" in capsys.readouterr().out

    def test_cli_json(self, capsys) -> None:
        sys.argv = ["run_python", "--sols", "30", "--predictions", "10",
                     "--seeds", "1", "--json"]
        assert main() == 0
        assert "proof_hash" in json.loads(capsys.readouterr().out)


class TestInvariants:

    def test_terrarium_pops_nonneg(self) -> None:
        for c in run_terrarium_only(sols=50, seed=42)["results"]["colonies"]:
            assert c["final_population"] >= 0

    def test_market_brier_nonneg(self) -> None:
        assert run_market_only(sols=50, n_predictions=30, n_seeds=1)["mean_brier"] >= 0.0

    def test_both_covers_all(self) -> None:
        report = run_proof(sols=50, n_predictions=20, n_seeds=1)
        assert report.terrarium["total_population"] > 0
        assert report.n_resolved > 0
        assert report.ci.score >= 0.0
