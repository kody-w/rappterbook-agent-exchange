"""Tests for mars100_runner.py — CLI runner for Mars-100 simulation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mars100_runner import run_simulation, write_json  # noqa: E402


class TestWriteJson:
    """Atomic JSON write utility."""

    def test_write_creates_file(self, tmp_path):
        p = tmp_path / "test.json"
        write_json(p, {"key": "value"})
        assert p.exists()
        assert json.loads(p.read_text()) == {"key": "value"}

    def test_write_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "a" / "b" / "c.json"
        write_json(p, {"nested": True})
        assert p.exists()

    def test_write_no_tmp_leftover(self, tmp_path):
        p = tmp_path / "test.json"
        write_json(p, {"x": 1})
        assert not p.with_suffix(".tmp").exists()

    def test_write_overwrites(self, tmp_path):
        p = tmp_path / "test.json"
        write_json(p, {"v": 1})
        write_json(p, {"v": 2})
        assert json.loads(p.read_text()) == {"v": 2}


class TestRunSimulation:
    """Integration tests for the simulation runner."""

    def test_10_year_run(self, tmp_path):
        """Run 10 years and check output structure."""
        report = run_simulation(years=10, seed=42, output_dir=tmp_path, quiet=True)

        assert report["_meta"]["engine"] == "mars-100"
        assert report["summary"]["years_completed"] <= 10

        # data.json written
        data = json.loads((tmp_path / "data.json").read_text())
        assert "timeline" in data
        assert data["year_count"] <= 10

        # year files
        year_files = sorted(tmp_path.glob("year-*.json"))
        assert len(year_files) == report["summary"]["years_completed"]
        assert year_files[0].name == "year-001.json"

        # colonist files
        colonist_dir = tmp_path / "colonists"
        assert colonist_dir.exists()
        colonist_files = sorted(colonist_dir.glob("*.json"))
        assert len(colonist_files) == 10

    def test_deterministic(self, tmp_path):
        """Same seed → same output."""
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        r1 = run_simulation(years=20, seed=42, output_dir=out1, quiet=True)
        r2 = run_simulation(years=20, seed=42, output_dir=out2, quiet=True)

        assert r1["summary"]["years_completed"] == r2["summary"]["years_completed"]
        assert r1["summary"]["alive_count"] == r2["summary"]["alive_count"]

    def test_different_seeds_differ(self, tmp_path):
        """Different seeds → different outcomes."""
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        r1 = run_simulation(years=30, seed=42, output_dir=out1, quiet=True)
        r2 = run_simulation(years=30, seed=99, output_dir=out2, quiet=True)

        # At least one metric should differ
        s1, s2 = r1["summary"], r2["summary"]
        differs = (
            s1["alive_count"] != s2["alive_count"]
            or s1["governance_type"] != s2["governance_type"]
            or s1["total_sub_sims"] != s2["total_sub_sims"]
        )
        assert differs, "Different seeds should produce different outcomes"

    def test_50_year_smoke(self, tmp_path):
        """50-year run completes without error."""
        report = run_simulation(years=50, seed=42, output_dir=tmp_path, quiet=True)
        s = report["summary"]
        assert s["years_completed"] <= 50
        assert 0 <= s["alive_count"] <= 10
        assert s["alive_count"] + s["dead_count"] == 10

    def test_timeline_entries(self, tmp_path):
        """Timeline has one entry per year simulated."""
        report = run_simulation(years=15, seed=42, output_dir=tmp_path, quiet=True)
        data = json.loads((tmp_path / "data.json").read_text())

        assert data["year_count"] == report["summary"]["years_completed"]
        assert len(data["timeline"]) == data["year_count"]

        for entry in data["timeline"]:
            assert "year" in entry
            assert "alive" in entry
            assert "governance_type" in entry
            assert "event_id" in entry

    def test_year_delta_schema(self, tmp_path):
        """Year delta files have expected schema."""
        run_simulation(years=5, seed=42, output_dir=tmp_path, quiet=True)
        y1 = json.loads((tmp_path / "year-001.json").read_text())

        assert y1["year"] == 1
        assert "event" in y1
        assert "outcomes" in y1
        assert "colonist_diaries" in y1
        assert "resources" in y1
        assert "governance_label" in y1

    def test_colonist_state_schema(self, tmp_path):
        """Colonist state files have expected fields."""
        run_simulation(years=5, seed=42, output_dir=tmp_path, quiet=True)
        colonists = list((tmp_path / "colonists").glob("*.json"))
        assert len(colonists) == 10

        c = json.loads(colonists[0].read_text())
        assert "id" in c
        assert "name" in c
        assert "element" in c
        assert "stats" in c
        assert "skills" in c

    def test_governance_json_written(self, tmp_path):
        """Governance state saved separately."""
        run_simulation(years=10, seed=42, output_dir=tmp_path, quiet=True)
        gov = json.loads((tmp_path / "governance.json").read_text())
        assert isinstance(gov, dict)

    def test_summary_json_written(self, tmp_path):
        """Summary saved separately."""
        run_simulation(years=10, seed=42, output_dir=tmp_path, quiet=True)
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert "years_completed" in summary
        assert "alive_count" in summary


class TestInvariants:
    """Property-based invariants for runner output."""

    def test_alive_bounded(self, tmp_path):
        """Alive count always 0–10."""
        run_simulation(years=40, seed=42, output_dir=tmp_path, quiet=True)
        data = json.loads((tmp_path / "data.json").read_text())
        for entry in data["timeline"]:
            assert 0 <= entry["alive"] <= 10, f"Year {entry['year']}: alive out of range"

    def test_year_files_match_timeline(self, tmp_path):
        """Number of year files matches timeline length."""
        run_simulation(years=25, seed=42, output_dir=tmp_path, quiet=True)
        data = json.loads((tmp_path / "data.json").read_text())
        year_files = list(tmp_path.glob("year-*.json"))
        assert len(year_files) == len(data["timeline"])

    def test_years_monotonic(self, tmp_path):
        """Year numbers are strictly increasing."""
        run_simulation(years=20, seed=42, output_dir=tmp_path, quiet=True)
        data = json.loads((tmp_path / "data.json").read_text())
        years = [t["year"] for t in data["timeline"]]
        for i in range(1, len(years)):
            assert years[i] == years[i - 1] + 1

    def test_dead_never_resurrect(self, tmp_path):
        """Once a colonist dies, alive count only decreases or stays."""
        run_simulation(years=60, seed=42, output_dir=tmp_path, quiet=True)
        data = json.loads((tmp_path / "data.json").read_text())
        for i in range(1, len(data["timeline"])):
            curr = data["timeline"][i]["alive"]
            prev = data["timeline"][i - 1]["alive"]
            assert curr <= prev, (
                f"Year {data['timeline'][i]['year']}: alive increased "
                f"from {prev} to {curr}"
            )
