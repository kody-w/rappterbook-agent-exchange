"""Tests for Mars-100 ensemble runner and amendment generator."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100_ensemble import (
    AMENDMENT_TEMPLATES,
    EnsembleResult,
    check_amendment_readiness,
    generate_amendment_markdown,
    governance_stability,
    run_ensemble,
    write_ensemble_output,
)


class TestEnsembleResult:
    def test_empty_to_dict(self):
        er = EnsembleResult()
        d = er.to_dict()
        assert d["num_runs"] == 0
        assert d["collapse_rate"] == 0

    def test_to_dict_keys(self):
        er = EnsembleResult(seeds=[1], survival_years=[50])
        d = er.to_dict()
        assert "num_runs" in d
        assert "mean_survival" in d


class TestSmallEnsemble:
    def test_run_3_seeds(self):
        ens = run_ensemble(num_runs=3, years=5, base_seed=100)
        assert len(ens.seeds) == 3

    def test_survival_bounded(self):
        ens = run_ensemble(num_runs=3, years=10, base_seed=200)
        for y in ens.survival_years:
            assert 1 <= y <= 10

    def test_governance_counts_sum(self):
        ens = run_ensemble(num_runs=4, years=10, base_seed=400)
        assert sum(ens.governance_counts.values()) == 4

    def test_deterministic(self):
        e1 = run_ensemble(num_runs=2, years=5, base_seed=600)
        e2 = run_ensemble(num_runs=2, years=5, base_seed=600)
        assert e1.survival_years == e2.survival_years


class TestGovernanceStability:
    def test_prevalence_bounded(self):
        ens = run_ensemble(num_runs=3, years=10, base_seed=800)
        stab = governance_stability(ens)
        for stats in stab.values():
            assert 0 <= stats["prevalence"] <= 1

    def test_max_streak_bounded(self):
        ens = run_ensemble(num_runs=3, years=10, base_seed=950)
        stab = governance_stability(ens)
        for stats in stab.values():
            assert 0 <= stats["max_streak"] <= 10


class TestAmendmentReadiness:
    def test_anarchy_excluded(self):
        ens = run_ensemble(num_runs=3, years=10, base_seed=1200)
        ready = check_amendment_readiness(ens, min_prevalence=0.0, min_streak=0)
        assert all(r["governance_type"] != "anarchy" for r in ready)

    def test_high_threshold_empty(self):
        ens = run_ensemble(num_runs=2, years=5, base_seed=1300)
        ready = check_amendment_readiness(ens, min_prevalence=1.0, min_streak=100)
        assert len(ready) == 0


class TestAmendmentMarkdown:
    def test_all_templates(self):
        for gov in ["council", "democracy", "technocracy", "monarchy", "theocracy"]:
            assert gov in AMENDMENT_TEMPLATES

    def test_generate(self):
        stats = {"prevalence": 0.6, "mean_duration": 35.0, "max_streak": 20}
        summary = {"num_runs": 10, "mean_survival": 65.0, "collapse_rate": 0.3}
        md = generate_amendment_markdown("council", stats, summary)
        assert "Amendment:" in md

    def test_fallback_type(self):
        stats = {"prevalence": 0.5, "mean_duration": 20.0, "max_streak": 10}
        summary = {"num_runs": 3, "mean_survival": 50.0, "collapse_rate": 0.5}
        md = generate_amendment_markdown("oligarchy", stats, summary)
        assert "Amendment:" in md


class TestOutputFiles:
    def test_writes_ensemble_json(self):
        ens = run_ensemble(num_runs=2, years=5, base_seed=1400)
        with tempfile.TemporaryDirectory() as tmpdir:
            written = write_ensemble_output(ens, tmpdir)
            assert "ensemble" in written
            data = json.loads(Path(written["ensemble"]).read_text())
            assert data["num_runs"] == 2

    def test_amendment_when_ready(self):
        ens = EnsembleResult(
            seeds=[1, 2, 3],
            governance_counts={"democracy": 3},
            survival_years=[80, 90, 70],
            governance_timelines=[["democracy"] * 80, ["democracy"] * 90, ["democracy"] * 70],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            written = write_ensemble_output(ens, tmpdir)
            assert "amendment" in written


class TestInvariants:
    def test_survival_monotonic(self):
        e_short = run_ensemble(num_runs=3, years=10, base_seed=2000)
        e_long = run_ensemble(num_runs=3, years=50, base_seed=2000)
        for short, long in zip(e_short.survival_years, e_long.survival_years):
            assert long >= short

    def test_births_non_negative(self):
        ens = run_ensemble(num_runs=3, years=20, base_seed=2100)
        assert all(b >= 0 for b in ens.total_births)

    def test_results_match_seeds(self):
        ens = run_ensemble(num_runs=4, years=5, base_seed=2300)
        assert len(ens.results) == len(ens.seeds) == 4
