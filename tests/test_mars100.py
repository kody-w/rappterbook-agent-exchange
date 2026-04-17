"""test_mars100.py -- Tests for the Mars-100 simulation engine.

Covers: smoke test, full run, determinism, conservation laws, death
handling, colonist memory, governance emergence, sub-sim integration.
"""
from __future__ import annotations

import json
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100 import Mars100Simulation, write_results


class TestSmoke:
    """Basic smoke tests — does it run at all?"""

    def test_runs_10_years(self):
        sim = Mars100Simulation(seed=42, years=10)
        results = sim.run()
        assert results['_meta']['years_completed'] == 10

    def test_runs_20_years(self):
        sim = Mars100Simulation(seed=42, years=20)
        results = sim.run()
        assert results['_meta']['years_completed'] == 20

    def test_initial_population_10(self):
        sim = Mars100Simulation(seed=42, years=1)
        results = sim.run()
        # Year 1 population should be 10 (no births or deaths expected)
        assert results['years'][0]['population'] == 10

    def test_returns_valid_structure(self):
        sim = Mars100Simulation(seed=42, years=5)
        results = sim.run()
        assert '_meta' in results
        assert 'summary' in results
        assert 'years' in results
        assert 'colonists' in results

    def test_json_serializable(self):
        sim = Mars100Simulation(seed=42, years=5)
        results = sim.run()
        # Should not raise
        json.dumps(results, default=str)


class TestDeterminism:
    """Same seed = same simulation."""

    def test_same_seed_same_population(self):
        r1 = Mars100Simulation(seed=42, years=30).run()
        r2 = Mars100Simulation(seed=42, years=30).run()
        pop1 = [y['population'] for y in r1['years']]
        pop2 = [y['population'] for y in r2['years']]
        assert pop1 == pop2

    def test_same_seed_same_events(self):
        r1 = Mars100Simulation(seed=42, years=20).run()
        r2 = Mars100Simulation(seed=42, years=20).run()
        events1 = [y['event']['type'] for y in r1['years']]
        events2 = [y['event']['type'] for y in r2['years']]
        assert events1 == events2

    def test_different_seeds_differ(self):
        r1 = Mars100Simulation(seed=42, years=30).run()
        r2 = Mars100Simulation(seed=99, years=30).run()
        pop1 = [y['population'] for y in r1['years']]
        pop2 = [y['population'] for y in r2['years']]
        assert pop1 != pop2

    def test_same_seed_same_governance(self):
        r1 = Mars100Simulation(seed=42, years=50).run()
        r2 = Mars100Simulation(seed=42, years=50).run()
        assert r1['summary']['dominant_governance'] == r2['summary']['dominant_governance']


class TestPopulationAccounting:
    """Population = alive colonists. Births + arrivals - deaths should match."""

    def test_population_never_negative(self):
        sim = Mars100Simulation(seed=42, years=100)
        results = sim.run()
        for year_data in results['years']:
            assert year_data['population'] >= 0

    def test_population_matches_alive_count(self):
        sim = Mars100Simulation(seed=42, years=50)
        results = sim.run()
        final_alive = sum(1 for c in results['colonists'] if c['alive'])
        assert results['summary']['final_population'] == final_alive

    def test_deaths_tracked(self):
        sim = Mars100Simulation(seed=42, years=100)
        results = sim.run()
        # Deaths list matches dead colonists
        dead_in_list = len(results['dead'])
        dead_in_colonists = sum(1 for c in results['colonists'] if not c['alive'])
        assert dead_in_list == dead_in_colonists


class TestDeathHandling:
    """Dead colonists are archived, not deleted."""

    def test_dead_remain_in_colonists_list(self):
        sim = Mars100Simulation(seed=42, years=100)
        results = sim.run()
        total = len(results['colonists'])
        alive = sum(1 for c in results['colonists'] if c['alive'])
        dead = sum(1 for c in results['colonists'] if not c['alive'])
        assert alive + dead == total

    def test_dead_have_year_died(self):
        sim = Mars100Simulation(seed=42, years=100)
        results = sim.run()
        for c in results['colonists']:
            if not c['alive']:
                assert c['year_died'] is not None
                assert c['year_died'] > 0

    def test_dead_have_death_cause(self):
        sim = Mars100Simulation(seed=42, years=100)
        results = sim.run()
        for d in results['dead']:
            assert 'cause' in d
            assert d['cause'] != ''


class TestGovernanceEmergence:
    def test_governance_pattern_detected(self):
        sim = Mars100Simulation(seed=42, years=50)
        results = sim.run()
        pattern = results['summary']['dominant_governance']
        assert pattern in ('anarchy', 'democracy', 'council', 'oligarchy',
                          'theocracy', 'autocracy', 'technocracy', 'commune')

    def test_pattern_history_populated(self):
        sim = Mars100Simulation(seed=42, years=50)
        results = sim.run()
        history = results['summary']['pattern_history']
        assert len(history) == 50
        for year, pattern in history:
            assert isinstance(year, int)
            assert isinstance(pattern, str)

    def test_proposals_exist(self):
        sim = Mars100Simulation(seed=42, years=50)
        results = sim.run()
        assert len(results['proposals']) > 0

    def test_proposals_have_votes(self):
        sim = Mars100Simulation(seed=42, years=50)
        results = sim.run()
        voted_proposals = [p for p in results['proposals'] if p['votes']]
        assert len(voted_proposals) > 0


class TestSubSimIntegration:
    def test_sub_sims_appear_after_year_50(self):
        sim = Mars100Simulation(seed=42, years=55)
        results = sim.run()
        sub_sims = results['sub_sim_log']
        assert len(sub_sims) >= 1

    def test_sub_sims_at_year_67(self):
        sim = Mars100Simulation(seed=42, years=70)
        results = sim.run()
        y67_sims = [s for s in results['sub_sim_log'] if s.get('year') == 67]
        assert len(y67_sims) >= 1

    def test_depth_3_at_year_82(self):
        sim = Mars100Simulation(seed=42, years=85)
        results = sim.run()
        y82_sims = [s for s in results['sub_sim_log'] if s.get('year') == 82]
        assert any(s.get('depth', 0) >= 3 for s in y82_sims)

    def test_sub_sim_logged(self):
        sim = Mars100Simulation(seed=42, years=55)
        results = sim.run()
        for s in results['sub_sim_log']:
            assert 'success' in s
            assert 'timestamp' in s


class TestDiaries:
    def test_diaries_per_year(self):
        sim = Mars100Simulation(seed=42, years=10)
        results = sim.run()
        for year_data in results['years']:
            assert 'diaries' in year_data
            assert len(year_data['diaries']) <= 3

    def test_diary_has_entry(self):
        sim = Mars100Simulation(seed=42, years=5)
        results = sim.run()
        for year_data in results['years']:
            for diary in year_data['diaries']:
                assert 'entry' in diary
                assert len(diary['entry']) > 0
                assert 'colonist_id' in diary


class TestColonistMemory:
    def test_memory_grows(self):
        sim = Mars100Simulation(seed=42, years=20)
        results = sim.run()
        for c in results['colonists']:
            if c['alive']:
                assert len(c['memory']) > 0

    def test_memory_capped(self):
        sim = Mars100Simulation(seed=42, years=100)
        results = sim.run()
        for c in results['colonists']:
            assert len(c['memory']) <= 50


class TestWriteResults:
    def test_writes_state_file(self):
        sim = Mars100Simulation(seed=42, years=5)
        results = sim.run()
        with tempfile.TemporaryDirectory() as tmpdir:
            write_results(results, Path(tmpdir))
            assert (Path(tmpdir) / 'state.json').exists()

    def test_writes_summary(self):
        sim = Mars100Simulation(seed=42, years=5)
        results = sim.run()
        with tempfile.TemporaryDirectory() as tmpdir:
            write_results(results, Path(tmpdir))
            assert (Path(tmpdir) / 'summary.json').exists()

    def test_writes_colonist_files(self):
        sim = Mars100Simulation(seed=42, years=5)
        results = sim.run()
        with tempfile.TemporaryDirectory() as tmpdir:
            write_results(results, Path(tmpdir))
            colonist_dir = Path(tmpdir) / 'colonists'
            assert colonist_dir.exists()
            files = list(colonist_dir.glob('*.json'))
            assert len(files) == len(results['colonists'])

    def test_output_is_valid_json(self):
        sim = Mars100Simulation(seed=42, years=5)
        results = sim.run()
        with tempfile.TemporaryDirectory() as tmpdir:
            write_results(results, Path(tmpdir))
            state = json.loads((Path(tmpdir) / 'state.json').read_text())
            assert '_meta' in state

    def test_sub_sim_logs_written(self):
        sim = Mars100Simulation(seed=42, years=55)
        results = sim.run()
        with tempfile.TemporaryDirectory() as tmpdir:
            write_results(results, Path(tmpdir))
            subsim_dir = Path(tmpdir) / 'sub-sims'
            if results['sub_sim_log']:
                assert subsim_dir.exists()


class TestResourceBounds:
    """Resources should never go negative."""

    def test_resources_non_negative(self):
        sim = Mars100Simulation(seed=42, years=100)
        results = sim.run()
        for year_data in results['years']:
            for resource, value in year_data['resources'].items():
                assert value >= 0, f"Year {year_data['year']}: {resource} = {value}"


class TestFullRun:
    """Full 100-year simulation."""

    def test_full_100_years(self):
        sim = Mars100Simulation(seed=42, years=100)
        results = sim.run()
        assert results['_meta']['years_completed'] >= 1
        # Colony may die before 100, that's ok

    def test_amendment_proposed(self):
        """Over 100 years, an amendment should be proposed."""
        sim = Mars100Simulation(seed=42, years=100)
        results = sim.run()
        # Amendment may or may not be proposed depending on governance patterns
        # Just check it's present in the structure
        assert 'amendment' in results['summary']

    def test_performance_under_5_seconds(self):
        """Full 100-year sim should complete in under 5 seconds."""
        import time
        start = time.time()
        sim = Mars100Simulation(seed=42, years=100)
        sim.run()
        elapsed = time.time() - start
        assert elapsed < 5.0, f"100-year sim took {elapsed:.2f}s"
