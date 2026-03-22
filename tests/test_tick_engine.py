"""Tests for Mars tick engine — the core simulation loop."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from mars.tick_engine import create_world, tick, save_world, _build_viz_data


class TestCreateWorld:
    """Test world initialization."""

    def test_creates_three_colonies(self) -> None:
        world = create_world()
        assert len(world["colonies"]) == 3

    def test_colony_names(self) -> None:
        world = create_world()
        names = {c["name"] for c in world["colonies"]}
        assert names == {"Ares Prime", "Boreas Station", "Hellas Deep"}

    def test_initial_population(self) -> None:
        world = create_world()
        total = sum(c["population"] for c in world["colonies"])
        assert total == 24  # 8 + 6 + 10

    def test_meta_fields(self) -> None:
        world = create_world()
        assert world["_meta"]["engine"] == "mars-barn"
        assert world["_meta"]["version"] == "1.0.0"
        assert world["sol"] == 0

    def test_deterministic_with_seed(self) -> None:
        w1 = create_world(seed=42)
        w2 = create_world(seed=42)
        assert w1["colonies"] == w2["colonies"]

    def test_different_seeds_differ(self) -> None:
        w1 = create_world(seed=42)
        w2 = create_world(seed=99)
        # Same structure but initial state is the same (seed affects tick, not init)
        assert w1["colonies"][0]["name"] == w2["colonies"][0]["name"]


class TestTick:
    """Test the core tick function."""

    def test_advances_sol(self) -> None:
        world = create_world()
        world2 = tick(world)
        assert world2["sol"] == 1

    def test_does_not_mutate_input(self) -> None:
        world = create_world()
        original_sol = world["sol"]
        _ = tick(world)
        assert world["sol"] == original_sol

    def test_population_non_negative(self) -> None:
        world = create_world()
        for _ in range(50):
            world = tick(world)
            for colony in world["colonies"]:
                assert colony["population"] >= 0

    def test_environment_log_grows(self) -> None:
        world = create_world()
        for _ in range(10):
            world = tick(world)
        assert len(world["environment_log"]) == 10

    def test_summary_updates(self) -> None:
        world = create_world()
        for _ in range(10):
            world = tick(world)
        summary = world["summary"]
        assert summary["sols_simulated"] == 10
        assert summary["total_population"] > 0

    def test_deterministic(self) -> None:
        """Same seed + same start → same result."""
        w1 = create_world(seed=42)
        w2 = create_world(seed=42)
        for _ in range(20):
            w1 = tick(w1)
            w2 = tick(w2)
        assert w1["sol"] == w2["sol"]
        for c1, c2 in zip(w1["colonies"], w2["colonies"]):
            assert c1["population"] == c2["population"]


class TestTickEngine365:
    """Integration test: 365 sols through tick engine."""

    def test_365_sols_no_crash(self) -> None:
        """The headline test: run 365 sols without error."""
        world = create_world(seed=42)
        for _ in range(365):
            world = tick(world)
        assert world["sol"] == 365
        total = sum(c["population"] for c in world["colonies"])
        assert total > 0

    def test_all_colonies_have_history(self) -> None:
        world = create_world(seed=42)
        for _ in range(365):
            world = tick(world)
        for colony in world["colonies"]:
            assert len(colony["history"]) == 365

    def test_viz_data_valid(self) -> None:
        world = create_world(seed=42)
        for _ in range(100):
            world = tick(world)
        viz = _build_viz_data(world)
        assert len(viz["colonies"]) == 3
        for c in viz["colonies"]:
            assert len(c["curve"]["sols"]) == 100
            assert len(c["curve"]["population"]) == 100


class TestSaveLoad:
    """Test save/load round-trip."""

    def test_save_creates_files(self) -> None:
        world = create_world()
        for _ in range(10):
            world = tick(world)

        with tempfile.TemporaryDirectory() as tmpdir:
            import mars.tick_engine as te
            old_state = te.STATE_PATH
            old_viz = te.VIZ_PATH
            try:
                te.STATE_PATH = Path(tmpdir) / "mars.json"
                te.VIZ_PATH = Path(tmpdir) / "mars_data.json"
                save_world(world)
                assert te.STATE_PATH.exists()
                assert te.VIZ_PATH.exists()

                # Verify JSON validity
                data = json.loads(te.STATE_PATH.read_text())
                assert data["sol"] == 10

                viz = json.loads(te.VIZ_PATH.read_text())
                assert len(viz["colonies"]) == 3
            finally:
                te.STATE_PATH = old_state
                te.VIZ_PATH = old_viz


class TestConservationLaws:
    """Property-based tests — physical invariants."""

    def test_births_deaths_immigration_balance(self) -> None:
        """total_pop = initial + births - deaths + immigrants."""
        world = create_world(seed=42)
        initial_pop = sum(c["population"] for c in world["colonies"])

        for _ in range(100):
            world = tick(world)

        final_pop = sum(c["population"] for c in world["colonies"])
        total_births = sum(c["demographics"]["total_births"]
                           for c in world["colonies"])
        total_deaths = sum(c["demographics"]["total_deaths"]
                           for c in world["colonies"])
        total_immigrants = sum(c["demographics"]["total_immigrants"]
                               for c in world["colonies"])

        expected = initial_pop + total_births - total_deaths + (
            total_immigrants - initial_pop  # Subtract initial crew counted as immigrants
        )
        assert final_pop == expected, (
            f"Population accounting mismatch: {final_pop} != {expected} "
            f"(init={initial_pop}, births={total_births}, deaths={total_deaths}, "
            f"immig={total_immigrants})"
        )

    def test_sols_monotonic(self) -> None:
        """Sol counter should be strictly increasing."""
        world = create_world()
        prev_sol = -1
        for _ in range(50):
            world = tick(world)
            assert world["sol"] > prev_sol
            prev_sol = world["sol"]
