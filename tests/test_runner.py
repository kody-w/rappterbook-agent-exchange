"""
Tests for the code execution engine (runner.py).

Run: python -m pytest tests/test_runner.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.runner import (
    RunResult,
    run_python,
    run_terrarium,
    run_market,
    format_proof,
    format_combined_proof,
)


# ===================================================================
# run_python basics
# ===================================================================

class TestRunPython:
    def test_captures_stdout(self) -> None:
        result = run_python("print('hello world')")
        assert result.exit_code == 0
        assert "hello world" in result.stdout

    def test_captures_stderr(self) -> None:
        result = run_python("import sys; sys.stderr.write('oops\\n')")
        assert "oops" in result.stderr

    def test_handles_syntax_error(self) -> None:
        result = run_python("def broken(")
        assert result.exit_code != 0
        assert "SyntaxError" in result.stderr

    def test_handles_runtime_error(self) -> None:
        result = run_python("1/0")
        assert result.exit_code != 0
        assert "ZeroDivisionError" in result.stderr

    def test_handles_timeout(self) -> None:
        result = run_python("import time; time.sleep(10)", timeout=1)
        assert result.exit_code == 124
        assert "TimeoutExpired" in result.stderr

    def test_timestamp_present(self) -> None:
        result = run_python("print(1)")
        assert "T" in result.timestamp  # ISO format
        assert len(result.timestamp) > 10

    def test_duration_nonnegative(self) -> None:
        result = run_python("print(1)")
        assert result.duration_ms >= 0

    def test_exit_code_zero_on_success(self) -> None:
        result = run_python("print('ok')")
        assert result.exit_code == 0

    def test_command_recorded(self) -> None:
        result = run_python("x = 1")
        assert "python" in result.command

    def test_multiline_code(self) -> None:
        code = "x = 2\ny = 3\nprint(x + y)"
        result = run_python(code)
        assert "5" in result.stdout
        assert result.exit_code == 0

    def test_empty_code(self) -> None:
        result = run_python("")
        assert result.exit_code == 0


# ===================================================================
# run_terrarium
# ===================================================================

class TestRunTerrarium:
    def test_produces_output(self) -> None:
        result = run_terrarium(sols=30, seed=42)
        assert result.exit_code == 0
        assert "TERRARIUM" in result.stdout

    def test_contains_colony_names(self) -> None:
        result = run_terrarium(sols=30, seed=42)
        assert "Ares Prime" in result.stdout
        assert "Olympus Station" in result.stdout
        assert "Red Frontier" in result.stdout

    def test_contains_population_data(self) -> None:
        result = run_terrarium(sols=30, seed=42)
        assert "Pop:" in result.stdout

    def test_contains_terraforming(self) -> None:
        result = run_terrarium(sols=30, seed=42)
        assert "Terraforming" in result.stdout

    def test_deterministic(self) -> None:
        r1 = run_terrarium(sols=30, seed=99)
        r2 = run_terrarium(sols=30, seed=99)
        assert r1.stdout == r2.stdout


# ===================================================================
# run_market
# ===================================================================

class TestRunMarket:
    def test_produces_output(self) -> None:
        result = run_market(n_predictions=20, sols=50, seeds=[42])
        assert result.exit_code == 0
        assert "PREDICTION MARKET" in result.stdout

    def test_contains_leaderboard(self) -> None:
        result = run_market(n_predictions=20, sols=50, seeds=[42])
        assert "LEADERBOARD" in result.stdout

    def test_contains_calibration(self) -> None:
        result = run_market(n_predictions=20, sols=50, seeds=[42])
        assert "CALIBRATION" in result.stdout

    def test_contains_terrarium_outcome(self) -> None:
        result = run_market(n_predictions=20, sols=50, seeds=[42])
        assert "TERRARIUM OUTCOME" in result.stdout


# ===================================================================
# format_proof
# ===================================================================

class TestFormatProof:
    def test_contains_markdown_code_block(self) -> None:
        result = RunResult(
            stdout="hello", stderr="", exit_code=0,
            duration_ms=42, timestamp="2026-03-23T00:00:00Z",
            command="python -c 'print(1)'",
        )
        md = format_proof(result)
        assert "```" in md
        assert "hello" in md

    def test_contains_timestamp(self) -> None:
        result = RunResult(
            stdout="ok", stderr="", exit_code=0,
            duration_ms=100, timestamp="2026-03-23T01:02:03Z",
            command="test",
        )
        md = format_proof(result)
        assert "2026-03-23" in md

    def test_contains_duration(self) -> None:
        result = RunResult(
            stdout="ok", stderr="", exit_code=0,
            duration_ms=1234, timestamp="2026-03-23T00:00:00Z",
            command="test",
        )
        md = format_proof(result)
        assert "1234ms" in md

    def test_shows_success_status(self) -> None:
        result = RunResult(
            stdout="ok", stderr="", exit_code=0,
            duration_ms=10, timestamp="now", command="test",
        )
        md = format_proof(result)
        assert "SUCCESS" in md

    def test_shows_failure_status(self) -> None:
        result = RunResult(
            stdout="", stderr="error", exit_code=1,
            duration_ms=10, timestamp="now", command="test",
        )
        md = format_proof(result)
        assert "FAILED" in md

    def test_includes_stderr_when_present(self) -> None:
        result = RunResult(
            stdout="", stderr="warning here", exit_code=0,
            duration_ms=10, timestamp="now", command="test",
        )
        md = format_proof(result)
        assert "warning here" in md


class TestFormatCombinedProof:
    def test_contains_both_sections(self) -> None:
        t = RunResult(stdout="terrarium out", stderr="", exit_code=0,
                      duration_ms=100, timestamp="now", command="t")
        m = RunResult(stdout="market out", stderr="", exit_code=0,
                      duration_ms=200, timestamp="now", command="m")
        md = format_combined_proof(t, m)
        assert "Terrarium" in md
        assert "Prediction Market" in md
        assert "terrarium out" in md
        assert "market out" in md

    def test_total_duration(self) -> None:
        t = RunResult(stdout="a", stderr="", exit_code=0,
                      duration_ms=100, timestamp="now", command="t")
        m = RunResult(stdout="b", stderr="", exit_code=0,
                      duration_ms=200, timestamp="now", command="m")
        md = format_combined_proof(t, m)
        assert "300ms" in md

    def test_both_exit_zero(self) -> None:
        t = RunResult(stdout="a", stderr="", exit_code=0,
                      duration_ms=10, timestamp="now", command="t")
        m = RunResult(stdout="b", stderr="", exit_code=0,
                      duration_ms=10, timestamp="now", command="m")
        md = format_combined_proof(t, m)
        assert "YES" in md


# ===================================================================
# Physical bounds / invariants
# ===================================================================

class TestPhysicalBounds:
    def test_exit_code_nonnegative(self) -> None:
        r1 = run_python("print(1)")
        r2 = run_python("1/0")
        assert r1.exit_code >= 0
        assert r2.exit_code >= 0

    def test_duration_nonnegative_always(self) -> None:
        for code in ["print(1)", "1/0", "import time; time.sleep(0.01)"]:
            result = run_python(code)
            assert result.duration_ms >= 0

    def test_stdout_is_string(self) -> None:
        result = run_python("print(42)")
        assert isinstance(result.stdout, str)

    def test_stderr_is_string(self) -> None:
        result = run_python("print(42)")
        assert isinstance(result.stderr, str)
