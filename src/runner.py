"""
runner.py — Code execution engine for the Mars Barn organism.

The organism's self-execution organ: run simulations, capture stdout,
format proof artifacts for posting to Discussions.

Usage:
    python -m src.runner terrarium
    python -m src.runner market
    python -m src.runner both
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RunResult:
    """Structured result of a code execution."""
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    timestamp: str
    command: str


def run_python(code: str, timeout: int = 60) -> RunResult:
    """Execute a Python code string in a subprocess, capture all output.

    Runs in the repo root so imports work. No eval/exec — always subprocess.
    """
    ts = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        elapsed = int((time.monotonic() - start) * 1000)
        return RunResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            duration_ms=elapsed,
            timestamp=ts,
            command=f"python -c '{code[:80]}...'" if len(code) > 80 else f"python -c '{code}'",
        )
    except subprocess.TimeoutExpired:
        elapsed = int((time.monotonic() - start) * 1000)
        return RunResult(
            stdout="",
            stderr=f"TimeoutExpired: code did not finish in {timeout}s",
            exit_code=124,
            duration_ms=elapsed,
            timestamp=ts,
            command=f"python -c (timeout={timeout}s)",
        )


def run_terrarium(sols: int = 365, seed: int = 42) -> RunResult:
    """Run the Mars Barn terrarium and capture output."""
    code = f"""
import sys, json
sys.path.insert(0, '.')
from src.tick_engine import Simulation
sim = Simulation(sols={sols}, env_seed={seed})
results = sim.run()
print('=' * 60)
print('TERRARIUM — {{}} colonies x {{}} sols (seed={seed})'.format(len(sim.colonies), {sols}))
print('=' * 60)
for s in results['summary']['colonies']:
    print(f"  {{s['name']}} ({{s['strategy']}})")
    print(f"    Pop: {{s['start_pop']}} -> {{s['end_pop']}} ({{s['growth_pct']:+.1f}}%)")
    print(f"    Peak: {{s['peak_pop']}}  Trough: {{s['min_pop']}}")
    print(f"    Births: {{s['total_births']}}  Deaths: {{s['total_deaths']}}")
    dc = {{k:v for k,v in s.get('death_causes',{{}}).items() if v > 0}}
    if dc:
        parts = ', '.join(f'{{k}}: {{v}}' for k,v in sorted(dc.items(), key=lambda x: -x[1]))
        print(f"    Causes: {{parts}}")
    print(f"    Techs: {{s['techs_unlocked']}}  Terraform: {{s['terraforming_output']:.6f}}")
tf = results['summary']['terraforming']
print(f"  Migrations: {{results['summary']['total_migrations']}}")
print(f"  Terraforming: {{tf['progress']*100:.4f}}% ({{tf['phase'] or 'none'}})")
for c in results['colonies']:
    tech = c.get('tech')
    if tech and tech.get('unlocked'):
        print(f"  Tech timeline [{{c['name']}}]:")
        for t in tech['unlocked']:
            print(f"    Sol {{t['sol']:>4}}: {{t['name']}} [{{t['branch']}}]")
print('=' * 60)
"""
    return run_python(code, timeout=120)


def run_market(n_predictions: int = 100, sols: int = 365, seeds: list | None = None) -> RunResult:
    """Run the prediction market and capture output."""
    seed_list = seeds or [42, 43, 44]
    code = f"""
import sys
sys.path.insert(0, '.')
from src.market_maker import run_market, _print_report
report = run_market(n_predictions={n_predictions}, sols={sols}, seeds={seed_list}, market_seed=0)
_print_report(report)
"""
    return run_python(code, timeout=120)


def format_proof(result: RunResult, title: str = "Execution Proof") -> str:
    """Format a RunResult as a markdown proof block for Discussions."""
    status = "SUCCESS" if result.exit_code == 0 else f"FAILED (exit {result.exit_code})"
    lines = [
        f"## {title}",
        "",
        f"**Status:** {status}  ",
        f"**Duration:** {result.duration_ms}ms  ",
        f"**Timestamp:** {result.timestamp}  ",
        f"**Command:** `{result.command}`",
        "",
    ]
    if result.stdout.strip():
        lines.extend([
            "### stdout",
            "```",
            result.stdout.strip(),
            "```",
            "",
        ])
    if result.stderr.strip():
        lines.extend([
            "### stderr",
            "```",
            result.stderr.strip(),
            "```",
            "",
        ])
    return "\n".join(lines)


def format_combined_proof(terrarium: RunResult, market: RunResult) -> str:
    """Format both results as a single proof artifact."""
    lines = [
        "# Execution Proof — Mars Barn + Prediction Market",
        "",
        f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        format_proof(terrarium, title="Terrarium (Mars Barn)"),
        "---",
        "",
        format_proof(market, title="Prediction Market"),
        "---",
        "",
        f"**Total duration:** {terrarium.duration_ms + market.duration_ms}ms  ",
        f"**Both exit 0:** {'YES' if terrarium.exit_code == 0 and market.exit_code == 0 else 'NO'}",
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.runner [terrarium|market|both]")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "terrarium":
        result = run_terrarium()
        print(format_proof(result))
    elif cmd == "market":
        result = run_market()
        print(format_proof(result))
    elif cmd == "both":
        t = run_terrarium()
        m = run_market()
        print(format_combined_proof(t, m))
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
