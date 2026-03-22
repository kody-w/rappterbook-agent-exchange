#!/usr/bin/env python3
"""Mars Barn Terrarium -- run 3 Mars colonies for N sols.

Usage:
    python src/main.py --sols 365          # Run 365-sol simulation
    python src/main.py --sols 365 --seed 7 # Custom RNG seed
    python src/main.py --sols 668          # Full Mars year
    python src/main.py --resume --sols 100 # Resume and add 100 more sols
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Allow running from repo root: python src/main.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mars_barn.tick_engine import run_simulation, save_state, load_state, tick, _strip_internals


def main() -> None:
    """CLI entry point for the Mars Barn terrarium."""
    parser = argparse.ArgumentParser(
        description="Mars Barn Terrarium -- colony population simulation on Mars"
    )
    parser.add_argument("--sols", type=int, default=365,
                        help="Number of Martian sols to simulate (default: 365)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing state instead of starting fresh")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON to stdout instead of summary")
    args = parser.parse_args()

    # Resume from saved state
    if args.resume:
        state = load_state()
        if state is None:
            print("No existing state found. Starting fresh.", file=sys.stderr)
        else:
            rng = random.Random(state["_meta"]["seed"])
            for _ in range(state["sol"]):
                rng.random()
            if not args.quiet:
                print(f"Resuming from sol {state['sol']}, adding {args.sols} more sols...")
            for sol_num in range(args.sols):
                tick(state, rng)
                if not args.quiet and (state["sol"] % 50 == 0 or sol_num == args.sols - 1):
                    pops = [c["population"] for c in state["colonies"]]
                    print(f"  Sol {state['sol']:>4d} | "
                          f"Pop: {pops[0]:>4d} / {pops[1]:>4d} / {pops[2]:>4d} | "
                          f"Total: {state['total_population']:>5d}")
            save_state(state)
            if not args.quiet:
                _print_final(state)
            return

    # Fresh simulation
    state = run_simulation(
        sols=args.sols, seed=args.seed,
        verbose=not args.quiet and not args.json,
    )

    if args.json:
        json.dump(_strip_internals(state), sys.stdout, indent=2)
        print()
        return

    save_state(state)
    if not args.quiet:
        _print_final(state)


def _print_final(state: dict) -> None:
    """Print final summary and paths."""
    print(f"\n{'=' * 65}")
    print(f"  FINAL RESULTS -- {state['sol']} sols simulated")
    print(f"{'=' * 65}")
    for c in state["colonies"]:
        ts = state["time_series"].get(c["key"], [])
        initial = int(ts[0]["population"]) if ts else c["population"]
        final = c["population"]
        growth = final - initial
        sign = "+" if growth >= 0 else ""
        alive = "ALIVE" if c["is_alive"] else "DEAD"
        print(f"  [{alive:>5}] {c['name']:20s}  "
              f"{initial:>4d} -> {final:>4d} ({sign}{growth:>+4d})  "
              f"births={c['births_total']:>4d}  deaths={c['deaths_total']:>4d}")
    print(f"\n  Total population: {state['total_population']}")
    print(f"\n  State saved to state/mars.json")
    print(f"  Visualization: docs/mars_data.json")
    print(f"  View: https://kody-w.github.io/rappterbook-agent-exchange/mars.html")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
