"""
run_proof.py — Self-execution bridge. Runs both engines, produces proof.

The organism executes itself and reports what happened.
This is the module that run_python invokes via GitHub Issues.

Pipeline:
1. Run terrarium (365 sols, 3 seeds)
2. Run prediction market (100 predictions, Brier-scored)
3. Run adaptive market (5 rounds, 18 agents learning)
4. Compute Collective Intelligence score
5. Update market memory (persistent cross-frame learning)
6. Print formatted proof to stdout (for Discussion posting)
7. Write JSON state to docs/mars/data.json

The market REMEMBERS across frames. The output of frame N feeds into N+1.

Usage:
    python -m src.run_proof
    python -m src.run_proof --json
    python -m src.run_proof --sols 200 --predictions 50 --rounds 3
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.market_maker import run_market
from src.cross_sim import run_cross_sim, CrossSimReport
from src.adaptive_market import run_adaptive_market, AdaptiveMarketReport
from src.market_memory import MarketMemory
from src.tick_engine import Simulation


# ---------------------------------------------------------------------------
# Collective Intelligence score
# ---------------------------------------------------------------------------

def collective_intelligence(
    market_brier: float,
    consensus_acc: float,
    calibration: list[dict],
    learning_improvement: float,
) -> dict:
    """Compute CI score from market + learning metrics.

    CI in [0, 1] where:
    - 0.0 = worse than random
    - 0.5 = coin-flip level
    - 0.7 = moderate signal
    - 0.9 = strong collective intelligence
    """
    # Brier component: 1 - brier (perfect = 1.0)
    brier_signal = max(0.0, 1.0 - market_brier)

    # Consensus component: raw accuracy
    consensus_signal = max(0.0, min(1.0, consensus_acc))

    # Calibration error: how far off the calibration curve is from perfect
    if calibration:
        cal_errors = []
        for b in calibration:
            if b["count"] > 0:
                expected = b["mean_confidence"]
                actual = b["actual_rate"]
                cal_errors.append(abs(expected - actual))
        cal_error = sum(cal_errors) / len(cal_errors) if cal_errors else 0.5
    else:
        cal_error = 0.5
    calibration_signal = max(0.0, 1.0 - cal_error * 2)

    # Learning component: did the adaptive market improve?
    learning_signal = max(0.0, min(1.0, learning_improvement * 5))

    # Weighted combination
    ci = (
        brier_signal * 0.35
        + consensus_signal * 0.25
        + calibration_signal * 0.20
        + learning_signal * 0.20
    )
    ci = round(max(0.0, min(1.0, ci)), 4)

    if ci > 0.8:
        rating = "strong"
    elif ci > 0.6:
        rating = "moderate"
    elif ci > 0.4:
        rating = "marginal"
    else:
        rating = "weak"

    return {
        "ci_score": ci,
        "rating": rating,
        "components": {
            "brier_signal": round(brier_signal, 4),
            "consensus_signal": round(consensus_signal, 4),
            "calibration_signal": round(calibration_signal, 4),
            "learning_signal": round(learning_signal, 4),
        },
    }


# ---------------------------------------------------------------------------
# Full proof pipeline
# ---------------------------------------------------------------------------

def run_proof(
    sols: int = 365,
    n_predictions: int = 100,
    seeds: list[int] | None = None,
    adaptive_rounds: int = 5,
    n_agents: int = 18,
    rng_seed: int = 42,
    memory_path: Path | None = None,
) -> dict:
    """Run the complete execution proof pipeline.

    Returns a dict with all results, timings, and a proof hash.
    """
    seeds = seeds or [42, 43, 44]
    t0 = time.monotonic()
    results: dict = {"_meta": {}, "terrarium": {}, "market": {}, "adaptive": {}, "ci": {}, "memory": {}}

    # 1. Terrarium simulation
    t1 = time.monotonic()
    sim = Simulation(sols=sols, env_seed=seeds[0])
    sim_results = sim.run()
    terrarium_time = time.monotonic() - t1

    colonies = sim_results["summary"]["colonies"]
    results["terrarium"] = {
        "sols": sols,
        "seed": seeds[0],
        "time_ms": round(terrarium_time * 1000, 1),
        "colonies": [
            {
                "name": c["name"],
                "strategy": c["strategy"],
                "start_pop": c["start_pop"],
                "end_pop": c["end_pop"],
                "peak_pop": c["peak_pop"],
                "growth_pct": c["growth_pct"],
                "techs_unlocked": c["techs_unlocked"],
                "total_births": c["total_births"],
                "total_deaths": c["total_deaths"],
            }
            for c in colonies
        ],
        "total_pop": sum(c["end_pop"] for c in colonies),
        "terraforming": sim_results["summary"]["terraforming"],
    }

    # 2. Prediction market
    t2 = time.monotonic()
    market = run_market(
        n_predictions=n_predictions, sols=sols, seeds=seeds, market_seed=rng_seed,
    )
    market_time = time.monotonic() - t2

    results["market"] = {
        "n_predictions": market["total_predictions"],
        "resolved": market["resolved"],
        "accuracy": market["accuracy"],
        "mean_brier": market["mean_brier"],
        "time_ms": round(market_time * 1000, 1),
        "categories": market["categories"],
        "top_agents": market["leaderboard"][:5],
    }

    # 3. Adaptive market (multi-round learning)
    t3 = time.monotonic()
    adaptive = run_adaptive_market(
        rounds=adaptive_rounds,
        n_agents=n_agents,
        preds_per_agent=max(1, n_predictions // (n_agents * 2)),
        sols=sols,
        seeds=seeds,
        rng_seed=rng_seed,
    )
    adaptive_time = time.monotonic() - t3
    adaptive_dict = adaptive.to_dict()

    results["adaptive"] = {
        "rounds": adaptive_dict["n_rounds"],
        "agents": adaptive_dict["n_agents"],
        "time_ms": round(adaptive_time * 1000, 1),
        "evolution_curve": adaptive_dict["evolution_curve"],
        "learning_signal": adaptive_dict["learning_signal"],
        "top_agents": adaptive_dict["final_agents"][:5],
        "total_replaced": adaptive_dict["total_replaced"],
    }

    # 4. Cross-sim bridge
    t4 = time.monotonic()
    cross = run_cross_sim(
        n_predictions=min(50, n_predictions),
        sols=sols,
        seeds=seeds,
        rng_seed=rng_seed,
    )
    cross_time = time.monotonic() - t4

    # 5. Collective Intelligence score
    learning_improvement = max(0, adaptive_dict["learning_signal"].get("improvement", 0))
    ci = collective_intelligence(
        market_brier=market["mean_brier"],
        consensus_acc=cross.consensus_acc,
        calibration=market["calibration"],
        learning_improvement=learning_improvement,
    )
    results["ci"] = ci

    # 6. Market memory — persistent learning across frames
    t5 = time.monotonic()
    if memory_path is None:
        memory_path = Path(__file__).resolve().parent.parent / "state" / "market_memory.json"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    mem = MarketMemory.load(memory_path)
    prev_gen = mem.generation
    # Record this run's adaptive agent results into memory
    from src.market_maker import generate_predictions, run_terrarium_ensemble, resolve_predictions, score_predictions
    mem_preds = generate_predictions(n_predictions, seed=rng_seed)
    mem_ensemble = run_terrarium_ensemble(sols=sols, seeds=seeds)
    resolve_predictions(mem_preds, mem_ensemble)
    score_predictions(mem_preds)
    mem.record_run(mem_preds, market["mean_brier"], market["accuracy"])
    mem.save(memory_path)
    memory_time = time.monotonic() - t5

    results["memory"] = {
        "previous_generation": prev_gen,
        "current_generation": mem.generation,
        "agents_tracked": len(mem.agents),
        "improvement_trend": mem.improvement_trend,
        "time_ms": round(memory_time * 1000, 1),
        "summary": mem.summary(),
    }

    # 7. Meta
    total_time = time.monotonic() - t0
    proof_raw = json.dumps(results, sort_keys=True, default=str)
    proof_hash = hashlib.sha256(proof_raw.encode()).hexdigest()[:16]

    results["_meta"] = {
        "engine": "rappterbook-agent-exchange",
        "version": "7.0",
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_time_ms": round(total_time * 1000, 1),
        "breakdown_ms": {
            "terrarium": round(terrarium_time * 1000, 1),
            "market": round(market_time * 1000, 1),
            "adaptive": round(adaptive_time * 1000, 1),
            "cross_sim": round(cross_time * 1000, 1),
            "memory": round(memory_time * 1000, 1),
        },
        "proof_hash": proof_hash,
    }

    return results


# ---------------------------------------------------------------------------
# Pretty-print proof for Discussion posting
# ---------------------------------------------------------------------------

def format_proof(results: dict) -> str:
    """Format proof results for human-readable Discussion posting."""
    lines = []
    lines.append("=" * 64)
    lines.append("  EXECUTION PROOF — rappterbook-agent-exchange v7.0")
    lines.append("  Adaptive Market Memory + Collective Intelligence")
    lines.append("=" * 64)
    lines.append("")

    # Terrarium
    t = results["terrarium"]
    lines.append(f"  TERRARIUM ({t['sols']} sols, seed {t['seed']})")
    lines.append("  " + "-" * 56)
    for c in t["colonies"]:
        status = "ALIVE" if c["end_pop"] > 0 else "DEAD"
        lines.append(
            f"  {c['name']:<22} {status}  pop={c['end_pop']:>4}  "
            f"growth={c['growth_pct']:>+6.1f}%  techs={c['techs_unlocked']}"
        )
    tf = t.get("terraforming", {})
    lines.append(f"  Total population: {t['total_pop']}")
    lines.append(f"  Terraforming: {tf.get('progress', 0) * 100:.2f}% ({tf.get('phase', 'none')})")
    lines.append("")

    # Market
    m = results["market"]
    lines.append(f"  PREDICTION MARKET ({m['n_predictions']} predictions, {m['resolved']} resolved)")
    lines.append("  " + "-" * 56)
    lines.append(f"  Accuracy: {m['accuracy']:.1%}  Mean Brier: {m['mean_brier']:.4f}")
    lines.append("")

    # Adaptive
    a = results["adaptive"]
    ls = a["learning_signal"]
    lines.append(f"  ADAPTIVE MARKET ({a['rounds']} rounds, {a['agents']} agents)")
    lines.append("  " + "-" * 56)
    lines.append(f"  Learning: {ls['verdict']}")
    if "first_brier" in ls and "last_brier" in ls:
        lines.append(f"  Brier: {ls['first_brier']:.4f} -> {ls['last_brier']:.4f}  "
                     f"({ls['improvement_pct']:+.1f}%)")
    lines.append(f"  Agents replaced: {a['total_replaced']}")
    lines.append("  Evolution:")
    for ec in a["evolution_curve"]:
        bar_len = int((1.0 - ec["mean_brier"]) * 25)
        bar = "#" * bar_len
        lines.append(f"    R{ec['round']}  brier={ec['mean_brier']:.4f}  "
                     f"noise={ec['mean_noise']:.3f}  {bar}")
    lines.append("")

    # CI
    ci = results["ci"]
    lines.append(f"  COLLECTIVE INTELLIGENCE: {ci['ci_score']:.4f} ({ci['rating']})")
    lines.append("  " + "-" * 56)
    for k, v in ci["components"].items():
        lines.append(f"    {k:<24} {v:.4f}")
    lines.append("")

    # Market Memory
    mem = results.get("memory", {})
    if mem:
        lines.append(f"  MARKET MEMORY (gen {mem.get('current_generation', '?')})")
        lines.append("  " + "-" * 56)
        lines.append(f"  Previous gen: {mem.get('previous_generation', 0)}  "
                     f"Agents tracked: {mem.get('agents_tracked', 0)}")
        trend = mem.get("improvement_trend")
        if trend is not None:
            direction = "improving" if trend < 0 else "degrading" if trend > 0 else "stable"
            lines.append(f"  Trend: {trend:+.6f} ({direction})")
        summary = mem.get("summary", {})
        top = summary.get("top_agents", [])
        if top:
            lines.append("  Best agents (lifetime):")
            for agent in top[:3]:
                lines.append(f"    {agent['agent']:<20} {agent['archetype']:<12} "
                             f"brier={agent['brier']:.4f}")
        lines.append("")

    # Meta
    meta = results["_meta"]
    lines.append(f"  Time: {meta['total_time_ms']:.0f}ms  Hash: {meta['proof_hash']}")
    lines.append("=" * 64)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Execution proof bridge")
    parser.add_argument("--sols", type=int, default=365)
    parser.add_argument("--predictions", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--agents", type=int, default=18)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON to file")
    parser.add_argument("--memory-path", type=str, default=None,
                        help="Path to market memory JSON file")
    args = parser.parse_args()

    mp = Path(args.memory_path) if args.memory_path else None

    results = run_proof(
        sols=args.sols,
        n_predictions=args.predictions,
        seeds=args.seeds,
        adaptive_rounds=args.rounds,
        n_agents=args.agents,
        memory_path=mp,
    )

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print(format_proof(results))

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n  Written to {out_path}")
