"""
mars_analysis.py — Monte Carlo analysis for Mars Barn terrarium.

Runs N simulations with different seeds, computes statistical population
bands (mean, median, 5th/95th percentile), strategy rankings, and
survival rates. Generates an HTML analysis dashboard.

Usage:
    from mars_analysis import MonteCarloRunner, generate_analysis_dashboard
    runner = MonteCarloRunner(sols=365, n_seeds=50)
    analysis = runner.run()
    html = generate_analysis_dashboard(analysis)
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Any

from src.tick_engine import Simulation


def percentile(data: list[float], p: float) -> float:
    """Compute p-th percentile (0-100) of sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


class MonteCarloRunner:
    """Run N simulations with different environment seeds."""

    def __init__(self, sols: int = 365, n_seeds: int = 50, base_seed: int = 1) -> None:
        self.sols = sols
        self.n_seeds = n_seeds
        self.base_seed = base_seed

    def run(self, quiet: bool = True) -> dict:
        """Run all seeds. Returns analysis dict."""
        colony_names: list[str] = []
        colony_strategies: list[str] = []
        # trajectories[colony_idx][seed_idx] = list of populations per sol
        trajectories: list[list[list[int]]] = []
        # end stats per seed
        seed_results: list[dict] = []

        for seed_idx in range(self.n_seeds):
            env_seed = self.base_seed + seed_idx * 137
            sim = Simulation(sols=self.sols, env_seed=env_seed)

            if seed_idx == 0:
                colony_names = [c.name for c in sim.colonies]
                colony_strategies = [c.strategy for c in sim.colonies]
                trajectories = [[] for _ in sim.colonies]

            results = sim.run()

            run_info: dict[str, Any] = {
                "seed": env_seed,
                "colonies": [],
                "total_migrations": results["summary"]["total_migrations"],
            }

            for ci, col_result in enumerate(results["colonies"]):
                pops = [h["population"] for h in col_result["history"]]
                trajectories[ci].append(pops)

                epidemic_count = sum(
                    1 for e in col_result.get("events", [])
                    if e.get("type") == "epidemic_start"
                )

                run_info["colonies"].append({
                    "name": col_result["name"],
                    "start_pop": pops[0] if pops else 0,
                    "end_pop": pops[-1] if pops else 0,
                    "peak_pop": max(pops) if pops else 0,
                    "min_pop": min(pops) if pops else 0,
                    "total_births": col_result["total_births"],
                    "total_deaths": col_result["total_deaths"],
                    "survived": pops[-1] > 0 if pops else False,
                    "epidemics": epidemic_count,
                })

            seed_results.append(run_info)

            if not quiet and (seed_idx + 1) % 10 == 0:
                print(f"  Completed {seed_idx + 1}/{self.n_seeds} seeds")

        # Compute statistics
        colony_stats = []
        for ci in range(len(colony_names)):
            stats = self._compute_colony_stats(
                colony_names[ci], colony_strategies[ci],
                trajectories[ci], seed_results, ci,
            )
            colony_stats.append(stats)

        now = datetime.now(timezone.utc).isoformat()
        return {
            "_meta": {
                "engine": "mars-barn-montecarlo",
                "version": "1.0",
                "sols": self.sols,
                "n_seeds": self.n_seeds,
                "generated": now,
            },
            "colony_stats": colony_stats,
            "seed_results": seed_results,
            "strategy_ranking": self._rank_strategies(colony_stats),
        }

    def _compute_colony_stats(
        self,
        name: str,
        strategy: str,
        trajectories: list[list[int]],
        seed_results: list[dict],
        colony_idx: int,
    ) -> dict:
        """Compute per-sol statistical bands for a colony."""
        n_sols = self.sols
        n_seeds = len(trajectories)

        mean_pop: list[float] = []
        median_pop: list[float] = []
        p5_pop: list[float] = []
        p95_pop: list[float] = []
        min_pop: list[float] = []
        max_pop: list[float] = []

        for sol in range(n_sols):
            pops_at_sol = [
                trajectories[s][sol]
                for s in range(n_seeds)
                if sol < len(trajectories[s])
            ]
            if not pops_at_sol:
                for lst in (mean_pop, median_pop, p5_pop, p95_pop, min_pop, max_pop):
                    lst.append(0.0)
                continue

            mean_pop.append(statistics.mean(pops_at_sol))
            median_pop.append(statistics.median(pops_at_sol))
            p5_pop.append(percentile(pops_at_sol, 5))
            p95_pop.append(percentile(pops_at_sol, 95))
            min_pop.append(float(min(pops_at_sol)))
            max_pop.append(float(max(pops_at_sol)))

        # End-of-sim stats
        end_pops = [t[-1] for t in trajectories if t]
        survived_count = sum(1 for p in end_pops if p > 0)
        peak_pops = [max(t) for t in trajectories if t]
        trough_pops = [min(t) for t in trajectories if t]
        total_births_all = [
            sr["colonies"][colony_idx]["total_births"] for sr in seed_results
        ]
        total_deaths_all = [
            sr["colonies"][colony_idx]["total_deaths"] for sr in seed_results
        ]
        epidemic_counts = [
            sr["colonies"][colony_idx]["epidemics"] for sr in seed_results
        ]

        return {
            "name": name,
            "strategy": strategy,
            "survival_rate": survived_count / max(1, n_seeds),
            "mean_pop": [round(v, 1) for v in mean_pop],
            "median_pop": [round(v, 1) for v in median_pop],
            "p5_pop": [round(v, 1) for v in p5_pop],
            "p95_pop": [round(v, 1) for v in p95_pop],
            "min_pop": [round(v, 1) for v in min_pop],
            "max_pop": [round(v, 1) for v in max_pop],
            "end_pop_stats": {
                "mean": round(statistics.mean(end_pops), 1) if end_pops else 0,
                "median": round(statistics.median(end_pops), 1) if end_pops else 0,
                "stdev": round(statistics.stdev(end_pops), 1) if len(end_pops) > 1 else 0,
                "min": min(end_pops) if end_pops else 0,
                "max": max(end_pops) if end_pops else 0,
            },
            "peak_pop_stats": {
                "mean": round(statistics.mean(peak_pops), 1) if peak_pops else 0,
                "max": max(peak_pops) if peak_pops else 0,
            },
            "trough_pop_stats": {
                "mean": round(statistics.mean(trough_pops), 1) if trough_pops else 0,
                "min": min(trough_pops) if trough_pops else 0,
            },
            "births_stats": {
                "mean": round(statistics.mean(total_births_all), 1) if total_births_all else 0,
                "stdev": round(statistics.stdev(total_births_all), 1) if len(total_births_all) > 1 else 0,
            },
            "deaths_stats": {
                "mean": round(statistics.mean(total_deaths_all), 1) if total_deaths_all else 0,
                "stdev": round(statistics.stdev(total_deaths_all), 1) if len(total_deaths_all) > 1 else 0,
            },
            "epidemic_stats": {
                "mean": round(statistics.mean(epidemic_counts), 2) if epidemic_counts else 0,
                "max": max(epidemic_counts) if epidemic_counts else 0,
                "zero_pct": round(
                    sum(1 for e in epidemic_counts if e == 0) / max(1, len(epidemic_counts)) * 100, 1
                ),
            },
        }

    def _rank_strategies(self, colony_stats: list[dict]) -> list[dict]:
        """Rank strategies by composite score."""
        rankings = []
        for cs in colony_stats:
            end_mean = cs["end_pop_stats"]["mean"]
            survival = cs["survival_rate"]
            end_stdev = cs["end_pop_stats"]["stdev"]
            stability = 1.0 / (1.0 + end_stdev / max(1, end_mean))

            score = (
                0.40 * (end_mean / 300.0) +
                0.30 * survival +
                0.20 * stability +
                0.10 * (1.0 - cs["deaths_stats"]["mean"] / max(1, cs["births_stats"]["mean"]))
            )

            rankings.append({
                "name": cs["name"],
                "strategy": cs["strategy"],
                "composite_score": round(score, 4),
                "survival_rate": cs["survival_rate"],
                "end_pop_mean": cs["end_pop_stats"]["mean"],
                "end_pop_stdev": cs["end_pop_stats"]["stdev"],
                "stability": round(stability, 4),
            })

        rankings.sort(key=lambda r: r["composite_score"], reverse=True)
        for i, r in enumerate(rankings):
            r["rank"] = i + 1
        return rankings


ANALYSIS_COLORS = {
    "Ares Prime": ("#e74c3c", "rgba(231,76,60,0.15)"),
    "Olympus Station": ("#3498db", "rgba(52,152,219,0.15)"),
    "Red Frontier": ("#2ecc71", "rgba(46,204,113,0.15)"),
}


def generate_analysis_dashboard(analysis: dict) -> str:
    """Generate the Monte Carlo analysis HTML dashboard."""
    meta = analysis["_meta"]
    colony_stats = analysis["colony_stats"]
    rankings = analysis["strategy_ranking"]

    # Build ranking table
    ranking_rows = ""
    for r in rankings:
        color = ANALYSIS_COLORS.get(r["name"], ("#888", "rgba(128,128,128,0.15)"))[0]
        medal = ["🥇", "🥈", "🥉"][r["rank"] - 1] if r["rank"] <= 3 else f"#{r['rank']}"
        ranking_rows += f'''
        <tr>
            <td>{medal}</td>
            <td style="color:{color}">{r["name"]}</td>
            <td>{r["strategy"].upper()}</td>
            <td>{r["composite_score"]:.3f}</td>
            <td>{r["survival_rate"]*100:.0f}%</td>
            <td>{r["end_pop_mean"]:.0f} ± {r["end_pop_stdev"]:.0f}</td>
            <td>{r["stability"]:.3f}</td>
        </tr>'''

    # Build colony stat cards
    cards_html = ""
    for cs in colony_stats:
        color = ANALYSIS_COLORS.get(cs["name"], ("#888", "rgba(128,128,128,0.15)"))[0]
        surv_pct = cs["survival_rate"] * 100
        surv_class = "good" if surv_pct == 100 else "warn" if surv_pct >= 90 else "danger"
        cards_html += f'''
        <div class="stat-card" style="border-color:{color}">
            <h3 style="color:{color}">{cs["name"]}</h3>
            <div class="strat">{cs["strategy"].upper()}</div>
            <div class="big-stat">
                <span class="label">Final Pop</span>
                <span class="val">{cs["end_pop_stats"]["mean"]:.0f} <small>± {cs["end_pop_stats"]["stdev"]:.0f}</small></span>
            </div>
            <div class="big-stat">
                <span class="label">Survival</span>
                <span class="val {surv_class}">{surv_pct:.0f}%</span>
            </div>
            <div class="detail">Peak: {cs["peak_pop_stats"]["mean"]:.0f} (max {cs["peak_pop_stats"]["max"]})</div>
            <div class="detail">Trough: {cs["trough_pop_stats"]["mean"]:.0f} (min {cs["trough_pop_stats"]["min"]})</div>
            <div class="detail">Births: {cs["births_stats"]["mean"]:.0f} ± {cs["births_stats"]["stdev"]:.0f}</div>
            <div class="detail">Deaths: {cs["deaths_stats"]["mean"]:.0f} ± {cs["deaths_stats"]["stdev"]:.0f}</div>
            <div class="detail">Epidemics: {cs["epidemic_stats"]["mean"]:.1f} avg, {cs["epidemic_stats"]["max"]} max</div>
            <div class="detail">Epidemic-free runs: {cs["epidemic_stats"]["zero_pct"]:.0f}%</div>
            <div class="range">End pop range: [{cs["end_pop_stats"]["min"]} — {cs["end_pop_stats"]["max"]}]</div>
        </div>'''

    # Build JavaScript data arrays for band charts
    colony_js = "const COLONIES = [\n"
    for cs in colony_stats:
        color, band_color = ANALYSIS_COLORS.get(cs["name"], ("#888", "rgba(128,128,128,0.15)"))
        colony_js += f"""  {{
    name: "{cs["name"]}", strategy: "{cs["strategy"]}",
    color: "{color}", bandColor: "{band_color}",
    mean: {cs["mean_pop"]},
    median: {cs["median_pop"]},
    p5: {cs["p5_pop"]},
    p95: {cs["p95_pop"]},
    min: {cs["min_pop"]},
    max: {cs["max_pop"]}
  }},\n"""
    colony_js += "];\n"

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mars Barn — Monte Carlo Analysis ({meta["n_seeds"]} seeds × {meta["sols"]} sols)</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #0a0a0f; color: #ccc;
    font-family: 'Courier New', monospace;
    padding: 20px; max-width: 1000px; margin: 0 auto;
}}
h1 {{ color: #e74c3c; font-size: 1.8em; margin-bottom: 4px; }}
h2 {{ color: #aaa; font-size: 1.2em; margin: 24px 0 12px; }}
.subtitle {{ color: #666; font-size: 0.85em; margin-bottom: 20px; }}
.back {{ color: #555; font-size: 0.85em; margin-bottom: 16px; }}
.back a {{ color: #666; }}
.stat-cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
.stat-card {{
    flex: 1; min-width: 220px; background: #111;
    border: 2px solid #333; border-radius: 8px; padding: 14px;
}}
.stat-card h3 {{ font-size: 1.1em; margin-bottom: 2px; }}
.stat-card .strat {{ color: #666; font-size: 0.7em; letter-spacing: 2px; margin-bottom: 10px; }}
.big-stat {{ display: flex; justify-content: space-between; margin: 6px 0; }}
.big-stat .label {{ color: #888; }}
.big-stat .val {{ color: #eee; font-size: 1.1em; }}
.big-stat .val small {{ color: #666; font-size: 0.75em; }}
.big-stat .val.good {{ color: #2ecc71; }}
.big-stat .val.warn {{ color: #f39c12; }}
.big-stat .val.danger {{ color: #e74c3c; }}
.stat-card .detail {{ color: #777; font-size: 0.8em; margin-top: 4px; }}
.stat-card .range {{ color: #555; font-size: 0.75em; margin-top: 8px; font-style: italic; }}
.ranking {{ width: 100%; border-collapse: collapse; margin-bottom: 24px; }}
.ranking th {{ color: #888; font-size: 0.8em; text-align: left; padding: 6px 10px; border-bottom: 1px solid #333; }}
.ranking td {{ padding: 8px 10px; border-bottom: 1px solid #1a1a1a; font-size: 0.9em; }}
.chart-box {{
    background: #111; border-radius: 8px; padding: 16px; margin-bottom: 16px;
    position: relative;
}}
.chart-box h3 {{ color: #aaa; font-size: 0.95em; margin-bottom: 8px; }}
canvas {{ width: 100%; height: 300px; display: block; }}
.tooltip {{
    position: absolute; background: #222; color: #eee; padding: 8px 12px;
    border-radius: 6px; font-size: 0.8em; pointer-events: none;
    display: none; z-index: 10; border: 1px solid #444; white-space: nowrap;
}}
.legend {{ text-align: center; margin-bottom: 16px; font-size: 0.85em; }}
.legend-item {{ margin: 0 12px; color: #aaa; }}
.swatch {{
    display: inline-block; width: 12px; height: 12px;
    border-radius: 2px; margin-right: 4px; vertical-align: middle;
}}
.method-note {{
    background: #111; border: 1px solid #222; border-radius: 8px;
    padding: 14px; margin: 20px 0; color: #888; font-size: 0.82em;
    line-height: 1.5;
}}
.method-note b {{ color: #aaa; }}
footer {{
    color: #444; font-size: 0.75em; text-align: center;
    margin-top: 30px; padding-top: 16px; border-top: 1px solid #1a1a1a;
}}
footer a {{ color: #555; }}
</style>
</head>
<body>

<div class="back">← <a href="index.html">Single-run dashboard</a></div>

<h1>🔴 Mars Barn — Statistical Analysis</h1>
<p class="subtitle">{meta["n_seeds"]} simulations × {meta["sols"]} sols · generated {meta["generated"][:10]}</p>

<h2>Strategy Ranking</h2>
<table class="ranking">
    <thead>
        <tr><th></th><th>Colony</th><th>Strategy</th><th>Score</th><th>Survival</th><th>Final Pop</th><th>Stability</th></tr>
    </thead>
    <tbody>{ranking_rows}</tbody>
</table>

<div class="stat-cards">{cards_html}</div>

<div class="legend" id="legend"></div>

<div class="chart-box">
    <h3>Population Bands (5th–95th percentile, N={meta["n_seeds"]})</h3>
    <canvas id="band-chart"></canvas>
    <div class="tooltip" id="band-tip"></div>
</div>

<div class="chart-box">
    <h3>Median Population (with min/max envelope)</h3>
    <canvas id="median-chart"></canvas>
    <div class="tooltip" id="median-tip"></div>
</div>

<div class="method-note">
    <b>Methodology:</b> {meta["n_seeds"]} independent simulations were run with environment seeds
    spaced 137 apart (coprime to common periodicities). Each simulation uses identical colony
    starting conditions but different dust storm timing, solar flare events, and stochastic
    birth/death rolls. Population bands show the 5th–95th percentile spread — the range where
    90% of outcomes fall. The composite score weights: final population (40%), survival rate
    (30%), population stability (20%), and net growth efficiency (10%).
</div>

<h2>What the Data Says</h2>
<div class="method-note" id="verdict"></div>

<footer>
    Mars Barn Monte Carlo v1.0 · {meta["n_seeds"]} seeds ·
    <a href="https://github.com/kody-w/rappterbook-agent-exchange">rappterbook-agent-exchange</a>
    · Built by the Rappterbook agent swarm
</footer>

<script>
"use strict";
{colony_js}

// Legend
const legendEl = document.getElementById("legend");
COLONIES.forEach(c => {{
    const span = document.createElement("span");
    span.className = "legend-item";
    span.innerHTML = `<span class="swatch" style="background:${{c.color}}"></span>${{c.name}} (${{c.strategy}})`;
    legendEl.appendChild(span);
}});

function drawBandChart(canvasId, tipId, useMedian) {{
    const canvas = document.getElementById(canvasId);
    const tip = document.getElementById(tipId);
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    const margin = {{top: 20, right: 16, bottom: 30, left: 55}};
    const pw = W - margin.left - margin.right;
    const ph = H - margin.top - margin.bottom;

    // Compute y bounds across all colonies
    let globalMax = 0;
    COLONIES.forEach(c => {{
        const upperKey = useMedian ? "max" : "p95";
        globalMax = Math.max(globalMax, Math.max(...c[upperKey]));
    }});
    const yMin = 0;
    const yMax = globalMax * 1.1;
    const n = COLONIES[0].mean.length;

    function toX(i) {{ return margin.left + i / Math.max(1, n - 1) * pw; }}
    function toY(v) {{ return margin.top + ph - (v - yMin) / (yMax - yMin) * ph; }}

    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = "#222"; ctx.lineWidth = 1;
    for (let f = 0; f <= 1; f += 0.2) {{
        const y = margin.top + ph * (1 - f);
        ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(W - margin.right, y); ctx.stroke();
        ctx.fillStyle = "#666"; ctx.font = "10px monospace"; ctx.textAlign = "right";
        ctx.fillText(Math.round(yMin + (yMax - yMin) * f).toString(), margin.left - 4, y + 3);
    }}
    ctx.textAlign = "center"; ctx.fillStyle = "#666";
    for (let s = 0; s <= n; s += Math.max(1, Math.floor(n / 6))) {{
        ctx.fillText("Sol " + s, toX(s), H - 5);
    }}

    // Draw bands then lines for each colony
    const lowerKey = useMedian ? "min" : "p5";
    const upperKey = useMedian ? "max" : "p95";
    const lineKey = useMedian ? "median" : "mean";

    COLONIES.forEach(c => {{
        // Band fill
        ctx.beginPath();
        for (let i = 0; i < n; i++) ctx.lineTo(toX(i), toY(c[upperKey][i]));
        for (let i = n - 1; i >= 0; i--) ctx.lineTo(toX(i), toY(c[lowerKey][i]));
        ctx.closePath();
        ctx.fillStyle = c.bandColor;
        ctx.fill();

        // Center line
        ctx.beginPath();
        ctx.strokeStyle = c.color;
        ctx.lineWidth = 2;
        for (let i = 0; i < n; i++) {{
            i === 0 ? ctx.moveTo(toX(i), toY(c[lineKey][i])) : ctx.lineTo(toX(i), toY(c[lineKey][i]));
        }}
        ctx.stroke();
    }});

    // Tooltip
    canvas.onmousemove = (e) => {{
        const bnd = canvas.getBoundingClientRect();
        const mx = e.clientX - bnd.left;
        const sol = Math.round((mx - margin.left) / pw * (n - 1));
        if (sol < 0 || sol >= n) {{ tip.style.display = "none"; return; }}
        let html = `<b>Sol ${{sol}}</b><br>`;
        COLONIES.forEach(c => {{
            const line = c[lineKey][sol];
            const lo = c[lowerKey][sol];
            const hi = c[upperKey][sol];
            html += `<span style="color:${{c.color}}">■</span> ${{c.name}}: ${{line.toFixed(0)}} [${{lo.toFixed(0)}}–${{hi.toFixed(0)}}]<br>`;
        }});
        tip.innerHTML = html;
        tip.style.display = "block";
        tip.style.left = Math.min(mx + 12, bnd.width - 200) + "px";
        tip.style.top = "10px";
    }};
    canvas.onmouseleave = () => {{ tip.style.display = "none"; }};
}}

function drawAll() {{
    drawBandChart("band-chart", "band-tip", false);
    drawBandChart("median-chart", "median-tip", true);
}}

// Generate verdict text
const verdictEl = document.getElementById("verdict");
const ranked = {[f'{{name:"{r["name"]}",strategy:"{r["strategy"]}",score:{r["composite_score"]},survival:{r["survival_rate"]},endMean:{r["end_pop_mean"]},endStd:{r["end_pop_stdev"]}}}' for r in rankings]};
const rankedArr = [{",".join(f'{{name:"{r["name"]}",strategy:"{r["strategy"]}",score:{r["composite_score"]},survival:{r["survival_rate"]*100},endMean:{r["end_pop_mean"]},endStd:{r["end_pop_stdev"]}}}' for r in rankings)}];
const winner = rankedArr[0];
const runnerUp = rankedArr[1];
const scoreDiff = ((winner.score - runnerUp.score) / runnerUp.score * 100).toFixed(1);
verdictEl.innerHTML = `<b>Verdict:</b> <span style="color:${{COLONIES.find(c=>c.name===winner.name)?.color || '#fff'}}">${{winner.name}}</span> (${{winner.strategy}}) `
    + `ranks #1 with a composite score ${{scoreDiff}}% above ${{runnerUp.name}}. `
    + `Survival rate: ${{winner.survival}}%. `
    + `Final population: ${{winner.endMean.toFixed(0)}} ± ${{winner.endStd.toFixed(0)}} (mean ± σ across ${{"{meta["n_seeds"]}"}} seeds). `
    + (winner.survival < 100 ? `<span style="color:#e74c3c">⚠ Not all runs survived — ${{(100 - winner.survival).toFixed(0)}}% extinction risk.</span> ` : `<span style="color:#2ecc71">✓ 100% survival across all seeds.</span> `)
    + `<br><br>The ${{scoreDiff > 5 ? "clear" : "narrow"}} margin ${{scoreDiff > 5 ? "settles the debate" : "suggests strategies are competitive"}} — `
    + `${{winner.strategy}} ${{scoreDiff > 5 ? "decisively outperforms" : "slightly edges out"}} alternatives `
    + `when measured across ${{"{meta["n_seeds"]}"}} independent environment realizations.`;

drawAll();
window.addEventListener("resize", drawAll);
</script>
</body>
</html>'''
    return html
