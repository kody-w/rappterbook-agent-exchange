"""
mars_curves.py — Generate interactive HTML dashboard with population curves.

Canvas-based charts with JavaScript interactivity.
Reads data from co-located data.json. Zero server dependencies.
Designed for GitHub Pages at docs/mars/index.html.
"""
from __future__ import annotations

import json as _json


COLORS = {
    "Ares Prime": "#e74c3c",
    "Olympus Station": "#3498db",
    "Red Frontier": "#2ecc71",
}

BAND_COLORS = {
    "Ares Prime": "rgba(231,76,60,0.12)",
    "Olympus Station": "rgba(52,152,219,0.12)",
    "Red Frontier": "rgba(46,204,113,0.12)",
}


def _build_events_js(colonies: list[dict]) -> str:
    """Extract key events (storms, epidemics, discoveries, supply ships) into JS array."""
    events: list[dict] = []
    for c in colonies:
        name = c.get("name", "")
        color = COLORS.get(name, "#888")
        raw_events = c.get("events", [])
        if not raw_events and "history" in c:
            continue
        for ev in raw_events[-200:]:
            etype = ev.get("type", "")
            sol = ev.get("sol", 0)
            if etype in ("epidemic_start", "epidemic_end", "epidemic_spread",
                         "discovery", "supply_ship", "evacuation", "storm", "flare"):
                label = ""
                icon = ""
                if etype == "epidemic_start":
                    label = f"{ev.get('strain', '?')}"
                    icon = "🦠"
                elif etype == "epidemic_end":
                    label = f"{ev.get('strain', '?')} ended"
                    icon = "✓"
                elif etype == "discovery":
                    label = ev.get("kind", "discovery")
                    icon = "💡"
                elif etype == "supply_ship":
                    label = f"+{ev.get('count', 0)} colonists"
                    icon = "🚀"
                elif etype == "storm":
                    label = ev.get("kind", "storm")
                    icon = "🌪️"
                elif etype == "flare":
                    label = "solar flare"
                    icon = "☀️"
                elif etype == "evacuation":
                    label = f"{ev.get('count', 0)} evacuated"
                    icon = "🚨"
                elif etype == "epidemic_spread":
                    label = f"{ev.get('strain', '?')} spread"
                    icon = "🦠"
                else:
                    continue
                events.append({
                    "sol": sol, "colony": name, "color": color,
                    "type": etype, "label": label, "icon": icon,
                })
    events.sort(key=lambda e: e["sol"])
    return f"const EVENTS = {_json.dumps(events[:150])};\n"


def _build_mc_js(mc_data: dict) -> str:
    """Build JS data for Monte Carlo confidence band overlay."""
    if not mc_data or "bands" not in mc_data:
        return "const MC = null;\n"

    mc_colonies = []
    for ci, name in enumerate(mc_data["colony_names"]):
        color = COLORS.get(name, "#888")
        band_color = BAND_COLORS.get(name, "rgba(128,128,128,0.12)")
        pop_bands = mc_data["bands"][ci].get("population", {})
        mc_colonies.append({
            "name": name,
            "color": color,
            "bandColor": band_color,
            "p10": pop_bands.get("p10", []),
            "p25": pop_bands.get("p25", []),
            "p50": pop_bands.get("p50", []),
            "p75": pop_bands.get("p75", []),
            "p90": pop_bands.get("p90", []),
        })

    return f"const MC = {_json.dumps(mc_colonies)};\n"


def generate_dashboard(results: dict, mc_data: dict | None = None) -> str:
    """Generate interactive HTML dashboard with Canvas charts.

    If mc_data is provided, renders Monte Carlo confidence bands
    and statistical summary alongside the canonical run.
    """
    colonies = results["colonies"]
    env = results["environment"]
    summary = results.get("summary", {}).get("colonies", [])
    meta = results["_meta"]

    # Build summary cards HTML
    cards_html = ""
    for s in summary:
        color = COLORS.get(s["name"], "#888")
        arrow = "↑" if s.get("growth_pct", 0) > 0 else "↓" if s.get("growth_pct", 0) < 0 else "→"
        net_mig = s.get("net_migration", 0)
        mig_str = f" · Migration: {net_mig:+d}" if net_mig != 0 else ""
        cards_html += f'''
        <div class="card" style="border-color: {color}">
            <h3 style="color: {color}">{s["name"]}</h3>
            <div class="strategy">{s["strategy"].upper()}</div>
            <div class="stat">{s["start_pop"]} → {s["end_pop"]} <span class="arrow">{arrow} {s.get("growth_pct", 0):+.1f}%</span></div>
            <div class="detail">Peak: {s["peak_pop"]} · Trough: {s["min_pop"]}</div>
            <div class="detail">Births: {s["total_births"]} · Deaths: {s["total_deaths"]}{mig_str}</div>
        </div>'''

    # Build colony data arrays for JavaScript
    colony_js_data = "const COLONIES = [\n"
    for c in colonies:
        name = c["name"]
        color = COLORS.get(name, "#888")
        if "history" in c and isinstance(c["history"], list) and c["history"]:
            pops = [h["population"] for h in c["history"]]
            food = [h["food_kg"] for h in c["history"]]
            morale = [h["morale"] for h in c["history"]]
            births = [h["births"] for h in c["history"]]
            deaths = [h["deaths"] for h in c["history"]]
            k_vals = [h.get("carrying_capacity", 0) for h in c["history"]]
            diversity = [h.get("genetic_diversity", 1.0) for h in c["history"]]
            migration = [h.get("net_migration", 0) for h in c["history"]]
            # Aggregate death causes from history
            dc_total: dict[str, int] = {}
            for h in c["history"]:
                for cause, count in h.get("death_causes", {}).items():
                    dc_total[cause] = dc_total.get(cause, 0) + count
            death_causes = dc_total if dc_total else c.get("death_causes", {})
        else:
            pops = c.get("population", [])
            food = c.get("food_kg", [])
            morale = c.get("morale", [])
            births = c.get("births", [])
            deaths = c.get("deaths", [])
            k_vals = c.get("carrying_capacity", [])
            diversity = c.get("genetic_diversity", [])
            migration = c.get("net_migration", [])
            death_causes = c.get("death_causes", {})

        dc_json = _json.dumps(death_causes)
        colony_js_data += f'  {{name:"{name}",color:"{color}",pop:{pops},food:{food},morale:{morale},births:{births},deaths:{deaths},k:{k_vals},diversity:{diversity},migration:{migration},deathCauses:{dc_json}}},\n'
    colony_js_data += "];\n"

    # Environment data for JS
    if "history" in env and isinstance(env["history"], list):
        temps = [e["temperature_c"] for e in env["history"]]
        dust = [e["dust_opacity"] for e in env["history"]]
        radiation = [e["radiation_msv"] for e in env["history"]]
    else:
        temps = env.get("temperature_c", [])
        dust = env.get("dust_opacity", [])
        radiation = env.get("radiation_msv", [])

    env_js_data = f"const ENV = {{temp:{temps},dust:{dust},radiation:{radiation}}};\n"

    # Extract key events for timeline annotations
    events_js = _build_events_js(colonies)

    # Monte Carlo band data (if available)
    mc_js = _build_mc_js(mc_data) if mc_data else "const MC = null;\n"

    total_mig = results.get("summary", {}).get("total_migrations", results.get("migration", {}).get("total_transfers", 0))

    mc_subtitle = f" · Monte Carlo: {mc_data['n_seeds']} seeds" if mc_data else ""

    # MC summary cards (survival rates, confidence intervals)
    mc_cards_html = ""
    if mc_data:
        mc_cards_html = '<div class="mc-section"><h2>📊 Monte Carlo Statistics</h2><div class="cards">'
        for ci, name in enumerate(mc_data["colony_names"]):
            color = COLORS.get(name, "#888")
            fps = mc_data["final_pop_stats"][ci]
            gps = mc_data["growth_pct_stats"][ci]
            surv = mc_data["survival_rates"][ci]
            surv_color = "#2ecc71" if surv >= 0.99 else "#f39c12" if surv >= 0.9 else "#e74c3c"
            mc_cards_html += f'''
            <div class="card" style="border-color: {color}">
                <h3 style="color: {color}">{name}</h3>
                <div class="stat" style="color:{surv_color}">{surv * 100:.0f}% survival</div>
                <div class="detail">Final pop: {fps["mean"]:.0f} ± {fps["stdev"]:.0f}</div>
                <div class="detail">Range: {fps["p10"]:.0f} — {fps["p90"]:.0f} (p10–p90)</div>
                <div class="detail">Growth: {gps["mean"]:+.1f}% ± {gps["stdev"]:.1f}%</div>
            </div>'''
        mc_cards_html += '</div></div>'

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mars Barn — Colony Population Curves</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #0a0a0f;
    color: #ccc;
    font-family: 'Courier New', monospace;
    padding: 20px;
    max-width: 960px;
    margin: 0 auto;
}}
h1 {{ color: #e74c3c; font-size: 1.8em; margin-bottom: 4px; }}
.subtitle {{ color: #666; font-size: 0.85em; margin-bottom: 20px; }}
.cards {{ display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }}
.card {{
    flex: 1; min-width: 200px; background: #111;
    border: 2px solid #333; border-radius: 8px; padding: 14px;
}}
.card h3 {{ font-size: 1.1em; margin-bottom: 4px; }}
.card .strategy {{ color: #666; font-size: 0.75em; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; }}
.card .stat {{ font-size: 1.3em; color: #eee; }}
.card .arrow {{ font-size: 0.9em; }}
.card .detail {{ color: #888; font-size: 0.8em; margin-top: 4px; }}
.chart-box {{
    background: #111; border-radius: 8px; padding: 16px; margin-bottom: 16px;
    position: relative;
}}
.chart-box h3 {{ color: #aaa; font-size: 0.95em; margin-bottom: 8px; }}
canvas {{ width: 100%; height: 260px; display: block; }}
.tooltip {{
    position: absolute; background: #222; color: #eee; padding: 8px 12px;
    border-radius: 6px; font-size: 0.8em; pointer-events: none;
    display: none; z-index: 10; border: 1px solid #444;
    white-space: nowrap;
}}
.legend {{
    text-align: center; margin-bottom: 16px; font-size: 0.85em;
}}
.legend-item {{ margin: 0 12px; color: #aaa; cursor: pointer; }}
.legend-item.hidden {{ opacity: 0.3; text-decoration: line-through; }}
.swatch {{
    display: inline-block; width: 12px; height: 12px;
    border-radius: 2px; margin-right: 4px; vertical-align: middle;
}}
.stats-bar {{
    display: flex; gap: 20px; justify-content: center;
    margin-bottom: 20px; font-size: 0.85em; color: #888;
}}
.stats-bar span {{ color: #aaa; }}
footer {{
    color: #444; font-size: 0.75em; text-align: center;
    margin-top: 30px; padding-top: 16px; border-top: 1px solid #1a1a1a;
}}
footer a {{ color: #555; }}
.mc-section {{ margin-bottom: 24px; }}
.mc-section h2 {{ color: #f39c12; font-size: 1.1em; margin-bottom: 12px; }}
.event-marker {{ position: absolute; width: 2px; background: rgba(255,255,255,0.15); top: 0; bottom: 30px; pointer-events: none; }}
</style>
</head>
<body>
<h1>🔴 Mars Barn</h1>
<p class="subtitle">{meta["sols"]} sols · 3 colonies · seed 42{mc_subtitle} · generated {meta["generated"][:10]}</p>

<div class="stats-bar">
    <div>Total migrations: <span>{total_mig}</span></div>
</div>

<div class="legend" id="legend"></div>

<div class="cards">{cards_html}</div>

{mc_cards_html}

<div class="chart-box">
    <h3>Population + Carrying Capacity (K)</h3>
    <canvas id="pop-chart"></canvas>
    <div class="tooltip" id="pop-tip"></div>
</div>
<div class="chart-box">
    <h3>Genetic Diversity</h3>
    <canvas id="diversity-chart"></canvas>
    <div class="tooltip" id="diversity-tip"></div>
</div>
<div class="chart-box" id="mc-pop-box" style="display:none">
    <h3>Population — Monte Carlo Confidence Bands (p10–p90)</h3>
    <canvas id="mc-pop-chart"></canvas>
    <div class="tooltip" id="mc-pop-tip"></div>
</div>
<div class="chart-box">
    <h3>Food Reserves (kg)</h3>
    <canvas id="food-chart"></canvas>
    <div class="tooltip" id="food-tip"></div>
</div>
<div class="chart-box">
    <h3>Colony Morale</h3>
    <canvas id="morale-chart"></canvas>
    <div class="tooltip" id="morale-tip"></div>
</div>
<div class="chart-box">
    <h3>Cumulative Births</h3>
    <canvas id="births-chart"></canvas>
    <div class="tooltip" id="births-tip"></div>
</div>
<div class="chart-box">
    <h3>Mars Surface Temperature (°C)</h3>
    <canvas id="temp-chart"></canvas>
    <div class="tooltip" id="temp-tip"></div>
</div>
<div class="chart-box">
    <h3>Death Causes by Colony</h3>
    <canvas id="death-chart" style="height: 200px"></canvas>
    <div class="tooltip" id="death-tip"></div>
</div>

<footer>
    Mars Barn Terrarium v3.0 · <a href="https://github.com/kody-w/rappterbook-agent-exchange">rappterbook-agent-exchange</a> · Built by the Rappterbook agent swarm
</footer>

<script>
"use strict";
{colony_js_data}
{env_js_data}
{events_js}
{mc_js}

// Visibility toggles
const visible = COLONIES.map(() => true);

// Build legend
const legendEl = document.getElementById("legend");
COLONIES.forEach((c, i) => {{
    const span = document.createElement("span");
    span.className = "legend-item";
    span.innerHTML = `<span class="swatch" style="background:${{c.color}}"></span>${{c.name}}`;
    span.onclick = () => {{
        visible[i] = !visible[i];
        span.classList.toggle("hidden");
        drawAll();
    }};
    legendEl.appendChild(span);
}});

function cumsum(arr) {{
    let s = 0;
    return arr.map(v => (s += v, s));
}}

function drawChart(canvasId, tipId, series, opts) {{
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

    // Compute bounds
    let allVals = [];
    series.forEach(s => {{ if (s.show !== false) allVals.push(...s.data); }});
    if (allVals.length === 0) return;
    let yMin = opts.yMin !== undefined ? opts.yMin : Math.min(...allVals);
    let yMax = opts.yMax !== undefined ? opts.yMax : Math.max(...allVals);
    const pad = (yMax - yMin) * 0.08 || 1;
    yMin -= pad; yMax += pad;
    const n = Math.max(...series.map(s => s.data.length));

    function toX(i) {{ return margin.left + i / Math.max(1, n - 1) * pw; }}
    function toY(v) {{ return margin.top + ph - (v - yMin) / (yMax - yMin) * ph; }}

    // Background
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = "#222"; ctx.lineWidth = 1;
    for (let f = 0; f <= 1; f += 0.25) {{
        const y = margin.top + ph * (1 - f);
        ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(W - margin.right, y); ctx.stroke();
        ctx.fillStyle = "#666"; ctx.font = "10px monospace"; ctx.textAlign = "right";
        ctx.fillText((yMin + (yMax - yMin) * f).toFixed(0), margin.left - 4, y + 3);
    }}
    // X labels
    ctx.textAlign = "center"; ctx.fillStyle = "#666";
    for (let s = 0; s <= n; s += Math.max(1, Math.floor(n / 6))) {{
        ctx.fillText("Sol " + s, toX(s), H - 5);
    }}

    // Dust overlay (orange fill) if provided
    if (opts.dustOverlay) {{
        const d = opts.dustOverlay;
        ctx.beginPath();
        ctx.moveTo(toX(0), toY(yMin));
        for (let i = 0; i < d.length; i++) {{
            const scaledDust = yMin + d[i] * (yMax - yMin);
            ctx.lineTo(toX(i), toY(scaledDust));
        }}
        ctx.lineTo(toX(d.length - 1), toY(yMin));
        ctx.closePath();
        ctx.fillStyle = "rgba(255,165,0,0.08)";
        ctx.fill();
    }}

    // Lines
    series.forEach(s => {{
        if (s.show === false) return;
        ctx.beginPath();
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.dashed ? 1 : 2;
        if (s.dashed) ctx.setLineDash([4, 4]); else ctx.setLineDash([]);
        for (let i = 0; i < s.data.length; i++) {{
            const x = toX(i), y = toY(s.data[i]);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }}
        ctx.stroke();
        ctx.setLineDash([]);
    }});

    // Tooltip on hover
    canvas.onmousemove = (e) => {{
        const bnd = canvas.getBoundingClientRect();
        const mx = e.clientX - bnd.left;
        const sol = Math.round((mx - margin.left) / pw * (n - 1));
        if (sol < 0 || sol >= n) {{ tip.style.display = "none"; return; }}
        let html = `<b>Sol ${{sol}}</b><br>`;
        series.forEach(s => {{
            if (s.show === false || sol >= s.data.length) return;
            const v = s.data[sol];
            html += `<span style="color:${{s.color}}">■</span> ${{s.name}}: ${{typeof v === "number" ? v.toFixed(1) : v}}<br>`;
        }});
        tip.innerHTML = html;
        tip.style.display = "block";
        tip.style.left = Math.min(mx + 12, bnd.width - 180) + "px";
        tip.style.top = "10px";
    }};
    canvas.onmouseleave = () => {{ tip.style.display = "none"; }};
}}

function drawAll() {{
    // Population + K chart
    const popSeries = COLONIES.map((c, i) => ({{
        name: c.name, color: c.color, data: c.pop, show: visible[i]
    }}));
    COLONIES.forEach((c, i) => {{
        if (c.k && c.k.some(v => v > 0)) {{
            popSeries.push({{name: c.name + " (K)", color: c.color, data: c.k, dashed: true, show: visible[i]}});
        }}
    }});
    drawChart("pop-chart", "pop-tip", popSeries, {{dustOverlay: ENV.dust}});

    // Diversity
    const divSeries = COLONIES.map((c, i) => ({{
        name: c.name, color: c.color, data: c.diversity, show: visible[i]
    }}));
    drawChart("diversity-chart", "diversity-tip", divSeries, {{yMin: 0, yMax: 1.1}});

    // Food
    const foodSeries = COLONIES.map((c, i) => ({{
        name: c.name, color: c.color, data: c.food, show: visible[i]
    }}));
    drawChart("food-chart", "food-tip", foodSeries, {{}});

    // Morale
    const moraleSeries = COLONIES.map((c, i) => ({{
        name: c.name, color: c.color, data: c.morale, show: visible[i]
    }}));
    drawChart("morale-chart", "morale-tip", moraleSeries, {{yMin: 0, yMax: 1.1}});

    // Cumulative births
    const birthSeries = COLONIES.map((c, i) => ({{
        name: c.name, color: c.color, data: cumsum(c.births), show: visible[i]
    }}));
    drawChart("births-chart", "births-tip", birthSeries, {{}});

    // Temperature
    drawChart("temp-chart", "temp-tip", [
        {{name: "Temperature", color: "#f39c12", data: ENV.temp}}
    ], {{}});

    // Monte Carlo population bands (if available)
    if (MC) {{
        document.getElementById("mc-pop-box").style.display = "block";
        drawBandChart("mc-pop-chart", "mc-pop-tip", MC, "population");
    }}

    // Draw event markers on population chart
    drawEventMarkers("pop-chart", EVENTS, COLONIES[0].pop.length);

    // Death causes stacked bar chart
    drawDeathCauses();
}}

function drawBandChart(canvasId, tipId, mc, metric) {{
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

    // Compute global bounds across all colonies
    let allVals = [];
    mc.bands.forEach(cb => {{
        const mb = cb[metric];
        if (mb) {{ allVals.push(...mb.p10, ...mb.p90); }}
    }});
    if (allVals.length === 0) return;
    let yMin = Math.min(...allVals);
    let yMax = Math.max(...allVals);
    const pad = (yMax - yMin) * 0.08 || 1;
    yMin -= pad; yMax += pad;
    const n = mc.bands[0][metric].p50.length;

    function toX(i) {{ return margin.left + i / Math.max(1, n - 1) * pw; }}
    function toY(v) {{ return margin.top + ph - (v - yMin) / (yMax - yMin) * ph; }}

    ctx.fillStyle = "#111"; ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = "#222"; ctx.lineWidth = 1;
    for (let f = 0; f <= 1; f += 0.25) {{
        const y = margin.top + ph * (1 - f);
        ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(W - margin.right, y); ctx.stroke();
        ctx.fillStyle = "#666"; ctx.font = "10px monospace"; ctx.textAlign = "right";
        ctx.fillText((yMin + (yMax - yMin) * f).toFixed(0), margin.left - 4, y + 3);
    }}
    ctx.textAlign = "center"; ctx.fillStyle = "#666";
    for (let s = 0; s <= n; s += Math.max(1, Math.floor(n / 6))) {{
        ctx.fillText("Sol " + s, toX(s), H - 5);
    }}

    // Draw bands for each colony
    const colors = mc.bands.map((_, i) => COLONIES[i] ? COLONIES[i].color : "#888");
    mc.bands.forEach((cb, ci) => {{
        if (!visible[ci]) return;
        const mb = cb[metric];
        if (!mb) return;
        const color = colors[ci];

        // p10-p90 band (lightest)
        ctx.beginPath();
        for (let i = 0; i < n; i++) ctx.lineTo(toX(i), toY(mb.p90[i]));
        for (let i = n - 1; i >= 0; i--) ctx.lineTo(toX(i), toY(mb.p10[i]));
        ctx.closePath();
        ctx.fillStyle = color + "15";
        ctx.fill();

        // p25-p75 band (medium)
        ctx.beginPath();
        for (let i = 0; i < n; i++) ctx.lineTo(toX(i), toY(mb.p75[i]));
        for (let i = n - 1; i >= 0; i--) ctx.lineTo(toX(i), toY(mb.p25[i]));
        ctx.closePath();
        ctx.fillStyle = color + "25";
        ctx.fill();

        // p50 median line
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        for (let i = 0; i < n; i++) {{
            i === 0 ? ctx.moveTo(toX(i), toY(mb.p50[i])) : ctx.lineTo(toX(i), toY(mb.p50[i]));
        }}
        ctx.stroke();
    }});

    // Tooltip
    canvas.onmousemove = (e) => {{
        const bnd = canvas.getBoundingClientRect();
        const mx = e.clientX - bnd.left;
        const sol = Math.round((mx - margin.left) / pw * (n - 1));
        if (sol < 0 || sol >= n) {{ tip.style.display = "none"; return; }}
        let html = `<b>Sol ${{sol}}</b> (n=${{mc.n_seeds}} seeds)<br>`;
        mc.bands.forEach((cb, ci) => {{
            if (!visible[ci]) return;
            const mb = cb[metric];
            if (!mb || sol >= mb.p50.length) return;
            const color = colors[ci];
            html += `<span style="color:${{color}}">■</span> ${{mc.colony_names[ci]}}: `;
            html += `${{mb.p50[sol].toFixed(0)}} (p10=${{mb.p10[sol].toFixed(0)}}, p90=${{mb.p90[sol].toFixed(0)}})<br>`;
        }});
        tip.innerHTML = html;
        tip.style.display = "block";
        tip.style.left = Math.min(mx + 12, bnd.width - 220) + "px";
        tip.style.top = "10px";
    }};
    canvas.onmouseleave = () => {{ tip.style.display = "none"; }};
}}

function drawEventMarkers(canvasId, events, nSols) {{
    const canvas = document.getElementById(canvasId);
    if (!canvas || !events || events.length === 0) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    const margin = {{top: 20, right: 16, bottom: 30, left: 55}};
    const pw = W - margin.left - margin.right;

    function toX(sol) {{ return margin.left + sol / Math.max(1, nSols - 1) * pw; }}

    const icons = {{
        "epidemic_start": {{symbol: "☣", color: "#e74c3c"}},
        "epidemic_end": {{symbol: "✓", color: "#2ecc71"}},
        "supply_ship": {{symbol: "🚀", color: "#3498db"}},
        "global_storm": {{symbol: "🌪", color: "#f39c12"}},
        "regional_storm": {{symbol: "💨", color: "#e67e22"}},
        "discovery": {{symbol: "⭐", color: "#f1c40f"}},
    }};

    ctx.save();
    ctx.scale(dpr, dpr);
    events.forEach(ev => {{
        const info = icons[ev.type] || {{symbol: "·", color: "#666"}};
        const x = toX(ev.sol);
        // Thin vertical line
        ctx.strokeStyle = info.color + "40";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, margin.top);
        ctx.lineTo(x, H - margin.bottom);
        ctx.stroke();
        // Symbol at top
        ctx.fillStyle = info.color;
        ctx.font = "10px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(info.symbol, x, margin.top - 4);
    }});
    ctx.restore();
}}

const CAUSE_COLORS = {{
    baseline: "#7f8c8d", starvation: "#e67e22", dehydration: "#3498db",
    power_failure: "#9b59b6", radiation: "#f1c40f", storm: "#e74c3c",
    epidemic: "#1abc9c", accident: "#95a5a6"
}};
const CAUSE_LABELS = {{
    baseline: "Natural", starvation: "Starvation", dehydration: "Dehydration",
    power_failure: "Power Fail", radiation: "Radiation", storm: "Storm",
    epidemic: "Epidemic", accident: "Accident"
}};

function drawDeathCauses() {{
    const canvas = document.getElementById("death-chart");
    const tip = document.getElementById("death-tip");
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    const margin = {{top: 20, right: 16, bottom: 40, left: 55}};
    const pw = W - margin.left - margin.right;
    const ph = H - margin.top - margin.bottom;

    const allCauses = new Set();
    COLONIES.forEach(c => {{
        if (c.deathCauses) Object.keys(c.deathCauses).forEach(k => {{ if (c.deathCauses[k] > 0) allCauses.add(k); }});
    }});
    const causes = [...allCauses].sort();
    if (causes.length === 0) return;

    const nc = COLONIES.length;
    const barGroupWidth = pw / nc;
    const barWidth = Math.min(60, barGroupWidth * 0.6);

    let maxTotal = 0;
    COLONIES.forEach(c => {{
        let t = 0;
        causes.forEach(k => {{ t += (c.deathCauses || {{}})[k] || 0; }});
        maxTotal = Math.max(maxTotal, t);
    }});
    if (maxTotal === 0) return;

    ctx.fillStyle = "#111"; ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = "#222"; ctx.lineWidth = 1;
    for (let f = 0; f <= 1; f += 0.25) {{
        const y = margin.top + ph * (1 - f);
        ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(W - margin.right, y); ctx.stroke();
        ctx.fillStyle = "#666"; ctx.font = "10px monospace"; ctx.textAlign = "right";
        ctx.fillText((maxTotal * f).toFixed(0), margin.left - 4, y + 3);
    }}

    COLONIES.forEach((c, ci) => {{
        const x = margin.left + ci * barGroupWidth + (barGroupWidth - barWidth) / 2;
        let yBottom = margin.top + ph;
        causes.forEach(cause => {{
            const v = (c.deathCauses || {{}})[cause] || 0;
            if (v === 0) return;
            const barH = (v / maxTotal) * ph;
            ctx.fillStyle = CAUSE_COLORS[cause] || "#888";
            ctx.fillRect(x, yBottom - barH, barWidth, barH);
            yBottom -= barH;
        }});
        ctx.fillStyle = c.color; ctx.font = "11px monospace"; ctx.textAlign = "center";
        ctx.fillText(c.name.split(" ")[0], x + barWidth / 2, H - 5);
    }});

    let lx = margin.left;
    ctx.font = "10px monospace"; ctx.textAlign = "left";
    causes.forEach(cause => {{
        ctx.fillStyle = CAUSE_COLORS[cause] || "#888";
        ctx.fillRect(lx, H - 22, 10, 10);
        ctx.fillStyle = "#888";
        ctx.fillText(CAUSE_LABELS[cause] || cause, lx + 13, H - 13);
        lx += ctx.measureText(CAUSE_LABELS[cause] || cause).width + 22;
    }});

    canvas.onmousemove = (e) => {{
        const bnd = canvas.getBoundingClientRect();
        const mx = e.clientX - bnd.left;
        const ci = Math.floor((mx - margin.left) / barGroupWidth);
        if (ci < 0 || ci >= nc) {{ tip.style.display = "none"; return; }}
        const c = COLONIES[ci];
        let html = `<b>${{c.name}}</b><br>`;
        causes.forEach(cause => {{
            const v = (c.deathCauses || {{}})[cause] || 0;
            if (v > 0) html += `<span style="color:${{CAUSE_COLORS[cause] || "#888"}}">■</span> ${{CAUSE_LABELS[cause] || cause}}: ${{v}}<br>`;
        }});
        tip.innerHTML = html;
        tip.style.display = "block";
        tip.style.left = Math.min(mx + 12, bnd.width - 180) + "px";
        tip.style.top = "10px";
    }};
    canvas.onmouseleave = () => {{ tip.style.display = "none"; }};
}}

drawAll();
window.addEventListener("resize", drawAll);
</script>
</body>
</html>'''
    return html
