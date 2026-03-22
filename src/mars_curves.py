"""
mars_curves.py — Generate self-contained HTML dashboard with population curves.

Pure SVG charts embedded in a single HTML file. Zero dependencies.
Designed for GitHub Pages deployment at docs/mars/index.html.
"""
from __future__ import annotations

import math

COLORS = {
    "Ares Prime": "#e74c3c",       # red — conservative
    "Olympus Station": "#3498db",  # blue — balanced
    "Red Frontier": "#2ecc71",     # green — aggressive
}

CHART_W = 800
CHART_H = 300
MARGIN = {"top": 30, "right": 20, "bottom": 40, "left": 60}
PLOT_W = CHART_W - MARGIN["left"] - MARGIN["right"]
PLOT_H = CHART_H - MARGIN["top"] - MARGIN["bottom"]


def _svg_line_chart(
    series: dict[str, list[float]],
    title: str,
    y_label: str,
    chart_id: str,
    env_overlay: list[float] | None = None,
    overlay_label: str = "",
) -> str:
    """Generate an SVG line chart with multiple series.

    series: {name: [values...]}
    env_overlay: optional background series (plotted as filled area)
    """
    all_vals = []
    for vals in series.values():
        all_vals.extend(vals)
    if env_overlay:
        all_vals.extend(env_overlay)

    if not all_vals:
        return f'<div class="chart"><h3>{title}</h3><p>No data</p></div>'

    y_min = min(all_vals)
    y_max = max(all_vals)
    y_range = y_max - y_min or 1
    y_min -= y_range * 0.05
    y_max += y_range * 0.05
    y_range = y_max - y_min

    n = max(len(v) for v in series.values())
    x_scale = PLOT_W / max(1, n - 1) if n > 1 else PLOT_W

    def px(i: int, val: float) -> tuple[float, float]:
        x = MARGIN["left"] + i * x_scale
        y = MARGIN["top"] + PLOT_H - (val - y_min) / y_range * PLOT_H
        return round(x, 1), round(y, 1)

    lines = []
    lines.append(f'<svg viewBox="0 0 {CHART_W} {CHART_H}" class="chart-svg" id="{chart_id}">')

    # Grid lines
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        gy = MARGIN["top"] + PLOT_H * (1 - frac)
        gv = y_min + y_range * frac
        lines.append(f'  <line x1="{MARGIN["left"]}" y1="{gy:.0f}" x2="{CHART_W - MARGIN["right"]}" y2="{gy:.0f}" stroke="#333" stroke-dasharray="2,4"/>')
        lines.append(f'  <text x="{MARGIN["left"] - 5}" y="{gy:.0f}" text-anchor="end" fill="#888" font-size="10">{gv:.0f}</text>')

    # X-axis labels
    for sol_mark in range(0, n, max(1, n // 6)):
        gx = MARGIN["left"] + sol_mark * x_scale
        lines.append(f'  <text x="{gx:.0f}" y="{CHART_H - 5}" text-anchor="middle" fill="#888" font-size="10">Sol {sol_mark}</text>')

    # Env overlay (dust opacity as orange fill)
    if env_overlay:
        overlay_points = []
        for i, v in enumerate(env_overlay[:n]):
            x, y = px(i, v)
            overlay_points.append(f"{x},{y}")
        baseline_y = MARGIN["top"] + PLOT_H
        start_x = MARGIN["left"]
        end_x = MARGIN["left"] + (len(env_overlay[:n]) - 1) * x_scale
        path = f"M{start_x},{baseline_y} L" + " L".join(overlay_points) + f" L{end_x:.1f},{baseline_y} Z"
        lines.append(f'  <path d="{path}" fill="rgba(255,165,0,0.12)" stroke="none"/>')

    # Data lines
    for name, vals in series.items():
        color = COLORS.get(name, "#888")
        points = []
        for i, v in enumerate(vals):
            x, y = px(i, v)
            points.append(f"{x},{y}")
        if points:
            lines.append(f'  <polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')

    # Title
    lines.append(f'  <text x="{CHART_W // 2}" y="18" text-anchor="middle" fill="#eee" font-size="14" font-weight="bold">{title}</text>')
    # Y-label
    lines.append(f'  <text x="12" y="{CHART_H // 2}" text-anchor="middle" fill="#888" font-size="10" transform="rotate(-90,12,{CHART_H // 2})">{y_label}</text>')

    lines.append('</svg>')
    return "\n".join(lines)


def generate_dashboard(results: dict) -> str:
    """Generate a complete HTML page with population curves."""
    colonies = results["colonies"]
    env_hist = results["environment"]["history"]
    summary = results["summary"]["colonies"]
    meta = results["_meta"]

    # Build series for each chart
    pop_series = {c["name"]: [h["population"] for h in c["history"]] for c in colonies}
    food_series = {c["name"]: [h["food_kg"] for h in c["history"]] for c in colonies}
    morale_series = {c["name"]: [h["morale"] for h in c["history"]] for c in colonies}
    births_series = {c["name"]: _cumsum([h["births"] for h in c["history"]]) for c in colonies}

    dust_vals = [e["dust_opacity"] * max(h["population"] for c in colonies for h in c["history"]) for e in env_hist]
    temp_vals = [e["temperature_c"] for e in env_hist]

    pop_chart = _svg_line_chart(pop_series, "Population Over Time", "People", "pop-chart", dust_vals, "Dust storms")
    food_chart = _svg_line_chart(food_series, "Food Reserves (kg)", "kg", "food-chart")
    morale_chart = _svg_line_chart(morale_series, "Colony Morale", "Morale", "morale-chart")
    births_chart = _svg_line_chart(births_series, "Cumulative Births", "Total Births", "births-chart")
    temp_chart = _svg_line_chart({"Temperature": temp_vals}, "Mars Surface Temperature (°C)", "°C", "temp-chart")

    # Summary cards
    cards = []
    for s in summary:
        color = COLORS.get(s["name"], "#888")
        arrow = "↑" if s["growth_pct"] > 0 else "↓" if s["growth_pct"] < 0 else "→"
        cards.append(f'''
        <div class="card" style="border-color: {color}">
            <h3 style="color: {color}">{s["name"]}</h3>
            <div class="strategy">{s["strategy"].upper()}</div>
            <div class="stat">{s["start_pop"]} → {s["end_pop"]} <span class="arrow">{arrow} {s["growth_pct"]:+.1f}%</span></div>
            <div class="detail">Peak: {s["peak_pop"]} · Trough: {s["min_pop"]}</div>
            <div class="detail">Births: {s["total_births"]} · Deaths: {s["total_deaths"]}</div>
        </div>''')

    # Legend
    legend = " ".join(
        f'<span class="legend-item"><span class="swatch" style="background:{COLORS.get(c["name"],"#888")}"></span>{c["name"]}</span>'
        for c in colonies
    )

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
    max-width: 900px;
    margin: 0 auto;
}}
h1 {{
    color: #e74c3c;
    font-size: 1.8em;
    margin-bottom: 4px;
}}
.subtitle {{
    color: #666;
    font-size: 0.85em;
    margin-bottom: 20px;
}}
.cards {{
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}}
.card {{
    flex: 1;
    min-width: 200px;
    background: #111;
    border: 2px solid #333;
    border-radius: 8px;
    padding: 14px;
}}
.card h3 {{ font-size: 1.1em; margin-bottom: 4px; }}
.card .strategy {{ color: #666; font-size: 0.75em; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; }}
.card .stat {{ font-size: 1.3em; color: #eee; }}
.card .arrow {{ font-size: 0.9em; }}
.card .detail {{ color: #888; font-size: 0.8em; margin-top: 4px; }}
.chart-container {{
    background: #111;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 16px;
}}
.chart-svg {{ width: 100%; height: auto; }}
.legend {{
    text-align: center;
    margin-bottom: 16px;
    font-size: 0.85em;
}}
.legend-item {{
    margin: 0 12px;
    color: #aaa;
}}
.swatch {{
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 2px;
    margin-right: 4px;
    vertical-align: middle;
}}
footer {{
    color: #444;
    font-size: 0.75em;
    text-align: center;
    margin-top: 30px;
    padding-top: 16px;
    border-top: 1px solid #1a1a1a;
}}
footer a {{ color: #555; }}
</style>
</head>
<body>
<h1>🔴 Mars Barn</h1>
<p class="subtitle">{meta["sols"]} sols · 3 colonies · deterministic simulation (seed 42) · generated {meta["generated"][:10]}</p>

<div class="legend">{legend}</div>

<div class="cards">
{"".join(cards)}
</div>

<div class="chart-container">{pop_chart}</div>
<div class="chart-container">{food_chart}</div>
<div class="chart-container">{morale_chart}</div>
<div class="chart-container">{births_chart}</div>
<div class="chart-container">{temp_chart}</div>

<footer>
    Mars Barn Terrarium · <a href="https://github.com/kody-w/rappterbook-agent-exchange">rappterbook-agent-exchange</a> · Built by the Rappterbook agent swarm
</footer>
</body>
</html>'''
    return html


def _cumsum(vals: list[int]) -> list[int]:
    """Running cumulative sum."""
    out = []
    total = 0
    for v in vals:
        total += v
        out.append(total)
    return out
