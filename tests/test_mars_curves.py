"""
Tests for src/mars_curves.py — HTML dashboard generator for Mars Barn.

Property-based invariants: output is valid HTML, colony colors appear,
data arrays are embedded, event counts bounded, Monte Carlo stats render.

Run: python -m pytest tests/test_mars_curves.py -v
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars_curves import COLORS, generate_dashboard, _build_events_js, _build_mc_js


# ── Fixtures: minimal data matching Simulation.run() output format ──


def _make_history(n_sols: int, base_pop: int = 100) -> list[dict]:
    """Generate a minimal sol-by-sol history list."""
    history = []
    pop = base_pop
    for sol in range(n_sols):
        pop = max(1, pop + (1 if sol % 3 == 0 else 0))
        history.append({
            "population": pop,
            "food_kg": 500.0 + sol,
            "morale": 0.7,
            "births": 1 if sol % 5 == 0 else 0,
            "deaths": 0,
            "carrying_capacity": 300,
            "genetic_diversity": 0.95,
            "net_migration": 0,
            "death_causes": {},
        })
    return history


def _make_colony(name: str, strategy: str, n_sols: int = 30,
                 base_pop: int = 100, events: list | None = None) -> dict:
    """Build a colony dict matching Simulation output."""
    hist = _make_history(n_sols, base_pop)
    return {
        "name": name,
        "strategy": strategy,
        "history": hist,
        "events": events or [],
        "death_causes": {"starvation": 0, "radiation": 0, "accident": 1},
        "tech": {"unlocked": []},
    }


def _make_env_history(n_sols: int = 30) -> list[dict]:
    """Build environment history matching MarsEnvironment output."""
    return [
        {
            "temperature_c": -60.0 + (sol * 0.01),
            "dust_opacity": 0.3,
            "radiation_msv": 0.7,
            "terraforming_progress": sol * 0.0001,
            "pressure_kpa": 0.636 + sol * 0.00001,
        }
        for sol in range(n_sols)
    ]


def _make_results(n_sols: int = 30) -> dict:
    """Build a full results dict matching Simulation.run() output."""
    colonies = [
        _make_colony("Ares Prime", "conservative", n_sols, 120),
        _make_colony("Olympus Station", "balanced", n_sols, 80),
        _make_colony("Red Frontier", "aggressive", n_sols, 60),
    ]
    return {
        "_meta": {"sols": n_sols, "seed": 42, "version": "test", "generated": "2026-03-26T00:00:00Z"},
        "colonies": colonies,
        "environment": {"history": _make_env_history(n_sols)},
        "summary": {
            "colonies": [
                {
                    "name": c["name"],
                    "strategy": c["strategy"],
                    "start_pop": c["history"][0]["population"],
                    "end_pop": c["history"][-1]["population"],
                    "peak_pop": max(h["population"] for h in c["history"]),
                    "min_pop": min(h["population"] for h in c["history"]),
                    "total_births": sum(h["births"] for h in c["history"]),
                    "total_deaths": sum(h["deaths"] for h in c["history"]),
                    "growth_pct": 10.0,
                    "net_migration": 0,
                    "death_causes": c["death_causes"],
                }
                for c in colonies
            ],
            "total_migrations": 0,
            "terraforming": {"progress": 0.003, "phase": "atmospheric"},
        },
        "migration": {"total_transfers": 0},
    }


def _make_mc_data() -> dict:
    """Build Monte Carlo data dict matching _serialize_ensemble() output."""
    return {
        "n_seeds": 10,
        "sols": 30,
        "colony_names": ["Ares Prime", "Olympus Station", "Red Frontier"],
        "colony_strategies": ["conservative", "balanced", "aggressive"],
        "bands": [
            {"population": {"p10": [100] * 30, "p50": [110] * 30, "p90": [120] * 30}},
            {"population": {"p10": [70] * 30, "p50": [80] * 30, "p90": [90] * 30}},
            {"population": {"p10": [50] * 30, "p50": [60] * 30, "p90": [70] * 30}},
        ],
        "final_pop_stats": [
            {"mean": 130.0, "stdev": 8.0, "p10": 120.0, "p90": 140.0},
            {"mean": 90.0, "stdev": 5.0, "p10": 84.0, "p90": 96.0},
            {"mean": 75.0, "stdev": 5.0, "p10": 68.0, "p90": 82.0},
        ],
        "growth_pct_stats": [
            {"mean": 8.3, "stdev": 2.0, "p10": 5.0, "p90": 11.0},
            {"mean": 12.5, "stdev": 3.0, "p10": 8.0, "p90": 17.0},
            {"mean": 25.0, "stdev": 5.0, "p10": 18.0, "p90": 32.0},
        ],
        "survival_rates": [1.0, 1.0, 0.95],
    }


# ═══════════════════════════════════════════════════════════════════════
# Section 1: generate_dashboard — basic structure
# ═══════════════════════════════════════════════════════════════════════


class TestDashboardBasicStructure:
    """The generated HTML must be a valid, complete page."""

    def test_returns_string(self):
        html = generate_dashboard(_make_results())
        assert isinstance(html, str)

    def test_starts_with_doctype(self):
        html = generate_dashboard(_make_results())
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_ends_with_html_close(self):
        html = generate_dashboard(_make_results())
        assert html.strip().endswith("</html>")

    def test_contains_head_and_body(self):
        html = generate_dashboard(_make_results())
        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html

    def test_contains_title(self):
        html = generate_dashboard(_make_results())
        assert "<title>" in html
        assert "Mars Barn" in html

    def test_contains_canvas_tags(self):
        html = generate_dashboard(_make_results())
        assert "<canvas" in html

    def test_contains_script_tag(self):
        html = generate_dashboard(_make_results())
        assert "<script>" in html
        assert "</script>" in html

    def test_contains_style_tag(self):
        html = generate_dashboard(_make_results())
        assert "<style>" in html


# ═══════════════════════════════════════════════════════════════════════
# Section 2: Colony data embedding
# ═══════════════════════════════════════════════════════════════════════


class TestColonyDataEmbedding:
    """Colony names, colors, and data arrays must appear in the output."""

    def test_all_colony_names_present(self):
        html = generate_dashboard(_make_results())
        assert "Ares Prime" in html
        assert "Olympus Station" in html
        assert "Red Frontier" in html

    def test_colony_colors_embedded(self):
        html = generate_dashboard(_make_results())
        for color in COLORS.values():
            assert color in html, f"Color {color} missing from dashboard"

    def test_colonies_js_array_present(self):
        html = generate_dashboard(_make_results())
        assert "const COLONIES" in html

    def test_env_js_object_present(self):
        html = generate_dashboard(_make_results())
        assert "const ENV" in html

    def test_terraform_js_present(self):
        html = generate_dashboard(_make_results())
        assert "const TERRAFORM" in html

    def test_population_data_in_output(self):
        """Population values from history must be embedded in JS."""
        results = _make_results()
        html = generate_dashboard(results)
        first_pop = results["colonies"][0]["history"][0]["population"]
        assert str(first_pop) in html

    def test_temperature_data_in_output(self):
        results = _make_results()
        html = generate_dashboard(results)
        assert "-60.0" in html or "-60" in html

    def test_dust_opacity_in_output(self):
        results = _make_results()
        html = generate_dashboard(results)
        assert "0.3" in html


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Summary cards
# ═══════════════════════════════════════════════════════════════════════


class TestSummaryCards:
    """Summary cards show colony stats correctly."""

    def test_strategy_labels_present(self):
        html = generate_dashboard(_make_results())
        assert "CONSERVATIVE" in html
        assert "BALANCED" in html
        assert "AGGRESSIVE" in html

    def test_growth_arrow_up(self):
        """Positive growth should show up arrow."""
        html = generate_dashboard(_make_results())
        assert "↑" in html

    def test_growth_arrow_down(self):
        """Negative growth should show down arrow."""
        results = _make_results()
        results["summary"]["colonies"][0]["growth_pct"] = -5.0
        html = generate_dashboard(results)
        assert "↓" in html

    def test_growth_arrow_flat(self):
        """Zero growth should show right arrow."""
        results = _make_results()
        results["summary"]["colonies"][0]["growth_pct"] = 0
        html = generate_dashboard(results)
        assert "→" in html

    def test_migration_shown_when_nonzero(self):
        results = _make_results()
        results["summary"]["colonies"][0]["net_migration"] = 5
        html = generate_dashboard(results)
        assert "Migration:" in html
        assert "+5" in html

    def test_migration_hidden_when_zero(self):
        results = _make_results()
        html = generate_dashboard(results)
        # Migration: string should not appear for zero migration
        card_section = html.split("const COLONIES")[0]
        assert "Migration:" not in card_section

    def test_death_causes_shown(self):
        results = _make_results()
        results["summary"]["colonies"][0]["death_causes"] = {"starvation": 5, "radiation": 2}
        html = generate_dashboard(results)
        assert "starvation" in html

    def test_births_and_deaths_shown(self):
        html = generate_dashboard(_make_results())
        assert "Births:" in html
        assert "Deaths:" in html


# ═══════════════════════════════════════════════════════════════════════
# Section 4: _build_events_js
# ═══════════════════════════════════════════════════════════════════════


class TestBuildEventsJs:
    """Event extraction from colony data for timeline annotations."""

    def test_returns_js_const(self):
        js = _build_events_js([])
        assert js.startswith("const EVENTS = ")
        assert js.endswith(";\n")

    def test_empty_colonies_gives_empty_array(self):
        js = _build_events_js([])
        assert "const EVENTS = [];" in js

    def test_storm_events_extracted(self):
        colony = {"events": [{"sol": 10, "type": "storm", "kind": "regional"}]}
        js = _build_events_js([colony])
        parsed = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert len(parsed) == 1
        assert parsed[0]["type"] == "regional_storm"
        assert parsed[0]["sol"] == 10

    def test_global_storm_type(self):
        colony = {"events": [{"sol": 50, "type": "storm", "kind": "global"}]}
        js = _build_events_js([colony])
        parsed = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert parsed[0]["type"] == "global_storm"

    def test_storm_default_kind_is_regional(self):
        colony = {"events": [{"sol": 5, "type": "storm"}]}
        js = _build_events_js([colony])
        parsed = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert parsed[0]["type"] == "regional_storm"

    def test_duplicate_storms_deduplicated(self):
        """Two storms at the same sol should be collapsed to one."""
        colonies = [
            {"events": [{"sol": 10, "type": "storm", "kind": "regional"}]},
            {"events": [{"sol": 10, "type": "storm", "kind": "regional"}]},
        ]
        js = _build_events_js(colonies)
        parsed = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        storm_at_10 = [e for e in parsed if e["sol"] == 10 and "storm" in e["type"]]
        assert len(storm_at_10) == 1

    def test_epidemic_events_extracted(self):
        colony = {"events": [
            {"sol": 20, "type": "epidemic_start", "strain": "Mars Flu"},
            {"sol": 35, "type": "epidemic_end", "strain": "Mars Flu"},
        ]}
        js = _build_events_js([colony])
        parsed = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        types = [e["type"] for e in parsed]
        assert "epidemic_start" in types
        assert "epidemic_end" in types

    def test_epidemic_label_uses_strain(self):
        colony = {"events": [{"sol": 20, "type": "epidemic_start", "strain": "Rad Fever"}]}
        js = _build_events_js([colony])
        parsed = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert parsed[0]["label"] == "Rad Fever"

    def test_supply_ship_extracted(self):
        colony = {"events": [{"sol": 120, "type": "supply_ship"}]}
        js = _build_events_js([colony])
        parsed = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert any(e["type"] == "supply_ship" for e in parsed)

    def test_supply_ship_label_fallback(self):
        colony = {"events": [{"sol": 120, "type": "supply_ship"}]}
        js = _build_events_js([colony])
        parsed = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        ship = [e for e in parsed if e["type"] == "supply_ship"][0]
        assert ship["label"] == "supply ship"

    def test_discovery_events_extracted(self):
        colony = {"events": [{"sol": 80, "type": "discovery", "kind": "ice deposit"}]}
        js = _build_events_js([colony])
        parsed = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert any(e["type"] == "discovery" for e in parsed)
        assert any(e["label"] == "ice deposit" for e in parsed)

    def test_tech_unlock_extracted(self):
        colony = {"name": "Ares Prime", "events": [
            {"sol": 100, "type": "tech_unlock", "name": "Advanced Recycling"}
        ]}
        js = _build_events_js([colony])
        parsed = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        tech = [e for e in parsed if e["type"] == "tech_unlock"]
        assert len(tech) == 1
        assert tech[0]["label"] == "Advanced Recycling"
        assert tech[0]["colony"] == "Ares Prime"

    def test_events_sorted_by_sol(self):
        colony = {"events": [
            {"sol": 200, "type": "supply_ship"},
            {"sol": 50, "type": "storm", "kind": "regional"},
            {"sol": 100, "type": "discovery", "kind": "cave"},
        ]}
        js = _build_events_js([colony])
        parsed = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        sols = [e["sol"] for e in parsed]
        assert sols == sorted(sols)

    def test_max_40_events(self):
        """Events should be capped at 40."""
        colony = {"events": [
            {"sol": i, "type": "supply_ship"} for i in range(50)
        ]}
        js = _build_events_js([colony])
        parsed = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert len(parsed) <= 40

    def test_priority_sorting_epidemics_first(self):
        """When truncating, higher-priority events survive."""
        events = []
        for i in range(45):
            events.append({"sol": i + 100, "type": "discovery", "kind": "rock"})
        events.append({"sol": 10, "type": "epidemic_start", "strain": "Rad Fever"})
        colony = {"events": events}
        js = _build_events_js([colony])
        parsed = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        types = [e["type"] for e in parsed]
        assert "epidemic_start" in types

    def test_non_list_events_skipped(self):
        """Colony with non-list events field should not crash."""
        colony = {"events": "not a list"}
        js = _build_events_js([colony])
        parsed = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert parsed == []

    def test_unknown_event_type_ignored(self):
        colony = {"events": [{"sol": 5, "type": "alien_invasion"}]}
        js = _build_events_js([colony])
        parsed = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert len(parsed) == 0

    def test_missing_events_key(self):
        """Colony with no events key should not crash."""
        colony = {"name": "Test"}
        js = _build_events_js([colony])
        parsed = json.loads(js.replace("const EVENTS = ", "").rstrip(";\n"))
        assert parsed == []

    def test_output_is_valid_json(self):
        colony = {"events": [
            {"sol": 10, "type": "storm", "kind": "regional"},
            {"sol": 50, "type": "epidemic_start", "strain": "Mars Flu"},
        ]}
        js = _build_events_js([colony])
        data_str = js.replace("const EVENTS = ", "").rstrip(";\n")
        parsed = json.loads(data_str)
        assert isinstance(parsed, list)


# ═══════════════════════════════════════════════════════════════════════
# Section 5: _build_mc_js
# ═══════════════════════════════════════════════════════════════════════


class TestBuildMcJs:
    """Monte Carlo JS serialization."""

    def test_returns_js_const(self):
        js = _build_mc_js({"n_seeds": 5})
        assert js.startswith("const MC = ")
        assert js.endswith(";\n")

    def test_output_is_valid_json(self):
        mc = _make_mc_data()
        js = _build_mc_js(mc)
        data_str = js.replace("const MC = ", "").rstrip(";\n")
        parsed = json.loads(data_str)
        assert isinstance(parsed, dict)

    def test_n_seeds_preserved(self):
        mc = _make_mc_data()
        js = _build_mc_js(mc)
        data_str = js.replace("const MC = ", "").rstrip(";\n")
        parsed = json.loads(data_str)
        assert parsed["n_seeds"] == 10

    def test_colony_names_preserved(self):
        mc = _make_mc_data()
        js = _build_mc_js(mc)
        data_str = js.replace("const MC = ", "").rstrip(";\n")
        parsed = json.loads(data_str)
        assert parsed["colony_names"] == mc["colony_names"]

    def test_compact_separators(self):
        """Output should use compact JSON (no spaces)."""
        mc = _make_mc_data()
        js = _build_mc_js(mc)
        # Compact JSON has no spaces after : or ,
        data_str = js.replace("const MC = ", "").rstrip(";\n")
        assert ": " not in data_str or ", " not in data_str

    def test_empty_dict_works(self):
        js = _build_mc_js({})
        assert "const MC = {};" in js


# ═══════════════════════════════════════════════════════════════════════
# Section 6: generate_dashboard with Monte Carlo data
# ═══════════════════════════════════════════════════════════════════════


class TestDashboardWithMonteCarlo:
    """Dashboard should render MC statistics when mc_data is provided."""

    def test_mc_null_without_data(self):
        html = generate_dashboard(_make_results())
        assert "const MC = null;" in html

    def test_mc_present_with_data(self):
        html = generate_dashboard(_make_results(), mc_data=_make_mc_data())
        assert "const MC = {" in html
        assert "const MC = null" not in html

    def test_mc_subtitle_present(self):
        html = generate_dashboard(_make_results(), mc_data=_make_mc_data())
        assert "Monte Carlo" in html
        assert "10 seeds" in html

    def test_mc_cards_section_present(self):
        html = generate_dashboard(_make_results(), mc_data=_make_mc_data())
        assert "Monte Carlo Statistics" in html

    def test_mc_survival_rate_shown(self):
        html = generate_dashboard(_make_results(), mc_data=_make_mc_data())
        assert "100% survival" in html or "95% survival" in html

    def test_mc_final_pop_stats_shown(self):
        html = generate_dashboard(_make_results(), mc_data=_make_mc_data())
        assert "130" in html  # mean of Ares Prime
        assert "± 8" in html  # stdev

    def test_mc_growth_stats_shown(self):
        html = generate_dashboard(_make_results(), mc_data=_make_mc_data())
        assert "+8.3%" in html or "8.3" in html

    def test_no_mc_cards_without_data(self):
        html = generate_dashboard(_make_results())
        assert "Monte Carlo Statistics" not in html


# ═══════════════════════════════════════════════════════════════════════
# Section 7: Edge cases and robustness
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """The dashboard should handle edge cases gracefully."""

    def test_single_sol(self):
        """One-sol simulation should not crash."""
        results = _make_results(n_sols=1)
        html = generate_dashboard(results)
        assert "<!DOCTYPE html>" in html

    def test_large_simulation(self):
        """668-sol simulation should produce valid output."""
        results = _make_results(n_sols=668)
        html = generate_dashboard(results)
        assert "<!DOCTYPE html>" in html
        assert len(html) > 1000

    def test_zero_deaths_no_crash(self):
        results = _make_results()
        for s in results["summary"]["colonies"]:
            s["death_causes"] = {}
        html = generate_dashboard(results)
        assert "<!DOCTYPE html>" in html

    def test_missing_growth_pct(self):
        results = _make_results()
        for s in results["summary"]["colonies"]:
            s.pop("growth_pct", None)
        html = generate_dashboard(results)
        assert "→" in html  # fallback to 0 = flat arrow

    def test_missing_net_migration(self):
        results = _make_results()
        for s in results["summary"]["colonies"]:
            s.pop("net_migration", None)
        html = generate_dashboard(results)
        assert "<!DOCTYPE html>" in html

    def test_unknown_colony_name_gets_fallback_color(self):
        results = _make_results()
        results["colonies"][0]["name"] = "Unknown Colony X"
        results["summary"]["colonies"][0]["name"] = "Unknown Colony X"
        html = generate_dashboard(results)
        assert "#888" in html

    def test_compact_data_format_accepted(self):
        """Results with pre-extracted arrays (no history key) should work."""
        results = _make_results()
        for c in results["colonies"]:
            hist = c.pop("history")
            c["population"] = [h["population"] for h in hist]
            c["food_kg"] = [h["food_kg"] for h in hist]
            c["morale"] = [h["morale"] for h in hist]
            c["births"] = [h["births"] for h in hist]
            c["deaths"] = [h["deaths"] for h in hist]
            c["carrying_capacity"] = [h.get("carrying_capacity", 0) for h in hist]
            c["genetic_diversity"] = [h.get("genetic_diversity", 1.0) for h in hist]
            c["net_migration"] = [h.get("net_migration", 0) for h in hist]
            c["cumulative_death_causes"] = c.pop("death_causes", {})
        env_hist = results["environment"].pop("history")
        results["environment"]["temperature_c"] = [e["temperature_c"] for e in env_hist]
        results["environment"]["dust_opacity"] = [e["dust_opacity"] for e in env_hist]
        results["environment"]["radiation_msv"] = [e["radiation_msv"] for e in env_hist]
        results["environment"]["terraforming_progress"] = [
            e.get("terraforming_progress", 0) for e in env_hist
        ]
        results["environment"]["pressure_kpa"] = [
            e.get("pressure_kpa", 0.636) for e in env_hist
        ]
        html = generate_dashboard(results)
        assert "const COLONIES" in html
        assert "const ENV" in html

    def test_events_js_embedded_in_dashboard(self):
        results = _make_results()
        results["colonies"][0]["events"] = [
            {"sol": 50, "type": "storm", "kind": "global"},
        ]
        html = generate_dashboard(results)
        assert "const EVENTS" in html


# ═══════════════════════════════════════════════════════════════════════
# Section 8: Property-based invariants
# ═══════════════════════════════════════════════════════════════════════


class TestPropertyInvariants:
    """Invariants that must hold for any valid input."""

    def test_output_length_scales_with_sols(self):
        """More sols = more data = longer HTML."""
        short = generate_dashboard(_make_results(n_sols=10))
        long = generate_dashboard(_make_results(n_sols=200))
        assert len(long) > len(short)

    def test_three_colonies_three_colors(self):
        """Each colony color must appear at least twice (card + JS data)."""
        html = generate_dashboard(_make_results())
        for name, color in COLORS.items():
            count = html.count(color)
            assert count >= 2, f"{name} color {color} appears only {count} time(s)"

    def test_no_python_objects_leak(self):
        """Python repr artifacts should not appear in JS output."""
        html = generate_dashboard(_make_results())
        assert "True" not in html.split("<script>")[1] or "true" in html
        assert "None" not in html.split("<script>")[1]
        assert "False" not in html.split("<script>")[1] or "false" in html

    def test_all_js_consts_present(self):
        """All required JS constants must be defined."""
        html = generate_dashboard(_make_results(), mc_data=_make_mc_data())
        required = ["const COLONIES", "const ENV", "const TERRAFORM",
                     "const EVENTS", "const MC"]
        for const in required:
            assert const in html, f"Missing JS constant: {const}"

    def test_html_tags_balanced(self):
        """Basic check: key HTML tags must be balanced."""
        html = generate_dashboard(_make_results())
        for tag in ["html", "head", "body", "style", "script"]:
            opens = html.count(f"<{tag}")
            closes = html.count(f"</{tag}>")
            assert opens == closes, f"Unbalanced <{tag}>: {opens} opens, {closes} closes"

    def test_deterministic_output(self):
        """Same input should produce same output."""
        r1 = _make_results()
        r2 = _make_results()
        html1 = generate_dashboard(r1)
        html2 = generate_dashboard(r2)
        assert html1 == html2

    def test_mc_survival_color_coding(self):
        """Survival ≥99% = green, ≥90% = orange, <90% = red."""
        mc = _make_mc_data()
        mc["survival_rates"] = [1.0, 0.95, 0.85]
        html = generate_dashboard(_make_results(), mc_data=mc)
        assert "#2ecc71" in html  # green for 100%
        assert "#f39c12" in html  # orange for 95%
        assert "#e74c3c" in html  # red for 85%


# ═══════════════════════════════════════════════════════════════════════
# Section 9: COLORS constant
# ═══════════════════════════════════════════════════════════════════════


class TestColorsConstant:
    """COLORS dict must have the three canonical colonies."""

    def test_three_colonies_defined(self):
        assert len(COLORS) == 3

    def test_ares_prime_red(self):
        assert COLORS["Ares Prime"] == "#e74c3c"

    def test_olympus_blue(self):
        assert COLORS["Olympus Station"] == "#3498db"

    def test_red_frontier_green(self):
        assert COLORS["Red Frontier"] == "#2ecc71"

    def test_all_values_hex(self):
        for name, color in COLORS.items():
            assert re.match(r"^#[0-9a-f]{6}$", color), f"{name} has invalid hex: {color}"


# ═══════════════════════════════════════════════════════════════════════
# Section 10: Smoke test — full pipeline
# ═══════════════════════════════════════════════════════════════════════


class TestSmokePipeline:
    """End-to-end: run a short simulation and generate the dashboard."""

    def test_10_sol_sim_produces_valid_dashboard(self):
        """Run the actual sim for 10 sols and generate a dashboard."""
        from src.tick_engine import Simulation
        sim = Simulation(sols=10, env_seed=42)
        results = sim.run()
        html = generate_dashboard(results)
        assert "<!DOCTYPE html>" in html
        assert "Ares Prime" in html
        assert "const COLONIES" in html
        assert len(html) > 5000

    def test_10_sol_with_mc_band_data(self):
        """Simulate + synthetic MC data produces valid dashboard."""
        from src.tick_engine import Simulation
        sim = Simulation(sols=10, env_seed=42)
        results = sim.run()
        mc = _make_mc_data()
        mc["sols"] = 10
        html = generate_dashboard(results, mc_data=mc)
        assert "Monte Carlo" in html
        assert "const MC = {" in html
